#!/usr/bin/env python3
"""All-shots azimuthal-covariance variant.

Uses the same lobe + background generator as
explanation_bootstrap_covariance.py but SKIPS the streak-finder gate
entirely — the covariance is built from every synthetic shot's per-shot
theta vector. Runs both Rayleigh scale = 0 (unstreaked) and Rayleigh
scale = 5 px (streaked) with N_SHOTS shots each.

Writes:
  explanation_bootstrap_covariance_unstreaked_allshots.png
      lobe-density + azimuthal covariance (unstreaked, all shots)
  explanation_bootstrap_covariance_diff_allshots.png
      2 x 3 grid: streaked / unstreaked / (streaked - unstreaked),
      raw on row 1, NaN-aware Gaussian smoothed (sigma = 1.5 bins) on
      row 2.  Guides at theta = 90 and 270 deg.
  explanation_bootstrap_covariance_allshots.npz
      cached H, cov, lobe density for both scales.

Requires conda env CXI.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
# Add batch_analysis/ to path so sibling imports work from plotting/.
sys.path.insert(0, os.path.dirname(_HERE))

from run_bootstrap_with_slurm_sigma import (
    NOISE_PROFILE_PATH,
    R_MAX_FROM_CENTER,
    SHELL_THICKNESS,
    _load_noise_profile,
    _sample_bg_from_profile,
    _sample_projected_shell_with_sin2,
)

# --- Sim knobs (mirror explanation_bootstrap_covariance.py) ---
N_SHOTS_DEFAULT = 1000
BG_N_MIN, BG_N_MAX = 50, 300
LOBE_FRACTION = 0.10
PEAK_BIN_MIN, PEAK_BIN_MAX = 16, 26
SIGMA_THETA_DEG = 25.0
RNG_SEED = 0

CX, CY = 67, 59
N_BINS = 36
DR_HALF = 3.0
RAW_CHANNELS_PER_BIN = 32
SPECTRUM_ROI_START = 13
SMOOTH_SIGMA_BINS = 1.5
GUIDES_DEG = (90.0, 270.0)

OUT_UNSTREAKED_PNG = os.path.join(
    _HERE, 'explanation_bootstrap_covariance_unstreaked_allshots.png')
OUT_DIFF_PNG = os.path.join(
    _HERE, 'explanation_bootstrap_covariance_diff_allshots.png')
OUT_NPZ = os.path.join(
    _HERE, 'explanation_bootstrap_covariance_allshots.npz')


def simulate(n_shots, rayleigh_scale, rng, profile):
    ny, nx = 140, 140
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]

    hits = np.zeros((n_shots, ny, nx), dtype=np.float32)
    mean_energy = np.zeros(n_shots, dtype=float)
    lobe_only_density = np.zeros((ny, nx), dtype=float)

    sigma_theta = np.deg2rad(SIGMA_THETA_DEG)
    r_max_sq = R_MAX_FROM_CENTER ** 2

    for shot in range(n_shots):
        peak_bin = int(rng.integers(PEAK_BIN_MIN, PEAK_BIN_MAX + 1))
        mock_energy = (peak_bin + SPECTRUM_ROI_START) * RAW_CHANNELS_PER_BIN
        re = (mock_energy / RAW_CHANNELS_PER_BIN
              - SPECTRUM_ROI_START) * 0.6 + 29.4
        mean_energy[shot] = mock_energy

        n_bg = int(rng.integers(BG_N_MIN, BG_N_MAX + 1))
        n_lobe = int(round(LOBE_FRACTION * n_bg))

        streak_mode = rng.uniform(-np.pi, np.pi)
        streak_radius = (rng.rayleigh(scale=rayleigh_scale)
                         if rayleigh_scale > 0 else 0.0)

        if n_lobe > 0:
            theta_streak = rng.normal(loc=streak_mode,
                                      scale=sigma_theta, size=n_lobe)
            lobe_cx = CX + streak_radius * np.cos(theta_streak)
            lobe_cy = CY + streak_radius * np.sin(theta_streak)
            lobe_x = np.empty(n_lobe)
            lobe_y = np.empty(n_lobe)
            filled = 0
            while filled < n_lobe:
                need = n_lobe - filled
                dx, dy = _sample_projected_shell_with_sin2(
                    n=need, re=re, dr=SHELL_THICKNESS,
                    r_max=R_MAX_FROM_CENTER, rng=rng,
                )
                cand_x = lobe_cx[filled:filled + need] + dx
                cand_y = lobe_cy[filled:filled + need] + dy
                keep = ((cand_x - CX) ** 2 + (cand_y - CY) ** 2) <= r_max_sq
                take = int(keep.sum())
                lobe_x[filled:filled + take] = cand_x[keep]
                lobe_y[filled:filled + take] = cand_y[keep]
                filled += take
        else:
            lobe_x = np.empty(0)
            lobe_y = np.empty(0)

        bg_x, bg_y = _sample_bg_from_profile(
            n=n_bg, profile=profile,
            cx=CX, cy=CY, r_max=R_MAX_FROM_CENTER, rng=rng,
        )

        if lobe_x.size > 0:
            H, _, _ = np.histogram2d(
                lobe_y, lobe_x,
                bins=(np.arange(ny + 1), np.arange(nx + 1)),
            )
            lobe_only_density += H

        all_x = np.concatenate([lobe_x, bg_x])
        all_y = np.concatenate([lobe_y, bg_y])

        img = np.zeros((ny, nx))
        for k in range(len(all_x)):
            sigma_e = rng.uniform(0.1, 0.8)
            amp = 1.0 / np.sqrt(2.0 * np.pi) / sigma_e
            gp = amp * np.exp(
                -((x_grid - all_x[k]) ** 2 + (y_grid - all_y[k]) ** 2)
                / (2.0 * sigma_e ** 2)
            )
            above = gp >= 0.2
            npix = int(above.sum())
            if npix > 0:
                img[above] += 1.0 / npix
        img[104:116, 92:101] = 0.0
        hits[shot] = img.astype(np.float32)

        if shot % 100 == 0:
            print(f'  synth [scale={rayleigh_scale:g}] shot {shot}/{n_shots}')

    return hits, mean_energy, lobe_only_density


def azimuthal_theta_matrix(hits, mean_energy):
    ny, nx = hits.shape[1:]
    y, x = np.mgrid[0:ny, 0:nx]
    r_map = np.hypot(x - CX, y - CY)
    theta = np.arctan2(y - CY, x - CX) + np.pi
    theta_bin_map = np.clip(
        np.floor(theta / (2.0 * np.pi / N_BINS)).astype(int),
        0, N_BINS - 1,
    )
    rows = np.empty((hits.shape[0], N_BINS), dtype=float)
    for i in range(hits.shape[0]):
        re = (mean_energy[i] / RAW_CHANNELS_PER_BIN
              - SPECTRUM_ROI_START) * 0.6 + 29.4
        annulus = (r_map >= re - DR_HALF) & (r_map < re + DR_HALF)
        img = hits[i]
        rows[i] = np.bincount(
            theta_bin_map[annulus],
            weights=img[annulus].astype(float, copy=False),
            minlength=N_BINS,
        )
    return rows


def nan_gaussian_filter(A, sigma):
    A = np.asarray(A, dtype=float)
    nan_mask = np.isnan(A)
    V = np.where(nan_mask, 0.0, A)
    W = (~nan_mask).astype(float)
    Vs = gaussian_filter(V, sigma=sigma, mode='wrap')
    Ws = gaussian_filter(W, sigma=sigma, mode='wrap')
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(Ws > 1e-12, Vs / Ws, np.nan)


def draw_lobe(ax, density, title):
    vmax = float(np.percentile(density, 99.5)) or 1.0
    ax.imshow(density, cmap='viridis', origin='lower', vmin=0, vmax=vmax)
    ring_theta = np.linspace(0, 2 * np.pi, 400)
    peak_bin_mid = (PEAK_BIN_MIN + PEAK_BIN_MAX) // 2
    re = ((peak_bin_mid + SPECTRUM_ROI_START)
          - SPECTRUM_ROI_START) * 0.6 + 29.4
    for r in (re - DR_HALF, re + DR_HALF):
        ax.plot(CX + r * np.cos(ring_theta),
                CY + r * np.sin(ring_theta),
                ls='--', color='white', lw=0.9, alpha=0.7)
    ax.plot(CX, CY, marker='+', color='white', ms=8, mew=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def draw_cov(ax, M, title, vmax=None):
    if vmax is None:
        vmax = float(np.nanmax(np.abs(M))) or 1e-12
    im = ax.imshow(M, cmap='RdBu_r', origin='lower',
                   extent=[0, 360, 0, 360],
                   vmin=-vmax, vmax=vmax, aspect='equal')
    for g in GUIDES_DEG:
        ax.axvline(g, color='k', ls=':', lw=0.8, alpha=0.55)
        ax.axhline(g, color='k', ls=':', lw=0.8, alpha=0.55)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_xlabel(r'$\theta_j$ [deg]')
    ax.set_ylabel(r'$\theta_i$ [deg]')
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-shots', type=int, default=N_SHOTS_DEFAULT)
    args = ap.parse_args()
    n_shots = args.n_shots

    profile = _load_noise_profile(NOISE_PROFILE_PATH)

    print(f'--- simulating {n_shots} unstreaked shots (Rayleigh=0) ---')
    rng = np.random.default_rng(RNG_SEED)
    hits_u, en_u, lobe_u = simulate(n_shots, 0.0, rng, profile)
    H_u = azimuthal_theta_matrix(hits_u, en_u)
    cov_u = np.cov(H_u, rowvar=False)

    print(f'--- simulating {n_shots} streaked shots (Rayleigh=5) ---')
    rng = np.random.default_rng(RNG_SEED)
    hits_s, en_s, lobe_s = simulate(n_shots, 5.0, rng, profile)
    H_s = azimuthal_theta_matrix(hits_s, en_s)
    cov_s = np.cov(H_s, rowvar=False)

    diag = np.eye(N_BINS, dtype=bool)
    cov_u_p = np.where(diag, np.nan, cov_u)
    cov_s_p = np.where(diag, np.nan, cov_s)
    diff = cov_s - cov_u
    diff_p = np.where(diag, np.nan, diff)

    cov_u_sm = nan_gaussian_filter(cov_u_p, SMOOTH_SIGMA_BINS)
    cov_s_sm = nan_gaussian_filter(cov_s_p, SMOOTH_SIGMA_BINS)
    diff_sm = nan_gaussian_filter(diff_p, SMOOTH_SIGMA_BINS)

    # ---- Unstreaked lobe + covariance figure ----
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    draw_lobe(
        axes[0], lobe_u,
        f'accumulated lobe density  ({n_shots} shots)\n'
        f'Rayleigh r_scale=0 px (unstreaked), '
        f'sigma_theta={SIGMA_THETA_DEG:g} deg',
    )
    draw_cov(
        axes[1], cov_u_p,
        f'azimuthal covariance, ALL shots  ({n_shots} shots, no gate)',
    )
    fig.suptitle(
        'Unstreaked bootstrap (all shots, no streak-finder gate)',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_UNSTREAKED_PNG, dpi=150)
    plt.close(fig)
    print(f'wrote {OUT_UNSTREAKED_PNG}')

    # ---- Diff figure (streaked vs unstreaked, all shots) ----
    v_raw = max(float(np.nanmax(np.abs(cov_s_p))),
                float(np.nanmax(np.abs(cov_u_p))))
    v_sm = max(float(np.nanmax(np.abs(cov_s_sm))),
               float(np.nanmax(np.abs(cov_u_sm))))
    v_diff_raw = float(np.nanmax(np.abs(diff_p)))
    v_diff_sm = float(np.nanmax(np.abs(diff_sm)))

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 11.2))
    draw_cov(axes[0, 0], cov_s_p,
             f'streaked (Rayleigh=5 px)  |  ALL {n_shots} shots', vmax=v_raw)
    draw_cov(axes[0, 1], cov_u_p,
             f'unstreaked (Rayleigh=0)  |  ALL {n_shots} shots', vmax=v_raw)
    draw_cov(axes[0, 2], diff_p,
             'streaked − unstreaked  (raw)', vmax=v_diff_raw)

    draw_cov(axes[1, 0], cov_s_sm,
             f'streaked (smoothed, σ={SMOOTH_SIGMA_BINS:g} bins)', vmax=v_sm)
    draw_cov(axes[1, 1], cov_u_sm,
             f'unstreaked (smoothed, σ={SMOOTH_SIGMA_BINS:g} bins)', vmax=v_sm)
    draw_cov(axes[1, 2], diff_sm,
             f'streaked − unstreaked (smoothed, σ={SMOOTH_SIGMA_BINS:g} bins)',
             vmax=v_diff_sm)

    fig.suptitle(
        f'Bootstrap covariance across ALL {n_shots} shots '
        '(no streak-finder gate)',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIFF_PNG, dpi=150)
    plt.close(fig)
    print(f'wrote {OUT_DIFF_PNG}')

    np.savez_compressed(
        OUT_NPZ,
        cov_streaked=cov_s,
        cov_unstreaked=cov_u,
        H_streaked=H_s,
        H_unstreaked=H_u,
        lobe_streaked=lobe_s.astype(np.float32),
        lobe_unstreaked=lobe_u.astype(np.float32),
        n_shots=n_shots,
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
    )
    print(f'wrote {OUT_NPZ}')


if __name__ == '__main__':
    main()
