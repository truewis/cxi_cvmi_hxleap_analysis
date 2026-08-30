#!/usr/bin/env python3
"""Sigma_theta sweep + unstreaked baseline, PROBABILITY-normalized shots.

Each shot's hit image is divided by its own total intensity before
computing the annulus θ-vector. That makes the per-pixel value a
probability, so shot brightness drops out of the covariance and only
the *shape* of the angular distribution enters.

Layout: 6 columns × 2 rows.
  col 0 : unstreaked (Rayleigh scale = 0)  [reference]
  col 1..5 : streaked (Rayleigh scale = 5 px) at
             sigma_theta ∈ {0, 5, 10, 25, 45} deg
  row 0 : smoothed azimuthal covariance
  row 1 : covariance − unstreaked baseline (smoothed diff)

Reads the SLURM-produced streaked stacks from
  batch_analysis/wiggler_sigma_sweep_metrics/shot_hits_full_sigma_<slug>deg.npz
Regenerates 1000 unstreaked shots inline (no such npz exists on disk).

Writes into
  batch_analysis_results/bootstrap_covariance_explanations/
    plot_sigma_sweep_covariance_prob.png
    plot_sigma_sweep_covariance_prob.npz

Requires conda env CXI.
"""
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

SIGMA_VALUES = [0, 5, 10, 25, 45]           # deg (streaked columns)
CX, CY = 67, 59
N_BINS = 36
DR_HALF = 3.0
RAW_CHANNELS_PER_BIN = 32
SPECTRUM_ROI_START = 13
SMOOTH_SIGMA_BINS = 1.5
GUIDES_DEG = (90.0, 270.0)
BG_N_MIN, BG_N_MAX = 50, 300
LOBE_FRACTION = 0.10
PEAK_BIN_MIN, PEAK_BIN_MAX = 16, 26
RAYLEIGH_SCALE_STREAKED = 5.0
N_UNSTREAKED_SHOTS = 1000
RNG_SEED_UNSTREAKED = 0

LEGACY_OUTPUTS = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
                  'batch_analysis_results/legacy_outputs')
SWEEP_DIR = os.path.join(LEGACY_OUTPUTS, 'wiggler_sigma_sweep_metrics')
OUT_DIR = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
           'batch_analysis_results/bootstrap_covariance_explanations')
OUT_PNG = os.path.join(OUT_DIR, 'plot_sigma_sweep_covariance_prob.png')
OUT_NPZ = os.path.join(OUT_DIR, 'plot_sigma_sweep_covariance_prob.npz')


def simulate_unstreaked(n_shots, rng, profile):
    """Match run_bootstrap_with_slurm_sigma.py at Rayleigh scale = 0 and
    sigma_theta = 0. Since streak_radius = 0, sigma_theta has no effect,
    so the choice is arbitrary — pick 0 for reference clarity."""
    ny, nx = 140, 140
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]
    hits = np.zeros((n_shots, ny, nx), dtype=np.float32)
    mean_energy = np.zeros(n_shots, dtype=float)
    r_max_sq = R_MAX_FROM_CENTER ** 2
    sigma_theta = 0.0

    for shot in range(n_shots):
        peak_bin = int(rng.integers(PEAK_BIN_MIN, PEAK_BIN_MAX + 1))
        mock_energy = (peak_bin + SPECTRUM_ROI_START) * RAW_CHANNELS_PER_BIN
        re = (mock_energy / RAW_CHANNELS_PER_BIN
              - SPECTRUM_ROI_START) * 0.6 + 29.4
        mean_energy[shot] = mock_energy

        n_bg = int(rng.integers(BG_N_MIN, BG_N_MAX + 1))
        n_lobe = int(round(LOBE_FRACTION * n_bg))

        # Rayleigh scale 0 => streak_radius = 0 always.
        streak_mode = rng.uniform(-np.pi, np.pi)
        streak_radius = 0.0

        if n_lobe > 0:
            # sigma_theta = 0: all lobe electrons share the streak_mode axis.
            theta_streak = np.full(n_lobe, streak_mode)
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
            print(f'  unstreaked shot {shot}/{n_shots}')
    return hits, mean_energy


def azimuthal_prob_matrix(hits, mean_energy):
    """Per-shot azimuthal PROBABILITY vector: normalize each image by its
    own total intensity before summing into θ-bins in the annulus."""
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
        img = hits[i].astype(np.float64, copy=False)
        total = img.sum()
        if total <= 0:
            rows[i] = 0.0
            continue
        prob = img / total          # each pixel is P(pixel | shot)
        re = (mean_energy[i] / RAW_CHANNELS_PER_BIN
              - SPECTRUM_ROI_START) * 0.6 + 29.4
        annulus = (r_map >= re - DR_HALF) & (r_map < re + DR_HALF)
        rows[i] = np.bincount(
            theta_bin_map[annulus],
            weights=prob[annulus],
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


def draw(ax, M, title, vmax):
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-15
    im = ax.imshow(M, cmap='RdBu_r', origin='lower',
                   extent=[0, 360, 0, 360],
                   vmin=-vmax, vmax=vmax, aspect='equal')
    for g in GUIDES_DEG:
        ax.axvline(g, color='k', ls=':', lw=0.7, alpha=0.55)
        ax.axhline(g, color='k', ls=':', lw=0.7, alpha=0.55)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_xlabel(r'$\theta_j$ [deg]', fontsize=8)
    ax.set_ylabel(r'$\theta_i$ [deg]', fontsize=8)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def main():
    diag = np.eye(N_BINS, dtype=bool)
    profile = _load_noise_profile(NOISE_PROFILE_PATH)

    print(f'--- synthesizing {N_UNSTREAKED_SHOTS} unstreaked shots '
          f'(Rayleigh=0) ---')
    rng = np.random.default_rng(RNG_SEED_UNSTREAKED)
    hits_u, en_u = simulate_unstreaked(N_UNSTREAKED_SHOTS, rng, profile)
    H_u = azimuthal_prob_matrix(hits_u, en_u)
    cov_u = np.cov(H_u, rowvar=False)
    cov_u_p = np.where(diag, np.nan, cov_u)
    cov_u_sm = nan_gaussian_filter(cov_u_p, SMOOTH_SIGMA_BINS)

    cov_s_sm = {}
    diff_sm = {}
    n_shots_by_sigma = {}
    for s in SIGMA_VALUES:
        npz = os.path.join(SWEEP_DIR, f'shot_hits_full_sigma_{s}deg.npz')
        d = np.load(npz)
        hits = d['hits']
        en = d['mean_energy']
        print(f'  sigma={s} deg: {hits.shape[0]} streaked shots')
        H = azimuthal_prob_matrix(hits, en)
        cov = np.cov(H, rowvar=False)
        cov_p = np.where(diag, np.nan, cov)
        cov_s_sm[s] = nan_gaussian_filter(cov_p, SMOOTH_SIGMA_BINS)
        diff_sm[s] = nan_gaussian_filter(cov_p - cov_u_p, SMOOTH_SIGMA_BINS)
        n_shots_by_sigma[s] = hits.shape[0]

    v_cov = max(
        [float(np.nanmax(np.abs(cov_u_sm)))]
        + [float(np.nanmax(np.abs(cov_s_sm[s]))) for s in SIGMA_VALUES]
    )
    v_diff = max(float(np.nanmax(np.abs(diff_sm[s]))) for s in SIGMA_VALUES)

    ncols = 1 + len(SIGMA_VALUES)
    fig, axes = plt.subplots(2, ncols, figsize=(3.5 * ncols, 7.4))
    draw(axes[0, 0], cov_u_sm,
         f'unstreaked (Rayleigh=0)\nN={N_UNSTREAKED_SHOTS}', v_cov)
    axes[1, 0].axis('off')
    axes[1, 0].set_title(
        '(reference — no diff)', fontsize=9, y=0.5)
    for c, s in enumerate(SIGMA_VALUES, start=1):
        draw(axes[0, c], cov_s_sm[s],
             f'σ_θ = {s}°  (Rayleigh=5 px)\nN={n_shots_by_sigma[s]}',
             v_cov)
        draw(axes[1, c], diff_sm[s],
             f'σ_θ = {s}° − unstreaked', v_diff)
    fig.suptitle(
        'Probability-normalized azimuthal covariance across '
        f'sigma_theta sweep (all shots, smoothed σ={SMOOTH_SIGMA_BINS:g} bins)',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f'wrote {OUT_PNG}')

    np.savez_compressed(
        OUT_NPZ,
        sigma_values=np.array(SIGMA_VALUES),
        cov_unstreaked=cov_u,
        cov_unstreaked_sm=np.nan_to_num(cov_u_sm),
        cov_streaked_sm=np.stack([np.nan_to_num(cov_s_sm[s])
                                  for s in SIGMA_VALUES]),
        diff_sm=np.stack([np.nan_to_num(diff_sm[s])
                          for s in SIGMA_VALUES]),
        rayleigh_scale_streaked=RAYLEIGH_SCALE_STREAKED,
        n_unstreaked_shots=N_UNSTREAKED_SHOTS,
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
    )
    print(f'wrote {OUT_NPZ}')


if __name__ == '__main__':
    main()
