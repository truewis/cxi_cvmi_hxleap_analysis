#!/usr/bin/env python3
"""Rayleigh=0 counterpart to circular_wiggler_combined_5000.

Synthesizes 1000 unstreaked shots (Rayleigh scale = 0) per sigma_theta
in {0, 5, 10, 15, 20} deg, then computes probability-normalized
azimuthal covariance and writes the three same-shape plots:

  plot_rayleigh0_covariance.png              (5-col x 2-row: cov, diff-vs-sigma0)
  plot_rayleigh0_covariance_extdiff.png      (3-row x 5-col: cov, ext-ref, diff)
  plot_rayleigh0_covariance_by_offset.png    (4-row x 5-col: r_off in {-6,-4,-2,0} x sigma)

Outputs into
  batch_analysis_results/circular_wiggler_rayleigh0_1000/

External reference in the ext-diff plot is the same as for the
combined_5000 run: the Rayleigh=0 1000-shot cov cached in
bootstrap_covariance_explanations/plot_sigma_sweep_covariance_prob.npz
(`cov_unstreaked`). Note that at Rayleigh scale = 0 this study is
effectively asking "how much does sigma_theta matter when the streak
radius is 0?" — the answer physically is "not at all", so the sigma
panels here should be statistically indistinguishable.

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

# --- Sim knobs (mirror explanation_bootstrap_covariance_allshots.py) ---
N_SHOTS = 1000
BG_N_MIN, BG_N_MAX = 50, 300
LOBE_FRACTION = 0.10
PEAK_BIN_MIN, PEAK_BIN_MAX = 16, 26
RAYLEIGH_SCALE = 0.0
SIGMA_VALUES = [0, 5, 10, 15, 20]
R_OFFSETS = [-6, -4, -2, 0]
RNG_SEED = 0

CX, CY = 67, 59
N_BINS = 36
DR_HALF = 3.0
RAW_CHANNELS_PER_BIN = 32
SPECTRUM_ROI_START = 13
SMOOTH_SIGMA_BINS = 1.5
GUIDES_DEG = (90.0, 270.0)

OUT_DIR = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
           'batch_analysis_results/circular_wiggler_rayleigh0_1000')
EXTREF_NPZ = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
              'batch_analysis_results/bootstrap_covariance_explanations/'
              'plot_sigma_sweep_covariance_prob.npz')


def simulate(n_shots, sigma_theta_deg, rayleigh_scale, rng, profile):
    ny, nx = 140, 140
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]
    hits = np.zeros((n_shots, ny, nx), dtype=np.float32)
    mean_energy = np.zeros(n_shots, dtype=float)
    r_max_sq = R_MAX_FROM_CENTER ** 2
    sigma_theta = np.deg2rad(sigma_theta_deg)

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
            print(f'  σ_θ={sigma_theta_deg:2d}, shot {shot}/{n_shots}')
    return hits, mean_energy


def azimuthal_prob_matrix(hits, mean_energy, r_offset=0.0):
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
        prob = img / total
        re = (mean_energy[i] / RAW_CHANNELS_PER_BIN
              - SPECTRUM_ROI_START) * 0.6 + 29.4 + r_offset
        annulus = (r_map >= re - DR_HALF) & (r_map < re + DR_HALF)
        if not annulus.any():
            rows[i] = 0.0
            continue
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
    os.makedirs(OUT_DIR, exist_ok=True)
    profile = _load_noise_profile(NOISE_PROFILE_PATH)
    diag = np.eye(N_BINS, dtype=bool)

    # 1) Synthesize hits stack per sigma_theta (Rayleigh scale = 0 always).
    hits_by_sigma = {}
    en_by_sigma = {}
    # Persist hit stacks under a wiggler_sigma_sweep_metrics/ subdir so the
    # layout mirrors the SLURM sweep outputs in
    # batch_analysis_results/legacy_outputs/circular_wiggler_run*/wiggler_sigma_sweep_metrics/.
    HITS_DIR = os.path.join(OUT_DIR, 'wiggler_sigma_sweep_metrics')
    os.makedirs(HITS_DIR, exist_ok=True)

    for s in SIGMA_VALUES:
        rng = np.random.default_rng(RNG_SEED + s)   # different seed per sigma
        print(f'--- simulating {N_SHOTS} unstreaked shots '
              f'(Rayleigh=0, σ_θ={s} deg) ---')
        hits, en = simulate(N_SHOTS, s, RAYLEIGH_SCALE, rng, profile)
        hits_by_sigma[s] = hits
        en_by_sigma[s] = en
        p = os.path.join(HITS_DIR, f'shot_hits_full_sigma_{s}deg.npz')
        np.savez_compressed(
            p,
            hits=hits.astype(np.float32),
            mean_energy=en,
            rayleigh_scale=RAYLEIGH_SCALE,
            sigma_theta_deg=s,
            rng_seed=RNG_SEED + s,
        )
        print(f'  saved {p}')

    # 2) Main sigma-sweep plot (r_off = 0).
    cov_by_sigma = {}
    cov_sm_by_sigma = {}
    for s in SIGMA_VALUES:
        H = azimuthal_prob_matrix(hits_by_sigma[s], en_by_sigma[s], 0.0)
        cov = np.cov(H, rowvar=False)
        cov_by_sigma[s] = cov
        cov_sm_by_sigma[s] = nan_gaussian_filter(
            np.where(diag, np.nan, cov), SMOOTH_SIGMA_BINS)

    diff_sm_by_sigma = {
        s: nan_gaussian_filter(
            np.where(diag, np.nan, cov_by_sigma[s] - cov_by_sigma[0]),
            SMOOTH_SIGMA_BINS)
        for s in SIGMA_VALUES
    }

    ncols = len(SIGMA_VALUES)
    v_cov = max(float(np.nanmax(np.abs(cov_sm_by_sigma[s]))) for s in SIGMA_VALUES)
    v_diff = max(float(np.nanmax(np.abs(diff_sm_by_sigma[s])))
                 for s in SIGMA_VALUES if s > 0) or 1e-15

    fig, axes = plt.subplots(2, ncols, figsize=(3.5 * ncols, 7.4))
    for c, s in enumerate(SIGMA_VALUES):
        draw(axes[0, c], cov_sm_by_sigma[s],
             f'σ_θ = {s}°  (smoothed)\nN = {N_SHOTS} shots', v_cov)
        if s == 0:
            axes[1, c].axis('off')
            axes[1, c].set_title('(reference — no diff)', fontsize=9, y=0.5)
        else:
            draw(axes[1, c], diff_sm_by_sigma[s],
                 f'σ_θ = {s}° − σ_θ = 0°  (smoothed diff)', v_diff)
    fig.suptitle(
        f'Rayleigh=0 {N_SHOTS}-shot bootstrap covariance sigma_theta sweep '
        f'(probability-normalized, smoothed σ={SMOOTH_SIGMA_BINS:g} bins, '
        'mode=wrap)',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT_DIR, 'plot_rayleigh0_covariance.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'wrote {p}')

    np.savez_compressed(
        os.path.join(OUT_DIR, 'plot_rayleigh0_covariance.npz'),
        sigma_values=np.array(SIGMA_VALUES),
        cov_raw=np.stack([cov_by_sigma[s] for s in SIGMA_VALUES]),
        cov_sm=np.stack([np.nan_to_num(cov_sm_by_sigma[s]) for s in SIGMA_VALUES]),
        diff_sm=np.stack([np.nan_to_num(diff_sm_by_sigma[s]) for s in SIGMA_VALUES]),
        n_shots=N_SHOTS,
        rayleigh_scale=RAYLEIGH_SCALE,
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
    )

    # 3) External-diff (vs cached 1000-shot Rayleigh=0 unstreaked cov).
    extref = np.load(EXTREF_NPZ)
    ext_cov = extref['cov_unstreaked']
    ext_n = int(extref['n_unstreaked_shots'])
    ext_sm = nan_gaussian_filter(np.where(diag, np.nan, ext_cov),
                                 SMOOTH_SIGMA_BINS)
    ext_diff_sm = {
        s: nan_gaussian_filter(np.where(diag, np.nan, cov_by_sigma[s] - ext_cov),
                                SMOOTH_SIGMA_BINS)
        for s in SIGMA_VALUES
    }
    v_ext_cov = max(v_cov, float(np.nanmax(np.abs(ext_sm))))
    v_ext_diff = max(float(np.nanmax(np.abs(ext_diff_sm[s]))) for s in SIGMA_VALUES) or 1e-15

    fig, axes = plt.subplots(3, ncols, figsize=(3.5 * ncols, 10.6))
    for c, s in enumerate(SIGMA_VALUES):
        draw(axes[0, c], cov_sm_by_sigma[s],
             f'Rayleigh=0 σ_θ = {s}°  (smoothed)\nN = {N_SHOTS}', v_ext_cov)
        if c == 0:
            draw(axes[1, c], ext_sm,
                 f'external ref (Rayleigh=0)\nN = {ext_n}', v_ext_cov)
        else:
            axes[1, c].axis('off')
            axes[1, c].set_title('(ext ref shown in col 0)', fontsize=9, y=0.5)
        draw(axes[2, c], ext_diff_sm[s],
             f'σ_θ = {s}° − ext ref  (smoothed diff)', v_ext_diff)
    fig.suptitle(
        f'Rayleigh=0 {N_SHOTS}-shot bootstrap covariance vs external ref  '
        f'(smoothed σ={SMOOTH_SIGMA_BINS:g} bins, mode=wrap)',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    p = os.path.join(OUT_DIR, 'plot_rayleigh0_covariance_extdiff.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'wrote {p}')

    np.savez_compressed(
        os.path.join(OUT_DIR, 'plot_rayleigh0_covariance_extdiff.npz'),
        sigma_values=np.array(SIGMA_VALUES),
        cov_ext_ref=ext_cov,
        cov_ext_ref_sm=np.nan_to_num(ext_sm),
        diff_sm=np.stack([np.nan_to_num(ext_diff_sm[s]) for s in SIGMA_VALUES]),
        n_ext_ref_shots=ext_n,
        n_shots=N_SHOTS,
    )

    # 4) Radial-offset sweep.
    cov_sm_off = {}
    for r_off in R_OFFSETS:
        for s in SIGMA_VALUES:
            if r_off == 0:
                cov_sm_off[(r_off, s)] = cov_sm_by_sigma[s]
                continue
            H = azimuthal_prob_matrix(hits_by_sigma[s], en_by_sigma[s], r_off)
            cov = np.cov(H, rowvar=False)
            cov_sm_off[(r_off, s)] = nan_gaussian_filter(
                np.where(diag, np.nan, cov), SMOOTH_SIGMA_BINS)
            print(f'  r_off={r_off:+d} σ_θ={s:2d} done '
                  f'(|cov_sm|_max={np.nanmax(np.abs(cov_sm_off[(r_off, s)])):.2e})')

    v_max = max(float(np.nanmax(np.abs(cov_sm_off[k]))) for k in cov_sm_off)
    nrows = len(R_OFFSETS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    for r_idx, r_off in enumerate(R_OFFSETS):
        for c_idx, s in enumerate(SIGMA_VALUES):
            label = f'r_off = {r_off:+d} px  |  σ_θ = {s}°'
            draw(axes[r_idx, c_idx], cov_sm_off[(r_off, s)], label, v_max)
    fig.suptitle(
        f'Rayleigh=0 {N_SHOTS}-shot bootstrap azimuthal covariance across '
        'annulus radial offsets\n'
        f'(annulus = [re + r_off − {DR_HALF:g}, re + r_off + {DR_HALF:g}) px, '
        f'smoothed σ={SMOOTH_SIGMA_BINS:g} bins, mode=wrap)',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT_DIR, 'plot_rayleigh0_covariance_by_offset.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'wrote {p}')

    np.savez_compressed(
        os.path.join(OUT_DIR, 'plot_rayleigh0_covariance_by_offset.npz'),
        sigma_values=np.array(SIGMA_VALUES),
        r_offsets=np.array(R_OFFSETS),
        cov_sm=np.stack([
            np.stack([np.nan_to_num(cov_sm_off[(r, s)]) for s in SIGMA_VALUES])
            for r in R_OFFSETS
        ]),
        n_shots=N_SHOTS,
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
    )


if __name__ == '__main__':
    main()
