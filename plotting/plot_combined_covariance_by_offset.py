#!/usr/bin/env python3
"""Combined 5000-shot azimuthal covariance across radial-offset annuli.

Same 5x1000-shot bootstrap sample as plot_combined_run1_4_covariance.py
(now really 5 runs), but sweeps the annulus radial offset. For each
offset r_off in {-6, -4, -2, 0} px, the annulus becomes
[re + r_off - DR_HALF, re + r_off + DR_HALF); at r_off = 0 this is the
default photoline annulus.

Layout: 4 rows (one per r_off) x 5 columns (one per sigma_theta).
        Row 0: r_off = -6 (inner background annulus)
        Row 1: r_off = -4
        Row 2: r_off = -2
        Row 3: r_off = 0   (photoline)

Cells: smoothed probability-normalized azimuthal covariance.

Outputs into
  batch_analysis_results/circular_wiggler_combined_5000/
    plot_combined_covariance_by_offset.png
    plot_combined_covariance_by_offset.npz

Conventions match plot_combined_run1_4_covariance.py:
  cx, cy = 67, 59; N_BINS = 36; DR_HALF = 3 px;
  probability-normalized shots; NaN-aware smoothing at sigma=1.5 bins
  with mode='wrap'; diagonal masked.

Requires conda env CXI.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

RUNS_ROOT = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
             'batch_analysis_results/legacy_outputs')
RUNS = ['circular_wiggler_run1', 'circular_wiggler_run2',
        'circular_wiggler_run3', 'circular_wiggler_run4',
        'circular_wiggler_run5']
SIGMA_VALUES = [0, 5, 10, 15, 20]
R_OFFSETS = [-6, -4, -2, 0]

OUT_DIR = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
           'batch_analysis_results/circular_wiggler_combined_5000')
OUT_PNG = os.path.join(OUT_DIR, 'plot_combined_covariance_by_offset.png')
OUT_NPZ = os.path.join(OUT_DIR, 'plot_combined_covariance_by_offset.npz')

CX, CY = 67, 59
N_BINS = 36
DR_HALF = 3.0
RAW_CHANNELS_PER_BIN = 32
SPECTRUM_ROI_START = 13
SMOOTH_SIGMA_BINS = 1.5
GUIDES_DEG = (90.0, 270.0)


def load_stack(run_name, sigma):
    p = os.path.join(RUNS_ROOT, run_name,
                     'wiggler_sigma_sweep_metrics',
                     f'shot_hits_full_sigma_{sigma}deg.npz')
    d = np.load(p)
    return d['hits'], d['mean_energy']


def azimuthal_prob_matrix(hits, mean_energy, r_offset):
    """Same as before but with an additive annulus offset r_offset."""
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
    diag = np.eye(N_BINS, dtype=bool)

    # Pre-load & concatenate per sigma once, then reuse for every r_offset.
    combined = {}
    for s in SIGMA_VALUES:
        hits_all, en_all = [], []
        for run in RUNS:
            hits, en = load_stack(run, s)
            hits_all.append(hits)
            en_all.append(en)
        combined[s] = (np.concatenate(hits_all, axis=0),
                       np.concatenate(en_all, axis=0))
        print(f'  loaded combined sigma={s} deg: '
              f'{combined[s][0].shape[0]} shots')

    cov_sm = {}   # (r_off, sigma) -> smoothed cov matrix
    for r_off in R_OFFSETS:
        for s in SIGMA_VALUES:
            hits, en = combined[s]
            H = azimuthal_prob_matrix(hits, en, r_off)
            cov = np.cov(H, rowvar=False)
            cov_p = np.where(diag, np.nan, cov)
            cov_sm[(r_off, s)] = nan_gaussian_filter(cov_p, SMOOTH_SIGMA_BINS)
            print(f'  r_off={r_off:+d}  sigma={s:2d}: done '
                  f'(|cov_sm|_max={np.nanmax(np.abs(cov_sm[(r_off, s)])):.2e})')

    v_max = max(float(np.nanmax(np.abs(cov_sm[k]))) for k in cov_sm)

    nrows = len(R_OFFSETS)
    ncols = len(SIGMA_VALUES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    for r_idx, r_off in enumerate(R_OFFSETS):
        for c_idx, s in enumerate(SIGMA_VALUES):
            label = f'r_off = {r_off:+d} px  |  σ_θ = {s}°'
            draw(axes[r_idx, c_idx], cov_sm[(r_off, s)], label, v_max)

    fig.suptitle(
        f'Combined {len(RUNS)}×1000 = {len(RUNS) * 1000}-shot bootstrap '
        'azimuthal covariance across annulus radial offsets\n'
        f'(annulus = [re + r_off − {DR_HALF:g}, re + r_off + {DR_HALF:g}) px, '
        f'smoothed σ={SMOOTH_SIGMA_BINS:g} bins, mode=wrap)',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f'wrote {OUT_PNG}')

    np.savez_compressed(
        OUT_NPZ,
        sigma_values=np.array(SIGMA_VALUES),
        r_offsets=np.array(R_OFFSETS),
        # Shape (n_r, n_sigma, N_BINS, N_BINS), NaN-diagonal replaced by 0
        cov_sm=np.stack([
            np.stack([np.nan_to_num(cov_sm[(r_off, s)]) for s in SIGMA_VALUES])
            for r_off in R_OFFSETS
        ]),
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
        n_shots_per_condition=len(RUNS) * 1000,
    )
    print(f'wrote {OUT_NPZ}')


if __name__ == '__main__':
    main()
