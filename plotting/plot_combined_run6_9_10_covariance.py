#!/usr/bin/env python3
"""Combined 60000-shot azimuthal covariance from
batch_analysis_results/legacy_outputs/circular_wiggler_run{6,9,10}/
wiggler_sigma_sweep_metrics/.

Each of the 3 SLURM runs saved shot_hits_full_sigma_<slug>deg.npz with
20 000 shots per sigma_theta. Concatenating gives a 60 000-shot sample
per sigma, then computes the same probability-normalized azimuthal
covariance as plot_combined_run1_4_covariance.py.

Produces three plots into
  batch_analysis_results/circular_wiggler_combined_60000/

  1. plot_combined_covariance.png            (5-col x 2-row)
     Row 0: smoothed cov per sigma.
     Row 1: (sigma − sigma=0) smoothed diff.
  2. plot_combined_covariance_extdiff.png    (3-row x 5-col)
     Row 0: 60k-shot smoothed cov per sigma.
     Row 1: external 1000-shot Rayleigh=0 reference (col 0 only).
     Row 2: (60k − external ref) smoothed diff.
  3. plot_combined_covariance_by_offset.png  (4-row x 5-col)
     Rows: r_off ∈ {-6, -4, -2, 0} px annulus offsets.
     Cols: σ_θ ∈ {0, 5, 10, 15, 20}° sigma_theta.

Memory strategy: concatenate all 3 stacks per sigma (~5 GB) but drop
after each sigma point so peak RSS stays bounded.

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
RUNS = ['circular_wiggler_run6', 'circular_wiggler_run9',
        'circular_wiggler_run10']
SIGMA_VALUES = [0, 5, 10, 15, 20]
R_OFFSETS = [-6, -4, -2, 0]

OUT_DIR = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
           'batch_analysis_results/circular_wiggler_combined_60000')

EXTREF_NPZ = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
              'batch_analysis_results/bootstrap_covariance_explanations/'
              'plot_sigma_sweep_covariance_prob.npz')

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


def load_combined(sigma):
    hits_all, en_all = [], []
    for run in RUNS:
        hits, en = load_stack(run, sigma)
        hits_all.append(hits)
        en_all.append(en)
        print(f'  loaded {run} sigma={sigma}deg: {hits.shape[0]} shots')
    return (np.concatenate(hits_all, axis=0),
            np.concatenate(en_all, axis=0))


def azimuthal_prob_matrix(hits, mean_energy, r_offset=0.0):
    """Per-shot azimuthal probability vector in the annulus
    [re + r_offset - DR_HALF, re + r_offset + DR_HALF)."""
    ny, nx = hits.shape[1:]
    y, x = np.mgrid[0:ny, 0:nx]
    r_map = np.hypot(x - CX, y - CY)
    theta = np.arctan2(y - CY, x - CX) + np.pi
    theta_bin_map = np.clip(
        np.floor(theta / (2.0 * np.pi / N_BINS)).astype(int),
        0, N_BINS - 1,
    )
    n = hits.shape[0]
    rows = np.empty((n, N_BINS), dtype=float)
    for i in range(n):
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

    # (sigma, r_off) -> raw cov ; keep only the cov matrices, drop hits.
    cov = {}
    n_shots_by_sigma = {}
    for s in SIGMA_VALUES:
        print(f'--- combining sigma={s}° ---')
        hits, en = load_combined(s)
        n_shots_by_sigma[s] = hits.shape[0]
        print(f'  combined sigma={s}° -> {hits.shape[0]} shots')
        for r_off in R_OFFSETS:
            H = azimuthal_prob_matrix(hits, en, r_off)
            c = np.cov(H, rowvar=False)
            cov[(r_off, s)] = c
            print(f'    r_off={r_off:+d} done')
        del hits, en

    # Smoothed variants
    cov_sm = {}
    for k, c in cov.items():
        cov_sm[k] = nan_gaussian_filter(np.where(diag, np.nan, c),
                                         SMOOTH_SIGMA_BINS)

    # ------ Plot 1: main sigma sweep (r_off = 0) ------
    diff_sm = {}
    ref = cov[(0, 0)]
    for s in SIGMA_VALUES:
        diff_sm[s] = nan_gaussian_filter(
            np.where(diag, np.nan, cov[(0, s)] - ref),
            SMOOTH_SIGMA_BINS)

    ncols = len(SIGMA_VALUES)
    v_cov = max(float(np.nanmax(np.abs(cov_sm[(0, s)]))) for s in SIGMA_VALUES)
    v_diff = (max(float(np.nanmax(np.abs(diff_sm[s])))
                  for s in SIGMA_VALUES if s > 0) or 1e-15)

    fig, axes = plt.subplots(2, ncols, figsize=(3.5 * ncols, 7.4))
    for c_idx, s in enumerate(SIGMA_VALUES):
        draw(axes[0, c_idx], cov_sm[(0, s)],
             f'σ_θ = {s}°  (smoothed)\nN = {n_shots_by_sigma[s]}', v_cov)
        if s == 0:
            axes[1, c_idx].axis('off')
            axes[1, c_idx].set_title('(reference — no diff)', fontsize=9, y=0.5)
        else:
            draw(axes[1, c_idx], diff_sm[s],
                 f'σ_θ = {s}° − σ_θ = 0°  (smoothed diff)', v_diff)
    fig.suptitle(
        f'Combined {len(RUNS)}×20 000 = {len(RUNS) * 20000}-shot bootstrap '
        'covariance (run6+9+10, probability-normalized, all shots, no gate; '
        f'smoothed σ={SMOOTH_SIGMA_BINS:g} bins, mode=wrap)',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT_DIR, 'plot_combined_covariance.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'wrote {p}')

    np.savez_compressed(
        os.path.join(OUT_DIR, 'plot_combined_covariance.npz'),
        sigma_values=np.array(SIGMA_VALUES),
        cov_raw=np.stack([cov[(0, s)] for s in SIGMA_VALUES]),
        cov_sm=np.stack([np.nan_to_num(cov_sm[(0, s)]) for s in SIGMA_VALUES]),
        diff_sm=np.stack([np.nan_to_num(diff_sm[s]) for s in SIGMA_VALUES]),
        n_shots_per_sigma=np.array([n_shots_by_sigma[s] for s in SIGMA_VALUES]),
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
    )

    # ------ Plot 2: external-ref diff (r_off = 0) ------
    extref = np.load(EXTREF_NPZ)
    ext_cov = extref['cov_unstreaked']
    ext_n = int(extref['n_unstreaked_shots'])
    ext_sm = nan_gaussian_filter(np.where(diag, np.nan, ext_cov),
                                 SMOOTH_SIGMA_BINS)
    ext_diff_sm = {}
    for s in SIGMA_VALUES:
        ext_diff_sm[s] = nan_gaussian_filter(
            np.where(diag, np.nan, cov[(0, s)] - ext_cov),
            SMOOTH_SIGMA_BINS)
    v_ext_cov = max(v_cov, float(np.nanmax(np.abs(ext_sm))))
    v_ext_diff = (max(float(np.nanmax(np.abs(ext_diff_sm[s])))
                      for s in SIGMA_VALUES) or 1e-15)

    fig, axes = plt.subplots(3, ncols, figsize=(3.5 * ncols, 10.6))
    for c_idx, s in enumerate(SIGMA_VALUES):
        draw(axes[0, c_idx], cov_sm[(0, s)],
             f'combined σ_θ = {s}°  (smoothed)\nN = {n_shots_by_sigma[s]}',
             v_ext_cov)
        if c_idx == 0:
            draw(axes[1, c_idx], ext_sm,
                 f'external ref (Rayleigh=0)\nN = {ext_n}', v_ext_cov)
        else:
            axes[1, c_idx].axis('off')
            axes[1, c_idx].set_title('(ext ref shown in col 0)', fontsize=9, y=0.5)
        draw(axes[2, c_idx], ext_diff_sm[s],
             f'combined σ_θ = {s}° − ext ref\n(smoothed diff)', v_ext_diff)
    fig.suptitle(
        f'Combined {len(RUNS) * 20000}-shot bootstrap covariance vs external '
        'Rayleigh=0 reference '
        f'(smoothed σ={SMOOTH_SIGMA_BINS:g} bins, mode=wrap)',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    p = os.path.join(OUT_DIR, 'plot_combined_covariance_extdiff.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'wrote {p}')

    np.savez_compressed(
        os.path.join(OUT_DIR, 'plot_combined_covariance_extdiff.npz'),
        sigma_values=np.array(SIGMA_VALUES),
        cov_combined_sm=np.stack([np.nan_to_num(cov_sm[(0, s)]) for s in SIGMA_VALUES]),
        cov_ext_ref=ext_cov,
        cov_ext_ref_sm=np.nan_to_num(ext_sm),
        diff_sm=np.stack([np.nan_to_num(ext_diff_sm[s]) for s in SIGMA_VALUES]),
        n_shots_per_sigma=np.array([n_shots_by_sigma[s] for s in SIGMA_VALUES]),
        n_ext_ref_shots=ext_n,
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
    )

    # ------ Plot 3: radial-offset sweep ------
    v_max = max(float(np.nanmax(np.abs(cov_sm[k]))) for k in cov_sm)
    nrows = len(R_OFFSETS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    for r_idx, r_off in enumerate(R_OFFSETS):
        for c_idx, s in enumerate(SIGMA_VALUES):
            label = f'r_off = {r_off:+d} px  |  σ_θ = {s}°'
            draw(axes[r_idx, c_idx], cov_sm[(r_off, s)], label, v_max)
    fig.suptitle(
        f'Combined {len(RUNS)}×20 000 = {len(RUNS) * 20000}-shot bootstrap '
        'azimuthal covariance across annulus radial offsets\n'
        f'(annulus = [re + r_off − {DR_HALF:g}, re + r_off + {DR_HALF:g}) px, '
        f'smoothed σ={SMOOTH_SIGMA_BINS:g} bins, mode=wrap)',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT_DIR, 'plot_combined_covariance_by_offset.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'wrote {p}')

    np.savez_compressed(
        os.path.join(OUT_DIR, 'plot_combined_covariance_by_offset.npz'),
        sigma_values=np.array(SIGMA_VALUES),
        r_offsets=np.array(R_OFFSETS),
        cov_sm=np.stack([
            np.stack([np.nan_to_num(cov_sm[(r, s)]) for s in SIGMA_VALUES])
            for r in R_OFFSETS
        ]),
        n_shots_per_sigma=np.array([n_shots_by_sigma[s] for s in SIGMA_VALUES]),
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
    )


if __name__ == '__main__':
    main()
