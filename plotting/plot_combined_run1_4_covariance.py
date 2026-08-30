#!/usr/bin/env python3
"""Combined 4000-shot azimuthal covariance from
batch_analysis_results/legacy_outputs/circular_wiggler_run{1..4}/wiggler_sigma_sweep_metrics/.

Each of the 4 SLURM runs saved shot_hits_full_sigma_<slug>deg.npz with
1000 shots. This script concatenates the four stacks per sigma to give
a 4000-shot sample, then computes the same probability-normalized
azimuthal covariance as plot_sigma_sweep_covariance_prob.py.

Outputs
-------
  batch_analysis_results/circular_wiggler_combined_4000/
    plot_combined_run1_4_covariance.png
        6-column x 2-row grid.
        Row 0 : smoothed covariance for sigma_theta in {0,5,10,15,20} deg
                + column 5 for the unstreaked 4000-shot sample (col 0
                doubles as reference; extra col intentionally left as
                a placeholder is not added — sigma=0 IS the unstreaked case).
        Row 1 : (sigma>0 minus sigma=0) smoothed diff per sigma.
    plot_combined_run1_4_covariance.npz
        stacked cov / cov_sm / diff_sm keyed by sigma.
    NOTES.md
        provenance + attribution.

Convention (same as plot_sigma_sweep_covariance_prob.py):
  cx, cy = 67, 59 on 140x140 grid
  N_BINS = 36 angular bins (10 deg each), guides at 90 and 270 deg
  annulus [re - DR_HALF, re + DR_HALF) with DR_HALF = 3 px
  probability normalization: hits[i] / hits[i].sum() per shot
  NaN-aware Gaussian smoothing (mode='wrap') at sigma = 1.5 bins
  diagonal of self-covariance masked

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

RUNS_ROOT = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
             'batch_analysis_results/legacy_outputs')
RUNS = ['circular_wiggler_run1', 'circular_wiggler_run2',
        'circular_wiggler_run3', 'circular_wiggler_run4',
        'circular_wiggler_run5']
SIGMA_VALUES = [0, 5, 10, 15, 20]           # deg — the intersection across all runs

OUT_DIR = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
           'batch_analysis_results/circular_wiggler_combined_5000')
OUT_PNG = os.path.join(OUT_DIR, 'plot_combined_covariance.png')
OUT_NPZ = os.path.join(OUT_DIR, 'plot_combined_covariance.npz')

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
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    d = np.load(p)
    return d['hits'], d['mean_energy']


def azimuthal_prob_matrix(hits, mean_energy):
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
    os.makedirs(OUT_DIR, exist_ok=True)
    diag = np.eye(N_BINS, dtype=bool)

    cov_by_sigma = {}
    cov_sm_by_sigma = {}
    n_shots_by_sigma = {}

    for s in SIGMA_VALUES:
        hits_all = []
        en_all = []
        for run in RUNS:
            hits, en = load_stack(run, s)
            hits_all.append(hits)
            en_all.append(en)
            print(f'  loaded {run} sigma={s} deg: {hits.shape[0]} shots')
        hits_cat = np.concatenate(hits_all, axis=0)
        en_cat = np.concatenate(en_all, axis=0)
        n_shots_by_sigma[s] = hits_cat.shape[0]
        print(f'  combined sigma={s} deg: {hits_cat.shape[0]} shots')
        H = azimuthal_prob_matrix(hits_cat, en_cat)
        cov = np.cov(H, rowvar=False)
        cov_p = np.where(diag, np.nan, cov)
        cov_by_sigma[s] = cov
        cov_sm_by_sigma[s] = nan_gaussian_filter(cov_p, SMOOTH_SIGMA_BINS)

    # Diff row: (sigma - sigma=0) smoothed
    diff_sm_by_sigma = {
        s: nan_gaussian_filter(np.where(diag, np.nan,
                                         cov_by_sigma[s] - cov_by_sigma[0]),
                                SMOOTH_SIGMA_BINS)
        for s in SIGMA_VALUES
    }

    ncols = len(SIGMA_VALUES)
    v_cov = max(float(np.nanmax(np.abs(cov_sm_by_sigma[s])))
                for s in SIGMA_VALUES)
    v_diff = max(float(np.nanmax(np.abs(diff_sm_by_sigma[s])))
                 for s in SIGMA_VALUES if s > 0) or 1e-15

    fig, axes = plt.subplots(2, ncols, figsize=(3.5 * ncols, 7.4))
    for c, s in enumerate(SIGMA_VALUES):
        draw(axes[0, c], cov_sm_by_sigma[s],
             f'σ_θ = {s}°  (smoothed)\nN = {n_shots_by_sigma[s]} shots',
             v_cov)
        if s == 0:
            axes[1, c].axis('off')
            axes[1, c].set_title(
                '(reference — no diff)', fontsize=9, y=0.5)
        else:
            draw(axes[1, c], diff_sm_by_sigma[s],
                 f'σ_θ = {s}° − σ_θ = 0°  (smoothed diff)', v_diff)

    fig.suptitle(
        f'Combined {len(RUNS)}×1000 = {len(RUNS) * 1000}-shot bootstrap covariance '
        f'(probability-normalized, all shots, no gate; smoothed '
        f'σ={SMOOTH_SIGMA_BINS:g} bins, mode=wrap)',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f'wrote {OUT_PNG}')

    np.savez_compressed(
        OUT_NPZ,
        sigma_values=np.array(SIGMA_VALUES),
        cov_raw=np.stack([cov_by_sigma[s] for s in SIGMA_VALUES]),
        cov_sm=np.stack([np.nan_to_num(cov_sm_by_sigma[s]) for s in SIGMA_VALUES]),
        diff_sm=np.stack([np.nan_to_num(diff_sm_by_sigma[s]) for s in SIGMA_VALUES]),
        n_shots_per_sigma=np.array([n_shots_by_sigma[s] for s in SIGMA_VALUES]),
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
    )
    print(f'wrote {OUT_NPZ}')


if __name__ == '__main__':
    main()
