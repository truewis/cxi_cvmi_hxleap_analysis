#!/usr/bin/env python3
"""Azimuthal covariance across sigma_theta sweep.

Reads the SLURM-produced full hit stacks from
    wiggler_sigma_sweep_metrics/shot_hits_full_sigma_<slug>deg.npz
for slug in {0, 5, 10, 25, 45}, computes the per-shot azimuthal
intensity vector across all 1000 shots per sigma point, and renders:

  plot_sigma_sweep_covariance_allshots.png
      5-column figure. Column c corresponds to sigma_theta =
      SIGMA_VALUES[c].  Row 0: raw covariance (diagonal masked).
      Row 1: NaN-aware Gaussian smoothed (sigma = 1.5 bins).

  plot_sigma_sweep_covariance_diff.png
      Row 0: raw covariance for each sigma.
      Row 1: (this-sigma - sigma=0) smoothed, showing what streaking
             adds relative to the unstreaked baseline.

Requires conda env CXI.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
LEGACY_OUTPUTS = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
                  'batch_analysis_results/legacy_outputs')
SWEEP_DIR = os.path.join(LEGACY_OUTPUTS, 'wiggler_sigma_sweep_metrics')
OUT_DIR = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
           'batch_analysis_results/bootstrap_covariance_explanations')

SIGMA_VALUES = [0, 5, 10, 25, 45]           # deg
CX, CY = 67, 59
N_BINS = 36
DR_HALF = 3.0
RAW_CHANNELS_PER_BIN = 32
SPECTRUM_ROI_START = 13
SMOOTH_SIGMA_BINS = 1.5
GUIDES_DEG = (90.0, 270.0)


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


def draw(ax, M, title, vmax):
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-12
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
    cov_raw = {}
    cov_sm = {}
    n_shots_by_sigma = {}
    for s in SIGMA_VALUES:
        path = os.path.join(SWEEP_DIR, f'shot_hits_full_sigma_{s}deg.npz')
        d = np.load(path)
        hits = d['hits']
        en = d['mean_energy']
        print(f'  sigma={s} deg: {hits.shape[0]} shots')
        H = azimuthal_theta_matrix(hits, en)
        cov = np.cov(H, rowvar=False)
        cov_plot = np.where(diag, np.nan, cov)
        cov_raw[s] = cov_plot
        cov_sm[s] = nan_gaussian_filter(cov_plot, SMOOTH_SIGMA_BINS)
        n_shots_by_sigma[s] = hits.shape[0]

    v_raw = max(float(np.nanmax(np.abs(cov_raw[s]))) for s in SIGMA_VALUES)
    v_sm = max(float(np.nanmax(np.abs(cov_sm[s]))) for s in SIGMA_VALUES)

    ncols = len(SIGMA_VALUES)
    fig, axes = plt.subplots(2, ncols, figsize=(3.6 * ncols, 7.4))
    for c, s in enumerate(SIGMA_VALUES):
        draw(axes[0, c], cov_raw[s],
             f'σ_θ = {s}°  (raw)  |  N = {n_shots_by_sigma[s]}', v_raw)
        draw(axes[1, c], cov_sm[s],
             f'σ_θ = {s}°  (smoothed σ={SMOOTH_SIGMA_BINS:g} bins)', v_sm)
    fig.suptitle(
        'Azimuthal covariance across sigma_theta sweep '
        f'(Rayleigh scale = 5 px, all shots, no gate)',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, 'plot_sigma_sweep_covariance_allshots.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'wrote {out}')

    # Diff row vs sigma = 0
    diff_sm = {s: nan_gaussian_filter(cov_raw[s] - cov_raw[0],
                                      SMOOTH_SIGMA_BINS)
               for s in SIGMA_VALUES}
    v_diff = max(float(np.nanmax(np.abs(diff_sm[s]))) for s in SIGMA_VALUES)

    fig, axes = plt.subplots(2, ncols, figsize=(3.6 * ncols, 7.4))
    for c, s in enumerate(SIGMA_VALUES):
        draw(axes[0, c], cov_sm[s],
             f'σ_θ = {s}°  (smoothed)', v_sm)
        draw(axes[1, c], diff_sm[s],
             f'σ_θ = {s}° − σ_θ = 0°  (smoothed diff)', v_diff)
    fig.suptitle(
        'Streaked covariance vs unstreaked baseline '
        f'(Rayleigh scale = 5 px, all shots)',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, 'plot_sigma_sweep_covariance_diff.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'wrote {out}')

    cache = os.path.join(OUT_DIR, 'plot_sigma_sweep_covariance.npz')
    np.savez_compressed(
        cache,
        sigma_values=np.array(SIGMA_VALUES),
        cov_raw=np.stack([np.where(diag, 0.0, cov_raw[s])
                          for s in SIGMA_VALUES]),
        cov_sm=np.stack([np.nan_to_num(cov_sm[s])
                        for s in SIGMA_VALUES]),
        diff_sm=np.stack([np.nan_to_num(diff_sm[s])
                          for s in SIGMA_VALUES]),
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
    )
    print(f'wrote {cache}')


if __name__ == '__main__':
    main()
