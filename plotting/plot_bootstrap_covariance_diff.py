#!/usr/bin/env python3
"""Streaked − unstreaked azimuthal-covariance difference, smoothed.

Reads the two cached .npz files written by
explanation_bootstrap_covariance.py:

  explanation_bootstrap_covariance.npz            (Rayleigh scale = 5 px)
  explanation_bootstrap_covariance_unstreaked.npz (Rayleigh scale = 0)

and produces:

  explanation_bootstrap_covariance_diff.png
      Row 1 : streaked / unstreaked / (streaked − unstreaked) raw
              covariance matrices, diagonal masked.
      Row 2 : NaN-aware Gaussian smoothed (sigma = 1.5 bins ≈ 15°)
              versions of the same three matrices.

  explanation_bootstrap_covariance_diff.npz
      { 'diff': cov_streaked - cov_unstreaked,
        'diff_smoothed': smoothed diff,
        'cov_streaked', 'cov_unstreaked',
        'n_bins', 'smooth_sigma_bins' }

Requires conda env CXI.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
               'batch_analysis_results/bootstrap_covariance_explanations')
STREAKED_NPZ = os.path.join(RESULTS_DIR, 'explanation_bootstrap_covariance.npz')
UNSTREAKED_NPZ = os.path.join(RESULTS_DIR,
                              'explanation_bootstrap_covariance_unstreaked.npz')
OUT_PNG = os.path.join(RESULTS_DIR, 'explanation_bootstrap_covariance_diff.png')
OUT_NPZ = os.path.join(RESULTS_DIR, 'explanation_bootstrap_covariance_diff.npz')

N_BINS = 36
SMOOTH_SIGMA_BINS = 1.5           # ~15 deg at 10 deg per bin
GUIDES_DEG = (90.0, 270.0)


def nan_gaussian_filter(A, sigma):
    """Gaussian filter that treats NaN as missing (renormalize weights)."""
    A = np.asarray(A, dtype=float)
    nan_mask = np.isnan(A)
    V = np.where(nan_mask, 0.0, A)
    W = (~nan_mask).astype(float)
    Vs = gaussian_filter(V, sigma=sigma, mode='wrap')
    Ws = gaussian_filter(W, sigma=sigma, mode='wrap')
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(Ws > 1e-12, Vs / Ws, np.nan)


def load_cov(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    cov = np.asarray(d['cov'], dtype=float)
    diag = np.eye(cov.shape[0], dtype=bool)
    return np.where(diag, np.nan, cov)


def draw(ax, M, title, vmax=None):
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
    cov_streaked = load_cov(STREAKED_NPZ)
    cov_unstreak = load_cov(UNSTREAKED_NPZ)
    assert cov_streaked.shape == cov_unstreak.shape == (N_BINS, N_BINS)

    diff = cov_streaked - cov_unstreak

    cov_streaked_sm = nan_gaussian_filter(cov_streaked, SMOOTH_SIGMA_BINS)
    cov_unstreak_sm = nan_gaussian_filter(cov_unstreak, SMOOTH_SIGMA_BINS)
    diff_sm = nan_gaussian_filter(diff, SMOOTH_SIGMA_BINS)

    # Shared vmax across the two same-magnitude rows so the streaked and
    # unstreaked panels are directly comparable.
    v_raw = max(float(np.nanmax(np.abs(cov_streaked))),
                float(np.nanmax(np.abs(cov_unstreak))))
    v_sm = max(float(np.nanmax(np.abs(cov_streaked_sm))),
               float(np.nanmax(np.abs(cov_unstreak_sm))))
    v_diff_raw = float(np.nanmax(np.abs(diff)))
    v_diff_sm = float(np.nanmax(np.abs(diff_sm)))

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 11.2))
    draw(axes[0, 0], cov_streaked,
         'streaked  (Rayleigh scale = 5 px)', vmax=v_raw)
    draw(axes[0, 1], cov_unstreak,
         'unstreaked  (Rayleigh scale = 0)', vmax=v_raw)
    draw(axes[0, 2], diff,
         'streaked − unstreaked  (raw)', vmax=v_diff_raw)

    draw(axes[1, 0], cov_streaked_sm,
         f'streaked (smoothed, σ={SMOOTH_SIGMA_BINS:g} bins)', vmax=v_sm)
    draw(axes[1, 1], cov_unstreak_sm,
         f'unstreaked (smoothed, σ={SMOOTH_SIGMA_BINS:g} bins)', vmax=v_sm)
    draw(axes[1, 2], diff_sm,
         f'streaked − unstreaked (smoothed, σ={SMOOTH_SIGMA_BINS:g} bins)',
         vmax=v_diff_sm)

    fig.suptitle(
        'Bootstrap covariance: streaked vs. unstreaked (diagonal masked)',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f'wrote {OUT_PNG}')

    np.savez_compressed(
        OUT_NPZ,
        diff=diff,
        diff_smoothed=diff_sm,
        cov_streaked=cov_streaked,
        cov_unstreaked=cov_unstreak,
        n_bins=N_BINS,
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
    )
    print(f'wrote {OUT_NPZ}')


if __name__ == '__main__':
    main()
