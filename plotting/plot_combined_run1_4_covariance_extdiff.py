#!/usr/bin/env python3
"""Diff the combined 4000-shot covariance against an EXTERNAL Rayleigh=0
unstreaked reference stored in
  batch_analysis_results/bootstrap_covariance_explanations/
    plot_sigma_sweep_covariance_prob.npz

That reference (`cov_unstreaked`) is the 1000-shot azimuthal covariance
of the pristine Rayleigh=0 lobe, generated inline by
`plot_sigma_sweep_covariance_prob.py`. Using it as the reference (vs.
the internal σ_θ=0 column of the combined sweep) has two advantages:
  * The external reference has Rayleigh scale exactly = 0 (delta at cx,
    cy), so it's the "true" unstreaked distribution.
  * The internal σ_θ=0 column still has Rayleigh scale = 5 px, which
    itself contributes a small streak signal; subtracting the external
    Rayleigh=0 pulls out the pure Rayleigh-radius contribution.

Reads:
  batch_analysis_results/circular_wiggler_combined_4000/
    plot_combined_run1_4_covariance.npz          (cov_raw per sigma)
  batch_analysis_results/bootstrap_covariance_explanations/
    plot_sigma_sweep_covariance_prob.npz         (cov_unstreaked)

Writes:
  batch_analysis_results/circular_wiggler_combined_4000/
    plot_combined_run1_4_covariance_extdiff.png
    plot_combined_run1_4_covariance_extdiff.npz

Requires conda env CXI.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

RESULTS = '/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/batch_analysis_results'
COMBINED_NPZ = os.path.join(RESULTS, 'circular_wiggler_combined_5000',
                            'plot_combined_covariance.npz')
EXTREF_NPZ = os.path.join(RESULTS, 'bootstrap_covariance_explanations',
                          'plot_sigma_sweep_covariance_prob.npz')

OUT_DIR = os.path.join(RESULTS, 'circular_wiggler_combined_5000')
OUT_PNG = os.path.join(OUT_DIR, 'plot_combined_covariance_extdiff.png')
OUT_NPZ = os.path.join(OUT_DIR, 'plot_combined_covariance_extdiff.npz')

N_BINS = 36
SMOOTH_SIGMA_BINS = 1.5
GUIDES_DEG = (90.0, 270.0)


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
    combined = np.load(COMBINED_NPZ)
    extref = np.load(EXTREF_NPZ)

    sigmas = list(combined['sigma_values'].tolist())
    cov_raw = combined['cov_raw']                     # (n_sig, N_BINS, N_BINS)
    n_shots_per_sigma = combined['n_shots_per_sigma'] # (n_sig,)

    ext_cov = extref['cov_unstreaked']                # (N_BINS, N_BINS)
    ext_n = int(extref['n_unstreaked_shots'])

    diag = np.eye(N_BINS, dtype=bool)
    diff_raw = {}
    diff_sm = {}
    cov_combined_sm = {}
    ext_sm = nan_gaussian_filter(np.where(diag, np.nan, ext_cov),
                                 SMOOTH_SIGMA_BINS)
    for i, s in enumerate(sigmas):
        c = cov_raw[i]
        cov_combined_sm[s] = nan_gaussian_filter(
            np.where(diag, np.nan, c), SMOOTH_SIGMA_BINS)
        diff = np.where(diag, np.nan, c - ext_cov)
        diff_raw[s] = diff
        diff_sm[s] = nan_gaussian_filter(diff, SMOOTH_SIGMA_BINS)

    ncols = len(sigmas)
    v_cov = max(
        [float(np.nanmax(np.abs(ext_sm)))]
        + [float(np.nanmax(np.abs(cov_combined_sm[s]))) for s in sigmas]
    )
    v_diff = max(float(np.nanmax(np.abs(diff_sm[s]))) for s in sigmas) or 1e-15

    # Layout: 3 rows x ncols. Row 0 = combined smoothed. Row 1 = external
    # reference (only col 0, rest blank). Row 2 = combined − external diff.
    fig, axes = plt.subplots(3, ncols, figsize=(3.5 * ncols, 10.6))
    for c, s in enumerate(sigmas):
        draw(axes[0, c], cov_combined_sm[s],
             f'combined σ_θ = {s}°  (smoothed)\nN = {n_shots_per_sigma[c]}',
             v_cov)
        if c == 0:
            draw(axes[1, c], ext_sm,
                 f'external ref (Rayleigh=0)\nN = {ext_n}', v_cov)
        else:
            axes[1, c].axis('off')
            axes[1, c].set_title(
                '(ext ref shown in col 0)', fontsize=9, y=0.5)
        draw(axes[2, c], diff_sm[s],
             f'combined σ_θ = {s}° − ext ref\n(smoothed diff)',
             v_diff)

    fig.suptitle(
        'Combined 5000-shot bootstrap covariance vs external '
        'Rayleigh=0 reference  '
        f'(probability-normalized, smoothed σ={SMOOTH_SIGMA_BINS:g} bins, '
        f'mode=wrap)',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f'wrote {OUT_PNG}')

    np.savez_compressed(
        OUT_NPZ,
        sigma_values=np.array(sigmas),
        cov_combined_sm=np.stack([np.nan_to_num(cov_combined_sm[s]) for s in sigmas]),
        cov_ext_ref=ext_cov,
        cov_ext_ref_sm=np.nan_to_num(ext_sm),
        diff_raw=np.stack([np.nan_to_num(diff_raw[s]) for s in sigmas]),
        diff_sm=np.stack([np.nan_to_num(diff_sm[s]) for s in sigmas]),
        n_shots_per_sigma=n_shots_per_sigma,
        n_ext_ref_shots=ext_n,
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
    )
    print(f'wrote {OUT_NPZ}')


if __name__ == '__main__':
    main()
