#!/usr/bin/env python3
"""
For every `azimuthal_covariance_*.npy` (self OR cross) under batch_analysis/,
generate two companion figures next to it:

  1. `<basename>_smoothed.png`
     - NaN-aware Gaussian smoothing at sigma=3 (the diagonal cells masked out
       in the plotted self-covariance version are excluded from the smoother
       so they do not bleed into off-diagonal cells).
     - Dashed guide lines at θ = 90° and θ = 270° on both axes.
     - Max-contrast diverging colormap: symmetric ±p98(|smoothed|) around zero
       so the middle 96% of the value range fills the palette.

  2. `<basename>_projections.png`
     - Four 1-D slices through the smoothed matrix at x = 90°, x = 270°,
       y = 90°, y = 270°.  Each panel plots covariance and correlation
       overlays where both are stored (self: full column/row; cross: same).
     - Vertical guides at 90° and 270° on each projection axis.

Requires conda env CXI.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


BATCH_ANALYSIS_DIR = '/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/batch_analysis_results/legacy_outputs'
SMOOTH_SIGMA = 3.0
CONTRAST_PERCENTILE = 98.0
GUIDE_DEG = (90.0, 270.0)


def nan_gaussian_filter(A, sigma):
    """Gaussian filter that treats NaN as missing: the kernel weight is
    renormalized by the local sum of valid weights, so masked-out cells (e.g.
    the diagonal of a self-covariance matrix) do not leak into neighbors."""
    A = np.asarray(A, dtype=float)
    nan_mask = np.isnan(A)
    V = np.where(nan_mask, 0.0, A)
    W = (~nan_mask).astype(float)
    Vs = gaussian_filter(V, sigma=sigma, mode='nearest')
    Ws = gaussian_filter(W, sigma=sigma, mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(Ws > 1e-12, Vs / Ws, np.nan)
    return out


def _symmetric_limits(arr, pct=CONTRAST_PERCENTILE):
    """Symmetric vmin/vmax at ±(pct-percentile of |arr|)."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1e-12, 1e-12
    a = float(np.percentile(np.abs(finite), pct))
    if a == 0:
        a = float(np.max(np.abs(finite))) or 1e-12
    return -a, a


def _load_matrix(npy_path):
    """Return (cov_plot, corr_plot, meta) where the _plot arrays have the
    diagonal masked for the self-covariance case. Cross-covariance keeps
    both matrices intact."""
    d = np.load(npy_path, allow_pickle=True).item()
    is_cross = 'cross_cov' in d or 'Hx' in d
    n_bins = int(d.get('n_bins', 90))

    if is_cross:
        cov = np.asarray(d['cross_cov'], dtype=float)
        corr = np.asarray(d['cross_corr'], dtype=float)
        n_shots = np.asarray(d.get('Hx', np.empty((0,)))).shape[0]
        r_x = d.get('r_offset_x', 0.0)
        r_y = d.get('r_offset_y', 0.0)
        cov_plot, corr_plot = cov, corr
    else:
        cov = np.asarray(d['cov'], dtype=float)
        corr = np.asarray(d['corr'], dtype=float)
        n_shots = np.asarray(d.get('H', np.empty((0,)))).shape[0]
        r_x = r_y = d.get('r_offset', 0.0)
        diag = np.eye(n_bins, dtype=bool)
        cov_plot = np.where(diag, np.nan, cov)
        corr_plot = np.where(diag, np.nan, corr)

    meta = {
        'is_cross': is_cross,
        'n_bins': n_bins,
        'n_shots': int(n_shots),
        'preset_name': d.get('preset_name', d.get('roi', '?')),
        'kind': d.get('kind', '?'),
        'r_off_x': float(r_x),
        'r_off_y': float(r_y),
        'dr_half': float(d.get('dr_half', 3.0)),
        'cx': float(d.get('cx', 67.0)),
        'cy': float(d.get('cy', 59.0)),
    }
    return cov_plot, corr_plot, meta


def _add_guides(ax):
    for d in GUIDE_DEG:
        ax.axvline(d, color='k', ls='--', lw=0.9, alpha=0.55)
        ax.axhline(d, color='k', ls='--', lw=0.9, alpha=0.55)


def render_smoothed(npy_path):
    cov_plot, corr_plot, meta = _load_matrix(npy_path)
    cov_sm = nan_gaussian_filter(cov_plot, SMOOTH_SIGMA)
    corr_sm = nan_gaussian_filter(corr_plot, SMOOTH_SIGMA)

    fig, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(15, 6.5))
    extent = [0.0, 360.0, 0.0, 360.0]
    for ax, arr, title in (
        (ax_c, cov_sm, 'covariance'),
        (ax_r, corr_sm, 'correlation'),
    ):
        vmin, vmax = _symmetric_limits(arr)
        im = ax.imshow(arr, cmap='RdBu_r', origin='lower', extent=extent,
                       vmin=vmin, vmax=vmax, aspect='equal',
                       interpolation='nearest')
        _add_guides(ax)
        if meta['is_cross']:
            ax.set_xlabel(r'$\theta_j$ [deg]  (X-annulus)')
            ax.set_ylabel(r'$\theta_i$ [deg]  (Y-annulus)')
        else:
            ax.set_xlabel(r'$\theta_j$ [deg]')
            ax.set_ylabel(r'$\theta_i$ [deg]')
        ax.set_title(f'{title} (Gaussian σ={SMOOTH_SIGMA} px, max-contrast p{CONTRAST_PERCENTILE:g})')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    tag = ('cross-annulus  '
           f'X: re{meta["r_off_x"]:+g}, Y: re{meta["r_off_y"]:+g}'
           if meta['is_cross']
           else f'annulus = re{meta["r_off_x"]:+g}')
    fig.suptitle(
        f'{meta["preset_name"]}  (kind={meta["kind"]}, {meta["n_shots"]} shots)  |  '
        f'{tag} ± {meta["dr_half"]:g} px  |  '
        f'center=({meta["cx"]:g}, {meta["cy"]:g})',
        fontweight='bold',
    )
    fig.tight_layout()
    out_png = npy_path.replace('.npy', '_smoothed.png')
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png, cov_sm, corr_sm, meta


def render_projections(npy_path, cov_sm, corr_sm, meta):
    """Four projection panels through the smoothed matrix:
       (0,0) horizontal cut at y=90°   → cov & corr vs theta_j (x-axis)
       (0,1) horizontal cut at y=270°  →       "
       (1,0) vertical cut at x=90°     → cov & corr vs theta_i (y-axis)
       (1,1) vertical cut at x=270°    →       "
    """
    n_bins = meta['n_bins']
    theta_edges = np.linspace(0.0, 360.0, n_bins + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    step = 360.0 / n_bins

    def _bin_of(deg):
        return int(np.clip(deg / step, 0, n_bins - 1))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for panel_idx, (cut_type, cut_deg) in enumerate((
        ('horizontal', 90.0),   # slice along row where y=90 -> cov_sm[bin_of(90), :]
        ('horizontal', 270.0),
        ('vertical', 90.0),
        ('vertical', 270.0),
    )):
        ax = axes[panel_idx // 2, panel_idx % 2]
        b = _bin_of(cut_deg)
        if cut_type == 'horizontal':
            cov_slice = cov_sm[b, :]
            corr_slice = corr_sm[b, :]
            x_axis = theta_centers
            xlabel = r'$\theta_j$ [deg]  (x-axis of matrix)'
            title_cut = fr'$\theta_i = {cut_deg:g}^\circ$  (row of matrix)'
        else:
            cov_slice = cov_sm[:, b]
            corr_slice = corr_sm[:, b]
            x_axis = theta_centers
            xlabel = r'$\theta_i$ [deg]  (y-axis of matrix)'
            title_cut = fr'$\theta_j = {cut_deg:g}^\circ$  (column of matrix)'

        # Plot covariance on the left axis, correlation on a twin axis.
        color_cov = 'tab:blue'
        color_corr = 'tab:orange'
        ax.plot(x_axis, cov_slice, color=color_cov, lw=1.4, label='covariance')
        ax.axhline(0.0, color='0.4', lw=0.6, alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('covariance', color=color_cov)
        ax.tick_params(axis='y', labelcolor=color_cov)

        ax_r = ax.twinx()
        ax_r.plot(x_axis, corr_slice, color=color_corr, lw=1.2, alpha=0.85,
                  label='correlation')
        ax_r.set_ylabel('correlation', color=color_corr)
        ax_r.tick_params(axis='y', labelcolor=color_corr)

        for d in GUIDE_DEG:
            ax.axvline(d, color='k', ls='--', lw=0.9, alpha=0.55)

        ax.set_title(title_cut, fontsize=11)
        ax.set_xlim(0, 360)
        ax.grid(True, alpha=0.25, linestyle=':')

    tag = ('cross-annulus  '
           f'X: re{meta["r_off_x"]:+g}, Y: re{meta["r_off_y"]:+g}'
           if meta['is_cross']
           else f'annulus = re{meta["r_off_x"]:+g}')
    fig.suptitle(
        f'{meta["preset_name"]}  (kind={meta["kind"]}, {meta["n_shots"]} shots)  |  '
        f'{tag} ± {meta["dr_half"]:g} px  |  '
        f'projections at θ=90°/270° through smoothed matrix (σ={SMOOTH_SIGMA} px)',
        fontweight='bold',
    )
    fig.tight_layout()
    out_png = npy_path.replace('.npy', '_projections.png')
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def walk_npys(root):
    """Yield every azimuthal_covariance_*.npy under `root`."""
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.startswith('azimuthal_covariance') and fn.endswith('.npy'):
                yield os.path.join(dirpath, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=BATCH_ANALYSIS_DIR,
                    help='directory to walk for azimuthal_covariance_*.npy')
    ap.add_argument('--only', default=None,
                    help='substring; only process npy files whose path contains this')
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    print(f'walking {root}')
    n = 0
    for npy_path in walk_npys(root):
        if args.only and args.only not in npy_path:
            continue
        try:
            out_sm, cov_sm, corr_sm, meta = render_smoothed(npy_path)
            out_pr = render_projections(npy_path, cov_sm, corr_sm, meta)
            n += 1
            if n <= 20 or n % 25 == 0:
                print(f'  [{n}] {os.path.relpath(npy_path, root)}')
        except Exception as exc:
            print(f'  [error] {npy_path}: {exc}')
    print(f'processed {n} npy files')


if __name__ == '__main__':
    main()
