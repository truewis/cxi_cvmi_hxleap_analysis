#!/usr/bin/env python3
"""Real-data probability-normalized azimuthal covariance.

Unstreaked pool: ALL goose shots that pass (goose mask) ∩ (is_gaussian)
                 across every step of every run present on disk. This is
                 the un-streaked reference.

Streaked pool  : ALL cotimed shots — (run, step) pairs from
                 aggregate_4roi_late_steps.PRESETS['cotimed'], kind='duck',
                 passing (duck mask) ∩ (is_gaussian) ∩ (step_axis == step_val).

Per shot: divide `hits[i]` by its own total intensity to get a per-pixel
probability, then sum the annulus [re-DR_HALF, re+DR_HALF) into 36
angular bins about (cx, cy) = (67, 59). re derived per shot from
`mean_energy` via the standard `(E / RAW_CHANNELS_PER_BIN
- SPECTRUM_ROI_START) * 0.6 + 29.4` formula.

Outputs land under
  batch_analysis_results/real_covariance_goose_vs_cotimed/
    plot_real_covariance_goose_vs_cotimed.png    1x3 smoothed cov+diff
    plot_real_covariance_goose_vs_cotimed.npz    cached matrices
    NOTES.md                                     provenance / conventions

Requires conda env CXI.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
# Add batch_analysis/ to path so sibling imports work from plotting/.
sys.path.insert(0, os.path.dirname(_HERE))
from aggregate_4roi_late_steps import PRESETS  # noqa: E402

PICKLE_DIR = '/sdf/scratch/users/j/jinseop'
OUT_DIR = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
           'batch_analysis_results/real_covariance_goose_vs_cotimed')
OUT_PNG = os.path.join(OUT_DIR, 'plot_real_covariance_goose_vs_cotimed.png')
OUT_NPZ = os.path.join(OUT_DIR, 'plot_real_covariance_goose_vs_cotimed.npz')

DEFAULT_CX, DEFAULT_CY = 67, 59   # cvmi default (unshifted)
CX, CY = DEFAULT_CX, DEFAULT_CY   # overwritten by CLI in main()
N_BINS = 36
DR_HALF = 3.0
RAW_CHANNELS_PER_BIN = 32
SPECTRUM_ROI_START = 13
SMOOTH_SIGMA_BINS = 1.5
GUIDES_DEG = (90.0, 270.0)
# When True, roll the covariance matrix by N_BINS/2 in each axis and
# display with extent [-180, 180] so that theta = 180 deg lands at the
# center of the plot. Toggled by --center-180.
CENTER_180 = False

# Runs to walk for the goose pool: everything the pipeline routinely
# aggregates (matches _enumerate_pairs('goose') in aggregate_4roi_late_steps).
GOOSE_RUNS = [145, 146, 147, 148, 149, 150, 152]

# Cotimed pairs from the module-level preset (5 pairs across runs 145–150).
COTIMED_PAIRS = PRESETS['cotimed']


def load_pickle(run, cache):
    if run in cache:
        return cache[run]
    p = os.path.join(PICKLE_DIR, f'preprocessed_run_{run}.pkl')
    if not os.path.exists(p):
        cache[run] = None
        return None
    print(f'  loading {p} ...')
    with open(p, 'rb') as f:
        cache[run] = pickle.load(f)
    return cache[run]


def azimuthal_prob_matrix_from_indices(hits_stack, mean_energy_arr, idx,
                                       theta_bin_map, r_map):
    """Return H shape (len(idx), N_BINS) of per-shot probability vectors."""
    rows = np.empty((len(idx), N_BINS), dtype=float)
    for row_i, i in enumerate(idx):
        img = hits_stack[i].astype(np.float64, copy=False)
        total = img.sum()
        if total <= 0:
            rows[row_i] = 0.0
            continue
        prob = img / total
        re = (mean_energy_arr[i] / RAW_CHANNELS_PER_BIN
              - SPECTRUM_ROI_START) * 0.6 + 29.4
        annulus = (r_map >= re - DR_HALF) & (r_map < re + DR_HALF)
        rows[row_i] = np.bincount(
            theta_bin_map[annulus],
            weights=prob[annulus],
            minlength=N_BINS,
        )
    return rows


def collect_goose(cache):
    rows_all = []
    for run in GOOSE_RUNS:
        d = load_pickle(run, cache)
        if d is None:
            continue
        ny, nx = d['hits'][0].shape
        y, x = np.mgrid[0:ny, 0:nx]
        r_map = np.hypot(x - CX, y - CY)
        theta = np.arctan2(y - CY, x - CX) + np.pi
        theta_bin_map = np.clip(
            np.floor(theta / (2.0 * np.pi / N_BINS)).astype(int),
            0, N_BINS - 1,
        )
        m = d['masks']['goose'].astype(bool) & d['is_gaussian'].astype(bool)
        idx = np.where(m)[0]
        print(f'  goose run {run}: {idx.size} shots')
        rows = azimuthal_prob_matrix_from_indices(
            d['hits'], d['mean_energy'], idx, theta_bin_map, r_map)
        rows_all.append(rows)
    return np.vstack(rows_all) if rows_all else np.empty((0, N_BINS))


def collect_cotimed(cache):
    rows_all = []
    for run, step in COTIMED_PAIRS:
        d = load_pickle(run, cache)
        if d is None:
            continue
        ny, nx = d['hits'][0].shape
        y, x = np.mgrid[0:ny, 0:nx]
        r_map = np.hypot(x - CX, y - CY)
        theta = np.arctan2(y - CY, x - CX) + np.pi
        theta_bin_map = np.clip(
            np.floor(theta / (2.0 * np.pi / N_BINS)).astype(int),
            0, N_BINS - 1,
        )
        step_axis = d['lxts']
        unique_steps = np.unique(step_axis)
        step_val = unique_steps[step]
        m = (d['masks']['duck'].astype(bool)
             & d['is_gaussian'].astype(bool)
             & (step_axis == step_val))
        idx = np.where(m)[0]
        print(f'  cotimed run {run} step {step} (lxt={step_val*1e12:+.3f} ps): '
              f'{idx.size} shots')
        rows = azimuthal_prob_matrix_from_indices(
            d['hits'], d['mean_energy'], idx, theta_bin_map, r_map)
        rows_all.append(rows)
    return np.vstack(rows_all) if rows_all else np.empty((0, N_BINS))


def nan_gaussian_filter(A, sigma):
    A = np.asarray(A, dtype=float)
    nan_mask = np.isnan(A)
    V = np.where(nan_mask, 0.0, A)
    W = (~nan_mask).astype(float)
    Vs = gaussian_filter(V, sigma=sigma, mode='wrap')
    Ws = gaussian_filter(W, sigma=sigma, mode='wrap')
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(Ws > 1e-12, Vs / Ws, np.nan)


def _recenter_180(M):
    """Roll a (N_BINS, N_BINS) covariance matrix so that bin index 0
    corresponds to physical theta = 180 deg. Extents / tick labels then
    span [-180, 180]."""
    shift = N_BINS // 2
    return np.roll(np.roll(M, shift, axis=0), shift, axis=1)


def draw(ax, M, title, vmax):
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-15
    if CENTER_180:
        M_disp = _recenter_180(M)
        extent = [-180, 180, -180, 180]
        ticks = [-180, -90, 0, 90, 180]
        guides = tuple(g - 180 for g in GUIDES_DEG)
    else:
        M_disp = M
        extent = [0, 360, 0, 360]
        ticks = [0, 90, 180, 270, 360]
        guides = GUIDES_DEG
    im = ax.imshow(M_disp, cmap='RdBu_r', origin='lower',
                   extent=extent,
                   vmin=-vmax, vmax=vmax, aspect='equal')
    for g in guides:
        ax.axvline(g, color='k', ls=':', lw=0.7, alpha=0.55)
        ax.axhline(g, color='k', ls=':', lw=0.7, alpha=0.55)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(r'$\theta_j$ [deg]', fontsize=9)
    ax.set_ylabel(r'$\theta_i$ [deg]', fontsize=9)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def main():
    global CX, CY, OUT_PNG, OUT_NPZ, CENTER_180
    ap = argparse.ArgumentParser()
    ap.add_argument('--cx', type=float, default=DEFAULT_CX,
                    help=f'x-center for r/θ (default {DEFAULT_CX}).')
    ap.add_argument('--cy', type=float, default=DEFAULT_CY,
                    help=f'y-center for r/θ (default {DEFAULT_CY}).')
    ap.add_argument('--center-180', action='store_true',
                    help='Roll each covariance axis by 180 deg so that '
                         'theta = 180 deg sits at the center of the plot.')
    ap.add_argument('--out-suffix', default='',
                    help='Suffix appended to output filenames. '
                         'Auto-set to reflect shifted center / center-180.')
    args = ap.parse_args()
    CX, CY = float(args.cx), float(args.cy)
    CENTER_180 = bool(args.center_180)
    suffix = args.out_suffix
    if not suffix:
        parts = []
        if (CX, CY) != (DEFAULT_CX, DEFAULT_CY):
            parts.append('shifted')
        if CENTER_180:
            parts.append('center180')
        suffix = '_'.join(parts)
    if suffix:
        base = f'plot_real_covariance_goose_vs_cotimed_{suffix}'
        OUT_PNG = os.path.join(OUT_DIR, f'{base}.png')
        OUT_NPZ = os.path.join(OUT_DIR, f'{base}.npz')
    print(f'--- azimuthal covariance center: ({CX:g}, {CY:g}), '
          f'center_180={CENTER_180} ---')

    os.makedirs(OUT_DIR, exist_ok=True)
    cache = {}

    print('--- collecting goose shots (all runs, all steps) ---')
    H_goose = collect_goose(cache)
    print(f'  total goose shots: {H_goose.shape[0]}')

    print('--- collecting cotimed shots (PRESETS["cotimed"]) ---')
    H_cotimed = collect_cotimed(cache)
    print(f'  total cotimed shots: {H_cotimed.shape[0]}')

    cov_goose = np.cov(H_goose, rowvar=False)
    cov_cotim = np.cov(H_cotimed, rowvar=False)
    diag = np.eye(N_BINS, dtype=bool)
    cov_g_p = np.where(diag, np.nan, cov_goose)
    cov_c_p = np.where(diag, np.nan, cov_cotim)
    diff_p = cov_c_p - cov_g_p

    cov_g_sm = nan_gaussian_filter(cov_g_p, SMOOTH_SIGMA_BINS)
    cov_c_sm = nan_gaussian_filter(cov_c_p, SMOOTH_SIGMA_BINS)
    diff_sm = nan_gaussian_filter(diff_p, SMOOTH_SIGMA_BINS)

    v_cov = max(float(np.nanmax(np.abs(cov_g_sm))),
                float(np.nanmax(np.abs(cov_c_sm))))
    v_diff = float(np.nanmax(np.abs(diff_sm)))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    draw(axes[0], cov_g_sm,
         f'unstreaked = goose (all runs/steps)\nN = {H_goose.shape[0]} shots',
         v_cov)
    draw(axes[1], cov_c_sm,
         f'streaked = cotimed (all shots)\nN = {H_cotimed.shape[0]} shots',
         v_cov)
    draw(axes[2], diff_sm,
         'cotimed − goose  (probability-normalized)', v_diff)
    axis_note = ('θ = 180° at plot origin (axes rolled ±180°)'
                 if CENTER_180 else 'θ = 0° at plot origin')
    fig.suptitle(
        'Real-data probability-normalized azimuthal covariance  '
        f'(smoothed σ={SMOOTH_SIGMA_BINS:g} bins, '
        f'center = ({CX:g}, {CY:g}), {axis_note})',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f'wrote {OUT_PNG}')

    np.savez_compressed(
        OUT_NPZ,
        cov_goose=cov_goose,
        cov_cotimed=cov_cotim,
        cov_goose_sm=np.nan_to_num(cov_g_sm),
        cov_cotimed_sm=np.nan_to_num(cov_c_sm),
        diff_sm=np.nan_to_num(diff_sm),
        n_goose_shots=H_goose.shape[0],
        n_cotimed_shots=H_cotimed.shape[0],
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
        cotimed_pairs=np.array(COTIMED_PAIRS),
        goose_runs=np.array(GOOSE_RUNS),
        cx=CX, cy=CY,
    )
    print(f'wrote {OUT_NPZ}')


if __name__ == '__main__':
    main()
