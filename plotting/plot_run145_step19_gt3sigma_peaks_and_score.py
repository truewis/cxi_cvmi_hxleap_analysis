#!/usr/bin/env python3
"""
For run 145 step 19 (duck), among shots with wiggle-analysis
significance > 3 sigma, produce:

  1. `peak_position_distribution.png` — 2D histogram of the analysis peak
     positions (peak_x, peak_y), centered at (cx, cy) = (67, 59).
  2. `avg_score_landscape.png` — mean `combined_sym_score_map` across those
     shots, recomputed shot-by-shot using the same inner loop as the wiggle
     finder (`_compute_score_maps`).

Also saves the underlying arrays as `.npy` alongside.

Outputs land in:
    batch_analysis/run145_step19_gt3sigma/

Runs under conda env CXI.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add batch_analysis/ to path so sibling imports work from plotting/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_sigma_sweep_center_accuracy import _compute_score_maps
from aggregate_4roi_by_energy import CX, CY


RUN = 145
KIND = 'duck'
SIG_CUT = 3.0
PICKLE_PATH = f'/sdf/scratch/users/j/jinseop/preprocessed_run_{RUN}.pkl'
# Default output goes under the stepwise root now that run145_step19_gt3sigma
# was moved into run145_stepwise_peak_sigma_y/.
DEFAULT_OUT_PARENT = (
    '/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
    'batch_analysis_results/legacy_outputs/run145_stepwise_peak_sigma_y'
)


def _batch_results_dir(step):
    return (
        f'/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
        f'batch_analysis_results/results_145_to_152_4_ROI/'
        f'circular_wiggler_{KIND}_run{RUN}_step{step}_batch_metrics'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', type=int, default=19)
    ap.add_argument('--out_parent', default=DEFAULT_OUT_PARENT,
                    help='parent directory for the run<RUN>_step<STEP>_gtXsigma folder')
    args = ap.parse_args()
    STEP = int(args.step)
    BATCH_RESULTS = _batch_results_dir(STEP)
    OUT_DIR = os.path.join(
        args.out_parent, f'run{RUN}_step{STEP}_gt{SIG_CUT:g}sigma'
    )
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Load saved wiggle-analysis peak positions & significance ---
    est_path = os.path.join(BATCH_RESULTS, f'data_peak_positions_run_{RUN}.npy')
    est = np.load(est_path, allow_pickle=True).item()
    x_est = np.asarray(est['x'], dtype=float)
    y_est = np.asarray(est['y'], dtype=float)
    sig = np.asarray(est['significance'], dtype=float)

    # --- Load the run pickle to get shot indices + hits + energies ---
    print(f'loading {PICKLE_PATH}')
    with open(PICKLE_PATH, 'rb') as f:
        data = pickle.load(f)
    unique_steps = np.unique(data['lxts'])
    step_val = unique_steps[STEP]
    mask = (data['masks'][KIND].astype(bool)
            & data['is_gaussian'].astype(bool)
            & (data['lxts'] == step_val))
    orig_idx = np.where(mask)[0]
    n_min = min(len(orig_idx), len(x_est))
    orig_idx = orig_idx[:n_min]
    x_est = x_est[:n_min]; y_est = y_est[:n_min]; sig = sig[:n_min]

    keep = sig > SIG_CUT
    kept_idx_local = np.where(keep)[0]
    kept_shot_idx = orig_idx[keep]
    kept_x = x_est[keep]
    kept_y = y_est[keep]
    kept_sig = sig[keep]
    n_kept = int(keep.sum())
    print(f'Run {RUN} step {STEP} (lxt={step_val:.3g} s)')
    print(f'  pool = {n_min} base-masked shots; {n_kept} pass >{SIG_CUT:g}σ '
          f'({100.0 * n_kept / n_min:.1f}% of pool)')

    # ---------- Plot 1: 2D peak distribution ----------
    x_bins = np.linspace(CX - 15, CX + 15, 61)
    y_bins = np.linspace(CY - 15, CY + 15, 61)
    H, xed, yed = np.histogram2d(kept_x, kept_y, bins=[x_bins, y_bins])
    # np.histogram2d returns (N_xbins, N_ybins) with axis 0 = x, axis 1 = y;
    # transpose so imshow(origin='lower') plots y on the vertical axis.
    H = H.T
    extent = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]

    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    im = ax.imshow(H, cmap='hot', origin='lower', extent=extent, aspect='equal')
    ax.axvline(CX, color='cyan', ls='--', alpha=0.6)
    ax.axhline(CY, color='cyan', ls='--', alpha=0.6)
    # dashed circle at r=3 (the near-center threshold in your earlier question)
    ax.add_patch(plt.Circle((CX, CY), 3.0, color='lime', fill=False,
                            ls='--', lw=1.4, label='r=3 px near-center'))
    ax.add_patch(plt.Circle((CX, CY), 10.0, color='white', fill=False,
                            ls=':', alpha=0.5))
    ax.set_xlabel('peak x [px]')
    ax.set_ylabel('peak y [px]')
    ax.set_title(f'run {RUN} step {STEP}  (lxt={step_val*1e15:+g} fs)  |  '
                 f'peak positions of {n_kept} shots >{SIG_CUT:g}σ',
                 fontsize=11, fontweight='bold')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('shots per (x, y) bin')
    ax.plot(np.mean(kept_x), np.mean(kept_y), 'x', color='blue', ms=10,
            mew=2, label=f'mean = ({np.mean(kept_x):.2f}, {np.mean(kept_y):.2f})')
    ax.legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    out_hist = os.path.join(OUT_DIR, 'peak_position_distribution.png')
    fig.savefig(out_hist, dpi=140)
    plt.close(fig)
    print(f'wrote {out_hist}')

    np.save(
        os.path.join(OUT_DIR, 'peak_position_distribution.npy'),
        {
            'H': H, 'x_edges': xed, 'y_edges': yed,
            'kept_shot_idx': kept_shot_idx,
            'kept_x': kept_x, 'kept_y': kept_y, 'kept_sig': kept_sig,
            'sig_cut': SIG_CUT, 'run': RUN, 'step': STEP,
            'cx': CX, 'cy': CY,
        },
        allow_pickle=True,
    )

    # ---------- Plot 2: average score landscape ----------
    #
    # Each shot's map lives on its own (xc, yc) grid because
    # wiggle_range = min(52-re, re-28) depends on the shot's mean_energy.
    # Build a common master grid (union of every shot's span) and embed
    # each local map centred on (cx, cy) before averaging. Unfilled cells
    # are NaN, and we average with nanmean so each cell reflects only the
    # shots whose grid actually covered it.
    print(f'recomputing score maps for {n_kept} shots ...')

    # First pass: determine the max wiggle_range across the surviving shots
    # so we know the master grid extent.
    from analysis_library.cvmi import (
        RAW_CHANNELS_PER_BIN, SPECTRUM_ROI_START,
    )
    def _re_of(e):
        return (float(e) / RAW_CHANNELS_PER_BIN - SPECTRUM_ROI_START) * 0.6 + 29.4
    max_wr = 0.0
    for shot_idx in kept_shot_idx:
        e = float(data['mean_energy'][shot_idx])
        re_ = _re_of(e)
        wr = min(52.0 - re_, re_ - 28.0)
        if wr > max_wr:
            max_wr = wr
    step_master = 0.4
    max_steps = int(np.ceil(max_wr / step_master))
    master_offsets = np.arange(-max_steps, max_steps + 1) * step_master
    x_centers_ref = CX + master_offsets
    y_centers_ref = CY + master_offsets
    N_master = master_offsets.size
    print(f'  master grid: {N_master}×{N_master}  (max_wr = {max_wr:.2f} px)')

    accum_sum = np.zeros((N_master, N_master), dtype=float)
    accum_cnt = np.zeros((N_master, N_master), dtype=int)
    n_valid = 0
    peak_scores = []
    for shot_idx in kept_shot_idx:
        img = data['hits'][shot_idx]
        energy = float(data['mean_energy'][shot_idx])
        x_centers, y_centers, m5, p5, combined = _compute_score_maps(
            img=img, energy=energy, cx=CX, cy=CY,
        )
        if combined.size == 0:
            continue
        # Embed this shot's map onto the master grid. Because both grids
        # are centred on (cx, cy) with the same step size, the local grid
        # is a centered sub-block of the master grid.
        n_local = x_centers.size
        assert x_centers.size == y_centers.size == combined.shape[0] == combined.shape[1]
        off = (N_master - n_local) // 2
        block = combined
        # Treat NaN in the local block as missing (do not add).
        mask_valid = np.isfinite(block)
        accum_sum[off:off + n_local, off:off + n_local] += np.where(mask_valid, block, 0.0)
        accum_cnt[off:off + n_local, off:off + n_local] += mask_valid.astype(int)
        n_valid += 1
        if mask_valid.any():
            peak_scores.append(float(np.nanmax(block)))

    if n_valid == 0:
        print('no valid score maps; aborting')
        return
    with np.errstate(invalid='ignore', divide='ignore'):
        avg_score = np.where(accum_cnt > 0, accum_sum / accum_cnt, np.nan)
    print(f'averaged {n_valid} score maps  '
          f'(peak of average = {np.nanmax(avg_score):.3e}, '
          f'min per-cell coverage = {accum_cnt[accum_cnt>0].min()}, '
          f'max = {accum_cnt.max()})')

    extent2 = [x_centers_ref[0], x_centers_ref[-1],
               y_centers_ref[0], y_centers_ref[-1]]

    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    im = ax.imshow(avg_score, cmap='magma', origin='lower', extent=extent2,
                   aspect='equal')
    ax.axvline(CX, color='cyan', ls='--', alpha=0.5, lw=0.9)
    ax.axhline(CY, color='cyan', ls='--', alpha=0.5, lw=0.9)
    ax.add_patch(plt.Circle((CX, CY), 3.0, color='lime', fill=False,
                            ls='--', lw=1.4, label='r=3 px'))
    # Mark the peak of the AVERAGE map (NaN-safe).
    avg_for_peak = np.where(np.isfinite(avg_score), avg_score, -np.inf)
    ipy, ipx = np.unravel_index(int(np.argmax(avg_for_peak)), avg_score.shape)
    px = x_centers_ref[ipx]; py = y_centers_ref[ipy]
    ax.plot(px, py, marker='+', ms=16, mew=2, color='cyan',
            label=f'peak of avg = ({px:.2f}, {py:.2f})')
    ax.set_xlabel('trial ring center x [px]')
    ax.set_ylabel('trial ring center y [px]')
    ax.set_title(f'run {RUN} step {STEP}  (lxt={step_val*1e15:+g} fs)  |  '
                 f'average combined_sym_score_map  ({n_valid} shots >{SIG_CUT:g}σ)',
                 fontsize=11, fontweight='bold')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('mean combined_sym_score_map')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.85)
    fig.tight_layout()
    out_avg = os.path.join(OUT_DIR, 'avg_score_landscape.png')
    fig.savefig(out_avg, dpi=140)
    plt.close(fig)
    print(f'wrote {out_avg}')

    np.save(
        os.path.join(OUT_DIR, 'avg_score_landscape.npy'),
        {
            'avg_score': avg_score,
            'x_centers': x_centers_ref, 'y_centers': y_centers_ref,
            'n_valid': n_valid, 'n_kept': n_kept, 'sig_cut': SIG_CUT,
            'peak_of_average_xy': (float(px), float(py)),
            'per_shot_peak_scores': np.asarray(peak_scores, dtype=float),
            'run': RUN, 'step': STEP, 'cx': CX, 'cy': CY,
        },
        allow_pickle=True,
    )


if __name__ == '__main__':
    main()
