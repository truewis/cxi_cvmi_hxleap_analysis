#!/usr/bin/env python3
"""
For every `step<NN>/` folder under `run145_stepwise_peak_sigma_y/`, pick
5 ROI_R + 5 ROI_L example shots (quartile-spaced across the sigma_y
distribution within each ROI) and render one figure per step:

    step<NN>/example_shots_hits_and_score.png     (10 rows × 2 columns)
    step<NN>/example_shots_hits_and_score.npy     picks + metadata

Column layout:
    left  — hitfinder image `data['hits'][shot_idx]`  (viridis, origin='lower')
    right — recomputed combined_sym_score_map        (magma), with a red '+'
             at the peak coordinate.

Row header labels the shot with its ROI, event index, sigma_y, and peak_score.

Requires conda env CXI.
"""
import argparse
import os
import pickle
import re
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
PICKLE_PATH = f'/sdf/scratch/users/j/jinseop/preprocessed_run_{RUN}.pkl'
STEPWISE_ROOT = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
                 'batch_analysis_results/legacy_outputs/run145_stepwise_peak_sigma_y')
STEP_RE = re.compile(r'^step(?P<idx>\d+)$')
N_PER_ROI = 5
ROI_COLORS = {'R': 'tab:orange', 'L': 'tab:blue'}


def lxt_label(lxt_s):
    if abs(float(lxt_s)) < 25e-15:
        return 'cotimed'
    return 'laser late' if float(lxt_s) < 0 else 'laser early'


def _pick_quartile(sy_arr, n=N_PER_ROI):
    """Return indices into sy_arr at quantiles evenly spanning the finite
    range: min, 1st quartile, median, 3rd quartile, max. Duplicates removed."""
    finite = np.isfinite(sy_arr)
    if finite.sum() == 0:
        return np.array([], dtype=int)
    idx_finite = np.where(finite)[0]
    sy_finite = sy_arr[idx_finite]
    order = np.argsort(sy_finite)
    order_full = idx_finite[order]
    m = order_full.size
    if m <= n:
        return order_full
    picks = np.linspace(0, m - 1, n).round().astype(int)
    picks = np.unique(picks)
    return order_full[picks]


def process_step(step_dir, hits, mean_energy):
    step_idx = int(STEP_RE.match(os.path.basename(step_dir)).group('idx'))
    p_r = os.path.join(step_dir, 'sigma_y_ROI_R.npy')
    p_l = os.path.join(step_dir, 'sigma_y_ROI_L.npy')
    if not (os.path.exists(p_r) and os.path.exists(p_l)):
        print(f'  [skip step {step_idx}] missing sigma_y files')
        return
    dr = np.load(p_r, allow_pickle=True).item()
    dl = np.load(p_l, allow_pickle=True).item()
    lxt_s = float(dr.get('lxt_s', 0.0))

    picks = []  # list of tuples (roi, shot_idx, sigma_y, peak_score, peak_x, peak_y)
    for roi, d in (('R', dr), ('L', dl)):
        sy = np.asarray(d['sigma_y'], dtype=float)
        si = np.asarray(d['shot_idx'], dtype=int)
        ps = np.asarray(d['peak_score'], dtype=float)
        px = np.asarray(d['peak_x'], dtype=float)
        py = np.asarray(d['peak_y'], dtype=float)
        chosen = _pick_quartile(sy, n=N_PER_ROI)
        for k in chosen:
            picks.append((roi, int(si[k]), float(sy[k]),
                          float(ps[k]), float(px[k]), float(py[k])))

    if not picks:
        print(f'  [skip step {step_idx}] no finite sigma_y shots')
        return

    n_rows = len(picks)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(9.6, 3.4 * n_rows), squeeze=False,
    )
    for r, (roi, shot_idx, sigma_y, peak_score, peak_x, peak_y) in enumerate(picks):
        # --- Left: hitfinder image ---
        ax_h = axes[r, 0]
        img = hits[shot_idx].astype(float, copy=False)
        vmax = float(np.percentile(img, 99.5)) if img.max() > 0 else 1.0
        ax_h.imshow(img, cmap='viridis', origin='lower', vmin=0.0, vmax=vmax)
        ax_h.axvline(CX, color='cyan', ls='--', alpha=0.5, lw=0.8)
        ax_h.axhline(CY, color='cyan', ls='--', alpha=0.5, lw=0.8)
        for rad in (20.0, 60.0):
            ax_h.add_patch(plt.Circle((CX, CY), radius=rad,
                                      color='white', fill=False, ls='--',
                                      alpha=0.35, lw=0.9))
        ax_h.plot(peak_x, peak_y, marker='+', ms=14, mew=2.0,
                  color=ROI_COLORS[roi])
        ax_h.set_xlabel('pixel x'); ax_h.set_ylabel('pixel y')
        ax_h.set_title(
            f'ROI_{roi}  |  shot {shot_idx}  |  σ_y = {sigma_y:.2f} px  |  '
            f'peak_score = {peak_score:.2e}',
            fontsize=10,
        )

        # --- Right: score map at the same shot's energy ---
        ax_s = axes[r, 1]
        energy = float(mean_energy[shot_idx])
        x_centers, y_centers, m5, p5, combined = _compute_score_maps(
            img=img, energy=energy, cx=CX, cy=CY,
        )
        extent = [x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]]
        im = ax_s.imshow(combined, cmap='magma', origin='lower', extent=extent,
                         aspect='equal')
        ax_s.plot(peak_x, peak_y, marker='+', ms=14, mew=2.0,
                  color=ROI_COLORS[roi])
        ax_s.axvline(CX, color='cyan', ls='--', alpha=0.4, lw=0.6)
        ax_s.axhline(CY, color='cyan', ls='--', alpha=0.4, lw=0.6)
        ax_s.set_xlabel('trial ring center x [px]')
        ax_s.set_ylabel('trial ring center y [px]')
        ax_s.set_title('combined_sym_score_map', fontsize=10)
        plt.colorbar(im, ax=ax_s, fraction=0.046, pad=0.03,
                     label='score')

    lxt_fs = f'{lxt_s * 1e15:+g} fs'
    fig.suptitle(
        f'run {RUN} duck  step {step_idx}  (lxt = {lxt_fs}, {lxt_label(lxt_s)})\n'
        f'Example ROI_R (top) & ROI_L (bottom) shots — hitfinder | score map',
        fontweight='bold',
    )
    fig.tight_layout()
    out_png = os.path.join(step_dir, 'example_shots_hits_and_score.png')
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    payload = {
        'run': RUN, 'step_idx': step_idx, 'lxt_s': lxt_s,
        'picks': picks,
        'columns': ['roi', 'shot_idx', 'sigma_y', 'peak_score', 'peak_x', 'peak_y'],
        'n_per_roi': N_PER_ROI,
    }
    np.save(
        os.path.join(step_dir, 'example_shots_hits_and_score.npy'),
        payload, allow_pickle=True,
    )
    print(f'  step {step_idx}: wrote {len(picks)} example shots → {out_png}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=STEPWISE_ROOT,
                    help='parent directory containing step<NN>/ subfolders')
    args = ap.parse_args()

    print(f'loading {PICKLE_PATH}')
    with open(PICKLE_PATH, 'rb') as f:
        data = pickle.load(f)
    hits = data['hits']
    mean_energy = data['mean_energy']

    root = os.path.abspath(args.root)
    step_dirs = sorted(
        os.path.join(root, name)
        for name in os.listdir(root)
        if STEP_RE.match(name)
    )
    print(f'found {len(step_dirs)} step directories under {root}')

    for sd in step_dirs:
        process_step(sd, hits, mean_energy)


if __name__ == '__main__':
    main()
