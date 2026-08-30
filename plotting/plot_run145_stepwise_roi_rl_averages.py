#!/usr/bin/env python3
"""
For each step folder under batch_analysis/run145_stepwise_peak_sigma_y/,
add a per-step figure showing the averaged hitfinder frame of ROI_R and
ROI_L shots (the same shots used for sigma_y).

Reads the shot indices from `sigma_y_ROI_R.npy` / `sigma_y_ROI_L.npy` and
pulls hits frames from the run-145 preprocessed pickle. Writes:

    <step_dir>/roi_rl_avg_hits.png
    <step_dir>/roi_rl_avg_hits.npy    dict{avg_R, avg_L, n_R, n_L, shot_idx_R, shot_idx_L, ...}

Runs under conda env CXI.
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
from aggregate_4roi_by_energy import CX, CY


RUN = 145
KIND = 'duck'
PICKLE_PATH = f'/sdf/scratch/users/j/jinseop/preprocessed_run_{RUN}.pkl'
STEPWISE_ROOT = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
                 'batch_analysis_results/legacy_outputs/run145_stepwise_peak_sigma_y')
STEP_RE = re.compile(r'^step(?P<idx>\d+)$')


def lxt_label(lxt_s):
    if abs(float(lxt_s)) < 25e-15:
        return 'cotimed'
    return 'laser late' if float(lxt_s) < 0 else 'laser early'


def _mean_stack(hits, shot_idx):
    if shot_idx.size == 0:
        return None
    frames = np.stack([hits[i].astype(float, copy=False) for i in shot_idx])
    return frames.mean(axis=0)


def process_step(step_dir, hits):
    step_idx = int(STEP_RE.match(os.path.basename(step_dir)).group('idx'))
    p_r = os.path.join(step_dir, 'sigma_y_ROI_R.npy')
    p_l = os.path.join(step_dir, 'sigma_y_ROI_L.npy')
    if not (os.path.exists(p_r) and os.path.exists(p_l)):
        print(f'  [skip step {step_idx}] missing sigma_y files')
        return

    dr = np.load(p_r, allow_pickle=True).item()
    dl = np.load(p_l, allow_pickle=True).item()
    idx_r = np.asarray(dr['shot_idx'], dtype=int)
    idx_l = np.asarray(dl['shot_idx'], dtype=int)
    lxt_s = float(dr.get('lxt_s', 0.0))

    avg_r = _mean_stack(hits, idx_r)
    avg_l = _mean_stack(hits, idx_l)

    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, avg, roi, n in ((ax_r, avg_r, 'R', idx_r.size),
                            (ax_l, avg_l, 'L', idx_l.size)):
        if avg is None:
            ax.text(0.5, 0.5, f'no ROI_{roi} shots', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        vmax = float(np.percentile(avg, 99.5)) if avg.max() > 0 else 1.0
        im = ax.imshow(avg, cmap='viridis', origin='lower', vmin=0.0, vmax=vmax)
        ax.axvline(CX, color='cyan', ls='--', alpha=0.5, lw=0.8)
        ax.axhline(CY, color='cyan', ls='--', alpha=0.5, lw=0.8)
        for rad in (20.0, 60.0):
            ax.add_patch(plt.Circle((CX, CY), radius=rad,
                                    color='white', fill=False, ls='--',
                                    alpha=0.35, lw=0.9))
        ax.set_title(f'ROI_{roi}  (mean of {n} shots)', fontsize=11)
        ax.set_xlabel('pixel x'); ax.set_ylabel('pixel y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                     label='mean hitfinder intensity')

    lxt_fs = f'{lxt_s * 1e15:+g} fs'
    fig.suptitle(
        f'run {RUN} duck  step {step_idx}  (lxt = {lxt_fs}, {lxt_label(lxt_s)})\n'
        f'Averaged hitfinder frames for ROI_R vs ROI_L shots',
        fontweight='bold',
    )
    fig.tight_layout()
    out_png = os.path.join(step_dir, 'roi_rl_avg_hits.png')
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    np.save(
        os.path.join(step_dir, 'roi_rl_avg_hits.npy'),
        {
            'avg_R': avg_r if avg_r is not None else np.zeros((0, 0)),
            'avg_L': avg_l if avg_l is not None else np.zeros((0, 0)),
            'n_R': int(idx_r.size), 'n_L': int(idx_l.size),
            'shot_idx_R': idx_r, 'shot_idx_L': idx_l,
            'run': RUN, 'step_idx': step_idx, 'lxt_s': lxt_s,
        },
        allow_pickle=True,
    )
    print(f'  step {step_idx}: R n={idx_r.size}, L n={idx_l.size}  →  {out_png}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=STEPWISE_ROOT,
                    help='parent directory containing step<NN>/ subfolders')
    args = ap.parse_args()

    print(f'loading {PICKLE_PATH}')
    with open(PICKLE_PATH, 'rb') as f:
        data = pickle.load(f)
    hits = data['hits']

    root = os.path.abspath(args.root)
    step_dirs = []
    for name in sorted(os.listdir(root)):
        m = STEP_RE.match(name)
        if m:
            step_dirs.append(os.path.join(root, name))
    print(f'found {len(step_dirs)} step directories under {root}')

    for sd in step_dirs:
        process_step(sd, hits)


if __name__ == '__main__':
    main()
