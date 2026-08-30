#!/usr/bin/env python3
"""
For every aggregate_4roi_* folder under batch_analysis/, generate a
run-step-proportion matrix showing what fraction of each ROI's total
pixel-sum came from each contributing (run, step) pair.

Two aggregate families are handled:

  * `_steps` / `run145_*` — produced by aggregate_4roi_late_steps.py.
    Their NOTES.md already lists per-ROI per-pair pixel sums; we parse
    those tables directly. The compare folder is skipped (it just
    re-plots two sibling folders).

  * `_by_energy(_shifted)` — produced by aggregate_4roi_by_energy.py.
    Only `aggregation_metadata.npy` carries the pair list; we reopen
    each pair's per-ROI source .npy from the batch_analysis_results
    tree to compute pixel sums.

Output: `run_step_proportion_matrix.png` and
`run_step_proportion_matrix.npy` inside each aggregate folder. Matrix
shape (4, N_pairs); each ROI row is normalized to sum = 1 so the
color represents that ROI's *composition* by (run, step). Pairs with
zero contribution are shown as NaN (blank cells).

Requires conda env CXI.
"""
import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BATCH_ANALYSIS_DIR = '/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/batch_analysis_results/legacy_outputs'
BATCH_RESULTS_ROOT = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
                     'batch_analysis_results/results_145_to_152_4_ROI')
ROIS = ('R', 'L', 'U', 'D')


# --- Family 1: _steps / run145_* — parse NOTES.md tables ---------------
def parse_notes_pairs_and_sums(notes_path):
    """Return {roi: {(run, step): sum}} by parsing the '### ROI_<x>' tables
    of a `_steps`-style NOTES.md. Missing entries are simply absent from
    the dict."""
    text = open(notes_path).read()
    # Split by ROI section headers; each ROI section holds its own table.
    per_roi = {r: {} for r in ROIS}
    section_re = re.compile(r'###\s+ROI_(?P<roi>[RLUD])\b(.*?)(?=###|\Z)',
                            re.DOTALL)
    row_re = re.compile(r'^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([-+0-9.eE]+)\s*\|',
                        re.MULTILINE)
    for m in section_re.finditer(text):
        roi = m.group('roi')
        body = m.group(2)
        for rm in row_re.finditer(body):
            run, step, s = int(rm.group(1)), int(rm.group(2)), float(rm.group(3))
            per_roi[roi][(run, step)] = s
    return per_roi


# --- Family 2: _by_energy — recompute pixel sums per (roi, pair) --------
def _roi_path(run, step, roi, kind):
    return os.path.join(
        BATCH_RESULTS_ROOT,
        f'circular_wiggler_{kind}_run{run}_step{step}_batch_metrics',
        f'data_accumulated_hits_ROI_{roi}_run_{run}.npy',
    )


def recompute_pairs_and_sums(pairs, kind):
    """For each ROI, compute pixel-sum per (run, step) by loading the
    source `data_accumulated_hits_ROI_*_run_*.npy`. Missing files are
    skipped."""
    per_roi = {r: {} for r in ROIS}
    for run, step in pairs:
        for roi in ROIS:
            p = _roi_path(run, step, roi, kind)
            if not os.path.exists(p):
                continue
            per_roi[roi][(run, step)] = float(np.load(p).sum())
    return per_roi


# --- Shared plotting ----------------------------------------------------
def plot_proportion_matrix(per_roi, aggregate_name, out_dir):
    """Compose the ROI x pair matrix and save it."""
    # Union of all pairs seen across ROIs.
    all_pairs = set()
    for m in per_roi.values():
        all_pairs.update(m.keys())
    if not all_pairs:
        print(f'  [warn] no pairs found for {aggregate_name}; skipping matrix')
        return None
    pairs = sorted(all_pairs)

    matrix = np.full((len(ROIS), len(pairs)), np.nan)
    for i, roi in enumerate(ROIS):
        d = per_roi[roi]
        # Row sum for normalization.
        row_sum = sum(d.values())
        if row_sum <= 0:
            continue
        for j, pair in enumerate(pairs):
            if pair in d:
                matrix[i, j] = d[pair] / row_sum

    fig_w = max(6.0, 0.55 * len(pairs) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, 3.2))
    im = ax.imshow(matrix, cmap='viridis', aspect='auto',
                   vmin=0.0, vmax=max(np.nanmax(matrix), 1e-6))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v):
                # Choose text color for contrast against viridis
                txt_color = 'white' if v < 0.5 * np.nanmax(matrix) else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        color=txt_color, fontsize=7)
    ax.set_xticks(np.arange(len(pairs)))
    ax.set_xticklabels([f'{r}:{s}' for r, s in pairs],
                       rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(ROIS)))
    ax.set_yticklabels([f'ROI_{r}' for r in ROIS])
    ax.set_xlabel('(run, step)')
    ax.set_title(f'{aggregate_name}\nrow-normalized ROI composition by (run, step)',
                 fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label('fraction of ROI pixel-sum', fontsize=8)
    fig.tight_layout()
    out_png = os.path.join(out_dir, 'run_step_proportion_matrix.png')
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    np.save(os.path.join(out_dir, 'run_step_proportion_matrix.npy'),
            {'matrix': matrix, 'rois': list(ROIS),
             'pairs': [list(p) for p in pairs]},
            allow_pickle=True)
    print(f'  wrote {out_png}')
    return out_png


# --- Family dispatch ----------------------------------------------------
def is_by_energy(name):
    return name.endswith('_by_energy') or name.endswith('_by_energy_shifted')


def is_compare(name):
    return 'compare_' in name


def is_steps(name):
    return name.endswith('_steps')


def process_folder(folder):
    name = os.path.basename(folder)
    print(f'=== {name} ===')

    if is_compare(name):
        # Compare folders don't have their own pair list; skip.
        print('  [skip] compare folder — no independent pair list')
        return

    if is_steps(name) or 'run145' in name:
        notes = os.path.join(folder, 'NOTES.md')
        if not os.path.exists(notes):
            print(f'  [skip] no NOTES.md in {folder}')
            return
        per_roi = parse_notes_pairs_and_sums(notes)
        plot_proportion_matrix(per_roi, name, folder)
        return

    if is_by_energy(name):
        meta_path = os.path.join(folder, 'aggregation_metadata.npy')
        if not os.path.exists(meta_path):
            print(f'  [skip] no aggregation_metadata.npy in {folder}')
            return
        meta = np.load(meta_path, allow_pickle=True).item()
        pairs = [tuple(p) for p in meta.get('pairs', [])]
        kind = meta.get('kind', 'duck')
        if not pairs:
            print(f'  [skip] empty pair list in {meta_path}')
            return
        per_roi = recompute_pairs_and_sums(pairs, kind)
        plot_proportion_matrix(per_roi, name, folder)
        return

    print(f'  [skip] unrecognized aggregate flavor: {name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None,
                    help='glob-style substring; only folders whose basename contains this run')
    args = ap.parse_args()

    folders = sorted(
        os.path.join(BATCH_ANALYSIS_DIR, d)
        for d in os.listdir(BATCH_ANALYSIS_DIR)
        if d.startswith('aggregated_4roi_')
    )
    if args.only:
        folders = [f for f in folders if args.only in os.path.basename(f)]

    for f in folders:
        process_folder(f)


if __name__ == '__main__':
    main()
