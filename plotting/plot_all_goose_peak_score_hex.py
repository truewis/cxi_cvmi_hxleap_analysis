#!/usr/bin/env python3
"""
Aggregate every goose (run, step) directory under
`results_145_to_152_4_ROI/` and produce one peak_score+significance hex-bin
heatmap identical in shape to
`plot_run145_stepwise_peak_score_hex.py`, but pooling ALL goose shots.

Cell = MEAN of peak_score (left panel) or significance (right panel) over
shots whose (peak_x, peak_y) lands inside that hex. Hex extent ≈ 3 px,
`mincnt = 5`. Window: ±15 px around (cx, cy) = (67, 59).

Output tree:
    batch_analysis/aggregated_all_goose_peak_score_hex/
        peak_score_hex.png
        peak_score_hex.npy
"""
import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


KIND = 'goose'
BATCH_RESULTS_ROOT = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
                     'batch_analysis_results/results_145_to_152_4_ROI')
DEFAULT_OUT_DIR = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
                   'batch_analysis_results/legacy_outputs/'
                   'aggregated_all_goose_peak_score_hex')

CX, CY = 67, 59
WINDOW_HALF = 15.0
HEX_PX = 3.0
MIN_MINCNT = 5


DIR_RE = re.compile(
    r'^circular_wiggler_(?P<kind>goose)_run(?P<run>\d+)_step(?P<step>\d+)_batch_metrics$'
)


def _walk_goose_dirs(root):
    """Yield (run, step, sweep_dir) for every goose folder under root."""
    for name in sorted(os.listdir(root)):
        m = DIR_RE.match(name)
        if not m:
            continue
        yield int(m.group('run')), int(m.group('step')), os.path.join(root, name)


def _load_shot_arrays(sweep_dir, run):
    """Return (x, y, sig, score) arrays for one (run, step), aligned; or None."""
    p_pos = os.path.join(sweep_dir, f'data_peak_positions_run_{run}.npy')
    p_sh  = os.path.join(sweep_dir, f'data_score_vs_hits_run_{run}.npy')
    if not (os.path.exists(p_pos) and os.path.exists(p_sh)):
        return None
    pos = np.load(p_pos, allow_pickle=True).item()
    sh  = np.load(p_sh,  allow_pickle=True).item()
    x = np.asarray(pos['x'], dtype=float)
    y = np.asarray(pos['y'], dtype=float)
    sig = np.asarray(pos['significance'], dtype=float)
    scores = np.asarray(sh['scores'], dtype=float)
    n = min(len(x), len(scores))
    return x[:n], y[:n], sig[:n], scores[:n]


def _hex_gridsize():
    return max(3, int(round(2 * WINDOW_HALF / HEX_PX)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    xs, ys, sigs, scores, run_step = [], [], [], [], []
    n_missing = 0
    for run, step, sd in _walk_goose_dirs(BATCH_RESULTS_ROOT):
        r = _load_shot_arrays(sd, run)
        if r is None:
            n_missing += 1
            continue
        x, y, s, sc = r
        xs.append(x); ys.append(y); sigs.append(s); scores.append(sc)
        run_step.extend([(run, step)] * len(x))

    if not xs:
        raise SystemExit('no goose shots found')
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    sig = np.concatenate(sigs)
    score = np.concatenate(scores)
    print(f'{len(x)} total goose shots pooled from '
          f'{len(xs)} (run, step) directories (missing files: {n_missing})')

    m = ((np.abs(x - CX) <= WINDOW_HALF) & (np.abs(y - CY) <= WINDOW_HALF)
         & np.isfinite(score) & np.isfinite(sig))
    xw = x[m]; yw = y[m]; sw = score[m]; gw = sig[m]
    print(f'{int(m.sum())} shots within ±{WINDOW_HALF:g} px of ({CX}, {CY})')

    gs = _hex_gridsize()
    x_range = (CX - WINDOW_HALF, CX + WINDOW_HALF)
    y_range = (CY - WINDOW_HALF, CY + WINDOW_HALF)

    fig, (ax_s, ax_g) = plt.subplots(1, 2, figsize=(15, 6.6))

    hb_s = ax_s.hexbin(
        xw, yw, C=sw, gridsize=gs, reduce_C_function=np.mean,
        extent=[*x_range, *y_range], mincnt=MIN_MINCNT, cmap='viridis',
    )
    ax_s.axvline(CX, color='cyan', ls='--', alpha=0.6, lw=0.9)
    ax_s.axhline(CY, color='cyan', ls='--', alpha=0.6, lw=0.9)
    ax_s.add_patch(plt.Circle((CX, CY), 3.0, color='lime',
                              fill=False, ls='--', lw=1.4))
    ax_s.add_patch(plt.Circle((CX, CY), 10.0, color='white',
                              fill=False, ls=':', alpha=0.5))
    ax_s.set_xlim(x_range); ax_s.set_ylim(y_range); ax_s.set_aspect('equal')
    ax_s.set_xlabel('peak x [px]'); ax_s.set_ylabel('peak y [px]')
    ax_s.set_title(f'mean peak_score  |  {int(m.sum())} shots  |  hex ≈ {HEX_PX:g} px')
    cb = fig.colorbar(hb_s, ax=ax_s, fraction=0.046, pad=0.03)
    cb.set_label('mean peak_score')

    hb_g = ax_g.hexbin(
        xw, yw, C=gw, gridsize=gs, reduce_C_function=np.mean,
        extent=[*x_range, *y_range], mincnt=MIN_MINCNT, cmap='inferno',
    )
    ax_g.axvline(CX, color='cyan', ls='--', alpha=0.6, lw=0.9)
    ax_g.axhline(CY, color='cyan', ls='--', alpha=0.6, lw=0.9)
    ax_g.add_patch(plt.Circle((CX, CY), 3.0, color='lime',
                              fill=False, ls='--', lw=1.4))
    ax_g.add_patch(plt.Circle((CX, CY), 10.0, color='white',
                              fill=False, ls=':', alpha=0.5))
    ax_g.set_xlim(x_range); ax_g.set_ylim(y_range); ax_g.set_aspect('equal')
    ax_g.set_xlabel('peak x [px]'); ax_g.set_ylabel('peak y [px]')
    ax_g.set_title(f'mean significance [σ]  |  {int(m.sum())} shots  |  hex ≈ {HEX_PX:g} px')
    cb2 = fig.colorbar(hb_g, ax=ax_g, fraction=0.046, pad=0.03)
    cb2.set_label('mean significance')

    fig.suptitle(
        f'all goose shots (runs 145-150+147, every step)  |  '
        f'peak_score and significance vs peak position\n'
        f'lime dashed = r=3 px, white dotted = r=10 px  |  mincnt={MIN_MINCNT}',
        fontweight='bold',
    )
    fig.tight_layout()
    out_png = os.path.join(out_dir, 'peak_score_hex.png')
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f'wrote {out_png}')

    np.save(
        os.path.join(out_dir, 'peak_score_hex.npy'),
        {
            'score_centers_xy': np.asarray(hb_s.get_offsets()),
            'score_mean':       np.asarray(hb_s.get_array()),
            'sig_centers_xy':   np.asarray(hb_g.get_offsets()),
            'sig_mean':         np.asarray(hb_g.get_array()),
            'gridsize': gs, 'hex_px': HEX_PX,
            'window_half': WINDOW_HALF, 'cx': CX, 'cy': CY,
            'n_total_pooled': int(len(x)),
            'n_in_window':    int(m.sum()),
            'per_shot_x': xw, 'per_shot_y': yw,
            'per_shot_score': sw, 'per_shot_sig': gw,
            'kind': KIND,
        },
        allow_pickle=True,
    )
    print(f'wrote {os.path.join(out_dir, "peak_score_hex.npy")}')


if __name__ == '__main__':
    main()
