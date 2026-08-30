#!/usr/bin/env python3
"""
For each LXT step of run 145 (duck), plot peak-score-vs-peak-position and
significance-vs-peak-position hex-bin heatmaps.

Cell metric = MEAN of the score (or significance) over all shots whose
`(peak_x, peak_y)` falls inside that hex bin. Hex bin extent ≈ 5 px on
the detector (matplotlib hexbin uses `gridsize`; for a ±15 px window
that means gridsize ≈ 6).

Shot pool: all base-masked duck shots  (masks['duck'] & is_gaussian & lxts=step).
No significance / ROI cut — this is a map of "where does the wiggle
finder point, and how confident is it?".

Data sources:
  - `data_peak_positions_run_<N>.npy` → x, y, significance
  - `data_score_vs_hits_run_<N>.npy`  → scores (per-shot peak_score),
                                        aligned with peak_positions
Output tree (one folder per step under the stepwise root):
    run145_stepwise_peak_sigma_y/step<NN>/peak_score_hex.png
                                        /peak_score_hex.npy
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


RUN = 145
KIND = 'duck'
BATCH_RESULTS_ROOT = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
                     'batch_analysis_results/results_145_to_152_4_ROI')
STEPWISE_ROOT = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
                 'batch_analysis_results/legacy_outputs/run145_stepwise_peak_sigma_y')
CX, CY = 67, 59
WINDOW_HALF = 15.0     # ± px around (cx, cy)
HEX_PX = 3.0           # target hex bin extent in detector px
MIN_MINCNT = 5         # require at least this many shots per hex


def _load_step(step_idx):
    """Load per-shot (peak_x, peak_y, peak_score, significance) for one step."""
    d = os.path.join(BATCH_RESULTS_ROOT,
                     f'circular_wiggler_{KIND}_run{RUN}_step{step_idx}_batch_metrics')
    p_pos = os.path.join(d, f'data_peak_positions_run_{RUN}.npy')
    p_sh  = os.path.join(d, f'data_score_vs_hits_run_{RUN}.npy')
    if not (os.path.exists(p_pos) and os.path.exists(p_sh)):
        return None
    pos = np.load(p_pos, allow_pickle=True).item()
    sh  = np.load(p_sh, allow_pickle=True).item()
    x = np.asarray(pos['x'], dtype=float)
    y = np.asarray(pos['y'], dtype=float)
    sig = np.asarray(pos['significance'], dtype=float)
    scores = np.asarray(sh['scores'], dtype=float)
    n = min(len(x), len(scores))
    return x[:n], y[:n], sig[:n], scores[:n]


def _lxt_of_step(step_idx):
    """Read the lxt value from the sigma_y npy if available, else return None."""
    p = os.path.join(STEPWISE_ROOT, f'step{step_idx:02d}', 'sigma_y_ROI_R.npy')
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True).item()
    return float(d.get('lxt_s', None)) if d.get('lxt_s') is not None else None


def _lxt_label(lxt_s):
    if lxt_s is None:
        return '?'
    if abs(lxt_s) < 25e-15:
        return 'cotimed'
    return 'laser late' if lxt_s < 0 else 'laser early'


def _hex_gridsize():
    """Number of hex bins along x for the ±WINDOW_HALF window at HEX_PX size."""
    return max(3, int(round(2 * WINDOW_HALF / HEX_PX)))


def process_step(step_idx, out_root):
    r = _load_step(step_idx)
    if r is None:
        print(f'  step {step_idx}: no source files — skipping')
        return None
    x, y, sig, score = r
    n_all = len(x)
    lxt_s = _lxt_of_step(step_idx)
    lxt_lbl = _lxt_label(lxt_s)
    lxt_fs = f'{lxt_s*1e15:+g} fs' if lxt_s is not None else '?'

    # Window mask — keep only shots inside the display window.
    m = ((np.abs(x - CX) <= WINDOW_HALF)
         & (np.abs(y - CY) <= WINDOW_HALF)
         & np.isfinite(score) & np.isfinite(sig))
    xw = x[m]; yw = y[m]; sw = score[m]; gw = sig[m]
    n_in = int(m.sum())
    print(f'  step {step_idx}: {n_all} shots pipelined, '
          f'{n_in} within ±{WINDOW_HALF:g} px of ({CX}, {CY})')

    out_dir = os.path.join(out_root, f'step{step_idx:02d}')
    os.makedirs(out_dir, exist_ok=True)

    gs = _hex_gridsize()
    x_range = (CX - WINDOW_HALF, CX + WINDOW_HALF)
    y_range = (CY - WINDOW_HALF, CY + WINDOW_HALF)

    fig, (ax_s, ax_g) = plt.subplots(1, 2, figsize=(15, 6.6))

    # ---- Panel 1: mean peak_score per hex ----
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
    ax_s.set_xlim(x_range); ax_s.set_ylim(y_range)
    ax_s.set_xlabel('peak x [px]'); ax_s.set_ylabel('peak y [px]')
    ax_s.set_aspect('equal')
    ax_s.set_title(f'mean peak_score  |  {n_in} shots  |  hex ≈ {HEX_PX:g} px')
    cb = fig.colorbar(hb_s, ax=ax_s, fraction=0.046, pad=0.03)
    cb.set_label('mean peak_score')

    # ---- Panel 2: mean significance per hex ----
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
    ax_g.set_xlim(x_range); ax_g.set_ylim(y_range)
    ax_g.set_xlabel('peak x [px]'); ax_g.set_ylabel('peak y [px]')
    ax_g.set_aspect('equal')
    ax_g.set_title(f'mean significance [σ]  |  {n_in} shots  |  hex ≈ {HEX_PX:g} px')
    cb2 = fig.colorbar(hb_g, ax=ax_g, fraction=0.046, pad=0.03)
    cb2.set_label('mean significance')

    fig.suptitle(
        f'run {RUN} duck  step {step_idx}  (lxt = {lxt_fs}, {lxt_lbl})  |  '
        f'peak_score and significance vs peak position\n'
        f'lime dashed = r=3 px, white dotted = r=10 px  |  mincnt={MIN_MINCNT}',
        fontweight='bold',
    )
    fig.tight_layout()
    out_png = os.path.join(out_dir, 'peak_score_hex.png')
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f'    → {out_png}')

    # Save the raw hex data (counts per bin + mean value per bin) alongside.
    # matplotlib hexbin.get_offsets() returns hex centers; get_array() returns
    # the reduced value (mean here) per bin.
    hb_data = {
        'score_centers_xy': np.asarray(hb_s.get_offsets()),
        'score_mean':       np.asarray(hb_s.get_array()),
        'sig_centers_xy':   np.asarray(hb_g.get_offsets()),
        'sig_mean':         np.asarray(hb_g.get_array()),
        'gridsize': gs, 'hex_px': HEX_PX,
        'window_half': WINDOW_HALF, 'cx': CX, 'cy': CY,
        'n_all': n_all, 'n_in': n_in,
        'lxt_s': lxt_s, 'step_idx': step_idx, 'run': RUN,
        'per_shot_x': xw, 'per_shot_y': yw,
        'per_shot_score': sw, 'per_shot_sig': gw,
    }
    np.save(os.path.join(out_dir, 'peak_score_hex.npy'), hb_data, allow_pickle=True)
    return {'step_idx': step_idx, 'n_in': n_in, 'out_png': out_png}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, nargs='+', default=None,
                    help='step indices to process (default: all 20)')
    ap.add_argument('--out_root', default=STEPWISE_ROOT)
    args = ap.parse_args()

    steps = args.steps if args.steps is not None else list(range(20))
    for s in steps:
        process_step(s, args.out_root)


if __name__ == '__main__':
    main()
