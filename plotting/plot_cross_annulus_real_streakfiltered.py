#!/usr/bin/env python3
"""Cross-annulus azimuthal covariance for streak-finder-filtered real data.

Same layout family as
  plotting/plot_cross_annulus_covariance_v2.py
  plotting/plot_cross_annulus_covariance_v2b.py
but the real-data pools are now subsets defined by the CVMI streak
finder result (`data_peak_positions_run_<N>.npy`), gated on:
  * significance > 2  (score sigma cut)
  * |r_est| > 5  px   (any-radius wiggle cut, per user request)
  * angle-fan filter (only for the ROI_R / ROI_L pools; the 'any'
    pool has no angle cut)

Three real pools per kind (duck, goose):
  - streak_R         : peak-angle in [-30°, +30°]
  - streak_L         : peak-angle in [150°, 180°] ∪ [-180°, -150°]
  - streak_any       : no angle cut (any streak direction)

Two dataset variants:
  - duck  (kind='duck', all (run, step) pairs present on disk)
  - goose (kind='goose', all (run, step) pairs present on disk)

Radial pairs (r_A_off, r_B_off) in px — same union as v2 + v2b:
  (-6,  0), (-4, +2), (-2, +4), (-4,  0), (-6, -2), (-2, +2)

Outputs into
  batch_analysis_results/cross_annulus_covariance_real_streakfiltered_more_cotimed_shots/

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
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from aggregate_4roi_late_steps import PRESETS

# Where the SLURM streak-finder wrote its per-(run, step) results.
BATCH_RESULTS_ROOT = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
                      'batch_analysis_results/results_145_to_152_4_ROI')
PICKLE_DIR = '/sdf/scratch/users/j/jinseop'

DIR_RE = re.compile(
    r'^circular_wiggler_(?P<kind>duck|goose)_run(?P<run>\d+)_step(?P<step>\d+)_batch_metrics$'
)

# Real-data timing groupings (see reference_lxt_convention.md + roi_timing_grouped_analysis.py:490):
#   'cotimed'          -> PRESETS['cotimed']  (|lxts| < 25 fs, 5 pairs)
#   'laser_early'      -> PRESETS['late']     (physical lxts > 0, 12 pairs)
#   'laser_late'       -> PRESETS['early']    (physical lxts < 0, 16 pairs)
# The label names use the physical laser-timing convention; the underlying
# PRESETS keys are inverted vs scan-order convention. Everything ELSE in the
# 'duck' universe (i.e. kind='duck' shots at (run, step) pairs not in any of
# these three PRESETS) is not part of this analysis.
DUCK_TIMING_PAIRS = {
    'duck_cotimed':      set((int(r), int(s)) for r, s in PRESETS['cotimed']),
    'duck_laser_early':  set((int(r), int(s)) for r, s in PRESETS['late']),
    'duck_laser_late':   set((int(r), int(s)) for r, s in PRESETS['early']),
}

# Streak-finder cuts (user request: sig > 2, |r| > 5; NO upper |r| bound).
SIG_CUT = 2.0
DR_MIN = 5.0
DR_MAX = np.inf

CX, CY = 67, 59
N_BINS = 36
DR_HALF = 3.0            # annulus half-width in px
RAW_CHANNELS_PER_BIN = 32
SPECTRUM_ROI_START = 13
SMOOTH_SIGMA_BINS = 1.5
GUIDES_DEG = (90.0, 270.0)

PAIRS = [(-6,  0), (-4, +2), (-2, +4), (-4,  0), (-6, -2), (-2, +2)]

OUT_DIR = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
           'batch_analysis_results/cross_annulus_covariance_real_streakfiltered_more_cotimed_shots')


def _roi_of_angle(angle_deg):
    """Match the ROI angle-fan convention used elsewhere in the pipeline
    (aggregate_4roi_by_energy.py:_roi_of_angle)."""
    if -30.0 <= angle_deg <= 30.0:
        return 'R'
    if angle_deg >= 150.0 or angle_deg <= -150.0:
        return 'L'
    if 60.0 <= angle_deg <= 120.0:
        return 'U'
    if -120.0 <= angle_deg <= -60.0:
        return 'D'
    return None


def enumerate_pairs():
    """Yield (kind, run, step) for every batch_metrics folder on disk."""
    for name in sorted(os.listdir(BATCH_RESULTS_ROOT)):
        m = DIR_RE.match(name)
        if not m:
            continue
        yield m.group('kind'), int(m.group('run')), int(m.group('step'))


def _load_estimates(kind, run, step):
    p = os.path.join(BATCH_RESULTS_ROOT,
                     f'circular_wiggler_{kind}_run{run}_step{step}_batch_metrics',
                     f'data_peak_positions_run_{run}.npy')
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True).item()


def _pickle_cache():
    cache = {}
    def get(run):
        if run in cache:
            return cache[run]
        p = os.path.join(PICKLE_DIR, f'preprocessed_run_{run}.pkl')
        print(f'  loading {p}')
        with open(p, 'rb') as f:
            cache[run] = pickle.load(f)
        return cache[run]
    return get


def _mask_shots(data, kind, step):
    step_axis = data['lxts']
    step_val = np.unique(step_axis)[step]
    return (data['masks'][kind].astype(bool)
            & data['is_gaussian'].astype(bool)
            & (step_axis == step_val))


def _annulus_mask(shape, cx, cy):
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    r_map = np.hypot(x - cx, y - cy)
    theta = np.arctan2(y - cy, x - cx) + np.pi
    theta_bin_map = np.clip(
        np.floor(theta / (2.0 * np.pi / N_BINS)).astype(int),
        0, N_BINS - 1,
    )
    return r_map, theta_bin_map


def _resolve_kind_dataset(kind_dataset):
    """Return (folder_kind, allowed_pairs_or_None) for a dataset label.

      - folder_kind: 'duck' or 'goose' -- matches the folder-name prefix.
      - allowed_pairs: None (no filter) or a set of (run, step) tuples the
        pool must be limited to.
    """
    if kind_dataset == 'goose':
        return 'goose', None
    if kind_dataset in DUCK_TIMING_PAIRS:
        return 'duck', DUCK_TIMING_PAIRS[kind_dataset]
    raise ValueError(f'unknown kind_dataset {kind_dataset!r}')


def collect_pool_theta(kind_dataset, roi_target, r_off, get):
    """For a given (kind_dataset, streak-ROI target), walk every
    (run, step) folder on disk that matches, apply
    sig>SIG_CUT & DR_MIN<|r|<DR_MAX & angle-in-target-fan, and build the
    per-shot azimuthal probability vector at annulus offset r_off.
    Returns H (N_shots, N_BINS)."""
    folder_kind, allowed = _resolve_kind_dataset(kind_dataset)
    rows = []
    for kind_dir, run, step in enumerate_pairs():
        if kind_dir != folder_kind:
            continue
        if allowed is not None and (run, step) not in allowed:
            continue
        est = _load_estimates(kind_dir, run, step)
        if est is None:
            continue
        data = get(run)
        mask = _mask_shots(data, kind=kind_dir, step=step)
        orig_idx = np.where(mask)[0]
        x_est = np.asarray(est['x'], dtype=float)
        y_est = np.asarray(est['y'], dtype=float)
        sig = np.asarray(est['significance'], dtype=float)
        n = min(len(orig_idx), len(x_est))
        if n == 0:
            continue
        oi = orig_idx[:n]; x_est = x_est[:n]; y_est = y_est[:n]; sig = sig[:n]

        dr = np.hypot(x_est - CX, y_est - CY)
        angle_deg = np.degrees(np.arctan2(y_est - CY, x_est - CX))

        # Precompute r/theta grids for this run's shape (all pickles are 125×125).
        r_map, theta_bin_map = _annulus_mask(data['hits'][oi[0]].shape, CX, CY)
        hits_run = data['hits']
        en_run = data['mean_energy']

        for k in range(n):
            if sig[k] <= SIG_CUT:
                continue
            if not (DR_MIN < dr[k] < DR_MAX):
                continue
            if roi_target is not None and _roi_of_angle(angle_deg[k]) != roi_target:
                continue
            img = hits_run[oi[k]].astype(np.float64, copy=False)
            total = img.sum()
            if total <= 0:
                continue
            prob = img / total
            re = (en_run[oi[k]] / RAW_CHANNELS_PER_BIN
                  - SPECTRUM_ROI_START) * 0.6 + 29.4 + r_off
            annulus = (r_map >= re - DR_HALF) & (r_map < re + DR_HALF)
            if not annulus.any():
                continue
            rows.append(np.bincount(
                theta_bin_map[annulus],
                weights=prob[annulus],
                minlength=N_BINS,
            ))
        print(f'    {kind_dataset} run {run} step {step} '
              f'(roi={roi_target or "any"}, r_off={r_off:+d}): {len(rows)} rows so far')
    return np.vstack(rows) if rows else np.empty((0, N_BINS))


def cross_covariance(H_A, H_B):
    n = H_A.shape[0]
    if n < 2 or H_B.shape[0] != n:
        raise ValueError(f'shots mismatch: {n} vs {H_B.shape[0]}')
    dA = H_A - H_A.mean(axis=0, keepdims=True)
    dB = H_B - H_B.mean(axis=0, keepdims=True)
    cov = (dB.T @ dA) / (n - 1)
    sA = dA.std(axis=0, ddof=1); sB = dB.std(axis=0, ddof=1)
    denom = np.outer(sB, sA); denom[denom == 0] = 1e-30
    corr = cov / denom
    return cov, corr


def zero_diag(M):
    diag = np.eye(M.shape[0], dtype=bool)
    return np.where(diag, np.nan, M)


def nan_gaussian_filter(A, sigma):
    A = np.asarray(A, dtype=float)
    nan_mask = np.isnan(A)
    V = np.where(nan_mask, 0.0, A); W = (~nan_mask).astype(float)
    Vs = gaussian_filter(V, sigma=sigma, mode='wrap')
    Ws = gaussian_filter(W, sigma=sigma, mode='wrap')
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(Ws > 1e-12, Vs / Ws, np.nan)


def draw(ax, M, title, vmax, xlabel='inner θ [deg]', ylabel='outer θ [deg]'):
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-30
    im = ax.imshow(M, cmap='RdBu_r', origin='lower',
                   extent=[0, 360, 0, 360],
                   vmin=-vmax, vmax=vmax, aspect='equal')
    for g in GUIDES_DEG:
        ax.axvline(g, color='k', ls=':', lw=0.7, alpha=0.55)
        ax.axhline(g, color='k', ls=':', lw=0.7, alpha=0.55)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def _pair_key(r_A, r_B):
    return f'r{r_A:+d}_vs_r{r_B:+d}'.replace('+', 'p').replace('-', 'm')


def collect_all_pools_by_r_off(kind_dataset, get):
    """For a given kind_dataset, precompute H at every unique r_off for
    the three streak pools. Returns nested dict pool -> r_off -> H."""
    unique_offsets = sorted({r for pair in PAIRS for r in pair})
    pools = {'streak_R': 'R', 'streak_L': 'L', 'streak_any': None}
    out = {}
    for pool_name, roi_target in pools.items():
        print(f'\n=== kind_dataset={kind_dataset}, pool={pool_name} ===')
        out[pool_name] = {}
        for r_off in unique_offsets:
            print(f'  r_off = {r_off:+d}')
            out[pool_name][r_off] = collect_pool_theta(
                kind_dataset, roi_target, r_off, get)
    return out


KIND_DATASETS_ALL = ['duck_cotimed', 'duck_laser_early', 'duck_laser_late', 'goose']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kinds', nargs='+', default=KIND_DATASETS_ALL,
                    choices=KIND_DATASETS_ALL)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    get = _pickle_cache()

    # Build H matrices per (kind_dataset, pool, r_off).
    H_by = {}
    for kind in args.kinds:
        H_by[kind] = collect_all_pools_by_r_off(kind, get)

    # Cross-cov per (kind_dataset, pool, pair).
    all_results = {}
    for kind in args.kinds:
        all_results[kind] = {}
        for pool in ('streak_R', 'streak_L', 'streak_any'):
            all_results[kind][pool] = {}
            for r_A, r_B in PAIRS:
                pair_key = _pair_key(r_A, r_B)
                H_A = H_by[kind][pool][r_A]
                H_B = H_by[kind][pool][r_B]
                if H_A.shape[0] < 2:
                    print(f'  [skip] {kind}/{pool}/{pair_key}: '
                          f'{H_A.shape[0]} shots (< 2)')
                    continue
                cov, corr = cross_covariance(H_A, H_B)
                cov_dz_sm = nan_gaussian_filter(zero_diag(cov), SMOOTH_SIGMA_BINS)
                corr_dz_sm = nan_gaussian_filter(zero_diag(corr), SMOOTH_SIGMA_BINS)
                all_results[kind][pool][pair_key] = {
                    'cov': cov, 'corr': corr,
                    'cov_dz_sm': cov_dz_sm, 'corr_dz_sm': corr_dz_sm,
                    'n_shots': H_A.shape[0],
                    'r_A': r_A, 'r_B': r_B,
                }
                print(f'  cov {kind}/{pool}/{pair_key}: '
                      f'N={H_A.shape[0]}  |cov_dz_sm|_max='
                      f'{np.nanmax(np.abs(cov_dz_sm)):.2e}')

    # Per-pair figures: 3 rows (pools) × N_kinds cols. Rows: streak_R,
    # streak_L, streak_any. Panels show diag-zero smoothed cov.
    pools_order = ['streak_R', 'streak_L', 'streak_any']
    for r_A, r_B in PAIRS:
        pair_key = _pair_key(r_A, r_B)
        cols = args.kinds
        fig, axes = plt.subplots(len(pools_order), len(cols),
                                  figsize=(3.7 * len(cols),
                                           3.7 * len(pools_order)),
                                  squeeze=False)
        # Shared vmax within each pair figure across every panel that has data.
        v_max = 0.0
        for kind in cols:
            for pool in pools_order:
                r = all_results[kind][pool].get(pair_key)
                if r is not None:
                    v_max = max(v_max, float(np.nanmax(np.abs(r['cov_dz_sm']))))
        v_max = v_max or 1e-30
        xlab = f'θ_inner (r={r_A:+d}) [deg]'
        ylab = f'θ_outer (r={r_B:+d}) [deg]'
        for r_idx, pool in enumerate(pools_order):
            for c_idx, kind in enumerate(cols):
                r = all_results[kind][pool].get(pair_key)
                ax = axes[r_idx, c_idx]
                if r is None:
                    ax.axis('off')
                    ax.set_title(f'{kind} / {pool}\n(no shots)', fontsize=9)
                    continue
                draw(ax, r['cov_dz_sm'],
                     f'{kind}  /  {pool}  |  N={r["n_shots"]}\n'
                     f'(r={r_A:+d} vs {r_B:+d}, diag=0)',
                     v_max, xlab, ylab)
        fig.suptitle(
            f'Real-data streak-filtered cross-annulus covariance  |  '
            f'inner r_A={r_A:+d}, outer r_B={r_B:+d} px  '
            f'(sig>2, |r|>5; smoothed σ={SMOOTH_SIGMA_BINS:g} bins, mode=wrap)',
            fontweight='bold', fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.955])
        p = os.path.join(OUT_DIR, f'cross_covariance_diagzero_{pair_key}.png')
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f'wrote {p}')

        # Diff figures per pair: each duck sub-pool minus goose per streak pool.
        # Grid: 3 rows (streak pools) × 3 cols (duck_cotimed, duck_laser_early,
        # duck_laser_late) — only if 'goose' is in the kind set.
        if 'goose' in args.kinds:
            duck_subs = [k for k in ('duck_cotimed', 'duck_laser_early',
                                     'duck_laser_late') if k in args.kinds]
            if duck_subs:
                fig, axes = plt.subplots(
                    len(pools_order), len(duck_subs),
                    figsize=(3.9 * len(duck_subs), 3.9 * len(pools_order)),
                    squeeze=False,
                )
                diffs_grid = {}
                v_diff = 0.0
                for r_idx, pool in enumerate(pools_order):
                    g_res = all_results['goose'][pool].get(pair_key)
                    for c_idx, d_kind in enumerate(duck_subs):
                        d_res = all_results[d_kind][pool].get(pair_key)
                        if d_res is None or g_res is None:
                            continue
                        diff = d_res['cov'] - g_res['cov']
                        diff_sm = nan_gaussian_filter(zero_diag(diff),
                                                       SMOOTH_SIGMA_BINS)
                        diffs_grid[(r_idx, c_idx, pool, d_kind)] = (
                            diff_sm, d_res['n_shots'], g_res['n_shots']
                        )
                        v_diff = max(v_diff, float(np.nanmax(np.abs(diff_sm))))
                v_diff = v_diff or 1e-30
                for r_idx, pool in enumerate(pools_order):
                    for c_idx, d_kind in enumerate(duck_subs):
                        ax = axes[r_idx, c_idx]
                        key = (r_idx, c_idx, pool, d_kind)
                        if key not in diffs_grid:
                            ax.axis('off'); continue
                        diff_sm, n_d, n_g = diffs_grid[key]
                        draw(ax, diff_sm,
                             f'{d_kind} − goose  ({pool})\n'
                             f'N_d={n_d}, N_g={n_g}, diag=0',
                             v_diff, xlab, ylab)
                fig.suptitle(
                    f'Real-data streak-filtered diffs (duck subpool − goose)  |  '
                    f'inner r_A={r_A:+d}, outer r_B={r_B:+d} px  '
                    f'(sig>2, |r|>5)',
                    fontweight='bold', fontsize=11,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.955])
                p = os.path.join(OUT_DIR,
                                 f'cross_covariance_diffs_diagzero_{pair_key}.png')
                fig.savefig(p, dpi=150); plt.close(fig)
                print(f'wrote {p}')

    # --- Summary grids: one per streak pool.
    # Rows = pairs; cols = duck_cotimed, duck_laser_early, duck_laser_late,
    # goose, then three diffs (duck_X − goose). Per-row shared vmax across
    # the cov columns; independent per diff column.
    for pool in pools_order:
        duck_subs = [k for k in ('duck_cotimed', 'duck_laser_early',
                                 'duck_laser_late') if k in args.kinds]
        has_goose = 'goose' in args.kinds
        cols_spec = []  # list of ('kind', kind_name)  or  ('diff', duck_sub)
        for k in duck_subs:
            cols_spec.append(('kind', k))
        if has_goose:
            cols_spec.append(('kind', 'goose'))
        if has_goose:
            for k in duck_subs:
                cols_spec.append(('diff', k))
        if not cols_spec:
            continue
        n_cols = len(cols_spec)
        fig, axes = plt.subplots(len(PAIRS), n_cols,
                                  figsize=(3.3 * n_cols, 3.3 * len(PAIRS)),
                                  squeeze=False)
        for r_idx, (r_A, r_B) in enumerate(PAIRS):
            pair_key = _pair_key(r_A, r_B)
            v_cov = 0.0
            for kind_col, val in cols_spec:
                if kind_col == 'diff':
                    continue
                r = all_results[val][pool].get(pair_key)
                if r is not None:
                    v_cov = max(v_cov, float(np.nanmax(np.abs(r['cov_dz_sm']))))
            v_cov = v_cov or 1e-30
            xlab = f'θ_inner (r={r_A:+d}) [deg]'
            ylab = f'θ_outer (r={r_B:+d}) [deg]'
            for c_idx, (kind_col, val) in enumerate(cols_spec):
                ax = axes[r_idx, c_idx]
                if kind_col == 'diff':
                    d_res = all_results[val][pool].get(pair_key)
                    g_res = all_results['goose'][pool].get(pair_key)
                    if d_res is None or g_res is None:
                        ax.axis('off'); continue
                    diff = d_res['cov'] - g_res['cov']
                    M = nan_gaussian_filter(zero_diag(diff), SMOOTH_SIGMA_BINS)
                    vmax = float(np.nanmax(np.abs(M))) or 1e-30
                    draw(ax, M,
                         f'{val} − goose  ({pool})\n'
                         f'(r={r_A:+d} vs {r_B:+d})',
                         vmax, xlab, ylab)
                else:
                    r = all_results[val][pool].get(pair_key)
                    if r is None:
                        ax.axis('off'); continue
                    draw(ax, r['cov_dz_sm'],
                         f'{val}  ({pool})  |  N={r["n_shots"]}\n'
                         f'(r={r_A:+d} vs {r_B:+d}, diag=0)',
                         v_cov, xlab, ylab)
        fig.suptitle(
            f'Real-data streak-filtered ({pool}) cross-annulus covariance  '
            f'(sig>2, |r|>5; smoothed σ={SMOOTH_SIGMA_BINS:g} bins, mode=wrap)',
            fontweight='bold', fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        p = os.path.join(OUT_DIR, f'summary_cross_covariance_{pool}.png')
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f'wrote {p}')

    # -----------------------------------------------------------------
    # SELF-COVARIANCE (same-annulus): produce covariance at each single
    # r_off with H_A == H_B, diagonal zeroed. Same layout family as the
    # cross-annulus outputs above.
    # -----------------------------------------------------------------
    SELF_OFFSETS = [-6, -4, -2, 0]
    self_results = {}
    for kind in args.kinds:
        self_results[kind] = {}
        for pool in ('streak_R', 'streak_L', 'streak_any'):
            self_results[kind][pool] = {}
            for r_off in SELF_OFFSETS:
                H = H_by[kind][pool][r_off]
                if H.shape[0] < 2:
                    print(f'  [skip self] {kind}/{pool}/r{r_off:+d}: '
                          f'{H.shape[0]} shots')
                    continue
                cov, corr = cross_covariance(H, H)   # self => symmetric
                cov_dz_sm = nan_gaussian_filter(zero_diag(cov), SMOOTH_SIGMA_BINS)
                corr_dz_sm = nan_gaussian_filter(zero_diag(corr), SMOOTH_SIGMA_BINS)
                self_results[kind][pool][r_off] = {
                    'cov': cov, 'corr': corr,
                    'cov_dz_sm': cov_dz_sm, 'corr_dz_sm': corr_dz_sm,
                    'n_shots': H.shape[0],
                }
                print(f'  self-cov {kind}/{pool}/r{r_off:+d}: '
                      f'N={H.shape[0]}  |cov_dz_sm|_max='
                      f'{np.nanmax(np.abs(cov_dz_sm)):.2e}')

    # Per-offset figures: 3 rows (pools) × N_kinds cols.
    for r_off in SELF_OFFSETS:
        off_key = f'r{r_off:+d}'.replace('+', 'p').replace('-', 'm')
        cols = args.kinds
        fig, axes = plt.subplots(len(pools_order), len(cols),
                                  figsize=(3.7 * len(cols),
                                           3.7 * len(pools_order)),
                                  squeeze=False)
        v_max = 0.0
        for kind in cols:
            for pool in pools_order:
                r = self_results[kind][pool].get(r_off)
                if r is not None:
                    v_max = max(v_max, float(np.nanmax(np.abs(r['cov_dz_sm']))))
        v_max = v_max or 1e-30
        xlab = f'θ_j (r={r_off:+d}) [deg]'
        ylab = f'θ_i (r={r_off:+d}) [deg]'
        for r_idx, pool in enumerate(pools_order):
            for c_idx, kind in enumerate(cols):
                r = self_results[kind][pool].get(r_off)
                ax = axes[r_idx, c_idx]
                if r is None:
                    ax.axis('off')
                    ax.set_title(f'{kind} / {pool}\n(no shots)', fontsize=9)
                    continue
                draw(ax, r['cov_dz_sm'],
                     f'{kind}  /  {pool}  |  N={r["n_shots"]}\n'
                     f'r={r_off:+d}  self-cov (diag=0)',
                     v_max, xlab, ylab)
        fig.suptitle(
            f'Real-data streak-filtered self-annulus covariance  |  '
            f'r_off={r_off:+d} px  '
            f'(sig>2, |r|>5; smoothed σ={SMOOTH_SIGMA_BINS:g} bins, mode=wrap)',
            fontweight='bold', fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.955])
        p = os.path.join(OUT_DIR, f'self_covariance_diagzero_{off_key}.png')
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f'wrote {p}')

        # Diff figure per offset: each duck sub-pool minus goose per streak pool.
        if 'goose' in args.kinds:
            duck_subs = [k for k in ('duck_cotimed', 'duck_laser_early',
                                     'duck_laser_late') if k in args.kinds]
            if duck_subs:
                fig, axes = plt.subplots(
                    len(pools_order), len(duck_subs),
                    figsize=(3.9 * len(duck_subs), 3.9 * len(pools_order)),
                    squeeze=False,
                )
                diffs_grid = {}
                v_diff = 0.0
                for r_idx, pool in enumerate(pools_order):
                    g_res = self_results['goose'][pool].get(r_off)
                    for c_idx, d_kind in enumerate(duck_subs):
                        d_res = self_results[d_kind][pool].get(r_off)
                        if d_res is None or g_res is None:
                            continue
                        diff = d_res['cov'] - g_res['cov']
                        diff_sm = nan_gaussian_filter(zero_diag(diff),
                                                       SMOOTH_SIGMA_BINS)
                        diffs_grid[(r_idx, c_idx, pool, d_kind)] = (
                            diff_sm, d_res['n_shots'], g_res['n_shots']
                        )
                        v_diff = max(v_diff, float(np.nanmax(np.abs(diff_sm))))
                v_diff = v_diff or 1e-30
                for r_idx, pool in enumerate(pools_order):
                    for c_idx, d_kind in enumerate(duck_subs):
                        ax = axes[r_idx, c_idx]
                        key = (r_idx, c_idx, pool, d_kind)
                        if key not in diffs_grid:
                            ax.axis('off'); continue
                        diff_sm, n_d, n_g = diffs_grid[key]
                        draw(ax, diff_sm,
                             f'{d_kind} − goose  ({pool})\n'
                             f'N_d={n_d}, N_g={n_g}, r={r_off:+d}, diag=0',
                             v_diff, xlab, ylab)
                fig.suptitle(
                    f'Real-data streak-filtered self-cov diffs (duck subpool − goose)  |  '
                    f'r_off={r_off:+d} px  (sig>2, |r|>5)',
                    fontweight='bold', fontsize=11,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.955])
                p = os.path.join(OUT_DIR, f'self_covariance_diffs_diagzero_{off_key}.png')
                fig.savefig(p, dpi=150); plt.close(fig)
                print(f'wrote {p}')

    # Summary grid per streak pool: rows = offsets, cols = same layout as
    # cross-cov summary (duck subpools, goose, then three diffs vs goose).
    for pool in pools_order:
        duck_subs = [k for k in ('duck_cotimed', 'duck_laser_early',
                                 'duck_laser_late') if k in args.kinds]
        has_goose = 'goose' in args.kinds
        cols_spec = []
        for k in duck_subs:
            cols_spec.append(('kind', k))
        if has_goose:
            cols_spec.append(('kind', 'goose'))
        if has_goose:
            for k in duck_subs:
                cols_spec.append(('diff', k))
        if not cols_spec:
            continue
        n_cols = len(cols_spec)
        fig, axes = plt.subplots(len(SELF_OFFSETS), n_cols,
                                  figsize=(3.3 * n_cols, 3.3 * len(SELF_OFFSETS)),
                                  squeeze=False)
        for r_idx, r_off in enumerate(SELF_OFFSETS):
            v_cov = 0.0
            for kind_col, val in cols_spec:
                if kind_col == 'diff':
                    continue
                r = self_results[val][pool].get(r_off)
                if r is not None:
                    v_cov = max(v_cov, float(np.nanmax(np.abs(r['cov_dz_sm']))))
            v_cov = v_cov or 1e-30
            xlab = f'θ_j (r={r_off:+d}) [deg]'
            ylab = f'θ_i (r={r_off:+d}) [deg]'
            for c_idx, (kind_col, val) in enumerate(cols_spec):
                ax = axes[r_idx, c_idx]
                if kind_col == 'diff':
                    d_res = self_results[val][pool].get(r_off)
                    g_res = self_results['goose'][pool].get(r_off)
                    if d_res is None or g_res is None:
                        ax.axis('off'); continue
                    diff = d_res['cov'] - g_res['cov']
                    M = nan_gaussian_filter(zero_diag(diff), SMOOTH_SIGMA_BINS)
                    vmax = float(np.nanmax(np.abs(M))) or 1e-30
                    draw(ax, M,
                         f'{val} − goose  ({pool})\n(r={r_off:+d}, diag=0)',
                         vmax, xlab, ylab)
                else:
                    r = self_results[val][pool].get(r_off)
                    if r is None:
                        ax.axis('off'); continue
                    draw(ax, r['cov_dz_sm'],
                         f'{val}  ({pool})  |  N={r["n_shots"]}\n'
                         f'r={r_off:+d} self-cov (diag=0)',
                         v_cov, xlab, ylab)
        fig.suptitle(
            f'Real-data streak-filtered ({pool}) self-annulus covariance  '
            f'(sig>2, |r|>5; smoothed σ={SMOOTH_SIGMA_BINS:g} bins, mode=wrap)',
            fontweight='bold', fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        p = os.path.join(OUT_DIR, f'summary_self_covariance_{pool}.png')
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f'wrote {p}')

    # Save every raw + smoothed matrix + shot counts.
    save = {}
    for kind, pool_res in all_results.items():
        for pool, pair_res in pool_res.items():
            for pair_key, r in pair_res.items():
                p_ = f'{kind}__{pool}__{pair_key}'
                save[f'{p_}__cov'] = r['cov']
                save[f'{p_}__corr'] = r['corr']
                save[f'{p_}__cov_dz_sm'] = np.nan_to_num(r['cov_dz_sm'])
                save[f'{p_}__corr_dz_sm'] = np.nan_to_num(r['corr_dz_sm'])
                save[f'{p_}__n_shots'] = r['n_shots']
    # Self-cov matrices too.
    for kind, pool_res in self_results.items():
        for pool, off_res in pool_res.items():
            for r_off, r in off_res.items():
                off_key = f'r{r_off:+d}'.replace('+', 'p').replace('-', 'm')
                p_ = f'{kind}__{pool}__self__{off_key}'
                save[f'{p_}__cov'] = r['cov']
                save[f'{p_}__corr'] = r['corr']
                save[f'{p_}__cov_dz_sm'] = np.nan_to_num(r['cov_dz_sm'])
                save[f'{p_}__corr_dz_sm'] = np.nan_to_num(r['corr_dz_sm'])
                save[f'{p_}__n_shots'] = r['n_shots']
    np.savez_compressed(os.path.join(OUT_DIR, 'cross_covariance.npz'),
                        pairs=np.array([f'{a:+d}_{b:+d}' for a, b in PAIRS]),
                        self_offsets=np.array(SELF_OFFSETS),
                        sig_cut=SIG_CUT,
                        dr_min=DR_MIN,
                        **save)
    print(f'wrote {os.path.join(OUT_DIR, "cross_covariance.npz")}')


if __name__ == '__main__':
    main()
