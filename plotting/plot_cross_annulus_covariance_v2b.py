#!/usr/bin/env python3
"""Additional cross-annulus plots on the v2 dataset pool.

Extends plot_cross_annulus_covariance_v2.py with three additions:

  1. Zero the DIAGONAL of every cross-covariance / cross-correlation
     matrix before smoothing.  Even though the diagonal of a cross-cov
     (between two different annuli) is not self-cov, it often carries
     a strong "same-θ different-radius" signal that dominates the
     colorscale and hides off-diagonal features. Zeroing exposes the
     off-diagonal structure.

  2. New radial pairs symmetric about the photoline (r_off = 0):
       (-6, +0)   inner background vs photoline
       (-4, +2)   shallow-inside  vs shallow-outside
       (-2, +4)   photoline near-side  vs outer wing
     ...alongside the original set. Same-pair keys are unique so both
     scripts' outputs coexist in the folder.

  3. NEW: full sigma_theta sweep for every radial pair from the
     streaked SLURM datasets (run6+9+10 have σ_θ ∈ {0,5,10,15,20};
     ~60 000 shots each). One figure per pair with 5 σ-panels showing
     the smoothed diag-zeroed cross-cov, all sharing a vmax.

Outputs into
  batch_analysis_results/cross_annulus_covariance_v2/
    cross_covariance_diagzero_<pair>.png                (all datasets)
    cross_covariance_diffs_diagzero_<pair>.png          (diffs)
    summary_cross_covariance_diagzero.png               (all pairs, diag-zero)
    cross_covariance_sigma_sweep_<pair>.png             (σ_θ x pair, streaked)
    cross_covariance_v2b.npz

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
sys.path.insert(0, os.path.dirname(_HERE))
from aggregate_4roi_late_steps import PRESETS

LEGACY = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
          'batch_analysis_results/legacy_outputs')

STREAKED_RUNS = ['circular_wiggler_run6', 'circular_wiggler_run9',
                 'circular_wiggler_run10']
UNSTREAKED_RUNS = ['circular_wiggler_run11', 'circular_wiggler_run12',
                   'circular_wiggler_run13']
SIGMA_VALUES = [0, 5, 10, 15, 20]

PICKLE_DIR = '/sdf/scratch/users/j/jinseop'
GOOSE_RUNS_REAL = [145, 146, 147, 148, 149, 150, 152]
COTIMED_PAIRS = PRESETS['cotimed']
LASER_LATE_PAIRS = PRESETS['early']

CX, CY = 67, 59
N_BINS = 36
DR_HALF = 3.0
RAW_CHANNELS_PER_BIN = 32
SPECTRUM_ROI_START = 13
SMOOTH_SIGMA_BINS = 1.5
GUIDES_DEG = (90.0, 270.0)

# Original v2 pairs + new v2b pairs. Same-key figures are prefixed
# 'diagzero' to disambiguate from v2's diagonal-included outputs.
PAIRS = [(-6,  0), (-4, +2), (-2, +4), (-4,  0), (-6, -2), (-2, +2)]

OUT_DIR = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
           'batch_analysis_results/cross_annulus_covariance_v2')


# ------------------------------------------------------------------
# Loaders (identical to v2 script)
# ------------------------------------------------------------------
def _skip_if_bad(d, fn, run_tag, expected_n=20000, want_rayleigh_zero=None):
    n = d['hits'].shape[0]
    if n < expected_n // 2:
        print(f'  [skip] {run_tag}/{fn}: {n} shots (partial)'); return True
    if want_rayleigh_zero is not None:
        max_streak = float(d['streak_radius_true_px'].max())
        if want_rayleigh_zero and max_streak > 0:
            print(f'  [skip] {run_tag}/{fn}: expected Rayleigh=0 but streak_r max={max_streak}')
            return True
        if (not want_rayleigh_zero) and max_streak == 0:
            print(f'  [skip] {run_tag}/{fn}: expected streaked but streak_r max=0')
            return True
    return False


def load_streaked_sigma(sigma):
    hs, es = [], []
    for run in STREAKED_RUNS:
        p = os.path.join(LEGACY, run, 'wiggler_sigma_sweep_metrics',
                         f'shot_hits_full_sigma_{sigma}deg.npz')
        if not os.path.exists(p):
            print(f'  [miss] {run}/shot_hits_full_sigma_{sigma}deg.npz')
            continue
        d = np.load(p)
        if _skip_if_bad(d, os.path.basename(p), run, want_rayleigh_zero=False):
            continue
        hs.append(d['hits']); es.append(d['mean_energy'])
    return (np.concatenate(hs, axis=0) if hs else np.empty((0, 140, 140)),
            np.concatenate(es, axis=0) if es else np.empty(0))


def load_unstreaked():
    hs, es = [], []
    for run in UNSTREAKED_RUNS:
        run_dir = os.path.join(LEGACY, run, 'wiggler_sigma_sweep_metrics')
        if not os.path.isdir(run_dir):
            continue
        for fn in sorted(os.listdir(run_dir)):
            if not fn.startswith('shot_hits_full_sigma_') or not fn.endswith('.npz'):
                continue
            d = np.load(os.path.join(run_dir, fn))
            if _skip_if_bad(d, fn, run, want_rayleigh_zero=True):
                continue
            hs.append(d['hits']); es.append(d['mean_energy'])
    return np.concatenate(hs, axis=0), np.concatenate(es, axis=0)


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


def _indices_goose(d):
    return np.where(d['masks']['goose'].astype(bool)
                    & d['is_gaussian'].astype(bool))[0]


def _indices_duck_step(d, step):
    step_axis = d['lxts']
    step_val = np.unique(step_axis)[step]
    return np.where(d['masks']['duck'].astype(bool)
                    & d['is_gaussian'].astype(bool)
                    & (step_axis == step_val))[0]


def _prob_matrix_indices(data, idx, r_off):
    hits, en = data['hits'], data['mean_energy']
    ny, nx = hits[0].shape
    y, x = np.mgrid[0:ny, 0:nx]
    r_map = np.hypot(x - CX, y - CY)
    theta = np.arctan2(y - CY, x - CX) + np.pi
    theta_bin_map = np.clip(
        np.floor(theta / (2.0 * np.pi / N_BINS)).astype(int),
        0, N_BINS - 1,
    )
    rows = np.empty((idx.size, N_BINS), dtype=float)
    for row_i, i in enumerate(idx):
        img = hits[i].astype(np.float64, copy=False)
        total = img.sum()
        if total <= 0:
            rows[row_i] = 0.0; continue
        prob = img / total
        re = (en[i] / RAW_CHANNELS_PER_BIN
              - SPECTRUM_ROI_START) * 0.6 + 29.4 + r_off
        annulus = (r_map >= re - DR_HALF) & (r_map < re + DR_HALF)
        if not annulus.any():
            rows[row_i] = 0.0; continue
        rows[row_i] = np.bincount(
            theta_bin_map[annulus],
            weights=prob[annulus],
            minlength=N_BINS,
        )
    return rows


def real_theta_matrix(pool, r_off, get):
    rows = []
    if pool == 'goose':
        for run in GOOSE_RUNS_REAL:
            d = get(run)
            idx = _indices_goose(d)
            rows.append(_prob_matrix_indices(d, idx, r_off))
    else:
        pairs = COTIMED_PAIRS if pool == 'cotimed' else LASER_LATE_PAIRS
        for run, step in pairs:
            d = get(run)
            idx = _indices_duck_step(d, step)
            rows.append(_prob_matrix_indices(d, idx, r_off))
    return np.vstack(rows) if rows else np.empty((0, N_BINS))


def azimuthal_prob_matrix(hits, mean_energy, r_off):
    ny, nx = hits.shape[1:]
    y, x = np.mgrid[0:ny, 0:nx]
    r_map = np.hypot(x - CX, y - CY)
    theta = np.arctan2(y - CY, x - CX) + np.pi
    theta_bin_map = np.clip(
        np.floor(theta / (2.0 * np.pi / N_BINS)).astype(int),
        0, N_BINS - 1,
    )
    n = hits.shape[0]
    rows = np.empty((n, N_BINS), dtype=float)
    for i in range(n):
        img = hits[i].astype(np.float64, copy=False)
        total = img.sum()
        if total <= 0:
            rows[i] = 0.0; continue
        prob = img / total
        re = (mean_energy[i] / RAW_CHANNELS_PER_BIN
              - SPECTRUM_ROI_START) * 0.6 + 29.4 + r_off
        annulus = (r_map >= re - DR_HALF) & (r_map < re + DR_HALF)
        if not annulus.any():
            rows[i] = 0.0; continue
        rows[i] = np.bincount(
            theta_bin_map[annulus],
            weights=prob[annulus],
            minlength=N_BINS,
        )
    return rows


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
    """Return a copy of M with the diagonal replaced by NaN so
    nan_gaussian_filter can renormalize weights around it."""
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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-real', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Load bootstrap datasets ---
    print('=== streaked (run6+9+10 σ=0) ===')
    hits_s0, en_s0 = load_streaked_sigma(0)
    print(f'  N = {hits_s0.shape[0]}')

    print('=== unstreaked (run11+12+13) ===')
    hits_u, en_u = load_unstreaked()
    print(f'  N = {hits_u.shape[0]}')

    real_pools = [] if args.no_real else ['goose', 'cotimed', 'laser_late']
    real_get = None if args.no_real else _pickle_cache()

    # Precompute per-dataset H matrices at every unique r_off we need.
    unique_offsets = sorted({r for pair in PAIRS for r in pair})
    print(f'  unique r_offsets needed: {unique_offsets}')

    H = {
        'streaked': {r: azimuthal_prob_matrix(hits_s0, en_s0, r) for r in unique_offsets},
        'unstreaked': {r: azimuthal_prob_matrix(hits_u, en_u, r) for r in unique_offsets},
    }
    n_shots = {
        'streaked': hits_s0.shape[0], 'unstreaked': hits_u.shape[0],
    }
    for pool in real_pools:
        H[f'real_{pool}'] = {r: real_theta_matrix(pool, r, real_get) for r in unique_offsets}
        n_shots[f'real_{pool}'] = H[f'real_{pool}'][unique_offsets[0]].shape[0]
        print(f'  real_{pool}: {n_shots[f"real_{pool}"]} shots')

    # --- Compute diag-zeroed cross-cov / corr per (pair, dataset) ---
    dataset_names = ['streaked', 'unstreaked'] + [f'real_{p}' for p in real_pools]
    all_results = {}
    for r_A, r_B in PAIRS:
        pair_key = _pair_key(r_A, r_B)
        print(f'\n--- pair {pair_key} (r_A={r_A}, r_B={r_B}) ---')
        pair_res = {}
        for name in dataset_names:
            H_A = H[name][r_A]; H_B = H[name][r_B]
            cov, corr = cross_covariance(H_A, H_B)
            pair_res[name] = {
                'cov': cov, 'corr': corr,
                'cov_dz_sm': nan_gaussian_filter(zero_diag(cov), SMOOTH_SIGMA_BINS),
                'corr_dz_sm': nan_gaussian_filter(zero_diag(corr), SMOOTH_SIGMA_BINS),
                'n_shots': H_A.shape[0],
            }
        # diffs (smoothed, on diag-zeroed raw)
        diffs = {}
        if 'streaked' in pair_res and 'unstreaked' in pair_res:
            diffs['streaked_minus_unstreaked'] = nan_gaussian_filter(
                zero_diag(pair_res['streaked']['cov']
                          - pair_res['unstreaked']['cov']),
                SMOOTH_SIGMA_BINS)
        if 'real_cotimed' in pair_res and 'real_goose' in pair_res:
            diffs['cotimed_minus_goose'] = nan_gaussian_filter(
                zero_diag(pair_res['real_cotimed']['cov']
                          - pair_res['real_goose']['cov']),
                SMOOTH_SIGMA_BINS)
        if 'real_laser_late' in pair_res and 'real_goose' in pair_res:
            diffs['laser_late_minus_goose'] = nan_gaussian_filter(
                zero_diag(pair_res['real_laser_late']['cov']
                          - pair_res['real_goose']['cov']),
                SMOOTH_SIGMA_BINS)
        pair_res['_diffs'] = diffs
        all_results[(r_A, r_B)] = pair_res

        # Figure: 2 rows × N cols (row 0 = cov diag-zero sm; row 1 = corr diag-zero sm)
        panels = dataset_names
        n_cols = len(panels)
        fig, axes = plt.subplots(2, n_cols, figsize=(3.6 * n_cols, 7.4))
        v_cov = max(float(np.nanmax(np.abs(pair_res[n]['cov_dz_sm']))) for n in panels) or 1e-30
        v_corr = max(float(np.nanmax(np.abs(pair_res[n]['corr_dz_sm']))) for n in panels) or 1e-30
        xlab = f'θ_inner (r={r_A:+d}) [deg]'
        ylab = f'θ_outer (r={r_B:+d}) [deg]'
        for c, name in enumerate(panels):
            r = pair_res[name]
            draw(axes[0, c], r['cov_dz_sm'],
                 f'{name}\ncross-cov (diag=0, smoothed σ={SMOOTH_SIGMA_BINS:g})\n'
                 f'N={r["n_shots"]}',
                 v_cov, xlab, ylab)
            draw(axes[1, c], r['corr_dz_sm'],
                 f'cross-corr (diag=0, smoothed)', v_corr, xlab, ylab)
        fig.suptitle(
            f'Diag-zero cross-annulus covariance  |  inner r_A={r_A:+d}, '
            f'outer r_B={r_B:+d} px  (probability-normalized, mode=wrap)',
            fontweight='bold', fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.955])
        p = os.path.join(OUT_DIR, f'cross_covariance_diagzero_{pair_key}.png')
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f'  wrote {p}')

        # Per-pair diffs (diag-zero)
        if diffs:
            fig, axes = plt.subplots(1, len(diffs),
                                      figsize=(4.2 * len(diffs), 4.6),
                                      squeeze=False)
            v_diff = max(float(np.nanmax(np.abs(v))) for v in diffs.values()) or 1e-30
            titles = {
                'streaked_minus_unstreaked':
                    'bootstrap:  streaked − unstreaked (diag=0)',
                'cotimed_minus_goose':
                    'real:  cotimed − goose (diag=0)',
                'laser_late_minus_goose':
                    'real:  laser_late − goose (diag=0)',
            }
            for c, name in enumerate(diffs):
                draw(axes[0, c], diffs[name], titles.get(name, name),
                     v_diff, xlab, ylab)
            fig.suptitle(
                f'Diag-zero cross-annulus diffs  |  inner r_A={r_A:+d}, '
                f'outer r_B={r_B:+d} px',
                fontweight='bold', fontsize=11,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            p = os.path.join(OUT_DIR, f'cross_covariance_diffs_diagzero_{pair_key}.png')
            fig.savefig(p, dpi=150); plt.close(fig)
            print(f'  wrote {p}')

    # --- Summary grid (diag-zero) ---
    col_specs = [
        ('streaked',                   'streaked'),
        ('unstreaked',                 'unstreaked'),
        ('_streaked_minus_unstreaked', 'streaked − unstreaked'),
    ]
    if real_pools:
        col_specs += [
            ('real_goose',              'real goose'),
            ('real_cotimed',            'real cotimed'),
            ('real_laser_late',         'real laser_late'),
            ('_cotimed_minus_goose',    'real cotimed − goose'),
            ('_laser_late_minus_goose', 'real laser_late − goose'),
        ]
    n_cols = len(col_specs)
    fig, axes = plt.subplots(len(PAIRS), n_cols,
                              figsize=(3.4 * n_cols, 3.4 * len(PAIRS)))
    if len(PAIRS) == 1:
        axes = axes[np.newaxis, :]
    for r_idx, (r_A, r_B) in enumerate(PAIRS):
        pair_res = all_results[(r_A, r_B)]
        cov_sm_names = [key for key, _ in col_specs if not key.startswith('_')]
        v_cov = 0.0
        for k in cov_sm_names:
            if k in pair_res:
                v_cov = max(v_cov, float(np.nanmax(np.abs(pair_res[k]['cov_dz_sm']))))
        v_cov = v_cov or 1e-30
        xlab = f'θ_inner (r={r_A:+d}) [deg]'
        ylab = f'θ_outer (r={r_B:+d}) [deg]'
        for c_idx, (key, title) in enumerate(col_specs):
            ax = axes[r_idx, c_idx]
            if key.startswith('_'):
                diff_key = key[1:]  # strip leading underscore
                M = pair_res['_diffs'].get(diff_key)
                if M is None:
                    ax.axis('off'); continue
                vmax = float(np.nanmax(np.abs(M))) or 1e-30
                sub = ''
            else:
                if key not in pair_res:
                    ax.axis('off'); continue
                M = pair_res[key]['cov_dz_sm']
                vmax = v_cov
                sub = f' | N={pair_res[key]["n_shots"]}'
            draw(ax, M, f'{title}{sub}\n(r={r_A:+d} vs {r_B:+d}, diag=0)',
                 vmax, xlab, ylab)
    fig.suptitle(
        f'Diag-zero cross-annulus covariance summary  '
        f'(smoothed σ={SMOOTH_SIGMA_BINS:g} bins, mode=wrap)',
        fontweight='bold', fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    p = os.path.join(OUT_DIR, 'summary_cross_covariance_diagzero.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'wrote {p}')

    # --- Sigma-theta sweep at every radial pair (streaked datasets only) ---
    # For each pair, load H at r_A and r_B for every sigma, build 5-panel
    # figure (one per sigma_theta), diag-zero smoothed. Shared vmax across
    # the 5 panels per pair. Reuse the sigma=0 stack we already loaded.
    print('\n=== sigma_theta sweep at each pair (streaked only) ===')
    # Per-sigma H caches to avoid recomputation across pairs.
    H_sigma = {0: H['streaked']}  # sigma=0 already cached
    sigma_covs = {}  # (pair, sigma) -> smoothed diag-zero cov
    sigma_ncts = {}  # sigma -> n_shots
    sigma_ncts[0] = n_shots['streaked']
    for sigma in SIGMA_VALUES:
        if sigma not in H_sigma:
            print(f'  loading streaked σ_θ={sigma}° ...')
            hits, en = load_streaked_sigma(sigma)
            sigma_ncts[sigma] = hits.shape[0]
            H_sigma[sigma] = {r: azimuthal_prob_matrix(hits, en, r) for r in unique_offsets}
            del hits, en   # free memory
        for r_A, r_B in PAIRS:
            H_A = H_sigma[sigma][r_A]; H_B = H_sigma[sigma][r_B]
            cov, _ = cross_covariance(H_A, H_B)
            sigma_covs[((r_A, r_B), sigma)] = nan_gaussian_filter(
                zero_diag(cov), SMOOTH_SIGMA_BINS)
        print(f'  σ_θ={sigma}° done')

    # One figure per pair: 1 row × 5 cols (one per sigma), shared vmax.
    for r_A, r_B in PAIRS:
        pair_key = _pair_key(r_A, r_B)
        v_max = max(float(np.nanmax(np.abs(sigma_covs[((r_A, r_B), s)])))
                    for s in SIGMA_VALUES) or 1e-30
        ncols = len(SIGMA_VALUES)
        fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3.9),
                                  squeeze=False)
        xlab = f'θ_inner (r={r_A:+d}) [deg]'
        ylab = f'θ_outer (r={r_B:+d}) [deg]'
        for c, s in enumerate(SIGMA_VALUES):
            draw(axes[0, c], sigma_covs[((r_A, r_B), s)],
                 f'σ_θ = {s}°  |  N = {sigma_ncts[s]}\n(r={r_A:+d} vs {r_B:+d}, diag=0)',
                 v_max, xlab, ylab)
        fig.suptitle(
            f'Streaked bootstrap sigma_theta sweep  |  inner r_A={r_A:+d}, '
            f'outer r_B={r_B:+d} px  (probability-normalized, smoothed '
            f'σ={SMOOTH_SIGMA_BINS:g} bins, mode=wrap, diag=0)',
            fontweight='bold', fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        p = os.path.join(OUT_DIR, f'cross_covariance_sigma_sweep_{pair_key}.png')
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f'  wrote {p}')

    # Grand sigma-sweep summary: pairs × sigmas (all in one figure).
    fig, axes = plt.subplots(len(PAIRS), len(SIGMA_VALUES),
                              figsize=(3.4 * len(SIGMA_VALUES),
                                       3.4 * len(PAIRS)))
    if len(PAIRS) == 1:
        axes = axes[np.newaxis, :]
    v_global = max(float(np.nanmax(np.abs(sigma_covs[k]))) for k in sigma_covs) or 1e-30
    for r_idx, (r_A, r_B) in enumerate(PAIRS):
        for c_idx, s in enumerate(SIGMA_VALUES):
            draw(axes[r_idx, c_idx], sigma_covs[((r_A, r_B), s)],
                 f'σ_θ = {s}°  |  r={r_A:+d} vs {r_B:+d}',
                 v_global,
                 f'θ_inner (r={r_A:+d}) [deg]',
                 f'θ_outer (r={r_B:+d}) [deg]')
    fig.suptitle(
        f'Streaked bootstrap sigma_theta sweep across all radial pairs  '
        f'(diag=0 smoothed σ={SMOOTH_SIGMA_BINS:g} bins, '
        'shared vmax across all panels)',
        fontweight='bold', fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    p = os.path.join(OUT_DIR, 'summary_cross_covariance_sigma_sweep.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'wrote {p}')

    # --- Save every matrix ---
    save = {}
    for (r_A, r_B), pair_res in all_results.items():
        pair_key = _pair_key(r_A, r_B)
        for name, r in pair_res.items():
            if name == '_diffs':
                for d_name, d_val in r.items():
                    save[f'{pair_key}__diff_{d_name}_dz_sm'] = np.nan_to_num(d_val)
                continue
            save[f'{pair_key}__{name}__cov'] = r['cov']
            save[f'{pair_key}__{name}__corr'] = r['corr']
            save[f'{pair_key}__{name}__cov_dz_sm'] = np.nan_to_num(r['cov_dz_sm'])
            save[f'{pair_key}__{name}__corr_dz_sm'] = np.nan_to_num(r['corr_dz_sm'])
            save[f'{pair_key}__{name}__n_shots'] = r['n_shots']
    for (r_A, r_B), s in [((rp, s), s)
                          for rp in [(r_A, r_B) for (r_A, r_B) in PAIRS]
                          for s in SIGMA_VALUES]:
        pass  # placeholder; below writes the σ sweep cache properly
    # Flat σ-sweep matrices.
    for (pair, sigma), M in sigma_covs.items():
        pkey = _pair_key(*pair)
        save[f'{pkey}__streaked_sigma{sigma}__cov_dz_sm'] = np.nan_to_num(M)
    np.savez_compressed(os.path.join(OUT_DIR, 'cross_covariance_v2b.npz'),
                        pairs=np.array([f'{a:+d}_{b:+d}' for a, b in PAIRS]),
                        sigma_values=np.array(SIGMA_VALUES),
                        **save)
    print(f'wrote {os.path.join(OUT_DIR, "cross_covariance_v2b.npz")}')


if __name__ == '__main__':
    main()
