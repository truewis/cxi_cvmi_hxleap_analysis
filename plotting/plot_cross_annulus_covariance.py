#!/usr/bin/env python3
"""Cross-annulus azimuthal covariance across three datasets.

For each dataset, compute per-shot azimuthal probability vectors at two
different annulus radii (H_A, H_B) and their sample cross-covariance
C[i, j] = cov(H_B[:, i], H_A[:, j]) / (n - 1) — so the y-axis indexes
the "outer" annulus theta bin and the x-axis indexes the "inner"
annulus theta bin.

Datasets
--------
- streaked_60k    : run6 + run9 + run10 sigma_theta=0 shots (60 000 total,
                    RAYLEIGH_SCALE=5 px)
- unstreaked_40k  : run11 + run12 Rayleigh=0 sigma_theta=0 shots (40 000)
- real            : goose (all runs/steps, ~7 000 shots) + cotimed
                    (PRESETS['cotimed'] pairs, ~2 800 shots). Same
                    selection as plot_real_covariance_goose_vs_cotimed.py

Pairs (r_off_A, r_off_B) [px]
-----------------------------
- ( -4, 0)
- ( -6, -2)
- ( -2, +2)

Layout
------
One figure per pair: 3 rows × N_ds cols. Rows = {raw cross-cov, smoothed
cross-cov, Pearson cross-correlation}. Each panel shows the (N_BINS×N_BINS)
matrix with guides at theta = 90° and 270°.

Additionally, a summary grid: (n_pairs × 4 col) with (streaked smoothed,
unstreaked smoothed, streaked - unstreaked smoothed diff, real
smoothed) per pair.

Outputs into
  batch_analysis_results/cross_annulus_covariance/

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
# Add batch_analysis/ to sys.path so we can import aggregate_4roi_late_steps.
sys.path.insert(0, os.path.dirname(_HERE))

from aggregate_4roi_late_steps import PRESETS

LEGACY = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
          'batch_analysis_results/legacy_outputs')

STREAKED_RUNS = ['circular_wiggler_run6', 'circular_wiggler_run9',
                 'circular_wiggler_run10']
UNSTREAKED_RUNS = ['circular_wiggler_run11', 'circular_wiggler_run12']

PICKLE_DIR = '/sdf/scratch/users/j/jinseop'
GOOSE_RUNS_REAL = [145, 146, 147, 148, 149, 150, 152]
COTIMED_PAIRS = PRESETS['cotimed']

CX, CY = 67, 59
N_BINS = 36
DR_HALF = 3.0
RAW_CHANNELS_PER_BIN = 32
SPECTRUM_ROI_START = 13
SMOOTH_SIGMA_BINS = 1.5
GUIDES_DEG = (90.0, 270.0)

PAIRS = [(-4, 0), (-6, -2), (-2, +2)]

OUT_DIR = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
           'batch_analysis_results/cross_annulus_covariance')


def load_streaked_sigma0():
    """Return concatenated (hits, mean_energy) for sigma_theta=0 across
    STREAKED_RUNS (60 000 shots at RAYLEIGH_SCALE=5)."""
    hs, es = [], []
    for run in STREAKED_RUNS:
        p = os.path.join(LEGACY, run, 'wiggler_sigma_sweep_metrics',
                         'shot_hits_full_sigma_0deg.npz')
        d = np.load(p)
        hs.append(d['hits'])
        es.append(d['mean_energy'])
        print(f'  streaked {run}: {d["hits"].shape[0]} shots')
    return np.concatenate(hs, axis=0), np.concatenate(es, axis=0)


def load_unstreaked():
    """Return concatenated (hits, mean_energy) for Rayleigh=0 across
    UNSTREAKED_RUNS (40 000 shots). Filters partial/streaked stacks."""
    hs, es = [], []
    for run in UNSTREAKED_RUNS:
        run_dir = os.path.join(LEGACY, run, 'wiggler_sigma_sweep_metrics')
        for fn in sorted(os.listdir(run_dir)):
            if not fn.startswith('shot_hits_full_sigma_') or not fn.endswith('.npz'):
                continue
            d = np.load(os.path.join(run_dir, fn))
            n = d['hits'].shape[0]
            if n < 10_000:
                print(f'  [skip] {run}/{fn}: {n} shots (partial)')
                continue
            max_streak = float(d['streak_radius_true_px'].max())
            if max_streak > 0:
                print(f'  [skip] {run}/{fn}: streak_r max={max_streak}')
                continue
            hs.append(d['hits'])
            es.append(d['mean_energy'])
            print(f'  unstreaked {run}/{fn}: {n} shots')
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


def _real_indices_goose(data):
    return np.where(data['masks']['goose'].astype(bool)
                    & data['is_gaussian'].astype(bool))[0]


def _real_indices_cotimed(data, step):
    step_axis = data['lxts']
    step_val = np.unique(step_axis)[step]
    return np.where(data['masks']['duck'].astype(bool)
                    & data['is_gaussian'].astype(bool)
                    & (step_axis == step_val))[0]


def real_theta_matrix(r_off):
    """Build per-shot theta vector matrix for the combined real pool
    (goose across GOOSE_RUNS_REAL + cotimed at COTIMED_PAIRS) at
    annulus offset r_off. Returns (H_goose, H_cotimed)."""
    get = _pickle_cache()
    rows_g, rows_c = [], []
    for run in GOOSE_RUNS_REAL:
        d = get(run)
        idx = _real_indices_goose(d)
        rows_g.append(_azimuthal_prob_matrix_indices(d, idx, r_off))
        print(f'  goose run {run} r_off={r_off:+d}: {idx.size} shots')
    for run, step in COTIMED_PAIRS:
        d = get(run)
        idx = _real_indices_cotimed(d, step)
        rows_c.append(_azimuthal_prob_matrix_indices(d, idx, r_off))
        print(f'  cotimed run {run} step {step} r_off={r_off:+d}: '
              f'{idx.size} shots')
    return (np.vstack(rows_g) if rows_g else np.empty((0, N_BINS)),
            np.vstack(rows_c) if rows_c else np.empty((0, N_BINS)))


def _azimuthal_prob_matrix_indices(data, idx, r_off):
    """Same math as azimuthal_prob_matrix but indexes into a preloaded
    pickle's hits + mean_energy arrays. (Used by real_theta_matrix.)"""
    hits = data['hits']
    en = data['mean_energy']
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
            rows[row_i] = 0.0
            continue
        prob = img / total
        re = (en[i] / RAW_CHANNELS_PER_BIN
              - SPECTRUM_ROI_START) * 0.6 + 29.4 + r_off
        annulus = (r_map >= re - DR_HALF) & (r_map < re + DR_HALF)
        if not annulus.any():
            rows[row_i] = 0.0
            continue
        rows[row_i] = np.bincount(
            theta_bin_map[annulus],
            weights=prob[annulus],
            minlength=N_BINS,
        )
    return rows


def azimuthal_prob_matrix(hits, mean_energy, r_off):
    """Standalone version for synthetic stacks (hits shape (N, ny, nx))."""
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
            rows[i] = 0.0
            continue
        prob = img / total
        re = (mean_energy[i] / RAW_CHANNELS_PER_BIN
              - SPECTRUM_ROI_START) * 0.6 + 29.4 + r_off
        annulus = (r_map >= re - DR_HALF) & (r_map < re + DR_HALF)
        if not annulus.any():
            rows[i] = 0.0
            continue
        rows[i] = np.bincount(
            theta_bin_map[annulus],
            weights=prob[annulus],
            minlength=N_BINS,
        )
    return rows


def cross_covariance(H_A, H_B):
    """Return (cross_cov, cross_corr) of shape (N_BINS, N_BINS).
    C[i, j] = cov(H_B[:, i], H_A[:, j]) / (n - 1)."""
    n = H_A.shape[0]
    if n < 2 or H_B.shape[0] != n:
        raise ValueError(f'shots mismatch: {n} vs {H_B.shape[0]}')
    dA = H_A - H_A.mean(axis=0, keepdims=True)
    dB = H_B - H_B.mean(axis=0, keepdims=True)
    cov = (dB.T @ dA) / (n - 1)
    sA = dA.std(axis=0, ddof=1)
    sB = dB.std(axis=0, ddof=1)
    denom = np.outer(sB, sA)
    denom[denom == 0] = 1e-30
    corr = cov / denom
    return cov, corr


def nan_gaussian_filter(A, sigma):
    A = np.asarray(A, dtype=float)
    nan_mask = np.isnan(A)
    V = np.where(nan_mask, 0.0, A)
    W = (~nan_mask).astype(float)
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
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def load_all_datasets(no_real=False):
    print('--- loading streaked (run6+9+10 sigma_theta=0) ---')
    hits_s, en_s = load_streaked_sigma0()
    print(f'  streaked total: {hits_s.shape[0]} shots')

    print('--- loading unstreaked (run11+12) ---')
    hits_u, en_u = load_unstreaked()
    print(f'  unstreaked total: {hits_u.shape[0]} shots')

    real = None
    if not no_real:
        print('--- loading real (goose all + cotimed all) ---')
        # We build H at each r_off on demand below (needs r_off arg).
        real = 'lazy'   # marker; concatenation happens per-pair inside main
    return {
        'streaked': (hits_s, en_s),
        'unstreaked': (hits_u, en_u),
        'real': real,
    }


def build_pair_matrices(datasets, r_A, r_B):
    """Return dict dataset_name -> (H_A, H_B, label) with matched shots."""
    out = {}
    for name in ('streaked', 'unstreaked'):
        hits, en = datasets[name]
        H_A = azimuthal_prob_matrix(hits, en, r_A)
        H_B = azimuthal_prob_matrix(hits, en, r_B)
        out[name] = (H_A, H_B, f'{name}  |  N = {hits.shape[0]}')
    if datasets.get('real') is not None:
        Hg_A, Hc_A = real_theta_matrix(r_A)
        Hg_B, Hc_B = real_theta_matrix(r_B)
        out['real_goose'] = (Hg_A, Hg_B, f'real goose  |  N = {Hg_A.shape[0]}')
        out['real_cotimed'] = (Hc_A, Hc_B, f'real cotimed  |  N = {Hc_A.shape[0]}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-real', action='store_true',
                    help='Skip real-data pools (faster; useful for debugging).')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    datasets = load_all_datasets(no_real=args.no_real)

    # Precompute cross-cov + corr per (pair, dataset).
    all_results = {}
    for r_A, r_B in PAIRS:
        pair_key = f'r{r_A:+d}_vs_r{r_B:+d}'.replace('+', 'p').replace('-', 'm')
        print(f'--- pair {pair_key} (r_A={r_A}, r_B={r_B}) ---')
        matrices = build_pair_matrices(datasets, r_A, r_B)
        pair_res = {}
        for name, (H_A, H_B, label) in matrices.items():
            cov, corr = cross_covariance(H_A, H_B)
            cov_sm = nan_gaussian_filter(cov, SMOOTH_SIGMA_BINS)
            corr_sm = nan_gaussian_filter(corr, SMOOTH_SIGMA_BINS)
            pair_res[name] = {
                'cov': cov, 'corr': corr,
                'cov_sm': cov_sm, 'corr_sm': corr_sm,
                'label': label, 'n_shots': H_A.shape[0],
            }
        all_results[(r_A, r_B)] = pair_res

        # One figure per pair: (3 rows) × (n_ds cols)
        names = list(pair_res.keys())
        ncols = len(names)
        fig, axes = plt.subplots(3, ncols, figsize=(3.6 * ncols, 10.5))
        v_cov = max(float(np.nanmax(np.abs(pair_res[n]['cov'])))
                    for n in names) or 1e-30
        v_cov_sm = max(float(np.nanmax(np.abs(pair_res[n]['cov_sm'])))
                        for n in names) or 1e-30
        v_corr_sm = max(float(np.nanmax(np.abs(pair_res[n]['corr_sm'])))
                         for n in names) or 1e-30
        for c, n_ in enumerate(names):
            r = pair_res[n_]
            xlabel = f'θ_inner (r_A = {r_A:+d} px) [deg]'
            ylabel = f'θ_outer (r_B = {r_B:+d} px) [deg]'
            draw(axes[0, c], r['cov'],
                 f'{r["label"]}\ncross-cov (raw)', v_cov,
                 xlabel=xlabel, ylabel=ylabel)
            draw(axes[1, c], r['cov_sm'],
                 f'cross-cov (smoothed σ={SMOOTH_SIGMA_BINS:g} bins)',
                 v_cov_sm, xlabel=xlabel, ylabel=ylabel)
            draw(axes[2, c], r['corr_sm'],
                 'cross-corr (smoothed)', v_corr_sm,
                 xlabel=xlabel, ylabel=ylabel)
        fig.suptitle(
            f'Cross-annulus azimuthal covariance  |  '
            f'inner r_A = re + {r_A:+d} px, outer r_B = re + {r_B:+d} px  '
            f'(probability-normalized, mode=wrap, diag NOT masked -- '
            'cross matrices have no self-cov diagonal)',
            fontweight='bold', fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.955])
        p = os.path.join(OUT_DIR, f'cross_covariance_{pair_key}.png')
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f'  wrote {p}')

    # ---- Summary grid: pairs × 4 columns (streaked, unstreaked, diff, real cotimed) ----
    ncols = 5  # streaked_sm, unstreaked_sm, streaked-unstreaked_sm, real_goose_sm, real_cotimed_sm
    dataset_columns = ['streaked', 'unstreaked', 'streaked_minus_unstreaked',
                       'real_goose', 'real_cotimed']
    if args.no_real:
        ncols = 3
        dataset_columns = ['streaked', 'unstreaked', 'streaked_minus_unstreaked']

    fig, axes = plt.subplots(len(PAIRS), ncols,
                              figsize=(3.6 * ncols, 3.6 * len(PAIRS)))
    for r_idx, (r_A, r_B) in enumerate(PAIRS):
        pair_res = all_results[(r_A, r_B)]
        s_sm = pair_res['streaked']['cov_sm']
        u_sm = pair_res['unstreaked']['cov_sm']
        diff_sm = nan_gaussian_filter(
            pair_res['streaked']['cov'] - pair_res['unstreaked']['cov'],
            SMOOTH_SIGMA_BINS)
        # Per-row shared vmax for cov columns; independent vmax for diff / real.
        v_cov = max(float(np.nanmax(np.abs(s_sm))),
                    float(np.nanmax(np.abs(u_sm)))) or 1e-30
        v_diff = float(np.nanmax(np.abs(diff_sm))) or 1e-30
        xlabel = f'θ_inner (r={r_A:+d}) [deg]'
        ylabel = f'θ_outer (r={r_B:+d}) [deg]'
        draw(axes[r_idx, 0], s_sm,
             f'streaked (r={r_A:+d} vs {r_B:+d})\nN={pair_res["streaked"]["n_shots"]}',
             v_cov, xlabel=xlabel, ylabel=ylabel)
        draw(axes[r_idx, 1], u_sm,
             f'unstreaked (r={r_A:+d} vs {r_B:+d})\nN={pair_res["unstreaked"]["n_shots"]}',
             v_cov, xlabel=xlabel, ylabel=ylabel)
        draw(axes[r_idx, 2], diff_sm,
             f'streaked − unstreaked  (r={r_A:+d} vs {r_B:+d})',
             v_diff, xlabel=xlabel, ylabel=ylabel)
        if not args.no_real:
            g = pair_res['real_goose']['cov_sm']
            c = pair_res['real_cotimed']['cov_sm']
            v_real = max(float(np.nanmax(np.abs(g))),
                         float(np.nanmax(np.abs(c)))) or 1e-30
            draw(axes[r_idx, 3], g,
                 f'real goose (r={r_A:+d} vs {r_B:+d})\n'
                 f'N={pair_res["real_goose"]["n_shots"]}',
                 v_real, xlabel=xlabel, ylabel=ylabel)
            draw(axes[r_idx, 4], c,
                 f'real cotimed (r={r_A:+d} vs {r_B:+d})\n'
                 f'N={pair_res["real_cotimed"]["n_shots"]}',
                 v_real, xlabel=xlabel, ylabel=ylabel)
    fig.suptitle(
        'Cross-annulus azimuthal covariance summary across datasets and radial pairs '
        f'(smoothed σ={SMOOTH_SIGMA_BINS:g} bins, mode=wrap)',
        fontweight='bold', fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    p = os.path.join(OUT_DIR, 'summary_cross_covariance.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'wrote {p}')

    # Save every raw + smoothed matrix + shot counts.
    save_dict = {}
    for (r_A, r_B), pair_res in all_results.items():
        pair_key = f'r{r_A:+d}_vs_r{r_B:+d}'.replace('+', 'p').replace('-', 'm')
        for name, r in pair_res.items():
            save_dict[f'{pair_key}__{name}__cov'] = r['cov']
            save_dict[f'{pair_key}__{name}__corr'] = r['corr']
            save_dict[f'{pair_key}__{name}__cov_sm'] = np.nan_to_num(r['cov_sm'])
            save_dict[f'{pair_key}__{name}__corr_sm'] = np.nan_to_num(r['corr_sm'])
            save_dict[f'{pair_key}__{name}__n_shots'] = r['n_shots']
    np.savez_compressed(os.path.join(OUT_DIR, 'cross_covariance.npz'),
                        pairs=np.array([f'{a:+d}_{b:+d}' for a, b in PAIRS]),
                        **save_dict)
    print(f'wrote {os.path.join(OUT_DIR, "cross_covariance.npz")}')


if __name__ == '__main__':
    main()
