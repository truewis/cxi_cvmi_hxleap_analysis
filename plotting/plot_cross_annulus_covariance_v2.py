#!/usr/bin/env python3
"""Cross-annulus azimuthal covariance across datasets and radial pairs.

Extension of plot_cross_annulus_covariance.py:
  1. Unstreaked pool now pulls run11 + run12 + run13 (60 000 shots)
     to match the streaked pool size (run6+9+10, 60 000 shots at
     sigma_theta = 0).
  2. Real-data pools expanded: goose (all), cotimed (PRESETS['cotimed'],
     duck), and laser_late (PRESETS['early'], duck).  See
     roi_timing_grouped_analysis.py for the label ↔ PRESETS mapping.
  3. Emits per-dataset cross-covariance plots (one per radial pair),
     the streaked − unstreaked bootstrap diff, AND real-data
     cotimed − goose, laser_late − goose diffs.

Datasets
--------
- streaked_60k     : run6 + run9 + run10, σ_θ=0, RAYLEIGH_SCALE=5 px
- unstreaked_60k   : run11 + run12 + run13, RAYLEIGH_SCALE=0
- real_goose       : masks['goose'] & is_gaussian over runs
                     {145,146,147,148,149,150,152}
- real_cotimed     : masks['duck'] & is_gaussian at PRESETS['cotimed']
                     pairs
- real_laser_late  : masks['duck'] & is_gaussian at PRESETS['early']
                     pairs (physical lxts < 0)

Radial pairs (r_A_off, r_B_off) in px
-------------------------------------
- (-4,  0)   inner background vs photoline
- (-6, -2)   inner background vs shallow-inside
- (-2, +2)   shallow-inside vs shallow-outside

Outputs into
  batch_analysis_results/cross_annulus_covariance_v2/

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

PICKLE_DIR = '/sdf/scratch/users/j/jinseop'
GOOSE_RUNS_REAL = [145, 146, 147, 148, 149, 150, 152]
COTIMED_PAIRS = PRESETS['cotimed']
# roi_timing_grouped_analysis.py:490 -- 'laser_late' physically maps to
# PRESETS['early'] (labels are inverted vs scan-order convention).
LASER_LATE_PAIRS = PRESETS['early']

CX, CY = 67, 59
N_BINS = 36
DR_HALF = 3.0
RAW_CHANNELS_PER_BIN = 32
SPECTRUM_ROI_START = 13
SMOOTH_SIGMA_BINS = 1.5
GUIDES_DEG = (90.0, 270.0)

PAIRS = [(-4, 0), (-6, -2), (-2, +2)]

OUT_DIR = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
           'batch_analysis_results/cross_annulus_covariance_v2')


def _skip_if_bad(d, fn, run_tag, expected_n=20000, want_rayleigh_zero=None):
    n = d['hits'].shape[0]
    if n < expected_n // 2:
        print(f'  [skip] {run_tag}/{fn}: {n} shots (partial/smoke)')
        return True
    if want_rayleigh_zero is not None:
        max_streak = float(d['streak_radius_true_px'].max())
        if want_rayleigh_zero and max_streak > 0:
            print(f'  [skip] {run_tag}/{fn}: expected Rayleigh=0 but streak_r max={max_streak}')
            return True
        if (not want_rayleigh_zero) and max_streak == 0:
            print(f'  [skip] {run_tag}/{fn}: expected streaked but streak_r max=0')
            return True
    return False


def load_streaked_sigma0():
    """Return concatenated (hits, mean_energy) for sigma_theta=0 across STREAKED_RUNS."""
    hs, es = [], []
    for run in STREAKED_RUNS:
        p = os.path.join(LEGACY, run, 'wiggler_sigma_sweep_metrics',
                         'shot_hits_full_sigma_0deg.npz')
        d = np.load(p)
        if _skip_if_bad(d, os.path.basename(p), run, want_rayleigh_zero=False):
            continue
        hs.append(d['hits']); es.append(d['mean_energy'])
        print(f'  streaked {run}: {d["hits"].shape[0]} shots')
    return np.concatenate(hs, axis=0), np.concatenate(es, axis=0)


def load_unstreaked():
    """Return concatenated (hits, mean_energy) for Rayleigh=0 stacks."""
    hs, es = [], []
    for run in UNSTREAKED_RUNS:
        run_dir = os.path.join(LEGACY, run, 'wiggler_sigma_sweep_metrics')
        if not os.path.isdir(run_dir):
            print(f'  [skip] {run}: no wiggler_sigma_sweep_metrics/')
            continue
        for fn in sorted(os.listdir(run_dir)):
            if not fn.startswith('shot_hits_full_sigma_') or not fn.endswith('.npz'):
                continue
            d = np.load(os.path.join(run_dir, fn))
            if _skip_if_bad(d, fn, run, want_rayleigh_zero=True):
                continue
            hs.append(d['hits']); es.append(d['mean_energy'])
            print(f'  unstreaked {run}/{fn}: {d["hits"].shape[0]} shots')
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


def _indices_goose(data):
    return np.where(data['masks']['goose'].astype(bool)
                    & data['is_gaussian'].astype(bool))[0]


def _indices_duck_at_step(data, step):
    step_axis = data['lxts']
    step_val = np.unique(step_axis)[step]
    return np.where(data['masks']['duck'].astype(bool)
                    & data['is_gaussian'].astype(bool)
                    & (step_axis == step_val))[0]


def _azimuthal_prob_matrix_indices(data, idx, r_off):
    """Build the per-shot azimuthal probability matrix from a preloaded
    pickle's arrays; indices `idx` select the shot subset."""
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
    """Build H shape (N_shots, N_BINS) at annulus offset r_off for the
    specified pool label. pool ∈ {'goose', 'cotimed', 'laser_late'}."""
    rows = []
    if pool == 'goose':
        for run in GOOSE_RUNS_REAL:
            d = get(run)
            idx = _indices_goose(d)
            rows.append(_azimuthal_prob_matrix_indices(d, idx, r_off))
            print(f'    goose run {run} r_off={r_off:+d}: {idx.size} shots')
    else:
        pairs = COTIMED_PAIRS if pool == 'cotimed' else LASER_LATE_PAIRS
        for run, step in pairs:
            d = get(run)
            if d is None:
                continue
            idx = _indices_duck_at_step(d, step)
            rows.append(_azimuthal_prob_matrix_indices(d, idx, r_off))
            print(f'    {pool} run {run} step {step} r_off={r_off:+d}: {idx.size} shots')
    return np.vstack(rows) if rows else np.empty((0, N_BINS))


def azimuthal_prob_matrix(hits, mean_energy, r_off):
    """Same math as _azimuthal_prob_matrix_indices but takes numpy arrays."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-real', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Bootstrap datasets loaded once ---
    print('=== loading streaked (run6+9+10 σ=0) ===')
    hits_s, en_s = load_streaked_sigma0()
    print(f'  total streaked: {hits_s.shape[0]}')
    print('=== loading unstreaked (run11+12+13) ===')
    hits_u, en_u = load_unstreaked()
    print(f'  total unstreaked: {hits_u.shape[0]}')

    real_get = None if args.no_real else _pickle_cache()
    real_pools = [] if args.no_real else ['goose', 'cotimed', 'laser_late']

    # --- Per pair: build every dataset's (H_A, H_B) and covariances ---
    all_results = {}
    for r_A, r_B in PAIRS:
        pair_key = f'r{r_A:+d}_vs_r{r_B:+d}'.replace('+', 'p').replace('-', 'm')
        print(f'\n--- pair {pair_key} (r_A={r_A}, r_B={r_B}) ---')
        results = {}

        # bootstrap
        for name, hits, en in [('streaked', hits_s, en_s),
                                ('unstreaked', hits_u, en_u)]:
            print(f'  building matrices for {name} ({hits.shape[0]} shots)')
            H_A = azimuthal_prob_matrix(hits, en, r_A)
            H_B = azimuthal_prob_matrix(hits, en, r_B)
            cov, corr = cross_covariance(H_A, H_B)
            results[name] = {
                'cov': cov, 'corr': corr,
                'cov_sm': nan_gaussian_filter(cov, SMOOTH_SIGMA_BINS),
                'corr_sm': nan_gaussian_filter(corr, SMOOTH_SIGMA_BINS),
                'n_shots': hits.shape[0],
            }

        # real
        for pool in real_pools:
            print(f'  building matrices for real {pool}')
            H_A = real_theta_matrix(pool, r_A, real_get)
            H_B = real_theta_matrix(pool, r_B, real_get)
            cov, corr = cross_covariance(H_A, H_B)
            results[f'real_{pool}'] = {
                'cov': cov, 'corr': corr,
                'cov_sm': nan_gaussian_filter(cov, SMOOTH_SIGMA_BINS),
                'corr_sm': nan_gaussian_filter(corr, SMOOTH_SIGMA_BINS),
                'n_shots': H_A.shape[0],
            }

        # ---- diffs (smoothed, per pair): streaked-unstreaked, cotimed-goose,
        # laser_late-goose. Only compute if both operands present.
        diffs = {}
        if 'streaked' in results and 'unstreaked' in results:
            diffs['streaked_minus_unstreaked'] = nan_gaussian_filter(
                results['streaked']['cov'] - results['unstreaked']['cov'],
                SMOOTH_SIGMA_BINS)
        if 'real_cotimed' in results and 'real_goose' in results:
            diffs['cotimed_minus_goose'] = nan_gaussian_filter(
                results['real_cotimed']['cov'] - results['real_goose']['cov'],
                SMOOTH_SIGMA_BINS)
        if 'real_laser_late' in results and 'real_goose' in results:
            diffs['laser_late_minus_goose'] = nan_gaussian_filter(
                results['real_laser_late']['cov'] - results['real_goose']['cov'],
                SMOOTH_SIGMA_BINS)
        results['_diffs'] = diffs

        all_results[(r_A, r_B)] = results

        # --- Per-pair figure: 3 rows × N_cols
        panels = ['streaked', 'unstreaked'] + [f'real_{p}' for p in real_pools]
        n_cols = len(panels)
        fig, axes = plt.subplots(3, n_cols, figsize=(3.6 * n_cols, 10.5))
        v_cov = max(float(np.nanmax(np.abs(results[n]['cov']))) for n in panels) or 1e-30
        v_cov_sm = max(float(np.nanmax(np.abs(results[n]['cov_sm']))) for n in panels) or 1e-30
        v_corr_sm = max(float(np.nanmax(np.abs(results[n]['corr_sm']))) for n in panels) or 1e-30
        xlab = f'θ_inner (r={r_A:+d}) [deg]'
        ylab = f'θ_outer (r={r_B:+d}) [deg]'
        for c, name in enumerate(panels):
            r = results[name]
            label = f'{name}  |  N = {r["n_shots"]}'
            draw(axes[0, c], r['cov'], f'{label}\ncross-cov (raw)', v_cov, xlab, ylab)
            draw(axes[1, c], r['cov_sm'],
                 f'cross-cov (smoothed σ={SMOOTH_SIGMA_BINS:g})',
                 v_cov_sm, xlab, ylab)
            draw(axes[2, c], r['corr_sm'],
                 'cross-corr (smoothed)', v_corr_sm, xlab, ylab)
        fig.suptitle(
            f'Cross-annulus azimuthal covariance  |  inner r_A={r_A:+d} px, '
            f'outer r_B={r_B:+d} px  (probability-normalized, mode=wrap)',
            fontweight='bold', fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.955])
        p = os.path.join(OUT_DIR, f'cross_covariance_{pair_key}.png')
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f'  wrote {p}')

        # --- Per-pair diff figure (1 row × 3 diff panels)
        diff_names = list(diffs.keys())
        if diff_names:
            fig, axes = plt.subplots(1, len(diff_names),
                                      figsize=(4.2 * len(diff_names), 4.6),
                                      squeeze=False)
            v_diff = max(float(np.nanmax(np.abs(diffs[k]))) for k in diff_names) or 1e-30
            for c, name in enumerate(diff_names):
                title_map = {
                    'streaked_minus_unstreaked':
                        'bootstrap:  streaked − unstreaked (smoothed)',
                    'cotimed_minus_goose':
                        'real:  cotimed − goose (smoothed)',
                    'laser_late_minus_goose':
                        'real:  laser_late − goose (smoothed)',
                }
                draw(axes[0, c], diffs[name], title_map.get(name, name),
                     v_diff, xlab, ylab)
            fig.suptitle(
                f'Cross-annulus covariance diffs  |  inner r_A={r_A:+d} px, '
                f'outer r_B={r_B:+d} px',
                fontweight='bold', fontsize=11,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            p = os.path.join(OUT_DIR, f'cross_covariance_diffs_{pair_key}.png')
            fig.savefig(p, dpi=150); plt.close(fig)
            print(f'  wrote {p}')

    # ---- Summary grid: pairs × 6 cols
    if not args.no_real:
        col_specs = [
            ('streaked_sm',                'streaked (smoothed)'),
            ('unstreaked_sm',              'unstreaked (smoothed)'),
            ('streaked_minus_unstreaked',  'streaked − unstreaked'),
            ('real_goose_sm',              'real goose (smoothed)'),
            ('real_cotimed_sm',            'real cotimed (smoothed)'),
            ('real_laser_late_sm',         'real laser_late (smoothed)'),
            ('cotimed_minus_goose',        'real cotimed − goose'),
            ('laser_late_minus_goose',     'real laser_late − goose'),
        ]
    else:
        col_specs = [
            ('streaked_sm', 'streaked (smoothed)'),
            ('unstreaked_sm', 'unstreaked (smoothed)'),
            ('streaked_minus_unstreaked', 'streaked − unstreaked'),
        ]

    n_cols = len(col_specs)
    fig, axes = plt.subplots(len(PAIRS), n_cols,
                              figsize=(3.4 * n_cols, 3.4 * len(PAIRS)))
    for r_idx, (r_A, r_B) in enumerate(PAIRS):
        res = all_results[(r_A, r_B)]
        xlab = f'θ_inner (r={r_A:+d}) [deg]'
        ylab = f'θ_outer (r={r_B:+d}) [deg]'
        # Per-row: shared vmax across the sm-cov columns; independent for
        # each diff/real column pair.
        cov_sm_cols = ['streaked_sm', 'unstreaked_sm',
                       'real_goose_sm', 'real_cotimed_sm', 'real_laser_late_sm']
        cov_sm_cols = [c for c in cov_sm_cols if any(cs[0] == c for cs in col_specs)]
        v_cov_sm = 0.0
        for c in cov_sm_cols:
            k = c[:-3]  # strip '_sm'
            if k in res:
                v_cov_sm = max(v_cov_sm, float(np.nanmax(np.abs(res[k]['cov_sm']))))
        v_cov_sm = v_cov_sm or 1e-30

        for c_idx, (key, title) in enumerate(col_specs):
            ax = axes[r_idx, c_idx]
            if key.endswith('_sm'):
                base = key[:-3]
                M = res[base]['cov_sm']
                vmax = v_cov_sm
                sub = f' | N={res[base]["n_shots"]}'
            elif key in res['_diffs']:
                M = res['_diffs'][key]
                vmax = float(np.nanmax(np.abs(M))) or 1e-30
                sub = ''
            else:
                ax.axis('off')
                continue
            draw(ax, M, f'{title}{sub}\n(r={r_A:+d} vs {r_B:+d})', vmax, xlab, ylab)

    fig.suptitle(
        'Cross-annulus azimuthal covariance summary  '
        f'(pairs × datasets/diffs; smoothed σ={SMOOTH_SIGMA_BINS:g} bins, '
        'mode=wrap)',
        fontweight='bold', fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    p = os.path.join(OUT_DIR, 'summary_cross_covariance.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'wrote {p}')

    # Save every matrix + shot counts.
    save_dict = {}
    for (r_A, r_B), pair_res in all_results.items():
        pair_key = f'r{r_A:+d}_vs_r{r_B:+d}'.replace('+', 'p').replace('-', 'm')
        for name, r in pair_res.items():
            if name == '_diffs':
                for d_name, d_val in r.items():
                    save_dict[f'{pair_key}__diff_{d_name}_sm'] = np.nan_to_num(d_val)
                continue
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
