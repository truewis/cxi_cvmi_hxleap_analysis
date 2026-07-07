#!/usr/bin/env python3
"""
Compare the wiggle analysis's estimated lobe centre to the ground-truth centre
that the bootstrap generator drew.

For each sigma_theta_deg sweep point (from run_bootstrap_with_slurm_sigma.py):

  * TRUTH  — stats_sigma_<slug>deg.pkl carries:
      cx_true, cy_true                 (per-shot lobe centre in detector px)
      streak_mode_true_rad             (true streak axis, rad)
      streak_radius_true_px            (Rayleigh-drawn radial displacement)
      cx_reference, cy_reference       (67, 59)

  * ESTIMATE  — data_peak_positions_run_<slug>.npy stored by
    compute_circular_wiggle_analysis in each per-sigma metrics dir:
      x, y, significance               (per-shot in the same order)

    The estimates are computed for every shot passing mask_array (here all
    shots), so we join 1-to-1 by index.

Metrics per shot:
      dx  = x_est - cx_true
      dy  = y_est - cy_true
      dr  = hypot(dx, dy)                (Euclidean centre error, px)
      dtheta = wrap_pi(theta_est - theta_true)   (streak-angle error, rad)
      dradial = r_est - r_true            (radial displacement error, px)

Outputs (in --out, default ./wiggler_sigma_sweep_metrics):
  * accuracy_sigma_<slug>deg.png      per-sigma diagnostic
  * accuracy_data_sigma_<slug>deg.npz per-sigma per-shot arrays
  * master_center_accuracy_summary.png overlay: |dr|, |dtheta|, |dradial| vs sigma
  * master_center_accuracy_data.npy   {sigma_theta_deg, dr, dtheta, dradial,
                                       significance, per sigma point}

Only shots with significance > SIG_CUT are included in the *summary* stats —
the analysis reports these as detected. All shots are dumped raw in the .npz.

Requires conda env CXI.
"""
import argparse
import os
import pickle
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


DIR_RE = re.compile(r'^circular_wiggler_sim_sigma_(?P<slug>[0-9p]+)deg_batch_metrics$')
SIG_CUT = 3.0        # significance threshold for "detected" shots
N_PANEL_SHOTS = 10   # number of per-run shot panels to render

# Matches analysis_library/cvmi.py so the recomputed score maps agree with
# what the wiggle analysis itself produced.
RAW_CHANNELS_PER_BIN = 32
SPECTRUM_ROI_START = 13
WIGGLE_STEP = 0.4


def _compute_score_maps(img, energy, cx, cy, r_adjustment=0.0, sigma_r=1.0):
    """Recompute the three per-shot score maps (score_map_minus5,
    score_map_plus5, combined_sym_score_map) for a single hitfinder image.

    Direct port of the inner loop of compute_circular_wiggle_analysis in
    analysis_library/cvmi.py. Returns (x_centers, y_centers, m5, p5, combined),
    where m5/p5/combined are shape (len(y_centers), len(x_centers))."""
    ny, nx = img.shape
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]
    re = (energy / RAW_CHANNELS_PER_BIN - SPECTRUM_ROI_START) * 0.6 + 29.4 + r_adjustment
    wiggle_range = min(52.0 - re, re - 28.0)
    max_steps = int(np.ceil(max(wiggle_range, 0.0) / WIGGLE_STEP))
    offsets = np.arange(-max_steps, max_steps + 1) * WIGGLE_STEP
    x_centers = cx + offsets
    y_centers = cy + offsets

    smoothed = gaussian_filter(img, sigma=1.0)

    m5 = np.zeros((len(y_centers), len(x_centers)))
    p5 = np.zeros((len(y_centers), len(x_centers)))
    sym = np.zeros((len(y_centers), len(x_centers)))

    for row_idx, yc in enumerate(y_centers):
        for col_idx, xc in enumerate(x_centers):
            r_center = np.sqrt((xc - cx) ** 2 + (yc - cy) ** 2)
            if r_center + re >= 52.0 or r_center - re >= 28.0:
                m5[row_idx, col_idx] = np.nan
                p5[row_idx, col_idx] = np.nan
                sym[row_idx, col_idx] = np.nan
                continue
            r_map = np.sqrt((x_grid - xc) ** 2 + (y_grid - yc) ** 2)
            t_map = np.arctan2(y_grid - yc, x_grid - xc)
            cos2 = np.sin(t_map) ** 2 - 0.5

            w_re = np.exp(-((r_map - re) ** 2) / (2 * sigma_r ** 2))
            w_bm = np.exp(-((r_map - (re - 5.0)) ** 2) / (2 * sigma_r ** 2))
            w_bp = np.exp(-((r_map - (re + 5.0)) ** 2) / (2 * sigma_r ** 2))
            w_re[w_re < 1e-4] = 0
            w_bm[w_bm < 1e-4] = 0
            w_bp[w_bp < 1e-4] = 0

            weighted_re = img * w_re
            n_total = weighted_re.sum()
            if n_total > 0:
                x_com = (weighted_re * x_grid).sum() / n_total
                y_com = (weighted_re * y_grid).sum() / n_total
                d_sq = (x_com - xc) ** 2 + (y_com - yc) ** 2
                sym_lh = np.exp(-(n_total * d_sq) / (re ** 2))
            else:
                sym_lh = 0.0
            sym[row_idx, col_idx] = sym_lh

            def _dot(w):
                if w.sum() > 0:
                    wn = cos2 - np.average(cos2, weights=w)
                    return float(np.average(smoothed * wn, weights=w))
                return 0.0

            s_re = _dot(w_re)
            s_bm = _dot(w_bm)
            s_bp = _dot(w_bp)
            m5[row_idx, col_idx] = s_re - s_bm
            p5[row_idx, col_idx] = s_re - s_bp

    floored_m5 = np.clip(m5, 0, None)
    floored_p5 = np.clip(p5, 0, None)
    combined = floored_m5 * floored_p5 * sym
    return x_centers, y_centers, m5, p5, combined


def slug_to_deg(slug):
    return float(slug.replace('p', '.'))


def wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _load_truth(pkl_path):
    with open(pkl_path, 'rb') as f:
        stats = pickle.load(f)
    return stats


def _load_estimate(npy_path):
    # np.save with a plain dict writes a 0-d ndarray; unwrap it.
    obj = np.load(npy_path, allow_pickle=True).item()
    return obj  # {'x': ..., 'y': ..., 'significance': ...}


def _load_shot_cache(truth_dir, slug, detected=False):
    """Return (shot_indices, hits_stack) from the cache written by the
    bootstrap generator, or (None, None) if it isn't available.

    Set detected=True to load `shot_cache_detected_sigma_<slug>deg.npz`
    (the >3σ detected shots), which lives alongside the uniform sample."""
    name = ('shot_cache_detected' if detected else 'shot_cache') + f'_sigma_{slug}deg.npz'
    p = os.path.join(truth_dir, name)
    if not os.path.exists(p):
        return None, None
    z = np.load(p)
    return np.asarray(z['shot_indices']), np.asarray(z['hits'])


def _pick_panel_shots(cache_indices, sig, strategy='span'):
    """Pick up to N_PANEL_SHOTS shot indices from the cache. Returns a list
    of (shot_idx, cache_slot) pairs where shot_idx is the original shot
    number and cache_slot is its position in the cached hits stack.

    strategy = 'span'    : cover the full significance range (mixed)
    strategy = 'strong'  : pick the top-N by significance (best detections)
    """
    if cache_indices is None or len(cache_indices) == 0:
        return []
    cache_sig = sig[cache_indices]
    order = np.argsort(cache_sig)  # ascending sigma
    n = min(N_PANEL_SHOTS, len(order))
    if n == 0:
        return []
    if strategy == 'strong':
        picks = order[-n:][::-1]  # top-n descending
    else:  # 'span'
        picks = np.linspace(0, len(order) - 1, n, dtype=int)
        picks = np.unique(picks)
        picks = order[picks]
    return [(int(cache_indices[p]), int(p)) for p in picks]


def _plot_shot_panels(hits_stack, cache_indices, sig, x_est, y_est,
                      cx_true, cy_true, cx_ref, cy_ref, energy,
                      slug, out_dir, tag='', strategy='span', title_suffix=''):
    """For up to N_PANEL_SHOTS cached shots, render a 4-column row with the
    raw hitfinder image, the (r_e - 5) score map, the (r_e + 5) score map,
    and the combined composite score map — each with the true (lime +) and
    estimated (crimson x) lobe centres overlaid. Only shots present in the
    input cache are eligible.

    `tag` and `strategy` control which cache to use and how to pick shots:
      - tag=''        strategy='span'   uniform sample across sigma range
      - tag='detected' strategy='strong' top-N shots by sigma from the >3σ cache
    """
    picks = _pick_panel_shots(cache_indices, sig, strategy=strategy)
    if not picks:
        return None

    nrows = len(picks)
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)

    for row, (shot_idx, slot) in enumerate(picks):
        img = hits_stack[slot]
        cxt, cyt = float(cx_true[shot_idx]), float(cy_true[shot_idx])
        cxe, cye = float(x_est[shot_idx]),   float(y_est[shot_idx])
        err = np.hypot(cxe - cxt, cye - cyt)

        # Column 0: raw hitfinder image in detector coordinates.
        ax = axes[row, 0]
        vmax = np.percentile(img, 99.5) if img.max() > 0 else 1.0
        ax.imshow(img, cmap='viridis', origin='lower', vmin=0, vmax=vmax)
        for rad in (20.0, 60.0):
            ax.add_patch(plt.Circle((cx_ref, cy_ref), radius=rad,
                                    color='white', fill=False, linestyle='--',
                                    alpha=0.25, linewidth=0.8))
        ax.plot([cxt, cxe], [cyt, cye], color='yellow', lw=0.8, alpha=0.7)
        ax.plot(cxt, cyt, marker='+', ms=14, mew=2.0, color='lime',    label='true')
        ax.plot(cxe, cye, marker='x', ms=12, mew=2.0, color='crimson', label='est.')
        ax.plot(cx_ref, cy_ref, marker='o', ms=6, mfc='none', mec='cyan', mew=1.2)
        ax.set_title(f'shot {shot_idx}  sig={sig[shot_idx]:.2f}σ  |Δr|={err:.2f}px',
                     fontsize=10)
        ax.set_xlim(0, img.shape[1] - 1); ax.set_ylim(0, img.shape[0] - 1)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.85)

        # Recompute the three score maps for this shot in the same (xc, yc)
        # coordinate system used by the wiggle analysis, so overlaying the
        # true/estimated *centre coordinates* is direct.
        x_centers, y_centers, m5, p5, combined = _compute_score_maps(
            img=img, energy=float(energy), cx=cx_ref, cy=cy_ref,
        )
        extent = [x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]]

        for j, (arr, title, cmap) in enumerate((
            (m5,        r'score($r_e-5$)', 'inferno'),
            (p5,        r'score($r_e+5$)', 'inferno'),
            (combined,  r'combined·sym',   'magma'),
        )):
            ax = axes[row, 1 + j]
            im = ax.imshow(arr, cmap=cmap, origin='lower', extent=extent)
            ax.plot(cxt, cyt, marker='+', ms=14, mew=2.0, color='lime')
            ax.plot(cxe, cye, marker='x', ms=12, mew=2.0, color='crimson')
            ax.axvline(cx_ref, color='cyan', ls='--', alpha=0.4, lw=0.6)
            ax.axhline(cy_ref, color='cyan', ls='--', alpha=0.4, lw=0.6)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(f'σ_θ = {slug_to_deg(slug):g}°  |  per-shot centre reconstruction'
                 f'{title_suffix}  (lime + = truth, crimson × = estimate)',
                 fontweight='bold')
    fig.tight_layout()
    suffix = f'_{tag}' if tag else ''
    out_png = os.path.join(out_dir, f'shot_panels{suffix}_sigma_{slug}deg.png')
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f'  wrote {out_png}')
    return out_png


def _sorted_sweep_dirs(root):
    entries = []
    for name in os.listdir(root):
        m = DIR_RE.match(name)
        if not m:
            continue
        slug = m.group('slug')
        try:
            deg = slug_to_deg(slug)
        except ValueError:
            continue
        entries.append((deg, slug, os.path.join(root, name)))
    entries.sort(key=lambda e: e[0])
    return entries


def process_sweep_point(sweep_dir, slug, truth_dir, out_dir):
    """Return (deg, per-shot arrays dict, summary dict) for one sigma point."""
    run_id = slug
    est_path = os.path.join(sweep_dir, f'data_peak_positions_run_{run_id}.npy')
    truth_path = os.path.join(truth_dir, f'stats_sigma_{slug}deg.pkl')

    if not os.path.exists(est_path):
        print(f'  [skip] no estimate file: {est_path}')
        return None
    if not os.path.exists(truth_path):
        print(f'  [skip] no truth pkl: {truth_path}')
        return None

    truth = _load_truth(truth_path)
    est = _load_estimate(est_path)
    cache_indices, cache_hits = _load_shot_cache(truth_dir, slug)
    detected_indices, detected_hits = _load_shot_cache(truth_dir, slug, detected=True)

    x_est = np.asarray(est['x'], dtype=float)
    y_est = np.asarray(est['y'], dtype=float)
    sig = np.asarray(est['significance'], dtype=float)

    cx_true = np.asarray(truth['cx_true'], dtype=float)
    cy_true = np.asarray(truth['cy_true'], dtype=float)
    r_true = np.asarray(truth['streak_radius_true_px'], dtype=float)
    theta_true = np.asarray(truth['streak_mode_true_rad'], dtype=float)
    cx_ref = float(truth.get('cx_reference', 67))
    cy_ref = float(truth.get('cy_reference', 59))
    # `mock_energy_value` is the (constant) raw-channel energy the bootstrap
    # generator picked so raw_energy_to_bin_idx() returns the same peak bin.
    # Older pickles may not have this — fall back to the value implied by
    # peak_bin=25 (= (25 + 13) * 32 = 1216).
    energy = float(truth.get('mock_energy_value', (25 + 13) * 32))

    if not (len(x_est) == len(cx_true) == len(sig)):
        print(f'  [warn] length mismatch: est={len(x_est)}  truth={len(cx_true)}  sig={len(sig)}')
        n = min(len(x_est), len(cx_true), len(sig))
        x_est = x_est[:n]; y_est = y_est[:n]; sig = sig[:n]
        cx_true = cx_true[:n]; cy_true = cy_true[:n]
        r_true = r_true[:n]; theta_true = theta_true[:n]

    dx = x_est - cx_true
    dy = y_est - cy_true
    dr = np.hypot(dx, dy)

    r_est = np.hypot(x_est - cx_ref, y_est - cy_ref)
    theta_est = np.arctan2(y_est - cy_ref, x_est - cx_ref)
    dtheta = wrap_pi(theta_est - theta_true)
    dradial = r_est - r_true

    mask_detected = sig > SIG_CUT
    n_all = int(len(sig))
    n_det = int(mask_detected.sum())

    summary = {
        'sigma_theta_deg': slug_to_deg(slug),
        'n_total': n_all,
        'n_detected': n_det,
        # medians on *detected* shots (analysis's own significance filter)
        'median_dr_detected_px':      float(np.median(dr[mask_detected])) if n_det else np.nan,
        'median_abs_dtheta_detected_rad': float(np.median(np.abs(dtheta[mask_detected]))) if n_det else np.nan,
        'median_abs_dradial_detected_px': float(np.median(np.abs(dradial[mask_detected]))) if n_det else np.nan,
        # medians over all shots (upper bound; includes non-detected)
        'median_dr_all_px':      float(np.median(dr)),
        'median_abs_dtheta_all_rad': float(np.median(np.abs(dtheta))),
        'median_abs_dradial_all_px': float(np.median(np.abs(dradial))),
    }

    # ---- per-sigma diagnostic figure ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.scatter(cx_true - cx_ref, cy_true - cy_ref, s=8, alpha=0.35, color='0.5', label='truth')
    ax.scatter(x_est - cx_ref, y_est - cy_ref, s=8, alpha=0.35, color='crimson', label='estimate')
    if n_det:
        ax.scatter(x_est[mask_detected] - cx_ref, y_est[mask_detected] - cy_ref,
                   s=14, facecolor='none', edgecolor='blue',
                   label=f'detected (>{SIG_CUT:g}σ, N={n_det})')
    ax.axhline(0, ls='--', color='k', alpha=0.3)
    ax.axvline(0, ls='--', color='k', alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('x − cx  [px]')
    ax.set_ylabel('y − cy  [px]')
    ax.set_title(f'truth vs estimated centres (relative to ({cx_ref:.0f}, {cy_ref:.0f}))')
    ax.legend(fontsize=8, loc='upper left')

    ax = axes[0, 1]
    bins = np.linspace(0, max(1.0, np.nanmax(dr) * 1.05), 40)
    ax.hist(dr, bins=bins, color='0.6', alpha=0.55, label=f'all (N={n_all})')
    if n_det:
        ax.hist(dr[mask_detected], bins=bins, color='crimson', alpha=0.75, label=f'detected (N={n_det})')
    ax.axvline(summary['median_dr_all_px'], ls='--', color='k',
               label=f'median (all) = {summary["median_dr_all_px"]:.2f} px')
    if n_det:
        ax.axvline(summary['median_dr_detected_px'], ls='--', color='crimson',
                   label=f'median (det) = {summary["median_dr_detected_px"]:.2f} px')
    ax.set_xlabel(r'centre error $|\Delta r|$ [px]')
    ax.set_ylabel('count')
    ax.set_title('|estimate − truth|')
    ax.legend(fontsize=8, loc='upper right')

    ax = axes[1, 0]
    bins_th = np.linspace(-np.pi, np.pi, 41)
    ax.hist(dtheta, bins=bins_th, color='0.6', alpha=0.55, label='all')
    if n_det:
        ax.hist(dtheta[mask_detected], bins=bins_th, color='crimson', alpha=0.75, label='detected')
    ax.axvline(0.0, ls='--', color='k', alpha=0.6)
    ax.set_xlabel(r'streak-angle error  $\Delta\theta$ [rad]')
    ax.set_ylabel('count')
    ax.set_title(r'wrap($\theta_{est} - \theta_{true}$)')
    ax.legend(fontsize=8, loc='upper right')

    ax = axes[1, 1]
    bins_dr = np.linspace(np.min(dradial), np.max(dradial), 41)
    ax.hist(dradial, bins=bins_dr, color='0.6', alpha=0.55, label='all')
    if n_det:
        ax.hist(dradial[mask_detected], bins=bins_dr, color='crimson', alpha=0.75, label='detected')
    ax.axvline(0.0, ls='--', color='k', alpha=0.6)
    ax.set_xlabel(r'radial error  $r_{est} - r_{true}$ [px]')
    ax.set_ylabel('count')
    ax.set_title('radial displacement error')
    ax.legend(fontsize=8, loc='upper right')

    fig.suptitle(rf'centre-recovery accuracy  |  $\sigma_\theta = {slug_to_deg(slug):g}^\circ$'
                 f'  |  detected fraction = {n_det}/{n_all}',
                 fontweight='bold')
    fig.tight_layout()
    out_png = os.path.join(out_dir, f'accuracy_sigma_{slug}deg.png')
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f'  wrote {out_png}')

    # ---- per-sigma 10-panel per-shot reconstruction figures ----
    # First figure: 10 shots spanning the significance range (mixed).
    if cache_indices is not None and cache_hits is not None:
        _plot_shot_panels(
            hits_stack=cache_hits,
            cache_indices=cache_indices,
            sig=sig,
            x_est=x_est,
            y_est=y_est,
            cx_true=cx_true,
            cy_true=cy_true,
            cx_ref=cx_ref,
            cy_ref=cy_ref,
            energy=energy,
            slug=slug,
            out_dir=out_dir,
        )
    else:
        print(f'  [note] no shot_cache_sigma_{slug}deg.npz found; skipping panels')

    # Second figure: top-10 detected shots (sigma > 3), from the separate
    # detected cache that the bootstrap writes after running the analysis.
    if detected_indices is not None and detected_hits is not None:
        _plot_shot_panels(
            hits_stack=detected_hits,
            cache_indices=detected_indices,
            sig=sig,
            x_est=x_est,
            y_est=y_est,
            cx_true=cx_true,
            cy_true=cy_true,
            cx_ref=cx_ref,
            cy_ref=cy_ref,
            energy=energy,
            slug=slug,
            out_dir=out_dir,
            tag='detected',
            strategy='strong',
            title_suffix=r' (>3$\sigma$ detected)',
        )
    else:
        print(f'  [note] no shot_cache_detected_sigma_{slug}deg.npz found; '
              'skipping detected-shot panels')

    # ---- per-sigma raw arrays ----
    npz_path = os.path.join(out_dir, f'accuracy_data_sigma_{slug}deg.npz')
    np.savez(npz_path,
             sigma_theta_deg=slug_to_deg(slug),
             cx_true=cx_true, cy_true=cy_true,
             x_est=x_est, y_est=y_est,
             significance=sig,
             dx=dx, dy=dy, dr=dr,
             dtheta=dtheta, dradial=dradial,
             r_true=r_true, theta_true=theta_true,
             sig_cut=SIG_CUT)
    print(f'  wrote {npz_path}')

    per_shot = {
        'sigma_theta_deg': slug_to_deg(slug),
        'dr': dr, 'dtheta': dtheta, 'dradial': dradial,
        'significance': sig, 'detected_mask': mask_detected,
    }
    return per_shot, summary


def main():
    parser = argparse.ArgumentParser(description='Centre-recovery accuracy vs sigma_theta.')
    parser.add_argument('--root', type=str, default='.',
                        help='Directory containing circular_wiggler_sim_sigma_*_batch_metrics '
                             'AND the wiggler_sigma_sweep_metrics/ folder with stats_sigma_*.pkl.')
    parser.add_argument('--truth_dir', type=str, default=None,
                        help='Directory holding stats_sigma_<slug>deg.pkl. Defaults to '
                             '<root>/wiggler_sigma_sweep_metrics.')
    parser.add_argument('--out', type=str, default=None,
                        help='Output directory. Defaults to --truth_dir.')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    truth_dir = os.path.abspath(args.truth_dir) if args.truth_dir else os.path.join(root, 'wiggler_sigma_sweep_metrics')
    out_dir = os.path.abspath(args.out) if args.out else truth_dir
    os.makedirs(out_dir, exist_ok=True)

    sweeps = _sorted_sweep_dirs(root)
    if not sweeps:
        raise SystemExit(f'no circular_wiggler_sim_sigma_*_batch_metrics under {root}')

    print(f'Found {len(sweeps)} sweep point(s).  truth_dir={truth_dir}  out_dir={out_dir}')

    per_shot_all = []
    summaries = []
    for deg, slug, sweep_dir in sweeps:
        print(f'Processing sigma_theta = {deg:g} deg ({sweep_dir})...')
        out = process_sweep_point(sweep_dir, slug, truth_dir, out_dir)
        if out is None:
            continue
        per_shot, summary = out
        per_shot_all.append(per_shot)
        summaries.append(summary)

    if not summaries:
        raise SystemExit('no sweep points produced accuracy data.')

    sigma_axis = np.array([s['sigma_theta_deg'] for s in summaries], dtype=float)
    med_dr_all = np.array([s['median_dr_all_px'] for s in summaries])
    med_dr_det = np.array([s['median_dr_detected_px'] for s in summaries])
    med_dth_all = np.array([s['median_abs_dtheta_all_rad'] for s in summaries])
    med_dth_det = np.array([s['median_abs_dtheta_detected_rad'] for s in summaries])
    med_drad_all = np.array([s['median_abs_dradial_all_px'] for s in summaries])
    med_drad_det = np.array([s['median_abs_dradial_detected_px'] for s in summaries])
    n_total = np.array([s['n_total'] for s in summaries])
    n_det = np.array([s['n_detected'] for s in summaries])

    # ---- master overlay figure ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(sigma_axis, med_dr_all, 'o-', color='0.5', label='all shots')
    ax.plot(sigma_axis, med_dr_det, 'o-', color='crimson', label=f'detected (>{SIG_CUT:g}σ)')
    ax.set_ylabel(r'median $|\Delta r|$  [px]')
    ax.set_xlabel(r'$\sigma_\theta$ [deg]')
    ax.set_title('centre-position error vs streak length')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(sigma_axis, np.rad2deg(med_dth_all), 'o-', color='0.5', label='all')
    ax.plot(sigma_axis, np.rad2deg(med_dth_det), 'o-', color='crimson', label='detected')
    ax.set_ylabel(r'median $|\Delta\theta|$  [deg]')
    ax.set_xlabel(r'$\sigma_\theta$ [deg]')
    ax.set_title('streak-angle error')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.plot(sigma_axis, med_drad_all, 'o-', color='0.5', label='all')
    ax.plot(sigma_axis, med_drad_det, 'o-', color='crimson', label='detected')
    ax.set_ylabel(r'median $|r_{est} - r_{true}|$  [px]')
    ax.set_xlabel(r'$\sigma_\theta$ [deg]')
    ax.set_title('radial-displacement error')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(sigma_axis, n_det / np.maximum(n_total, 1), 'o-', color='steelblue')
    ax.set_ylabel(f'detected fraction (>{SIG_CUT:g}σ)')
    ax.set_xlabel(r'$\sigma_\theta$ [deg]')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('shots surviving the analysis significance gate')
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('Master: centre-recovery accuracy across $\\sigma_\\theta$ sweep', fontweight='bold')
    fig.tight_layout()
    master_png = os.path.join(out_dir, 'master_center_accuracy_summary.png')
    fig.savefig(master_png, dpi=150)
    plt.close(fig)
    print(f'wrote {master_png}')

    payload = {
        'sigma_theta_deg': sigma_axis,
        'n_total': n_total,
        'n_detected': n_det,
        'median_dr_all_px': med_dr_all,
        'median_dr_detected_px': med_dr_det,
        'median_abs_dtheta_all_rad': med_dth_all,
        'median_abs_dtheta_detected_rad': med_dth_det,
        'median_abs_dradial_all_px': med_drad_all,
        'median_abs_dradial_detected_px': med_drad_det,
        'sig_cut': SIG_CUT,
    }
    npy_path = os.path.join(out_dir, 'master_center_accuracy_data.npy')
    np.save(npy_path, payload, allow_pickle=True)
    print(f'wrote {npy_path}')


if __name__ == '__main__':
    main()
