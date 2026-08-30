#!/usr/bin/env python3
"""Post-hoc streak-finder over previously-generated bootstrap shot stacks.

Consumes `shot_hits_full_sigma_<slug>deg.npz` files written by
`run_bootstrap_with_slurm_sigma.py --no-analysis` (or by any run that
persisted the full hit stack in the same schema) and feeds them into
`compute_circular_wiggle_analysis` exactly the same way the inline
sweep does. Splits the sim generation from the analysis so a fast
'generate only' pass can be repeated cheaply while the (much slower)
streak finder runs later, once per (dataset, parameter) combination.

Reads:
  <shot_dir>/shot_hits_full_sigma_<slug>deg.npz
    keys used: hits, mean_energy
    (any additional keys such as cx_true, cy_true, streak_mode_true_rad,
     streak_radius_true_px, peak_bin, re, total_hits_within_mask are
     preserved into the summary pickle for downstream accuracy plots.)

Writes to `LEGACY_OUTPUTS_ROOT/`:
  circular_wiggler_sim_sigma_<slug>deg_batch_metrics/       (from cvmi)
  <shot_dir>/stats_sigma_<slug>deg.pkl                      (updated)
  <shot_dir>/distribution_sigma_<slug>deg.png               (histograms)
  <shot_dir>/shot_cache_detected_sigma_<slug>deg.npz        (top-N cache)

Requires conda env CXI.

Typical usage
-------------
Fast generate:
    python run_bootstrap_with_slurm_sigma.py --sigma_theta_deg 0 5 10 15 20 \
        --rayleigh_scale 5.0 --iterations 1000 --no-analysis
Later, run the streak finder:
    python run_analysis_on_bootstrap_hits.py \
        --shot_dir /path/to/wiggler_sigma_sweep_metrics \
        --sigma_theta_deg 0 5 10 15 20
"""
import argparse
import os
import pickle
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis_library.cvmi import (
    compute_circular_wiggle_analysis,
    LEGACY_OUTPUTS_ROOT,
)

SIGMA_FILE_RE = re.compile(r'^shot_hits_full_sigma_(?P<slug>[0-9p]+)deg\.npz$')


def _discover_sigmas(shot_dir):
    slugs = []
    for fn in os.listdir(shot_dir):
        m = SIGMA_FILE_RE.match(fn)
        if m:
            slugs.append(m.group('slug'))
    return sorted(slugs, key=lambda s: float(s.replace('p', '.')))


def _slug_from_sigma(sigma):
    """Filesystem-friendly encoding used by the bootstrap writer."""
    return f'{float(sigma):g}'.replace('.', 'p')


def _load_stack(shot_dir, slug):
    p = os.path.join(shot_dir, f'shot_hits_full_sigma_{slug}deg.npz')
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.load(p, allow_pickle=False)


def analyse_sigma(shot_dir, slug):
    stack = _load_stack(shot_dir, slug)
    hits = stack['hits']
    mean_energy = np.asarray(stack['mean_energy'], dtype=float)
    n = hits.shape[0]
    mask_array = np.ones(n, dtype=bool)
    is_gaussian = np.ones(n, dtype=bool)
    original_event_number = np.arange(n)

    # total_hits_within_mask is used only for scatter x-axis in cvmi's
    # summary; fall back to img.sum() when the writer didn't include it.
    if 'total_hits_within_mask' in stack.files:
        total_hits = np.asarray(stack['total_hits_within_mask'], dtype=float)
    else:
        total_hits = hits.reshape(n, -1).sum(axis=1).astype(float)

    print(f'[sigma_slug={slug}] running streak finder on {n} shots ...')
    scores, displacements_full = compute_circular_wiggle_analysis(
        mask_array=mask_array,
        run_id=slug,
        output_dir_suffix=f'sim_sigma_{slug}deg',
        images=hits.astype(float, copy=False),
        hits=hits.astype(float, copy=False),
        mean_energy=mean_energy,
        is_gaussian=is_gaussian,
        total_hit_within_mask=total_hits,
        original_event_number=original_event_number,
        annulus_mask=None,
        max_plots=20,
    )
    displacements = displacements_full[~np.isnan(displacements_full)]

    # --- Update / write the stats pickle in-place ---
    stats_path = os.path.join(shot_dir, f'stats_sigma_{slug}deg.pkl')
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
    else:
        stats = {}
    stats.update({
        'sigma_theta_deg': float(slug.replace('p', '.')),
        'scores': scores.tolist() if isinstance(scores, np.ndarray) else scores,
        'displacements': displacements.tolist(),
        'no_analysis': False,
        'shot_dir': os.path.abspath(shot_dir),
    })
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f'  updated {stats_path}')

    # --- Detected-shot cache (>3σ) ---
    DETECTED_SIG_THRESHOLD = 3.0
    DETECTED_CACHE_N = 30
    peak_pos_path = os.path.join(
        LEGACY_OUTPUTS_ROOT,
        f'circular_wiggler_sim_sigma_{slug}deg_batch_metrics',
        f'data_peak_positions_run_{slug}.npy',
    )
    try:
        peak_data = np.load(peak_pos_path, allow_pickle=True).item()
        sig_arr = np.asarray(peak_data['significance'], dtype=float)
        detected = np.where(sig_arr > DETECTED_SIG_THRESHOLD)[0]
        detected = detected[np.argsort(-sig_arr[detected])][:DETECTED_CACHE_N]
        if detected.size:
            detected = np.sort(detected)
            np.savez_compressed(
                os.path.join(shot_dir,
                             f'shot_cache_detected_sigma_{slug}deg.npz'),
                shot_indices=detected,
                hits=hits[detected].astype(np.float32),
                significance=sig_arr[detected].astype(np.float32),
                threshold=DETECTED_SIG_THRESHOLD,
            )
            print(f'  cached {detected.size} detected shots')
        else:
            print(f'  [note] no shots exceed {DETECTED_SIG_THRESHOLD}σ')
    except Exception as exc:
        print(f'  [warn] could not build detected shot cache: {exc}')

    # --- Trend histograms ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    ax1.hist(scores, bins=30, color='crimson', alpha=0.75)
    ax1.set_yscale('log')
    ax1.set_title(f'Composite Score  |  sigma_slug={slug}')
    ax1.set_xlabel('score')
    ax2.hist(displacements, bins=30, color='teal', alpha=0.75)
    ax2.set_title(f'Wiggle offsets |r|  |  N={len(displacements)}')
    ax2.set_xlabel('|r| [px]')
    ax3.hist(total_hits, bins=30, color='slategray', alpha=0.75)
    ax3.set_title('per-shot total hits')
    plt.tight_layout()
    out = os.path.join(shot_dir, f'distribution_sigma_{slug}deg.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'  wrote {out}')


def main():
    ap = argparse.ArgumentParser(
        description='Post-hoc streak finder on cached bootstrap shot stacks.')
    ap.add_argument('--shot_dir', required=True,
                    help='Directory containing shot_hits_full_sigma_*deg.npz '
                         '(typically <run_root>/wiggler_sigma_sweep_metrics).')
    ap.add_argument('--sigma_theta_deg', type=float, nargs='+', default=None,
                    help='Subset of sigma_theta values to run. Default: '
                         'process every shot_hits_full_sigma_*.npz in --shot_dir.')
    args = ap.parse_args()

    shot_dir = os.path.abspath(args.shot_dir)
    if not os.path.isdir(shot_dir):
        raise SystemExit(f'not a directory: {shot_dir}')

    all_slugs = _discover_sigmas(shot_dir)
    if not all_slugs:
        raise SystemExit(f'no shot_hits_full_sigma_*deg.npz found under {shot_dir}')

    if args.sigma_theta_deg is None:
        slugs = all_slugs
    else:
        want = {_slug_from_sigma(s) for s in args.sigma_theta_deg}
        slugs = [s for s in all_slugs if s in want]
        missing = want - set(slugs)
        for m in sorted(missing):
            print(f'[skip] no shot stack for sigma_slug={m} in {shot_dir}')

    print(f'analysing {len(slugs)} sigma point(s): {slugs}')
    for slug in slugs:
        analyse_sigma(shot_dir, slug)


if __name__ == '__main__':
    main()
