#!/usr/bin/env python3
"""
Extract an empirical background radial density from run 145 / step 19 and
save it as `noise_profile_run145_step19.npy` under batch_analysis/.

Recipe:
  * Load `data_accumulated_hits_run_145.npy` from
    results_145_to_152_4_ROI/circular_wiggler_duck_run145_step19_batch_metrics/
    — this is the total accumulated hitfinder image summed over all shots
    that passed the mask.
  * Divide by n_shots so the profile is per-shot (comes from
    data_score_vs_hits_run_145.npy which stores one entry per shot).
  * Keep only pixels with |theta - 0| < ang_half_width_deg OR
    |theta - pi| < ang_half_width_deg, i.e. the ±x fans. This avoids the
    unstreaked lobe signal which points along y (|sin theta| ~ 1).
  * Average within radial bins r in [0, R_MAX] to build density(r)
    (units: counts per pixel per shot at radius r).

The saved payload is a .npy pickle:
    {
      'r_centers':      float64 [N_R]      radial bin centers (px)
      'density':        float64 [N_R]      counts/pixel/shot  (nan where empty)
      'pixel_counts':   int64  [N_R]       # of pixels contributing per bin
      'n_shots':        int                shots in the underlying accumulation
      'ang_half_width_deg': float          wedge half-width used (both fans)
      'cx':             int                center x (67)
      'cy':             int                center y (59)
      'source_path':    str                path to source .npy
    }

Run inside conda env CXI.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_ROOT = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
                'batch_analysis_results/results_145_to_152_4_ROI/'
                'circular_wiggler_duck_run145_step19_batch_metrics')

# Output path — sits next to run_bootstrap_with_slurm_sigma.py so the bootstrap
# can find it with a hard-coded relative path.
DEFAULT_OUT = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
               'batch_analysis/noise_profile_run145_step19.npy')

CX, CY = 67, 59
R_MAX = 65.0
R_STEP = 1.0
ANG_HALF_WIDTH_DEG = 10.0


def build_profile(src_dir, ang_half_width_deg=ANG_HALF_WIDTH_DEG):
    hits_path = os.path.join(src_dir, 'data_accumulated_hits_run_145.npy')
    score_path = os.path.join(src_dir, 'data_score_vs_hits_run_145.npy')

    acc = np.load(hits_path).astype(float)
    ny, nx = acc.shape

    # n_shots: the score-vs-hits dump is one entry per shot passing the mask.
    n_shots = None
    if os.path.exists(score_path):
        try:
            obj = np.load(score_path, allow_pickle=True).item()
            if 'hits' in obj:
                n_shots = int(len(obj['hits']))
            elif 'scores' in obj:
                n_shots = int(len(obj['scores']))
        except Exception as exc:
            print(f'  [warn] could not parse {score_path}: {exc}')
    if n_shots is None:
        print('  [warn] falling back to n_shots=1 — profile will be in per-run '
              'units, not per-shot.')
        n_shots = 1

    per_shot = acc / float(n_shots)

    y_grid, x_grid = np.mgrid[0:ny, 0:nx]
    dx = x_grid - CX
    dy = y_grid - CY
    r_map = np.hypot(dx, dy).astype(float)
    theta = np.arctan2(dy, dx)                       # [-pi, pi]
    ang = np.deg2rad(ang_half_width_deg)

    # Wedge mask: near +x  (|theta| < ang)  OR  near -x  (|theta - pi| < ang,
    # wrapped to (|pi - |theta|| < ang)).
    near_x = (np.abs(theta) < ang) | (np.pi - np.abs(theta) < ang)

    r_edges = np.arange(0.0, R_MAX + R_STEP, R_STEP)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    density = np.full(r_centers.size, np.nan)
    counts = np.zeros(r_centers.size, dtype=int)

    for i in range(r_centers.size):
        m = near_x & (r_map >= r_edges[i]) & (r_map < r_edges[i + 1])
        n = int(m.sum())
        counts[i] = n
        if n > 0:
            density[i] = per_shot[m].mean()

    # Interpolate NaN bins from neighbors (bins should almost always be filled
    # given the wedge covers thousands of pixels; but the innermost bin can be
    # tiny).
    if np.isnan(density).any():
        good = ~np.isnan(density)
        if good.any():
            density = np.interp(r_centers, r_centers[good], density[good])

    density = np.clip(density, 0.0, None)  # negative densities aren't physical

    return {
        'r_centers':          r_centers,
        'density':            density,
        'pixel_counts':       counts,
        'n_shots':            int(n_shots),
        'ang_half_width_deg': float(ang_half_width_deg),
        'cx':                 int(CX),
        'cy':                 int(CY),
        'source_path':        os.path.abspath(hits_path),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default=DEFAULT_ROOT,
                   help='Directory holding data_accumulated_hits_run_145.npy and '
                        'data_score_vs_hits_run_145.npy from run 145 step 19.')
    p.add_argument('--out', default=DEFAULT_OUT,
                   help='Where to save the resulting noise_profile_run145_step19.npy.')
    p.add_argument('--ang_half_width_deg', type=float, default=ANG_HALF_WIDTH_DEG,
                   help='Half-width in degrees of the ±x wedges used to sample '
                        'background pixels (default 10).')
    args = p.parse_args()

    print(f'building profile from {args.src}')
    profile = build_profile(args.src, ang_half_width_deg=args.ang_half_width_deg)

    print(f'n_shots={profile["n_shots"]}  ang±{profile["ang_half_width_deg"]:g}°  '
          f'r bins={profile["r_centers"].size}')
    np.save(args.out, profile, allow_pickle=True)
    print(f'wrote {args.out}')

    # Diagnostic plot alongside the .npy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(profile['r_centers'], profile['density'], color='steelblue', lw=1.5)
    ax1.set_xlabel('r [px]')
    ax1.set_ylabel('counts / pixel / shot')
    ax1.set_title(f'background radial density  |  ±x wedge ±{profile["ang_half_width_deg"]:g}°'
                  f'  |  n_shots={profile["n_shots"]}')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')

    ax2.plot(profile['r_centers'], profile['pixel_counts'], color='goldenrod', lw=1.5)
    ax2.set_xlabel('r [px]')
    ax2.set_ylabel('# pixels contributing')
    ax2.set_title('per-bin pixel count')
    ax2.grid(True, alpha=0.3, linestyle='--')

    diag = args.out.replace('.npy', '.png')
    fig.tight_layout()
    fig.savefig(diag, dpi=140)
    print(f'wrote {diag}')


if __name__ == '__main__':
    main()
