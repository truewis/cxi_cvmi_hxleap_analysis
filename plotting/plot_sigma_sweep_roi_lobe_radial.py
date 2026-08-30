#!/usr/bin/env python3
"""
Post-process the sigma-sweep bootstrap output (see run_bootstrap_with_slurm_sigma.py).

For each sigma_theta_deg sweep point:
  * load the normalized+smoothed accumulated hit-density image for ROI_R and ROI_L
  * compute a y-lobe-weighted (sin^2 theta) radial density profile:

        S(r) = < img * sin^2(theta) >_{|r - r_pix| < dr/2}

    where theta is measured from the ROI center
      - ROI_R:  (cx + 5, cy) = (72, 59)
      - ROI_L:  (cx - 5, cy) = (62, 59)
    i.e. sin^2(theta) = 0 on the horizontal ray through that center and = 1
    on the vertical rays. This weights the up/down lobes and suppresses the
    left/right dead zones.

  * plot S(r) vs r for the sweep point, and save the underlying arrays as .npy

Also produces two master overlay figures (one per ROI) covering every sigma
sweep point present under --root.

Requires conda env CXI (numpy/scipy/matplotlib).
"""
import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


SMOOTH_SIGMA_PX = 0.3
CX, CY = 67, 59  # match analysis_library/cvmi.py
ROI_CENTERS = {
    'R': (CX + 5, CY),
    'L': (CX - 5, CY),
}

# Slug convention shared with run_bootstrap_with_slurm_sigma.py:
# f"{sigma_theta_deg:g}".replace('.', 'p')
DIR_RE = re.compile(r'^circular_wiggler_sim_sigma_(?P<slug>[0-9p]+)deg_batch_metrics$')

# Radial sweep: r = 0..50 in 1 px steps
R_MIN, R_MAX, R_STEP = 0.0, 50.0, 1.0


def slug_to_deg(slug):
    return float(slug.replace('p', '.'))


def load_and_normalize(path):
    """Load a saved accumulated-hits .npy, apply Gaussian smoothing, and
    normalize so the (smoothed) image sums to 1. Returns None on absence."""
    if not os.path.exists(path):
        return None
    img = np.load(path).astype(float)
    img = gaussian_filter(img, sigma=SMOOTH_SIGMA_PX)
    s = img.sum()
    if s > 0:
        img = img / s
    return img


def sin2_radial_profile(img, r_centers, cx_roi, cy_roi, dr):
    """For each r in r_centers, compute
        S(r) = < img * sin^2(theta) >  averaged over pixels with
                r - dr/2 <= |(x,y) - (cx_roi, cy_roi)| < r + dr/2

    theta = atan2(y - cy_roi, x - cx_roi). If an annulus contains no pixels,
    the corresponding S(r) is NaN."""
    ny, nx = img.shape
    y, x = np.mgrid[0:ny, 0:nx]
    dx = x - cx_roi
    dy = y - cy_roi
    r_map = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    sin2 = np.sin(theta) ** 2

    weighted = img * sin2
    S = np.full(r_centers.size, np.nan)
    counts = np.zeros(r_centers.size, dtype=int)
    for i, rc in enumerate(r_centers):
        m = (r_map >= rc - dr / 2.0) & (r_map < rc + dr / 2.0)
        n = int(m.sum())
        counts[i] = n
        if n > 0:
            S[i] = weighted[m].mean()
    return S, counts


def process_sweep_point(sweep_dir, slug, out_dir, r_centers):
    """Compute R and L radial profiles for one sweep point and write the
    per-sigma figure. Returns (deg, R_profile, L_profile)."""
    run_id = slug  # compute_circular_wiggle_analysis stores .npy files with run_id
    p_r = os.path.join(sweep_dir, f'data_accumulated_hits_ROI_R_run_{run_id}.npy')
    p_l = os.path.join(sweep_dir, f'data_accumulated_hits_ROI_L_run_{run_id}.npy')

    img_r = load_and_normalize(p_r)
    img_l = load_and_normalize(p_l)

    profile_r = counts_r = None
    profile_l = counts_l = None

    if img_r is not None:
        cx_r, cy_r = ROI_CENTERS['R']
        profile_r, counts_r = sin2_radial_profile(img_r, r_centers, cx_r, cy_r, R_STEP)
    if img_l is not None:
        cx_l, cy_l = ROI_CENTERS['L']
        profile_l, counts_l = sin2_radial_profile(img_l, r_centers, cx_l, cy_l, R_STEP)

    fig, ax = plt.subplots(figsize=(8, 5))
    if profile_r is not None:
        ax.plot(r_centers, profile_r,
                label=f'ROI R  @ ({ROI_CENTERS["R"][0]}, {ROI_CENTERS["R"][1]})',
                color='tab:orange', lw=1.6)
    if profile_l is not None:
        ax.plot(r_centers, profile_l,
                label=f'ROI L  @ ({ROI_CENTERS["L"][0]}, {ROI_CENTERS["L"][1]})',
                color='tab:blue', lw=1.6)
    ax.set_xlabel('integration radius r [px]')
    ax.set_ylabel(r'$\langle \mathrm{img} \cdot \sin^2\theta \rangle$  per pixel  (image sums to 1)')
    ax.set_title(fr'$\sigma_\theta = {slug_to_deg(slug):g}^\circ$'
                 f'  |  sin^2 radial profile  (gaussian sigma={SMOOTH_SIGMA_PX} px)')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    fig.tight_layout()
    out_png = os.path.join(out_dir, f'sin2_radial_ROI_RL_sigma_{slug}deg.png')
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f'  wrote {out_png}')

    return profile_r, profile_l, counts_r, counts_l


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


def main():
    parser = argparse.ArgumentParser(description='sin^2(theta) radial profiles of ROI R/L for sigma-sweep bootstrap.')
    parser.add_argument('--root', type=str, default='.',
                        help='Directory containing the circular_wiggler_sim_sigma_*_batch_metrics folders.')
    parser.add_argument('--out', type=str, default='./wiggler_sigma_sweep_metrics',
                        help='Directory to write per-sigma PNGs, overlays, and .npy dumps into.')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    r_centers = np.arange(R_MIN, R_MAX + R_STEP * 0.5, R_STEP)

    sweeps = _sorted_sweep_dirs(root)
    if not sweeps:
        raise SystemExit(f'no circular_wiggler_sim_sigma_*_batch_metrics directories under {root}')

    print(f'Found {len(sweeps)} sigma sweep point(s):')
    for deg, slug, _ in sweeps:
        print(f'  sigma_theta = {deg:g} deg  (slug={slug})')

    collected_deg = []
    collected_slugs = []
    R_profiles = []
    L_profiles = []
    R_counts = []
    L_counts = []
    for deg, slug, sweep_dir in sweeps:
        print(f'Processing sigma_theta = {deg:g} deg ({sweep_dir})...')
        pr, pl, cr, cl = process_sweep_point(sweep_dir, slug, out_dir, r_centers)
        collected_deg.append(deg)
        collected_slugs.append(slug)
        R_profiles.append(pr if pr is not None else np.full(r_centers.size, np.nan))
        L_profiles.append(pl if pl is not None else np.full(r_centers.size, np.nan))
        R_counts.append(cr if cr is not None else np.zeros(r_centers.size, dtype=int))
        L_counts.append(cl if cl is not None else np.zeros(r_centers.size, dtype=int))

    R_stack = np.stack(R_profiles)
    L_stack = np.stack(L_profiles)

    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0.0, 0.95, len(collected_deg)))

    def overlay(stack, title, out_png):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for i, deg in enumerate(collected_deg):
            row = stack[i]
            m = np.isfinite(row)
            if not m.any():
                continue
            ax.plot(r_centers[m], row[m], color=colors[i], lw=1.6, label=f'{deg:g} deg')
        ax.set_xlabel('integration radius r [px]')
        ax.set_ylabel(r'$\langle \mathrm{img} \cdot \sin^2\theta \rangle$  per pixel')
        ax.set_title(f'{title}  — sin^2 radial profile overlay  (gaussian sigma={SMOOTH_SIGMA_PX} px)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(title=r'$\sigma_\theta$', ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f'wrote {out_png}')

    overlay(R_stack, 'ROI R (right fan, center=(cx+5, cy))', os.path.join(out_dir, 'master_sin2_radial_ROI_R.png'))
    overlay(L_stack, 'ROI L (left fan,  center=(cx-5, cy))', os.path.join(out_dir, 'master_sin2_radial_ROI_L.png'))

    payload = {
        'r_centers': r_centers,
        'sigma_theta_deg': np.array(collected_deg, dtype=float),
        'sigma_slugs': np.array(collected_slugs),
        'roi_R_profile': R_stack,
        'roi_L_profile': L_stack,
        'roi_R_pixel_counts': np.stack(R_counts),
        'roi_L_pixel_counts': np.stack(L_counts),
        'roi_R_center': ROI_CENTERS['R'],
        'roi_L_center': ROI_CENTERS['L'],
        'smooth_sigma_px': SMOOTH_SIGMA_PX,
        'r_step': R_STEP,
    }
    npy_path = os.path.join(out_dir, 'master_sin2_radial_ROI_RL_data.npy')
    np.save(npy_path, payload, allow_pickle=True)
    print(f'wrote {npy_path}')


if __name__ == '__main__':
    main()
