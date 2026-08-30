#!/usr/bin/env python3
"""
Hyperspectral azimuthal covariance map for each preset.

For every shot in a preset's (run, step) pool (base-masked:
`data["masks"][kind] & data["is_gaussian"] & (data["lxts"] == step_val)`),
compute a per-shot angular density in the photoline annulus:

    h_theta_bin(shot) = sum of hitfinder-image pixels with
        (re − DR_HALF) <= r < (re + DR_HALF)   AND   theta in bin

theta = atan2(y − cy, x − cx), wrapped to [0, 2*pi], divided into
`N_BINS = 90` equal-width bins (4° each). `re` is derived per shot
from `mean_energy` via `(E / RAW_CHANNELS_PER_BIN − SPECTRUM_ROI_START)
* 0.6 + 29.4` — matches analysis_library/cvmi.py.

Then across the preset's shots, compute the sample covariance matrix
of the 90-dimensional theta vector and plot it as a diverging heatmap.

One figure + one .npy per preset (four presets total).

Requires conda env CXI.
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
# Add batch_analysis/ to path so sibling imports work from plotting/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aggregate_4roi_late_steps import PRESETS, _enumerate_pairs  # noqa: E402


PICKLE_DIR = '/sdf/scratch/users/j/jinseop/'
CX, CY = 67, 59
RAW_CHANNELS_PER_BIN = 32
SPECTRUM_ROI_START = 13
N_BINS = 90                  # 4 deg per bin
DR_HALF = 3.0                # annulus half-width around re
# Default radial offsets applied to the annulus center. The annulus is
# [re + r_offset − DR_HALF,  re + r_offset + DR_HALF).
DEFAULT_R_OFFSETS = (-6.0, -3.0, 0.0, 3.0, 6.0)

# Cross-covariance pairs (annulus_x_offset, annulus_y_offset). X-axis θ is
# taken from the inner annulus (re + off_x), Y-axis θ from the outer annulus
# (re + off_y).
DEFAULT_CROSS_PAIRS = ((-3.0, 3.0), (-6.0, 6.0))


def re_from_energy(energy):
    return (float(energy) / RAW_CHANNELS_PER_BIN - SPECTRUM_ROI_START) * 0.6 + 29.4


def _load_pickle_cache():
    cache = {}

    def get(run):
        if run not in cache:
            p = os.path.join(PICKLE_DIR, f'preprocessed_run_{run}.pkl')
            if not os.path.exists(p):
                cache[run] = None
                return None
            print(f'  loading {p} ...')
            with open(p, 'rb') as f:
                cache[run] = pickle.load(f)
        return cache[run]

    return get


def _mask_shots(data, kind, step):
    unique_steps = np.unique(data['lxts']) if 'lxts' in data else np.unique(data.get('step', [0]))
    step_val = unique_steps[step]
    step_axis = data['lxts'] if 'lxts' in data else data.get('step')
    return (
        data['masks'][kind].astype(bool)
        & data['is_gaussian'].astype(bool)
        & (step_axis == step_val)
    )


def _theta_bin_and_r(shape, cx, cy):
    """Precompute r-map and theta-bin index for the detector grid. All shots
    from the same run share the same grid, so this can be cached per shape."""
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    r = np.hypot(x - cx, y - cy)
    theta = np.arctan2(y - cy, x - cx)      # [-pi, pi]
    theta_shifted = theta + np.pi           # [0, 2pi]
    theta_bin = np.floor(theta_shifted / (2.0 * np.pi / N_BINS)).astype(int)
    theta_bin = np.clip(theta_bin, 0, N_BINS - 1)
    return r, theta_bin


def build_theta_matrix(pairs, kind, r_offset=0.0, cx=CX, cy=CY):
    """Return H shape (N_shots, N_BINS): each row is a shot's summed
    hitfinder intensity per θ-bin in the annulus
        [re + r_offset − DR_HALF,  re + r_offset + DR_HALF).

    cx / cy default to the module-level (67, 59). Passing different values
    (e.g. the (69, 61) shifted center) recomputes r and θ about that origin."""
    get_pkl = _load_pickle_cache()
    rows = []

    for run, step in pairs:
        data = get_pkl(run)
        if data is None:
            print(f'  [warn] no pickle for run {run}; skipping ({run}, {step})')
            continue
        mask = _mask_shots(data, kind=kind, step=step)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue

        # Precompute r/theta grids for this run's shape (all pickles use 125×125).
        r_map, theta_bin_map = _theta_bin_and_r(data['hits'][idx[0]].shape, cx, cy)

        for i in idx:
            re = re_from_energy(data['mean_energy'][i]) + r_offset
            annulus = (r_map >= re - DR_HALF) & (r_map < re + DR_HALF)
            if not annulus.any():
                continue
            img = data['hits'][i]
            h = np.bincount(theta_bin_map[annulus],
                            weights=img[annulus].astype(float, copy=False),
                            minlength=N_BINS)
            rows.append(h)

    if not rows:
        return np.empty((0, N_BINS), dtype=float)
    return np.vstack(rows)


def plot_covariance(H, title, out_png):
    n_shots = H.shape[0]
    if n_shots < 2:
        print(f'  [skip] only {n_shots} shots; covariance undefined')
        return None

    cov = np.cov(H, rowvar=False)  # (N_BINS, N_BINS)
    corr = np.corrcoef(H, rowvar=False)

    # Mask the diagonal so its self-variance / self-correlation doesn't
    # dominate the colormap. The saved .npy still carries the unmasked
    # matrices; only the plotted view has the diagonal removed.
    diag_mask = np.eye(N_BINS, dtype=bool)
    cov_plot = np.where(diag_mask, np.nan, cov)
    corr_plot = np.where(diag_mask, np.nan, corr)

    fig, (ax_cov, ax_corr) = plt.subplots(1, 2, figsize=(15, 6.5))
    extent = [0.0, 360.0, 0.0, 360.0]

    off_max = float(np.nanmax(np.abs(cov_plot)))
    if not np.isfinite(off_max) or off_max == 0:
        off_max = 1e-12
    im_c = ax_cov.imshow(cov_plot, cmap='RdBu_r', origin='lower', extent=extent,
                         vmin=-off_max, vmax=off_max, aspect='equal')
    ax_cov.set_xlabel(r'$\theta_j$ [deg]')
    ax_cov.set_ylabel(r'$\theta_i$ [deg]')
    ax_cov.set_title(f'covariance (diagonal masked)  |  {n_shots} shots')
    plt.colorbar(im_c, ax=ax_cov, fraction=0.046, pad=0.03)

    corr_max = float(np.nanmax(np.abs(corr_plot)))
    if not np.isfinite(corr_max) or corr_max == 0:
        corr_max = 1e-12
    im_r = ax_corr.imshow(corr_plot, cmap='RdBu_r', origin='lower', extent=extent,
                          vmin=-corr_max, vmax=corr_max, aspect='equal')
    ax_corr.set_xlabel(r'$\theta_j$ [deg]')
    ax_corr.set_ylabel(r'$\theta_i$ [deg]')
    ax_corr.set_title('correlation (diagonal masked)')
    plt.colorbar(im_r, ax=ax_corr, fraction=0.046, pad=0.03)

    fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f'  wrote {out_png}')
    return cov, corr


def plot_cross_covariance(Hx, Hy, title, out_png):
    """Cross-covariance between two per-shot theta vectors from two different
    annuli. Hx / Hy are shape (N_shots, N_BINS); shots must be aligned.

    The matrix element C[i, j] = cov(Hy[:, i], Hx[:, j]) so the y-axis
    corresponds to Hy's annulus and the x-axis to Hx's annulus. Pearson
    correlation is computed by dividing by outer(std_y, std_x)."""
    n_shots = Hx.shape[0]
    if n_shots < 2 or Hy.shape[0] != n_shots:
        print(f'  [skip cross] shots mismatch: Hx={Hx.shape[0]}, Hy={Hy.shape[0]}')
        return None

    mean_x = Hx.mean(axis=0, keepdims=True)
    mean_y = Hy.mean(axis=0, keepdims=True)
    dx = Hx - mean_x
    dy = Hy - mean_y
    # (N_BINS x N_BINS) — rows index Hy's θ, cols index Hx's θ.
    cross_cov = (dy.T @ dx) / (n_shots - 1)
    std_x = dx.std(axis=0, ddof=1)
    std_y = dy.std(axis=0, ddof=1)
    denom = np.outer(std_y, std_x)
    denom[denom == 0] = 1e-12
    cross_corr = cross_cov / denom

    fig, (ax_cov, ax_corr) = plt.subplots(1, 2, figsize=(15, 6.5))
    extent = [0.0, 360.0, 0.0, 360.0]

    cov_max = float(np.nanmax(np.abs(cross_cov)))
    if not np.isfinite(cov_max) or cov_max == 0:
        cov_max = 1e-12
    im_c = ax_cov.imshow(cross_cov, cmap='RdBu_r', origin='lower', extent=extent,
                         vmin=-cov_max, vmax=cov_max, aspect='equal')
    ax_cov.set_xlabel(r'$\theta_j$ [deg]  (X-annulus)')
    ax_cov.set_ylabel(r'$\theta_i$ [deg]  (Y-annulus)')
    ax_cov.set_title(f'cross-covariance  |  {n_shots} shots')
    plt.colorbar(im_c, ax=ax_cov, fraction=0.046, pad=0.03)

    corr_max = float(np.nanmax(np.abs(cross_corr)))
    if not np.isfinite(corr_max) or corr_max == 0:
        corr_max = 1e-12
    im_r = ax_corr.imshow(cross_corr, cmap='RdBu_r', origin='lower', extent=extent,
                          vmin=-corr_max, vmax=corr_max, aspect='equal')
    ax_corr.set_xlabel(r'$\theta_j$ [deg]  (X-annulus)')
    ax_corr.set_ylabel(r'$\theta_i$ [deg]  (Y-annulus)')
    ax_corr.set_title('cross-correlation (Pearson)')
    plt.colorbar(im_r, ax=ax_corr, fraction=0.046, pad=0.03)

    fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f'  wrote {out_png}')
    return cross_cov, cross_corr


def _preset_pairs(name):
    if name == 'all_goose':
        return _enumerate_pairs('goose')
    return PRESETS[name]


def _offset_slug(r_offset):
    """Filesystem-friendly slug for the offset: 0 -> 'r0', -6 -> 'r-6',
    +3.5 -> 'r+3p5'."""
    s = f'{r_offset:+g}'.replace('.', 'p')
    if r_offset == 0:
        return 'r0'
    return f'r{s}'


def _pair_slug(off_x, off_y):
    def _one(o):
        return f'r{o:+g}'.replace('.', 'p')
    return f'cross_{_one(off_x)}_{_one(off_y)}'


def process(preset_name, kind, out_base, r_offsets, cross_pairs=(),
            cx=CX, cy=CY):
    pairs = _preset_pairs(preset_name)
    print(f'=== {preset_name} (kind={kind}, {len(pairs)} pairs, '
          f'center=({cx}, {cy})) ===')

    # Choose output folder to match how the by-energy aggregator names things.
    # Default center writes into the base *_by_energy folder; a shifted center
    # writes into *_by_energy_shifted (which already exists on disk).
    base_name = ('aggregated_4roi_goose_all_goose_by_energy'
                 if preset_name == 'all_goose'
                 else f'aggregated_4roi_{kind}_{preset_name}_by_energy')
    if (cx, cy) != (CX, CY):
        base_name = f'{base_name}_shifted'
    out_dir = os.path.join(out_base, base_name)
    if not os.path.isdir(out_dir):
        out_dir = os.path.join(out_base, f'azimuthal_covariance_{kind}_{preset_name}')
        os.makedirs(out_dir, exist_ok=True)

    for r_off in r_offsets:
        slug = _offset_slug(r_off)
        print(f'  --- annulus center = re{r_off:+g} px (slug={slug}) ---')
        H = build_theta_matrix(pairs, kind=kind, r_offset=r_off, cx=cx, cy=cy)
        print(f'    built theta matrix: shape={H.shape}')

        title = (f'Hyperspectral azimuthal covariance  |  {preset_name}  '
                 f'(kind={kind})\nannulus = re{r_off:+g} ± {DR_HALF:g} px, '
                 f'{N_BINS} angular bins (4° each), center=({cx}, {cy})')
        out_png = os.path.join(out_dir, f'azimuthal_covariance_{slug}.png')
        result = plot_covariance(H, title, out_png)
        if result is None:
            continue
        cov, corr = result

        np.save(os.path.join(out_dir, f'azimuthal_covariance_{slug}.npy'),
                {'H': H, 'cov': cov, 'corr': corr,
                 'n_bins': N_BINS, 'dr_half': DR_HALF, 'r_offset': r_off,
                 'cx': cx, 'cy': cy, 'kind': kind,
                 'preset_name': preset_name, 'pairs': pairs,
                 'theta_edges_deg': np.linspace(0.0, 360.0, N_BINS + 1)},
                allow_pickle=True)
        print(f'    wrote {os.path.join(out_dir, f"azimuthal_covariance_{slug}.npy")}')

    for off_x, off_y in cross_pairs:
        slug = _pair_slug(off_x, off_y)
        print(f'  --- cross annulus  X: re{off_x:+g}  Y: re{off_y:+g}  (slug={slug}) ---')
        Hx = build_theta_matrix(pairs, kind=kind, r_offset=off_x, cx=cx, cy=cy)
        Hy = build_theta_matrix(pairs, kind=kind, r_offset=off_y, cx=cx, cy=cy)
        if Hx.shape[0] != Hy.shape[0]:
            print(f'    [warn] shot-count mismatch Hx={Hx.shape[0]} Hy={Hy.shape[0]}; '
                  'trimming to the shorter one')
            n = min(Hx.shape[0], Hy.shape[0])
            Hx = Hx[:n]; Hy = Hy[:n]

        title = (f'Hyperspectral azimuthal cross-covariance  |  {preset_name}  '
                 f'(kind={kind})\nX annulus = re{off_x:+g} ± {DR_HALF:g} px, '
                 f'Y annulus = re{off_y:+g} ± {DR_HALF:g} px, '
                 f'{N_BINS} angular bins (4° each), center=({cx}, {cy})')
        out_png = os.path.join(out_dir, f'azimuthal_covariance_{slug}.png')
        result = plot_cross_covariance(Hx, Hy, title, out_png)
        if result is None:
            continue
        cross_cov, cross_corr = result
        np.save(os.path.join(out_dir, f'azimuthal_covariance_{slug}.npy'),
                {'Hx': Hx, 'Hy': Hy,
                 'cross_cov': cross_cov, 'cross_corr': cross_corr,
                 'n_bins': N_BINS, 'dr_half': DR_HALF,
                 'r_offset_x': off_x, 'r_offset_y': off_y,
                 'cx': cx, 'cy': cy, 'kind': kind,
                 'preset_name': preset_name, 'pairs': pairs,
                 'theta_edges_deg': np.linspace(0.0, 360.0, N_BINS + 1)},
                allow_pickle=True)
        print(f'    wrote {os.path.join(out_dir, f"azimuthal_covariance_{slug}.npy")}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--presets', nargs='+',
                    default=['late', 'cotimed', 'early', 'all_goose'])
    ap.add_argument('--r_offsets', type=float, nargs='+',
                    default=[],
                    help='Same-annulus offsets (in px). Pass "-1e9" or leave empty '
                         'to skip the same-annulus loop.')
    ap.add_argument('--cross_pairs', type=float, nargs='*',
                    default=[o for pair in DEFAULT_CROSS_PAIRS for o in pair],
                    help='Flat list of (x_offset, y_offset) pairs for cross-annulus '
                         'covariance; even count required. Pass "--cross_pairs" '
                         'with no arguments to skip the cross-annulus loop. '
                         f'Default {list(DEFAULT_CROSS_PAIRS)}.')
    ap.add_argument('--cx', type=float, default=None,
                    help=f'x center for r/θ. Default {CX}.')
    ap.add_argument('--cy', type=float, default=None,
                    help=f'y center for r/θ. Default {CY}.')
    ap.add_argument('--out_base', default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    if len(args.cross_pairs) % 2 != 0:
        raise SystemExit('--cross_pairs must be a flat even-length list of x,y offsets')
    cross_pairs = list(zip(args.cross_pairs[0::2], args.cross_pairs[1::2]))

    cx = CX if args.cx is None else float(args.cx)
    cy = CY if args.cy is None else float(args.cy)

    for name in args.presets:
        kind = 'goose' if name == 'all_goose' else 'duck'
        process(name, kind, args.out_base, args.r_offsets,
                cross_pairs=cross_pairs, cx=cx, cy=cy)


if __name__ == '__main__':
    main()
