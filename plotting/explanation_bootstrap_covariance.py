#!/usr/bin/env python3
"""Full-simulation explanation figure.

Uses the same lobe generator as run_bootstrap_with_slurm_sigma.py
(Rayleigh streak radius + Gaussian angular sweep + shell-projected sin^2
lobes + empirical run-145/step-19 background), pipes the synthetic hit
stack through the streak finder (`compute_circular_wiggle_analysis`),
then computes the per-shot azimuthal covariance across only shots that
the finder tagged with |r| > 5 px AND significance > 2.

Outputs
-------
  batch_analysis/explanation_bootstrap_covariance.png
      Left  : cumulative lobe-only density (all shots) with re annulus.
      Right : sample azimuthal covariance across the DETECTED subset,
              36 angular bins, guides at theta = 90 and 270 deg.

Deterministic (rng seed = 0).  Requires conda env CXI.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
# Add batch_analysis/ to path so sibling imports work from plotting/.
sys.path.insert(0, os.path.dirname(_HERE))

from run_bootstrap_with_slurm_sigma import (
    NOISE_PROFILE_PATH,
    R_MAX_FROM_CENTER,
    SHELL_THICKNESS,
    _load_noise_profile,
    _sample_bg_from_profile,
    _sample_projected_shell_with_sin2,
)
import analysis_library.cvmi as _cvmi_mod
from analysis_library.cvmi import compute_circular_wiggle_analysis

# cvmi.compute_circular_wiggle_analysis has a latent UnboundLocalError:
# it references `display_idx` inside the ROI-append block, but the
# variable is only assigned inside the `plotted_count < max_plots`
# plotting branch. Setting max_plots ≥ N_SHOTS ensures the plotting
# branch always runs and assigns `display_idx`. We stub the drawing
# so it costs nothing on disk / cpu — but ONLY around the streak
# finder call so we don't cripple our own matplotlib usage.
from contextlib import contextmanager
from unittest.mock import MagicMock as _MagicMock


def _fast_subplots(*args, **kwargs):
    nrows = args[0] if args else kwargs.get('nrows', 1)
    ncols = args[1] if len(args) > 1 else kwargs.get('ncols', 1)
    total = int(nrows) * int(ncols)
    fig = _MagicMock()
    axes = tuple(_MagicMock() for _ in range(total))
    return fig, axes if total > 1 else axes[0]


@contextmanager
def _quiet_cvmi_plots():
    """Scope the plotting stubs so they don't leak into our own drawing."""
    orig_subplots = _cvmi_mod.plt.subplots
    orig_savefig = _cvmi_mod.plt.Figure.savefig
    orig_tight = _cvmi_mod.plt.tight_layout
    orig_close = _cvmi_mod.plt.close
    _cvmi_mod.plt.subplots = _fast_subplots
    _cvmi_mod.plt.Figure.savefig = lambda self, *a, **kw: None
    _cvmi_mod.plt.tight_layout = lambda *a, **kw: None
    _cvmi_mod.plt.close = lambda *a, **kw: None
    try:
        yield
    finally:
        _cvmi_mod.plt.subplots = orig_subplots
        _cvmi_mod.plt.Figure.savefig = orig_savefig
        _cvmi_mod.plt.tight_layout = orig_tight
        _cvmi_mod.plt.close = orig_close


# --- Simulation knobs ---
N_SHOTS = 1000
BG_N_MIN, BG_N_MAX = 50, 300
LOBE_FRACTION = 0.10
PEAK_BIN_MIN, PEAK_BIN_MAX = 16, 26
STREAK_RADIUS_SCALE = 5.0        # Rayleigh scale (px). 0 => unstreaked.
SIGMA_THETA_DEG = 25.0           # angular sweep width per shot
RNG_SEED = 0

# Populated by argparse in main(); simulate() reads them from here so
# the scale is a proper knob rather than a module-level constant.
CFG = {
    'rayleigh_scale': STREAK_RADIUS_SCALE,
    'sigma_theta_deg': SIGMA_THETA_DEG,
    'n_shots': N_SHOTS,
    'out_suffix': '',
}

# --- Detection cut ---
DR_CUT = 5.0
SIG_CUT = 2.0

# --- Covariance / annulus (match plot_azimuthal_covariance.py conventions) ---
CX, CY = 67, 59
N_BINS = 36
DR_HALF = 3.0
RAW_CHANNELS_PER_BIN = 32
SPECTRUM_ROI_START = 13

SIGMA_SLUG = 'explain'


def _out_paths(suffix):
    tag = f'_{suffix}' if suffix else ''
    return {
        'png': os.path.join(_HERE, f'explanation_bootstrap_covariance{tag}.png'),
        'npz': os.path.join(_HERE, f'explanation_bootstrap_covariance{tag}.npz'),
        'sigma_slug': f'{SIGMA_SLUG}{tag}',
    }


def simulate(rng):
    ny, nx = 140, 140
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]

    hits = np.zeros((N_SHOTS, ny, nx), dtype=float)
    mean_energy = np.zeros(N_SHOTS, dtype=float)
    total_hits_within = np.zeros(N_SHOTS, dtype=float)
    is_gaussian = np.ones(N_SHOTS, dtype=bool)
    mask_array = np.ones(N_SHOTS, dtype=bool)
    original_event_number = np.arange(N_SHOTS)

    lobe_only_density = np.zeros((ny, nx), dtype=float)

    profile = _load_noise_profile(NOISE_PROFILE_PATH)
    sigma_theta = np.deg2rad(CFG['sigma_theta_deg'])
    rayleigh_scale = float(CFG['rayleigh_scale'])
    r_max_sq = R_MAX_FROM_CENTER ** 2

    for shot in range(N_SHOTS):
        peak_bin = int(rng.integers(PEAK_BIN_MIN, PEAK_BIN_MAX + 1))
        mock_energy_value = (peak_bin + SPECTRUM_ROI_START) * RAW_CHANNELS_PER_BIN
        re = (mock_energy_value / RAW_CHANNELS_PER_BIN
              - SPECTRUM_ROI_START) * 0.6 + 29.4
        mean_energy[shot] = mock_energy_value

        n_bg = int(rng.integers(BG_N_MIN, BG_N_MAX + 1))
        n_lobe = int(round(LOBE_FRACTION * n_bg))

        streak_mode = rng.uniform(-np.pi, np.pi)
        streak_radius = (rng.rayleigh(scale=rayleigh_scale)
                         if rayleigh_scale > 0 else 0.0)

        if n_lobe > 0:
            theta_streak = rng.normal(loc=streak_mode,
                                      scale=sigma_theta, size=n_lobe)
            lobe_cx = CX + streak_radius * np.cos(theta_streak)
            lobe_cy = CY + streak_radius * np.sin(theta_streak)
            lobe_x = np.empty(n_lobe)
            lobe_y = np.empty(n_lobe)
            filled = 0
            while filled < n_lobe:
                need = n_lobe - filled
                dx, dy = _sample_projected_shell_with_sin2(
                    n=need, re=re, dr=SHELL_THICKNESS,
                    r_max=R_MAX_FROM_CENTER, rng=rng,
                )
                cand_x = lobe_cx[filled:filled + need] + dx
                cand_y = lobe_cy[filled:filled + need] + dy
                keep = ((cand_x - CX) ** 2 + (cand_y - CY) ** 2) <= r_max_sq
                take = int(keep.sum())
                lobe_x[filled:filled + take] = cand_x[keep]
                lobe_y[filled:filled + take] = cand_y[keep]
                filled += take
        else:
            lobe_x = np.empty(0)
            lobe_y = np.empty(0)

        bg_x, bg_y = _sample_bg_from_profile(
            n=n_bg, profile=profile,
            cx=CX, cy=CY, r_max=R_MAX_FROM_CENTER, rng=rng,
        )

        # accumulate lobe-only density for the illustration image
        if lobe_x.size > 0:
            H, _, _ = np.histogram2d(
                lobe_y, lobe_x,
                bins=(np.arange(ny + 1), np.arange(nx + 1))
            )
            lobe_only_density += H

        all_x = np.concatenate([lobe_x, bg_x])
        all_y = np.concatenate([lobe_y, bg_y])

        img = np.zeros((ny, nx))
        for k in range(len(all_x)):
            sigma_e = rng.uniform(0.1, 0.8)
            amp = 1.0 / np.sqrt(2.0 * np.pi) / sigma_e
            gaussian_peak = amp * np.exp(
                -((x_grid - all_x[k]) ** 2 + (y_grid - all_y[k]) ** 2)
                / (2.0 * sigma_e ** 2)
            )
            above = gaussian_peak >= 0.2
            npix = int(above.sum())
            if npix > 0:
                img[above] += 1.0 / npix
        img[104:116, 92:101] = 0.0
        hits[shot] = img
        total_hits_within[shot] = len(all_x)

        if shot % 100 == 0:
            print(f'  synth shot {shot}/{N_SHOTS}')

    return {
        'hits': hits,
        'mean_energy': mean_energy,
        'is_gaussian': is_gaussian,
        'mask_array': mask_array,
        'total_hits_within': total_hits_within,
        'original_event_number': original_event_number,
        'lobe_density': lobe_only_density,
        'ny': ny,
        'nx': nx,
    }


def azimuthal_theta_matrix(hits, mean_energy, keep_idx):
    """Per-shot azimuthal intensity in the annulus [re-DR_HALF, re+DR_HALF).
    Returns H shape (K, N_BINS) where K = len(keep_idx)."""
    ny, nx = hits.shape[1:]
    y, x = np.mgrid[0:ny, 0:nx]
    r_map = np.hypot(x - CX, y - CY)
    theta = np.arctan2(y - CY, x - CX) + np.pi
    theta_bin_map = np.clip(
        np.floor(theta / (2.0 * np.pi / N_BINS)).astype(int),
        0, N_BINS - 1,
    )
    rows = []
    for i in keep_idx:
        re = (mean_energy[i] / RAW_CHANNELS_PER_BIN
              - SPECTRUM_ROI_START) * 0.6 + 29.4
        annulus = (r_map >= re - DR_HALF) & (r_map < re + DR_HALF)
        if not annulus.any():
            continue
        img = hits[i]
        rows.append(np.bincount(
            theta_bin_map[annulus],
            weights=img[annulus].astype(float, copy=False),
            minlength=N_BINS,
        ))
    if not rows:
        return np.empty((0, N_BINS))
    return np.vstack(rows)


def draw_lobe(ax, density, title):
    vmax = float(np.percentile(density, 99.5)) or 1.0
    ax.imshow(density, cmap='viridis', origin='lower', vmin=0, vmax=vmax)
    ring_theta = np.linspace(0, 2 * np.pi, 400)
    # Trace annulus at re for the median peak_bin (illustrative only).
    peak_bin_mid = (PEAK_BIN_MIN + PEAK_BIN_MAX) // 2
    re = ((peak_bin_mid + SPECTRUM_ROI_START)
          - SPECTRUM_ROI_START) * 0.6 + 29.4
    for r in (re - DR_HALF, re + DR_HALF):
        ax.plot(CX + r * np.cos(ring_theta),
                CY + r * np.sin(ring_theta),
                ls='--', color='white', lw=0.9, alpha=0.7)
    ax.plot(CX, CY, marker='+', color='white', ms=8, mew=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def draw_covariance(ax, cov, title):
    off_max = float(np.nanmax(np.abs(cov)))
    if not np.isfinite(off_max) or off_max == 0:
        off_max = 1e-12
    im = ax.imshow(cov, cmap='RdBu_r', origin='lower',
                   extent=[0, 360, 0, 360],
                   vmin=-off_max, vmax=off_max, aspect='equal')
    for g in (90.0, 270.0):
        ax.axvline(g, color='k', ls=':', lw=0.8, alpha=0.55)
        ax.axhline(g, color='k', ls=':', lw=0.8, alpha=0.55)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_xlabel(r'$\theta_j$ [deg]')
    ax.set_ylabel(r'$\theta_i$ [deg]')
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def main():
    global N_SHOTS
    ap = argparse.ArgumentParser()
    ap.add_argument('--rayleigh-scale', type=float, default=STREAK_RADIUS_SCALE,
                    help='Rayleigh scale (px) for the streak radius. '
                         '0 => unstreaked.')
    ap.add_argument('--sigma-theta-deg', type=float, default=SIGMA_THETA_DEG,
                    help='Angular sweep width per shot (deg).')
    ap.add_argument('--n-shots', type=int, default=N_SHOTS,
                    help='Number of synthetic shots.')
    ap.add_argument('--out-suffix', default='',
                    help='Suffix appended to output filenames; '
                         'empty preserves the original streaked output name.')
    args = ap.parse_args()

    CFG['rayleigh_scale'] = args.rayleigh_scale
    CFG['sigma_theta_deg'] = args.sigma_theta_deg
    CFG['n_shots'] = args.n_shots
    CFG['out_suffix'] = args.out_suffix
    N_SHOTS = args.n_shots
    paths = _out_paths(args.out_suffix)
    sigma_slug = paths['sigma_slug']

    rng = np.random.default_rng(RNG_SEED)
    print(f'--- simulating {N_SHOTS} shots '
          f'(Rayleigh scale={CFG["rayleigh_scale"]} px, '
          f'sigma_theta={CFG["sigma_theta_deg"]} deg) ---')
    sim = simulate(rng)

    print('--- running compute_circular_wiggle_analysis ---')
    # cvmi writes to LEGACY_OUTPUTS_ROOT/circular_wiggler_<suffix>_batch_metrics/
    # unconditionally now, so no chdir dance is needed.
    with _quiet_cvmi_plots():
        compute_circular_wiggle_analysis(
            mask_array=sim['mask_array'],
            run_id=sigma_slug,
            output_dir_suffix=f'sim_sigma_{sigma_slug}deg',
            images=sim['hits'],
            hits=sim['hits'],
            mean_energy=sim['mean_energy'],
            is_gaussian=sim['is_gaussian'],
            total_hit_within_mask=sim['total_hits_within'],
            original_event_number=sim['original_event_number'],
            annulus_mask=None,
            max_plots=N_SHOTS + 1,
        )

    peak_pos = np.load(os.path.join(
        _cvmi_mod.LEGACY_OUTPUTS_ROOT,
        f'circular_wiggler_sim_sigma_{sigma_slug}deg_batch_metrics',
        f'data_peak_positions_run_{sigma_slug}.npy',
    ), allow_pickle=True).item()
    x_est = np.asarray(peak_pos['x'], dtype=float)
    y_est = np.asarray(peak_pos['y'], dtype=float)
    sig = np.asarray(peak_pos['significance'], dtype=float)
    n = min(len(x_est), N_SHOTS)
    x_est, y_est, sig = x_est[:n], y_est[:n], sig[:n]
    dr = np.hypot(x_est - CX, y_est - CY)

    keep = np.where((dr > DR_CUT) & (sig > SIG_CUT))[0]
    print(f'  detected {len(keep)} / {n} shots  '
          f'(|r|>{DR_CUT}, sigma>{SIG_CUT})')

    H = azimuthal_theta_matrix(sim['hits'], sim['mean_energy'], keep)
    print(f'  covariance built from H.shape={H.shape}')
    cov = np.cov(H, rowvar=False)
    diag = np.eye(N_BINS, dtype=bool)
    cov_plot = np.where(diag, np.nan, cov)

    # Cache everything expensive so re-plotting doesn't require another
    # streak-finder run.
    cache_path = paths['npz']
    np.savez_compressed(
        cache_path,
        lobe_density=sim['lobe_density'].astype(np.float32),
        cov=cov,
        H=H,
        keep_idx=keep,
        peak_x=x_est, peak_y=y_est, significance=sig, dr=dr,
        mean_energy=sim['mean_energy'],
        n_shots=N_SHOTS,
        rayleigh_scale=CFG['rayleigh_scale'],
        sigma_theta_deg=CFG['sigma_theta_deg'],
    )
    print(f'  cached intermediates to {cache_path}')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    draw_lobe(
        axes[0], sim['lobe_density'],
        f'accumulated lobe density  ({N_SHOTS} shots)\n'
        f'Rayleigh r_scale={CFG["rayleigh_scale"]:g} px, '
        f'sigma_theta={CFG["sigma_theta_deg"]:g} deg',
    )
    draw_covariance(
        axes[1], cov_plot,
        f'azimuthal covariance, detected shots only  ({len(keep)} shots)\n'
        f'|r|>{DR_CUT:g} px  &  sigma>{SIG_CUT:g}',
    )
    fig.suptitle(
        'Full-simulation bootstrap → streak finder → covariance of '
        'detected shots',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(paths['png'], dpi=150)
    plt.close(fig)
    print(f'wrote {paths["png"]}')


if __name__ == '__main__':
    main()
