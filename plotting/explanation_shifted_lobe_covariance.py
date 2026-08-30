#!/usr/bin/env python3
"""Pedagogical figure: how a shifted lobe distorts the azimuthal covariance.

Generates two figures under batch_analysis/:

  explanation_unshifted.png
      Row of 2 panels.
      Left  — noise-free density map of the un-streaked lobe (sin^2 phi
              along y) centered on (cx, cy), with the analysis annulus
              [re - DR_HALF, re + DR_HALF) traced in dashed white.
      Right — sample azimuthal covariance across 100 synthetic shots
              drawn from the same lobe (Poisson counts per pixel),
              90 x 90 bins (4 deg each), diagonal masked.

  explanation_shifted_lr.png
      2 x 2 grid.  Row 0 : lobe shifted by (-5, 0) px in the sample
      frame; Row 1 : lobe shifted by (+5, 0) px.  The analysis annulus
      stays centered on the un-shifted origin, so the lobe intensity
      is now non-uniform around the annulus and the covariance
      picks up a stripe pattern.

Reuses conventions from analysis_library / plot_azimuthal_covariance.py:
  * re = 30 px
  * DR_HALF = 3 px, N_BINS = 36 (10 deg each)
  * lobe drawn with the 3D uniform shell + sin^2(phi_2d) weight, exactly
    the un-streaked draw from run_bootstrap_with_slurm_sigma.py.

Deterministic (rng seed = 0).  Requires conda env CXI.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# --- Grid + analysis constants (match plot_azimuthal_covariance.py) ---
NY = NX = 125
CX = CY = NX // 2                 # 62
RE = 30.0                         # photoline radius in px
DR_HALF = 3.0
N_BINS = 36                       # 10 deg per bin
SHELL_THICKNESS = 5.0             # matches _sample_unstreaked_lobe
N_SHOTS = 100
HITS_PER_SHOT = 200               # Poisson-mean per shot (arbitrary but realistic)
SHIFT_PX = 5.0

OUT_DIR = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
           'batch_analysis_results/bootstrap_covariance_explanations')


def _sample_unstreaked_lobe(re, n_samples, rng):
    """Sample n_samples (dx, dy) offsets from a 3D uniform shell of radius re,
    thickness SHELL_THICKNESS, projected to xy, with sin^2(phi_2d) azimuthal
    weighting (lobes along y). Identical to run_bootstrap_with_slurm_sigma.py."""
    out_x = np.empty(n_samples)
    out_y = np.empty(n_samples)
    filled = 0
    batch = int(n_samples * 3) + 1024
    while filled < n_samples:
        r3d = rng.uniform(re - SHELL_THICKNESS / 2.0,
                          re + SHELL_THICKNESS / 2.0, size=batch)
        cos_t = rng.uniform(-1.0, 1.0, size=batch)
        sin_t = np.sqrt(np.clip(1.0 - cos_t ** 2, 0.0, 1.0))
        phi_3d = rng.uniform(0.0, 2.0 * np.pi, size=batch)
        dx = r3d * sin_t * np.cos(phi_3d)
        dy = r3d * sin_t * np.sin(phi_3d)
        phi_2d = np.arctan2(dy, dx)
        u = rng.uniform(0.0, 1.0, size=batch)
        keep = u < np.sin(phi_2d) ** 2
        take = min(int(keep.sum()), n_samples - filled)
        idx = np.flatnonzero(keep)[:take]
        out_x[filled:filled + take] = dx[idx]
        out_y[filled:filled + take] = dy[idx]
        filled += take
    return out_x, out_y


def lobe_density_map(shift_x=0.0, shift_y=0.0, n_samples=200_000, rng=None):
    """High-statistics density image of the (possibly shifted) lobe on the
    detector grid. Lobe is drawn about (CX + shift_x, CY + shift_y)."""
    if rng is None:
        rng = np.random.default_rng(0)
    dx, dy = _sample_unstreaked_lobe(RE, n_samples, rng)
    x = CX + shift_x + dx
    y = CY + shift_y + dy
    H, _, _ = np.histogram2d(y, x,
                             bins=(np.arange(NY + 1), np.arange(NX + 1)))
    return H


def theta_bin_and_r():
    """Grid r and theta_bin about the fixed analysis center (CX, CY)."""
    y, x = np.mgrid[0:NY, 0:NX]
    r = np.hypot(x - CX, y - CY)
    theta = np.arctan2(y - CY, x - CX) + np.pi        # [0, 2*pi)
    theta_bin = np.clip(
        np.floor(theta / (2.0 * np.pi / N_BINS)).astype(int),
        0, N_BINS - 1,
    )
    return r, theta_bin


def synthesize_shots(shift_x, shift_y, rng):
    """Return H shape (N_SHOTS, N_BINS): per-shot summed intensity per θ-bin
    in the annulus, where each shot is a Poisson-count draw from the lobe."""
    r_map, theta_bin_map = theta_bin_and_r()
    annulus = (r_map >= RE - DR_HALF) & (r_map < RE + DR_HALF)
    tb_ann = theta_bin_map[annulus]                     # 1-D per-pixel θ-bin
    rows = np.zeros((N_SHOTS, N_BINS), dtype=float)
    for k in range(N_SHOTS):
        n_hits = rng.poisson(HITS_PER_SHOT)
        if n_hits == 0:
            continue
        dx, dy = _sample_unstreaked_lobe(RE, n_hits, rng)
        x = CX + shift_x + dx
        y = CY + shift_y + dy
        img, _, _ = np.histogram2d(
            y, x, bins=(np.arange(NY + 1), np.arange(NX + 1))
        )
        rows[k] = np.bincount(tb_ann,
                              weights=img[annulus].astype(float, copy=False),
                              minlength=N_BINS)
    return rows


def sample_covariance(H):
    if H.shape[0] < 2:
        return None
    cov = np.cov(H, rowvar=False)
    diag = np.eye(N_BINS, dtype=bool)
    return np.where(diag, np.nan, cov)


def draw_lobe(ax, density, title):
    vmax = float(np.percentile(density, 99.5)) or 1.0
    ax.imshow(density, cmap='viridis', origin='lower', vmin=0, vmax=vmax)
    theta = np.linspace(0, 2 * np.pi, 400)
    for r in (RE - DR_HALF, RE + DR_HALF):
        ax.plot(CX + r * np.cos(theta), CY + r * np.sin(theta),
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


def figure_unshifted(rng):
    density = lobe_density_map(0.0, 0.0, n_samples=200_000,
                               rng=np.random.default_rng(0))
    H = synthesize_shots(0.0, 0.0, rng)
    cov = sample_covariance(H)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    draw_lobe(axes[0], density,
              f'unshifted lobe  (re={RE:g} px)\nannulus [re-{DR_HALF:g}, '
              f're+{DR_HALF:g}) traced')
    draw_covariance(axes[1], cov,
                    f'azimuthal covariance  ({N_SHOTS} synthetic shots, '
                    f'Poisson({HITS_PER_SHOT}) hits/shot)')
    fig.suptitle('Unshifted lobe: symmetric two-blob azimuthal covariance',
                 fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, 'explanation_unshifted.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'wrote {out}')


def figure_shifted_lr(rng):
    left_density = lobe_density_map(-SHIFT_PX, 0.0,
                                    n_samples=200_000,
                                    rng=np.random.default_rng(1))
    right_density = lobe_density_map(+SHIFT_PX, 0.0,
                                     n_samples=200_000,
                                     rng=np.random.default_rng(2))
    H_left = synthesize_shots(-SHIFT_PX, 0.0, rng)
    H_right = synthesize_shots(+SHIFT_PX, 0.0, rng)
    cov_left = sample_covariance(H_left)
    cov_right = sample_covariance(H_right)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.4))
    draw_lobe(axes[0, 0], left_density,
              f'lobe shifted by ({-SHIFT_PX:+g}, 0) px  '
              '(annulus stays on unshifted center)')
    draw_covariance(axes[0, 1], cov_left,
                    f'azimuthal covariance, {N_SHOTS} shots')
    draw_lobe(axes[1, 0], right_density,
              f'lobe shifted by ({+SHIFT_PX:+g}, 0) px  '
              '(annulus stays on unshifted center)')
    draw_covariance(axes[1, 1], cov_right,
                    f'azimuthal covariance, {N_SHOTS} shots')
    fig.suptitle(
        f'Lobe shifted ±{SHIFT_PX:g} px: azimuthal covariance picks up '
        'a directional stripe',
        fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, 'explanation_shifted_lr.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'wrote {out}')


def main():
    rng = np.random.default_rng(0)
    figure_unshifted(rng)
    figure_shifted_lr(rng)


if __name__ == '__main__':
    main()
