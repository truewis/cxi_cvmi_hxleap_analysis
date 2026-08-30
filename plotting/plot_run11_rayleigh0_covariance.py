#!/usr/bin/env python3
"""40 000-shot Rayleigh=0 azimuthal covariance from run11.

Under
  batch_analysis_results/legacy_outputs/circular_wiggler_run11/
    wiggler_sigma_sweep_metrics/
the SLURM sweep wrote 2 shot stacks at Rayleigh scale = 0:
  shot_hits_full_sigma_0deg.npz  (20 000 shots)
  shot_hits_full_sigma_5deg.npz  (20 000 shots)
Both are Rayleigh=0 so sigma_theta has no physical effect (streak arm
has zero length); the two stacks are independent draws of the same
distribution. Pooling gives a 40 000-shot unstreaked sample.

Produces:
  batch_analysis_results/circular_wiggler_rayleigh0_run11/
    plot_rayleigh0_covariance.png            (single-panel smoothed cov)
    plot_rayleigh0_covariance_by_offset.png  (1x4: r_off = {-6,-4,-2,0})
    plot_rayleigh0_extdiff_vs_combined_60000.png
        3-col: this run's 40k cov, the streaked 60k combined cov
        (sigma_theta=0), and their smoothed diff.
    *.npz  cached matrices for each of the above.

Conventions match plot_combined_run6_9_10_covariance.py:
  cx, cy = 67, 59; N_BINS = 36 (10 deg per bin); DR_HALF = 3 px;
  probability-normalized shots; NaN-aware smoothing at sigma=1.5 bins
  with mode='wrap'; diagonal masked.

Requires conda env CXI.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

RUN_DIRS = [
    ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
     'batch_analysis_results/legacy_outputs/circular_wiggler_run11/'
     'wiggler_sigma_sweep_metrics'),
    ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
     'batch_analysis_results/legacy_outputs/circular_wiggler_run12/'
     'wiggler_sigma_sweep_metrics'),
]
COMBINED_60K_NPZ = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
                    'batch_analysis_results/circular_wiggler_combined_60000/'
                    'plot_combined_covariance.npz')

OUT_DIR = ('/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/'
           'batch_analysis_results/circular_wiggler_rayleigh0_run11_12')

CX, CY = 67, 59
N_BINS = 36
DR_HALF = 3.0
RAW_CHANNELS_PER_BIN = 32
SPECTRUM_ROI_START = 13
SMOOTH_SIGMA_BINS = 1.5
GUIDES_DEG = (90.0, 270.0)
R_OFFSETS = [-6, -4, -2, 0]


def load_pool():
    """Return (hits, mean_energy) from every Rayleigh=0 stack across
    RUN_DIRS. Filters out stacks that don't have the expected shot count
    (defensive against partial smoke-test writes) or that have nonzero
    streak_radius (defensive against RAYLEIGH_SCALE mis-set)."""
    stacks_h, stacks_e = [], []
    expected_n = 20000
    for run_dir in RUN_DIRS:
        run_tag = os.path.basename(os.path.dirname(run_dir))  # circular_wiggler_run<N>
        for fn in sorted(os.listdir(run_dir)):
            if not fn.startswith('shot_hits_full_sigma_') or not fn.endswith('.npz'):
                continue
            p = os.path.join(run_dir, fn)
            d = np.load(p)
            n = d['hits'].shape[0]
            if n < expected_n // 2:
                print(f'  [skip] {run_tag}/{fn}: only {n} shots (partial/smoke-test)')
                continue
            max_streak = float(d['streak_radius_true_px'].max())
            if max_streak > 0:
                print(f'  [skip] {run_tag}/{fn}: streak_radius max={max_streak:.3g} != 0')
                continue
            stacks_h.append(d['hits'])
            stacks_e.append(d['mean_energy'])
            print(f'  loaded {run_tag}/{fn}: {n} shots')
    return (np.concatenate(stacks_h, axis=0),
            np.concatenate(stacks_e, axis=0))


def azimuthal_prob_matrix(hits, mean_energy, r_offset=0.0):
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
              - SPECTRUM_ROI_START) * 0.6 + 29.4 + r_offset
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


def nan_gaussian_filter(A, sigma):
    A = np.asarray(A, dtype=float)
    nan_mask = np.isnan(A)
    V = np.where(nan_mask, 0.0, A)
    W = (~nan_mask).astype(float)
    Vs = gaussian_filter(V, sigma=sigma, mode='wrap')
    Ws = gaussian_filter(W, sigma=sigma, mode='wrap')
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(Ws > 1e-12, Vs / Ws, np.nan)


def draw(ax, M, title, vmax):
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-15
    im = ax.imshow(M, cmap='RdBu_r', origin='lower',
                   extent=[0, 360, 0, 360],
                   vmin=-vmax, vmax=vmax, aspect='equal')
    for g in GUIDES_DEG:
        ax.axvline(g, color='k', ls=':', lw=0.7, alpha=0.55)
        ax.axhline(g, color='k', ls=':', lw=0.7, alpha=0.55)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_xlabel(r'$\theta_j$ [deg]', fontsize=8)
    ax.set_ylabel(r'$\theta_i$ [deg]', fontsize=8)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    diag = np.eye(N_BINS, dtype=bool)

    print('--- pooling run11 stacks ---')
    hits, en = load_pool()
    n_shots = hits.shape[0]
    print(f'  pooled: {n_shots} shots')

    # Cov for every r_offset from the same pooled stack.
    cov_raw = {}
    cov_sm = {}
    for r_off in R_OFFSETS:
        H = azimuthal_prob_matrix(hits, en, r_off)
        c = np.cov(H, rowvar=False)
        cov_raw[r_off] = c
        cov_sm[r_off] = nan_gaussian_filter(
            np.where(diag, np.nan, c), SMOOTH_SIGMA_BINS)
        print(f'  r_off={r_off:+d} done '
              f'(|cov_sm|_max={np.nanmax(np.abs(cov_sm[r_off])):.2e})')

    # ---- Plot 1: single-panel smoothed cov (r_off = 0) ----
    v0 = float(np.nanmax(np.abs(cov_sm[0])))
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.2))
    draw(ax, cov_sm[0],
         f'Rayleigh=0 pooled cov (r_off=0)\nN = {n_shots} shots '
         '(run11: 2×20 000 pooled)',
         v0)
    fig.suptitle(
        f'40 000-shot Rayleigh=0 azimuthal covariance (probability-normalized, '
        f'smoothed σ={SMOOTH_SIGMA_BINS:g} bins, mode=wrap)',
        fontweight='bold', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT_DIR, 'plot_rayleigh0_covariance.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'wrote {p}')

    # ---- Plot 2: radial-offset sweep (1 x 4) ----
    v_max = max(float(np.nanmax(np.abs(cov_sm[r]))) for r in R_OFFSETS)
    fig, axes = plt.subplots(1, len(R_OFFSETS), figsize=(3.5 * len(R_OFFSETS), 3.9))
    for c_idx, r_off in enumerate(R_OFFSETS):
        label = f'Rayleigh=0  |  r_off = {r_off:+d} px  |  N = {n_shots}'
        draw(axes[c_idx], cov_sm[r_off], label, v_max)
    fig.suptitle(
        f'{n_shots}-shot Rayleigh=0 azimuthal covariance across annulus radial '
        f'offsets\n(annulus = [re + r_off − {DR_HALF:g}, re + r_off + {DR_HALF:g}) px, '
        f'smoothed σ={SMOOTH_SIGMA_BINS:g} bins, mode=wrap)',
        fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(OUT_DIR, 'plot_rayleigh0_covariance_by_offset.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'wrote {p}')

    # ---- Plot 3: diff vs streaked 60 000-shot (sigma_theta = 0) ----
    combined = np.load(COMBINED_60K_NPZ)
    idx0 = int(np.where(combined['sigma_values'] == 0)[0][0])
    streaked_cov = combined['cov_raw'][idx0]                       # (N_BINS, N_BINS)
    streaked_n = int(combined['n_shots_per_sigma'][idx0])
    streaked_sm = nan_gaussian_filter(
        np.where(diag, np.nan, streaked_cov), SMOOTH_SIGMA_BINS)
    diff_raw = cov_raw[0] - streaked_cov
    diff_sm = nan_gaussian_filter(np.where(diag, np.nan, diff_raw),
                                   SMOOTH_SIGMA_BINS)
    v_cov = max(v0, float(np.nanmax(np.abs(streaked_sm))))
    v_diff = float(np.nanmax(np.abs(diff_sm))) or 1e-15

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    draw(axes[0], cov_sm[0],
         f'Rayleigh=0 (run11)\nN = {n_shots}', v_cov)
    draw(axes[1], streaked_sm,
         f'streaked σ_θ=0 (run6+9+10)\nN = {streaked_n}', v_cov)
    draw(axes[2], diff_sm,
         'Rayleigh=0 − streaked σ_θ=0  (smoothed diff)', v_diff)
    fig.suptitle(
        'Rayleigh=0 40 000-shot vs streaked 60 000-shot σ_θ=0 covariance '
        f'(probability-normalized, smoothed σ={SMOOTH_SIGMA_BINS:g} bins, '
        'mode=wrap)',
        fontweight='bold', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT_DIR, 'plot_rayleigh0_extdiff_vs_combined_60000.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f'wrote {p}')

    np.savez_compressed(
        os.path.join(OUT_DIR, 'plot_rayleigh0_covariance.npz'),
        cov_raw=cov_raw[0],
        cov_sm=np.nan_to_num(cov_sm[0]),
        n_shots=n_shots,
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
    )
    np.savez_compressed(
        os.path.join(OUT_DIR, 'plot_rayleigh0_covariance_by_offset.npz'),
        r_offsets=np.array(R_OFFSETS),
        cov_raw=np.stack([cov_raw[r] for r in R_OFFSETS]),
        cov_sm=np.stack([np.nan_to_num(cov_sm[r]) for r in R_OFFSETS]),
        n_shots=n_shots,
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
    )
    np.savez_compressed(
        os.path.join(OUT_DIR, 'plot_rayleigh0_extdiff_vs_combined_60000.npz'),
        cov_rayleigh0=cov_raw[0],
        cov_streaked=streaked_cov,
        cov_rayleigh0_sm=np.nan_to_num(cov_sm[0]),
        cov_streaked_sm=np.nan_to_num(streaked_sm),
        diff_raw=diff_raw,
        diff_sm=np.nan_to_num(diff_sm),
        n_rayleigh0_shots=n_shots,
        n_streaked_shots=streaked_n,
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
    )


if __name__ == '__main__':
    main()
