#!/usr/bin/env python3
"""
Bootstrap variant: sweep the streaking length (sigma_theta_deg) per SLURM run
rather than a fixed background electron density.

Per shot:
  - background electrons are drawn from an *empirical* radial density profile
    extracted from run 145 / step 19 (see extract_noise_profile_run145_step19.py
    and NOISE_PROFILE_PATH below). The profile gives per-shot counts per pixel
    at radius r; we integrate 2·π·r·d(r) to get the 1D radial law used to
    sample r, then draw theta uniformly in [-pi, pi]. n_bg per shot is drawn
    from Uniform{BG_N_MIN..BG_N_MAX}.
  - lobe electron count scales with the shot's background:
        n_lobe = round(LOBE_FRACTION * n_bg)
  - streak-mode angle ~ Uniform(-pi, pi)
  - streak radius r ~ Rayleigh(scale=STREAK_RADIUS_SCALE)  (Jun Wang thesis p.78)
  - per lobe electron: theta ~ Normal(streak_mode, sigma_theta) with
    sigma_theta = sweep parameter for this run
  - lobe centroid displaced from (cx, cy) by (r*cos, r*sin) at the per-electron theta
  - electron position sampled from a 3D emission shell of radius re and
    thickness SHELL_THICKNESS, projected to xy, with sin^2(phi_2d) azimuthal
    lobing kept via rejection sampling
  - hard cutoff |(x,y) - (cx,cy)| <= R_MAX_FROM_CENTER on every electron

Larger sigma_theta => broader lobes => less-resolvable wiggle direction.
"""
import os
import argparse
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis_library.cvmi import compute_circular_wiggle_analysis

# --- Constants shared with run_bootstrap_with_slurm_lobe.py ---
STREAK_RADIUS_SCALE = 5.0
SHELL_THICKNESS = 5.0
R_MAX_FROM_CENTER = 65.0

# --- Per-shot sampling ranges ---
BG_N_MIN, BG_N_MAX = 50, 300         # inclusive-inclusive uniform for background n
LOBE_FRACTION = 0.15                 # n_lobe = round(LOBE_FRACTION * n_bg)

# --- Empirical background radial density ---
# Built by extract_noise_profile_run145_step19.py; the .npy sits next to this
# script. If missing at import time the bootstrap raises with a helpful message.
NOISE_PROFILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'noise_profile_run145_step19.npy',
)


def _load_noise_profile(path):
    """Load the empirical background radial density and return an
    inverse-CDF-ready payload:

        {
          'r_centers':   (N_R,)   radial bin centers used by the CDF,
          'density':     (N_R,)   counts/pixel/shot at r_centers,
          'r_grid':      (N_R+1,) monotonic r grid (bin edges),
          'cdf':         (N_R+1,) empirical CDF of 2·π·r·density(r)
                                   evaluated on r_grid, normalized to [0, 1],
          'n_bg_mean_per_shot': float,   expected number of BG electrons/shot
                                          predicted by the profile (informational).
        }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'noise profile not found at {path!r}. Generate it with '
            'extract_noise_profile_run145_step19.py before running the bootstrap.'
        )
    payload = np.load(path, allow_pickle=True).item()
    r_centers = np.asarray(payload['r_centers'], dtype=float)
    density = np.asarray(payload['density'], dtype=float)
    density = np.clip(density, 0.0, None)  # defensive

    # Marginal radial law for a rotationally-symmetric density: p(r) ∝ 2·π·r·d(r).
    weight = 2.0 * np.pi * r_centers * density
    # CDF on the same r_centers grid, evaluated as a trapezoid running sum.
    # We use r_centers as knots for interpolation; the CDF is monotonically
    # nondecreasing so inverse sampling is straightforward.
    dr = np.diff(r_centers)
    # cumulative trapezoid: 0 at r_centers[0], increasing with r.
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (weight[:-1] + weight[1:]) * dr)])
    total = cum[-1]
    if total <= 0:
        raise ValueError(f'noise profile at {path!r} integrates to zero — nothing to sample.')
    cdf = cum / total
    return {
        'r_centers': r_centers,
        'density': density,
        'r_grid': r_centers.copy(),  # inverse-CDF interpolation nodes
        'cdf': cdf,
        'n_bg_mean_per_shot': float(total),
        'source_path': payload.get('source_path', path),
        'ang_half_width_deg': payload.get('ang_half_width_deg', None),
    }


def _sample_bg_from_profile(n, profile, cx, cy, r_max, rng):
    """Draw `n` background electron (x, y) positions from the empirical
    profile. r ~ inverse-CDF-of(2π r d(r)); θ ~ Uniform(-pi, pi); rejects
    positions outside |r| <= r_max."""
    if n <= 0:
        return np.empty(0), np.empty(0)
    r_grid = profile['r_grid']
    cdf = profile['cdf']

    out_x = np.empty(n)
    out_y = np.empty(n)
    filled = 0
    batch = int(n * 1.2) + 32
    while filled < n:
        u = rng.uniform(0.0, 1.0, size=batch)
        r = np.interp(u, cdf, r_grid)
        theta = rng.uniform(-np.pi, np.pi, size=batch)
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        keep = ((x - cx) ** 2 + (y - cy) ** 2) <= r_max * r_max
        take = min(int(keep.sum()), n - filled)
        idx = np.flatnonzero(keep)[:take]
        out_x[filled:filled + take] = x[idx]
        out_y[filled:filled + take] = y[idx]
        filled += take
    return out_x, out_y


def _sample_projected_shell_with_sin2(n, re, dr, r_max, rng):
    """Sample n (dx, dy) offsets from a 3D shell of radius re, thickness dr,
    projected to xy, restricted to sin^2(phi_2d) lobes and |offset| <= r_max."""
    if n <= 0:
        return np.empty(0), np.empty(0)
    out_x = np.empty(n)
    out_y = np.empty(n)
    filled = 0
    batch = int(n * 3) + 64
    while filled < n:
        r3d = rng.uniform(re - dr / 2.0, re + dr / 2.0, size=batch)
        cos_t = rng.uniform(-1.0, 1.0, size=batch)
        sin_t = np.sqrt(np.clip(1.0 - cos_t ** 2, 0.0, 1.0))
        phi_3d = rng.uniform(0.0, 2.0 * np.pi, size=batch)
        dx = r3d * sin_t * np.cos(phi_3d)
        dy = r3d * sin_t * np.sin(phi_3d)
        phi_2d = np.arctan2(dy, dx)
        u = rng.uniform(0.0, 1.0, size=batch)
        keep = (u < np.sin(phi_2d) ** 2) & ((dx * dx + dy * dy) <= r_max * r_max)
        take = min(int(keep.sum()), n - filled)
        idx = np.flatnonzero(keep)[:take]
        out_x[filled:filled + take] = dx[idx]
        out_y[filled:filled + take] = dy[idx]
        filled += take
    return out_x, out_y


def run_bootstrap_simulation(sigma_theta_deg_values, num_runs_per_sigma=1000):
    output_dir_experiment = './wiggler_sigma_sweep_metrics'
    os.makedirs(output_dir_experiment, exist_ok=True)

    cx, cy = 67, 59
    ny, nx = 140, 140
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]

    peak_bin = 25
    mock_energy_value = (peak_bin + 13) * 32
    re = (mock_energy_value / 32 - 13) * 0.6 + 29.4

    rng = np.random.default_rng()
    r_max_sq = R_MAX_FROM_CENTER ** 2

    noise_profile = _load_noise_profile(NOISE_PROFILE_PATH)
    print(f"loaded background profile from {NOISE_PROFILE_PATH}  "
          f"(implied ~{noise_profile['n_bg_mean_per_shot']:.1f} bg electrons/shot)")

    for sigma_theta_deg in sigma_theta_deg_values:
        sigma_theta = np.deg2rad(sigma_theta_deg)
        print(f"\n=== Sweep sigma_theta = {sigma_theta_deg:g} deg ({sigma_theta:.3f} rad) ===")

        sim_hits = np.zeros((num_runs_per_sigma, ny, nx))
        sim_images = np.zeros((num_runs_per_sigma, ny, nx))
        sim_mean_energy = np.full((num_runs_per_sigma,), mock_energy_value, dtype=float)
        sim_is_gaussian = np.full((num_runs_per_sigma,), True, dtype=bool)
        sim_mask_array = np.full((num_runs_per_sigma,), True, dtype=bool)
        sim_total_hit_within_mask = np.zeros((num_runs_per_sigma,))
        sim_original_event_number = np.arange(num_runs_per_sigma)
        sim_n_lobe = np.zeros(num_runs_per_sigma, dtype=int)
        sim_n_bg = np.zeros(num_runs_per_sigma, dtype=int)

        # --- Ground truth per shot (known because we generate it) ---
        # `streak_mode_true` is the streak-axis angle (rad) drawn once per shot.
        # `streak_radius_true` is the Rayleigh-drawn displacement (px) of the
        # lobe centroid from (cx, cy). The truth centre is:
        #     cx_true = cx + streak_radius_true * cos(streak_mode_true)
        #     cy_true = cy + streak_radius_true * sin(streak_mode_true)
        sim_streak_mode_true = np.zeros(num_runs_per_sigma, dtype=float)
        sim_streak_radius_true = np.zeros(num_runs_per_sigma, dtype=float)
        sim_cx_true = np.zeros(num_runs_per_sigma, dtype=float)
        sim_cy_true = np.zeros(num_runs_per_sigma, dtype=float)

        for run_idx in range(num_runs_per_sigma):
            img = np.zeros((ny, nx))

            # Per-shot electron counts.
            n_bg = int(rng.integers(BG_N_MIN, BG_N_MAX + 1))
            n_lobe = int(round(LOBE_FRACTION * n_bg))
            sim_n_bg[run_idx] = n_bg
            sim_n_lobe[run_idx] = n_lobe

            # --- Streaked lobe electrons ---
            streak_mode = rng.uniform(-np.pi, np.pi)
            streak_radius = rng.rayleigh(scale=STREAK_RADIUS_SCALE)
            sim_streak_mode_true[run_idx] = streak_mode
            sim_streak_radius_true[run_idx] = streak_radius
            sim_cx_true[run_idx] = cx + streak_radius * np.cos(streak_mode)
            sim_cy_true[run_idx] = cy + streak_radius * np.sin(streak_mode)

            if n_lobe > 0:
                theta_streak = rng.normal(loc=streak_mode, scale=sigma_theta, size=n_lobe)
                lobe_cx_arr = cx + streak_radius * np.cos(theta_streak)
                lobe_cy_arr = cy + streak_radius * np.sin(theta_streak)

                lobe_x = np.empty(n_lobe)
                lobe_y = np.empty(n_lobe)
                filled = 0
                while filled < n_lobe:
                    need = n_lobe - filled
                    dx, dy = _sample_projected_shell_with_sin2(
                        n=need, re=re, dr=SHELL_THICKNESS,
                        r_max=R_MAX_FROM_CENTER, rng=rng,
                    )
                    cand_x = lobe_cx_arr[filled:filled + need] + dx
                    cand_y = lobe_cy_arr[filled:filled + need] + dy
                    keep = ((cand_x - cx) ** 2 + (cand_y - cy) ** 2) <= r_max_sq
                    take = int(keep.sum())
                    lobe_x[filled:filled + take] = cand_x[keep]
                    lobe_y[filled:filled + take] = cand_y[keep]
                    filled += take
            else:
                lobe_x = np.empty(0)
                lobe_y = np.empty(0)

            # --- Background electrons drawn from the empirical radial density
            # extracted from run 145 step 19. r ~ inverse-CDF of 2π·r·d(r);
            # theta uniform. All electrons enforced within R_MAX_FROM_CENTER.
            bg_x, bg_y = _sample_bg_from_profile(
                n=n_bg, profile=noise_profile,
                cx=cx, cy=cy, r_max=R_MAX_FROM_CENTER, rng=rng,
            )

            all_x = np.concatenate([lobe_x, bg_x])
            all_y = np.concatenate([lobe_y, bg_y])

            # Hitfinder engine projection
            for k in range(len(all_x)):
                sigma_e = rng.uniform(0.1, 0.8)
                amp = 1.0 / np.sqrt(2.0 * np.pi) / sigma_e
                gaussian_peak = amp * np.exp(-((x_grid - all_x[k]) ** 2 + (y_grid - all_y[k]) ** 2) / (2.0 * sigma_e ** 2))
                above_threshold = (gaussian_peak >= 0.2)
                Npix = int(np.sum(above_threshold))
                if Npix > 0:
                    img[above_threshold] += 1.0 / Npix

            # Zero artifact block region
            img[104:116, 92:101] = 0

            sim_hits[run_idx] = img
            sim_images[run_idx] = img
            sim_total_hit_within_mask[run_idx] = len(all_x)

        # --- Run analysis in batch mode ---
        print(f"Piping {num_runs_per_sigma} shots into the wiggle analysis (sigma_theta={sigma_theta_deg:g} deg)...")
        # Encode sigma with a filesystem-friendly slug (dot -> p) so multiple sweep points don't collide.
        sigma_slug = f"{sigma_theta_deg:g}".replace('.', 'p')
        scores, displacements_full = compute_circular_wiggle_analysis(
            mask_array=sim_mask_array,
            run_id=sigma_slug,
            output_dir_suffix=f"sim_sigma_{sigma_slug}deg",
            images=sim_images,
            hits=sim_hits,
            mean_energy=sim_mean_energy,
            is_gaussian=sim_is_gaussian,
            total_hit_within_mask=sim_total_hit_within_mask,
            original_event_number=sim_original_event_number,
            annulus_mask=None,
            max_plots=20,
        )

        displacements = displacements_full[~np.isnan(displacements_full)]

        # --- Save summary ---
        summary_stats = {
            'sigma_theta_deg': sigma_theta_deg,
            'streak_radius_scale': STREAK_RADIUS_SCALE,
            'shell_thickness': SHELL_THICKNESS,
            'r_max_from_center': R_MAX_FROM_CENTER,
            'bg_n_range': [BG_N_MIN, BG_N_MAX],
            'lobe_fraction': LOBE_FRACTION,
            'noise_profile_source': noise_profile.get('source_path',
                                                      NOISE_PROFILE_PATH),
            'noise_profile_ang_half_width_deg': noise_profile.get('ang_half_width_deg', None),
            're': re,
            'mock_energy_value': mock_energy_value,
            'n_bg_per_shot': sim_n_bg.tolist(),
            'n_lobe_per_shot': sim_n_lobe.tolist(),
            # Ground-truth centers per shot (used for accuracy assessment
            # against the (peak_x, peak_y) that compute_circular_wiggle_analysis
            # writes to data_peak_positions_run_{run_id}.npy). cx/cy = (67, 59).
            'streak_mode_true_rad': sim_streak_mode_true.tolist(),
            'streak_radius_true_px': sim_streak_radius_true.tolist(),
            'cx_true': sim_cx_true.tolist(),
            'cy_true': sim_cy_true.tolist(),
            'cx_reference': 67,
            'cy_reference': 59,
            'shot_indices': sim_original_event_number.tolist(),
            'total_hits_per_shot': sim_total_hit_within_mask.tolist(),
            'scores': scores.tolist() if isinstance(scores, np.ndarray) else scores,
            'displacements': displacements.tolist() if isinstance(displacements, np.ndarray) else displacements,
        }
        with open(os.path.join(output_dir_experiment, f'stats_sigma_{sigma_slug}deg.pkl'), 'wb') as f:
            pickle.dump(summary_stats, f)

        # Cache a subsample of the raw generated hit images so the accuracy
        # post-processing script can render true-vs-estimated centre overlays
        # per shot. Sampling ~30 shots is enough for a 10-panel diagnostic and
        # keeps the on-disk footprint modest (~3 MB at 140x140 float32).
        cache_n = min(30, num_runs_per_sigma)
        cache_indices = np.linspace(0, num_runs_per_sigma - 1, cache_n, dtype=int)
        cache_indices = np.unique(cache_indices)
        np.savez_compressed(
            os.path.join(output_dir_experiment, f'shot_cache_sigma_{sigma_slug}deg.npz'),
            shot_indices=cache_indices,
            hits=sim_hits[cache_indices].astype(np.float32),
        )

        # Also cache a set of >3σ *detected* shots so the accuracy plotter can
        # render a second panel figure showing how well truly-detected shots
        # are reconstructed. Reads back the significance array that
        # compute_circular_wiggle_analysis just wrote to disk.
        DETECTED_SIG_THRESHOLD = 3.0
        DETECTED_CACHE_N = 30
        peak_pos_path = os.path.join(
            f'circular_wiggler_sim_sigma_{sigma_slug}deg_batch_metrics',
            f'data_peak_positions_run_{sigma_slug}.npy',
        )
        try:
            peak_data = np.load(peak_pos_path, allow_pickle=True).item()
            sig_arr = np.asarray(peak_data['significance'], dtype=float)
            detected = np.where(sig_arr > DETECTED_SIG_THRESHOLD)[0]
            # Sort by descending sigma so the cache carries the strongest shots
            # in case there are more than DETECTED_CACHE_N of them.
            detected = detected[np.argsort(-sig_arr[detected])]
            detected = detected[:DETECTED_CACHE_N]
            if detected.size:
                # Restore original-index order so panels traverse shots in
                # time rather than by significance.
                detected = np.sort(detected)
                np.savez_compressed(
                    os.path.join(output_dir_experiment,
                                 f'shot_cache_detected_sigma_{sigma_slug}deg.npz'),
                    shot_indices=detected,
                    hits=sim_hits[detected].astype(np.float32),
                    significance=sig_arr[detected].astype(np.float32),
                    threshold=DETECTED_SIG_THRESHOLD,
                )
            else:
                print(f'  [note] no shots exceed {DETECTED_SIG_THRESHOLD}σ; '
                      'detected shot cache not written.')
        except Exception as exc:
            print(f'  [warn] could not build detected shot cache: {exc}')

        # --- Trend histograms ---
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
        ax1.hist(scores, bins=30, color='crimson', alpha=0.75)
        ax1.set_yscale('log')
        ax1.set_title(f'Composite Score  |  sigma_theta={sigma_theta_deg:g} deg')
        ax1.set_xlabel('score')

        ax2.hist(displacements, bins=30, color='teal', alpha=0.75)
        ax2.set_title(f'Wiggle offsets |r|  |  N={len(displacements)}')
        ax2.set_xlabel('|r| [px]')

        ax3.hist(sim_n_bg, bins=np.arange(BG_N_MIN, BG_N_MAX + 2, 10),
                 color='slategray', alpha=0.75, label='n_bg')
        ax3.hist(sim_n_lobe, bins=np.arange(0, int(BG_N_MAX * LOBE_FRACTION) + 5),
                 color='goldenrod', alpha=0.75, label='n_lobe')
        ax3.set_title('per-shot electron counts')
        ax3.legend()

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir_experiment, f'distribution_sigma_{sigma_slug}deg.png'), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap SLURM: sweep sigma_theta (streak length).")
    parser.add_argument('--sigma_theta_deg', type=float, nargs='+', default=None,
                        help="Streak-length sigma values in degrees to sweep")
    parser.add_argument('--iterations', type=int, default=1000,
                        help="Number of Monte-Carlo shots per sigma point")
    args = parser.parse_args()

    default_array = [1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0]
    selected = args.sigma_theta_deg if args.sigma_theta_deg is not None else default_array

    run_bootstrap_simulation(selected, num_runs_per_sigma=args.iterations)
