#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the notebook function (configured to handle batch array processing)
from analysis_library.cvmi import compute_circular_wiggle_analysis

# Lobe streaking configuration:
#   - per image, draw one streak-mode angle uniformly in [-pi, pi]
#   - per image, draw one streak radius r ~ Rayleigh(scale=STREAK_RADIUS_SCALE)
#     (Var(r^2)/<r^4> = 1/2, consistent with 2D Gaussian streak components)
#   - per lobe electron, sample its OWN streak angle ~ Normal(streak_mode, SIGMA_THETA_DEG),
#     and displace its lobe centroid from (cx, cy) by that shot's r at the per-electron angle
#   - then sample the electron position around its displaced centroid by drawing an
#     isotropic point on a 3D spherical shell of radius re and thickness SHELL_THICKNESS,
#     projecting to the xy plane, and rejection-keeping on sin^2(phi_2d) so the projected
#     angular pattern remains lobes along the y-axis (matching analysis_library.cvmi).
#   - Every electron (lobe + backgrounds) is constrained to lie within R_MAX_FROM_CENTER
#     of (cx, cy).
SIGMA_THETA_DEG = 10.0
STREAK_RADIUS_SCALE = 5.0  # Rayleigh scale parameter sigma (mean r = sigma*sqrt(pi/2))
SHELL_THICKNESS = 5.0      # dr: uniform-in-radius thickness of the 3D emission shell
R_MAX_FROM_CENTER = 65.0   # hard cutoff on |(x,y) - (cx,cy)| for every simulated electron


def _sample_projected_shell_with_sin2(n, re, dr, cx, cy, r_max, rng):
    """Sample n points from a uniform 3D shell of radius re and thickness dr,
    project to the xy plane, and reject to keep a sin^2(phi_2d) azimuthal
    distribution (lobes along y). Returns dx, dy offsets from (cx, cy),
    already guaranteed to lie within r_max of (cx, cy) (since re + dr/2 < r_max
    in this simulation, but we still enforce it defensively)."""
    if n <= 0:
        return np.empty(0), np.empty(0)
    out_x = np.empty(n)
    out_y = np.empty(n)
    filled = 0
    batch = int(n * 3) + 64  # sin^2 acceptance is ~1/2 on average
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


def run_bootstrap_simulation(i_values, num_runs_per_i=100, num_lobe_electrons=20):
    output_dir_experiment = './wiggler_density_sweep_metrics_lobe'
    os.makedirs(output_dir_experiment, exist_ok=True)

    cx, cy = 67, 59
    ny, nx = 140, 140
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]

    peak_bin = 25
    exclusion_radius_sq = 20**2
    bg_sigma = 25.0

    # Back-calculate energy value so raw_energy_to_bin_idx(energy) returns exactly peak_bin (25)
    mock_energy_value = (peak_bin + 13) * 32

    # re must match analysis_library.cvmi: re = (energy/32 - 13)*0.6 + 29.4
    re = (mock_energy_value / 32 - 13) * 0.6 + 29.4

    sigma_theta = np.deg2rad(SIGMA_THETA_DEG)
    rng = np.random.default_rng()
    r_max_sq = R_MAX_FROM_CENTER ** 2

    for i_val in i_values:
        print(f"Generating simulated data structures for i = {i_val}...")

        num_gaussian_electrons = 4 * i_val
        num_uniform_electrons = 6 * i_val

        # Pre-allocate batch arrays to match the experimental function's expectations
        sim_hits = np.zeros((num_runs_per_i, ny, nx))
        sim_images = np.zeros((num_runs_per_i, ny, nx))  # Required for tracking frame shapes
        sim_mean_energy = np.full((num_runs_per_i,), mock_energy_value, dtype=float)
        sim_is_gaussian = np.full((num_runs_per_i,), True, dtype=bool)
        sim_mask_array = np.full((num_runs_per_i,), True, dtype=bool)
        sim_total_hit_within_mask = np.zeros((num_runs_per_i,))
        sim_original_event_number = np.arange(num_runs_per_i)

        for run_idx in range(num_runs_per_i):
            img = np.zeros((ny, nx))

            # --- Streaked lobe electrons ---
            # Step 1: pick one streak-mode angle uniformly per image
            streak_mode = np.random.uniform(-np.pi, np.pi)

            # Step 1b: draw one streak radius per image from Rayleigh(scale=STREAK_RADIUS_SCALE)
            # Jun Wang's thesis, p.78
            streak_radius = np.random.rayleigh(scale=STREAK_RADIUS_SCALE)

            # Step 2: per electron, draw its own streak angle ~ Normal(streak_mode, sigma_theta)
            # and displace its OWN lobe centroid from (cx, cy) by this shot's streak_radius
            theta_streak = np.random.normal(loc=streak_mode, scale=sigma_theta, size=num_lobe_electrons)
            lobe_cx_arr = cx + streak_radius * np.cos(theta_streak)
            lobe_cy_arr = cy + streak_radius * np.sin(theta_streak)

            # Step 3: sample each lobe electron around its displaced centroid.
            # Physical model: emission from a 3D spherical shell of radius re and
            # thickness SHELL_THICKNESS, projected to the detector plane. Rejection
            # sampling on the projected azimuth keeps a sin^2 lobe pattern along y.
            # All electrons are also constrained to lie within R_MAX_FROM_CENTER of
            # (cx, cy) (measured about the *shot* center, not the displaced centroid).
            lobe_x = np.empty(num_lobe_electrons)
            lobe_y = np.empty(num_lobe_electrons)
            filled = 0
            while filled < num_lobe_electrons:
                need = num_lobe_electrons - filled
                dx, dy = _sample_projected_shell_with_sin2(
                    n=need, re=re, dr=SHELL_THICKNESS,
                    cx=cx, cy=cy, r_max=R_MAX_FROM_CENTER, rng=rng,
                )
                cand_x = lobe_cx_arr[filled:filled + need] + dx
                cand_y = lobe_cy_arr[filled:filled + need] + dy
                keep = ((cand_x - cx) ** 2 + (cand_y - cy) ** 2) <= r_max_sq
                take = int(keep.sum())
                lobe_x[filled:filled + take] = cand_x[keep]
                lobe_y[filled:filled + take] = cand_y[keep]
                filled += take

            # --- Background Generation Logic ---
            # Additionally enforce a hard cutoff at R_MAX_FROM_CENTER for all electrons.
            gaussian_x_list, gaussian_y_list = [], []
            while len(gaussian_x_list) < num_gaussian_electrons:
                missing = num_gaussian_electrons - len(gaussian_x_list)
                tx = np.random.normal(loc=cx, scale=bg_sigma, size=missing)
                ty = np.random.normal(loc=cy, scale=bg_sigma, size=missing)
                dsq = (cx - tx)**2 + (cy - ty)**2
                valid = (dsq >= exclusion_radius_sq) & (dsq <= r_max_sq)
                gaussian_x_list.extend(tx[valid])
                gaussian_y_list.extend(ty[valid])

            uniform_x_list, uniform_y_list = [], []
            while len(uniform_x_list) < num_uniform_electrons:
                missing = num_uniform_electrons - len(uniform_x_list)
                tx = np.random.uniform(0, nx, size=missing)
                ty = np.random.uniform(0, ny, size=missing)
                dsq = (cx - tx)**2 + (cy - ty)**2
                valid = (dsq >= exclusion_radius_sq) & (dsq <= r_max_sq)
                uniform_x_list.extend(tx[valid])
                uniform_y_list.extend(ty[valid])

            all_x = np.concatenate([
                lobe_x,
                uniform_x_list[:num_uniform_electrons],
                gaussian_x_list[:num_gaussian_electrons],
            ])
            all_y = np.concatenate([
                lobe_y,
                uniform_y_list[:num_uniform_electrons],
                gaussian_y_list[:num_gaussian_electrons],
            ])

            # Hitfinder engine projection
            for k in range(len(all_x)):
                sigma_e = np.random.uniform(0.1, 0.8)
                amp = 1.0 / np.sqrt(2.0 * np.pi) / sigma_e
                gaussian_peak = amp * np.exp(-((x_grid - all_x[k])**2 + (y_grid - all_y[k])**2) / (2.0 * sigma_e**2))
                above_threshold = (gaussian_peak >= 0.2)
                N = np.sum(above_threshold)
                if N > 0:
                    img[above_threshold] += 1.0 / N

            # Zero artifact block region
            img[104:116, 92:101] = 0

            # Assign single frame to batch matrix
            sim_hits[run_idx] = img
            sim_images[run_idx] = img
            sim_total_hit_within_mask[run_idx] = len(all_x)

        # --- RUN ANALYSIS (BATCH MODE) ---
        print(f"Piping entire batch array into the analysis function...")

        scores_for_i, displacements_full = compute_circular_wiggle_analysis(
            mask_array=sim_mask_array,
            run_id=i_val,
            output_dir_suffix=f"sim_density_lobe_i_{i_val}",
            images=sim_images,
            hits=sim_hits,
            mean_energy=sim_mean_energy,
            is_gaussian=sim_is_gaussian,
            total_hit_within_mask=sim_total_hit_within_mask,
            original_event_number=sim_original_event_number,
            annulus_mask=None,
            max_plots=20
        )

        # Filter out NaN elements from the displacements array to match original logic
        displacements_for_i = displacements_full[~np.isnan(displacements_full)]

        # --- Save Summary Pickles ---
        summary_stats = {
            'i_val': i_val,
            'num_lobe_electrons': num_lobe_electrons,
            'sigma_theta_deg': SIGMA_THETA_DEG,
            'streak_radius_scale': STREAK_RADIUS_SCALE,
            'shell_thickness': SHELL_THICKNESS,
            'r_max_from_center': R_MAX_FROM_CENTER,
            're': re,
            'scores': scores_for_i.tolist() if isinstance(scores_for_i, np.ndarray) else scores_for_i,
            'displacements': displacements_for_i.tolist() if isinstance(displacements_for_i, np.ndarray) else displacements_for_i
        }
        with open(os.path.join(output_dir_experiment, f'stats_i_{i_val}.pkl'), 'wb') as f:
            pickle.dump(summary_stats, f)

        # Draw trend histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.hist(scores_for_i, bins=8, color='crimson', alpha=0.7)
        ax1.set_title(f'Composite Score Distribution (i={i_val}, lobe={num_lobe_electrons})')
        ax2.hist(displacements_for_i, bins=8, color='teal', alpha=0.7)
        ax2.set_title(f'Offsets |r| Distribution (i={i_val}, lobe={num_lobe_electrons})')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir_experiment, f'distribution_i_{i_val}.png'), dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Array Framework for SLURM Simulations (with streaked lobe electrons)")
    parser.add_argument('--i_values', type=int, nargs='+', default=None)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--num_lobe_electrons', type=int, default=50)
    args = parser.parse_args()

    default_array = [5, 8, 11, 14, 17, 20, 23, 26]
    selected_i = args.i_values if args.i_values is not None else default_array

    run_bootstrap_simulation(selected_i, num_runs_per_i=args.iterations, num_lobe_electrons=args.num_lobe_electrons)
