#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the unmodified original notebook function
from xleap_analysis import compute_circular_wiggle_analysis

def run_bootstrap_simulation(i_values, num_runs_per_i=100):
    output_dir_experiment = './wiggler_density_sweep_metrics'
    os.makedirs(output_dir_experiment, exist_ok=True)
    
    cx, cy = 67, 60
    ny, nx = 140, 140
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]
    
    peak_bin = 25
    exclusion_radius_sq = 20**2
    bg_sigma = 25.0
    
    # Back-calculate energy value so raw_energy_to_bin_idx(energy) returns exactly peak_bin (25)
    # Formula: bin = int((energy / 32) - 13) -> energy = (bin + 13) * 32
    mock_energy_value = (peak_bin + 13) * 32

    for i_val in i_values:
        print(f"Generating simulated data structures for i = {i_val}...")
        
        num_gaussian_electrons = 4 * i_val
        num_uniform_electrons = 6 * i_val
        
        # Pre-allocate batch arrays to match the experimental function's expectations
        sim_hits = np.zeros((num_runs_per_i, ny, nx))
        sim_images = np.zeros((num_runs_per_i, ny, nx)) # Required for tracking frame shapes
        sim_mean_energy = np.full((num_runs_per_i,), mock_energy_value, dtype=float)
        sim_is_gaussian = np.full((num_runs_per_i,), True, dtype=bool)
        sim_mask_array = np.full((num_runs_per_i,), True, dtype=bool)
        sim_total_hit_within_mask = np.zeros((num_runs_per_i,))
        sim_original_event_number = np.arange(num_runs_per_i)

        for run_idx in range(num_runs_per_i):
            img = np.zeros((ny, nx))
            
            # --- Background Generation Logic ---
            gaussian_x_list, gaussian_y_list = [], []
            while len(gaussian_x_list) < num_gaussian_electrons:
                missing = num_gaussian_electrons - len(gaussian_x_list)
                tx = np.random.normal(loc=cx, scale=bg_sigma, size=missing)
                ty = np.random.normal(loc=cy, scale=bg_sigma, size=missing)
                valid = ((cx - tx)**2 + (cy - ty)**2) >= exclusion_radius_sq
                gaussian_x_list.extend(tx[valid])
                gaussian_y_list.extend(ty[valid])
            
            uniform_x_list, uniform_y_list = [], []
            while len(uniform_x_list) < num_uniform_electrons:
                missing = num_uniform_electrons - len(uniform_x_list)
                tx = np.random.uniform(0, nx, size=missing)
                ty = np.random.uniform(0, ny, size=missing)
                valid = ((cx - tx)**2 + (cy - ty)**2) >= exclusion_radius_sq
                uniform_x_list.extend(tx[valid])
                uniform_y_list.extend(ty[valid])
                
            all_x = np.concatenate([uniform_x_list[:num_uniform_electrons], gaussian_x_list[:num_gaussian_electrons]])
            all_y = np.concatenate([uniform_y_list[:num_uniform_electrons], gaussian_y_list[:num_gaussian_electrons]])

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

        # --- RUN ANALYSIS ---
        # Call the unmodified function by passing a 1-element slice loop or letting it process the batch
        scores_for_i = []
        displacements_for_i = []
        
        print(f"Piping batch array into the analysis function...")
        for run_idx in range(num_runs_per_i):
            # Create a single element mock array slice to isolate one event at a time
            max_val, dr = compute_circular_wiggle_analysis(
                mask_array=sim_mask_array[run_idx:run_idx+1],
                run_id=i_val,
                output_dir_suffix=f"sim_density_i_{i_val}",
                images=sim_images[run_idx:run_idx+1],
                hits=sim_hits[run_idx:run_idx+1],
                mean_energy=sim_mean_energy[run_idx:run_idx+1],
                is_gaussian=sim_is_gaussian[run_idx:run_idx+1],
                total_hit_within_mask=sim_total_hit_within_mask[run_idx:run_idx+1],
                original_event_number=sim_original_event_number[run_idx:run_idx+1],
                annulus_mask=None, # Simulation template handles zeroing manually
                max_plots=1 # Prevent it from generating 100 images per density unless needed
            )
            scores_for_i.append(max_val)
            if not np.isnan(dr):
                displacements_for_i.append(dr)

        # --- Save Summary Pickles ---
        summary_stats = {
            'i_val': i_val,
            'scores': scores_for_i,
            'displacements': displacements_for_i
        }
        with open(os.path.join(output_dir_experiment, f'stats_i_{i_val}.pkl'), 'wb') as f:
            pickle.dump(summary_stats, f)

        # Draw trend histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.hist(scores_for_i, bins=8, color='crimson', alpha=0.7)
        ax1.set_title(f'Composite Score Distribution (i={i_val})')
        ax2.hist(displacements_for_i, bins=8, color='teal', alpha=0.7)
        ax2.set_title(f'Offsets |r| Distribution (i={i_val})')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir_experiment, f'distribution_i_{i_val}.png'), dpi=130)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Array Framework for SLURM Simulations")
    parser.add_argument('--i_values', type=int, nargs='+', default=None)
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()

    default_array = [5, 8, 11, 14, 17, 20, 23, 26]
    selected_i = args.i_values if args.i_values is not None else default_array
    
    run_bootstrap_simulation(selected_i, num_runs_per_i=args.iterations)
