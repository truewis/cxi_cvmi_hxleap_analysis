# analysis_library/cvmi.py
import os
import numpy as np
import matplotlib
# Enforce a non-interactive backend safe for headless SLURM nodes
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter  
from tqdm import tqdm

# --- Global Spectrum Binning Configuration ---
RAW_CHANNELS_PER_BIN = 32  
TOTAL_COARSE_BINS = 64
SPECTRUM_ROI_START = 13  
SPECTRUM_ROI_END = 45  

def raw_energy_to_bin_idx(mean_energy):
    """Converts raw mean energy values to ROI spectrum bin indices."""
    coarse_bin = np.floor(mean_energy / RAW_CHANNELS_PER_BIN).astype(int)
    final_bin_idx = coarse_bin - SPECTRUM_ROI_START
    is_out_of_roi = (coarse_bin < SPECTRUM_ROI_START) | (coarse_bin >= SPECTRUM_ROI_END)
    if np.isscalar(final_bin_idx):
        if is_out_of_roi: return -1
    else:
        final_bin_idx[is_out_of_roi] = -1
    return final_bin_idx

def bootstrapped_final_score_significance(n_electron, score):
    """Calculates statistical significance Z-score against bootstrapping background baseline."""
    return (score - n_electron * 4e-8 + 7e-9) / (2.0e-8 * n_electron + 2e-7) # ( x - mu ) / sigma

def compute_circular_wiggle_analysis(
    mask_array, 
    run_id, 
    target_step=0,
    images=None, 
    hits=None, 
    mean_energy=None, 
    is_gaussian=None, 
    total_hit_within_mask=None, 
    original_event_number=None,
    annulus_mask=None,
    cx=67, 
    cy=60,
    output_prefix="duck",
    output_dir_suffix=None,
    max_plots=50,
    r_adjustment = 0
):
    """
    Executes circular wiggle matrix search. Fully cross-compatible with 
    both high-throughput experimental batches and single-frame bootstrap loops.
    """
    # Dynamically resolve output directory naming
    if output_dir_suffix is not None:
        output_dir_metrics = f'./circular_wiggler_{output_dir_suffix}_batch_metrics'
    else:
        output_dir_metrics = f'./circular_wiggler_{output_prefix}_run{run_id}_step{target_step}_batch_metrics'
        
    os.makedirs(output_dir_metrics, exist_ok=True)
    
    # Precompute fixed detector grid coordinates
    ny, nx = images[0].shape
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]
    
    valid_indices = np.where(mask_array)[0]
    
    if len(valid_indices) == 0:
        return np.nan, np.nan

    # --- Precompute Global Maximum Grid Extents to Prevent Dimension Mismatches ---
    max_possible_wiggle = 0.0
    for idx in valid_indices:
        peak_bin = raw_energy_to_bin_idx(mean_energy[idx])
        if peak_bin <= 19:
            continue
        energy = mean_energy[idx]
        re = (energy / RAW_CHANNELS_PER_BIN - SPECTRUM_ROI_START) * 0.6 + 29.4 + r_adjustment
        wiggle_range = min(52.0 - re, re - 28.0)
        if wiggle_range > max_possible_wiggle:
            max_possible_wiggle = wiggle_range

    wiggle_step = 0.4
    max_steps = int(np.ceil(max_possible_wiggle / wiggle_step))
    master_offsets = np.arange(-max_steps, max_steps + 1) * wiggle_step
    
    master_x_centers = cx + master_offsets
    master_y_centers = cy + master_offsets
    
    # Initialize the absolute accumulation matrices to zero
    accumulated_grid_scores = np.zeros((len(master_y_centers), len(master_x_centers)))
    sigma_r = 1.0  
    
    # Metric Storage Populations
    collected_scores = []
    collected_hit_counts = []
    collected_peak_x = []
    collected_peak_y = []
    collected_dr = []
    collected_significance = []
    
    plotted_count = 0
    
    # --- Load and Process Batch ---
    for idx in valid_indices:
        img = hits[idx].copy()
        if annulus_mask is not None:
            img[~annulus_mask] = 0
        
        energy = mean_energy[idx] 
        single_spike = is_gaussian[idx]
        peak_bin = raw_energy_to_bin_idx(mean_energy[idx])
        
        if peak_bin <= 19:
            continue
            
        smoothed_img = gaussian_filter(img, sigma=1.0)
        re = (energy / RAW_CHANNELS_PER_BIN - SPECTRUM_ROI_START) * 0.6 + 29.4 + r_adjustment
        wiggle_range = min(52.0 - re, re - 28.0)
        
        valid_x_mask = np.abs(master_x_centers - cx) <= (wiggle_range + 1e-6)
        valid_y_mask = np.abs(master_y_centers - cy) <= (wiggle_range + 1e-6)
        
        x_centers = master_x_centers[valid_x_mask]
        y_centers = master_y_centers[valid_y_mask]
        
        x_indices = np.where(valid_x_mask)[0]
        y_indices = np.where(valid_y_mask)[0]
        x_start_idx, x_end_idx = x_indices[0], x_indices[-1] + 1
        y_start_idx, y_end_idx = y_indices[0], y_indices[-1] + 1
        
        score_map_minus5 = np.zeros((len(y_centers), len(x_centers)))
        score_map_plus5 = np.zeros((len(y_centers), len(x_centers)))
        
        for row_idx, yc in enumerate(y_centers):
            for col_idx, xc in enumerate(x_centers):
                r_center = np.sqrt((xc - cx)**2 + (yc - cy)**2)
                if r_center + re >= 52.0 or r_center - re >= 28.0:
                    score_map_minus5[row_idx, col_idx] = np.nan
                    score_map_plus5[row_idx, col_idx] = np.nan
                    continue
                    
                r_map_try = np.sqrt((x_grid - xc)**2 + (y_grid - yc)**2)
                t_map_try = np.arctan2(y_grid - yc, x_grid - xc)
                cos2_map_try = np.sin(t_map_try) ** 2 - 0.5
                
                weight_re = np.exp(-((r_map_try - re) ** 2) / (2 * sigma_r ** 2))
                weight_bg_minus = np.exp(-((r_map_try - (re - 5.0)) ** 2) / (2 * sigma_r ** 2))
                weight_bg_plus = np.exp(-((r_map_try - (re + 5.0)) ** 2) / (2 * sigma_r ** 2))
                
                weight_re[weight_re < 1e-4] = 0
                weight_bg_minus[weight_bg_minus < 1e-4] = 0
                weight_bg_plus[weight_bg_plus < 1e-4] = 0
                
                if np.sum(weight_re) > 0:
                    w_re_norm = cos2_map_try - np.average(cos2_map_try, weights=weight_re)
                    score_re = np.average(smoothed_img * w_re_norm, weights=weight_re)
                else:
                    score_re = 0.0
                    
                if np.sum(weight_bg_minus) > 0:
                    w_bm_norm = cos2_map_try - np.average(cos2_map_try, weights=weight_bg_minus)
                    score_bg_minus = np.average(smoothed_img * w_bm_norm, weights=weight_bg_minus)
                else:
                    score_bg_minus = 0.0
                    
                if np.sum(weight_bg_plus) > 0:
                    w_bp_norm = cos2_map_try - np.average(cos2_map_try, weights=weight_bg_plus)
                    score_bg_plus = np.average(smoothed_img * w_bp_norm, weights=weight_bg_plus)
                else:
                    score_bg_plus = 0.0
                
                score_map_minus5[row_idx, col_idx] = score_re - score_bg_minus
                score_map_plus5[row_idx, col_idx] = score_re - score_bg_plus
    
        floored_minus5 = np.clip(score_map_minus5, 0, None)
        floored_plus5 = np.clip(score_map_plus5, 0, None)
        combined_score_map = floored_minus5 * floored_plus5
        
        master_slice = accumulated_grid_scores[y_start_idx:y_end_idx, x_start_idx:x_end_idx]
        valid_mask = ~np.isnan(combined_score_map)
        master_slice[valid_mask] += combined_score_map[valid_mask]
        
        if not np.all(np.isnan(combined_score_map)):
            max_combined_val = np.nanmax(combined_score_map)
            max_y_idx, max_x_idx = np.unravel_index(np.nanargmax(combined_score_map), combined_score_map.shape)
            actual_max_x = x_centers[max_x_idx]
            actual_max_y = y_centers[max_y_idx]
            dr = np.sqrt((actual_max_x - cx)**2 + (actual_max_y - cy)**2)
            significance_sigma_value = bootstrapped_final_score_significance(n_electron=np.sum(img), score=max_combined_val)
            
            # Draw individual panel plots only if significant and under max plot limit
            if significance_sigma_value > 3 and plotted_count < max_plots:
                plotted_count += 1
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 5))
                extent_box = [x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]]
                
                im1 = ax1.imshow(img, cmap='viridis', origin='lower')
                fig.colorbar(im1, ax=ax1, label='Intensity')
                ax1.set_title(f'Hitfinder Output | Hits: {total_hit_within_mask[idx]:.0f}')
                ax1.set_xlabel('Pixel X')
                ax1.set_ylabel('Pixel Y')
                
                for rad in [20, 60]:
                    ax1.add_patch(plt.Circle((cx, cy), radius=rad, color='white', fill=False, linestyle='--', alpha=0.3, linewidth=1.0))
                ax1.add_patch(plt.Circle((actual_max_x, actual_max_y), radius=re-5.0, color='orange', fill=False, linestyle=':', alpha=0.5, linewidth=1.2))
                ax1.add_patch(plt.Circle((actual_max_x, actual_max_y), radius=re+5.0, color='magenta', fill=False, linestyle='-.', alpha=0.5, linewidth=1.2))
                
                im2 = ax2.imshow(score_map_minus5, cmap='inferno', origin='lower', extent=extent_box)
                fig.colorbar(im2, ax=ax2, label='Δ Score ($r_e$ - $[r_e-5]$)')
                ax2.set_title('Scan Landscape ($r_e - 5$)')
                ax2.set_xlabel('Test Center X')
                ax2.axvline(cx, color='cyan', linestyle='--', alpha=0.5)
                ax2.axhline(cy, color='cyan', linestyle='--', alpha=0.5)
                if not np.all(np.isnan(score_map_minus5)):
                    m_y2, m_x2 = np.unravel_index(np.nanargmax(score_map_minus5), score_map_minus5.shape)
                    ax2.plot(x_centers[m_x2], y_centers[m_y2], 'go', markersize=5)
                
                im3 = ax3.imshow(score_map_plus5, cmap='inferno', origin='lower', extent=extent_box)
                fig.colorbar(im3, ax=ax3, label='Δ Score ($r_e$ - $[r_e+5]$)')
                ax3.set_title('Scan Landscape ($r_e + 5$)')
                ax3.set_xlabel('Test Center X')
                ax3.axvline(cx, color='cyan', linestyle='--', alpha=0.5)
                ax3.axhline(cy, color='cyan', linestyle='--', alpha=0.5)
                if not np.all(np.isnan(score_map_plus5)):
                    m_y3, m_x3 = np.unravel_index(np.nanargmax(score_map_plus5), score_map_plus5.shape)
                    ax3.plot(x_centers[m_x3], y_centers[m_y3], 'go', markersize=5)
            
                im4 = ax4.imshow(combined_score_map, cmap='magma', origin='lower', extent=extent_box)
                fig.colorbar(im4, ax=ax4, label='Product Score ($\Delta_{-5} \cdot \Delta_{+5}$)')
                ax4.set_title('Combined Product Map')
                ax4.set_xlabel('Test Center X')
                ax4.axvline(cx, color='cyan', linestyle='--', alpha=0.5)
                ax4.axhline(cy, color='cyan', linestyle='--', alpha=0.5)
                if not np.all(np.isnan(combined_score_map)) and max_combined_val > 0:
                    max_y_idx, max_x_idx = np.unravel_index(np.nanargmax(combined_score_map), combined_score_map.shape)
                    ax4.plot(x_centers[max_x_idx], y_centers[max_y_idx], 'go', markersize=6, label=f'Peak ({x_centers[max_x_idx]:.1f}, {y_centers[max_y_idx]:.1f})')
                    ax4.legend(loc='lower left')
                
                try:
                    display_idx = original_event_number[idx]
                except:
                    display_idx = idx
                    
                fig.suptitle(
                    f'{output_prefix.upper()} Circular Sweep - Run: {run_id}, Event: {display_idx} | Bin: {peak_bin} | Calculated $r_e$: {re:.2f} | Max Combined Value: {max_combined_val:.3e} ({significance_sigma_value:.1f}$\sigma$) | (XLEAP: {single_spike})\n',
                    fontsize=12, y=1.02
                )
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir_metrics, f'four_panel_sweep_{display_idx}.png'), dpi=150, bbox_inches='tight')
                plt.close(fig)
        else:
            max_combined_val = 0.0
            actual_max_x, actual_max_y = cx, cy
            dr = np.nan
            significance_sigma_value = 0.0
            
        collected_scores.append(max_combined_val)
        collected_hit_counts.append(total_hit_within_mask[idx])
        collected_peak_x.append(actual_max_x)
        collected_peak_y.append(actual_max_y)
        collected_dr.append(dr)
        collected_significance.append(significance_sigma_value)
    
    collected_scores = np.array(collected_scores)
    collected_hit_counts = np.array(collected_hit_counts)
    collected_peak_x = np.array(collected_peak_x)
    collected_peak_y = np.array(collected_peak_y)
    collected_dr = np.array(collected_dr)
    collected_significance = np.array(collected_significance)
    total_samples = len(collected_scores)
    
    # --- BATCH SUMMARY PLOTS (Skipped during rapid single-shot simulation loops) ---
    if total_samples > 1:
        # Plot 1: Peak Score vs Number of Hits Scatter Plot
        np.save(os.path.join(output_dir_metrics, f'data_score_vs_hits_run_{run_id}.npy'), 
                {'hits': collected_hit_counts, 'scores': collected_scores, 'significance': collected_significance})
        
        fig1, ax_scat = plt.subplots(figsize=(8, 6))
        ax_scat.scatter(collected_hit_counts, collected_scores, color='purple', alpha=0.6, edgecolors='black', linewidths=0.5, label='Processed Shots')
        sorted_hit_axis = np.sort(collected_hit_counts)
        three_sigma_score_cutoff = 3.0 * (1.2e-8 * sorted_hit_axis + 4e-7) + (sorted_hit_axis * 4e-8)
        ax_scat.plot(sorted_hit_axis, three_sigma_score_cutoff, color='red', linestyle='--', linewidth=2.0, label=r'$3\sigma$ Significance Threshold')
        ax_scat.set_title(f'Run {run_id} | Composite Peak Score vs. Hit Count Density\n(Total Samples N = {total_samples})', fontsize=12, fontweight='bold')
        ax_scat.set_xlabel('Total Mask Hits ($N$)')
        ax_scat.set_ylabel('Max Multiplied Combined Value ($\Delta_{-5} \cdot \Delta_{+5}$)')
        ax_scat.set_yscale('log')
        ax_scat.grid(True, which="both", linestyle='--', alpha=0.5)
        ax_scat.legend(loc='upper left')
        plt.tight_layout()
        fig1.savefig(os.path.join(output_dir_metrics, f'score_vs_hits_run_{run_id}.png'), dpi=150)
        plt.close(fig1)
        
        # Plot 2: Peak Center Positioning 2D Histograms
        np.save(os.path.join(output_dir_metrics, f'data_peak_positions_run_{run_id}.npy'), 
                {'x': collected_peak_x, 'y': collected_peak_y, 'significance': collected_significance})
        
        sig_mask = collected_significance > 3.0
        sig_peak_x = collected_peak_x[sig_mask]
        sig_peak_y = collected_peak_y[sig_mask]
        num_sig_samples = len(sig_peak_x)
        
        fig2, (ax_all, ax_sig) = plt.subplots(1, 2, figsize=(15, 6.5))
        x_bins = np.linspace(cx - 15, cx + 15, 20)
        y_bins = np.linspace(cy - 15, cy + 15, 20)
        
        im_all = ax_all.hist2d(collected_peak_x, collected_peak_y, bins=[x_bins, y_bins], cmap='hot')
        fig2.colorbar(im_all[3], ax=ax_all, label='Event Occurrence Counts')
        ax_all.axvline(cx, color='cyan', linestyle='--', alpha=0.7, label='Reference Center')
        ax_all.axhline(cy, color='cyan', linestyle='--', alpha=0.7)
        ax_all.plot(np.mean(collected_peak_x), np.mean(collected_peak_y), 'bx', markersize=10, markeredgewidth=2, label=f'Mean ({np.mean(collected_peak_x):.2f}, {np.mean(collected_peak_y):.2f})')
        ax_all.set_title(f'All Passing Landscapes\n(N = {total_samples})', fontsize=11, fontweight='bold')
        ax_all.set_xlabel('Optimized Peak Center X')
        ax_all.set_ylabel('Optimized Peak Center Y')
        ax_all.set_aspect('equal')
        ax_all.legend(loc='lower left')
        
        if num_sig_samples > 0:
            im_sig = ax_sig.hist2d(sig_peak_x, sig_peak_y, bins=[x_bins, y_bins], cmap='hot')
            fig2.colorbar(im_sig[3], ax=ax_sig, label='Event Occurrence Counts')
            ax_sig.plot(np.mean(sig_peak_x), np.mean(sig_peak_y), 'bx', markersize=10, markeredgewidth=2, label=f'Mean ({np.mean(sig_peak_x):.2f}, {np.mean(sig_peak_y):.2f})')
        else:
            ax_sig.text(0.5, 0.5, 'No Spots Exceeding $3\sigma$', horizontalalignment='center', verticalalignment='center', transform=ax_sig.transAxes, color='red')
        ax_sig.axvline(cx, color='cyan', linestyle='--', alpha=0.7)
        ax_sig.axhline(cy, color='cyan', linestyle='--', alpha=0.7)
        ax_sig.set_title(f'Significant Landscapes Only ($>3\sigma$)\n(N = {num_sig_samples})', fontsize=11, fontweight='bold')
        ax_sig.set_xlabel('Optimized Peak Center X')
        ax_sig.set_aspect('equal')
        if num_sig_samples > 0: ax_sig.legend(loc='lower left')
        
        fig2.suptitle(f'Run {run_id} | Wiggle Track Landscape Peak Spatial Density Profiles', fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig2.savefig(os.path.join(output_dir_metrics, f'peak_positions_2d_hist_run_{run_id}.png'), dpi=150)
        plt.close(fig2)
        
        # Plot 3: Summed Wiggle Grid Landscape Score Mapping
        np.save(os.path.join(output_dir_metrics, f'data_average_grid_scores_run_{run_id}.npy'), accumulated_grid_scores)
        fig3, ax_avg = plt.subplots(figsize=(8, 6.5))
        extent_avg = [master_x_centers[0], master_x_centers[-1], master_y_centers[0], master_y_centers[-1]]
        im_avg = ax_avg.imshow(accumulated_grid_scores, cmap='jet', origin='lower', extent=extent_avg)
        fig3.colorbar(im_avg, ax=ax_avg, label='Summed Combined Product Score Matrix')
        ax_avg.axvline(cx, color='white', linestyle='--', alpha=0.6, label='Reference Center')
        ax_avg.axhline(cy, color='white', linestyle='--', alpha=0.6)
        ax_avg.set_title(f'Run {run_id} | Total Summed Wiggle-Grid Landscape\n(Accumulated Shots N = {total_samples})', fontsize=11, fontweight='bold')
        ax_avg.set_xlabel('Wiggle Grid Center X')
        ax_avg.set_ylabel('Wiggle Grid Center Y')
        ax_avg.legend(loc='lower left')
        plt.tight_layout()
        fig3.savefig(os.path.join(output_dir_metrics, f'average_wiggle_grid_landscape_run_{run_id}.png'), dpi=150)
        plt.close(fig3)

    # Return unpackable scalars if evaluating single item, or arrays if evaluating batch
    if total_samples == 1:
        return collected_scores[0], collected_dr[0]
    else:
        return collected_scores, collected_dr
