# cvmi.py
import os
import numpy as np
import matplotlib
# Force headless matplotlib backend so it doesn't try to open windows under SLURM
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# These numbers are for XRT Spectrometer. Modify as needed.
RAW_CHANNELS_PER_BIN = 32
SPECTRUM_ROI_START = 13
def raw_energy_to_bin_idx(energy):
    """Converts mean energy to its corresponding bin index."""
    return int((energy / RAW_CHANNELS_PER_BIN) - SPECTRUM_ROI_START)

def compute_circular_wiggle_analysis(
    mask_array, 
    run_id, 
    output_dir_suffix, 
    images, 
    hits, 
    mean_energy, 
    is_gaussian, 
    total_hit_within_mask, 
    original_event_number,
    annulus_mask,
    cx=67, 
    cy=60, 
    max_plots=50
):
    """
    Executes the circular wiggle landscape score mapping pipeline.
    Saves diagnostic 4-panel figures to a local directory.
    """
    output_dir_wiggle = f'./circular_wiggler_{output_dir_suffix}_run{run_id}_gaussian_radial_mask_hitfinder'
    os.makedirs(output_dir_wiggle, exist_ok=True)
    
    ny, nx = images[0].shape
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]
    
    valid_indices = np.where(mask_array.astype(bool) & is_gaussian)[0]
    print(f"Found {len(valid_indices)} shots for classification: '{output_dir_suffix}'.")
    
    plotted_count = 0
    sigma_r = 1.0 
    
    for idx in valid_indices:
        if plotted_count >= max_plots:  
            break
            
        img = hits[idx].copy()
        img[~annulus_mask] = 0
        
        energy = mean_energy[idx] 
        single_spike = is_gaussian[idx]
        peak_bin = raw_energy_to_bin_idx(energy)
        
        if peak_bin <= 19:
            continue
            
        plotted_count += 1
        smoothed_img = gaussian_filter(img, sigma=1.0)
        
        if single_spike:
            re = (energy / RAW_CHANNELS_PER_BIN - SPECTRUM_ROI_START) * 0.6 + 29.4 
        else:
            re = peak_bin * 0.6 + 29.4
        
        max_allowed_r = 60.0 - re  
        wiggle_range = min(15.0, max(5.0, max_allowed_r + 2.0))
        wiggle_step = 0.4
        
        x_centers = np.arange(cx - wiggle_range, cx + wiggle_range + wiggle_step, wiggle_step)
        y_centers = np.arange(cy - wiggle_range, cy + wiggle_range + wiggle_step, wiggle_step)
        
        score_map_minus5 = np.zeros((len(y_centers), len(x_centers)))
        score_map_plus5 = np.zeros((len(y_centers), len(x_centers)))
        
        for row_idx, yc in enumerate(y_centers):
            for col_idx, xc in enumerate(x_centers):
                r_center = np.sqrt((xc - cx)**2 + (yc - cy)**2)
                
                if r_center + re >= 60.0:
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
        
        max_combined_val = np.nanmax(combined_score_map) if not np.all(np.isnan(combined_score_map)) else 0.0
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 5))
        extent_box = [x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]]
        
        im1 = ax1.imshow(img, cmap='viridis', origin='lower')
        fig.colorbar(im1, ax=ax1, label='Intensity')
        ax1.set_title(f'Hits: {total_hit_within_mask[idx]}')
        
        for rad in [20, 30, 40, 50, 60]:
            ax1.add_patch(plt.Circle((cx, cy), radius=rad, color='white', fill=False, linestyle='--', alpha=0.3))
        ax1.add_patch(plt.Circle((cx, cy), radius=re, color='red', fill=False, linestyle='-', alpha=0.6))
        
        im2 = ax2.imshow(score_map_minus5, cmap='inferno', origin='lower', extent=extent_box)
        fig.colorbar(im2, ax=ax2)
        
        im3 = ax3.imshow(score_map_plus5, cmap='inferno', origin='lower', extent=extent_box)
        fig.colorbar(im3, ax=ax3)
        
        im4 = ax4.imshow(combined_score_map, cmap='magma', origin='lower', extent=extent_box)
        fig.colorbar(im4, ax=ax4)
        
        try:
            display_idx = original_event_number[idx]
        except:
            display_idx = idx
            
        fig.suptitle(f'{output_dir_suffix.upper()} Event: {display_idx} | Bin: {peak_bin} | Max Combined: {max_combined_val:.3e}')
        plt.tight_layout()
        
        fig.savefig(os.path.join(output_dir_wiggle, f'shot_{display_idx}_four_panel_wiggle_map.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    print(f"Analysis complete for target layout: {output_dir_suffix}")
