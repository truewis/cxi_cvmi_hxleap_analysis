#!/usr/bin/env python3
"""
Regenerate every plot in every aggregate_4roi_* folder from the on-disk
`.npy` data, swapping the words 'early' <-> 'late' in the titles only.

Motivation: the 'early' preset was originally defined by low LXT step
indices, which correspond to negative LXT (laser arriving LATE relative
to the X-rays). Semantically that should be labelled 'late'. Folder
names stay the same (they are historical); only plot title text is
corrected.

Handles four plot families:
  * panels_by_energy_with_lobe_contour.png / panels_density_anomaly.png
    inside aggregated_4roi_*_by_energy(_shifted) folders — uses the
    stored per-bin arrays plus a recomputed un-streaked lobe map.
  * run_step_proportion_matrix.png anywhere it exists.
  * azimuthal_covariance_*.png inside the by-energy folders.
  * aggregated_ROI_panels.png / aggregated_total_hits.png inside the
    _steps / run145_*_steps folders.
  * compare_ROI_early_vs_late.png / compare_total_hits_early_vs_late.png
    inside the compare folder (reads its two sibling folders' arrays).

Requires conda env CXI.
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add batch_analysis/ to path so sibling imports work from plotting/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aggregate_4roi_by_energy import (
    CX, CY, RAW_CHANNELS_PER_BIN, SPECTRUM_ROI_START,
    DENSITY_BLUR_SIGMA, ANOMALY_VMIN, ANOMALY_VMAX,
    N_ENERGY_BINS, ROIS,
    _lobe_density_map,
)


BATCH_ANALYSIS_DIR = '/sdf/data/lcls/ds/cxi/cxi100895124/results/jinseop/batch_analysis_results/legacy_outputs'


def swap_early_late(text):
    """Swap the words 'early' and 'late' as whole words, preserving case."""
    def _sub(m):
        w = m.group(0)
        low = w.lower()
        replacement = 'late' if low == 'early' else 'early'
        if w.isupper():
            return replacement.upper()
        if w[0].isupper():
            return replacement.capitalize()
        return replacement
    return re.sub(r'\b(early|late)\b', _sub, text, flags=re.IGNORECASE)


def _parse_re_median(notes_path):
    if not os.path.exists(notes_path):
        return []
    txt = open(notes_path).read()
    m = re.search(r're_median\[px\]\s*=\s*(\[.*?\])', txt, re.DOTALL)
    if not m:
        return []
    return [float(x) for x in re.findall(r'-?\d+\.\d+|-?\d+', m.group(1))]


def regen_panels_by_energy(folder):
    meta = np.load(os.path.join(folder, 'aggregation_metadata.npy'),
                   allow_pickle=True).item()
    preset_name = meta['preset_name']
    kind = meta['kind']
    edges = meta['quantile_edges']
    cx = float(meta.get('cx', CX))
    cy = float(meta.get('cy', CY))
    n_shots = meta['n_shots_per_bin_per_roi']

    aggregates = {roi: [np.load(os.path.join(folder, f'aggregated_ROI_{roi}_ebin{b}.npy'))
                        for b in range(N_ENERGY_BINS)] for roi in ROIS}
    ny, nx = aggregates['R'][0].shape

    bin_re_median = _parse_re_median(os.path.join(folder, 'NOTES.md'))
    if len(bin_re_median) < N_ENERGY_BINS:
        bin_re_median = bin_re_median + [float('nan')] * (N_ENERGY_BINS - len(bin_re_median))

    rng = np.random.default_rng(0)
    lobe_maps = []
    for b in range(N_ENERGY_BINS):
        r = bin_re_median[b]
        if not np.isfinite(r):
            lobe_maps.append(np.zeros((ny, nx)))
        else:
            lobe_maps.append(_lobe_density_map(r, (ny, nx), cx, cy, rng))

    fig, axes = plt.subplots(len(ROIS), N_ENERGY_BINS,
                             figsize=(3.6 * N_ENERGY_BINS, 3.4 * len(ROIS)),
                             squeeze=False)
    for i, roi in enumerate(ROIS):
        for b in range(N_ENERGY_BINS):
            ax = axes[i, b]
            arr = aggregates[roi][b]
            vmax = float(np.percentile(arr, 99.5)) if arr.max() > 0 else 1.0
            ax.imshow(arr, cmap='viridis', origin='lower', vmin=0.0, vmax=vmax)
            lobe = lobe_maps[b]
            if lobe.max() > 0:
                ax.contour(lobe, levels=[0.5 * lobe.max()],
                           colors='white', linewidths=1.2, alpha=0.85)
            ax.axvline(cx, color='cyan', ls='--', alpha=0.4, lw=0.6)
            ax.axhline(cy, color='cyan', ls='--', alpha=0.4, lw=0.6)
            if i == 0:
                lo, hi = edges[b], edges[b + 1]
                ax.set_title(f'e in [{lo:.0f}, {hi:.0f}]  '
                             f'(re≈{bin_re_median[b]:.1f} px)\n'
                             f'{n_shots[roi][b]} shots', fontsize=10)
            else:
                ax.set_title(f'{n_shots[roi][b]} shots', fontsize=10)
            if b == 0:
                ax.set_ylabel(f'ROI_{roi}', fontsize=11, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

    display_preset = swap_early_late(preset_name)
    fig.suptitle(f'Aggregated ROIs sliced by peak-energy quantile bin  '
                 f'({display_preset}, kind={kind}, center=({cx:g}, {cy:g}))\n'
                 f'White contour = unstreaked lobe half-max (from 3D shell projection)',
                 fontweight='bold')
    fig.tight_layout()
    out = os.path.join(folder, 'panels_by_energy_with_lobe_contour.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'  wrote {out}')


def regen_panels_density_anomaly(folder):
    meta = np.load(os.path.join(folder, 'aggregation_metadata.npy'),
                   allow_pickle=True).item()
    preset_name = meta['preset_name']
    kind = meta['kind']
    edges = meta['quantile_edges']
    cx = float(meta.get('cx', CX))
    cy = float(meta.get('cy', CY))
    n_shots = meta['n_shots_per_bin_per_roi']

    alpha_path = os.path.join(folder, 'lobe_fraction_alpha.npy')
    if not os.path.exists(alpha_path):
        print(f'  [skip anomaly] no lobe_fraction_alpha.npy in {folder}')
        return
    alpha_grid = np.load(alpha_path)

    anomaly = {roi: [np.load(os.path.join(folder, f'anomaly_ROI_{roi}_ebin{b}.npy'))
                     for b in range(N_ENERGY_BINS)] for roi in ROIS}
    ny, nx = anomaly['R'][0].shape

    bin_re_median = _parse_re_median(os.path.join(folder, 'NOTES.md'))
    if len(bin_re_median) < N_ENERGY_BINS:
        bin_re_median = bin_re_median + [float('nan')] * (N_ENERGY_BINS - len(bin_re_median))

    rng = np.random.default_rng(0)
    lobe_maps = []
    for b in range(N_ENERGY_BINS):
        r = bin_re_median[b]
        if not np.isfinite(r):
            lobe_maps.append(np.zeros((ny, nx)))
        else:
            lobe_maps.append(_lobe_density_map(r, (ny, nx), cx, cy, rng))

    fig, axes = plt.subplots(len(ROIS), N_ENERGY_BINS,
                             figsize=(3.6 * N_ENERGY_BINS, 3.4 * len(ROIS)),
                             squeeze=False)
    for i, roi in enumerate(ROIS):
        for b in range(N_ENERGY_BINS):
            ax = axes[i, b]
            im = ax.imshow(anomaly[roi][b], cmap='RdBu_r', origin='lower',
                           vmin=ANOMALY_VMIN, vmax=ANOMALY_VMAX)
            lobe = lobe_maps[b]
            if lobe.max() > 0:
                ax.contour(lobe, levels=[0.5 * lobe.max()],
                           colors='black', linewidths=0.8, alpha=0.6)
            ax.axvline(cx, color='k', ls='--', alpha=0.25, lw=0.5)
            ax.axhline(cy, color='k', ls='--', alpha=0.25, lw=0.5)
            alpha = alpha_grid[i, b]
            if i == 0:
                lo, hi = edges[b], edges[b + 1]
                ax.set_title(f'e in [{lo:.0f}, {hi:.0f}]  '
                             f'(re≈{bin_re_median[b]:.1f} px)\n'
                             f'{n_shots[roi][b]} shots  |  α={alpha:.3f}',
                             fontsize=10)
            else:
                ax.set_title(f'{n_shots[roi][b]} shots  |  α={alpha:.3f}',
                             fontsize=10)
            if b == 0:
                ax.set_ylabel(f'ROI_{roi}', fontsize=11, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    display_preset = swap_early_late(preset_name)
    fig.suptitle(
        f'Density anomaly with fitted mixture baseline  '
        f'({display_preset}, kind={kind}, center=({cx:g}, {cy:g}))\n'
        f'expected = α·(un-streaked lobe) + (1−α)·(empirical background), '
        f'α fit per panel to zero the residual inside the lobe half-max\n'
        f'σ={DENSITY_BLUR_SIGMA} px  |  fixed colour scale [{ANOMALY_VMIN:g}, {ANOMALY_VMAX:g}]  |  '
        'black contour = un-streaked lobe half-max',
        fontweight='bold',
    )
    fig.tight_layout()
    out = os.path.join(folder, 'panels_density_anomaly.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'  wrote {out}')


def regen_proportion_matrix(folder):
    npy = os.path.join(folder, 'run_step_proportion_matrix.npy')
    if not os.path.exists(npy):
        return
    data = np.load(npy, allow_pickle=True).item()
    matrix = data['matrix']
    rois = data['rois']
    pairs = data['pairs']
    aggregate_name = swap_early_late(os.path.basename(folder))

    fig_w = max(6.0, 0.55 * len(pairs) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, 3.2))
    im = ax.imshow(matrix, cmap='viridis', aspect='auto',
                   vmin=0.0, vmax=max(np.nanmax(matrix), 1e-6))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v):
                txt_color = 'white' if v < 0.5 * np.nanmax(matrix) else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        color=txt_color, fontsize=7)
    ax.set_xticks(np.arange(len(pairs)))
    ax.set_xticklabels([f'{p[0]}:{p[1]}' for p in pairs], rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(rois)))
    ax.set_yticklabels([f'ROI_{r}' for r in rois])
    ax.set_xlabel('(run, step)')
    ax.set_title(f'{aggregate_name}\nrow-normalized ROI composition by (run, step)',
                 fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label('fraction of ROI pixel-sum', fontsize=8)
    fig.tight_layout()
    out = os.path.join(folder, 'run_step_proportion_matrix.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'  wrote {out}')


def regen_azimuthal_covariance(folder):
    for fname in sorted(os.listdir(folder)):
        if not (fname.startswith('azimuthal_covariance') and fname.endswith('.npy')):
            continue
        npy_path = os.path.join(folder, fname)
        data = np.load(npy_path, allow_pickle=True).item()
        preset_name = data.get('preset_name', '?')
        kind = data.get('kind', '?')
        dr_half = data.get('dr_half', 3.0)
        cx = data.get('cx', CX)
        cy = data.get('cy', CY)
        n_bins = data.get('n_bins', 90)
        is_cross = 'cross_cov' in data or 'Hx' in data

        if is_cross:
            cov = data['cross_cov']
            corr = data['cross_corr']
            Hx = data['Hx']
            n_shots = Hx.shape[0]
            r_off_x = data.get('r_offset_x', 0.0)
            r_off_y = data.get('r_offset_y', 0.0)
            # No diagonal masking for cross — full matrix is meaningful.
            cov_plot = cov
            corr_plot = corr
            cov_title = f'cross-covariance  |  {n_shots} shots'
            corr_title = 'cross-correlation (Pearson)'
            xlabel = r'$\theta_j$ [deg]  (X-annulus)'
            ylabel = r'$\theta_i$ [deg]  (Y-annulus)'
        else:
            cov = data['cov']
            corr = data['corr']
            H = data['H']
            n_shots = H.shape[0]
            r_off = data.get('r_offset', 0.0)
            diag_mask = np.eye(n_bins, dtype=bool)
            cov_plot = np.where(diag_mask, np.nan, cov)
            corr_plot = np.where(diag_mask, np.nan, corr)
            cov_title = f'covariance (diagonal masked)  |  {n_shots} shots'
            corr_title = 'correlation (diagonal masked)'
            xlabel = r'$\theta_j$ [deg]'
            ylabel = r'$\theta_i$ [deg]'

        fig, (ax_cov, ax_corr) = plt.subplots(1, 2, figsize=(15, 6.5))
        extent = [0.0, 360.0, 0.0, 360.0]
        off_max = float(np.nanmax(np.abs(cov_plot)))
        if not np.isfinite(off_max) or off_max == 0:
            off_max = 1e-12
        im_c = ax_cov.imshow(cov_plot, cmap='RdBu_r', origin='lower', extent=extent,
                             vmin=-off_max, vmax=off_max, aspect='equal')
        ax_cov.set_xlabel(xlabel)
        ax_cov.set_ylabel(ylabel)
        ax_cov.set_title(cov_title)
        plt.colorbar(im_c, ax=ax_cov, fraction=0.046, pad=0.03)

        corr_max = float(np.nanmax(np.abs(corr_plot)))
        if not np.isfinite(corr_max) or corr_max == 0:
            corr_max = 1e-12
        im_r = ax_corr.imshow(corr_plot, cmap='RdBu_r', origin='lower', extent=extent,
                              vmin=-corr_max, vmax=corr_max, aspect='equal')
        ax_corr.set_xlabel(xlabel)
        ax_corr.set_ylabel(ylabel)
        ax_corr.set_title(corr_title)
        plt.colorbar(im_r, ax=ax_corr, fraction=0.046, pad=0.03)

        display_preset = swap_early_late(preset_name)
        if is_cross:
            title = (f'Hyperspectral azimuthal cross-covariance  |  {display_preset}  '
                     f'(kind={kind})\nX annulus = re{r_off_x:+g} ± {dr_half:g} px, '
                     f'Y annulus = re{r_off_y:+g} ± {dr_half:g} px, '
                     f'{n_bins} angular bins (4° each), center=({cx}, {cy})')
        else:
            title = (f'Hyperspectral azimuthal covariance  |  {display_preset}  '
                     f'(kind={kind})\nannulus = re{r_off:+g} ± {dr_half:g} px, '
                     f'{n_bins} angular bins (4° each), center=({cx}, {cy})')
        fig.suptitle(title, fontweight='bold')
        fig.tight_layout()
        out = os.path.join(folder, fname.replace('.npy', '.png'))
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f'  wrote {out}')


def regen_steps_plots(folder):
    label = swap_early_late(os.path.basename(folder))
    rois = ('R', 'L', 'U', 'D')
    arrs = {}
    for roi in rois:
        p = os.path.join(folder, f'aggregated_ROI_{roi}.npy')
        if os.path.exists(p):
            arrs[roi] = np.load(p).astype(float)
    if arrs:
        n_valid = len(arrs)
        fig, axes = plt.subplots(1, n_valid, figsize=(4.2 * n_valid, 4.6), squeeze=False)
        for j, roi in enumerate(arrs):
            ax = axes[0, j]
            im = ax.imshow(arrs[roi], cmap='viridis', origin='lower')
            ax.axvline(CX, color='white', ls='--', alpha=0.5, lw=0.7)
            ax.axhline(CY, color='white', ls='--', alpha=0.5, lw=0.7)
            ax.set_title(f'ROI_{roi}', fontsize=10)
            ax.set_xlabel('pixel x'); ax.set_ylabel('pixel y')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label='accumulated hits')
        fig.suptitle(f'Aggregated ROI accumulations  |  label = {label}',
                     fontweight='bold')
        fig.tight_layout()
        out = os.path.join(folder, 'aggregated_ROI_panels.png')
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f'  wrote {out}')

    p_total = os.path.join(folder, 'aggregated_total_hits.npy')
    if os.path.exists(p_total):
        arr = np.load(p_total).astype(float)
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        im = ax.imshow(arr, cmap='viridis', origin='lower')
        ax.axvline(CX, color='white', ls='--', alpha=0.5, lw=0.7)
        ax.axhline(CY, color='white', ls='--', alpha=0.5, lw=0.7)
        ax.set_title(f'aggregated total hits  |  label = {label}')
        ax.set_xlabel('pixel x'); ax.set_ylabel('pixel y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label='accumulated hits')
        fig.tight_layout()
        out = os.path.join(folder, 'aggregated_total_hits.png')
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f'  wrote {out}')


def regen_compare(folder):
    """Special case: the compare folder pulls arrays from two sibling folders
    and displays them side by side. Swap 'early' ↔ 'late' in the row/panel
    labels."""
    parent = os.path.dirname(folder)
    early_dir = os.path.join(parent, 'aggregated_4roi_goose_run145_early_steps')
    late_dir  = os.path.join(parent, 'aggregated_4roi_goose_run145_late_steps')
    if not (os.path.isdir(early_dir) and os.path.isdir(late_dir)):
        return
    rois = ('R', 'L', 'U', 'D')

    # Row labels: what the sibling folders BECOME after swap.
    # Sibling 'early_steps' folder contains steps 0-5 → after swap this is 'late'.
    label_early_dir = swap_early_late('early (step 0-5)')   # -> 'late (step 0-5)'
    label_late_dir  = swap_early_late('late (step 14-19)')  # -> 'early (step 14-19)'

    fig, axes = plt.subplots(2, len(rois), figsize=(4.2 * len(rois), 8.2))
    for j, roi in enumerate(rois):
        for i, (label, src) in enumerate(((label_early_dir, early_dir),
                                          (label_late_dir,  late_dir))):
            p = os.path.join(src, f'aggregated_ROI_{roi}.npy')
            if not os.path.exists(p):
                axes[i, j].axis('off')
                continue
            arr = np.load(p).astype(float)
            ax = axes[i, j]
            vmax = float(np.percentile(arr, 99.5)) if arr.max() > 0 else 1.0
            im = ax.imshow(arr, cmap='viridis', origin='lower', vmin=0, vmax=vmax)
            ax.axvline(CX, color='white', ls='--', alpha=0.5, lw=0.7)
            ax.axhline(CY, color='white', ls='--', alpha=0.5, lw=0.7)
            ax.set_title(f'ROI_{roi} | {label}\nsum={arr.sum():.0f}', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    fig.suptitle(swap_early_late('Run 145 goose — early (step 0-5) vs late (step 14-19)'),
                 fontweight='bold')
    fig.tight_layout()
    out = os.path.join(folder, 'compare_ROI_early_vs_late.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'  wrote {out}')

    # Total-hits comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for i, (label, src) in enumerate(((label_early_dir, early_dir),
                                      (label_late_dir,  late_dir))):
        p = os.path.join(src, 'aggregated_total_hits.npy')
        if not os.path.exists(p):
            axes[i].axis('off')
            continue
        arr = np.load(p).astype(float)
        ax = axes[i]
        vmax = float(np.percentile(arr, 99.5)) if arr.max() > 0 else 1.0
        im = ax.imshow(arr, cmap='viridis', origin='lower', vmin=0, vmax=vmax)
        ax.axvline(CX, color='white', ls='--', alpha=0.5, lw=0.7)
        ax.axhline(CY, color='white', ls='--', alpha=0.5, lw=0.7)
        ax.set_title(f'Total hits | {label} | sum={arr.sum():.0f}', fontsize=11)
        ax.set_xlabel('pixel x'); ax.set_ylabel('pixel y')
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle(swap_early_late('Run 145 goose — total hits early vs late'),
                 fontweight='bold')
    fig.tight_layout()
    out = os.path.join(folder, 'compare_total_hits_early_vs_late.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'  wrote {out}')


def main():
    folders = sorted(
        os.path.join(BATCH_ANALYSIS_DIR, d)
        for d in os.listdir(BATCH_ANALYSIS_DIR)
        if d.startswith('aggregated_4roi_')
    )
    for f in folders:
        name = os.path.basename(f)
        print(f'=== {name} ===')
        has_meta = os.path.exists(os.path.join(f, 'aggregation_metadata.npy'))
        has_roi = os.path.exists(os.path.join(f, 'aggregated_ROI_R.npy'))
        has_prop = os.path.exists(os.path.join(f, 'run_step_proportion_matrix.npy'))
        has_azim = any(fn.startswith('azimuthal_covariance') and fn.endswith('.npy')
                       for fn in os.listdir(f))
        is_compare = 'compare_' in name

        if has_meta:
            regen_panels_by_energy(f)
            regen_panels_density_anomaly(f)
        elif has_roi and not is_compare:
            regen_steps_plots(f)

        if is_compare:
            regen_compare(f)

        if has_prop:
            regen_proportion_matrix(f)

        if has_azim:
            regen_azimuthal_covariance(f)


if __name__ == '__main__':
    main()
