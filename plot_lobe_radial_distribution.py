#!/usr/bin/env python3
"""
Compare the current vs. proposed lobe-electron radial distribution used in
run_bootstrap_with_slurm_lobe.py.

Current  : r_2d ~ Normal(re, 2.0);  theta_2d ~ sin^2(theta_2d)  (independent 2D)
Proposed : uniform 3D shell of radius re, thickness dr=5, projected to the xy
           plane. i.e. sample r_3d ~ U(re - dr/2, re + dr/2), pick an isotropic
           direction on the unit sphere, take (x, y) = r_3d * (n_x, n_y).
           The 2D angular pattern (sin^2 lobes along y) is kept as-is by
           accepting/rejecting on the projected azimuth phi_2d = atan2(y, x).

The marginal Abel-projected radial law for a thin shell of radius R is
    P(r_2d) = r_2d / (R * sqrt(R^2 - r_2d^2))          (r_2d in [0, R])
which piles up at r_2d = R and falls off toward the origin. That is the
qualitative shape we want to see.

Runs under conda env CXI.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Match constants used in run_bootstrap_with_slurm_lobe.py ---
peak_bin = 25
mock_energy_value = (peak_bin + 13) * 32
re = (mock_energy_value / 32 - 13) * 0.6 + 29.4          # -> 29.4
dr = 5.0                                                  # NEW: shell thickness
SIGMA_R_CURRENT = 2.0                                     # current Gaussian sigma
N = 300_000

rng = np.random.default_rng(0)

# ---------------------------------------------------------------
# Current implementation (Gaussian in radial, sin^2 in 2D angle)
# ---------------------------------------------------------------
theta_space = np.linspace(-np.pi, np.pi, 2000)
p_theta = np.sin(theta_space) ** 2
p_theta /= p_theta.sum()

theta_2d_cur = rng.choice(theta_space, size=N, p=p_theta)
r_2d_cur     = rng.normal(loc=re, scale=SIGMA_R_CURRENT, size=N)
x_cur = r_2d_cur * np.cos(theta_2d_cur)
y_cur = r_2d_cur * np.sin(theta_2d_cur)

# ---------------------------------------------------------------
# Proposed: 3D uniform shell of thickness dr, projected to xy.
# Keep the 2D sin^2 azimuthal pattern by rejection on phi_2d = atan2(y,x).
# ---------------------------------------------------------------
def sample_projected_shell(n, re, dr, rng):
    """Sample (x, y) from a uniform 3D shell of radius re and thickness dr,
    then project to xy. Isotropic on the sphere so far — no angular lobing."""
    r3d = rng.uniform(re - dr / 2.0, re + dr / 2.0, size=n)
    cos_t = rng.uniform(-1.0, 1.0, size=n)                 # z / r_3d
    sin_t = np.sqrt(np.clip(1.0 - cos_t ** 2, 0.0, 1.0))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = r3d * sin_t * np.cos(phi)
    y = r3d * sin_t * np.sin(phi)
    return x, y

def sample_projected_shell_with_sin2(n, re, dr, rng, batch=None):
    """Same as above, but retain sin^2(phi_2d) lobing in the projected plane
    via rejection sampling on phi_2d = atan2(y, x). Max of sin^2 is 1, so the
    acceptance rate is 0.5 on average."""
    if batch is None:
        batch = int(n * 2.6) + 1024
    out_x = np.empty(n)
    out_y = np.empty(n)
    filled = 0
    while filled < n:
        x, y = sample_projected_shell(batch, re, dr, rng)
        phi2 = np.arctan2(y, x)
        u = rng.uniform(0.0, 1.0, size=batch)
        keep = u < np.sin(phi2) ** 2
        take = min(int(keep.sum()), n - filled)
        idx = np.flatnonzero(keep)[:take]
        out_x[filled:filled + take] = x[idx]
        out_y[filled:filled + take] = y[idx]
        filled += take
    return out_x, out_y

x_iso, y_iso = sample_projected_shell(N, re, dr, rng)
r_2d_iso = np.hypot(x_iso, y_iso)

x_lobe, y_lobe = sample_projected_shell_with_sin2(N, re, dr, rng)
r_2d_lobe = np.hypot(x_lobe, y_lobe)

# ---------------------------------------------------------------
# Analytic Abel curve for the mean-radius shell (dr -> 0) as a reference
# ---------------------------------------------------------------
def abel_shell_pdf(r_grid, R):
    y = np.zeros_like(r_grid)
    m = r_grid < R
    y[m] = r_grid[m] / (R * np.sqrt(np.clip(R ** 2 - r_grid[m] ** 2, 1e-12, None)))
    return y

r_grid = np.linspace(0.01, re + 3 * dr, 800)
abel = abel_shell_pdf(r_grid, re)

# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

ax = axes[0]
bins = np.linspace(0, re + 4 * dr, 120)
ax.hist(r_2d_cur,  bins=bins, alpha=0.55, density=True, color='steelblue',
        label=f'current  ~ Normal(re={re:.1f}, s={SIGMA_R_CURRENT})')
ax.hist(r_2d_iso,  bins=bins, alpha=0.55, density=True, color='seagreen',
        label=f'proposed isotropic shell (dr={dr})')
ax.hist(r_2d_lobe, bins=bins, alpha=0.55, density=True, color='crimson',
        label=f'proposed shell + sin^2 lobe (dr={dr})')
ax.plot(r_grid, abel, ':', color='k', label='Abel P(r) for zero-thickness shell')
ax.axvline(re, ls='--', color='k', alpha=0.5)
ax.set_xlabel('r_2d  [detector px]')
ax.set_ylabel('density')
ax.set_xlim(0, re + 4 * dr)
ax.legend(fontsize=8, loc='upper left')
ax.set_title('Radial distribution of lobe electrons')

def scatter_panel(ax, x, y, title, color):
    idx = rng.integers(0, len(x), size=8000)
    ax.scatter(x[idx], y[idx], s=1.2, alpha=0.35, color=color)
    ax.set_aspect('equal')
    ax.set_xlim(-1.4 * re, 1.4 * re)
    ax.set_ylim(-1.4 * re, 1.4 * re)
    ax.axhline(0, color='gray', lw=0.4)
    ax.axvline(0, color='gray', lw=0.4)
    ax.set_title(title)

scatter_panel(axes[1], x_cur, y_cur, 'current: N(re,2) x sin^2(theta)', 'steelblue')
scatter_panel(axes[2], x_lobe, y_lobe, f'proposed: shell(re,dr={dr}) proj. + sin^2 lobe', 'crimson')

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'lobe_radial_distribution.png')
fig.savefig(out, dpi=140)
print(f'Wrote {out}')
