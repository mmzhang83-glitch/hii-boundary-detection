"""Chart generation functions for the test framework."""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

from test_analysis import ExpectedBoundary


def _ensure_dir(save_path: Path) -> None:
    """Create parent directories for save_path if they don't exist."""
    save_path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Boundary overlay
# ---------------------------------------------------------------------------

def plot_boundary_overlay(
    image: np.ndarray,
    xc: float,
    yc: float,
    r_detected: np.ndarray,
    angles: np.ndarray,
    r_expected: float,
    save_path,
    title: str = "",
    boundary_uncertainty=None,
    expected_xc=None,
    expected_yc=None,
    show_expected: bool = True,
    polar_image: np.ndarray = None,
    polar_rr: np.ndarray = None,
) -> Path:
    """Overlay: full image + zoom + optional uncertainty panel.

    Parameters
    ----------
    polar_image, polar_rr : ndarray or None
        Original-resolution polar ROI and radial grid for uncertainty panel
        background.  Only used when boundary_uncertainty is given.
    """
    from matplotlib.patches import Polygon
    from matplotlib.gridspec import GridSpec
    from astropy.visualization import ZScaleInterval

    _ensure_dir(save_path)

    if expected_xc is None:
        expected_xc = xc
    if expected_yc is None:
        expected_yc = yc

    vmin, vmax = np.percentile(image, [2, 98])
    _has_unc = boundary_uncertainty is not None

    if _has_unc:
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 0.55])
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
    else:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(8, 4))

    if title:
        fig.suptitle(title)

    # --- Left: full image ---
    ax_left.imshow(image, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    xd = xc + r_detected * np.cos(angles)
    yd = yc + r_detected * np.sin(angles)
    ax_left.plot(xd, yd, color='red', linewidth=1.0, label='Detected')

    _poly = None
    if _has_unc:
        r_upper = r_detected + boundary_uncertainty
        r_lower = np.maximum(r_detected - boundary_uncertainty, 0)
        x_upper = xc + r_upper * np.cos(angles)
        y_upper = yc + r_upper * np.sin(angles)
        x_lower = xc + r_lower * np.cos(angles)
        y_lower = yc + r_lower * np.sin(angles)
        poly_x = np.concatenate([x_upper, x_lower[::-1]])
        poly_y = np.concatenate([y_upper, y_lower[::-1]])
        _poly = Polygon(np.column_stack([poly_x, poly_y]),
                        color='pink', alpha=0.8, linewidth=0, label='±1σ')
        ax_left.add_patch(_poly)

    _expected_kw = dict(color='black', linestyle='--', linewidth=1.5,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
    if show_expected:
        theta_c = np.linspace(0, 2 * np.pi, 361)
        ax_left.plot(expected_xc + r_expected * np.cos(theta_c),
                     expected_yc + r_expected * np.sin(theta_c),
                     label='Expected', **_expected_kw)

    ax_left.plot(xc, yc, 'rx', markersize=8, markeredgewidth=2)
    ax_left.set_title('Full Image')
    ax_left.legend(fontsize=8)
    ax_left.set_aspect('equal')

    # --- Right: zoomed ---
    _zoom_r = r_expected if show_expected else float(np.mean(r_detected))
    zoom = 1.5 * _zoom_r
    ax_right.imshow(image, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    ax_right.plot(xd, yd, color='red', linewidth=1.0)
    if _poly is not None:
        ax_right.add_patch(Polygon(np.column_stack([poly_x, poly_y]),
                            color='pink', alpha=0.25, linewidth=0))
    if show_expected:
        ax_right.plot(expected_xc + r_expected * np.cos(theta_c),
                      expected_yc + r_expected * np.sin(theta_c),
                      **_expected_kw)
    ax_right.plot(xc, yc, 'rx', markersize=8, markeredgewidth=2)
    ax_right.set_xlim(xc - zoom, xc + zoom)
    ax_right.set_ylim(yc - zoom, yc + zoom)
    ax_right.set_title('Zoom (1.5x radius)')
    ax_right.set_aspect('equal')

    # --- Bottom: uncertainty panel ---
    if _has_unc:
        ax3 = fig.add_subplot(gs[1, :])
        angle_deg = np.degrees(angles)
        r_upper = r_detected + boundary_uncertainty
        r_lower = np.maximum(r_detected - boundary_uncertainty, 0)

        if polar_image is not None and polar_rr is not None:
            v1, v2 = ZScaleInterval().get_limits(polar_image)
            ax3.imshow(polar_image, origin='lower', cmap='viridis',
                       extent=[0, 360, polar_rr[0], polar_rr[-1]],
                       aspect='auto', vmin=v1, vmax=v2, zorder=0)

        _mu = float(np.mean(boundary_uncertainty))
        ax3.plot(angle_deg, r_detected, 'r-', linewidth=1.5, label='Boundary')
        ax3.fill_between(angle_deg, r_lower, r_upper, alpha=0.3, color='pink',
                         label=f'±1σ (mean={_mu:.2f} px)')
        ax3.set_xlabel('Angle (deg)')
        ax3.set_ylabel('Radius (pixels)')
        ax3.set_xlim(0, 360)
        _pad = max(2, np.percentile(boundary_uncertainty, 95) * 0.5)
        _y_lo = max(np.min(r_lower) - _pad, 0)
        _y_hi = np.max(r_upper) + _pad
        # Ensure at least 10 px y-range
        if _y_hi - _y_lo < 10:
            _mid = (_y_lo + _y_hi) / 2
            _y_lo = max(_mid - 5, 0)
            _y_hi = _mid + 5
        ax3.set_ylim(_y_lo, _y_hi)
        ax3.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# 2. Radial profile diagnostic (4-panel)
# ---------------------------------------------------------------------------

def plot_radial_profile(
    expected: ExpectedBoundary,
    r_detected_mean: float,
    save_path,
    title: str = "",
) -> Path:
    """Four-panel vertical stack: f_raw/f_smooth, gradient, score, cost.

    Parameters
    ----------
    expected : ExpectedBoundary
        The analytically computed expected boundary.
    r_detected_mean : float
        Mean detected boundary radius for comparison.
    save_path : Path
        Output PNG path.
    title : str
        Optional suptitle.
    """
    _ensure_dir(save_path)

    r_grid = expected.r_grid
    f_raw = expected.f_raw
    f_smooth = expected.f_smooth
    G = expected.gradient
    score = expected.score
    r_exp = expected.radius

    fig, axes = plt.subplots(4, 1, figsize=(5, 8), sharex=True)
    if title:
        fig.suptitle(title)

    # --- Panel 1: f_raw and f_smooth ---
    ax = axes[0]
    ax.step(r_grid, f_raw, where='pre', color='gray', alpha=0.6, linewidth=1.0, label='f_raw(r)')
    ax.plot(r_grid, f_smooth, color='blue', linewidth=1.5, label='f_smooth(r)')
    ax.axvline(r_exp, color='green', linestyle='--', linewidth=1.5, label='Expected')
    ax.axvline(r_detected_mean, color='red', linestyle=':', linewidth=1.5, label='Detected')
    ax.set_ylabel('Intensity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Gradient G(r) ---
    ax = axes[1]
    ax.plot(r_grid, G, color='darkorange', linewidth=1.5, label='G(r)')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(r_exp, color='green', linestyle='--', linewidth=1.5)
    ax.axvline(r_detected_mean, color='red', linestyle=':', linewidth=1.5)
    ax.set_ylabel('Gradient')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Score(r) ---
    ax = axes[2]
    ax.plot(r_grid, score, color='purple', linewidth=1.5, label='Score(r)')
    ax.axvline(r_exp, color='green', linestyle='--', linewidth=1.5)
    ax.axvline(r_detected_mean, color='red', linestyle=':', linewidth=1.5)
    ax.set_ylabel('Score')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Cost equivalent ---
    ax = axes[3]
    # For G>0 regions, cost = 1 - normalized_log(score + 1e-6); else 1.0
    cost = np.ones_like(score)
    pos_mask = G > 0
    if np.any(pos_mask):
        log_score = np.log(np.maximum(score[pos_mask], 1e-30) + 1e-6)
        if np.ptp(log_score) > 1e-12:
            norm_log = log_score / np.max(log_score)
            cost_pos = 1.0 - norm_log
        else:
            cost_pos = np.zeros_like(log_score)
        cost[pos_mask] = cost_pos
    ax.plot(r_grid, cost, color='brown', linewidth=1.5, label='Cost equiv.')
    ax.axvline(r_exp, color='green', linestyle='--', linewidth=1.5)
    ax.axvline(r_detected_mean, color='red', linestyle=':', linewidth=1.5)
    ax.set_xlabel('Radius (pixels)')
    ax.set_ylabel('Cost equiv.')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# 3. Cost map with DP path
# ---------------------------------------------------------------------------

def plot_cost_map_with_path(
    cost_map: np.ndarray,
    cost_map_radii: np.ndarray,
    boundary_radii: np.ndarray,
    angles: np.ndarray,
    rmin: float,
    save_path,
) -> Path:
    """Polar cost map with the DP-optimal boundary path overlaid.

    Parameters
    ----------
    cost_map : 2D array (n_radii, n_angles)
        The cost matrix used by the DP.
    cost_map_radii : 1D array
        Radius values for each row of cost_map.
    boundary_radii : 1D array
        Optimal boundary radius at each angle (length n_angles).
    angles : 1D array
        Angles in radians (length n_angles, may differ from cost_map columns).
    save_path : Path
        Output PNG path.
    """
    _ensure_dir(save_path)

    r_start = cost_map_radii[0]
    r_end = cost_map_radii[-1]

    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(
        cost_map,
        origin='lower',
        extent=[0, 360, r_start, r_end],
        aspect='auto',
        cmap='viridis',
        interpolation='nearest',
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cost')

    # Overlay DP path in cyan - interpolate angles to 0..360 degrees
    angle_deg = np.degrees(angles) % 360
    # Sort by angle for proper line plot
    sort_idx = np.argsort(angle_deg)
    ax.plot(angle_deg[sort_idx], boundary_radii[sort_idx],
            color='cyan', linewidth=1.5, label='DP path')

    ax.axhline(rmin, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Radius (pixels)')
    ax.set_title('Cost map with DP path')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# 4. Error vs angle
# ---------------------------------------------------------------------------

def plot_error_vs_angle(
    angles: np.ndarray,
    r_detected: np.ndarray,
    r_expected: float | np.ndarray,
    save_path,
    y_range_factor: float = 3.0,
    argmax_curves: dict | None = None,
) -> Path:
    """Error = r_detected - r_expected vs angle.

    Optionally overlays argmax baseline error curves for comparison.

    Parameters
    ----------
    argmax_curves : dict or None
        If provided, keys are labels (e.g. "argmax_final") and values are
        1D arrays of boundary radii. Error curves are overlaid as dashed lines.
    """
    _ensure_dir(save_path)

    angle_deg = np.degrees(angles)
    error = r_detected - r_expected

    # Sort by angle
    sort_idx = np.argsort(angle_deg)
    angle_sorted = angle_deg[sort_idx]
    error_sorted = error[sort_idx]

    fig, ax = plt.subplots(figsize=(5, 2.5))

    # DP error (solid blue)
    ax.plot(angle_sorted, error_sorted, color='blue', linewidth=1.0,
            label=f'DP (MRE={np.mean(error):+.2f})')
    ax.axhline(0, color='black', linewidth=0.5)

    # Argmax overlay curves
    argmax_colors = {"argmax_final": "red", "argmax_raw": "orange"}
    if argmax_curves:
        for label, r_argmax in argmax_curves.items():
            err_argmax = r_argmax - r_expected
            err_sort = err_argmax[sort_idx]
            color = argmax_colors.get(label, "gray")
            ax.plot(angle_sorted, err_sort, color=color, linewidth=1.0,
                    linestyle='--', alpha=0.8,
                    label=f'{label} (MRE={np.mean(err_argmax):+.2f})')

    # Dynamic y-axis range: expand to cover all curves
    all_errors = [error_sorted]
    if argmax_curves:
        for r_argmax in argmax_curves.values():
            all_errors.append((r_argmax - r_expected)[sort_idx])
    combined = np.concatenate(all_errors)
    error_min, error_max = combined.min(), combined.max()
    error_center = (error_min + error_max) / 2
    error_half_range = max((error_max - error_min) / 2 * y_range_factor, 0.5)
    ax.set_ylim(error_center - error_half_range, error_center + error_half_range)

    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Error (pixels)')
    ax.set_title('Detected - Expected Radius vs Angle')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# 5. Multi-realization overlay
# ---------------------------------------------------------------------------

def plot_multi_realization_overlay(
    image: np.ndarray,
    xc: float,
    yc: float,
    all_r_detected: Sequence[np.ndarray],
    angles: np.ndarray,
    r_expected: float,
    save_path,
    title: str = "",
) -> Path:
    """Overlay all detected boundaries on the true image with expected circle.

    Parameters
    ----------
    image : 2D array
        The true (noise-free) image.
    xc, yc : float
        Center coordinates (pixels).
    all_r_detected : sequence of 1D arrays
        One array of boundary radii per realization.
    angles : 1D array
        Angles in radians, matching each r_detected array.
    r_expected : float
        Expected boundary radius.
    save_path : Path
        Output PNG path.
    title : str
        Optional suptitle.
    """
    _ensure_dir(save_path)

    vmin, vmax = np.percentile(image, [2, 98])

    fig, ax = plt.subplots(figsize=(5, 5))
    if title:
        ax.set_title(title)

    ax.imshow(image, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)

    # Draw each detected boundary with semi-transparent red
    n = len(all_r_detected)
    for i, r_det in enumerate(all_r_detected):
        xd = xc + r_det * np.cos(angles)
        yd = yc + r_det * np.sin(angles)
        alpha = 0.3 if n > 1 else 1.0
        ax.plot(xd, yd, color='red', linewidth=0.6, alpha=alpha)

    # Expected circle
    theta_c = np.linspace(0, 2 * np.pi, 361)
    ax.plot(xc + r_expected * np.cos(theta_c),
            yc + r_expected * np.sin(theta_c),
            color='green', linestyle='--', linewidth=1.5, label=f'Expected ({n} runs)')
    ax.plot(xc, yc, 'rx', markersize=8, markeredgewidth=2)
    ax.set_aspect('equal')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# 6. MRE vs noise
# ---------------------------------------------------------------------------

def plot_mre_vs_noise(
    noise_levels: np.ndarray,
    mre_means: np.ndarray,
    mre_stds: np.ndarray,
    save_path,
    title: str = "",
    xlabel: str = "Noise σ",
) -> Path:
    """MRE vs noise level with error bars.

    Parameters
    ----------
    noise_levels : 1D array
        Noise values tested.
    mre_means : 1D array
        Mean MRE at each noise level.
    mre_stds : 1D array
        Standard deviation of MRE at each noise level.
    save_path : Path
        Output PNG path.
    title : str
        Optional suptitle.
    xlabel : str
        X-axis label.
    """
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(5, 3))
    if title:
        ax.set_title(title)

    ax.errorbar(noise_levels, mre_means, yerr=mre_stds,
                fmt='o-', capsize=4, markersize=6,
                color='steelblue', ecolor='gray',
                label='MRE ± 1σ')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('MRE (pixels)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# 7. Noise examples grid
# ---------------------------------------------------------------------------

def plot_noise_examples(
    noisy_images: Sequence[np.ndarray],
    noise_levels: np.ndarray,
    save_path,
    ncols: int = 3,
) -> Path:
    """Grid of noisy images showing increasing noise levels.

    Parameters
    ----------
    noisy_images : sequence of 2D arrays
        One image per noise level.
    noise_levels : 1D array
        Noise values corresponding to each image.
    save_path : Path
        Output PNG path.
    ncols : int
        Number of columns in the grid.
    """
    _ensure_dir(save_path)

    n = len(noisy_images)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i in range(n):
        ax = axes[i]
        im = ax.imshow(noisy_images[i], origin='lower', cmap='gray')
        ax.set_title(f'σ = {noise_levels[i]:.2f}')
        ax.axis('off')

    # Turn off extra subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# 8. Parameter heatmap
# ---------------------------------------------------------------------------

def plot_param_heatmap(
    x_values: np.ndarray,
    y_values: np.ndarray,
    mre_grid: np.ndarray,
    xlabel: str,
    ylabel: str,
    save_path,
) -> Path:
    """Heatmap of MRE over a 2D parameter grid.

    Parameters
    ----------
    x_values : 1D array
        X-axis parameter values (len = ncols).
    y_values : 1D array
        Y-axis parameter values (len = nrows).
    mre_grid : 2D array (nrows, ncols)
        MRE values at each (y, x) combination.
    xlabel, ylabel : str
        Axis labels.
    save_path : Path
        Output PNG path.
    """
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(5, 4))

    mesh = ax.pcolormesh(x_values, y_values, mre_grid,
                         cmap='RdYlGn_r', shading='auto')
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('MRE (pixels)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'MRE vs {xlabel} and {ylabel}')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# 9. Boundary shift curves
# ---------------------------------------------------------------------------

def plot_boundary_shift_curves(
    angles: np.ndarray,
    all_boundaries: Dict[str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]],
    save_path,
    title: str = "",
    y_range_factor: float = 3.0,
) -> Path:
    """Overlaid boundary radius vs angle curves for multiple runs.

    Parameters
    ----------
    angles : 1D array
        Default angles in radians (used when all_boundaries values are plain arrays).
    all_boundaries : dict of str → 1D array or (1D array, 1D array)
        Mapping from label to either:
        - boundary radius array (uses shared `angles`)
        - (angles, radii) tuple (each boundary has its own angles)
    save_path : Path
        Output PNG path.
    title : str
        Optional suptitle.
    y_range_factor : float
        Expand y-axis range by this factor beyond data min/max.
    """
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(6, 3))
    if title:
        ax.set_title(title)

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(all_boundaries))))
    all_r_min, all_r_max = np.inf, -np.inf

    for idx, (label, val) in enumerate(all_boundaries.items()):
        if isinstance(val, tuple) and len(val) == 2:
            a, r_det = val
        else:
            a, r_det = angles, val
        angle_deg = np.degrees(a)
        sort_idx = np.argsort(angle_deg)
        ax.plot(angle_deg[sort_idx], r_det[sort_idx],
                color=colors[idx % len(colors)],
                linewidth=1.0, alpha=0.8, label=label)
        all_r_min = min(all_r_min, r_det.min())
        all_r_max = max(all_r_max, r_det.max())

    # Dynamic y-axis range: expand data range by y_range_factor
    r_center = (all_r_min + all_r_max) / 2
    r_half_range = (all_r_max - all_r_min) / 2 * y_range_factor
    ax.set_ylim(r_center - r_half_range, r_center + r_half_range)

    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Boundary radius (pixels)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def _set_symmetric_ticks(cb, vmax: float, n_ticks: int = 5):
    """Set symmetric colorbar ticks with nicely-rounded values (linear scale)."""
    vmax = abs(vmax) if np.isfinite(vmax) else 0.0
    if vmax < 1e-12:
        cb.set_ticks([0])
        return
    half = (n_ticks - 1) // 2
    raw_step = vmax / half
    exp = 10 ** np.floor(np.log10(max(raw_step, 1e-300)))
    if exp < 1e-300:
        cb.set_ticks([0])
        return
    mant = raw_step / exp
    if mant <= 1.2:
        nice_step = exp
    elif mant <= 2.1:
        nice_step = 2 * exp
    elif mant <= 3.5:
        nice_step = 2.5 * exp
    else:
        nice_step = 5 * exp
    ticks = np.arange(-half * nice_step, (half + 0.1) * nice_step, nice_step)
    cb.set_ticks([float(t) for t in ticks])
    cb.ax.set_yticklabels([f"{t:g}" for t in ticks])


def plot_polar_pipeline(
    polar_roi: np.ndarray,
    polar_smooth: np.ndarray,
    score_map: np.ndarray,
    dv_map: np.ndarray,
    cost_map: np.ndarray,
    cost_map_radii: np.ndarray,
    boundary_radii: np.ndarray,
    angles: np.ndarray,
    rr_roi: np.ndarray,
    save_path: Path,
    title: str = ""
) -> Path:
    """
    6-panel pipeline: raw polar → smoothed → gradient → score → cost → cross-section.
    The detected boundary is overlaid on all 2D panels.
    """
    _ensure_dir(save_path)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    ang_deg = angles / np.pi * 180
    extent_r = [0, 360, rr_roi[0], rr_roi[-1]]
    extent_cost = [0, 360, cost_map_radii[0], cost_map_radii[-1]]

    LOG_RATIO_THRESHOLD = 5  # switch to log scale when dynamic range > 5×

    # Panel 1: Raw polar image
    ax = axes[0, 0]
    v1, v2 = np.nanpercentile(polar_roi, [0.5, 99.5])
    _use_log1 = v2 / max(abs(v1), 1e-10) > LOG_RATIO_THRESHOLD
    _norm1 = SymLogNorm(linthresh=max(abs(v1), 1e-10) * 2, vmin=v1, vmax=v2) if _use_log1 else None
    _im_kw1 = dict(norm=_norm1) if _use_log1 else dict(vmin=v1, vmax=v2)
    im = ax.imshow(polar_roi, origin='lower', cmap='viridis',
                   extent=extent_r, aspect='auto', **_im_kw1)
    ax.set_title('1. Raw Polar Image (ROI)')
    ax.set_xlabel('Angle (deg)'); ax.set_ylabel('Radius (px)')
    plt.colorbar(im, ax=ax)

    # Panel 2: Smoothed polar
    ax = axes[0, 1]
    v1, v2 = np.nanpercentile(polar_smooth, [2, 98])
    _use_log2 = v2 / max(abs(v1), 1e-10) > LOG_RATIO_THRESHOLD
    _norm2 = SymLogNorm(linthresh=max(abs(v1), 1e-10) * 2, vmin=v1, vmax=v2) if _use_log2 else None
    _im_kw2 = dict(norm=_norm2) if _use_log2 else dict(vmin=v1, vmax=v2)
    im = ax.imshow(polar_smooth, origin='lower', cmap='viridis',
                   extent=extent_r, aspect='auto', **_im_kw2)
    ax.plot(ang_deg, boundary_radii, 'r-', linewidth=1.0)
    ax.set_title('2. Smoothed Polar Image')
    ax.set_xlabel('Angle (deg)')
    plt.colorbar(im, ax=ax)

    # Panel 3: Sobel gradient
    ax = axes[0, 2]
    vmax_abs = abs(dv_map).max()
    vmin_abs = max(abs(dv_map[dv_map != 0]).min() if np.any(dv_map != 0) else 1e-10, 1e-10)
    _use_log3 = vmax_abs / vmin_abs > LOG_RATIO_THRESHOLD
    _norm3 = SymLogNorm(linthresh=vmin_abs * 10, vmin=-vmax_abs, vmax=vmax_abs) if _use_log3 else None
    _im_kw3 = dict(norm=_norm3) if _use_log3 else dict(vmin=-vmax_abs, vmax=vmax_abs)
    im = ax.imshow(dv_map, origin='lower', cmap='RdBu_r',
                   extent=extent_r, aspect='auto', **_im_kw3)
    ax.plot(ang_deg, boundary_radii, 'c-', linewidth=1.5)
    ax.set_title('3. Sobel Gradient (dv_map)')
    ax.set_xlabel('Angle (deg)')
    cb = plt.colorbar(im, ax=ax)
    if _use_log3:
        # Log scale: one tick per decade (powers of 10), mirror for negatives
        decades = np.arange(np.ceil(np.log10(_norm3.linthresh)),
                            np.floor(np.log10(vmax_abs)) + 1)
        pos_ticks = 10.0 ** decades
        cb.set_ticks(sorted([float(-t) for t in pos_ticks] + [0.0] + [float(t) for t in pos_ticks]))
    else:
        _set_symmetric_ticks(cb, float(np.nanmax(np.abs(dv_map))))

    # Panel 4: Score map
    ax = axes[1, 0]
    sp = score_map[score_map > 0]
    v1 = np.percentile(sp, 5) if len(sp) > 0 else 0
    v2 = np.percentile(score_map, 95)
    im = ax.imshow(score_map, origin='lower', cmap='hot',
                   extent=extent_r, aspect='auto', vmin=v1, vmax=v2)
    ax.plot(ang_deg, boundary_radii, 'c-', linewidth=1.5)
    ax.set_title('4. Score Map = G² / pixel_dr')
    ax.set_xlabel('Angle (deg)'); ax.set_ylabel('Radius (px)')
    plt.colorbar(im, ax=ax)

    # Panel 5: Cost map with DP path
    ax = axes[1, 1]
    im = ax.imshow(cost_map, origin='lower', cmap='magma_r',
                   extent=extent_cost, aspect='auto')
    ax.plot(ang_deg, boundary_radii, 'c-', linewidth=1.5)
    ax.set_title('5. Cost Map with DP Path')
    ax.set_xlabel('Angle (deg)')
    plt.colorbar(im, ax=ax)

    # Panel 6: Cross-section
    ax = axes[1, 2]
    mid_col = score_map.shape[1] // 2
    rr_disp = np.arange(len(rr_roi)) + rr_roi[0]
    ax.step(rr_disp, polar_roi[:, mid_col], where='pre', color='gray', alpha=0.6, lw=0.8, label='Raw')
    ax.plot(rr_disp, polar_smooth[:, mid_col], 'b-', lw=1.5, label='Smoothed')
    if np.max(score_map[:, mid_col]) > 0:
        sn = score_map[:, mid_col] / np.max(score_map[:, mid_col])
        ax.plot(rr_disp, sn * np.max(polar_smooth[:, mid_col]), 'r--', lw=1, label='Score (norm)')
    ax.axvline(boundary_radii[mid_col], color='r', linestyle='-', lw=2,
               label=f'Boundary ({boundary_radii[mid_col]:.1f} px)')
    ax.set_xlim(rr_roi[0], rr_roi[-1])  # override step() auto-zoom on flat data
    ax.set_title('6. Cross-Section at Center')
    ax.set_xlabel('Radius (px)'); ax.set_ylabel('Value')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_model_summary(
    image: np.ndarray,
    xc: float,
    yc: float,
    r_expected: float,
    expected,
    params: dict,
    save_path: Path,
    title: str = "",
    error_map: np.ndarray = None,
    rmax: float = None,
    rmin: float = None,
    detect_rising_edge: bool = True,
) -> Path:
    """
    2-column layout: left = model image (RdBu_r, equal aspect),
    right = 4-row radial profile stack, or zoomed image if error_map provided.

    The left panel shows the expected (ground-truth) boundary as a dashed circle.
    The right panel shows the analytical radial profile stack (f_raw, f_smooth,
    gradient, score) with expected boundary marker, or a zoomed view of the
    image when error_map is provided.

    When *rmax* is provided, a dashed circle at rmax is drawn on the
    left image and a vertical marker on the radial profile panels.
    When *rmin* is provided, a dotted circle at rmin is drawn on the
    left image.
    """
    _ensure_dir(save_path)
    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    r_grid = expected.r_grid
    f_raw = expected.f_raw
    f_smooth = expected.f_smooth
    G = expected.gradient
    score = expected.score

    # --- Left: 2D image ---
    ax_img = fig.add_subplot(gs[0, 0])
    vmin, vmax = np.percentile(image[~np.isnan(image)], [2, 98])
    ax_img.imshow(image, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax,
                  aspect='equal')
    theta = np.linspace(0, 2 * np.pi, 360)
    ax_img.plot(xc + r_expected * np.cos(theta), yc + r_expected * np.sin(theta),
                'g--', linewidth=1.5, label=f'Expected (r={r_expected:.1f})')
    ax_img.plot(xc, yc, 'r+', markersize=10)
    if rmax is not None:
        theta_rmax = np.linspace(0, 2 * np.pi, 360)
        ax_img.plot(xc + rmax * np.cos(theta_rmax),
                    yc + rmax * np.sin(theta_rmax),
                    'c--', linewidth=1.0, alpha=0.8,
                    label=f'rmax={rmax:.0f}')
    if rmin is not None:
        theta_rmin = np.linspace(0, 2 * np.pi, 360)
        ax_img.plot(xc + rmin * np.cos(theta_rmin),
                    yc + rmin * np.sin(theta_rmin),
                    'm:', linewidth=1.0, alpha=0.7,
                    label=f'rmin={rmin:.1f}')
    ax_img.legend(fontsize=8, loc='upper right')
    param_text = ', '.join(f'{k}={v}' for k, v in params.items())
    ax_img.set_title(f'{title}\n{param_text}', fontsize=11)
    ax_img.set_xlabel('x (px)'); ax_img.set_ylabel('y (px)')

    # --- Right: radial profile or zoomed image ---
    if error_map is not None:
        ax_right = fig.add_subplot(gs[0, 1])
        zoom_half = r_expected * 1.5
        ax_right.imshow(image, origin='lower', cmap='viridis',
                        vmin=vmin, vmax=vmax)
        ax_right.set_xlim(xc - zoom_half, xc + zoom_half)
        ax_right.set_ylim(yc - zoom_half, yc + zoom_half)
        theta = np.linspace(0, 2 * np.pi, 360)
        ax_right.plot(xc + r_expected * np.cos(theta), yc + r_expected * np.sin(theta),
                      'g--', linewidth=1.5, label=f'Expected (r={r_expected:.1f})')
        ax_right.set_title(f'Zoom (noise σ = {np.max(error_map):.3f})')
        ax_right.set_xlabel('x (px)'); ax_right.set_ylabel('y (px)')
        ax_right.legend(fontsize=7, loc='upper right')
    else:
        gs_right = gs[0, 1].subgridspec(4, 1, hspace=0.08)
        ax0 = fig.add_subplot(gs_right[0])
        axes = [ax0] + [fig.add_subplot(gs_right[i], sharex=ax0) for i in range(1, 4)]

        # Row 1: f_raw + f_smooth
        ax = axes[0]
        ax.step(r_grid, f_raw, where='pre', color='gray', alpha=0.6, lw=1.0, label='f_raw(r)')
        ax.plot(r_grid, f_smooth, 'b-', lw=1.5, label='f_smooth(r)')
        ax.axvline(r_expected, color='green', linestyle='--', lw=1.5)
        if rmax is not None:
            ax.axvline(rmax, color='gray', linestyle=':', lw=1.0, alpha=0.6, label=f'rmax={rmax:.0f}')
        ax.set_ylabel('Value'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Row 2: Gradient
        ax = axes[1]
        ax.plot(r_grid, G, color='darkorange', lw=1.5)
        ax.axhline(0, color='black', lw=0.5)
        ax.axvline(r_expected, color='green', linestyle='--', lw=1.5)
        ax.set_ylabel('G(r)'); ax.grid(True, alpha=0.3)

        # Row 3: Score
        ax = axes[2]
        ax.plot(r_grid, score, color='purple', lw=1.5)
        ax.axvline(r_expected, color='green', linestyle='--', lw=1.5)
        ax.set_ylabel('Score'); ax.grid(True, alpha=0.3)

        # Row 4: Cost equivalent (matches actual algorithm)
        ax = axes[3]
        cost = np.ones_like(score)
        if detect_rising_edge:
            grad_mask = G > 0
        else:
            grad_mask = G < 0
        if np.any(grad_mask):
            valid_scores = score[grad_mask]
            epsilon = 1e-10
            log_scores = np.log(valid_scores + epsilon)
            min_log = np.min(log_scores)
            max_log = np.max(log_scores)
            if (max_log - min_log) > 1e-12:
                normalized_log = (log_scores - min_log) / (max_log - min_log)
                cost[grad_mask] = (1.0 - normalized_log) + 1e-6
        ax.plot(r_grid, cost, color='brown', lw=1.5)
        ax.axvline(r_expected, color='green', linestyle='--', lw=1.5)
        ax.set_xlabel('Radius (px)'); ax.set_ylabel('Cost'); ax.grid(True, alpha=0.3)

        # Hide x-tick labels on top 3 rows; only bottom shows numbers
        for a in axes[:-1]:
            plt.setp(a.get_xticklabels(), visible=False)
            a.set_xlabel('')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path
