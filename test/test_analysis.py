"""Analytical boundary computation and error metrics for test validation."""

import numpy as np
from dataclasses import dataclass
from typing import Callable
from scipy.ndimage import gaussian_filter1d
import pdb


@dataclass
class ExpectedBoundary:
    """Result of analytical boundary computation."""
    radius: float           # Expected boundary radius (pixels)
    r_grid: np.ndarray      # 1D radius array used for computation
    f_raw: np.ndarray       # Raw radial profile sampled at r_grid
    f_smooth: np.ndarray    # After Gaussian smoothing with sigma_r
    gradient: np.ndarray    # Sobel gradient G(r)
    score: np.ndarray       # Score = G^2 / pixel_dr for G>0, else 0


def compute_expected_boundary(
    radial_func: Callable,
    rmin: float,
    rmax: float,
    sigma_r: float,
    num_radii_samples: int = None,
    pixel_dr: float = 2.0,
    detect_rising_edge: bool = True,
) -> ExpectedBoundary:
    """
    Simulate extract_circle_boundary's processing chain on a radially
    symmetric profile to compute the expected boundary location.

    Uses the SAME polar transform grid as the algorithm:
    n = ceil(rmax) + 1, r_i = i * rmax / (n - 1), spacing = 1.0 px.

    The processing chain for a radially symmetric image:
    1. Polar transform → f(r) sampled on the same grid as manual_polar_transform
    2. Radial Gaussian smoothing with sigma_r
    3. Sobel axis=0: G(r_i) = 4 * (f_smooth(r_{i+1}) - f_smooth(r_{i-1}))
    4. Score(r_i) = G(r_i)^2 / pixel_dr  for G(r_i) > 0, else 0
    5. The DP finds the lowest-cost path → highest-score radius

    Parameters:
        radial_func: f(r) → value, the radial profile.
        rmin, rmax: Search range in physical pixel units.
        sigma_r: Radial smoothing sigma (= smoothing_fwhm / 2.355).
        num_radii_samples: Number of radial samples (= int(ceil(rmax))).
        pixel_dr: Sobel kernel effective distance (default 2.0, matches code).
    """
    # Build the same polar grid used by manual_polar_transform
    if num_radii_samples is None:
        num_radii_samples = int(np.ceil(rmax)) + 1
    full_r_grid = np.linspace(0, rmax, num_radii_samples)
    # Sample with padding beyond ROI (by index, not pixel) to avoid edge effects
    pad = max(10, int(np.ceil(3 * max(sigma_r, 1.0))))
    r_start = max(0, int(np.floor(rmin)) - pad)
    r_end = min(int(np.ceil(rmax)) + pad, num_radii_samples - 1)

    r_grid = full_r_grid[r_start:r_end + 1]
    f_raw = np.asarray(radial_func(r_grid), dtype=float)

    # 1. Radial Gaussian smoothing (matches axis=0 gaussian_filter1d in code)
    if sigma_r > 0:
        f_smooth = gaussian_filter1d(f_raw.astype(float), sigma=sigma_r, axis=0, mode='mirror')
    else:
        f_smooth = f_raw.astype(float).copy()

    # 2. Discrete Sobel operator along axis=0
    # For a θ-independent function, G(r_i) = 4 * (f_smooth(r_{i-1}) - f_smooth(r_{i+1}))
    G = np.zeros_like(f_smooth)
    G[1:-1] = 4.0 * (f_smooth[2:] - f_smooth[:-2])

    # 3. Score: detect_rising_edge selects positive or negative gradients
    if detect_rising_edge:
        score = np.where(G > 0, G ** 2 / pixel_dr, 0.0)
    else:
        score = np.where(G < 0, G ** 2 / pixel_dr, 0.0)

    # 4. Find argmax within ROI [rmin, rmax-1]
    #    (rmax-1 because cost_map excludes the last radius row)
    roi_mask = (r_grid >= rmin) & (r_grid <= rmax - 1)
    roi_indices = np.where(roi_mask)[0]
    if len(roi_indices) == 0:
        # No valid ROI points
        best_idx = len(r_grid) // 2
    else:
        best_idx = roi_indices[np.argmax(score[roi_indices])]

    return ExpectedBoundary(
        radius=r_grid[best_idx],
        r_grid=r_grid,
        f_raw=f_raw,
        f_smooth=f_smooth,
        gradient=G,
        score=score
    )


def compare_boundaries(r_detected: np.ndarray, r_expected: float) -> dict:
    """
    Compute error metrics between detected boundaries and expected radius.

    Parameters:
        r_detected: 1D array of detected boundary radii (one per angle).
        r_expected: Single expected radius, or array of per-angle expected radii.

    Returns dict with keys: mre, rms, max_error, angular_std,
        r_detected_mean, r_detected_min, r_detected_max.

    Note: mre is signed (positive = overestimate, negative = underestimate).
    """
    signed_errors = r_detected - r_expected
    abs_errors = np.abs(signed_errors)
    return {
        'mre': float(np.mean(signed_errors)),
        'rms': float(np.sqrt(np.mean(signed_errors ** 2))),
        'max_error': float(np.max(abs_errors)),
        'angular_std': float(np.std(r_detected)),
        'r_detected_mean': float(np.mean(r_detected)),
        'r_detected_min': float(np.min(r_detected)),
        'r_detected_max': float(np.max(r_detected)),
    }
