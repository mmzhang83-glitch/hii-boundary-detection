"""Argmax baseline: per-angle independent gradient argmax, zero parameters."""

import numpy as np


def compute_argmax_boundary(
    polar_gradient: np.ndarray,   # (n_radii, n_angles)
    rr_roi: np.ndarray,           # (n_radii,), radius value for each gradient row
    rmin: float,
    rmax: float,
    detect_rising_edge: bool = True,
) -> np.ndarray:
    """Per-angle argmax of gradient → boundary radii.

    For each angle θ_j:  r_j = rr_roi[ argmax_r G(r, θ_j) ]

    The argmax is constrained to r ∈ [rmin, rmax-1] to match the DP
    search range (the cost_map excludes the last radius row).

    Parameters
    ----------
    polar_gradient : (n_radii, n_angles) ndarray
        Gradient map (dv_map or dv_map_raw from extract_circle_boundary).
    rr_roi : (n_radii,) ndarray
        Physical radius values corresponding to each row.
    rmin : float
        Lower bound of search range.
    rmax : float
        Upper bound of search range (last row excluded).
    detect_rising_edge : bool
        If True, argmax of positive gradient. If False, argmax of negative.

    Returns
    -------
    boundary_radii : (n_angles,) ndarray
        Detected boundary radius at each angle.
    """
    n_radii = polar_gradient.shape[0]

    # Constrain to same ROI as DP: rows with radius in [rmin, rmax)
    r_min_idx = int(np.searchsorted(rr_roi, rmin))
    r_max_idx = int(np.searchsorted(rr_roi, rmax))

    if detect_rising_edge:
        gradient = np.where(polar_gradient > 0, polar_gradient, -np.inf)
    else:
        gradient = np.where(polar_gradient < 0, -polar_gradient, -np.inf)

    # Mask rows outside [rmin, rmax-1]
    valid_mask = np.zeros(n_radii, dtype=bool)
    valid_mask[r_min_idx:r_max_idx] = True
    gradient_masked = np.where(valid_mask[:, np.newaxis], gradient, -np.inf)

    # Per-angle argmax
    best_rows = np.argmax(gradient_masked, axis=0)
    return rr_roi[best_rows]
