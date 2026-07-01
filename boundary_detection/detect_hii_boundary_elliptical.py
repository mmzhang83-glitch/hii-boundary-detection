"""Unified HII region boundary detection — elliptical prior.

Independent entry point for elliptical prior search region.
Uses elliptical polar sampling → Sobel → DP → stable boundary scan → bootstrap.

All algorithm parameters are loaded from hii_detection_config.yaml.
Model-specific parameters (xc, yc, a, b, phi) must be provided explicitly.
This module does NOT depend on any test configuration.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict
import yaml
import logging

from .find_stable_boundary import find_stable_boundary
from .bootstrap_boundary import bootstrap_boundary_uncertainty

logger = logging.getLogger("hii_boundary")


def detect_hii_boundary_elliptical(
    data: np.ndarray,
    xc: Optional[float] = None,
    yc: Optional[float] = None,
    a: Optional[float] = None,
    b: Optional[float] = None,
    phi: Optional[float] = None,
    *,
    error_map: Optional[np.ndarray] = None,
    clean_image: Optional[np.ndarray] = None,
    config_path: Optional[str | Path] = None,
    n_bootstrap: Optional[int] = None,
    n_workers: Optional[int] = None,
    seed: int = 42,
    detect_rising_edge: Optional[bool] = None,
    # Algorithm parameters — None = use config/default
    method: Optional[str] = None,
    smoothing_fwhm: Optional[float] = None,
    cost_map_smoothing_sigma: Optional[float] = None,
    gradient_smoothing_sigma: Optional[float] = None,
    boundary_smoothing_sigma: Optional[float] = None,
    f_min_start_ratio: Optional[float] = None,
    f_min_min_pixels: Optional[float] = None,
    f_max_limit_ratio: Optional[float] = None,
    angular_snr_weighting: Optional[bool] = None,
    angular_snr_sigma: Optional[float] = None,
    coherence_penalty_weight: Optional[float] = None,
    coherence_sigma: Optional[float] = None,
    stable_window: Optional[int] = None,
    stable_threshold: Optional[float] = None,
    n_steps: Optional[int] = None,
    gradient_strip_width: Optional[int] = None,
    selection_weight_cost: Optional[float] = None,
    selection_weight_length: Optional[float] = None,
    selection_weight_std: Optional[float] = None,
    contrast_min: Optional[float] = None,
    contrast_strip_width: Optional[int] = None,
    show_diagnostics: bool = False,
    showfile: Optional[Path] = None,
) -> Dict:
    """Detect HII region boundary with elliptical prior.

    Parameters
    ----------
    data : np.ndarray
        2D image data.
    xc, yc : float
        Ellipse center coordinates (pixels).
    a : float
        Semi-major axis (pixels).
    b : float
        Semi-minor axis (pixels).
    phi : float
        Rotation angle (radians), 0 = major axis aligned with x-axis.
    error_map, clean_image, config_path, n_bootstrap, n_workers, seed :
        See detect_hii_boundary.
    method, smoothing_fwhm, ... (algorithm params) :
        Same as detect_hii_boundary. None = use config/default.
    f_min_start_ratio, f_min_min_pixels, f_max_limit_ratio :
        Elliptical scan parameters.

    Returns
    -------
    dict with same keys as detect_hii_boundary.
    """
    # xc, yc, a, b, phi are model-specific — must be provided explicitly
    if xc is None or yc is None or a is None or b is None:
        raise ValueError("xc, yc, a, b must be provided explicitly")
    _xc, _yc, _a, _b = xc, yc, a, b
    _phi = phi if phi is not None else 0.0

    # Load ALL algorithm params from hii_detection_config.yaml
    algo_config = Path(__file__).parent / 'hii_detection_config.yaml'
    if algo_config.exists():
        with open(algo_config, 'r') as f:
            algo_data = yaml.safe_load(f) or {}
    else:
        algo_data = {}

    # Build explicit kwargs — only algorithm params, no model params
    algo_explicit = {
        'method': method,
        'smoothing_fwhm': smoothing_fwhm,
        'cost_map_smoothing_sigma': cost_map_smoothing_sigma,
        'gradient_smoothing_sigma': gradient_smoothing_sigma,
        'boundary_smoothing_sigma': boundary_smoothing_sigma,
        'f_min_start_ratio': f_min_start_ratio,
        'f_min_min_pixels': f_min_min_pixels,
        'f_max_limit_ratio': f_max_limit_ratio,
        'angular_snr_weighting': angular_snr_weighting,
        'angular_snr_sigma': angular_snr_sigma,
        'coherence_penalty_weight': coherence_penalty_weight,
        'coherence_sigma': coherence_sigma,
        'stable_window': stable_window,
        'stable_threshold': stable_threshold,
        'n_steps': n_steps,
        'n_bootstrap': n_bootstrap,
        'n_workers': n_workers,
        'detect_rising_edge': detect_rising_edge,
        'gradient_strip_width': gradient_strip_width,
        'selection_weight_cost': selection_weight_cost,
        'selection_weight_length': selection_weight_length,
        'selection_weight_std': selection_weight_std,
        'contrast_min': contrast_min,
        'contrast_strip_width': contrast_strip_width,
    }

    # Merge: algo defaults + explicit overrides
    params = dict(algo_data)
    for key, value in algo_explicit.items():
        if value is not None:
            params[key] = value

    # Extract f_min params (map to rmin_* for find_stable_boundary)
    _f_min_start = params.pop('f_min_start_ratio', 0.05)
    _f_min_pixels = params.pop('f_min_min_pixels', 5.0)
    _f_max_limit = params.pop('f_max_limit_ratio', 0.7)
    params['rmin_start_ratio'] = _f_min_start
    params['rmin_min_pixels'] = _f_min_pixels
    params['rmax_limit_ratio'] = _f_max_limit

    # Floor f_min_start_ratio
    if _a * _f_min_start < _f_min_pixels:
        params['rmin_start_ratio'] = _f_min_pixels / _a

    _n_bootstrap = params.pop('n_bootstrap')
    _n_workers = params.pop('n_workers', 1)
    _outlier_removal = params.pop('outlier_removal', True)
    _outlier_k = params.pop('outlier_k', 3.0)

    # Use fractional scan mode with reference rmax = a (semi-major axis)
    params['use_fractional_scan'] = True
    params['fractional_rmax'] = _a
    params['ellipse'] = (_a, _b, _phi)

    logger.info(
        "HII elliptical detection — center=(%.0f, %.0f), a=%.0f, b=%.0f, phi=%.2f, n_bootstrap=%d",
        _xc, _yc, _a, _b, _phi, _n_bootstrap,
    )

    # Step 1: Stable boundary detection
    if showfile is not None:
        Path(showfile).parent.mkdir(parents=True, exist_ok=True)

    stable_result = find_stable_boundary(
        data=data,
        xc=_xc,
        yc=_yc,
        rmax=_a,  # rmax = semi-major axis
        error_map=error_map,
        show_diagnostics=show_diagnostics,
        showfile=showfile,
        **params,
    )

    logger.info(
        "Stable boundary: mean=%.1f px, type=%s, f_min=%.3f",
        float(np.mean(stable_result['boundary_radii'])),
        stable_result.get('boundary_type', 'iterative'),
        stable_result.get('f_min_final', 0),
    )

    # Step 2: Bootstrap uncertainty (if applicable)
    bootstrap_result = None
    if error_map is not None and _n_bootstrap > 0:
        bootstrap_result = bootstrap_boundary_uncertainty(
            data=data,
            xc=_xc,
            yc=_yc,
            rmax=_a,
            n_bootstrap=_n_bootstrap,
            n_workers=_n_workers,
            error_map=error_map,
            clean_image=clean_image,
            seed=seed,
            **params,
        )

    # Step 3: Build unified result
    result = {
        'boundary_radii': stable_result['boundary_radii'],
        'boundary_angles': stable_result['boundary_angles'],
        'boundary_x': stable_result.get('boundary_x'),
        'boundary_y': stable_result.get('boundary_y'),
        'xc': _xc,
        'yc': _yc,
        'rmax': _a,
        'rmin_final': stable_result.get('rmin_final'),
        'f_min_final': stable_result.get('f_min_final'),
        'boundary_type': stable_result.get('boundary_type', 'iterative'),
        'has_stable_region': stable_result.get('has_stable_region', False),
        'fallback_used': stable_result.get('fallback_used', False),
        'stable_regions_info': stable_result.get('stable_regions_info', []),
        'ellipse': (_a, _b, _phi),
        '_stable_result': stable_result,
        '_resolved_params': params,
    }

    if bootstrap_result is not None:
        result['bootstrap_boundaries'] = bootstrap_result['bootstrap_boundaries']
        result['scenario'] = bootstrap_result['scenario']
        result['_bootstrap_result'] = bootstrap_result

        # Select valid iterations (has_stable filter)
        has_stable = bootstrap_result.get('bootstrap_has_stable')
        all_bounds = bootstrap_result['bootstrap_boundaries']
        if has_stable is not None and np.any(has_stable):
            valid_bounds = all_bounds[has_stable]
        else:
            valid_bounds = all_bounds

        n_all_valid = len(valid_bounds)
        n_total = len(all_bounds)
        n_filtered_stable = n_total - n_all_valid

        # Outlier 剔除（可配置）
        n_outliers = 0
        if _outlier_removal:
            from .bootstrap_boundary import remove_outliers_mad
            mask = remove_outliers_mad(valid_bounds, k=_outlier_k)
            n_outliers = len(mask) - np.sum(mask)
            if n_outliers > 0:
                logger.info("Bootstrap: removed %d outlier iterations (k=%.1f)", n_outliers, _outlier_k)
            valid_bounds = valid_bounds[mask]

        # 用 MAD 过滤后的边界计算中位数和不确定性
        median_radii = np.median(valid_bounds, axis=0)
        result['boundary_radii'] = median_radii
        result['boundary_x'] = _xc + median_radii * np.cos(result['boundary_angles'])
        result['boundary_y'] = _yc + median_radii * np.sin(result['boundary_angles'])

        # 不确定性：在 MAD 剔除之后重新计算
        correction = 1.0 / np.sqrt(2.0) if bootstrap_result.get('correction_applied') else 1.0
        sigma_raw = np.std(valid_bounds, axis=0)
        sigma = sigma_raw * correction
        result['boundary_uncertainty'] = sigma
        result['mean_uncertainty'] = float(np.mean(sigma))
        result['max_uncertainty'] = float(np.max(sigma))
        result['n_valid_bootstrap'] = len(valid_bounds)
        result['n_filtered_bootstrap'] = n_filtered_stable + n_outliers

        mu = result['mean_uncertainty']
        logger.info("Bootstrap: scenario %s, %d/%d valid (stable filter %d, MAD filter %d), mean σ=%.2f px, max σ=%.2f px",
                      result['scenario'], len(valid_bounds), n_total,
                      n_filtered_stable, n_outliers, mu, result['max_uncertainty'])
    else:
        result['boundary_uncertainty'] = None
        result['bootstrap_boundaries'] = None
        result['mean_uncertainty'] = None
        result['max_uncertainty'] = None
        result['scenario'] = 'C'

    return result
