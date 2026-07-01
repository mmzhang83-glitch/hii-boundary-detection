"""Unified HII region boundary detection interface.

Single entry point that wraps find_stable_boundary + bootstrap_boundary_uncertainty.
Resolves parameters by priority: explicit arg > config file > built-in defaults.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict
import yaml
import logging

from .find_stable_boundary import find_stable_boundary
from .bootstrap_boundary import bootstrap_boundary_uncertainty

logger = logging.getLogger("hii_boundary")


# Built-in defaults — used when no config file and no explicit arg
_DEFAULTS: Dict = {
    'method': 'scan',
    'smoothing_fwhm': 2.0,
    'cost_map_smoothing_sigma': 5.0,
    'gradient_smoothing_sigma': 0.0,
    'boundary_smoothing_sigma': 0.0,
    'rmin_start_ratio': 0.05,
    'rmin_min_pixels': 5.0,
    'rmax_limit_ratio': 0.7,
    'angular_snr_weighting': False,
    'angular_snr_sigma': 3.0,
    'coherence_penalty_weight': 0.0,
    'coherence_sigma': 3.0,
    'stable_window': 5,
    'stable_threshold': 0.02,
    'n_steps': 50,
    'n_bootstrap': 100,
    'n_workers': 1,
    'detect_rising_edge': True,
    'gradient_strip_width': 3,
    'selection_weight_cost': 0.4,
    'selection_weight_length': 0.3,
    'selection_weight_std': 0.3,
    'contrast_min': 0.01,
    'contrast_strip_width': 4,
}


def _resolve_params(
    config_path: Optional[Path],
    explicit_kwargs: Dict,
) -> Dict:
    """Resolve parameters by priority chain.

    explicit_kwargs > config_path > auto-detected config > _DEFAULTS
    Only keys with non-None values in explicit_kwargs override.
    """
    # Start with built-in defaults
    resolved = dict(_DEFAULTS)

    # Auto-detect config file in same directory as this module
    auto_config = Path(__file__).parent / 'hii_detection_config.yaml'
    if config_path is not None:
        config_file = Path(config_path)
    elif auto_config.exists():
        config_file = auto_config
    else:
        config_file = None

    if config_file is not None and config_file.exists():
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f) or {}
        resolved.update(config_data)

    # Explicit args (non-None) have highest priority
    for key, value in explicit_kwargs.items():
        if value is not None:
            resolved[key] = value

    return resolved


# ---------------------------------------------------------------------------
# HDF5 save helper
# ---------------------------------------------------------------------------

def _save_result_h5(result: dict, path: Path):
    """Save detect_hii_boundary result dict as self-describing HDF5 file."""
    import h5py

    DESCRIPTIONS = {
        "boundary_radii": "边界半径（像素），每角度一个值（bootstrap 中位数），shape=(360,)",
        "boundary_angles": "对应角度（弧度，0→2π），shape=(360,)",
        "boundary_x": "边界笛卡尔 x 坐标（像素），shape=(360,)",
        "boundary_y": "边界笛卡尔 y 坐标（像素），shape=(360,)",
        "boundary_uncertainty": "每角度 1σ 不确定度（像素），shape=(360,) 或 None",
        "bootstrap_boundaries": "全部有效 bootstrap 迭代边界，shape=(N,360)",
        "bootstrap_has_stable": "各 bootstrap 迭代是否找到稳定区，shape=(N,)",
        "xc": "检测中心 x 坐标（像素）",
        "yc": "检测中心 y 坐标（像素）",
        "rmax": "搜索半径上限（像素）",
        "rmin_final": "最终内半径 rmin（像素）",
        "boundary_type": "边界类型：single_stable / multiple_stable / fallback",
        "has_stable_region": "是否找到至少一个稳定区",
        "fallback_used": "是否因无稳定区退化为 fallback",
        "stable_regions_info": "各候选稳定区详情（list[dict]）",
        "boundary_confidence": "边界置信度（每角度，0-1）",
        "boundary_valid": "每角度边界是否有效（bool）",
        "confidence_threshold": "置信度阈值",
        "scenario": "Bootstrap 场景：A（有 clean 图）/ B（无 clean 图，√2 校正）/ C（无 error_map）",
        "correction_applied": "是否应用了 √2 校正",
        "mean_uncertainty": "平均 1σ 不确定度（像素）",
        "max_uncertainty": "最大 1σ 不确定度（像素）",
        "n_valid_bootstrap": "有效 bootstrap 迭代数",
        "n_filtered_bootstrap": "被过滤的 bootstrap 迭代数（稳定区+MAD）",
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, val in result.items():
            if key.startswith("_"):
                continue  # skip internal
            try:
                if isinstance(val, np.ndarray):
                    ds = f.create_dataset(key, data=val)
                elif val is None:
                    ds = f.create_dataset(key, data=np.array(0, dtype=np.float32))
                    ds.attrs["is_none"] = True
                elif isinstance(val, (list, dict)):
                    raise TypeError("non-serializable")
                else:
                    ds = f.create_dataset(key, data=val)
                ds.attrs["description"] = DESCRIPTIONS.get(key, "")
            except (TypeError, ValueError):
                f.attrs[f"skipped_{key}"] = str(type(val).__name__)
    logger.info("Saved result to %s", path)


def detect_hii_boundary(
    data: np.ndarray,
    xc: float,
    yc: float,
    rmax: float,
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
    rmin_start_ratio: Optional[float] = None,
    rmin_min_pixels: Optional[float] = None,
    rmax_limit_ratio: Optional[float] = None,
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
    save_h5_path: Optional[Path] = None,
) -> Dict:
    """Detect HII region boundary with optional bootstrap uncertainty.

    Parameters
    ----------
    data : np.ndarray
        2D image data.
    xc, yc : float
        Center coordinates (pixels).
    rmax : float
        Maximum search radius (pixels).
    error_map : np.ndarray or None
        Per-pixel 1sigma measurement uncertainty.  None -> skip bootstrap.
    clean_image : np.ndarray or None
        Noise-free reference image for bootstrap scenario A.
    config_path : str or None
        Path to YAML config file.  None -> auto-detect hii_detection_config.yaml.
    n_bootstrap : int or None
        Bootstrap iterations.  None -> use config/default.  0 -> skip.
    seed : int
        Random seed for reproducibility.
    method : str or None
        'scan' or 'iterative'.
    smoothing_fwhm, cost_map_smoothing_sigma, gradient_smoothing_sigma,
    boundary_smoothing_sigma, rmin_start_ratio, rmin_min_pixels,
    rmax_limit_ratio, angular_snr_weighting, angular_snr_sigma,
    coherence_penalty_weight, coherence_sigma :
        See find_stable_boundary.
    stable_window, stable_threshold, n_steps :
        Scan-method specific.  See find_stable_boundary_by_scan.
    show_diagnostics : bool
        If True, generate diagnostic plots.
    showfile : Path or None
        Diagnostic plot output path.

    Returns
    -------
    dict with keys:
        boundary_radii, boundary_angles : polar coordinates (360,)
        boundary_x, boundary_y : Cartesian coordinates (360,)
        xc, yc, rmax : center and search radius
        boundary_uncertainty : per-angle 1sigma (360,) or None
        bootstrap_boundaries : all bootstrap boundaries (N, 360)
        mean_uncertainty, max_uncertainty : float or None
        scenario : 'A' | 'B' | 'C'
        correction_applied : bool
        rmin_final : float
        boundary_type : str
        has_stable_region : bool
        fallback_used : bool
        stable_regions_info : list of dict
        _stable_result, _bootstrap_result : internal diagnostics
    """
    # Collect explicit kwargs for resolution (only algorithm params)
    explicit = {
        'method': method,
        'smoothing_fwhm': smoothing_fwhm,
        'cost_map_smoothing_sigma': cost_map_smoothing_sigma,
        'gradient_smoothing_sigma': gradient_smoothing_sigma,
        'boundary_smoothing_sigma': boundary_smoothing_sigma,
        'rmin_start_ratio': rmin_start_ratio,
        'rmin_min_pixels': rmin_min_pixels,
        'rmax_limit_ratio': rmax_limit_ratio,
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
        'save_h5_path': save_h5_path,
    }

    config_file = Path(config_path) if config_path else None
    params = _resolve_params(config_file, explicit)

    # Extract n_bootstrap from resolved params (not passed to find_stable_boundary)
    _n_bootstrap = params.pop('n_bootstrap')
    _n_workers = params.pop('n_workers', 1)

    # Extract HDF5 save path (not passed to find_stable_boundary)
    _save_h5_path = params.pop('save_h5_path', None)

    # Extract outlier removal params (not passed to find_stable_boundary)
    _outlier_removal = params.pop('outlier_removal', True)
    _outlier_k = params.pop('outlier_k', 3.0)

    # Remove elliptical-only params (not used in circular mode)
    params.pop('f_min_start_ratio', None)
    params.pop('f_min_min_pixels', None)
    params.pop('f_max_limit_ratio', None)

    logger.info("HII boundary detection — center=(%.0f, %.0f), rmax=%.0f, n_bootstrap=%d",
                  xc, yc, rmax, _n_bootstrap)

    # Step 1: Stable boundary detection
    if showfile is not None:
        Path(showfile).parent.mkdir(parents=True, exist_ok=True)
    stable_result = find_stable_boundary(
        data=data,
        xc=xc,
        yc=yc,
        rmax=rmax,
        error_map=error_map,
        show_diagnostics=show_diagnostics,
        showfile=showfile,
        **params,
    )

    logger.info("Stable boundary: mean=%.1f px, type=%s, rmin=%.1f",
                  float(np.mean(stable_result['boundary_radii'])),
                  stable_result.get('boundary_type', 'iterative'),
                  stable_result.get('rmin_final', 0))

    # Step 2: Bootstrap uncertainty (if applicable)
    bootstrap_result = None
    if error_map is not None and _n_bootstrap > 0:
        bootstrap_result = bootstrap_boundary_uncertainty(
            data=data,
            xc=xc,
            yc=yc,
            rmax=rmax,
            n_bootstrap=_n_bootstrap,
            n_workers=_n_workers,
            error_map=error_map,
            clean_image=clean_image,
            seed=seed,
            **params,
        )

    # Step 3: Build unified result
    result = {
        # Polar
        'boundary_radii': stable_result['boundary_radii'],
        'boundary_angles': stable_result['boundary_angles'],
        # Cartesian
        'boundary_x': stable_result.get('boundary_x'),
        'boundary_y': stable_result.get('boundary_y'),
        # Center
        'xc': xc,
        'yc': yc,
        'rmax': rmax,
        # Stable boundary metadata
        'rmin_final': stable_result.get('rmin_final'),
        'boundary_type': stable_result.get('boundary_type', 'iterative'),
        'has_stable_region': stable_result.get('has_stable_region', False),
        'fallback_used': stable_result.get('fallback_used', False),
        'stable_regions_info': stable_result.get('stable_regions_info', []),
        # Per-angle boundary reliability
        'boundary_confidence': stable_result.get('boundary_confidence'),
        'boundary_valid': stable_result.get('boundary_valid'),
        'confidence_threshold': stable_result.get('confidence_threshold'),
        # Internal
        '_stable_result': stable_result,
        '_bootstrap_result': bootstrap_result,
    }

    # Merge bootstrap info
    if bootstrap_result is not None:
        result['bootstrap_boundaries'] = bootstrap_result['bootstrap_boundaries']
        result['bootstrap_has_stable'] = bootstrap_result.get('bootstrap_has_stable')
        result['scenario'] = bootstrap_result['scenario']
        result['correction_applied'] = bootstrap_result['correction_applied']

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
        result['boundary_x'] = xc + median_radii * np.cos(result['boundary_angles'])
        result['boundary_y'] = yc + median_radii * np.sin(result['boundary_angles'])

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
        result['bootstrap_boundaries'] = np.empty((0, len(stable_result['boundary_radii'])))
        result['mean_uncertainty'] = None
        result['max_uncertainty'] = None
        result['scenario'] = 'C' if error_map is None else 'no_bootstrap'
        result['correction_applied'] = False

    # Final result summary
    mean_r = float(np.mean(result['boundary_radii']))
    mu = result.get('mean_uncertainty')
    if mu is not None:
        logger.info("Result: boundary at %.1f ± %.2f px", mean_r, mu)
    else:
        logger.info("Result: boundary at %.1f px (no uncertainty)", mean_r)

    # Save resolved params for reporting (shows actual values used)
    # Add n_bootstrap back for reporting (it was popped before find_stable_boundary)
    report_params = dict(params)
    report_params['n_bootstrap'] = _n_bootstrap
    result['_resolved_params'] = report_params

    # Optionally save result as HDF5
    if _save_h5_path:
        _save_result_h5(result, Path(_save_h5_path))

    return result
