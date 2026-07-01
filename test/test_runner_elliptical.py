"""Test functions for elliptical prior boundary detection."""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from test_analysis import compare_boundaries
from test_models import ModelImage
from test_diagnostics import (
    plot_boundary_overlay,
    plot_error_vs_angle,
)


@dataclass
class TestResult:
    name: str
    model: ModelImage
    test_type: str
    params: dict
    result: dict
    error_metrics: dict
    bootstrap_metrics: Optional[dict] = None
    plots: dict = field(default_factory=dict)
    passed: bool = True
    expected_boundary: Optional[object] = None
    sigma_r: Optional[float] = None
    argmax_metrics: Optional[dict] = None


def _get_signal_amplitude(model, config):
    if "Gaussian Ring" in model.name:
        return config.get("ring_b_max", 1.0)
    return config["crater_b"] - config["crater_a"]


def _shift_image(image):
    img_min = np.min(image)
    if img_min <= 0:
        return image + (0.01 - img_min)
    return image


def _compute_argmax_metrics(result, r_expected, detect_rising_edge=True):
    """Extract gradients from _extract_result, compute argmax MRE."""
    stable = result.get("_stable_result", {})
    extr = stable.get("_extract_result", {}) if isinstance(stable, dict) else {}
    dv_map = extr.get("dv_map")
    dv_map_raw = extr.get("dv_map_raw")
    rr_roi = extr.get("rr_roi")
    if dv_map is None or rr_roi is None:
        return None

    from test_argmax_baseline import compute_argmax_boundary
    rmin = result.get("rmin_final", 0)
    rmax_val = result["rmax"]

    metrics = {}
    for key, grad in [("final", dv_map), ("raw", dv_map_raw)]:
        if grad is None:
            continue
        radii = compute_argmax_boundary(grad, rr_roi, rmin, rmax_val, detect_rising_edge)
        metrics[key] = compare_boundaries(radii, r_expected)
    return metrics if metrics else None


def _compute_expected_boundary_elliptical(model, config):
    """Compute expected boundary for elliptical model.

    For an elliptical model, the expected boundary at each angle
    follows r_ell(theta) = a*b / sqrt((b*cos(theta-phi))^2 + (a*sin(theta-phi))^2)
    """
    a = model.params.get('a', model.expected_radius)
    b = model.params.get('b', model.expected_radius)
    phi = model.params.get('phi', 0.0)

    n_angles = 360
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    cos_t = np.cos(angles - phi)
    sin_t = np.sin(angles - phi)
    r_expected = a * b / np.sqrt((b * cos_t) ** 2 + (a * sin_t) ** 2)

    class ExpectedBoundary:
        pass
    eb = ExpectedBoundary()
    eb.radius = r_expected
    return eb


def _make_diagnostic_plots(result, expected_radii, exp_boundary, image,
                           model, out_dir, prefix, title):
    """Generate diagnostic plots for elliptical tests."""
    plots = {}
    try:
        overlay = out_dir / f"{prefix}_overlay.png"
        plot_boundary_overlay(
            image, result['boundary_x'], result['boundary_y'],
            result.get('boundary_uncertainty'),
            result['xc'], result['yc'],
            title=title, save_path=overlay,
        )
        plots['overlay'] = overlay
    except Exception:
        pass
    try:
        err = out_dir / f"{prefix}_error.png"
        plot_error_vs_angle(
            result['boundary_radii'], expected_radii,
            result.get('boundary_uncertainty'),
            title=title, save_path=err,
        )
        plots['error_vs_angle'] = err
    except Exception:
        pass
    return plots


def test_baseline_elliptical(detect_fn, model, plots_dir, config, seed=42):
    """Clean-image detection for elliptical prior."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    exp_boundary = _compute_expected_boundary_elliptical(model, config)

    ell = model.params
    result = detect_fn(
        data=_shift_image(model.clean_image),
        xc=model.xc, yc=model.yc,
        a=ell.get('a'), b=ell.get('b'), phi=ell.get('phi', 0.0),
        n_bootstrap=0, seed=seed,
        show_diagnostics=True,
        showfile=plots_dir / "baseline_algo_diag.png",
    )
    em = compare_boundaries(result['boundary_radii'], exp_boundary.radius)
    detect_rising = config.get("detect_rising_edge", True)
    argmax_em = _compute_argmax_metrics(result, exp_boundary.radius, detect_rising)
    plots = _make_diagnostic_plots(
        result, exp_boundary.radius, exp_boundary,
        model.clean_image, model, plots_dir, "baseline", model.name,
    )
    return TestResult(
        name=model.name, model=model, test_type="baseline",
        params={"noise_level": 0.0}, result=result,
        error_metrics=em, bootstrap_metrics=None,
        plots=plots, passed=abs(em["mre"]) < 2.0,
        expected_boundary=exp_boundary,
        argmax_metrics=argmax_em,
    )


def test_noise_elliptical(detect_fn, model, noise_levels, plots_dir, config, seed=42):
    """Noise sweep for elliptical prior."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    results = []
    shape = model.clean_image.shape
    sigma_err = _get_signal_amplitude(model, config)
    rng = np.random.default_rng(seed)
    exp_boundary = _compute_expected_boundary_elliptical(model, config)
    ell = model.params

    for level in noise_levels:
        pct = int(level * 100)
        noise_sigma = level * sigma_err
        noisy = model.clean_image + rng.normal(0, noise_sigma, shape)
        error_map = np.full(shape, noise_sigma, dtype=np.float64)

        result = detect_fn(
            data=_shift_image(noisy),
            xc=model.xc, yc=model.yc,
            a=ell.get('a'), b=ell.get('b'), phi=ell.get('phi', 0.0),
            error_map=error_map, clean_image=model.clean_image,
            n_bootstrap=config.get("bootstrap_n", 100),
            seed=seed + pct,
            show_diagnostics=True,
            showfile=plots_dir / f"noise_{pct}pct_algo_diag.png",
        )
        em = compare_boundaries(result['boundary_radii'], exp_boundary.radius)
        detect_rising = config.get("detect_rising_edge", True)
        argmax_em = _compute_argmax_metrics(result, exp_boundary.radius, detect_rising)
        title = f"{model.name} (sigma={pct}%)"
        plots = _make_diagnostic_plots(
            result, exp_boundary.radius, exp_boundary, noisy,
            model, plots_dir, f"noise_{pct}pct", title,
        )
        bs = None
        if result.get('boundary_uncertainty') is not None:
            bs = {
                'mean_uncertainty': result.get('mean_uncertainty'),
                'max_uncertainty': result.get('max_uncertainty'),
                'scenario': result.get('scenario'),
            }
        results.append(TestResult(
            name=title, model=model, test_type="noise",
            params={"noise_level": level, "noise_sigma": noise_sigma},
            result=result, error_metrics=em, bootstrap_metrics=bs,
            plots=plots, passed=abs(em["mre"]) < 5.0,
            expected_boundary=exp_boundary,
            argmax_metrics=argmax_em,
        ))
    return results


def test_center_offset_elliptical(detect_fn, model, offsets, plots_dir, config,
                                   noise_level=0.30, seed=42):
    """Center offset sensitivity for elliptical prior."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    results = []
    shape = model.clean_image.shape
    sigma_err = _get_signal_amplitude(model, config)
    noise_sigma = noise_level * sigma_err
    rng = np.random.default_rng(seed)
    error_map = np.full(shape, noise_sigma, dtype=np.float64)
    exp_boundary = _compute_expected_boundary_elliptical(model, config)
    ell = model.params

    for offset_frac in offsets:
        direction = rng.uniform(0, 2 * np.pi)
        dx = offset_frac * ell['a'] * np.cos(direction)
        dy = offset_frac * ell['a'] * np.sin(direction)
        cx = model.xc + dx
        cy = model.yc + dy
        noisy = model.clean_image + rng.normal(0, noise_sigma, shape)

        result = detect_fn(
            data=_shift_image(noisy),
            xc=cx, yc=cy,
            a=ell.get('a'), b=ell.get('b'), phi=ell.get('phi', 0.0),
            error_map=error_map, clean_image=model.clean_image,
            n_bootstrap=config.get("bootstrap_n", 100),
            seed=seed,
            show_diagnostics=True,
            showfile=plots_dir / f"center_{int(offset_frac*100)}pct_algo_diag.png",
        )
        em = compare_boundaries(result['boundary_radii'], exp_boundary.radius)
        detect_rising = config.get("detect_rising_edge", True)
        argmax_em = _compute_argmax_metrics(result, exp_boundary.radius, detect_rising)
        pct = int(offset_frac * 100)
        title = f"{model.name} (offset={pct}%)"
        plots = _make_diagnostic_plots(
            result, exp_boundary.radius, exp_boundary, noisy,
            model, plots_dir, f"center_{pct}pct", title,
        )
        bs = None
        if result.get('boundary_uncertainty') is not None:
            bs = {
                'mean_uncertainty': result.get('mean_uncertainty'),
                'max_uncertainty': result.get('max_uncertainty'),
                'scenario': result.get('scenario'),
            }
        results.append(TestResult(
            name=title, model=model, test_type="center",
            params={"offset_frac": offset_frac, "dx": dx, "dy": dy,
                    "noise_level": noise_level},
            result=result, error_metrics=em, bootstrap_metrics=bs,
            plots=plots, passed=True,
            expected_boundary=exp_boundary,
            argmax_metrics=argmax_em,
        ))
    return results


def test_f_max_sensitivity(detect_fn, model, f_max_values, plots_dir, config,
                            noise_level=0.30, seed=42):
    """Scale factor sensitivity for elliptical prior."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    results = []
    shape = model.clean_image.shape
    sigma_err = _get_signal_amplitude(model, config)
    noise_sigma = noise_level * sigma_err
    rng = np.random.default_rng(seed)
    error_map = np.full(shape, noise_sigma, dtype=np.float64)
    exp_boundary = _compute_expected_boundary_elliptical(model, config)
    ell = model.params

    for scale in f_max_values:
        a_scaled = ell['a'] * scale
        b_scaled = ell['b'] * scale
        noisy = model.clean_image + rng.normal(0, noise_sigma, shape)

        result = detect_fn(
            data=_shift_image(noisy),
            xc=model.xc, yc=model.yc,
            a=a_scaled, b=b_scaled, phi=ell.get('phi', 0.0),
            error_map=error_map, clean_image=model.clean_image,
            n_bootstrap=config.get("bootstrap_n", 100),
            seed=seed,
        )
        em = compare_boundaries(result['boundary_radii'], exp_boundary.radius)
        detect_rising = config.get("detect_rising_edge", True)
        argmax_em = _compute_argmax_metrics(result, exp_boundary.radius, detect_rising)
        title = f"{model.name} (f_max={scale:.1f})"
        results.append(TestResult(
            name=title, model=model, test_type="f_max",
            params={"scale": scale, "noise_level": noise_level},
            result=result, error_metrics=em, bootstrap_metrics=None,
            plots={}, passed=True,
            expected_boundary=exp_boundary,
            argmax_metrics=argmax_em,
        ))
    return results
