"""Standard test functions for HII boundary detection.

Each test function receives detect_hii_boundary as a parameter so the
test layer is decoupled from the concrete implementation.
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import logging
from astropy.io import fits

logger = logging.getLogger("hii_boundary.test")


def _get_signal_amplitude(model: 'ModelImage', config: dict) -> float:
    """根据模型类型返回信号幅度（噪声基准）。

    Crater 类模型：crater_b - crater_a
    Gaussian Ring 模型：ring_b_max
    """
    if "Gaussian Ring" in model.name:
        return config.get("ring_b_max", 1.0)
    return config["crater_b"] - config["crater_a"]


class TestLoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter that prepends test context to each message.

    Usage::

        log = TestLoggerAdapter(logger, {"test": "Sharp Step (noise σ=10%)"})
        log.info("processing...")  # -> "[Sharp Step (noise σ=10%)] processing..."
    """

    def process(self, msg, kwargs):
        ctx = ", ".join(f"{k}={v}" for k, v in self.extra.items())
        return f"[{ctx}] {msg}", kwargs


def _shift_image(image: np.ndarray) -> np.ndarray:
    """Shift image so min > 0, avoiding extract_circle_boundary zero→NaN conversion."""
    img_min = np.min(image)
    if img_min <= 0:
        return image + (0.01 - img_min)
    return image


def _compute_argmax_metrics(result: dict, r_expected, detect_rising_edge: bool = True) -> Optional[dict]:
    """Extract gradients from _extract_result, compute argmax MRE for both gradient versions.

    Returns dict with keys "final" (dv_map) and "raw" (dv_map_raw), each a
    compare_boundaries dict. Returns None if gradients are unavailable.
    """
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


from test_analysis import compare_boundaries, compute_expected_boundary
from test_models import ModelImage
from test_diagnostics import (
    plot_boundary_overlay,
    plot_error_vs_angle,
    plot_model_summary,
    plot_polar_pipeline,
    plot_boundary_shift_curves,
)


@dataclass
class TestResult:
    """Structured result from a single test run."""
    name: str
    model: ModelImage
    test_type: str               # "baseline" | "noise" | "center" | "rmax"
    params: dict
    result: dict                 # raw detect_hii_boundary return
    error_metrics: dict
    bootstrap_metrics: Optional[dict]
    plots: dict[str, Path]
    passed: bool
    expected_boundary: Optional[object] = None  # ExpectedBoundary from test_analysis
    sigma_r: float = 0.0
    argmax_metrics: Optional[dict] = None
    # {"final": {"mre": ..., "rms": ..., ...},
    #  "raw":   {"mre": ..., "rms": ..., ...}}


# ---- Plot helpers ----

def _make_diagnostic_plots(result, expected, expected_boundary, image, model,
                           plots_dir, prefix, title, argmax_em=None):
    """Generate all diagnostic plots for a single detection result."""
    plots = {}
    sigma = result.get("boundary_uncertainty")
    angles = result["boundary_angles"]
    r_det = result["boundary_radii"]
    xc = result["xc"]
    yc = result["yc"]
    # Optional per-angle radii for error computation (e.g. center offset test
    # where detected radii are in a shifted frame and must be transformed to
    # model-center frame for fair comparison with expected radius).


    # Model summary (image + radial profile) — displayed first
    if expected_boundary is not None:
        plots["model_summary"] = str(plot_model_summary(
            image, model.xc, model.yc, expected,
            expected_boundary, model.params,
            save_path=plots_dir / f"{prefix}_model_summary.png",
            title=title,
            detect_rising_edge=result.get("detect_rising_edge", True),
        ))

    # Polar pipeline visualization
    stable = result.get("_stable_result", {})
    extr = stable.get("_extract_result", {}) if isinstance(stable, dict) else {}
    if "polar_smooth" in extr:
        polar_roi = extr.get("polar_roi", extr.get("polar_image_full"))
        polar_smooth = extr.get("polar_smooth", polar_roi)
        score_map = extr.get("score_map", np.zeros((1, 1)))
        dv_map = extr.get("dv_map", np.zeros((1, 1)))
        cost_map = extr.get("cost_map", np.zeros((1, 1)))
        cost_map_radii = extr.get("cost_map_radii",
                                  np.linspace(0, 1, cost_map.shape[0]))
        rr_roi = extr.get("rr_roi", np.arange(cost_map.shape[0]))
        # Use the scan's own boundary (not bootstrap median) for pipeline
        scan_r = stable.get("boundary_radii", r_det)
        plots["polar_pipeline"] = str(plot_polar_pipeline(
            polar_roi, polar_smooth, score_map, dv_map,
            cost_map, cost_map_radii,
            scan_r, angles, rr_roi,
            save_path=plots_dir / f"{prefix}_pipeline.png",
            title=title,
        ))

    # Stable boundary diagnostic
    stable_diag = plots_dir / f"{prefix}_algo_diag.png"
    if stable_diag.exists():
        plots["stable_diag"] = str(stable_diag)

    # Boundary overlay with error band (includes uncertainty panel when sigma given)
    _stable = result.get("_stable_result") or {}
    _extract = _stable.get("_extract_result") or {}
    plots["overlay"] = plot_boundary_overlay(
        image, xc, yc, r_det, angles,
        expected, plots_dir / f"{prefix}_overlay.png",
        title=title, boundary_uncertainty=sigma,
        expected_xc=model.xc, expected_yc=model.yc,
        polar_image=_extract.get("polar_roi"),
        polar_rr=_extract.get("rr_roi"),
    )

    # Build argmax curves for error_vs_angle overlay
    argmax_curves = None
    if argmax_em is not None:
        from test_argmax_baseline import compute_argmax_boundary
        argmax_curves = {}
        stable = result.get("_stable_result", {})
        extr = stable.get("_extract_result", {}) if isinstance(stable, dict) else {}
        dv_map_f = extr.get("dv_map")
        dv_map_r = extr.get("dv_map_raw")
        rr_roi = extr.get("rr_roi")
        if dv_map_f is not None and rr_roi is not None:
            rmin_val = result.get("rmin_final", 0)
            rmax_val = result["rmax"]
            dre = result.get("detect_rising_edge", True)
            r_final = compute_argmax_boundary(dv_map_f, rr_roi, rmin_val, rmax_val, dre)
            argmax_curves["argmax_final"] = r_final
            if dv_map_r is not None:
                r_raw = compute_argmax_boundary(dv_map_r, rr_roi, rmin_val, rmax_val, dre)
                argmax_curves["argmax_raw"] = r_raw

    # Error vs angle
    # For center offset tests, use shifted-center comparison
    if "_r_expected_shifted" in result:
        # Center offset test: compare in shifted-center coordinate system
        angles_err = angles  # use shifted-center angles
        r_det_err = r_det  # use shifted-center detected boundary
        r_expected = result["_r_expected_shifted"]
        plots["error_vs_angle"] = plot_error_vs_angle(
            angles_err, r_det_err, r_expected, plots_dir / f"{prefix}_error.png",
            argmax_curves=argmax_curves,
        )
    else:
        # Other tests: compare in original coordinate system
        plots["error_vs_angle"] = plot_error_vs_angle(
            angles, r_det, expected, plots_dir / f"{prefix}_error.png",
            argmax_curves=argmax_curves,
        )

    return plots


def _compute_expected_boundary(model, config):
    """Compute the analytically expected boundary for a model."""
    sigma_r = config.get("smoothing_fwhm", 2.0) / 2.355
    rmin = max(model.rmax * 0.05, 2.0)
    return compute_expected_boundary(
        model.radial_func, rmin, model.rmax, sigma_r,
        detect_rising_edge=config.get("detect_rising_edge", True),
    )


def transform_expected_boundary_to_shifted_center(
    expected_boundary: float | np.ndarray,
    model_xc: float, model_yc: float,
    shifted_xc: float, shifted_yc: float,
    angles: np.ndarray,
    model_type: str = "circle",
) -> np.ndarray:
    """将期望边界转换到偏移中心坐标系。

    Parameters
    ----------
    expected_boundary : float or array
        圆形半径或边界点数组。
    model_xc, model_yc : float
        真实模型中心坐标。
    shifted_xc, shifted_yc : float
        偏移中心坐标。
    angles : np.ndarray
        检测边界的极角（弧度）。
    model_type : str
        "circle" 或 "arbitrary"。

    Returns
    -------
    r_expected_shifted : np.ndarray
        期望边界在偏移中心坐标系下的半径。
    """
    if model_type == "circle":
        # 数值转换：在模型坐标系下生成期望边界点，转换到偏移中心坐标系
        R0 = expected_boundary
        n_samples = 10000  # 高密度采样以保证精度
        theta_true = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)

        # 模型坐标系下的边界点
        x_true = model_xc + R0 * np.cos(theta_true)
        y_true = model_yc + R0 * np.sin(theta_true)

        # 转换到偏移中心极坐标
        r_shifted = np.sqrt((x_true - shifted_xc)**2 + (y_true - shifted_yc)**2)
        theta_shifted = (np.arctan2(y_true - shifted_yc, x_true - shifted_xc) + 2 * np.pi) % (2 * np.pi)

        # 按 theta_shifted 排序以便插值
        sort_idx = np.argsort(theta_shifted)
        theta_shifted_sorted = theta_shifted[sort_idx]
        r_shifted_sorted = r_shifted[sort_idx]

        # 处理周期性边界：在两端添加点
        theta_extended = np.concatenate([
            [theta_shifted_sorted[0] - 2 * np.pi],
            theta_shifted_sorted,
            [theta_shifted_sorted[-1] + 2 * np.pi],
        ])
        r_extended = np.concatenate([
            [r_shifted_sorted[0]],
            r_shifted_sorted,
            [r_shifted_sorted[-1]],
        ])

        # 插值到检测边界的角度采样
        from scipy.interpolate import interp1d
        interp_func = interp1d(theta_extended, r_extended, kind='linear')
        r_expected_shifted = interp_func(angles % (2 * np.pi))

        return r_expected_shifted
    else:
        # 数值转换 + 插值（适用于任意形状）
        # TODO: 实现数值转换逻辑
        raise NotImplementedError("Non-circle models not yet supported")


# ---- Test functions ----

def test_baseline(
    detect_fn: Callable,
    model: ModelImage,
    plots_dir: Path,
    config: dict,
    seed: int = 42,
) -> TestResult:
    """Clean-image detection without bootstrap (no error_map, no noise)."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    exp_boundary = _compute_expected_boundary(model, config)

    result = detect_fn(
        data=_shift_image(model.clean_image),
        xc=model.xc, yc=model.yc,
        rmax=model.rmax,
        n_bootstrap=0,  # 基准测试：无噪声，不需要误差估计
        seed=seed,
        show_diagnostics=True,
        showfile=plots_dir / "baseline_algo_diag.png",
    )

    em = compare_boundaries(result["boundary_radii"], exp_boundary.radius)
    detect_rising = config.get("detect_rising_edge", True)
    argmax_em = _compute_argmax_metrics(result, exp_boundary.radius, detect_rising)
    plots = _make_diagnostic_plots(
        result, exp_boundary.radius, exp_boundary, model.clean_image,
        model, plots_dir, "baseline", model.name,
        argmax_em=argmax_em,
    )

    return TestResult(
        name=model.name,
        model=model,
        test_type="baseline",
        params={"noise_level": 0.0},
        result=result,
        error_metrics=em,
        bootstrap_metrics=None,  # 基准测试不做 bootstrap
        plots=plots,
        passed=abs(em["mre"]) < 2.0,
        expected_boundary=exp_boundary,
        sigma_r=config.get("smoothing_fwhm", 2.0) / 2.355,
        argmax_metrics=argmax_em,
    )


def test_noise(
    detect_fn: Callable,
    model: ModelImage,
    noise_levels: list[float],
    plots_dir: Path,
    config: dict,
    seed: int = 42,
) -> list[TestResult]:
    """Bootstrap scenario A across noise levels.  One result per level."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    results = []
    shape = model.clean_image.shape
    sigma_err = _get_signal_amplitude(model, config)
    rng = np.random.default_rng(seed)
    exp_boundary = _compute_expected_boundary(model, config)

    for level in noise_levels:
        pct = int(level * 100)
        noise_sigma = level * sigma_err
        noisy = model.clean_image + rng.normal(0, noise_sigma, shape)
        error_map = np.full(shape, noise_sigma, dtype=np.float64)

        result = detect_fn(
            data=_shift_image(noisy),
            xc=model.xc, yc=model.yc,
            rmax=model.rmax,
            error_map=error_map,
            clean_image=model.clean_image,
            n_bootstrap=config.get("bootstrap_n", 100),
            seed=seed + pct,
            show_diagnostics=True,
            showfile=plots_dir / f"noise_{pct}pct_algo_diag.png",
        )

        em = compare_boundaries(result["boundary_radii"], exp_boundary.radius)
        detect_rising = config.get("detect_rising_edge", True)
        argmax_em = _compute_argmax_metrics(result, exp_boundary.radius, detect_rising)
        title = f"{model.name} (σ={pct}%)"
        plots = _make_diagnostic_plots(
            result, exp_boundary.radius, exp_boundary, noisy,
            model, plots_dir, f"noise_{pct}pct", title,
            argmax_em=argmax_em,
        )

        results.append(TestResult(
            name=title,
            model=model,
            test_type="noise",
            params={"noise_level": level, "noise_sigma": noise_sigma},
            result=result,
            error_metrics=em,
            bootstrap_metrics={
                "mean_uncertainty": result.get("mean_uncertainty"),
                "max_uncertainty": result.get("max_uncertainty"),
                "scenario": result.get("scenario"),
            } if result.get("boundary_uncertainty") is not None else None,
            plots=plots,
            passed=abs(em["mre"]) < 5.0,
            expected_boundary=exp_boundary,
            sigma_r=config.get("smoothing_fwhm", 2.0) / 2.355,
            argmax_metrics=argmax_em,
        ))

    return results


def test_center_offset(
    detect_fn: Callable,
    model: ModelImage,
    offsets: list[float],
    plots_dir: Path,
    config: dict,
    noise_level: float = 0.30,
    seed: int = 42,
) -> list[TestResult]:
    """Random-direction center offset sensitivity."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    results = []
    shape = model.clean_image.shape
    sigma_err = _get_signal_amplitude(model, config)
    noise_sigma = noise_level * sigma_err
    rng = np.random.default_rng(seed)
    error_map = np.full(shape, noise_sigma, dtype=np.float64)
    exp_boundary = _compute_expected_boundary(model, config)

    for offset_frac in offsets:
        direction = rng.uniform(0, 2 * np.pi)
        dx = offset_frac * model.expected_radius * np.cos(direction)
        dy = offset_frac * model.expected_radius * np.sin(direction)
        cx = model.xc + dx
        cy = model.yc + dy

        noisy = model.clean_image + rng.normal(0, noise_sigma, shape)

        result = detect_fn(
            data=_shift_image(noisy),
            xc=cx, yc=cy,
            rmax=model.rmax,
            error_map=error_map,
            clean_image=model.clean_image,
            n_bootstrap=config.get("bootstrap_n", 100),
            seed=seed,
            show_diagnostics=True,
            showfile=plots_dir / f"center_{int(offset_frac*100)}pct_algo_diag.png",
        )

        # Transform expected boundary to shifted-center coordinate system
        # This avoids quantization and interpolation errors from coordinate transformation
        r_expected_shifted = transform_expected_boundary_to_shifted_center(
            exp_boundary.radius, model.xc, model.yc, cx, cy,
            result["boundary_angles"],
        )
        result["_r_expected_shifted"] = r_expected_shifted

        # Compute error in shifted-center coordinate system
        em = compare_boundaries(result["boundary_radii"], r_expected_shifted)
        detect_rising = config.get("detect_rising_edge", True)
        argmax_em = _compute_argmax_metrics(result, r_expected_shifted, detect_rising)
        pct = int(offset_frac * 100)
        title = f"{model.name} (offset={pct}%)"
        plots = _make_diagnostic_plots(
            result, exp_boundary.radius, exp_boundary, noisy,
            model, plots_dir, f"center_{pct}pct", title,
            argmax_em=argmax_em,
        )

        results.append(TestResult(
            name=title,
            model=model,
            test_type="center",
            params={"offset_frac": offset_frac, "dx": dx, "dy": dy, "noise_level": noise_level},
            result=result,
            error_metrics=em,
            bootstrap_metrics={
                "mean_uncertainty": result.get("mean_uncertainty"),
                "max_uncertainty": result.get("max_uncertainty"),
                "scenario": result.get("scenario"),
            } if result.get("boundary_uncertainty") is not None else None,
            plots=plots,
            passed=True,  # sensitivity tests report metrics, not pass/fail
            expected_boundary=exp_boundary,
            sigma_r=config.get("smoothing_fwhm", 2.0) / 2.355,
            argmax_metrics=argmax_em,
        ))

    # Shift curves — all offsets overlaid
    # Transform detected boundaries to true-center coordinates with uniform sampling
    if len(offsets) >= 2:
        all_bounds = {}
        for i, offset_frac in enumerate(offsets):
            pct = int(offset_frac * 100)
            label = f"offset {pct}%"

            # Get shifted-center boundary
            r_shifted = results[i].result["boundary_radii"]
            theta_shifted = results[i].result["boundary_angles"]

            # Transform to true-center Cartesian
            cx = results[i].result["xc"]
            cy = results[i].result["yc"]
            x = cx + r_shifted * np.cos(theta_shifted)
            y = cy + r_shifted * np.sin(theta_shifted)

            # Convert to true-center polar
            r_true = np.sqrt((x - model.xc)**2 + (y - model.yc)**2)
            theta_true = (np.arctan2(y - model.yc, x - model.xc) + 2 * np.pi) % (2 * np.pi)

            # Interpolate to uniform angle sampling
            sort_idx = np.argsort(theta_true)
            theta_sorted = theta_true[sort_idx]
            r_sorted = r_true[sort_idx]

            theta_extended = np.concatenate([
                [theta_sorted[0] - 2 * np.pi],
                theta_sorted,
                [theta_sorted[-1] + 2 * np.pi],
            ])
            r_extended = np.concatenate([
                [r_sorted[0]],
                r_sorted,
                [r_sorted[-1]],
            ])
            from scipy.interpolate import interp1d
            interp_func = interp1d(theta_extended, r_extended, kind='linear')

            n_angles = len(theta_shifted)
            theta_uniform = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
            r_uniform = np.round(interp_func(theta_uniform), 2)

            all_bounds[label] = (theta_uniform, r_uniform)

        # Also add expected boundary (circle in true-center coordinates)
        theta_expected = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        r_expected = np.full(n_angles, exp_boundary.radius)
        all_bounds["expected"] = (theta_expected, r_expected)

        sc_path = plot_boundary_shift_curves(
            theta_uniform,  # default angles (ignored for tuples)
            all_bounds,
            plots_dir / "center_shift_curves.png",
            title=f"{model.name} — Center Offset Sensitivity",
        )
        for r in results:
            r.plots["shift_curves"] = str(sc_path)

    return results


def test_rmax_sensitivity(
    detect_fn: Callable,
    model: ModelImage,
    rmax_values: list[float],
    plots_dir: Path,
    config: dict,
    noise_level: float = 0.30,
    seed: int = 42,
) -> list[TestResult]:
    """Rmax (search radius) sensitivity."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    results = []
    shape = model.clean_image.shape
    sigma_err = _get_signal_amplitude(model, config)
    noise_sigma = noise_level * sigma_err
    rng = np.random.default_rng(seed)
    error_map = np.full(shape, noise_sigma, dtype=np.float64)

    noisy = model.clean_image + rng.normal(0, noise_sigma, shape)
    exp_boundary = _compute_expected_boundary(model, config)

    for rmax_val in rmax_values:
        result = detect_fn(
            data=_shift_image(noisy.copy()),
            xc=model.xc, yc=model.yc,
            rmax=rmax_val,
            error_map=error_map,
            clean_image=model.clean_image,
            n_bootstrap=config.get("bootstrap_n", 100),
            seed=seed,
            show_diagnostics=True,
            showfile=plots_dir / f"rmax_{rmax_val:.0f}_algo_diag.png",
        )

        em = compare_boundaries(result["boundary_radii"], exp_boundary.radius)
        detect_rising = config.get("detect_rising_edge", True)
        argmax_em = _compute_argmax_metrics(result, exp_boundary.radius, detect_rising)
        title = f"{model.name} (rmax={rmax_val:.0f})"
        plots = _make_diagnostic_plots(
            result, exp_boundary.radius, exp_boundary, noisy,
            model, plots_dir, f"rmax_{rmax_val:.0f}", title,
            argmax_em=argmax_em,
        )

        results.append(TestResult(
            name=title,
            model=model,
            test_type="rmax",
            params={"rmax": rmax_val, "noise_level": noise_level},
            result=result,
            error_metrics=em,
            bootstrap_metrics={
                "mean_uncertainty": result.get("mean_uncertainty"),
                "max_uncertainty": result.get("max_uncertainty"),
                "scenario": result.get("scenario"),
            } if result.get("boundary_uncertainty") is not None else None,
            plots=plots,
            passed=True,
            expected_boundary=exp_boundary,
            sigma_r=config.get("smoothing_fwhm", 2.0) / 2.355,
            argmax_metrics=argmax_em,
        ))

    # Shift curves — all rmax values overlaid
    if len(rmax_values) >= 2:
        all_bounds = {}
        for i, rv in enumerate(rmax_values):
            label = f"rmax={rv:.0f}"
            all_bounds[label] = results[i].result["boundary_radii"]
        sc_path = plot_boundary_shift_curves(
            results[0].result["boundary_angles"],
            all_bounds,
            plots_dir / "rmax_shift_curves.png",
            title=f"{model.name} — Rmax Sensitivity",
        )
        for r_item in results:
            r_item.plots["shift_curves"] = str(sc_path)

    return results
