#!/usr/bin/env python3
"""HII Boundary Detection Test Plan — Orchestrator.

Three-layer architecture:
  test_models.py  — model image generators
  test_runner.py  — standard test functions via detect_hii_boundary
  run_test_plan.py (this file) — orchestrator: model x test -> HTML report
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import logging

from boundary_detection.detect_hii_boundary import detect_hii_boundary
from test_models import build_model_list, ModelImage
from test_runner import (
    test_baseline,
    test_noise,
    test_center_offset,
    test_rmax_sensitivity,
    TestResult,
)
from test_report import build_md_report, build_html_report, write_report_files
from logging_setup import _setup_test_logging

logger = logging.getLogger("hii_boundary.test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(val, digits=3):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def get_git_commit() -> str:
    """Return the current git commit hash (short), or 'N/A' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent,
        )
        return result.stdout.strip() or "N/A"
    except Exception:
        return "N/A"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "test_config.yaml") -> dict:
    """Load test configuration from YAML file.

    Resolution order:
    1. Explicit path as-is (absolute or relative to CWD)
    2. Relative to this source file's directory (for packaged/module execution)
    """
    path_obj = Path(path)
    if not path_obj.exists():
        # Try relative to this file's location
        sibling = Path(__file__).parent / path
        if sibling.exists():
            path_obj = sibling
    with open(path_obj, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("shape", [300, 300])
    cfg["shape"] = tuple(cfg["shape"])
    return cfg


# ---------------------------------------------------------------------------
# Test result -> report dict conversion
# ---------------------------------------------------------------------------

def _test_result_to_report_dict(tr: TestResult) -> dict:
    """Convert a TestResult to the dict format expected by build_md_report."""
    plots = {k: str(v) for k, v in tr.plots.items()}
    bootstrap = tr.bootstrap_metrics or {}
    error_metrics = dict(tr.error_metrics)
    error_metrics.update(bootstrap)
    # For center-offset tests, use the shifted-coordinate expected mean
    # so it's comparable with r_detected_mean (also in shifted coords)
    if tr.test_type == "center" and "_r_expected_shifted" in tr.result:
        expected_radius = float(np.mean(tr.result["_r_expected_shifted"]))
    else:
        expected_radius = tr.expected_boundary.radius if tr.expected_boundary else tr.model.expected_radius
    result = {
        "name": tr.name,
        "model_description": f"{tr.model.name} ({tr.test_type})",
        "params": tr.params,
        "expected_radius": expected_radius,
        "error_metrics": error_metrics,
        "passed": tr.passed,
        "plots": plots,
    }
    # Include expected boundary data for report tables
    if tr.expected_boundary is not None:
        result["expected"] = tr.expected_boundary
    if tr.sigma_r:
        result["sigma_r"] = tr.sigma_r
    # Argmax baseline comparison metrics
    if tr.argmax_metrics is not None:
        result["argmax_metrics"] = tr.argmax_metrics
    return result


# ---------------------------------------------------------------------------
# Core collection: run all 4 test types on all models
# ---------------------------------------------------------------------------

def _collect_all_results(models: list, config: dict, output_dir: Path, args) -> dict:
    """Run all 4 test types on all models."""
    all_results = {}
    rng = np.random.default_rng(config.get("seed", 42))

    for model in models:
        safe_name = (model.name
                     .replace(" ", "_")
                     .replace("(", "").replace(")", "")
                     .replace("=", "").replace(",", ""))
        model_dir = output_dir / safe_name

        # 1. Baseline
        base_dir = model_dir / "baseline"
        try:
            tr = test_baseline(detect_hii_boundary, model, base_dir, config,
                               seed=config.get("seed", 42))
            all_results[tr.name] = tr
            logger.info("  baseline: %s -- %s", tr.name, "PASS" if tr.passed else "FAIL")
        except Exception as e:
            logger.error("baseline ERROR for %s: %s", model.name, e)

        # 2. Noise sweep
        if not getattr(args, 'skip_noise', False):
            noise_dir = model_dir / "noise"
            try:
                noise_results = test_noise(
                    detect_hii_boundary, model,
                    config["noise_levels"], noise_dir, config,
                    seed=config.get("seed", 42),
                )
                for nr in noise_results:
                    all_results[nr.name] = nr
                logger.info("  noise: %d levels", len(noise_results))
            except Exception as e:
                logger.error("noise ERROR for %s: %s", model.name, e)

        # 3. Center offset sensitivity
        if getattr(args, 'sensitivity', True):
            sens_dir = model_dir / "sensitivity"
            try:
                center_dir = sens_dir / "center"
                center_results = test_center_offset(
                    detect_hii_boundary, model,
                    config["center_offsets"], center_dir, config,
                    noise_level=config.get("sensitivity_noise_level", 0.30),
                    seed=int(rng.integers(0, 2**31)),
                )
                for cr in center_results:
                    all_results[cr.name] = cr
                logger.info("  center: %d offsets", len(center_results))
            except Exception as e:
                logger.error("center sensitivity ERROR for %s: %s", model.name, e)

            try:
                rmax_dir = sens_dir / "rmax"
                rmax_results = test_rmax_sensitivity(
                    detect_hii_boundary, model,
                    config["rmax_values"], rmax_dir, config,
                    noise_level=config.get("sensitivity_noise_level", 0.30),
                    seed=int(rng.integers(0, 2**31)),
                )
                for rr in rmax_results:
                    all_results[rr.name] = rr
                logger.info("  rmax: %d values", len(rmax_results))
            except Exception as e:
                logger.error("rmax sensitivity ERROR for %s: %s", model.name, e)

    return all_results


def _build_summary_table(all_results: dict) -> list:
    """Build a per-test summary table."""
    rows = []
    for label, tr in all_results.items():
        if not isinstance(tr, TestResult):
            continue
        em = tr.error_metrics
        boot = tr.bootstrap_metrics or {}
        rows.append({
            "name": label,
            "test_type": tr.test_type,
            "expected": tr.model.expected_radius,
            "mre": em["mre"],
            "rms": em["rms"],
            "max_error": em["max_error"],
            "mean_uncertainty": boot.get("mean_uncertainty"),
            "scenario": boot.get("scenario"),
            "passed": tr.passed,
        })
    return rows


def _build_report_config(config: dict, resolved_params: dict = None) -> dict:
    """Build report-level configuration for build_md_report.

    Uses resolved_params (actual values used by detect_hii_boundary) if available,
    falls back to hii_detection_config.yaml raw values.
    """
    if resolved_params:
        algo_params = resolved_params
    else:
        algo_path = Path("hii_detection_config.yaml")
        if algo_path.exists():
            with open(algo_path, 'r') as f:
                algo_params = yaml.safe_load(f) or {}
        else:
            algo_params = {}

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(),
        "python_version": sys.version.split()[0],
        "shape": str(config.get("shape", "N/A")),
        "algo_params": algo_params,
    }


# ---------------------------------------------------------------------------
# Core runner — callable from Python or CLI
# ---------------------------------------------------------------------------

def run_tests(
    config_path: str = "test_config.yaml",
    output_dir: str | Path | None = None,
    model: str | None = None,
    bootstrap_n: int | None = None,
    seed: int | None = None,
    skip_noise: bool = False,
    sensitivity: bool = True,
    skip_reports: bool = False,
    verbose: bool = True,
) -> dict:
    """Run the full test plan and return structured results.

    Parameters
    ----------
    config_path : str
        Path to test config YAML.
    output_dir : str, Path or None
        Directory for plots and reports.  None = read from config file
        (key ``output_dir``, default ``"test_plots"``).
    model : str or None
        Substring filter for model names (e.g. "Sharp", "Gaussian").
        None = run all models.
    bootstrap_n : int or None
        Override bootstrap_n in config.  None = use config value.
    seed : int or None
        Override random seed.  None = use config value.
    skip_noise : bool
        Skip noise sweep.
    sensitivity : bool
        Run center-offset and rmax sensitivity tests.
    skip_reports : bool
        Skip HTML/MD report generation.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    dict with keys:
        models          — list of model names tested
        all_results     — {label: TestResult} dict
        summary         — list of per-test summary rows
        total / passed / failed — counts
        report_md / report_html — Path or None
        elapsed_sec     — float
    """
    config = load_config(config_path)
    if bootstrap_n is not None:
        config["bootstrap_n"] = bootstrap_n
    if seed is not None:
        config["seed"] = seed
    config.setdefault("seed", 42)
    config.setdefault("bootstrap_n", 0)
    config.setdefault("noise_levels", [0.05, 0.10, 0.30, 0.60, 1.00])
    config.setdefault("center_offsets", [0.05, 0.10, 0.50])
    config.setdefault("rmax_values", [65.0, 72.0, 80.0, 100.0, 150.0])

    if output_dir is None:
        output_dir = config.get("output_dir", "test_plots")
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    # Build models
    if verbose:
        logger.info("Building models...")
    models = build_model_list(config)
    if model:
        models = [m for m in models if model.lower() in m.name.lower()]
        if not models:
            if verbose:
                logger.warning("No models matching '%s' — exiting.", model)
            return {
                "models": [], "all_results": {}, "summary": [],
                "total": 0, "passed": 0, "failed": 0,
                "report_md": None, "report_html": None, "elapsed_sec": 0,
            }
    if verbose:
        logger.info("  %d models ready", len(models))

    # Report suffix
    if model:
        report_suffix = "_" + model.lower().replace(" ", "_").replace("(", "").replace(")", "")
    else:
        first_model = models[0].name
        report_suffix = "_" + first_model.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(",", "")[:20]

    # Collection args (skip_noise / sensitivity are the only flags _collect_all_results uses)
    class _Args:
        pass
    args = _Args()
    args.skip_noise = skip_noise
    args.sensitivity = sensitivity

    # Set up test logging (replaces handlers on hii_boundary root)
    if verbose:
        _setup_test_logging(out)

    # Run all tests
    t0 = time.time()
    all_results = _collect_all_results(models, config, out, args)
    elapsed = time.time() - t0
    if verbose:
        logger.info("All tests complete (%.0fs)", elapsed)

    # Summary
    rows = _build_summary_table(all_results)
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    failed = sum(1 for r in rows if not r["passed"])

    if verbose:
        logger.info("Total: %d, Passed: %d, Failed: %d", total, passed, failed)
        logger.info("Per-test Error Metrics:")
        logger.info("  %-30s %8s %8s %8s %8s %6s", "Test", "Expected", "MRE", "RMS", "MaxErr", "Status")
        logger.info("  %s", "-"*68)
        for r in rows:
            status = "PASS" if r["passed"] else "FAIL"
            logger.info("  %-30s %8.3f %8.3f %8.3f %8.3f %6s",
                          r["name"], r["expected"], r["mre"], r["rms"], r["max_error"], status)

    # Reports
    report_md = None
    report_html = None
    if not skip_reports:
        # Extract resolved params from a non-baseline test result (actual values used)
        # Skip baseline since it hardcodes n_bootstrap=0
        resolved_params = {}
        for tr in all_results.values():
            if isinstance(tr, TestResult) and '_resolved_params' in tr.result:
                if tr.test_type != 'baseline':
                    resolved_params = tr.result['_resolved_params']
                    break
        # Fallback to first result if no non-baseline found
        if not resolved_params:
            for tr in all_results.values():
                if isinstance(tr, TestResult) and '_resolved_params' in tr.result:
                    resolved_params = tr.result['_resolved_params']
                    break
        report_config = _build_report_config(config, resolved_params)
        report_results = {}
        for label, tr in all_results.items():
            if isinstance(tr, TestResult):
                report_results[label] = _test_result_to_report_dict(tr)

        if verbose:
            logger.info("Generating reports...")
        try:
            md_content = build_md_report(report_results, report_config)
            html_content = build_html_report(md_content, out)
            md_path, html_path = write_report_files(
                md_content, html_content, out, suffix=report_suffix,
            )
            report_md = md_path
            report_html = html_path
            if verbose:
                logger.info("OK  (MD: %s, HTML: %s)", md_path.name, html_path.name)
        except Exception as e:
            if verbose:
                logger.error("report generation failed: %s", e)

    if verbose:
        logger.info("Done.")

    return {
        "models": [m.name for m in models],
        "all_results": all_results,
        "summary": rows,
        "total": total,
        "passed": passed,
        "failed": failed,
        "report_md": report_md,
        "report_html": report_html,
        "elapsed_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HII Boundary Detection Test Plan")
    parser.add_argument("--config", default="test_config.yaml",
                        help="Test config YAML")
    parser.add_argument("--output", default=None,
                        help="Output directory (default from config or test_plots)")
    parser.add_argument("--skip-noise", action="store_true",
                        help="Skip noise sweep")
    parser.add_argument("--sensitivity", action="store_true", default=True,
                        help="Run sensitivity tests (default: True)")
    parser.add_argument("--no-sensitivity", action="store_false", dest="sensitivity",
                        help="Skip sensitivity tests")
    parser.add_argument("--bootstrap-n", type=int, default=None,
                        help="Override bootstrap_n in config")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed in config")
    parser.add_argument("--skip-reports", action="store_true",
                        help="Skip HTML/MD report generation")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only models matching this substring (e.g. 'Sharp', 'Gaussian', 'Sigmoid')")
    args = parser.parse_args()

    run_tests(
        config_path=args.config,
        output_dir=args.output,
        model=args.model,
        bootstrap_n=args.bootstrap_n,
        seed=args.seed,
        skip_noise=args.skip_noise,
        sensitivity=args.sensitivity,
        skip_reports=args.skip_reports,
        verbose=True,
    )


if __name__ == "__main__":
    main()
