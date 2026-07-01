#!/usr/bin/env python3
"""Elliptical HII Boundary Detection Test Plan — Orchestrator.

Runs test_runner_elliptical tests on elliptical model variants.
Generates MD + HTML reports via test_report.
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from boundary_detection.detect_hii_boundary_elliptical import detect_hii_boundary_elliptical
from test_models import make_sigmoid_elliptical
from test_runner_elliptical import (
    test_baseline_elliptical,
    test_noise_elliptical,
    test_center_offset_elliptical,
    test_f_max_sensitivity,
    TestResult,
)
from test_report import build_md_report, build_html_report, write_report_files


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent,
        )
        return result.stdout.strip() or "N/A"
    except Exception:
        return "N/A"


def load_config(path="test_config.yaml"):
    """Load test configuration from YAML file.

    Resolution order:
    1. Explicit path as-is (absolute or relative to CWD)
    2. Relative to this source file's directory (for packaged/module execution)
    """
    path_obj = Path(path)
    if not path_obj.exists():
        sibling = Path(__file__).parent / path
        if sibling.exists():
            path_obj = sibling
    with open(path_obj, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("shape", (300, 300))
    cfg["shape"] = tuple(cfg["shape"])
    return cfg


def build_elliptical_models(config):
    """Build elliptical test models from test config."""
    shape = tuple(config["shape"])
    xc = config["xc"]
    yc = config["yc"]
    a = config["a"]
    phi = config.get("phi", 0.0)
    ab_ratios = config.get("ab_ratios", [0.9, 0.7, 0.5])
    k = config.get("sigmoid_k", 5.0)

    models = []
    for ratio in ab_ratios:
        b = a * ratio
        models.append(make_sigmoid_elliptical(shape, config, k=k, a=a, b=b, phi=phi))
    return models


def main():
    parser = argparse.ArgumentParser(description="Elliptical HII Boundary Test Plan")
    parser.add_argument("--config", default="test_config_elliptical.yaml")
    parser.add_argument("--output", default="test_plots_elliptical")
    parser.add_argument("--skip-noise", action="store_true")
    parser.add_argument("--no-sensitivity", action="store_false", dest="sensitivity")
    parser.add_argument("--bootstrap-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-reports", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.bootstrap_n is not None:
        config["bootstrap_n"] = args.bootstrap_n
    if args.seed is not None:
        config["seed"] = args.seed
    config.setdefault("seed", 42)
    config.setdefault("bootstrap_n", 10)
    config.setdefault("noise_levels", [0.10, 0.30])
    config.setdefault("center_offsets", [0.10, 0.50])
    config.setdefault("f_max_values", [0.8, 1.0, 1.2])
    config.setdefault("sensitivity_noise_level", 0.30)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    models = build_elliptical_models(config)
    all_results = {}
    rng = np.random.default_rng(config["seed"])

    for model in models:
        safe_name = model.name.replace(" ", "_").replace("(", "").replace(")", "")
        model_dir = out / safe_name

        # 1. Baseline
        try:
            tr = test_baseline_elliptical(
                detect_hii_boundary_elliptical, model,
                model_dir / "baseline", config,
                seed=config["seed"],
            )
            all_results[tr.name] = tr
            print(f"  baseline: {tr.name} -- {'PASS' if tr.passed else 'FAIL'}")
        except Exception as e:
            print(f"baseline ERROR for {model.name}: {e}")

        # 2. Noise sweep
        if not args.skip_noise:
            try:
                noise_dir = model_dir / "noise"
                for nr in test_noise_elliptical(
                    detect_hii_boundary_elliptical, model,
                    config["noise_levels"], noise_dir, config,
                    seed=config["seed"],
                ):
                    all_results[nr.name] = nr
                print(f"  noise: {len(config['noise_levels'])} levels")
            except Exception as e:
                print(f"noise ERROR for {model.name}: {e}")

        # 3. Sensitivity
        if args.sensitivity:
            try:
                center_dir = model_dir / "sensitivity" / "center"
                for cr in test_center_offset_elliptical(
                    detect_hii_boundary_elliptical, model,
                    config["center_offsets"], center_dir, config,
                    noise_level=config.get("sensitivity_noise_level", 0.30),
                    seed=int(rng.integers(0, 2**31)),
                ):
                    all_results[cr.name] = cr
                print(f"  center: {len(config['center_offsets'])} offsets")
            except Exception as e:
                print(f"center ERROR for {model.name}: {e}")

            try:
                f_max_dir = model_dir / "sensitivity" / "f_max"
                for fr in test_f_max_sensitivity(
                    detect_hii_boundary_elliptical, model,
                    config["f_max_values"], f_max_dir, config,
                    noise_level=config.get("sensitivity_noise_level", 0.30),
                    seed=int(rng.integers(0, 2**31)),
                ):
                    all_results[fr.name] = fr
                print(f"  f_max: {len(config['f_max_values'])} values")
            except Exception as e:
                print(f"f_max ERROR for {model.name}: {e}")

    elapsed = time.time() - t0

    # Reports
    if not args.skip_reports and all_results:
        algo_path = Path("hii_detection_config.yaml")
        algo_params = {}
        if algo_path.exists():
            with open(algo_path) as f:
                algo_params = yaml.safe_load(f) or {}

        report_config = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": get_git_commit(),
            "python_version": sys.version.split()[0],
            "shape": str(config.get("shape", "N/A")),
            "algo_params": algo_params,
        }

        report_results = {}
        for label, tr in all_results.items():
            if isinstance(tr, TestResult):
                plots = {k: str(v) for k, v in tr.plots.items()}
                bs = tr.bootstrap_metrics or {}
                em = dict(tr.error_metrics)
                em.update(bs)
                if tr.expected_boundary is not None:
                    exp_r = float(np.mean(tr.expected_boundary.radius))
                else:
                    exp_r = tr.model.expected_radius
                report_results[label] = {
                    "name": tr.name,
                    "model_description": f"{tr.model.name} ({tr.test_type})",
                    "params": tr.params,
                    "expected_radius": exp_r,
                    "error_metrics": em,
                    "passed": tr.passed,
                    "plots": plots,
                    "argmax_metrics": tr.argmax_metrics if tr.argmax_metrics is not None else {},
                }

        try:
            md_content = build_md_report(report_results, report_config)
            html_content = build_html_report(md_content, out)
            md_path, html_path = write_report_files(md_content, html_content, out, suffix="_elliptical")
            print(f"Reports: {md_path.name}, {html_path.name}")
        except Exception as e:
            print(f"report ERROR: {e}")

    total = len(all_results)
    passed = sum(1 for tr in all_results.values() if isinstance(tr, TestResult) and tr.passed)
    print(f"\nTotal: {total}, Passed: {passed}, Failed: {total - passed}")
    print(f"Elapsed: {elapsed:.0f}s")
    print("Done.")


if __name__ == '__main__':
    main()
