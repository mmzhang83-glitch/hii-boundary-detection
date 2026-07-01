#!/usr/bin/env python3
"""Run full real-data pipeline: catalog → download → preprocess → detect → report.

Usage:
    python run_test_plan_real.py                          # 使用 test_config_real.yaml
    python run_test_plan_real.py --config my_config.yaml   # 自定义配置
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger("hii_boundary.real_data.pipeline")

# Cached package data dir lookup
_pkg_data_dir: Path | None = None


def _get_package_data_dir() -> Path | None:
    """Return path to bundled package data, or None if not available."""
    global _pkg_data_dir
    if _pkg_data_dir is None:
        try:
            from boundary_detection import __file__ as _pkg_file
            _candidate = Path(_pkg_file).parent / 'data'
            _pkg_data_dir = _candidate if _candidate.is_dir() else None
        except Exception:
            _pkg_data_dir = None
    return _pkg_data_dir


def get_git_commit(base_dir: Path) -> str:
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(base_dir),
        )
        return result.stdout.strip() or "N/A"
    except Exception:
        return "N/A"


def _run_only_plot(target_name: str, output_dir: Path, config: dict,
                   base_dir: Path, config_path: str) -> dict:
    """Regenerate all plots and HTML report from saved intermediate data."""

    pp_dir = output_dir / "preprocess"
    _det_cfg = dict(config.get("detection") or {})
    _ds = _det_cfg.get("cleanmap_downsamp_scale", 0)
    det_dir = output_dir / "detection" / f"ds{int(_ds)}"
    img_dir = output_dir / "images"

    # ── 1. Load preprocess result ──
    pp_json_path = pp_dir / f"{target_name}_result.json"
    if not pp_json_path.exists():
        raise FileNotFoundError(f"Preprocess result not found: {pp_json_path}")
    with open(pp_json_path) as f:
        pp_result = json.load(f)
    logger.info("  loaded preprocess result: %s", pp_json_path)

    # ── 2. Load detection result ──
    det_json_path = det_dir / f"{target_name}_result.json"
    if not det_json_path.exists():
        raise FileNotFoundError(f"Detection result not found: {det_json_path}")
    with open(det_json_path) as f:
        det_result = json.load(f)
    logger.info("  loaded detection result: %s", det_json_path)

    # ── 3. Load NPZ arrays ──
    npz_path = det_dir / f"{target_name}_arrays.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Arrays not found: {npz_path}")
    arrays = dict(np.load(npz_path))
    logger.info("  loaded arrays from %s: %s", npz_path, list(arrays.keys()))

    # ── 4. Find original science FITS from manifest ──
    manifest_path = img_dir / "download_manifest.json"
    pkg_data = _get_package_data_dir()
    _manifest_src = manifest_path
    if not _manifest_src.exists() and pkg_data:
        _pkg_manifest = pkg_data / "images" / "download_manifest.json"
        if _pkg_manifest.exists():
            _manifest_src = _pkg_manifest
    found_science = None
    xc, yc = None, None
    pixel_scale_arcmin = 0.02
    if _manifest_src.exists():
        with open(_manifest_src) as f:
            manifest = json.load(f)
        for e in manifest:
            if e["name"] == target_name:
                s = Path(e["fits_science"])
                found_science = s if s.is_absolute() else base_dir / s
                if not found_science.exists():
                    # Fallback 1: same filename in images/ dir
                    found_science = img_dir / s.name
                if not found_science.exists() and pkg_data:
                    # Fallback 2: bundled package data
                    found_science = pkg_data / "images" / s.name
                xc = e.get("xc_pixel")
                yc = e.get("yc_pixel")
                pixel_scale_arcmin = e.get("pixel_scale_arcmin", 0.02)
                break

    # ── 5. Re-plot preprocess diagnostic ──
    w1_path_str = pp_result.get("w1_precleaned_path")
    cleaned_path = Path(pp_result["cleaned_path"]) if pp_result.get("cleaned_path") else None
    sigma_path = Path(pp_result["sigma_path"]) if pp_result.get("sigma_path") else None

    if found_science and w1_path_str and cleaned_path and sigma_path:
        from astropy.io import fits
        from astropy.visualization import ZScaleInterval
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        original_data = fits.getdata(str(found_science)).astype(np.float64)
        if original_data.ndim == 3:
            original_data = original_data[0]
        cleaned_data = fits.getdata(str(cleaned_path))
        sigma_map_data = fits.getdata(str(sigma_path))

        # Load point source mask if available, else fallback to w1
        mask_path_str = pp_result.get("ps_mask_path")
        if mask_path_str and Path(mask_path_str).exists():
            ps_mask = fits.getdata(mask_path_str).astype(bool)
        else:
            ps_mask = None

        diagnostic_path = pp_dir / f"{target_name}_preprocess_diagnostic.png"
        zscale = ZScaleInterval()
        fig, axes = plt.subplots(2, 2, figsize=(9, 9))
        v1, v2 = zscale.get_limits(original_data)
        axes[0, 0].imshow(original_data, cmap="viridis", vmin=v1, vmax=v2, origin="lower")
        axes[0, 0].set_title(f"Original ({target_name}, GLIMPSE 8μm)")
        if ps_mask is not None:
            axes[0, 1].imshow(ps_mask.astype(float), cmap="Reds", origin="lower")
        else:
            w1_data = fits.getdata(w1_path_str)
            v1, v2 = zscale.get_limits(w1_data)
            axes[0, 1].imshow(w1_data, cmap="viridis", vmin=v1, vmax=v2, origin="lower")
        axes[0, 1].set_title(f"Point Source Mask — {target_name}")
        v1, v2 = zscale.get_limits(cleaned_data)
        axes[1, 0].imshow(cleaned_data, cmap="viridis", vmin=v1, vmax=v2, origin="lower")
        axes[1, 0].set_title(f"Cleaned ({target_name})")
        v1, v2 = zscale.get_limits(sigma_map_data)
        im = axes[1, 1].imshow(sigma_map_data, cmap="viridis", vmin=v1, vmax=v2, origin="lower")
        axes[1, 1].set_title(f"σ_map ({target_name})")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(diagnostic_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  saved preprocess diagnostic: %s", diagnostic_path)

        # Scales diagnostic from persisted FITS
        file_labels = ["work_data", "precleaned", "w1", "w2", "w3", "w4", "w5", "c5"]
        scale_paths = [pp_dir / f"{target_name}_scale_{lbl}.fits" for lbl in file_labels]
        if all(p.exists() for p in scale_paths):
            scales_path = pp_dir / f"{target_name}_scales_diagnostic.png"
            fig2, axs = plt.subplots(2, 4, figsize=(14, 7))
            for idx, (ax, fp, tl) in enumerate(zip(axs.flat, scale_paths, file_labels)):
                arr = fits.getdata(str(fp))
                v1, v2 = zscale.get_limits(arr)
                ax.imshow(arr, cmap="viridis", vmin=v1, vmax=v2, origin="lower")
                ax.set_title(tl)
                ax.set_xticks([]); ax.set_yticks([])
            fig2.tight_layout()
            fig2.savefig(scales_path, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            logger.info("  saved scales diagnostic: %s", scales_path.name)

    # ── 6. Re-plot detection plots ──
    boundary_radii = arrays["boundary_radii"]
    boundary_angles = arrays["boundary_angles"]
    boundary_uncertainty = arrays.get("boundary_uncertainty")

    overlay_image = None
    if found_science:
        from astropy.io import fits
        overlay_image = fits.getdata(str(found_science)).astype(np.float64)
        if overlay_image.ndim == 3:
            overlay_image = overlay_image[0]
    if overlay_image is None and cleaned_path and cleaned_path.exists():
        overlay_image = fits.getdata(str(cleaned_path))

    if overlay_image is not None and xc is not None:
        from test_diagnostics import plot_boundary_overlay, plot_polar_pipeline

        r_expected_px = det_result.get("R_arcmin", 6.55) / pixel_scale_arcmin

        overlay_path = det_dir / f"{target_name}_overlay.png"
        plot_boundary_overlay(
            image=overlay_image,
            xc=xc, yc=yc,
            r_detected=boundary_radii,
            angles=boundary_angles,
            r_expected=r_expected_px,
            save_path=overlay_path,
            title=f"{target_name} — Detected Boundary ±1σ",
            boundary_uncertainty=boundary_uncertainty,
            show_expected=False,
            polar_image=arrays.get("polar_roi_orig"),
            polar_rr=arrays.get("rr_roi_orig"),
        )
        logger.info("  saved overlay: %s", overlay_path)


        pipeline_keys = ["polar_roi", "polar_smooth", "score_map", "dv_map",
                         "cost_map", "cost_map_radii", "rr_roi"]
        if all(k in arrays for k in pipeline_keys):
            pipeline_path = det_dir / f"{target_name}_pipeline.png"
            # Use unscaled boundary for pipeline plot (intermediates are in ds space)
            _pipeline_r = arrays.get("boundary_radii_ds", boundary_radii)
            plot_polar_pipeline(
                polar_roi=arrays["polar_roi"],
                polar_smooth=arrays["polar_smooth"],
                score_map=arrays["score_map"],
                dv_map=arrays["dv_map"],
                cost_map=arrays["cost_map"],
                cost_map_radii=arrays["cost_map_radii"],
                boundary_radii=_pipeline_r,
                angles=boundary_angles,
                rr_roi=arrays["rr_roi"],
                save_path=pipeline_path,
                title=f"{target_name} — Polar Pipeline",
            )
            logger.info("  saved pipeline: %s", pipeline_path)

    # ── 7. Generate HTML report ──
    from test_report import build_real_report
    from astropy.table import Table

    # Build minimal detection result object for HTML
    class _DetResult:
        pass
    _dr = _DetResult()
    _dr.name = det_result.get("name", target_name)
    _dr.R_arcmin = det_result.get("R_arcmin", 0)
    _dr.boundary_mean_r_pixel = det_result.get("boundary_mean_r_pixel", 0)
    _dr.boundary_mean_r_arcmin = det_result.get("boundary_mean_r_arcmin", 0)
    _dr.uncertainty_mean = det_result.get("uncertainty_mean")
    _dr.scenario = det_result.get("scenario")
    _dr.success = det_result.get("success", True)
    _dr.error_message = None
    _dr.plots = {"overlay": str(det_dir / f"{target_name}_overlay.png"),
                  "pipeline": str(det_dir / f"{target_name}_pipeline.png"),
                  "algo_diag": str(det_dir / f"{target_name}_algo_diag.png")}

    catalog_path = output_dir / "catalog" / "bubble_catalog.csv"
    if catalog_path.exists():
        catalog = Table.read(str(catalog_path), format="ascii.csv")
    else:
        catalog = Table(names=["Name", "GLON", "GLAT", "<R>", "MFlags"])

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = []

    _n_cat = len(catalog) if catalog_path.exists() else 0
    _n_dl = len(manifest)
    # Check actual files in images/ dir (manifest may have stale paths)
    _dl_ok = 0
    for e in manifest:
        _p = Path(e["fits_science"])
        if not _p.exists():
            _p = img_dir / _p.name
        if _p.exists():
            _dl_ok += 1

    phases = {
        "catalog": {"status": "done", "n_bubbles": f"{_n_cat}"},
        "download": {"status": "done", "n_downloaded": f"{_dl_ok}/{_n_dl}"},
        "preprocess": {"status": "done", "result": pp_result},
        "detection": {"status": "done", "result": _dr},
    }

    report_config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(base_dir),
        "python_version": sys.version.split()[0],
        "config_path": config_path,
        "algo_params": config.get("detection") or {},
        "debug_scales": config.get("preprocess", {}).get("debug_scales", False),
    }

    html_content = build_real_report(
        phases=phases,
        catalog=catalog,
        manifest=manifest,
        config=report_config,
        output_dir=output_dir,
    )

    html_path = output_dir / "result.html"
    html_path.write_text(html_content, encoding='utf-8')
    logger.info("  saved HTML report: %s", html_path)
    logger.info("=" * 60)
    logger.info("Done.  report=%s", html_path)

    return {
        "config": config,
        "catalog": catalog if catalog_path.exists() else None,
        "manifest": manifest,
        "preprocess_result": pp_result,
        "detection_result": det_result,
        "report_html": html_path,
        "elapsed_sec": 0.0,
    }


def load_config(config_path: str) -> dict:
    base_dir = Path(__file__).parent
    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = base_dir / cfg_file
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config not found: {cfg_file}")
    with open(cfg_file, 'r') as f:
        config = yaml.safe_load(f) or {}
    return config, base_dir


def run_test_plan_real(config_path: str = "test_config_real.yaml",
                       only_plot: bool | None = None) -> dict:
    """Run full real-data pipeline: catalog → download → preprocess → detect → report.

    Parameters
    ----------
    config_path : str
        Path to test_config_real.yaml.
    only_plot : bool or None
        If True, skip computation and regenerate plots/HTML from saved data.
        If None, read from config (default False).

    Returns
    -------
    dict with keys:
        config, catalog, manifest, preprocess_result, detection_result, report_html, elapsed_sec
    """
    config, base_dir = load_config(config_path)
    t_start = time.time()

    pipe_cfg = config.get("pipeline", {})
    output_dir = (base_dir / pipe_cfg.get("output_dir", "test_plots_real")).expanduser()
    target_name = pipe_cfg.get("name")
    skip_download = pipe_cfg.get("skip_download", False)
    skip_preprocess = pipe_cfg.get("skip_preprocess", False)

    if only_plot is None:
        only_plot = pipe_cfg.get("only_plot", False)

    cat_dir = output_dir / "catalog"
    img_dir = output_dir / "images"
    pp_dir = output_dir / "preprocess"
    _det_cfg = dict(config.get("detection") or {})
    _ds = _det_cfg.get("cleanmap_downsamp_scale", 0)
    det_dir = output_dir / "detection" / f"ds{int(_ds)}"
    for d in [cat_dir, img_dir, pp_dir, det_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Phase tracking for the report
    phases = {k: {"status": "pending"} for k in ["catalog", "download", "preprocess", "detection"]}

    if only_plot:
        logger.info("=" * 60)
        logger.info("ONLY-PLOT mode: regenerating all plots and HTML from saved data")
        return _run_only_plot(target_name, output_dir, config, base_dir, config_path)

    # ── Phase 1: Catalog ──
    logger.info("=" * 60)
    logger.info("[1/4] Catalog fetch")
    phases["catalog"]["status"] = "running"

    from fetch_bubble_catalog import fetch_bubble_catalog
    catalog_n = config.get("catalog", {}).get("n_bubbles", 5)
    catalog_path = str(cat_dir / "bubble_catalog.csv")

    # Hydrate catalog from package data if missing locally
    pkg_data = _get_package_data_dir()
    if not Path(catalog_path).exists() and pkg_data:
        _pkg_cat = pkg_data / "catalog" / "bubble_catalog.csv"
        if _pkg_cat.exists():
            cat_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_pkg_cat, catalog_path)
            logger.info("  hydrated catalog from package data")

    if catalog_path and Path(catalog_path).exists() and skip_download:
        from astropy.table import Table
        catalog = Table.read(catalog_path, format="ascii.csv")
        logger.info("  skipped (catalog exists, skip_download=true)")
    else:
        catalog = fetch_bubble_catalog(n=catalog_n, output_path=catalog_path)

    n_bubbles = len(catalog)
    phases["catalog"] = {"status": "done", "n_bubbles": n_bubbles, "path": catalog_path}
    logger.info("  catalog: %d bubbles → %s", n_bubbles, catalog_path)

    # ── Phase 2: Download ──
    logger.info("[2/4] Image download")
    phases["download"]["status"] = "running"
    manifest_path = img_dir / "download_manifest.json"

    # Hydrate manifest from package data if missing locally
    if not manifest_path.exists() and pkg_data:
        _pkg_manifest = pkg_data / "images" / "download_manifest.json"
        if _pkg_manifest.exists():
            with open(_pkg_manifest) as f:
                _manifest = json.load(f)
            # Rewrite fits_science to point to bundled FITS
            for _e in _manifest:
                _fn = Path(_e["fits_science"]).name
                _e["fits_science"] = str(pkg_data / "images" / _fn)
            img_dir.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, 'w') as f:
                json.dump(_manifest, f, indent=2)
            logger.info("  hydrated manifest from package data")

    if manifest_path.exists() and skip_download:
        with open(manifest_path) as f:
            manifest = json.load(f)
        logger.info("  skipped (manifest exists, skip_download=true)")
    else:
        from download_glimpse_images import download_glimpse_images
        download_cfg = config.get("download", {})
        fov_factor = download_cfg.get("fov_factor", 6)
        manifest = download_glimpse_images(
            catalog_path=catalog_path,
            image_dir=str(img_dir),
            fov_factor=fov_factor,
        )

    phases["download"] = {
        "status": "done",
        "n_downloaded": len(manifest),
        "manifest_path": str(manifest_path),
    }
    logger.info("  download: %d images", len(manifest))

    # ── Phase 3: Preprocess ──
    # Only process the target bubble specified by pipeline.name
    logger.info("[3/4] Preprocess")
    phases["preprocess"]["status"] = "running"
    from preprocess_glimpse import preprocess_single_bubble

    target_entry = None
    for e in manifest:
        if e["name"] == target_name:
            target_entry = e
            break

    if target_entry is None:
        raise ValueError(f"Bubble '{target_name}' not found in download manifest")

    fits_science = Path(target_entry["fits_science"])
    if not fits_science.is_absolute():
        fits_science = base_dir / fits_science
    if not fits_science.exists():
        fits_science = img_dir / fits_science.name  # fallback to migrated images/ dir
    if not fits_science.exists() and pkg_data:
        fits_science = pkg_data / "images" / fits_science.name  # fallback to bundled data

    cleaned_path = pp_dir / f"{target_name}_8um_cleaned.fits"
    sigma_path = pp_dir / f"{target_name}_8um_sigma.fits"

    if cleaned_path.exists() and sigma_path.exists() and skip_preprocess:
        logger.info("  skipped (cleaned/sigma FITS exist, skip_preprocess=true)")
        pp_result = {
            "name": target_name,
            "cleaned_path": cleaned_path,
            "sigma_path": sigma_path,
            "diagnostic_path": pp_dir / f"{target_name}_preprocess_diagnostic.png",
            "n_sources": None,
        }
    else:
        pp_config = config.get("preprocess", {})
        pp_result = preprocess_single_bubble(
            target_name, fits_science, pp_dir, pp_config,
        )

    phases["preprocess"] = {"status": "done", "result": pp_result}
    logger.info("  preprocess: %s → %s", target_name, cleaned_path)

    # ── Phase 4: Detection ──
    logger.info("[4/4] Boundary detection")
    phases["detection"]["status"] = "running"
    from test_real_bubbles import test_single_bubble
    from test_diagnostics import plot_polar_pipeline

    cleaned_exists = pp_result["cleaned_path"].exists()

    if not cleaned_exists:
        raise FileNotFoundError(f"Cleaned FITS not found: {pp_result['cleaned_path']}")

    target = target_entry
    bscfg = dict(config.get("bootstrap", {}) if config else {})
    n_bootstrap = bscfg.pop("n_bootstrap", None)

    algo_diag_path = det_dir / f"{target_name}_algo_diag.png"
    rr = test_single_bubble(
        name=target_name,
        glon=target["glon"],
        glat=target["glat"],
        Rout_arcmin=target["R_arcmin"],
        fits_cleaned=pp_result["cleaned_path"],
        fits_sigma=pp_result["sigma_path"],
        config=config,
        output_dir=det_dir,
        n_bootstrap=n_bootstrap,
        showfile=algo_diag_path,
    )

    # Generate pipeline.png (intermediate products in _extract_result)
    if rr.success:
        stable = rr.result.get("_stable_result", {})
        extr = stable.get("_extract_result", {})
        if all(k in extr for k in ["polar_roi", "polar_smooth", "score_map", "dv_map",
                                    "cost_map", "cost_map_radii", "boundary_radii", "rr_roi"]):
            pipeline_path = det_dir / f"{target_name}_pipeline.png"
            plot_polar_pipeline(
                polar_roi=extr["polar_roi"],
                polar_smooth=extr["polar_smooth"],
                score_map=extr["score_map"],
                dv_map=extr["dv_map"],
                cost_map=extr["cost_map"],
                cost_map_radii=extr["cost_map_radii"],
                boundary_radii=extr["boundary_radii"],
                angles=extr.get("boundary_angles",
                    np.linspace(0, 2*np.pi, len(extr["boundary_radii"]))),
                rr_roi=extr["rr_roi"],
                save_path=pipeline_path,
                title=f"{target_name} — Polar Pipeline",
            )
            rr.plots["pipeline"] = str(pipeline_path)

    # ── Persist results for --only-plot mode ──
    if rr.success:
        det_summary = {
            "name": target_name,
            "glon": target["glon"],
            "glat": target["glat"],
            "R_arcmin": target["R_arcmin"],
            "boundary_mean_r_pixel": rr.boundary_mean_r_pixel,
            "boundary_mean_r_arcmin": rr.boundary_mean_r_arcmin,
            "uncertainty_mean": rr.uncertainty_mean,
            "scenario": rr.scenario,
            "success": True,
        }
        det_json_path = det_dir / f"{target_name}_result.json"
        det_json_path.write_text(json.dumps(det_summary, indent=2), encoding="utf-8")
        logger.info("  saved: %s", det_json_path.name)

        # NPZ arrays
        stable = rr.result.get("_stable_result", {})
        extr = stable.get("_extract_result", {})
        b_radii = rr.result.get("boundary_radii")
        b_uncertainty = rr.result.get("boundary_uncertainty")
        ang = np.linspace(0, 2 * np.pi, len(b_radii) if b_radii is not None else 360)

        npz_dict = {}
        if b_radii is not None:
            npz_dict["boundary_radii"] = b_radii
            npz_dict["boundary_angles"] = ang
        # Also save unscaled boundary_radii for pipeline (in downsampled space)
        if "boundary_radii" in extr:
            npz_dict["boundary_radii_ds"] = extr["boundary_radii"]
        if b_uncertainty is not None:
            npz_dict["boundary_uncertainty"] = b_uncertainty
        # Also save unscaled uncertainty for plots with polar background
        if int(_ds) > 0 and b_uncertainty is not None:
            npz_dict["boundary_uncertainty_ds"] = b_uncertainty / int(_ds)
        for k in ["polar_roi", "polar_smooth", "score_map", "dv_map",
                   "cost_map", "cost_map_radii", "rr_roi"]:
            if k in extr:
                npz_dict[k] = extr[k]

        # Compute original-resolution polar ROI for uncertainty plot background
        _cleaned_path = pp_result.get("cleaned_path")
        if _cleaned_path and Path(_cleaned_path).exists() and "rmin_final" in rr.result:
            from astropy.io import fits as _fits
            _orig_data = _fits.getdata(str(_cleaned_path)).astype(np.float64)
            if _orig_data.ndim == 3:
                _orig_data = _orig_data[0]
            _orig_xc = float(target_entry.get("xc_pixel", _orig_data.shape[1] / 2))
            _orig_yc = float(target_entry.get("yc_pixel", _orig_data.shape[0] / 2))
            _orig_ps = float(target_entry.get("pixel_scale_arcmin", 0.02))
            _orig_rmax = float(target_entry["R_arcmin"]) * 1.2 / _orig_ps
            _orig_rmin = float(rr.result.get("rmin_final", _orig_rmax * 0.05))
            from boundary_detection.extract_circle_boundary import manual_polar_transform
            _n_radii = max(int(_orig_rmax - _orig_rmin), 20)
            _polar_orig = manual_polar_transform(
                _orig_data, center=(_orig_xc, _orig_yc),
                output_shape=(_n_radii, 360),
                radius_range=(_orig_rmin, _orig_rmax))
            _rr_orig = np.linspace(_orig_rmin, _orig_rmax, _n_radii)
            npz_dict["polar_roi_orig"] = _polar_orig
            npz_dict["rr_roi_orig"] = _rr_orig
            logger.info("  computed original-resolution polar ROI: %s", _polar_orig.shape)

        if npz_dict:
            npz_path = det_dir / f"{target_name}_arrays.npz"
            np.savez_compressed(str(npz_path), **npz_dict)
            logger.info("  saved: %s (%d arrays)", npz_path.name, len(npz_dict))

    phases["detection"] = {"status": "done", "result": rr}

    # ── Phase 5: Report ──
    logger.info("Generating HTML report...")
    from test_report import build_real_report

    report_config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": get_git_commit(base_dir),
        "python_version": sys.version.split()[0],
        "config_path": config_path,
        "algo_params": _det_cfg,
    }

    html_content = build_real_report(
        phases=phases,
        catalog=catalog,
        manifest=manifest,
        config=report_config,
        output_dir=output_dir,
    )

    html_path = output_dir / "result.html"
    html_path.write_text(html_content, encoding='utf-8')

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Done.  elapsed=%.0fs  report=%s", elapsed, html_path)
    if rr.success:
        logger.info("  %s: R=%.2f px  σ=%.2f px  status=OK",
                    target_name,
                    rr.boundary_mean_r_pixel,
                    rr.uncertainty_mean or 0)
    else:
        logger.info("  %s: status=FAIL  error=%s", target_name, rr.error_message)

    return {
        "config": config,
        "catalog": catalog,
        "manifest": manifest,
        "preprocess_result": pp_result,
        "detection_result": rr,
        "report_html": html_path,
        "elapsed_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser(
        description="Run full real-data pipeline: catalog → download → preprocess → detect → report"
    )
    p.add_argument(
        "--config", type=str, default="test_config_real.yaml",
        help="Configuration file (default: test_config_real.yaml)",
    )
    p.add_argument(
        "--only-plot", action="store_true", default=None,
        help="Skip all computation, regenerate plots and HTML from saved data",
    )
    args = p.parse_args()

    result = run_test_plan_real(config_path=args.config, only_plot=args.only_plot)
    print(f"\nReport: {result['report_html']}")
