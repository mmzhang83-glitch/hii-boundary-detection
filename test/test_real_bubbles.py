"""Run boundary detection on preprocessed GLIMPSE bubble images.

Uses cleaned images (+ sigma_map) from preprocess_glimpse.py.
Coordinates derived from galactic (glon, glat) via WCS pixel mapping.
Search radius rmax = Rout (catalog outer radius) × rmax_scale.

Usage:
    python test_real_bubbles.py [n_bootstrap] [--rmax-scale 1.2] [--output-dir test_plots_real]

    n_bootstrap defaults to 10 for quick verification.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from boundary_detection.detect_hii_boundary import detect_hii_boundary, _save_result_h5
from test_diagnostics import plot_boundary_overlay

logger = logging.getLogger("hii_boundary.real_data")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RealBubbleResult:
    """Result of boundary detection on a single real bubble image."""

    name: str
    glon: float
    glat: float
    R_arcmin: float  # catalog <R>
    fits_science: Path
    fits_uncertainty: Optional[Path]
    result: dict  # raw detect_hii_boundary output
    boundary_mean_r_pixel: float  # detected mean boundary radius (pixels)
    boundary_mean_r_arcmin: float  # detected mean boundary radius (arcmin)
    uncertainty_mean: Optional[float]  # mean bootstrap uncertainty (pixels), None if no bootstrap
    scenario: Optional[str]  # bootstrap scenario (A/B/C)
    plots: dict = field(default_factory=dict)  # {"overlay": path, "uncertainty": path}
    success: bool = True
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Per-bubble detection
# ---------------------------------------------------------------------------


def test_single_bubble(
    name: str,
    glon: float,
    glat: float,
    Rout_arcmin: float,
    fits_cleaned: Path,
    fits_sigma: Path,
    config: dict,
    output_dir: Path,
    rmax_scale: float = 1.2,
    n_bootstrap: Optional[int] = None,
    showfile: Optional[Path] = None,
) -> RealBubbleResult:
    """Detect boundary on one bubble using preprocessed images.

    Uses cleaned image (point sources removed) and sigma_map for
    error-aware detection. Coordinates are derived from galactic (glon, glat)
    via WCS pixel mapping, and search radius from Rout × scale.

    Parameters
    ----------
    name : str
        Bubble name (e.g. "N131").
    glon, glat : float
        Galactic centre from catalog (degrees).
    Rout_arcmin : float
        Outer radius from catalog (arcmin).
    fits_cleaned : Path
        Path to preprocessed cleaned FITS image (with WCS header).
    fits_sigma : Path
        Path to preprocessed sigma_map FITS image.
    config : dict
        Algorithm configuration override.
    output_dir : Path
        Directory for diagnostic plots.
    rmax_scale : float
        Scale factor applied to Rout for search radius (default 1.2).
    n_bootstrap : int, optional
        Bootstrap iterations. If None, read from config bootstrap section
        (default 50 if not in config). 0 = skip.

    Returns
    -------
    RealBubbleResult
    """
    logger.info("=" * 60)
    logger.info("Processing %s: glon=%.3f glat=%.3f Rout=%.2f'",
                name, glon, glat, Rout_arcmin)

    try:
        # ── 1. Load cleaned image + WCS ──
        logger.info("  Loading cleaned: %s", fits_cleaned)
        with fits.open(fits_cleaned) as hdul:
            cleaned_data = hdul[0].data.astype(np.float64)
            if cleaned_data.ndim == 3:
                cleaned_data = cleaned_data[0]
            wcs = WCS(hdul[0].header)

        # ── 2. Convert galactic → pixel coordinates ──
        gc = SkyCoord(l=glon, b=glat, unit="deg", frame="galactic")
        icrs = gc.transform_to("icrs")
        xc, yc = wcs.world_to_pixel(icrs)
        logger.info("  Galactic (%.3f, %.3f) → ICRS (%.5f°, %.5f°) → pixel (%.1f, %.1f)",
                    glon, glat, icrs.ra.deg, icrs.dec.deg, xc, yc)

        # ── 3. Compute pixel scale from WCS CDELT ──
        cdelt1_deg = abs(wcs.wcs.cdelt[0])
        pixel_scale_arcmin = cdelt1_deg * 60.0  # deg/pixel → arcmin/pixel
        logger.info("  Pixel scale: %.4f arcmin/px  (|CDELT1|=%.6f°)",
                    pixel_scale_arcmin, cdelt1_deg)

        # ── 4. rmax from Rout × scale ──
        rmax_arcmin = Rout_arcmin * rmax_scale
        rmax_pixel = rmax_arcmin / pixel_scale_arcmin
        logger.info("  Rout=%.2f' × %.1f = %.2f'  →  rmax=%.1f px",
                    Rout_arcmin, rmax_scale, rmax_arcmin, rmax_pixel)

        # Validate rmax fits inside image
        max_allowed = min(xc, yc, cleaned_data.shape[1] - xc, cleaned_data.shape[0] - yc)
        if rmax_pixel > max_allowed:
            old = rmax_pixel
            rmax_pixel = max_allowed * 0.95
            logger.warning("  rmax %.0f clipped to %.0f (image edge)", old, rmax_pixel)

        # ── 5. Load sigma_map ──
        logger.info("  Loading sigma: %s", fits_sigma)
        with fits.open(fits_sigma) as hdul:
            error_map = hdul[0].data.astype(np.float64)
            if error_map.ndim == 3:
                error_map = error_map[0]
        logger.info("  sigma_map: mean=%.4f  median=%.4f  max=%.4f",
                    error_map.mean(), np.median(error_map), error_map.max())

        # ── 6. Extract detection config ──
        det_cfg = (config.get("detection") or {}) if config else {}
        bscfg = dict(config.get("bootstrap", {}) if config else {})
        if n_bootstrap is None:
            n_bootstrap = bscfg.pop("n_bootstrap", None)
        else:
            bscfg.pop("n_bootstrap", None)  # avoid duplicate in **bscfg

        # ── 7. Optional downsampling for fast-test mode ──
        downsamp = int(det_cfg.pop("cleanmap_downsamp_scale", 0))
        _ds_info = None  # stored for downstream rescaling of diagnostics
        if downsamp > 0:
            from scipy.ndimage import gaussian_filter
            scale = 1.0 / downsamp
            # Keep original data for overlay plotting and diagnostic
            _cleaned_orig = cleaned_data.copy()
            _error_orig = error_map.copy()
            _xc_orig, _yc_orig = float(xc), float(yc)
            # Gaussian anti-aliasing before subsampling
            sigma = downsamp / 2.0
            cleaned_smooth = gaussian_filter(cleaned_data, sigma=sigma)
            cleaned_data_ds = cleaned_smooth[::downsamp, ::downsamp]
            # sigma_map: gaussian smooth then subsample; √N correction for binning
            error_smooth = gaussian_filter(error_map, sigma=sigma)
            error_map_ds = error_smooth[::downsamp, ::downsamp] * np.sqrt(downsamp)
            # Rescale coordinates and pixel scale
            xc_ds = float(xc) * scale
            yc_ds = float(yc) * scale
            rmax_pixel_ds = rmax_pixel * scale
            pixel_scale_arcmin_ds = pixel_scale_arcmin * downsamp  # each pixel covers more
            _ds_info = {
                "scale": downsamp,
                "original_shape": cleaned_data.shape,
                "downsampled_shape": cleaned_data_ds.shape,
                "original_xc": float(xc), "original_yc": float(yc),
                "original_rmax": rmax_pixel,
                "original_pixel_scale": pixel_scale_arcmin,
            }
            logger.info("  Downsampled %dx (σ=%.1f px) → shape %s (rmax=%.1f→%.1f px)",
                        downsamp, sigma, cleaned_data_ds.shape,
                        rmax_pixel, rmax_pixel_ds)
            # Scale absolute-pixel detection params to match downsampled image
            # (hii_detection_config.yaml defaults not in user config)
            _user_det = config.get("detection") or {}
            _hii_path = Path(__file__).parent / "hii_detection_config.yaml"
            if _hii_path.exists():
                _hii_defaults = yaml.safe_load(_hii_path.read_text()) or {}
                for _px_key in ("rmin_min_pixels", "gradient_strip_width", "contrast_strip_width"):
                    if _px_key not in _user_det:
                        _default = _hii_defaults.get(_px_key, 0)
                        if _default and _default > 0:
                            val = _default / downsamp
                            if _px_key == "rmin_min_pixels":
                                det_cfg[_px_key] = val  # float is fine for comparison checks
                            else:
                                det_cfg[_px_key] = max(int(np.ceil(val)), 1)  # must be int for slice indices
            # Diagnostic: 2×2 original vs downsampled (cleaned + sigma)
            from astropy.visualization import ZScaleInterval
            zs = ZScaleInterval()
            fig_ds, axes = plt.subplots(2, 2, figsize=(9, 9))
            panels = [
                (_cleaned_orig, f"Cleaned ({_ds_info['original_shape'][1]}×{_ds_info['original_shape'][0]})"),
                (cleaned_data_ds, f"Cleaned ds{downsamp} ({cleaned_data_ds.shape[1]}×{cleaned_data_ds.shape[0]})"),
                (_error_orig, f"σ_map ({_ds_info['original_shape'][1]}×{_ds_info['original_shape'][0]})"),
                (error_map_ds, f"σ_map ds{downsamp} ({error_map_ds.shape[1]}×{error_map_ds.shape[0]})"),
            ]
            for ax, (arr, title) in zip(axes.flat, panels):
                v1, v2 = zs.get_limits(arr)
                ax.imshow(arr, cmap="viridis", vmin=v1, vmax=v2, origin="lower")
                ax.set_title(title)
                ax.set_xticks([]); ax.set_yticks([])
            fig_ds.tight_layout()
            ds_diag_path = output_dir / f"{name}_downsamp_diagnostic.png"
            fig_ds.savefig(ds_diag_path, dpi=150, bbox_inches="tight")
            plt.close(fig_ds)
            logger.info("  saved downsamp diagnostic: %s", ds_diag_path.name)

            # Save downsampled FITS
            fits.PrimaryHDU(data=cleaned_data_ds.astype(np.float32)).writeto(
                str(output_dir / f"{name}_cleaned_ds{downsamp}.fits"), overwrite=True)
            fits.PrimaryHDU(data=error_map_ds.astype(np.float32)).writeto(
                str(output_dir / f"{name}_sigma_ds{downsamp}.fits"), overwrite=True)

            # Swap in downsampled versions
            cleaned_data, error_map = cleaned_data_ds, error_map_ds
            xc, yc, rmax_pixel, pixel_scale_arcmin = xc_ds, yc_ds, rmax_pixel_ds, pixel_scale_arcmin_ds
        else:
            logger.info("  Image shape: %s (no downsampling)", cleaned_data.shape)

        # ── 8. Call detect_hii_boundary ──
        logger.info("  Running boundary detection (rmax=%.1f, bootstrap=%d) ...",
                    rmax_pixel, n_bootstrap)
        result = detect_hii_boundary(
            cleaned_data,
            float(xc),
            float(yc),
            float(rmax_pixel),
            error_map=error_map,
            n_bootstrap=n_bootstrap,
            showfile=showfile,
            show_diagnostics=showfile is not None,
            **det_cfg,
            **bscfg,
        )
        logger.info("  Detection complete. scenario=%s", result.get("scenario"))

        # ── 9. Extract metrics ──
        boundary_radii = result.get("boundary_radii")
        if boundary_radii is None:
            raise ValueError("boundary_radii is None — detection failed")

        # If downsampled, rescale all per-pixel results back to original image space
        _orig_pixel_scale = pixel_scale_arcmin  # default
        if _ds_info is not None:
            s = _ds_info["scale"]
            # Per-angle arrays
            boundary_radii = np.asarray(boundary_radii) * s
            for _k in ("boundary_radii", "boundary_uncertainty", "boundary_x", "boundary_y"):
                if result.get(_k) is not None:
                    result[_k] = np.asarray(result[_k]) * s
            # Scalar values
            for _k in ("mean_uncertainty", "max_uncertainty", "rmin_final", "xc", "yc", "rmax"):
                if result.get(_k) is not None:
                    result[_k] = result[_k] * s
            # _stable_result scalars
            _sr = result.get("_stable_result", {})
            if "rmin_final" in _sr:
                _sr["rmin_final"] = _sr["rmin_final"] * s
            _orig_pixel_scale = _ds_info["original_pixel_scale"]

        boundary_mean_r_pixel = float(np.mean(boundary_radii))
        boundary_mean_r_arcmin = boundary_mean_r_pixel * _orig_pixel_scale

        uncertainty = result.get("boundary_uncertainty")
        if uncertainty is not None and np.all(uncertainty == 0):
            uncertainty_mean = None
        elif uncertainty is not None:
            uncertainty_mean = float(np.mean(uncertainty))
        else:
            uncertainty_mean = None

        scenario = result.get("scenario", "C")

        logger.info("  Detected R = %.2f px (%.3f arcmin)  Rout_catalog = %.2f arcmin",
                    boundary_mean_r_pixel, boundary_mean_r_arcmin, Rout_arcmin)
        if uncertainty_mean is not None:
            logger.info("  Mean uncertainty = %.3f px", uncertainty_mean)
        else:
            logger.info("  No bootstrap uncertainty")

        # ── 9.5. Save result as HDF5 ──
        save_h5 = output_dir / f"{name}_result.h5"
        _save_result_h5(result, save_h5)
        logger.info("  saved HDF5: %s", save_h5)

        # ── 10. Generate diagnostic plots ──
        plots = {}
        angles = np.linspace(0, 2 * np.pi, len(boundary_radii))
        r_expected_px = Rout_arcmin / _orig_pixel_scale

        # Overlay plot (use original image if downsampled)
        overlay_path = output_dir / f"{name}_overlay.png"
        _ovl_img = _cleaned_orig if _ds_info else cleaned_data
        _ovl_xc = _xc_orig if _ds_info else float(xc)
        _ovl_yc = _yc_orig if _ds_info else float(yc)
        # Polar background for uncertainty panel
        _pb, _prr = None, None
        if uncertainty is not None and not np.all(uncertainty == 0):
            _img_p = _cleaned_orig if _ds_info else cleaned_data
            _xc_p = _xc_orig if _ds_info else float(xc)
            _yc_p = _yc_orig if _ds_info else float(yc)
            _rmax_p = _ds_info["original_rmax"] if _ds_info else rmax_pixel
            _rmin_p = result.get("rmin_final", _rmax_p * 0.05)
            from boundary_detection.extract_circle_boundary import manual_polar_transform
            _n_radii = max(int(_rmax_p - _rmin_p), 20)
            _prr = np.linspace(_rmin_p, _rmax_p, _n_radii)
            _pb = manual_polar_transform(
                _img_p, center=(_xc_p, _yc_p),
                output_shape=(_n_radii, 360),
                radius_range=(_rmin_p, _rmax_p))
        plot_boundary_overlay(
            image=_ovl_img,
            xc=_ovl_xc,
            yc=_ovl_yc,
            r_detected=boundary_radii,
            angles=angles,
            r_expected=r_expected_px,
            save_path=overlay_path,
            title=f"{name} — Detected Boundary ±1σ",
            boundary_uncertainty=uncertainty,
            show_expected=False,
            polar_image=_pb,
            polar_rr=_prr,
        )
        plots["overlay"] = str(overlay_path)
        logger.info("  Saved overlay: %s", overlay_path)

        # algo_diag (generated by detect_hii_boundary if showfile was passed)
        if showfile is not None:
            plots["algo_diag"] = str(showfile)
            logger.info("  Saved algo_diag: %s", showfile)

        return RealBubbleResult(
            name=name,
            glon=glon,
            glat=glat,
            R_arcmin=Rout_arcmin,
            fits_science=fits_cleaned,
            fits_uncertainty=fits_sigma,
            result=result,
            boundary_mean_r_pixel=boundary_mean_r_pixel,
            boundary_mean_r_arcmin=boundary_mean_r_arcmin,
            uncertainty_mean=uncertainty_mean,
            scenario=scenario,
            plots=plots,
            success=True,
        )

    except Exception as e:
        logger.error("  FAILED: %s — %s", name, e, exc_info=True)
        return RealBubbleResult(
            name=name,
            glon=glon,
            glat=glat,
            R_arcmin=Rout_arcmin,
            fits_science=fits_cleaned,
            fits_uncertainty=fits_sigma,
            result={},
            boundary_mean_r_pixel=0.0,
            boundary_mean_r_arcmin=0.0,
            uncertainty_mean=None,
            scenario=None,
            plots={},
            success=False,
            error_message=str(e),
        )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def test_all_bubbles(
    manifest_path: str = "real_data/download_manifest.json",
    catalog_path: str = "bubble_catalog.csv",
    output_dir: str = "test_plots_real",
    config_path: str = "test_config_real.yaml",
    rmax_scale: float = 1.2,
    n_bootstrap: Optional[int] = None,
) -> list[RealBubbleResult]:
    """Iterate all bubbles, run detection on preprocessed images.

    Reads galactic coordinates from manifest, Rout from catalog CSV,
    loads preprocessed cleaned + sigma FITS, computes pixel coords
    via WCS, and runs detect_hii_boundary.

    Parameters
    ----------
    manifest_path : str
        Path to download_manifest.json.
    catalog_path : str
        Path to bubble_catalog.csv (contains Rout).
    output_dir : str
        Output directory for plots.
    config_path : str
        Path to test_config_real.yaml.
    rmax_scale : float
        Scale factor for Rout → rmax (default 1.2).
    n_bootstrap : int, optional
        Bootstrap iterations. If None, read from config bootstrap section.

    Returns
    -------
    list of RealBubbleResult
    """
    base_dir = Path(__file__).parent

    def _resolve(p: str) -> Path:
        return p if Path(p).is_absolute() else base_dir / p

    manifest_file = _resolve(manifest_path)
    if not manifest_file.exists():
        # Fallback: new directory structure
        manifest_file = base_dir / "test_plots_real" / "images" / "download_manifest.json"
    if not manifest_file.exists():
        # Fallback: old directory structure
        manifest_file = base_dir / "real_data" / "download_manifest.json"
    catalog_file = _resolve(catalog_path)
    config_file = _resolve(config_path)
    out_dir = _resolve(output_dir)

    # ── Load manifest ──
    logger.info("Loading manifest: %s", manifest_file)
    with open(manifest_file, "r") as f:
        manifest = json.load(f)
    logger.info("Found %d bubbles in manifest", len(manifest))

    # ── Load catalog CSV → {name: Rout_arcmin} ──
    logger.info("Loading catalog: %s", catalog_file)
    rout_map: dict[str, float] = {}
    with open(catalog_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rout_map[row["[CPA2006]"]] = float(row["Rout"])
    logger.info("Loaded Rout for %d bubbles", len(rout_map))

    # ── Load config ──
    if config_file.exists():
        with open(config_file, "r") as f:
            config = yaml.safe_load(f) or {}
        logger.info("Loaded config: %s", config_file)
    else:
        config = {}
        logger.info("Config not found: %s — using defaults", config_file)

    # ── Create output directory ──
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    # ── Process each bubble ──
    results: list[RealBubbleResult] = []
    for i, entry in enumerate(manifest):
        name = entry["name"]
        logger.info("[%d/%d] Bubble: %s", i + 1, len(manifest), name)

        # Look up Rout from catalog
        if name not in rout_map:
            logger.warning("  %s not found in catalog — skipping", name)
            continue
        Rout_arcmin = rout_map[name]

        # Resolve cleaned + sigma FITS paths (prefer new directory structure)
        fits_cleaned = base_dir / "test_plots_real" / "preprocess" / f"{name}_8um_cleaned.fits"
        if not fits_cleaned.exists():
            fits_cleaned = base_dir / "real_data" / f"{name}_8um_cleaned.fits"
        fits_sigma = base_dir / "test_plots_real" / "preprocess" / f"{name}_8um_sigma.fits"
        if not fits_sigma.exists():
            fits_sigma = base_dir / "real_data" / f"{name}_8um_sigma.fits"

        if not fits_cleaned.exists():
            logger.warning("  Cleaned FITS not found: %s — run preprocess_glimpse.py first", fits_cleaned)
            results.append(RealBubbleResult(
                name=name, glon=entry["glon"], glat=entry["glat"],
                R_arcmin=Rout_arcmin, fits_science=fits_cleaned, fits_uncertainty=fits_sigma,
                result={}, boundary_mean_r_pixel=0.0, boundary_mean_r_arcmin=0.0,
                uncertainty_mean=None, scenario=None, plots={}, success=False,
                error_message="cleaned FITS missing",
            ))
            continue

        r = test_single_bubble(
            name=name,
            glon=entry["glon"],
            glat=entry["glat"],
            Rout_arcmin=Rout_arcmin,
            fits_cleaned=fits_cleaned,
            fits_sigma=fits_sigma,
            config=config,
            output_dir=out_dir,
            rmax_scale=rmax_scale,
            n_bootstrap=n_bootstrap,
        )
        results.append(r)

    # ── Print summary ──
    n_success = sum(1 for r in results if r.success)
    logger.info("=" * 60)
    logger.info("SUMMARY: %d/%d successful", n_success, len(results))
    logger.info("-" * 60)
    logger.info("%-8s %8s %8s %12s %12s %10s %8s",
                "Name", "glon", "glat", "Rout(')", "R_det(')", "σ(px)", "Status")
    logger.info("-" * 60)
    for r in results:
        status = "OK" if r.success else "FAIL"
        r_det_arcmin = f"{r.boundary_mean_r_arcmin:.3f}" if r.success else "N/A"
        sigma_str = f"{r.uncertainty_mean:.3f}" if r.uncertainty_mean is not None else "N/A"
        logger.info("%-8s %8.3f %8.3f %12.2f %12s %10s %8s",
                    r.name, r.glon, r.glat, r.R_arcmin,
                    r_det_arcmin, sigma_str, status)
    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Run boundary detection on preprocessed GLIMPSE bubble images"
    )
    p.add_argument(
        "n_bootstrap", type=int, nargs="?", default=None,
        help="Bootstrap iterations (default from config or 50)",
    )
    p.add_argument(
        "--rmax-scale", type=float, default=1.2,
        help="Rout scale factor for search radius rmax (default: 1.2)",
    )
    p.add_argument(
        "--output-dir", type=str, default="test_plots_real",
        help="Output directory (default: test_plots_real)",
    )
    p.add_argument(
        "--config", type=str, default="test_config_real.yaml",
        help="Config file (default: test_config_real.yaml)",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = test_all_bubbles(
        n_bootstrap=args.n_bootstrap,
        rmax_scale=args.rmax_scale,
        output_dir=args.output_dir,
        config_path=args.config,
    )
    n_ok = sum(1 for r in results if r.success)
    print(f"\nDone. {n_ok}/{len(results)} successful")
