"""GLIMPSE 8μm image preprocessing: point source removal + noise estimation.

Uses à trous wavelet decomposition (watroo, B3 spline) to separate
compact point sources (stars, PSF ~2-10 px) from extended emission.
Point sources are detected on the finest wavelet scale w₁ and inpainted
with local median on the original image. A spatially adaptive noise map
is derived via 31×31 MAD on w₁.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
from skimage.filters import median as sk_median
from skimage.morphology import disk
from watroo import AtrousTransform, B3spline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("hii_boundary.preprocess")


def preprocess_glimpse_image(
    data: np.ndarray,
    n_scales: int = 5,
    sigma_threshold: float = 3.0,
    max_area: int = 200,
    mad_window: int = 101,
    sigma_stride: int = 10,
    detection_threshold: float = 3.0,
    max_ellipticity: float = 0.3,
    bright_mask_radius_max: int = 35,
    bright_mask_radius_min: int = 3,
    peak_threshold_low: float = 1000.0,
    peak_threshold_high: float = 2000.0,
    peak_radius_low: float = 5.0,
    peak_radius_mid: float = 10.0,
    peak_radius_high: float = 40.0,
    refine_radius: float = 10.0,
    ring_min_area: int = 5,
    ring_max_area: int = 1000,
    bright_fill_method: str = "biharmonic",
    bright_fill_expand: int = 3,
    bright_fill_outlier_sigma: float = 3.0,
    n_inpaint_iters: int = 10,
    inpaint_lambda_w1: float = 0.3,
    inpaint_lambda_w2: float = 0.3,
    n_refine_iters: int = 1,
    refine_detection_threshold: float = 2.5,
    debug_scales: bool = True,
    output_dir: str | Path | None = None,
    debug_prefix: str | None = None,
) -> dict:
    """Remove point sources and estimate noise from GLIMPSE 8μm image.

    Uses photutils segmentation for bright source detection (ellipticity-filtered),
    biharmonic inpainting for bright source filling, and wavelet iterative
    soft-thresholding inpainting for final point source removal.

    Parameters
    ----------
    data : np.ndarray
        2D float64 input image.
    n_scales : int
        A trous decomposition levels (default 5, covers ~2 to 32 px).
    sigma_threshold : float
        Detection threshold on w1 in units of sigma_local.
    max_area : int
        Maximum connected area in pixels for a point source.
    mad_window : int
        Sliding MAD window size in pixels.
    sigma_stride : int
        Sparse grid stride for sigma_map computation (px). Larger = faster
        but coarser. Default 10.
    detection_threshold : float
        photutils segmentation detection threshold in sigma units.
    max_ellipticity : float
        Maximum roundness for bright sources (0=circular, used as DAOStarFinder roundlo/roundhi).
    bright_mask_radius_max : int
        Maximum mask radius cap in pixels.
    bright_mask_radius_min : int
        Minimum mask radius floor in pixels.
    peak_threshold_low : float
        Peak threshold between low and mid bins. Default 1000.0.
    peak_threshold_high : float
        Peak threshold between mid and high bins. Default 2000.0.
    peak_radius_low : float
        Mask radius for sources with peak < peak_threshold_low.
        Default 5.0 px.
    peak_radius_mid : float
        Mask radius for peak_threshold_low ≤ peak < peak_threshold_high.
        Default 10.0 px.
    peak_radius_high : float
        Mask radius for peak ≥ peak_threshold_high. Default 40.0 px.
    refine_radius : float
        Mask radius for all refine (Step 8b) sources. Default 10.0 px.
    ring_min_area : int
        Minimum hole area for negative ring detection. 0 = disabled.
    ring_max_area : int
        Maximum hole area for negative ring detection (excludes large diffuse structures).
    bright_fill_method : str
        Bright source fill method: "biharmonic" (default) or "bg_noise".
    bright_fill_expand : int
        For "bg_noise": pixels to dilate mask for background annulus. Default 3.
    bright_fill_outlier_sigma : float
        For "bg_noise": sigma threshold for outlier removal in annulus. Default 3.0.
    n_inpaint_iters : int
        Number of wavelet iterative soft-thresholding inpainting iterations.
    inpaint_lambda_w1 : float
        Fraction of |w1| removed per inpainting iteration (default 0.3 = 30%).
    inpaint_lambda_w2 : float
        Fraction of |w2| removed per inpainting iteration (default 0.3).
    n_refine_iters : int
        Number of refine iterations: re-run DAOStarFinder on cleaned image
        to catch residual bright sources, then re-inpaint (default 1).
    refine_detection_threshold : float
        DAOStarFinder detection threshold for refine (× rms_clean). Lower
        than initial detection_threshold because residuals are dimmer.
        Default 3.0.
    debug_scales : bool
        If True, save all à trous decomposition scales as FITS files.
    output_dir : str or Path, optional
        Directory for debug FITS output.

    Returns
    -------
    dict
        Dictionary with keys:
        - cleaned_data : np.ndarray — point-source-free image
        - sigma_map : np.ndarray — spatially adaptive noise map
        # bg_map removed — no longer computed
        - point_source_mask : np.ndarray — boolean mask, True = point source
        - bright_mask : np.ndarray — photutils-detected bright source mask
        - w1_precleaned : np.ndarray — pre-cleaned (biharmonic-filled) w1 wavelet scale
        - n_sources : int — total number of detected point sources
    """
    # ── Shared utilities ──
    def _soft_threshold(arr, lam):
        return np.sign(arr) * np.maximum(np.abs(arr) - lam, 0)
    yy, xx = np.ogrid[:data.shape[0], :data.shape[1]]

    # ── Validate debug output early ──
    if debug_scales and output_dir is None:
        raise ValueError("output_dir is required when debug_scales=True")
    if debug_scales:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: negative value handling ──
    data_min = float(np.min(data))
    shifted = data_min <= 0
    if shifted:
        offset = -data_min + 1.0
        work_data = data + offset
    else:
        offset = 0.0
        work_data = data.copy()

    # ── Save original for reference ──
    original_work = work_data.copy()

    # ── Step 2: à trous on original → save w1 for later sigma_map ──
    transform = AtrousTransform(B3spline)
    coeffs_orig = transform(work_data, level=n_scales)
    w1_orig = coeffs_orig.data[0]
    selem = disk(mad_window // 2)

    # ── Step 3: DAOStarFinder bright source detection ──
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats

    _, median_bg, bg_rms = sigma_clipped_stats(work_data, sigma=3.0, maxiters=5)
    abs_threshold = bg_rms * detection_threshold
    logger.debug("Background: median=%.2f, RMS=%.4f, threshold=%dx=%.2f",
                 median_bg, bg_rms, detection_threshold, abs_threshold)

    finder = DAOStarFinder(
        fwhm=3.0,
        threshold=abs_threshold,
        roundlo=-max_ellipticity,
        roundhi=max_ellipticity,
        sharplo=0.2,
        sharphi=1.0,
    )
    stars = finder(work_data - median_bg)
    bright_mask = np.zeros(work_data.shape, dtype=bool)
    if stars is not None and len(stars) > 0:
        xs = np.round(stars['xcentroid']).astype(int)
        ys = np.round(stars['ycentroid']).astype(int)
        peaks = stars['peak']

        # Peak-based brightness binning for mask radius
        radii = np.full(len(xs), peak_radius_low, dtype=float)
        radii[(peaks >= peak_threshold_low) & (peaks < peak_threshold_high)] = peak_radius_mid
        radii[peaks >= peak_threshold_high] = peak_radius_high
        radii = np.clip(
            radii, bright_mask_radius_min, bright_mask_radius_max,
        ).astype(int)

        n_low = int(np.sum(peaks < peak_threshold_low))
        n_mid = int(np.sum((peaks >= peak_threshold_low) & (peaks < peak_threshold_high)))
        n_high = int(np.sum(peaks >= peak_threshold_high))

        for xc, yc, r in zip(xs, ys, radii):
            if 0 <= yc < work_data.shape[0] and 0 <= xc < work_data.shape[1]:
                rr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
                bright_mask |= rr <= r
        logger.info(
            "DAOStarFinder: %d bright stars, peak<%d:%d r=%.0f, %d≤peak<%d:%d r=%.0f, peak≥%d:%d r=%.0f "
            "(radius %d-%d px, thr=%.2f, roundness≤%.1f)",
            len(stars),
            peak_threshold_low, n_low, peak_radius_low,
            peak_threshold_low, peak_threshold_high, n_mid, peak_radius_mid,
            peak_threshold_high, n_high, peak_radius_high,
            radii.min(), radii.max(), abs_threshold, max_ellipticity,
        )
    else:
        logger.info("DAOStarFinder: no bright stars found (thr=%.2f)", abs_threshold)

    # ── Step 4: fill bright sources ──
    if bright_mask.any():
        if bright_fill_method == "bg_noise":
            # Per-component background+noise fill
            labeled, n_comp = ndimage.label(bright_mask)
            fill_data = work_data.copy()
            rng = np.random.default_rng()
            for comp_id in range(1, n_comp + 1):
                comp_mask = labeled == comp_id
                # Expand to define annulus
                dilated = ndimage.binary_dilation(
                    comp_mask, structure=np.ones((3, 3), dtype=bool),
                    iterations=bright_fill_expand,
                )
                annulus = dilated & ~comp_mask
                annulus_vals = work_data[annulus]
                if len(annulus_vals) >= 5:
                    # Sigma-clip outliers
                    med = float(np.median(annulus_vals))
                    std = float(np.std(annulus_vals))
                    clipped = annulus_vals[
                        np.abs(annulus_vals - med) < bright_fill_outlier_sigma * std
                    ]
                    if len(clipped) >= 3:
                        med = float(np.median(clipped))
                        std = float(np.std(clipped))
                else:
                    med = float(np.median(work_data[~bright_mask]))
                    std = float(np.std(work_data[~bright_mask]))
                # Fill with background + noise
                n_pix = comp_mask.sum()
                fill_data[comp_mask] = med + rng.normal(0.0, max(std, 1e-6), n_pix)
            work_data = fill_data
            logger.info("bg+noise filled %d bright sources (%d px), expand=%d, outlier_sigma=%.1f",
                        n_comp, bright_mask.sum(), bright_fill_expand, bright_fill_outlier_sigma)
        else:
            from skimage.restoration import inpaint_biharmonic
            work_data = inpaint_biharmonic(work_data.astype(np.float64), bright_mask)
            logger.info("Biharmonic inpainted %d bright source pixels", bright_mask.sum())
    else:
        logger.info("No bright sources to inpaint")

    # ── Step 5: à trous on pre-cleaned data (for detection) ──
    coeffs = transform(work_data, level=n_scales)
    w_scales = [coeffs.data[j] for j in range(n_scales)]
    w1 = w_scales[0].copy()
    w2 = w_scales[1]

    # ── Debug: save scales with prefix ──
    if debug_scales:
        _pfx = f"{debug_prefix}_" if debug_prefix else ""
        scale_names = [f"w{j+1}" for j in range(n_scales)] + [f"c{n_scales}"]
        for nm, arr in zip(scale_names, coeffs.data):
            fpath = out_dir / f"{_pfx}scale_{nm}.fits"
            fits.writeto(str(fpath), arr.astype(np.float32), overwrite=True)
        fits.writeto(str(out_dir / f"{_pfx}scale_precleaned.fits"),
                     work_data.astype(np.float32), overwrite=True)
        fits.writeto(str(out_dir / f"{_pfx}scale_work_data.fits"),
                     original_work.astype(np.float32), overwrite=True)
        logger.info("Saved %d wavelet scales + precleaned + work_data to %s",
                    n_scales + 2, out_dir)

    # ── Step 6: σ_local on pre-cleaned w1/w2 (for detection only) ──
    w1_med = sk_median(w1, selem)
    sigma_local_w1 = 1.4826 * sk_median(np.abs(w1 - w1_med), selem)
    w2_med = sk_median(w2, selem)
    sigma_local_w2 = 1.4826 * sk_median(np.abs(w2 - w2_med), selem)

    # ── Step 7: w1 + w2 joint point source detection ──
    candidates_w1 = np.abs(w1) > sigma_threshold * sigma_local_w1
    candidates_w2 = np.abs(w2) > 5.0 * sigma_local_w2
    candidates_wavelet = candidates_w1 | candidates_w2
    if candidates_wavelet.any():
        labeled_w, n_labels_w = ndimage.label(candidates_wavelet)
        label_ids_w = np.arange(1, n_labels_w + 1)
        areas_w = ndimage.sum(
            np.ones_like(candidates_wavelet, dtype=np.int32), labeled_w, label_ids_w,
        )
        wavelet_mask = np.isin(labeled_w, label_ids_w[areas_w <= max_area])
    else:
        wavelet_mask = np.zeros(work_data.shape, dtype=bool)
        n_labels_w = 0
    # ── Step 7b: negative ring detection via binary_fill_holes ──
    # Rings = negative w1 surrounding positive core. fill_holes finds enclosed
    # holes; keep only holes that contain positive pixels (genuine core).
    ring_mask = np.zeros(work_data.shape, dtype=bool)
    n_holes_total = 0
    if ring_min_area > 0:
        neg = w1 < -1.0 * sigma_local_w1
        if neg.any():
            filled = ndimage.binary_fill_holes(neg)
            holes = filled & ~neg
            if holes.any():
                hole_labeled, n_holes_total = ndimage.label(holes)
                hole_ids = np.arange(1, n_holes_total + 1)
                hole_areas = ndimage.sum(
                    np.ones_like(holes, dtype=np.int32), hole_labeled, hole_ids,
                )
                # Require hole to enclose positive w1 pixels (a genuine core)
                pos = w1 > 0
                pos_in_hole = ndimage.sum(
                    pos.astype(np.int32), hole_labeled, hole_ids,
                )
                holes_with_core = hole_ids[
                    (hole_areas >= ring_min_area) & (hole_areas <= ring_max_area)
                    & (pos_in_hole > 0)
                ]
                if len(holes_with_core) > 0:
                    hole_mask = np.isin(hole_labeled, holes_with_core)
                    # Dilate hole → intersect with neg → ring mask
                    dilated = ndimage.binary_dilation(
                        hole_mask, structure=np.ones((3, 3), dtype=bool), iterations=1,
                    )
                    ring_mask = dilated & neg
    logger.info(
        "Negative ring: %d holes with core in [%d,%d], %d ring px",
        n_holes_total, ring_min_area, ring_max_area, ring_mask.sum(),
    )

    point_source_mask = wavelet_mask | bright_mask | ring_mask
    _, n_sources = ndimage.label(point_source_mask)
    logger.info("Detected %d point sources (%d wavelet + bright + %d ring, area<=%d)",
                n_sources, n_labels_w, int(ring_mask.any()), max_area)

    # ── Step 8: wavelet iterative soft-thresholding inpainting ──
    # Start from pre-cleaned work_data (biharmonic-filled) to avoid point source
    # leakage into wavelet scales during inpainting reconstruction.
    filled_work = work_data.copy()
    if n_inpaint_iters > 0 and point_source_mask.any():
        for it in range(n_inpaint_iters):
            coeffs_inp = transform(filled_work, level=n_scales)
            # Proportional lambda: remove fixed fraction of |w1|/|w2| per iteration
            lam1 = inpaint_lambda_w1 * np.abs(coeffs_inp.data[0])
            lam2 = inpaint_lambda_w2 * np.abs(coeffs_inp.data[1])
            coeffs_inp.data[0] = _soft_threshold(coeffs_inp.data[0], lam1)
            coeffs_inp.data[1] = _soft_threshold(coeffs_inp.data[1], lam2)
            reconstructed = coeffs_inp.data.sum(axis=0)
            filled_work[point_source_mask] = reconstructed[point_source_mask]
        logger.info("Wavelet inpainting: %d iterations complete", n_inpaint_iters)
    elif point_source_mask.any():
        filled_work[point_source_mask] = np.nan
        bg = ndimage.median_filter(
            np.nan_to_num(filled_work, nan=np.nanmedian(filled_work)), size=9,
        )
        filled_work[np.isnan(filled_work)] = bg[np.isnan(filled_work)]

    # ── Step 8b: refine — re-run DAOStarFinder on cleaned, catch residuals ──
    for refine_iter in range(n_refine_iters):
        # Run DAOStarFinder on current cleaned image (revert offset for detection)
        clean_detect = filled_work - offset if shifted else filled_work.copy()
        _, med_clean, rms_clean = sigma_clipped_stats(clean_detect, sigma=3.0, maxiters=5)
        # Relaxed roundness: inpainting makes residuals less circular
        refine_finder = DAOStarFinder(
            fwhm=3.0,
            threshold=refine_detection_threshold * rms_clean,
            roundlo=-0.9, roundhi=0.9,
            sharplo=0.0, sharphi=1.0,
        )
        refine_stars = refine_finder(clean_detect - med_clean)
        refine_mask = np.zeros(work_data.shape, dtype=bool)
        if refine_stars is not None and len(refine_stars) > 0:
            xs = np.round(refine_stars['xcentroid']).astype(int)
            ys = np.round(refine_stars['ycentroid']).astype(int)
            # All refine sources get the same fixed radius
            radii = np.full(len(xs), refine_radius, dtype=float)
            radii = np.clip(
                radii, bright_mask_radius_min, bright_mask_radius_max,
            ).astype(int)

            for xc, yc, r in zip(xs, ys, radii):
                if 0 <= yc < refine_mask.shape[0] and 0 <= xc < refine_mask.shape[1]:
                    rr = np.sqrt((xx - xc)**2 + (yy - yc)**2)
                    refine_mask |= rr <= r
        # Only keep sources NOT already in mask
        new_mask = refine_mask & ~point_source_mask
        if new_mask.any():
            point_source_mask |= new_mask
            # Re-inpaint: start from pre-cleaned work_data
            filled_work = work_data.copy()
            if n_inpaint_iters > 0:
                for it in range(n_inpaint_iters):
                    coeffs_inp = transform(filled_work, level=n_scales)
                    coeffs_inp.data[0] = _soft_threshold(
                        coeffs_inp.data[0], inpaint_lambda_w1 * np.abs(coeffs_inp.data[0]))
                    coeffs_inp.data[1] = _soft_threshold(
                        coeffs_inp.data[1], inpaint_lambda_w2 * np.abs(coeffs_inp.data[1]))
                    reconstructed = coeffs_inp.data.sum(axis=0)
                    filled_work[point_source_mask] = reconstructed[point_source_mask]
            else:
                filled_work[point_source_mask] = np.nan
                bg_f = ndimage.median_filter(
                    np.nan_to_num(filled_work, nan=np.nanmedian(filled_work)), size=9)
                filled_work[np.isnan(filled_work)] = bg_f[np.isnan(filled_work)]
            logger.info("Refine %d/%d: %d new sources, %d px, re-inpainted",
                        refine_iter + 1, n_refine_iters,
                        len(refine_stars) if refine_stars is not None else 0,
                        new_mask.sum())
        else:
            logger.info("Refine %d/%d: no new sources found", refine_iter + 1, n_refine_iters)

    # Re-count sources after refine
    _, n_sources = ndimage.label(point_source_mask)

    # ── Step 9: sigma_map via sparse-grid NaN-aware MAD ──
    # Compute sliding MAD on sparse grid (stride px), then interpolate to
    # full resolution. Dramatically faster than per-pixel generic_filter and
    # naturally avoids NaN valleys from large masks.
    w1_masked = w1_orig.copy()
    w1_masked[point_source_mask] = np.nan
    hw = mad_window // 2

    # Sparse grid
    grid_ys = np.arange(hw, w1_orig.shape[0] - hw, sigma_stride)
    grid_xs = np.arange(hw, w1_orig.shape[1] - hw, sigma_stride)
    sigma_grid = np.full((len(grid_ys), len(grid_xs)), np.nan)
    min_valid = int(mad_window**2 * 0.3)  # require ≥30% valid pixels

    for i, gy in enumerate(grid_ys):
        y1, y2 = gy - hw, gy + hw + 1
        for j, gx in enumerate(grid_xs):
            x1, x2 = gx - hw, gx + hw + 1
            win = w1_masked[y1:y2, x1:x2]
            n_valid = np.sum(~np.isnan(win))
            if n_valid < min_valid:
                continue
            # MAD from the full nan-aware window
            med = float(np.nanmedian(win))
            sigma_grid[i, j] = 1.4826 * float(np.nanmedian(np.abs(win - med)))

    # Interpolate to full resolution (linear, nearest for edge pixels)
    from scipy.interpolate import griddata
    gyv, gxv = np.meshgrid(grid_ys, grid_xs, indexing='ij')
    valid = ~np.isnan(sigma_grid)
    if valid.sum() < 4:
        # Fallback: global sigma
        sigma_map = np.full(w1_orig.shape, float(np.nanmedian(np.abs(w1_orig[~point_source_mask])) * 1.4826))
    else:
        points = np.column_stack([gyv[valid], gxv[valid]])
        values = sigma_grid[valid]
        yy_full, xx_full = np.mgrid[:w1_orig.shape[0], :w1_orig.shape[1]]
        sigma_map = griddata(points, values, (yy_full, xx_full), method='linear')
        # Fill any remaining NaN (edges) with nearest-neighbor interpolation
        if np.isnan(sigma_map).any():
            sigma_map_nn = griddata(points, values, (yy_full, xx_full), method='nearest')
            sigma_map = np.where(np.isnan(sigma_map), sigma_map_nn, sigma_map)
    logger.info("  sigma_map: stride=%d, %d grid points, %d valid → interpolated",
                sigma_stride, sigma_grid.size, valid.sum())

    # # ── Step 9b: bg_map — fast sliding median on original image ──
    # # Fill point_source_mask with global median → C-optimised
    # # ndimage.median_filter avoids the 135s generic_filter overhead.
    # data_for_bg = original_work.copy()
    # bg_global = float(np.median(data_for_bg[~point_source_mask]))
    # data_for_bg[point_source_mask] = bg_global
    # bg_map_raw = ndimage.median_filter(data_for_bg, size=mad_window)
    # # Revert offset to original flux units
    # bg_map = bg_map_raw - offset if shifted else bg_map_raw

    # ── Step 10: revert offset ──
    cleaned_data = filled_work
    if shifted:
        cleaned_data = filled_work - offset

    return {
        "cleaned_data": cleaned_data,
        "sigma_map": sigma_map,
        # "bg_map": bg_map,
        "point_source_mask": point_source_mask,
        "bright_mask": bright_mask,
        "w1_precleaned": w1,
        "n_sources": n_sources,
        # For scales diagnostic
        "work_data": original_work,
        "precleaned_work": filled_work,
        "scales": coeffs.data,  # w1..wN, cN
    }


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _load_preprocess_config(
    config_path: str | Path | None = None,
) -> dict:
    """Load preprocessing parameters from a YAML config file.

    Priority: explicit config_path > test_config_real.yaml in base dir > built-in defaults.

    Parameters
    ----------
    config_path : str, Path, or None
        Explicit path to config file. If None, looks for test_config_real.yaml.

    Returns
    -------
    dict with keys matching preprocess_glimpse_image() kwargs.
    """
    import yaml

    base_dir = Path(__file__).parent

    # Resolve config path
    if config_path is not None:
        cfg_file = Path(config_path)
        if not cfg_file.is_absolute():
            cfg_file = base_dir / cfg_file
    else:
        cfg_file = base_dir / "test_config_real.yaml"

    if not cfg_file.exists():
        logger.debug("Config file not found: %s — using built-in defaults", cfg_file)
        return {}

    with open(cfg_file) as f:
        raw = yaml.safe_load(f) or {}

    preprocess_cfg = raw.get("preprocess", {})
    if not preprocess_cfg:
        return {}

    # Map config keys to function kwargs
    params = {}
    key_map = {
        "sigma_threshold": "sigma_threshold",
        "max_area": "max_area",
        "mad_window": "mad_window",
        "sigma_stride": "sigma_stride",
        "n_scales": "n_scales",
        "detection_threshold": "detection_threshold",
        "max_ellipticity": "max_ellipticity",
        "bright_mask_radius_max": "bright_mask_radius_max",
        "bright_mask_radius_min": "bright_mask_radius_min",
        "peak_threshold_low": "peak_threshold_low",
        "peak_threshold_high": "peak_threshold_high",
        "peak_radius_low": "peak_radius_low",
        "peak_radius_mid": "peak_radius_mid",
        "peak_radius_high": "peak_radius_high",
        "refine_radius": "refine_radius",
        "bright_fill_method": "bright_fill_method",
        "bright_fill_expand": "bright_fill_expand",
        "bright_fill_outlier_sigma": "bright_fill_outlier_sigma",
        "ring_min_area": "ring_min_area",
        "ring_max_area": "ring_max_area",
        "n_inpaint_iters": "n_inpaint_iters",
        "inpaint_lambda_w1": "inpaint_lambda_w1",
        "n_refine_iters": "n_refine_iters",
        "refine_detection_threshold": "refine_detection_threshold",
        "inpaint_lambda_w2": "inpaint_lambda_w2",
        "debug_scales": "debug_scales",
    }
    for cfg_key, fn_key in key_map.items():
        if cfg_key in preprocess_cfg:
            params[fn_key] = preprocess_cfg[cfg_key]

    # output_dir comes from pipeline section
    pipeline_cfg = raw.get("pipeline", {})
    if "output_dir" in pipeline_cfg:
        params["output_dir"] = str(base_dir / pipeline_cfg["output_dir"])

    logger.info("Loaded preprocess config: %s", params)
    return params


# ---------------------------------------------------------------------------
# Programmatic entry point (for run_test_plan_real.py)
# ---------------------------------------------------------------------------


def preprocess_single_bubble(
    name: str,
    fits_science: Path,
    output_dir: Path,
    config: dict,
) -> dict:
    """预处理单个 bubble：加载科学图像 → 去点源 → 保存 cleaned/sigma FITS + 诊断图。

    Parameters
    ----------
    name : str
        Bubble 名称，如 "N131"。
    fits_science : Path
        科学图像 FITS 文件的绝对路径。
    output_dir : Path
        输出文件保存目录（cleaned/sigma FITS + diagnostic PNG 写入此处）。
    config : dict
        preprocess 段的参数字典（key 为函数 kwarg 名，如 sigma_threshold 等）。

    Returns
    -------
    dict with keys:
        name : str
        cleaned_path : Path       — cleaned FITS
        sigma_path : Path         — sigma_map FITS
        diagnostic_path : Path    — 2×2 诊断 PNG
        n_sources : int
        n_bright_sources : int
        sigma_mean : float
        sigma_median : float
        elapsed_sec : float
    """
    from astropy.visualization import ZScaleInterval
    import time

    # ── 1. Load FITS ──
    logger.info("preprocess_single_bubble: %s → %s", name, fits_science)
    with fits.open(str(fits_science)) as hdul:
        data = hdul[0].data.astype(np.float64)
        if data.ndim == 3:
            data = data[0]
        original_header = hdul[0].header.copy()
    for _strip_key in ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
                        "EXTEND", "BSCALE", "BZERO", "BLANK", "DATAMIN", "DATAMAX"):
        original_header.pop(_strip_key, None)
    logger.info("  image shape: %s, range: [%.2f, %.2f]", data.shape, data.min(), data.max())

    # ── 2. Run preprocessing ──
    if config.get("debug_scales"):
        config["output_dir"] = str(output_dir)  # ensure scales go to preprocess dir
    t0 = time.perf_counter()
    result = preprocess_glimpse_image(data, **config, debug_prefix=name)
    elapsed = time.perf_counter() - t0

    # ── 3. Save FITS ──
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = output_dir / f"{name}_8um_cleaned.fits"
    sigma_path = output_dir / f"{name}_8um_sigma.fits"

    fits.PrimaryHDU(
        data=result["cleaned_data"].astype(np.float32),
        header=original_header,
    ).writeto(str(cleaned_path), overwrite=True)
    fits.PrimaryHDU(
        data=result["sigma_map"].astype(np.float32),
        header=original_header,
    ).writeto(str(sigma_path), overwrite=True)

    logger.info("  saved: %s", cleaned_path.name)
    logger.info("  saved: %s", sigma_path.name)

    # ── 4. Generate 2×2 diagnostic plot ──
    diagnostic_path = output_dir / f"{name}_preprocess_diagnostic.png"
    zscale = ZScaleInterval()
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    # Panel 1: Original
    v1, v2 = zscale.get_limits(data)
    axes[0, 0].imshow(data, cmap="viridis", vmin=v1, vmax=v2, origin="lower")
    axes[0, 0].set_title(f"Original ({name}, GLIMPSE 8μm)")

    # Panel 2: Point source mask
    ps_mask = result.get("point_source_mask", np.zeros_like(data, dtype=bool))
    axes[0, 1].imshow(ps_mask.astype(float), cmap="Reds", origin="lower")
    axes[0, 1].set_title(f"Point Source Mask — {name}")

    # Panel 3: Cleaned
    cleaned = result["cleaned_data"]
    v1, v2 = zscale.get_limits(cleaned)
    axes[1, 0].imshow(cleaned, cmap="viridis", vmin=v1, vmax=v2, origin="lower")
    axes[1, 0].set_title(f"Cleaned ({name})")

    # Panel 4: sigma_map
    sigma_map = result["sigma_map"]
    v1, v2 = zscale.get_limits(sigma_map)
    im = axes[1, 1].imshow(sigma_map, cmap="viridis", vmin=v1, vmax=v2, origin="lower")
    axes[1, 1].set_title(f"σ_map ({name})")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(diagnostic_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  saved diagnostic: %s", diagnostic_path.name)

    # ── 4b. Generate scales diagnostic + persist for --only-plot ──
    scales = result.get("scales")
    work_raw = result.get("work_data")
    precleaned_raw = result.get("precleaned_work")
    if scales is not None and len(scales) >= 6 and work_raw is not None:
        n_sc = len(scales) - 1  # w1..wN, cN
        scale_labels = [f"w{j+1}" for j in range(n_sc)] + [f"c{n_sc}"]
        panels = [work_raw, precleaned_raw] + [scales[j] for j in range(n_sc)] + [scales[-1]]
        panel_titles = ["work_data", "precleaned"] + scale_labels
        file_labels = ["work_data", "precleaned"] + scale_labels

        # Save scale FITS for --only-plot
        for label, arr in zip(file_labels, panels):
            fpath = output_dir / f"{name}_scale_{label}.fits"
            fits.PrimaryHDU(data=arr.astype(np.float32), header=original_header).writeto(
                str(fpath), overwrite=True)

        # Diagnostic plot
        scales_path = output_dir / f"{name}_scales_diagnostic.png"
        fig2, axs = plt.subplots(2, 4, figsize=(14, 7))
        for idx, (ax, arr, tl) in enumerate(zip(axs.flat, panels, panel_titles)):
            v1, v2 = zscale.get_limits(arr)
            ax.imshow(arr, cmap="viridis", vmin=v1, vmax=v2, origin="lower")
            ax.set_title(tl)
            ax.set_xticks([]); ax.set_yticks([])
        fig2.tight_layout()
        fig2.savefig(scales_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        logger.info("  saved scales diagnostic: %s", scales_path.name)

    # ── 5. Save intermediates for --only-plot mode ──
    w1_precleaned = result.get("w1_precleaned")
    if w1_precleaned is not None:
        w1_path = output_dir / f"{name}_w1_precleaned.fits"
        fits.PrimaryHDU(
            data=w1_precleaned.astype(np.float32),
            header=original_header,
        ).writeto(str(w1_path), overwrite=True)
        logger.info("  saved: %s", w1_path.name)
    else:
        w1_path = None

    ps_mask = result.get("point_source_mask")
    if ps_mask is not None:
        mask_path = output_dir / f"{name}_ps_mask.fits"
        fits.PrimaryHDU(
            data=ps_mask.astype(np.uint8),
            header=original_header,
        ).writeto(str(mask_path), overwrite=True)
        logger.info("  saved: %s", mask_path.name)
    else:
        mask_path = None

    # ── 6. Save result summary for --only-plot mode ──
    import json as _json
    rr = {
        "name": name,
        "n_sources": result["n_sources"],
        "sigma_mean": float(sigma_map.mean()),
        "sigma_median": float(np.median(sigma_map)),
        "elapsed_sec": elapsed,
        "cleaned_path": str(cleaned_path),
        "sigma_path": str(sigma_path),
        "w1_precleaned_path": str(w1_path) if w1_path else None,
        "ps_mask_path": str(mask_path) if mask_path else None,
        "diagnostic_path": str(diagnostic_path),
    }
    result_json_path = output_dir / f"{name}_result.json"
    result_json_path.write_text(_json.dumps(rr, indent=2, default=str), encoding="utf-8")
    logger.info("  saved: %s", result_json_path.name)

    return {
        "name": name,
        "cleaned_path": cleaned_path,
        "sigma_path": sigma_path,
        "diagnostic_path": diagnostic_path,
        "w1_precleaned_path": w1_path,
        "ps_mask_path": mask_path,
        "result_json_path": result_json_path,
        "n_sources": result["n_sources"],
        "n_bright_sources": result.get("n_bright_sources", 0),
        "sigma_mean": float(sigma_map.mean()),
        "sigma_median": float(np.median(sigma_map)),
        "elapsed_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# CLI test entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser(
        description="GLIMPSE 8μm point source removal + noise estimation"
    )
    p.add_argument(
        "--config",
        default=None,
        help="YAML config file (default: test_config_real.yaml)",
    )
    p.add_argument(
        "--name",
        default="N131",
        help="Bubble name to process (from download_manifest.json)",
    )
    p.add_argument(
        "--sigma-threshold",
        type=float,
        default=None,
        help="Detection threshold on w1 (units of sigma_local)",
    )
    p.add_argument(
        "--max-area",
        type=int,
        default=None,
        help="Max connected area (px) for a point source",
    )
    p.add_argument(
        "--mad-window",
        type=int,
        default=None,
        help="Sliding MAD window size (px)",
    )
    p.add_argument(
        "--sigma-stride",
        type=int,
        default=None,
        help="Sparse grid stride for sigma_map (default: 10, larger=faster)",
    )
    p.add_argument(
        "--debug-scales",
        action="store_true",
        default=None,
        help="Save all à trous scales as FITS (default from config or True)",
    )
    p.add_argument(
        "--no-debug-scales",
        action="store_false",
        dest="debug_scales",
        help="Disable saving à trous scale FITS files",
    )
    p.add_argument(
        "--peak-threshold-low",
        type=float,
        default=None,
        help="Peak threshold between low/mid bins (default: 1000)",
    )
    p.add_argument(
        "--peak-threshold-high",
        type=float,
        default=None,
        help="Peak threshold between mid/high bins (default: 2000)",
    )
    p.add_argument(
        "--peak-radius-low",
        type=float,
        default=None,
        help="Mask radius for low-peak sources (default: 5.0 px)",
    )
    p.add_argument(
        "--peak-radius-mid",
        type=float,
        default=None,
        help="Mask radius for mid-peak sources (default: 10.0 px)",
    )
    p.add_argument(
        "--peak-radius-high",
        type=float,
        default=None,
        help="Mask radius for high-peak sources (default: 40.0 px)",
    )
    p.add_argument(
        "--refine-radius",
        type=float,
        default=None,
        help="Mask radius for all refine sources (default: 10.0 px)",
    )
    p.add_argument(
        "--bright-fill-method",
        default=None,
        help="Bright source fill method: biharmonic (default) or bg_noise",
    )
    p.add_argument(
        "--bright-fill-expand",
        type=int,
        default=None,
        help="Pixels to expand mask for bg_noise annulus (default: 3)",
    )
    p.add_argument(
        "--bright-fill-outlier-sigma",
        type=float,
        default=None,
        help="Outlier removal sigma for bg_noise annulus (default: 3.0)",
    )
    p.add_argument(
        "--inpaint-iters",
        type=int,
        default=None,
        help="Wavelet inpainting iterations (default: 10, 0=skip)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files (default from config or test_plots_real)",
    )
    args = p.parse_args()
    base_dir = Path(__file__).parent

    # ── Load params: CLI > config file > built-in defaults ──
    config_params = _load_preprocess_config(args.config)

    sigma_thr = (
        args.sigma_threshold
        if args.sigma_threshold is not None
        else config_params.get("sigma_threshold", 3.0)
    )
    max_a = (
        args.max_area
        if args.max_area is not None
        else config_params.get("max_area", 200)
    )
    mad_w = (
        args.mad_window
        if args.mad_window is not None
        else config_params.get("mad_window", 101)
    )
    sig_stride = (
        args.sigma_stride
        if args.sigma_stride is not None
        else config_params.get("sigma_stride", 10)
    )
    n_sc = config_params.get("n_scales", 5)
    det_thr = config_params.get("detection_threshold", 3.0)
    max_ell = config_params.get("max_ellipticity", 0.3)
    mask_rad_max = config_params.get("bright_mask_radius_max", 35)
    mask_rad_min = config_params.get("bright_mask_radius_min", 3)
    pk_thr_lo = (
        args.peak_threshold_low
        if args.peak_threshold_low is not None
        else config_params.get("peak_threshold_low", 1000.0)
    )
    pk_thr_hi = (
        args.peak_threshold_high
        if args.peak_threshold_high is not None
        else config_params.get("peak_threshold_high", 2000.0)
    )
    pk_r_lo = (
        args.peak_radius_low
        if args.peak_radius_low is not None
        else config_params.get("peak_radius_low", 5.0)
    )
    pk_r_mid = (
        args.peak_radius_mid
        if args.peak_radius_mid is not None
        else config_params.get("peak_radius_mid", 10.0)
    )
    pk_r_hi = (
        args.peak_radius_high
        if args.peak_radius_high is not None
        else config_params.get("peak_radius_high", 40.0)
    )
    ref_r = (
        args.refine_radius
        if args.refine_radius is not None
        else config_params.get("refine_radius", 10.0)
    )
    fill_method = (
        args.bright_fill_method
        if args.bright_fill_method is not None
        else config_params.get("bright_fill_method", "biharmonic")
    )
    fill_expand = (
        args.bright_fill_expand
        if args.bright_fill_expand is not None
        else config_params.get("bright_fill_expand", 3)
    )
    fill_sigma = (
        args.bright_fill_outlier_sigma
        if args.bright_fill_outlier_sigma is not None
        else config_params.get("bright_fill_outlier_sigma", 3.0)
    )
    ring_area_min = config_params.get("ring_min_area", 5)
    ring_area_max = config_params.get("ring_max_area", 1000)
    n_iter = (
        args.inpaint_iters
        if args.inpaint_iters is not None
        else config_params.get("n_inpaint_iters", 10)
    )
    lam_w1 = config_params.get("inpaint_lambda_w1", 0.3)
    lam_w2 = config_params.get("inpaint_lambda_w2", 0.3)
    n_refine = config_params.get("n_refine_iters", 1)
    refine_det_thr = config_params.get("refine_detection_threshold", 2.5)
    debug_scales = (
        args.debug_scales
        if args.debug_scales is not None
        else config_params.get("debug_scales", True)
    )

    scale_output_dir = (
        args.output_dir
        if args.output_dir is not None
        else config_params.get("output_dir", str(base_dir / "test_plots_real"))
    )

    print(f"Parameters: sigma_thr={sigma_thr} max_area={max_a} mad_win={mad_w} sigma_stride={sig_stride} n_scales={n_sc}")
    print(f"  detection_thr={det_thr} max_roundness={max_ell} mask_radius=[{mask_rad_min},{mask_rad_max}]")
    print(f"  peak bins: <{pk_thr_lo}→{pk_r_lo}, {pk_thr_lo}-{pk_thr_hi}→{pk_r_mid}, ≥{pk_thr_hi}→{pk_r_hi}, refine→{ref_r}")
    print(f"  bright fill: {fill_method}" + (f" expand={fill_expand} outlier_sigma={fill_sigma}" if fill_method == "bg_noise" else ""))
    print(f"  inpaint_iters={n_iter} lam_w1={lam_w1} lam_w2={lam_w2}")
    print(f"Debug: scales={debug_scales}, output_dir={scale_output_dir}")

    # ── Resolve output directory ──
    from pathlib import Path as _Path
    preprocess_out = _Path(scale_output_dir) / "preprocess"

    # ── Find bubble in manifest ──
    manifest_path = base_dir / "test_plots_real" / "images" / "download_manifest.json"
    if not manifest_path.exists():
        manifest_path = base_dir / "real_data" / "download_manifest.json"  # legacy
    with open(manifest_path) as f:
        manifest = json.load(f)
    entry = [e for e in manifest if e["name"] == args.name]
    if not entry:
        raise SystemExit(f"{args.name} not found in download_manifest.json")
    entry = entry[0]

    fits_path = _Path(entry["fits_science"])
    if not fits_path.is_absolute():
        fits_path = base_dir / fits_path

    # ── Build config dict for preprocess_single_bubble ──
    pp_config = {
        "sigma_threshold": sigma_thr,
        "max_area": max_a,
        "mad_window": mad_w,
        "sigma_stride": sig_stride,
        "n_scales": n_sc,
        "detection_threshold": det_thr,
        "max_ellipticity": max_ell,
        "bright_mask_radius_max": mask_rad_max,
        "bright_mask_radius_min": mask_rad_min,
        "peak_threshold_low": pk_thr_lo,
        "peak_threshold_high": pk_thr_hi,
        "peak_radius_low": pk_r_lo,
        "peak_radius_mid": pk_r_mid,
        "peak_radius_high": pk_r_hi,
        "refine_radius": ref_r,
        "bright_fill_method": fill_method,
        "bright_fill_expand": fill_expand,
        "bright_fill_outlier_sigma": fill_sigma,
        "ring_min_area": ring_area_min,
        "ring_max_area": ring_area_max,
        "n_inpaint_iters": n_iter,
        "inpaint_lambda_w1": lam_w1,
        "inpaint_lambda_w2": lam_w2,
        "n_refine_iters": n_refine,
        "refine_detection_threshold": refine_det_thr,
        "debug_scales": debug_scales,
        "output_dir": scale_output_dir if debug_scales else None,
    }

    summary = preprocess_single_bubble(args.name, fits_path, preprocess_out, pp_config)
    elapsed = summary["elapsed_sec"]

    print(f"\n{'='*50}")
    print(f"{args.name} Preprocessing Results")
    print(f"{'='*50}")
    print(f"Point sources detected: {summary['n_sources']}")
    print(f"σ_map mean: {summary['sigma_mean']:.4f}")
    print(f"σ_map median: {summary['sigma_median']:.4f}")
    print(f"Execution time: {elapsed:.1f}s")
    print(f"Cleaned: {summary['cleaned_path']}")
    print(f"Sigma:   {summary['sigma_path']}")
    print(f"{'='*50}")
