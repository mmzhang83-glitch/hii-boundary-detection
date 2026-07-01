"""Download Spitzer GLIMPSE 8μm images for each bubble in bubble_catalog.csv.

Queries IRSA SIA across multiple Spitzer collections (glimpse, mipsgal, seip),
downloads multi-part images, mosaics with SWarp, and reprojects onto a standard
TAN-projection grid centred on each bubble.

Download logic adapted from getspitzer.py:
  - Multi-collection SIA query with pickle caching
  - Band filtering via energy_bandpassname (not URL patterns)
  - Quality-based row selection (skip short / prefer median for SEIP;
    prefer corr for non-SEIP)
  - Multi-part download → SWarp mosaic → reproject_to_box

Usage:
    python download_glimpse_images.py [catalog_path] [image_dir] [fov_factor]

Returns
    list[dict] — manifest written to {image_dir}/download_manifest.json
"""

import json
import logging
import os
import pickle
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.utils.data import download_file
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

try:
    from astroquery.ipac.irsa import Irsa
except ImportError:  # pre-0.4.7 astroquery
    from astroquery.irsa import Irsa  # noqa: F811

from reproject import reproject_interp

warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)

logger = logging.getLogger("hii_boundary.real_data")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RADIUS_CANDIDATES = ["<R>", "R", "Radius", "rad", "AvgR"]
NAME_CANDIDATES = ["Name", "CPA2006", "___", "Seq"]

# Spitzer collections queried via SIA (per getspitzer.py)
SIA_COLLECTIONS = ["spitzer_glimpse", "spitzer_mipsgal", "spitzer_seip"]

# IRAC Channel 4 (8 μm) — energy_bandpassname value
IRAC4 = "IRAC4"

# dataproduct_subtype substrings that indicate an uncertainty / error frame
UNC_SUBSTRINGS = ["uncertainty", "sigma", "error", "rms", "noise"]


# ===================================================================
# Helpers — catalog parsing
# ===================================================================


def _resolve_radius_column(table: Table) -> str | None:
    """Find the mean-radius column under various possible names."""
    for col in table.colnames:
        for candidate in RADIUS_CANDIDATES:
            if col.strip() == candidate or col.strip().lower() == candidate.lower():
                return col
    return None


def _resolve_name(row, table: Table, idx: int) -> str:
    """Return a human-readable bubble name.

    Tries *NAME_CANDIDATES* columns in order; falls back to ``bubble_{idx:03d}``.
    """
    for col in table.colnames:
        for candidate in NAME_CANDIDATES:
            if candidate in col:
                val = str(row[col]).strip()
                if val:
                    return val
    return f"bubble_{idx:03d}"


# ===================================================================
# Helpers — FITS metadata
# ===================================================================


def _extract_pixel_scale(header: fits.Header) -> float:
    """Extract pixel scale in arcmin / pixel from a FITS header.

    Resolution order
    ----------------
    1. ``CDELT1``                              (deg → × 60)
    2. sqrt(|det(CD matrix)|)                   (deg → × 60)
    3. ``CD1_1`` or ``PC1_1``                  (deg → × 60, isotropic approx)
    """
    if "CDELT1" in header:
        return abs(header["CDELT1"]) * 60.0

    cd_keys = ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]
    if all(k in header for k in cd_keys):
        cd = np.array(
            [
                [header["CD1_1"], header["CD1_2"]],
                [header["CD2_1"], header["CD2_2"]],
            ]
        )
        det = np.linalg.det(cd)
        return float(np.sqrt(abs(det)) * 60.0)

    for key in ["CD1_1", "PC1_1"]:
        if key in header:
            return abs(header[key]) * 60.0

    raise KeyError(
        "Cannot determine pixel scale from FITS header: "
        "no CDELT1, CD matrix, CD1_1, or PC1_1 found"
    )


def _compute_pixel_coords(header: fits.Header, coord_icrs: SkyCoord):
    """Return (xc, yc) float pixel coordinates for *coord_icrs* using WCS."""
    w = WCS(header)
    if w.naxis >= 3:
        try:
            w = w.celestial
        except Exception:
            axis_types = w.get_axis_types()
            for ax in range(w.naxis):
                if axis_types[ax].get("coordinate_type") is not None and "celestial" not in str(
                    axis_types[ax].get("coordinate_type", "")
                ):
                    w = w.dropaxis(ax)
    px, py = w.world_to_pixel(coord_icrs)
    return float(px), float(py)


# ===================================================================
# SIA query — multi-collection + pickle caching (per getspitzer.py)
# ===================================================================


def _query_sia_cached(
    coord_icrs: SkyCoord,
    search_size: u.Quantity,
    cache_dir: str,
    cache_key: str,
) -> Table:
    """Query multiple Spitzer collections via SIA, with pickle caching.

    Queries *SIA_COLLECTIONS* and stacks results with ``vstack``.
    Caches the combined table to avoid repeated queries on re-run.

    Parameters
    ----------
    coord_icrs : SkyCoord
        Centre coordinate in ICRS frame.
    search_size : Quantity
        SIA search radius (angular).
    cache_dir : str
        Directory for pickle cache files.
    cache_key : str
        Unique key for this query (e.g. bubble name).

    Returns
    -------
    Table
        Stacked SIA results; may be empty (len=0) if nothing found.
    """
    cache_path = os.path.join(cache_dir, f"sia_{cache_key}.pkl")

    if os.path.isfile(cache_path):
        logger.info("  Loading cached SIA results ← %s", os.path.basename(cache_path))
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    irsa = Irsa()
    irsa.ROW_LIMIT = 20000

    tables: list[Table] = []
    #pdb.set_trace()
    for collection in SIA_COLLECTIONS:
        try:
            tbl = irsa.query_sia(pos=(coord_icrs, search_size), collection=collection)
            if tbl is not None and len(tbl) > 0:
                logger.info("  %s: %d images", collection, len(tbl))
                tables.append(tbl)
            else:
                logger.info("  %s: 0 images", collection)
        except Exception as exc:
            logger.warning("  %s query failed: %s", collection, exc)
        
    #pdb.set_trace()

    if not tables:
        # Fallback — spitzer_seip alone
        logger.info("  Fallback: spitzer_seip only")
        try:
            tbl = irsa.query_sia(pos=(coord_icrs, search_size), collection="spitzer_seip")
            if tbl is not None and len(tbl) > 0:
                tables = [tbl]
        except Exception:
            pass

    image_table = vstack(tables, join_type="outer") if len(tables) > 1 else (tables[0] if tables else Table())

    with open(cache_path, "wb") as f:
        pickle.dump(image_table, f)
    logger.info("  Cached SIA → %s  (%d rows)", os.path.basename(cache_path), len(image_table))

    return image_table


# ===================================================================
# Row selection (per getspitzer.py quality logic)
# ===================================================================


def _select_science_rows(image_table: Table, bandpass: str) -> Table:
    """Select science-quality rows for *bandpass* using getspitzer.py criteria.

    Filter chain
    ------------
    1. ``energy_bandpassname`` == *bandpass*
    2. ``dataproduct_subtype`` == "science"
    3. Pixel-scale gate: keep SEIP; keep non-SEIP only if ``s_pixel_scale > 1``
       (filters out high-res GLIMPSE single-epoch images at ~0.6 arcsec)
    4. URL quality:
       - **SEIP**:  skip ``short``, keep ``median``
       - **non-SEIP**: skip rows without ``corr`` in the URL (prefer calibrated
         mosaics over single-epoch frames)
    """
    if len(image_table) == 0:
        return image_table

    #pdb.set_trace()
    # --- 1 & 2: bandpass + subtype -------------------------------------------
    has_bp = "energy_bandpassname" in image_table.colnames
    has_st = "dataproduct_subtype" in image_table.colnames

    if has_bp:
        bp_ok = np.array([str(v).strip() == bandpass for v in image_table["energy_bandpassname"]])
    else:
        bp_ok = np.ones(len(image_table), dtype=bool)

    if has_st:
        st_ok = np.array([str(v).lower() == "science" for v in image_table["dataproduct_subtype"]])
    else:
        st_ok = np.ones(len(image_table), dtype=bool)

    rows = image_table[bp_ok & st_ok]

    #pdb.set_trace()
    if len(rows) == 0:
        return rows

    # --- 3: pixel-scale gate -------------------------------------------------
    has_coll = "obs_collection" in rows.colnames
    has_ps = "s_pixel_scale" in rows.colnames

    if has_coll and has_ps:
        is_seip = np.array(
            [str(v) == "spitzer_seip" for v in rows["obs_collection"]]
        )
        ps = np.array([float(v) for v in rows["s_pixel_scale"]])
        keep = is_seip | ((~is_seip) & (ps > 1.0))
        rows = rows[keep]

    if len(rows) == 0:
        return rows

    #pdb.set_trace()
    # --- 4: URL quality ------------------------------------------------------
    if "access_url" in rows.colnames and has_coll:
        final_keep = np.ones(len(rows), dtype=bool)
        for i, row in enumerate(rows):
            url = str(row["access_url"]).lower()
            if str(row["obs_collection"]) == "spitzer_seip":
                # SEIP: skip short-exposure, keep median mosaics
                if "short" in url or "median" not in url:
                    final_keep[i] = False
            else:
                # non-SEIP (glimpse / mipsgal): prefer calibrated (corr) frames
                if "v3.5" not in url:
                    final_keep[i] = False
        rows = rows[final_keep]

    # --- 5: prefer non-SEIP over SEIP -----------------------------------------
    # spitzer_seip is a reprocessed product; GLIMPSE/MIPSGAL originals are
    # preferred.  Only fall back to SEIP if nothing else is available.
    if has_coll and len(rows) > 0:
        is_seip = np.array([str(v) == "spitzer_seip" for v in rows["obs_collection"]])
        non_seip = rows[~is_seip]
        if len(non_seip) > 0:
            rows = non_seip

    return rows


def _select_uncertainty_rows(image_table: Table, bandpass: str) -> Table:
    """Select uncertainty/error rows for *bandpass*.

    Simpler than the science filter — fewer rows available, so only
    bandpass + subtype matching is applied (no pixel-scale or URL gate).
    """
    if len(image_table) == 0:
        return image_table

    has_bp = "energy_bandpassname" in image_table.colnames
    has_st = "dataproduct_subtype" in image_table.colnames

    if has_bp:
        bp_ok = np.array([str(v).strip() == bandpass for v in image_table["energy_bandpassname"]])
    else:
        bp_ok = np.ones(len(image_table), dtype=bool)

    if has_st:
        st_ok = np.array(
            [any(u in str(v).lower() for u in UNC_SUBSTRINGS) for v in image_table["dataproduct_subtype"]]
        )
    else:
        st_ok = np.zeros(len(image_table), dtype=bool)

    return image_table[bp_ok & st_ok]


# ===================================================================
# SWarp mosaic + reproject_to_box
# ===================================================================


def _swarp_mosaic(image_paths: list[str], output_path: str) -> None:
    """Mosaic multiple FITS images into one with SWarp.

    Uses bilinear resampling, equatorial celestial coordinates, background
    subtraction, and a 512-pixel background mesh.
    """
    img_input = " ".join(image_paths)
    cmd = (
        f"swarp {img_input} -IMAGEOUT_NAME {output_path}"
        " -CELESTIAL_TYPE EQUATORIAL -RESAMPLE Y"
        " -RESAMPLING_TYPE BILINEAR -SUBTRACT_BACK Y -WRITE_XML N"
        " -BACK_SIZE 512"
    )
    logger.info("  SWarp: %d parts → %s", len(image_paths), os.path.basename(output_path))
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"SWarp failed with exit code {ret}")


def _reproject_to_box(
    center_coord: SkyCoord,
    size: u.Quantity,
    input_fits_path: str,
    projection_type: str = "TAN",
) -> fits.PrimaryHDU:
    """Reproject a FITS image onto a standard square grid.

    The output image is centred on *center_coord*, has side length *size*,
    and uses the *projection_type* WCS projection.

    Pixel scale is inherited from the input image (average of CDELTs).
    """
    if not os.path.exists(input_fits_path):
        raise FileNotFoundError(f"Input FITS not found: {input_fits_path}")

    logger.info("  Reproject: %s", os.path.basename(input_fits_path))
    with fits.open(input_fits_path) as hdul:
        input_hdu = hdul[0]
        input_wcs = WCS(input_hdu.header)
        input_data = input_hdu.data

    # Preserve input pixel scale
    scales = proj_plane_pixel_scales(input_wcs)
    pixel_scale_deg = float(np.mean(scales)) * u.deg / u.pixel

    naxis = int(np.round((size / pixel_scale_deg).to_value(u.pixel)))

    target_wcs = WCS(naxis=2)
    target_wcs.wcs.crval = [center_coord.ra.deg, center_coord.dec.deg]
    target_wcs.wcs.crpix = [naxis / 2.0 + 0.5, naxis / 2.0 + 0.5]
    target_wcs.wcs.ctype = [f"RA---{projection_type}", f"DEC--{projection_type}"]
    target_wcs.wcs.cdelt = [-pixel_scale_deg.value, pixel_scale_deg.value]

    target_header = target_wcs.to_header()
    target_header["NAXIS1"] = naxis
    target_header["NAXIS2"] = naxis
    target_header["NAXIS"] = 2

    output_data, _footprint = reproject_interp((input_data, input_wcs), target_header)
    logger.info("  Reprojected → %d × %d px", naxis, naxis)

    return fits.PrimaryHDU(data=output_data, header=target_header)


def _robust_download(
    url: str,
    *,
    cache: bool = True,
    timeout_per_chunk: float = 30.0,
    retries: int = 5,
    retry_delay: float = 10.0,
) -> str:
    """Download a file with retry on connection errors / short reads.

    Uses ``requests`` for streaming download with a ``tqdm`` progress bar.
    Retries are folded into the same bar — on failure the bar resets and
    continues, with the attempt counter shown in the bar description.

    Parameters
    ----------
    url : str
        The URL to download.
    cache : bool
        If True, use astropy's download cache (instant return on cache hit).
    timeout_per_chunk : float
        Timeout in seconds for each chunk read.
    retries : int
        Maximum number of download attempts.
    retry_delay : float
        Seconds to wait between retries.

    Returns
    -------
    str
        Path to the downloaded (or cached) file.
    """
    import time as _time

    # --- streaming download with retry + tqdm ----------------------------------
    import requests as _requests
    from tqdm import tqdm

    # Extract a short filename for the progress-bar label
    fname = url.rstrip("/").split("/")[-1][:60]

    cache_dir = os.path.expanduser(os.path.join("~", ".astropy", "cache", "download"))
    os.makedirs(cache_dir, exist_ok=True)

    last_err = None
    for attempt in range(retries):
        desc = f"  {fname}" if attempt == 0 else f"  {fname} [retry {attempt}]"
        try:
            resp = _requests.get(url, stream=True, timeout=(30, timeout_per_chunk))
            resp.raise_for_status()

            total = int(resp.headers.get("Content-Length", 0))

            import tempfile
            tmp_fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".fits")

            with os.fdopen(tmp_fd, "wb") as f, \
                 tqdm(desc=desc, total=total, unit="iB", unit_scale=True,
                      leave=False, position=0, file=sys.stderr) as pbar:
                for chunk in resp.iter_content(chunk_size=512 * 1024):  # 512 KB
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Verify file is complete (if Content-Length was provided)
            actual = os.path.getsize(tmp_path)
            if total and actual < total:
                raise OSError(f"Incomplete download: got {actual} / {total} bytes")

            # Store in astropy cache so cache=True hits next time
            if cache:
                try:
                    cached = os.path.join(cache_dir, "url", url.replace("/", "_")[:200])
                    os.makedirs(os.path.dirname(cached), exist_ok=True)
                    shutil.move(tmp_path, cached)
                    return cached
                except Exception:
                    pass
            return tmp_path

        except Exception as exc:
            last_err = exc
            if attempt < retries - 1:
                _time.sleep(retry_delay)

    raise RuntimeError(f"Download failed after {retries} attempts: {last_err}")


def _download_parts_and_assemble(
    rows: Table,
    name: str,
    kind: str,
    work_dir: str,
    coord_icrs: SkyCoord,
    output_size: u.Quantity,
    final_path: Path,
) -> None:
    """Download all parts from *rows*, SWarp-mosaic, and reproject.

    Parameters
    ----------
    rows : Table
        SIA rows to download (already filtered to one bandpass + subtype).
    name : str
        Bubble name (for logging / temp-file naming).
    kind : str
        ``"science"`` or ``"uncertainty"`` (for temp-file naming).
    work_dir : str
        Directory for scratch files (cleaned up).
    coord_icrs : SkyCoord
        Target centre for reprojection.
    output_size : Quantity
        Angular side length of the output square image.
    final_path : Path
        Destination path for the reprojected FITS file.
    """
    downloaded: list[str] = []

    for i_row in range(len(rows)):
        url = str(rows[i_row]["access_url"])
        coll = rows[i_row].get("obs_collection", "?")
        logger.info("  ↓ %s part %d/%d  [%s]  %s", kind, i_row + 1, len(rows), coll, url[:70])
        tmp = _robust_download(url, cache=True)
        part_path = os.path.join(work_dir, f"{name}_{kind}_part{i_row:03d}.fits")
        shutil.copy(tmp, part_path)
        downloaded.append(part_path)

    # --- mosaic ---------------------------------------------------------------
    if len(downloaded) == 1:
        mosaic_path = str(final_path).replace(".fits", "_mosaic_tmp.fits")
        shutil.copy(downloaded[0], mosaic_path)
        logger.info("  Single part — skip SWarp")
    else:
        mosaic_path = os.path.join(work_dir, f"{name}_{kind}_mosaic.fits")
        _swarp_mosaic(downloaded, mosaic_path)

    # --- reproject ------------------------------------------------------------
    reproj_hdu = _reproject_to_box(coord_icrs, output_size, mosaic_path)
    reproj_hdu.writeto(str(final_path), overwrite=True)

    # --- cleanup --------------------------------------------------------------
    for p in downloaded:
        try:
            os.remove(p)
        except OSError:
            pass
    if os.path.exists(mosaic_path):
        try:
            os.remove(mosaic_path)
        except OSError:
            pass


# ===================================================================
# Main entry point
# ===================================================================


def download_glimpse_images(
    catalog_path: str = "bubble_catalog.csv",
    image_dir: str = "real_data",
    fov_factor: float = 6.0,
) -> list[dict]:
    """Download Spitzer GLIMPSE 8 μm images for each bubble in *catalog_path*.

    Download strategy (adapted from getspitzer.py)
    ----------------------------------------------
    - Queries ``spitzer_glimpse``, ``spitzer_mipsgal``, ``spitzer_seip``
      via IRSA SIA; caches results as pickle files.
    - Filters science rows by ``energy_bandpassname == "IRAC4"`` with
      quality gates (pixel scale, URL patterns).
    - Downloads all matching parts → SWarp mosaic → reproject to a
      TAN-projection square grid centred on the bubble.
    - Handles uncertainty frames separately.

    Parameters
    ----------
    catalog_path : str
        Path to the CSV produced by ``fetch_bubble_catalog.py``.
    image_dir : str
        Directory where FITS files and the manifest are written.
    fov_factor : float
        Output image side length = *R_arcmin* × *fov_factor*.

    Returns
    -------
    list[dict]
        Manifest list; also saved to ``{image_dir}/download_manifest.json``.
    """
    # ------------------------------------------------------------------
    # 1. Read catalog
    # ------------------------------------------------------------------
    catalog = Table.read(catalog_path, format="ascii.csv")
    logger.info("Read %d rows from %s", len(catalog), catalog_path)
    logger.info("Columns: %s", catalog.colnames)

    radius_col = _resolve_radius_column(catalog)
    if radius_col is None:
        raise KeyError(f"Cannot find radius column among {catalog.colnames}")
    logger.info("Radius column: '%s'", radius_col)

    out_dir = Path(image_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Process each bubble
    # ------------------------------------------------------------------
    manifest: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for i, row in enumerate(catalog):
        name = _resolve_name(row, catalog, i)

        # --- parse row ----------------------------------------------------
        try:
            glon = float(row["GLON"])
            glat = float(row["GLAT"])
            r_arcmin = float(row[radius_col])
        except (KeyError, ValueError) as exc:
            logger.warning("[%s] Cannot parse coordinates/radius: %s — skip", name, exc)
            skipped.append((name, f"bad coordinates/radius: {exc}"))
            continue

        logger.info("─" * 60)
        logger.info("[%s] GLON=%.3f  GLAT=%.3f  R=%.2f arcmin", name, glon, glat, r_arcmin)

        # --- coordinate conversion ----------------------------------------
        coord_gal = SkyCoord(glon, glat, frame="galactic", unit="deg")
        coord_icrs = coord_gal.icrs
        logger.info("  ICRS: %.4f°, %.4f°", coord_icrs.ra.deg, coord_icrs.dec.deg)

        # --- output paths -------------------------------------------------
        fits_science = out_dir / f"{name}_8um_science.fits"
        # fits_unc = out_dir / f"{name}_8um_unc.fits"  # 不需要下载误差图

        # --- SIA query (cached) -------------------------------------------
        search_size = 5.0 * u.arcmin  # fixed search radius (getspitzer style)
        try:
            image_table = _query_sia_cached(coord_icrs, search_size, str(out_dir), name)
        except Exception as exc:
            logger.error("[%s] SIA query failed — skip", name, exc_info=True)
            skipped.append((name, f"SIA query: {exc}"))
            continue
        
        #pdb.set_trace()
        if image_table is None or len(image_table) == 0:
            logger.warning("[%s] SIA returned no results — skip", name)
            skipped.append((name, "SIA empty"))
            continue

        logger.info("  SIA rows: %d", len(image_table))
        #pdb.set_trace()
        # --- select science rows ------------------------------------------
        science_rows = _select_science_rows(image_table, IRAC4)
        #pdb.set_trace()
        if len(science_rows) == 0:
            # Fallback: bandpass + subtype only, no quality gates
            logger.warning("[%s] No science rows after quality filter — fallback", name)
            bp = "energy_bandpassname" in image_table.colnames
            st = "dataproduct_subtype" in image_table.colnames
            bp_ok = (
                np.array([str(v).strip() == IRAC4 for v in image_table["energy_bandpassname"]])
                if bp
                else np.ones(len(image_table), dtype=bool)
            )
            st_ok = (
                np.array([str(v).lower() == "science" for v in image_table["dataproduct_subtype"]])
                if st
                else np.ones(len(image_table), dtype=bool)
            )
            science_rows = image_table[bp_ok & st_ok]

        if len(science_rows) == 0:
            logger.warning("[%s] No IRAC4 science images — skip", name)
            skipped.append((name, "no IRAC4 science image"))
            continue

        logger.info("  IRAC4 science: %d row(s)", len(science_rows))

        # --- select uncertainty rows (disabled: 不需要误差图) --------------
        # unc_rows = _select_uncertainty_rows(image_table, IRAC4)
        # if len(unc_rows) > 0:
        #     logger.info("  IRAC4 uncertainty: %d row(s)", len(unc_rows))
        #pdb.set_trace()
        # --- output image size --------------------------------------------
        output_size = r_arcmin * fov_factor * u.arcmin

        # --- download & assemble science ----------------------------------
        if fits_science.exists():
            logger.info("  ✓ Science FITS cached: %s", fits_science.name)
        else:
            try:
                _download_parts_and_assemble(
                    science_rows, name, "science",
                    str(out_dir), coord_icrs, output_size, fits_science,
                )
            except Exception as exc:
                logger.error("[%s] Science assembly failed — skip", name, exc_info=True)
                skipped.append((name, f"science: {exc}"))
                continue

        # --- uncertainty (disabled: 不需要下载误差图) ----------------------
        fits_unc_str: str | None = None

        #pdb.set_trace()
        # --- extract metadata from science FITS ---------------------------
        with fits.open(fits_science) as hdul:
            header = hdul[0].header
            pixel_scale = _extract_pixel_scale(header)
            logger.info("  Pixel scale: %.4f arcmin/px", pixel_scale)

            xc, yc = _compute_pixel_coords(header, coord_icrs)
            logger.info("  Centre pixel: (%.1f, %.1f)", xc, yc)

            naxis1 = header.get("NAXIS1", 0)
            naxis2 = header.get("NAXIS2", 0)

        rmax_pixel = r_arcmin / pixel_scale * 1.5
        logger.info("  rmax_pixel: %.1f", rmax_pixel)

        # --- append to manifest -------------------------------------------
        entry = {
            "name": name,
            "glon": glon,
            "glat": glat,
            "R_arcmin": r_arcmin,
            "fits_science": str(fits_science),
            "fits_uncertainty": fits_unc_str,
            "pixel_scale_arcmin": pixel_scale,
            "xc_pixel": xc,
            "yc_pixel": yc,
            "rmax_pixel": rmax_pixel,
            "image_shape": [int(naxis1), int(naxis2)],
        }
        manifest.append(entry)
        logger.info("  ✓ Done [%s]", name)

    # ------------------------------------------------------------------
    # 3. Save manifest
    # ------------------------------------------------------------------
    manifest_path = out_dir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Manifest saved → %s  (%d entries)", manifest_path, len(manifest))

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Download: %d ok  /  %d skipped", len(manifest), len(skipped))
    if skipped:
        logger.info("Skipped:")
        for nm, reason in skipped:
            logger.info("  - %s: %s", nm, reason)

    return manifest


# ===================================================================
# CLI
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,  # log → stdout, tqdm → stderr, 互不干扰
    )

    catalog_path = sys.argv[1] if len(sys.argv) > 1 else "bubble_catalog.csv"
    image_dir = sys.argv[2] if len(sys.argv) > 2 else "real_data"
    fov_factor = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0

    manifest = download_glimpse_images(
        catalog_path=catalog_path,
        image_dir=image_dir,
        fov_factor=fov_factor,
    )
    print(f"\nDownloaded {len(manifest)} images.  See {image_dir}/download_manifest.json")
