"""Fetch the Churchwell+2006 Milky Way bubble catalog from Vizier.

Queries the Vizier catalog J/ApJ/649/759/bubbles (322 Galactic bubbles),
merges returned tables, filters for closed-ring morphology (MFlags="C"),
sorts by mean radius <R>, and saves the top N rows to CSV.

Usage:
    python fetch_bubble_catalog.py [n]

Returns
    astropy.table.Table with columns: GLON, GLAT, <R>, Rin, rout, Ecc, MFlags, ...
"""

import logging
import sys
from pathlib import Path
import pdb

from astropy.table import Table, vstack
from astroquery.vizier import Vizier

log = logging.getLogger("hii_boundary.real_data")


def _resolve_radius_column(table: Table) -> str | None:
    """Find the mean radius column in the table under various possible names.

    Column <R> from Vizier may appear as ``<R>``, ``R``, ``Radius``, ``AvgR``
    or similar in edge cases.  This helper tries known candidates and returns
    the first match, or *None* if none is found.
    """
    candidates = ["<R>", "R", "Radius", "rad", "AvgR"]
    for col in table.colnames:
        for candidate in candidates:
            if col.strip() == candidate or col.strip().lower() == candidate.lower():
                return col
    return None


def fetch_bubble_catalog(n: int = 5, output_path: str = "bubble_catalog.csv") -> Table:
    """Fetch the Churchwell+2006 bubble catalog, filter & sort, save to CSV.

    Parameters
    ----------
    n : int
        Number of top rows (by <R> descending) to keep.  Default 5.
    output_path : str
        Path for the output CSV file.  All original columns are preserved.

    Returns
    -------
    astropy.table.Table
        Filtered, sorted table with *n* rows (or fewer if not enough matches).
    """
    # ------------------------------------------------------------------
    # 1. Query Vizier — ROW_LIMIT=-1 means all rows
    # ------------------------------------------------------------------
    v = Vizier()
    v.ROW_LIMIT = -1

    log.info("Querying Vizier catalog J/ApJ/649/759/bubbles ...")
    tables = v.get_catalogs("J/ApJ/649/759/bubbles")

    if not tables:
        raise RuntimeError(
            "Vizier returned no tables for catalog J/ApJ/649/759/bubbles"
        )

    log.info("Received %d table(s) from Vizier", len(tables))

    #pdb.set_trace()
    # ------------------------------------------------------------------
    # 2. Merge all returned tables (north + south, same columns)
    # ------------------------------------------------------------------
    if len(tables) == 1:
        catalog = tables[0]
    else:
        # Verify column compatibility before stacking
        master_cols = set(tables[0].colnames)
        for i, t in enumerate(tables[1:], start=2):
            if set(t.colnames) != master_cols:
                log.warning(
                    "Table %d has different columns (%s vs %s); will convert",
                    i,
                    t.colnames,
                    tables[0].colnames,
                )
        catalog = vstack(tables, metadata_conflicts="silent")

    log.info("Merged catalog: %d rows, %d columns", len(catalog), len(catalog.colnames))
    log.info("Columns: %s", catalog.colnames)
    #pdb.set_trace()
    # ------------------------------------------------------------------
    # 3. Filter to MFlags == "C" (closed ring)
    # ------------------------------------------------------------------
    if "MFlags" in catalog.colnames:
        n_before = len(catalog)
        catalog = catalog[catalog["MFlags"] == "C"]
        log.info(
            "Filtered MFlags=='C': %d / %d rows kept",
            len(catalog),
            n_before,
        )

        if len(catalog) == 0:
            raise RuntimeError("No bubbles remain after MFlags=='C' morphology filter")
    else:
        log.warning("Column 'MFlags' not found — proceeding without filter")
        log.warning("Available columns: %s", catalog.colnames)
    #pdb.set_trace()
    # ------------------------------------------------------------------
    # 4. Sort by <R> descending
    # ------------------------------------------------------------------
    radius_col = _resolve_radius_column(catalog)
    if radius_col is None:
        raise KeyError(
            f"Cannot find radius column among names in {catalog.colnames}"
        )

    log.info("Using radius column: '%s'", radius_col)
    catalog.sort(radius_col, reverse=True)

    # ------------------------------------------------------------------
    # 5. Take top-n
    # ------------------------------------------------------------------
    n_actual = min(n, len(catalog))
    catalog = catalog[:n_actual]
    log.info("Top %d rows by %s:", n_actual, radius_col)

    for row in catalog:
        log.info(
            "  GLON=%-10s  GLAT=%-10s  <%s>=%-10s  MFlags=%s",
            row["GLON"] if "GLON" in catalog.colnames else "?",
            row["GLAT"] if "GLAT" in catalog.colnames else "?",
            radius_col,
            row[radius_col],
            row["MFlags"] if "MFlags" in catalog.colnames else "?",
        )

    #pdb.set_trace()
    # ------------------------------------------------------------------
    # 6. Save to CSV
    # ------------------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.write(str(output_path), format="csv", overwrite=True)
    log.info("Saved to %s", output_path.resolve())

    return catalog


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    result = fetch_bubble_catalog(n=n)

    print(f"\n{'='*60}")
    print(f"Fetched {len(result)} bubble(s)")
    print(f"Columns: {result.colnames}")
    print(f"{'='*60}")
