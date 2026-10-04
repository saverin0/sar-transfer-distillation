"""The pairing rules. No network and no model, only arrays and tables.

Two steps
- `pair_index` works on the two index tables, before anything is fetched. It pairs by
  grid_cell, takes the closest Sentinel-1 date and applies the 7 day rule.
- `make_pair` works on the two fetched cells. It cuts Sentinel-1 by the MAP COORDINATES of the
  Sentinel-2 crop (from the affine transforms, never by pixel index) and applies the other rules.

`make_pair` never raises for a cell that breaks a rule. It returns a `records.Skip` with one of
`records.SKIP_REASONS`. The rules are checked in this fixed order and the first one that fails
is the reason that is counted

    no_s1_for_cell, gap_gt_7d, s1_not_1068_10m, crs_differs, s2_band_shape,
    crop_outside_s1, s1_nodata_in_crop, s2_nodata_in_crop, cloud_gt_10pct

A wrong ARGUMENT is not a fault of the cell. An unusable list of cloud codes or an unknown
`offset` (`make_pair`), a missing date (`remove_offset`) or a product id without a processing
baseline (`offset_by_baseline`) is a ValueError, it is never passed over in silence.

Units. Sentinel-1 goes in as linear power and comes out as dB (float16). Sentinel-2 stays the
stored digital numbers (uint16) at native resolution, on one scale. The +1000 offset is taken
off by PROCESSING BASELINE (`remove_offset_by_baseline`, the rule decided on 2026-09-30 after
the stage 0 run), unless `make_pair` is called with offset=None. Map coordinates are metres.

Band shift. The stored transforms give every band of a cell the same corner, but in the image
data the 20 m and 60 m bands do not lie where the 10 m bands lie (measured on 433 real cells
on 2026-09-30). `s2_to_grid` corrects it with the corner rule (`band_shift_rule`) when it gets
the corner and the CRS of the pair, decided 2026-09-30.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .records import (CROP_PX, PIXEL_M, S1_CELL_PX, S1_CHANNELS, S2_BANDS, S2_RES_M,
                      Pair, S1Cell, S2Cell, Skip)

PAIR_COLUMNS = ("grid_cell", "s2_url", "s2_row", "s2_product_id", "s2_datetime",
                "s1_url", "s1_row", "s1_product_id", "s1_datetime", "gap_days", "s2_year")
PIXEL_TOL_M = 1e-6       # a pixel size counts as 10 m when it is this close to 10 m
MAP_TOL_M = 1e-3         # two map positions count as the same when they are this close
FILE_FORMAT = 1          # version of the .npz layout written by `save_pair`

# The Sentinel-2 offset by processing baseline, decided 2026-09-30 from the 481 pairs of the
# stage 0 run. The reading per image is the pre-registered one of
# `stage0.offset_check`, the same numbers, so the two must not drift apart.
OFFSET_DN = 1000                     # what is taken off, the -1000 of processing baseline 04.00
OFFSET_BANDS = ("B02", "B04", "B08")     # the bands of the per-image reading
OFFSET_PERCENTILE = 0.1                  # of the non-zero pixels of each band
OFFSET_MIN_DN = 950                  # the image carries the offset when all three are at least this
BASELINE_WITH_OFFSET = (400, 599)    # N0400 to N0599 carry the offset (181 of 188 stage 0 images)
BASELINE_PER_IMAGE = "N9999"         # decided per image with the reading (17 of 117 carried it)
OFFSET_RULES = ("baseline", None)    # what `make_pair` takes as `offset`

# The band shift by the corner rule, decided 2026-09-30 from the 433 measured cells of the
# stage 0 run. Every band was cut on the pixel lattice of its own resolution, the
# stored corner rounded to the nearest multiple of it. South of the equator the false northing
# of a UTM zone is not a multiple of 60 m, so the 60 m lattice sits at 40 mod 60 m there.
LATTICE_RES_M = (10, 20, 60)         # the resolutions of the 12 bands, one lattice each
SOUTH_FALSE_NORTHING_M = 10_000_000  # y of the equator in a UTM south zone (EPSG:327xx)
UTM_NORTH_EPSG = range(32601, 32661)  # the UTM north zones, lattices at 0 mod r
UTM_SOUTH_EPSG = range(32701, 32761)  # the UTM south zones, y lattice at false northing mod r


# --------------------------------------------------------------------------- index tables

def _index_part(index: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """The columns of one index table under the names used in `pairs`, dates in UTC."""
    return pd.DataFrame({
        "grid_cell": index["grid_cell"],
        f"{prefix}_url": index["parquet_url"],
        f"{prefix}_row": index["parquet_row"],
        f"{prefix}_product_id": index["product_id"],
        # a date without a time zone is read as UTC
        f"{prefix}_datetime": pd.to_datetime(index["product_datetime"], utc=True),
    })


def pair_index(s2_index: pd.DataFrame, s1_index: pd.DataFrame,
               max_gap_days: float | None = 7.0) -> tuple[pd.DataFrame, dict]:
    """One pair per grid cell, from the two index tables of `majortom`.

    For every grid cell the Sentinel-2 row and the Sentinel-1 row with the smallest date gap
    are taken. The pair is kept when the gap is at most `max_gap_days` (7 days and 0 seconds
    passes). None or inf keeps every cell that has Sentinel-1, which is what the gap report
    of stage 0 needs. A row without a date never passes.

    Returns (pairs, counts). `pairs` has the columns of `PAIR_COLUMNS`, dates in UTC,
    gap_days as float days, sorted by grid_cell. `counts` holds
        s2_rows, s2_cells, s1_rows, cells_with_s1, no_s1_for_cell, gap_gt_7d, pairs, max_gap_days
    with s2_cells = no_s1_for_cell + gap_gt_7d + pairs. The key gap_gt_7d is the name of the
    skip reason, the limit that was used is in max_gap_days.
    """
    limit = float("inf") if max_gap_days is None else float(max_gap_days)
    s2, s1 = _index_part(s2_index, "s2"), _index_part(s1_index, "s1")
    both = s2.merge(s1, on="grid_cell", how="inner")
    gap = (both["s2_datetime"] - both["s1_datetime"]).abs()
    both["gap_days"] = gap.dt.total_seconds() / 86400.0
    both = both.sort_values(["grid_cell", "gap_days", "s2_datetime", "s1_datetime",
                             "s2_product_id", "s1_product_id"], kind="stable")
    best = both.drop_duplicates("grid_cell", keep="first")
    pairs = best[best["gap_days"] <= limit].copy()
    pairs["s2_year"] = pairs["s2_datetime"].dt.year.astype(int)
    pairs = pairs[list(PAIR_COLUMNS)].reset_index(drop=True)

    s2_cells = int(s2["grid_cell"].nunique())
    counts = {
        "s2_rows": int(len(s2_index)),
        "s2_cells": s2_cells,
        "s1_rows": int(len(s1_index)),
        "cells_with_s1": int(len(best)),
        "no_s1_for_cell": s2_cells - int(len(best)),
        "gap_gt_7d": int(len(best) - len(pairs)),
        "pairs": int(len(pairs)),
        "max_gap_days": limit,
    }
    return pairs, counts


def s2_row_of(pair_row: Mapping) -> dict:
    """The Sentinel-2 side of one row of `pairs` (a Series or a dict), for `read_s2_cell`."""
    return {"grid_cell": pair_row["grid_cell"], "product_id": pair_row["s2_product_id"],
            "product_datetime": pair_row["s2_datetime"], "parquet_url": pair_row["s2_url"],
            "parquet_row": int(pair_row["s2_row"])}


def s1_row_of(pair_row: Mapping) -> dict:
    """The Sentinel-1 side of one row of `pairs` (a Series or a dict), for `read_s1_cell`."""
    return {"grid_cell": pair_row["grid_cell"], "product_id": pair_row["s1_product_id"],
            "product_datetime": pair_row["s1_datetime"], "parquet_url": pair_row["s1_url"],
            "parquet_row": int(pair_row["s1_row"])}


def sample_pairs(pairs: pd.DataFrame, n: int = 500, seed: int = 0) -> pd.DataFrame:
    """`n` rows of `pairs`, spread over s2_year, the same rows for the same seed.

    Every year gets the same share. A year with fewer pairs gives all it has and what is left
    is shared by the other years. The draw does not depend on the order of the rows.
    Returns all rows when `pairs` has at most `n`. Sorted by s2_year, then grid_cell.
    """
    if len(pairs) == 0 or n <= 0:
        return pairs.iloc[:0].reset_index(drop=True)
    groups = {int(year): g.sort_values(["grid_cell", "s2_product_id", "s1_product_id"])
              for year, g in pairs.groupby("s2_year")}
    take = dict.fromkeys(sorted(groups), 0)
    left = min(int(n), len(pairs))
    while left > 0:
        open_years = [y for y in take if take[y] < len(groups[y])]
        share = max(left // len(open_years), 1)
        for year in open_years:
            k = min(share, len(groups[year]) - take[year], left)
            take[year] += k
            left -= k
    parts = []
    for year, k in take.items():
        rng = np.random.default_rng([int(seed), year])
        parts.append(groups[year].iloc[np.sort(rng.choice(len(groups[year]), k, replace=False))])
    return pd.concat(parts).sort_values(["s2_year", "grid_cell"]).reset_index(drop=True)


# --------------------------------------------------------------------------- one pair

def _utc_time(value: Any) -> pd.Timestamp:
    """One date as a UTC timestamp. A date without a time zone is read as UTC."""
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _gap_days(a: Any, b: Any) -> float:
    """Distance of two dates in days, nan when one of them is missing or cannot be read."""
    try:
        return abs((_utc_time(a) - _utc_time(b)).total_seconds()) / 86400.0
    except (TypeError, ValueError, AttributeError):
        return float("nan")


def _corner(transform: Any, pixel_m: float) -> tuple[float, float] | None:
    """(left, top) in metres of a north-up affine with square pixels of `pixel_m`, else None."""
    try:
        a, b, c, d, e, f = (float(v) for v in tuple(transform)[:6])
    except (TypeError, ValueError):
        return None
    if abs(a - pixel_m) > PIXEL_TOL_M or abs(e + pixel_m) > PIXEL_TOL_M or b != 0 or d != 0:
        return None
    return (c, f) if np.isfinite(c) and np.isfinite(f) else None


def _s2_problem(s2: S2Cell) -> tuple[str, tuple[float, float] | None]:
    """What is wrong with the Sentinel-2 cell ("" when nothing is) and its (left, top) corner.

    Every band must be uint16, have its native size (384, 192 or 64 px for 10, 20 or 60 m)
    and cover the same square of 3,840 m, which must be `s2.bounds`.
    """
    side_m = CROP_PX * PIXEL_M
    ref = None
    for name in S2_BANDS:
        if name not in s2.bands:
            return f"{name} is missing", None
        band, res = s2.bands[name], S2_RES_M[name]
        n = int(round(side_m / res))
        if np.shape(band) != (n, n):
            return f"{name} is {np.shape(band)}, expected {(n, n)}", None
        if getattr(band, "dtype", None) != np.uint16:
            return f"{name} is {getattr(band, 'dtype', type(band).__name__)}, expected uint16", None
        corner = _corner(s2.transforms.get(name), res)
        if corner is None:
            return f"{name} transform {s2.transforms.get(name)} is not north-up at {res} m", None
        ref = ref or corner
        if max(abs(corner[0] - ref[0]), abs(corner[1] - ref[1])) > MAP_TOL_M:
            return f"{name} starts at {corner}, {S2_BANDS[0]} at {ref}", None
    left, top = ref
    expected = (left, top - side_m, left + side_m, top)
    try:
        off = max(abs(float(v) - w) for v, w in zip(s2.bounds, expected, strict=True))
    except (TypeError, ValueError):
        off = float("inf")
    if not off <= MAP_TOL_M:
        return f"bounds {s2.bounds} do not match the transforms, expected {expected}", None
    if s2.cloud_mask is not None and np.shape(s2.cloud_mask) != (CROP_PX, CROP_PX):
        return f"cloud_mask is {np.shape(s2.cloud_mask)}, expected {(CROP_PX, CROP_PX)}", None
    return "", ref


def _cloud_codes(cloud_codes: Any) -> list[int]:
    """The cloud codes as a sorted list of int without doubles, ValueError when unusable.

    The cloud mask is uint8, so a code is a whole number from 0 to 255. An empty list, a
    text, a number with a fraction or a number outside that range can never mean what was
    meant, and the rule would count as applied without being able to skip a cell.
    """
    problem = f"cloud_codes {cloud_codes!r} must be a list of whole numbers from 0 to 255"
    if isinstance(cloud_codes, (str, bytes)):
        raise ValueError(problem)
    try:
        given = list(cloud_codes)
        codes = sorted({int(c) for c in given})
        whole = all(float(c) == int(c) for c in given)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(problem) from None
    if not codes or not whole or codes[0] < 0 or codes[-1] > 255:
        raise ValueError(problem + (", an empty list applies no rule, use None to leave "
                                    "the rule off" if not codes else ""))
    return codes


def to_db(linear: Any) -> np.ndarray:
    """Linear power to dB, 10 * log10(linear), float32, same shape.

    Zero gives -inf and a negative value gives nan, so `make_pair` checks the crop first,
    on the values as float32 (the type the dB is computed from).
    Every positive float32 lies between -449 and +386 dB, so dB never overflows float16
    (largest value 65,504). float16 keeps steps of 1/64 dB below 32 dB and 1/32 dB below 64 dB.
    """
    return (10.0 * np.log10(np.asarray(linear, dtype=np.float32))).astype(np.float32)


def make_pair(s2: S2Cell, s1: S1Cell | None, cloud_codes: Iterable[int] | None = None,
              max_cloud: float = 0.10, max_gap_days: float | None = 7.0,
              offset: str | None = "baseline") -> Pair | Skip:
    """One Sentinel-2 cell and one Sentinel-1 cell become a `Pair`, or a `Skip` with the reason.

    The Sentinel-1 window is found from the two affine transforms. Its corner is the corner of
    the Sentinel-2 crop in Sentinel-1 pixels, rounded to the nearest pixel. What rounding cut
    off goes into `meta["shift_px"]` = (rows, columns), exact position minus used position,
    between -0.5 and 0.5. So "fully inside" means inside within half a pixel.

    cloud_codes   the codes of `cloud_mask` that count as cloud (`records.CLOUD_CODES` is the
                  decided list). None = the rule is NOT applied and `meta["cloud_rule"]` =
                  "not applied". With codes, a cell without a cloud mask is a skip
                  (read_error), it is never kept unchecked. The codes must be whole numbers
                  from 0 to 255 and at least one, else ValueError, before any rule is looked
                  at. An empty list is NOT the same as None.
    max_cloud     largest cloud share of the crop that is kept (0.10 passes, more does not,
                  and with a limit that is not a number no cell passes)
    max_gap_days  largest date gap in days that is kept, None = no limit
    offset        "baseline" (the default) takes the +1000 offset off the Sentinel-2 bands of
                  an image whose processing baseline carries it (`remove_offset_by_baseline`),
                  so every stored pair is on one scale. None stores the values as read, as
                  stage 0 did. Anything else is a ValueError before any rule is looked at.
                  With the rule on, a product id without a processing baseline or with one the
                  rule does not cover is a skip (read_error), it is never kept unchecked.

    The Sentinel-1 values are checked as float32, the type the dB is computed from. So a
    value that is 0 or infinite as float32 is a skip, whatever type the cell holds.
    A Sentinel-2 pixel of 0 is the L2A no-data value. One such pixel in any of the 12 bands
    of the crop is a skip (s2_nodata_in_crop), the same kind of rule as for Sentinel-1.

    meta keys - shift_px, s1_window (row, column of the corner in the Sentinel-1 cell),
    cloud_rule, cloud_codes and cloud_fraction (only when applied), s2_product_id,
    s1_product_id, orbit_state, baseline (the Nxxyy part of the Sentinel-2 product id, None
    when the name has none), offset_removed (bool) and offset_how ("baseline", "per_image"
    for N9999, "none" with offset=None).
    """
    codes = None if cloud_codes is None else _cloud_codes(cloud_codes)
    if offset not in OFFSET_RULES:
        raise ValueError(f"offset {offset!r} must be one of {OFFSET_RULES}")
    cell = s2.grid_cell
    if s1 is None or s1.grid_cell != cell:
        return Skip(cell, "no_s1_for_cell", f"Sentinel-1 cell is {getattr(s1, 'grid_cell', None)}")

    limit = float("inf") if max_gap_days is None else float(max_gap_days)
    gap = _gap_days(s2.datetime, s1.datetime)
    if not gap <= limit:
        return Skip(cell, "gap_gt_7d", f"{gap:.3f} days, limit {limit:g}")

    s1_corner = _corner(s1.transform, PIXEL_M)
    full = (S1_CELL_PX, S1_CELL_PX)
    if np.shape(s1.vv) != full or np.shape(s1.vh) != full or s1_corner is None:
        return Skip(cell, "s1_not_1068_10m",
                    f"vv {np.shape(s1.vv)}, vh {np.shape(s1.vh)}, transform {s1.transform}")

    crs_s2, crs_s1 = (str(c or "").strip().upper() for c in (s2.crs, s1.crs))
    if not crs_s2 or crs_s2 != crs_s1:
        return Skip(cell, "crs_differs", f"Sentinel-2 {s2.crs!r}, Sentinel-1 {s1.crs!r}")

    problem, s2_corner = _s2_problem(s2)
    if problem:
        return Skip(cell, "s2_band_shape", problem)

    row = (s1_corner[1] - s2_corner[1]) / PIXEL_M      # the top edge is the larger y
    col = (s2_corner[0] - s1_corner[0]) / PIXEL_M
    row0, col0 = int(np.floor(row + 0.5)), int(np.floor(col + 0.5))
    if min(row0, col0) < 0 or max(row0, col0) + CROP_PX > S1_CELL_PX:
        return Skip(cell, "crop_outside_s1",
                    f"crop corner at row {row:.2f}, column {col:.2f} of the Sentinel-1 cell")

    window = np.s_[row0:row0 + CROP_PX, col0:col0 + CROP_PX]
    cuts = []
    for name, channel in zip(S1_CHANNELS, (s1.vv, s1.vh)):
        stored = np.asarray(channel)[window]
        with np.errstate(over="ignore", invalid="ignore"):
            cut = stored.astype(np.float32)          # the values the dB is computed from
        bad = ~np.isfinite(cut) | (cut <= 0)
        if s1.nodata is not None:
            bad |= stored == s1.nodata
        if bad.any():
            return Skip(cell, "s1_nodata_in_crop", f"{name} has {int(bad.sum())} of {bad.size} "
                        f"pixels that are no-data ({s1.nodata}), not finite or not positive "
                        "as float32")
        cuts.append(cut)

    for name in S2_BANDS:                    # every band is an uint16 array here, checked above
        zeros = int(np.count_nonzero(s2.bands[name] == 0))
        if zeros:
            return Skip(cell, "s2_nodata_in_crop", f"{name} has {zeros} of {s2.bands[name].size} "
                        "pixels that are 0 (the L2A no-data value)")

    meta: dict[str, Any] = {"shift_px": (round(row - row0, 6), round(col - col0, 6)),
                            "s1_window": (row0, col0), "cloud_rule": "not applied"}
    if codes is not None:
        if s2.cloud_mask is None:
            return Skip(cell, "read_error", "the cloud rule is on and the cell has no cloud mask")
        share = float(np.isin(s2.cloud_mask, codes).mean())
        if not share <= max_cloud:
            return Skip(cell, "cloud_gt_10pct", f"{share:.4f} of the crop has the codes {codes}, "
                        f"limit {max_cloud:g}")
        meta.update(cloud_rule="applied", cloud_codes=codes, cloud_fraction=share)

    bands = {name: s2.bands[name] for name in S2_BANDS}
    baseline, removed, how = baseline_of(s2.product_id), False, "none"
    if offset == "baseline":
        try:
            bands, removed, baseline = remove_offset_by_baseline(bands, s2.product_id)
        except ValueError as err:
            return Skip(cell, "read_error", f"the offset rule is on and {err}")
        how = "per_image" if baseline == BASELINE_PER_IMAGE else "baseline"
    meta.update(s2_product_id=s2.product_id, s1_product_id=s1.product_id,
                orbit_state=s1.orbit_state, baseline=baseline, offset_removed=removed,
                offset_how=how)

    return Pair(grid_cell=cell,
                s1_db=np.stack([to_db(c) for c in cuts]).astype(np.float16),
                s2=bands,
                cloud_mask=s2.cloud_mask,
                s2_datetime=s2.datetime, s1_datetime=s1.datetime, gap_days=gap,
                crs=s2.crs, bounds=tuple(float(v) for v in s2.bounds), meta=meta)


# --------------------------------------------------------------------------- Sentinel-2 values

def _take_off(bands: Mapping[str, np.ndarray], offset: int) -> dict[str, np.ndarray]:
    """`offset` off every band, clipped at 0, dtype kept, the input arrays are not changed."""
    out = {}
    for name, band in bands.items():
        band = np.asarray(band)
        wide = band.astype(np.int64) if band.dtype.kind in "ui" else band
        out[name] = np.maximum(wide - offset, 0).astype(band.dtype)
    return out


def remove_offset(bands: dict[str, np.ndarray], datetime: Any, offset_from: Any = None,
                  offset: int = 1000) -> dict[str, np.ndarray]:
    """Take the Sentinel-2 offset off the images taken on or after `offset_from`. The DATE rule.

    NOT the rule of the project. Stage 0 found on 2026-09-30 that the offset follows the
    processing baseline and not the date, so the rule of the project is
    `remove_offset_by_baseline`, and `make_pair` applies that one. This function stays as
    it was, for its tests.

    offset_from = None (the default) changes nothing. With a date, an image whose `datetime`
    is on or after it gets `offset` taken off every band. Values are clipped at 0, and 0
    (no-data) stays 0, so a stored value of 1 to `offset` also ends at 0. The dtype is kept and
    the input arrays are not changed. Dates without a time zone are read as UTC.
    With a rule (offset_from is not None) both dates must be known. A missing date (None,
    NaT, an empty text) is a ValueError, because no answer to "on or after" exists for it.

    What the ESA pages say, read on 2026-09-29. None of it is used as a default here, because
    no page says which products Major-TOM holds. Stage 0 measured it.
    - Processing baseline 04.00 went into production on 25 January 2022. From it on the
      Level-2A values are shifted by BOA_ADD_OFFSET, reflectance = (DN + BOA_ADD_OFFSET) /
      QUANTIFICATION_VALUE, and DN 0 stays no-data. The page gives no number for the offset,
      it points to the product metadata.
      https://sentiwiki.copernicus.eu/web/s2-products
    - The number is -1000 at Level-2A (and at Level-1C). Stated by the Sentinel-2 technical
      manager of the Optical Mission Performance Cluster on the ESA STEP forum.
      https://forum.step.esa.int/t/info-introduction-of-additional-radiometric-offset-in-pb04-00-products/35431
    - The offset belongs to the processing baseline, not to the day the image was taken.
      Reprocessed products (Collection-1, baselines 05.00, 05.10 and 05.11) cover images from
      4 July 2015 on and have the product format "with radiometric offset", so an older image
      can carry the offset too. The baseline is the Nxxyy part of the product name.
      https://sentiwiki.copernicus.eu/web/s2-processing
    """
    if offset_from is None:
        return dict(bands)
    taken, first = _utc_time(datetime), _utc_time(offset_from)
    if pd.isna(taken) or pd.isna(first):
        raise ValueError(f"remove_offset needs both dates, got datetime {datetime!r} and "
                         f"offset_from {offset_from!r}")
    if taken < first:
        return dict(bands)
    return _take_off(bands, offset)


def baseline_of(product_id: Any) -> str | None:
    """The processing baseline of a Sentinel-2 product id, its Nxxyy part. None when absent.

    S2A_MSIL2A_20160109T022052_N0201_R003_T50NMF_20160109T053423 gives "N0201".
    """
    found = re.search(r"_(N\d{4})_", str(product_id or ""))
    return found.group(1) if found else None


def offset_by_values(bands: Mapping[str, np.ndarray]) -> bool:
    """The per-image reading of stage 0. True when the values say the image carries the offset.

    The 0.1 percentile (`OFFSET_PERCENTILE`) of B02, B04 and B08 (`OFFSET_BANDS`) over their
    non-zero pixels is at least 950 (`OFFSET_MIN_DN`) in all three. With the offset a
    reflectance of 0 is stored as 1000, so only the few pixels below -0.005 lie under 950.
    A band without a non-zero pixel gives False. The same reading as `stage0.offset_check`,
    pre-registered on 2026-09-29. A missing band is a ValueError.
    """
    missing = [name for name in OFFSET_BANDS if name not in bands]
    if missing:
        raise ValueError(f"bands are missing: {missing}")
    for name in OFFSET_BANDS:
        values = np.asarray(bands[name])
        values = values[values != 0]
        if values.size == 0 or np.percentile(values, OFFSET_PERCENTILE) < OFFSET_MIN_DN:
            return False
    return True


def offset_by_baseline(product_id: Any, bands: Mapping[str, np.ndarray]) -> bool:
    """Whether the image of this product id carries the +1000 offset. The rule of the project.

    Decided 2026-09-30 from the 481 pairs of the stage 0 run. The processing
    baseline is the Nxxyy part of the product id (`baseline_of`).
    - N0400 up to N0599 (`BASELINE_WITH_OFFSET`) carry the offset, 181 of 188 images did.
    - below N0400 does not, 9 of 176 images did.
    - N9999 (`BASELINE_PER_IMAGE`) is decided per image by `offset_by_values`, 17 of 117
      images carried it. `bands` is read only for this case.
    A product id without a baseline or with one the rule does not cover (N0600 to N9998) is
    a ValueError. Nothing is guessed.
    """
    baseline = baseline_of(product_id)
    if baseline is None:
        raise ValueError(f"product id {product_id!r} has no processing baseline (Nxxyy)")
    if baseline == BASELINE_PER_IMAGE:
        return offset_by_values(bands)
    number = int(baseline[1:])
    if number < BASELINE_WITH_OFFSET[0]:
        return False
    if number <= BASELINE_WITH_OFFSET[1]:
        return True
    low, high = (f"N{n:04d}" for n in BASELINE_WITH_OFFSET)
    raise ValueError(f"processing baseline {baseline} of {product_id!r} is not covered by the "
                     f"offset rule (below {low}, {low} to {high}, or {BASELINE_PER_IMAGE})")


def remove_offset_by_baseline(bands: dict[str, np.ndarray], product_id: Any,
                              ) -> tuple[dict[str, np.ndarray], bool, str]:
    """The bands on one scale, by the rule of `offset_by_baseline`.

    Returns (bands, carries, baseline). When the image carries the offset, `OFFSET_DN` is
    taken off every band and clipped at 0, so 0 (no-data) stays 0 and a stored value of 1
    to 1000 ends at 0 too. The dtype is kept and the input arrays are not changed. Without
    the offset the same arrays come back in a new dict. baseline is the Nxxyy text.
    ValueError as in `offset_by_baseline`.
    """
    carries = offset_by_baseline(product_id, bands)
    out = _take_off(bands, OFFSET_DN) if carries else dict(bands)
    return out, carries, baseline_of(product_id)


def _epsg(crs: Any) -> int:
    """The code number of an EPSG text, ValueError for anything else.

    "EPSG:32633" gives 32633. Spaces around it and lower case are fine.
    """
    found = re.fullmatch(r"EPSG:(\d+)", str(crs or "").strip().upper())
    if not found:
        raise ValueError(f"crs {crs!r} is not a text like 'EPSG:32633'")
    return int(found.group(1))


def _nearest(value: float, step: float) -> float:
    """`value` rounded to the nearest multiple of `step`, a half goes up.

    The arithmetic of the measurement (measure_band_shift.py, 2026-09-30), floor(q + 0.5).
    """
    return math.floor(value / step + 0.5) * step


def band_shift_rule(corner: tuple[float, float], crs: Any) -> dict[str, tuple[float, float]]:
    """Where each band really lies against the 10 m bands, (dx, dy) in 10 m pixels per band.

    The corner rule, measured on 433 real cells and decided on 2026-09-30. Each
    band was cut on its own pixel lattice, its corner rounded to the nearest multiple of its
    resolution r (`LATTICE_RES_M`), x = 0 mod r and y = 0 mod r in a UTM north zone. In a UTM
    south zone y counts from the false northing (`SOUTH_FALSE_NORTHING_M`), so there the 60 m
    lattice sits at y = 40 mod 60 (the 10 m and 20 m lattices stay at 0). The stored
    transforms claim all bands share the 10 m corner, so the difference of the two roundings
    is where a band really sits. It explains 91 % of the clear 20 m band cells and 82 % of
    the 60 m ones, the 20 m error is 0.04 px median after it and 1.00 px before.

    corner  (left, top) of the crop in map units, Pair.bounds[0] and Pair.bounds[3]
    crs     the EPSG text of the pair, "EPSG:32633" (north) or "EPSG:32733" (south).
            Anything but a UTM zone is a ValueError, the rule was measured on UTM only.

    Sign convention of the measurement, positive dx = the band's content appears EAST of
    where the 10 m bands have it, positive dy = SOUTH, band[r, c] ~ ref[r - dy, c - dx].
    The 10 m bands get (0.0, 0.0). `s2_to_grid` moves the sampling positions by (dx, dy).
    """
    try:
        left, top = (float(v) for v in corner)
    except (TypeError, ValueError):
        raise ValueError(f"corner {corner!r} must be (left, top) in map units") from None
    if not (math.isfinite(left) and math.isfinite(top)):
        raise ValueError(f"corner {corner!r} must be (left, top) in map units")
    code = _epsg(crs)
    if code in UTM_NORTH_EPSG:
        south = False
    elif code in UTM_SOUTH_EPSG:
        south = True
    else:
        raise ValueError(f"crs {crs!r} is not a UTM zone (EPSG:32601 to 32660 or 32701 to "
                         "32760), the corner rule was measured on UTM only")
    x_10, y_10 = _nearest(left, PIXEL_M), _nearest(top, PIXEL_M)
    out = {}
    for name in S2_BANDS:
        res = float(S2_RES_M[name])
        anchor = (SOUTH_FALSE_NORTHING_M % res) if south else 0.0
        x_r = _nearest(left, res)
        y_r = _nearest(top - anchor, res) + anchor
        # the band's true left is (x_r - x_10) east of the 10 m bands' left, so its content
        # appears that far west; its true top is (y_r - y_10) north, so its content appears
        # that far south
        out[name] = (-(x_r - x_10) / PIXEL_M + 0.0, (y_r - y_10) / PIXEL_M + 0.0)
    return out


def _upsample(band: np.ndarray, size: int, shift: tuple[float, float] = (0.0, 0.0)
              ) -> np.ndarray:
    """Bilinear interpolation of a square (n, n) array to (size, size), float64.

    An output pixel reads the input at the map position of its own centre,
    (i + 0.5) * n / size - 0.5 in input pixels. Outside the outermost input centres the edge
    value is kept. A linear ramp stays the same ramp (no half-pixel shift).

    `shift` = (rows, columns) in OUTPUT pixels moves every sampling position, output pixel i
    reads the input where output pixel i + shift has its centre. A sampling position outside
    the outermost input centres is clamped to the edge, so the border repeats the edge value.
    (0.0, 0.0) gives the same numbers as no shift, bit for bit.
    """
    n = band.shape[0]
    band = band.astype(np.float64)
    axes = []
    for s in shift:
        src = np.clip((np.arange(size) + 0.5 + s) * n / size - 0.5, 0, n - 1)
        i0 = np.floor(src).astype(int)
        i1 = np.minimum(i0 + 1, n - 1)
        axes.append((i0, i1, src - i0))
    (r0, r1, wr), (c0, c1, wc) = axes
    rows = band[r0] * (1 - wr)[:, None] + band[r1] * wr[:, None]       # (size, n)
    return rows[:, c0] * (1 - wc) + rows[:, c1] * wc


def s2_to_grid(bands: dict[str, np.ndarray], size: int = CROP_PX,
               corner: tuple[float, float] | None = None, crs: Any = None) -> np.ndarray:
    """The 12 bands on the 10 m grid, float32 (12, size, size), band order `S2_BANDS`.

    The side of the whole crop is read from B02 (384 px in a pair). A band of resolution r
    must be square with 10 / r of that side (192 px at 20 m, 64 px at 60 m), else
    ValueError. The 10 m bands are copied, the 20 m and 60 m bands are interpolated
    bilinearly with pixel centres at their map positions. With `size`
    below the side of the crop the central `size` px are returned, the same cut as
    `center_crop` makes. Values are not scaled and 0 (no-data) is not treated.

    corner, crs   both None (the default) = the bands are interpolated as the stored
                  transforms place them, all 12 with the same corner. With corner = (left,
                  top) of the crop in map units (Pair.bounds[0], Pair.bounds[3]) and crs the
                  EPSG text of the pair ("EPSG:32633"), the 20 m and 60 m bands are read where
                  the corner rule says they really lie (`band_shift_rule`, decided
                  2026-09-30), so their content comes back in register with the 10 m bands.
                  The shift is a fraction of a coarse pixel and goes into the interpolation.
                  A sampling position outside the band is clamped to its edge, so up to a few
                  border pixels repeat the edge value. The 10 m bands are copied either way.
                  One of the two without the other is a ValueError.
    """
    if (corner is None) != (crs is None):
        raise ValueError("corner and crs go together, give both or neither")
    shifts = None if corner is None else band_shift_rule(corner, crs)
    missing = [name for name in S2_BANDS if name not in bands]
    if missing:
        raise ValueError(f"bands are missing: {missing}")
    full = np.shape(bands["B02"])[0] if np.ndim(bands["B02"]) == 2 else 0
    out = np.empty((len(S2_BANDS), full, full), dtype=np.float32)
    for i, name in enumerate(S2_BANDS):
        n, rest = divmod(full * int(PIXEL_M), S2_RES_M[name])
        band = np.asarray(bands[name])
        if rest or band.shape != (n, n):
            raise ValueError(f"{name} is {band.shape}, B02 is {np.shape(bands['B02'])}, a "
                             f"{S2_RES_M[name]} m band needs {int(PIXEL_M)} / {S2_RES_M[name]} "
                             "of the side of B02")
        if n == full:
            out[i] = band
        else:
            dx, dy = shifts[name] if shifts else (0.0, 0.0)
            out[i] = _upsample(band, full, (dy, dx))
    return np.ascontiguousarray(center_crop(out, size))


def center_crop(arr: Any, size: int) -> Any:
    """The central (size, size) part of the last two axes, as a view. Any leading axes.

    The corner is at (side - size) // 2. ValueError when the array is smaller than `size`.
    """
    h, w = arr.shape[-2:]
    if size < 1 or size > h or size > w:
        raise ValueError(f"cannot cut {size} px from an array of {h} x {w} px")
    top, left = (h - size) // 2, (w - size) // 2
    return arr[..., top:top + size, left:left + size]


# --------------------------------------------------------------------------- pair files

def _to_json(value: Any) -> Any:
    """A value of `Pair` as JSON types. Tuples and dates are marked, so they come back the same."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dt.datetime):
        t = pd.Timestamp(value)
        return {"__time__": t.isoformat(), "unit": t.unit}
    if isinstance(value, tuple):
        return {"__tuple__": [_to_json(v) for v in value]}
    if isinstance(value, list):
        return [_to_json(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _to_json(v) for k, v in value.items()}
    raise TypeError(f"cannot store a value of type {type(value).__name__} in a pair file")


def _from_json(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_json(v) for v in value]
    if isinstance(value, dict):
        if "__tuple__" in value:
            return tuple(_from_json(v) for v in value["__tuple__"])
        if "__time__" in value:
            t = pd.Timestamp(value["__time__"]).as_unit(value["unit"])
            return t if t.tzinfo is None else t.tz_convert("UTC")
        return {k: _from_json(v) for k, v in value.items()}
    return value


def save_pair(pair: Pair, path: str | os.PathLike) -> None:
    """Write one pair to one .npz file at exactly `path`, without compression.

    Arrays keep their dtype. Dates, bounds and meta go into one JSON text, so the file is read
    without pickle. meta may hold None, bool, int, float, str, dates, lists, tuples and dicts
    with str keys, anything else is a TypeError before a file is written. The file is written
    under a temporary name and then renamed, so a stopped run leaves no half file at `path`.
    """
    path = Path(path)
    info = {"format": FILE_FORMAT, "grid_cell": pair.grid_cell, "gap_days": pair.gap_days,
            "s2_datetime": pair.s2_datetime, "s1_datetime": pair.s1_datetime, "crs": pair.crs,
            "bounds": tuple(pair.bounds), "meta": pair.meta, "s2_bands": list(pair.s2)}
    arrays = {"info": np.array(json.dumps(_to_json(info))), "s1_db": pair.s1_db}
    arrays.update({f"s2_{name}": band for name, band in pair.s2.items()})
    if pair.cloud_mask is not None:
        arrays["cloud_mask"] = pair.cloud_mask
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_info(z: Any, path: Any) -> dict:
    """The info of an open pair file, checked for the layout version."""
    info = _from_json(json.loads(str(z["info"][()])))
    if info.get("format") != FILE_FORMAT:
        raise ValueError(f"{path} has pair format {info.get('format')!r}, "
                         f"this code reads {FILE_FORMAT}")
    return info


def load_pair_info(path: str | os.PathLike) -> dict:
    """The info of a file written by `save_pair`, without its arrays.

    A dict with format, grid_cell, gap_days, s2_datetime, s1_datetime, crs, bounds, meta and
    s2_bands, the values as `load_pair` gives them. Only the small info entry of the file is
    read, so checking the meta of many pairs on disk costs no array read.
    """
    with np.load(Path(path), allow_pickle=False) as z:
        return _read_info(z, path)


def load_pair(path: str | os.PathLike) -> Pair:
    """Read a file written by `save_pair`. Dates come back as pandas.Timestamp in UTC."""
    with np.load(Path(path), allow_pickle=False) as z:
        info = _read_info(z, path)
        return Pair(grid_cell=info["grid_cell"], s1_db=z["s1_db"],
                    s2={name: z[f"s2_{name}"] for name in info["s2_bands"]},
                    cloud_mask=z["cloud_mask"] if "cloud_mask" in z.files else None,
                    s2_datetime=info["s2_datetime"], s1_datetime=info["s1_datetime"],
                    gap_days=info["gap_days"], crs=info["crs"], bounds=info["bounds"],
                    meta=info["meta"])
