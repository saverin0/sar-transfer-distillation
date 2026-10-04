"""Pairing rules, one check per rule. Synthetic data, no network."""
import os
import _setup  # noqa: F401

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sartransfer.distil import pairing as P
from sartransfer.distil import records as R

TMP = _setup.TMP_BASE / "pairing"
TMP.mkdir(exist_ok=True)
for old in TMP.iterdir():
    old.unlink()

CELL = "433U_118R"
LEFT, TOP = 500000.0, 5200000.0                # corner of the Sentinel-2 crop, metres


def s2_id(baseline: str) -> str:
    """A Sentinel-2 product id with this processing baseline (the Nxxyy part)."""
    return f"S2A_MSIL2A_20210610T100031_{baseline}_R122_T33UUU_20210610T130000"


S2_ID = s2_id("N0300")                         # the regular cell, a baseline without the offset
RNG = np.random.default_rng(0)
VV = RNG.uniform(0.01, 0.5, (1068, 1068)).astype(np.float32)      # linear power
VH = RNG.uniform(0.001, 0.1, (1068, 1068)).astype(np.float32)
VV.setflags(write=False)
VH.setflags(write=False)


def s2_cell(date="2021-06-10T10:00:00Z", **change) -> R.S2Cell:
    """A regular Sentinel-2 cell, the crop starts at (LEFT, TOP). `change` sets fields."""
    bands, transforms = {}, {}
    for i, name in enumerate(R.S2_BANDS):
        res = R.S2_RES_M[name]
        bands[name] = np.full((3840 // res, 3840 // res), 1000 + i, np.uint16)
        transforms[name] = (float(res), 0.0, LEFT, 0.0, -float(res), TOP)
    cell = R.S2Cell(CELL, S2_ID, pd.Timestamp(date), bands, transforms, "EPSG:32633",
                    np.zeros((384, 384), np.uint8), (LEFT, TOP - 3840.0, LEFT + 3840.0, TOP))
    for key, value in change.items():
        setattr(cell, key, value)
    return cell


def s1_cell(row0=342.0, col0=342.0, date="2021-06-12T05:00:00Z", **change) -> R.S1Cell:
    """A regular Sentinel-1 cell, placed so that the Sentinel-2 crop starts at its pixel
    (row0, col0). The place comes from the transform alone."""
    transform = (10.0, 0.0, LEFT - col0 * 10.0, 0.0, -10.0, TOP + row0 * 10.0)
    cell = R.S1Cell(CELL, "S1A_TEST", pd.Timestamp(date), VV, VH, transform, "EPSG:32633",
                    -32768.0, "ascending")
    for key, value in change.items():
        setattr(cell, key, value)
    return cell


def reason(result) -> str:
    assert isinstance(result, R.Skip), f"expected a Skip, got {type(result).__name__}"
    assert result.grid_cell == CELL and result.reason in R.SKIP_REASONS and result.detail
    return result.reason


def kept(result) -> R.Pair:
    assert isinstance(result, R.Pair), f"expected a Pair, got {result}"
    return result


def raises(kind, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except kind:
        return
    raise SystemExit(f"{fn.__name__} did not raise {kind.__name__}")


# ---------------------------------------------------------------- pair_index
def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


s2_index = pd.DataFrame(
    [("A", "S2_A", ts("2021-06-10 10:00"), "s2_0.parquet", 0),     # three Sentinel-1 rows
     ("B", "S2_B", ts("2022-03-01 10:00"), "s2_0.parquet", 1),     # no Sentinel-1
     ("C", "S2_C", ts("2019-01-10 10:00"), "s2_1.parquet", 0),     # exactly 7 days
     ("D", "S2_D", ts("2019-01-10 10:00"), "s2_1.parquet", 1),     # 7 days and 1 second
     ("E", "S2_E", ts("2016-12-31 23:00"), "s2_1.parquet", 2),     # Sentinel-1 in the next year
     ("F", "S2_F1", ts("2020-05-01 10:00"), "s2_2.parquet", 0),    # two Sentinel-2 rows
     ("F", "S2_F2", ts("2020-05-20 10:00"), "s2_2.parquet", 1),
     ("G", "S2_G", ts("2023-08-01 12:00"), "s2_2.parquet", 2)],    # two rows at the same gap
    columns=["grid_cell", "product_id", "product_datetime", "parquet_url", "parquet_row"])
s1_rows = [("A", "S1_A1", ts("2021-06-05 05:00"), "s1_0.parquet", 4),
           ("A", "S1_A2", ts("2021-06-12 05:00"), "s1_0.parquet", 5),   # closest, 1.79 days
           ("A", "S1_A3", ts("2021-06-16 05:00"), "s1_0.parquet", 6),
           ("C", "S1_C", ts("2019-01-03 10:00"), "s1_1.parquet", 0),
           ("D", "S1_D", ts("2019-01-17 10:00:01"), "s1_1.parquet", 1),
           ("E", "S1_E", ts("2017-01-01 01:00"), "s1_1.parquet", 2),
           ("F", "S1_F", ts("2020-05-18 10:00"), "s1_2.parquet", 0),    # 2 days from S2_F2
           ("G", "S1_G2", ts("2023-08-02 12:00"), "s1_2.parquet", 2),
           ("G", "S1_G1", ts("2023-07-31 12:00"), "s1_2.parquet", 1),
           ("Z", "S1_Z", ts("2021-06-10 10:00"), "s1_2.parquet", 3)]    # no Sentinel-2
s1_index = pd.DataFrame(
    s1_rows, columns=["grid_cell", "product_id", "product_datetime", "parquet_url", "parquet_row"])
s1_index = s1_index.assign(nodata=-32768.0, orbit_state="ascending", centre_lat=0.0,
                           centre_lon=0.0, crs="EPSG:32633")

pairs, counts = P.pair_index(s2_index, s1_index)
assert tuple(pairs.columns) == P.PAIR_COLUMNS == (
    "grid_cell", "s2_url", "s2_row", "s2_product_id", "s2_datetime", "s1_url", "s1_row",
    "s1_product_id", "s1_datetime", "gap_days", "s2_year")
by_cell = pairs.set_index("grid_cell")
assert list(pairs["grid_cell"]) == ["A", "C", "E", "F", "G"], list(pairs["grid_cell"])
assert by_cell.loc["A", "s1_product_id"] == "S1_A2"                  # the closest date
assert abs(by_cell.loc["A", "gap_days"] - (43 / 24)) < 1e-9
assert (by_cell.loc["A", "s1_url"], by_cell.loc["A", "s1_row"]) == ("s1_0.parquet", 5)
assert (by_cell.loc["A", "s2_url"], by_cell.loc["A", "s2_row"]) == ("s2_0.parquet", 0)
assert by_cell.loc["C", "gap_days"] == 7.0                           # 7 days passes
assert "D" not in by_cell.index                                      # 7 days and 1 s does not
assert "B" not in by_cell.index and "Z" not in by_cell.index
assert by_cell.loc["E", "s2_year"] == 2016 and by_cell.loc["E", "s1_datetime"].year == 2017
assert by_cell.loc["F", "s2_product_id"] == "S2_F2" and by_cell.loc["F", "gap_days"] == 2.0
assert by_cell.loc["G", "s1_product_id"] == "S1_G1"                  # same gap, the earlier one
assert str(pairs["s2_datetime"].dt.tz) == "UTC" and str(pairs["s1_datetime"].dt.tz) == "UTC"
assert (pairs["gap_days"] <= 7.0).all() and (pairs["gap_days"] >= 0).all()
assert counts == {"s2_rows": 8, "s2_cells": 7, "s1_rows": 10, "cells_with_s1": 6,
                  "no_s1_for_cell": 1, "gap_gt_7d": 1, "pairs": 5, "max_gap_days": 7.0}, counts
assert counts["s2_cells"] == counts["no_s1_for_cell"] + counts["gap_gt_7d"] + counts["pairs"]
assert "no_s1_for_cell" in R.SKIP_REASONS and "gap_gt_7d" in R.SKIP_REASONS

# the order of the rows does not matter
mixed, mixed_counts = P.pair_index(
    s2_index.sample(frac=1.0, random_state=1).reset_index(drop=True),
    s1_index.sample(frac=1.0, random_state=2).reset_index(drop=True))
pd.testing.assert_frame_equal(mixed, pairs)
assert mixed_counts == counts

# no limit keeps every cell that has Sentinel-1
for no_limit in (None, float("inf")):
    every, every_counts = P.pair_index(s2_index, s1_index, max_gap_days=no_limit)
    assert list(every["grid_cell"]) == ["A", "C", "D", "E", "F", "G"]
    assert every_counts["gap_gt_7d"] == 0 and every_counts["pairs"] == 6
    assert abs(every.set_index("grid_cell").loc["D", "gap_days"] - (7 + 1 / 86400)) < 1e-9
tight, tight_counts = P.pair_index(s2_index, s1_index, max_gap_days=1.0)
assert list(tight["grid_cell"]) == ["E", "G"] and tight_counts["gap_gt_7d"] == 4

# dates without a time zone are read as UTC, a missing date never passes
naive_s1 = s1_index.assign(product_datetime=s1_index["product_datetime"].dt.tz_localize(None))
pd.testing.assert_frame_equal(P.pair_index(s2_index, naive_s1)[0], pairs)
no_date = s1_index.copy()
no_date.loc[no_date["product_id"] == "S1_C", "product_datetime"] = pd.NaT
lost, lost_counts = P.pair_index(s2_index, no_date)
assert "C" not in set(lost["grid_cell"]) and lost_counts["gap_gt_7d"] == 2
lost, lost_counts = P.pair_index(s2_index, no_date, max_gap_days=None)
assert "C" not in set(lost["grid_cell"]) and lost_counts["gap_gt_7d"] == 1

empty, empty_counts = P.pair_index(s2_index, s1_index.iloc[:0])
assert len(empty) == 0 and tuple(empty.columns) == P.PAIR_COLUMNS
assert empty_counts["no_s1_for_cell"] == 7 and empty_counts["pairs"] == 0

row_a = pairs.iloc[0]
for row in (row_a, row_a.to_dict()):
    assert P.s2_row_of(row) == {"grid_cell": "A", "product_id": "S2_A", "parquet_row": 0,
                                "product_datetime": ts("2021-06-10 10:00"),
                                "parquet_url": "s2_0.parquet"}
    assert P.s1_row_of(row) == {"grid_cell": "A", "product_id": "S1_A2", "parquet_row": 5,
                                "product_datetime": ts("2021-06-12 05:00"),
                                "parquet_url": "s1_0.parquet"}
    assert type(P.s2_row_of(row)["parquet_row"]) is int
    assert type(P.s1_row_of(row)["parquet_row"]) is int
print("pair_index: ok")

# ---------------------------------------------------------------- sample_pairs
years = np.repeat([2016, 2019, 2022], [5, 300, 2000])
big = pd.DataFrame({"grid_cell": [f"{i:05d}" for i in range(len(years))], "s2_year": years,
                    "s2_product_id": "a", "s1_product_id": "b"})
picked = P.sample_pairs(big, n=90, seed=0)
per_year = picked["s2_year"].value_counts().to_dict()
assert len(picked) == 90 and picked["grid_cell"].is_unique
assert set(picked["grid_cell"]) <= set(big["grid_cell"])
assert per_year[2016] == 5                       # a small year gives all it has
assert abs(per_year[2019] - per_year[2022]) <= 1 and per_year[2019] + per_year[2022] == 85
assert list(picked.columns) == list(big.columns)
assert picked.equals(picked.sort_values(["s2_year", "grid_cell"]).reset_index(drop=True))
pd.testing.assert_frame_equal(P.sample_pairs(big, n=90, seed=0), picked)
mixed_big = big.sample(frac=1.0, random_state=3)                 # the order does not matter,
pd.testing.assert_frame_equal(P.sample_pairs(mixed_big, 90, 0), picked)
pd.testing.assert_frame_equal(P.sample_pairs(mixed_big.reset_index(drop=True), 90, 0), picked)
assert set(P.sample_pairs(big, n=90, seed=1)["grid_cell"]) != set(picked["grid_cell"])
nine = pd.DataFrame({"grid_cell": [f"{i:05d}" for i in range(9 * 400)],
                     "s2_year": np.repeat(np.arange(2016, 2025), 400),
                     "s2_product_id": "a", "s1_product_id": "b"})
default = P.sample_pairs(nine)
assert len(default) == 500 and set(default["s2_year"].value_counts()) <= {55, 56}
assert len(P.sample_pairs(big.iloc[:40], n=500)) == 40           # fewer than n, all rows
assert len(P.sample_pairs(big.iloc[:0], n=500)) == 0 and len(P.sample_pairs(big, n=0)) == 0
assert len(P.sample_pairs(pairs, n=3)) == 3 and tuple(P.sample_pairs(pairs, 3)) == P.PAIR_COLUMNS
print("sample_pairs: ok")

# ---------------------------------------------------------------- to_db and float16
assert P.to_db(np.float32(1.0)) == 0.0 and abs(P.to_db(10.0) - 10.0) < 1e-6
db = P.to_db(np.array([[0.01, 0.1], [1.0, 100.0]], np.float64))
assert db.dtype == np.float32 and db.shape == (2, 2)
assert np.allclose(db, [[-20.0, -10.0], [0.0, 20.0]], atol=1e-5)
with np.errstate(divide="ignore", invalid="ignore"):
    assert np.isneginf(P.to_db(0.0)) and np.isnan(P.to_db(-1.0))

lin = np.logspace(-6, 3, 4001).astype(np.float32)            # -60 dB to +30 dB
db = P.to_db(lin)
half = db.astype(np.float16)
assert db.min() < -59.9 and db.max() > 29.9
assert np.isfinite(half).all()
assert np.abs(half.astype(np.float32) - db).max() <= 1 / 64 + 1e-6      # half a step of 1/32 dB
info = np.finfo(np.float32)
edge = P.to_db(np.array([info.smallest_subnormal, info.tiny, info.max], np.float32))
assert np.isfinite(edge).all() and edge.min() > -449 and edge.max() < 386
assert np.isfinite(edge.astype(np.float16)).all()            # no positive float32 overflows
with np.errstate(over="ignore"):
    assert not np.isfinite(np.float32(1e5).astype(np.float16))   # above 65,504 it does overflow
print("to_db: ok")

# ---------------------------------------------------------------- make_pair, the cut
for row0, col0 in ((342, 342), (0, 0), (684, 684), (100, 250), (684, 0)):
    pair = kept(P.make_pair(s2_cell(), s1_cell(row0, col0)))
    want = np.stack([P.to_db(VV[row0:row0 + 384, col0:col0 + 384]),
                     P.to_db(VH[row0:row0 + 384, col0:col0 + 384])]).astype(np.float16)
    assert pair.s1_db.dtype == np.float16 and pair.s1_db.shape == (2, 384, 384)
    assert np.array_equal(pair.s1_db, want), (row0, col0)     # cut by map position, (VV, VH)
    assert pair.meta["s1_window"] == (row0, col0) and pair.meta["shift_px"] == (0.0, 0.0)
    assert np.isfinite(pair.s1_db).all()

# a shift by a part of a pixel is rounded to the nearest pixel and stored
for d_row, d_col, used, shift in ((0.2, 0.3, (100, 250), (0.2, 0.3)),
                                  (0.7, 0.6, (101, 251), (-0.3, -0.4)),
                                  (-0.4, 0.0, (100, 250), (-0.4, 0.0)),
                                  (0.5, -0.5, (101, 250), (-0.5, -0.5))):
    pair = kept(P.make_pair(s2_cell(), s1_cell(100 + d_row, 250 + d_col)))
    assert pair.meta["s1_window"] == used, (pair.meta, used)
    assert np.allclose(pair.meta["shift_px"], shift, atol=1e-6), (pair.meta, shift)
    assert type(pair.meta["shift_px"]) is tuple and len(pair.meta["shift_px"]) == 2
    r, c = used
    assert np.array_equal(pair.s1_db[0], P.to_db(VV[r:r + 384, c:c + 384]).astype(np.float16))

# transforms of two real Major-TOM cells (read by the majortom builder on 2026-09-29). The
# corners are not multiples of 10 m and the crop is the middle of the Sentinel-1 cell
marked = np.full((1068, 1068), 0.1, np.float32)
marked[542, 442] = 10.0                                  # row 200, column 100 of the crop
for s2_corner, s1_corner in (((506162.5228299328, 2122324.170112551),
                              (502742.5228299328, 2125744.170112551)),
                             ((600143.6932120417, 2092760.9913528187),
                              (600143.6932120417 - 3420.0, 2092760.9913528187 + 3420.0))):
    real = s2_cell(bounds=(s2_corner[0], s2_corner[1] - 3840.0, s2_corner[0] + 3840.0,
                           s2_corner[1]))
    for name in R.S2_BANDS:
        res = float(R.S2_RES_M[name])
        real.transforms[name] = (res, 0.0, s2_corner[0], 0.0, -res, s2_corner[1])
    pair = kept(P.make_pair(real, s1_cell(
        vv=marked, transform=(10.0, 0.0, s1_corner[0], 0.0, -10.0, s1_corner[1]))))
    assert pair.meta["s1_window"] == (342, 342) and pair.meta["shift_px"] == (0.0, 0.0)
    assert np.argwhere(pair.s1_db[0] > 5.0).tolist() == [[200, 100]]
    assert pair.bounds == real.bounds

# what a pair holds
source = s2_cell()
pair = kept(P.make_pair(source, s1_cell()))
assert pair.grid_cell == CELL and list(pair.s2) == list(R.S2_BANDS)
for name in R.S2_BANDS:
    assert pair.s2[name].dtype == np.uint16 and pair.s2[name] is source.bands[name]
    assert pair.s2[name].shape == (3840 // R.S2_RES_M[name],) * 2
assert pair.cloud_mask is source.cloud_mask and pair.crs == "EPSG:32633"
assert pair.bounds == (LEFT, TOP - 3840.0, LEFT + 3840.0, TOP) and type(pair.bounds) is tuple
assert pair.s2_datetime == ts("2021-06-10 10:00") and pair.s1_datetime == ts("2021-06-12 05:00")
assert abs(pair.gap_days - 43 / 24) < 1e-9
assert pair.meta["cloud_rule"] == "not applied" and "cloud_fraction" not in pair.meta
assert pair.meta["s2_product_id"] == S2_ID and pair.meta["s1_product_id"] == "S1A_TEST"
assert pair.meta["orbit_state"] == "ascending"
assert (pair.meta["baseline"], pair.meta["offset_removed"], pair.meta["offset_how"]) == (
    "N0300", False, "baseline")                        # N0300 carries no offset, values as stored
assert VV.flags.writeable is False                     # the inputs are not written to

# real value range, end to end. Very small and very large power stays finite in float16
wide = VV.copy()
wide[342:726, 342:726] = np.logspace(-38, 38, 384 * 384).reshape(384, 384).astype(np.float32)
pair = kept(P.make_pair(s2_cell(), s1_cell(vv=wide)))
assert np.isfinite(pair.s1_db).all()
assert pair.s1_db[0].min() < -379 and pair.s1_db[0].max() > 379
print("make_pair, cut and contents: ok")

# ---------------------------------------------------------------- make_pair, every skip
# no Sentinel-1 for the cell
assert reason(P.make_pair(s2_cell(), None)) == "no_s1_for_cell"
assert reason(P.make_pair(s2_cell(), s1_cell(grid_cell="000U_000R"))) == "no_s1_for_cell"

# dates at most 7 days apart
assert kept(P.make_pair(s2_cell(), s1_cell(date="2021-06-17T10:00:00Z"))).gap_days == 7.0
assert kept(P.make_pair(s2_cell(), s1_cell(date="2021-06-03T10:00:00Z"))).gap_days == 7.0
assert reason(P.make_pair(s2_cell(), s1_cell(date="2021-06-17T10:00:01Z"))) == "gap_gt_7d"
assert reason(P.make_pair(s2_cell(), s1_cell(date="2021-06-03T09:59:59Z"))) == "gap_gt_7d"
assert reason(P.make_pair(s2_cell(), s1_cell(date="2022-06-10T10:00:00Z"))) == "gap_gt_7d"
late = s1_cell(date="2021-06-19T10:00:00Z")
assert kept(P.make_pair(s2_cell(), late, max_gap_days=10.0)).gap_days == 9.0
assert kept(P.make_pair(s2_cell(), late, max_gap_days=None)).gap_days == 9.0
assert reason(P.make_pair(s2_cell(), s1_cell(), max_gap_days=1.0)) == "gap_gt_7d"
naive = s1_cell(datetime=pd.Timestamp("2021-06-12T05:00:00"))         # no time zone = UTC
assert abs(kept(P.make_pair(s2_cell(), naive)).gap_days - 43 / 24) < 1e-9
for missing in (pd.NaT, None):
    assert reason(P.make_pair(s2_cell(), s1_cell(datetime=missing))) == "gap_gt_7d"
    assert reason(P.make_pair(s2_cell(datetime=missing), s1_cell(),
                              max_gap_days=None)) == "gap_gt_7d"

# Sentinel-1 cell of 1,068 px at 10 m
antarctic = s1_cell(vv=np.ones((1424, 1424), np.float32), vh=np.ones((1424, 1424), np.float32),
                    transform=(7.5, 0.0, LEFT - 3420.0, 0.0, -7.5, TOP + 3420.0))
for bad in (antarctic,
            s1_cell(transform=(7.5, 0.0, LEFT - 3420.0, 0.0, -7.5, TOP + 3420.0)),
            s1_cell(transform=(20.0, 0.0, LEFT - 3420.0, 0.0, -20.0, TOP + 3420.0)),
            s1_cell(transform=(10.0, 0.0, LEFT - 3420.0, 0.0, 10.0, TOP + 3420.0)),    # south-up
            s1_cell(transform=(10.0, 0.1, LEFT - 3420.0, 0.0, -10.0, TOP + 3420.0)),   # turned
            s1_cell(transform=(LEFT - 3420.0, 10.0, 0.0, TOP + 3420.0, 0.0, -10.0)),   # GDAL order
            s1_cell(transform=(10.0, 0.0, float("nan"), 0.0, -10.0, TOP + 3420.0)),
            s1_cell(transform=(10.0, 0.0, LEFT)), s1_cell(transform=None),
            s1_cell(vv=VV[:1000], vh=VH[:1000]), s1_cell(vv=VV[:, :1067]),
            s1_cell(vh=VH[:534, :534]), s1_cell(vv=None)):
    assert reason(P.make_pair(s2_cell(), bad)) == "s1_not_1068_10m"
nine = (10.0, 0.0, LEFT - 3420.0, 0.0, -10.0, TOP + 3420.0, 0.0, 0.0, 1.0)     # all 9 numbers
assert kept(P.make_pair(s2_cell(), s1_cell(transform=nine))).meta["s1_window"] == (342, 342)

# the same coordinate system
assert reason(P.make_pair(s2_cell(), s1_cell(crs="EPSG:32634"))) == "crs_differs"
assert reason(P.make_pair(s2_cell(crs="EPSG:32733"), s1_cell())) == "crs_differs"
assert reason(P.make_pair(s2_cell(crs=""), s1_cell(crs=""))) == "crs_differs"
assert reason(P.make_pair(s2_cell(crs=None), s1_cell(crs=None))) == "crs_differs"
kept(P.make_pair(s2_cell(), s1_cell(crs=" epsg:32633 ")))

# every Sentinel-2 band at its native size


def s2_with(name: str, band=None, transform=None, **change) -> R.S2Cell:
    cell = s2_cell(**change)
    if band is not None:
        cell.bands[name] = band
    if transform is not None:
        cell.transforms[name] = transform
    return cell


dropped = s2_cell()
del dropped.bands["B01"]
no_transform = s2_cell()
del no_transform.transforms["B12"]
for bad in (s2_with("B05", np.full((384, 384), 7, np.uint16)),           # 20 m band, 10 m size
            s2_with("B01", np.full((192, 192), 7, np.uint16)),           # 60 m band, 20 m size
            s2_with("B02", np.full((192, 192), 7, np.uint16)),           # 10 m band, 20 m size
            s2_with("B08", np.full((384, 383), 7, np.uint16)),
            s2_with("B03", np.full((1068, 1068), 7, np.uint16)),         # a full cell, not the crop
            s2_with("B04", np.full((3, 384, 384), 7, np.uint16)),
            s2_with("B8A", np.full((192, 192), 7, np.int16)),
            s2_with("B11", np.full((192, 192), 7.0, np.float32)),
            s2_with("B12", [[7] * 192] * 192),
            s2_with("B11", transform=(10.0, 0.0, LEFT, 0.0, -10.0, TOP)),        # wrong pixel size
            s2_with("B09", transform=(60.0, 0.0, LEFT + 60.0, 0.0, -60.0, TOP)),  # moved
            s2_with("B06", transform=(20.0, 0.0, LEFT, 0.0, -20.0, TOP - 20.0)),
            s2_with("B07", transform=(20.0, 0.0, LEFT, 0.0, 20.0, TOP)),
            dropped, no_transform,
            s2_cell(bounds=(LEFT, TOP - 3840.0, LEFT + 3850.0, TOP)),
            s2_cell(bounds=(LEFT, TOP, LEFT + 3840.0, TOP - 3840.0)),
            s2_cell(bounds=(LEFT, TOP - 3840.0, LEFT + 3840.0)), s2_cell(bounds=None),
            s2_cell(cloud_mask=np.zeros((192, 192), np.uint8))):
    assert reason(P.make_pair(bad, s1_cell())) == "s2_band_shape"

# the crop fully inside the Sentinel-1 cell (within half a pixel)
for row0, col0 in ((-1, 342), (342, -1), (685, 342), (342, 685), (-0.6, 342), (342, 684.6),
                   (-500, -500), (2000, 342)):
    assert reason(P.make_pair(s2_cell(), s1_cell(row0, col0))) == "crop_outside_s1", (row0, col0)
for row0, col0 in ((0, 342), (342, 0), (684, 342), (342, 684), (-0.4, 342), (342, 684.4)):
    kept(P.make_pair(s2_cell(), s1_cell(row0, col0)))

# no no-data, non-finite or non-positive Sentinel-1 pixel in the crop
for value in (-32768.0, np.nan, np.inf, -np.inf, 0.0, -0.001):
    for channel in ("vv", "vh"):
        for spot in ((342, 342), (725, 725), (342, 725), (500, 600)):      # inside the crop
            arr = (VV if channel == "vv" else VH).copy()
            arr[spot] = value
            result = P.make_pair(s2_cell(), s1_cell(**{channel: arr}))
            assert reason(result) == "s1_nodata_in_crop", (value, channel, spot)
            assert channel.upper() in result.detail
        for spot in ((341, 342), (342, 341), (726, 500), (500, 726), (0, 0), (1067, 1067)):
            arr = (VV if channel == "vv" else VH).copy()
            arr[spot] = value                                               # outside the crop
            kept(P.make_pair(s2_cell(), s1_cell(**{channel: arr})))
arr = VV.copy()
arr[400, 400] = 9999.0                                   # a no-data value that is positive
assert reason(P.make_pair(s2_cell(), s1_cell(vv=arr, nodata=9999.0))) == "s1_nodata_in_crop"
kept(P.make_pair(s2_cell(), s1_cell(vv=arr, nodata=-32768.0)))
kept(P.make_pair(s2_cell(), s1_cell(vv=arr, nodata=None)))
kept(P.make_pair(s2_cell(), s1_cell(nodata=np.nan)))
moved = VV.copy()
moved[120, 270] = 0.0                                    # the window follows the map position
assert reason(P.make_pair(s2_cell(), s1_cell(100, 250, vv=moved))) == "s1_nodata_in_crop"
kept(P.make_pair(s2_cell(), s1_cell(342, 342, vv=moved)))
# the values are checked as float32, the type the dB is computed from. A float64 value that
# is positive and finite but 0 or infinite as float32 must not end as -inf or inf in a pair
for value in (1e-50, 1e60):
    for channel in ("vv", "vh"):
        arr = (VV if channel == "vv" else VH).astype(np.float64)
        arr[500, 600] = value
        assert np.isfinite(arr).all() and (arr > 0).all()
        result = P.make_pair(s2_cell(), s1_cell(**{channel: arr}))
        assert reason(result) == "s1_nodata_in_crop", (value, channel)
        arr[500, 600], arr[10, 10] = 0.2, value                          # outside the crop
        pair = kept(P.make_pair(s2_cell(), s1_cell(**{channel: arr})))
        assert pair.s1_db.dtype == np.float16 and np.isfinite(pair.s1_db).all()
wide64 = kept(P.make_pair(s2_cell(), s1_cell(vv=VV.astype(np.float64), vh=VH.astype(np.float64))))
assert np.array_equal(wide64.s1_db, kept(P.make_pair(s2_cell(), s1_cell())).s1_db)

# no pixel of 0 (the L2A no-data value) in any Sentinel-2 band of the crop
assert R.SKIP_REASONS.index("s2_nodata_in_crop") == R.SKIP_REASONS.index("s1_nodata_in_crop") + 1
one_zero = s2_cell()
one_zero.bands["B09"][10, 20] = 0                        # one pixel of the 60 m band
result = P.make_pair(one_zero, s1_cell())
assert reason(result) == "s2_nodata_in_crop"
assert result.detail == "B09 has 1 of 4096 pixels that are 0 (the L2A no-data value)", result.detail
for name in R.S2_BANDS:
    for spot in ((0, 0), (-1, -1), (5, 17)):
        cell = s2_cell()
        cell.bands[name][spot] = 0
        result = P.make_pair(cell, s1_cell())
        assert reason(result) == "s2_nodata_in_crop", (name, spot)
        assert result.detail.startswith(f"{name} has 1 of {cell.bands[name].size} pixels")
many = s2_cell()
many.bands["B05"][:3] = 0                                # 3 rows of 192 px
many.bands["B12"][0, 0] = 0                              # the first band in S2_BANDS order is named
assert P.make_pair(many, s1_cell()).detail.startswith("B05 has 576 of 36864 pixels")
lowest = s2_cell()
lowest.bands["B02"][0, 0] = 1                            # 1 is a value, not no-data
kept(P.make_pair(lowest, s1_cell()))
clean = s2_cell()
assert not any((band == 0).any() for band in clean.bands.values())
assert kept(P.make_pair(clean, s1_cell())).s2["B09"] is clean.bands["B09"]

# at most 10 % cloud in the crop


def clouds(n: int, code: int = 3) -> np.ndarray:
    mask = np.zeros(384 * 384, np.uint8)
    mask[:n] = code
    return RNG.permutation(mask).reshape(384, 384)


assert 384 * 384 == 147456                                # 10 % is 14,745.6 px
pair = kept(P.make_pair(s2_cell(cloud_mask=clouds(14745)), s1_cell(), cloud_codes=(3, 7)))
assert pair.meta["cloud_rule"] == "applied" and pair.meta["cloud_codes"] == [3, 7]
assert abs(pair.meta["cloud_fraction"] - 14745 / 147456) < 1e-12
assert reason(P.make_pair(s2_cell(cloud_mask=clouds(14746)), s1_cell(),
                          cloud_codes=(3, 7))) == "cloud_gt_10pct"
assert reason(P.make_pair(s2_cell(cloud_mask=clouds(14746, 7)), s1_cell(),
                          cloud_codes=(3, 7))) == "cloud_gt_10pct"
assert reason(P.make_pair(s2_cell(cloud_mask=clouds(147456)), s1_cell(),
                          cloud_codes=[3])) == "cloud_gt_10pct"
both_codes = clouds(7400, 3)
both_codes[both_codes == 0] = RNG.permutation(np.repeat([7, 0], [7400, 147456 - 14800]))
assert reason(P.make_pair(s2_cell(cloud_mask=both_codes), s1_cell(),
                          cloud_codes=(3, 7))) == "cloud_gt_10pct"         # 3 and 7 together
kept(P.make_pair(s2_cell(cloud_mask=both_codes), s1_cell(), cloud_codes=(3,)))
# only the given codes count
pair = kept(P.make_pair(s2_cell(cloud_mask=clouds(147456, 5)), s1_cell(), cloud_codes=(3, 7)))
assert pair.meta["cloud_fraction"] == 0.0
# the limit itself passes, one pixel more does not
quarter = 147456 // 4
kept(P.make_pair(s2_cell(cloud_mask=clouds(quarter)), s1_cell(), cloud_codes=(3,),
                 max_cloud=0.25))
assert reason(P.make_pair(s2_cell(cloud_mask=clouds(quarter + 1)), s1_cell(), cloud_codes=(3,),
                          max_cloud=0.25)) == "cloud_gt_10pct"
# without codes the rule is not applied, and that is written down
pair = kept(P.make_pair(s2_cell(cloud_mask=clouds(147456)), s1_cell()))
assert pair.meta["cloud_rule"] == "not applied"
assert "cloud_codes" not in pair.meta and "cloud_fraction" not in pair.meta
assert inspect.signature(P.make_pair).parameters["cloud_codes"].default is None
assert inspect.signature(P.make_pair).parameters["max_cloud"].default == 0.10
assert inspect.signature(P.make_pair).parameters["max_gap_days"].default == 7.0
# with codes and without a mask the cell cannot be checked, so it is not kept
assert reason(P.make_pair(s2_cell(cloud_mask=None), s1_cell(), cloud_codes=(3,))) == "read_error"
assert kept(P.make_pair(s2_cell(cloud_mask=None), s1_cell())).cloud_mask is None
# codes that cannot work are an error of the call, the rule never counts as applied with
# them. An empty list is not None. The error comes before any rule is looked at
all_cloud = clouds(147456)
for bad_codes in ([], (), iter(()), "3", "12", b"3", [3.7], [300], [-1], [3, 256], [None],
                  ["x"], [float("nan")], [float("inf")], 3, 3.0):
    raises(ValueError, P.make_pair, s2_cell(cloud_mask=all_cloud), s1_cell(),
           cloud_codes=bad_codes)
    raises(ValueError, P.make_pair, s2_cell(cloud_mask=all_cloud), None, cloud_codes=bad_codes)
for fine_codes, stored_codes in (([3], [3]), ((7, 3, 3), [3, 7]), ([0, 255], [0, 255]),
                                 ([3.0, np.uint8(1)], [1, 3]), ((c for c in (2, 1)), [1, 2]),
                                 (np.array([3, 2]), [2, 3])):
    pair = kept(P.make_pair(s2_cell(), s1_cell(), cloud_codes=fine_codes, max_cloud=1.0))
    assert pair.meta["cloud_codes"] == stored_codes, (pair.meta, stored_codes)
    assert all(type(c) is int for c in pair.meta["cloud_codes"])
assert kept(P.make_pair(s2_cell(), s1_cell(), cloud_codes=[0],
                        max_cloud=1.0)).meta["cloud_fraction"] == 1.0      # code 0 is a code
# a limit that is not a number lets no cell pass
assert reason(P.make_pair(s2_cell(), s1_cell(), cloud_codes=[3],
                          max_cloud=float("nan"))) == "cloud_gt_10pct"

# the first rule that fails is the one that is counted
cloudy = clouds(147456)
hole = VV.copy()
hole[500, 500] = np.nan
small = np.full((384, 384), 7, np.uint16)
zero_b09 = np.full((64, 64), 7, np.uint16)
zero_b09[10, 20] = 0
order = [
    ("no_s1_for_cell", dict(grid_cell="000U_000R")),
    ("gap_gt_7d", dict(date="2021-08-01T00:00:00Z")),
    ("s1_not_1068_10m", dict(vh=VH[:1000])),
    ("crs_differs", dict(crs="EPSG:32634")),
    ("s2_band_shape", None),
    ("crop_outside_s1", dict(row0=900)),
    ("s1_nodata_in_crop", dict(vv=hole)),
    ("s2_nodata_in_crop", None),
    ("cloud_gt_10pct", None),
]
for first in range(len(order)):
    s1_change, bands = {}, {}
    for name, change in order[first:]:
        if name == "s2_band_shape":
            bands["B05"] = small
        elif name == "s2_nodata_in_crop":
            bands["B09"] = zero_b09
        elif change:
            s1_change.update(change)
    cell = s2_cell(cloud_mask=cloudy)
    cell.bands.update(bands)
    result = P.make_pair(cell, s1_cell(**s1_change), cloud_codes=(3,))
    assert reason(result) == order[first][0], (order[first][0], result)
assert {name for name, _ in order} | {"read_error"} == set(R.SKIP_REASONS)
assert len(order) + 1 == len(R.SKIP_REASONS)
print("make_pair, skips: ok")

# ---------------------------------------------------------------- remove_offset
assert inspect.signature(P.remove_offset).parameters["offset_from"].default is None
assert inspect.signature(P.remove_offset).parameters["offset"].default == 1000
raw = {"B02": np.array([[0, 1, 999], [1000, 1001, 65535]], np.uint16),
       "B8A": np.array([[0, 5000]], np.uint16)}
saved = {k: v.copy() for k, v in raw.items()}
after = ts("2023-05-01 10:00")


def same_bands(a: dict, b: dict) -> bool:
    return list(a) == list(b) and all(
        a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]) for k in a)


assert same_bands(P.remove_offset(raw, after), saved)                 # default, nothing changes
assert same_bands(P.remove_offset(raw, after, offset_from=None), saved)
out = P.remove_offset(raw, after, offset_from="2022-01-25")
assert out["B02"].dtype == np.uint16 and out["B8A"].dtype == np.uint16
assert out["B02"].tolist() == [[0, 0, 0], [0, 1, 64535]]              # clipped, 0 stays 0
assert out["B8A"].tolist() == [[0, 4000]]
assert same_bands(raw, saved)                                         # the input is not changed
# on the date, and one second before it
on_day = P.remove_offset(raw, ts("2022-01-25 00:00:00"), offset_from="2022-01-25")
assert on_day["B02"].tolist() == [[0, 0, 0], [0, 1, 64535]]
before = P.remove_offset(raw, ts("2022-01-24 23:59:59"), offset_from="2022-01-25")
assert same_bands(before, saved)
assert same_bands(P.remove_offset(raw, ts("2016-07-01 10:00"), offset_from="2022-01-25"), saved)
# time zones. 00:30 in Berlin on 25 January is 23:30 UTC on 24 January
berlin = pd.Timestamp("2022-01-25 00:30", tz="Europe/Berlin")
assert same_bands(P.remove_offset(raw, berlin, offset_from="2022-01-25"), saved)
naive_day = P.remove_offset(raw, pd.Timestamp("2022-01-25 00:30"), offset_from=ts("2022-01-25"))
assert naive_day["B8A"].tolist() == [[0, 4000]]
assert P.remove_offset(raw, after, offset_from="2022-01-25", offset=500)["B02"].tolist() == [
    [0, 0, 499], [500, 501, 65035]]
assert P.remove_offset(raw, after, offset_from="2022-01-25", offset=0)["B02"].tolist() == [
    [0, 1, 999], [1000, 1001, 65535]]
# with a rule both dates must be known. A missing date never counts as "on or after"
for missing in (None, pd.NaT, "", "NaT", float("nan"), np.datetime64("NaT", "s")):
    raises(ValueError, P.remove_offset, raw, missing, offset_from="2022-01-25")
    assert same_bands(P.remove_offset(raw, missing), saved)           # no rule, nothing to decide
    if missing is not None:                                           # None means no rule
        for image_date in (ts("2019-08-11 16:28"), after):
            raises(ValueError, P.remove_offset, raw, image_date, offset_from=missing)
assert same_bands(raw, saved)
print("remove_offset: ok")

# ---------------------------------------------------------------- the offset by processing baseline
assert P.OFFSET_DN == 1000 and P.OFFSET_BANDS == ("B02", "B04", "B08")
assert P.OFFSET_PERCENTILE == 0.1 and P.OFFSET_MIN_DN == 950
assert P.BASELINE_WITH_OFFSET == (400, 599) and P.BASELINE_PER_IMAGE == "N9999"
assert P.OFFSET_RULES == ("baseline", None)
assert P.baseline_of("S2A_MSIL2A_20160109T022052_N0201_R003_T50NMF_20160109T053423") == "N0201"
assert P.baseline_of(s2_id("N9999")) == "N9999" and P.baseline_of(S2_ID) == "N0300"
for without in ("S2A_TEST", "", None, "N0400", "S2A_N0400", "S2A_MSIL2A_N040_R003_", float("nan")):
    assert P.baseline_of(without) is None, without

with_off = s2_cell().bands                       # every value 1000 + i, the reading says offset
without_off = {name: np.full(band.shape, 100 + i, np.uint16)
               for i, (name, band) in enumerate(with_off.items())}
assert P.offset_by_values(with_off) is True and P.offset_by_values(without_off) is False
# the per-image reading. 950 is at least 950, 949 is not, zeros do not count, one band decides
edge = {name: np.full((8, 8), 5000, np.uint16) for name in R.S2_BANDS}
edge["B02"][:] = 950
assert P.offset_by_values(edge) is True
edge["B08"][:] = 949
assert P.offset_by_values(edge) is False
holed = {name: np.full((8, 8), 1000, np.uint16) for name in R.S2_BANDS}
holed["B04"][:3] = 0                                     # 3 rows of no-data
assert P.offset_by_values(holed) is True
holed["B04"][:] = 0                                      # no data at all
assert P.offset_by_values(holed) is False
raises(ValueError, P.offset_by_values, {"B02": edge["B02"], "B04": edge["B04"]})

# the rule per baseline. The values are read for N9999 only
for baseline, want in (("N0201", False), ("N0301", False), ("N0399", False), ("N0400", True),
                       ("N0500", True), ("N0510", True), ("N0599", True)):
    for bands in (with_off, without_off, {}):
        assert P.offset_by_baseline(s2_id(baseline), bands) is want, baseline
assert P.offset_by_baseline(s2_id("N9999"), with_off) is True
assert P.offset_by_baseline(s2_id("N9999"), without_off) is False
for bad in (s2_id("N0600"), s2_id("N9998"), "S2A_TEST", "", None):
    raises(ValueError, P.offset_by_baseline, bad, with_off)
    raises(ValueError, P.remove_offset_by_baseline, with_off, bad)
raises(ValueError, P.offset_by_baseline, s2_id("N9999"), {})

# remove_offset_by_baseline. Clipped at 0, 0 stays 0, dtype kept, the input is not changed
out, carries, baseline = P.remove_offset_by_baseline(raw, s2_id("N0400"))
assert (carries, baseline) == (True, "N0400")
assert out["B02"].tolist() == [[0, 0, 0], [0, 1, 64535]] and out["B8A"].tolist() == [[0, 4000]]
assert out["B02"].dtype == np.uint16 and out["B8A"].dtype == np.uint16
assert same_bands(raw, saved)
out, carries, baseline = P.remove_offset_by_baseline(raw, s2_id("N0301"))
assert (carries, baseline) == (False, "N0301") and same_bands(out, saved)
assert out["B02"] is raw["B02"] and out is not raw                 # the same arrays, a new dict
out, carries, baseline = P.remove_offset_by_baseline(with_off, s2_id("N9999"))
assert (carries, baseline) == (True, "N9999")
assert [int(out[name][0, 0]) for name in R.S2_BANDS] == list(range(12))
assert all(int(with_off[name][0, 0]) == 1000 + i for i, name in enumerate(R.S2_BANDS))
out, carries, baseline = P.remove_offset_by_baseline(without_off, s2_id("N9999"))
assert (carries, baseline) == (False, "N9999") and same_bands(out, without_off)
out, carries, baseline = P.remove_offset_by_baseline(holed, s2_id("N0510"))
assert carries and out["B04"].tolist() == [[0] * 8] * 8 and out["B02"].tolist() == [[0] * 8] * 8

# make_pair takes the offset off by baseline unless told not to
assert inspect.signature(P.make_pair).parameters["offset"].default == "baseline"
for baseline, removed, how in (("N0201", False, "baseline"), ("N0300", False, "baseline"),
                               ("N0400", True, "baseline"), ("N0510", True, "baseline"),
                               ("N9999", True, "per_image")):
    source = s2_cell(product_id=s2_id(baseline))
    pair = kept(P.make_pair(source, s1_cell()))
    assert (pair.meta["baseline"], pair.meta["offset_removed"], pair.meta["offset_how"]) == (
        baseline, removed, how), (baseline, pair.meta)
    for i, name in enumerate(R.S2_BANDS):
        assert pair.s2[name].dtype == np.uint16 and pair.s2[name].shape == source.bands[name].shape
        assert (pair.s2[name] == (i if removed else 1000 + i)).all(), (baseline, name)
        assert (pair.s2[name] is source.bands[name]) is not removed
        assert (source.bands[name] == 1000 + i).all()      # the cell is not changed
    plain_pair = kept(P.make_pair(source, s1_cell(), offset=None))
    assert (plain_pair.meta["baseline"], plain_pair.meta["offset_removed"],
            plain_pair.meta["offset_how"]) == (baseline, False, "none")
    assert all(plain_pair.s2[name] is source.bands[name] for name in R.S2_BANDS)
    assert pair.meta["s2_product_id"] == plain_pair.meta["s2_product_id"] == s2_id(baseline)
low = s2_cell(product_id=s2_id("N9999"))                 # N9999 whose values carry no offset
for name in R.S2_BANDS:
    low.bands[name][:] = 100
pair = kept(P.make_pair(low, s1_cell()))
assert (pair.meta["offset_removed"], pair.meta["offset_how"]) == (False, "per_image")
assert (pair.s2["B02"] == 100).all() and pair.s2["B02"] is low.bands["B02"]
# with the rule on, a product id without a baseline, or with one the rule does not cover,
# cannot be put on one scale, so the cell is not kept. Without the rule it is
for product in ("S2A_TEST", "", s2_id("N0600")):
    result = P.make_pair(s2_cell(product_id=product), s1_cell())
    assert reason(result) == "read_error" and result.detail.startswith("the offset rule is on and")
    pair = kept(P.make_pair(s2_cell(product_id=product), s1_cell(), offset=None))
    assert pair.meta["baseline"] == P.baseline_of(product) and pair.meta["offset_how"] == "none"
# the offset comes after the other rules, a cloudy cell is counted as cloudy
assert reason(P.make_pair(s2_cell(product_id="S2A_TEST", cloud_mask=clouds(147456)), s1_cell(),
                          cloud_codes=(3,))) == "cloud_gt_10pct"
# anything but "baseline" or None is an error of the call, before any rule is looked at
for bad in ("date", "", "Baseline", True, 1000):
    raises(ValueError, P.make_pair, s2_cell(), s1_cell(), offset=bad)
    raises(ValueError, P.make_pair, s2_cell(), None, offset=bad)
print("offset by baseline: ok")

# ---------------------------------------------------------------- s2_to_grid


def ramp(n: int, res: int, clip_to: int | None = None) -> np.ndarray:
    """100 + 3 x + 7 y at the pixel centres of an (n, n) band of `res` m pixels.
    x and y are map positions in units of 10 m. `clip_to` keeps the positions between the
    outermost centres of a band of that resolution."""
    c = (np.arange(n) + 0.5) * res / 10.0
    if clip_to:
        c = np.clip(c, clip_to / 20.0, 384 - clip_to / 20.0)
    return 100.0 + 3.0 * c[None, :] + 7.0 * c[:, None]


ramps = {name: ramp(3840 // res, res) for name, res in R.S2_RES_M.items()}
assert all(np.array_equal(v, np.round(v)) and v.max() < 65535 for v in ramps.values())
grid = P.s2_to_grid({name: v.astype(np.uint16) for name, v in ramps.items()})
assert grid.dtype == np.float32 and grid.shape == (12, 384, 384)
fine = ramp(384, 10)
for i, name in enumerate(R.S2_BANDS):
    res = R.S2_RES_M[name]
    if res == 10:
        assert np.array_equal(grid[i], fine), name                # 10 m bands are copied
        continue
    edge = res // 20                                  # 10 m pixels outside the outermost centres
    inner = np.s_[edge:384 - edge, edge:384 - edge]
    assert np.abs(grid[i][inner] - fine[inner]).max() < 1e-3, name     # the same ramp, no shift
    assert np.abs(grid[i] - ramp(384, 10, clip_to=res)).max() < 1e-3, name   # edge value kept
    assert np.abs(grid[i] - fine).max() > 1.0, name           # the border is where it differs

# the same numbers as torch with align_corners=False, and not the ones of align_corners=True
noise = {name: RNG.integers(0, 65536, (3840 // res,) * 2).astype(np.uint16)
         for name, res in R.S2_RES_M.items()}
grid = P.s2_to_grid(noise)
for i, name in enumerate(R.S2_BANDS):
    t = torch.from_numpy(noise[name].astype(np.float64))[None, None]
    ref = torch.nn.functional.interpolate(t, size=(384, 384), mode="bilinear",
                                          align_corners=False)[0, 0].numpy()
    assert np.abs(grid[i] - ref).max() < 0.01, name
    if R.S2_RES_M[name] == 10:
        assert np.array_equal(grid[i], noise[name].astype(np.float32))
    else:
        wrong = torch.nn.functional.interpolate(t, size=(384, 384), mode="bilinear",
                                                align_corners=True)[0, 0].numpy()
        assert np.abs(grid[i] - wrong).max() > 100.0, name
    assert grid[i].min() >= noise[name].min() and grid[i].max() <= noise[name].max()

# band order and size
flat = {name: np.full((3840 // R.S2_RES_M[name],) * 2, 100 * i, np.uint16)
        for i, name in enumerate(R.S2_BANDS)}
grid = P.s2_to_grid(dict(reversed(list(flat.items()))))            # dict order does not matter
assert [float(grid[i, 5, 5]) for i in range(12)] == [100.0 * i for i in range(12)]
assert np.array_equal(grid, np.broadcast_to(grid[:, :1, :1], grid.shape))
grid = P.s2_to_grid(kept(P.make_pair(s2_cell(), s1_cell())).s2)     # a pair goes in as it is
assert grid.shape == (12, 384, 384) and [int(grid[i, 0, 0]) for i in (0, 11)] == [1000, 1011]
# a smaller size is the central part of the whole crop, cut after the interpolation
whole = P.s2_to_grid(noise)
for size in (256, 192, 383, 1):
    part = P.s2_to_grid(noise, size=size)
    assert part.dtype == np.float32 and part.shape == (12, size, size)
    assert np.array_equal(part, P.center_crop(whole, size)), size
    assert part.flags.c_contiguous and part.base is None       # not a view of the whole crop
near = (384 - 256) // 2
part = P.s2_to_grid({name: v.astype(np.uint16) for name, v in ramps.items()}, size=256)
for i, name in enumerate(R.S2_BANDS):                # 256 px is inside every outermost centre
    assert np.abs(part[i] - fine[near:near + 256, near:near + 256]).max() < 1e-3, name
half_size = {name: np.full((1920 // R.S2_RES_M[name],) * 2, 9, np.uint16) for name in R.S2_BANDS}
assert P.s2_to_grid(half_size, size=192).shape == (12, 192, 192)
assert np.array_equal(P.s2_to_grid(half_size, size=64), np.full((12, 64, 64), 9, np.float32))
raises(ValueError, P.s2_to_grid, half_size)                         # 192 px bands, size 384
for bad_size in (385, 0, -1):
    raises(ValueError, P.s2_to_grid, flat, size=bad_size)
for name, shape in (("B05", (384, 384)), ("B01", (64, 63)), ("B01", (192, 192)),
                    ("B08", (192, 192)), ("B12", (191, 191)), ("B09", (3, 64, 64))):
    raises(ValueError, P.s2_to_grid, {**flat, name: np.zeros(shape, np.uint16)})
for shape in ((192, 192), (384, 383), (12, 384, 384), (384,), ()):
    raises(ValueError, P.s2_to_grid, {**flat, "B02": np.zeros(shape, np.uint16)})
no_sixth = {name: np.zeros((2560 // R.S2_RES_M[name],) * 2, np.uint16) for name in R.S2_BANDS}
raises(ValueError, P.s2_to_grid, no_sixth, size=256)       # 256 px at 10 m is 42.67 px at 60 m
raises(ValueError, P.s2_to_grid, {k: v for k, v in flat.items() if k != "B09"})
print("s2_to_grid: ok")

# ---------------------------------------------------------------- s2_to_grid, the corner rule


def block(a: np.ndarray, k: int) -> np.ndarray:
    """Mean over k x k blocks that start at pixel 0."""
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def box_mean(a: np.ndarray, k: int) -> np.ndarray:
    """Mean of `a` over a k x k window centred on each pixel, the ends weighted by a half when
    k is even. The same blur as `block` at every position, so it has no lattice phase. The
    k // 2 outermost rows and columns are nan."""
    h = k // 2
    out = a.astype(np.float64)
    for axis in (0, 1):
        x = np.moveaxis(out, axis, 0)
        n = x.shape[0]
        s = np.zeros((n + 1,) + x.shape[1:])
        s[1:] = x.cumsum(0)
        win = s[2 * h + 1:n + 1] - s[0:n - 2 * h]                  # k + 1 terms around pixel i
        if k % 2 == 0:
            win = win - 0.5 * (x[0:n - 2 * h] + x[2 * h:n])
        res = np.full_like(x, np.nan)
        res[h:n - h] = win / k
        out = np.moveaxis(res, 0, axis)
    return out


def peak(ref: np.ndarray, band: np.ndarray, reach: int = 4) -> tuple[int, int]:
    """(dy, dx) of the largest correlation of `band` with `ref` over whole shifts of up to
    `reach` px, the convention of the measurement, band[r, c] ~ ref[r - dy, c - dx]."""
    h, w = ref.shape
    core = band[reach:h - reach, reach:w - reach].astype(np.float64)
    bb = core - core.mean()
    best, at = -2.0, None
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            aa = ref[reach - dy:h - reach - dy, reach - dx:w - reach - dx].astype(np.float64)
            aa = aa - aa.mean()
            r = float((aa * bb).sum() / np.sqrt((aa ** 2).sum() * (bb ** 2).sum()))
            if r > best:
                best, at = r, (dy, dx)
    return at


def cut_bands(world: np.ndarray, origin: int, offsets: dict) -> dict:
    """The 12 bands of a crop of `world` whose 10 m corner is at (origin, origin). A 20 m or
    60 m band is cut `offsets[res]` = (east, south) 10 m px away from that corner, on its own
    lattice, as block means, like the self-check of the measurement."""
    bands = {}
    for name, res in R.S2_RES_M.items():
        east, south = offsets.get(res, (0, 0))
        r0, c0 = origin + south, origin + east
        bands[name] = block(world[r0:r0 + 384, c0:c0 + 384], res // 10).astype(np.uint16)
    return bands


assert P.LATTICE_RES_M == (10, 20, 60) and P.SOUTH_FALSE_NORTHING_M == 10_000_000
assert sorted(set(R.S2_RES_M.values())) == list(P.LATTICE_RES_M)
assert inspect.signature(P.s2_to_grid).parameters["corner"].default is None
assert inspect.signature(P.s2_to_grid).parameters["crs"].default is None

# a synthetic world, the 10 m crop starts 12 px inside it so coarse bands can be cut around it.
# The reference for a coarse band is the world with the same blur at every position, so the
# correlation peak of the band on the 10 m grid tells where its content lies
WORLD = RNG.integers(1000, 5000, (408, 408)).astype(np.float64)
ORIGIN = 12
fine = WORLD[ORIGIN:ORIGIN + 384, ORIGIN:ORIGIN + 384]
refs = {res: box_mean(WORLD, res // 10)[ORIGIN:ORIGIN + 384, ORIGIN:ORIGIN + 384]
        for res in (20, 60)}
assert all(np.isfinite(ref).all() for ref in refs.values())
# north, corner (100052, 5000008). Nearest lattice points, 10 m (100050, 5000010), 20 m
# (100060, 5000000), 60 m (100080, 4999980). So the 20 m bands were cut 10 m east and south
# of the 10 m bands, the 60 m bands 30 m east and south.
# south, corner (100052, 4999992). 10 m (100050, 4999990), 20 m (100060, 5000000), 60 m at
# y = 40 mod 60 (100080, 5000020), where the north rule would say (100080, 4999980). So the
# 20 m bands were cut 10 m east and 10 m north, the 60 m bands 30 m east and 30 m north
for crs, corner, cut, want in (
        ("EPSG:32633", (100052.0, 5000008.0), {20: (1, 1), 60: (3, 3)},
         {20: (-1.0, -1.0), 60: (-3.0, -3.0)}),
        ("EPSG:32733", (100052.0, 4999992.0), {20: (1, -1), 60: (3, -3)},
         {20: (-1.0, 1.0), 60: (-3.0, 3.0)})):
    rule = P.band_shift_rule(corner, crs)
    assert list(rule) == list(R.S2_BANDS)
    for name in R.S2_BANDS:
        assert rule[name] == want.get(R.S2_RES_M[name], (0.0, 0.0)), (crs, name, rule[name])
        assert all(type(v) is float for v in rule[name])
    bands = cut_bands(WORLD, ORIGIN, cut)
    as_stored = P.s2_to_grid(bands)
    fixed = P.s2_to_grid(bands, corner=corner, crs=crs)
    assert fixed.dtype == np.float32 and fixed.shape == (12, 384, 384)
    for i, name in enumerate(R.S2_BANDS):
        res = R.S2_RES_M[name]
        if res == 10:
            assert np.array_equal(fixed[i], fine) and np.array_equal(as_stored[i], fine), name
            continue
        east, south = cut[res]
        assert peak(refs[res], as_stored[i]) == (-south, -east), (crs, name)   # off, as measured
        assert peak(refs[res], fixed[i]) == (0, 0), (crs, name)                 # in register
        assert fixed[i].min() >= bands[name].min() and fixed[i].max() <= bands[name].max()
    assert np.array_equal(P.s2_to_grid(bands, size=256, corner=corner, crs=crs),
                          P.center_crop(fixed, 256))
    assert P.band_shift_rule(corner, f" {crs.lower()} ") == rule
# the south anchor moves the 60 m lattice by 40 m, the 20 m lattice not at all
assert P.band_shift_rule((100052.0, 4999992.0), "EPSG:32633")["B01"] == (-3.0, -1.0)
assert P.band_shift_rule((100052.0, 4999992.0), "EPSG:32633")["B05"] == (-1.0, 1.0)
assert P.band_shift_rule((100052.0, 5000008.0), "EPSG:32733")["B09"] == (-3.0, 1.0)

# corner=None gives the numbers of the function as it was before the rule, bit for bit


def old_upsample(band: np.ndarray, size: int) -> np.ndarray:
    """`_upsample` as it was before the corner rule (2026-09-30)."""
    n = band.shape[0]
    src = np.clip((np.arange(size) + 0.5) * n / size - 0.5, 0, n - 1)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = src - i0
    band = band.astype(np.float64)
    rows = band[i0] * (1 - w)[:, None] + band[i1] * w[:, None]
    return rows[:, i0] * (1 - w) + rows[:, i1] * w


grid = P.s2_to_grid(noise)
for i, name in enumerate(R.S2_BANDS):
    old = noise[name] if R.S2_RES_M[name] == 10 else old_upsample(noise[name], 384)
    assert grid[i].tobytes() == old.astype(np.float32).tobytes(), name
assert P.s2_to_grid(noise, size=256).tobytes() == P.center_crop(grid, 256).tobytes()

# a corner on every lattice gives no shift and the same numbers as corner=None
for corner, crs in (((600000.0, 5400000.0), "EPSG:32633"), ((600000.0, 9000040.0), "EPSG:32733"),
                    ((600000, 5400000), "EPSG:32601"), ((0.0, 0.0), "EPSG:32660")):
    rule = P.band_shift_rule(corner, crs)
    assert all(v == (0.0, 0.0) for v in rule.values()), (corner, crs, rule)
    assert P.s2_to_grid(noise, corner=corner, crs=crs).tobytes() == grid.tobytes(), (corner, crs)
# 40 mod 60 is not on the northern 60 m lattice, 0 mod 60 not on the southern one
assert P.band_shift_rule((600000.0, 9000040.0), "EPSG:32633")["B01"] == (0.0, 2.0)
assert P.band_shift_rule((600000.0, 5400000.0), "EPSG:32733")["B09"] == (0.0, -2.0)
assert P.band_shift_rule((600000.0, 5400000.0), "EPSG:32733")["B8A"] == (0.0, 0.0)
# a half rounds up, as in the measurement
assert P.band_shift_rule((100010.0, 100010.0), "EPSG:32633")["B05"] == (-1.0, 1.0)
assert P.band_shift_rule((100003.0, 100003.0), "EPSG:32633")["B05"] == (0.0, 0.0)
assert P.band_shift_rule((100007.0, 100007.0), "EPSG:32633")["B05"] == (1.0, -1.0)

# corner and crs go together, and only a UTM zone is known
CORNER = (100052.0, 5000008.0)
raises(ValueError, P.s2_to_grid, noise, corner=CORNER)
raises(ValueError, P.s2_to_grid, noise, crs="EPSG:32633")
for bad in ("EPSG:4326", "EPSG:3031", "EPSG:32600", "EPSG:32661", "EPSG:32700", "EPSG:32761",
            "32633", "", None, "UTM 33N"):
    raises(ValueError, P.band_shift_rule, CORNER, bad)
    raises(ValueError, P.s2_to_grid, noise, corner=CORNER, crs=bad)
for bad in ((1.0,), (1.0, 2.0, 3.0), None, (float("nan"), 1.0), (1.0, float("inf")), ("a", 1.0),
            (100052.0, 5000008.0, 4996168.0, 103892.0)):
    raises(ValueError, P.band_shift_rule, bad, "EPSG:32633")
    raises(ValueError, P.s2_to_grid, noise, corner=bad, crs="EPSG:32633")

# the rule column of the measured table of part three's stage 0, every real cell in it. The
# file is named by SARTRANSFER_TEST_BAND_SHIFT (a local path, not part of this repository);
# skipped without it
table = Path(os.environ.get("SARTRANSFER_TEST_BAND_SHIFT", ""))
if table.name and table.is_file():
    measured = pd.read_csv(table)
    checked = 0
    for (cell, crs, left, top), rows in measured.groupby(["grid_cell", "crs", "left", "top"]):
        rule = P.band_shift_rule((left, top), crs)
        for _, row in rows.iterrows():
            want = (float(row["rule_nearest_s40_dx"]), float(row["rule_nearest_s40_dy"]))
            assert rule[row["band"]] == want, (cell, row["band"], rule[row["band"]], want)
        checked += 1
    assert checked >= 20, checked
    print(f"band_shift_rule against the measured table: {checked} cells match")
else:
    print(f"band_shift_rule against the measured table: skipped, {table} is not here")
print("s2_to_grid, the corner rule: ok")

# ---------------------------------------------------------------- center_crop
cube = np.arange(12 * 384 * 384, dtype=np.int32).reshape(12, 384, 384)
cut = P.center_crop(cube, 256)
assert cut.shape == (12, 256, 256) and np.array_equal(cut, cube[:, 64:320, 64:320])
assert np.shares_memory(cut, cube)
assert P.center_crop(cube, 384).shape == (12, 384, 384)
assert np.array_equal(P.center_crop(cube[0], 2), cube[0, 191:193, 191:193])
five = np.arange(25).reshape(5, 5)
assert np.array_equal(P.center_crop(five, 2), five[1:3, 1:3])        # odd rest, rounded down
assert np.array_equal(P.center_crop(five, 3), five[1:4, 1:4])
wide_arr = np.arange(24).reshape(4, 6)
assert np.array_equal(P.center_crop(wide_arr, 2), wide_arr[1:3, 2:4])
assert P.center_crop(np.zeros((3, 2, 12, 64, 64)), 32).shape == (3, 2, 12, 32, 32)
tensor = torch.arange(2 * 384 * 384).reshape(2, 384, 384)
assert torch.equal(P.center_crop(tensor, 256), tensor[:, 64:320, 64:320])
for bad_size in (385, 0, -1):
    raises(ValueError, P.center_crop, cube, bad_size)
raises(ValueError, P.center_crop, wide_arr, 5)
# Sentinel-1 and Sentinel-2 of one pair are cut to the same 256 px
pair = kept(P.make_pair(s2_cell(), s1_cell(100, 250)))
assert np.array_equal(P.center_crop(pair.s1_db, 256)[0],
                      P.to_db(VV[164:420, 314:570]).astype(np.float16))
assert P.center_crop(P.s2_to_grid(pair.s2), 256).shape == (12, 256, 256)
print("center_crop: ok")

# ---------------------------------------------------------------- save_pair and load_pair


def same(a, b) -> bool:
    """Equal in value AND in type, all the way down."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, pd.Timestamp):
        return a == b and a.unit == b.unit and a.utcoffset() == b.utcoffset()
    if isinstance(a, float) and a != a:
        return b != b
    return a == b


def same_pair(a: R.Pair, b: R.Pair) -> bool:
    return all(same(getattr(a, f), getattr(b, f)) for f in a.__dataclass_fields__)


noisy = s2_cell(date="2022-02-03T10:20:31.123456Z", cloud_mask=clouds(9000),
                product_id=s2_id("N0509"))
for name in R.S2_BANDS:
    noisy.bands[name] = RNG.integers(1, 65536, noisy.bands[name].shape).astype(np.uint16)
pair = kept(P.make_pair(noisy, s1_cell(100.2, 250.7, date="2022-02-01T05:10:11Z"),
                        cloud_codes=(3, 7)))
assert pair.meta["offset_removed"] is True
assert int(pair.s2["B02"].max()) == int(noisy.bands["B02"].max()) - 1000
pair.meta.update(note="text with a : and an ä", nothing=None, flag=True, count=3, ratio=0.1,
                 not_a_number=float("nan"), nested={"a": (1, 2.5, "z", None), "b": [1, (2, 3)]},
                 when=ts("2026-09-29 12:00"), empty_tuple=(), empty_list=[])
path = TMP / "pair_a.npz"
P.save_pair(pair, path)
assert path.is_file() and sorted(p.name for p in TMP.iterdir()) == ["pair_a.npz"]
back = P.load_pair(path)
assert same_pair(back, pair)
assert back.s1_db.dtype == np.float16 and back.s1_db.shape == (2, 384, 384)
assert back.cloud_mask.dtype == np.uint8 and list(back.s2) == list(R.S2_BANDS)
assert all(back.s2[n].dtype == np.uint16 for n in R.S2_BANDS)
assert back.s2_datetime == ts("2022-02-03 10:20:31.123456")
assert back.s2_datetime.microsecond == 123456
assert back.s1_datetime == ts("2022-02-01 05:10:11") and str(back.s1_datetime.tz) == "UTC"
assert type(back.bounds) is tuple and type(back.meta["shift_px"]) is tuple
assert type(back.meta["s1_window"]) is tuple and type(back.meta["cloud_codes"]) is list
assert back.meta["flag"] is True and type(back.meta["count"]) is int
assert back.meta["when"] == ts("2026-09-29 12:00") and back.meta["nested"]["b"][1] == (2, 3)
assert back.meta["baseline"] == "N0509" and back.meta["offset_removed"] is True
assert back.meta["offset_how"] == "baseline"
assert not same_pair(back, kept(P.make_pair(noisy, s1_cell(100.2, 250.7,
                                                           date="2022-02-01T05:10:12Z"),
                                            cloud_codes=(3, 7))))     # the check can fail
with np.load(path, allow_pickle=False) as z:                         # readable without pickle
    assert set(z.files) == {"info", "s1_db", "cloud_mask"} | {f"s2_{n}" for n in R.S2_BANDS}
# the info alone, the same values as the pair, without any array
info = P.load_pair_info(path)
assert list(info) == ["format", "grid_cell", "gap_days", "s2_datetime", "s1_datetime", "crs",
                      "bounds", "meta", "s2_bands"], list(info)
assert info["format"] == P.FILE_FORMAT and info["grid_cell"] == CELL
assert same(info["meta"], back.meta) and same(info["bounds"], back.bounds)
assert same(info["s2_datetime"], back.s2_datetime) and same(info["s1_datetime"], back.s1_datetime)
assert info["gap_days"] == back.gap_days and info["crs"] == back.crs
assert info["s2_bands"] == list(R.S2_BANDS)
with np.load(path, allow_pickle=False) as z:                         # a file with the info only
    np.savez(TMP / "info_only.npz", info=z["info"])
assert same(P.load_pair_info(TMP / "info_only.npz"), info)           # the arrays are not touched
raises(KeyError, P.load_pair, TMP / "info_only.npz")
(TMP / "info_only.npz").unlink()

# the file is at exactly the given path, also without a suffix, and an old file is replaced
bare = TMP / "pair_without_suffix"
plain = kept(P.make_pair(s2_cell(cloud_mask=None), s1_cell()))
P.save_pair(plain, bare)
P.save_pair(plain, path)
assert sorted(p.name for p in TMP.iterdir()) == ["pair_a.npz", "pair_without_suffix"]
stored = plain.s1_db.nbytes + sum(band.nbytes for band in plain.s2.values())
assert stored == 2 * 384 * 384 * 2 + (4 * 384 * 384 + 6 * 192 * 192 + 2 * 64 * 64) * 2
assert stored <= bare.stat().st_size < stored + 20000            # no compression
for p in (bare, str(path)):
    back = P.load_pair(p)
    assert same_pair(back, plain) and back.cloud_mask is None
    assert back.meta["cloud_rule"] == "not applied" and back.meta["orbit_state"] == "ascending"
    assert (back.meta["baseline"], back.meta["offset_removed"], back.meta["offset_how"]) == (
        "N0300", False, "baseline")
# the meta of a pair made without the offset rule, and without a baseline, comes back the same
none = kept(P.make_pair(s2_cell(product_id="S2A_TEST"), s1_cell(), offset=None))
P.save_pair(none, path)
back = P.load_pair(path)
assert same_pair(back, none)
assert back.meta["baseline"] is None and back.meta["offset_removed"] is False
assert back.meta["offset_how"] == "none"

# a value that cannot be stored is an error before a file is written
plain.meta["bad"] = {1, 2}
raises(TypeError, P.save_pair, plain, TMP / "never.npz")
plain.meta["bad"] = {1: "key is not text"}
raises(TypeError, P.save_pair, plain, TMP / "never.npz")
plain.meta["bad"] = np.zeros(3)
raises(TypeError, P.save_pair, plain, TMP / "never.npz")
assert sorted(p.name for p in TMP.iterdir()) == ["pair_a.npz", "pair_without_suffix"]
del plain.meta["bad"]
plain.meta["from_numpy"] = np.float32(0.5)               # numpy numbers come back as numbers
P.save_pair(plain, path)
assert P.load_pair(path).meta["from_numpy"] == 0.5

# a file of another layout is refused
other = TMP / "other_format.npz"
np.savez(other, info=np.array('{"format": 99}'), s1_db=np.zeros((2, 4, 4), np.float16))
raises(ValueError, P.load_pair, other)
raises(ValueError, P.load_pair_info, other)
for old in TMP.iterdir():
    old.unlink()
print("save_pair and load_pair: ok")

print("test_pairing: all checks passed")
