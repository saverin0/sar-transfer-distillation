"""Stage 0 measurements and checks, one check per rule and guard. Synthetic data, no network."""
import _setup  # noqa: F401

import gc
import inspect
import json
import math
import os
import shutil
import sys
import threading
import time
import weakref

import numpy as np
import pandas as pd
import torch

from sartransfer.distil import stage0 as Z

# ---- the other modules are imported inside the functions, not with stage0 -----------------
LAZY = ("sartransfer.distil.majortom", "sartransfer.distil.pairing",
        "sartransfer.distil.student", "matplotlib", "psutil")
early = [name for name in LAZY if name in sys.modules]
assert not early, f"stage0 imported {early} when it was imported itself"

from sartransfer.distil import majortom as M  # noqa: E402
from sartransfer.distil import pairing as P  # noqa: E402
from sartransfer.distil import records as R  # noqa: E402
from sartransfer.distil.student import MAJORTOM_MODES, Heads, tiny_student  # noqa: E402
from sartransfer.distil.teachers import Teacher, TeacherOut, band_index  # noqa: E402

TMP = _setup.TMP_BASE / "stage0"
for _ in range(40):                    # Windows may take a moment to let go of a folder
    shutil.rmtree(TMP, ignore_errors=True)
    if not TMP.exists():
        break
    time.sleep(0.25)
TMP.mkdir(parents=True)                # files of an older run would change what is tested
NAN = float("nan")


def raises(kind, fn, *args, says: str = "", **kwargs) -> None:
    """`fn` must stop with `kind`, and with `says` in the message when that is given."""
    try:
        fn(*args, **kwargs)
    except kind as err:
        if says not in str(err):
            raise SystemExit(f"wrong message from {fn.__name__}, wanted {says!r}, got {err}")
        return
    raise SystemExit(f"{fn.__name__} did not raise {kind.__name__}")


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def near(a, b, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


# ---- the rules are the pre-registered ones ------------------------------------------------
assert Z.OFFSET_BANDS == ("B02", "B04", "B08") and Z.OFFSET_PERCENTILES == (0.1, 1.0, 50.0)
assert Z.OFFSET_MIN_DN == 950 and Z.OFFSET_FROM == "2022-01-25"
assert (Z.OFFSET_SHARE_HIGH, Z.OFFSET_SHARE_LOW) == (0.90, 0.10)
assert "sentiwiki.copernicus.eu/web/s2-products" in Z.OFFSET_SOURCE
assert "sentiwiki.copernicus.eu/web/s2-processing" in Z.OFFSET_SOURCE
assert "25th January 2022" in Z.OFFSET_SOURCE and "January 24th" in Z.OFFSET_SOURCE
assert Z.TERRAMIND_S1RTC == {"mean_db": (-10.93, -17.329), "sd_db": (4.391, 4.459)}
assert Z.S1_MAX_MEAN_DIFF_DB == 3.0 and Z.S1_SD_RATIO == (0.5, 2.0)
assert "terramind_register.py" in Z.TERRAMIND_SOURCE
assert "github.com/torchgeo/terratorch" in Z.TERRAMIND_SOURCE      # where the file is now
assert Z.MAX_GAP_DAYS == 7.0 and Z.MAX_CLOUD == 0.10
assert inspect.signature(P.make_pair).parameters["max_cloud"].default == Z.MAX_CLOUD
assert Z.ALL_SHARDS == {"s2": 498, "s1": 2930} and Z.SENSORS == ("s2", "s1")
assert inspect.signature(P.pair_index).parameters["max_gap_days"].default == Z.MAX_GAP_DAYS
assert inspect.signature(P.make_pair).parameters["max_gap_days"].default == Z.MAX_GAP_DAYS
assert Z.WARMUP >= 1 and Z.STRETCH_PERCENTILES == (1.0, 99.0)
assert Z.MB == 1e6 and Z.GB == 1e9 and Z.FLOAT16_BYTES == np.dtype(np.float16).itemsize

# ---- titles have no colon -------------------------------------------------------------------
PUBLIC = ("runtime_report", "fetch_pairs", "offset_check", "offset_by_year", "offset_verdict",
          "offset_by_baseline_table", "s1_stats", "gap_report", "time_teacher", "time_student",
          "pca_rgb", "feature_pictures", "budget_table", "save_report")
# the notebook calls them like this, names and order are fixed
ARGUMENTS = {"runtime_report": [], "fetch_pairs": ["pairs", "out_dir", "cloud_codes", "workers",
                                                   "token"],
             "offset_check": ["pair_paths"], "offset_by_year": ["df"], "offset_verdict": ["df"],
             "offset_by_baseline_table": ["df"],
             "s1_stats": ["pair_paths"], "gap_report": ["pairs"],
             "time_teacher": ["teacher", "batch_sizes", "n_batches", "size"],
             "time_student": ["student", "heads", "teacher_outs_like", "batch_sizes", "n_batches",
                              "device"],
             "pca_rgb": ["spatial"],
             "feature_pictures": ["name", "build_fn", "s2_batch", "rules", "out_png"],
             "budget_table": ["teacher_times", "student_times", "n_cells", "epochs"],
             "save_report": ["out_dir"]}
assert set(ARGUMENTS) == set(PUBLIC)
for name, want in ARGUMENTS.items():
    found = list(inspect.signature(getattr(Z, name)).parameters)
    assert found[:len(want)] == want, f"{name} takes {found}"
values = inspect.signature(Z.fetch_pairs).parameters
assert (values["cloud_codes"].default, values["workers"].default, values["token"].default) \
    == (None, 8, None)
assert values["offset"].default is None, "the stage 0 notebook stores the values as read"
# the per-image reading of the offset is the same in pairing and here, one rule, two places
assert (P.OFFSET_BANDS, P.OFFSET_PERCENTILE, P.OFFSET_MIN_DN) == (Z.OFFSET_BANDS,
                                                                  Z.OFFSET_PERCENTILES[0],
                                                                  Z.OFFSET_MIN_DN)
assert "s2_nodata_in_crop" in R.SKIP_REASONS and R.CLOUD_CODES == (1, 2, 3)
for name in (*PUBLIC, "cloud_report"):
    title = getattr(Z, name).__doc__.strip().splitlines()[0]
    assert title and ":" not in title, f"{name}: title {title!r}"
assert ":" not in Z.__doc__.strip().splitlines()[0]

# ---- runtime_report -------------------------------------------------------------------------
import psutil  # noqa: E402

gone = TMP / "not there"
report = Z.runtime_report([TMP, gone])
assert set(report) == {"gpu", "gpu_memory_gb", "gpu_free_gb", "ram_gb", "ram_free_gb", "cpus",
                       "disk"}
assert report["ram_gb"] == psutil.virtual_memory().total / 1e9
assert 0 < report["ram_free_gb"] <= report["ram_gb"]
assert list(report["disk"]) == [str(TMP), str(gone)]
assert report["disk"][str(TMP)]["total_gb"] == shutil.disk_usage(TMP).total / 1e9
assert 0 < report["disk"][str(TMP)]["free_gb"] <= report["disk"][str(TMP)]["total_gb"]
assert report["disk"][str(gone)] == {"total_gb": None, "free_gb": None}
if torch.cuda.is_available():
    assert report["gpu"] and 0 < report["gpu_free_gb"] <= report["gpu_memory_gb"]
else:
    assert report["gpu"] is None and report["gpu_memory_gb"] is None
    assert report["gpu_free_gb"] is None
assert os.getcwd() in Z.runtime_report()["disk"]
json.dumps(report)

# ---- fetch_pairs, with readers made for the test ------------------------------------------------
LEFT, TOP = 500000.0, 5200000.0                # corner of the Sentinel-2 crop, metres
RNG = np.random.default_rng(0)
VV = RNG.uniform(0.01, 0.5, (1068, 1068)).astype(np.float32)      # linear power
VH = RNG.uniform(0.001, 0.1, (1068, 1068)).astype(np.float32)
FAKE_TOKEN = "FAKE-TOKEN-ONLY-FOR-THIS-TEST"
CELLS = [f"10U_{k}R" for k in range(1, 8)]
# cell 1, 2 fine | 3 no-data in the crop | 4 Sentinel-2 read fails | 5 half cloud
# cell 6 Sentinel-1 read fails and the error text holds the token | 7 saving fails
# What the readers made for the test say that one read cost, (megabytes, seconds)
COST = {"s2": {"footer": (4.0, 0.5), "row": (1.5, 0.25), "images": 13},
        "s1": {"footer": (1.0, 0.125), "row": (10.0, 1.0), "images": 2}}
DECODE_S = 0.0625                              # per image
STATS_KEYS = ("requests", "bytes", "seconds", "footers", "footer_requests", "footer_bytes",
              "footer_seconds", "row_requests", "row_bytes", "row_seconds", "retries",
              "wait_seconds", "decoded", "decode_seconds")
assert set(STATS_KEYS) <= set(M.STATS), "majortom no longer counts what stage0 reads"


def s2_cell(cell: str, k: int) -> R.S2Cell:
    """A regular Sentinel-2 cell, every pixel is 1000 + k. Cell 5 is half cloud (code 3)."""
    bands, transforms = {}, {}
    for name in R.S2_BANDS:
        res = R.S2_RES_M[name]
        bands[name] = np.full((3840 // res, 3840 // res), 1000 + k, np.uint16)
        transforms[name] = (float(res), 0.0, LEFT, 0.0, -float(res), TOP)
    mask = np.zeros((384, 384), np.uint8)
    if k == 5:
        mask[:192] = 3
    return R.S2Cell(cell, f"S2_{cell}", ts("2021-06-10 10:00"), bands, transforms, "EPSG:32633",
                    mask, (LEFT, TOP - 3840.0, LEFT + 3840.0, TOP))


def s1_cell(cell: str, k: int) -> R.S1Cell:
    """A regular Sentinel-1 cell, VV is VV * k. The crop starts at its pixel (342, 342)."""
    vv = VV * np.float32(k)  # noqa: F821  del below
    if k == 3:
        vv[500, 500] = -32768.0
    transform = (10.0, 0.0, LEFT - 3420.0, 0.0, -10.0, TOP + 3420.0)
    return R.S1Cell(cell, f"S1_{cell}", ts("2021-06-12 05:00"), vv, VH, transform, "EPSG:32633",  # noqa: F821  del below
                    -32768.0, "ascending")


CALLS: list[tuple] = []
LOCK = threading.Lock()


def note(kind: str, row, token) -> int:
    """Write the call down and count one footer, one row and the images of the cell."""
    (footer_mb, footer_s), (row_mb, row_s) = COST[kind]["footer"], COST[kind]["row"]
    images = COST[kind]["images"]
    more = {"requests": 3, "bytes": int((footer_mb + row_mb) * 1e6), "seconds": footer_s + row_s,
            "footers": 1, "footer_requests": 2, "footer_bytes": int(footer_mb * 1e6),
            "footer_seconds": footer_s, "row_requests": 1, "row_bytes": int(row_mb * 1e6),
            "row_seconds": row_s, "retries": 1, "wait_seconds": 2.0, "decoded": images,
            "decode_seconds": images * DECODE_S}
    with LOCK:
        CALLS.append((kind, row["grid_cell"], threading.get_ident(), token, dict(row)))
        for name, value in more.items():
            if name in M.STATS:
                M.STATS[name] += value
    time.sleep(0.05)
    return CELLS.index(row["grid_cell"]) + 1


def fake_read_s2(row, token=None) -> R.S2Cell:
    k = note("s2", row, token)
    if k == 4:
        raise OSError("the shard is gone")
    return s2_cell(row["grid_cell"], k)


def fake_read_s1(row, token=None) -> R.S1Cell:
    k = note("s1", row, token)
    if k == 6:
        raise PermissionError(f"HTTP 401 for the request with {token}")
    return s1_cell(row["grid_cell"], k)


REAL_SAVE = P.save_pair


def save_or_fail(pair, path) -> None:
    if pair.grid_cell == CELLS[6]:
        raise OSError("the disk is full")
    REAL_SAVE(pair, path)


def calls_of(kind: str, since: int = 0) -> list[str]:
    return sorted(c[1] for c in CALLS[since:] if c[0] == kind)


fetch_table = pd.DataFrame({
    "grid_cell": CELLS,
    "s2_url": [f"s2_{k // 2}.parquet" for k in range(7)], "s2_row": [10 + k for k in range(7)],
    "s2_product_id": [f"S2_{c}" for c in CELLS], "s2_datetime": ts("2021-06-10 10:00"),
    "s1_url": [f"s1_{k}.parquet" for k in range(7)], "s1_row": [20 + k for k in range(7)],
    "s1_product_id": [f"S1_{c}" for c in CELLS], "s1_datetime": ts("2021-06-12 05:00"),
    "gap_days": 1.7917, "s2_year": 2021})
REAL_READERS = (M.read_s2_cell, M.read_s1_cell)
REAL_STATS = M.STATS
M.read_s2_cell, M.read_s1_cell, P.save_pair = fake_read_s2, fake_read_s1, save_or_fail
try:
    out = TMP / "pairs" / "run"                   # the folder is made by fetch_pairs
    paths, skips, timing = Z.fetch_pairs(fetch_table, out, cloud_codes=(c for c in [3, 3]),
                                         workers=4, token=FAKE_TOKEN, progress=False)

    # what was saved and what was skipped, with the reasons
    assert paths == [out / f"{c}.npz" for c in CELLS[:2]] and all(p.is_file() for p in paths)
    assert sorted(p.name for p in out.iterdir()) == [f"{c}.npz" for c in CELLS[:2]]
    assert list(skips.columns) == list(Z.SKIP_COLUMNS) and list(skips["grid_cell"]) == CELLS[2:]
    assert list(skips["reason"]) == ["s1_nodata_in_crop", "read_error", "cloud_gt_10pct",
                                     "read_error", "read_error"]
    assert set(skips["reason"]) <= set(R.SKIP_REASONS)
    detail = dict(zip(skips["grid_cell"], skips["detail"]))
    assert detail[CELLS[3]] == "read_s2 failed, OSError, the shard is gone"
    assert detail[CELLS[5]].startswith("read_s1 failed, PermissionError, HTTP 401")
    assert detail[CELLS[6]] == "save failed, OSError, the disk is full"
    assert all(detail.values())

    # the token reaches the readers and no table
    assert {c[3] for c in CALLS} == {FAKE_TOKEN}
    assert "<token>" in detail[CELLS[5]]
    for table in (skips, timing):
        assert FAKE_TOKEN not in table.to_csv()
        assert FAKE_TOKEN not in json.dumps(Z._plain(table.attrs))

    # the readers got the rows of their own sensor
    sent = {(c[0], c[1]): c[4] for c in CALLS}
    assert sent[("s2", CELLS[1])] == {
        "grid_cell": CELLS[1], "product_id": f"S2_{CELLS[1]}", "parquet_url": "s2_0.parquet",
        "product_datetime": ts("2021-06-10 10:00"), "parquet_row": 11}
    assert sent[("s1", CELLS[1])] == {
        "grid_cell": CELLS[1], "product_id": f"S1_{CELLS[1]}", "parquet_url": "s1_1.parquet",
        "product_datetime": ts("2021-06-12 05:00"), "parquet_row": 21}
    assert calls_of("s2") == CELLS and calls_of("s1") == CELLS[:3] + CELLS[4:]

    # each file holds the two images of its own cell
    for k, path in enumerate(paths, start=1):
        pair = P.load_pair(path)
        assert pair.grid_cell == CELLS[k - 1] and pair.s1_db.shape == (2, 384, 384)
        assert all((pair.s2[name] == 1000 + k).all() for name in R.S2_BANDS)
        want = P.to_db(VV[342:726, 342:726] * np.float32(k)).astype(np.float16)
        assert np.array_equal(pair.s1_db[0], want)
        assert pair.meta["cloud_rule"] == "applied" and pair.meta["cloud_codes"] == [3]
        assert (pair.meta["baseline"], pair.meta["offset_removed"], pair.meta["offset_how"]) == (
            None, False, "none")                         # no rule, the product id has no baseline

    # the timing table
    assert list(timing.columns) == list(Z.TIMING_COLUMNS) and list(timing["grid_cell"]) == CELLS
    assert list(timing["status"]) == ["saved"] * 2 + ["skipped"] * 5
    steps = [f"{s}_s" for s in Z.FETCH_STEPS]
    assert steps == ["read_s2_s", "read_s1_s", "make_pair_s", "save_s"]
    reached = {1: 4, 2: 4, 3: 3, 4: 0, 5: 3, 6: 1, 7: 3}      # steps that ended, per cell
    for k, row in enumerate(timing.to_dict("records"), start=1):
        done = [not math.isnan(row[s]) for s in steps]
        assert done == [True] * reached[k] + [False] * (4 - reached[k]), (k, row)
        assert all(row[s] >= 0 for s in steps[:reached[k]])
        assert row["total_s"] >= sum(row[s] for s in steps[:reached[k]]) and row["total_s"] > 0
        if reached[k] >= 1:
            assert 0.05 <= row["read_s2_s"] < 5
        if reached[k] >= 2:
            assert 0.05 <= row["read_s1_s"] < 5
            # 12 bands 1,638,400 + cloud mask 147,456 + VV and VH 9,124,992 bytes
            assert row["mb_decoded"] == 10.910848
        else:
            assert math.isnan(row["mb_decoded"])
        if row["status"] == "saved":
            assert row["mb_saved"] == paths[k - 1].stat().st_size / 1e6 and row["mb_saved"] > 2
        else:
            assert math.isnan(row["mb_saved"])

    # threads, and the totals of the call. 7 Sentinel-2 reads and 6 Sentinel-1 reads
    assert len({c[2] for c in CALLS}) > 1, "4 workers used one thread"
    totals = timing.attrs
    assert (totals["cells"], totals["cells_read"], totals["workers"]) == (7, 7, 4)
    assert totals["cells_saved"] == 2
    assert (totals["shards_s2"], totals["shards_s1"]) == (4, 7)
    assert 0 < totals["wall_s"] < timing["total_s"].sum() == totals["thread_s"]
    assert near(totals["cells_per_s"], 7 / totals["wall_s"])
    assert set(totals["counted"]) == set(M.STATS)
    assert totals["requests"] == 39 and totals["footers"] == 13 and totals["retries"] == 13
    assert all(type(totals[name]) is int for name in ("requests", "footers", "retries"))
    for name, want in (("mb_footers", 7 * 4.0 + 6 * 1.0), ("mb_rows", 7 * 1.5 + 6 * 10.0),
                       ("mb_fetched", 7 * 5.5 + 6 * 11.0),
                       ("footer_s", 7 * 0.5 + 6 * 0.125), ("row_s", 7 * 0.25 + 6 * 1.0),
                       ("decode_s", (7 * 13 + 6 * 2) * DECODE_S), ("wait_s", 26.0),
                       ("s_per_footer", (7 * 0.5 + 6 * 0.125) / 13),
                       ("mb_per_footer", (7 * 4.0 + 6 * 1.0) / 13),
                       ("mb_rows_per_cell", (7 * 1.5 + 6 * 10.0) / 7),
                       ("mb_fetched_per_cell", (7 * 5.5 + 6 * 11.0) / 7)):
        assert near(totals[name], want), (name, totals[name], want)
    per_cell = totals["s_per_cell"]
    assert list(per_cell) == ["fetch_rows", "decode", "make_pair", "save", "wait", "rest"]
    assert near(per_cell["fetch_rows"], 7.75 / 7) and near(per_cell["wait"], 26.0 / 7)
    assert near(per_cell["decode"], 103 * DECODE_S / 7)
    assert near(per_cell["make_pair"], timing["make_pair_s"].sum() / 7)
    assert near(per_cell["save"], timing["save_s"].sum() / 7) and per_cell["save"] > 0
    # the steps and the footers add up to the time of the threads
    assert near(sum(per_cell.values()) * 7 + totals["footer_s"], totals["thread_s"])
    json.dumps(Z._plain(totals))

    # resumable - a pair on disk is not read again, a skipped cell is
    stamp = [p.stat().st_mtime_ns for p in paths]
    n_calls = len(CALLS)
    paths2, skips2, timing2 = Z.fetch_pairs(fetch_table, out, cloud_codes=[3], workers=1,
                                            token=FAKE_TOKEN, progress=False)
    assert calls_of("s2", n_calls) == CELLS[2:], "a pair on disk was read again"
    assert paths2 == paths and [p.stat().st_mtime_ns for p in paths] == stamp
    assert skips2.equals(skips)
    assert list(timing2["status"]) == ["on_disk"] * 2 + ["skipped"] * 5
    on_disk = timing2[timing2["status"] == "on_disk"]
    assert on_disk[steps + ["total_s", "mb_decoded"]].isna().all().all()
    assert list(on_disk["mb_saved"]) == [p.stat().st_size / 1e6 for p in paths]
    assert (timing2.attrs["cells"], timing2.attrs["cells_read"]) == (7, 5)
    assert timing2.attrs["cells_saved"] == 0
    assert (timing2.attrs["shards_s2"], timing2.attrs["shards_s1"]) == (3, 5)
    assert near(timing2.attrs["mb_fetched"], 5 * 5.5 + 4 * 11.0)
    assert near(timing2.attrs["mb_rows_per_cell"], (5 * 1.5 + 4 * 10.0) / 5)
    assert near(timing2.attrs["mb_fetched_per_cell"], (5 * 5.5 + 4 * 11.0) / 5)
    assert near(timing2.attrs["s_per_cell"]["fetch_rows"], (5 * 0.25 + 4 * 1.0) / 5)
    assert near(timing2.attrs["s_per_cell"]["save"], 0.0) and timing2.attrs["footers"] == 9
    assert len({c[2] for c in CALLS[n_calls:]}) == 1, "1 worker used more than one thread"

    # a pair on disk that was made under another cloud rule stops the call, nothing is read
    n_calls = len(CALLS)
    for other in (None, [2], [2, 3]):
        raises(ValueError, Z.fetch_pairs, fetch_table, out, cloud_codes=other, progress=False,
               says="another folder")
    assert len(CALLS) == n_calls, "cells were read although the folder holds another rule"
    again = Z.fetch_pairs(fetch_table.iloc[:2], out, cloud_codes=[3.0], progress=False)
    assert again[0] == paths and len(CALLS) == n_calls

    # without cloud codes the cloud rule is not applied
    paths3, skips3, _ = Z.fetch_pairs(fetch_table.iloc[[4]], TMP / "pairs" / "no_codes",
                                      workers=2, progress=False)
    assert [p.name for p in paths3] == [f"{CELLS[4]}.npz"] and len(skips3) == 0
    assert P.load_pair(paths3[0]).meta["cloud_rule"] == "not applied"
    assert CALLS[-1][3] is None
    raises(ValueError, Z.fetch_pairs, fetch_table.iloc[[4]], TMP / "pairs" / "no_codes",
           cloud_codes=[3], progress=False, says="another folder")
    # a file on disk that is no pair stops the call and is named
    n_calls = len(CALLS)
    (TMP / "pairs" / "broken").mkdir()
    (TMP / "pairs" / "broken" / f"{CELLS[1]}.npz").write_bytes(b"this is not a pair")
    raises(ValueError, Z.fetch_pairs, fetch_table.iloc[:3], TMP / "pairs" / "broken",
           progress=False, says=f"{CELLS[1]}.npz is on disk and cannot be read")
    assert len(CALLS) == n_calls

    # guards, all of them before anything is read and before the folder is made
    n_calls = len(CALLS)
    raises(ValueError, Z.fetch_pairs, pd.concat([fetch_table, fetch_table.iloc[[0]]]),
           TMP / "pairs" / "guard", progress=False, says=CELLS[0])
    for column in ("s2_url", "s1_row", "s1_product_id"):
        raises(KeyError, Z.fetch_pairs, fetch_table.drop(columns=column),
               TMP / "pairs" / "guard", progress=False, says=column)
    for codes in ([], [3.5], [-1], [256], "3", ["cloud"], 3):
        raises(ValueError, Z.fetch_pairs, fetch_table, TMP / "pairs" / "guard", cloud_codes=codes,
               progress=False, says="whole numbers from 0 to 255")
    for rule in ("date", "", True, "Baseline"):
        raises(ValueError, Z.fetch_pairs, fetch_table, TMP / "pairs" / "guard", offset=rule,
               progress=False, says="offset")
    assert len(CALLS) == n_calls and not (TMP / "pairs" / "guard").exists()
    # with the offset rule on, a product id without a baseline cannot be put on one scale
    n_calls = len(CALLS)
    scaled = Z.fetch_pairs(fetch_table.iloc[:2], TMP / "pairs" / "scaled", offset="baseline",
                           progress=False)
    assert scaled[0] == [] and list(scaled[1]["reason"]) == ["read_error"] * 2
    assert all(d.startswith("the offset rule is on and") for d in scaled[1]["detail"])
    assert calls_of("s2", n_calls) == CELLS[:2]
    none = Z.fetch_pairs(fetch_table.iloc[:0], TMP / "pairs" / "empty", progress=False)
    assert none[0] == [] and len(none[1]) == 0 and len(none[2]) == 0
    assert list(none[1].columns) == list(Z.SKIP_COLUMNS)
    assert list(none[2].columns) == list(Z.TIMING_COLUMNS)
    assert none[2].attrs["cells_read"] == 0 and none[2].attrs["mb_rows_per_cell"] is None
    assert none[2].attrs["s_per_cell"]["fetch_rows"] is None

    # a library that is missing is no fault of a cell. It stops the run, and the cells that
    # wait are dropped
    def no_library(row, token=None):
        note("s2", row, token)
        raise ModuleNotFoundError("No module named 'rasterio'")

    M.read_s2_cell = no_library
    n_calls = len(CALLS)
    raises(ImportError, Z.fetch_pairs, fetch_table, TMP / "pairs" / "no_library", workers=1,
           progress=False, says="rasterio")
    assert 1 <= len(CALLS) - n_calls <= 3, "the run went on without the library"
    assert not list((TMP / "pairs" / "no_library").iterdir())
    M.read_s2_cell = fake_read_s2

    # a majortom that counts less, or nothing, gives None and never a made-up number
    M.STATS = {"requests": 0, "bytes": 0}
    few = Z.fetch_pairs(fetch_table.iloc[[0]], TMP / "pairs" / "few", progress=False)[2].attrs
    assert few["counted"] == {"requests": 6, "bytes": 16_500_000} and few["mb_fetched"] == 16.5
    assert few["mb_fetched_per_cell"] == 16.5 and few["cells_saved"] == 1
    for name in ("footers", "footer_s", "mb_footers", "row_s", "mb_rows", "decode_s", "wait_s",
                 "retries", "s_per_footer", "mb_per_footer", "mb_rows_per_cell"):
        assert few[name] is None, name
    assert few["s_per_cell"]["rest"] is None and few["s_per_cell"]["fetch_rows"] is None
    assert few["s_per_cell"]["make_pair"] > 0
    del M.STATS
    attrs = Z.fetch_pairs(fetch_table.iloc[[0]], out, cloud_codes=[3], progress=False)[2].attrs
    assert attrs["counted"] is None and attrs["mb_fetched"] is None and attrs["requests"] is None
    assert attrs["cells_read"] == 0 and attrs["cells_per_s"] == 0
finally:
    (M.read_s2_cell, M.read_s1_cell), P.save_pair = REAL_READERS, REAL_SAVE
    M.STATS = REAL_STATS
fake_timing = timing
del VV, VH

# ---- fetch_pairs end to end, local shards through the real majortom and pairing -------------
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import Affine  # noqa: E402

SHARDS = TMP / "shards"
SHARDS.mkdir()
E_LEFT, E_TOP = 500002.5, 4000004.25        # a corner that is not a multiple of 10 m
# cell 0 fine | 1 no-data in the crop | 2 half cloud | 3 Sentinel-1 cell of 96 px
# cell 4 the two dates are 15 days apart | 5 fine, the crop is not in the middle
S1_CORNER = {5: (E_LEFT - 1000.0, E_TOP + 2000.0)}            # else the middle, 342 px in
S2_DAY = ["20220101", "20220102", "20220103", "20220104", "20220105", "20230106"]
S1_DAY = ["20220102", "20220103", "20220104", "20220105", "20220120", "20230107"]


def tif(array, res=10.0, left=E_LEFT, top=E_TOP, nodata=None) -> bytes:
    """One GeoTIFF in memory, one layer, LZW as in Major-TOM."""
    with rasterio.MemoryFile() as memory:
        with memory.open(driver="GTiff", count=1, height=array.shape[0], width=array.shape[1],
                         dtype=array.dtype, crs="EPSG:32633", nodata=nodata, compress="lzw",
                         transform=Affine(res, 0.0, left, 0.0, -res, top)) as dst:
            dst.write(array[None])
        return memory.read()


def s2_stored(i: int) -> dict:
    rng = np.random.default_rng(i)
    day = S2_DAY[i]
    baseline = "N9999" if i == 5 else "N0400"          # cell 5 is decided per image
    row = {"product_id": f"S2A_MSIL2A_{day}T101010_{baseline}_R022_T33TUM_{day}T121212",
           "grid_cell": f"{100 + i}U_{200 + i}R", "product_datetime": f"{day}T101010"}
    for band in R.S2_BANDS:
        px = R.CROP_PX * 10 // R.S2_RES_M[band]
        values = 1000 + 100 * i + rng.integers(0, 8, (px, px))
        row[band] = tif(values.astype(np.uint16), R.S2_RES_M[band])
    mask = np.zeros((R.CROP_PX, R.CROP_PX), np.uint8)
    if i == 2:
        mask[:192] = 3
    row["cloud_mask"] = tif(mask)
    row["thumbnail"] = b"not a picture " * 5000       # 70 kB that must not be fetched
    return row


def s1_stored(i: int) -> dict:
    rng = np.random.default_rng(100 + i)
    size = 96 if i == 3 else R.S1_CELL_PX
    left, top = S1_CORNER.get(i, (E_LEFT - 3420.0, E_TOP + 3420.0))
    row = {"product_id": f"S1A_IW_GRDH_1SDV_{S1_DAY[i]}T051500_x_rtc",
           "grid_cell": f"{100 + i}U_{200 + i}R", "product_datetime": f"{S1_DAY[i]}T051500"}
    for k, name in enumerate(("vv", "vh")):
        values = rng.choice(np.array([0.01, 0.02, 0.05, 0.1], np.float32) / (1 + 3 * k),
                            (size, size))
        if i == 1 and name == "vh":
            values[400, 400] = -32768.0
        row[name] = tif(values.astype(np.float32), left=left, top=top, nodata=-32768.0)
    row["thumbnail"] = b"not a picture " * 5000
    return row


def write_shard(name: str, rows: list) -> str:
    """One row per row group, as in Major-TOM."""
    schema = pa.schema([(k, pa.string() if k in M.META_COLUMNS else pa.binary())
                        for k in rows[0]])
    with pq.ParquetWriter(SHARDS / name, schema, compression="snappy") as writer:
        for row in rows:
            writer.write_table(pa.Table.from_pylist([row], schema=schema))
    return str(SHARDS / name)


S2_STORED, S1_STORED = [s2_stored(i) for i in range(6)], [s1_stored(i) for i in range(6)]
S2_A, S2_B = write_shard("s2_a.parquet", S2_STORED[:3]), write_shard("s2_b.parquet", S2_STORED[3:])
S1_A, S1_B = write_shard("s1_a.parquet", S1_STORED[:2]), write_shard("s1_b.parquet", S1_STORED[2:])
s2_index = M.build_s2_index(shards=[S2_A, S2_B], progress=False, verify=0)
pd.DataFrame({
    "grid_cell": [r["grid_cell"] for r in S1_STORED], "grid_row_u": 0, "grid_col_r": 0,
    "product_id": [r["product_id"] for r in S1_STORED],
    "timestamp": [f"{day}T051512" for day in S1_DAY],        # 12 s after the stored date
    "nodata": 0.0, "orbit_state": "ascending", "centre_lat": 36.1, "centre_lon": 15.0,
    "crs": "EPSG:EPSG:32633", "parquet_url": [S1_A] * 2 + [S1_B] * 4,
    "parquet_row": [0, 1, 0, 1, 2, 3]}).to_parquet(SHARDS / "metadata.parquet", index=False)
s1_index = M.load_s1_index(source=SHARDS / "metadata.parquet")
all_pairs, pair_counts = P.pair_index(s2_index, s1_index, max_gap_days=None)
assert pair_counts["pairs"] == 6 and list(all_pairs.columns) == list(P.PAIR_COLUMNS)
real_gaps = dict(zip(*Z.gap_report(all_pairs)[["what", "pairs"]].to_numpy().T))
assert real_gaps["within 7 days"] == 5 and real_gaps["14 to 30 days"] == 1
assert real_gaps["different years"] == 0 and real_gaps["all pairs"] == 6


def stored_bytes(shard: str, row: int, columns) -> int:
    """Bytes of these columns of one row in the file, from the footer read by pyarrow."""
    footer = pq.read_metadata(shard)
    group = footer.row_group(row)
    return sum(group.column(footer.schema.names.index(c)).total_compressed_size for c in columns)


def footer_bytes(shard: str) -> int:
    """Bytes that one fetch of this footer takes, the first request or the whole footer."""
    size = os.path.getsize(shard)
    return max(min(M.FOOTER_GUESS, size), pq.read_metadata(shard).serialized_size + 8)


S2_ASKED = [*M.META_COLUMNS, *R.S2_BANDS, "cloud_mask"]
S1_ASKED = [*M.META_COLUMNS, "vv", "vh"]
want_rows = sum(stored_bytes(row["s2_url"], row["s2_row"], S2_ASKED)
                + stored_bytes(row["s1_url"], row["s1_row"], S1_ASKED)
                for row in all_pairs.to_dict("records"))
want_footers = sum(footer_bytes(shard) for shard in (S2_A, S2_B, S1_A, S1_B))

M._FOOTERS.clear()                             # no footer is in memory
before = dict(M.STATS)
real_out = TMP / "pairs" / "real"
paths, skips, timing = Z.fetch_pairs(all_pairs, real_out, cloud_codes=[3], workers=2,
                                     progress=False)
counted = {name: M.STATS[name] - before[name] for name in M.STATS}
cell = [row["grid_cell"] for row in S2_STORED]
assert paths == [real_out / f"{cell[0]}.npz", real_out / f"{cell[5]}.npz"]
assert dict(zip(skips["grid_cell"], skips["reason"])) == {
    cell[1]: "s1_nodata_in_crop", cell[2]: "cloud_gt_10pct", cell[3]: "s1_not_1068_10m",
    cell[4]: "gap_gt_7d"}
assert "VH has 1 of 147456" in skips["detail"][0] and "0.5000" in skips["detail"][1]
assert list(timing["status"]) == ["saved"] + ["skipped"] * 4 + ["saved"]
assert timing[["read_s2_s", "read_s1_s", "make_pair_s", "total_s"]].gt(0).all().all()
assert timing["mb_decoded"].tolist() == [10.910848] * 3 + [1.859584] + [10.910848] * 2

# the pair is the stored data, Sentinel-1 cut by the map position of the Sentinel-2 crop
for i, window in ((0, np.s_[342:726, 342:726]), (5, np.s_[200:584, 100:484])):
    pair = P.load_pair(real_out / f"{cell[i]}.npz")
    for k, name in enumerate(("vv", "vh")):
        stored = M.decode_tif(S1_STORED[i][name])[0]
        assert np.array_equal(pair.s1_db[k], P.to_db(stored[window]).astype(np.float16)), (i, name)
    for band in R.S2_BANDS:
        assert np.array_equal(pair.s2[band], M.decode_tif(S2_STORED[i][band])[0]), (i, band)
    assert pair.s2_datetime == pd.Timestamp(f"{S2_DAY[i]}T101010", tz="UTC")
    assert pair.s1_datetime == pd.Timestamp(f"{S1_DAY[i]}T051500", tz="UTC")     # as stored
    assert near(pair.gap_days, 1.0 - (4 * 3600 + 55 * 60 + 10) / 86400)
    assert pair.meta["s2_product_id"] == S2_STORED[i]["product_id"]
    assert pair.meta["shift_px"] == (0.0, 0.0) and pair.meta["s1_window"] == (window[0].start,
                                                                             window[1].start)

# the totals are the counters of majortom, footers apart from rows
totals = timing.attrs
assert totals["counted"] == counted and set(STATS_KEYS) <= set(counted)
assert (totals["cells_read"], totals["cells_saved"]) == (6, 2)
assert (totals["shards_s2"], totals["shards_s1"]) == (2, 2)
assert totals["footers"] == 4, "one footer per shard, not one per cell"
assert totals["mb_footers"] == want_footers / 1e6 == counted["footer_bytes"] / 1e6
assert totals["mb_rows"] == want_rows / 1e6 == counted["row_bytes"] / 1e6
assert totals["mb_fetched"] == (want_footers + want_rows) / 1e6
footer_requests = sum(1 if footer_bytes(s) <= min(M.FOOTER_GUESS, os.path.getsize(s)) else 2
                      for s in (S2_A, S2_B, S1_A, S1_B))
assert totals["requests"] == footer_requests + 12 and counted["row_requests"] == 12
assert totals["mb_rows_per_cell"] == want_rows / 1e6 / 6
assert totals["mb_fetched_per_cell"] == (want_footers + want_rows) / 1e6 / 6
assert totals["mb_per_footer"] == want_footers / 1e6 / 4
assert totals["footer_s"] == counted["footer_seconds"] > 0
assert totals["row_s"] == counted["row_seconds"] > 0
assert totals["decode_s"] == counted["decode_seconds"] > 0
assert counted["decoded"] == 6 * (13 + 2) and totals["retries"] == 0 == totals["wait_s"]
assert totals["s_per_footer"] == counted["footer_seconds"] / 4
per_cell = totals["s_per_cell"]
assert per_cell["fetch_rows"] == counted["row_seconds"] / 6
assert per_cell["decode"] == counted["decode_seconds"] / 6 and per_cell["wait"] == 0
assert near(sum(per_cell.values()) * 6 + totals["footer_s"], totals["thread_s"])
assert per_cell["rest"] > 0, "more time in the parts than in the whole"
json.dumps(Z._plain(totals))
real_timing = timing

# a second call finds the footers in memory, and the pairs on disk
again = Z.fetch_pairs(all_pairs, real_out, cloud_codes=[3], workers=2, progress=False)
assert again[0] == paths and again[1].equals(skips)
assert list(again[2]["status"]) == ["on_disk"] + ["skipped"] * 4 + ["on_disk"]
assert again[2].attrs["footers"] == 0 and again[2].attrs["mb_footers"] == 0
assert again[2].attrs["s_per_footer"] is None and again[2].attrs["cells_read"] == 4
no_footer_timing = again[2]

# without the offset rule (the default) the pairs hold the values as stored
for i in (0, 5):
    meta = P.load_pair(real_out / f"{cell[i]}.npz").meta
    assert (meta["baseline"], meta["offset_removed"], meta["offset_how"]) == (
        "N0400" if i == 0 else "N9999", False, "none")
# with the rule the same cells come out on one scale, by baseline and, for N9999, per image
scaled_out = TMP / "pairs" / "real_scaled"
scaled, scaled_skips, _ = Z.fetch_pairs(all_pairs, scaled_out, cloud_codes=[3], workers=2,
                                        progress=False, offset="baseline")
assert [p.name for p in scaled] == [p.name for p in paths] and scaled_skips.equals(skips)
for i, how in ((0, "baseline"), (5, "per_image")):
    pair = P.load_pair(scaled_out / f"{cell[i]}.npz")
    assert (pair.meta["baseline"], pair.meta["offset_removed"], pair.meta["offset_how"]) == (
        "N0400" if i == 0 else "N9999", True, how)
    for band in R.S2_BANDS:
        stored = M.decode_tif(S2_STORED[i][band])[0].astype(np.int64)
        want_band = np.maximum(stored - 1000, 0).astype(np.uint16)
        assert np.array_equal(pair.s2[band], want_band), (i, band)
        assert pair.s2[band].dtype == np.uint16
    assert pair.meta["s2_product_id"] == S2_STORED[i]["product_id"]

# the checks read these pairs
real_check = Z.offset_check(paths)
assert list(real_check["baseline"]) == ["04.00", "99.99"]
assert list(real_check["year"]) == [2022, 2023]
assert list(real_check["period"]) == ["before", "from"] and real_check["carries_offset"].all()
assert list(real_check["B02_p0.1"]) == [1000.0, 1500.0]
assert Z.offset_by_baseline_table(real_check).to_dict("list") == {
    "baseline": ["N0400", "N9999"], "images": [1, 1], "with_offset": [1, 1],
    "share_with_offset": [1.0, 1.0]}
# on one scale the same images carry no offset any more
assert not Z.offset_check(scaled)["carries_offset"].any()
real_s1 = Z.s1_stats(paths)
assert real_s1["pairs"] == 2 and real_s1["pixels"] == {"VV": 294912, "VH": 294912}
assert -21 < real_s1["mean_db"]["VV"] < -9 and -27 < real_s1["mean_db"]["VH"] < -15

# ---- small pairs for the data checks --------------------------------------------------------
def small_pair(cell: str, date, b02=None, b04=None, b08=None, s1_db=None,
               product: str = "", mask=None) -> R.Pair:
    """A pair with small arrays. The bands that are not given are copies of B02."""
    b02 = np.ones((4, 4), np.uint16) if b02 is None else b02
    s2 = {name: np.ones((4, 4), np.uint16) for name in R.S2_BANDS}
    s2.update(B02=b02, B04=b02.copy() if b04 is None else b04,
              B08=b02.copy() if b08 is None else b08)
    s1_db = np.zeros((2, 4, 4)) if s1_db is None else s1_db
    return R.Pair(cell, np.asarray(s1_db).astype(np.float16), s2, mask, date, date, 0.0,
                  "EPSG:32633", (0.0, 0.0, 1.0, 1.0), {"s2_product_id": product} if product else {})


def saved(pair: R.Pair, folder) -> "os.PathLike":
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{pair.grid_cell}.npz"
    P.save_pair(pair, path)
    return path


def plain() -> np.ndarray:
    """80 x 80 px without the offset, values 1 to 2,999."""
    return RNG.integers(1, 3000, (80, 80)).astype(np.uint16)


def shifted() -> np.ndarray:
    """80 x 80 px with the offset, values 1,000 to 3,999."""
    return (1000 + RNG.integers(0, 3000, (80, 80))).astype(np.uint16)


def percentile(values: np.ndarray, q: float) -> float:
    """Linear interpolation between the sorted values, written out by hand."""
    s = np.sort(values.astype(np.float64))
    pos = q / 100.0 * (len(s) - 1)
    i = int(pos)
    return float(s[i] + (pos - i) * (s[min(i + 1, len(s) - 1)] - s[i]))


# ---- cloud_report -----------------------------------------------------------------------------
def mask_of(**pixels: int) -> np.ndarray:
    """A mask of 10 x 10 px, code 0 but for the given counts, mask_of(c1=10) has 10 px of 1."""
    mask = np.zeros(100, np.uint8)
    at = 0
    for code, n in pixels.items():
        mask[at:at + n] = int(code[1:])
        at += n
    return RNG.permutation(mask).reshape(10, 10)


day = ts("2021-06-10")
cloud_paths = [saved(pair, TMP / "clouds") for pair in (
    small_pair("m1", day, mask=mask_of()),
    small_pair("m2", day, mask=mask_of(c1=10, c2=11)),    # 10 % is not over the limit, 11 % is
    small_pair("m3", day, mask=mask_of(c1=5, c3=6)),      # only together over the limit
    small_pair("m4", day))]                               # no mask
CLOUD_COLUMNS = ["what", "pixels", "pixel_share", "images", "images_over", "image_share"]
clouds = Z.cloud_report(iter(cloud_paths))
assert list(clouds.columns) == CLOUD_COLUMNS
assert clouds.to_dict("list") == {
    "what": ["code 0", "code 1", "code 2", "code 3", "no cloud mask"],
    "pixels": [268, 15, 11, 6, 0], "pixel_share": [268 / 300, 15 / 300, 11 / 300, 6 / 300, 0.0],
    "images": [3, 2, 1, 1, 1], "images_over": [3, 0, 1, 0, 0],
    "image_share": [1.0, 0.0, 1 / 3, 0.0, 0.0]}
assert clouds["pixels"].sum() == 300 and not any(":" in what for what in clouds["what"])
together = Z.cloud_report(cloud_paths, cloud_codes=(3, 1, 1))
assert list(together["what"]) == ["code 0", "code 1", "code 2", "code 3", "codes 1 3 together",
                                  "no cloud mask"]
assert together.iloc[4].tolist() == ["codes 1 3 together", 21, 21 / 300, 2, 1, 1 / 3]
assert together.drop(index=4).reset_index(drop=True).equals(clouds)
# the same count as the rule of make_pair, which keeps 10 % and skips 11 %
for pixels, skipped in ((10, 0), (11, 1)):
    path = saved(small_pair("m5", day, mask=mask_of(c7=pixels)), TMP / "clouds" / str(pixels))
    assert Z.cloud_report([path], [7])["images_over"].tolist() == [1, skipped, skipped, 0]
without = Z.cloud_report(cloud_paths[3:], [1])
assert without.to_dict("list") == {"what": ["no cloud mask"], "pixels": [0], "pixel_share": [0.0],
                                   "images": [1], "images_over": [0], "image_share": [0.0]}
assert Z.cloud_report([])["images"].tolist() == [0]
raises(ValueError, Z.cloud_report, cloud_paths, [], says="whole numbers from 0 to 255")
raises(ValueError, Z.cloud_report, cloud_paths, [300], says="whole numbers from 0 to 255")
real_clouds = Z.cloud_report(paths, [3])                  # the pairs fetched end to end
assert real_clouds["what"].tolist() == ["code 0", "codes 3 together", "no cloud mask"]
assert real_clouds["pixels"].tolist() == [2 * 384 * 384, 0, 0]

# ---- check 1, the Sentinel-2 offset ---------------------------------------------------------
holes = shifted()
holes[:24] = 0                                            # 30 % no-data
few_low, many_low = shifted(), shifted()
few_low.flat[:3] = 400                                    # 0.05 % of the pixels
many_low.flat[:20] = 400                                  # 0.31 % of the pixels
full = lambda v: np.full((80, 80), v, np.uint16)          # noqa: E731
NAME = "S2B_MSIL2A_20230301T101839_N0509_R065_T32TQM_20230301T140000"
images = [
    small_pair("a1", ts("2019-05-01 10:00"), plain()),
    small_pair("a2", ts("2021-05-01 10:00"), plain()),
    small_pair("a3", ts("2022-01-24 23:59:59"), plain()),              # last second before
    small_pair("b1", ts("2022-01-25 00:00:00"), shifted()),            # first second from
    small_pair("b2", ts("2023-03-01 10:00"), holes, shifted(), shifted(), product=NAME),
    small_pair("b3", ts("2023-05-01 10:00"), full(950)),               # exactly at the limit
    small_pair("b4", ts("2024-05-01 10:00"), full(950), full(950), full(949)),
    small_pair("b5", ts("2024-06-01 10:00"), few_low, shifted(), shifted()),
    small_pair("b6", ts("2024-07-01 10:00"), many_low, shifted(), shifted()),
    small_pair("b7", ts("2024-08-01 10:00"), shifted(), full(0), shifted()),
    small_pair("b8", pd.Timestamp("2022-01-25 00:00:00"), shifted()),  # no time zone
]
offset_paths = [saved(p, TMP / "offset") for p in images]
check = Z.offset_check(iter(offset_paths))
assert list(check["grid_cell"]) == [p.grid_cell for p in images] and len(check) == 11
by_cell = check.set_index("grid_cell")
assert list(check["period"]) == ["before"] * 3 + ["from"] * 8
assert list(check["year"]) == [2019, 2021, 2022, 2022, 2023, 2023, 2024, 2024, 2024, 2024, 2022]
assert all(t.tzinfo is not None for t in check["s2_datetime"])
assert by_cell.loc["b8", "s2_datetime"] == ts("2022-01-25 00:00:00")
carries = dict(zip(check["grid_cell"], check["carries_offset"]))
assert carries == {"a1": False, "a2": False, "a3": False, "b1": True,
                   "b2": True,       # zeros are no-data and do not count
                   "b3": True,       # 950 is at least 950
                   "b4": False,      # one band at 949 is enough to fail
                   "b5": True,       # 3 low pixels do not move the 0.1 percentile
                   "b6": False,      # 20 low pixels do
                   "b7": False,      # B04 has no data
                   "b8": True}
assert by_cell.loc["b2", "baseline"] == "05.09" and by_cell.loc["b2", "s2_product_id"] == NAME
assert pd.isna(by_cell.loc["a1", "baseline"]) and pd.isna(by_cell.loc["a1", "s2_product_id"])
assert Z._baseline("S2A_MSIL2A_20190501T101031_N0211_R022_T32TQM_x") == "02.11"
assert Z._baseline("no baseline here") is None and Z._baseline(None) is None

for cell, band, values in (("b2", "B02", holes), ("b5", "B02", few_low), ("b6", "B02", many_low),
                           ("a1", "B08", images[0].s2["B08"])):
    data = values[values != 0]
    for column, want in ((f"{band}_p0.1", percentile(data, 0.1)),
                         (f"{band}_p1", percentile(data, 1.0)),
                         (f"{band}_median", percentile(data, 50.0)),
                         (f"{band}_mean", data.astype(np.float64).mean())):
        assert near(by_cell.loc[cell, column], want), (cell, column)
assert by_cell.loc["b2", "B02_p0.1"] >= 1000 and by_cell.loc["b2", "nonzero_px"] == 56 * 80
assert by_cell.loc["b5", "B02_p0.1"] >= 1000 and by_cell.loc["b6", "B02_p0.1"] == 400
assert by_cell.loc["b3", "B08_p0.1"] == 950 and by_cell.loc["b4", "B08_p0.1"] == 949
assert by_cell.loc["b7", "nonzero_px"] == 0 and by_cell.loc["a1", "nonzero_px"] == 6400
assert check.loc[check["grid_cell"] == "b7", [f"B04_{s}" for s in
                                              ("p0.1", "p1", "median", "mean")]].isna().all().all()
assert not check.drop(index=9)[[c for c in check.columns if c[:3] in Z.OFFSET_BANDS]
                               ].isna().any().any()
assert len(Z.offset_check([])) == 0 and list(Z.offset_check([]).columns) == list(check.columns)

years = Z.offset_by_year(check)
assert list(years["year"]) == [2019, 2021, 2022, 2023, 2024]
assert list(years["images"]) == [1, 1, 3, 2, 3], "the image without data was counted"
assert list(years["with_offset"]) == [0, 0, 2, 2, 1]
assert list(years["share_with_offset"]) == [0.0, 0.0, 2 / 3, 1.0, 1 / 3]
y2024 = by_cell.loc[["b4", "b5", "b6"]]
row2024 = years.set_index("year").loc[2024]
assert row2024["B02_p0.1_min"] == 400 == y2024["B02_p0.1"].min()
assert row2024["B08_p0.1_min"] == 949
assert near(row2024["B02_p0.1_median"], y2024["B02_p0.1"].median())
assert near(row2024["B04_p1_median"], y2024["B04_p1"].median())
assert near(row2024["B04_mean"], y2024["B04_mean"].mean())
assert years.set_index("year").loc[2019, "B02_p0.1_min"] < 100

# per processing baseline, read from the product id. Only b2 has one, b7 has no data
by_baseline = Z.offset_by_baseline_table(check)
assert list(by_baseline.columns) == ["baseline", "images", "with_offset", "share_with_offset"]
assert by_baseline.to_dict("list") == {"baseline": ["N0509", "no baseline"], "images": [1, 9],
                                       "with_offset": [1, 4], "share_with_offset": [1.0, 4 / 9]}


def id_of(baseline: str) -> str:
    return f"S2A_MSIL2A_20230301T101839_{baseline}_R065_T32TQM_20230301T140000"


made = pd.DataFrame({
    "s2_product_id": [id_of("N0400"), id_of("N0201"), id_of("N9999"), id_of("N0400"), None,
                      id_of("N9999"), id_of("N0510"), id_of("N0400"), "no baseline here"],
    "nonzero_px": [100, 100, 100, 100, 100, 0, 100, 100, 100],
    "carries_offset": [True, False, True, True, False, True, True, False, True]})
got = Z.offset_by_baseline_table(made.sample(frac=1.0, random_state=0))   # the order is free
assert got.to_dict("list") == {
    "baseline": ["N0201", "N0400", "N0510", "N9999", "no baseline"], "images": [1, 3, 1, 1, 2],
    "with_offset": [0, 2, 1, 1, 1], "share_with_offset": [0.0, 2 / 3, 1.0, 1.0, 0.5]}, got
assert got["images"].dtype.kind == "i" and got["with_offset"].dtype.kind == "i"
empty_table = Z.offset_by_baseline_table(check.iloc[:0])
assert len(empty_table) == 0 and list(empty_table.columns) == list(by_baseline.columns)
raises(KeyError, Z.offset_by_baseline_table, check.drop(columns="s2_product_id"))

verdict = Z.offset_verdict(check)
assert verdict["verdict"] == "unclear, the user decides" and verdict["why"]
assert (verdict["images_before"], verdict["with_offset_before"]) == (3, 0)
assert (verdict["images_from"], verdict["with_offset_from"]) == (7, 5)
assert verdict["share_before"] == 0.0 and verdict["share_from"] == 5 / 7
assert verdict["images_without_data"] == 1 and verdict["baseline_date"] == "2022-01-25"
assert verdict["source"] == Z.OFFSET_SOURCE and "950" in verdict["rule"]
json.dumps(verdict)
only = lambda cells: check[check["grid_cell"].isin(cells)]          # noqa: E731
assert Z.offset_verdict(only(["a1", "a2", "a3", "b1", "b2", "b3", "b5", "b8"]))["verdict"] \
    == "offset present from 2022-01-25"
assert Z.offset_verdict(only(["a1", "a2", "a3", "b4", "b6"]))["verdict"] == "no offset"


def shares(before_with: int, before_n: int, from_with: int, from_n: int,
           no_data: int = 0) -> dict:
    """The verdict for a table with these counts of images."""
    rows = ([("before", i < before_with, 100) for i in range(before_n)]
            + [("from", i < from_with, 100) for i in range(from_n)]
            + [("from", False, 0)] * no_data)
    table = pd.DataFrame(rows, columns=["period", "carries_offset", "nonzero_px"])
    return Z.offset_verdict(table.sample(frac=1.0, random_state=0))


PRESENT, NO, UNCLEAR = "offset present from 2022-01-25", "no offset", "unclear, the user decides"
for counts, want in (((0, 10, 10, 10), PRESENT),
                     ((0, 10, 9, 10), PRESENT),          # exactly 90 % from the date on
                     ((1, 10, 9, 10), PRESENT),          # exactly 10 % before
                     ((0, 10, 89, 100), UNCLEAR),        # 89 %
                     ((11, 100, 10, 10), UNCLEAR),       # 11 % before
                     ((0, 10, 0, 10), NO),
                     ((1, 10, 1, 10), NO),               # exactly 10 % in both
                     ((0, 10, 11, 100), UNCLEAR),        # 11 % from the date on
                     ((11, 100, 0, 10), UNCLEAR),
                     ((10, 10, 10, 10), UNCLEAR),        # all images, old ones too
                     ((10, 10, 0, 10), UNCLEAR),
                     ((0, 10, 5, 10), UNCLEAR)):
    got = shares(*counts)
    assert got["verdict"] == want, (counts, got["verdict"])
    assert bool(got["why"]) == (want == UNCLEAR)
empty_before, empty_from = shares(0, 0, 10, 10), shares(0, 10, 0, 0)
assert empty_before["verdict"] == UNCLEAR and empty_before["why"] == "no image before the date"
assert empty_before["share_before"] is None and empty_before["share_from"] == 1.0
assert empty_from["verdict"] == UNCLEAR and empty_from["why"] == "no image from the date on"
assert shares(0, 0, 0, 0)["why"] == "no image before the date and no image from the date on"
with_empty = shares(0, 10, 9, 10, no_data=5)
assert with_empty["verdict"] == PRESENT and with_empty["images_without_data"] == 5
assert with_empty["images_from"] == 10


# ---- check 2, Sentinel-1 statistics -----------------------------------------------------------
def two_values(mean: float, sd: float, shape=(64, 64)) -> np.ndarray:
    """Half the pixels at mean - sd, half at mean + sd, so both numbers are exact."""
    out = np.full(shape, mean - sd)
    out.flat[::2] = mean + sd
    return out


def s1_of(name: str, *images) -> dict:
    """`s1_stats` of pairs with these (2, h, w) arrays, VV first."""
    folder = TMP / "s1" / name
    files = [saved(small_pair(f"c{i}", ts("2021-06-10"), s1_db=np.stack(image)), folder)
             for i, image in enumerate(images)]
    return Z.s1_stats(iter(files))


# numbers that float16 holds exactly, close to TerraMind's
good = s1_of("good", (two_values(-10.9375, 4.375), two_values(-17.328125, 4.453125)),
             (two_values(-10.9375, 4.375), two_values(-17.328125, 4.453125)))
assert good["pairs"] == 2 and good["pixels"] == {"VV": 8192, "VH": 8192}
assert good["not_finite"] == {"VV": 0, "VH": 0}
assert good["mean_db"] == {"VV": -10.9375, "VH": -17.328125}
assert good["sd_db"] == {"VV": 4.375, "VH": 4.453125}
assert good["terramind_mean_db"] == {"VV": -10.93, "VH": -17.329}
assert good["terramind_sd_db"] == {"VV": 4.391, "VH": 4.459}
assert near(good["mean_diff_db"]["VV"], -0.0075) and near(good["mean_diff_db"]["VH"], 0.000875)
assert near(good["sd_ratio"]["VV"], 4.375 / 4.391) and near(good["sd_ratio"]["VH"],
                                                           4.453125 / 4.459)
assert good["flag"] == "none" and good["flagged"] == []
assert good["source"] == Z.TERRAMIND_SOURCE and "3 dB" in good["rule"]
json.dumps(good)

# every pixel has the same weight, a large image counts more than a small one
flat8, flat16 = np.full((64, 64), -8.0), np.full((32, 32), -16.0)
pooled = s1_of("pooled", (flat8, flat8 - 6.5), (flat16, flat16 - 6.5))
assert pooled["pixels"] == {"VV": 5120, "VH": 5120}
assert near(pooled["mean_db"]["VV"], -9.6) and near(pooled["mean_db"]["VH"], -16.1)
assert near(pooled["sd_db"]["VV"], 3.2) and near(pooled["sd_db"]["VH"], 3.2)
assert pooled["flag"] == "none"

# pixels that are not finite are counted and left out
broken = two_values(-10.9375, 4.375)
broken[0, 0], broken[0, 1] = np.nan, -np.inf              # one of each of the two values
holes_s1 = s1_of("holes", (broken, two_values(-17.328125, 4.453125)))
assert holes_s1["not_finite"] == {"VV": 2, "VH": 0}
assert holes_s1["pixels"] == {"VV": 4094, "VH": 4096}
assert holes_s1["mean_db"]["VV"] == -10.9375 and holes_s1["flag"] == "none"

# unit errors
linear = s1_of("linear", (np.full((8, 8), 0.08), np.full((8, 8), 0.02)))
assert linear["flag"] == "unit error suspected" and len(linear["flagged"]) == 4
assert near(linear["mean_diff_db"]["VV"], 0.08 - (-10.93), 1e-3)
swapped = s1_of("swapped", (two_values(-17.328125, 4.453125), two_values(-10.9375, 4.375)))
assert swapped["flag"] == "unit error suspected"
assert swapped["flagged"] == ["VV mean differs by -6.40 dB", "VH mean differs by +6.39 dB"]
halved = s1_of("halved", (two_values(-5.46875, 2.1875), two_values(-8.65625, 2.21875)))
assert halved["flag"] == "unit error suspected" and len(halved["flagged"]) == 4

MEAN, SD = Z.TERRAMIND_S1RTC["mean_db"], Z.TERRAMIND_S1RTC["sd_db"]
assert Z.s1_flags(MEAN, SD) == []
assert Z.s1_flags((MEAN[0] + 2.99, MEAN[1] - 2.99), SD) == []
assert Z.s1_flags((MEAN[0] + 3.01, MEAN[1]), SD) == ["VV mean differs by +3.01 dB"]
assert Z.s1_flags((MEAN[0], MEAN[1] - 3.01), SD) == ["VH mean differs by -3.01 dB"]
assert Z.s1_flags(MEAN, (SD[0] * 0.5, SD[1] * 2.0)) == []          # the limits are inside
assert Z.s1_flags(MEAN, (SD[0] * 0.49, SD[1])) == ["VV SD ratio is 0.49"]
assert Z.s1_flags(MEAN, (SD[0], SD[1] * 2.01)) == ["VH SD ratio is 2.01"]
assert len(Z.s1_flags((NAN, MEAN[1]), (SD[0], NAN))) == 2         # nan is never fine
raises(ValueError, Z.s1_stats, [], says="no pair")
bad_shape = saved(small_pair("c9", ts("2021-06-10"), s1_db=np.zeros((3, 4, 4))), TMP / "s1" / "bad")
raises(ValueError, Z.s1_stats, [bad_shape], says="expected (2, h, w)")

# ---- check 3, the date gap --------------------------------------------------------------------
GAPS = [0.0, 0.5, 1.0, 1.0001, 2.5, 6.99, 7.0, 7.0001, 13.0, 29.0, 89.0, 364.0, 400.0, NAN]
s2_time = ts("2021-12-30 12:00")                          # 1.5 days before the new year
gap_table = pd.DataFrame({
    "grid_cell": [f"g{i}" for i in range(len(GAPS))], "gap_days": GAPS, "s2_datetime": s2_time,
    "s1_datetime": [pd.NaT if math.isnan(g) else s2_time + pd.Timedelta(days=g) for g in GAPS]})
gaps = Z.gap_report(gap_table.sample(frac=1.0, random_state=1))
assert list(gaps.columns) == ["kind", "what", "pairs", "share"]
counts = dict(zip(gaps["what"], gaps["pairs"]))
assert counts == {"0 to 1 days": 3, "1 to 2 days": 1, "2 to 3 days": 1, "3 to 5 days": 0,
                  "5 to 7 days": 2,                      # 7 days and 0 seconds is inside
                  "7 to 14 days": 2, "14 to 30 days": 1, "30 to 90 days": 1,
                  "90 to 365 days": 1, "more than 365 days": 1,
                  "within 7 days": 7, "more than 7 days": 6, "no date": 1,
                  "different years": 9, "all pairs": 14}, counts
assert list(gaps["kind"]) == ["bin"] * 10 + ["total"] * 5
bins = gaps[gaps["kind"] == "bin"]
assert bins["pairs"].sum() + counts["no date"] == counts["all pairs"]
assert bins["pairs"].iloc[:5].sum() == counts["within 7 days"]
share = dict(zip(gaps["what"], gaps["share"]))
assert share["within 7 days"] == 0.5 and share["all pairs"] == 1.0
assert near(gaps["share"].iloc[:10].sum() + share["no date"], 1.0)
assert not any(":" in what for what in gaps["what"])
no_pairs = Z.gap_report(gap_table.iloc[:0])
assert (no_pairs["pairs"] == 0).all() and no_pairs["share"].isna().all()
raises(KeyError, Z.gap_report, gap_table.drop(columns="gap_days"))
raises(KeyError, Z.gap_report, gap_table.drop(columns="s1_datetime"))

# ---- the clock --------------------------------------------------------------------------------
EVENTS: list = []
cuda = (torch.cuda.synchronize, torch.cuda.reset_peak_memory_stats,
        torch.cuda.max_memory_allocated)
torch.cuda.synchronize = lambda device=None: EVENTS.append(("sync", str(device)))
torch.cuda.reset_peak_memory_stats = lambda device=None: EVENTS.append(("reset", str(device)))
torch.cuda.max_memory_allocated = lambda device=None: 3_500_000_000
try:
    Z._sync("cpu")
    Z._peak_reset("cpu")
    assert EVENTS == [] and math.isnan(Z._peak_gb("cpu")) and not Z._on_gpu("cpu")
    assert Z._on_gpu("cuda") and Z._on_gpu("cuda:1") and Z._on_gpu(torch.device("cuda"))
    seconds, peak = Z._time_calls(lambda: EVENTS.append("call"), 3, "cuda:0")
    # warm-up, then the peak is reset, then the clock runs between two synchronize calls
    assert EVENTS == ["call"] * Z.WARMUP + [("reset", "cuda:0"), ("sync", "cuda:0")] \
        + ["call"] * 3 + [("sync", "cuda:0")], EVENTS
    assert peak == 3.5 and 0 <= seconds < 1
finally:
    (torch.cuda.synchronize, torch.cuda.reset_peak_memory_stats,
     torch.cuda.max_memory_allocated) = cuda

slow = iter([0.3] * Z.WARMUP + [0.04] * 3)
seconds, peak = Z._time_calls(lambda: time.sleep(next(slow)), 3, "cpu")
assert 0.04 <= seconds < 0.15, f"{seconds} s per call, the warm-up was counted or the sum given"
assert math.isnan(peak)

REAL_SYNC = Z._sync
Z._sync = lambda device: EVENTS.append("sync")


# ---- time_teacher -----------------------------------------------------------------------------
class Toy(Teacher):
    """4 x 4 grid of 8 features that follow the input. Slow in the warm-up calls."""

    name, dim, patch, input_size = "toy", 8, 16, 64
    bands, rules = ("B04", "B03", "B02"), ("unit", "half")
    rule_sources = {"unit": "made for the test", "half": "made for the test"}
    licence = "made for the test"
    largest = 10 ** 9                  # a batch above this does not fit

    def __init__(self, rule: str = "unit"):
        torch.manual_seed(0)
        super().__init__(torch.nn.Linear(3, 8), rule, device="cpu", dtype=torch.float32)
        self.inputs: list[tuple] = []

    def prepare(self, s2):
        self.inputs.append((tuple(s2.shape), s2.dtype, float(s2.min()), float(s2.max())))
        return s2[:, band_index(self.bands)] / (10000.0 if self.rule == "unit" else 5000.0)

    def _forward(self, x):
        b = x.shape[0]
        if b > self.largest:
            raise Z._OOM("the toy is full")
        EVENTS.append(("call", b))
        seen = sum(1 for shape, *_ in self.inputs if shape[0] == b)
        time.sleep(0.3 if seen <= Z.WARMUP else 0.02)
        spatial = self.model(torch.nn.functional.adaptive_avg_pool2d(x, 4).permute(0, 2, 3, 1))
        return TeacherOut(spatial.mean((1, 2)), spatial)


EVENTS.clear()
toy = Toy("half")
t_times = Z.time_teacher(toy, batch_sizes=(2, 4), n_batches=3, size=64)
assert EVENTS == [e for b in (2, 4) for e in
                  [("call", b)] * Z.WARMUP + ["sync"] + [("call", b)] * 3 + ["sync"]], EVENTS
assert list(t_times["batch_size"]) == [2, 4] and len(t_times) == 2
for row in t_times.to_dict("records"):
    assert (row["teacher"], row["rule"], row["device"], row["dtype"]) == ("toy", "half", "cpu",
                                                                         "float32")
    assert (row["input_size"], row["grid"], row["dim"]) == (64, 4, 8)
    assert row["params_m"] == 32 / 1e6 and (row["n_batches"], row["warmup"]) == (3, Z.WARMUP)
    assert 0.02 <= row["seconds_per_batch"] < 0.15, "the warm-up was counted"
    assert near(row["images_per_s"], row["batch_size"] / row["seconds_per_batch"])
    assert math.isnan(row["peak_gpu_gb"]) and row["note"] == ""
assert [i[0] for i in toy.inputs] == [(2, 12, 64, 64)] * 5 + [(4, 12, 64, 64)] * 5
assert all(i[1] == torch.float32 and 0 <= i[2] < 100 and 9900 < i[3] <= 10000
           for i in toy.inputs), "the input is not like digital numbers"

toy = Toy()
toy.largest = 4
full_gpu = Z.time_teacher(toy, batch_sizes=(8, 2), n_batches=1, size=64)
assert list(full_gpu["note"]) == ["out of memory", ""]
assert full_gpu[["seconds_per_batch", "images_per_s"]].isna().iloc[0].all()
assert full_gpu["images_per_s"].iloc[1] > 0
raises(ValueError, Z.time_teacher, toy, batch_sizes=(2,), n_batches=0, says="n_batches")
raises(ValueError, Z.time_teacher, toy, batch_sizes=(2, 0), says="batch sizes")
raises(ValueError, Z.time_teacher, toy, batch_sizes=(), says="batch sizes")


# ---- time_student -----------------------------------------------------------------------------
class ToyStudent(torch.nn.Module):
    """2 x 2 grid of 6 features, one input layer per sensor. Slow in the warm-up steps."""

    img = 32

    def __init__(self, group: int):
        super().__init__()
        self.e1 = torch.nn.Conv2d(2, 6, 16, stride=16)
        self.e2 = torch.nn.Conv2d(12, 6, 16, stride=16)
        self.group = group             # calls per batch size and mode
        self.seen: list[tuple] = []

    def forward(self, s1=None, s2=None):
        EVENTS.append(("call", self.training, torch.is_grad_enabled()))
        self.seen.append(tuple(None if x is None else tuple(x.shape) for x in (s1, s2)))
        time.sleep(0.3 if (len(self.seen) - 1) % self.group < Z.WARMUP else 0.02)
        x = sum(layer(x) for layer, x in ((self.e1, s1), (self.e2, s2)) if x is not None)
        spatial = x.permute(0, 2, 3, 1)
        return spatial.mean((1, 2)), spatial


torch.manual_seed(0)
like = {"a": TeacherOut(torch.zeros(1, 8), torch.zeros(1, 4, 4, 8)),      # grid above ours
        "b": TeacherOut(torch.zeros(3, 5), torch.zeros(3, 1, 1, 5))}      # grid below ours
student, heads = ToyStudent(Z.WARMUP + 2).eval(), Heads({"a": 8, "b": 5}, student_dim=6).eval()
start = {k: v.clone() for module in (student, heads) for k, v in module.state_dict().items()}
EVENTS.clear()
s_times = Z.time_student(student, heads, like, batch_sizes=(2, 3), n_batches=2, device="cpu")
assert MAJORTOM_MODES == ("s1", "s2", "both")
assert list(s_times["mode"]) == list(MAJORTOM_MODES) * 2
assert list(s_times["batch_size"]) == [2] * 3 + [3] * 3
step = ("call", True, True)                               # training mode, with gradients
assert EVENTS == ([step] * Z.WARMUP + ["sync"] + [step] * 2 + ["sync"]) * 6, EVENTS
want_seen = []
for b in (2, 3):
    s1_shape, s2_shape = (b, 2, 32, 32), (b, 12, 32, 32)
    for shapes in ((s1_shape, None), (None, s2_shape), (s1_shape, s2_shape)):
        want_seen += [shapes] * (Z.WARMUP + 2)
assert student.seen == want_seen
n_params = sum(p.numel() for module in (student, heads) for p in module.parameters())
for row in s_times.to_dict("records"):
    assert (row["device"], row["img"], row["grid"], row["teachers"]) == ("cpu", 32, 2, "a b")
    assert row["params_m"] == n_params / 1e6 and (row["n_batches"], row["warmup"]) == (2, 2)
    assert 0.02 <= row["seconds_per_step"] < 0.2, "the warm-up was counted"
    assert near(row["images_per_s"], row["batch_size"] / row["seconds_per_step"])
    assert math.isnan(row["peak_gpu_gb"]) and row["note"] == ""
after = {k: v for module in (student, heads) for k, v in module.state_dict().items()}
still = [k for k in start if torch.equal(start[k], after[k])]
assert not still, f"AdamW did not change {still}"
assert student.training and heads.training

raises(ValueError, Z.time_student, student, heads, {}, batch_sizes=(2,), device="cpu",
       says="no teacher output")
raises(ValueError, Z.time_student, student, heads, like, batch_sizes=(2,), n_batches=0,
       device="cpu", says="n_batches")
raises(ValueError, Z.time_student, student, heads, like, batch_sizes=(0,), device="cpu",
       says="batch sizes")
raises(ValueError, Z.time_student, student, heads, {"a": like["a"]}, batch_sizes=(2,),
       n_batches=1, device="cpu")                         # heads and targets do not match

Z._sync = REAL_SYNC

# the real small student and heads of the student module, one step per mode
real = tiny_student(img=64)
real_heads = Heads({"x": 16, "y": 24}, student_dim=real.dim)
real_like = {"x": TeacherOut(torch.zeros(1, 16), torch.zeros(1, 4, 4, 16)),
             "y": TeacherOut(torch.zeros(1, 24), torch.zeros(1, 7, 7, 24))}
real_times = Z.time_student(real, real_heads, real_like, batch_sizes=(2,), n_batches=1,
                            device="cpu")
assert list(real_times["mode"]) == list(MAJORTOM_MODES) and (real_times["images_per_s"] > 0).all()
assert (real_times["grid"] == 4).all() and (real_times["img"] == 64).all()
assert near(real_times["params_m"].iloc[0], (real.n_params() + real_heads.n_params()) / 1e6)
del real, real_heads

# ---- pca_rgb ----------------------------------------------------------------------------------
rng = np.random.default_rng(1)
B, G, D = 3, 6, 10
N = B * G * G
axes = np.linalg.qr(rng.normal(size=(D, D)))[0][:, :3]                  # (D, 3), orthonormal
parts = rng.normal(size=(N, 3))
parts = np.linalg.qr(parts - parts.mean(0))[0] * np.array([5.0, 2.0, 1.0])    # no correlation
feats = (parts @ axes.T + 7.0).reshape(B, G, G, D)
rgb = Z.pca_rgb(feats)
assert rgb.shape == (B, G, G, 3) and rgb.dtype == np.uint8
flat = rgb.reshape(N, 3).astype(int)
for k in range(3):                     # the component with the most variance is red, and so on
    sign = 1.0 if axes[np.abs(axes[:, k]).argmax(), k] > 0 else -1.0
    part = sign * parts[:, k]
    lo, hi = percentile(part, 1.0), percentile(part, 99.0)
    want = np.round(np.clip((part - lo) / (hi - lo), 0.0, 1.0) * 255.0)
    assert np.abs(flat[:, k] - want).max() <= 1, f"colour {k}"
    assert flat[:, k].min() == 0 and flat[:, k].max() == 255
    # 1 % of 108 patches lies outside at each end, a neighbour may round to the same value
    assert 2 <= (flat[:, k] == 0).sum() <= 4 and 2 <= (flat[:, k] == 255).sum() <= 4


def same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and int(np.abs(a.astype(int) - b.astype(int)).max()) <= 1


assert same(Z.pca_rgb(feats * 3.0 + 100.0), rgb)          # centred, and the scale is free
assert same(Z.pca_rgb(torch.from_numpy(feats).float()), rgb)
assert same(Z.pca_rgb(torch.from_numpy(feats).float().requires_grad_()), rgb)
assert same(Z.pca_rgb(-feats), 255 - rgb)                 # the sign follows the axis rule
assert not same(Z.pca_rgb(feats[:1]), rgb[:1])            # one PCA for the whole batch
assert np.array_equal(Z.pca_rgb(feats), rgb)
assert not Z.pca_rgb(np.full((2, 3, 3, 5), 4.0)).any()    # no range, no division by zero
raises(ValueError, Z.pca_rgb, feats[0], says="expected (B, g, g, dim)")
raises(ValueError, Z.pca_rgb, feats[..., :2], says="at least 3 features")
raises(ValueError, Z.pca_rgb, feats[:1, :1, :2], says="at least 3 features and 3 patches")
with_nan = feats.copy()
with_nan[0, 0, 0, 0] = np.nan
raises(ValueError, Z.pca_rgb, with_nan, says="not finite")

# ---- feature_pictures ---------------------------------------------------------------------------
from PIL import Image  # noqa: E402

assert "matplotlib" not in sys.modules
ramp = torch.linspace(0.0, 1.0, 64)
s2_batch = torch.zeros(2, 12, 64, 64)
s2_batch[0] = 500.0 + 3000.0 * ramp[None, None, :] + 200.0 * torch.rand(12, 64, 64)
s2_batch[1] = 500.0 + 3000.0 * ramp[None, :, None] + 200.0 * torch.rand(12, 64, 64)
s2_batch[:, 3] += 2000.0                                  # B04, red
s2_batch[:, 1] -= 400.0                                   # B02, blue

preview = Z._preview_rgb(s2_batch)
assert preview.shape == (2, 64, 64, 3) and preview.dtype == np.uint8
red, green, blue = (preview[..., k].astype(float).mean() for k in range(3))
assert red > green > blue, "the preview is not B04, B03, B02"
assert preview.min() == 0 and preview.max() == 255
no_data = s2_batch.clone()
no_data[1] = 0.0
assert not Z._preview_rgb(no_data)[1].any() and Z._preview_rgb(no_data)[0].any()

built: list[tuple] = []


def build_toy(rule: str) -> Toy:
    gc.collect()
    alive = [name for name, ref in built if ref() is not None]
    assert not alive, f"the teacher of {alive} was not freed before {rule} was built"
    teacher = Toy(rule)
    built.append((rule, weakref.ref(teacher)))
    return teacher


png = TMP / "pictures" / "deep" / "toy.png"
got = Z.feature_pictures("toy", build_toy, s2_batch, ("unit", "half"), png)
assert got == png and png.is_file()
assert [rule for rule, _ in built] == ["unit", "half"], "one build per rule, in their order"
gc.collect()
assert all(ref() is None for _, ref in built), "a teacher was not freed"
with Image.open(png) as picture:
    # 3 columns and 2 rows of 2.2 inches, 0.6 inches for the title, 150 dots per inch
    assert abs(picture.size[0] - 990) <= 1 and abs(picture.size[1] - 750) <= 1, picture.size
    colours = np.unique(np.asarray(picture.convert("RGB")).reshape(-1, 3), axis=0)
    assert len(colours) > 100, "the picture is empty"

columns = [("RGB preview B04 B03 B02", preview), ("unit", rgb[:2]), ("half: b", rgb[:2])]
figure = Z._draw_grid("toy: patch features", columns)
assert len(figure.axes) == 6                              # rows = images, one column per entry
titles = [ax.get_title() for ax in figure.axes]
assert titles == ["RGB preview B04 B03 B02", "unit", "half - b", "", "", ""]
assert figure.get_suptitle() == "toy - patch features"
assert [len(ax.images) for ax in figure.axes] == [1] * 6
assert np.array_equal(figure.axes[3].images[0].get_array(), preview[1])
assert np.array_equal(figure.axes[1].images[0].get_array(), rgb[0])

raises(ValueError, Z.feature_pictures, "toy", build_toy, s2_batch, (), png, says="no rule")
raises(ValueError, Z.feature_pictures, "toy", build_toy, s2_batch[:, :3], ("unit",), png,
       says="expected (B, 12, H, W)")
raises(ValueError, Z.feature_pictures, "toy", build_toy, s2_batch[:0], ("unit",), png)
assert len(built) == 2, "a teacher was built for a bad input"
other = TMP / "pictures" / "not written.png"
raises(ValueError, Z.feature_pictures, "toy", build_toy, s2_batch, ("unit", "no such rule"),
       other, says="unknown rule")
gc.collect()
assert len(built) == 3 and built[2][1]() is None, "not freed after an error"
assert not other.exists()

# ---- two teachers of the plan, small and with random weights ---------------------------------
# The registry holds the six teachers (part four). Timed and drawn here are
# the two Sentinel-2 readers that were in this repository before part four's new modules;
# time_teacher and feature_pictures feed Sentinel-2, which the radar teacher does not read.
from sartransfer.distil.teachers import TEACHERS, build_teacher  # noqa: E402

assert sorted(TEACHERS) == ["dinov3-l-sat", "pe-spatial-l", "siglip2-so400m-256", "siglip2-so400m-naflex",
                            "terrafm-b", "terrafm-b-radar"]
S2_TEACHERS = ("dinov3-l-sat", "terrafm-b")
SMALL = dict(device="cpu", dtype=torch.float32, tiny=True)
tiny_times, tiny_outs = [], {}
for name in S2_TEACHERS:
    teacher = build_teacher(name, **SMALL)
    table = Z.time_teacher(teacher, batch_sizes=(2,), n_batches=1, size=64)
    row = table.iloc[0].to_dict()
    assert (row["teacher"], row["rule"], row["dtype"]) == (name, teacher.rule, "float32")
    assert (row["input_size"], row["grid"], row["dim"]) == (teacher.input_size, teacher.grid,
                                                           teacher.dim)
    assert row["params_m"] == teacher.n_params() / 1e6 and row["images_per_s"] > 0
    tiny_times.append(table)
    tiny_outs[name] = teacher(s2_batch)
    assert tiny_outs[name].spatial.shape == (2, teacher.grid, teacher.grid, teacher.dim)
    rules = list(teacher.rules)
    assert len(rules) >= 2 and rules[0] != rules[1]
    seen: list[tuple] = []

    def build_small(rule: str, name: str = name) -> Teacher:
        gc.collect()
        assert all(ref() is None for _, ref in seen), f"{name} was not freed before {rule}"
        made = build_teacher(name, rule=rule, **SMALL)
        assert made.rule == rule
        seen.append((rule, weakref.ref(made)))
        return made

    del teacher
    png = Z.feature_pictures(name, build_small, s2_batch, rules, TMP / "pictures" / f"{name}.png")
    assert [rule for rule, _ in seen] == rules
    gc.collect()
    assert all(ref() is None for _, ref in seen), f"a teacher of {name} was not freed"
    with Image.open(png) as picture:                      # the preview and one column per rule
        assert abs(picture.size[0] - 330 * (1 + len(rules))) <= 1, (name, picture.size)
tiny_times = pd.concat(tiny_times, ignore_index=True)
small_student = tiny_student(img=64)
small_heads = Heads({name: out.summary.shape[1] for name, out in tiny_outs.items()},
                    student_dim=small_student.dim)
tiny_steps = Z.time_student(small_student, small_heads, tiny_outs, batch_sizes=(2,), n_batches=1,
                            device="cpu")
assert list(tiny_steps["mode"]) == list(MAJORTOM_MODES) and (tiny_steps["images_per_s"] > 0).all()
assert (tiny_steps["teachers"] == " ".join(S2_TEACHERS)).all() and (tiny_steps["grid"] == 4).all()
tiny_budget = Z.budget_table(tiny_times, tiny_steps, 1000, epochs=(2,))
# the grids are 16, 2 and 3 against the 4 of the student, the smaller one is stored
stored = sum((min(g, 4) ** 2 + 1) * d for g, d in zip(tiny_times["grid"], tiny_times["dim"]))
assert near(tiny_budget["features_gb"].iloc[1], stored * 2 * 1000 / 1e9)
del small_student, small_heads, tiny_outs


# ---- budget_table -------------------------------------------------------------------------------
def teacher_rows(*rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["teacher", "batch_size", "images_per_s", "grid", "dim",
                                       "device"])


def student_rows(*rows) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=["mode", "batch_size", "images_per_s"])
    return table.assign(grid=16, device="A100")


# three teacher rows with speeds, grids and widths made for the test (the names are the teachers', the numbers are not measured)
t_rows = teacher_rows(("dinov3-l-sat", 16, 400.0, 16, 1024, "A100"),
                      ("dinov3-l-sat", 64, 500.0, 16, 1024, "A100"),
                      ("terrafm-b", 16, 1000.0, 14, 768, "A100"),
                      ("terrafm-b", 64, 800.0, 14, 768, "A100"),
                      ("siglip2-so400m-256", 16, 250.0, 27, 1152, "A100"),
                      ("siglip2-so400m-256", 64, NAN, 27, 1152, "A100"))    # out of memory
s_rows = student_rows(("s1", 64, 600.0), ("s2", 64, 600.0), ("both", 64, 300.0),
                      ("s1", 128, 1000.0), ("s2", 128, 1000.0), ("both", 128, 500.0))
budget = Z.budget_table(t_rows, s_rows, 162_000)
assert list(budget["route"]) == ["live", "live", "stored", "stored"]
assert list(budget["epochs"]) == [20, 50, 20, 50] and (budget["n_cells"] == 162_000).all()
# T = 1/500 + 1/1000 + 1/250 = 0.007 s, S = (1/1000 + 1/1000 + 1/500) / 3 = 0.004 / 3 s
for column, want in (("teacher_hours", [6.3, 15.75, 0.315, 0.315]),
                     ("student_hours", [1.2, 3.0, 1.2, 3.0]),
                     ("hours", [7.5, 18.75, 1.515, 3.315]),
                     # (257 * 1024 + 197 * 768 + 257 * 1152) * 2 bytes * 162,000 cells
                     ("features_gb", [0.0, 0.0, 230.211072, 230.211072]),
                     ("teacher_images_per_s", [1 / 0.007] * 4),
                     ("student_images_per_s", [750.0] * 4)):
    assert all(near(a, b) for a, b in zip(budget[column], want)), (column, list(budget[column]))
assert (budget["student_batch_size"] == 128).all() and (budget["measured_on"] == "A100").all()
# without the summary vectors the storage is the 229 GB
summaries_gb = (1024 + 768 + 1152) * 2 * 162_000 / 1e9
assert near(budget["features_gb"].iloc[2] - summaries_gb, 229.257216)

one = Z.budget_table(t_rows, s_rows, 1000, epochs=(1,))
assert near(one["hours"].iloc[0], one["hours"].iloc[1])   # one epoch, the routes cost the same
assert near(one["hours"].iloc[0], 1000 * (0.007 + 0.004 / 3) / 3600)
# a student grid below a teacher grid stores the smaller one
small_grid = Z.budget_table(t_rows, s_rows.assign(grid=8), 1000, epochs=(1,))
assert near(small_grid["features_gb"].iloc[1], 65 * (1024 + 768 + 1152) * 2 * 1000 / 1e9)
# a batch size that did not fit in one mode is not used
partly = s_rows.copy()
partly.loc[5, "images_per_s"] = NAN
fallback = Z.budget_table(t_rows, partly, 162_000)
assert (fallback["student_batch_size"] == 64).all()
assert near(fallback["student_images_per_s"].iloc[0], 3 / (1 / 600 + 1 / 600 + 1 / 300))
mixed = Z.budget_table(t_rows.assign(device=["A100"] * 5 + ["cpu"]), s_rows.assign(device="T4"),
                       1000)
assert (mixed["measured_on"] == "A100, T4").all()

raises(ValueError, Z.budget_table, t_rows.iloc[[0, 2, 5]], s_rows, 1000, says="siglip2-so400m-256")
raises(ValueError, Z.budget_table, t_rows.iloc[:0], s_rows, 1000, says="no measured speed")
raises(ValueError, Z.budget_table, t_rows, s_rows.iloc[[0, 1, 3, 4]], 1000, says="all 3 modes")
raises(ValueError, Z.budget_table, t_rows, s_rows, 0, says="above 0")
raises(ValueError, Z.budget_table, t_rows, s_rows, 1000, epochs=(), says="above 0")
raises(ValueError, Z.budget_table, t_rows, s_rows, 1000, epochs=(20, 0), says="above 0")
grids = s_rows.copy()
grids["grid"] = [16, 16, 16, 16, 16, 8]
raises(ValueError, Z.budget_table, t_rows, grids, 1000, says="different grids")
# the tables of the two timing functions have the columns that the budget reads
for table in (Z.budget_table(t_times.assign(teacher=["p", "q"]), s_times, 1000),
              Z.budget_table(full_gpu, real_times, 1000)):
    assert (table["hours"] > 0).all() and (table["measured_on"] == "cpu").all()

# the data fetch. Without a fetch table the columns are there and empty
FETCH_COLUMNS = ["fetch_cells", "fetch_shards", "fetch_hours", "fetch_gb"]
assert list(Z.FETCH_BUDGET) == FETCH_COLUMNS and list(budget.columns[-4:]) == FETCH_COLUMNS
assert budget[FETCH_COLUMNS].isna().all().all()


def measured(**change) -> pd.DataFrame:
    """A fetch table that holds the totals of a run made for the test.

    100 cells read and 80 saved, 8 threads that took 400 s together in 50 s. 150 footers
    took 100 s and 300 MB, so one took 2/3 s and 2 MB. A cell took 3 s and 12 MB without.
    """
    table = pd.DataFrame(columns=list(Z.TIMING_COLUMNS))
    table.attrs = {"wall_s": 50.0, "thread_s": 400.0, "workers": 8, "cells_read": 100,
                   "cells_saved": 80, "footers": 150, "footer_s": 100.0, "mb_footers": 300.0,
                   "mb_rows_per_cell": 12.0, **change}
    return table


with_fetch = Z.budget_table(t_rows, s_rows, 1000, fetch=measured(), shards={"s2": 20, "s1": 30})
assert list(with_fetch.columns) == list(budget.columns) and len(with_fetch) == 4
unchanged = [c for c in budget.columns if c not in FETCH_COLUMNS]
assert with_fetch[unchanged].equals(Z.budget_table(t_rows, s_rows, 1000)[unchanged])
# 1,000 pairs need 1,250 cells. 1,250 cells of 3 s and 50 footers of 2/3 s are 3,783.3 s in the
# threads, 1/8 of it on the clock. 1,250 cells of 12 MB and 50 footers of 2 MB are 15.1 GB
for column, want in (("fetch_cells", 1250), ("fetch_shards", 50),
                     ("fetch_hours", (1250 * 3.0 + 50 * 100.0 / 150) / 8 / 3600),
                     ("fetch_gb", 15.1)):
    assert all(near(v, want) for v in with_fetch[column]), (column, list(with_fetch[column]))
# a footer is paid once per shard. Paid once per cell, as in the measured run, it would be
# 1,250 cells of 4 s and of 15 MB
assert with_fetch["fetch_hours"].iloc[0] < 0.8 * (1250 * 4.0) / 8 / 3600
assert with_fetch["fetch_gb"].iloc[0] < 0.85 * 1250 * 15.0 / 1000
more_cells = Z.budget_table(t_rows, s_rows, 2000, fetch=measured(), shards={"s2": 20, "s1": 30})
assert near(more_cells["fetch_gb"].iloc[0] - 15.1, 1250 * 12.0 / 1000), "footers grew with cells"
more_shards = Z.budget_table(t_rows, s_rows, 1000, fetch=measured(),
                             shards={"s2": 20, "s1": 130})
assert near(more_shards["fetch_gb"].iloc[0] - 15.1, 100 * 2.0 / 1000)
assert near(more_shards["fetch_hours"].iloc[0] - with_fetch["fetch_hours"].iloc[0],
            100 * (100.0 / 150) / 8 / 3600)
no_shards = Z.budget_table(t_rows, s_rows, 1000, fetch=measured(), shards={"s2": 0, "s1": 0})
assert near(no_shards["fetch_gb"].iloc[0], 15.0)
# without `shards` the upper limit is taken, all shards of a set or one per cell
full = Z.budget_table(t_rows, s_rows, 162_000, fetch=measured())
assert (full["fetch_cells"] == 202_500).all() and (full["fetch_shards"] == 498 + 2930).all()
assert near(full["fetch_gb"].iloc[0], (202_500 * 12.0 + 3428 * 2.0) / 1000)
few_cells = Z.budget_table(t_rows, s_rows, 100, fetch=measured())
assert (few_cells["fetch_cells"] == 125).all() and (few_cells["fetch_shards"] == 250).all()
exact = Z.budget_table(t_rows, s_rows, 3, fetch=measured(cells_read=7, cells_saved=2),
                       shards={"s2": 1, "s1": 1})
assert (exact["fetch_cells"] == 11).all()                 # 10.5 cells, rounded up
raises(ValueError, Z.budget_table, t_rows, s_rows, 1000, shards={"s2": 20, "s1": 30},
       says="without the fetch table")
raises(ValueError, Z.budget_table, t_rows, s_rows, 1000, fetch=measured(), shards={"s2": 20},
       says="a count for each")
raises(ValueError, Z.budget_table, t_rows, s_rows, 1000, fetch=measured(),
       shards={"s2": 20, "s1": -1}, says="a count for each")
raises(ValueError, Z.budget_table, t_rows, s_rows, 1000, fetch=measured(cells_saved=0),
       says="no cell that was read and saved")
raises(ValueError, Z.budget_table, t_rows, s_rows, 1000, fetch=measured(footers=0),
       says="no footer was fetched")
for name in ("wall_s", "thread_s", "cells_read", "footers", "footer_s", "mb_footers",
             "mb_rows_per_cell"):
    raises(ValueError, Z.budget_table, t_rows, s_rows, 1000, fetch=measured(**{name: None}),
           says=name)
raises(ValueError, Z.budget_table, t_rows, s_rows, 1000,
       fetch=pd.DataFrame(columns=list(Z.TIMING_COLUMNS)), says="as fetch_pairs returned it")

# the tables of fetch_pairs have what the budget reads, here the run on the local shards
totals = real_timing.attrs
real_budget = Z.budget_table(t_rows, s_rows, 1000, fetch=real_timing, shards={"s2": 3, "s1": 5})
assert (real_budget["fetch_cells"] == 3000).all()         # 2 of 6 cells were saved
thread = (3000 * (totals["thread_s"] - totals["footer_s"]) / 6
          + 8 * totals["footer_s"] / 4)
assert near(real_budget["fetch_hours"].iloc[0],
            thread * totals["wall_s"] / totals["thread_s"] / 3600)
assert near(real_budget["fetch_gb"].iloc[0],
            (3000 * want_rows / 6 + 8 * want_footers / 4) / 1e9)
raises(ValueError, Z.budget_table, t_rows, s_rows, 1000, fetch=no_footer_timing,
       says="no cell that was read and saved")            # the pairs were on disk

# ---- save_report --------------------------------------------------------------------------------
folder = TMP / "report" / "deep"
facts = {"when": ts("2022-01-25 00:00"), "count": np.int64(3), "share": np.float32(0.5),
         "missing": NAN, "far": math.inf, "values": np.arange(3), "folder": folder,
         "nested": {"pair": (1, 2.5), 7: None, "ok": np.bool_(True)}, "text": "späť"}
files = Z.save_report(folder, gaps=gaps, facts=facts, totals=fake_timing.attrs, offsets=check)
assert files == [folder / "gaps.csv", folder / "facts.json", folder / "totals.json",
                 folder / "offsets.csv"]
assert sorted(p.name for p in folder.iterdir()) == ["facts.json", "gaps.csv", "offsets.csv",
                                                    "totals.json"]
back = pd.read_csv(files[0])
assert list(back.columns) == list(gaps.columns) and len(back) == len(gaps)
assert list(back["what"]) == list(gaps["what"]) and list(back["pairs"]) == list(gaps["pairs"])
assert np.allclose(back["share"], gaps["share"])
offsets_back = pd.read_csv(files[3])
assert list(offsets_back.columns) == list(check.columns) and len(offsets_back) == 11
assert list(offsets_back["carries_offset"]) == list(check["carries_offset"])
# the stage A notebook prints the table from offset_images.csv of stage 0, read back like this
pd.testing.assert_frame_equal(Z.offset_by_baseline_table(offsets_back), by_baseline)
assert "\r" not in files[0].read_bytes().decode("utf-8")
read = json.loads(files[1].read_text(encoding="utf-8"))
assert read == {"when": "2022-01-25T00:00:00+00:00", "count": 3, "share": 0.5, "missing": None,
                "far": None, "values": [0, 1, 2], "folder": str(folder),
                "nested": {"pair": [1, 2.5], "7": None, "ok": True}, "text": "späť"}
assert "NaN" not in files[1].read_text(encoding="utf-8")
assert json.loads(files[2].read_text(encoding="utf-8"))["cells"] == 7

again = Z.save_report(folder, gaps=gaps.iloc[:2])          # the same name is replaced
assert again == [folder / "gaps.csv"] and len(pd.read_csv(again[0])) == 2
nothing = TMP / "report" / "nothing"
raises(TypeError, Z.save_report, nothing, gaps=gaps, wrong=[1, 2, 3], says="wrong")
raises(TypeError, Z.save_report, nothing, facts={"model": torch.nn.Linear(1, 1)}, gaps=gaps)
assert not nothing.exists(), "a part of a report was written"
assert Z.save_report(nothing) == [] and nothing.is_dir()

shutil.rmtree(TMP, ignore_errors=True)
print("test_stage0: all checks passed")
