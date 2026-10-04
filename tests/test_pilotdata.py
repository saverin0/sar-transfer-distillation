"""Pilot data of stage A, on synthetic shards through the real majortom, pairing and stage0.
No network. The synthetic shard helpers are those of test_majortom.py by name, as in test_stage0."""
import _setup  # noqa: F401

import inspect
import io
import json
import math
import os
import re
import shutil
import tarfile
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.transform import Affine

from sartransfer.distil import majortom as M
from sartransfer.distil import pairing as P
from sartransfer.distil import pilotdata as D
from sartransfer.distil import records as R
from sartransfer.distil import stage0 as Z

TMP = _setup.TMP_BASE / "pilotdata"
for _ in range(40):                    # Windows may take a moment to let go of a folder
    shutil.rmtree(TMP, ignore_errors=True)
    if not TMP.exists():
        break
    time.sleep(0.25)
TMP.mkdir(parents=True)


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


def quiet(fn, *args, **kwargs) -> tuple:
    """(result, what majortom counted, what was printed)."""
    before = dict(M.STATS)
    with redirect_stdout(io.StringIO()) as buf:
        out = fn(*args, **kwargs)
    return out, {name: M.STATS[name] - before[name] for name in M.STATS}, buf.getvalue()


# ---- the constants are the plan, titles have no colon ------------------------------------------
assert (D.N_PILOT, D.N_HELD_OUT, D.N_STATS, D.SEED, D.CHUNK) == (26_000, 2_000, 2_000, 0, 2_000)
# 26,000 CELLS are drawn for the 20,000 PAIRS. Stage 0 (2026-09-30) kept
# 393 of 500 cells under the rules of stage A, so 26,000 x 393 / 500 = 20,436 pairs, 13 chunks
assert D.N_PAIRS == 20_000 and D.N_PILOT == 13 * D.CHUNK
assert 20_000 <= round(D.N_PILOT * 393 / 500) < 21_000, "N_PILOT no longer gives about 20,000 pairs"
assert all(w in D.__doc__ for w in ("26,000", "20,000 PAIRS")), "the module docstring does not say why"
SOURCE_TEXT = Path(D.__file__).read_text(encoding="utf-8")
assert all(w in SOURCE_TEXT.split("N_PILOT = ")[0]
           for w in ("393 of 500", "78.6 %", "26,000 x 0.786 = 20,436")), "the arithmetic is not at N_PILOT"
# 28 fetch threads, the reason next to the number (the token limit of the Hub, the measured run)
assert D.WORKERS == 28
before_workers = SOURCE_TEXT.split("WORKERS = 28")[0].split("CHUNK = 2_000")[1]
assert all(w in before_workers for w in ("5,000 requests per 5 minutes", "3,000", "2,400", "16 threads",
                                         "wait_seconds", "retries")), "the reason for 28 threads is not there"
assert D.FETCH_ORDER == ("s2_url", "s1_url", "s2_row") and D.RAM_LIMIT_GB == 120.0
assert D.PAIR_BYTES == (2 + 12) * 384 * 384 * 2 == 4_128_768
assert D.MB == Z.MB and D.GB == Z.GB and D.SEP == "::"
assert D.TIMING_COLUMNS == ("chunk", "source", "cells", "saved", "skipped", "wall_s", "footers",
                            "mb_footers", "mb_rows", "requests", "wait_s", "retries")
assert D.INFO_COLUMNS == ("chunk", "file", "offset", "bytes", *D.META_COLUMNS, "s1_product_id",
                          "cloud_rule", "cloud_codes")
values = inspect.signature(D.fetch_pilot).parameters
assert values["cloud_codes"].default == R.CLOUD_CODES == (1, 2, 3)
assert (values["offset"].default, values["workers"].default, values["token"].default) == (
    "baseline", 28, None)
assert inspect.signature(D.pilot_sample).parameters["n"].default == 26_000
assert inspect.signature(D.split_held_out).parameters["n"].default == 2_000
assert inspect.signature(D.load_pairs_to_ram).parameters["limit_gb"].default == 120.0
# 16 threads read the pairs into RAM, the reason next to the number (archives in order, the GIL, a guess)
assert inspect.signature(D.load_pairs_to_ram).parameters["workers"].default == D.RAM_WORKERS == 16
before_ram = SOURCE_TEXT.split("RAM_WORKERS = 16")[0].split("RAM_LIMIT_GB = ")[1]
assert all(w in before_ram for w in ("13", "GIL", "guess", "MB and the pairs per second")), \
    "the reason for 16 threads is not there"
assert D.OFFSET_HOW == {"baseline": ("baseline", "per_image"), None: ("none",)}
for name in ("pilot_sample", "fetch_pilot", "split_held_out", "s2_grid", "channel_stats",
             "load_pairs_to_ram", "pilot_summary", "open_pair", "pair_cell", "split_location",
             "pair_location", "member_bytes", "pair_info_at", "tar_index", "read_infos", "chunk_infos",
             "load_infos", "locations_of", "infos_file", "pair_locations", "ChunkReader"):
    title = getattr(D, name).__doc__.strip().splitlines()[0]
    assert title and ":" not in title, f"{name}: title {title!r}"
assert ":" not in D.__doc__.strip().splitlines()[0]
assert "no-data" in D.ZERO_NOTE and "offset" in D.ZERO_NOTE

# ---- pilot_sample, the draw of stage 0 in the order of the shards, in chunks ------------------
rng = np.random.default_rng(3)
years = np.repeat([2016, 2020, 2024], 10)
table = pd.DataFrame({
    "grid_cell": [f"{i}U_{i}R" for i in range(30)],
    "s2_url": [f"s2_{k}.parquet" for k in rng.integers(0, 4, 30)],
    "s2_row": rng.integers(0, 500, 30),
    "s2_product_id": [f"S2_{i}" for i in range(30)],
    "s2_datetime": [ts(f"{y}-06-01") for y in years],
    "s1_url": [f"s1_{k}.parquet" for k in rng.integers(0, 6, 30)],
    "s1_row": rng.integers(0, 500, 30),
    "s1_product_id": [f"S1_{i}" for i in range(30)],
    "s1_datetime": [ts(f"{y}-06-02") for y in years],
    "gap_days": 1.0, "s2_year": years})
D.CHUNK = 4
sample = D.pilot_sample(table, n=12, seed=0)
want = P.sample_pairs(table, n=12, seed=0)
assert list(sample.columns) == [*P.PAIR_COLUMNS, "chunk"] and len(sample) == 12
assert sorted(sample["grid_cell"]) == sorted(want["grid_cell"])
assert sample["s2_year"].value_counts().to_dict() == {2016: 4, 2020: 4, 2024: 4}
keys = list(zip(sample["s2_url"], sample["s1_url"], sample["s2_row"]))
assert keys == sorted(keys), "not sorted by s2_url, s1_url, s2_row"
assert list(sample["chunk"]) == [0] * 4 + [1] * 4 + [2] * 4
# the same rows and order from a table in another order
shuffled = D.pilot_sample(table.sample(frac=1.0, random_state=5), n=12, seed=0)
assert shuffled.equals(sample)
# stable, rows with the same shards keep the order of sample_pairs (s2_year, grid_cell)
tied = table.assign(s2_url="one", s1_url="one", s2_row=0)
assert list(D.pilot_sample(tied, 12, 0)["grid_cell"]) == list(
    P.sample_pairs(tied, 12, 0)["grid_cell"])
assert D.pilot_sample(table, 12, 1)["grid_cell"].tolist() != sample["grid_cell"].tolist()
everything = D.pilot_sample(table, n=100)
assert len(everything) == 30 and list(everything["chunk"]) == [i // 4 for i in range(30)]
assert len(D.pilot_sample(table, n=0)) == 0 and "chunk" in D.pilot_sample(table, n=0).columns
print("pilot_sample: ok")

# ---- locations, "<tar>::<member>" or a plain file --------------------------------------------
loc = D.pair_location(Path("D:/x y/chunk_3.tar"), "chunk_3/12U_34R.npz")
assert loc == f"{Path('D:/x y/chunk_3.tar')}::chunk_3/12U_34R.npz"
assert D.split_location(loc) == (Path("D:/x y/chunk_3.tar"), "chunk_3/12U_34R.npz")
assert D.pair_cell(loc) == "12U_34R"
assert D.split_location(Path("a/b/5U_6R.npz")) == (Path("a/b/5U_6R.npz"), None)
assert D.split_location("a/b/5U_6R.npz") == (Path("a/b/5U_6R.npz"), None)
assert D.pair_cell(Path("a/b/5U_6R.npz")) == "5U_6R" == D.pair_cell("a/b/5U_6R.npz")
raises(ValueError, D.split_location, "::chunk_3/x.npz", says="not a location")
raises(ValueError, D.split_location, "a.tar::", says="not a location")
print("locations: ok")

# ---- synthetic shards, 8 cells in 2 chunks -------------------------------------------------------
SHARDS = TMP / "shards"
SHARDS.mkdir()
E_LEFT, E_TOP = 500002.5, 4000004.25        # a corner that is not a multiple of 10 m
# cell 0 fine | 1 Sentinel-1 no-data in the crop | 2 half cloud | 3 fine
# cell 4 fine | 5 N9999, the offset is decided per image | 6 N0201, no offset | 7 fine
BASELINE = {5: "N9999", 6: "N0201"}
S2_DAY = [f"2022010{i + 1}" for i in range(8)]
S1_DAY = [f"2022010{i + 2}" for i in range(8)]


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
    day, baseline = S2_DAY[i], BASELINE.get(i, "N0400")
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
    row["thumbnail"] = b"not a picture " * 50
    return row


def s1_stored(i: int) -> dict:
    rng = np.random.default_rng(100 + i)
    left, top = E_LEFT - 3420.0, E_TOP + 3420.0
    row = {"product_id": f"S1A_IW_GRDH_1SDV_{S1_DAY[i]}T051500_x_rtc",
           "grid_cell": f"{100 + i}U_{200 + i}R", "product_datetime": f"{S1_DAY[i]}T051500"}
    for k, name in enumerate(("vv", "vh")):
        values = rng.choice(np.array([0.01, 0.02, 0.05, 0.1], np.float32) / (1 + 3 * k),
                            (R.S1_CELL_PX, R.S1_CELL_PX))
        if i == 1 and name == "vh":
            values[400, 400] = -32768.0
        row[name] = tif(values.astype(np.float32), left=left, top=top, nodata=-32768.0)
    row["thumbnail"] = b"not a picture " * 50
    return row


def write_shard(name: str, rows: list) -> str:
    """One row per row group, as in Major-TOM."""
    schema = pa.schema([(k, pa.string() if k in M.META_COLUMNS else pa.binary())
                        for k in rows[0]])
    with pq.ParquetWriter(SHARDS / name, schema, compression="snappy") as writer:
        for row in rows:
            writer.write_table(pa.Table.from_pylist([row], schema=schema))
    return str(SHARDS / name)


S2_STORED, S1_STORED = [s2_stored(i) for i in range(8)], [s1_stored(i) for i in range(8)]
S2_A, S2_B = write_shard("s2_a.parquet", S2_STORED[:4]), write_shard("s2_b.parquet", S2_STORED[4:])
S1_A, S1_B = write_shard("s1_a.parquet", S1_STORED[:2]), write_shard("s1_b.parquet", S1_STORED[2:6])
S1_C = write_shard("s1_c.parquet", S1_STORED[6:])
s2_index = M.build_s2_index(shards=[S2_A, S2_B], progress=False, verify=0)
pd.DataFrame({
    "grid_cell": [r["grid_cell"] for r in S1_STORED], "grid_row_u": 0, "grid_col_r": 0,
    "product_id": [r["product_id"] for r in S1_STORED],
    "timestamp": [f"{day}T051512" for day in S1_DAY],
    "nodata": 0.0, "orbit_state": "ascending", "centre_lat": 36.1, "centre_lon": 15.0,
    "crs": "EPSG:EPSG:32633", "parquet_url": [S1_A] * 2 + [S1_B] * 4 + [S1_C] * 2,
    "parquet_row": [0, 1, 0, 1, 2, 3, 0, 1]}).to_parquet(SHARDS / "metadata.parquet", index=False)
s1_index = M.load_s1_index(source=SHARDS / "metadata.parquet")
all_pairs, pair_counts = P.pair_index(s2_index, s1_index)
assert pair_counts["pairs"] == 8
cell = [r["grid_cell"] for r in S2_STORED]
sample = D.pilot_sample(all_pairs, n=8, seed=0)
assert list(sample["grid_cell"]) == cell and list(sample["chunk"]) == [0] * 4 + [1] * 4
assert list(sample["s2_url"]) == [S2_A] * 4 + [S2_B] * 4
assert list(sample["s1_url"]) == [S1_A] * 2 + [S1_B] * 4 + [S1_C] * 2
KEPT = [0, 3, 4, 5, 6, 7]
IMAGES_PER_CELL = len(R.S2_BANDS) + 1 + 2                # 12 bands, the mask, VV and VH

# ---- fetch_pilot end to end, two chunks --------------------------------------------------------
WORK, ARCHIVE = TMP / "work", TMP / "archive"

# The archive goes to the archive folder (Drive) as the stage 0 notebook wrote its archive. The
# tar is built on the local disk (WORK), copied next to its final name as .part, renamed, then
# the JSON and the infos table. Watched from outside through shutil.copyfile, os.replace and
# _write_json. Nothing under the final name may be created or changed by the copy, and after the
# call neither a local tar nor a .part nor the local pair files are left
EVENTS: list = []
REAL_COPY, REAL_REPLACE, REAL_JSON = shutil.copyfile, os.replace, D._write_json


def copy_seen(src, dst, *args, **kwargs):
    src, dst = Path(src), Path(dst)
    watched = dst.parent == ARCHIVE
    if watched:
        final = dst.with_name(dst.name.removesuffix(".part"))
        assert dst.name.endswith(".part") and src.parent == WORK and src.name == final.name, (src, dst)
        before = final.stat().st_mtime_ns if final.exists() else None
        EVENTS.append(("copy", final.name))
    out = REAL_COPY(src, dst, *args, **kwargs)
    if watched:
        after = final.stat().st_mtime_ns if final.exists() else None
        assert after == before, f"{final} was made or changed by the copy"
        assert dst.stat().st_size == src.stat().st_size
    return out


def replace_seen(src, dst, *args, **kwargs):
    src, dst = Path(src), Path(dst)
    if dst.parent == ARCHIVE and dst.suffix == ".tar":
        assert src == dst.with_name(dst.name + ".part"), (src, dst)
        EVENTS.append(("rename", dst.name))
    return REAL_REPLACE(src, dst, *args, **kwargs)


def json_seen(path, value):
    if Path(path).parent == ARCHIVE:
        EVENTS.append(("json", Path(path).name))
    return REAL_JSON(path, value)


def location(k: int, i: int, archive: Path = ARCHIVE) -> str:
    return D.pair_location(archive / f"chunk_{k}.tar", f"chunk_{k}/{cell[i]}.npz")


def no_pair_read(*args, **kwargs):
    raise SystemExit("a pair was read although every chunk has its archive and its infos")


def arrays_equal(a: R.Pair, b: R.Pair) -> bool:
    return (np.array_equal(a.s1_db, b.s1_db) and list(a.s2) == list(b.s2)
            and all(np.array_equal(a.s2[band], b.s2[band]) for band in a.s2)
            and a.meta == b.meta and a.bounds == b.bounds and a.grid_cell == b.grid_cell
            and a.s2_datetime == b.s2_datetime and a.s1_datetime == b.s1_datetime)


def pair_from_bytes(data: bytes, where: str) -> R.Pair:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return D._pair_of(z, where)


shutil.copyfile, os.replace, D._write_json = copy_seen, replace_seen, json_seen
M._FOOTERS.clear()
(paths, skips, timing), cost, text = quiet(D.fetch_pilot, sample, WORK, ARCHIVE, workers=2,
                                           cloud_codes=[3], offset="baseline", progress=False)
assert paths == [location(i // 4, i) for i in KEPT]
assert all(isinstance(p, str) and D.SEP in p for p in paths)
assert list(skips.columns) == list(D.SKIP_COLUMNS)
assert skips[["chunk", "grid_cell", "reason"]].values.tolist() == [
    [0, cell[1], "s1_nodata_in_crop"], [0, cell[2], "cloud_gt_10pct"]]
assert "VH has 1 of 147456" in skips["detail"][0] and "0.5000" in skips["detail"][1]
assert list(timing.columns) == list(D.TIMING_COLUMNS)
assert timing[["chunk", "source", "cells", "saved", "skipped"]].values.tolist() == [
    [0, "fetched", 4, 2, 2], [1, "fetched", 4, 4, 0]]
assert (timing["wall_s"] > 0).all() and (timing["mb_rows"] > 0).all()
assert (timing["mb_footers"] > 0).all()
# chunk 0 pays the footers of s2_a, s1_a and s1_b. Chunk 1 those of s2_b and s1_c, s1_b is
# still in memory. So 3 and 2, and 5 in all, one per shard and never one per cell
assert timing["footers"].tolist() == [3, 2] and cost["footers"] == 5
assert cost["decoded"] == 8 * IMAGES_PER_CELL and cost["row_requests"] == 16
# the requests per chunk are counted and printed with the rate and the waits
assert int(timing["requests"].sum()) == cost["requests"] and (timing["requests"] > 0).all()
assert timing["wait_s"].tolist() == [0.0, 0.0] and timing["retries"].tolist() == [0, 0]
assert re.search(r"chunk 0  fetched .* \d+ requests, \d+\.\d per second, waited 0 s in 0 retries", text), text
totals = timing.attrs
assert {k: totals[k] for k in totals if k != "wall_s"} == {
    "chunks": 2, "fetched": 2, "from_archive": 0, "infos_built": 0, "cells": 8, "pairs": 6,
    "skipped": 2}
assert totals["wall_s"] > 0
assert ("chunk 0  fetched  cells 4  saved 2  skipped 2 (s1_nodata_in_crop 1, cloud_gt_10pct 1)"
        in text)
assert "chunk 1  fetched  cells 4  saved 4  skipped 0  " in text and " MB  footers " in text
assert "pilot  6 pairs of 8 cells, 2 skipped, 2 chunks fetched, 0 from archives, 0 infos tables built" in text
assert EVENTS == [("copy", "chunk_0.tar"), ("rename", "chunk_0.tar"), ("json", "chunk_0.json"),
                  ("copy", "chunk_1.tar"), ("rename", "chunk_1.tar"), ("json", "chunk_1.json")], EVENTS
# the local pair files and the local tar are gone, the archive on Drive holds the chunk
assert WORK.is_dir() and sorted(p.name for p in WORK.iterdir()) == [], "local files were left"
assert not list(ARCHIVE.glob("*.part")) and not list(ARCHIVE.glob("*.tmp"))

# the archives, one tar, one JSON and one infos table per chunk, nothing else
assert sorted(p.name for p in ARCHIVE.iterdir()) == [
    "chunk_0.json", "chunk_0.tar", "chunk_0_infos.csv", "chunk_1.json", "chunk_1.tar", "chunk_1_infos.csv"]
for k, cells in ((0, [cell[0], cell[3]]), (1, cell[4:])):
    tar, info = ARCHIVE / f"chunk_{k}.tar", ARCHIVE / f"chunk_{k}.json"
    meta = json.loads(info.read_text(encoding="utf-8"))
    assert meta["files"] == [f"{c}.npz" for c in cells] and meta["bytes"] == tar.stat().st_size
    assert (meta["chunk"], meta["cells"], meta["pairs"]) == (k, 4, len(cells))
    assert meta["cloud_codes"] == [3] and meta["offset"] == "baseline"
    assert meta["fetch"]["wall_s"] > 0 and meta["fetch"]["footers"] == timing["footers"][k]
    assert meta["fetch"]["workers"] == 2 and meta["fetch"]["cells_read"] == 4
    assert meta["fetch"]["requests"] == timing["requests"][k] and meta["fetch"]["retries"] == 0
    with tarfile.open(tar) as archive:
        assert sorted(archive.getnames()) == sorted(f"chunk_{k}/{c}.npz" for c in cells)
        for c in cells:
            held = archive.extractfile(f"chunk_{k}/{c}.npz").read()
            assert held == D.member_bytes(D.pair_location(tar, f"chunk_{k}/{c}.npz"))
    # the infos table, one row per pair in the order of the tar, the rules and the place in the tar
    infos = D.read_infos(D.infos_file(ARCHIVE, k))
    assert list(infos.columns) == list(D.INFO_COLUMNS) and len(infos) == len(cells)
    assert infos["file"].tolist() == [f"chunk_{k}/{c}.npz" for c in cells]
    assert infos["grid_cell"].tolist() == cells and (infos["chunk"] == k).all()
    scanned = D._scan_tar(tar)
    assert [(o, b) for o, b in zip(infos["offset"], infos["bytes"])] == [scanned[f] for f in infos["file"]]
    assert (infos["cloud_rule"] == "applied").all() and (infos["cloud_codes"] == "[3]").all()
    assert infos["s2_product_id"].tolist() == [S2_STORED[cell.index(c)]["product_id"] for c in cells]
    assert infos["s1_product_id"].tolist() == [S1_STORED[cell.index(c)]["product_id"] for c in cells]
    assert (infos["crs"] == "EPSG:32633").all() and (infos["cloud_fraction"] == 0.0).all()
    assert infos["s2_datetime"].tolist() == [ts(f"{S2_DAY[cell.index(c)]}T101010") for c in cells]
    assert (infos["gap_days"] > 0).all()
chunk0 = json.loads((ARCHIVE / "chunk_0.json").read_text(encoding="utf-8"))
assert [(s["grid_cell"], s["reason"]) for s in chunk0["skips"]] == [
    (cell[1], "s1_nodata_in_crop"), (cell[2], "cloud_gt_10pct")]
assert all(s["detail"] for s in chunk0["skips"])
assert json.loads((ARCHIVE / "chunk_1.json").read_text(encoding="utf-8"))["skips"] == []
infos_all = D.load_infos(ARCHIVE, [1, 0])
assert infos_all["grid_cell"].tolist() == [cell[i] for i in KEPT] and infos_all["chunk"].tolist() == [0, 0, 1, 1, 1, 1]
assert D.locations_of(infos_all, ARCHIVE) == paths
assert infos_all["baseline"].tolist() == ["N0400", "N0400", "N0400", "N9999", "N0201", "N0400"]
assert infos_all["offset_removed"].tolist() == [True, True, True, True, False, True]
assert infos_all["offset_how"].tolist() == ["baseline"] * 3 + ["per_image", "baseline", "baseline"]
assert len(D.load_infos(ARCHIVE, [])) == 0 and list(D.load_infos(ARCHIVE, []).columns) == list(D.INFO_COLUMNS)

# the rules were applied once, in make_pair. Offset by baseline, per image for N9999, not below.
# The pairs are read from the archives in place
HOW = {"N0400": "baseline", "N9999": "per_image", "N0201": "baseline"}
for i, path in zip(KEPT, paths):
    pair = D.open_pair(path)
    baseline = BASELINE.get(i, "N0400")
    removed = baseline != "N0201"
    assert (pair.meta["baseline"], pair.meta["offset_removed"], pair.meta["offset_how"]) == (
        baseline, removed, HOW[baseline]), i
    for band in R.S2_BANDS:
        stored = M.decode_tif(S2_STORED[i][band])[0].astype(np.int64)
        want_band = np.maximum(stored - 1000, 0) if removed else stored
        assert np.array_equal(pair.s2[band], want_band.astype(np.uint16)), (i, band)
    assert pair.meta["cloud_rule"] == "applied" and pair.meta["cloud_codes"] == [3]
    assert pair.s1_db.shape == (2, 384, 384) and pair.s1_db.dtype == np.float16
    assert pair.grid_cell == cell[i]
    assert pair.bounds == (E_LEFT, E_TOP - 3840.0, E_LEFT + 3840.0, E_TOP)
    assert D.pair_info_at(path)["meta"] == pair.meta
first_bytes = {p: D.member_bytes(p) for p in paths}
stamps = {p: p.stat().st_mtime_ns for p in ARCHIVE.iterdir()}

# a copy to Drive that fails leaves nothing under the final name, no .part, no local tar and
# no JSON. The pairs on disk stay. The next call writes the archive and reads only the cells
# that had left no file (the skipped ones), never a pair on disk, then deletes the local files
FAIL_ARCHIVE = TMP / "archive fail"
first_chunk = sample[sample["chunk"] == 0]


def copy_fails(src, dst, *args, **kwargs):
    if Path(dst).parent == FAIL_ARCHIVE:
        raise OSError("the copy to Drive failed")
    return copy_seen(src, dst, *args, **kwargs)


shutil.copyfile = copy_fails
with redirect_stdout(io.StringIO()):
    raises(OSError, D.fetch_pilot, first_chunk, WORK, FAIL_ARCHIVE, workers=2, cloud_codes=[3],
           offset="baseline", progress=False, says="the copy to Drive failed")
shutil.copyfile = copy_seen
assert sorted(p.name for p in FAIL_ARCHIVE.iterdir()) == [], list(FAIL_ARCHIVE.iterdir())
assert sorted(p.name for p in WORK.iterdir()) == ["chunk_0"], "the local tar was left or the pairs are gone"
on_disk = {p.name: p.read_bytes() for p in (WORK / "chunk_0").iterdir()}
assert sorted(on_disk) == [f"{cell[0]}.npz", f"{cell[3]}.npz"]
(paths_f, skips_f, timing_f), cost_f, _ = quiet(D.fetch_pilot, first_chunk, WORK, FAIL_ARCHIVE, workers=2,
                                                cloud_codes=[3], offset="baseline", progress=False)
assert paths_f == [location(0, 0, FAIL_ARCHIVE), location(0, 3, FAIL_ARCHIVE)]
assert list(timing_f["source"]) == ["fetched"] and len(skips_f) == 2
assert cost_f["decoded"] == 2 * IMAGES_PER_CELL, "a pair on disk was read again"
assert sorted(p.name for p in FAIL_ARCHIVE.iterdir()) == ["chunk_0.json", "chunk_0.tar", "chunk_0_infos.csv"]
assert (FAIL_ARCHIVE / "chunk_0.tar").stat().st_size == (ARCHIVE / "chunk_0.tar").stat().st_size
with tarfile.open(FAIL_ARCHIVE / "chunk_0.tar") as archive:
    assert sorted(archive.getnames()) == sorted(f"chunk_0/{c}.npz" for c in (cell[0], cell[3]))
for p in paths_f:                                      # the pair files on disk went into the tar as they were
    assert D.member_bytes(p) == on_disk[Path(D.split_location(p)[1]).name]
assert sorted(p.name for p in WORK.iterdir()) == [], "the local files were not deleted after the archive"
assert {p: D.member_bytes(p) for p in paths} == first_bytes

# a second call reads nothing, unpacks nothing and opens no pair, every chunk is read in place
REAL_OPEN, REAL_INFO_AT, REAL_BYTES = D.open_pair, D.pair_info_at, D.member_bytes
D.open_pair = D.pair_info_at = D.member_bytes = no_pair_read
try:
    (paths2, skips2, timing2), cost2, text2 = quiet(D.fetch_pilot, sample, WORK, ARCHIVE, workers=2,
                                                    cloud_codes=[3], offset="baseline", progress=False)
finally:
    D.open_pair, D.pair_info_at, D.member_bytes = REAL_OPEN, REAL_INFO_AT, REAL_BYTES
assert paths2 == paths and skips2.equals(skips)
assert cost2["requests"] == 0 and cost2["decoded"] == 0, "cells were read again"
assert list(timing2["source"]) == ["archive"] * 2
assert timing2.drop(columns="source").equals(timing.drop(columns="source")), "the JSON numbers"
assert {p: p.stat().st_mtime_ns for p in ARCHIVE.iterdir()} == stamps, "an archive was written"
assert sorted(p.name for p in WORK.iterdir()) == [], "something was unpacked"
assert (timing2.attrs["from_archive"], timing2.attrs["fetched"], timing2.attrs["pairs"],
        timing2.attrs["infos_built"]) == (2, 0, 6, 0)
assert "chunk 0  archive  pairs 2, skipped 2, chunk_0.tar read in place" in text2
assert "unpacked" not in text2 and "2 from archives, 0 infos tables built" in text2
# a deleted work folder changes nothing, the archives are the source
shutil.rmtree(WORK)
(paths3, skips3, timing3), cost3, text3 = quiet(D.fetch_pilot, sample, WORK, ARCHIVE, workers=2,
                                                cloud_codes=[3], offset="baseline", progress=False)
assert paths3 == paths and skips3.equals(skips) and cost3["requests"] == 0 and cost3["decoded"] == 0
assert list(timing3["source"]) == ["archive"] * 2 and not WORK.exists()
assert {p: D.member_bytes(p) for p in paths} == first_bytes

# a chunk without its infos table is not complete. The table is built ONCE from the tar (the info
# entry of every member, nothing unpacked, no cell read), the time is printed, nothing else is done
built_before = D.read_infos(D.infos_file(ARCHIVE, 1))
D.infos_file(ARCHIVE, 1).unlink()
stamps = {p: p.stat().st_mtime_ns for p in ARCHIVE.iterdir()}
(paths4, _, timing4), cost4, text4 = quiet(D.fetch_pilot, sample, WORK, ARCHIVE, workers=2,
                                           cloud_codes=[3], offset="baseline", progress=False)
assert paths4 == paths and cost4["requests"] == 0 and cost4["decoded"] == 0
assert timing4.attrs["infos_built"] == 1 and list(timing4["source"]) == ["archive"] * 2
assert re.search(r"chunk 1  infos built from chunk_1\.tar, 4 pairs, \d+\.\d s, written to chunk_1_infos\.csv", text4), text4
assert "chunk 0  infos built" not in text4 and "1 infos tables built" in text4
assert {p: p.stat().st_mtime_ns for p in ARCHIVE.iterdir() if p.name != "chunk_1_infos.csv"} == {
    p: s for p, s in stamps.items()}, "a tar or a JSON was written"
assert D.read_infos(D.infos_file(ARCHIVE, 1)).equals(built_before), "the rebuilt table differs"
assert not WORK.exists()
# a table that does not list the members of the JSON is built again too
short = built_before.iloc[:-1]
D._write_infos(D.infos_file(ARCHIVE, 1), short)
(paths4b, _, timing4b), cost4b, text4b = quiet(D.fetch_pilot, sample, WORK, ARCHIVE, workers=2,
                                               cloud_codes=[3], offset="baseline", progress=False)
assert paths4b == paths and cost4b["requests"] == 0 and timing4b.attrs["infos_built"] == 1
assert D.read_infos(D.infos_file(ARCHIVE, 1)).equals(built_before)
# the same through chunk_infos and load_infos alone, a chunk that is not complete stops them
D.infos_file(ARCHIVE, 0).unlink()
with redirect_stdout(io.StringIO()) as buf:
    table0, built0 = D.chunk_infos(ARCHIVE, 0)
assert built0 is True and "chunk 0  infos built from chunk_0.tar, 2 pairs" in buf.getvalue()
assert D.chunk_infos(ARCHIVE, 0)[1] is False and table0.equals(D.read_infos(D.infos_file(ARCHIVE, 0)))
raises(FileNotFoundError, D.chunk_infos, ARCHIVE, 7, says="not complete")
raises(FileNotFoundError, D.load_infos, ARCHIVE, [0, 7], says="not complete")

# a cut archive is refused and the chunk is done again. Its local files are gone, so every cell
# of the chunk is read again, and the tar, the JSON and the infos table come back whole
tar0 = ARCHIVE / "chunk_0.tar"
whole = tar0.stat().st_size
with open(tar0, "r+b") as f:
    f.truncate(whole // 2)
(paths5, skips5, timing5), cost5, text5 = quiet(D.fetch_pilot, sample, WORK, ARCHIVE, workers=2,
                                                cloud_codes=[3], offset="baseline", progress=False)
assert paths5 == paths and skips5.equals(skips)
assert list(timing5["source"]) == ["fetched", "archive"]
assert cost5["decoded"] == 4 * IMAGES_PER_CELL, "not every cell of the chunk was read again"
assert ("chunk 0  the archive chunk_0.tar is missing its JSON or is cut, the chunk is fetched "
        "again") in text5
assert tar0.stat().st_size == json.loads((ARCHIVE / "chunk_0.json").read_text())["bytes"]
assert tar0.stat().st_size > whole // 2
with tarfile.open(tar0) as archive:
    assert sorted(archive.getnames()) == sorted(f"chunk_0/{c}.npz" for c in (cell[0], cell[3]))
for p in paths[:2]:                                    # the same pairs, saved again
    assert arrays_equal(D.open_pair(p), pair_from_bytes(first_bytes[p], p))
infos0 = D.read_infos(D.infos_file(ARCHIVE, 0))
assert infos0["grid_cell"].tolist() == [cell[0], cell[3]]
assert [(o, b) for o, b in zip(infos0["offset"], infos0["bytes"])] == [D._scan_tar(tar0)[f] for f in infos0["file"]]
assert sorted(p.name for p in WORK.iterdir()) == []
# a tar without its JSON is not complete either. Every cell of the chunk is read
(ARCHIVE / "chunk_1.json").unlink()
(paths6, _, timing6), cost6, text6 = quiet(D.fetch_pilot, sample, WORK, ARCHIVE, workers=2,
                                           cloud_codes=[3], offset="baseline", progress=False)
assert paths6 == paths and list(timing6["source"]) == ["archive", "fetched"]
assert cost6["decoded"] == 4 * IMAGES_PER_CELL and (ARCHIVE / "chunk_1.json").is_file()
assert "chunk 1  the archive chunk_1.tar is missing its JSON or is cut" in text6
assert sorted(p.name for p in ARCHIVE.iterdir()) == [
    "chunk_0.json", "chunk_0.tar", "chunk_0_infos.csv", "chunk_1.json", "chunk_1.tar", "chunk_1_infos.csv"]
for p in paths[2:]:                                    # the same pairs, saved again
    assert arrays_equal(D.open_pair(p), pair_from_bytes(first_bytes[p], p))
first_bytes = {p: D.member_bytes(p) for p in paths}

# a folder made under another offset rule is refused before anything is read or written
WORK2, ARCHIVE2 = TMP / "work2", TMP / "archive2"
raw = Z.fetch_pairs(sample[sample["chunk"] == 0], WORK2 / "chunk_0", cloud_codes=[3], workers=2,
                    progress=False, offset=None)
assert len(raw[0]) == 2 and P.load_pair(raw[0][0]).meta["offset_how"] == "none"
before = dict(M.STATS)
raises(ValueError, D.fetch_pilot, sample, WORK2, ARCHIVE2, workers=2, cloud_codes=[3],
       offset="baseline", progress=False, says="offset rule 'none', asked for is offset='baseline'")
assert M.STATS == before and not list(ARCHIVE2.glob("*.tar")), "read or written all the same"
WORK3 = TMP / "work3"
Z.fetch_pairs(sample[sample["chunk"] == 0], WORK3 / "chunk_0", cloud_codes=[3], workers=2,
              progress=False, offset="baseline")
before = dict(M.STATS)
raises(ValueError, D.fetch_pilot, sample, WORK3, ARCHIVE / "other", workers=2, cloud_codes=[3],
       offset=None, progress=False, says="Use another folder for another rule")
assert M.STATS == before
# the cloud rule the same way
raises(ValueError, D.fetch_pilot, sample, WORK3, ARCHIVE / "codes", workers=2, cloud_codes=[2],
       offset="baseline", progress=False, says="another folder")
assert M.STATS == before and not list((ARCHIVE / "other").glob("*")) and not list((ARCHIVE / "codes").glob("*"))
# an archive made under other rules is refused before anything is read
shutil.rmtree(WORK, ignore_errors=True)
for codes, rule in (([3], None), ([2, 3], "baseline"), (None, "baseline")):
    raises(ValueError, D.fetch_pilot, sample, WORK, ARCHIVE, workers=2, cloud_codes=codes,
           offset=rule, progress=False, says="another archive folder")
assert not WORK.exists() and M.STATS == before
# guards, before anything happens
raises(ValueError, D.fetch_pilot, all_pairs, WORK, ARCHIVE, progress=False, says="chunk")
for codes in ([], [3.5], [-1], [256], "3", ["cloud"], 3):
    raises(ValueError, D.fetch_pilot, sample, WORK, ARCHIVE, cloud_codes=codes, progress=False,
           says="whole numbers from 0 to 255")
for rule in ("date", "", True, "Baseline"):
    raises(ValueError, D.fetch_pilot, sample, WORK, ARCHIVE, offset=rule, progress=False,
           says="offset")
assert not WORK.exists() and M.STATS == before
# the pairs come back for the next parts, and a generator of codes is fine
(paths7, skips7, timing7), cost7, _ = quiet(D.fetch_pilot, sample, WORK, ARCHIVE, workers=2,
                                            cloud_codes=(c for c in [3, 3]), progress=False)
assert paths7 == paths and skips7.equals(skips) and cost7["requests"] == 0
assert list(timing7["source"]) == ["archive"] * 2

# every pair of a COMPLETE archive of the right rules is checked too, from its infos row.
# Another offset rule, no offset_how entry (a pair made before the rule existed) and another
# cloud rule each stop the call, nothing is read, the archive stays
good_infos = D.read_infos(D.infos_file(ARCHIVE, 0))
for change, says in ((dict(offset_how="none"), "offset rule 'none', asked for is offset='baseline'"),
                     (dict(offset_how=None), "no offset_how entry"),
                     (dict(cloud_rule="not applied", cloud_codes=None),
                      "cloud rule ('not applied', None), asked for is ('applied', [3])")):
    changed = good_infos.copy()
    for key, value in change.items():
        changed.loc[0, key] = value
    D._write_infos(D.infos_file(ARCHIVE, 0), changed)
    before = dict(M.STATS)
    stamps = {p: p.stat().st_mtime_ns for p in ARCHIVE.iterdir()}
    raises(ValueError, D.fetch_pilot, sample, WORK, ARCHIVE, workers=2, cloud_codes=[3],
           offset="baseline", progress=False, says=says)
    assert M.STATS == before, "a cell was read"
    assert {p: p.stat().st_mtime_ns for p in ARCHIVE.iterdir()} == stamps, "an archive was written"
    assert {p: D.member_bytes(p) for p in paths} == first_bytes
D._write_infos(D.infos_file(ARCHIVE, 0), good_infos)
assert D.read_infos(D.infos_file(ARCHIVE, 0)).equals(good_infos)
# an archive whose JSON names the right rules but holds a pair of another rule is refused when
# its infos are built from the tar, nothing is unpacked. Folders with a space in the name
pair0 = D.open_pair(paths[0])


def pair_with(meta: dict) -> R.Pair:
    return R.Pair(pair0.grid_cell, pair0.s1_db, pair0.s2, pair0.cloud_mask, pair0.s2_datetime,
                  pair0.s1_datetime, pair0.gap_days, pair0.crs, pair0.bounds, meta)


NET_WORK, NET_ARCHIVE = TMP / "work net", TMP / "archive net"
NET_ARCHIVE.mkdir()
(NET_WORK / "chunk_0").mkdir(parents=True)
NAME0 = f"{cell[0]}.npz"
P.save_pair(pair_with(dict(pair0.meta, offset_how="none")), NET_WORK / "chunk_0" / NAME0)
size, index = D._write_tar(NET_ARCHIVE / "chunk_0.tar", 0, [NET_WORK / "chunk_0" / NAME0], NET_WORK)
assert sorted(p.name for p in NET_WORK.iterdir()) == ["chunk_0"], "the local tar was left"
assert list(index) == [f"chunk_0/{NAME0}"]
assert index[f"chunk_0/{NAME0}"] == D._scan_tar(NET_ARCHIVE / "chunk_0.tar")[f"chunk_0/{NAME0}"]
D._write_json(NET_ARCHIVE / "chunk_0.json", {
    "chunk": 0, "files": [NAME0], "bytes": size, "cells": 1, "pairs": 1,
    "cloud_codes": [3], "offset": "baseline", "skips": [], "fetch": {}})
shutil.rmtree(NET_WORK)
before = dict(M.STATS)
with redirect_stdout(io.StringIO()) as buf:
    raises(ValueError, D.fetch_pilot, sample[sample["chunk"] == 0], NET_WORK, NET_ARCHIVE, workers=2,
           cloud_codes=[3], offset="baseline", progress=False, says="offset rule 'none'")
assert M.STATS == before and not NET_WORK.exists()
assert "chunk 0  infos built from chunk_0.tar, 1 pairs" in buf.getvalue()
assert (NET_ARCHIVE / "chunk_0_infos.csv").is_file()
# asked for the rule of the pair, the JSON still names the other one, so it is refused first
raises(ValueError, D.fetch_pilot, sample[sample["chunk"] == 0], NET_WORK, NET_ARCHIVE, workers=2,
       cloud_codes=[3], offset=None, progress=False, says="another archive folder")
# with a JSON that names the rule the pair was made under, the archived pair is fine
D._write_json(NET_ARCHIVE / "chunk_0.json", {
    "chunk": 0, "files": [NAME0], "bytes": size, "cells": 1, "pairs": 1,
    "cloud_codes": [3], "offset": None, "skips": [], "fetch": {}})
(out_net, _, timing_net), cost_net, _ = quiet(D.fetch_pilot, sample[sample["chunk"] == 0], NET_WORK,
                                              NET_ARCHIVE, workers=2, cloud_codes=[3], offset=None,
                                              progress=False)
assert out_net == [location(0, 0, NET_ARCHIVE)] and cost_net["requests"] == 0
assert list(timing_net["source"]) == ["archive"] and not NET_WORK.exists()
assert timing_net["requests"].isna().all(), "an old JSON without the request count gives None"
shutil.copyfile, os.replace, D._write_json = REAL_COPY, REAL_REPLACE, REAL_JSON
print("fetch_pilot: ok")

# ---- the same pairs as plain files, byte for byte -----------------------------------------------
PLAIN = TMP / "plain"
for k in (0, 1):
    with tarfile.open(ARCHIVE / f"chunk_{k}.tar") as archive:
        archive.extractall(PLAIN, filter="data")
plain = [PLAIN / f"chunk_{i // 4}" / f"{cell[i]}.npz" for i in KEPT]
assert all(p.is_file() for p in plain)
for loc, file in zip(paths, plain):
    assert D.member_bytes(loc) == file.read_bytes(), "a tar member differs from the pair file"
    assert D.pair_cell(loc) == file.stem
    assert arrays_equal(D.open_pair(loc), P.load_pair(file)) and arrays_equal(D.open_pair(file), P.load_pair(file))
    assert D.pair_info_at(loc) == P.load_pair_info(file) == D.pair_info_at(file)
    assert D.open_pair(str(file)).grid_cell == file.stem
raises(FileNotFoundError, D.open_pair, D.pair_location(ARCHIVE / "chunk_0.tar", "chunk_0/nobody.npz"),
       says="does not hold")
# the index of a tar is scanned once per process and again when the tar changes
D._INDEX.clear()
scans = []
REAL_SCAN = D._scan_tar
D._scan_tar = lambda tar: (scans.append(Path(tar).name), REAL_SCAN(tar))[1]
try:
    D.tar_index(ARCHIVE / "chunk_0.tar")
    D.tar_index(ARCHIVE / "chunk_0.tar")
    D.open_pair(paths[0])
    assert scans == ["chunk_0.tar"]
    os.utime(ARCHIVE / "chunk_0.tar")
    D.open_pair(paths[0])
    assert scans == ["chunk_0.tar"] * 2
    D.chunk_infos(ARCHIVE, 1)                          # the table primes the index, no scan
    D.open_pair(paths[2])
    assert scans == ["chunk_0.tar"] * 2
finally:
    D._scan_tar = REAL_SCAN
print("open_pair: ok")

# ---- split_held_out, on the grid cells of the locations ----------------------------------------
train, held = D.split_held_out(paths, n=2, seed=0, archive_dir=ARCHIVE)
assert len(train) == 4 and len(held) == 2 and not set(train) & set(held)
assert set(train) | set(held) == set(paths)
assert train == [p for p in paths if p in set(train)]
assert held == [p for p in paths if p in set(held)]
# by hand, a draw on the sorted cells
ordered = sorted(D.pair_cell(p) for p in paths)
picks = np.random.default_rng(0).choice(len(ordered), 2, replace=False)
assert sorted(D.pair_cell(p) for p in held) == sorted(ordered[i] for i in picks)
# the same split whatever the order given, and from the plain files (the same cells), another
# seed gives another one
train_b, held_b = D.split_held_out(list(reversed(paths)), n=2, seed=0)
assert held_b == list(reversed(held)) and train_b == list(reversed(train))
train_p, held_p = D.split_held_out(plain, n=2, seed=0)
assert [p.stem for p in held_p] == [D.pair_cell(p) for p in held]
assert [p.stem for p in train_p] == [D.pair_cell(p) for p in train]
assert len({tuple(sorted(D.pair_cell(p) for p in D.split_held_out(paths, n=2, seed=s)[1]))
            for s in range(6)}) > 1
# the two files next to the archives, one sorted cell per line
assert (ARCHIVE / D.TRAIN_FILE).read_text(encoding="utf-8").split("\n")[:-1] == sorted(
    D.pair_cell(p) for p in train)
assert (ARCHIVE / D.HELD_OUT_FILE).read_text(encoding="utf-8").split("\n")[:-1] == sorted(
    D.pair_cell(p) for p in held)
assert D.split_held_out(paths, n=0) == (paths, [])
raises(ValueError, D.split_held_out, paths, n=6, says="cannot hold out 6 of 6")
raises(ValueError, D.split_held_out, paths, n=-1)
raises(ValueError, D.split_held_out, paths + paths[:1], n=1, says="more than once")
print("split_held_out: ok")

# ---- s2_grid, the corner rule through a stub both ways -----------------------------------------
pair0 = D.open_pair(paths[0])
REAL_GRID = P.s2_to_grid
HAS_CORNER = "corner" in inspect.signature(REAL_GRID).parameters
calls: list = []


def with_corner(bands, size=R.CROP_PX, corner=None, crs=None):
    calls.append((corner, crs))
    return REAL_GRID(bands, size) + 1.0


def without_corner(bands, size=R.CROP_PX):
    calls.append(("plain",))
    return REAL_GRID(bands, size)


def broken(bands, size=R.CROP_PX, corner=None, crs=None):
    raise TypeError("something else is wrong")


P.s2_to_grid = with_corner
try:
    D._SAID.clear()
    with redirect_stdout(io.StringIO()) as buf:
        grid, used = D.s2_grid(pair0)
    assert used is True and calls == [((E_LEFT, E_TOP), "EPSG:32633")] and buf.getvalue() == ""
    assert calls[0][0] == (pair0.bounds[0], pair0.bounds[3]) and calls[0][1] == pair0.crs
    assert grid.dtype == np.float32 and grid.shape == (12, 384, 384)
    assert np.array_equal(grid, REAL_GRID(pair0.s2) + 1.0)
    calls.clear()
    grid, used = D.s2_grid(pair0, corner_rule=False)             # the stub, without the keywords
    assert used is False and calls == [(None, None)]
    assert np.array_equal(grid, REAL_GRID(pair0.s2) + 1.0)
    P.s2_to_grid = without_corner
    calls.clear()
    D._SAID.clear()
    with redirect_stdout(io.StringIO()) as buf:
        grid, used = D.s2_grid(pair0)
        grid2, used2 = D.s2_grid(pair0)
    assert used is False and used2 is False and calls == [("plain",)] * 2
    assert np.array_equal(grid, REAL_GRID(pair0.s2)) and np.array_equal(grid2, grid)
    assert buf.getvalue().count("NOT applied") == 1 and D.NO_CORNER_RULE in buf.getvalue()
    P.s2_to_grid = broken
    raises(TypeError, D.s2_grid, pair0, says="something else")       # not passed over
finally:
    P.s2_to_grid = REAL_GRID
D._SAID.clear()
with redirect_stdout(io.StringIO()) as buf:
    grid, used = D.s2_grid(pair0)
if HAS_CORNER:                       # pairing.s2_to_grid takes the keywords (2026-09-30)
    assert used is True and buf.getvalue() == ""
    assert np.array_equal(grid, REAL_GRID(pair0.s2, corner=(E_LEFT, E_TOP), crs="EPSG:32633"))
    assert not np.array_equal(grid, REAL_GRID(pair0.s2)), "the corner rule changed nothing"
else:
    assert used is False and D.NO_CORNER_RULE in buf.getvalue()
    assert np.array_equal(grid, REAL_GRID(pair0.s2))
assert np.array_equal(grid[1], pair0.s2["B02"]), "a 10 m band is copied"
print(f"s2_grid: ok (the installed s2_to_grid takes corner and crs: {HAS_CORNER})")

# ---- load_pairs_to_ram, from the archives in place and from the plain files -------------------
D._SAID.clear()
with redirect_stdout(io.StringIO()) as buf:
    ram = D.load_pairs_to_ram(paths, progress=False, workers=3)
assert set(ram) == {"s1", "s2", "grid_cell", "meta", "corner_rule", "bytes"}
assert ram["s1"].shape == (6, 2, 384, 384) and ram["s1"].dtype == np.float16
assert ram["s2"].shape == (6, 12, 384, 384) and ram["s2"].dtype == np.uint16
assert ram["grid_cell"] == [D.pair_cell(p) for p in paths] and ram["bytes"] == 6 * D.PAIR_BYTES
assert ram["corner_rule"] is HAS_CORNER
assert "load_pairs_to_ram  6 pairs x 4.13 MB = 0.02 GB, limit 120 GB" in buf.getvalue()
members_mb = sum(len(D.member_bytes(p)) for p in paths) / D.MB          # the rate line, the bytes of the members
assert re.search(rf"^pairs to RAM  6 pairs, {members_mb:,.0f} MB in \d+\.\d s, [\d,]+ MB per second, "
                 r"[\d,]+\.\d pairs per second, 3 threads$", buf.getvalue(), re.M), buf.getvalue()
for i, path in enumerate(plain):
    pair = P.load_pair(path)
    assert np.array_equal(ram["s1"][i], pair.s1_db)
    assert np.array_equal(ram["s2"][i], np.rint(D.s2_grid(pair)[0]).astype(np.uint16))
    assert np.array_equal(ram["s2"][i, 1], pair.s2["B02"])         # stored values, no offset again
meta = ram["meta"]
assert list(meta.columns) == list(D.META_COLUMNS) and list(meta["grid_cell"]) == ram["grid_cell"]
assert meta["baseline"].tolist() == ["N0400", "N0400", "N0400", "N9999", "N0201", "N0400"]
assert meta["offset_removed"].tolist() == [True, True, True, True, False, True]
assert meta["offset_how"].tolist() == ["baseline"] * 3 + ["per_image", "baseline", "baseline"]
assert (meta["cloud_fraction"] == 0.0).all() and (meta["crs"] == "EPSG:32633").all()
assert meta["s2_datetime"].tolist() == [ts(f"{S2_DAY[i]}T101010") for i in KEPT]
assert meta["s1_datetime"].tolist() == [ts(f"{S1_DAY[i]}T051500") for i in KEPT]
assert (meta["gap_days"] > 0).all() and (meta["gap_days"] < 1).all()
assert meta["s2_product_id"].tolist() == [S2_STORED[i]["product_id"] for i in KEPT]
# the same arrays from the plain files, and from the locations in another order
with redirect_stdout(io.StringIO()):
    ram_files = D.load_pairs_to_ram(plain, progress=False, workers=3)
    ram_mixed = D.load_pairs_to_ram([paths[3], plain[0], paths[1]], progress=False, workers=2)
assert np.array_equal(ram_files["s1"], ram["s1"]) and np.array_equal(ram_files["s2"], ram["s2"])
assert ram_files["grid_cell"] == ram["grid_cell"] and ram_files["meta"].equals(ram["meta"])
assert ram_mixed["grid_cell"] == [ram["grid_cell"][i] for i in (3, 0, 1)]
assert np.array_equal(ram_mixed["s2"], ram["s2"][[3, 0, 1]]) and np.array_equal(ram_mixed["s1"], ram["s1"][[3, 0, 1]])
with redirect_stdout(io.StringIO()) as buf:                          # the default threads, the same arrays
    ram_16 = D.load_pairs_to_ram([paths[3], plain[0], paths[1]], progress=False)
assert "16 threads" in buf.getvalue() and ram_16["grid_cell"] == ram_mixed["grid_cell"]
assert np.array_equal(ram_16["s2"], ram_mixed["s2"]) and np.array_equal(ram_16["s1"], ram_mixed["s1"])
del ram_files, ram_mixed, ram_16
# the stub both ways
P.s2_to_grid = with_corner
try:
    calls.clear()
    with redirect_stdout(io.StringIO()):
        ram_c = D.load_pairs_to_ram(paths[:2], progress=False)
    assert ram_c["corner_rule"] is True and calls == [((E_LEFT, E_TOP), "EPSG:32633")] * 2
    assert np.array_equal(ram_c["s2"][0], np.rint(REAL_GRID(pair0.s2) + 1.0).astype(np.uint16))
    assert np.array_equal(ram_c["s1"], ram["s1"][:2])
    calls.clear()
    with redirect_stdout(io.StringIO()):
        ram_n = D.load_pairs_to_ram(paths[:2], corner_rule=False, progress=False)
    assert ram_n["corner_rule"] is False and calls == [(None, None)] * 2
    assert np.array_equal(ram_n["s2"][0], np.rint(REAL_GRID(pair0.s2) + 1.0).astype(np.uint16))
    P.s2_to_grid = without_corner
    calls.clear()
    D._SAID.clear()
    with redirect_stdout(io.StringIO()) as buf:
        ram_w = D.load_pairs_to_ram(paths[:2], progress=False)
    assert ram_w["corner_rule"] is False and calls == [("plain",)] * 2
    assert buf.getvalue().count("NOT applied") == 1
    assert np.array_equal(ram_w["s2"][0], np.rint(REAL_GRID(pair0.s2)).astype(np.uint16))
finally:
    P.s2_to_grid = REAL_GRID
# the memory is printed and checked before anything is loaded
loads: list = []
REAL_OPEN = D.open_pair
D.open_pair = lambda where: (loads.append(where), REAL_OPEN(where))[1]
try:
    with redirect_stdout(io.StringIO()) as buf:
        raises(MemoryError, D.load_pairs_to_ram, paths, limit_gb=0.01, progress=False,
               says="6 pairs need 0.02 GB, the limit is 0.01 GB")
    assert loads == [] and "= 0.02 GB, limit 0.01 GB" in buf.getvalue()
    with redirect_stdout(io.StringIO()):
        assert D.load_pairs_to_ram(paths[:1], limit_gb=0.005, progress=False)["s1"].shape[0] == 1
    assert loads == paths[:1]
finally:
    D.open_pair = REAL_OPEN
with redirect_stdout(io.StringIO()):
    empty = D.load_pairs_to_ram([], progress=False)
assert empty["s1"].shape == (0, 2, 384, 384) and empty["grid_cell"] == []
assert len(empty["meta"]) == 0
assert list(empty["meta"].columns) == list(D.META_COLUMNS)
small = R.Pair("tiny", np.zeros((2, 4, 4), np.float16), pair0.s2, None, pair0.s2_datetime,
               pair0.s1_datetime, 1.0, pair0.crs, pair0.bounds, dict(pair0.meta))
(TMP / "small").mkdir()
P.save_pair(small, TMP / "small" / "tiny.npz")
with redirect_stdout(io.StringIO()):
    raises(ValueError, D.load_pairs_to_ram, [TMP / "small" / "tiny.npz"], progress=False,
           says="expected (2, 384, 384)")
print("load_pairs_to_ram: ok")

# ---- ChunkReader, the same arrays archive by archive, in any order of arrival --------------------
# the locations from the JSONs alone are those of the infos tables, in the same order
where = D.pair_locations(ARCHIVE, [1, 0, 1])
assert list(where.values()) == paths and list(where) == [cell[i] for i in KEPT]
assert D.pair_locations(ARCHIVE, []) == {}
raises(FileNotFoundError, D.pair_locations, ARCHIVE, [7])
BAD_JSON = TMP / "bad json"
BAD_JSON.mkdir()
(BAD_JSON / "chunk_0.json").write_text(json.dumps({"chunk": 0}), encoding="utf-8")
raises(ValueError, D.pair_locations, BAD_JSON, [0], says="names no file list")
(BAD_JSON / "chunk_0.json").write_text(json.dumps({"chunk": 0, "files": ["x.npz"]}), encoding="utf-8")
(BAD_JSON / "chunk_1.json").write_text(json.dumps({"chunk": 1, "files": ["x.npz"]}), encoding="utf-8")
raises(ValueError, D.pair_locations, BAD_JSON, [0, 1], says="grid cell x is in two archives")
assert inspect.signature(D.ChunkReader).parameters["workers"].default == D.RAM_WORKERS
assert inspect.signature(D.ChunkReader).parameters["limit_gb"].default == D.RAM_LIMIT_GB

# two lists whose rows are in another order than the archives, chunk 1 handed over first. Each list
# equals load_pairs_to_ram on it bit for bit, and the pairs are read in the reader's threads
LISTS = {"train": [paths[4], paths[0], paths[5], paths[2]], "held_out": [paths[3], paths[1]]}
with redirect_stdout(io.StringIO()):
    SEQ = {name: D.load_pairs_to_ram(locations, progress=False, workers=2) for name, locations in LISTS.items()}


def same_ram(a: dict, b: dict) -> bool:
    return (set(a) == set(b) and all(a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]) for k in ("s1", "s2"))
            and a["grid_cell"] == b["grid_cell"] and a["meta"].equals(b["meta"])
            and a["corner_rule"] is b["corner_rule"] and a["bytes"] == b["bytes"])


reading: list = []                                   # (thread, location) of every pair read
REAL_OPEN = D.open_pair
D.open_pair = lambda where: (reading.append((threading.get_ident(), str(where))), REAL_OPEN(where))[1]
try:
    with redirect_stdout(io.StringIO()) as buf:
        reader = D.ChunkReader(LISTS, workers=3)
        reader.arrived(ARCHIVE / "chunk_1.tar")
        reader.arrived(str(ARCHIVE / "chunk_0.tar"))
        got = reader.finish()
    printed = buf.getvalue()
    assert set(got) == set(LISTS) and all(same_ram(got[name], SEQ[name]) for name in LISTS)
    assert sorted(w for _, w in reading) == sorted(paths) and threading.get_ident() not in {t for t, _ in reading}
    # chunk 1 first, in the order of its members
    assert [w for _, w in reading][:4] != [] and {w for _, w in reading[:4]} == set(paths[2:])
    assert "ChunkReader  6 pairs in 2 lists x 4.13 MB = 0.02 GB, limit 120 GB" in printed
    mb = {k: sum(len(D.member_bytes(p)) for p in paths if f"chunk_{k}.tar::" in p) / D.MB for k in (0, 1)}
    first = re.search(rf"^chunk 1  4 pairs read from chunk_1\.tar in \d+\.\d s, {mb[1]:,.0f} MB, [\d,]+ MB per second$",
                      printed, re.M)
    second = re.search(rf"^chunk 0  2 pairs read from chunk_0\.tar in \d+\.\d s, {mb[0]:,.0f} MB, [\d,]+ MB per second$",
                       printed, re.M)
    assert first and second and first.start() < second.start(), printed
    assert re.search(rf"^pairs to RAM  6 pairs of 2 lists from 2 archives, {mb[0] + mb[1]:,.0f} MB in \d+\.\d s of "
                     r"reading, [\d,]+ MB per second, [\d,]+\.\d pairs per second, 3 threads$", printed, re.M), printed
    assert reader.pairs == 6 and abs(reader.read_mb - mb[0] - mb[1]) < 1e-9 and reader.read_s > 0
    assert ":" not in printed
    # it keeps no array afterwards and takes nothing more
    assert reader._rows is None and not reader._thread.is_alive()
    raises(RuntimeError, reader.arrived, ARCHIVE / "chunk_0.tar", says="finished or stopped")
    raises(RuntimeError, reader.finish, says="finished or stopped")

    # the corner rule as asked, and the default threads
    with redirect_stdout(io.StringIO()) as buf:
        reader = D.ChunkReader({"x": paths[:2]}, corner_rule=False)
        reader.arrived(ARCHIVE / "chunk_0.tar")
        no_rule = reader.finish()["x"]
        assert "16 threads" in buf.getvalue()
        assert same_ram(no_rule, D.load_pairs_to_ram(paths[:2], corner_rule=False, progress=False))
    assert no_rule["corner_rule"] is False

    # an archive from another folder (Drive instead of the runtime disk) with no infos table there. The
    # table is built from that tar and written next to it, the arrays are the same
    OTHER = TMP / "other folder"
    OTHER.mkdir()
    for name in ("chunk_1.tar", "chunk_1.json"):
        shutil.copyfile(ARCHIVE / name, OTHER / name)
    with redirect_stdout(io.StringIO()) as buf:
        reader = D.ChunkReader(LISTS, workers=2)
        reader.arrived(OTHER / "chunk_1.tar")
        reader.arrived(ARCHIVE / "chunk_0.tar")
        got = reader.finish()
    assert all(same_ram(got[name], SEQ[name]) for name in LISTS)
    assert "chunk 1  infos built from chunk_1.tar, 4 pairs" in buf.getvalue() and D.infos_file(OTHER, 1).is_file()

    # an archive never handed over - finish names it, and the arrays are let go
    with redirect_stdout(io.StringIO()):
        reader = D.ChunkReader(LISTS)
        reader.arrived(ARCHIVE / "chunk_0.tar")
        raises(RuntimeError, reader.finish,
               says="train 3 pairs in ['chunk_1.tar'], held_out 1 pairs in ['chunk_1.tar']")
    assert reader._rows is None

    # a cut archive is refused by its JSON when its turn comes, the error comes out of finish, the archive
    # after it is not read
    CUT = TMP / "cut folder"
    CUT.mkdir()
    shutil.copyfile(ARCHIVE / "chunk_0.json", CUT / "chunk_0.json")
    (CUT / "chunk_0.tar").write_bytes((ARCHIVE / "chunk_0.tar").read_bytes()[:1000])
    reading.clear()
    with redirect_stdout(io.StringIO()):
        reader = D.ChunkReader(LISTS)
        reader.arrived(CUT / "chunk_0.tar")
        reader.arrived(ARCHIVE / "chunk_1.tar")
        raises(FileNotFoundError, reader.finish, says="is not complete")
    assert reading == [] and reader._rows is None and not reader._thread.is_alive()

    # a pair that does not fit its row (a wrong shape) - the error of the reading comes out of finish
    BAD_TAR = TMP / "bad tar"
    BAD_TAR.mkdir()
    with tarfile.open(BAD_TAR / "chunk_5.tar", "w") as archive:
        archive.add(TMP / "small" / "tiny.npz", arcname="chunk_5/tiny.npz")
    (BAD_TAR / "chunk_5.json").write_text(json.dumps(
        {"chunk": 5, "files": ["tiny.npz"], "bytes": (BAD_TAR / "chunk_5.tar").stat().st_size}), encoding="utf-8")
    with redirect_stdout(io.StringIO()):
        reader = D.ChunkReader({"x": list(D.pair_locations(BAD_TAR, [5]).values())})
        reader.arrived(BAD_TAR / "chunk_5.tar")
        raises(ValueError, reader.finish, says="expected (2, 384, 384)")

    # refused at once - an entry that is not in a chunk archive, an archive handed over twice, a file that is
    # not a chunk archive. stop ends the reading and lets go of the arrays, a second stop does nothing
    with redirect_stdout(io.StringIO()):
        raises(ValueError, D.ChunkReader, {"x": [plain[0]]}, says="not a pair in a chunk archive")
        reader = D.ChunkReader(LISTS)
        reader.arrived(ARCHIVE / "chunk_0.tar")
        raises(ValueError, reader.arrived, OTHER / "chunk_0.tar", says="handed over twice")
        raises(ValueError, reader.arrived, ARCHIVE / "chunk_0.json", says="not a chunk archive")
        reader.stop()
        reader.stop()
    assert reader._rows is None and not reader._thread.is_alive()
    raises(RuntimeError, reader.arrived, ARCHIVE / "chunk_1.tar", says="finished or stopped")

    # the memory is printed and checked before anything is made or read
    reading.clear()
    with redirect_stdout(io.StringIO()) as buf:
        raises(MemoryError, D.ChunkReader, LISTS, limit_gb=0.01, says="6 pairs need 0.02 GB, the limit is 0.01 GB")
    assert reading == [] and "= 0.02 GB, limit 0.01 GB" in buf.getvalue()
finally:
    D.open_pair = REAL_OPEN
assert not [t for t in threading.enumerate() if t.name == "ChunkReader"], "a reader thread is still running"
print("ChunkReader: ok")

# ---- channel_stats -----------------------------------------------------------------------------
out = TMP / "stats" / "deep" / "channel_stats.json"
with redirect_stdout(io.StringIO()):
    stats = D.channel_stats(train, n=3, seed=0, out_path=out, progress=False)
assert set(stats) == {"seed", "n", "pairs_given", "cells_used", "corner_rule", "s1", "s2", "aerial"}
assert (stats["seed"], stats["n"], stats["pairs_given"], stats["aerial"]) == (0, 3, 4, None)
assert len(stats["cells_used"]) == 3 and stats["cells_used"] == sorted(stats["cells_used"])
assert set(stats["cells_used"]) < {D.pair_cell(p) for p in train}
assert stats["s1"]["channels"] == ["VV", "VH"] and stats["s1"]["unit"] == "dB"
assert len(stats["s1"]["mean"]) == 2 == len(stats["s1"]["sd"])
assert stats["s2"]["channels"] == list(R.S2_BANDS)
assert len(stats["s2"]["mean"]) == 12 == len(stats["s2"]["sd"]) == len(stats["s2"]["zero_pixels"])
assert stats["s1"]["pixels"] == 3 * 384 * 384 == stats["s2"]["pixels"]
assert stats["s2"]["note"] == D.ZERO_NOTE and "offset" in stats["s2"]["unit"]
assert stats["corner_rule"] is HAS_CORNER
# the same numbers as computed over the same pairs by hand
chosen = [p for p in train if D.pair_cell(p) in stats["cells_used"]]
db = np.stack([D.open_pair(p).s1_db.astype(np.float64) for p in chosen])
assert near(stats["s1"]["mean"][0], db[:, 0].mean())
assert near(stats["s1"]["mean"][1], db[:, 1].mean())
assert near(stats["s1"]["sd"][0], db[:, 0].std()) and near(stats["s1"]["sd"][1], db[:, 1].std())
assert -25 < stats["s1"]["mean"][0] < -5 and stats["s1"]["sd"][0] > 1
grids = np.stack([D.s2_grid(D.open_pair(p))[0].astype(np.float64) for p in chosen])
assert np.allclose(stats["s2"]["mean"], grids.mean((0, 2, 3)), rtol=1e-9, atol=1e-9)
assert np.allclose(stats["s2"]["sd"], grids.std((0, 2, 3)), rtol=1e-9, atol=1e-9)
assert stats["s2"]["zero_pixels"] == (grids == 0).sum((0, 2, 3)).tolist() == [0] * 12
assert json.loads(out.read_text(encoding="utf-8")) == stats and not list(out.parent.glob("*.tmp"))
# the same draw for the same seed, whatever the order. All pairs when n is larger
again = D.channel_stats(list(reversed(train)), n=3, seed=0, progress=False)
assert again["cells_used"] == stats["cells_used"] and again["s2"]["mean"] == stats["s2"]["mean"]
assert D.channel_stats(train, n=10, progress=False)["n"] == 4
assert D.channel_stats(train, n=4, corner_rule=False, progress=False)["corner_rule"] is False
# the same numbers from the plain files (the old way), the same pairs
from_files = D.channel_stats(train_p, n=3, seed=0, progress=False)
assert from_files["cells_used"] == stats["cells_used"]
for key in ("s1", "s2"):
    for what in ("mean", "sd"):
        assert np.allclose(from_files[key][what], stats[key][what], rtol=1e-12, atol=1e-12), (key, what)
    assert from_files[key]["pixels"] == stats[key]["pixels"]
assert from_files["s2"]["zero_pixels"] == stats["s2"]["zero_pixels"]


# a hand-computed value, two pairs with flat bands and three aerial images
def flat_pair(name: str, s1_value: float, s2_value: int):
    s2 = {band: np.full((3840 // R.S2_RES_M[band],) * 2, s2_value, np.uint16)
          for band in R.S2_BANDS}
    s1 = np.full((2, 384, 384), s1_value, np.float16)
    s1[1] -= 10                                            # VH is 10 dB below VV
    pair = R.Pair(name, s1, s2, None, ts("2021-06-10"), ts("2021-06-11"), 1.0, "EPSG:32633",
                  (E_LEFT, E_TOP - 3840.0, E_LEFT + 3840.0, E_TOP),
                  {"baseline": "N0400", "offset_removed": True, "offset_how": "baseline"})
    (TMP / "flat").mkdir(exist_ok=True)
    P.save_pair(pair, TMP / "flat" / f"{name}.npz")
    return TMP / "flat" / f"{name}.npz"


flat = [flat_pair("f1", -10.0, 100), flat_pair("f2", -20.0, 300)]
aerial = np.zeros((3, 4, 256, 256), np.uint8)
for c in range(4):
    aerial[0, c] = 10 * (c + 1)                            # one image lit, two dark
with redirect_stdout(io.StringIO()):
    hand = D.channel_stats(flat, n=2, aerial=aerial, progress=False)
assert hand["s1"]["mean"] == [-15.0, -25.0] and hand["s1"]["sd"] == [5.0, 5.0]
assert all(near(m, 200.0) for m in hand["s2"]["mean"])
assert all(near(s, 100.0) for s in hand["s2"]["sd"])
assert hand["s2"]["zero_pixels"] == [0] * 12 and hand["s2"]["pixels"] == 2 * 384 * 384
for c in range(4):                                         # mean v/3, SD v * sqrt(2) / 3
    v = 10 * (c + 1)
    assert near(hand["aerial"]["mean"][c], v / 3)
    assert near(hand["aerial"]["sd"][c], v * math.sqrt(2) / 3)
assert hand["aerial"]["channels"] == ["R", "G", "B", "NIR"] and hand["aerial"]["unit"] == "8 bit"
assert (hand["aerial"]["images"], hand["aerial"]["pixels"]) == (3, 3 * 256 * 256)
assert hand["cells_used"] == ["f1", "f2"] and hand["n"] == 2
# blocks of 256 images give the same numbers as one block
many = np.concatenate([aerial] * 100)                      # 300 images
blocks = D.channel_stats(flat, n=2, aerial=many, progress=False)["aerial"]
assert blocks["mean"] == hand["aerial"]["mean"] and blocks["images"] == 300
# zeros after the offset rule are real pixels and are counted
zero_path = flat_pair("f0", -10.0, 0)
zeros = D.channel_stats([zero_path, flat[0]], n=2, progress=False)
assert zeros["s2"]["zero_pixels"] == [384 * 384] * 12
assert all(near(m, 50.0) for m in zeros["s2"]["mean"])
for bad in (aerial.astype(np.uint16), aerial[:, :3], aerial[0]):
    raises(ValueError, D.channel_stats, flat, n=2, aerial=bad, progress=False,
           says="aerial must be")
raises(ValueError, D.channel_stats, [], progress=False, says="no training pair")
raises(ValueError, D.channel_stats, flat + flat[:1], progress=False, says="more than once")
print("channel_stats: ok")

# ---- pilot_summary, from the infos table and from the meta table of the RAM ------------------
with redirect_stdout(io.StringIO()) as buf:
    summary = D.pilot_summary(D.load_infos(ARCHIVE, [0, 1]), skips)
assert list(summary.columns) == list(D.SUMMARY_COLUMNS)
assert list(summary["kind"]) == ["pairs"] * 2 + ["skip"] * 2 + ["year"] + ["baseline"] * 3 \
    + ["offset"] * 3 + ["cloud"] * 2
rows = {(k, w): (p, s) for k, w, p, s in summary.itertuples(index=False)}
assert rows[("pairs", "kept")] == (6, 0.75) and rows[("pairs", "skipped")] == (2, 0.25)
assert rows[("skip", "s1_nodata_in_crop")] == (1, 0.125)
assert rows[("skip", "cloud_gt_10pct")] == (1, 0.125)
assert rows[("year", "2022")] == (6, 1.0)
assert rows[("baseline", "N0201")] == (1, 1 / 6) and rows[("baseline", "N0400")] == (4, 4 / 6)
assert rows[("baseline", "N9999")] == (1, 1 / 6)
assert rows[("offset", "removed by baseline")] == (4, 4 / 6)
assert rows[("offset", "removed per image (N9999)")] == (1, 1 / 6)
assert rows[("offset", "not removed")] == (1, 1 / 6)
assert rows[("cloud", "pairs with any cloud")] == (0, 0.0)
assert rows[("cloud", "mean cloud fraction")] == (6, 0.0)
assert summary["pairs"].dtype.kind == "i" and not any(":" in w for w in summary["what"])
printed = buf.getvalue()
assert "kept" in printed and "N9999" in printed and "0.750" in printed and "share" in printed
with redirect_stdout(io.StringIO()):
    pd.testing.assert_frame_equal(D.pilot_summary(ram["meta"], skips), summary)
# without skips, and with a cloud fraction and no baseline
with redirect_stdout(io.StringIO()):
    alone = D.pilot_summary(ram["meta"])
assert alone[alone["kind"] == "pairs"]["pairs"].tolist() == [6, 0]
assert "skip" not in set(alone["kind"])
changed = ram["meta"].copy()
changed.loc[0, "cloud_fraction"] = 0.05
changed.loc[1, "baseline"] = None
with redirect_stdout(io.StringIO()):
    cloudy = D.pilot_summary(changed, skips.iloc[:0])
rows = {(k, w): (p, s) for k, w, p, s in cloudy.itertuples(index=False)}
assert rows[("cloud", "pairs with any cloud")] == (1, 1 / 6)
assert near(rows[("cloud", "mean cloud fraction")][1], 0.05 / 6)
assert rows[("baseline", "no baseline")] == (1, 1 / 6) and rows[("baseline", "N0400")] == (3, 0.5)
assert ("skip", "s1_nodata_in_crop") not in rows and rows[("pairs", "kept")] == (6, 1.0)
with redirect_stdout(io.StringIO()):
    nothing = D.pilot_summary(ram["meta"].iloc[:0], skips)
assert nothing[nothing["kind"] == "pairs"]["pairs"].tolist() == [0, 2]
assert math.isnan(nothing[nothing["what"] == "not removed"]["share"].iloc[0])
print("pilot_summary: ok")

del ram, ram_c, ram_n, ram_w, empty
shutil.rmtree(TMP, ignore_errors=True)
print("test_pilotdata: all checks passed")
