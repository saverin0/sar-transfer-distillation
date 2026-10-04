"""Major-TOM index and single-cell reads, on synthetic shards and a server on this machine."""
import _setup

import http.server
import json
import os
import re
import shutil
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import requests
from rasterio.transform import Affine

from sartransfer.distil import majortom as M
from sartransfer.distil import records as R

DIR = _setup.TMP_BASE / "majortom"
shutil.rmtree(DIR, ignore_errors=True)
DIR.mkdir(parents=True)
M.BACKOFF_S = 0.01
DEFAULTS = {k: getattr(M, k) for k in ("HF", "MERGE_GAP", "FOOTER_GUESS", "FOOTER_CACHE")}
LEFT, TOP = 500002.5, 4000004.25          # a corner that is not a multiple of 10 m, as in the data


def fails(kind, fn, *args, **kwargs) -> str:
    """The call must raise `kind`. Returns the error text."""
    try:
        fn(*args, **kwargs)
    except kind as err:
        return str(err)
    raise SystemExit(f"{getattr(fn, '__name__', fn)}{args[1:]} did not raise {kind.__name__}")


def fresh() -> None:
    """Forget the footers and go back to the module's own settings."""
    M._FOOTERS.clear()
    for k, v in DEFAULTS.items():
        setattr(M, k, v)


def cost(fn, *args, **kwargs) -> dict:
    """What STATS counted while fn ran."""
    before = dict(M.STATS)
    fn(*args, **kwargs)
    return {k: M.STATS[k] - before[k] for k in M.STATS}


# ------------------------------------------------------------------ synthetic data

def tif(array, res=10.0, left=LEFT, top=TOP, crs="EPSG:32633", nodata=None,
        turn=(0.0, 0.0)) -> bytes:
    """One GeoTIFF as bytes. `turn` = the two terms of the affine that are 0 when north is up."""
    array = array if array.ndim == 3 else array[None]
    with rasterio.MemoryFile() as memory:
        with memory.open(driver="GTiff", count=array.shape[0], height=array.shape[1],
                         width=array.shape[2], dtype=array.dtype, crs=crs, nodata=nodata,
                         transform=Affine(res, turn[0], left, turn[1], -res, top),
                         compress="lzw") as dst:
            dst.write(array)
        return memory.read()


def s2_row(i: int, **change) -> dict:
    rng = np.random.default_rng(i)
    day = f"2022010{i + 1}"
    row = {"product_id": f"S2A_MSIL2A_{day}T101010_N0400_R022_T33TUM_{day}T121212",
           "grid_cell": f"{100 + i}U_{200 + i}R", "product_datetime": f"{day}T101010"}
    for band in R.S2_BANDS:
        px = R.CROP_PX * 10 // R.S2_RES_M[band]
        row[band] = tif((1000 + rng.integers(0, 8, (px, px))).astype(np.uint16), R.S2_RES_M[band])
    row["cloud_mask"] = tif(rng.integers(0, 4, (R.CROP_PX, R.CROP_PX)).astype(np.uint8))
    row["thumbnail"] = b"not a picture " * 50
    row.update(change)
    return row


def s1_row(i: int, size: int = 96, **change) -> dict:
    rng = np.random.default_rng(100 + i)
    day = f"2022010{i + 1}"
    row = {"product_id": f"S1A_IW_GRDH_1SDV_{day}T051500_{day}T051525_041000_04E000_rtc",
           "grid_cell": f"{100 + i}U_{200 + i}R", "product_datetime": f"{day}T051500"}
    for name in ("vv", "vh"):
        row[name] = tif(rng.uniform(0.001, 0.5, (size, size)).astype(np.float32),
                        left=LEFT - 3420, top=TOP + 3420, nodata=-32768.0)
    row["thumbnail"] = b"not a picture " * 50
    row.update(change)
    return row


def write_shard(name: str, rows: list, stats: bool = True, one_group: bool = False) -> str:
    """One row per row group, as in Major-TOM. `one_group` breaks that rule on purpose."""
    path = DIR / name
    schema = pa.schema([(k, pa.string() if k in M.META_COLUMNS else pa.binary()) for k in rows[0]])
    with pq.ParquetWriter(path, schema, compression="snappy", write_statistics=stats) as writer:
        for group in ([rows] if one_group else [[r] for r in rows]):
            writer.write_table(pa.Table.from_pylist(group, schema=schema))
    return str(path)


def row_of(shard: str, row: dict, number: int = 0, **more) -> dict:
    """What an index row looks like for `row`, stored as row `number` of `shard`."""
    return {"grid_cell": row["grid_cell"], "product_id": row["product_id"],
            "product_datetime": pd.Timestamp(row["product_datetime"], tz="UTC"),
            "parquet_url": shard, "parquet_row": number, **more}


S2_ROWS = [s2_row(i) for i in range(8)]
S2_ROWS[7]["grid_cell"] = "107U_20"              # looks cut short, the footer is not trusted
A = write_shard("s2_a.parquet", S2_ROWS[:4])
B = write_shard("s2_b.parquet", S2_ROWS[4:6], stats=False)        # no statistics in the footer
C = write_shard("s2_c.parquet", S2_ROWS[6:])
TWO = write_shard("two_in_one.parquet", S2_ROWS[:2], one_group=True)
(DIR / "text.parquet").write_bytes(b"this is not a parquet file " * 10)

# ------------------------------------------------------------------ decode_tif

a16 = (np.arange(64 * 48).reshape(64, 48) % 5000).astype(np.uint16)
array, transform, crs, nodata = M.decode_tif(tif(a16, 20.0))
assert array.dtype == np.uint16 and array.shape == (64, 48) and (array == a16).all()
assert transform == (20.0, 0.0, LEFT, 0.0, -20.0, TOP) and all(type(v) is float for v in transform)
assert crs == "EPSG:32633" and nodata is None
a32 = np.linspace(0.001, 2.0, 2500, dtype=np.float32).reshape(50, 50)
array, _, crs, nodata = M.decode_tif(tif(a32, nodata=-32768.0, crs="EPSG:32752"))
assert array.dtype == np.float32 and (array == a32).all() and nodata == -32768.0
assert crs == "EPSG:32752"
assert M.decode_tif(tif(np.stack([a16, a16])))[0].shape == (2, 64, 48)
fails(ValueError, M.decode_tif, b"this is not a picture")

# ------------------------------------------------------------------ read_row, local file

got = M.read_row(A, 2, ["grid_cell", "B02", "cloud_mask"])
assert set(got) == {"grid_cell", "B02", "cloud_mask"}                # only the asked columns
assert all(got[k] == S2_ROWS[2][k] for k in got)                     # values as stored
assert isinstance(got["grid_cell"], str) and isinstance(got["B02"], bytes)
whole = M.read_row(A, 0)
assert set(whole) == set(S2_ROWS[0]) and all(whole[k] == S2_ROWS[0][k] for k in whole)

# only the bytes of the asked columns of the asked row are fetched
ranges = []
real_range = M._read_range


def logged_range(url, start, end, token=None, kind="row"):
    assert kind == "row"                     # the footer is in memory, only the row is fetched
    ranges.append((start, end))
    return real_range(url, start, end, token, kind)


M._read_range = logged_range
M.MERGE_GAP = 0
footer = pq.read_metadata(A)
names = footer.schema.names
sizes = {c: footer.row_group(3).column(names.index(c)).total_compressed_size for c in names}
before = dict(M.STATS)
assert M.read_row(A, 3, ["B08"])["B08"] == S2_ROWS[3]["B08"]
assert len(ranges) == 1 and ranges[0][1] - ranges[0][0] == sizes["B08"]
assert M.STATS["bytes"] - before["bytes"] == sizes["B08"] < os.path.getsize(A) / 20
assert M.STATS["requests"] - before["requests"] == 1
ranges.clear()
M.read_row(A, 3, ["B02", "B03"])                                     # neighbours, one request
assert len(ranges) == 1 and ranges[0][1] - ranges[0][0] == sizes["B02"] + sizes["B03"]
ranges.clear()
M.read_row(A, 3, ["B02", "B04"])                                     # B03 lies between them
assert [b - a for a, b in ranges] == [sizes["B02"], sizes["B04"]]
ranges.clear()
M.MERGE_GAP = DEFAULTS["MERGE_GAP"]
got = M.read_row(A, 3, ["B02", "B04"])                               # small gap, fetched as one
assert len(ranges) == 1
assert ranges[0][1] - ranges[0][0] == sum(sizes[c] for c in ("B02", "B03", "B04"))
assert got["B02"] == S2_ROWS[3]["B02"] and got["B04"] == S2_ROWS[3]["B04"]
M._read_range = real_range

assert "no row 4" in fails(IndexError, M.read_row, A, 4)
assert "no row -1" in fails(IndexError, M.read_row, A, -1)
assert "the file has" in fails(KeyError, M.read_row, A, 0, ["B10"])
assert "holds 2 rows" in fails(ValueError, M.read_row, TWO, 0, ["grid_cell"])
assert "PAR1" in fails(ValueError, M.read_row, str(DIR / "text.parquet"), 0)
(DIR / "long_footer.parquet").write_bytes(b"x" * 20 + struct.pack("<I", 10 ** 6) + b"PAR1")
assert "footer says" in fails(ValueError, M.read_row, str(DIR / "long_footer.parquet"), 0)
(DIR / "four_bytes.parquet").write_bytes(b"PAR1")                     # too short for a footer
assert "PAR1" in fails(ValueError, M.read_row, str(DIR / "four_bytes.parquet"), 0)
fails(FileNotFoundError, M.read_row, str(DIR / "nothing_here.parquet"), 0)
size = os.path.getsize(A)
assert len(M._read_range(A, size - 10, size)) == 10
assert "got 10 bytes" in fails(OSError, M._read_range, A, size - 10, size + 10)   # never padded

# the footer is kept in memory, at most FOOTER_CACHE of them
fresh()
tails = []
real_tail = M._read_tail
M._read_tail = lambda url, n, token=None: (tails.append(url), real_tail(url, n, token))[1]
M.read_row(A, 0, ["grid_cell"])
M.read_row(A, 1, ["grid_cell"])
assert tails == [A]
M.FOOTER_CACHE = 1
M.read_row(C, 0, ["grid_cell"])
M.read_row(A, 1, ["grid_cell"])
assert tails == [A, C, A] and list(M._FOOTERS) == [A]
M._read_tail = real_tail
fresh()

# ------------------------------------------------------------------ what a read costs

first = cost(M.read_row, A, 3, ["B08"])                # the footer is not in memory, it is paid
tail = min(M.FOOTER_GUESS, os.path.getsize(A))
assert (first["footers"], first["footer_requests"], first["footer_bytes"]) == (1, 1, tail)
assert (first["row_requests"], first["row_bytes"]) == (1, sizes["B08"])
assert (first["requests"], first["bytes"]) == (2, tail + sizes["B08"])
assert first["footer_seconds"] > 0 and first["row_seconds"] > 0
assert abs(first["seconds"] - first["footer_seconds"] - first["row_seconds"]) < 1e-9
assert first["retries"] == 0 and first["wait_seconds"] == 0 and first["decoded"] == 0
again = cost(M.read_row, A, 2, ["B08"])                # same shard, the footer costs nothing now
assert (again["footers"], again["footer_requests"], again["footer_bytes"]) == (0, 0, 0)
assert again["footer_seconds"] == 0
assert (again["requests"], again["bytes"]) == (again["row_requests"], again["row_bytes"])
assert again["row_bytes"] > 0 and again["seconds"] == again["row_seconds"] > 0
fresh()
M.FOOTER_GUESS = 64                                    # a footer in two parts is ONE footer
need = struct.unpack("<I", (DIR / "s2_a.parquet").read_bytes()[-8:-4])[0] + 8
two = cost(M._footer, A)
assert (two["footers"], two["footer_requests"], two["footer_bytes"]) == (1, 2, need)
assert (two["requests"], two["bytes"], two["row_requests"]) == (2, need, 0)
fresh()

# ------------------------------------------------------------------ small parts

parts = M._Parts(100, [(10, b"abcdefghij")], "x")
parts.seek(12)
assert parts.read(3) == b"cde" and parts.tell() == 15
assert parts.seek(0, 2) == 100 and parts.read(5) == b""
for position, n in ((5, 10), (15, 10), (50, 1)):                     # not held, never made up
    parts.seek(position)
    assert "not fetched" in fails(OSError, parts.read, n)
M.MERGE_GAP = 10
assert M._merge([(50, 60), (0, 10), (20, 30), (41, 45)]) == [(0, 30), (41, 60)]
fresh()

good = SimpleNamespace(has_min_max=True, min="921D_252L", max="921D_252L", has_null_count=True,
                       null_count=0)
assert M._stat_value(good) == "921D_252L"
assert M._stat_value(None) is None
for change in ({"has_min_max": False}, {"max": "921D_252M"}, {"null_count": 1},
               {"min": "", "max": ""}, {"min": b"921D_252L", "max": b"921D_252L"}):
    assert M._stat_value(SimpleNamespace(**{**vars(good), **change})) is None, change

# ------------------------------------------------------------------ build_s2_index

groups = []
real_groups = M._read_groups


def logged_groups(url, rows, columns=None, token=None):
    groups.append((url, list(rows), list(columns or [])))
    return real_groups(url, rows, columns, token)


paid = cost(M.build_s2_index, shards=[A], workers=1, progress=False, verify=0)
assert (paid["footers"], paid["row_requests"], paid["row_bytes"]) == (1, 0, 0)   # footer alone
fresh()
M._read_groups = logged_groups
index = M.build_s2_index(shards=[A, B, C], workers=4, progress=False, verify=0)
assert list(index.columns) == list(M.S2_INDEX_COLUMNS)
assert len(index) == 8
for column in ("grid_cell", "product_id"):
    assert index[column].tolist() == [r[column] for r in S2_ROWS]
assert index["grid_cell"][7] == "107U_20"                            # as stored, not repaired
assert str(index["product_datetime"].dt.tz) == "UTC"
assert index["product_datetime"].tolist() == [pd.Timestamp(r["product_datetime"], tz="UTC")
                                              for r in S2_ROWS]
assert index["product_datetime"][0] == pd.Timestamp("2022-01-01 10:10:10", tz="UTC")
assert index["parquet_row"].tolist() == [0, 1, 2, 3, 0, 1, 0, 1]
assert index["parquet_url"].tolist() == [A] * 4 + [B] * 2 + [C] * 2
assert index.attrs["read_by_columns"] == [B, C]
# A came from its footer alone. B (no statistics) and C (a value that looks cut) were read
assert sorted(groups) == [(B, [0, 1], list(M.META_COLUMNS)), (C, [0, 1], list(M.META_COLUMNS))]

groups.clear()
M.build_s2_index(shards=[A], workers=1, progress=False, verify=3)     # three rows are read back
assert [g[1] for g in groups] == [[0], [1], [3]] and all(g[0] == A for g in groups)
groups.clear()
M.build_s2_index(shards=[B, A, C], workers=1, progress=False, verify=3)
# B and C were read by their columns, their rows already are the stored values. So the three
# rows that are read back and compared are all rows of A, which came from its footer
assert [g[:2] for g in groups if len(g[1]) == 1] == [(A, [0]), (A, [1]), (A, [3])]
real_row = M.read_row
M.read_row = lambda *a, **k: {**real_row(*a, **k), "product_id": "another product"}
assert "differ" in fails(RuntimeError, M.build_s2_index, shards=[A], progress=False)
M.read_row = lambda *a, **k: {**real_row(*a, **k), "product_datetime": "20220101T101011"}
assert "differ" in fails(RuntimeError, M.build_s2_index, shards=[A], progress=False)
M.read_row = real_row
M._read_groups = real_groups

cache = DIR / "cache" / "s2_index.parquet"
first = M.build_s2_index(cache=cache, shards=[A, B, C], workers=2, progress=True)
assert cache.is_file() and [p.name for p in cache.parent.iterdir()] == [cache.name]
real_shard = M._shard_index
M._shard_index = lambda *a, **k: fails(SystemExit, lambda: None)      # any use stops the test
again = M.build_s2_index(cache=cache, shards=[A, B, C], progress=False)
assert list(again.columns) == list(M.S2_INDEX_COLUMNS)
assert first.attrs["read_by_columns"] == again.attrs["read_by_columns"] == [B, C]
bare = again.copy()
bare.attrs = {}
bare.to_parquet(DIR / "bare.parquet")                  # a cache that does not say how it was read
assert M.build_s2_index(cache=DIR / "bare.parquet", progress=False).attrs == {
    "read_by_columns": None}
assert again.to_dict("list") == first.to_dict("list") == index.to_dict("list")
assert str(again["product_datetime"].dt.tz) == "UTC"
assert len(M.build_s2_index(cache=cache, progress=False)) == 8     # no list given, cache as it is
fails(ValueError, M.build_s2_index, cache=cache, shards=[A], progress=False)   # other shards
again.drop(columns="parquet_row").to_parquet(DIR / "cache" / "broken.parquet")
text = fails(KeyError, M.build_s2_index, cache=DIR / "cache" / "broken.parquet", progress=False)
assert "broken.parquet" in text and "parquet_row" in text           # says which file, which column
M._shard_index = real_shard

# a write that stops half way leaves no cache behind
real_to_parquet = pd.DataFrame.to_parquet


def stopped(self, path, **kwargs):
    open(path, "wb").write(b"half a file")
    raise OSError("the disk is full")


pd.DataFrame.to_parquet = stopped
fails(OSError, M.build_s2_index, cache=cache.with_name("stopped.parquet"), shards=[A],
      progress=False)
pd.DataFrame.to_parquet = real_to_parquet
assert sorted(p.name for p in cache.parent.iterdir()) == ["broken.parquet", cache.name]

# a footer value that was cut short is not used, the stored value is read instead
real_stat = M._stat_value
for cut_what in (r"\d{8}T\d{6}", r"\d+[UD]_\d+[LR]"):
    M._stat_value = lambda s: (v := real_stat(s)) and (v[:-2] if re.fullmatch(cut_what, v) else v)
    cut = M.build_s2_index(shards=[A], progress=False, verify=0)
    assert cut.attrs["read_by_columns"] == [A], cut_what
    assert cut.to_dict("list") == index[:4].to_dict("list"), cut_what
M._stat_value = real_stat

bad_date = write_shard("s2_bad_date.parquet",
                       [dict(S2_ROWS[0], product_datetime="2022-01-01 10:10")])
assert "20221115T161819" in fails(ValueError, M.build_s2_index, shards=[bad_date], progress=False)
no_cell = write_shard("s2_no_cell.parquet",
                      [{k: v for k, v in S2_ROWS[0].items() if k != "grid_cell"}])
fails(KeyError, M.build_s2_index, shards=[no_cell], progress=False)
assert "2 rows in 1 row groups" in fails(ValueError, M.build_s2_index, shards=[TWO], progress=False)
# a row without a cell name or a product name is an error, it never goes into the index
for column in ("grid_cell", "product_id"):
    for value in (None, ""):
        fresh()
        gap = write_shard("s2_gap.parquet", [S2_ROWS[0], dict(S2_ROWS[1], **{column: value})])
        text = fails(ValueError, M.build_s2_index, shards=[gap], progress=False, verify=0)
        assert "s2_gap.parquet row 1" in text and f"{column} is missing" in text, text
assert "no shards" in fails(ValueError, M.build_s2_index, shards=[], progress=False)
fresh()

# ------------------------------------------------------------------ load_s1_index

S1_ROWS = [s1_row(0), s1_row(1)]
S1 = write_shard("s1_a.parquet", S1_ROWS)
meta = pd.DataFrame({
    "grid_cell": [r["grid_cell"] for r in S1_ROWS] + ["874D_289R"],
    "grid_row_u": [100, 101, -874], "grid_col_r": [200, 201, 289],
    "product_id": [r["product_id"] for r in S1_ROWS] + ["S1B_IW_GRDH_1SDV_x_rtc"],
    "timestamp": ["20220101T051512", "20220102T051512", "20191105T144255"],
    "nodata": [0.0, 0.0, 0.0], "orbit_state": ["ascending", "descending", "ascending"],
    "centre_lat": [36.1, 36.2, -78.45], "centre_lon": [15.0, 15.1, 130.43],
    "crs": ["EPSG:EPSG:32633", "EPSG:EPSG:32633", "EPSG:EPSG:32752"],
    "parquet_url": [S1, S1, "https://example.invalid/part_00001.parquet"],
    "parquet_row": [0, 1, 0]})
meta.to_parquet(DIR / "metadata.parquet", index=False)
s1_index = M.load_s1_index(source=DIR / "metadata.parquet")
assert list(s1_index.columns) == list(M.S1_INDEX_COLUMNS) and len(s1_index) == 3
assert s1_index["crs"].tolist() == ["EPSG:32633", "EPSG:32633", "EPSG:32752"]
assert str(s1_index["product_datetime"].dt.tz) == "UTC"
assert s1_index["product_datetime"][2] == pd.Timestamp("2019-11-05 14:42:55", tz="UTC")
for column in ("grid_cell", "product_id", "parquet_url", "parquet_row", "nodata", "orbit_state",
               "centre_lat", "centre_lon"):
    assert s1_index[column].tolist() == meta[column].tolist(), column       # as stored

s1_cache = DIR / "cache" / "s1_index.parquet"
first = M.load_s1_index(cache=s1_cache, source=DIR / "metadata.parquet")
assert s1_cache.is_file() and not list(s1_cache.parent.glob("*.part"))
again = M.load_s1_index(cache=s1_cache, source=DIR / "nothing_here.parquet")   # source not touched
assert again.to_dict("list") == first.to_dict("list") == s1_index.to_dict("list")
assert str(again["product_datetime"].dt.tz) == "UTC"
assert "broken.parquet" in fails(KeyError, M.load_s1_index, cache=DIR / "cache" / "broken.parquet")
meta.drop(columns="orbit_state").to_parquet(DIR / "metadata_short.parquet", index=False)
text = fails(KeyError, M.load_s1_index, source=DIR / "metadata_short.parquet")
assert "orbit_state" in text and "the file has" in text
meta.assign(timestamp="2019-11-05").to_parquet(DIR / "metadata_date.parquet", index=False)
fails(ValueError, M.load_s1_index, source=DIR / "metadata_date.parquet")

# ------------------------------------------------------------------ read_s2_cell

M._read_groups = logged_groups
groups.clear()
before = dict(M.STATS)
cell = M.read_s2_cell(index.iloc[2])
assert M.STATS["decoded"] - before["decoded"] == len(R.S2_BANDS) + 1      # 12 bands, the mask
assert M.STATS["decode_seconds"] > before["decode_seconds"]
assert "thumbnail" not in groups[0][2] and groups[0][1] == [2] and len(groups) == 1
M._read_groups = real_groups
assert isinstance(cell, R.S2Cell)
assert cell.grid_cell == S2_ROWS[2]["grid_cell"] and cell.product_id == S2_ROWS[2]["product_id"]
assert cell.datetime == pd.Timestamp("2022-01-03 10:10:10", tz="UTC")
assert set(cell.bands) == set(cell.transforms) == set(R.S2_BANDS)
for band in R.S2_BANDS:
    res = float(R.S2_RES_M[band])
    px = R.CROP_PX * 10 // R.S2_RES_M[band]
    assert cell.bands[band].dtype == np.uint16 and cell.bands[band].shape == (px, px)
    assert (cell.bands[band] == M.decode_tif(S2_ROWS[2][band])[0]).all()
    assert cell.transforms[band] == (res, 0.0, LEFT, 0.0, -res, TOP)
assert cell.crs == "EPSG:32633"
assert cell.bounds == (LEFT, TOP - 3840.0, LEFT + 3840.0, TOP)       # left, bottom, right, top
assert cell.cloud_mask.dtype == np.uint8 and cell.cloud_mask.shape == (384, 384)
assert (cell.cloud_mask == M.decode_tif(S2_ROWS[2]["cloud_mask"])[0]).all()
assert M.read_s2_cell(dict(index.iloc[2])).grid_cell == cell.grid_cell        # a plain dict works
other_day = {**dict(index.iloc[2]), "product_datetime": pd.Timestamp("2000-01-01", tz="UTC")}
assert M.read_s2_cell(other_day).datetime == cell.datetime           # the date stored in the shard

# the shard must hold the cell that the index row names
assert "grid_cell" in fails(ValueError, M.read_s2_cell,
                            {**dict(index.iloc[2]), "grid_cell": "1U_1R"})
assert "product_id" in fails(ValueError, M.read_s2_cell, {**dict(index.iloc[2]), "product_id": "x"})
assert "grid_cell" in fails(ValueError, M.read_s2_cell, {**dict(index.iloc[2]), "parquet_row": 1})

small = (1000 + np.arange(100 * 100).reshape(100, 100) % 7).astype(np.uint16)
mask8 = np.zeros((R.CROP_PX, R.CROP_PX), np.uint8)
BAD_S2 = {
    "B05 is int16": {"B05": tif(small.astype(np.int16), 20.0)},
    "B03 has two layers": {"B03": tif(np.stack([small, small]))},
    "B8A in another CRS": {"B8A": tif(small, 20.0, crs="EPSG:32634")},
    "B02 is not north up, first term": {"B02": tif(small, turn=(0.5, 0.0)), "cloud_mask": None},
    "B02 is not north up, second term": {"B02": tif(small, turn=(0.0, 0.5)), "cloud_mask": None},
    "cloud mask starts 500 m further east": {"cloud_mask": tif(mask8, left=LEFT + 500)},
    "cloud mask starts half a pixel further south": {"cloud_mask": tif(mask8, top=TOP - 5)},
    "cloud mask in another CRS": {"cloud_mask": tif(mask8, crs="EPSG:32634")},
    "cloud mask names no CRS": {"cloud_mask": tif(mask8, crs=None)},
    "cloud mask at 20 m": {"cloud_mask": tif(mask8[:192, :192], 20.0)},
    "no band names a CRS": {b: tif(small, R.S2_RES_M[b], crs=None) for b in R.S2_BANDS},
    "B01 is empty": {"B01": None},
    "cloud mask is uint16": {"cloud_mask": tif(small)},
    "the date has another form": {"product_datetime": "2022-01-01T10:10:10Z"},
}
for number, (what, change) in enumerate(BAD_S2.items()):
    row = dict(S2_ROWS[0], **change)
    shard = write_shard(f"s2_bad_{number}.parquet", [row])
    text = fails(ValueError, M.read_s2_cell, {**row_of(shard, S2_ROWS[0])})
    assert shard in text, what
    assert ("north up" in text) == ("north up" in what) and ("grid of B02" in text) == (
        "cloud mask" in what and "uint16" not in what), (what, text)
row = dict(S2_ROWS[0], cloud_mask=None, B02=tif(small))
cell = M.read_s2_cell(row_of(write_shard("s2_no_mask.parquet", [row]), row))
assert cell.cloud_mask is None
assert cell.bands["B02"].shape == (100, 100)          # sizes are the business of the pairing rules
assert cell.bounds == (LEFT, TOP - 1000.0, LEFT + 1000.0, TOP)
# a mask that is off by less than GRID_TOL_M is on the grid (a rounding error, not a shift)
row = dict(S2_ROWS[0], cloud_mask=tif(mask8, left=LEFT + M.GRID_TOL_M / 2))
cell = M.read_s2_cell(row_of(write_shard("s2_near_mask.parquet", [row]), row))
assert cell.cloud_mask.shape == (R.CROP_PX, R.CROP_PX)

# ------------------------------------------------------------------ read_s1_cell

cell = M.read_s1_cell(s1_index.iloc[1])
assert isinstance(cell, R.S1Cell)
assert cell.grid_cell == S1_ROWS[1]["grid_cell"] and cell.product_id == S1_ROWS[1]["product_id"]
assert cell.datetime == pd.Timestamp("2022-01-02 05:15:00", tz="UTC")       # as stored in the shard
assert s1_index["product_datetime"][1] - cell.datetime == pd.Timedelta(seconds=12)
for name, layer in (("vv", cell.vv), ("vh", cell.vh)):
    assert layer.dtype == np.float32 and layer.shape == (96, 96)
    assert (layer == M.decode_tif(S1_ROWS[1][name])[0]).all()            # linear power, untouched
assert cell.transform == (10.0, 0.0, LEFT - 3420, 0.0, -10.0, TOP + 3420)
assert cell.crs == "EPSG:32633" and cell.nodata == -32768.0 and type(cell.nodata) is float
assert cell.orbit_state == "descending"
assert M.read_s1_cell(row_of(S1, S1_ROWS[0])).orbit_state is None         # not known, not made up
assert M.read_s1_cell(row_of(S1, S1_ROWS[0], orbit_state=float("nan"))).orbit_state is None
assert "grid_cell" in fails(ValueError, M.read_s1_cell, row_of(S1, S1_ROWS[0], 1))

wave = np.linspace(0.001, 0.5, 96 * 96, dtype=np.float32).reshape(96, 96)
corner = {"left": LEFT - 3420, "top": TOP + 3420, "nodata": -32768.0}
BAD_S1 = {
    "vh has another size": {"vh": tif(wave[:90], **corner)},
    "vv is float64": {"vv": tif(wave.astype(np.float64), **corner)},
    "both have two layers": {k: tif(np.stack([wave, wave]), **corner) for k in ("vv", "vh")},
    "vh in another CRS": {"vh": tif(wave, crs="EPSG:32634", **corner)},
    "vh starts 10 m further east": {"vh": tif(wave, **{**corner, "left": LEFT - 3410})},
    "vh names another no-data value": {"vh": tif(wave, **{**corner, "nodata": 0.0})},
    "vh names no no-data value": {"vh": tif(wave, **{**corner, "nodata": None})},
    "no layer names a CRS": {k: tif(wave, crs=None, **corner) for k in ("vv", "vh")},
    "vv is empty": {"vv": None},
    "vh is empty": {"vh": None},
    "the date has another form": {"product_datetime": "2022-01-01"},
}
for number, (what, change) in enumerate(BAD_S1.items()):
    row = dict(S1_ROWS[0], **change)
    shard = write_shard(f"s1_bad_{number}.parquet", [row])
    text = fails(ValueError, M.read_s1_cell, row_of(shard, S1_ROWS[0]))
    assert shard in text, what
# NaN as the no-data value of both layers is the same value, although NaN != NaN
row = dict(S1_ROWS[0], **{k: tif(wave, **{**corner, "nodata": float("nan")}) for k in ("vv", "vh")})
assert np.isnan(M.read_s1_cell(row_of(write_shard("s1_nan.parquet", [row]), row)).nodata)

# ------------------------------------------------------------------ tries and waits, no network


class Answer:
    """What a session gives back, as far as `_get` looks at it."""

    def __init__(self, code: int = 200, body: bytes = b"", **headers):
        self.status_code, self.content = code, body
        self.headers = {"Content-Length": str(len(body)), **{k.replace("_", "-"): v
                                                             for k, v in headers.items()}}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def slow(*args, **kwargs):
    raise requests.Timeout("too slow")


def gone(*args, **kwargs):
    raise requests.ConnectionError("no line")


def play(steps: list, **kwargs):
    """Run `_get` against a list of answers, returns (result or error text, waits, steps left)."""
    waits = []
    M._session = lambda: SimpleNamespace(get=lambda *a, **k: (
        step(*a, **k) if callable(step := steps.pop(0)) else step))
    M.time = SimpleNamespace(sleep=waits.append, perf_counter=time.perf_counter)
    try:
        out = M._get("http://example.invalid/x", **kwargs)[0]
    except OSError as err:
        out = err
    M._session, M.time = real_session, time
    return out, waits, steps


real_session = M._session
full = Answer(200, b"1234567890")
assert play([full]) == (b"1234567890", [], [])
before = dict(M.STATS)
assert play([slow, gone, Answer(503), Answer(408), full]) == (b"1234567890",
                                                               [0.01, 0.02, 0.04, 0.08], [])
counted = {k: M.STATS[k] - before[k] for k in M.STATS}
assert (counted["requests"], counted["bytes"], counted["retries"]) == (1, 10, 4)
assert abs(counted["wait_seconds"] - 0.15) < 1e-9 and counted["seconds"] > 0
assert counted["footer_requests"] == counted["row_requests"] == 0     # neither footer nor row
before = dict(M.STATS)
out, waits, left = play([slow] * 9)
assert isinstance(out, ConnectionError) and "Timeout" in str(out) and "5 tries" in str(out)
assert waits == [0.01, 0.02, 0.04, 0.08] and len(left) == 9 - M.RETRIES
# a read that failed in the end is no request, but its tries and waits are counted
assert M.STATS["requests"] == before["requests"] and M.STATS["bytes"] == before["bytes"]
assert M.STATS["retries"] - before["retries"] == 4
assert abs(M.STATS["wait_seconds"] - before["wait_seconds"] - 0.15) < 1e-9
# a body that is shorter than the answer says is tried again, never handed on
short = Answer(200, b"12345", Content_Length="10")
assert play([short, full]) == (b"1234567890", [0.01], [])
out, _, _ = play([short] * 5)
assert isinstance(out, ConnectionError) and "cut answer" in str(out)
# the wait that the server names is kept, up to MAX_WAIT_S
assert play([Answer(429, RateLimit='"resolvers";r=0;t=7'), full])[1] == [7.0]
assert play([Answer(429, Retry_After="3"), full])[1] == [3.0]
assert play([Answer(429, RateLimit='"resolvers";r=0;t=100000'), full])[1] == [M.MAX_WAIT_S]
assert play([Answer(429), Answer(429, Retry_After="soon"), full])[1] == [0.01, 0.02]
# other answers stop at once, and a range must come back as a range
for code, kind in ((404, FileNotFoundError), (401, PermissionError), (403, PermissionError),
                   (400, OSError), (206, OSError)):
    out, waits, left = play([Answer(code), full])
    assert type(out) is kind and waits == [] and left == [full], code
out, waits, left = play([Answer(200, b"the whole file"), full], byte_range="bytes=0-3")
assert type(out) is OSError and "206" in str(out) and left == [full]
assert play([Answer(206, b"1234")], byte_range="bytes=0-3")[0] == b"1234"

# ------------------------------------------------------------------ over HTTP, on this machine

HITS = []              # (path, headers) of every request
AWAY = []              # the same for the second server, which stands for another host
PLAN = {}              # path -> answers to give before the file is served ("503", "429", "cut")
LISTING = {"empty": False}
FAKE = "not-a-real-token"


class Handler(http.server.BaseHTTPRequestHandler):
    seen = HITS

    def log_message(self, *args):
        pass

    def plain(self, code: int, headers: dict | None = None, body: bytes = b"") -> None:
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve(self, name: str, follow_range: bool = True, cut: bool = False) -> None:
        file = DIR / name.rpartition("/")[2]
        if not file.is_file():
            return self.plain(404)
        data = file.read_bytes()
        asked = self.headers.get("Range")
        if asked and follow_range:
            a, _, b = asked.removeprefix("bytes=").partition("-")
            start = max(0, len(data) - int(b)) if a == "" else int(a)
            end = len(data) if a == "" else min(int(b) + 1, len(data))
            body, code = data[start:end], 206
            headers = {"Content-Range": f"bytes {start}-{end - 1}/{len(data)}"}
        else:
            body, code, headers = data, 200, {}
        self.send_response(code)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body[:len(body) // 2] if cut else body)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        self.seen.append((path, dict(self.headers)))
        kind, _, name = path.lstrip("/").partition("/")
        todo = PLAN.get(path) or []
        step = todo.pop(0) if todo else ""
        if kind == "slow":
            time.sleep(0.3)
        if kind == "down" or step == "503":
            self.plain(503)
        elif kind == "moved":                  # as the Hub does, the file is on another host
            self.plain(302, {"Location": f"{ELSEWHERE}/files/{name}"})
        elif step == "429":
            self.plain(429, {"RateLimit": '"resolvers";r=0;t=1'})
        elif kind == "missing":
            self.plain(404)
        elif kind == "secret":
            self.plain(401)
        elif kind == "nosize":
            self.plain(206, body=b"PAR1")
        elif kind == "api" and LISTING["empty"]:
            self.plain(200, body=b"[]")
        elif kind == "api" and query == "":
            page = [{"type": "file", "path": "images_249k/s2_b.parquet"},
                    {"type": "directory", "path": "images_249k/old"},
                    {"type": "file", "path": "images_249k/notes.txt"}]
            link = f'<{BASE}{path}?cursor=2>; rel="next"'
            self.plain(200, {"Link": link}, json.dumps(page).encode())
        elif kind == "api":
            page = [{"type": "file", "path": "images_249k/s2_a.parquet"}]
            self.plain(200, body=json.dumps(page).encode())
        else:
            self.serve(name, follow_range=kind != "norange", cut=step == "cut")


class Away(Handler):
    seen = AWAY


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{server.server_address[1]}"
away = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Away)
threading.Thread(target=away.serve_forever, daemon=True).start()
ELSEWHERE = f"http://127.0.0.1:{away.server_address[1]}"
# From here on a netrc file names a login for this machine. No request may carry it.
(DIR / "netrc").write_text("machine 127.0.0.1 login somebody password not-a-real-password\n")
os.environ["NETRC"] = str(DIR / "netrc")


def hits(path: str = "") -> list:
    return [h for p, h in HITS if not path or p == path]


# the same values as from the file, fetched by byte ranges only
fresh()
before = dict(M.STATS)
asked = ["grid_cell", "B02", "B04"]
assert M.read_row(f"{BASE}/files/s2_a.parquet", 1, asked) == M.read_row(A, 1, asked)
assert len(hits()) == 2 and all("Range" in h for h in hits())         # footer, then the row
assert M.STATS["requests"] - before["requests"] == 2 + 2     # two over HTTP, two from the file
assert hits()[0]["Range"] == f"bytes=-{M.FOOTER_GUESS}"
assert all(h["Accept-Encoding"] == "identity" for h in hits())     # bytes as they are in the file
assert not any("Authorization" in h for h in hits())    # no token, no header, netrc not read

# a footer that is longer than the first guess takes a second request, not more
fresh()
HITS.clear()
M.FOOTER_GUESS = 64
before = dict(M.STATS)
md, size = M._footer(f"{BASE}/files/s2_a.parquet")
assert len(hits()) == 2 and size == os.path.getsize(A) and md.num_row_groups == 4
assert md.row_group(3).column(1).statistics.min == S2_ROWS[3]["grid_cell"]
paid = {k: M.STATS[k] - before[k] for k in M.STATS}
assert (paid["footers"], paid["footer_requests"], paid["footer_bytes"]) == (1, 2, need)
assert (paid["requests"], paid["bytes"], paid["row_requests"]) == (2, need, 0)
assert 0 < paid["footer_seconds"] == paid["seconds"]
fresh()

# a cell of a shard that was not read before pays the footer, the next cell of that shard
# pays its row only. This is what makes a sample dearer per cell than a full run
HITS.clear()
one = cost(M.read_s2_cell, row_of(f"{BASE}/files/s2_a.parquet", S2_ROWS[0], 0))
two = cost(M.read_s2_cell, row_of(f"{BASE}/files/s2_a.parquet", S2_ROWS[1], 1))
assert (one["footers"], two["footers"]) == (1, 0) and len(hits()) == 3
assert one["footer_bytes"] == min(M.FOOTER_GUESS, os.path.getsize(A)) and two["footer_bytes"] == 0
assert one["bytes"] == one["footer_bytes"] + one["row_bytes"] and two["bytes"] == two["row_bytes"]
assert one["row_bytes"] > 0 and two["row_bytes"] > 0 and one["decoded"] == two["decoded"] == 13
fresh()

# threads that start on the same shard at the same time fetch its footer ONCE
HITS.clear()
with ThreadPoolExecutor(8) as pool:
    got = list(pool.map(lambda i: M.read_row(f"{BASE}/slow/s2_a.parquet", i % 4, ["grid_cell"]),
                        range(8)))
assert [g["grid_cell"] for g in got] == [S2_ROWS[i % 4]["grid_cell"] for i in range(8)]
assert [h["Range"].startswith("bytes=-") for h in hits()].count(True) == 1
assert len(hits()) == 1 + 8 and M._FETCHING == {}
# when the fetch fails, every thread gets the error and nothing is left locked
fresh()
HITS.clear()
with ThreadPoolExecutor(4) as pool:
    jobs = [pool.submit(M.read_row, f"{BASE}/missing/s2_a.parquet", 0) for _ in range(4)]
assert all(type(j.exception()) is FileNotFoundError for j in jobs) and M._FETCHING == {}
fresh()

# the index over HTTP equals the index from the files
HITS.clear()
urls = [f"{BASE}/files/{n}" for n in ("s2_a.parquet", "s2_b.parquet", "s2_c.parquet")]
over_http = M.build_s2_index(shards=urls, workers=3, progress=False, verify=3)
for column in (*M.META_COLUMNS, "parquet_row"):
    assert over_http[column].tolist() == index[column].tolist(), column
assert over_http["parquet_url"].tolist() == [urls[0]] * 4 + [urls[1]] * 2 + [urls[2]] * 2
assert over_http.attrs["read_by_columns"] == urls[1:]
assert len(hits("/files/s2_a.parquet")) == 1 + 3         # the footer, then 3 rows read back

# network trouble is tried again, with a wait, and the wait the server asks for is kept
fresh()
HITS.clear()
PLAN["/flaky/s2_a.parquet"] = ["503", "cut", "429"]
start = time.perf_counter()
before = dict(M.STATS)
got = M.read_row(f"{BASE}/flaky/s2_a.parquet", 0, ["product_id"])
assert got["product_id"] == S2_ROWS[0]["product_id"]
assert time.perf_counter() - start >= 1.0                 # the 429 asked for one second
assert len(hits()) == 3 + 1 + 1                           # three failures, the footer, the row
assert M.STATS["retries"] - before["retries"] == 3        # the waits are counted, apart from
assert M.STATS["wait_seconds"] - before["wait_seconds"] >= 1.0        # the seconds of the reads
assert M.STATS["requests"] - before["requests"] == 2

fresh()
HITS.clear()
text = fails(ConnectionError, M.read_row, f"{BASE}/down/s2_a.parquet", 0)
assert len(hits()) == M.RETRIES and "HTTP 503" in text and f"{M.RETRIES} tries" in text
HITS.clear()
PLAN["/flaky/s2_b.parquet"] = ["cut"] * M.RETRIES
text = fails(ConnectionError, M.read_row, f"{BASE}/flaky/s2_b.parquet", 0)
assert len(hits()) == M.RETRIES and f"{M.RETRIES} tries" in text

# answers that another try cannot change stop at once
for kind, error in (("missing", FileNotFoundError), ("secret", PermissionError)):
    HITS.clear()
    fails(error, M.read_row, f"{BASE}/{kind}/s2_a.parquet", 0)
    assert len(hits()) == 1, kind
HITS.clear()
assert "206" in fails(OSError, M.read_row, f"{BASE}/norange/s2_a.parquet", 0)      # range ignored
assert len(hits()) == 1
assert "file size" in fails(OSError, M.read_row, f"{BASE}/nosize/s2_a.parquet", 0)

# a token goes to addresses under HF only, and never into an error text
fresh()
M.HF = f"{BASE}/hub"
HITS.clear()
M.read_row(f"{BASE}/hub/s2_a.parquet", 0, ["grid_cell"], token=FAKE)
assert len(hits()) == 2 and all(h.get("Authorization") == f"Bearer {FAKE}" for h in hits())
for other in ("files", "hubx"):                           # hubx only starts like HF
    HITS.clear()
    M.read_row(f"{BASE}/{other}/s2_a.parquet", 0, ["grid_cell"], token=FAKE)
    assert len(hits()) == 2 and not any("Authorization" in h for h in hits()), other
# the Hub sends a read on to another host. The range goes along, the token does not
fresh()
M.HF = BASE
HITS.clear()
got = M.read_row(f"{BASE}/moved/s2_a.parquet", 0, ["grid_cell"], token=FAKE)
assert got["grid_cell"] == S2_ROWS[0]["grid_cell"]
assert len(hits()) == 2 and all(h.get("Authorization") == f"Bearer {FAKE}" for h in hits())
assert len(AWAY) == 2 and all("Range" in h for _, h in AWAY)
assert not any("Authorization" in h for _, h in AWAY)
for kind, error in (("missing", FileNotFoundError), ("secret", PermissionError),
                    ("down", ConnectionError), ("norange", OSError)):
    M.HF = f"{BASE}/{kind}"
    assert FAKE not in fails(error, M.read_row, f"{BASE}/{kind}/s2_a.parquet", 0, None, FAKE)

# the list of shards, over two pages, and the two indexes from the repository addresses
fresh()
M.HF = BASE
HITS.clear()
listed = M.list_s2_shards()
assert listed == [f"{BASE}/datasets/{M.S2_REPO}/resolve/main/images_249k/{n}"
                  for n in ("s2_a.parquet", "s2_b.parquet")]
assert [p for p, _ in HITS] == [f"/api/datasets/{M.S2_REPO}/tree/main/images_249k"] * 2
from_list = M.build_s2_index(progress=False)
assert from_list["grid_cell"].tolist() == index["grid_cell"].tolist()[:6]
assert from_list["parquet_url"].tolist() == [listed[0]] * 4 + [listed[1]] * 2
LISTING["empty"] = True
fails(FileNotFoundError, M.list_s2_shards)
LISTING["empty"] = False
HITS.clear()
assert M.load_s1_index().to_dict("list") == s1_index.to_dict("list")
assert [p for p, _ in HITS] == [f"/datasets/{M.S1_REPO}/resolve/main/metadata.parquet"]
assert "Range" not in hits()[0]
fresh()
server.shutdown()
away.shutdown()
del os.environ["NETRC"]

# ------------------------------------------------------------------ real rows, only when allowed

if _setup.NET:
    M.BACKOFF_S = 1.0
    at_start = dict(M.STATS)
    shards = M.list_s2_shards()
    assert len(shards) == 498 and shards[249].endswith("images_249k/part_00250.parquet")
    before = dict(M.STATS)
    part = M.build_s2_index(shards=shards[249:250], workers=1, progress=False, verify=2)
    assert len(part) == 500 and part.attrs["read_by_columns"] == []
    assert part["grid_cell"][231] == "213U_978L"
    paid = {k: M.STATS[k] - before[k] for k in M.STATS}
    assert (paid["footers"], paid["footer_requests"], paid["row_requests"]) == (1, 2, 2)
    assert 2e6 < paid["footer_bytes"] < 6e6 and paid["row_bytes"] < 1e4    # a footer of 2 to 5 MB
    before = dict(M.STATS)
    s2 = M.read_s2_cell(part.iloc[231])      # fails when the cloud mask is not on the grid of B02
    paid = {k: M.STATS[k] - before[k] for k in M.STATS}
    assert (paid["footers"], paid["footer_bytes"], paid["row_requests"]) == (0, 0, 1)
    assert 0.5e6 < paid["row_bytes"] < 3e6 and paid["decoded"] == 13
    print(f"Sentinel-2 cell, row {paid['row_bytes'] / 1e6:.2f} MB in {paid['row_seconds']:.2f} s, "
          f"decoding {paid['decode_seconds']:.2f} s, footer in memory")
    assert [s2.bands[b].shape[0] for b in ("B02", "B05", "B01")] == [384, 192, 64]
    assert s2.cloud_mask.shape == (384, 384)
    assert s2.datetime == pd.Timestamp("2019-08-11 16:28:39", tz="UTC")
    before = dict(M.STATS)
    s1 = M.read_s1_cell({
        "grid_cell": "213U_978L", "parquet_row": 71, "product_datetime": None,
        "product_id": "S1A_IW_GRDH_1SDV_20190811T120131_20190811T120200_028521_033984_rtc",
        "parquet_url": f"{M.HF}/datasets/{M.S1_REPO}/resolve/main/images/part_01344.parquet"})
    paid = {k: M.STATS[k] - before[k] for k in M.STATS}
    assert (paid["footers"], paid["footer_requests"], paid["row_requests"]) == (1, 1, 1)
    assert paid["footer_bytes"] == M.FOOTER_GUESS and 5e6 < paid["row_bytes"] < 20e6
    print(f"Sentinel-1 cell, row {paid['row_bytes'] / 1e6:.2f} MB in {paid['row_seconds']:.2f} s, "
          f"footer {paid['footer_bytes'] / 1e6:.2f} MB in {paid['footer_seconds']:.2f} s, "
          f"decoding {paid['decode_seconds']:.2f} s")
    assert s1.vv.shape == s1.vh.shape == (1068, 1068) and s1.nodata == -32768.0
    assert s1.crs == s2.crs == "EPSG:32615" and s1.transform[0] == s2.transforms["B02"][0] == 10.0
    assert s1.datetime == pd.Timestamp("2019-08-11 12:01:31", tz="UTC")
    # the Sentinel-2 crop is the middle of the Sentinel-1 cell, on the same 10 m grid
    assert (s2.transforms["B02"][2] - s1.transform[2]) / 10 == 342.0
    assert (s1.transform[5] - s2.transforms["B02"][5]) / 10 == 342.0
    print(f"network checks done, {(M.STATS['bytes'] - at_start['bytes']) / 1e6:.1f} MB read in "
          f"{M.STATS['requests'] - at_start['requests']} requests, "
          f"{M.STATS['retries'] - at_start['retries']} retries")
else:
    print("network checks skipped (SARTRANSFER_TEST_NET is not 1)")

shutil.rmtree(DIR, ignore_errors=True)
print("test_majortom: all checks passed")
