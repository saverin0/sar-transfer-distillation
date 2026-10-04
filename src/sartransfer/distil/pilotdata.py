"""The pilot data of stage A. Major-TOM pairs in archives, their infos, in RAM.

Steps, in the order the data notebook calls them
- `pilot_sample`        26,000 of the 150,309 pairs (seed 0), spread over the years as in
                        stage 0, in the order of the shards, cut into chunks of 2,000 rows.
                        26,000 cells are drawn for the 20,000 PAIRS,
                        the arithmetic is at `N_PILOT`
- `fetch_pilot`         the chunks one after the other through `stage0.fetch_pairs`, each one
                        packed into a tar on the local disk and copied into the archive folder
                        (Drive) as the stage 0 notebook did, with a JSON and an infos table
                        (`INFO_COLUMNS`, one row per pair) next to it. Resumable per chunk.
                        The local pair files of a chunk are deleted once its archive is
                        complete, so the disk holds one chunk at a time. An archive is never
                        unpacked, a pair is read from it in place (`open_pair`)
- `load_infos`          the infos tables of the chunks, built from a tar when a chunk has none
- `split_held_out`      2,000 pairs (seed 0) for the distillation loss, the rest for training
- `channel_stats`       mean and SD per channel over the training pairs, saved once
- `load_pairs_to_ram`   the pairs as two arrays, Sentinel-2 on the 10 m grid, read in
                        `RAM_WORKERS` threads archive by archive, the MB per second printed
- `ChunkReader`         the same arrays for several lists at once, filled one archive at a
                        time as the archives arrive, so the data notebook reads an archive
                        while the next ones are still being fetched; `pair_locations` finds
                        every pair from the chunk JSONs before any archive is on the disk
- `pilot_summary`       one table of a pilot, printed

Where a pair is. A location is a text "<tar path>::<member name>" for a pair inside an
archive, or a plain file path. `open_pair` reads either, `pair_cell` gives the grid cell,
`split_location` takes a location apart. A tar member is read through its own file handle
at the offset the tar index gives (`tar_index`, scanned once per tar per process, or taken
from the infos table), so many threads read one archive without unpacking it. Read bytes are
the bytes of the pair file, so a pair from a tar member equals the pair from the file.

Rules. Cloud = `records.CLOUD_CODES` over 10 %, the offset by processing baseline, stored at
384 px with the native bands (`pairing.save_pair`). Every rule is applied ONCE, in
`pairing.make_pair`, when a pair is fetched. A pair that comes back from an archive is taken
as it is. A folder or an archive that was made under another rule is refused (ValueError),
it is never corrected and never read on.

The band shift of the 20 m and 60 m bands is corrected by the corner
rule when the bands are put on the 10 m grid, in `s2_grid`, through the keywords corner and
crs of `pairing.s2_to_grid`. An installed `s2_to_grid` without these keywords is called
without them, and one line says that the correction is not applied.

Why shard by shard. `majortom` keeps 8 footers in memory (`majortom.FOOTER_CACHE`) and a
Sentinel-2 footer is 2 to 5 MB. Stage 0 read 500 cells in the order of the years and paid a
footer for nearly every cell (5 of 16.4 MB per cell). Sorted by shard a footer is paid about
once per shard.

Units. Megabytes are 10^6 bytes and GB are 10^9 bytes, as in stage0. Times are seconds.
The modules majortom, pairing and stage0 are imported inside the functions that need them.
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
import shutil
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .records import CLOUD_CODES, CROP_PX, S1_CHANNELS, S2_BANDS, SKIP_REASONS, Pair

N_PAIRS = 20_000             # pairs of the pilot
# Cells drawn for about N_PAIRS pairs. A drawn cell becomes a pair only when it passes the pairing
# rules, so more cells than pairs are drawn. The share comes from the stage 0 run of
# 2026-09-30 (500 cells). 481 of 500 passed the stage 0 rules (17 crs_differs, 2
# s1_nodata_in_crop). Of these 481 crops 48 hold a Sentinel-2 zero (s2_nodata_in_crop) and 43
# are over 10 % cloud with the codes 1, 2, 3 (cloud_gt_10pct), 3 of them both, so 88 more are
# skipped under the rules of stage A. 393 of 500 = 78.6 % become a pair. 26,000 x 0.786 = 20,436
# pairs, 13 chunks of CHUNK. 20,000 cells would give about 15,700 pairs. The share is an
# estimate from 500 cells, the real count is printed by the fetch.
N_PILOT = 26_000             # cells drawn, about N_PAIRS pairs after the rules (13 chunks)
N_HELD_OUT = 2_000           # pairs held out for the distillation loss
N_STATS = 2_000              # pairs the channel statistics are computed over
SEED = 0
CHUNK = 2_000                # consecutive rows per chunk, one tar each
# Threads of the fetch. The read token is sent with every Major-TOM read, and the Hugging Face
# limit with a token was read as 5,000 requests per 5 minutes (3,000 anonymous). The run of
# 2026-09-30 at 16 threads made about 2,400 requests per 5 minutes (about 8 per second, 4
# cells per second, 470 to 576 s per chunk of 2,000 cells) without a wait. 28 threads aim at
# about 14 per second, still under the limit. The fetch prints the requests per second and
# the waits per chunk (majortom.STATS wait_seconds and retries), so the real limit shows.
WORKERS = 28
RAM_LIMIT_GB = 120.0         # `load_pairs_to_ram` refuses above this (High-RAM printed 179 GB)
# Threads of `load_pairs_to_ram`. The pairs are read in the order of the archives (13 for the
# pilot) and of their members, so the threads read one archive at a time at neighbouring
# offsets. Each read waits for the disk (Drive through its mount, or the runtime disk) and then
# puts 12 bands on the 10 m grid, numpy work that releases the GIL, so more threads than the 12
# CPUs of the A100 runtime keep the CPUs busy while others wait for bytes. 16 is a guess, not
# measured, the loader prints the MB and the pairs per second so the next run can set it. Part
# three's data notebook read its pairs with 16 threads at 213 MB/s in 3.7 min (A100 runtime);
# whether more threads would help was not measured. `ChunkReader` uses the same count per archive.
RAM_WORKERS = 16
FETCH_ORDER = ("s2_url", "s1_url", "s2_row")     # consecutive rows share the shards
MB = 1e6
GB = 1e9
PAIR_BYTES = (len(S1_CHANNELS) + len(S2_BANDS)) * CROP_PX * CROP_PX * 2   # float16 + uint16 in RAM
TRAIN_FILE = "train_cells.txt"
HELD_OUT_FILE = "held_out_cells.txt"
AERIAL_CHANNELS = ("R", "G", "B", "NIR")
OFFSET_HOW = {"baseline": ("baseline", "per_image"), None: ("none",)}   # what a pair on disk says
SEP = "::"                   # "<tar path>::<member name>", the location of a pair in an archive

TIMING_COLUMNS = ("chunk", "source", "cells", "saved", "skipped", "wall_s", "footers",
                  "mb_footers", "mb_rows", "requests", "wait_s", "retries")
SKIP_COLUMNS = ("chunk", "grid_cell", "reason", "detail")
META_COLUMNS = ("grid_cell", "s2_datetime", "s1_datetime", "gap_days", "crs", "baseline",
                "offset_removed", "offset_how", "cloud_fraction", "s2_product_id")
# One row per pair of a chunk, chunk_<k>_infos.csv next to the archive. What `pairing.load_pair_info`
# gives (the info entry of the pair file, no array), the rules the pair was made under, and where
# the pair is in the tar (the member name, the offset of its bytes and their count).
INFO_COLUMNS = ("chunk", "file", "offset", "bytes", *META_COLUMNS, "s1_product_id", "cloud_rule",
                "cloud_codes")
INFO_TEXT = ("file", "grid_cell", "crs", "baseline", "offset_how", "s2_product_id", "s1_product_id",
             "cloud_rule", "cloud_codes")
SUMMARY_COLUMNS = ("kind", "what", "pairs", "share")

NO_CORNER_RULE = ("the installed pairing.s2_to_grid takes no corner and crs, the band shift "
                  "correction (the corner rule) is NOT applied")
ZERO_NOTE = ("no pixel is left out. A stored 0 (the L2A no-data value) does not occur, the "
             "pairing rule s2_nodata_in_crop skipped every crop with one. A 0 here is a real "
             "pixel, a stored value of 1 to 1000 in an image that carried the offset, clipped "
             "at 0 when the offset was taken off. zero_pixels counts them per band")

_SAID: set[str] = set()
_SAY_LOCK = threading.Lock()
_INDEX: dict[tuple, dict[str, tuple[int, int]]] = {}    # (tar, bytes, mtime) -> member -> (offset, bytes)
_INDEX_LOCK = threading.Lock()


def _say_once(text: str) -> None:
    """Print `text` once per process, also from many threads."""
    with _SAY_LOCK:
        if text in _SAID:
            return
        _SAID.add(text)
    print(text)


# --------------------------------------------------------------------------- the sample

def pilot_sample(pairs: pd.DataFrame, n: int = N_PILOT, seed: int = SEED) -> pd.DataFrame:
    """`n` pairs drawn as in stage 0, in the order of the shards, with a column "chunk".

    The draw is `pairing.sample_pairs` (spread over s2_year, the same rows for the same
    seed). The rows are then sorted by s2_url, s1_url and s2_row (stable), so that
    consecutive cells share the Sentinel-2 shard and then the Sentinel-1 shard, and
    "chunk" numbers groups of `CHUNK` consecutive rows from 0. The columns of
    `pairing.PAIR_COLUMNS` plus chunk.
    """
    from . import pairing

    out = pairing.sample_pairs(pairs, n=n, seed=seed)
    out = out.sort_values(list(FETCH_ORDER), kind="stable").reset_index(drop=True)
    out["chunk"] = np.arange(len(out)) // int(CHUNK)
    return out


# --------------------------------------------------------------------------- where a pair is

def pair_location(tar: str | os.PathLike, member: str) -> str:
    """The location text of a pair inside an archive, the tar path, `SEP` and the member name."""
    return f"{Path(tar)}{SEP}{member}"


def split_location(location: str | os.PathLike) -> tuple[Path, str | None]:
    """The tar path and member name of a location in an archive, the file path and None of a plain file."""
    if isinstance(location, os.PathLike):
        return Path(location), None
    text = str(location)
    if SEP not in text:
        return Path(text), None
    tar, _, member = text.rpartition(SEP)
    if not tar or not member:
        raise ValueError(f"{text!r} is not a location, expected <tar path>{SEP}<member name>")
    return Path(tar), member


def pair_cell(location: str | os.PathLike) -> str:
    """The grid cell of a location, the stem of the file name."""
    path, member = split_location(location)
    return Path(member).stem if member is not None else path.stem


def _stat_key(tar: Path) -> tuple:
    st = tar.stat()
    return (os.path.abspath(tar), st.st_size, st.st_mtime_ns)


def _scan_tar(tar: Path) -> dict[str, tuple[int, int]]:
    """member name -> (offset of its bytes in the tar, bytes), from the tar headers."""
    with tarfile.open(tar) as archive:
        return {m.name: (int(m.offset_data), int(m.size)) for m in archive.getmembers() if m.isfile()}


def tar_index(tar: str | os.PathLike) -> dict[str, tuple[int, int]]:
    """The members of a tar with the offset and the bytes of each, scanned once per process.

    The index is kept by the tar's path, size and change time, so a tar written again is
    scanned again. `_prime_index` puts the index of an infos table there without a scan.
    """
    tar = Path(tar)
    key = _stat_key(tar)
    with _INDEX_LOCK:
        found = _INDEX.get(key)
    if found is not None:
        return found
    index = _scan_tar(tar)
    with _INDEX_LOCK:
        _INDEX[key] = index
    return index


def _prime_index(tar: Path, infos: pd.DataFrame) -> None:
    """The index of `tar` from the file, offset and bytes columns of its infos table."""
    index = {str(f): (int(o), int(b)) for f, o, b in zip(infos["file"], infos["offset"], infos["bytes"])}
    with _INDEX_LOCK:
        _INDEX[_stat_key(tar)] = index


def _member_at(tar: Path, member: str) -> tuple[int, int]:
    index = tar_index(tar)
    if member not in index:
        raise FileNotFoundError(f"{tar} does not hold {member}, delete it and its JSON, the chunk "
                                "is then fetched again")
    return index[member]


class _Window(io.RawIOBase):
    """A read-only view of `size` bytes of a file from `offset`, with its own file handle.

    A tar member as a file object that can seek, so np.load reads one entry of a pair file
    (the info) without the rest. One window per read, never shared between threads.
    """

    def __init__(self, path: Path, offset: int, size: int) -> None:
        super().__init__()
        self._file = open(path, "rb")
        self._start, self._size, self._pos = int(offset), int(size), 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, pos: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: self._size}[whence]
        self._pos = max(0, base + int(pos))
        return self._pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self._size - self._pos)
        if n <= 0:
            return 0
        self._file.seek(self._start + self._pos)
        data = self._file.read(n)
        buffer[:len(data)] = data
        self._pos += len(data)
        return len(data)

    def close(self) -> None:
        if not self.closed:
            self._file.close()
        super().close()


def member_bytes(location: str | os.PathLike) -> bytes:
    """The bytes of a pair at `location`, the pair file itself, from a tar or from a plain file."""
    path, member = split_location(location)
    if member is None:
        return path.read_bytes()
    offset, size = _member_at(path, member)
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(size)
    if len(data) != size:
        raise OSError(f"{path} is cut, {member} has {len(data)} of {size} bytes")
    return data


def _pair_of(z: Any, where: Any) -> Pair:
    """A `Pair` from an open pair file, as `pairing.load_pair` builds it."""
    from . import pairing

    info = pairing._read_info(z, where)
    return Pair(grid_cell=info["grid_cell"], s1_db=z["s1_db"],
                s2={name: z[f"s2_{name}"] for name in info["s2_bands"]},
                cloud_mask=z["cloud_mask"] if "cloud_mask" in z.files else None,
                s2_datetime=info["s2_datetime"], s1_datetime=info["s1_datetime"],
                gap_days=info["gap_days"], crs=info["crs"], bounds=info["bounds"], meta=info["meta"])


def open_pair(location: str | os.PathLike) -> Pair:
    """The pair at `location`, a plain file (`pairing.load_pair`) or a tar member read in place.

    A tar member is read in one piece through its own file handle, so threads can read one
    archive at once. The bytes are those of the pair file, the pair is the same.
    """
    from . import pairing

    path, member = split_location(location)
    if member is None:
        return pairing.load_pair(path)
    with np.load(io.BytesIO(member_bytes(location)), allow_pickle=False) as z:
        return _pair_of(z, location)


def pair_info_at(location: str | os.PathLike) -> dict:
    """The info of the pair at `location` (`pairing.load_pair_info`), from a tar only the info entry."""
    from . import pairing

    path, member = split_location(location)
    if member is None:
        return pairing.load_pair_info(path)
    offset, size = _member_at(path, member)
    with _Window(path, offset, size) as window, np.load(window, allow_pickle=False) as z:
        return pairing._read_info(z, location)


def _by_tar(locations: Sequence) -> list[int]:
    """The positions of `locations` in reading order, by tar and then by offset in it."""
    keys = []
    for i, location in enumerate(locations):
        path, member = split_location(location)
        offset = _member_at(path, member)[0] if member is not None else 0
        keys.append((str(path), offset, i))
    return [i for _, _, i in sorted(keys)]


# --------------------------------------------------------------------------- fetch per chunk

def _cloud_codes(cloud_codes: Iterable[int] | None) -> list[int] | None:
    """The cloud codes as a sorted list without doubles, None = the rule is off.

    Whole numbers from 0 to 255 and at least one, as `stage0.fetch_pairs` asks for. Checked
    here before an archive is compared with them, so a wrong list never matches one.
    """
    if cloud_codes is None:
        return None
    problem = f"cloud_codes {cloud_codes!r} must be whole numbers from 0 to 255, at least one"
    if isinstance(cloud_codes, (str, bytes)):
        raise ValueError(problem)
    try:
        given = list(cloud_codes)
        codes = sorted({int(c) for c in given})
        whole = all(float(c) == int(c) for c in given)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(problem) from None
    if not codes or not whole or codes[0] < 0 or codes[-1] > 255:
        raise ValueError(problem)
    return codes


def _chunk_files(archive_dir: Path, k: int) -> tuple[Path, Path]:
    """The tar and the JSON of chunk `k`."""
    return archive_dir / f"chunk_{k}.tar", archive_dir / f"chunk_{k}.json"


def infos_file(archive_dir: str | os.PathLike, k: int) -> Path:
    """The infos table of chunk `k`, chunk_<k>_infos.csv next to its tar."""
    return Path(archive_dir) / f"chunk_{k}_infos.csv"


def _archive_info(tar: Path, info: Path) -> dict | None:
    """The JSON of a complete archive, None when the tar or the JSON is missing or cut.

    Complete = both files exist, the JSON can be read and names a file list, and the tar
    has exactly the bytes the JSON says. The infos table is checked apart (`chunk_infos`),
    a missing one is built from the tar, nothing else is done again.
    """
    if not tar.is_file() or not info.is_file():
        return None
    try:
        found = json.loads(info.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(found, dict) or not isinstance(found.get("files"), list):
        return None
    if tar.stat().st_size != found.get("bytes"):
        return None
    return found


def _write_json(path: Path, value: Mapping) -> None:
    """Write next to the target first, then rename, so a stopped write leaves no half file."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(value, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_tar(tar: Path, k: int, paths: Sequence[Path], work_dir: Path
               ) -> tuple[int, dict[str, tuple[int, int]]]:
    """The pair files of chunk `k` into `tar` as chunk_<k>/<name>, built on the local disk first.

    The way the stage 0 notebook wrote its archive. The tar is built as <work_dir>/<name of
    tar> (no compression, the pair files are not compressed either), copied next to `tar` as
    <name of tar>.part, then renamed to `tar`. So nothing ever sits under the final name
    before it is whole, a stopped run leaves at most the .part and the local tar, and both
    are removed here, also when the copy fails. A copy that does not have the bytes of the
    tar is an OSError. Returns the bytes of the tar and its index (member name -> offset,
    bytes), scanned from the local tar before it is deleted.
    """
    local, part = work_dir / tar.name, tar.with_name(tar.name + ".part")
    try:
        local.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(local, "w") as archive:
            for path in paths:
                archive.add(path, arcname=f"chunk_{k}/{path.name}")
        size = local.stat().st_size
        index = _scan_tar(local)
        shutil.copyfile(local, part)
        copied = part.stat().st_size
        if copied != size:
            raise OSError(f"{part} has {copied} bytes after the copy, the tar has {size}")
        os.replace(part, tar)
    finally:
        for path in (local, part):
            path.unlink(missing_ok=True)
    return size, index


def _check_meta(where: Any, meta: Mapping, cloud: tuple, want: tuple, folder_word: str) -> None:
    """The rules a pair was made under, from its meta, against the rules asked for."""
    found = (meta.get("cloud_rule"), meta.get("cloud_codes"))
    if found != cloud:
        raise ValueError(f"{where} was made with the cloud rule {found}, asked for is {cloud}. "
                         f"Use another {folder_word} for another rule")
    how = meta.get("offset_how")
    if how not in want:
        why = "" if how is not None else (" (no offset_how entry, made before the offset rule "
                                         "existed)")
        raise ValueError(f"{where} was made with the offset rule {how!r}{why}, asked for is "
                         f"offset={_offset_of(want)!r} ({' or '.join(want)}). Use another "
                         f"{folder_word} for another rule")


def _offset_of(want: tuple) -> str | None:
    return next(rule for rule, hows in OFFSET_HOW.items() if hows == want)


def _check_rules(paths: Iterable[Path], codes: list[int] | None, offset: str | None) -> None:
    """Every pair file of `paths` that exists must have been made under the asked rules.

    The cloud rule as `stage0.fetch_pairs` checks it (`Pair.meta` cloud_rule and cloud_codes)
    and the offset rule through `Pair.meta["offset_how"]`. Only the info of a file is read
    (`pairing.load_pair_info`), not its arrays. A pair of another rule, one without an
    offset_how entry (made before the offset rule existed) or one that cannot be read is a
    ValueError, the pair is never handed on. Use another folder for another rule.
    """
    from . import pairing

    cloud = ("not applied", None) if codes is None else ("applied", list(codes))
    want = OFFSET_HOW[offset]
    for path in paths:
        if not path.is_file():
            continue
        try:
            meta = pairing.load_pair_info(path)["meta"]
        except Exception as err:
            raise ValueError(f"{path} is on disk and cannot be read, {type(err).__name__}, "
                             f"{err}. Delete it, the cell is then read again") from None
        _check_meta(path, meta, cloud, want, "folder")


def _text(value: Any) -> str | None:
    """A csv value as text, None for an empty one."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    return str(value)


def _check_infos(infos: pd.DataFrame, tar: Path, codes: list[int] | None, offset: str | None) -> None:
    """Every pair of an infos table must have been made under the asked rules, as `_check_rules`."""
    cloud = ("not applied", None) if codes is None else ("applied", list(codes))
    want = OFFSET_HOW[offset]
    for row in infos.itertuples(index=False):
        listed = _text(row.cloud_codes)
        meta = {"cloud_rule": _text(row.cloud_rule),
                "cloud_codes": json.loads(listed) if listed else None,
                "offset_how": _text(row.offset_how)}
        _check_meta(f"{row.file} in {tar}", meta, cloud, want, "archive folder")


def _info_row(k: int, member: str, offset: int, size: int, info: Mapping) -> dict:
    """One row of `INFO_COLUMNS` from the info of a pair file."""
    meta = info["meta"]
    codes = meta.get("cloud_codes")
    return {"chunk": int(k), "file": member, "offset": int(offset), "bytes": int(size),
            "grid_cell": info["grid_cell"], "s2_datetime": info["s2_datetime"],
            "s1_datetime": info["s1_datetime"], "gap_days": info["gap_days"], "crs": info["crs"],
            "baseline": meta.get("baseline"), "offset_removed": meta.get("offset_removed"),
            "offset_how": meta.get("offset_how"),
            "cloud_fraction": meta.get("cloud_fraction", float("nan")),
            "s2_product_id": meta.get("s2_product_id"), "s1_product_id": meta.get("s1_product_id"),
            "cloud_rule": meta.get("cloud_rule"),
            "cloud_codes": json.dumps(list(codes)) if codes is not None else None}


def _write_infos(path: Path, infos: pd.DataFrame) -> None:
    """The infos table as csv, written next to the target first and renamed."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        infos[list(INFO_COLUMNS)].to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_infos(path: str | os.PathLike) -> pd.DataFrame:
    """An infos table from its csv, columns `INFO_COLUMNS`, dates as pandas Timestamps in UTC.

    Text columns stay text (an empty one is None), offset_removed is bool. A file without
    the columns is a ValueError.
    """
    df = pd.read_csv(path, dtype={name: str for name in INFO_TEXT}, keep_default_na=True)
    missing = [c for c in INFO_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} has no column {missing}")
    df = df[list(INFO_COLUMNS)].copy()
    for name in INFO_TEXT:
        df[name] = df[name].astype(object).where(df[name].notna(), None)
    for name in ("s2_datetime", "s1_datetime"):
        df[name] = pd.to_datetime(df[name], utc=True)
    df["offset_removed"] = df["offset_removed"].astype(bool)
    for name in ("chunk", "offset", "bytes"):
        df[name] = df[name].astype(np.int64)
    return df


def _infos_match(infos: pd.DataFrame, k: int, members: Sequence[str]) -> bool:
    return (infos["chunk"] == int(k)).all() and list(infos["file"]) == list(members)


def chunk_infos(archive_dir: str | os.PathLike, k: int, workers: int = 8, progress: bool = False,
                found: Mapping | None = None) -> tuple[pd.DataFrame, bool]:
    """The infos table of chunk `k`, read from chunk_<k>_infos.csv, else built from the tar and written.

    The chunk must be complete (tar and JSON, `_archive_info`), else FileNotFoundError. A
    table on disk that does not list the members of the JSON, in their order, is built again.
    The build reads the info entry of every member straight out of the tar, in `workers`
    threads through windows on the tar (nothing is unpacked), and prints how long it took.
    A member the tar does not hold is a RuntimeError. Returns (infos, built), and the index
    of the tar is primed from the table, so no later read scans the tar.
    """
    archive_dir = Path(archive_dir)
    tar, info = _chunk_files(archive_dir, k)
    if found is None:
        found = _archive_info(tar, info)
    if found is None:
        raise FileNotFoundError(f"{tar} is not complete (the tar or its JSON is missing, or the tar "
                                "is cut), run the fetch again")
    members = [f"chunk_{k}/{name}" for name in found["files"]]
    path = infos_file(archive_dir, k)
    if path.is_file():
        try:
            infos = read_infos(path)
        except (OSError, ValueError):
            infos = None
        if infos is not None and _infos_match(infos, k, members):
            _prime_index(tar, infos)
            return infos, False
    start = time.perf_counter()
    index = tar_index(tar)
    missing = [m for m in members if m not in index]
    if missing:
        raise RuntimeError(f"{tar} does not hold {missing[:3]}, delete it and its JSON, the chunk "
                           "is then fetched again")

    def one(member: str) -> dict:
        offset, size = index[member]
        return _info_row(k, member, offset, size, pair_info_at(pair_location(tar, member)))

    rows = _map(one, members, workers, progress, f"infos of chunk {k}")
    infos = pd.DataFrame(rows, columns=list(INFO_COLUMNS))
    _write_infos(path, infos)
    infos = read_infos(path)
    _prime_index(tar, infos)
    print(f"chunk {k}  infos built from {tar.name}, {len(infos):,} pairs, {time.perf_counter() - start:.1f} s, "
          f"written to {path.name}")
    return infos, True


def load_infos(archive_dir: str | os.PathLike, chunks: Iterable[int], workers: int = 8,
               progress: bool = False) -> pd.DataFrame:
    """The infos tables of `chunks` as one table, in chunk order, built where a chunk has none."""
    tables = [chunk_infos(archive_dir, int(k), workers, progress)[0]
              for k in sorted({int(c) for c in chunks})]
    if not tables:
        return pd.DataFrame(columns=list(INFO_COLUMNS))
    return pd.concat(tables, ignore_index=True)


def locations_of(infos: pd.DataFrame, archive_dir: str | os.PathLike) -> list[str]:
    """The location of every pair of an infos table, chunk_<k>.tar under `archive_dir` and the member."""
    archive_dir = Path(archive_dir)
    return [pair_location(archive_dir / f"chunk_{int(k)}.tar", str(f))
            for k, f in zip(infos["chunk"], infos["file"])]


def pair_locations(archive_dir: str | os.PathLike, chunks: Iterable[int]) -> dict[str, str]:
    """The location of every pair of the chunk archives by grid cell, from their JSON files alone.

    The JSON of chunk k names its pair files (<grid cell>.npz) and its archive holds them as
    chunk_<k>/<file>, so every pair has its location before any archive is on the disk (what
    `ChunkReader` needs before the fetch). The locations are those `locations_of` gives from
    the infos tables, in the same order (chunk, then the order of the JSON), and the grid cell
    is the file name without .npz, as `pair_cell` gives it and the lists of `split_held_out`
    hold it. A missing JSON is a FileNotFoundError, one without a file list a ValueError, a
    cell in two archives a ValueError.
    """
    archive_dir = Path(archive_dir)
    out: dict[str, str] = {}
    for k in sorted({int(c) for c in chunks}):
        tar, info = _chunk_files(archive_dir, k)
        found = json.loads(info.read_text(encoding="utf-8"))
        files = found.get("files") if isinstance(found, dict) else None
        if not isinstance(files, list):
            raise ValueError(f"{info} names no file list, rebuild the chunk with notebook 01")
        for name in files:
            location = pair_location(tar, f"chunk_{k}/{name}")
            cell = pair_cell(location)
            if cell in out:
                raise ValueError(f"grid cell {cell} is in two archives, {out[cell]} and {location}")
            out[cell] = location
    return out


def _skips_text(counts: Mapping[str, int]) -> str:
    """"cloud_gt_10pct 60, crs_differs 17" in the order of `records.SKIP_REASONS`."""
    return ", ".join(f"{r} {counts[r]:,}" for r in SKIP_REASONS if counts.get(r))


def _mb(value: Any) -> str:
    return "?" if value is None else f"{value:,.0f} MB"


def _rate_text(totals: Mapping) -> str:
    """"1,234 requests, 8.2 per second, waited 0 s in 0 retries" from the totals of a fetch."""
    requests, wall = totals.get("requests"), totals.get("wall_s")
    if requests is None:
        return "requests not counted"
    rate = f"{requests / wall:.1f}" if wall else "?"
    wait, retries = totals.get("wait_s"), totals.get("retries")
    return (f"{int(requests):,} requests, {rate} per second, waited "
            f"{'?' if wait is None else f'{wait:,.0f}'} s in {'?' if retries is None else int(retries)} retries")


def fetch_pilot(pairs: pd.DataFrame, work_dir: str | os.PathLike, archive_dir: str | os.PathLike,
                workers: int = WORKERS, token: str | None = None,
                cloud_codes: Iterable[int] | None = CLOUD_CODES, offset: str | None = "baseline",
                progress: bool = True) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Fetch the pilot chunk by chunk, each chunk into one tar. Resumable at chunk level.

    pairs        the table of `pilot_sample` (it needs the column chunk)
    work_dir     local folder, the pairs of chunk k are fetched into <work_dir>/chunk_<k>/
                 <grid_cell>.npz and the tar of the chunk is built there while it is copied.
                 Both are deleted once the archive is complete, so the disk holds one chunk
    archive_dir  folder for the archives (Drive), chunk_<k>.tar, chunk_<k>.json and
                 chunk_<k>_infos.csv
    workers      threads of `stage0.fetch_pairs` (`WORKERS`, the reason is written there)
    cloud_codes  the cloud rule, `records.CLOUD_CODES` is the decided list, None = not applied
    offset       the offset rule of `pairing.make_pair`, "baseline" (decided) or None

    Per chunk, in the order of the chunk numbers
    - A complete archive (the tar has the bytes the JSON says) made under the same rules is
      used in place ("archive"). Nothing is fetched and nothing is unpacked. Its infos table
      is read, or built from the tar when it is missing (`chunk_infos`, printed with its
      time), and every pair is checked against the rules from that table.
    - Else the rows of the chunk go through `stage0.fetch_pairs` into <work_dir>/chunk_<k>/
      ("fetched"), then the tar is built on the local disk, copied to archive_dir as
      chunk_<k>.tar.part and renamed (`_write_tar`, as the stage 0 notebook did, the local
      tar is deleted after the copy), then the JSON (files, bytes, skips, the rules, the
      fetch totals) and the infos table, then the local pair files are deleted. A cut or
      half-written archive is therefore not complete and the chunk is done again. Inside a
      chunk `fetch_pairs` resumes at pair level, a pair on disk is not read again.
    - The rules are applied once, by `pairing.make_pair` in the fetch. Every pair that is
      handed on is checked against them from its info (cloud_rule, cloud_codes and
      offset_how of `Pair.meta`), a pair on disk before the fetch through `_check_rules`,
      a pair of an archive through its infos row. A pair of another rule stops the call
      (ValueError), and so does an archive whose JSON names other rules, before anything is
      read. Nothing is fetched in these cases, use another folder for another rule.
    - One line is printed per chunk, cells, saved, skips per reason, seconds and megabytes
      (rows and footers apart), the requests per second and the waits (majortom.STATS
      wait_seconds and retries), from the totals of `fetch_pairs`.

    Returns (paths, skips, timing).
    paths    the location of every pair of the table, in table order, as
             "<archive_dir>/chunk_<k>.tar::chunk_<k>/<grid_cell>.npz" (`pair_location`).
             `open_pair` reads one, `pair_cell` gives its grid cell. Every chunk is read
             from its archive, also one fetched now, its local files are gone
    skips    columns chunk, grid_cell, reason, detail, also for chunks from an archive
    timing   one row per chunk, columns chunk, source (fetched, archive), cells, saved,
             skipped, wall_s, footers, mb_footers, mb_rows, requests, wait_s, retries. For a
             chunk from an archive the numbers are those of the fetch that made it, from its
             JSON (None where that JSON does not hold them). attrs holds wall_s of this call,
             chunks, fetched, from_archive, infos_built, cells, pairs, skipped.
    """
    from . import pairing, stage0

    if "chunk" not in pairs.columns:
        raise ValueError("pairs has no column chunk, give the table of pilot_sample")
    codes = _cloud_codes(cloud_codes)
    if offset not in OFFSET_HOW:
        raise ValueError(f"offset {offset!r} must be one of {tuple(OFFSET_HOW)}")
    work_dir, archive_dir = Path(work_dir), Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    rules = {"cloud_codes": codes, "offset": offset}
    counted = ("wall_s", "workers", "cells_read", "footers", "mb_footers", "mb_rows", "requests",
               "wait_s", "retries")

    start = time.perf_counter()
    paths: list[str] = []
    skip_rows: list[dict] = []
    timing_rows: list[dict] = []
    built = 0
    for k in sorted(int(c) for c in pairs["chunk"].unique()):
        chunk_rows = pairs[pairs["chunk"] == k]
        rows = chunk_rows.to_dict("records")
        folder = work_dir / f"chunk_{k}"
        tar, info = _chunk_files(archive_dir, k)
        found = _archive_info(tar, info)
        if found is not None:
            made = {"cloud_codes": found.get("cloud_codes"), "offset": found.get("offset")}
            if made != rules:
                raise ValueError(f"{tar} was made with the rules {made}, asked for is {rules}. "
                                 "Use another archive folder for another rule")
            infos, was_built = chunk_infos(archive_dir, k, workers=min(8, max(1, int(workers))),
                                           progress=False, found=found)
            built += was_built
            _check_infos(infos, tar, codes, offset)
            files = list(found["files"])
            chunk_skips = [{"chunk": k, **s} for s in found.get("skips", [])]
            totals = found.get("fetch", {})
            source = "archive"
            print(f"chunk {k}  archive  pairs {len(files):,}, skipped {len(chunk_skips):,}, "
                  f"{tar.name} read in place")
        else:
            if tar.is_file() or info.is_file():
                print(f"chunk {k}  the archive {tar.name} is missing its JSON or is cut, "
                      "the chunk is fetched again")
            _check_rules([folder / f"{row['grid_cell']}.npz" for row in rows], codes, offset)
            fetched, chunk_skips_df, chunk_timing = stage0.fetch_pairs(
                chunk_rows, folder, cloud_codes=codes, workers=workers, token=token,
                progress=progress, offset=offset)
            fetched = [Path(f) for f in fetched]
            files = [f.name for f in fetched]
            skips_plain = chunk_skips_df[list(stage0.SKIP_COLUMNS)].to_dict("records")
            t = chunk_timing.attrs
            totals = {name: t.get(name) for name in counted}
            size, index = _write_tar(tar, k, fetched, work_dir)
            _write_json(info, {"chunk": k, "files": files, "bytes": size,
                               "cells": len(rows), "pairs": len(files), **rules,
                               "skips": skips_plain, "fetch": totals})
            infos = pd.DataFrame([_info_row(k, f"chunk_{k}/{f.name}", *index[f"chunk_{k}/{f.name}"],
                                            pairing.load_pair_info(f)) for f in fetched],
                                 columns=list(INFO_COLUMNS))
            _write_infos(infos_file(archive_dir, k), infos)
            _prime_index(tar, infos)
            shutil.rmtree(folder, ignore_errors=True)       # the archive on Drive holds the chunk
            chunk_skips = [{"chunk": k, **s} for s in skips_plain]
            source = "fetched"
            counts = pd.Series([s["reason"] for s in skips_plain]).value_counts().to_dict()
            print(f"chunk {k}  fetched  cells {len(rows):,}  saved {len(files):,}  skipped "
                  f"{len(skips_plain):,}" + (f" ({_skips_text(counts)})" if counts else "")
                  + f"  {totals['wall_s']:,.0f} s  rows {_mb(totals['mb_rows'])}  footers "
                  f"{_mb(totals['mb_footers'])}  {_rate_text(totals)}")
        paths += [pair_location(tar, f"chunk_{k}/{name}") for name in files]
        skip_rows += chunk_skips
        timing_rows.append({"chunk": k, "source": source, "cells": len(rows),
                            "saved": len(files), "skipped": len(chunk_skips),
                            **{name: totals.get(name) for name in
                               ("wall_s", "footers", "mb_footers", "mb_rows", "requests", "wait_s",
                                "retries")}})

    wall = time.perf_counter() - start
    skips = pd.DataFrame(skip_rows, columns=list(SKIP_COLUMNS))
    timing = pd.DataFrame(timing_rows, columns=list(TIMING_COLUMNS))
    sources = timing["source"].value_counts().to_dict()
    timing.attrs = {"wall_s": wall, "chunks": len(timing_rows),
                    "fetched": int(sources.get("fetched", 0)),
                    "from_archive": int(sources.get("archive", 0)), "infos_built": int(built),
                    "cells": int(len(pairs)), "pairs": len(paths), "skipped": len(skips)}
    print(f"pilot  {len(paths):,} pairs of {len(pairs):,} cells, {len(skips):,} skipped, "
          f"{timing.attrs['fetched']} chunks fetched, {timing.attrs['from_archive']} from "
          f"archives, {built} infos tables built, {wall:,.0f} s")
    return paths, skips, timing


# --------------------------------------------------------------------------- held out

def _cells_of(paths: Iterable[str | os.PathLike]) -> tuple[list, list[str]]:
    """The locations and their grid cells (the file names). A cell twice is a ValueError."""
    paths = list(paths)
    cells = [pair_cell(p) for p in paths]
    seen: set[str] = set()
    for cell in cells:
        if cell in seen:
            raise ValueError(f"grid cell {cell} is in the list more than once")
        seen.add(cell)
    return paths, cells


def split_held_out(paths: Iterable[str | os.PathLike], n: int = N_HELD_OUT, seed: int = SEED,
                   archive_dir: str | os.PathLike | None = None) -> tuple[list, list]:
    """`n` pairs held out for the distillation loss, the rest for training.

    `paths` are locations (of `fetch_pilot` or `locations_of`) or plain files, only their
    grid cells count. The draw is made on the SORTED list of grid cells, so the same cells
    give the same split whatever the order of `paths`. Returns (train_paths, held_paths),
    each in the order of `paths`. With `archive_dir` the cells are written there as
    `TRAIN_FILE` and `HELD_OUT_FILE`, one sorted cell per line, next to the archives.
    ValueError when a cell is given twice or `n` leaves no training pair.
    """
    paths, cells = _cells_of(paths)
    if not 0 <= int(n) < len(cells):
        raise ValueError(f"cannot hold out {n} of {len(cells)} pairs")
    ordered = sorted(cells)
    rng = np.random.default_rng(int(seed))
    held = {ordered[i] for i in rng.choice(len(ordered), int(n), replace=False)}
    train_paths = [p for p, c in zip(paths, cells) if c not in held]
    held_paths = [p for p, c in zip(paths, cells) if c in held]
    if archive_dir is not None:
        archive_dir = Path(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        for name, chosen in ((TRAIN_FILE, [c for c in ordered if c not in held]),
                             (HELD_OUT_FILE, [c for c in ordered if c in held])):
            (archive_dir / name).write_text("".join(f"{c}\n" for c in chosen), encoding="utf-8")
    return train_paths, held_paths


# --------------------------------------------------------------------------- the 10 m grid

def s2_grid(pair: Pair, corner_rule: bool = True) -> tuple[np.ndarray, bool]:
    """The 12 bands of a pair on the 10 m grid, float32 (12, 384, 384), and whether the corner
    rule was applied.

    With `corner_rule` the call is pairing.s2_to_grid(pair.s2, size=384, corner=(left, top),
    crs=pair.crs), left and top from `pair.bounds`. That is the band shift correction
    decided on 2026-09-30. An installed `s2_to_grid` that does not take these keywords yet
    gets the call without them, once per process one line says that the correction is not
    applied, and the second value is False. No offset is removed here, the stored pair is
    on one scale already (`pair.meta["offset_how"]`).
    """
    from . import pairing

    if corner_rule:
        try:
            grid = pairing.s2_to_grid(pair.s2, size=CROP_PX,
                                      corner=(pair.bounds[0], pair.bounds[3]), crs=pair.crs)
            return np.asarray(grid, dtype=np.float32), True
        except TypeError as err:
            if "unexpected keyword" not in str(err):
                raise
            _say_once(NO_CORNER_RULE)
    return np.asarray(pairing.s2_to_grid(pair.s2, size=CROP_PX), dtype=np.float32), False


def _meta_row(pair: Pair) -> dict:
    """One row of the meta table of `load_pairs_to_ram`."""
    meta = pair.meta
    return {"grid_cell": pair.grid_cell, "s2_datetime": pair.s2_datetime,
            "s1_datetime": pair.s1_datetime, "gap_days": pair.gap_days, "crs": pair.crs,
            "baseline": meta.get("baseline"), "offset_removed": meta.get("offset_removed"),
            "offset_how": meta.get("offset_how"),
            "cloud_fraction": meta.get("cloud_fraction", float("nan")),
            "s2_product_id": meta.get("s2_product_id")}


def _map(fn: Any, items: Sequence, workers: int, progress: bool, desc: str) -> list:
    """`fn` over `items` in a thread pool, results in order. An error in one item is raised."""
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        jobs = pool.map(fn, items)
        if progress:
            from tqdm.auto import tqdm
            jobs = tqdm(jobs, total=len(items), desc=desc, unit="pair")
        return list(jobs)


# --------------------------------------------------------------------------- statistics

def _mean_sd(total: np.ndarray, squares: np.ndarray, count: int) -> tuple[list, list]:
    mean = total / count
    sd = np.sqrt(np.maximum(squares / count - mean ** 2, 0.0))
    return mean.tolist(), sd.tolist()


def channel_stats(train_paths: Iterable[str | os.PathLike], n: int = N_STATS, seed: int = SEED,
                  aerial: np.ndarray | None = None, out_path: str | os.PathLike | None = None,
                  corner_rule: bool = True, workers: int = 8, progress: bool = True) -> dict:
    """Mean and SD per channel over `n` random training pairs, the standardisation of the student.

    Sentinel-1 the 2 dB channels as stored (VV, VH). Sentinel-2 the 12 bands on the 10 m
    grid after the offset rule (in the stored pair) and the corner rule (`s2_grid`), in DN.
    Aerial, when given as a uint8 array (N, 4, 256, 256), the 4 channels over all N images.
    Every pixel has the same weight, the SD is that of the pixels.

    The `n` pairs are drawn with `seed` from the list sorted by grid cell, so the same
    pairs give the same draw. All pairs when there are at most `n`. `train_paths` are
    locations (`open_pair`), read in `workers` threads, archive by archive in the order of
    the members, so each archive is opened once and never unpacked.
    Values of 0 in Sentinel-2 are not left out and not treated. A stored 0 (no-data) does not
    occur any more, the pairing rule skipped such crops. A 0 after the offset rule is a real
    pixel (`ZERO_NOTE`), zero_pixels counts them per band.

    Returns a dict with seed, n, pairs_given, cells_used, corner_rule (applied to every pair
    or not), then s1, s2 and aerial (None when not given), each with channels, unit, mean,
    sd and pixels, s2 also zero_pixels and note. With `out_path` the dict is written there
    as JSON.
    """
    paths, cells = _cells_of(train_paths)
    if not paths:
        raise ValueError("no training pair given")
    order = np.argsort(cells, kind="stable")
    k = min(int(n), len(paths))
    rng = np.random.default_rng(int(seed))
    picks = order[np.sort(rng.choice(len(paths), k, replace=False))]
    chosen = [paths[i] for i in picks]
    reading = [chosen[i] for i in _by_tar(chosen)]

    def one(location: Any) -> tuple:
        pair = open_pair(location)
        db = np.asarray(pair.s1_db, dtype=np.float64).reshape(len(S1_CHANNELS), -1)
        grid, used = s2_grid(pair, corner_rule)
        g = grid.astype(np.float64).reshape(len(S2_BANDS), -1)
        return (db.sum(1), (db * db).sum(1), db.shape[1], g.sum(1), (g * g).sum(1), g.shape[1],
                (g == 0).sum(1), used)

    s1_sum, s1_sq = np.zeros(len(S1_CHANNELS)), np.zeros(len(S1_CHANNELS))
    s2_sum, s2_sq = np.zeros(len(S2_BANDS)), np.zeros(len(S2_BANDS))
    zeros = np.zeros(len(S2_BANDS), np.int64)
    s1_px = s2_px = 0
    applied = True
    for a, b, c, d, e, f, z, used in _map(one, reading, workers, progress, "channel stats"):
        s1_sum, s1_sq, s1_px = s1_sum + a, s1_sq + b, s1_px + c
        s2_sum, s2_sq, s2_px = s2_sum + d, s2_sq + e, s2_px + f
        zeros += z
        applied = applied and used
    s1_mean, s1_sd = _mean_sd(s1_sum, s1_sq, s1_px)
    s2_mean, s2_sd = _mean_sd(s2_sum, s2_sq, s2_px)

    aerial_stats = None
    if aerial is not None:
        a = np.asarray(aerial)
        if a.dtype != np.uint8 or a.ndim != 4 or a.shape[1] != len(AERIAL_CHANNELS):
            raise ValueError(f"aerial must be uint8 (N, {len(AERIAL_CHANNELS)}, h, w), got "
                             f"{a.dtype} {a.shape}")
        total, squares = np.zeros(a.shape[1]), np.zeros(a.shape[1])
        for i in range(0, len(a), 256):                    # blocks, never a float copy of all
            x = a[i:i + 256].astype(np.float64).reshape(-1, a.shape[1], a.shape[2] * a.shape[3])
            total += x.sum((0, 2))
            squares += (x * x).sum((0, 2))
        px = len(a) * a.shape[2] * a.shape[3]
        mean, sd = _mean_sd(total, squares, px) if px else ([float("nan")] * a.shape[1],) * 2
        aerial_stats = {"channels": list(AERIAL_CHANNELS), "unit": "8 bit", "mean": mean,
                        "sd": sd, "images": int(len(a)), "pixels": int(px)}

    out = {"seed": int(seed), "n": k, "pairs_given": len(paths),
           "cells_used": [pair_cell(p) for p in chosen], "corner_rule": bool(applied),
           "s1": {"channels": list(S1_CHANNELS), "unit": "dB", "mean": s1_mean, "sd": s1_sd,
                  "pixels": int(s1_px)},
           "s2": {"channels": list(S2_BANDS),
                  "unit": "DN on the 10 m grid, offset taken off by processing baseline",
                  "mean": s2_mean, "sd": s2_sd, "pixels": int(s2_px),
                  "zero_pixels": zeros.tolist(), "note": ZERO_NOTE},
           "aerial": aerial_stats}
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(out_path, out)
    return out


# --------------------------------------------------------------------------- RAM

def _location_bytes(location: str | os.PathLike) -> int:
    """The bytes of the pair file at `location`, of the tar member or of the plain file."""
    path, member = split_location(location)
    return _member_at(path, member)[1] if member is not None else path.stat().st_size


def load_pairs_to_ram(paths: Iterable[str | os.PathLike], corner_rule: bool = True,
                      progress: bool = True, limit_gb: float = RAM_LIMIT_GB,
                      workers: int = RAM_WORKERS) -> dict:
    """All pairs as two arrays in memory, read in a thread pool straight from the archives.

    paths      locations (`open_pair`), tar members or plain files. Tar members are read in
               place through their own file handles, archive by archive in the order of the
               members, nothing is written to disk. The arrays are in the order of `paths`
    workers    threads (`RAM_WORKERS`, the reason is written there). At the end one line says
               the MB read, the seconds, the MB and the pairs per second and the threads
    s1         float16 (N, 2, 384, 384), dB, as stored
    s2         uint16 (N, 12, 384, 384), the 12 bands on the 10 m grid, rounded to whole
               numbers. The offset is already off in the stored pair, the corner rule is
               applied here (`s2_grid`, `corner_rule` = False leaves it off)
    grid_cell  list of N, in the order of `paths`
    meta       one row per pair, columns `META_COLUMNS` (baseline, dates, gap, crs, the
               offset and cloud entries of the pair)
    corner_rule  whether the correction was applied to every pair
    bytes      what s1 and s2 take together

    The memory, N x `PAIR_BYTES` (4.13 MB per pair), is printed before anything is
    allocated. Above `limit_gb` the call is refused with a MemoryError.
    """
    paths = list(paths)
    n = len(paths)
    need = n * PAIR_BYTES
    print(f"load_pairs_to_ram  {n:,} pairs x {PAIR_BYTES / MB:.2f} MB = {need / GB:.2f} GB, "
          f"limit {limit_gb:g} GB")
    if need > float(limit_gb) * GB:
        raise MemoryError(f"{n:,} pairs need {need / GB:.2f} GB, the limit is {limit_gb:g} GB. "
                          "Give fewer pairs or a higher limit")
    rows = _Rows(paths, corner_rule)
    start = time.perf_counter()
    order = _by_tar(paths)
    _map(rows.fill, order, workers, progress, "pairs to RAM")
    wall = time.perf_counter() - start
    read_mb = sum(_location_bytes(p) for p in paths) / MB
    print(f"pairs to RAM  {n:,} pairs, {read_mb:,.0f} MB in {wall:,.1f} s, "
          f"{read_mb / max(wall, 1e-9):,.0f} MB per second, {n / max(wall, 1e-9):,.1f} pairs per second, "
          f"{max(1, int(workers))} threads")
    return rows.result()


class _Rows:
    """The arrays of one list of pairs, made once at full size, and the reading of one pair into
    its row. `fill` is the one way a pair enters RAM, for `load_pairs_to_ram` and `ChunkReader`
    alike, so both give the same arrays bit for bit."""

    def __init__(self, locations: Sequence, corner_rule: bool) -> None:
        self.locations = list(locations)
        n = len(self.locations)
        self.corner_rule = corner_rule
        self.s1 = np.empty((n, len(S1_CHANNELS), CROP_PX, CROP_PX), np.float16)
        self.s2 = np.empty((n, len(S2_BANDS), CROP_PX, CROP_PX), np.uint16)
        self.applied: list[bool] = [False] * n
        self.metas: list[dict] = [{}] * n

    def fill(self, i: int, location: str | os.PathLike | None = None) -> None:
        """Row `i` from the pair at `location`, at its own location when None."""
        where = self.locations[i] if location is None else location
        pair = open_pair(where)
        db = np.asarray(pair.s1_db)
        if db.shape != self.s1.shape[1:]:
            raise ValueError(f"{where}: s1_db is {db.shape}, expected {self.s1.shape[1:]}")
        grid, used = s2_grid(pair, self.corner_rule)
        self.s1[i] = db
        self.s2[i] = np.rint(grid).astype(np.uint16)
        self.applied[i], self.metas[i] = used, _meta_row(pair)

    def result(self) -> dict:
        """The dict `load_pairs_to_ram` returns."""
        meta = pd.DataFrame(self.metas, columns=list(META_COLUMNS))
        return {"s1": self.s1, "s2": self.s2, "grid_cell": [m["grid_cell"] for m in self.metas],
                "meta": meta, "corner_rule": bool(self.corner_rule and all(self.applied)),
                "bytes": len(self.locations) * PAIR_BYTES}


_CHUNK_TAR = re.compile(r"chunk_(\d+)\.tar")


class ChunkReader:
    """The pairs of named lists into RAM, one chunk archive at a time, in the order the archives arrive.

    The data notebook reads each archive while the next ones are still being fetched
    (hub.fetch_pilot hands every archive over as soon as it is whole on the disk), instead of
    reading all of them after the whole fetch.

        reader = ChunkReader({"train": train_locations, "held_out": held_locations})
        reader.arrived(path)         for every chunk archive, as it arrives
        ram = reader.finish()        name -> the dict of `load_pairs_to_ram` for that list

    lists        name -> locations of pairs in chunk archives ("<folder>/chunk_<k>.tar::chunk_<k>/
                 <cell>.npz", from `pair_locations` or `locations_of`). Every list gets its own
                 arrays in its own order, made once at full size here; the memory is printed
                 first, and above `limit_gb` the reader is refused (MemoryError) before anything
                 is made, as in `load_pairs_to_ram`
    arrived(tar) hands over the archive chunk_<k>.tar, whole on the disk, with its JSON next to
                 it. Its folder may be another one than the locations name (the runtime disk or
                 Drive), only the archive name and the member names count. Returns at once
    reading      one thread of the reader takes the archives in the order they were handed
                 over. Per archive the infos table is read, or built from the tar and written
                 next to it (`chunk_infos`, which also refuses an archive that is cut or has no
                 JSON), then every pair of every list in that archive goes into its row in the
                 order of the members, `workers` threads at once (`RAM_WORKERS`, the reason is
                 written there), and one line gives the pairs, the MB, the seconds and the MB per
                 second of that archive
    finish()     waits until every archive handed over is read, raises the first error of the
                 reading, refuses (RuntimeError) when a pair was never read because its archive
                 was not handed over, prints one line for all and returns name -> the dict of
                 `load_pairs_to_ram` (s1, s2, grid_cell, meta, corner_rule, bytes). The reader
                 keeps no array afterwards
    stop()       ends the reading without the rest and lets go of the arrays, so after an error
                 of the notebook cell no thread goes on and no arrays stay in RAM
    read_s, read_mb, pairs   the seconds the reading of all archives took (the time an archive
                 waited for the fetch not counted), the MB of the pair files read, the pairs read

    The arrays equal those of `load_pairs_to_ram` on each list bit for bit, whatever order the
    archives arrive in, because a pair enters its row the same way (`_Rows.fill`).
    """

    def __init__(self, lists: Mapping[str, Sequence], corner_rule: bool = True,
                 limit_gb: float = RAM_LIMIT_GB, workers: int = RAM_WORKERS) -> None:
        lists = {str(name): list(locations) for name, locations in lists.items()}
        n = sum(len(locations) for locations in lists.values())
        need = n * PAIR_BYTES
        print(f"ChunkReader  {n:,} pairs in {len(lists)} lists x {PAIR_BYTES / MB:.2f} MB = {need / GB:.2f} GB, "
              f"limit {limit_gb:g} GB")
        if need > float(limit_gb) * GB:
            raise MemoryError(f"{n:,} pairs need {need / GB:.2f} GB, the limit is {limit_gb:g} GB. "
                              "Give fewer pairs or a higher limit")
        todo: dict[str, list[tuple[str, int, str]]] = {}      # archive name -> (list, row, member)
        for name, locations in lists.items():
            for i, location in enumerate(locations):
                tar, member = split_location(location)
                if member is None or not _CHUNK_TAR.fullmatch(tar.name):
                    raise ValueError(f"{location} is not a pair in a chunk archive chunk_<k>.tar, "
                                     "ChunkReader reads archives only")
                todo.setdefault(tar.name, []).append((name, i, member))
        self.workers = max(1, int(workers))
        self.read_s = self.read_mb = 0.0
        self.pairs = 0
        self._todo = todo
        self._rows = {name: _Rows(locations, corner_rule) for name, locations in lists.items()}
        self._done = {name: np.zeros(len(locations), bool) for name, locations in lists.items()}
        self._given: list[str] = []
        self._queue: queue.Queue = queue.Queue()
        self._error: BaseException | None = None
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name="ChunkReader", daemon=True)
        self._thread.start()

    def arrived(self, tar: str | os.PathLike) -> None:
        """Hand over the chunk archive `tar`, whole on the disk. Returns at once."""
        tar = Path(tar)
        if self._rows is None:
            raise RuntimeError("the reader is finished or stopped, make a new one")
        if not _CHUNK_TAR.fullmatch(tar.name):
            raise ValueError(f"{tar} is not a chunk archive chunk_<k>.tar")
        if tar.name in self._given:
            raise ValueError(f"{tar.name} was handed over twice")
        self._given.append(tar.name)
        self._queue.put(tar)

    def _run(self) -> None:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            while (tar := self._queue.get()) is not None:
                if self._error is not None or self._stopping:
                    continue                              # the rest is not read
                try:
                    self._read_archive(tar, pool)
                except BaseException as err:              # kept for finish()
                    self._error = err

    def _read_archive(self, tar: Path, pool: ThreadPoolExecutor) -> None:
        start = time.perf_counter()
        k = int(_CHUNK_TAR.fullmatch(tar.name).group(1))
        chunk_infos(tar.parent, k, workers=self.workers)  # checks the archive, primes its index
        placed = sorted((*_member_at(tar, member), name, i, member)
                        for name, i, member in self._todo.get(tar.name, []))

        def one(job: tuple) -> None:
            _, _, name, i, member = job
            if self._stopping:
                return
            self._rows[name].fill(i, pair_location(tar, member))
            self._done[name][i] = True

        list(pool.map(one, placed))
        seconds = time.perf_counter() - start
        mb = sum(job[1] for job in placed) / MB
        self.read_s += seconds
        self.read_mb += mb
        self.pairs += len(placed)
        print(f"chunk {k}  {len(placed):,} pairs read from {tar.name} in {seconds:.1f} s, {mb:,.0f} MB, "
              f"{mb / max(seconds, 1e-9):,.0f} MB per second")

    def finish(self) -> dict[str, dict]:
        """Wait for the reading, check it, return name -> the dict of `load_pairs_to_ram`."""
        if self._rows is None:
            raise RuntimeError("the reader is finished or stopped already")
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            err = self._error
            self._drop()
            raise err
        missing = {}
        for name, done in self._done.items():
            rows = np.flatnonzero(~done)
            if len(rows):
                locations = self._rows[name].locations
                missing[name] = (len(rows), sorted({split_location(locations[i])[0].name for i in rows}))
        if missing:
            self._drop()
            raise RuntimeError("pairs were not read because their archives were never handed over, "
                               + ", ".join(f"{name} {n:,} pairs in {archives}" for name, (n, archives) in missing.items()))
        out = {name: rows.result() for name, rows in self._rows.items()}
        print(f"pairs to RAM  {self.pairs:,} pairs of {len(out)} lists from {len(self._given)} archives, "
              f"{self.read_mb:,.0f} MB in {self.read_s:,.1f} s of reading, "
              f"{self.read_mb / max(self.read_s, 1e-9):,.0f} MB per second, "
              f"{self.pairs / max(self.read_s, 1e-9):,.1f} pairs per second, {self.workers} threads")
        self._drop()
        return out

    def stop(self) -> None:
        """End the reading without the rest and let go of the arrays. Safe to call again."""
        self._stopping = True
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()
        self._drop()

    def _drop(self) -> None:
        self._rows = self._done = None


# --------------------------------------------------------------------------- summary

def pilot_summary(meta: pd.DataFrame, skips: pd.DataFrame | None = None) -> pd.DataFrame:
    """One table of a pilot, printed and returned.

    meta    the infos table (`load_infos`, one row per kept pair) or the meta table of
            `load_pairs_to_ram`, only the columns `META_COLUMNS` are read
    skips   the skips table of `fetch_pilot`, None = no skipped cell

    Columns kind, what, pairs, share. kind is
        pairs      kept and skipped, share of all cells
        skip       one row per reason that occurred, share of all cells
        year       Sentinel-2 acquisition year of the kept pairs, share of the kept pairs
        baseline   processing baseline, share of the kept pairs
        offset     removed by baseline, removed per image (N9999), not removed
        cloud      pairs with any cloud pixel, and the mean cloud fraction (pairs = the pairs
                   with a cloud fraction, share = the mean)
    """
    n = len(meta)
    n_skip = 0 if skips is None else len(skips)
    total = n + n_skip
    rows: list[tuple] = [("pairs", "kept", n, n / total if total else float("nan")),
                         ("pairs", "skipped", n_skip, n_skip / total if total else float("nan"))]
    if skips is not None and n_skip:
        counts = skips["reason"].value_counts().to_dict()
        rows += [("skip", r, int(counts[r]), counts[r] / total) for r in SKIP_REASONS
                 if counts.get(r)]
    share = (lambda k: k / n) if n else (lambda k: float("nan"))
    years = pd.to_datetime(meta["s2_datetime"], utc=True).dt.year if n else pd.Series(dtype=int)
    rows += [("year", str(int(y)), int(k), share(k))
             for y, k in sorted(years.value_counts().items())]
    baselines = meta["baseline"].fillna("no baseline").astype(str) if n else pd.Series(dtype=str)
    rows += [("baseline", b, int(k), share(k))
             for b, k in sorted(baselines.value_counts().items())]
    removed = meta["offset_removed"].astype(bool) if n else pd.Series(dtype=bool)
    how = meta["offset_how"].astype(str) if n else pd.Series(dtype=str)
    for what, flag in (("removed by baseline", removed & (how == "baseline")),
                       ("removed per image (N9999)", removed & (how == "per_image")),
                       ("not removed", ~removed)):
        rows.append(("offset", what, int(flag.sum()), share(int(flag.sum()))))
    cloud = pd.to_numeric(meta["cloud_fraction"], errors="coerce") if n else pd.Series(dtype=float)
    with_rule = cloud.notna()
    any_cloud = int((cloud > 0).sum())
    rows.append(("cloud", "pairs with any cloud", any_cloud, share(any_cloud)))
    rows.append(("cloud", "mean cloud fraction", int(with_rule.sum()),
                 float(cloud[with_rule].mean()) if with_rule.any() else float("nan")))
    table = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    print(table.to_string(index=False, formatters={"share": "{:.3f}".format,
                                                   "pairs": "{:,}".format}))
    return table
