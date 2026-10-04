"""Index of the Major-TOM cells and single-cell reads over HTTP.

Two public sets of ESA Major-TOM are read, both CC-BY-SA-4.0

    Core-S2L2A-249k   parquet shards under images_249k/, no metadata file
    Core-S1RTC        metadata.parquet and parquet shards under images/

What the files looked like on 2026-09-29. The code relies on these points and checks them.

- Every shard holds ONE ROW PER ROW GROUP, so `parquet_row` is the number of a row group.
  A shard that breaks this is refused.
- Every image column holds one GeoTIFF as bytes. Sentinel-2 bands are uint16 at native
  resolution, the cloud mask is uint8, Sentinel-1 VV and VH are float32 linear power.
- grid_cell, product_id and product_datetime are strings. The date looks like
  20221115T161819 (no zone written, read as UTC). Any other form is an error, it is not guessed.
- The Sentinel-2 card describes a metadata_249k.parquet that is not in the repository. The
  index is built from the FOOTERS of the shards instead. With one row per row group the
  footer statistics (min = max) of the three text columns already are the values.
  A footer is 2 to 5 MB (2.1 to 5.2 MB in 12 shards, mean 4.0 MB), because its statistics
  also hold copies of cloud masks and thumbnails. A Sentinel-1 footer is 0.3 MB.
- In the Sentinel-1 metadata the column `timestamp` is some seconds later than the
  `product_datetime` stored next to the image (12 and 15 s in two cells, and 4 to 19 s
  later than the start time in the product name over all rows), `nodata` is 0.0
  in every row (it is not the no-data value of the GeoTIFF, which is -32768) and `crs` is
  written as EPSG:EPSG:32752.

A "url" may also be a local file path. Both go through the same code, only the two
functions that fetch bytes differ.

Nothing here prints or stores a token. With `token=None` every read is anonymous, no login
is taken from the environment (no netrc file). A token is sent to addresses under `HF`
only. When the Hub sends a request on to another host, the token does not go along.

What a read costs is counted in `STATS`, footers apart from rows. Reading one cell costs
the footer of its shard on top of the row, unless the footer is still in memory, and only
FOOTER_CACHE footers are kept. The cells of a sample are spread over the shards, so nearly
every cell of a sample pays a footer of its own. A run over all cells pays one footer per
shard, but only when it reads the cells shard by shard (sorted by parquet_url). Sorted by
grid_cell the shard changes at about every third cell (counted in the Sentinel-1 metadata).
Use the row counters, not the totals of a sample, to estimate a full run.
"""

from __future__ import annotations

import json
import os
import re
import struct
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from .records import S2_BANDS, S1Cell, S2Cell

S2_REPO = "Major-TOM/Core-S2L2A-249k"
S1_REPO = "Major-TOM/Core-S1RTC"
S2_SHARD_DIR = "images_249k"
HF = "https://huggingface.co"

META_COLUMNS = ("grid_cell", "product_id", "product_datetime")
S2_INDEX_COLUMNS = (*META_COLUMNS, "parquet_url", "parquet_row")
S1_INDEX_COLUMNS = (*S2_INDEX_COLUMNS, "nodata", "orbit_state", "centre_lat", "centre_lon", "crs")
S1_METADATA_COLUMNS = ("grid_cell", "product_id", "timestamp", "parquet_url", "parquet_row",
                       "nodata", "orbit_state", "centre_lat", "centre_lon", "crs")
DATETIME_FORMAT = "%Y%m%dT%H%M%S"              # as stored in both sets, e.g. 20221115T161819
DATETIME_FORM = r"\d{8}T\d{6}"
GRID_CELL = re.compile(r"\d+[UD]_\d+[LR]")     # as stored in both sets, e.g. 921D_252L

# Our own settings for the transfer, not facts about the data.
RETRIES = 5                  # tries per request
BACKOFF_S = 1.0              # wait before the second try, doubled before every further try
TIMEOUT_S = 60.0             # seconds, per request
MAX_WAIT_S = 300.0           # longest wait that a server may ask for (the Hub counts in 5 minutes)
FOOTER_GUESS = 1 << 20       # bytes asked from the end of a file in the first footer request
MERGE_GAP = 1 << 18          # byte ranges closer than this are fetched as one request
FOOTER_CACHE = 8             # parsed footers kept in memory (a Sentinel-2 footer is 2 to 5 MB)
GRID_TOL_M = 1e-3            # two grids count as the same when their six numbers are this close

# What was read since the start, over HTTP and from local files. Take a copy before a run and
# one after it, the difference is the cost of the run.
#   requests, bytes, seconds    every read that worked. The seconds of a read hold the tries
#                               that failed before it, not the waits between the tries
#   footer_*                    the part of these that fetched parquet footers, its seconds
#                               also hold the parsing. `footers` = footers fetched (one or
#                               two requests each), a footer found in memory costs nothing
#   row_*                       the part that fetched the column chunks of rows
#   the rest                    the list of shards and the Sentinel-1 metadata file
#   retries, wait_seconds       tries after the first one and the seconds waited before them,
#                               also for a read that failed in the end
#   decoded, decode_seconds     images decoded by `decode_tif`
# Seconds are measured inside the thread that reads and are summed over all threads, like
# the seconds per cell that a caller measures inside its threads. They are not wall time.
STATS = {"requests": 0, "bytes": 0, "seconds": 0.0,
         "footers": 0, "footer_requests": 0, "footer_bytes": 0, "footer_seconds": 0.0,
         "row_requests": 0, "row_bytes": 0, "row_seconds": 0.0,
         "retries": 0, "wait_seconds": 0.0,
         "decoded": 0, "decode_seconds": 0.0}

_LOCAL = threading.local()
_LOCK = threading.Lock()
_FOOTERS: OrderedDict[str, tuple[pq.FileMetaData, int]] = OrderedDict()
_FETCHING: dict[str, threading.Lock] = {}      # one lock per footer that is being fetched


# ------------------------------------------------------------------ fetching bytes

def _is_http(url: str) -> bool:
    return str(url).startswith(("http://", "https://"))


def _add(**more: float) -> None:
    with _LOCK:
        for name, value in more.items():
            STATS[name] += value


def _count(kind: str, n_bytes: int, seconds: float, n_requests: int = 1) -> None:
    """One read that worked, under the totals and under its kind ("footer", "row", "other")."""
    more = {"requests": n_requests, "bytes": n_bytes, "seconds": seconds}
    if kind in ("footer", "row"):
        more.update({f"{kind}_{name}": value for name, value in list(more.items())})
    _add(**more)


def _session() -> requests.Session:
    """One session per thread, so connections are kept open and never shared.

    The session does not read the environment. requests would else add the login of a netrc
    file to a request on its own, also in the place of a given token and also after the
    request was sent on to another host. Proxy settings of the environment are not read
    either.
    """
    if not hasattr(_LOCAL, "session"):
        _LOCAL.session = requests.Session()
        _LOCAL.session.trust_env = False
    return _LOCAL.session


def _get(url: str, token: str | None = None, byte_range: str | None = None,
         kind: str = "other") -> tuple[bytes, Mapping]:
    """One GET with retries, returns (body, headers). `byte_range` is e.g. "bytes=0-99".

    Network errors, cut answers and the codes 408, 429 and 5xx are tried again, RETRIES tries
    in all, waiting BACKOFF_S and then twice as long each time, or as long as the server asks
    (RateLimit or Retry-After, at most MAX_WAIT_S). Other codes stop at once.
    A read that fails in the end raises, it never comes back empty.
    A range request must be answered with 206, so a server that ignores the range never
    sends a whole 5 GB file. Error texts hold the url and the code, never the token.
    `kind` says where the read is counted in STATS.
    """
    headers = {"Accept-Encoding": "identity"}
    if byte_range:
        headers["Range"] = byte_range
    if token and url.startswith(HF + "/"):
        headers["Authorization"] = f"Bearer {token}"
    want = 206 if byte_range else 200
    last, wait, spent = "", 0.0, 0.0
    for attempt in range(RETRIES):
        if attempt:
            _add(retries=1, wait_seconds=wait)
            time.sleep(wait)
        wait = BACKOFF_S * 2 ** attempt
        start = time.perf_counter()
        try:
            with _session().get(url, headers=headers, timeout=TIMEOUT_S, stream=True) as r:
                code, answer = r.status_code, r.headers      # header names match without case
                body = r.content if code == want else b""
        except (requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError) as err:
            last = type(err).__name__
            continue
        finally:
            spent += time.perf_counter() - start
        if code in (408, 429) or code >= 500:
            last = f"HTTP {code}"
            asked = (re.search(r"\bt=(\d+)", str(answer.get("RateLimit", "")))
                     or re.fullmatch(r"\s*(\d+)\s*", str(answer.get("Retry-After", ""))))
            if asked:
                wait = max(wait, min(float(asked.group(1)), MAX_WAIT_S))
            continue
        if code == 404:
            raise FileNotFoundError(f"{url}: HTTP 404, no such file")
        if code in (401, 403):
            raise PermissionError(f"{url}: HTTP {code}, this file needs a token with read access")
        if code != want:
            raise OSError(f"{url}: HTTP {code}, expected {want}"
                          + (" (the server did not follow the byte range)" if byte_range else ""))
        length = answer.get("Content-Length")
        if length is not None and int(length) != len(body):
            last = f"cut answer ({len(body)} of {length} bytes)"
            continue
        _count(kind, len(body), spent)
        return body, answer
    raise ConnectionError(f"{url}: gave up after {RETRIES} tries, the last error was {last}")


def _read_range(url: str, start: int, end: int, token: str | None = None,
                kind: str = "row") -> bytes:
    """Bytes start to end - 1 of a file or url. `kind` says where the read is counted."""
    if _is_http(url):
        data, _ = _get(url, token, f"bytes={start}-{end - 1}", kind)
    else:
        begin = time.perf_counter()
        with open(url, "rb") as f:
            f.seek(start)
            data = f.read(end - start)
        _count(kind, len(data), time.perf_counter() - begin)
    if len(data) != end - start:
        raise OSError(f"{url}: asked for bytes {start} to {end - 1}, got {len(data)} bytes")
    return data


def _read_tail(url: str, n: int, token: str | None = None) -> tuple[bytes, int]:
    """The last n bytes of a file or url (all of it when it is shorter) and its size in bytes.

    Only footers are read this way, so the read is counted as a footer.
    """
    if _is_http(url):
        data, answer = _get(url, token, f"bytes=-{n}", "footer")
        total = str(answer.get("Content-Range", "")).rpartition("/")[2]
        if not total.isdigit():
            raise OSError(f"{url}: the answer does not say the file size (Content-Range)")
        return data, int(total)
    begin = time.perf_counter()
    size = os.path.getsize(url)
    with open(url, "rb") as f:
        f.seek(max(0, size - n))
        data = f.read()
    _count("footer", len(data), time.perf_counter() - begin)
    return data, size


class _Parts:
    """A read-only file that holds only some byte ranges of a larger file.

    pyarrow reads the column chunks from it. A read outside the held ranges is an error,
    it is never answered with made-up bytes.
    """

    closed = False

    def __init__(self, size: int, parts: list[tuple[int, bytes]], name: str):
        self.size, self.parts, self.name, self.pos = size, parts, name, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def close(self) -> None:
        self.closed = True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset + (0, self.pos, self.size)[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.pos + n, self.size)
        if end <= self.pos:
            return b""
        for start, data in self.parts:
            if start <= self.pos and end <= start + len(data):
                out = data[self.pos - start:end - start]
                self.pos = end
                return out
        raise OSError(f"{self.name}: bytes {self.pos} to {end - 1} were not fetched")


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted (start, end) ranges, neighbours closer than MERGE_GAP joined."""
    out: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= MERGE_GAP:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


# ------------------------------------------------------------------ parquet, footer first

def _kept_footer(url: str) -> tuple[pq.FileMetaData, int] | None:
    """The footer of `url` when it is in memory. Call it with _LOCK held."""
    if url in _FOOTERS:
        _FOOTERS.move_to_end(url)
        return _FOOTERS[url]
    return None


def _fetch_footer(url: str, token: str | None = None) -> tuple[pq.FileMetaData, int]:
    tail, size = _read_tail(url, FOOTER_GUESS, token)
    if len(tail) < 8 or tail[-4:] != b"PAR1":
        raise ValueError(f"{url}: not a parquet file (no PAR1 at the end)")
    need = struct.unpack("<I", tail[-8:-4])[0] + 8
    if need > size:
        raise ValueError(f"{url}: the footer says {need} bytes, the file has {size}")
    if need > len(tail):
        tail = _read_range(url, size - need, size - len(tail), token, "footer") + tail
    begin = time.perf_counter()
    md = pq.read_metadata(pa.BufferReader(tail[-need:]))
    _count("footer", 0, time.perf_counter() - begin, 0)
    _add(footers=1)
    return md, size


def _footer(url: str, token: str | None = None) -> tuple[pq.FileMetaData, int]:
    """The parsed footer of a parquet file and the file size, in one or two range requests.

    The last FOOTER_CACHE footers are kept in memory. Threads that ask for the same footer
    at the same time wait for ONE fetch, they do not fetch it once each.
    """
    url = str(url)
    with _LOCK:
        found = _kept_footer(url)
        if found is not None:
            return found
        gate = _FETCHING.setdefault(url, threading.Lock())
    with gate:
        try:
            with _LOCK:
                found = _kept_footer(url)
            if found is None:
                found = _fetch_footer(url, token)
                with _LOCK:
                    _FOOTERS[url] = found
                    while len(_FOOTERS) > FOOTER_CACHE:
                        _FOOTERS.popitem(last=False)
        finally:
            with _LOCK:
                if _FETCHING.get(url) is gate:
                    del _FETCHING[url]
    return found


def _column_numbers(md: pq.FileMetaData, columns: Sequence[str], url: str) -> list[int]:
    names = md.schema.names
    missing = [c for c in columns if c not in names]
    if missing:
        raise KeyError(f"{url}: no column {missing}, the file has {names}")
    return [names.index(c) for c in columns]


def _read_groups(url: str, rows: Sequence[int], columns: Sequence[str] | None = None,
                 token: str | None = None) -> pa.Table:
    """The asked columns of the asked row groups. Only their bytes are fetched.

    Every asked row group must hold exactly one row. Columns are flat (no nested types).
    """
    url = str(url)
    md, size = _footer(url, token)
    columns = list(md.schema.names if columns is None else columns)
    numbers = _column_numbers(md, columns, url)
    spans = []
    for row in rows:
        if not 0 <= row < md.num_row_groups:
            raise IndexError(f"{url}: no row {row}, the file has {md.num_row_groups} row groups")
        group = md.row_group(row)
        if group.num_rows != 1:
            raise ValueError(f"{url}: row group {row} holds {group.num_rows} rows, expected "
                             "one row per row group")
        for j in numbers:
            col = group.column(j)
            start = col.data_page_offset
            if col.has_dictionary_page and 0 < col.dictionary_page_offset < start:
                start = col.dictionary_page_offset
            spans.append((start, start + col.total_compressed_size))
    parts = [(a, _read_range(url, a, b, token)) for a, b in _merge(spans)]
    source = pq.ParquetFile(_Parts(size, parts, url), metadata=md, pre_buffer=False)
    return source.read_row_groups(list(rows), columns=columns)


def read_row(parquet_url: str, parquet_row: int, columns: Sequence[str] | None = None,
             token: str | None = None) -> dict:
    """One row of a shard, only the asked columns (None = all), values as stored.

    Image columns come back as bytes (one GeoTIFF or PNG each), text columns as str.
    Costs the footer of the shard (kept in memory for the next reads of that shard) plus the
    row itself. Measured without thumbnails, 1.2 MB for a Sentinel-2 cell behind a footer of
    2 to 5 MB, and 10.2 MB for a Sentinel-1 cell behind a footer of 0.3 MB.
    STATS counts the two apart (footer_bytes and footer_seconds, row_bytes and row_seconds).
    """
    table = _read_groups(parquet_url, [int(parquet_row)], columns, token)
    return {name: table.column(name)[0].as_py() for name in table.column_names}


# ------------------------------------------------------------------ the two indexes

def list_s2_shards(token: str | None = None) -> list[str]:
    """URLs of the parquet shards under images_249k/, sorted by name."""
    url = f"{HF}/api/datasets/{S2_REPO}/tree/main/{S2_SHARD_DIR}"
    paths: list[str] = []
    while url:
        body, answer = _get(url, token)
        paths += [e["path"] for e in json.loads(body)
                  if e.get("type") == "file" and e["path"].endswith(".parquet")]
        link = re.search(r'<([^>]+)>;\s*rel="next"', str(answer.get("Link", "")))
        url = link.group(1) if link else ""
    if not paths:
        raise FileNotFoundError(f"{S2_REPO}: no parquet file under {S2_SHARD_DIR}/")
    return [f"{HF}/datasets/{S2_REPO}/resolve/main/{p}" for p in sorted(paths)]


def _stat_value(stats: Any) -> str | None:
    """The one value behind the footer statistics of a one-row column chunk, or None.

    None = the statistics are missing, count a null, or min and max differ. A writer that
    cuts long values short stores a lower min and a higher max, so they differ then.
    """
    if stats is None or not stats.has_min_max or stats.min != stats.max:
        return None
    if stats.has_null_count and stats.null_count:
        return None
    value = stats.min
    return value if isinstance(value, str) and value else None


def _parse_dates(values: Sequence[str], errors: str = "raise") -> pd.Series:
    """Stored date strings as UTC timestamps. Another form raises, or gives NaT with "coerce".

    The form is checked character by character first, because the parser alone would read a
    value that was cut short (20221115T1618) as another time.
    """
    text = pd.Series(list(values), dtype="object")
    formed = text.str.fullmatch(DATETIME_FORM).fillna(False).astype(bool)
    if errors == "raise" and not formed.all():
        raise ValueError(f"{text[~formed].iloc[0]!r} is not like 20221115T161819")
    return pd.to_datetime(text.where(formed), format=DATETIME_FORMAT, utc=True, errors=errors)


def _shard_index(url: str, token: str | None = None) -> tuple[pd.DataFrame, str]:
    """The index rows of one shard and how they were read ("footer" or "columns")."""
    md, _ = _footer(url, token)
    n = md.num_row_groups
    if md.num_rows != n:
        raise ValueError(f"{url}: {md.num_rows} rows in {n} row groups, expected one row "
                         "per row group")
    numbers = _column_numbers(md, META_COLUMNS, url)
    values: dict[str, list] = {c: [] for c in META_COLUMNS}
    how = "footer"
    for i in range(n):
        group = md.row_group(i)
        found = [_stat_value(group.column(j).statistics) for j in numbers]
        if None in found:
            how = "columns"
            break
        for c, v in zip(META_COLUMNS, found):
            values[c].append(v)
    if how == "footer" and n:
        cells_ok = all(GRID_CELL.fullmatch(g) for g in values["grid_cell"])
        dates_ok = _parse_dates(values["product_datetime"], errors="coerce").notna().all()
        if not (cells_ok and dates_ok):
            how = "columns"
    if how == "columns":
        table = _read_groups(url, range(n), META_COLUMNS, token)
        values = {c: table.column(c).to_pylist() for c in META_COLUMNS}
    for c in ("grid_cell", "product_id"):
        for i, value in enumerate(values[c]):
            if not isinstance(value, str) or not value:
                # such a row could not be paired and would drop out without a count
                raise ValueError(f"{url} row {i}: {c} is missing")
    try:
        dates = _parse_dates(values["product_datetime"])
    except (ValueError, TypeError):
        raise ValueError(f"{url}: a stored product_datetime is not like 20221115T161819") from None
    df = pd.DataFrame({"grid_cell": values["grid_cell"], "product_id": values["product_id"],
                       "product_datetime": dates, "parquet_url": [url] * n,
                       "parquet_row": np.arange(n, dtype=np.int64)})
    return df[list(S2_INDEX_COLUMNS)], how


def _check_columns(df: pd.DataFrame, columns: Sequence[str], where: Any) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{where}: no column {missing}")


def _save(df: pd.DataFrame, cache: str | Path) -> None:
    """Write next to the target first, then rename, so a stopped write leaves no broken cache."""
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    part = cache.with_name(cache.name + ".part")
    try:
        df.to_parquet(part, index=False)
        os.replace(part, cache)
    finally:
        part.unlink(missing_ok=True)


def build_s2_index(cache: str | Path | None = None, token: str | None = None,
                   shards: Sequence[str] | None = None, workers: int = 16,
                   progress: bool = True, verify: int = 3) -> pd.DataFrame:
    """One row per Sentinel-2 cell, read from the footers of the shards.

    Columns grid_cell, product_id, product_datetime (UTC), parquet_url, parquet_row.

    cache     one parquet file. If it exists it is returned as it is, else it is written.
              Use one cache file per list of shards.
    shards    urls or local paths, None = all shards of the repository
    workers   threads, one footer each (2 to 5 MB per shard, two requests)
    verify    this many rows, spread over the rows that came from footers, are read from
              the shards and compared with the footer values. A difference is an error.

    A shard whose footer statistics are missing or cannot be trusted is read by its three
    text columns instead (one small request per row, slow). `attrs["read_by_columns"]` lists
    these shards. It is saved with the cache, and it is None when a cache does not say.
    A row without a grid_cell or a product_id is an error, it is not left out.
    """
    if cache is not None and Path(cache).is_file():
        df = pd.read_parquet(cache)
        _check_columns(df, S2_INDEX_COLUMNS, cache)
        if shards is not None and set(df["parquet_url"]) != {str(s) for s in shards}:
            raise ValueError(f"{cache}: this cache was built from other shards")
        if progress:
            print(f"Sentinel-2 index from {cache}, {len(df):,} rows, "
                  f"{df['parquet_url'].nunique()} shards")
        by_columns = df.attrs.get("read_by_columns")
        df = df[list(S2_INDEX_COLUMNS)]
        df.attrs["read_by_columns"] = None if by_columns is None else list(by_columns)
        return df

    urls = [str(s) for s in shards] if shards is not None else list_s2_shards(token)
    if not urls:
        raise ValueError("no shards given")
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        jobs = pool.map(lambda u: _shard_index(u, token), urls)
        if progress:
            from tqdm.auto import tqdm
            jobs = tqdm(jobs, total=len(urls), desc="Sentinel-2 footers", unit="shard")
        done = list(jobs)
    df = pd.concat([d for d, _ in done], ignore_index=True)
    by_columns = [u for u, (_, how) in zip(urls, done) if how == "columns"]

    # only rows that came from a footer, the other rows already are the stored values
    rows = np.flatnonzero(~df["parquet_url"].isin(by_columns).to_numpy())
    picks = np.linspace(0, len(rows) - 1, max(0, int(verify))).astype(int) if len(rows) else []
    picks = sorted(set(int(rows[i]) for i in picks))
    for i in picks:
        row = df.iloc[i]
        stored = read_row(row["parquet_url"], int(row["parquet_row"]), META_COLUMNS, token)
        same = (stored["grid_cell"] == row["grid_cell"]
                and stored["product_id"] == row["product_id"]
                and _parse_dates([stored["product_datetime"]])[0] == row["product_datetime"])
        if not same:
            raise RuntimeError(f"{row['parquet_url']} row {row['parquet_row']}: the footer "
                               "statistics differ from the stored values")
    if progress:
        print(f"Sentinel-2 index, {len(df):,} rows from {len(urls)} shards, {len(by_columns)} "
              f"shards read by columns, {len(picks)} rows compared with the stored values")
    df.attrs["read_by_columns"] = by_columns
    if cache is not None:
        _save(df, cache)
    return df


def load_s1_index(cache: str | Path | None = None, token: str | None = None,
                  source: str | Path | None = None) -> pd.DataFrame:
    """One row per Sentinel-1 cell, from metadata.parquet of Core-S1RTC (about 80 MB).

    Columns grid_cell, product_id, product_datetime (UTC), parquet_url, parquet_row, nodata,
    orbit_state, centre_lat, centre_lon, crs.

    cache     one parquet file. If it exists it is returned as it is, else it is written.
    source    url or local path of the metadata file, None = the file of the repository

    Two columns are changed, the others are as stored. `timestamp` is parsed and renamed to
    product_datetime. `crs` is stored as EPSG:EPSG:32752, the doubled prefix is removed.
    """
    if cache is not None and Path(cache).is_file():
        df = pd.read_parquet(cache)
        _check_columns(df, S1_INDEX_COLUMNS, cache)
        return df[list(S1_INDEX_COLUMNS)]

    if source is None:
        source = f"{HF}/datasets/{S1_REPO}/resolve/main/metadata.parquet"
    source = str(source)
    handle = pa.BufferReader(_get(source, token)[0]) if _is_http(source) else source
    file = pq.ParquetFile(handle)
    missing = [c for c in S1_METADATA_COLUMNS if c not in file.schema_arrow.names]
    if missing:
        raise KeyError(f"{source}: no column {missing}, the file has {file.schema_arrow.names}")
    df = file.read(columns=list(S1_METADATA_COLUMNS)).to_pandas()
    df = df.rename(columns={"timestamp": "product_datetime"})
    try:
        df["product_datetime"] = _parse_dates(df["product_datetime"])
    except (ValueError, TypeError):
        raise ValueError(f"{source}: a stored timestamp is not like 20221115T161819") from None
    df["crs"] = df["crs"].str.replace("EPSG:EPSG:", "EPSG:", regex=False)
    df = df[list(S1_INDEX_COLUMNS)]
    if cache is not None:
        _save(df, cache)
    return df


# ------------------------------------------------------------------ one cell

def decode_tif(data: bytes) -> tuple[np.ndarray, tuple, str, float | None]:
    """One GeoTIFF from memory, returns (array, transform6, crs_string, nodata).

    array        (h, w) for one band, (bands, h, w) for more, dtype as stored
    transform6   the affine (a, b, c, d, e, f) in the units of the CRS,
                 x = a * col + b * row + c and y = d * col + e * row + f,
                 (c, f) is the outer corner of the upper left pixel
    crs_string   "EPSG:32633" when the CRS has an EPSG code, else its WKT, "" when there is none
    nodata       the no-data value written in the file, None when there is none
    """
    import rasterio   # not shipped by Colab, the notebook installs it

    begin = time.perf_counter()
    try:
        with rasterio.MemoryFile(bytes(data)) as memory, memory.open() as src:
            array = src.read()
            transform = tuple(float(v) for v in tuple(src.transform)[:6])
            crs = src.crs.to_string() if src.crs else ""
            nodata = src.nodata
    except rasterio.errors.RasterioIOError:
        raise ValueError(f"these {len(data)} bytes are not an image that can be read") from None
    if array.shape[0] == 1:
        array = array[0]
    _add(decoded=1, decode_seconds=time.perf_counter() - begin)
    return array, transform, crs, nodata


def _same_cell(row: Mapping[str, Any], stored: Mapping[str, Any]) -> None:
    """The shard must hold the cell that the index row names."""
    for key in ("grid_cell", "product_id"):
        if str(stored[key]) != str(row[key]):
            raise ValueError(f"{row['parquet_url']} row {row['parquet_row']}: {key} is "
                             f"{stored[key]!r} in the shard and {row[key]!r} in the index")


def _stored_date(stored: Mapping[str, Any], row: Mapping[str, Any]) -> pd.Timestamp:
    try:
        return _parse_dates([stored["product_datetime"]])[0]
    except (ValueError, TypeError):
        raise ValueError(f"{row['parquet_url']} row {row['parquet_row']}: product_datetime "
                         f"{stored['product_datetime']!r} is not like 20221115T161819") from None


def read_s2_cell(row: Mapping[str, Any], token: str | None = None) -> S2Cell:
    """One Sentinel-2 cell, 12 bands at native resolution and the cloud mask, in memory.

    row = mapping with grid_cell, product_id, product_datetime, parquet_url, parquet_row.
    The date of the cell is the one stored in the shard. Sizes are NOT checked here (the
    pairing rules do that), but every band must be one uint16 layer, north up, in one CRS.
    The cloud mask must lie on the grid of B02 (same CRS, same six numbers of the transform).
    The cell does not carry the transform of its mask, so nothing later could check that.
    """
    where = f"{row['parquet_url']} row {row['parquet_row']}"
    stored = read_row(row["parquet_url"], int(row["parquet_row"]),
                      [*META_COLUMNS, *S2_BANDS, "cloud_mask"], token)
    _same_cell(row, stored)
    bands, transforms, systems = {}, {}, set()
    for name in S2_BANDS:
        if stored[name] is None:
            raise ValueError(f"{where}: band {name} is empty")
        array, transform, crs, _ = decode_tif(stored[name])
        if array.ndim != 2 or array.dtype != np.uint16:
            raise ValueError(f"{where}: band {name} is {array.dtype} {array.shape}, expected one "
                             "uint16 layer")
        if transform[1] != 0 or transform[3] != 0:
            raise ValueError(f"{where}: band {name} is not north up, transform {transform}")
        bands[name], transforms[name] = array, transform
        systems.add(crs)
    if len(systems) != 1 or "" in systems:
        raise ValueError(f"{where}: the bands do not share one CRS, found {sorted(systems)}")
    mask = None
    if stored["cloud_mask"] is not None:
        mask, transform, crs, _ = decode_tif(stored["cloud_mask"])
        if mask.ndim != 2 or mask.dtype != np.uint8:
            raise ValueError(f"{where}: cloud_mask is {mask.dtype} {mask.shape}, expected one "
                             "uint8 layer")
        apart = max(abs(m - b) for m, b in zip(transform, transforms["B02"], strict=True))
        if crs not in systems or not apart <= GRID_TOL_M:
            raise ValueError(f"{where}: cloud_mask is not on the grid of B02, {transform} "
                             f"{crs!r} and {transforms['B02']} {sorted(systems)}")
    a, _, c, _, e, f = transforms["B02"]
    h, w = bands["B02"].shape
    ys, xs = (f, f + e * h), (c, c + a * w)
    return S2Cell(grid_cell=str(stored["grid_cell"]), product_id=str(stored["product_id"]),
                  datetime=_stored_date(stored, row), bands=bands, transforms=transforms,
                  crs=systems.pop(), cloud_mask=mask,
                  bounds=(min(xs), min(ys), max(xs), max(ys)))


def read_s1_cell(row: Mapping[str, Any], token: str | None = None) -> S1Cell:
    """One Sentinel-1 cell, VV and VH in linear power as stored, in memory.

    row = mapping with grid_cell, product_id, product_datetime, parquet_url, parquet_row and,
    if known, orbit_state. The date of the cell is the one stored in the shard (the start of
    the product, a few seconds before the `timestamp` of the metadata file). `nodata` is the
    value written in the GeoTIFF. The size is NOT checked here (the pairing rules do that),
    but VV and VH must be float32 layers on the same grid.
    """
    where = f"{row['parquet_url']} row {row['parquet_row']}"
    stored = read_row(row["parquet_url"], int(row["parquet_row"]),
                      [*META_COLUMNS, "vv", "vh"], token)
    _same_cell(row, stored)
    if stored["vv"] is None or stored["vh"] is None:
        raise ValueError(f"{where}: vv or vh is empty")
    vv, transform, crs, nodata = decode_tif(stored["vv"])
    vh, transform_vh, crs_vh, nodata_vh = decode_tif(stored["vh"])
    for name, array in (("vv", vv), ("vh", vh)):
        if array.ndim != 2 or array.dtype != np.float32:
            raise ValueError(f"{where}: {name} is {array.dtype} {array.shape}, expected one "
                             "float32 layer")
    if vv.shape != vh.shape or transform != transform_vh or crs != crs_vh or not crs:
        raise ValueError(f"{where}: vv and vh are not on the same grid, {vv.shape} {transform} "
                         f"{crs!r} and {vh.shape} {transform_vh} {crs_vh!r}")
    both_nan = None not in (nodata, nodata_vh) and np.isnan(nodata) and np.isnan(nodata_vh)
    if nodata != nodata_vh and not both_nan:
        raise ValueError(f"{where}: vv and vh name different no-data values, {nodata} and "
                         f"{nodata_vh}")
    orbit = row.get("orbit_state") if hasattr(row, "get") else None
    return S1Cell(grid_cell=str(stored["grid_cell"]), product_id=str(stored["product_id"]),
                  datetime=_stored_date(stored, row), vv=vv, vh=vh, transform=transform, crs=crs,
                  nodata=None if nodata is None else float(nodata),
                  orbit_state=None if orbit is None or pd.isna(orbit) else str(orbit))
