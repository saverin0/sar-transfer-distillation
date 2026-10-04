"""The measurements and checks of stage 0. Nothing is trained or scored.

Measured
- `runtime_report`     GPU, RAM and free disk of this runtime
- `fetch_pairs`        seconds per step and megabytes per cell while the pairs are fetched,
                       footers, rows and decoding apart
- `cloud_report`       the codes of the cloud masks and how many crops each one fills
- `time_teacher`       images per second and peak GPU memory of one frozen teacher
- `time_student`       the same for one full training step of the student, in each mode
- `budget_table`       hours and storage of the two teacher routes, from the measured speeds,
                       and the time and download of the data fetch

Checked
- `offset_check`       check 1, the Sentinel-2 offset, with `offset_by_year`, `offset_verdict`
                       and `offset_by_baseline_table` (the grouping that decided the rule)
- `s1_stats`           check 2, Sentinel-1 mean and SD in dB next to TerraMind's constants
- `gap_report`         check 3, the date gap of every pair
- `feature_pictures`   checks 4 and 5, patch features of one teacher under each input rule

The rules of the three data checks are the constants below. They were fixed BEFORE any image
was read, each one names its source.

Units. Megabytes are 10^6 bytes and GB are 10^9 bytes, as. Times are
seconds, budgets are hours. Sentinel-1 is dB, Sentinel-2 is the stored digital number (DN).

The modules majortom, pairing and student, and matplotlib, psutil and tqdm, are imported
inside the functions that need them.

`fetch_pairs` takes footers, rows and decoding apart with the counters of majortom
(`majortom.STATS`, a copy before the run and one after it). No private name is used.
"""

from __future__ import annotations

import datetime as dt
import gc
import json
import math
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .records import S1_CHANNELS, S2_BANDS, Skip
from .teachers.base import Teacher, TeacherOut, band_index

MB = 1e6
GB = 1e9

WORK_FOLDERS = ("/content", "/content/drive/MyDrive")      # Colab, the runtime disk and Drive

SENSORS = ("s2", "s1")
FETCH_STEPS = ("read_s2", "read_s1", "make_pair", "save")
TIMING_COLUMNS = ("grid_cell", "status", "read_s2_s", "read_s1_s", "make_pair_s", "save_s",
                  "total_s", "mb_decoded", "mb_saved")
SKIP_COLUMNS = ("grid_cell", "reason", "detail")
FETCH_BUDGET = ("fetch_cells", "fetch_shards", "fetch_hours", "fetch_gb")

# Parquet files of the two sets. The most shards a full run can touch.
ALL_SHARDS = {"s2": 498, "s1": 2930}
MAX_CLOUD = 0.10                 # the cloud rule, share of the crop

# ---- check 1, the Sentinel-2 offset. Fixed before any image was read.
OFFSET_BANDS = ("B02", "B04", "B08")
OFFSET_PERCENTILES = (0.1, 1.0, 50.0)
OFFSET_MIN_DN = 950              # lowest 0.1 percentile of an image that carries the offset
OFFSET_FROM = "2022-01-25"       # first day of processing baseline 04.00, 00:00 UTC
OFFSET_SHARE_HIGH = 0.90
OFFSET_SHARE_LOW = 0.10
OFFSET_UNCLEAR = "unclear, the user decides"
OFFSET_SOURCE = (
    "Copernicus SentiWiki, S2 Products, read 2026-09-29, "
    "https://sentiwiki.copernicus.eu/web/s2-products - starting with the Processing Baseline "
    "04.00 (25th January 2022) the Level-2A values are shifted by BOA_ADD_OFFSET, reflectance "
    "= (DN + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE, DN 0 stays no-data. The page gives no "
    "number for the offset. In its part on quality checks the same page names January 24th "
    "2022 for the same baseline. The 25th is also on the page S2 Processing, "
    "https://sentiwiki.copernicus.eu/web/s2-processing (in operations since 25 January 2022), "
    "which also says that reprocessed products of older images (baselines 05.00 to 05.11) "
    "carry the offset, so the offset belongs to the baseline and not to the day the image "
    "was taken. The number -1000 at Level-2A is from the post of the Sentinel-2 "
    "technical manager of the Optical Mission Performance Cluster on the ESA STEP forum, "
    "https://forum.step.esa.int/t/info-introduction-of-additional-radiometric-offset-in-"
    "pb04-00-products/35431 . The same SentiWiki page gives the product name as "
    "MMM_MSIXXX_YYYYMMDDHHMMSS_Nxxyy_ROOO_Txxxxx_<Product Discriminator>, Nxxyy is the "
    "processing baseline.")

# ---- check 2, Sentinel-1 statistics. Fixed before any image was read.
TERRAMIND_S1RTC = {"mean_db": (-10.93, -17.329), "sd_db": (4.391, 4.459)}      # (VV, VH)
S1_MAX_MEAN_DIFF_DB = 3.0
S1_SD_RATIO = (0.5, 2.0)
S1_FLAG = "unit error suspected"
TERRAMIND_SOURCE = (
    "TerraMind pre-training statistics, key untok_sen1rtc@224 of v1_pretraining_mean and "
    "v1_pretraining_std, band order VV, VH (PRETRAINED_BANDS), file terratorch/models/"
    "backbones/terramind/model/terramind_register.py of github.com/torchgeo/terratorch (the "
    "addresses github.com/IBM/terratorch and github.com/terrastackai/terratorch lead to the "
    "same repository), last changed in commit 49d8d9986af4, read 2026-09-29. The TerraMind "
    "model card sends the reader to this file for the statistics, under the terrastackai "
    "address. The file does not say dB, the means are below "
    "zero, so they cannot be linear power. That the RTC images are the Major-TOM ones is "
    "NOT said in one sentence on a page that was read. The TerraMesh card names SSL4EO-S12 "
    "v1.1 and MajorTOM-Core as its two sources, the TerraMesh paper says each of them holds "
    "one Sentinel-1 product, and the Major-TOM set is named Core-S1RTC.")

# ---- check 3, the date gap. The limit is the pairing rule.
MAX_GAP_DAYS = 7.0
GAP_EDGES_DAYS = (1.0, 2.0, 3.0, 5.0, 7.0, 14.0, 30.0, 90.0, 365.0)    # our choice of bins

# ---- timing. Our own settings, none of them is a fact about a model.
WARMUP = 2                       # calls before the clock starts, not counted
TIMING_LR = 1e-4                 # AdamW needs one, it has no effect on the time
DN_MAX = 10000.0                 # random teacher inputs are drawn between 0 and this

# ---- pictures. Our own settings, for display only.
STRETCH_PERCENTILES = (1.0, 99.0)
PREVIEW_BANDS = ("B04", "B03", "B02")
PICTURE_CELL_IN = 2.2            # side of one picture in inches
PICTURE_DPI = 150

FLOAT16_BYTES = 2                # stored teacher features are float16

_OOM = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


# --------------------------------------------------------------------------- runtime

def runtime_report(folders: Sequence[str | os.PathLike] | None = None) -> dict:
    """GPU name and memory, system RAM and free disk, as this runtime reports them.

    folders   the working folders. None = the current folder and, where they exist, the
              runtime disk and Drive of Colab (`WORK_FOLDERS`).

    Sizes are GB = 10^9 bytes. Without a GPU the three GPU entries are None. A folder that
    cannot be read gets None for both sizes, it is not left out.
    """
    import psutil

    if folders is None:
        folders = [os.getcwd(), *(f for f in WORK_FOLDERS if os.path.isdir(f))]
    disk = {}
    for folder in folders:
        try:
            use = shutil.disk_usage(folder)
            disk[str(folder)] = {"total_gb": use.total / GB, "free_gb": use.free / GB}
        except OSError:
            disk[str(folder)] = {"total_gb": None, "free_gb": None}
    ram = psutil.virtual_memory()
    report = {"gpu": None, "gpu_memory_gb": None, "gpu_free_gb": None,
              "ram_gb": ram.total / GB, "ram_free_gb": ram.available / GB,
              "cpus": os.cpu_count(), "disk": disk}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        report.update(gpu=torch.cuda.get_device_name(0), gpu_memory_gb=total / GB,
                      gpu_free_gb=free / GB)
    return report


# --------------------------------------------------------------------------- fetch

def _cell_bytes(cell: Any) -> int:
    """Bytes of the arrays of one fetched cell, as they sit in memory."""
    arrays = [*getattr(cell, "bands", {}).values(),
              *(getattr(cell, key, None) for key in ("cloud_mask", "vv", "vh"))]
    return sum(a.nbytes for a in arrays if isinstance(a, np.ndarray))


def _fetch_one(row: Mapping, path: Path, cloud_codes: list[int] | None,
               token: str | None, offset: str | None = None) -> dict:
    """One row of `pairs`. Returns its timing row with reason and detail.

    An error of a cell makes it a skip. A library that is missing is no error of a cell,
    it is raised.
    """
    from . import majortom, pairing

    out: dict[str, Any] = dict.fromkeys(TIMING_COLUMNS, float("nan"))
    out.update(grid_cell=row["grid_cell"], status="skipped", reason="", detail="")
    if path.is_file():
        out.update(status="on_disk", mb_saved=path.stat().st_size / MB)
        return out

    step = FETCH_STEPS[0]

    def timed(name: str, fn: Callable, *args, **kwargs) -> Any:
        nonlocal step
        step = name
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        out[f"{name}_s"] = time.perf_counter() - start
        return result

    start = time.perf_counter()
    try:
        s2 = timed("read_s2", majortom.read_s2_cell, pairing.s2_row_of(row), token=token)
        s1 = timed("read_s1", majortom.read_s1_cell, pairing.s1_row_of(row), token=token)
        out["mb_decoded"] = (_cell_bytes(s2) + _cell_bytes(s1)) / MB
        made = timed("make_pair", pairing.make_pair, s2, s1, cloud_codes=cloud_codes,
                     offset=offset)
        if isinstance(made, Skip):
            out.update(reason=made.reason, detail=made.detail)
        else:
            timed("save", pairing.save_pair, made, path)
            out.update(status="saved", mb_saved=path.stat().st_size / MB)
    except ImportError:                 # the runtime lacks a library, every cell would fail
        raise
    except Exception as err:            # one failing cell never stops the run
        detail = f"{step} failed, {type(err).__name__}, {err}"
        if token:
            detail = detail.replace(token, "<token>")
        out.update(status="skipped", reason="read_error", detail=detail)
    out["total_s"] = time.perf_counter() - start
    return out


def _cloud_codes(cloud_codes: Iterable[int] | None) -> list[int] | None:
    """The cloud codes as a sorted list without doubles. None = the rule is off.

    Whole numbers from 0 to 255 and at least one, as `pairing.make_pair` asks for. It is
    checked here before a cell is read. Left to `make_pair`, the error would come after
    the download and would turn every cell into a skip.
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


def _cloud_rule(codes: list[int] | None) -> tuple[str, list[int] | None]:
    """The cloud rule as `pairing.make_pair` writes it into the meta of a pair."""
    return ("not applied", None) if codes is None else ("applied", list(codes))


def _per(total: float | None, n: float | None) -> float | None:
    """total / n, None when one of them is not known or n is 0."""
    return None if total is None or not n else total / n


def fetch_pairs(pairs: pd.DataFrame, out_dir: str | os.PathLike,
                cloud_codes: Iterable[int] | None = None, workers: int = 8,
                token: str | None = None, progress: bool = True, offset: str | None = None,
                ) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
    """Fetch the cells of `pairs`, make the pairs and save them, one .npz per pair.

    pairs        rows of `pairing.pair_index` or `pairing.sample_pairs`, one per grid cell
    out_dir      local folder, a pair is saved as <out_dir>/<grid_cell>.npz
    cloud_codes  passed to `pairing.make_pair`, None = the cloud rule is not applied
    workers      threads, one cell each
    offset       passed to `pairing.make_pair`. None (the default) stores the Sentinel-2
                 values as read, as the stage 0 notebook did. "baseline" takes the +1000
                 offset off by processing baseline, the rule decided on 2026-09-30, so the
                 pairs of a folder are on one scale. Anything else is a ValueError before a
                 cell is read. The pairs on disk are NOT checked against it, use another
                 folder for another rule.

    Resumable. A pair that is already on disk is not read again, it is listed with the
    status "on_disk". A cell that was skipped left no file, so it is read again.
    One failing cell never stops the run. An error in any step makes that cell a skip with
    the reason read_error, and the detail names the step and the error.
    The token is never written into a table.

    What stops the call, because it is no fault of a cell
    - before anything is read. A grid cell that is in the table twice (ValueError), a table
      without the columns of `pairing.pair_index` (KeyError), cloud codes that are not
      whole numbers from 0 to 255 or an empty list (ValueError), and a pair on disk that
      cannot be read or was made under another cloud rule than the one asked for
      (ValueError). Such a pair is never handed on unchecked, use another folder for
      another rule.
    - while it runs. A library that is missing (ImportError). The cells that have not
      started are dropped, the pairs saved so far stay on disk.

    Returns (paths, skips, timing).
    paths    the pair files of this table, saved now or on disk before, in table order
    skips    columns grid_cell, reason (one of `records.SKIP_REASONS`), detail
    timing   one row per cell, columns
             status       saved, skipped or on_disk
             read_s2_s, read_s1_s, make_pair_s, save_s     seconds per step, nan = not reached
             total_s      seconds for the cell
             mb_decoded   megabytes of the two cells as arrays in memory
             mb_saved     megabytes of the pair file
             A read is the fetch of the row, its decoding and, when the footer of the shard
             was not in memory, the fetch of that footer. The totals take them apart.
    timing.attrs holds the totals of this call
        wall_s, thread_s (sum of total_s), workers, cells, cells_read (not on disk before),
        cells_saved, cells_per_s (cells_read / wall_s)
        shards_s2, shards_s1      different shards of the cells that were read
        counted                   everything `majortom.STATS` counted during the call
        and from these counters, None when majortom does not count them
        footers, footer_s, mb_footers     footers fetched, their seconds and megabytes
        row_s, mb_rows                    the rows of the cells, without any footer
        decode_s                          decoding the GeoTIFFs
        wait_s, retries                   waits before a request was tried again
        requests, mb_fetched              all reads, footers inside
        s_per_footer, mb_per_footer       footer_s and mb_footers per footer
        mb_rows_per_cell                  mb_rows / cells_read
        mb_fetched_per_cell               mb_fetched / cells_read, footers inside
        s_per_cell                        thread seconds per cell that was read, by step
                                          fetch_rows, decode, make_pair, save, wait and rest
                                          (all that is left of the time without footers,
                                          such as unpacking the row and the checks)

    How to read the numbers. All seconds but wall_s are measured inside the threads and
    summed, so with 8 threads the run takes about 1/8 of thread_s, wall_s is the real time.
    A footer is needed once per shard and a row once per cell. A sample of 500 cells
    touches almost as many shards, so nearly every cell pays for a footer (2 to 5 MB for
    Sentinel-2). A full run has many cells per shard. That is why the footers are counted
    apart, and why `budget_table` counts them per shard and the rows per cell.
    mb_fetched_per_cell is what THIS sample cost, it is too high for a full run.
    """
    from . import majortom, pairing

    if pairs["grid_cell"].duplicated().any():
        twice = pairs.loc[pairs["grid_cell"].duplicated(), "grid_cell"].iloc[0]
        raise ValueError(f"grid cell {twice} is in the table more than once")
    rows = pairs.to_dict("records")
    if rows:
        try:
            pairing.s2_row_of(rows[0]), pairing.s1_row_of(rows[0])
        except KeyError as err:
            raise KeyError(f"pairs has no column {err}, it must be a table of "
                           "pairing.pair_index") from None
    codes = _cloud_codes(cloud_codes)
    if offset not in pairing.OFFSET_RULES:
        raise ValueError(f"offset {offset!r} must be one of {pairing.OFFSET_RULES}")
    out_dir = Path(out_dir)
    for row in rows:
        path = out_dir / f"{row['grid_cell']}.npz"
        if path.is_file():
            try:
                meta = pairing.load_pair(path).meta
            except Exception as err:
                raise ValueError(f"{path} is on disk and cannot be read, {type(err).__name__}, "
                                 f"{err}. Delete it, the cell is then read again") from None
            found = (meta.get("cloud_rule"), meta.get("cloud_codes"))
            if found != _cloud_rule(codes):
                raise ValueError(f"{path} was made with the cloud rule {found}, asked for is "
                                 f"{_cloud_rule(codes)}. Use another folder for another rule")
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(workers))

    counted = getattr(majortom, "STATS", None)
    before = dict(counted) if counted else {}
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # when a cell raises, `map` drops the cells that have not started
        jobs = pool.map(
            lambda row: _fetch_one(row, out_dir / f"{row['grid_cell']}.npz", codes, token, offset),
            rows)
        if progress:
            from tqdm.auto import tqdm
            jobs = tqdm(jobs, total=len(rows), desc="pairs", unit="cell")
        done = list(jobs)
    wall = time.perf_counter() - start
    delta = {name: counted[name] - before[name] for name in before} if counted else None

    table = pd.DataFrame(done, columns=[*TIMING_COLUMNS, "reason", "detail"])
    kept = table["status"] != "skipped"
    paths = [out_dir / f"{cell}.npz" for cell in table.loc[kept, "grid_cell"]]
    skips = table.loc[~kept, list(SKIP_COLUMNS)].reset_index(drop=True)
    timing = table[list(TIMING_COLUMNS)].copy()
    was_read = (table["status"] != "on_disk").to_numpy()
    read = int(was_read.sum())
    thread_s = float(timing["total_s"].sum())

    def of(name: str) -> float | None:
        """One counter of majortom, None when it is not counted."""
        return None if not delta or name not in delta else delta[name]

    totals: dict[str, Any] = {
        "wall_s": wall, "thread_s": thread_s, "workers": workers, "cells": len(rows),
        "cells_read": read, "cells_saved": int((table["status"] == "saved").sum()),
        "cells_per_s": read / wall if wall > 0 else None}
    for sensor in SENSORS:
        urls = pairs[f"{sensor}_url"].to_numpy()[was_read] if rows else []
        totals[f"shards_{sensor}"] = len(set(urls))
    totals.update(
        counted=delta, footers=of("footers"), footer_s=of("footer_seconds"),
        mb_footers=_per(of("footer_bytes"), MB), row_s=of("row_seconds"),
        mb_rows=_per(of("row_bytes"), MB), decode_s=of("decode_seconds"),
        wait_s=of("wait_seconds"), retries=of("retries"), requests=of("requests"),
        mb_fetched=_per(of("bytes"), MB))
    totals.update(
        s_per_footer=_per(totals["footer_s"], totals["footers"]),
        mb_per_footer=_per(totals["mb_footers"], totals["footers"]),
        mb_rows_per_cell=_per(totals["mb_rows"], read),
        mb_fetched_per_cell=_per(totals["mb_fetched"], read))
    steps = {"fetch_rows": totals["row_s"], "decode": totals["decode_s"],
             "make_pair": float(timing["make_pair_s"].sum()),
             "save": float(timing["save_s"].sum()), "wait": totals["wait_s"]}
    known = [totals["footer_s"], *steps.values()]
    steps["rest"] = None if None in known else thread_s - sum(known)
    totals["s_per_cell"] = {name: _per(value, read) for name, value in steps.items()}
    timing.attrs = totals
    return paths, skips, timing


# --------------------------------------------------------------------------- cloud codes

def cloud_report(pair_paths: Iterable[str | os.PathLike],
                 cloud_codes: Iterable[int] | None = None) -> pd.DataFrame:
    """The codes of the cloud masks, how common each one is and how many crops it fills.

    Stage 0 counts the codes. No page gives their meaning, so no
    code is called cloud here. `cloud_codes` adds one row for these codes taken together,
    which is what `pairing.make_pair` checks when it gets the same codes.

    One row per code that was seen, sorted, then the row of `cloud_codes` if given, then
    the row "no cloud mask". Columns
        what           "code 3", "codes 2 3 together", "no cloud mask"
        pixels         pixels with the code, over all masks
        pixel_share    pixels / all pixels of all masks
        images         pairs whose mask holds the code at least once
        images_over    pairs where the code fills more than 10 % of the crop (`MAX_CLOUD`),
                       these are skipped when the code counts as cloud
        image_share    images_over / pairs with a mask
    The row "no cloud mask" counts the pairs without a mask in `images`, the rest is 0.
    """
    from .pairing import load_pair

    codes = _cloud_codes(cloud_codes)
    pixels: dict[Any, int] = {}
    images: dict[Any, int] = {}
    over: dict[Any, int] = {}
    n_pixels = n_masks = n_without = 0

    def count(key: Any, n: int, size: int) -> None:
        pixels[key] = pixels.get(key, 0) + n
        images[key] = images.get(key, 0) + (n > 0)
        over[key] = over.get(key, 0) + (n / size > MAX_CLOUD)

    for path in pair_paths:
        mask = load_pair(path).cloud_mask
        if mask is None:
            n_without += 1
            continue
        n_masks += 1
        n_pixels += mask.size
        values, counts = np.unique(mask, return_counts=True)
        for value, n in zip(values.tolist(), counts.tolist()):
            count(int(value), n, mask.size)
        if codes is not None:
            count("set", int(np.isin(mask, codes).sum()), mask.size)

    keys = sorted(k for k in pixels if k != "set") + (["set"] if "set" in pixels else [])
    rows = []
    for key in keys:
        what = f"code {key}" if key != "set" else f"codes {' '.join(map(str, codes))} together"
        rows.append({"what": what, "pixels": pixels[key], "pixel_share": pixels[key] / n_pixels,
                     "images": images[key], "images_over": over[key],
                     "image_share": over[key] / n_masks})
    rows.append({"what": "no cloud mask", "pixels": 0, "pixel_share": 0.0, "images": n_without,
                 "images_over": 0, "image_share": 0.0})
    return pd.DataFrame(rows, columns=["what", "pixels", "pixel_share", "images", "images_over",
                                       "image_share"])


# --------------------------------------------------------------------------- check 1

def _utc(value: Any) -> pd.Timestamp:
    """One date as a UTC timestamp. A date without a time zone is read as UTC."""
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _baseline(product_id: Any) -> str | None:
    """The processing baseline in a product name, _N0510_ gives "05.10". None when absent."""
    found = re.search(r"_N(\d{2})(\d{2})_", str(product_id or ""))
    return f"{found.group(1)}.{found.group(2)}" if found else None


def offset_check(pair_paths: Iterable[str | os.PathLike]) -> pd.DataFrame:
    """Check 1, the low values of B02, B04 and B08 of every Sentinel-2 image.

    The rule, FIXED before any image was read
    - Only non-zero pixels count, 0 is no-data.
    - An image CARRIES THE OFFSET when the 0.1 percentile of ALL three bands is at least
      950 (`OFFSET_MIN_DN`). With the offset a reflectance of 0 is stored as 1000, so only
      the few pixels with a reflectance below -0.005 lie under 950.
    - The images are split at the first day of processing baseline 04.00, 25 January 2022,
      00:00 UTC (`OFFSET_FROM`, source in `OFFSET_SOURCE`). "before" is before that day,
      "from" is that day or later.
    - `offset_verdict` turns the share of images that carry the offset, per period, into
      the verdict.

    One row per image, columns
        grid_cell, s2_product_id, baseline, s2_datetime (UTC), year, period, nonzero_px,
        <band>_p0.1, <band>_p1, <band>_median, <band>_mean   for B02, B04 and B08, in DN,
        carries_offset
    nonzero_px is the count of non-zero pixels of the band that has the fewest. An image
    with nonzero_px = 0 has nan values, never carries the offset and is left out of every
    share. baseline is read from the product name (_N0510_ gives 05.10) and is missing
    when the name has none. It is for the reader, the verdict does not use it.
    Percentiles are interpolated linearly.
    """
    from .pairing import load_pair

    first_day = _utc(OFFSET_FROM)
    rows = []
    for path in pair_paths:
        pair = load_pair(path)
        when = _utc(pair.s2_datetime)
        product = pair.meta.get("s2_product_id")
        row = {"grid_cell": pair.grid_cell, "s2_product_id": product,
               "baseline": _baseline(product), "s2_datetime": when, "year": int(when.year),
               "period": "from" if when >= first_day else "before", "nonzero_px": 0}
        counts, lows = [], []
        for name in OFFSET_BANDS:
            values = np.asarray(pair.s2[name])
            values = values[values != 0]
            stats = [float("nan")] * 4
            if values.size:
                stats = [*np.percentile(values, OFFSET_PERCENTILES),
                         values.mean(dtype=np.float64)]
            row.update(zip((f"{name}_p0.1", f"{name}_p1", f"{name}_median", f"{name}_mean"),
                           (float(v) for v in stats)))
            counts.append(int(values.size))
            lows.append(stats[0])
        row["nonzero_px"] = min(counts)
        row["carries_offset"] = bool(min(counts) > 0 and min(lows) >= OFFSET_MIN_DN)
        rows.append(row)
    columns = ["grid_cell", "s2_product_id", "baseline", "s2_datetime", "year", "period",
               "nonzero_px",
               *(f"{b}_{s}" for b in OFFSET_BANDS for s in ("p0.1", "p1", "median", "mean")),
               "carries_offset"]
    return pd.DataFrame(rows, columns=columns)


def offset_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Per acquisition year the lowest values and the mean of B02, B04 and B08.

    `df` is the table of `offset_check`. Images without a non-zero pixel are left out.
    Columns year, images, with_offset, share_with_offset and per band, in DN
        <band>_p0.1_min      the lowest 0.1 percentile among the images of the year
        <band>_p0.1_median   the median of their 0.1 percentiles
        <band>_p1_median     the median of their 1 percentiles
        <band>_mean          the mean of their means
    """
    data = df[df["nonzero_px"] > 0]
    rows = []
    for year, group in data.groupby("year"):
        with_offset = int(group["carries_offset"].astype(bool).sum())
        row = {"year": int(year), "images": len(group), "with_offset": with_offset,
               "share_with_offset": with_offset / len(group)}
        for band in OFFSET_BANDS:
            row[f"{band}_p0.1_min"] = float(group[f"{band}_p0.1"].min())
            row[f"{band}_p0.1_median"] = float(group[f"{band}_p0.1"].median())
            row[f"{band}_p1_median"] = float(group[f"{band}_p1"].median())
            row[f"{band}_mean"] = float(group[f"{band}_mean"].mean())
        rows.append(row)
    columns = ["year", "images", "with_offset", "share_with_offset",
               *(f"{b}_{s}" for b in OFFSET_BANDS
                 for s in ("p0.1_min", "p0.1_median", "p1_median", "mean"))]
    return pd.DataFrame(rows, columns=columns)


def offset_by_baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per processing baseline how many images carry the offset. The grouping behind the rule.

    `df` is the table of `offset_check`, also as read back from offset_images.csv. The
    baseline is read from the column s2_product_id with `pairing.baseline_of` (the Nxxyy
    text, "no baseline" when the name has none). Images without a non-zero pixel are left
    out, as in `offset_by_year`. One row per baseline, sorted, "no baseline" last.
    Columns baseline, images, with_offset, share_with_offset.
    """
    from .pairing import baseline_of

    data = df[df["nonzero_px"] > 0]
    labels = data["s2_product_id"].map(lambda product: baseline_of(product) or "no baseline")
    rows = []
    for baseline, flags in data["carries_offset"].astype(bool).groupby(labels):
        with_offset = int(flags.sum())
        rows.append({"baseline": baseline, "images": int(len(flags)), "with_offset": with_offset,
                     "share_with_offset": with_offset / len(flags)})
    rows.sort(key=lambda row: (row["baseline"] == "no baseline", row["baseline"]))
    return pd.DataFrame(rows, columns=["baseline", "images", "with_offset", "share_with_offset"])


def offset_verdict(df: pd.DataFrame) -> dict:
    """The verdict of check 1 from the table of `offset_check`.

    The rule, FIXED before any image was read. share = images that carry the offset /
    images with data, per period.
        "offset present from 2022-01-25"   share from the date on is at least 0.90 AND
                                           share before the date is at most 0.10
        "no offset"                        both shares are at most 0.10
        "unclear, the user decides"        anything else, also when a period has no image

    Returns verdict, why (empty unless unclear), baseline_date, images_before, images_from,
    with_offset_before, with_offset_from, share_before, share_from (None without images),
    images_without_data, rule and source.
    """
    data = df[df["nonzero_px"] > 0]
    out: dict[str, Any] = {"verdict": OFFSET_UNCLEAR, "why": "", "baseline_date": OFFSET_FROM}
    share = {}
    for period in ("before", "from"):
        flags = data.loc[data["period"] == period, "carries_offset"].astype(bool)
        n, k = len(flags), int(flags.sum())
        share[period] = k / n if n else None
        out.update({f"images_{period}": n, f"with_offset_{period}": k,
                    f"share_{period}": share[period]})
    out["images_without_data"] = int(len(df) - len(data))

    words = {"before": "before the date", "from": "from the date on"}
    empty = [words[p] for p in words if share[p] is None]
    if empty:
        out["why"] = "no image " + " and no image ".join(empty)
    elif share["from"] >= OFFSET_SHARE_HIGH and share["before"] <= OFFSET_SHARE_LOW:
        out["verdict"] = f"offset present from {OFFSET_FROM}"
    elif share["from"] <= OFFSET_SHARE_LOW and share["before"] <= OFFSET_SHARE_LOW:
        out["verdict"] = "no offset"
    else:
        out["why"] = (f"share before the date {share['before']:.3f}, from the date on "
                      f"{share['from']:.3f}")
    out["rule"] = (f"an image carries the offset when the 0.1 percentile of "
                   f"{', '.join(OFFSET_BANDS)} over non-zero pixels is at least {OFFSET_MIN_DN} "
                   f"in all of them. Present when the share is at least {OFFSET_SHARE_HIGH:g} "
                   f"from {OFFSET_FROM} on and at most {OFFSET_SHARE_LOW:g} before. No offset "
                   f"when at most {OFFSET_SHARE_LOW:g} in both. Else unclear.")
    out["source"] = OFFSET_SOURCE
    return out


# --------------------------------------------------------------------------- check 2

def s1_flags(mean_db: Sequence[float], sd_db: Sequence[float]) -> list[str]:
    """What breaks the rule of check 2, one text per channel and rule. Empty = nothing.

    The rule, FIXED before any image was read. Per channel (VV, VH), against
    `TERRAMIND_S1RTC`
        the mean differs by MORE than 3 dB, or
        our SD / TerraMind's SD is below 0.5 or above 2
    """
    found = []
    low, high = S1_SD_RATIO
    for i, name in enumerate(S1_CHANNELS):
        diff = float(mean_db[i]) - TERRAMIND_S1RTC["mean_db"][i]
        ratio = float(sd_db[i]) / TERRAMIND_S1RTC["sd_db"][i]
        if not abs(diff) <= S1_MAX_MEAN_DIFF_DB:
            found.append(f"{name} mean differs by {diff:+.2f} dB")
        if not low <= ratio <= high:
            found.append(f"{name} SD ratio is {ratio:.2f}")
    return found


def s1_stats(pair_paths: Iterable[str | os.PathLike]) -> dict:
    """Check 2, mean and SD of Sentinel-1 in dB per channel, over all pixels of all pairs.

    Every pixel has the same weight. The SD is that of the pixels, not of image means.
    Pixels that are not finite are counted in not_finite and left out.

    Returns pairs, pixels and not_finite (per channel), mean_db, sd_db, terramind_mean_db,
    terramind_sd_db, mean_diff_db (ours - TerraMind's), sd_ratio (ours / TerraMind's),
    each a dict with the keys VV and VH, then flag ("unit error suspected" or "none"),
    flagged (the texts of `s1_flags`), rule and source.

    Why a flag means a unit error. Linear power read as dB has a mean near 0, amplitude
    turned into dB with the factor 10 has half the mean and half the SD, and the factor 20
    on power doubles both. A small gap is no error, TerraMind's numbers are from its own
    sample of cells.
    """
    from .pairing import load_pair

    n = 0
    count, bad = np.zeros(2, np.int64), np.zeros(2, np.int64)
    total, squares = np.zeros(2), np.zeros(2)
    for path in pair_paths:
        db = np.asarray(load_pair(path).s1_db)
        if db.ndim != 3 or db.shape[0] != len(S1_CHANNELS):
            raise ValueError(f"{path}: s1_db is {db.shape}, expected (2, h, w)")
        db = db.reshape(2, -1).astype(np.float64)
        ok = np.isfinite(db)
        db = np.where(ok, db, 0.0)
        count += ok.sum(1)
        bad += (~ok).sum(1)
        total += db.sum(1)
        squares += (db * db).sum(1)
        n += 1
    if n == 0 or count.min() == 0:
        raise ValueError("no pair or no finite Sentinel-1 pixel given")
    mean = total / count
    sd = np.sqrt(np.maximum(squares / count - mean ** 2, 0.0))

    def per_channel(values: Iterable) -> dict:
        return {name: v.item() if isinstance(v, np.generic) else v
                for name, v in zip(S1_CHANNELS, values)}

    ref_mean, ref_sd = TERRAMIND_S1RTC["mean_db"], TERRAMIND_S1RTC["sd_db"]
    flagged = s1_flags(mean, sd)
    return {"pairs": n, "pixels": per_channel(count), "not_finite": per_channel(bad),
            "mean_db": per_channel(mean), "sd_db": per_channel(sd),
            "terramind_mean_db": per_channel(ref_mean), "terramind_sd_db": per_channel(ref_sd),
            "mean_diff_db": per_channel(mean - np.array(ref_mean)),
            "sd_ratio": per_channel(sd / np.array(ref_sd)),
            "flag": S1_FLAG if flagged else "none", "flagged": flagged,
            "rule": (f"flag when a mean differs by more than {S1_MAX_MEAN_DIFF_DB:g} dB or an "
                     f"SD ratio is outside {S1_SD_RATIO[0]:g} to {S1_SD_RATIO[1]:g}"),
            "source": TERRAMIND_SOURCE}


# --------------------------------------------------------------------------- check 3

def gap_report(pairs: pd.DataFrame) -> pd.DataFrame:
    """Check 3, the date gap of every pair. Give the FULL table, made without a gap limit.

    `pairs` is the table of `pairing.pair_index(..., max_gap_days=None)`, it needs the
    columns gap_days, s2_datetime and s1_datetime.

    Columns kind, what, pairs, share (of all pairs).
    kind "bin"     one row per gap bin. A gap equal to an upper edge is inside the bin, so
                   7 days and 0 seconds is within 7 days, as in the pairing rule.
    kind "total"   within 7 days, more than 7 days, no date (the gap is not a number),
                   different years (the two images are from different calendar years, UTC)
                   and all pairs
    The bins and "no date" add up to all pairs.
    """
    gap = pd.to_numeric(pairs["gap_days"], errors="coerce").astype(float)
    n = len(pairs)
    rows = []
    lower = -math.inf
    for upper in (*GAP_EDGES_DAYS, math.inf):
        if lower < 0:
            what = f"0 to {upper:g} days"
        elif upper == math.inf:
            what = f"more than {lower:g} days"
        else:
            what = f"{lower:g} to {upper:g} days"
        rows.append(("bin", what, int(((gap > lower) & (gap <= upper)).sum())))
        lower = upper
    years_differ = (pd.to_datetime(pairs["s2_datetime"], utc=True).dt.year
                    != pd.to_datetime(pairs["s1_datetime"], utc=True).dt.year)
    rows += [("total", f"within {MAX_GAP_DAYS:g} days", int((gap <= MAX_GAP_DAYS).sum())),
             ("total", f"more than {MAX_GAP_DAYS:g} days", int((gap > MAX_GAP_DAYS).sum())),
             ("total", "no date", int(gap.isna().sum())),
             ("total", "different years", int((years_differ & gap.notna()).sum())),
             ("total", "all pairs", n)]
    out = pd.DataFrame(rows, columns=["kind", "what", "pairs"])
    out["share"] = out["pairs"] / n if n else float("nan")
    return out


# --------------------------------------------------------------------------- timing

def _on_gpu(device: Any) -> bool:
    return str(device).startswith("cuda")


def _sync(device: Any) -> None:
    """Wait until the GPU has finished its work. Does nothing on the CPU."""
    if _on_gpu(device):
        torch.cuda.synchronize(torch.device(device))


def _peak_reset(device: Any) -> None:
    if _on_gpu(device):
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def _peak_gb(device: Any) -> float:
    """Peak GPU memory held by tensors since the last reset, in GB. nan on the CPU."""
    if _on_gpu(device):
        return torch.cuda.max_memory_allocated(torch.device(device)) / GB
    return float("nan")


def _device_name(device: Any) -> str:
    return torch.cuda.get_device_name(torch.device(device)) if _on_gpu(device) else "cpu"


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _time_calls(call: Callable[[], Any], n: int, device: Any) -> tuple[float, float]:
    """Seconds per call and the peak GPU memory in GB (nan on the CPU).

    `WARMUP` calls come first and are not counted. The clock starts after one
    `torch.cuda.synchronize` and stops after another, so the GPU work of the counted calls
    is inside the time and the work of the warm-up is not. The peak is that of the counted
    calls. It holds all tensors on the GPU at that time, the weights and the batch too.
    """
    for _ in range(WARMUP):
        call()
    _peak_reset(device)
    _sync(device)
    start = time.perf_counter()
    for _ in range(n):
        call()
    _sync(device)
    seconds = (time.perf_counter() - start) / n
    return seconds, _peak_gb(device)


def _check_timing_input(batch_sizes: Sequence[int], n_batches: int) -> list[int]:
    sizes = [int(b) for b in batch_sizes]
    if not sizes or min(sizes) < 1:
        raise ValueError(f"batch sizes must be whole numbers above 0, got {list(batch_sizes)}")
    if int(n_batches) < 1:
        raise ValueError(f"n_batches must be at least 1, got {n_batches}")
    return sizes


MODALITIES = ("s2", "radar", "aerial")      # what a teacher can read, teachers/base.py


def random_input(modality: str, b: int, size: int, device: Any) -> torch.Tensor:
    """One random batch the way a teacher receives it - "s2" (B, 12, size, size) digital
    numbers between 0 and 10,000; "radar" (B, 2, size, size) dB around -15 with spread 5;
    "aerial" (B, 4, size, size) float 8-bit values between 0 and 255."""
    if modality == "s2":
        return torch.rand(b, len(S2_BANDS), size, size, device=device) * DN_MAX
    if modality == "radar":
        return torch.randn(b, 2, size, size, device=device) * 5.0 - 15.0
    if modality == "aerial":
        return torch.randint(0, 256, (b, 4, size, size), device=device).float()
    raise ValueError(f"unknown modality {modality!r}, known: {MODALITIES}")


def teacher_call(teacher: Teacher, modality: str) -> Callable[[torch.Tensor], Any]:
    """The reader of `teacher` for `modality` - the teacher itself for Sentinel-2, its
    `radar` or its `aerial` method (teachers/base.py). TypeError when it cannot read it."""
    if modality == "s2":
        if not getattr(teacher, "reads_s2", True):
            raise TypeError(f"{teacher.name} does not read Sentinel-2")
        return teacher
    if modality == "radar":
        if not getattr(teacher, "reads_radar", False):
            raise TypeError(f"{teacher.name} does not read Sentinel-1")
        return teacher.radar
    if modality == "aerial":
        if not getattr(teacher, "reads_aerial", False):
            raise TypeError(f"{teacher.name} gives no aerial target")
        return teacher.aerial
    raise ValueError(f"unknown modality {modality!r}, known: {MODALITIES}")


def time_teacher(teacher: Teacher, batch_sizes: Sequence[int] = (16, 64), n_batches: int = 5,
                 size: int = 256, modality: str = "s2") -> pd.DataFrame:
    """Images per second and peak GPU memory of one teacher, per batch size.

    The input is one random batch of `modality` (`random_input`, Sentinel-2 digital numbers
    by default), made on the teacher's device before the clock starts. A counted call is the
    whole call of the teacher's reader for that modality (`teacher_call`), which is the
    preparation and the forward pass. Reading data and moving it to the GPU are not inside.
    2 warm-up batches are not counted (`WARMUP`).

    One row per batch size, columns teacher, modality, rule, device, dtype, input_size,
    grid, dim, params_m (millions), batch_size, n_batches, warmup, seconds_per_batch,
    images_per_s, peak_gpu_gb (nan on the CPU), note. A batch size that does not fit the
    GPU gets nan and the note "out of memory", the other batch sizes still run.
    """
    sizes = _check_timing_input(batch_sizes, n_batches)
    n = int(n_batches)
    device = teacher.device
    call = teacher_call(teacher, modality)
    rows = []
    for b in sizes:
        row = {"teacher": teacher.name, "modality": modality, "rule": teacher.rule,
               "device": _device_name(device),
               "dtype": str(teacher.dtype).replace("torch.", ""),
               "input_size": teacher.input_size, "grid": teacher.grid, "dim": teacher.dim,
               "params_m": teacher.n_params() / 1e6, "batch_size": b, "n_batches": n,
               "warmup": WARMUP, "seconds_per_batch": float("nan"),
               "images_per_s": float("nan"), "peak_gpu_gb": float("nan"), "note": ""}
        x = None
        try:
            x = random_input(modality, b, size, device)
            seconds, peak = _time_calls(lambda: call(x), n, device)  # noqa: F821  del below
            row.update(seconds_per_batch=seconds, images_per_s=b / seconds, peak_gpu_gb=peak)
        except _OOM:
            row["note"] = "out of memory"
        del x
        _free()
        rows.append(row)
    return pd.DataFrame(rows)


def time_student(student: torch.nn.Module, heads: torch.nn.Module,
                 teacher_outs_like: Mapping[str, TeacherOut],
                 batch_sizes: Sequence[int] = (64, 128), n_batches: int = 5,
                 device: str = "cuda") -> pd.DataFrame:
    """Images per second and peak GPU memory of one full training step, per mode.

    A counted step is zero_grad, the forward pass of the student and the heads (under
    bfloat16 autocast on a GPU), `student.distil_loss` in float32, backward and one AdamW
    step over the student and the heads. The teachers do not run, their outputs are random
    tensors with the shapes of `teacher_outs_like` (name -> TeacherOut of any batch size).
    The inputs are random too, (B, 2, img, img) and (B, 12, img, img) with img =
    `student.img` (256 when the model has no such field). Reading data, moving it to the
    GPU and PHI-S are not inside. 2 warm-up steps are not counted (`WARMUP`).

    The weights of `student` and `heads` ARE changed and both are left in training mode
    on `device`. Use models that are thrown away after.

    One row per batch size and Major-TOM mode ("s1", "s2", "both"; the aerial mode of stage A
    is not timed here), columns mode, device, img, grid
    (of the student), params_m (student and heads, millions), teachers, batch_size,
    n_batches, warmup, seconds_per_step, images_per_s, peak_gpu_gb (nan on the CPU), note.
    A batch size that does not fit the GPU gets nan and the note "out of memory".
    """
    from .student import MAJORTOM_MODES, distil_loss

    sizes = _check_timing_input(batch_sizes, n_batches)
    n = int(n_batches)
    if not teacher_outs_like:
        raise ValueError("no teacher output given")
    student.to(device).train()
    heads.to(device).train()
    params = [p for module in (student, heads) for p in module.parameters()]
    optim = torch.optim.AdamW(params, lr=TIMING_LR)
    img = int(getattr(student, "img", 256))
    chans = {"s1": len(S1_CHANNELS), "s2": len(S2_BANDS)}
    seen: dict[str, int] = {}

    rows = []
    for b in sizes:
        for mode in MAJORTOM_MODES:
            row = {"mode": mode, "device": _device_name(device), "img": img, "grid": None,
                   "params_m": sum(p.numel() for p in params) / 1e6,
                   "teachers": " ".join(teacher_outs_like), "batch_size": b, "n_batches": n,
                   "warmup": WARMUP, "seconds_per_step": float("nan"),
                   "images_per_s": float("nan"), "peak_gpu_gb": float("nan"), "note": ""}
            inputs = target = None
            try:
                inputs = {k: torch.randn(b, c, img, img, device=device)
                          for k, c in chans.items() if mode in (k, "both")}
                target = {name: TeacherOut(
                              torch.randn(b, *like.summary.shape[1:], device=device),
                              torch.randn(b, *like.spatial.shape[1:], device=device))
                          for name, like in teacher_outs_like.items()}

                def step() -> None:
                    optim.zero_grad(set_to_none=True)
                    if _on_gpu(device):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            pred = heads(*student(**inputs))  # noqa: F821  del below
                    else:
                        pred = heads(*student(**inputs))  # noqa: F821  del below
                    seen["grid"] = int(next(iter(pred.values()))[1].shape[1])
                    loss, _ = distil_loss(pred, target)  # noqa: F821  del below
                    loss.backward()
                    optim.step()

                seconds, peak = _time_calls(step, n, device)
                row.update(seconds_per_step=seconds, images_per_s=b / seconds,
                           peak_gpu_gb=peak, grid=seen["grid"])
            except _OOM:
                row["note"] = "out of memory"
                optim.zero_grad(set_to_none=True)
            del inputs, target
            _free()
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- pictures

def _stretch(values: np.ndarray, lo: Any, hi: Any) -> np.ndarray:
    """Values between lo and hi as 0 to 255, uint8. Outside is clipped, hi = lo gives 0."""
    span = np.where(np.asarray(hi) > np.asarray(lo), np.asarray(hi) - np.asarray(lo), 1.0)
    return np.round(np.clip((values - lo) / span, 0.0, 1.0) * 255.0).astype(np.uint8)


def pca_rgb(spatial: torch.Tensor | np.ndarray) -> np.ndarray:
    """Patch features (B, g, g, dim) as colours, uint8 (B, g, g, 3).

    ONE principal component analysis over all patches of the batch, so the same colour
    means the same feature in every image of the batch. The features are centred, the first
    three components become red, green and blue. Each component is scaled on its own, its
    1 percentile becomes 0 and its 99 percentile becomes 255, what lies outside is clipped.
    The sign of a component is free in a PCA. Here the entry of its axis with the largest
    size is made positive, so the same features always give the same colours.

    ValueError when the array is not 4-d, holds a value that is not finite, or has fewer
    than 3 features or 3 patches. Colours of different batches or rules cannot be compared.
    """
    if isinstance(spatial, torch.Tensor):
        spatial = spatial.detach().float().cpu().numpy()
    x = np.asarray(spatial, dtype=np.float64)
    if x.ndim != 4:
        raise ValueError(f"expected (B, g, g, dim), got {x.shape}")
    flat = x.reshape(-1, x.shape[-1])
    if flat.shape[1] < 3 or flat.shape[0] < 3:
        raise ValueError(f"3 colours need at least 3 features and 3 patches, got {x.shape}")
    if not np.isfinite(flat).all():
        raise ValueError("the features hold values that are not finite")
    flat = flat - flat.mean(0)
    axes = np.linalg.svd(flat, full_matrices=False)[2][:3]                # (3, dim)
    largest = axes[np.arange(3), np.abs(axes).argmax(1)]
    axes = axes * np.where(largest < 0, -1.0, 1.0)[:, None]
    parts = flat @ axes.T                                                 # (patches, 3)
    lo, hi = np.percentile(parts, STRETCH_PERCENTILES, axis=0)
    return _stretch(parts, lo, hi).reshape(*x.shape[:3], 3)


def _preview_rgb(s2_batch: torch.Tensor) -> np.ndarray:
    """RGB preview (B, H, W, 3) uint8 from B04, B03 and B02 of a batch (B, 12, H, W).

    Per image the three bands share ONE range, the 1 to 99 percentile of their non-zero
    values, so the colours keep their balance. For the eye only, no teacher sees it.
    """
    rgb = s2_batch[:, band_index(PREVIEW_BANDS)].detach().float().cpu().numpy()
    rgb = rgb.transpose(0, 2, 3, 1)
    out = np.zeros(rgb.shape, np.uint8)
    for i, image in enumerate(rgb):
        values = image[image != 0]
        if values.size:
            lo, hi = np.percentile(values, STRETCH_PERCENTILES)
            out[i] = _stretch(image, lo, hi)
    return out


def _preview_radar(s1_batch: torch.Tensor) -> np.ndarray:
    """Preview (B, H, W, 3) uint8 of a radar batch (B, 2, H, W) in dB - red VV, green VH,
    blue VV minus VH, each stretched on its own 1 to 99 percentile. For the eye only."""
    s1 = s1_batch.detach().float().cpu().numpy()
    out = np.zeros((s1.shape[0], s1.shape[2], s1.shape[3], 3), np.uint8)
    for i, (vv, vh) in enumerate(s1):
        for c, plane in enumerate((vv, vh, vv - vh)):
            finite = plane[np.isfinite(plane)]
            if finite.size:
                lo, hi = np.percentile(finite, STRETCH_PERCENTILES)
                out[i, :, :, c] = _stretch(np.nan_to_num(plane, nan=lo), lo, hi)
    return out


def _preview_aerial(aerial_batch: torch.Tensor) -> np.ndarray:
    """Preview (B, H, W, 3) uint8 of an aerial batch (B, 4 or 3, H, W) of 8-bit values, the
    RGB channels as they are. For the eye only."""
    rgb = aerial_batch[:, :3].detach().float().cpu().numpy().transpose(0, 2, 3, 1)
    return np.round(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)


def preview(batch: torch.Tensor, modality: str = "s2") -> np.ndarray:
    """The preview of a batch for `modality`, (B, H, W, 3) uint8."""
    if modality == "s2":
        return _preview_rgb(batch)
    if modality == "radar":
        return _preview_radar(batch)
    if modality == "aerial":
        return _preview_aerial(batch)
    raise ValueError(f"unknown modality {modality!r}, known: {MODALITIES}")


def _no_colon(text: Any) -> str:
    return str(text).replace(":", " -")


def _draw_grid(title: str, columns: Sequence[tuple[str, np.ndarray]]) -> Any:
    """One figure, rows = images, one column per entry (heading, (B, h, w, 3) uint8)."""
    from matplotlib.figure import Figure

    n_rows, n_cols = len(columns[0][1]), len(columns)
    fig = Figure(figsize=(PICTURE_CELL_IN * n_cols, PICTURE_CELL_IN * n_rows + 0.6),
                 layout="constrained")
    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    fig.suptitle(_no_colon(title), fontsize=11)
    for c, (heading, images) in enumerate(columns):
        for r in range(n_rows):
            axes[r][c].imshow(images[r], interpolation="nearest")
            axes[r][c].set_axis_off()
        axes[0][c].set_title(_no_colon(heading), fontsize=9)
    return fig


def feature_pictures(name: str, build_fn: Callable[[str], Teacher], s2_batch: torch.Tensor,
                     rules: Sequence[str], out_png: str | os.PathLike,
                     modality: str = "s2") -> Path:
    """One PNG of the patch features of one teacher under each of its input rules.

    name       the teacher's name, for the title
    build_fn   build_fn(rule) returns the teacher with that input rule
    s2_batch   the batch the teacher receives - (B, 12, H, W) float digital numbers for
               "s2", (B, 2, H, W) dB for "radar", (B, 4, H, W) 8-bit values for "aerial"
    rules      the rules to draw, one column each
    out_png    the file to write, its folder is made
    modality   which reader of the teacher runs (`teacher_call`), "s2" by default

    Rows are the images. The first column is a preview of the input (`preview`), then one
    column per rule with `pca_rgb` of that teacher's patch grid. Every column has its own
    PCA, so colours are compared inside a column, between columns only the shapes.
    The teacher is built once per rule and freed before the next one is built.
    The choice of a rule from this picture is a judgement by eye, not a score.
    """
    rules = list(rules)
    if not rules:
        raise ValueError(f"{name}: no rule given")
    channels = {"s2": len(S2_BANDS), "radar": 2, "aerial": (3, 4)}[modality]
    ok = s2_batch.ndim == 4 and s2_batch.shape[0] > 0 and (
        s2_batch.shape[1] in channels if isinstance(channels, tuple) else s2_batch.shape[1] == channels)
    if not ok:
        raise ValueError(f"{name}: expected (B, {channels}, H, W) for {modality}, "
                         f"got {tuple(s2_batch.shape)}")
    heading = {"s2": f"RGB preview {' '.join(PREVIEW_BANDS)}", "radar": "radar preview VV VH VV-VH",
               "aerial": "aerial RGB"}[modality]
    columns = [(heading, preview(s2_batch, modality))]
    for rule in rules:
        teacher = build_fn(rule)
        try:
            columns.append((rule, pca_rgb(teacher_call(teacher, modality)(s2_batch).spatial)))
        finally:
            del teacher
            _free()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    _draw_grid(f"{name} patch features as PCA colours", columns).savefig(out_png,
                                                                         dpi=PICTURE_DPI)
    return out_png


# ---------------------------------------------------------------- scene check (part four)
# How the papers draw feature maps (research of 2026-10-04) - the Perception Encoder
# paper fits one PCA per image on 0.5 x + 0.5 blur(x) and warns that a plain PCA "can be
# misleading" for models with global tokens; the DINOv3 notebook fits a whitened PCA on one
# image and maps it to colour with sigmoid(2 x). One PCA over a batch of different images, as
# `pca_rgb` does, spends its colours on the differences between the images.
SCENE_BLUR = (0.27407, 0.45186, 0.27407)   # 3-tap Gaussian, sigma 1 (exp(-1/2) normalised)
SCENE_PAIRS = 4000                          # patch pairs per image for the scene number


def _blur_grid(x: torch.Tensor) -> torch.Tensor:
    """(B, g, g, d) blurred over the two grid axes with `SCENE_BLUR`, edges repeated."""
    k = torch.tensor(SCENE_BLUR, dtype=x.dtype, device=x.device)
    y = x.permute(0, 3, 1, 2)                                            # (B, d, g, g)
    b, d, g, h = y.shape
    y = torch.nn.functional.pad(y.reshape(b * d, 1, g, h), (1, 1, 1, 1), mode="replicate")
    y = torch.nn.functional.conv2d(y, k.view(1, 1, 1, 3))
    y = torch.nn.functional.conv2d(y, k.view(1, 1, 3, 1))
    return y.reshape(b, d, g, h).permute(0, 2, 3, 1)


def pca_rgb_per_image(spatial: torch.Tensor | np.ndarray) -> np.ndarray:
    """Patch features (B, g, g, dim) as colours, uint8 (B, g, g, 3), ONE PCA PER IMAGE.

    Per image - 0.5 x + 0.5 blur(x) (Perception Encoder paper), every channel z-scored over
    the image's patches (guards against a few channels with large values, as DINOv3 uses its
    final norm for), the first three components, each scaled to unit SD (whitened), then
    sigmoid(2 x) as colour (DINOv3 notebook). The sign of a component is fixed so that its
    largest axis entry is positive. Colours are compared inside one image only.
    """
    if isinstance(spatial, np.ndarray):
        spatial = torch.from_numpy(spatial)
    x = spatial.detach().float().cpu()
    if x.ndim != 4 or x.shape[-1] < 3 or x.shape[1] * x.shape[2] < 3:
        raise ValueError(f"expected (B, g, g, dim) with dim and g * g at least 3, got {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise ValueError("the features hold values that are not finite")
    x = 0.5 * x + 0.5 * _blur_grid(x)
    out = np.zeros((*x.shape[:3], 3), np.uint8)
    for i, image in enumerate(x):
        flat = image.reshape(-1, image.shape[-1]).double()
        flat = (flat - flat.mean(0)) / flat.std(0).clamp_min(1e-6)
        axes = torch.linalg.svd(flat, full_matrices=False)[2][:3]           # (3, dim)
        largest = axes[torch.arange(3), axes.abs().argmax(1)]
        axes = axes * torch.where(largest < 0, -1.0, 1.0).double()[:, None]
        parts = flat @ axes.T
        parts = parts / parts.std(0).clamp_min(1e-12)
        rgb = torch.sigmoid(2.0 * parts).reshape(*image.shape[:2], 3)
        out[i] = (rgb * 255.0).round().clamp(0, 255).byte().numpy()
    return out


def patch_descriptors(batch: torch.Tensor, modality: str, grid: int) -> torch.Tensor:
    """What a patch of the INPUT looks like, (B, grid, grid, 2 C) - the mean and the SD of each
    input channel inside the patch, the input pooled onto the teacher's grid. Sentinel-2 uses
    B04 B03 B02 as stored (digital numbers), radar VV and VH in dB, aerial its RGB."""
    if modality == "s2":
        x = batch[:, band_index(PREVIEW_BANDS)]
    elif modality == "radar":
        x = batch[:, :2]
    elif modality == "aerial":
        x = batch[:, :3]
    else:
        raise ValueError(f"unknown modality {modality!r}, known: {MODALITIES}")
    x = torch.nan_to_num(x.detach().float().cpu())
    mean = torch.nn.functional.adaptive_avg_pool2d(x, grid)
    square = torch.nn.functional.adaptive_avg_pool2d(x * x, grid)
    sd = (square - mean * mean).clamp_min(0).sqrt()
    return torch.cat([mean, sd], 1).permute(0, 2, 3, 1)


def scene_correlation(spatial: torch.Tensor, descriptors: torch.Tensor, n_pairs: int = SCENE_PAIRS,
                      seed: int = 0) -> list[float]:
    """Per image, whether the teacher maps similar patches to similar features.

    Over `n_pairs` random pairs of patches of one image - the distance of their features (1 -
    cosine of the features with the image's mean token taken out) against the distance of
    their input descriptors (`patch_descriptors`, each column z-scored over the image,
    Euclidean). The Pearson correlation of the two. About 0 when the features do not follow
    the picture (noise, or a pattern of their own), higher when they do. Not a score of the
    teacher's quality - a teacher can see more than colour and texture - but noise gives 0.
    """
    f = spatial.detach().float().cpu()
    dsc = descriptors.detach().float().cpu()
    if f.shape[:3] != dsc.shape[:3]:
        raise ValueError(f"features {tuple(f.shape)} and descriptors {tuple(dsc.shape)} differ in grid")
    g = torch.Generator().manual_seed(seed)
    out = []
    for fi, di in zip(f, dsc):
        fi = fi.reshape(-1, fi.shape[-1])
        fi = torch.nn.functional.normalize(fi - fi.mean(0), dim=1)
        di = di.reshape(-1, di.shape[-1])
        di = (di - di.mean(0)) / di.std(0).clamp_min(1e-6)
        n = fi.shape[0]
        a = torch.randint(0, n, (n_pairs,), generator=g)
        b = torch.randint(0, n, (n_pairs,), generator=g)
        keep = a != b
        a, b = a[keep], b[keep]
        feat_dist = 1.0 - (fi[a] * fi[b]).sum(1)
        input_dist = (di[a] - di[b]).norm(dim=1)
        fd, idist = feat_dist - feat_dist.mean(), input_dist - input_dist.mean()
        denom = (fd.norm() * idist.norm()).item()
        out.append(float((fd * idist).sum().item() / denom) if denom > 0 else 0.0)
    return out


INPUT_NAMES = {"s2": "Sentinel-2 RGB thumbnail", "radar": "Sentinel-1 VV VH", "aerial": "aerial RGB 8 bit / 255"}


def scene_pictures(teacher: Teacher, batch: torch.Tensor, modality: str,
                   out_png: str | os.PathLike) -> tuple[Path, list[float]]:
    """One PNG and one number per image for one teacher on one input.

    Rows are the images, the first column is the input (`preview`), the second the teacher's
    patch features with one PCA per image (`pca_rgb_per_image`). Its heading names the model
    and the input, and every row carries its scene number (`scene_correlation`). Returns the
    file and the numbers in image order.
    """
    out = teacher_call(teacher, modality)(batch)
    numbers = scene_correlation(out.spatial, patch_descriptors(batch, modality, teacher.grid))
    columns = [(INPUT_NAMES[modality], preview(batch, modality)),
               (f"{teacher.name} reading {INPUT_NAMES[modality]}", pca_rgb_per_image(out.spatial))]
    fig = _draw_grid(f"{teacher.name}, one PCA per image, scene number per row", columns)
    for r, value in enumerate(numbers):                 # axes are row major, 2 per row
        head = f"{_no_colon(columns[1][0])}\n" if r == 0 else ""
        fig.axes[r * 2 + 1].set_title(f"{head}scene {value:.2f}", fontsize=8)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=PICTURE_DPI)
    return out_png, numbers


# --------------------------------------------------------------------------- budget

def _one_name(values: pd.Series) -> str:
    return ", ".join(sorted({str(v) for v in values}))


def _fetch_budget(fetch: pd.DataFrame, n_cells: int,
                  shards: Mapping[str, int] | None) -> dict:
    """The data fetch of a full run, from the timing table of `fetch_pairs`.

    Footers are counted once per SHARD, everything else once per cell that is read.
    The formulas are in the docstring of `budget_table`.
    """
    totals = fetch.attrs
    needed = ("wall_s", "thread_s", "cells_read", "cells_saved", "footers", "footer_s",
              "mb_footers", "mb_rows_per_cell")
    missing = [name for name in needed if totals.get(name) is None]
    if missing:
        raise ValueError(f"the fetch table does not say {missing}. Give the timing table as "
                         "fetch_pairs returned it (the numbers are in its attrs), measured on "
                         "cells that were not on disk, with the counters of majortom")
    if totals["cells_saved"] < 1 or totals["thread_s"] <= 0:
        raise ValueError("the fetch table holds no cell that was read and saved")
    cells = math.ceil(n_cells * totals["cells_read"] / totals["cells_saved"])
    if shards is None:
        shards = {s: min(cells, ALL_SHARDS[s]) for s in SENSORS}
    if set(shards) != set(SENSORS) or min(int(n) for n in shards.values()) < 0:
        raise ValueError(f"shards must hold a count for each of {SENSORS}, got {dict(shards)}")
    n_shards = sum(int(n) for n in shards.values())
    if n_shards and totals["footers"] < 1:
        raise ValueError("no footer was fetched in the measured run, so the cost of one is "
                         "not known")

    per_cell_s = (totals["thread_s"] - totals["footer_s"]) / totals["cells_read"]
    seconds = cells * per_cell_s
    megabytes = cells * totals["mb_rows_per_cell"]
    if n_shards:
        seconds += n_shards * totals["footer_s"] / totals["footers"]
        megabytes += n_shards * totals["mb_footers"] / totals["footers"]
    return {"fetch_cells": cells, "fetch_shards": n_shards,
            "fetch_hours": seconds * totals["wall_s"] / totals["thread_s"] / 3600.0,
            "fetch_gb": megabytes * MB / GB}


def budget_table(teacher_times: pd.DataFrame, student_times: pd.DataFrame, n_cells: int,
                 epochs: Sequence[int] = (20, 50), fetch: pd.DataFrame | None = None,
                 shards: Mapping[str, int] | None = None) -> pd.DataFrame:
    """Hours and storage of the two teacher routes, from measured speeds.

    teacher_times   rows of `time_teacher`, all teachers in one table
    student_times   rows of `time_student`
    n_cells         number of training cells, one image per cell and epoch
    epochs          one row per route and number of epochs
    fetch           the timing table of `fetch_pairs`, as returned (the numbers are in its
                    attrs). None = the four fetch columns are nan
    shards          {"s2": n, "s1": n}, the shards that the full run reads, for example
                    pairs["s2_url"].nunique() and pairs["s1_url"].nunique() of the full
                    pairs table. None = the upper limit, all shards of a set (498 and
                    2,930, `ALL_SHARDS`) or one per cell when the cells are fewer

    Formulas. Rows without a speed (out of memory) are left out first.
        r_t   images per second of teacher t, the best of its rows
        T     = sum over the teachers of 1 / r_t         seconds per image, all teachers
        S     = mean over the 3 modes of 1 / r_mode      seconds per image, one training
                step, at the batch size where this mean is smallest (every sample gets one
                mode with equal chance)
        g_t   = min(grid of teacher t, grid of the student)    

        route "live"     teacher_hours = epochs * n_cells * T / 3600
        route "stored"   teacher_hours = n_cells * T / 3600      the teachers run once
        both routes      student_hours = epochs * n_cells * S / 3600
                         hours         = teacher_hours + student_hours
        route "stored"   features_gb   = n_cells * sum over the teachers of
                                         (g_t * g_t + 1) * dim_t * 2 bytes / 10^9
                         float16, the + 1 is the summary vector. has
                         229 GB for 162,000 cells, which is this sum without the + 1.
        route "live"     features_gb   = 0

    The data fetch, the same in every row. It is done once, in a session without a GPU,
    so fetch_hours is NOT a part of `hours`. All numbers are totals of the measured run.
        fetch_cells  = n_cells * cells_read / cells_saved, rounded up
                       the cells to read for n_cells pairs, skipped cells cost a read too
        fetch_shards = the sum of `shards`
        c            = (thread_s - footer_s) / cells_read
                       thread seconds per cell without any footer
        thread       = fetch_cells * c + fetch_shards * footer_s / footers
        fetch_hours  = thread * wall_s / thread_s / 3600
                       wall_s / thread_s is what the measured number of threads gained
        fetch_gb     = (fetch_cells * mb_rows_per_cell
                        + fetch_shards * mb_footers / footers) / 1000
    A footer is counted once per SHARD, never once per cell. This holds for a full run
    that reads its cells shard by shard or keeps every footer in memory. majortom keeps 8
    footers, a run that jumps between more shards than that fetches footers again. For
    such a run give the number of footer fetches that you expect in `shards`.
    The cost of one footer is the mean of the measured run, where about half the footers
    are Sentinel-2 ones (2 to 5 MB) and half Sentinel-1 ones (1 MB fetched). A full run
    has more Sentinel-1 shards, so its footers cost a little less than counted here.
    Waits after a rate limit are inside c, as they were met in the measured run.

    Not inside, because stage 0 does not measure it - moving data to the GPU, PHI-S,
    saving checkpoints, and for the stored route reading the features back.
    The hours are A100 hours only when the speeds were measured on an A100, `measured_on`
    names the device.

    Columns route, epochs, n_cells, teacher_hours, student_hours, hours, features_gb,
    teacher_images_per_s (1 / T), student_images_per_s (1 / S), student_batch_size,
    measured_on, fetch_cells, fetch_shards, fetch_hours, fetch_gb.
    """
    n_cells = int(n_cells)
    if n_cells < 1 or not list(epochs) or min(int(e) for e in epochs) < 1:
        raise ValueError(f"n_cells and epochs must be above 0, got {n_cells} and {list(epochs)}")

    fast = teacher_times[teacher_times["images_per_s"] > 0]
    missing = sorted(set(teacher_times["teacher"]) - set(fast["teacher"]))
    if fast.empty or missing:
        raise ValueError(f"no measured speed for the teachers {missing or 'at all'}")
    best = fast.sort_values("images_per_s").groupby("teacher").tail(1)
    t_image = float((1.0 / best["images_per_s"]).sum())

    steps = student_times[student_times["images_per_s"] > 0]
    per_batch = {}
    for b, group in steps.groupby("batch_size"):
        if group["mode"].nunique() == 3 and len(group) == 3:
            per_batch[int(b)] = float((1.0 / group["images_per_s"]).mean())
    if not per_batch:
        raise ValueError("no batch size of the student has a measured speed in all 3 modes")
    batch = min(per_batch, key=per_batch.get)
    s_image = per_batch[batch]

    grids = steps.loc[steps["batch_size"] == batch, "grid"].astype(int)
    if grids.nunique() != 1:
        raise ValueError(f"the student rows have different grids {sorted(set(grids))}")
    g_student = int(grids.iloc[0])
    g = np.minimum(best["grid"].astype(int), g_student)
    bytes_per_cell = float(((g * g + 1) * best["dim"].astype(int) * FLOAT16_BYTES).sum())

    fetched = dict.fromkeys(FETCH_BUDGET, float("nan"))
    if fetch is not None:
        fetched = _fetch_budget(fetch, n_cells, shards)
    elif shards is not None:
        raise ValueError("shards given without the fetch table")

    measured_on = _one_name(pd.concat([best["device"], steps["device"]]))
    rows = []
    for route in ("live", "stored"):
        for e in epochs:
            e = int(e)
            teacher_h = (e if route == "live" else 1) * n_cells * t_image / 3600.0
            student_h = e * n_cells * s_image / 3600.0
            stored_gb = n_cells * bytes_per_cell / GB if route == "stored" else 0.0
            rows.append({"route": route, "epochs": e, "n_cells": n_cells,
                         "teacher_hours": teacher_h, "student_hours": student_h,
                         "hours": teacher_h + student_h, "features_gb": stored_gb,
                         "teacher_images_per_s": 1.0 / t_image,
                         "student_images_per_s": 1.0 / s_image,
                         "student_batch_size": batch, "measured_on": measured_on, **fetched})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report

def _plain(value: Any) -> Any:
    """A value as JSON types. nan and inf become null, dates become ISO text."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, os.PathLike):
        return str(value)
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    raise TypeError(f"cannot write a value of type {type(value).__name__} into a report")


def save_report(out_dir: str | os.PathLike, **tables: Any) -> list[Path]:
    """Write every table as <out_dir>/<name>.csv and every dict as <out_dir>/<name>.json.

    A DataFrame becomes a CSV without its index, a dict becomes JSON (nan as null, dates as
    ISO text). Anything else is a TypeError, and then nothing is written at all.
    Files of the same name are replaced. Returns the paths in the order given.
    """
    out_dir = Path(out_dir)
    texts = {}
    for name, value in tables.items():
        if isinstance(value, pd.DataFrame):
            texts[f"{name}.csv"] = value.to_csv(index=False, lineterminator="\n")
        elif isinstance(value, Mapping):
            texts[f"{name}.json"] = json.dumps(_plain(value), indent=2, ensure_ascii=False)
        else:
            raise TypeError(f"{name} is a {type(value).__name__}, expected a DataFrame or a dict")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for file_name, text in texts.items():
        path = out_dir / file_name
        path.write_text(text, encoding="utf-8", newline="")
        paths.append(path)
    return paths
