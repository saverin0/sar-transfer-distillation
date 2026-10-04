"""FLAIR-HUB aerial patches, RGB + NIR at 0.4 m, the third input of the student.

The data is IGN's FLAIR-HUB on Hugging Face, IGNF/FLAIR-HUB, not gated, Etalab Open Licence
2.0 (attribution). Everything the code relies on was read on 2026-09-30 from these pages and
from the files themselves, at the pinned revision HF_REVISION (last modified 2026-06-05)

    card       https://huggingface.co/datasets/IGNF/FLAIR-HUB (the raw README.md)
    listing    https://huggingface.co/api/datasets/IGNF/FLAIR-HUB/tree/main/data (and ?blobs=true)
    code       https://github.com/IGNF/FLAIR-HUB (README.md, flair-hub-HF-dl.py, csv/FLAIR-HUB_TOY.zip,
               src/flair_hub/data/dataloader.py, src/flair_hub/data/utils_data/label.py and io.py)
    licence    https://github.com/etalab/licence-ouverte/blob/master/open-licence.md
    files      D004-2021_AERIAL_LABEL-COSIA.zip, D080-2017_AERIAL_RGBI.zip and its label zip were
               downloaded and read; the central directories of all 148 aerial zips were read by
               byte ranges (157 MB)

Zips (listing). data/ holds 885 zips, one per department, year and modality, named
data/<DEPT>-<YEAR>_<MODALITY>.zip, plus data/GLOBAL_ALL_MTD.zip (2.1 GB, not used). The aerial
ones are <DEPT>-<YEAR>_AERIAL_RGBI.zip (74 files, 241.2 GB, 0.10 to 13.95 GB each) and
<DEPT>-<YEAR>_AERIAL_LABEL-COSIA.zip (74 files, 1.8 GB). 70 department codes, four of them with
two years (D017 2018 and 2021, D033 2018 and 2021, D044 2020 and 2022, D080 2017 and 2021), so
the card's "74 departments" are the 74 zips. Four codes join two departments (D024047, D025039,
D054057, D059062). The listed size equals the real file size for all 148 aerial zips.

Inside a zip (read from the files, the card only draws the tree). No directory entries, every
member deflated (method 8), one folder per ROI, one GeoTIFF per patch

    <DEPT>-<YEAR>_<MODALITY>/<ROI>/<DEPT>-<YEAR>_<MODALITY>_<ROI>_<ROW>-<COL>.tif

for example D004-2021_AERIAL_RGBI/AA-S1-32/D004-2021_AERIAL_RGBI_AA-S1-32_1-1.tif. ROI is
letters, digits and dashes (never an underscore), ROW and COL run from 1 to 30. All 241,100
member names of the RGBI zips have this form, the label zips hold the same ids (equal per zip),
2,822 ROI folders in all, as the card counts. 11 RGBI zips are zip64 (the ones over 4 GB). A
central directory is at most 1.9 MB.

Patch id, as in their csv files (csv/FLAIR-HUB_TOY.zip on GitHub, column patch_id)
<DEPT>-<YEAR>_<ROI>_<ROW>-<COL>, for example D012-2019_AF-S1-27_5-10. Unique over the dataset.

Image (AERIAL_RGBI, read from the files). GeoTIFF, uint8, 4 bands in the order R, G, B, NIR,
512 x 512 px at 0.2 m, EPSG:2154 (Lambert-93), LZW, no no-data value, 82 KB to 1.33 MB
deflated. The card says "4 bands (R, G, B, NIR)"; the files carry no band names (GDAL tags the
fourth as alpha). In D080-2017 the fourth band is 115 to 127 over the vegetation classes
against 76 to 95 for the first, and 34 over water, so it is near-infrared.

Label (AERIAL_LABEL-COSIA, read from the files). GeoTIFF, uint8, one band, 512 x 512 px on the
grid of the image (same corner and CRS; the pixel size is stored as 0.2 in some files and as
0.19999999999999998 in others, image and label files alike, so grids are compared within
GRID_TOL_M), LZW, no no-data value, values 0 to 18 as stored (0 in 86 of 200 files of D004,
nothing above 18). Their code reads them as they are (one-hot over range(19), no shift,
dataloader.py and label.py), so the stored value IS the class code of the table in their README
(config_supervision.yaml), CLASSES below. 0 = building is a class, there is no no-data code,
IGNORE is None. Their training gives weight 0 to the codes 15, 16, 17 and 18 (clear cut,
ligneous, mixed, undefined) and the card says "15-19 classes"; the plan scores all 19.

Departments (card, "official split ... split_1 fold"). TRAIN / VALIDATION 54 codes and TEST
16 codes, TRAIN_VAL_DEPARTMENTS and TEST_DEPARTMENTS, together the 70 codes of the listing. No
test department has two years. Patches per zip, counted in the central directories

    D004-2021 3,725, D005-2018 2,350, D006-2020 3,850, D007-2020 2,125, D008-2019 2,975, D009-2019 3,250
    D010-2019 1,375, D011-2021 2,175, D012-2019 2,625, D013-2020 7,525, D014-2020 3,700, D015-2020 3,500
    D016-2020 3,125, D017-2018 3,150, D017-2021 1,225, D018-2020 1,050, D020-2019 4,450, D021-2020 3,500
    D022-2021 2,775, D023-2020 2,600, D024047-2021 3,825, D025039-2020 2,925, D026-2020 5,025, D029-2021 4,300
    D030-2021 3,575, D031-2019 3,775, D032-2019 14,300, D033-2018 375, D033-2021 4,800, D034-2021 4,800
    D035-2020 4,975, D036-2020 1,975, D037-2021 1,425, D038-2021 3,550, D040-2021 1,700, D041-2021 2,750
    D044-2020 4,275, D044-2022 2,800, D045-2020 850, D046-2019 3,125, D049-2020 3,500, D051-2019 2,675
    D052-2019 2,825, D054057-2018 3,075, D055-2018 2,650, D056-2019 1,250, D058-2020 3,700, D059062-2021 4,425
    D060-2021 3,400, D061-2020 2,675, D063-2019 3,825, D064-2021 2,225, D065-2019 1,125, D066-2021 4,250
    D067-2021 3,250, D068-2021 2,450, D069-2020 4,900, D070-2020 3,350, D071-2020 3,500, D072-2019 4,025
    D073-2022 1,250, D074-2020 5,125, D075-2021 2,300, D076-2019 3,275, D077-2021 5,075, D078-2021 2,325
    D080-2017 100, D080-2021 2,825, D081-2020 3,425, D083-2020 5,300, D084-2021 3,450, D085-2019 3,475
    D086-2020 2,000, D091-2021 1,925

241,100 in all (as the card says), TRAIN / VALIDATION 190,400, TEST 50,700. The smallest
department is D045 (850), the smallest test department D073 (1,250).

What this module makes. Each 512 px patch is halved
to 256 px at 0.4 m, the image by the mean of each 2 x 2 block (rounded half up), the label by
the majority of the 2 x 2 block, a tie going to the smallest class code. Five sets, planned
before any download from the patch ids in the central directories, then built zip by zip

    distil_train     19,000 unlabelled TRAIN / VALIDATION patches (seed 0)
    distil_heldout    1,000 more of the same draw of 20,000, held out for the loss
    probe             4,000 labelled TRAIN / VALIDATION patches (seed 0), may overlap distil
    val               2,000 labelled TEST patches (seed 0)
    test              2,000 labelled TEST patches (seed 1), disjoint from val

How the patches are fetched. The central directory of a zip (read by byte ranges, no zip is
downloaded) gives every member's local header offset, deflated size, method and CRC-32. These
go into the patch index and so into the plan. A set is then built patch by patch, one range
request per member (local header plus deflated bytes, a second request only when the local
extra field is longer than LOCAL_SLACK allows), inflated with zlib, the stored name and the
CRC-32 checked, decoded with majortom.decode_tif. The old way, whole zips downloaded and read
in place, is kept as how="zips". What the real plan costs either way is in RANGE_COST below,
computed from the real central directories on 2026-09-30.

Nothing here prints or stores a token. With token=None every read is anonymous. A token is
sent to addresses under majortom.HF only, never on to the host the Hub redirects to. Reads
are counted in majortom.STATS, patch members as rows (row_requests, row_bytes, row_seconds),
listings, central directories and whole zips under "other".
"""

from __future__ import annotations

import json
import os
import re
import struct
import tarfile
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import IO, Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import requests

from . import majortom as M

HF_REPO = "IGNF/FLAIR-HUB"
HF_REVISION = "4c86dd8a227dcf74148f490d87285e404b94661d"     # the commit of 2026-06-05
HF_UPDATED = "2026-06-05"
MODALITIES = ("AERIAL_RGBI", "AERIAL_LABEL-COSIA")
BANDS = ("R", "G", "B", "NIR")
PATCH_PX = 512               # side of a stored patch
OUT_PX = 256                 # side after halving
SOURCE_PIXEL_M = 0.2
PIXEL_M = 0.4

CLASSES = ("building", "greenhouse", "swimming_pool", "impervious surface", "pervious surface",
           "bare soil", "water", "snow", "herbaceous vegetation", "agricultural land",
           "plowed land", "vineyard", "deciduous", "coniferous", "brushwood", "clear cut",
           "ligneous", "mixed", "undefined")
N_CLASSES = 19
IGNORE = None                # the files hold no no-data code, every value 0 to 18 is a class

TRAIN_VAL_DEPARTMENTS = (
    "D004", "D005", "D006", "D007", "D008", "D009", "D010", "D011", "D013", "D014", "D016",
    "D017", "D018", "D020", "D021", "D023", "D024047", "D025039", "D029", "D030", "D031",
    "D032", "D033", "D034", "D035", "D037", "D038", "D040", "D041", "D044", "D045", "D046",
    "D049", "D051", "D052", "D054057", "D055", "D056", "D058", "D059062", "D060", "D063",
    "D065", "D066", "D067", "D070", "D072", "D074", "D077", "D078", "D080", "D081", "D086",
    "D091")
TEST_DEPARTMENTS = ("D012", "D015", "D022", "D026", "D036", "D061", "D064", "D068", "D069",
                    "D071", "D073", "D075", "D076", "D083", "D084", "D085")

# The sets. "distil" is the draw of 20,000, of which "distil_heldout"
# are held out and the rest is distil_train.
SET_SIZES = {"distil": 20000, "distil_heldout": 1000, "probe": 4000, "val": 2000, "test": 2000}
SET_SEEDS = {"distil": 0, "probe": 0, "val": 0, "test": 1}
SETS = ("distil_train", "distil_heldout", "probe", "val", "test")
LABELLED = ("probe", "val", "test")

# What the real plan (SET_SIZES, SET_SEEDS) costs, computed on 2026-09-30 from the real central
# directories, no patch downloaded. The plan has 28,000 rows but 24,000 distinct patches: the
# probe draw (seed 0) lies inside the distil draw (seed 0), same pool, same permutation.
RANGE_COST = {"plan_rows": 28000, "rgbi_members": 24000, "label_members": 8000,
              "requests": 32000, "rgbi_gb_deflated": 23.970, "label_gb_deflated": 0.057,
              "gb_with_headers_and_slack": 24.034, "members_per_zip_min_median_max": (12, 370, 371),
              "zips_way_zips": 148, "zips_way_gb": 242.963}

LISTING_COLUMNS = ("department", "year", "modality", "path", "bytes")
# offset = the member's local header in the zip, size = its deflated bytes, method = 8
# (deflated) or 0 (stored), crc = its CRC-32, all from the central directory, as text
MEMBER_FIELDS = ("offset", "size", "method", "crc")
PATCH_COLUMNS = ("department", "year", "patch_id", "rgbi_zip", *(f"rgbi_{f}" for f in MEMBER_FIELDS),
                 "label_zip", *(f"label_{f}" for f in MEMBER_FIELDS))
PLAN_COLUMNS = ("set", *PATCH_COLUMNS)

ZIP_NAME = re.compile(r"data/(D\d+)-(\d{4})_(AERIAL_RGBI|AERIAL_LABEL-COSIA)\.zip")
MEMBER = re.compile(r"(?P<stem>D\d+-\d{4}_(?:AERIAL_RGBI|AERIAL_LABEL-COSIA))/(?P<roi>[A-Za-z0-9-]+)/"
                    r"(?P=stem)_(?P=roi)_(?P<rc>\d+-\d+)\.tif")
PATCH_ID = re.compile(r"(D\d+-\d{4})_([A-Za-z0-9-]+)_(\d+-\d+)")

# Our own settings for the transfer, not facts about the data.
TAIL_GUESS = 1 << 21         # bytes asked from the end of a zip for its central directory
GRID_TOL_M = 1e-3            # a label is on the grid of its image when the six numbers are this close
LOCAL_SLACK = 128            # bytes asked past the local header for its extra field (a zip64 extra is 20)
CHUNK = 1 << 22              # bytes per write while a zip is downloaded

_LISTINGS: dict[str, pd.DataFrame] = {}      # url of the listing -> the listing


# ------------------------------------------------------------------ small helpers

def _check_columns(df: pd.DataFrame, columns: Sequence[str], where) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{where}: no column {missing}")


def _save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write next to the target first, then rename, so a stopped write leaves no broken file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        df.to_csv(part, index=False)
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def _read_csv(path: str | Path, columns: Sequence[str]) -> pd.DataFrame:
    """A csv written here, every column as text (an empty label_zip stays "")."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    _check_columns(df, columns, path)
    return df[list(columns)]


def _zip_url(path: str) -> str:
    return f"{M.HF}/datasets/{HF_REPO}/resolve/{HF_REVISION}/{path}"


def _zip_stem(path: str | Path) -> str:
    """D004-2021_AERIAL_RGBI for data/D004-2021_AERIAL_RGBI.zip (also for a local path)."""
    name = Path(str(path)).name
    return name[:-4] if name.endswith(".zip") else name


def member_name(zip_path: str | Path, patch_id: str) -> str:
    """The name of a patch inside a zip, by the layout in the module docstring."""
    stem = _zip_stem(zip_path)
    found = PATCH_ID.fullmatch(patch_id)
    if not found:
        raise ValueError(f"{patch_id!r} is not a patch id like D012-2019_AF-S1-27_5-10")
    if not stem.startswith(found.group(1) + "_"):
        raise ValueError(f"{patch_id} is not a patch of {stem}")
    return f"{stem}/{found.group(2)}/{stem}_{found.group(2)}_{found.group(3)}.tif"


def _members(infos: Sequence[zipfile.ZipInfo], stem: str, where) -> list[tuple[str, zipfile.ZipInfo]]:
    """(patch id, member) for the members of one zip, in member order. Another form is an error."""
    out = []
    for info in infos:
        found = MEMBER.fullmatch(info.filename)
        if info.is_dir() or not found or found.group("stem") != stem:
            raise ValueError(f"{where}: member {info.filename!r} is not "
                             f"{stem}/<roi>/{stem}_<roi>_<row>-<col>.tif")
        out.append((f"{stem.partition('_')[0]}_{found.group('roi')}_{found.group('rc')}", info))
    if len({patch_id for patch_id, _ in out}) != len(out):
        raise ValueError(f"{where}: a patch is stored twice")
    return out


def _fields(info: zipfile.ZipInfo | None) -> dict[str, str]:
    """The four member fields of the index, as text ("" without a member)."""
    if info is None:
        return {f: "" for f in MEMBER_FIELDS}
    return {"offset": str(info.header_offset), "size": str(info.compress_size),
            "method": str(info.compress_type), "crc": str(info.CRC)}


def _paid(before: Mapping[str, float]) -> dict[str, float]:
    return {k: M.STATS[k] - before[k] for k in M.STATS}


def _cost(paid: Mapping[str, float], wall: float) -> str:
    return (f"{int(paid['requests'])} requests, {paid['bytes'] / 1e6:.1f} MB, {paid['seconds']:.1f} s "
            f"in threads, {int(paid['retries'])} retries, {paid['wait_seconds']:.1f} s waited, "
            f"{wall:.1f} s wall")


# ------------------------------------------------------------------ the listing

def list_zips(token: str | None = None, cache: str | Path | None = None) -> pd.DataFrame:
    """The aerial zips of the repository at HF_REVISION, one row each.

    Columns department, year, modality, path, bytes (department, year and modality as text),
    sorted by path. The listing is kept in memory, and in `cache` (a csv file) when given: an
    existing cache is returned as it is.
    """
    if cache is not None and Path(cache).is_file():
        df = _read_csv(cache, LISTING_COLUMNS)
        df["bytes"] = df["bytes"].astype(np.int64)
        return df
    url = f"{M.HF}/api/datasets/{HF_REPO}/tree/{HF_REVISION}/data"
    if url in _LISTINGS:
        df = _LISTINGS[url].copy()
    else:
        rows, first = [], url
        while url:
            body, answer = M._get(url, token)
            for entry in json.loads(body):
                found = ZIP_NAME.fullmatch(entry.get("path", "")) if entry.get("type") == "file" else None
                if found:
                    rows.append({"department": found.group(1), "year": found.group(2),
                                 "modality": found.group(3), "path": entry["path"],
                                 "bytes": int(entry["size"])})
            link = re.search(r'<([^>]+)>;\s*rel="next"', str(answer.get("Link", "")))
            url = link.group(1) if link else ""
        if not rows:
            raise FileNotFoundError(f"{HF_REPO}: no aerial zip under data/")
        df = pd.DataFrame(rows, columns=list(LISTING_COLUMNS)).sort_values("path", ignore_index=True)
        _LISTINGS[first] = df.copy()
    if cache is not None:
        _save_csv(df, cache)
    return df


def _listed(listing: pd.DataFrame | None, path: str, token: str | None) -> int:
    """The listed size of one zip."""
    listing = list_zips(token) if listing is None else listing
    _check_columns(listing, LISTING_COLUMNS, "listing")
    hit = listing.loc[listing["path"] == path, "bytes"]
    if hit.empty:
        raise KeyError(f"{path} is not in the listing")
    return int(hit.iloc[0])


# ------------------------------------------------------------------ central directories

def _central_directory(url: str, token: str | None = None) -> tuple[list[zipfile.ZipInfo], int]:
    """The members of a zip on a server, and its size, from its central directory alone.

    One range request for the last TAIL_GUESS bytes, a second one when the central directory
    starts earlier. zip64 archives are read the same way. Nothing else of the zip is fetched.
    """
    tail, answer = M._get(url, token, f"bytes=-{TAIL_GUESS}", "other")
    total = str(answer.get("Content-Range", "")).rpartition("/")[2]
    if not total.isdigit():
        raise OSError(f"{url}: the answer does not say the file size (Content-Range)")
    size = int(total)
    base = size - len(tail)
    end = tail.rfind(b"PK\x05\x06")
    if end < 0:
        raise ValueError(f"{url}: not a zip file (no end of central directory in the last "
                         f"{len(tail)} bytes)")
    cd_size, cd_start = struct.unpack("<II", tail[end + 12:end + 20])
    if 0xFFFFFFFF in (cd_size, cd_start):                     # zip64
        locator = tail.rfind(b"PK\x06\x07", 0, end)
        if locator < 0:
            raise ValueError(f"{url}: zip64 archive without a zip64 locator")
        record = struct.unpack("<Q", tail[locator + 8:locator + 16])[0]
        if record < base:
            tail = M._read_range(url, record, base, token, "other") + tail
            base = record
        at = record - base
        if tail[at:at + 4] != b"PK\x06\x06":
            raise ValueError(f"{url}: no zip64 end of central directory at byte {record}")
        cd_size, cd_start = struct.unpack("<QQ", tail[at + 40:at + 56])
    if cd_start + cd_size > size:
        raise ValueError(f"{url}: the central directory ends after the file")
    if cd_start < base:
        tail = M._read_range(url, cd_start, base, token, "other") + tail
        base = cd_start
    try:
        with zipfile.ZipFile(M._Parts(size, [(base, tail)], url)) as z:
            return z.infolist(), size
    except zipfile.BadZipFile as err:
        raise ValueError(f"{url}: {err}") from None


def list_patches(listing: pd.DataFrame, cache: str | Path | None = None,
                 token: str | None = None, workers: int = 8,
                 progress: bool = True) -> pd.DataFrame:
    """One row per patch of the RGBI zips, from the central directories. No zip is downloaded.

    Columns PATCH_COLUMNS, every value as text, sorted by patch_id: department, year, patch_id,
    then for the RGBI member and for the label member the zip path and the member's offset,
    size, method and crc (see MEMBER_FIELDS). The label fields are "" when the label zip of
    that department and year does not hold the patch. A label without an image is not
    listed. A member name of another form than the one in the module docstring is an error.
    Costs one or two range requests of TAIL_GUESS bytes per zip. `cache` (a csv file) is
    returned as it is when it exists, else written.
    """
    if cache is not None and Path(cache).is_file():
        return _read_csv(cache, PATCH_COLUMNS)
    _check_columns(listing, LISTING_COLUMNS, "listing")
    zips = listing[listing["modality"].isin(MODALITIES)].sort_values("path")
    if zips.empty:
        raise ValueError("the listing has no aerial zip")
    begin = time.perf_counter()
    before = dict(M.STATS)

    def read(path: str) -> tuple[str, list[tuple[str, zipfile.ZipInfo]]]:
        infos, size = _central_directory(_zip_url(path), token)
        if size != _listed(zips, path, token):
            raise OSError(f"{path}: {size} bytes on the server, the listing says "
                          f"{_listed(zips, path, token)}")
        return path, _members(infos, _zip_stem(path), path)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        found = dict(pool.map(read, zips["path"].tolist()))
    labels: dict[tuple[str, str], tuple[str, dict[str, zipfile.ZipInfo]]] = {}
    for row in zips.itertuples(index=False):
        if row.modality == "AERIAL_LABEL-COSIA":
            labels[(row.department, row.year)] = (row.path, dict(found[row.path]))
    rows = []
    for row in zips.itertuples(index=False):
        if row.modality != "AERIAL_RGBI":
            continue
        label_zip, label_members = labels.get((row.department, row.year), ("", {}))
        for patch_id, info in found[row.path]:
            label = label_members.get(patch_id)
            rows.append({"department": row.department, "year": row.year, "patch_id": patch_id,
                         "rgbi_zip": row.path, **{f"rgbi_{k}": v for k, v in _fields(info).items()},
                         "label_zip": label_zip if label is not None else "",
                         **{f"label_{k}": v for k, v in _fields(label).items()}})
    df = pd.DataFrame(rows, columns=list(PATCH_COLUMNS))
    if df["patch_id"].duplicated().any():
        raise ValueError("a patch id is in two zips")
    df = df.sort_values("patch_id", ignore_index=True)
    if progress:
        print(f"{len(df):,} patches in {len(zips)} zips, {(df['label_zip'] != '').sum():,} with a "
              f"label, {(M.STATS['bytes'] - before['bytes']) / 1e6:.0f} MB read in "
              f"{time.perf_counter() - begin:.0f} s")
    if cache is not None:
        _save_csv(df, cache)
    return df


# ------------------------------------------------------------------ one member, one patch

def fetch_member(url: str, offset: int | str, size: int | str, name: str, method: int | str,
                 crc: int | str, token: str | None = None, end: int | None = None) -> bytes:
    """The bytes of one member of a zip on a server, without downloading the zip.

    offset, size, method, crc   the member's fields from the central directory (MEMBER_FIELDS)
    name                        the member's name, checked against the local header
    end                         the size of the zip when known, so no range reaches past it

    One range request for the local header plus the deflated bytes (LOCAL_SLACK bytes more for
    the local extra field; a longer one costs a second request), inflated with zlib for method
    8, as they are for method 0. The name stored in the local header must be `name` and the
    CRC-32 of the result must be `crc`, else the bytes are not the member's and it is an
    error. Retries and waits are majortom's (`_get`). Counted as a row in majortom.STATS.
    """
    offset, size, method, crc = int(offset), int(size), int(method), int(crc)
    if method not in (0, 8):
        raise ValueError(f"{url} {name}: compression method {method} is neither stored nor deflated")
    wanted = name.encode("utf-8")
    stop = offset + 30 + len(wanted) + LOCAL_SLACK + size
    if end is not None:
        stop = min(stop, int(end))
    data = M._read_range(url, offset, stop, token, "row")
    if len(data) < 30 or data[:4] != b"PK\x03\x04":
        raise ValueError(f"{url}: no local file header at byte {offset}")
    n, m = struct.unpack("<HH", data[26:30])
    stored = data[30:30 + n]
    if stored != wanted:
        raise ValueError(f"{url}: byte {offset} holds {stored.decode('utf-8', 'replace')!r}, "
                         f"not {name!r}")
    start = 30 + n + m
    if start + size > len(data):
        data += M._read_range(url, offset + len(data), offset + start + size, token, "row")
    raw = data[start:start + size]
    if method == 8:
        try:
            out = zlib.decompress(raw, -15)
        except zlib.error as err:
            raise ValueError(f"{url} {name}: damaged ({err})") from None
    else:
        out = bytes(raw)
    if zlib.crc32(out) != crc:
        raise ValueError(f"{url} {name}: the CRC-32 differs, the bytes are not the member's")
    return out


def _decode(data: bytes, where: str) -> tuple[np.ndarray, tuple]:
    """One GeoTIFF as an array, and its grid (transform6, crs)."""
    try:
        array, transform, crs, _ = M.decode_tif(data)
    except ValueError:
        raise ValueError(f"{where}: not an image that can be read") from None
    return array, (transform, crs)


def _check_image(image: np.ndarray, where: str) -> None:
    if image.ndim != 3 or image.shape != (4, PATCH_PX, PATCH_PX) or image.dtype != np.uint8:
        raise ValueError(f"{where}: image is {image.dtype} {image.shape}, expected uint8 "
                         f"(4, {PATCH_PX}, {PATCH_PX})")


def _check_label(label: np.ndarray, label_grid: tuple, grid: tuple, where: str) -> None:
    if label.ndim != 2 or label.shape != (PATCH_PX, PATCH_PX) or label.dtype != np.uint8:
        raise ValueError(f"{where}: label is {label.dtype} {label.shape}, expected uint8 "
                         f"({PATCH_PX}, {PATCH_PX})")
    apart = max(abs(a - b) for a, b in zip(label_grid[0], grid[0], strict=True))
    if label_grid[1] != grid[1] or not apart <= GRID_TOL_M:
        raise ValueError(f"{where}: the label is not on the grid of the image, {label_grid} and {grid}")
    if label.max() >= N_CLASSES:
        raise ValueError(f"{where}: label code {label.max()} is not one of the {N_CLASSES} classes")


def fetch_patch(zip_path: str, patch_id: str, offset: int | str, size: int | str,
                method: int | str, crc: int | str, token: str | None = None,
                listing: pd.DataFrame | None = None) -> tuple[np.ndarray, tuple]:
    """One patch of a zip of the repository by a range request, as (array, grid).

    zip_path and the member fields as in the patch index. The array is uint8 (4, 512, 512) for
    an RGBI zip and (512, 512) for a label zip, checked. `listing` (None = fetched) gives the
    size of the zip.
    """
    name = member_name(zip_path, patch_id)
    where = f"{zip_path} {name}"
    data = fetch_member(_zip_url(zip_path), offset, size, name, method, crc, token,
                        _listed(listing, zip_path, token))
    array, grid = _decode(data, where)
    if "AERIAL_RGBI" in _zip_stem(zip_path):
        _check_image(array, where)
    elif array.ndim != 2 or array.shape != (PATCH_PX, PATCH_PX) or array.dtype != np.uint8:
        raise ValueError(f"{where}: label is {array.dtype} {array.shape}, expected uint8 "
                         f"({PATCH_PX}, {PATCH_PX})")
    return array, grid


# ------------------------------------------------------------------ patches from local zips

def iter_patches(rgbi_zip: str | Path, label_zip: str | Path | None = None,
                 ids: Sequence[str] | None = None
                 ) -> Iterator[tuple[str, np.ndarray, np.ndarray | None]]:
    """(patch_id, image, label) for patches read straight from local zips.

    image   uint8 (4, PATCH_PX, PATCH_PX), bands R, G, B, NIR as stored
    label   uint8 (PATCH_PX, PATCH_PX) from label_zip, None without one
    ids     the patches to read, in that order; None = every member of the RGBI zip in name order

    Every patch is checked (dtype, shape, the label on the grid of the image, no label code
    above N_CLASSES - 1). A zip that is not a zip, a damaged member, a missing member or a
    patch of another form stops with an error that names the file. Nothing is repaired.
    """
    rgbi_zip = Path(rgbi_zip)
    label_zip = None if label_zip is None else Path(label_zip)
    try:
        with zipfile.ZipFile(rgbi_zip) as images, (
                zipfile.ZipFile(label_zip) if label_zip else nullcontext()) as labels:
            if ids is None:
                ids = [patch_id for patch_id, _ in
                       _members(sorted(images.infolist(), key=lambda i: i.filename),
                                _zip_stem(rgbi_zip), rgbi_zip)]
            for patch_id in ids:
                image, grid = _read_member(images, rgbi_zip, patch_id)
                _check_image(image, f"{rgbi_zip} {patch_id}")
                label = None
                if labels is not None:
                    label, label_grid = _read_member(labels, label_zip, patch_id)
                    _check_label(label, label_grid, grid, f"{label_zip} {patch_id}")
                yield patch_id, image, label
    except zipfile.BadZipFile as err:
        raise ValueError(f"{rgbi_zip}: not a zip file or damaged ({err})") from None


def _read_member(archive: zipfile.ZipFile, path: Path, patch_id: str) -> tuple[np.ndarray, tuple]:
    """One GeoTIFF of a local zip as an array, and its grid (transform6, crs)."""
    name = member_name(path, patch_id)
    try:
        data = archive.read(name)
    except KeyError:
        raise KeyError(f"{path}: no member {name}") from None
    except (zipfile.BadZipFile, zlib.error) as err:
        raise ValueError(f"{path}: member {name} is damaged ({err})") from None
    return _decode(data, f"{path}: member {name}")


# ------------------------------------------------------------------ halving

def halve_image(x: np.ndarray) -> np.ndarray:
    """uint8 (..., h, w) -> uint8 (..., h / 2, w / 2), the mean of each 2 x 2 block.

    The mean is rounded half up (sum + 2) // 4, so 2.5 becomes 3. Sides must be even.
    """
    x = np.asarray(x)
    if x.dtype != np.uint8 or x.ndim < 2 or x.shape[-1] % 2 or x.shape[-2] % 2:
        raise ValueError(f"expected uint8 with even sides, got {x.dtype} {x.shape}")
    total = (x[..., 0::2, 0::2].astype(np.uint16) + x[..., 0::2, 1::2]
             + x[..., 1::2, 0::2] + x[..., 1::2, 1::2])
    return ((total + 2) // 4).astype(np.uint8)


def halve_label(y: np.ndarray) -> np.ndarray:
    """uint8 (..., h, w) -> uint8 (..., h / 2, w / 2), the majority of each 2 x 2 block.

    The code that appears most often in the block wins. A tie (2 against 2, or four different
    codes) goes to the smallest class code. Sides must be even.
    """
    y = np.asarray(y)
    if y.dtype != np.uint8 or y.ndim < 2 or y.shape[-1] % 2 or y.shape[-2] % 2:
        raise ValueError(f"expected uint8 with even sides, got {y.dtype} {y.shape}")
    four = np.stack([y[..., 0::2, 0::2], y[..., 0::2, 1::2], y[..., 1::2, 0::2], y[..., 1::2, 1::2]])
    counts = (four[:, None] == four[None]).sum(0, dtype=np.int32)      # how many equal each one
    key = counts * 256 + (255 - four.astype(np.int32))                 # more first, then smaller
    best = key.argmax(0)
    return np.take_along_axis(four, best[None], 0)[0]


# ------------------------------------------------------------------ the plan

def _spread(n: int, counts: Mapping[str, int]) -> dict[str, int]:
    """How many of n to take per department, as even as the counts allow.

    n // D each and one more for the first n % D departments in the given order. A department
    with fewer patches than its share gives all it has, and what is missing is spread again
    the same way over the departments that still have patches, until n are taken. Fewer than
    n patches in all is an error.
    """
    total = sum(counts.values())
    if n < 0 or n > total:
        raise ValueError(f"{n} patches asked, the departments hold {total}")
    take = {d: 0 for d in counts}
    left = n
    while left:
        room = [d for d in counts if take[d] < counts[d]]
        base, extra = divmod(left, len(room))
        for i, d in enumerate(room):
            got = min(base + (i < extra), counts[d] - take[d])
            take[d] += got
            left -= got
    return take


def _draw(pool: pd.DataFrame, n: int, seed: int, what: str) -> pd.DataFrame:
    """n rows of the pool, spread over its departments by _spread, drawn with one generator."""
    if pool.empty:
        raise ValueError(f"{what}: no patch to draw from")
    rng = np.random.default_rng(seed)
    groups = {d: g.sort_values("patch_id") for d, g in pool.groupby("department", sort=True)}
    try:
        take = _spread(n, {d: len(g) for d, g in groups.items()})
    except ValueError as err:
        raise ValueError(f"{what}: {err}") from None
    parts = [g.iloc[rng.permutation(len(g))[:take[d]]] for d, g in groups.items()]
    return pd.concat(parts, ignore_index=True)


def plan_sets(patches: pd.DataFrame, sizes: Mapping[str, int] | None = None,
              seeds: Mapping[str, int] | None = None) -> pd.DataFrame:
    """The five sets, one row per patch, fixed before any download.

    patches   from list_patches
    sizes     SET_SIZES with these changed (tests use small ones)
    seeds     SET_SEEDS with these changed

    Columns PLAN_COLUMNS = set and the columns of the patch index (so a row carries what the
    range read needs). Rows are in draw order, the sets in the order of SETS.

    Rule per set. The pool is every patch of the set's departments (for a labelled set, every
    patch with a label). The patches are spread over the departments as evenly as the counts
    allow (_spread, departments in code order, both years of a department in one pool).
    Inside a department the patches are sorted by id and drawn by numpy's
    default_rng(seed).permutation, one generator per set, departments in code order.
    distil draws sizes["distil"] with seeds["distil"]; sizes["distil_heldout"] of them, spread
    over the departments the same way and taken from the front of each department's draw, are
    distil_heldout, the rest distil_train. probe is a draw of its own (it may overlap distil).
    val is drawn from the test departments, test from the test departments without val.
    """
    sizes = {**SET_SIZES, **(sizes or {})}
    seeds = {**SET_SEEDS, **(seeds or {})}
    if sizes["distil_heldout"] > sizes["distil"]:
        raise ValueError("more held out than drawn for distil")
    patches = pd.DataFrame(patches)
    _check_columns(patches, PATCH_COLUMNS, "patches")
    patches = patches[list(PATCH_COLUMNS)].fillna("").astype(str)
    if patches["patch_id"].duplicated().any():
        raise ValueError("a patch id is listed twice")
    unknown = sorted(set(patches["department"]) - set(TRAIN_VAL_DEPARTMENTS) - set(TEST_DEPARTMENTS))
    if unknown:
        raise ValueError(f"departments that are not on the card: {unknown}")
    train = patches[patches["department"].isin(TRAIN_VAL_DEPARTMENTS)]
    test = patches[patches["department"].isin(TEST_DEPARTMENTS)]

    distil = _draw(train, sizes["distil"], seeds["distil"], "distil")
    held = _spread(sizes["distil_heldout"],
                   {d: int(k) for d, k in distil.groupby("department", sort=True).size().items()})
    is_held = np.zeros(len(distil), dtype=bool)
    for d, k in held.items():
        is_held[np.flatnonzero((distil["department"] == d).to_numpy())[:k]] = True
    distil.insert(0, "set", np.where(is_held, "distil_heldout", "distil_train"))
    probe = _draw(train[train["label_zip"] != ""], sizes["probe"], seeds["probe"], "probe")
    val = _draw(test[test["label_zip"] != ""], sizes["val"], seeds["val"], "val")
    rest = test[(test["label_zip"] != "") & ~test["patch_id"].isin(val["patch_id"])]
    tst = _draw(rest, sizes["test"], seeds["test"], "test")
    parts = [distil[distil["set"] == "distil_train"], distil[distil["set"] == "distil_heldout"],
             probe.assign(set="probe"), val.assign(set="val"), tst.assign(set="test")]
    plan = pd.concat(parts, ignore_index=True)[list(PLAN_COLUMNS)]
    return _checked_plan(plan)


def _checked_plan(plan: pd.DataFrame) -> pd.DataFrame:
    plan = pd.DataFrame(plan)
    _check_columns(plan, PLAN_COLUMNS, "plan")
    plan = plan[list(PLAN_COLUMNS)].fillna("").astype(str).reset_index(drop=True)
    unknown = sorted(set(plan["set"]) - set(SETS))
    if unknown:
        raise ValueError(f"unknown sets {unknown}, the sets are {list(SETS)}")
    if plan.duplicated(["set", "patch_id"]).any():
        raise ValueError("a patch is twice in one set")
    if plan.drop_duplicates(list(PATCH_COLUMNS))["patch_id"].duplicated().any():
        raise ValueError("a patch has two different entries")
    labelled = plan["set"].isin(LABELLED)
    if (plan.loc[labelled, "label_zip"] == "").any():
        raise ValueError("a patch of a labelled set has no label zip")
    for kind in ("rgbi", "label"):
        columns = [f"{kind}_{f}" for f in MEMBER_FIELDS]
        given = plan[columns].ne("").sum(axis=1)
        if kind == "rgbi":
            if (given != len(columns)).any():
                raise ValueError("an rgbi member field is missing")
        elif ((given != len(columns)) & (plan["label_zip"] != "")).any() or (
                (given != 0) & (plan["label_zip"] == "")).any():
            raise ValueError("the label member fields do not go with label_zip")
        for column in columns:
            if not plan[column].str.fullmatch(r"\d*").all():
                raise ValueError(f"{column} is not a whole number")
    for name, group in plan.groupby("set"):
        if not group["patch_id"].str.fullmatch(PATCH_ID.pattern).all():
            raise ValueError(f"{name}: a patch id is not like D012-2019_AF-S1-27_5-10")
    return plan


def write_plan(plan: pd.DataFrame, path: str | Path) -> None:
    """The plan as a csv file, written whole or not at all."""
    _save_csv(_checked_plan(plan), path)


def read_plan(path: str | Path) -> pd.DataFrame:
    return _checked_plan(_read_csv(path, PLAN_COLUMNS))


# ------------------------------------------------------------------ whole zips

def _download(url: str, part: Path, expected: int, token: str | None = None) -> None:
    """Stream url into `part`, going on from what is there, until it holds `expected` bytes.

    Network errors, cut answers and the codes 408, 429 and 5xx are tried again as in
    majortom._get (RETRIES tries in a row without progress, BACKOFF_S doubled, or the wait the
    server asks for up to MAX_WAIT_S). A try that brought bytes does not count as a failure.
    Other codes stop at once. A server that ignores the range (200) is read from the start.
    A size that differs from `expected` is an error. Error texts never hold the token.
    """
    have = part.stat().st_size if part.exists() else 0
    if have > expected:
        part.unlink()
        have = 0
    failures, wait, last = 0, 0.0, ""
    while have < expected:
        if failures:
            if failures >= M.RETRIES:
                raise ConnectionError(f"{url}: gave up after {M.RETRIES} tries in a row, the last "
                                      f"error was {last}")
            M._add(retries=1, wait_seconds=wait)
            time.sleep(wait)
        wait = M.BACKOFF_S * 2 ** failures
        headers = {"Accept-Encoding": "identity", "Range": f"bytes={have}-"}
        if token and url.startswith(M.HF + "/"):
            headers["Authorization"] = f"Bearer {token}"
        start, got = time.perf_counter(), 0
        try:
            with M._session().get(url, headers=headers, timeout=M.TIMEOUT_S, stream=True) as r:
                code, answer = r.status_code, r.headers
                if code in (408, 429) or code >= 500:
                    last = f"HTTP {code}"
                    asked = (re.search(r"\bt=(\d+)", str(answer.get("RateLimit", "")))
                             or re.fullmatch(r"\s*(\d+)\s*", str(answer.get("Retry-After", ""))))
                    if asked:
                        wait = max(wait, min(float(asked.group(1)), M.MAX_WAIT_S))
                    failures += 1
                    continue
                if code == 404:
                    raise FileNotFoundError(f"{url}: HTTP 404, no such file")
                if code in (401, 403):
                    raise PermissionError(f"{url}: HTTP {code}, this file needs a token with read access")
                if code == 200:
                    mode, have = "wb", 0
                elif code == 206:
                    span = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", str(answer.get("Content-Range", "")))
                    if not span or int(span.group(1)) != have:
                        raise OSError(f"{url}: the answer does not start at byte {have} "
                                      f"(Content-Range {answer.get('Content-Range')!r})")
                    if int(span.group(3)) != expected:
                        raise OSError(f"{url}: the file has {span.group(3)} bytes on the server, "
                                      f"the listing says {expected}")
                    mode = "ab"
                else:
                    raise OSError(f"{url}: HTTP {code}, expected 206")
                length = answer.get("Content-Length")
                if length is not None and int(length) != expected - have:
                    raise OSError(f"{url}: the server sends {int(length) + have} bytes, the listing "
                                  f"says {expected}")
                with open(part, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        got += len(chunk)
        except (requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError) as err:
            last = type(err).__name__
            failures = 0 if got else failures + 1
            continue
        finally:
            M._count("other", got, time.perf_counter() - start)
            have = part.stat().st_size if part.exists() else 0
        if have < expected:
            last = f"cut answer ({have} of {expected} bytes)"
            failures = 0 if got else failures + 1
    if have != expected:
        raise OSError(f"{url}: {have} bytes came, the listing says {expected}")


def download_zip(path: str, dest_dir: str | Path, token: str | None = None,
                 listing: pd.DataFrame | None = None, progress: bool = True) -> Path:
    """One zip of the repository into dest_dir, returns its path.

    path      as in the listing, e.g. data/D004-2021_AERIAL_RGBI.zip
    listing   from list_zips, None = fetched (kept in memory)

    A file that is there with the listed size is not fetched again. A stopped download is
    taken up where it stopped (the .part file next to the target). The size is checked
    against the listing, then the file is renamed into place. A file of another size is an
    error, it is not overwritten.
    """
    expected = _listed(listing, path, token)
    dest = Path(dest_dir) / Path(path).name
    if dest.is_file():
        if dest.stat().st_size == expected:
            return dest
        raise OSError(f"{dest}: has {dest.stat().st_size} bytes, the listing says {expected}; "
                      "remove it first")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    begin = time.perf_counter()
    _download(_zip_url(path), part, expected, token)
    os.replace(part, dest)
    if progress:
        print(f"{dest.name}: {expected / 1e9:.2f} GB in {time.perf_counter() - begin:.0f} s")
    return dest


# ------------------------------------------------------------------ building and loading

def _file_name(name: str, department: str, year: str) -> str:
    return f"{name}__{department}-{year}.npz"


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """An npz as numpy writes it, but with a fixed date, so equal arrays give equal bytes."""
    part = path.with_name(path.name + ".part")
    try:
        with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for key, value in arrays.items():
                info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                with z.open(info, "w", force_zip64=True) as f:
                    np.lib.format.write_array(f, np.asarray(value), allow_pickle=False)
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def _npz_headers(source, keys: Sequence[str], where: str) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Shape and dtype of the arrays `keys` of an npz from their headers alone, nothing else is read.

    `source` is a file path or an open binary file object that can seek, read from its start
    (a tar member from tarfile.extractfile). `where` names it in the errors.
    """
    readers = {(1, 0): np.lib.format.read_array_header_1_0,
               (2, 0): np.lib.format.read_array_header_2_0}
    out = {}
    with zipfile.ZipFile(source) as z:
        names = z.namelist()
        for key in keys:
            if f"{key}.npy" not in names:
                raise RuntimeError(f"{where}: holds no {key}, a file of another build; remove it")
            with z.open(f"{key}.npy") as f:
                version = np.lib.format.read_magic(f)
                if version not in readers:
                    raise RuntimeError(f"{where}: {key} has npy format {version}, not 1.0 or 2.0")
                shape, _, dtype = readers[version](f)
            out[key] = (tuple(shape), dtype)
    return out


def _npz_header(path: Path, key: str) -> tuple[tuple[int, ...], np.dtype]:
    """Shape and dtype of one array of an npz file from its header alone, the array is not read."""
    return _npz_headers(path, [key], Path(path).name)[key]


def _check_file(path: Path, labelled: bool) -> int:
    """A built file must hold ids (N,), images uint8 (N, 4, OUT_PX, OUT_PX) and, when labelled,
    labels uint8 (N, OUT_PX, OUT_PX). Anything else is a file of another build. Returns N."""
    n = _npz_header(path, "ids")[0]
    if len(n) != 1:
        raise RuntimeError(f"{path.name}: ids are {n}, expected one row per patch")
    wanted = {"images": (n[0], len(BANDS), OUT_PX, OUT_PX)}
    if labelled:
        wanted["labels"] = (n[0], OUT_PX, OUT_PX)
    for key, shape in wanted.items():
        found, dtype = _npz_header(path, key)
        if found != shape or dtype != np.uint8:
            raise RuntimeError(f"{path.name}: {key} are {dtype} {found}, expected uint8 {shape}, "
                               "a file of another build; remove it")
    return n[0]


def _one(values: pd.Series, what: str) -> str:
    found = sorted(set(values))
    if len(found) != 1:
        raise ValueError(f"{what} is not one value, found {found}")
    return found[0]


def _read_by_ranges(rows: pd.DataFrame, labelled_ids: Sequence[str], plain_ids: Sequence[str],
                    token: str | None, listing: pd.DataFrame | None, workers: int
                    ) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """The planned patches of one zip, halved, by range requests in threads."""
    stem = f"{rows['department'].iloc[0]}-{rows['year'].iloc[0]}"
    fields = rows.drop_duplicates("patch_id").set_index("patch_id")
    rgbi_zip = _one(rows["rgbi_zip"], f"{stem}: rgbi_zip")
    rgbi_url, rgbi_end = _zip_url(rgbi_zip), _listed(listing, rgbi_zip, token)
    labelled = set(labelled_ids)
    if labelled:
        label_zip = _one(rows.loc[rows["patch_id"].isin(labelled), "label_zip"], f"{stem}: label_zip")
        label_url, label_end = _zip_url(label_zip), _listed(listing, label_zip, token)

    def one(patch_id: str) -> tuple[str, tuple[np.ndarray, np.ndarray | None]]:
        row = fields.loc[patch_id]
        name = member_name(rgbi_zip, patch_id)
        image, grid = _decode(fetch_member(rgbi_url, row["rgbi_offset"], row["rgbi_size"], name,
                                           row["rgbi_method"], row["rgbi_crc"], token, rgbi_end),
                              f"{rgbi_zip} {name}")
        _check_image(image, f"{rgbi_zip} {name}")
        label = None
        if patch_id in labelled:
            name = member_name(label_zip, patch_id)
            label, label_grid = _decode(fetch_member(label_url, row["label_offset"], row["label_size"],
                                                     name, row["label_method"], row["label_crc"],
                                                     token, label_end), f"{label_zip} {name}")
            _check_label(label, label_grid, grid, f"{label_zip} {name}")
        return patch_id, (halve_image(image), None if label is None else halve_label(label))

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        return dict(pool.map(one, [*labelled_ids, *plain_ids]))


def _read_from_zips(rows: pd.DataFrame, labelled_ids: Sequence[str], plain_ids: Sequence[str],
                    token: str | None, listing: pd.DataFrame | None, work_dir: Path,
                    keep_zips: bool, progress: bool
                    ) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """The planned patches of one zip, halved, from the whole zips downloaded to work_dir."""
    stem = f"{rows['department'].iloc[0]}-{rows['year'].iloc[0]}"
    rgbi_zip = _one(rows["rgbi_zip"], f"{stem}: rgbi_zip")
    label_zip = (_one(rows.loc[rows["patch_id"].isin(labelled_ids), "label_zip"], f"{stem}: label_zip")
                 if labelled_ids else None)
    rgbi_path = download_zip(rgbi_zip, work_dir, token, listing, progress)
    label_path = download_zip(label_zip, work_dir, token, listing, progress) if label_zip else None
    read = {}
    for patch_id, image, label in iter_patches(rgbi_path, label_path, labelled_ids):
        read[patch_id] = (halve_image(image), halve_label(label))
    for patch_id, image, _ in iter_patches(rgbi_path, None, plain_ids):
        read[patch_id] = (halve_image(image), None)
    if not keep_zips:
        rgbi_path.unlink()
        if label_path is not None:
            label_path.unlink()
    return read


def build_sets(plan: pd.DataFrame, work_dir: str | Path, out_dir: str | Path,
               token: str | None = None, listing: pd.DataFrame | None = None,
               how: str = "ranges", workers: int = 8, keep_zips: bool = False,
               progress: bool = True) -> dict:
    """Reads the planned patches zip by zip, halves them, writes the sets.

    how       "ranges" = one range request per member, in `workers` threads, nothing is
              downloaded whole (the default). "zips" = each needed zip is downloaded once to
              work_dir and read in place, then deleted unless keep_zips.

    One file out_dir/<set>__<DEPT>-<YEAR>.npz per set and zip, holding images uint8
    (N, 4, OUT_PX, OUT_PX), labels uint8 (N, OUT_PX, OUT_PX) for the labelled sets, ids (N,)
    and the department and year, rows in the order of the plan; the same bytes either way.
    A file that is there is not made again (a file is renamed into place only when complete),
    so a stopped build goes on where it stopped and a finished one reads nothing. Labels are
    read only for the labelled sets that are still to be made. What was read is printed per
    department and in total from majortom.STATS. Returns the manifest, also written to
    out_dir/manifest.json (sets, files, counts, class shares, the pinned revision).
    """
    if how not in ("ranges", "zips"):
        raise ValueError(f"how is {how!r}, expected 'ranges' or 'zips'")
    plan = _checked_plan(plan)
    work_dir, out_dir = Path(work_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    at_start, wall_start, patches = dict(M.STATS), time.perf_counter(), 0
    for (department, year), rows in plan.groupby(["department", "year"], sort=True):
        todo = {name: group for name, group in rows.groupby("set", sort=True)
                if not (out_dir / _file_name(name, department, year)).is_file()}
        if not todo:
            continue
        with_labels = [name for name in todo if name in LABELLED]
        labelled_ids = sorted(set(pd.concat([todo[n] for n in with_labels])["patch_id"])) if with_labels else []
        plain_ids = sorted(set(pd.concat(todo.values())["patch_id"]) - set(labelled_ids))
        before, begin = dict(M.STATS), time.perf_counter()
        if how == "ranges":
            read = _read_by_ranges(rows, labelled_ids, plain_ids, token, listing, workers)
        else:
            read = _read_from_zips(rows, labelled_ids, plain_ids, token, listing, work_dir,
                                   keep_zips, progress)
        for name, group in todo.items():
            picked = group["patch_id"].tolist()
            arrays = {"images": np.stack([read[p][0] for p in picked]),
                      "ids": np.array(picked, dtype=str), "department": department, "year": year}
            if name in LABELLED:
                arrays["labels"] = np.stack([read[p][1] for p in picked])
            _save_npz(out_dir / _file_name(name, department, year), arrays)
        patches += len(read)
        if progress:
            print(f"{department}-{year}: {len(read)} patches into {len(todo)} sets, "
                  f"{_cost(_paid(before), time.perf_counter() - begin)}")
    if progress:
        print(f"built {patches} patches, {_cost(_paid(at_start), time.perf_counter() - wall_start)}")
    manifest = _manifest(plan, out_dir)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _manifest(plan: pd.DataFrame, out_dir: Path) -> dict:
    """The sets as built, checked against the plan (every planned patch, nothing else) and
    against the layout (every file holds arrays of this build's shape, checked from the npy
    headers, so a leftover file of another build stops here and not at load time)."""
    sets = {}
    for name, group in plan.groupby("set", sort=False):
        zips = sorted(set(zip(group["department"], group["year"])))
        files = [_file_name(name, d, y) for d, y in zips]
        ids, pixels = [], np.zeros(N_CLASSES, dtype=np.int64)
        for file in files:
            _check_file(out_dir / file, name in LABELLED)
            with np.load(out_dir / file) as z:
                ids += z["ids"].tolist()
                if name in LABELLED:
                    pixels += np.bincount(z["labels"].ravel(), minlength=N_CLASSES)[:N_CLASSES]
        if sorted(ids) != sorted(group["patch_id"]):
            raise RuntimeError(f"{name}: the files hold other patches than the plan, "
                               f"{len(ids)} against {len(group)}")
        sets[name] = {"count": len(ids), "files": files,
                      "departments": sorted(set(group["department"])),
                      "class_shares": ((pixels / pixels.sum()).round(6).tolist()
                                       if name in LABELLED else None)}
    return {"repo": HF_REPO, "revision": HF_REVISION, "updated": HF_UPDATED, "bands": list(BANDS),
            "pixel_m": PIXEL_M, "out_px": OUT_PX, "classes": list(CLASSES),
            "halving": {"image": "mean of each 2 x 2 block, rounded half up",
                        "label": "majority of each 2 x 2 block, a tie goes to the smallest code"},
            "sets": sets}


def _load_set(name: str, sources: Sequence[tuple[str, Callable[[], IO[bytes]]]]) -> dict:
    """The arrays of one set from its npz files, each given as (its name, a function that opens it).

    The function gives a fresh binary file object that can seek, closed here after use. The
    arrays are made once at their full size from the npy headers and the files are copied
    into them one at a time, so the peak memory is the set plus one file, not twice the set
    (19,000 patches of 256 px are 5 GB).
    """
    keys = ("images", "labels") if name in LABELLED else ("images",)
    heads = []
    for where, opener in sources:
        with opener() as f:
            heads.append(_npz_headers(f, ("ids", *keys), where))
    counts = [h["ids"][0][0] for h in heads]
    out = {}
    for key in keys:
        shape, dtype = heads[0][key]
        for (where, _), k, h in zip(sources, counts, heads, strict=True):
            s, d = h[key]
            if s != (k, *shape[1:]) or d != dtype:
                raise RuntimeError(f"{where}: {key} are {d} {s}, the first file of the set has "
                                   f"{dtype} {shape}, files of two builds; remove the wrong one")
        out[key] = np.empty((sum(counts), *shape[1:]), dtype=dtype)
    ids, departments, at = [], [], 0
    for (where, opener), k in zip(sources, counts, strict=True):
        with opener() as f, np.load(f) as z:
            for key in keys:
                out[key][at:at + k] = z[key]
            ids.append(z["ids"])
            departments.append(np.repeat(str(z["department"]), k))
        at += k
    return {"images": out["images"], "labels": out.get("labels"),
            "ids": np.concatenate(ids), "departments": np.concatenate(departments)}


def load_set(out_dir: str | Path, name: str) -> dict:
    """One built set, over all its files in a folder.

    images (N, 4, OUT_PX, OUT_PX) uint8, labels (N, OUT_PX, OUT_PX) uint8 or None for an
    unlabelled set, ids (N,) str, departments (N,) str. Needs the manifest, so only a finished
    build loads. `load_set_from_tar` gives the same arrays straight out of a set's archive.

    The arrays are made once at their full size from the npy headers and the files are copied
    into them one at a time, so the peak memory is the set plus one file, not twice the set
    (19,000 patches of 256 px are 5 GB).
    """
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    if name not in manifest["sets"]:
        raise KeyError(f"{out_dir}: no set {name!r}, the manifest has {sorted(manifest['sets'])}")
    files = [out_dir / f for f in manifest["sets"][name]["files"]]
    return _load_set(name, [(f.name, lambda f=f: open(f, "rb")) for f in files])


def load_set_from_tar(archive: str | Path, name: str, files: Sequence[str]) -> dict:
    """One built set straight out of its tar archive, nothing is unpacked to the disk.

    `files` are the npz files of the set as the manifest lists them (its `files` entry, in
    order). Each member is read through tarfile.extractfile, a file object over the member's
    bytes that can seek, so np.load reads the npz in place, and the arrays are filled as in
    `load_set` (made once at full size, one file at a time). The arrays equal those of
    `load_set` on the unpacked files. A member the tar does not hold is a FileNotFoundError,
    a tar that cannot be read a tarfile.ReadError.
    """
    archive = Path(archive)
    files = list(files)
    with tarfile.open(archive) as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        missing = [f for f in files if f not in members]
        if missing:
            raise FileNotFoundError(f"{archive} does not hold {missing[:3]}, run notebook 01 again")
        return _load_set(name, [(f"{archive.name}::{f}", lambda f=f: tar.extractfile(members[f]))
                                for f in files])


# ------------------------------------------------------------------ licence

ATTRIBUTION = (
    "Aerial patches from FLAIR-HUB by IGN (Institut national de l'information geographique et "
    "forestiere), Etalab Open Licence 2.0 - original data downloaded from "
    "https://huggingface.co/datasets/IGNF/FLAIR-HUB (revision " + HF_REVISION[:12]
    + ", updated on " + HF_UPDATED + "), patches halved to 0.4 m by this project. "
    "Cite Garioud, Giordano, David and Gonthier, FLAIR-HUB - Large-scale multimodal dataset for "
    "land cover and crop mapping, ISPRS Journal of Photogrammetry and Remote Sensing 237, "
    "271-300, 2026, doi 10.1016/j.isprsjprs.2026.04.017 (arXiv 2506.07080)."
)


def print_attribution() -> str:
    """Prints and returns the licence and attribution text to put next to any published number."""
    print(ATTRIBUTION)
    return ATTRIBUTION
