"""Dry run of notebooks/01_data.ipynb on this PC, no network, no GPU, no real data.

Every code cell but the mount cell, the sync cell and the flush cell (drive.flush_and_unmount)
is executed in order and unchanged.
What a cell would reach on Colab is replaced from OUTSIDE, by patching names of the modules

    folders /content and Drive    stage0.WORK_FOLDERS, a temp folder on D
    .env file and token           env.load_env reads a fake .env of the temp folder that holds
                                  the two data settings DATA_HF_REPO and DATA_DRIVE_DIR (the
                                  real parser, made-up values), env.require gives a made-up
                                  token. The Drive layout and the repo id follow those values
    pip install                   subprocess.run, a pip call is noted and not run
    Major-TOM index and rows      majortom.list_s2_shards and majortom.load_s1_index, small
                                  local shards made with the helpers of test_majortom.py
    FLAIR-HUB                     majortom.HF points at a server on this machine that serves
                                  small zips with the real layout, as test_aerial.py does
    the sizes of the plan         pilotdata.CHUNK, N_HELD_OUT and N_STATS, aerial.SET_SIZES,
                                  PATCH_PX and OUT_PX, small so the cells run in seconds. The
                                  notebook prints whatever the modules hold, it retypes nothing
    load_pairs_to_ram             replaced by a function that stops the test, the notebook
                                  must never load all pairs into RAM
    the private Hugging Face repo hub.hub_api gives a stand-in HfApi (create_repo, repo_info,
                                  list_repo_tree, upload_file over a dict), so the real
                                  upload_pilot runs against it

A connection to another machine stops the test. The made-up tokens must not show up in
anything the cells print or save. The upload cell runs without the write token (it says so
and goes on), then alone with the token (everything uploaded, the card written to Drive),
again (everything skipped), against a public repo and against a Hub that fails (both
printed, nothing stops).

Runs. The first one from nothing. The second one as on a new runtime, where the local disk
is empty and Drive is kept, so nothing may be fetched and no archive written. The third one
in the same session with the index files moved to the stage 0 folder. Then the guards, a
sample and a plan on Drive that differ from the draw, a cut aerial archive and a cut chunk
archive on Drive (the set and the chunk are made again, nothing else), a chunk archive on
Drive without its infos table (built once from the tar, nothing fetched), and the setup cell
with the library missing, with Drive and the runtime disk nearly full and without Drive.
One synthetic cell holds a band that is not an image, so the read_error path of the fetch
cell (the error table, the warning that names the chunk) and of the summary runs too.

STAND-INS ONLY, so NOT tested against the real thing by this file - the mount cell, the
sync cell and the flush cell as cells, pip, the real .env file and the tokens, every read from Hugging Face (shard
list, footers, rows, metadata, the FLAIR-HUB listing and zips), the real sizes (26,000 cells,
2,000 held out, the aerial sets), the speed of 28 threads and the Hub's rate limit, Drive as a file system, the RAM
of a standard runtime, and the library versions of Colab.
"""
import _setup

import ast
import contextlib
import http.server
import importlib.util
import io
import json
import os
import re
import shutil
import socket
import subprocess
import tarfile
import threading
import time
import traceback
import zipfile
from pathlib import Path

os.environ["IPYTHONDIR"] = str(_setup.TMP_BASE / "ipython")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import Affine  # noqa: E402

from sartransfer import env as E  # noqa: E402
from sartransfer import sync as SYNC  # noqa: E402
from sartransfer.distil import aerial as A  # noqa: E402
from sartransfer.distil import majortom as M  # noqa: E402
from sartransfer.distil import pilotdata as D  # noqa: E402
from sartransfer.distil import records as R  # noqa: E402
from sartransfer.distil import stage0 as Z  # noqa: E402
from sartransfer.domains import hub as H  # noqa: E402

NOTEBOOK = _setup.REPO / "notebooks" / "01_data.ipynb"
TMP = _setup.TMP_BASE / "notebook01"
for _ in range(40):                    # Windows may take a moment to let go of a folder
    shutil.rmtree(TMP, ignore_errors=True)
    if not TMP.exists():
        break
    time.sleep(0.25)
TMP.mkdir(parents=True)                # files of an older run would change what is tested
CONTENT = TMP / "content"                                  # stands for /content
MYDRIVE = CONTENT / "drive" / "MyDrive"                    # stands for /content/drive/MyDrive
FAKE_REPO = "made-up-owner/made-up-pilot-data"       # DATA_HF_REPO of the fake .env, owner/name
FAKE_DRIVE_DIR = "made-up-data-folder"               # DATA_DRIVE_DIR of the fake .env, under MyDrive
PILOT = MYDRIVE / FAKE_DRIVE_DIR / "pilot"           # the notebook derives these from the settings
STAGE0 = MYDRIVE / FAKE_DRIVE_DIR / "stage0"
DRIVE_MT, DRIVE_AERIAL = PILOT / "majortom", PILOT / "aerial"
WORK = CONTENT / FAKE_DRIVE_DIR / "pilot"
FAKE_ENV = TMP / "fake.env"                          # the two data settings, no token
SHARDS = TMP / "majortom"
AERIAL = TMP / "aerial"
for folder in (MYDRIVE, SHARDS, AERIAL / "data"):
    folder.mkdir(parents=True)
FAKE_TOKEN = "MADE-UP-TOKEN-OF-THE-DRY-RUN"
FAKE_WRITE_TOKEN = "MADE-UP-WRITE-TOKEN-OF-THE-DRY-RUN"
HAD_CONTENT = Path("/content").exists()         # the folder that a cell must not write to
HAD_WRITE_TOKEN = os.environ.pop("HF_WRITE_TOKEN", None)   # the upload cell reads it from os.environ
HAD_DATA = {k: os.environ.pop(k, None) for k in ("DATA_HF_REPO", "DATA_DRIVE_DIR")}   # the fake .env sets them
FAKE_ENV.write_text(f"DATA_HF_REPO={FAKE_REPO}\nDATA_DRIVE_DIR={FAKE_DRIVE_DIR}\n", encoding="utf-8")

# ---- the file -----------------------------------------------------------------------------
NAMES = ["kept", "intro", "mount", "sync", "setup", "index", "sample", "fetch", "held-out", "aerial-plan",
         "aerial-build", "stats", "summary", "upload", "flush"]
(KEPT, INTRO, MOUNT, SYNC_CELL, SETUP, INDEX, SAMPLE, FETCH, HELD_OUT, AERIAL_PLAN, AERIAL_BUILD, STATS, SUMMARY,
 UPLOAD, FLUSH) = range(len(NAMES))
STEPS = ["setup", "index", "sample", "fetch", "held_out", "aerial_plan", "aerial_build", "stats",
         "summary", "upload"]
text = NOTEBOOK.read_text(encoding="utf-8")
nb = json.loads(text)
assert (nb["nbformat"], nb["nbformat_minor"]) == (4, 5)
assert nb["metadata"]["kernelspec"]["name"] == "python3"
# the short markdown cell before each code cell, id md-<code cell id>, no colon, no output
notes = [c for c in nb["cells"] if c["cell_type"] == "markdown" and c["id"].startswith("md-")]
assert len(notes) == 13 and all(":" not in "".join(c["source"]) for c in notes), "a colon in a markdown cell"
cells = [c for c in nb["cells"] if not (c["cell_type"] == "markdown" and c["id"].startswith("md-"))]
assert [c["cell_type"] for c in cells] == ["markdown", "markdown"] + ["code"] * 13
assert [c["id"] for c in notes] == ["md-" + n for n in NAMES[MOUNT:]]
assert [c["id"] for c in cells] == NAMES
SOURCE = ["".join(c["source"]) for c in cells]
for k, cell in enumerate(cells[MOUNT:], MOUNT):
    assert cell["outputs"] == [] and cell["execution_count"] is None, f"cell {k} holds output"
try:
    import nbformat
except ImportError:
    print("nbformat is not installed, the file was not checked against the notebook schema")
else:
    nbformat.validate(nbformat.reads(text, as_version=4))

# the text a reader sees. One line at the top says the notebook is kept and not run here. Then
# the intro, no colon at all, and the facts that the task asks for
kept = SOURCE[KEPT]
assert len(kept.strip().splitlines()) == 1 and ":" not in kept, "the kept line is not one line"
for words in ("rebuilt", "NOT run in part four", "stays where it is", ".env file"):
    assert words in kept, f"the kept line does not say {words!r}"
intro = SOURCE[INTRO]
assert intro.startswith("# ") and ":" not in intro, "a colon in the markdown cell"
for words in ("standard runtime", "no GPU", "12.7 GB", "100 GB", "16 MB per cell", "7 s per cell",
              "1.1 cells per second", "28 threads", f"{D.N_PILOT:,} cells", f"about {D.N_PAIRS:,} pairs",
              "393 of 500", "13 chunks", "estimate", "300 GB", "up to 62 GB", "about 20 GB", "up to 70 GB",
              "28 threads", "about 1.5 hours", "built on the runtime disk and copied to Drive",
              # measured on 2026-09-30 with 16 threads, the reason for 28, what a re-run does
              "Measured on 2026-09-30 with 16 threads", "470 to 576 s", "about 8 requests per second",
              "5,000 requests per 5 minutes", "requests per second and the waits", "11 of 13 chunks",
              "infos table", "one chunk at a time", "deleted once its archive is complete", "never unpacked",
              "read in place", "nothing is fetched or unpacked", "built from the tar once",
              "32,000 range requests", "24 GB", "DATA_DRIVE_DIR of the .env file under MyDrive", "majortom/",
              "aerial/", "VS Code", "never printed", "2,000 pairs held out", "seed 0", "read_error",
              "flush of Drive", "run the mount cell again",
              # the upload to the private repo, as part two did
              "named by DATA_HF_REPO in the .env file", "HF_WRITE_TOKEN", "same size is skipped", "4 files at once",
              "Without the token the cell says so", "refuses a public one", "README.md",
              "notebooks 02 and 03 read from the repo first",
              # the data settings, the notebook names no folder and no repo
              "DATA_HF_REPO (the private repo of the pilot data, owner/name)",
              "DATA_DRIVE_DIR (the folder under MyDrive that holds pilot/ and stage0/)",
              "names no folder and no repo"):
    assert words in intro, f"the first cell does not say {words!r}"
# no name of another repository or of its folders in any cell (the sync cell holds base64 only)
FORBIDDEN = _setup.private_names() + ("HF_OWNER", "part three")
for k, source in enumerate(SOURCE):
    if k != SYNC_CELL:
        hits = [w for w in FORBIDDEN if w in source]
        assert not hits, f"cell {k} ({NAMES[k]}) names another repository or its folder, {hits}"
for k in range(MOUNT, len(cells)):
    first = SOURCE[k].splitlines()[0]
    assert first.startswith("# ") and ":" not in first, f"cell {k} starts with {first!r}"

assert "google.colab" in SOURCE[MOUNT] and 'drive.mount("/content/drive")' in SOURCE[MOUNT]
# the last cell flushes Drive and says that the mount cell must be run again. Not run here
assert "google.colab" in SOURCE[FLUSH] and "drive.flush_and_unmount()" in SOURCE[FLUSH]
assert "mount cell" in SOURCE[FLUSH] and ":" not in SOURCE[FLUSH] and "HF_TOKEN" not in SOURCE[FLUSH]
assert not re.search(r"^\s*[!%]", SOURCE[FLUSH], re.M) and "print(" in SOURCE[FLUSH]
lines = SOURCE[SYNC_CELL].splitlines()
assert lines[0] == SYNC.BEGIN and lines[-1] == SYNC.END
assert nb["cells"][SYNC.cell_index(nb)]["id"] == "sync" and cells[SYNC_CELL]["id"] == "sync"
SYNC_STATE = _setup.check_sync_cell(nb, SYNC)            # packed and current, or cleared in a published copy

CODE = {}
for k in range(SETUP, FLUSH):
    source = SOURCE[k]
    assert not re.search(r"^\s*[!%]", source, re.M), f"cell {k} holds a notebook magic"
    assert not re.search(r"\bC:[\\/]", source), f"cell {k} names a path on C"
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", source), f"cell {k} holds a token"
    assert "getpass" not in source, f"cell {k} asks for a token on its own"
    assert "git clone" not in source and "torch.hub" not in source
    assert "print(HF_TOKEN" not in source and "{HF_TOKEN" not in source
    assert "print(WRITE_TOKEN" not in source and "{WRITE_TOKEN" not in source
    assert re.search(r'print\(took\(start, "[a-z_]+"\)\)$', source.rstrip()), f"cell {k} does not print its time"
    assert "start = time.perf_counter()" in source
    CODE[k] = compile(source, f"01_data cell {k} ({NAMES[k]})", "exec")
# what the setup cell asks of the runtime, and what it must not do
for words in ('("rasterio",)', "env.load_env()", 'env.require("HF_TOKEN"', "stage0.runtime_report(",
              "DRIVE_NEED_GB = 70", "DISK_NEED_GB = 20", "WORKERS = 28", "AERIAL_WORKERS = 28",
              "5,000", "3,000 anonymous", "2,400", "def paid_since(before)",
              "pilotdata.N_PILOT",
              "pilotdata.N_PAIRS", "pilotdata.N_HELD_OUT", "pilotdata.N_STATS", "pilotdata.SEED",
              "aerial.SET_SIZES", "from sartransfer.domains import hub",
              # the data settings of the .env file, the Drive folders and the repo derived from them
              'DATA_DIR = env.require("DATA_DRIVE_DIR"', "DATA_REPO = hub.data_repo_id()",
              'PILOT = MYDRIVE / DATA_DIR / "pilot"', 'STAGE0 = MYDRIVE / DATA_DIR / "stage0"',
              'WORK = CONTENT / DATA_DIR / "pilot"'):
    assert words in SOURCE[SETUP], f"the setup cell has no {words}"
# the upload cell, the write token from the environment, never asked for, gone at the end
for words in ('os.environ.get("HF_WRITE_TOKEN")', "hub.upload_pilot(PILOT, token=WRITE_TOKEN)",
              "hub.list_pilot(token=WRITE_TOKEN)", "del WRITE_TOKEN", "except Exception",
              "{DATA_REPO}", "hub.UPLOAD_WORKERS"):
    assert words in SOURCE[UPLOAD], f"the upload cell has no {words}"
assert "require(" not in SOURCE[UPLOAD], "the upload cell asks for the write token"
assert "torch" not in "".join(SOURCE[SETUP:]), "the data notebook imports torch"
assert "cuda" not in "".join(SOURCE[SETUP:]), "the data notebook asks for a GPU"
assert not re.search(r"\bpip\b", "".join(SOURCE[SETUP + 1:])), "pip outside the setup cell"
assert "load_pairs_to_ram" not in "".join(SOURCE), "the data notebook loads all pairs into RAM"
assert "extractall" not in "".join(SOURCE[SETUP:AERIAL_PLAN]) and "extractall" not in "".join(SOURCE[STATS:]), \
    "a pair archive is unpacked"
assert "ThreadPoolExecutor" not in "".join(SOURCE), "the summary reads the pair files again"
for words in ("26000", "26_000", "20000", "20_000", "2000", "2_000", "(1, 2, 3)", "[1, 2, 3]"):
    assert words not in "".join(SOURCE[SETUP:]), f"a pre-registered value is retyped, {words}"
# the token goes to the Major-TOM reads and to the FLAIR-HUB reads, the rules come from the modules
for k, call in ((INDEX, "build_s2_index(cache=index_files[\"s2\"], token=HF_TOKEN)"),
                (INDEX, "load_s1_index(cache=index_files[\"s1\"], token=HF_TOKEN)"),
                (INDEX, "max_gap_days=stage0.MAX_GAP_DAYS"),
                (SAMPLE, "pilot_sample(pairs, n=N_PILOT, seed=SEED)"),
                (FETCH, "cloud_codes=CLOUD_CODES"), (FETCH, 'offset="baseline"'), (FETCH, "workers=WORKERS"),
                (FETCH, "pilotdata.N_PAIRS"), (FETCH, "chunk_<k>.tar and chunk_<k>.json"),
                (FETCH, 'pilotdata.load_infos(DRIVE_MT, sample["chunk"].unique())'),
                (FETCH, "pilotdata.locations_of(infos, DRIVE_MT) == paths"), (FETCH, "5,000 per 5 minutes"),
                (AERIAL_BUILD, "paid_since(before)"), (STATS, "channel_stats(train_paths, n=N_STATS"),
                (SUMMARY, "infos[list(pilotdata.META_COLUMNS)]"), (SUMMARY, "pilotdata.pair_cell(p) for p in held_paths"),
                (HELD_OUT, "n=N_HELD_OUT, seed=SEED"), (AERIAL_PLAN, "list_zips(token=HF_TOKEN"),
                (AERIAL_PLAN, "list_patches(listing, cache="), (AERIAL_PLAN, "plan_sets(patches)"),
                (AERIAL_BUILD, 'how="ranges"'), (AERIAL_BUILD, "workers=AERIAL_WORKERS"), (STATS, "n=N_STATS, seed=SEED"),
                (STATS, 'aerial=distil_train["images"]'), (SUMMARY, "pilot_summary(meta, skips)")):
    assert call in SOURCE[k], f"cell {k} has no {call}"
print(f"the file is a notebook of {len(cells)} cells without outputs, the sync cell is up to date")

# ---- synthetic Major-TOM shards, made with the helpers of test_majortom.py -----------------
HELPERS = ("tif", "s2_row", "s1_row", "write_shard")
tree = ast.parse((_setup.REPO / "tests" / "test_majortom.py").read_text(encoding="utf-8"))


def is_corner(node) -> bool:
    """The line LEFT, TOP = ... of that file."""
    target = node.targets[0] if isinstance(node, ast.Assign) else None
    return (isinstance(target, ast.Tuple)
            and [getattr(e, "id", "") for e in target.elts] == ["LEFT", "TOP"])


taken = [node for node in tree.body
         if isinstance(node, ast.FunctionDef) and node.name in HELPERS or is_corner(node)]
found = {getattr(node, "name", "LEFT, TOP") for node in taken}
assert found == {*HELPERS, "LEFT, TOP"}, f"test_majortom.py no longer has {HELPERS}, found {found}"
helpers = {"np": np, "pd": pd, "pa": pa, "pq": pq, "rasterio": rasterio, "Affine": Affine,
           "M": M, "R": R, "DIR": SHARDS}
exec(compile(ast.Module(taken, []), "helpers of test_majortom.py", "exec"), helpers)
tif, s2_row, s1_row, write_shard = (helpers[name] for name in HELPERS)
LEFT, TOP = helpers["LEFT"], helpers["TOP"]
S1_CORNER = {"left": LEFT - 3420, "top": TOP + 3420, "nodata": -32768.0}

# cell  Sentinel-2 day, baseline  Sentinel-1 day   what
PLAN = [("20170610", "0205", "20170611", "pair"),
        ("20180610", "0208", "20180612", "pair"),
        ("20190610", "0213", "20190609", "pair"),
        ("20200610", "0214", "20200611", "pair"),
        ("20210610", "0300", "20210611", "pair"),
        ("20220310", "0400", "20220311", "pair, offset"),
        ("20230610", "0509", "20230611", "pair, offset"),
        ("20240610", "0510", "20240611", "pair, offset"),
        ("20220810", "0400", "20220811", "Sentinel-1 no-data inside the crop"),
        ("20230910", "0510", "20230911", "pair, offset"),
        ("20210710", "0301", "20210711", "cloudy"),
        ("20210720", "0301", "20210721", "a Sentinel-2 zero inside the crop"),
        ("20230920", "0510", "20230921", "pair, offset"),
        ("20210910", "0301", "20210930", "20 days apart"),
        ("20211010", "0301", None, "no Sentinel-1"),
        ("20210810", "0301", "20210811", "a band that is not an image, a read_error")]
CELL = [f"{100 + i}U_{200 + i}R" for i in range(len(PLAN))]
KEPT = [0, 1, 2, 3, 4, 5, 6, 7, 9, 12]
NO_DATA, CLOUDY, ZERO, FAR, ALONE, BROKEN = 8, 10, 11, 13, 14, 15
PAIRED = [*range(13), BROKEN]                     # the cells of the index within 7 days
FETCH_SKIPS = {CELL[NO_DATA]: "s1_nodata_in_crop", CELL[CLOUDY]: "cloud_gt_10pct",
               CELL[ZERO]: "s2_nodata_in_crop", CELL[BROKEN]: "read_error"}

s2_rows, s1_rows = [], []
for i, (day, baseline, s1_day, what) in enumerate(PLAN):
    rng = np.random.default_rng(1000 + i)
    change = {"product_id": f"S2A_MSIL2A_{day}T101010_N{baseline}_R022_T33TUM_{day}T121212",
              "product_datetime": f"{day}T101010"}
    low, high = (1100, 4000) if int(baseline) >= 400 else (100, 3000)     # the offset, or not
    for band in R.S2_BANDS:
        px = R.CROP_PX * 10 // R.S2_RES_M[band]
        values = rng.integers(low, high, (px, px)).astype(np.uint16)
        if i == ZERO and band == "B05":
            values[10, 10] = 0
        change[band] = tif(values, R.S2_RES_M[band])
        if i == BROKEN and band == "B8A":
            change[band] = b"not a GeoTIFF at all"
    mask = np.zeros((R.CROP_PX, R.CROP_PX), np.uint8)
    if i == CLOUDY:                                    # 20 % thick cloud, a little of the rest
        mask[:77] = 2
        mask[77:80] = 1
        mask[80:83] = 3
    change["cloud_mask"] = tif(mask)
    s2_rows.append(s2_row(i, **change))
    assert s2_rows[-1]["grid_cell"] == CELL[i]
    if s1_day is None:
        continue
    change = {"product_id": f"S1A_IW_GRDH_1SDV_{s1_day}T051500_{s1_day}T051525_0_0_rtc",
              "product_datetime": f"{s1_day}T051500"}
    if i == NO_DATA:
        holes = rng.uniform(0.001, 0.5, (R.S1_CELL_PX, R.S1_CELL_PX)).astype(np.float32)
        holes[500, 500] = -32768.0
        change["vh"] = tif(holes, **S1_CORNER)
    s1_rows.append(s1_row(i, size=R.S1_CELL_PX, **change))
    assert s1_rows[-1]["grid_cell"] == CELL[i]

S2_SHARDS = [write_shard(f"s2_{k}.parquet", s2_rows[a:b]) for k, (a, b) in
             enumerate(((0, 5), (5, 10), (10, 16)))]
S1_PARTS = ((0, 4), (4, 8), (8, 15))
S1_SHARDS = [write_shard(f"s1_{k}.parquet", s1_rows[a:b]) for k, (a, b) in enumerate(S1_PARTS)]
METADATA = SHARDS / "metadata.parquet"
pd.DataFrame({
    "grid_cell": [row["grid_cell"] for row in s1_rows],
    "product_id": [row["product_id"] for row in s1_rows],
    "timestamp": [row["product_datetime"][:-2] + "12" for row in s1_rows],   # 12 s after
    "nodata": 0.0, "orbit_state": "ascending", "centre_lat": 36.1, "centre_lon": 15.0,
    "crs": "EPSG:EPSG:32633",
    "parquet_url": [S1_SHARDS[k] for k, (a, b) in enumerate(S1_PARTS) for _ in range(a, b)],
    "parquet_row": [n for a, b in S1_PARTS for n in range(b - a)],
}).to_parquet(METADATA, index=False)
del s2_rows, s1_rows, holes
megabytes = sum(p.stat().st_size for p in SHARDS.iterdir()) / 1e6
print(f"synthetic Major-TOM, {len(PLAN)} cells in {len(S2_SHARDS)} + {len(S1_SHARDS)} shards, "
      f"{megabytes:.0f} MB in {SHARDS}")

# ---- synthetic FLAIR-HUB zips with the real layout, served from this machine --------------
PX = 32                                  # small patches, halved to 16 px
SIZES = {"distil": 20, "distil_heldout": 4, "probe": 8, "val": 6, "test": 6}
FAKE_ZIPS = {("D004", "2021"): {"AA-S1-32": 6, "AB-S1-7": 6}, ("D017", "2018"): {"UU-S1-3": 6},
             ("D017", "2021"): {"UU-S1-3": 6}, ("D045", "2020"): {"CC-S1-1": 4},
             ("D012", "2019"): {"AF-S1-27": 8}, ("D036", "2020"): {"ZZ-S1-9": 8}}


def aerial_tif(array, left, top, res=0.2) -> bytes:
    """One GeoTIFF as bytes, as in the zips (uint8, Lambert-93, 0.2 m, LZW)."""
    array = array if array.ndim == 3 else array[None]
    with rasterio.MemoryFile() as memory:
        with memory.open(driver="GTiff", count=array.shape[0], height=array.shape[1],
                         width=array.shape[2], dtype=array.dtype, crs="EPSG:2154",
                         transform=Affine(res, 0.0, left, 0.0, -res, top), compress="lzw") as dst:
            dst.write(array)
        return memory.read()


TRUTH = {}                               # patch id -> (image, label)
seed = 0
for (dept, year), rois in FAKE_ZIPS.items():
    members = {"AERIAL_RGBI": {}, "AERIAL_LABEL-COSIA": {}}
    for roi, n in rois.items():
        for k in range(n):
            patch_id = f"{dept}-{year}_{roi}_{k // 3 + 1}-{k % 3 + 1}"
            rng = np.random.default_rng(seed)
            image = rng.integers(0, 256, (4, PX, PX), dtype=np.uint8)
            label = rng.integers(0, A.N_CLASSES, (PX, PX), dtype=np.uint8)
            seed += 1
            TRUTH[patch_id] = (image, label)
            left, top = 900000.0 + 100 * seed, 6300000.0 - 100 * seed
            for modality, array in (("AERIAL_RGBI", image), ("AERIAL_LABEL-COSIA", label)):
                stem = f"{dept}-{year}_{modality}"
                members[modality][A.member_name(stem, patch_id)] = aerial_tif(array, left, top)
    for modality, files in members.items():
        with zipfile.ZipFile(AERIAL / "data" / f"{dept}-{year}_{modality}.zip", "w",
                             zipfile.ZIP_DEFLATED) as z:
            for name, data in files.items():
                z.writestr(name, data)
LISTING = pd.DataFrame([{"department": d, "year": y, "modality": m, "path": f"data/{d}-{y}_{m}.zip",
                         "bytes": os.path.getsize(AERIAL / "data" / f"{d}-{y}_{m}.zip")}
                        for d, y in FAKE_ZIPS for m in A.MODALITIES]).sort_values("path", ignore_index=True)
HITS = []                                # (path, headers) of every request to the server


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        path, _, _ = self.path.partition("?")
        HITS.append((path, dict(self.headers)))
        if "/api/datasets/" in path:
            entries = [{"type": "file", "path": p, "size": int(b), "oid": "x"}
                       for p, b in zip(LISTING["path"], LISTING["bytes"])]
            return self.answer(200, json.dumps(entries).encode())
        file = AERIAL / "data" / path.rpartition("/")[2]
        if not file.is_file():
            return self.answer(404)
        data = file.read_bytes()
        asked = self.headers.get("Range")
        if not asked:
            return self.answer(200, data)
        a, _, b = asked.removeprefix("bytes=").partition("-")
        start = max(0, len(data) - int(b)) if a == "" else int(a)
        end = len(data) if a == "" or b == "" else min(int(b) + 1, len(data))
        self.answer(206, data[start:end], {"Content-Range": f"bytes {start}-{end - 1}/{len(data)}"})

    def answer(self, code, body=b"", headers=None):
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{server.server_address[1]}"


def zip_hits() -> dict:
    """How often every zip was asked for."""
    out = {}
    for path, _ in HITS:
        if path.endswith(".zip"):
            name = path.rpartition("/")[2]
            out[name] = out.get(name, 0) + 1
    return out


# ---- the stand-in Hub for the upload cell ---------------------------------------------------
class FakeHub:
    """create_repo, repo_info, list_repo_tree, upload_file over a dict of bytes."""

    class Entry:
        def __init__(self, path, size):
            self.path, self.size = path, size

    def __init__(self):
        self.store, self.private, self.created, self.uploads, self.fail = {}, True, [], [], False

    def create_repo(self, rid, repo_type, private, exist_ok):
        assert (repo_type, private, exist_ok) == ("dataset", True, True)
        if self.fail:
            raise OSError(f"stand-in, the Hub is down, the token was {FAKE_WRITE_TOKEN}")
        self.created.append(rid)

    def repo_info(self, rid, repo_type):
        return type("Info", (), {"private": self.private})()

    def list_repo_tree(self, rid, repo_type, recursive):
        assert repo_type == "dataset" and recursive is True
        return [self.Entry(p, len(b)) for p, b in self.store.items()]

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        assert repo_type == "dataset" and repo_id == FAKE_REPO
        self.store[path_in_repo] = Path(path_or_fileobj).read_bytes()
        self.uploads.append(path_in_repo)


hub_fake = FakeHub()
HUB_TOKENS = []


def fake_hub_api(token):
    HUB_TOKENS.append(token)
    assert token == FAKE_WRITE_TOKEN, "the upload used another token than the write token"
    return hub_fake


# ---- the stand-ins, patched in from outside --------------------------------------------------
CALLS = {"pip": [], "list_s2_shards": 0, "s1_metadata_read": 0, "network": [], "env": []}
WRONG = {"libraries missing": False, "drive nearly full": False, "disk nearly full": False,
         "drive not mounted": False}
PATCHED = []


def patch(owner, name: str, value) -> None:
    PATCHED.append((owner, name, getattr(owner, name)))
    setattr(owner, name, value)


def unpatch(since: int = 0) -> None:
    while len(PATCHED) > since:
        owner, name, old = PATCHED.pop()
        setattr(owner, name, old)


REAL = {"run": subprocess.run, "connect": socket.socket.connect, "lookup": socket.getaddrinfo,
        "load_s1_index": M.load_s1_index, "find_spec": importlib.util.find_spec,
        "disk_usage": shutil.disk_usage, "load_pairs_to_ram": D.load_pairs_to_ram, "hub_api": H.hub_api,
        "load_env": E.load_env}
HERE = ("127.0.0.1", "::1", "localhost")


def run_but_pip(args, *more, **kwargs):
    if isinstance(args, (list, tuple)) and "pip" in [str(a) for a in args]:
        CALLS["pip"].append(list(args))
        return subprocess.CompletedProcess(args, 0)
    return REAL["run"](args, *more, **kwargs)


def connect_here_only(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if host not in HERE:
        CALLS["network"].append(str(host))
        raise ConnectionRefusedError(f"the dry run must not use the network, asked for {host}")
    return REAL["connect"](self, address)


def lookup_here_only(host, *args, **kwargs):
    if host not in (*HERE, None, ""):
        CALLS["network"].append(str(host))
        raise socket.gaierror(f"the dry run must not use the network, asked for {host}")
    return REAL["lookup"](host, *args, **kwargs)


def fake_env(*args, **kwargs) -> list:
    """The real parser on the fake .env of the dry run (the two data settings, no token)."""
    CALLS["env"].append("load_env")
    print("stand-in, the fake .env of the dry run is read, not the one on Drive")
    return REAL["load_env"](FAKE_ENV, verbose=False)


def made_up_token(key: str, why: str = "") -> str:
    CALLS["env"].append("require")
    if key == "DATA_DRIVE_DIR":                 # from the fake .env, nothing is asked for
        return os.environ[key]
    assert key == "HF_TOKEN", key
    return FAKE_TOKEN


def local_shards(token=None) -> list:
    assert token == FAKE_TOKEN, "the token was not passed to the index"
    CALLS["list_s2_shards"] += 1
    return list(S2_SHARDS)


def local_s1_index(cache=None, token=None, source=None):
    assert token == FAKE_TOKEN and source is None and cache is not None
    CALLS["s1_metadata_read"] += not Path(cache).is_file()
    return REAL["load_s1_index"](cache=cache, source=METADATA)


def never_in_ram(*args, **kwargs):
    raise SystemExit("the data notebook called load_pairs_to_ram, it must never load all pairs into RAM")


def find_or_not(name, *args, **kwargs):
    if WRONG["libraries missing"] and name == "rasterio":
        return None
    return REAL["find_spec"](name, *args, **kwargs)


def space_or_not(folder):
    use = REAL["disk_usage"](folder)
    on_drive = MYDRIVE in Path(folder).parents
    if (WRONG["drive nearly full"] and on_drive) or (WRONG["disk nearly full"] and not on_drive):
        return type(use)(use.total, use.total - 3 * 10 ** 9, 3 * 10 ** 9)
    return use


def run_cells(which, names: dict, title: str) -> str:
    """Execute these code cells in `names`, unchanged. Returns what they printed."""
    printed = []
    for k in which:
        out = io.StringIO()
        begin = time.perf_counter()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(CODE[k], names)
        except BaseException:
            print(out.getvalue())
            traceback.print_exc()
            raise SystemExit(f"{title}, cell {k} ({NAMES[k]}) raised") from None
        text = out.getvalue()
        print(f"\n==== {title}, cell {k} ({NAMES[k]}), {time.perf_counter() - begin:.1f} s")
        print("\n".join("    | " + line for line in text.splitlines() if "\r" not in line))
        assert FAKE_TOKEN not in text, f"cell {k} printed the token"
        assert FAKE_WRITE_TOKEN not in text, f"cell {k} printed the write token"
        assert "cell time" in text.splitlines()[-1], f"cell {k} did not print its time"
        printed.append(text)
    return "\n".join(printed)


def must_stop(k: int, names: dict, title: str, says: str) -> None:
    """Cell `k` must raise an AssertionError that says `says`."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(CODE[k], names)
    except AssertionError as err:
        assert says in str(err), f"{title}, cell {k} stopped with {err}"
        print(f"\n==== {title}, cell {k} ({NAMES[k]}) stopped as it must, {says!r}")
        return
    raise SystemExit(f"{title}, cell {k} ({NAMES[k]}) ran to its end")


def files_of(folder: Path) -> set:
    return {p.name for p in folder.iterdir()}


def stamps_of(folder: Path, pattern: str) -> dict:
    return {p.name: p.stat().st_mtime_ns for p in folder.glob(pattern)}


def stats_delta(before: dict) -> dict:
    return {k: M.STATS[k] - before[k] for k in M.STATS}


ALL = list(range(SETUP, FLUSH))
at_start = dict(os.environ)
plan_rows = None
try:
    patch(subprocess, "run", run_but_pip)
    patch(socket.socket, "connect", connect_here_only)
    patch(socket, "getaddrinfo", lookup_here_only)
    patch(Z, "WORK_FOLDERS", (str(CONTENT), str(MYDRIVE)))
    patch(E, "load_env", fake_env)
    patch(E, "require", made_up_token)
    patch(H, "hub_api", fake_hub_api)
    patch(M, "list_s2_shards", local_shards)
    patch(M, "load_s1_index", local_s1_index)
    patch(M, "HF", f"{BASE}/hub")
    patch(D, "load_pairs_to_ram", never_in_ram)
    patch(D, "CHUNK", 4)
    patch(D, "N_HELD_OUT", 2)
    patch(D, "N_STATS", 4)
    patch(A, "PATCH_PX", PX)
    patch(A, "OUT_PX", PX // 2)
    patch(A, "SET_SIZES", dict(SIZES))
    patch(importlib.util, "find_spec", find_or_not)
    patch(shutil, "disk_usage", space_or_not)
    A._LISTINGS.clear()

    # ---- first run, from nothing ------------------------------------------------------------
    M._FOOTERS.clear()
    one: dict = {"__name__": "__main__"}
    said = run_cells(ALL, one, "first run")

    assert (one["PILOT"], one["WORK"], one["DRIVE_MT"], one["STAGE0"]) == (PILOT, WORK, DRIVE_MT, STAGE0)
    assert (one["DATA_DIR"], one["DATA_REPO"]) == (FAKE_DRIVE_DIR, FAKE_REPO), "the settings of the fake .env"
    assert os.environ["DATA_HF_REPO"] == FAKE_REPO and os.environ["DATA_DRIVE_DIR"] == FAKE_DRIVE_DIR
    assert one["HF_TOKEN"] == FAKE_TOKEN and one["WORKERS"] == 28 and one["AERIAL_WORKERS"] == 28
    assert (one["N_PILOT"], one["N_HELD_OUT"], one["N_STATS"], one["SEED"]) == (D.N_PILOT, 2, 4, 0)
    assert D.N_PILOT == 26_000 and D.N_PAIRS == 20_000
    assert not CALLS["pip"], f"pip was called for {CALLS['pip']}, rasterio is here"
    assert CALLS["env"] == ["load_env", "require", "require"], CALLS["env"]     # the token, the data folder
    assert "GPU     none, none is needed here" in said
    assert "WARNING Drive" not in said and "WARNING the runtime disk" not in said
    assert (f"pilot {D.N_PILOT:,} cells drawn for about {D.N_PAIRS:,} pairs, 2 held out, statistics over "
            "4 pairs, seed 0, chunks of 4 cells") in said
    assert "cloud codes [1, 2, 3] over 10% of the crop, offset by processing baseline" in said
    assert f"aerial sets {SIZES}" in said and "installing" not in said
    assert set(one["runtime"]["disk"]) == {str(WORK), str(PILOT)} and one["runtime"]["gpu"] is None

    # the index, built now and kept in the pilot folder because stage 0 has none on Drive
    assert CALLS["list_s2_shards"] == 1 and CALLS["s1_metadata_read"] == 1
    assert "index s2  not on Drive, built now" in said and "index s1  not on Drive, built now" in said
    assert (PILOT / "index_s2.parquet").is_file() and (PILOT / "index_s1.parquet").is_file()
    counts = one["pair_counts"]
    assert (counts["s2_cells"], counts["cells_with_s1"], counts["pairs"]) == (16, 15, 14)
    assert (counts["no_s1_for_cell"], counts["gap_gt_7d"]) == (1, 1)
    assert CELL[FAR] not in set(one["pairs"]["grid_cell"]) and CELL[ALONE] not in set(one["pairs"]["grid_cell"])
    assert "s2_index" not in one and "s1_index" not in one, "the index tables were not freed"

    # the sample, 13 cells in 4 chunks, written to Drive
    sample = one["sample"]
    assert len(sample) == 14 and list(sample["chunk"]) == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 2
    assert sorted(sample["grid_cell"]) == sorted(CELL[i] for i in PAIRED)
    assert "sample of 14 cells written to pilot_sample.csv" in said
    assert "4 chunks of up to 4 cells" in said
    saved_sample = pd.read_csv(DRIVE_MT / "pilot_sample.csv")
    assert saved_sample["grid_cell"].tolist() == sample["grid_cell"].tolist()

    # the fetch. 10 pairs, 3 skips with the decided rules and one read_error, one archive per chunk
    # on Drive. The read_error is named with its chunk and how to read it again
    kept = [CELL[i] for i in KEPT]
    assert sorted(D.pair_cell(p) for p in one["paths"]) == sorted(kept)
    # every pair is a member of its archive on Drive, read in place, nothing on the runtime disk
    assert all(D.split_location(p)[0].parent == DRIVE_MT and D.split_location(p)[1] for p in one["paths"])
    assert files_of(WORK / "majortom") == set(), "pair files were left on the runtime disk"
    infos = one["infos"]
    assert list(infos.columns) == list(D.INFO_COLUMNS) and sorted(infos["grid_cell"]) == sorted(kept)
    assert D.locations_of(infos, DRIVE_MT) == one["paths"] and infos["chunk"].tolist() == [0] * 4 + [1] * 4 + [2, 3]
    assert "infos of 10 pairs in 4 tables" in said
    assert dict(zip(one["skips"]["grid_cell"], one["skips"]["reason"])) == FETCH_SKIPS
    assert (f"10 of 14 cells became a pair (the plan asks for about {D.N_PAIRS:,}), 4 chunks fetched now, "
            "0 read from Drive in place, 0 infos tables built") in said
    assert "the chunks fetched in this run, 14 cells in" in said and "the first read errors" in said
    # the requests per second and the waits, per chunk and for the run
    assert len(re.findall(r"^chunk \d  fetched .* \d+ requests, \d+\.\d per second, waited \d+ s in \d+ retries$",
                          said, re.M)) == 4, said
    assert re.search(r"\d+ requests, \d+\.\d per second, waited 0 s in 0 retries \(the Hub allows 5,000 per 5 minutes",
                     said), said
    assert re.search(rf"^ *3 +{CELL[BROKEN]} +read_error +read_s2 failed, ValueError, ", said, re.M), said
    assert ("WARNING 1 cells could not be read (chunks 3). Their archives on Drive are complete, so these "
            "cells are not read again on their own. To read them again delete chunk_<k>.tar and "
            f"chunk_<k>.json of these chunks in {DRIVE_MT} and run this cell again") in said
    for reason in R.SKIP_REASONS:
        assert re.search(rf"^  {reason} +\d+$", said, re.M), f"{reason} is not in the skip table"
    assert re.search(r"^  cloud_gt_10pct +1$", said, re.M) and re.search(r"^  s2_nodata_in_crop +1$", said, re.M)
    assert re.search(r"^  read_error +1$", said, re.M)
    info = json.loads((DRIVE_MT / "chunk_3.json").read_text(encoding="utf-8"))
    assert info["cells"] == 2 and [s["reason"] for s in info["skips"]] == ["read_error"]
    assert files_of(DRIVE_MT) >= {f"chunk_{k}{end}" for k in range(4) for end in (".tar", ".json", "_infos.csv")}
    for k, n_pairs in ((0, 4), (1, 4), (2, 1), (3, 1)):
        info = json.loads((DRIVE_MT / f"chunk_{k}.json").read_text(encoding="utf-8"))
        assert (info["pairs"], info["cloud_codes"], info["offset"]) == (n_pairs, [1, 2, 3], "baseline"), info
        assert (DRIVE_MT / f"chunk_{k}.tar").stat().st_size == info["bytes"]
    # the chunk tars were built on the runtime disk and copied to Drive, no local tar and no .part left
    assert not list((WORK / "majortom").glob("*.tar")), "a local tar was left"
    assert not list(DRIVE_MT.glob("*.part")) and not list(DRIVE_MT.glob("*.tmp"))
    for k in range(4):                            # the infos tables list the members of the tars
        table = D.read_infos(DRIVE_MT / f"chunk_{k}_infos.csv")
        with tarfile.open(DRIVE_MT / f"chunk_{k}.tar") as tar:
            assert table["file"].tolist() == [m.name for m in tar.getmembers()]
    for path in one["paths"]:                     # the rules were applied once, in make_pair
        pair = D.open_pair(path)
        assert pair.meta["cloud_rule"] == "applied" and pair.meta["cloud_codes"] == [1, 2, 3]
        assert pair.meta["offset_how"] in ("baseline", "per_image")
        assert pair.meta["offset_removed"] == (int(pair.meta["baseline"][1:]) >= 400)
    assert set(pd.read_csv(DRIVE_MT / "fetch_skips.csv")["reason"]) == set(FETCH_SKIPS.values())
    assert list(pd.read_csv(DRIVE_MT / "fetch_timing.csv")["source"]) == ["fetched"] * 4

    # held out, 2 of 10, the lists on Drive
    assert len(one["train_paths"]) == 8 and len(one["held_paths"]) == 2 == D.N_HELD_OUT
    assert len(one["train_paths"]) + len(one["held_paths"]) == len(one["paths"]) == 10
    train_cells = (DRIVE_MT / D.TRAIN_FILE).read_text(encoding="utf-8").split()
    held_cells = (DRIVE_MT / D.HELD_OUT_FILE).read_text(encoding="utf-8").split()
    assert sorted(train_cells + held_cells) == sorted(kept) and len(held_cells) == 2
    assert "8 training pairs, 2 held out, seed 0" in said

    # the aerial plan, the listings cached on Drive, the plan on Drive
    assert files_of(DRIVE_AERIAL) >= {"zips.csv", "patches.csv", "plan.csv"}
    plan_rows = one["plan"]
    assert plan_rows["set"].value_counts().to_dict() == {"distil_train": 16, "distil_heldout": 4,
                                                          "probe": 8, "val": 6, "test": 6}
    assert A.read_plan(DRIVE_AERIAL / "plan.csv").equals(plan_rows)
    assert len(one["patches"]) == len(TRUTH) and "12 aerial zips, 6 with images" in said
    assert "plan written to plan.csv" in said and f"{len(TRUTH)} patches in the index" in said

    # the aerial build, one file per set and department, one archive per set on Drive, the rate printed
    manifest = one["manifest"]
    assert re.search(r"range reads of this build with 28 threads and the token, \d+ requests, \d+\.\d per second, "
                     r"waited 0 s in 0 retries, \d+ MB in \d+ s", said), said
    assert sorted(manifest["sets"]) == sorted(A.SETS)
    assert files_of(DRIVE_AERIAL) >= {f"{name}.tar" for name in A.SETS} | {"manifest.json"}
    assert json.loads((DRIVE_AERIAL / "manifest.json").read_text(encoding="utf-8")) == manifest
    for name in A.SETS:
        entry = manifest["sets"][name]
        assert f"{name:<15} {entry['count']:>6,} patches, {len(entry['files'])} files" in said
        assert f"copied to Drive as {name}.tar" in said
        with tarfile.open(DRIVE_AERIAL / f"{name}.tar") as tar:
            assert sorted(tar.getnames()) == sorted(entry["files"]), name
        loaded = A.load_set(WORK / "aerial" / "sets", name)
        assert loaded["images"].shape == (entry["count"], 4, PX // 2, PX // 2)
        for i, patch_id in enumerate(loaded["ids"]):
            assert (loaded["images"][i] == A.halve_image(TRUTH[patch_id][0])).all()
    assert not (DRIVE_AERIAL / "val.tar.part").exists() and not (WORK / "aerial" / "val.tar").exists()
    labelled = plan_rows[plan_rows["set"].isin(A.LABELLED)]
    expected = {}
    for r in plan_rows.drop_duplicates("patch_id").itertuples():
        expected[r.rgbi_zip.rpartition("/")[2]] = expected.get(r.rgbi_zip.rpartition("/")[2], 0) + 1
    for r in labelled.drop_duplicates("patch_id").itertuples():
        expected[r.label_zip.rpartition("/")[2]] = expected.get(r.label_zip.rpartition("/")[2], 0) + 1
    for zip_path in LISTING["path"]:                              # plus one central directory per zip
        expected[zip_path.rpartition("/")[2]] = expected.get(zip_path.rpartition("/")[2], 0) + 1
    assert zip_hits() == expected, (zip_hits(), expected)         # one range request per member
    assert all(h.get("Authorization") == f"Bearer {FAKE_TOKEN}" for _, h in HITS)

    # the statistics, once, with the aerial training images
    stats = one["stats"]
    assert (stats["n"], stats["seed"], stats["pairs_given"]) == (4, 0, 8)
    # the same numbers as from the pair files unpacked by hand (the archives are read in place)
    by_hand = TMP / "by hand"
    for k in range(4):
        with tarfile.open(DRIVE_MT / f"chunk_{k}.tar") as tar:
            tar.extractall(by_hand, filter="data")
    plain_train = [by_hand / D.split_location(p)[1] for p in one["train_paths"]]
    with contextlib.redirect_stdout(io.StringIO()):
        again = D.channel_stats(plain_train, n=4, seed=0, aerial=A.load_set(WORK / "aerial" / "sets", "distil_train")["images"],
                                progress=False)
    assert again["cells_used"] == stats["cells_used"]
    for key in ("s1", "s2", "aerial"):
        assert np.allclose(again[key]["mean"], stats[key]["mean"], rtol=1e-12, atol=1e-12), key
        assert np.allclose(again[key]["sd"], stats[key]["sd"], rtol=1e-12, atol=1e-12), key
    shutil.rmtree(by_hand)
    assert stats["aerial"]["images"] == 16 and stats["aerial"]["channels"] == ["R", "G", "B", "NIR"]
    assert len(stats["s2"]["mean"]) == 12 and len(stats["s1"]["sd"]) == 2 and stats["corner_rule"] is True
    assert json.loads((DRIVE_MT / "channel_stats.json").read_text(encoding="utf-8")) == stats
    assert "computed over 4 pairs (seed 0) and 16 aerial images" in said

    # the summary, from the pair infos, everything on Drive, the token in none of the files
    meta = one["meta"]
    assert len(meta) == 10 and int(meta["held_out"].sum()) == 2 and list(meta.columns)[:-1] == list(D.META_COLUMNS)
    summary = one["summary"]
    rows = {(k, w): (p, s) for k, w, p, s in summary.itertuples(index=False)}
    assert rows[("pairs", "kept")] == (10, 10 / 14) and rows[("pairs", "skipped")] == (4, 4 / 14)
    assert rows[("skip", "cloud_gt_10pct")] == (1, 1 / 14) and rows[("offset", "not removed")] == (5, 0.5)
    assert rows[("skip", "read_error")] == (1, 1 / 14)
    assert rows[("year", "2023")] == (3, 0.3)
    assert set(one["TIMES"]) == set(STEPS) and one["sizes"]["pairs"] == 10
    assert "pairs_on_disk_gb" not in one["sizes"] and one["sizes"]["pairs_in_archives_gb"] > 0
    assert one["sizes"]["pair_archives_on_drive_gb"] > one["sizes"]["pairs_in_archives_gb"]
    assert one["sizes"]["aerial_patches"] == {n: manifest["sets"][n]["count"] for n in A.SETS}
    assert "Etalab Open Licence 2.0" in said and "files on Drive under" in said
    WANTED = {"runtime.json", "pair_counts.json", "index_s2.parquet", "index_s1.parquet",
              "pilot_summary.csv", "pilot_meta.csv", "sizes.json", "times.json"}
    assert files_of(PILOT) == WANTED | {"majortom", "aerial"}, files_of(PILOT)
    assert files_of(DRIVE_MT) == {*(f"chunk_{k}{e}" for k in range(4) for e in (".tar", ".json", "_infos.csv")),
                                  "pilot_sample.csv", "fetch_skips.csv", "fetch_timing.csv",
                                  D.TRAIN_FILE, D.HELD_OUT_FILE, "channel_stats.json"}, files_of(DRIVE_MT)
    assert files_of(DRIVE_AERIAL) == {*(f"{n}.tar" for n in A.SETS), "manifest.json", "zips.csv",
                                      "patches.csv", "plan.csv"}, files_of(DRIVE_AERIAL)
    for path in PILOT.rglob("*"):
        if path.suffix in (".csv", ".json", ".txt"):
            assert FAKE_TOKEN not in path.read_text(encoding="utf-8"), path
        if path.is_file():
            assert path.relative_to(PILOT).as_posix() in said.split("files on Drive under")[1], path
    assert json.loads((PILOT / "times.json").read_text(encoding="utf-8")).keys() == set(STEPS) - {"summary", "upload"}
    assert files_of(CONTENT) == {"drive", FAKE_DRIVE_DIR}
    assert files_of(WORK) == {"majortom", "aerial"} and files_of(WORK / "aerial") == {"sets"}
    # the upload without the write token, it says so and goes on, nothing reached the Hub
    assert "no HF_WRITE_TOKEN in the .env file, nothing is uploaded" in said
    assert f"to fill {FAKE_REPO}" in said and HUB_TOKENS == [] and hub_fake.store == {}
    assert "WRITE_TOKEN" not in one and not (PILOT / "README.md").exists()
    print("\nfirst run done, every cell ran and the files on Drive are what the synthetic data gives")

    # ---- the upload cell alone with the write token. Everything uploaded, then all skipped, then a
    # public repo and a Hub that fails, both printed and nothing stops
    os.environ["HF_WRITE_TOKEN"] = FAKE_WRITE_TOKEN
    said = run_cells([UPLOAD], one, "upload with the write token")
    PILOT_FILES = {*(f"majortom/chunk_{k}{e}" for k in range(4) for e in (".tar", ".json", "_infos.csv")),
                   "majortom/pilot_sample.csv", f"majortom/{D.TRAIN_FILE}", f"majortom/{D.HELD_OUT_FILE}",
                   "majortom/channel_stats.json", *(f"aerial/{n}.tar" for n in A.SETS), "aerial/manifest.json",
                   "aerial/plan.csv", "README.md"}
    assert set(hub_fake.uploads) == PILOT_FILES == set(hub_fake.store), set(hub_fake.store) ^ PILOT_FILES
    assert len([n for n in hub_fake.store if n.endswith("_infos.csv")]) == 4, "the infos tables are uploaded too"
    assert hub_fake.created == [FAKE_REPO] and set(HUB_TOKENS) == {FAKE_WRITE_TOKEN}
    for name in PILOT_FILES:
        assert hub_fake.store[name] == (PILOT / name).read_bytes(), name
    card = (PILOT / "README.md").read_text(encoding="utf-8")
    assert "Etalab Open Licence 2.0" in card and "CC-BY-SA-4.0" in card and "10 paired Major-TOM cells" in card
    assert f"uploaded and checked, {len(PILOT_FILES)} files, 0 skipped, sizes equal" in said
    # the files went up in threads, the rate per file and in all is printed
    assert re.search(rf"uploaded {len(PILOT_FILES)} of {len(PILOT_FILES)} files, [\d,]+ MB in \d+ s, [\d,]+ MB per second "
                     "with 4 threads", said), said
    assert said.count("MB per second") == len(PILOT_FILES) + 1 and "4 files at once" in said
    assert f"files in {FAKE_REPO}" in said and f"GB in {len(PILOT_FILES)} files" in said
    assert "UPLOAD FAILED" not in said and "WRITE_TOKEN" not in one
    hub_fake.uploads.clear()
    said = run_cells([UPLOAD], one, "upload again, all skipped")
    assert hub_fake.uploads == [] and f"{len(PILOT_FILES)} skipped" in said
    hub_fake.private = False
    said = run_cells([UPLOAD], one, "upload into a public repo")
    assert "UPLOAD FAILED, RuntimeError" in said and "PUBLIC, refusing" in said and hub_fake.uploads == []
    hub_fake.private = True
    hub_fake.fail = True
    said = run_cells([UPLOAD], one, "upload, the Hub is down")
    assert "UPLOAD FAILED, OSError, stand-in, the Hub is down, the token was <token>" in said
    hub_fake.fail = False
    print("\nupload done, with the token everything went up once, then skipped, refusals printed")

    # ---- second run, as on a new runtime. Drive is kept, the local disk is empty ------------
    shutil.rmtree(CONTENT / FAKE_DRIVE_DIR)
    M._FOOTERS.clear()
    before_calls = {k: (list(v) if isinstance(v, list) else v) for k, v in CALLS.items()}
    before_stats = dict(M.STATS)
    HITS.clear()
    stamps = {**stamps_of(DRIVE_MT, "*.tar"), **stamps_of(DRIVE_MT, "*.json"), **stamps_of(DRIVE_MT, "*_infos.csv"),
              **stamps_of(DRIVE_AERIAL, "*.tar"), **stamps_of(DRIVE_AERIAL, "*.csv"),
              "pilot_sample.csv": (DRIVE_MT / "pilot_sample.csv").stat().st_mtime_ns}
    two: dict = {"__name__": "__main__"}
    said = run_cells(ALL, two, "second run")

    assert CALLS == {**before_calls, "env": before_calls["env"] * 2}, f"work was repeated, {CALLS}"
    assert stats_delta(before_stats)["requests"] == 0, "a Major-TOM row or footer was read again"
    assert HITS == [], "a FLAIR-HUB zip was asked for again"
    changed = [name for name, stamp in stamps.items()
               if (DRIVE_MT / name if (DRIVE_MT / name).exists() else DRIVE_AERIAL / name).stat().st_mtime_ns != stamp]
    assert not changed, f"written again in the second run, {changed}"
    assert "index s2  found on Drive, own copy" in said and "index s1  found on Drive, own copy" in said
    assert "the sample on Drive is the same draw, 14 cells" in said
    assert "0 chunks fetched now, 4 read from Drive in place, 0 infos tables built" in said
    assert "the chunks fetched in this run" not in said and "infos built" not in said
    assert "the plan on Drive is the same draw" in said
    assert said.count(".tar read in place") == 4 and "unpacked from chunk_" not in said
    assert said.count("files unpacked from") == len(A.SETS)
    assert files_of(WORK / "majortom") == set(), "a pair archive was unpacked"
    assert said.count("the archive is on Drive already") == len(A.SETS)
    assert "found on Drive, not computed again" in said
    assert two["paths"] == one["paths"] and two["infos"].equals(one["infos"])
    assert two["stats"] == one["stats"] and two["manifest"] == one["manifest"]
    pd.testing.assert_frame_equal(two["summary"], one["summary"], obj="summary")
    assert two["pair_counts"] == one["pair_counts"] and two["plan"].equals(one["plan"])
    assert (DRIVE_MT / D.HELD_OUT_FILE).read_text(encoding="utf-8").split() == held_cells
    assert files_of(WORK / "aerial") == {"sets"}, "a side folder of the unpacking was left behind"
    assert hub_fake.uploads == [] and f"{len(PILOT_FILES)} skipped" in said, "the second run uploaded again"
    assert "README.md" in said.split("files on Drive under")[1]
    print("\nsecond run done, everything came from Drive, nothing was fetched or written again")

    # ---- third run, the same session, the index files in the stage 0 folder -----------------
    STAGE0.mkdir(parents=True)
    for name in ("index_s2.parquet", "index_s1.parquet"):
        os.replace(PILOT / name, STAGE0 / name)
    before_stats = dict(M.STATS)
    HITS.clear()
    said = run_cells(ALL, two, "third run")
    assert "index s2  found on Drive, from stage 0" in said and "index s1  found on Drive, from stage 0" in said
    assert not (PILOT / "index_s2.parquet").exists(), "the index was built again"
    assert "0 chunks fetched now, 4 read from Drive in place, 0 infos tables built" in said
    assert "unpacked from chunk_" not in said and "files unpacked from" not in said
    assert files_of(WORK / "majortom") == set()
    assert said.count("the archive is on Drive already") == len(A.SETS)
    assert stats_delta(before_stats)["requests"] == 0 and HITS == []
    assert CALLS["list_s2_shards"] == 1 and CALLS["s1_metadata_read"] == 1
    print("\nthird run done, the index of stage 0 was reused, everything was on disk")

    # ---- the guards. A sample and a plan on Drive that differ from the draw stop the notebook
    good = (DRIVE_MT / "pilot_sample.csv").read_bytes()
    lines = good.decode("utf-8").splitlines(True)
    (DRIVE_MT / "pilot_sample.csv").write_bytes("".join(lines[:-1]).encode("utf-8"))
    before_stats = dict(M.STATS)
    must_stop(SAMPLE, two, "sample guard", "holds another sample than this run draws")
    assert stats_delta(before_stats)["requests"] == 0
    (DRIVE_MT / "pilot_sample.csv").write_bytes(good)
    good = (DRIVE_AERIAL / "plan.csv").read_bytes()
    lines = good.decode("utf-8").splitlines(True)
    (DRIVE_AERIAL / "plan.csv").write_bytes("".join(lines[:-1]).encode("utf-8"))
    HITS.clear()
    must_stop(AERIAL_PLAN, two, "plan guard", "holds another plan than this run draws")
    assert not [p for p, _ in HITS if p.endswith(".zip")], "a zip was read after the plan guard"
    (DRIVE_AERIAL / "plan.csv").write_bytes(good)
    run_cells([SAMPLE, AERIAL_PLAN], two, "guards restored")

    # ---- a cut aerial archive on Drive. Only that set is read again, the archive comes back whole
    whole = (DRIVE_AERIAL / "val.tar").read_bytes()
    (DRIVE_AERIAL / "val.tar").write_bytes(whole[:len(whole) * 2 // 3])
    shutil.rmtree(WORK / "aerial" / "sets")
    (WORK / "aerial" / "sets").mkdir()
    HITS.clear()
    said = run_cells([AERIAL_BUILD], two, "cut aerial archive")
    assert "val.tar on Drive cannot be read, ReadError" in said
    assert said.count("files unpacked from") == len(A.SETS) - 1 and "copied to Drive as val.tar" in said
    assert said.count("the archive is on Drive already") == len(A.SETS) - 1
    val_rows = plan_rows[plan_rows["set"] == "val"]
    want = {}
    for r in val_rows.itertuples():
        for zip_path in (r.rgbi_zip, r.label_zip):
            want[zip_path.rpartition("/")[2]] = want.get(zip_path.rpartition("/")[2], 0) + 1
    assert zip_hits() == want, (zip_hits(), want)
    with tarfile.open(DRIVE_AERIAL / "val.tar") as tar:
        assert sorted(tar.getnames()) == sorted(manifest["sets"]["val"]["files"])
    assert two["manifest"] == manifest and files_of(WORK / "aerial") == {"sets"}
    print("\ncut aerial archive done, the set was built again and the archive is whole")

    # ---- a chunk archive on Drive without its infos table (the archives of the first real run). The
    # table is built once from the tar, nothing is fetched, the tar and the JSON stay as they are
    was = D.read_infos(DRIVE_MT / "chunk_2_infos.csv")
    (DRIVE_MT / "chunk_2_infos.csv").unlink()
    stamps = {**stamps_of(DRIVE_MT, "*.tar"), **stamps_of(DRIVE_MT, "*.json")}
    before_stats = dict(M.STATS)
    said = run_cells([FETCH], two, "chunk archive without its infos table")
    assert re.search(r"^chunk 2  infos built from chunk_2\.tar, 1 pairs, \d+\.\d s, written to chunk_2_infos\.csv$", said, re.M), said
    assert "0 chunks fetched now, 4 read from Drive in place, 1 infos tables built" in said
    assert said.count("infos built from") == 1 and stats_delta(before_stats)["requests"] == 0
    assert {**stamps_of(DRIVE_MT, "*.tar"), **stamps_of(DRIVE_MT, "*.json")} == stamps, "a tar or a JSON was written"
    assert D.read_infos(DRIVE_MT / "chunk_2_infos.csv").equals(was), "the rebuilt table differs"
    assert two["paths"] == one["paths"] and two["infos"].equals(one["infos"])
    print("\nmissing infos table done, built once from the tar, nothing fetched")

    # ---- a cut chunk archive on Drive. That chunk is fetched again, the others are not ------
    whole = (DRIVE_MT / "chunk_1.tar").read_bytes()
    (DRIVE_MT / "chunk_1.tar").write_bytes(whole[:len(whole) // 2])
    M._FOOTERS.clear()
    before_stats = dict(M.STATS)
    said = run_cells([FETCH], two, "cut chunk archive")
    assert "chunk 1  the archive chunk_1.tar is missing its JSON or is cut, the chunk is fetched again" in said
    assert "1 chunks fetched now, 3 read from Drive in place, 0 infos tables built" in said
    delta = stats_delta(before_stats)
    assert delta["row_requests"] == 8 and delta["decoded"] == 4 * (len(R.S2_BANDS) + 3), delta
    info = json.loads((DRIVE_MT / "chunk_1.json").read_text(encoding="utf-8"))
    assert (DRIVE_MT / "chunk_1.tar").stat().st_size == info["bytes"] and info["pairs"] == 4
    assert sorted(D.pair_cell(p) for p in two["paths"]) == sorted(kept)
    assert files_of(WORK / "majortom") == set() and not list(DRIVE_MT.glob("*.part"))
    assert D.read_infos(DRIVE_MT / "chunk_1_infos.csv")["grid_cell"].tolist() == one["infos"].query("chunk == 1")["grid_cell"].tolist()
    print("\ncut chunk archive done, the chunk was fetched again and the archive is whole")

    # ---- the setup cell with the library missing, Drive and the runtime disk nearly full --------
    WRONG.update({"libraries missing": True, "drive nearly full": True, "disk nearly full": True})
    said = run_cells([SETUP], two, "setup, library missing, Drive and the runtime disk nearly full")
    WRONG.update({"libraries missing": False, "drive nearly full": False, "disk nearly full": False})
    assert [call[-1] for call in CALLS["pip"]] == ["rasterio"] and "installing rasterio" in said
    assert all(call[1:5] == ["-m", "pip", "install", "-q"] for call in CALLS["pip"])
    assert "WARNING Drive has 3.0 GB free. A first run writes about 70 GB there" in said
    assert "WARNING the runtime disk has 3.0 GB free. A run keeps up to 20 GB there" in said
    WRONG.update({"disk nearly full": True})            # the runtime disk alone
    said = run_cells([SETUP], two, "setup, the runtime disk nearly full")
    WRONG.update({"disk nearly full": False})
    assert "WARNING the runtime disk has 3.0 GB free" in said and "WARNING Drive" not in said

    # ---- the setup cell on a runtime without Drive. It stops before the token is asked for --
    since = len(PATCHED)
    patch(Z, "WORK_FOLDERS", (str(CONTENT), str(TMP / "no-drive" / "MyDrive")))
    asked = len(CALLS["env"])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(CODE[SETUP], {"__name__": "__main__"})
    except AssertionError as err:
        assert "Google Drive is not mounted" in str(err), err
    else:
        raise AssertionError("the setup cell ran to its end without Drive")
    unpatch(since)
    assert CALLS["env"][asked:] == [], f"without Drive the cell went on to {CALLS['env'][asked:]}"
    print("\nwithout Drive the setup cell stops at once and asks for no token")

    assert not CALLS["network"], f"the dry run asked for the network, {CALLS['network']}"
    assert len(CALLS["pip"]) == 1
    assert set(HUB_TOKENS) == {FAKE_WRITE_TOKEN}, "another token reached the Hub"
finally:
    unpatch()
    server.shutdown()
    os.environ.pop("HF_WRITE_TOKEN", None)
    if HAD_WRITE_TOKEN is not None:
        os.environ["HF_WRITE_TOKEN"] = HAD_WRITE_TOKEN
    for k in HAD_DATA:                     # the settings of the fake .env leave with the test
        os.environ.pop(k, None)

assert not PATCHED and M.load_s1_index is REAL["load_s1_index"] and D.load_pairs_to_ram is REAL["load_pairs_to_ram"]
assert subprocess.run is REAL["run"] and socket.getaddrinfo is REAL["lookup"] and H.hub_api is REAL["hub_api"]
assert E.load_env is REAL["load_env"]
assert importlib.util.find_spec is REAL["find_spec"] and shutil.disk_usage is REAL["disk_usage"]
assert D.CHUNK == 2000 and D.N_HELD_OUT == 2000 and A.SET_SIZES["distil"] == 20000 and A.PATCH_PX == 512
assert D.N_PILOT == 26000 and D.N_PAIRS == 20000
for key in ("HF_HOME", "HF_TOKEN", "HF_HUB_DISABLE_SYMLINKS_WARNING", "DATA_HF_REPO", "DATA_DRIVE_DIR"):
    assert os.environ.get(key) == at_start.get(key), f"a cell changed {key}"
for k, v in HAD_DATA.items():
    if v is not None:
        os.environ[k] = v
assert Path("/content").exists() == HAD_CONTENT, "a cell wrote to the folder /content"

shutil.rmtree(TMP, ignore_errors=True)
print("test_notebook01: all checks passed")
