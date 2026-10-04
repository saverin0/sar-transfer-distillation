"""Dry run of notebooks/02_distil.ipynb on this PC, no network, no GPU, no real weights.

Every code cell but the mount cell, the sync cell and the flush cell is executed in order and unchanged.
What a cell would reach on Colab is replaced from OUTSIDE, by patching names of the modules

    folders /content and Drive    stage0.WORK_FOLDERS, a temp folder on D. Drive holds what part
                                  three's data notebook wrote, under the folder named by
                                  DATA_DRIVE_DIR (chunk archives and their JSON files, no infos
                                  tables, so the data cell builds them from the tars, the two
                                  lists of cells, the channel statistics, the aerial archives and
                                  manifest), made here from synthetic pairs
    .env file and token           env.load_env reads a fake .env of the temp folder that holds
                                  DATA_HF_REPO and DATA_DRIVE_DIR (the real parser, made-up
                                  values), env.require gives a made-up token,
                                  env.model_cache_on_drive a folder of the temp Drive
    the GPU check                 three names of torch.cuda, only while the setup cell runs
    the RAM of the runtime        stage0.runtime_report, the real one with the RAM of the
                                  High-RAM runtime (179 GB, 170 free) put in, or too little
    the free GPU memory           train.free_gpu_gb, a number of this file (70 GB, so the joint
                                  loop is taken on the CPU; 30 GB for the fallback)
    the teachers                  teachers.build_teacher, the TINY build of the real wrapper of
                                  each of the five names, on the CPU in float32
    the model cache               teachers.warm_weights, the real function on a fake Hub cache of
                                  the temp folder with made-up weight files at the pinned
                                  revisions (PE's file left out, so it is reported as missing)
    the training values           train.TrainConfig, a subclass with tiny defaults (12 steps,
                                  batch 8, crop 32 px, a tiny student). The notebook prints
                                  whatever the class holds, it retypes nothing
    the training run              train.train_students, the real function on the CPU. The
                                  notebook asks for cuda. In the measurement run the data share
                                  of the logs is set from outside, so both branches of the
                                  advice are seen
    the private Hugging Face repo hub.hub_api and hub.hub_download over a dict that holds the
                                  same files as the synthetic Drive, so the real fetch_pilot
                                  runs with the repo id of the fake .env. It can be told to
                                  fail, then the data cell must take the archives on Drive, or
                                  to refuse single files, then those archives come from Drive.
                                  One aerial archive arrives slowly in the first run, so the
                                  reading of the pairs is seen to start before the fetch ends

A connection to another machine stops the test. The made-up token must not show up in anything
the cells print or save.

Runs. The first one from nothing (the joint loop, the three students in one cell). The second one
as on a new runtime, where the local disk is empty and Drive holds the three finished students,
so no data is loaded, no teacher built, nothing measured or trained. Then the data cell with a Hub
that fails (Drive is taken), with nothing on Drive either (it stops), and with a Hub that answers but
two archives that do not arrive (they come from Drive, the arrays are the same). The third run resumes the
students from a cut joint run where C lags behind F and N (its last.pt deleted), so all go on from
the smaller step and give the weights of the first run bit for bit (the CPU is deterministic);
with a checkpoint on Drive the measurement is not taken again. Then the measurement with a
checkpoint there but its file gone (measured again), the fallback with too little free GPU memory
(one student after the other, the same weights), the measurement with a data path that is too
slow (the notebook stops with the advice), the measurement kept when last_prev.pt alone is on
Drive, the resume from last_prev.pt alone, the data cell with the statistics missing, a cut
archive, a cell without a pair file and a cut aerial archive, the data cell a second time in the
same session (the arrays of the first load must be gone before the new load), the data cell on a
runtime with too little RAM (it stops before anything is allocated), one student whose final.pt is
missing (it alone goes on), and the setup cell without Drive. The last cell is checked as text, it
is the shared flush cell (tests/_setup.FLUSH_SOURCE), and never run.

STAND-INS ONLY, so NOT tested against the real thing by this file - the mount cell, the sync cell
and the flush cell as cells (the packed source is unpacked and compared here, the cells are not
run), pip, the real .env file and the token, the model cache on Drive, the real weights and the
full size teachers and student, every CUDA call (bfloat16 autocast, memory counters, fused AdamW,
the free memory, non-deterministic kernels), the real sizes (20,000 pairs in 88 GB of RAM), Drive
as a file system, and the library versions of Colab.
"""
import _setup

import contextlib
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
import weakref
from dataclasses import dataclass, field
from pathlib import Path

os.environ["IPYTHONDIR"] = str(_setup.TMP_BASE / "ipython")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from sartransfer import env as E  # noqa: E402
from sartransfer import sync as SYNC  # noqa: E402
from sartransfer.distil import aerial as A  # noqa: E402
from sartransfer.distil import pairing as P  # noqa: E402
from sartransfer.distil import pilotdata as D  # noqa: E402
from sartransfer.distil import records as R  # noqa: E402
from sartransfer.distil import stage0 as Z  # noqa: E402
from sartransfer.distil import teachers as TEACH  # noqa: E402
from sartransfer.distil import train as T  # noqa: E402
from sartransfer.domains import hub as H  # noqa: E402

NOTEBOOK = _setup.REPO / "notebooks" / "02_distil.ipynb"
TMP = _setup.TMP_BASE / "notebook02"
for _ in range(40):                    # Windows may take a moment to let go of a folder
    shutil.rmtree(TMP, ignore_errors=True)
    if not TMP.exists():
        break
    time.sleep(0.25)
TMP.mkdir(parents=True)                # files of an older run would change what is tested
CONTENT = TMP / "content"                                  # stands for /content
MYDRIVE = CONTENT / "drive" / "MyDrive"                    # stands for /content/drive/MyDrive
FAKE_REPO = "made-up-owner/made-up-pilot-data"             # DATA_HF_REPO of the fake .env
FAKE_DRIVE_DIR = "made-up-data-folder"                     # DATA_DRIVE_DIR of the fake .env, under MyDrive
PILOT = MYDRIVE / FAKE_DRIVE_DIR / "pilot"                 # the notebook derives these from the setting
DRIVE_MT, DRIVE_AERIAL = PILOT / "majortom", PILOT / "aerial"
RUNS = MYDRIVE / "sar-transfer-distillation" / "runs_12"          # one folder per schedule, named by the steps (TinyConfig, 12)
KEYS = ("F", "N", "C")
OUT = {key: RUNS / f"student_{key}" for key in KEYS}
WORK = CONTENT / "sar-transfer-distillation" / "distil"
HUB_DIR = WORK / "hub"
STAGING = TMP / "staging"
FAKE_ENV = TMP / "fake.env"                                # the two data settings, no token
for folder in (DRIVE_MT, DRIVE_AERIAL, STAGING):
    folder.mkdir(parents=True)
FAKE_TOKEN = "MADE-UP-TOKEN-OF-THE-DRY-RUN"
HAD_CONTENT = Path("/content").exists()         # the folder that a cell must not write to
HAD_DATA = {k: os.environ.pop(k, None) for k in ("DATA_HF_REPO", "DATA_DRIVE_DIR")}   # the fake .env sets them
FAKE_ENV.write_text(f"DATA_HF_REPO={FAKE_REPO}\nDATA_DRIVE_DIR={FAKE_DRIVE_DIR}\n", encoding="utf-8")

# the teachers and students, as this file expects them
SG, SN, DINO, PE, RADAR = ("siglip2-so400m-256", "siglip2-so400m-naflex", "dinov3-l-sat", "pe-spatial-l",
                           "terrafm-b-radar")
TEACHER_NAMES = [SG, SN, DINO, PE, RADAR]
SUMMARY_ONLY = (SG, SN)
STUDENTS = {"F": [SG, DINO, PE, RADAR], "N": [SN, DINO, PE, RADAR], "C": [SG, DINO, PE]}
SPECS = [(f"student_{key}", names) for key, names in STUDENTS.items()]
FOLDERS = [name for name, _ in SPECS]
PHIS = {key: {f"phis_{n}.pt" for n in names if n not in SUMMARY_ONLY} for key, names in STUDENTS.items()}


def log_columns(key: str) -> list:
    parts = [f"{n}/{p}" for n in STUDENTS[key] for p in (("summary",) if n in SUMMARY_ONLY else ("summary", "spatial"))]
    return ["step", "lr", "loss", *parts, "grad_norm", "step_s", "data_s", "teacher_s", "student_s", "img_s", "gpu_gb"]


# ---- the file -----------------------------------------------------------------------------
NAMES = ["intro", "mount", "sync", "setup", "data", "teachers", "measure", "students", "summary", "flush"]
INTRO, MOUNT, SYNC_CELL, SETUP, DATA, TEACHERS, MEASURE_CELL, STUDENTS_CELL, SUMMARY, FLUSH = range(len(NAMES))
STEPS = ["setup", "data", "teachers", "measure", "students", "summary"]
text = NOTEBOOK.read_text(encoding="utf-8")
nb = json.loads(text)
assert (nb["nbformat"], nb["nbformat_minor"]) == (4, 5)
assert nb["metadata"]["kernelspec"]["name"] == "python3"
cells = nb["cells"]
assert [c["cell_type"] for c in cells] == ["markdown"] + ["code"] * 9
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

# the text a reader sees. No colon at all, and the facts that the task asks for
intro = SOURCE[INTRO]
assert intro.startswith("# ") and ":" not in intro, "a colon in the markdown cell"
for words in (
        # the students and the teachers
        "Student F copies SigLIP2 FixRes, the satellite DINOv3, PE-Spatial and TerraFM-B reading Sentinel-1",
        "Student N is F with SigLIP2 NaFlex in place of FixRes", "Student C has the three C-RADIOv4 slots without the radar",
        "the five teachers run once per step for all of them", "SigLIP2 gives its summary only",
        "", "summary_only argument of the loop", "never retypes them",
        "PE-Spatial-L at 224 px", "rows whose mode is s1 or both only",
        # the data, from the settings of the .env, nothing named or uploaded
        "DATA_HF_REPO names the private Hugging Face repo of the pilot data",
        "DATA_DRIVE_DIR the folder under MyDrive that holds it in pilot/",
        "never names another project's repo or folders itself", "nothing is uploaded",
        "with the read token first", "32 at once (hub.SMALL_FETCH_WORKERS)", "4 files at once (hub.FETCH_WORKERS)",
        "the pair archives in chunk order", "archives on Drive when the fetch fails",
        "an archive that did not arrive is read from Drive alone",
        "which source was used and the MB per second", "16 threads (pilotdata.RAM_WORKERS)", "nothing is unpacked",
        "Each archive is read as soon as it is on the disk", "(pilotdata.ChunkReader)", "the seconds this saved",
        "found by grid cell in the JSON files of the chunks", "infos tables", "checks that the RAM holds them",
        "side by side, one thread per file (warm_weights, nothing fetched)", "from memory",
        # the measurement, the loop, the summary, the flush
        "measures 20 training steps of the joint loop", "student F, student N, student C, rest",
        "the images per second, the GPU memory peak", "sequential one", "more than 15 percent",
        "train.train_students", "fused AdamW", "(DINOv3, PE-Spatial, TerraFM-B)", "every folder that needs it",
        "every 1,000 steps", "last_prev.pt", "held-out losses per student and teacher", "releases the runtime",
        # the runtime, the estimates, the folders, the token
        "The A100 with High-RAM.", "about 88 GB of RAM, part three measured it", "82.6 GB",
        "the data cell stops with a message when they do not fit", "about 53 GB", "29 to 31 GB by arithmetic",
        "train.JOINT_MIN_GB", f"at least {T.JOINT_MIN_GB} GB are free", "80 GB and on the 40 GB A100",
        "train.train_student with the same results", "VS Code",
        "Estimates until the measurement cell runs", "from two earlier measurement runs of the teachers",
        "0.90 to 0.95 s per step", "0.54 to 0.58 s", "about 1.45 to 1.52 s per step and 2.5 to 2.6 h for the 6,250 steps",
        "plus the held-out passes and the checkpoints", "took 279 s",
        "412 s", "213 MB/s", "about 4.5 to 5 min by arithmetic", "prints the measured overlap",
        "23 to 65 MB/s", "70 to 95 s for a 4.54 GB SigLIP2 file", "the cell prints both", "every 100 steps", "by arithmetic",
        "MyDrive/sar-transfer-distillation/runs_6250/", "student_F, student_N, student_C", "final.pt",
        "the data is not loaded, the teachers are not built and nothing is measured", "smallest step",
        "computes the steps in between again", "same in distribution on the GPU", "measurement is not repeated",
        "naming the student", "END_RUNTIME = False", "run the mount cell again", "Run the cells from top to bottom",
        "never printed", "a teacher whose files are not there yet is fetched once at its pinned revision"):
    assert words in intro, f"the first cell does not say {words!r}"
# no name of another repository or of its folders in any cell (the sync cell holds base64 only)
FORBIDDEN = _setup.private_names() + ("HF_OWNER",)
for k, source in enumerate(SOURCE):
    if k != SYNC_CELL:
        hits = [w for w in FORBIDDEN if w in source]
        assert not hits, f"cell {k} ({NAMES[k]}) names another repository or its folder, {hits}"
for k in range(MOUNT, len(cells)):
    first = SOURCE[k].splitlines()[0]
    assert first.startswith("# ") and ":" not in first, f"cell {k} starts with {first!r}"

assert "google.colab" in SOURCE[MOUNT] and 'drive.mount("/content/drive")' in SOURCE[MOUNT]
# the last cell is the shared flush cell, word for word - it flushes Drive, prints, then releases the
# machine. Not run here
assert cells[FLUSH]["id"] == "flush" and SOURCE[FLUSH] == _setup.FLUSH_SOURCE, "the last cell differs from the shared flush cell"
assert "google.colab" in SOURCE[FLUSH] and "drive.flush_and_unmount()" in SOURCE[FLUSH]
assert "mount cell" in SOURCE[FLUSH] and "HF_TOKEN" not in SOURCE[FLUSH] and "print(" in SOURCE[FLUSH]
assert "END_RUNTIME = True" in SOURCE[FLUSH] and "runtime.unassign()" in SOURCE[FLUSH] and "if END_RUNTIME" in SOURCE[FLUSH]
assert SOURCE[FLUSH].index("drive.flush_and_unmount()") < SOURCE[FLUSH].index("print(") \
    < SOURCE[FLUSH].index("runtime.unassign()"), "flush, then the message, then the release"
assert not any(":" in part for part in [ln for ln in SOURCE[FLUSH].splitlines() if ln.lstrip().startswith("#")]
               + re.findall(r'"([^"]*)"', SOURCE[FLUSH])), "a colon in the text of the flush cell"
assert not re.search(r"^\s*[!%]", SOURCE[FLUSH], re.M)
lines = SOURCE[SYNC_CELL].splitlines()
assert lines[0] == SYNC.BEGIN and lines[-1] == SYNC.END
assert SYNC.cell_index(nb) == SYNC_CELL
SYNC_STATE = _setup.check_sync_cell(nb, SYNC)            # packed and current, or cleared in a published copy

CODE = {}
for k in range(SETUP, FLUSH):
    source = SOURCE[k]
    assert not re.search(r"^\s*[!%]", source, re.M), f"cell {k} holds a notebook magic"
    assert not re.search(r"\bC:[\\/]", source), f"cell {k} names a path on C"
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", source), f"cell {k} holds a token"
    assert "getpass" not in source, f"cell {k} asks for a token on its own"
    assert "git clone" not in source and "torch.hub" not in source and "pip" not in source
    assert "print(HF_TOKEN" not in source and "{HF_TOKEN" not in source
    assert re.search(r'print\(took\(start, "[a-zA-Z_]+"\)\)$', source.rstrip()), f"cell {k} does not print its time"
    assert "start = time.perf_counter()" in source
    CODE[k] = compile(source, f"02_distil cell {k} ({NAMES[k]})", "exec")
for words in ("env.load_env()", 'env.require("HF_TOKEN"', 'DATA_DIR = env.require("DATA_DRIVE_DIR"',
              "env.model_cache_on_drive()", "assert torch.cuda.is_available()", "stage0.runtime_report([WORK, RUNS])",
              "stage0.WORK_FOLDERS", "CFG = train.TrainConfig()", "N_MEASURE = 20", "DATA_SHARE_LIMIT = 0.15",
              "RAM_SPARE_GB = 10", "from sartransfer.domains import hub", "hub.data_repo_id()",
              'PILOT = MYDRIVE / DATA_DIR / "pilot"', 'RUNS = MYDRIVE / "sar-transfer-distillation" / f"runs_{CFG.steps}"',
              'WORK = CONTENT / "sar-transfer-distillation" / "distil"', 'HUB_DIR = WORK / "hub"',
              'SUMMARY_ONLY = ("siglip2-so400m-256", "siglip2-so400m-naflex")',
              "SPECS = [(OUT[key].name, names) for key, names in STUDENTS.items()]",
              "from sartransfer.distil.teachers import build_teacher, warm_weights"):
    assert words in SOURCE[SETUP], f"the setup cell has no {words}"
assert "assert folder.is_dir()" not in SOURCE[SETUP], "the setup cell stops without the Drive folders"
# the pre-registered values are never retyped, they come from TrainConfig. The memory rule is not retyped
# either, the cells name train.JOINT_MIN_GB
for words in ("6250", "6_250", "9375", "9_375", "5e-4", "0.05", "500", "1000", "1_000", "2048", "2_048", "128", "96",
              "36"):
    assert not re.search(rf"(?<![\w.]){re.escape(words)}(?![\w.])", "".join(SOURCE[SETUP:])), \
        f"a pre-registered value is retyped, {words}"
for k, call in ((DATA, 'pilotdata.ChunkReader({"train": lists["train"], "held_out": lists["held_out"]}, corner_rule=True)'),
                (DATA, "reader.finish()"), (DATA, "reader.stop()"), (DATA, "except BaseException"),
                (DATA, "pilotdata.pair_locations(SOURCE_MT, numbers)"), (DATA, "reader.arrived(archive)"),
                (DATA, "MB per second"), (DATA, "pilotdata.RAM_WORKERS"), (DATA, "hub.FETCH_WORKERS"),
                (DATA, 'aerial.load_set_from_tar(archive, name, manifest["sets"][name]["files"])'),
                (DATA, "hub.fetch_pilot(HUB_DIR, token=HF_TOKEN, want=SMALL, workers=hub.SMALL_FETCH_WORKERS)"),
                (DATA, "hub.fetch_pilot(HUB_DIR, token=HF_TOKEN, want=ARCHIVES, on_file=arrived)"),
                (DATA, "saved"), (TEACHERS, "warm_weights(TEACHER_NAMES)"), (TEACHERS, "teachers.WARM_WORKERS"),
                (DATA, "SOURCE_MT, SOURCE_AERIAL = DRIVE_MT, DRIVE_AERIAL"), (DATA, "is not "),
                (DATA, "pilotdata.PAIR_BYTES"), (DATA, "need_gb + RAM_SPARE_GB <= free_gb"),
                (DATA, "data.check(CFG.crop)"), (DATA, "globals().pop(_name, None)"),
                (TEACHERS, "build_teacher(name, device=DEVICE, token=HF_TOKEN)"),
                (TEACHERS, "train.as_targets(teachers, summary_only=SUMMARY_ONLY)"),
                (TEACHERS, "torch.cuda.empty_cache()"),
                (MEASURE_CELL, "train.train_students(SPECS, CFG, data, targets, scratch, resume=False, device=DEVICE,"),
                (MEASURE_CELL, "stop_after=N_MEASURE, summary_only=SUMMARY_ONLY)"), (MEASURE_CELL, "hours_joint"),
                (MEASURE_CELL, "hours_seq"), (MEASURE_CELL, '"teacher_s"'), (MEASURE_CELL, '"student_s"'),
                (MEASURE_CELL, "split of a step"), (MEASURE_CELL, "images per second"),
                (STUDENTS_CELL, "train.train_students(todo, CFG, data, targets, RUNS, resume=True, device=DEVICE,"),
                (STUDENTS_CELL, "summary_only=SUMMARY_ONLY)"), (STUDENTS_CELL, "train.JOINT_MIN_GB"),
                (STUDENTS_CELL, "del trained"), (STUDENTS_CELL, "images per second"),
                (SUMMARY, '"teacher_s", "student_s"'), (SUMMARY, "held_out_last=")):
    assert call in SOURCE[k], f"cell {k} ({NAMES[k]}) has no {call}"
assert "extractall" not in SOURCE[DATA] and "unpack(" not in SOURCE[DATA], "the data cell unpacks an archive"
assert "load_pairs_to_ram(" not in SOURCE[DATA], "the data cell reads the pairs after the fetch, not while it runs"
assert SOURCE[TEACHERS].index("warm_weights(TEACHER_NAMES)") < SOURCE[TEACHERS].index("build_teacher(name"), \
    "the weights are read ahead after the builds"
assert "train_student(" not in "".join(SOURCE[SETUP:]), "a cell calls the single loop itself"
assert "copyfile" not in SOURCE[STUDENTS_CELL], "the PHI-S copy is train_students' job"
print(f"the file is a notebook of {len(cells)} cells without outputs, the sync cell is up to date, "
      "the last cell is the shared flush cell")

# ---- what part three's data notebook left on Drive, made here from synthetic pairs ----------------
LEFT, TOP = 500002.5, 4000004.25          # a corner that is not a multiple of 10 m, as in the data
N_TRAIN, N_HELD, PX = 12, 4, 32
CELLS = [f"{100 + i}U_{200 + i}R" for i in range(N_TRAIN + N_HELD)]


def fake_pair(i: int) -> R.Pair:
    rng = np.random.default_rng(50 + i)
    s1 = rng.normal(-15.0, 5.0, (2, R.CROP_PX, R.CROP_PX)).astype(np.float16)
    s2 = {band: rng.integers(100, 3000, (R.CROP_PX * 10 // R.S2_RES_M[band],) * 2).astype(np.uint16)
          for band in R.S2_BANDS}
    day = f"2022{(i % 12) + 1:02d}15"
    meta = {"cloud_rule": "applied", "cloud_codes": [1, 2, 3], "cloud_fraction": 0.0,
            "s2_product_id": f"S2A_MSIL2A_{day}T101010_N0400_R022_T33TUM_{day}T121212",
            "baseline": "N0400", "offset_removed": True, "offset_how": "baseline"}
    return R.Pair(CELLS[i], s1, s2, np.zeros((R.CROP_PX, R.CROP_PX), np.uint8),
                  pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:]}T10:10:10", tz="UTC"),
                  pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:]}T05:15:00", tz="UTC"), 0.2, "EPSG:32633",
                  (LEFT, TOP - 3840.0, LEFT + 3840.0, TOP), meta)


PAIR_FILES = {}                           # cell -> the staged pair file
for k, cells_of_chunk in enumerate((CELLS[:8], CELLS[8:])):
    folder = STAGING / f"chunk_{k}"
    folder.mkdir()
    files = []
    for cell in cells_of_chunk:
        P.save_pair(fake_pair(CELLS.index(cell)), folder / f"{cell}.npz")
        files.append(f"{cell}.npz")
        PAIR_FILES[cell] = folder / f"{cell}.npz"
    tar_path = DRIVE_MT / f"chunk_{k}.tar"
    with tarfile.open(tar_path, "w") as tar:
        for name in files:
            tar.add(folder / name, arcname=f"chunk_{k}/{name}")
    (DRIVE_MT / f"chunk_{k}.json").write_text(json.dumps({
        "chunk": k, "files": files, "bytes": tar_path.stat().st_size, "cells": len(files),
        "pairs": len(files), "cloud_codes": [1, 2, 3], "offset": "baseline", "skips": [],
        "fetch": {"wall_s": 1.0}}, indent=1), encoding="utf-8")
TRAIN_CELLS, HELD_CELLS = sorted(CELLS[:N_TRAIN]), sorted(CELLS[N_TRAIN:])
(DRIVE_MT / D.TRAIN_FILE).write_text("".join(f"{c}\n" for c in TRAIN_CELLS), encoding="utf-8")
(DRIVE_MT / D.HELD_OUT_FILE).write_text("".join(f"{c}\n" for c in HELD_CELLS), encoding="utf-8")


def aerial_file(path: Path, n: int, seed: int, department: str, year: str) -> None:
    rng = np.random.default_rng(seed)
    np.savez(path, images=rng.integers(0, 256, (n, 4, PX, PX), dtype=np.uint8),
             ids=np.array([f"{department}-{year}_AA-S1-1_{i}-1" for i in range(n)]),
             department=department, year=year)


AERIAL_SETS = {"distil_train": [("D004", "2021", 6), ("D017", "2018", 6)],
               "distil_heldout": [("D004", "2021", 2)]}
manifest = {"repo": A.HF_REPO, "revision": A.HF_REVISION, "bands": list(A.BANDS), "out_px": PX,
            "classes": list(A.CLASSES), "sets": {}}
for name, parts in AERIAL_SETS.items():
    files = []
    for j, (dept, year, n) in enumerate(parts):
        file = f"{name}__{dept}-{year}.npz"
        aerial_file(STAGING / file, n, 100 + j + len(name), dept, year)
        files.append(file)
    with tarfile.open(DRIVE_AERIAL / f"{name}.tar", "w") as tar:
        for file in files:
            tar.add(STAGING / file, arcname=file)
    manifest["sets"][name] = {"count": sum(p[2] for p in parts), "files": files,
                              "departments": sorted({p[0] for p in parts}), "class_shares": None}
(DRIVE_AERIAL / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
(STAGING / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
train_images = A.load_set(STAGING, "distil_train")["images"]
with contextlib.redirect_stdout(io.StringIO()):
    STATS = D.channel_stats([PAIR_FILES[c] for c in TRAIN_CELLS], n=4, aerial=train_images,
                            out_path=DRIVE_MT / "channel_stats.json", progress=False)
assert STATS["aerial"]["images"] == 12 and len(STATS["s2"]["mean"]) == 12
N_PATCHES = sum(n for parts in AERIAL_SETS.values() for _, _, n in parts)
NEED_GB = (len(CELLS) * D.PAIR_BYTES + N_PATCHES * 4 * PX * PX) / 1e9      # what the RAM check must compute
print(f"synthetic Drive of part three, {N_TRAIN} + {N_HELD} pairs in 2 archives, "
      f"{len(AERIAL_SETS)} aerial archives, in {PILOT}")


# ---- the model cache, a fake Hub cache with made-up weight files at the pinned revisions -------------------
FAKE_CACHE = TMP / "hf_cache" / "hub"
WARM_FILES = {}                           # (repo, file) -> path, PE's file is left out
for _name in TEACHER_NAMES:
    for _repo, _revision, _file in TEACH.weight_files(_name):
        if _name == PE:
            continue
        _path = FAKE_CACHE / f"models--{_repo.replace('/', '--')}" / "snapshots" / _revision / _file
        _path.parent.mkdir(parents=True, exist_ok=True)
        _path.write_bytes(bytes(range(256)) * (8 + len(WARM_FILES)))
        WARM_FILES[(_repo, _file)] = _path
assert len(WARM_FILES) == 9               # SigLIP2 3 + 3, DINOv3 2, TerraFM 1


# ---- the private Hugging Face repo, a stand-in over the same files ----------------------------
class FakeHub:
    """list_repo_tree over a dict of bytes. `fail` = the repo cannot be reached."""

    class Entry:
        def __init__(self, path, size):
            self.path, self.size = path, size

    def __init__(self, store):
        self.store, self.fail = store, False

    def list_repo_tree(self, rid, repo_type, recursive):
        assert repo_type == "dataset" and recursive is True and rid == FAKE_REPO, rid
        if self.fail:
            raise FileNotFoundError(f"stand-in, {rid} is not on the Hub, the token was {FAKE_TOKEN}")
        return [self.Entry(p, len(b)) for p, b in self.store.items()]


HUB_STORE = {f"{folder.name}/{p.name}": p.read_bytes() for folder in (DRIVE_MT, DRIVE_AERIAL)
             for p in folder.iterdir() if p.is_file()}
hub_fake = FakeHub(HUB_STORE)
HUB_CALLS = {"tokens": [], "downloads": [], "done_at": {}}
HUB_SLOW, HUB_REFUSE = {}, set()          # seconds a file takes, files that never arrive
READ_AT = []                              # when each pair was read, filled in the first run


def fake_hub_api(token):
    HUB_CALLS["tokens"].append(token)
    assert token == FAKE_TOKEN, "the fetch used another token than the read token"
    return hub_fake


def fake_hub_download(rid, name, dest, token):
    assert token == FAKE_TOKEN and rid == FAKE_REPO, rid
    HUB_CALLS["downloads"].append(name)
    if name in HUB_SLOW:
        # the slow archive waits until the reader has read a pair (or HUB_SLOW seconds at most), so the
        # overlap of fetch and read shows whatever the load of the machine (a fixed sleep failed when
        # three tests ran at once on the laptop)
        deadline = time.perf_counter() + HUB_SLOW[name]
        while not READ_AT and time.perf_counter() < deadline:
            time.sleep(0.01)
    if name in HUB_REFUSE:
        raise OSError(f"stand-in, {name} did not arrive")
    target = Path(dest) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(HUB_STORE[name])
    HUB_CALLS["done_at"][name] = time.perf_counter()
    return target


# ---- the stand-ins, patched in from outside --------------------------------------------------
@dataclass
class TinyConfig(T.TrainConfig):
    """The fixed values replaced by tiny ones. The notebook builds train.TrainConfig()."""

    steps: int = 12
    batch: int = 8
    n_aerial: int = 2
    crop: int = PX
    warmup: int = 3
    checkpoint_every: int = 5
    held_out_every: int = 5
    phis_samples: int = 12
    bf16: bool = False
    student: dict = field(default_factory=lambda: dict(img=PX, patch=16, dim=32, depth=2, heads=4))


assert RUNS.name == f"runs_{TinyConfig().steps}", "the run folder of the dry run is named by the tiny steps"


CALLS = {"build_teacher": [], "warm": [], "train": [], "network": [], "env": []}
MEASURE = {"data_share": 0.05}          # the data share written into the logs of a measurement
FREE = {"gb": 70.0}                     # what train.free_gpu_gb reports, 70 = the joint loop is taken
RAM = {"total": 179.0, "free": 170.0}   # what the runtime reports, the High-RAM A100 printed 179 GB
WRONG = {"drive not mounted": False}
GPU = {"name": "stand-in GPU of the dry run"}
PATCHED = []


def patch(owner, name: str, value) -> None:
    PATCHED.append((owner, name, getattr(owner, name)))
    setattr(owner, name, value)


def unpatch(since: int = 0) -> None:
    while len(PATCHED) > since:
        owner, name, old = PATCHED.pop()
        setattr(owner, name, old)


REAL = {"run": subprocess.run, "connect": socket.socket.connect, "lookup": socket.getaddrinfo,
        "load_env": E.load_env, "build_teacher": TEACH.build_teacher, "warm_weights": TEACH.warm_weights,
        "ChunkReader": D.ChunkReader, "open_pair": D.open_pair, "train_students": T.train_students,
        "free_gpu_gb": T.free_gpu_gb, "TrainConfig": T.TrainConfig, "runtime_report": Z.runtime_report,
        "load_pairs_to_ram": D.load_pairs_to_ram, "hub_api": H.hub_api, "hub_download": H.hub_download}
HERE = ("127.0.0.1", "::1", "localhost")


def run_but_pip(args, *more, **kwargs):
    if isinstance(args, (list, tuple)) and "pip" in [str(a) for a in args]:
        raise SystemExit(f"the notebook called pip, {args}")
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


def cache_in_temp(*args, **kwargs) -> Path | None:
    CALLS["env"].append("model_cache_on_drive")
    if WRONG["drive not mounted"]:       # what env.model_cache_on_drive gives without Drive
        return None
    root = MYDRIVE / "model_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root                         # HF_HOME is left as it is


def tiny_teacher(name: str, **asked):
    CALLS["build_teacher"].append(name)
    assert asked.pop("device") == "cuda" and asked.pop("token") == FAKE_TOKEN, asked
    assert not asked, f"build_teacher got {asked}"
    torch.manual_seed(list(TEACH.TEACHERS).index(name))     # the same random weights every time
    return REAL["build_teacher"](name, device="cpu", dtype=torch.float32, tiny=True)


def warm_on_fake_cache(names, **asked):
    """The real warm_weights on the fake cache. The notebook asks for the default threads and cache."""
    assert not asked, f"warm_weights got {asked}"
    CALLS["warm"].append(list(names))
    return REAL["warm_weights"](names, cache_dir=FAKE_CACHE)


def not_again(what: str):
    """A ChunkReader that must not be made."""
    def refuse(*args, **kwargs):
        raise AssertionError(what)
    return refuse


def free_gb_stand_in(device=None) -> float:
    return FREE["gb"]


def ram_of_the_runtime(folders=None) -> dict:
    """The real report of this PC with the RAM of the Colab runtime put in."""
    report = REAL["runtime_report"](folders)
    report.update(ram_gb=RAM["total"], ram_free_gb=RAM["free"])
    return report


def train_students_on_cpu(specs, config, data, teachers, out_root, resume=True, device=None,
                          stop_after=None, **kw):
    """The real train_students on the CPU. The notebook asks for cuda and passes the summary-only names,
    so the memory rule (the patched reader) decides between the joint loop and the fallback."""
    assert device == "cuda", f"the notebook asked for the device {device!r}"
    assert isinstance(config, TinyConfig) and kw == {"summary_only": SUMMARY_ONLY}, (type(config), kw)
    CALLS["train"].append((Path(out_root).name, [s[0] for s in specs], resume, stop_after))
    result = REAL["train_students"](specs, config, data, teachers, out_root, resume=resume, device="cpu",
                                    stop_after=stop_after, **kw)
    if stop_after is not None and MEASURE["data_share"] is not None:    # the measurement run
        for one in result.values():
            log = pd.read_csv(one["log"])
            log["data_s"] = MEASURE["data_share"] * log["step_s"]
            log.to_csv(one["log"], index=False)
    return result


@contextlib.contextmanager
def gpu_stand_in():
    """The runtime says it has a GPU. Only while the setup cell runs."""
    since = len(PATCHED)
    patch(torch.cuda, "is_available", lambda: True)
    patch(torch.cuda, "mem_get_info", lambda device=None: (30 * 10 ** 9, 40 * 10 ** 9))
    patch(torch.cuda, "get_device_name", lambda device=None: GPU["name"])
    try:
        yield
    finally:
        unpatch(since)


def run_cells(which, names: dict, title: str) -> str:
    """Execute these code cells in `names`, unchanged. Returns what they printed."""
    printed = []
    for k in which:
        out = io.StringIO()
        begin = time.perf_counter()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                with gpu_stand_in() if k == SETUP else contextlib.nullcontext():
                    exec(CODE[k], names)
        except BaseException:
            print(out.getvalue())
            traceback.print_exc()
            raise SystemExit(f"{title}, cell {k} ({NAMES[k]}) raised") from None
        text = out.getvalue()
        print(f"\n==== {title}, cell {k} ({NAMES[k]}), {time.perf_counter() - begin:.1f} s")
        print("\n".join("    | " + line for line in text.splitlines() if "\r" not in line))
        assert FAKE_TOKEN not in text, f"cell {k} printed the token"
        assert "cell time" in text.splitlines()[-1], f"cell {k} did not print its time"
        printed.append(text)
    return "\n".join(printed)


def must_stop(k: int, names: dict, title: str, kind, says: str) -> str:
    """Cell `k` must raise `kind` with `says` in the message. Returns what it printed."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            with gpu_stand_in() if k == SETUP else contextlib.nullcontext():
                exec(CODE[k], names)
    except kind as err:
        assert says in str(err), f"{title}, cell {k} stopped with {err}"
        assert FAKE_TOKEN not in out.getvalue() and FAKE_TOKEN not in str(err)
        print(f"\n==== {title}, cell {k} ({NAMES[k]}) stopped as it must, {says!r}")
        return out.getvalue()
    raise SystemExit(f"{title}, cell {k} ({NAMES[k]}) ran to its end")


def files_of(folder: Path) -> set:
    return {p.name for p in folder.iterdir()}


def stamps_of(folder: Path) -> dict:
    return {p.name: p.stat().st_mtime_ns for p in folder.iterdir() if p.is_file()}


def weights_of(path: Path) -> dict:
    return T.load_checkpoint(path)["student"]


def same_weights(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def log_rows(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def joint_line(gb: float) -> str:
    return (f"students: {', '.join(FOLDERS)} in one loop, the teachers run once per step, {gb:.1f} GB of GPU "
            f"memory free, at least {T.JOINT_MIN_GB} needed")


def unlink(*files: str, keys=KEYS) -> None:
    for key in keys:
        for file in files:
            (OUT[key] / file).unlink(missing_ok=True)


def cut_joint_run(names: dict, stop_after: int) -> None:
    """The real joint loop on Drive, cut after `stop_after` steps, all students at the same step."""
    unlink("final.pt", "last.pt", "last_prev.pt")
    with contextlib.redirect_stdout(io.StringIO()):
        REAL["train_students"](names["SPECS"], names["CFG"], names["data"], names["targets"], RUNS, resume=False,
                               device="cpu", stop_after=stop_after, joint=True, summary_only=SUMMARY_ONLY)
    for out in OUT.values():
        assert T.load_checkpoint(out / "last.pt")["step"] == stop_after - stop_after % 5
        assert not (out / "final.pt").exists()


def same_as_first(title: str) -> None:
    for key in KEYS:
        assert same_weights(weights_of(OUT[key] / "final.pt"), WEIGHTS[key]), f"{title}, student {key} differs"


ALL = list(range(SETUP, FLUSH))
WEIGHTS: dict = {}
at_start = dict(os.environ)
try:
    patch(subprocess, "run", run_but_pip)
    patch(socket.socket, "connect", connect_here_only)
    patch(socket, "getaddrinfo", lookup_here_only)
    patch(Z, "WORK_FOLDERS", (str(CONTENT), str(MYDRIVE)))
    patch(Z, "runtime_report", ram_of_the_runtime)
    patch(E, "load_env", fake_env)
    patch(E, "require", made_up_token)
    patch(E, "model_cache_on_drive", cache_in_temp)
    patch(TEACH, "build_teacher", tiny_teacher)
    patch(TEACH, "warm_weights", warm_on_fake_cache)
    patch(T, "TrainConfig", TinyConfig)
    patch(T, "train_students", train_students_on_cpu)
    patch(T, "free_gpu_gb", free_gb_stand_in)
    patch(H, "hub_api", fake_hub_api)
    patch(H, "hub_download", fake_hub_download)

    # ---- first run, from nothing ------------------------------------------------------------
    one: dict = {"__name__": "__main__"}
    READ_AT.clear()                           # when each pair was read, to see the overlap with the fetch
    since = len(PATCHED)
    patch(D, "open_pair", lambda where: (READ_AT.append(time.perf_counter()), REAL["open_pair"](where))[1])
    HUB_SLOW["aerial/distil_train.tar"] = 30.0           # at most, it waits for the first read pair
    said = run_cells(ALL, one, "first run")
    HUB_SLOW.clear()
    unpatch(since)

    # setup. The settings of the fake .env, the folders derived from them, the stand-in GPU and RAM
    assert (one["PILOT"], one["RUNS"], one["WORK"], one["HUB_DIR"]) == (PILOT, RUNS, WORK, HUB_DIR)
    assert one["DATA_DIR"] == FAKE_DRIVE_DIR and os.environ["DATA_HF_REPO"] == FAKE_REPO
    assert one["DEVICE"] == "cuda" and one["HF_TOKEN"] == FAKE_TOKEN
    assert (one["N_MEASURE"], one["DATA_SHARE_LIMIT"], one["RAM_SPARE_GB"]) == (20, 0.15, 10)
    assert isinstance(one["CFG"], TinyConfig) and one["CFG"].steps == 12
    assert list(one["TEACHER_NAMES"]) == TEACHER_NAMES and set(TEACHER_NAMES) == set(TEACH.TEACHERS) - {"terrafm-b"}
    assert one["SUMMARY_ONLY"] == SUMMARY_ONLY and one["STUDENTS"] == STUDENTS and one["SPECS"] == SPECS
    assert one["OUT"] == OUT
    assert CALLS["env"] == ["model_cache_on_drive", "load_env", "require", "require"], CALLS["env"]
    assert "stand-in GPU of the dry run, 40.0 GB, free 30.0 GB" in said and "RAM     179.0 GB, free 170.0 GB" in said
    assert "steps 12, batch 8 = 6 Major-TOM + 2 aerial, crop 32 px" in said
    assert "checkpoint every 5 steps, held-out loss every 5, progress line every 100, PHI-S on 12 samples per teacher" in said
    assert f"student F  {SG} (summary only), {DINO}, {PE}, {RADAR}" in said
    assert f"student N  {SN} (summary only), {DINO}, {PE}, {RADAR}" in said
    assert f"student C  {SG} (summary only), {DINO}, {PE}" in said
    assert "on Drive      pilot/majortom of DATA_DRIVE_DIR (the pair archives)" in said
    assert "on Drive      pilot/aerial of DATA_DRIVE_DIR (the aerial archives)" in said and "NOT on Drive" not in said
    assert json.loads((RUNS / "runtime.json").read_text(encoding="utf-8"))["ram_gb"] == 179.0
    # the data. From the private repo of DATA_HF_REPO (the stand-in), with the read token, into the hub folder
    WANTED_FROM_HUB = {*(f"majortom/chunk_{k}.{e}" for k in range(2) for e in ("tar", "json")),
                       f"majortom/{D.TRAIN_FILE}", f"majortom/{D.HELD_OUT_FILE}", "majortom/channel_stats.json",
                       "aerial/manifest.json", "aerial/distil_train.tar", "aerial/distil_heldout.tar"}
    assert set(HUB_CALLS["downloads"]) == WANTED_FROM_HUB and HUB_CALLS["tokens"] == [FAKE_TOKEN] * 2
    assert (one["SOURCE_MT"], one["SOURCE_AERIAL"]) == (HUB_DIR / "majortom", HUB_DIR / "aerial")
    # the small files first in one call (32 at once), then the archives (4 at once, the pair archives in chunk order)
    N_SMALL = len([n for n in WANTED_FROM_HUB if not n.endswith(".tar")])
    assert HUB_CALLS["downloads"][N_SMALL:] and all(n.endswith(".tar") for n in HUB_CALLS["downloads"][N_SMALL:])
    assert re.search(rf"^pilot data from the private Hugging Face repo of DATA_HF_REPO, {N_SMALL} small files in \d+\.\d s, "
                     "the archives follow while the pairs are read$", said, re.M), said
    assert "from the archives on Drive" not in said and all((HUB_DIR / n).is_file() for n in WANTED_FROM_HUB)
    assert re.search(rf"fetched {N_SMALL} of {N_SMALL} files, [\d,]+ MB in \d+ s, [\d,]+ MB per second with 32 threads",
                     said), said
    assert re.search(r"fetched 4 of 4 files, [\d,]+ MB in \d+ s, [\d,]+ MB per second with 4 threads", said), said
    assert one["ARRIVED"] == {n for n in WANTED_FROM_HUB if n.endswith(".tar")}
    # the overlap - the pairs were read while the slow aerial archive was still being fetched
    assert READ_AT and min(READ_AT) < HUB_CALLS["done_at"]["aerial/distil_train.tar"], "the reading waited for the fetch"
    # the pairs straight from the fetched archives into RAM (the repo has no infos tables, they are built once
    # from the tars next to the fetched archives), the aerial sets, the statistics
    assert re.search(r"^chunk 0  infos built from chunk_0\.tar, 8 pairs, \d+\.\d s, written to chunk_0_infos\.csv$", said, re.M), said
    assert "chunk 1  infos built from chunk_1.tar, 8 pairs" in said
    assert "chunk 0  8 pairs in chunk_0.tar, read in place" in said and "chunk 1  8 pairs in chunk_1.tar, read in place" in said
    assert "16 pairs in 2 archives" in said
    assert files_of(HUB_DIR / "majortom") == {p.split("/")[1] for p in WANTED_FROM_HUB if p.startswith("majortom/")} | {
        "chunk_0_infos.csv", "chunk_1_infos.csv"}, "a pair archive was unpacked to the runtime disk"
    # the RAM check before anything is allocated, with the numbers of the lists and the manifest
    assert (f"the arrays will take {NEED_GB:.1f} GB of RAM (16 pairs, {N_PATCHES} aerial patches), 170.0 GB are free, "
            "10 GB must stay free") in said, said
    assert said.index("the arrays will take") < said.index("ChunkReader  16 pairs in 2 lists x 4.13 MB")
    members_mb = sum(PAIR_FILES[c].stat().st_size for c in CELLS) / 1e6
    for k in (0, 1):                          # one line per archive as it is read, 8 pairs each (both lists)
        assert re.search(rf"^chunk {k}  8 pairs read from chunk_{k}\.tar in \d+\.\d s, \d+ MB, [\d,]+ MB per second$",
                         said, re.M), said
    assert re.search(rf"^pairs to RAM  16 pairs of 2 lists from 2 archives, {members_mb:,.0f} MB in \d+\.\d s of reading, "
                     r"[\d,]+ MB per second, [\d,]+\.\d pairs per second, 16 threads$", said, re.M), said
    assert re.search(rf"^pairs in RAM, {members_mb:,.0f} MB read from the archives in \d+ s, [\d,]+ MB per second, "
                     r"training \(12, 12, 384, 384\), held-out \(4, 12, 384, 384\), corner rule applied True, 16 threads$",
                     said, re.M), said
    assert re.search(r"^fetch and read side by side \d+ s, one after the other \d+ s \(\d+ s fetching the archives, \d+ s "
                     r"reading the pairs and the aerial sets\), -?\d+ s saved$", said, re.M), said
    assert list(one["where"]) == CELLS and all(D.split_location(p)[0].parent == HUB_DIR / "majortom"
                                                for p in one["where"].values())
    # the arrays equal the sequential path (load_pairs_to_ram on each list after the fetch) bit for bit
    with contextlib.redirect_stdout(io.StringIO()):
        sequential = {name: REAL["load_pairs_to_ram"](one["lists"][name], corner_rule=True, progress=False)
                      for name in ("train", "held_out")}
    for name, got in (("train", one["ram_train"]), ("held_out", one["ram_held"])):
        want = sequential[name]
        assert got["s1"].dtype == want["s1"].dtype and got["s2"].dtype == want["s2"].dtype
        assert np.array_equal(got["s1"], want["s1"]) and np.array_equal(got["s2"], want["s2"]), name
        assert got["grid_cell"] == want["grid_cell"] and got["meta"].equals(want["meta"]), name
        assert got["corner_rule"] is want["corner_rule"] and got["bytes"] == want["bytes"], name
    del sequential, want, got
    assert one["reader"]._rows is None, "the reader keeps the arrays"
    assert not [t for t in threading.enumerate() if t.name == "ChunkReader"], "a reader thread is still running"
    assert all(D.split_location(p)[0].parent == HUB_DIR / "majortom" for p in one["lists"]["train"])
    assert "train     12 pairs from train_cells.txt" in said and "held_out  4 pairs from held_out_cells.txt" in said
    assert re.search(r"^aerial distil_train    read in place from distil_train\.tar  \(12, 4, 32, 32\), \d+ MB in "
                     r"\d+\.\d s, [\d,]+ MB per second$", said, re.M), said
    assert re.search(r"^aerial distil_heldout  read in place from distil_heldout\.tar  \(2, 4, 32, 32\), \d+ MB in "
                     r"\d+\.\d s, [\d,]+ MB per second$", said, re.M), said
    data = one["data"]
    assert data.s1.shape == (12, 2, 384, 384) and data.s2.shape == (12, 12, 384, 384)
    assert data.held_s1.shape == (4, 2, 384, 384) and data.aerial.shape == (12, 4, 32, 32)
    assert data.held_aerial.shape == (2, 4, 32, 32) and data.stats == STATS
    assert one["ram_train"]["grid_cell"] == TRAIN_CELLS and one["ram_held"]["grid_cell"] == HELD_CELLS
    for i, cell in enumerate(TRAIN_CELLS):            # the pairs came back whole from the archives
        pair = P.load_pair(PAIR_FILES[cell])
        assert np.array_equal(data.s1[i], pair.s1_db) and np.array_equal(data.s2[i, 1], pair.s2["B02"])
    # the teachers, tiny, built once, as targets. SigLIP2 summary only, the radar teacher reads Sentinel-1 and
    # no aerial, every other teacher Sentinel-2 and aerial
    assert CALLS["build_teacher"] == TEACHER_NAMES, CALLS["build_teacher"]
    targets = one["targets"]
    assert [t.name for t in targets] == TEACHER_NAMES and one["t"] is None
    assert [t.summary_only for t in targets] == [True, True, False, False, False]
    assert [t.radar is not None for t in targets] == [False, False, False, False, True]
    assert [t.aerial is not None for t in targets] == [True, True, True, True, False]
    for t in targets:
        reads = "Sentinel-1, aerial target no" if t.name == RADAR else "Sentinel-2, aerial target yes"
        kind = "summary only" if t.name in SUMMARY_ONLY else "summary and patches"
        assert f"{t.name:<22} reads {reads}, {kind}" in said, t.name
        assert re.search(rf"^{re.escape(t.name)} +loaded in \d+\.\d s, [\d,]+ parameters, rule \w+, input \d+ px, grid "
                         r"\d+ x \d+, width \d+$", said, re.M), t.name
    assert re.search(r"^5 teachers, [\d,]+ M parameters in \d+\.\d s \(\d+\.\d s reading ahead, -?\d+\.\d s building\), "
                     r"[\d,]+ M parameters per second$", said, re.M), said
    # the weight files read ahead once on the fake cache, before the first build, PE's file named as missing
    assert CALLS["warm"] == [TEACHER_NAMES], CALLS["warm"]
    warm_bytes = sum(p.stat().st_size for p in WARM_FILES.values())
    assert re.search(rf"^weights read ahead, 9 files, {warm_bytes / 1e9:.2f} GB in \d+\.\d s, [\d,]+ MB per second with "
                     r"5 threads, 1 not in the cache, 0 not readable$", said, re.M), said
    assert f"not in the cache at revision {TEACH.weight_files(PE)[0][1][:8]}, its build fetches it" in said
    assert said.index("weights read ahead") < said.index(" loaded in "), "a teacher was built before the read ahead"
    assert one["warmed"]["bytes"] == warm_bytes and len(one["warmed"]["files"]) == 9
    # the measurement, the joint loop in a scratch folder that is gone afterwards, its numbers on Drive
    assert CALLS["train"][0] == ("measure", FOLDERS, False, 20)
    assert said.count(joint_line(70.0)) == 2, said                     # the measurement and the long run
    measured = one["measured"]
    assert measured["mode"] == "joint" and measured["steps"] == 12 and measured["steps_counted"] == 7
    assert measured["batch"] == 8 and measured["students"] == STUDENTS and set(measured["student_s"]) == set(KEYS)
    assert abs(measured["data_share"] - 0.05) < 1e-9 and measured["device"] == GPU["name"]
    assert measured["hours_projected"] == measured["hours_joint"] > 0 and measured["images_per_s"] > 0
    assert abs(measured["images_per_s"] - 8 / measured["step_s"]) < 1e-9
    assert measured["hours_sequential"] > measured["hours_joint"]        # the data and the teachers three times
    assert measured["step_s"] == measured["one_step_s"] > sum(measured["student_s"].values())
    assert set(measured["loss_first"]) == set(measured["loss_last"]) == set(KEYS)
    assert abs(measured["step_s"] - (measured["data_s"] + measured["teacher_s"] + sum(measured["student_s"].values())
                                     + measured["rest_s"])) < 1e-9
    assert "12 steps measured (joint), the last 7 counted" in said
    assert re.search(r"^split of a step    data \d\.\d{3} s \(5%\), teachers \d\.\d{3} s \(\d+%\), student F \d\.\d{3} s "
                     r"\(\d+%\), student N \d\.\d{3} s \(\d+%\), student C \d\.\d{3} s \(\d+%\), rest -?\d\.\d{3} s "
                     r"\(-?\d+%\)$", said, re.M), said
    assert re.search(r"^images per second  [\d,]+, GPU memory peak 0\.0 GB$", said, re.M), said
    assert re.search(r"^projected          joint \d+\.\d\d h \(measured\), one after the other \d+\.\d\d h \(estimated\), "
                     r"for 12 steps without the held-out losses$", said, re.M), said
    assert re.search(r"^loss               F \d\.\d{4} at the first step and \d\.\d{4} at the last, N \d\.\d{4} at the "
                     r"first step and \d\.\d{4} at the last, C \d\.\d{4} at the first step and \d\.\d{4} at the last$",
                     said, re.M), said
    assert "the data path takes 5% of a step, under 15%, the long run can start" in said
    assert "ADVICE" not in said and not (WORK / "measure").exists()
    assert json.loads((RUNS / "measurement.json").read_text(encoding="utf-8")) == json.loads(json.dumps(measured))
    assert "trained" not in one and not any({"student", "heads", "phis"} & set(v) for v in one["results"].values()), \
        "the students and heads stay in the notebook, on the GPU"
    assert not {"logs", "steady"} & set(one), "the measured logs stay in the notebook"
    # the three students in one call, PHI-S fitted once per teacher whose patches are learned and copied,
    # none for SigLIP2, checkpoints and final.pt on Drive
    assert CALLS["train"][1] == ("runs_12", FOLDERS, True, None) and len(CALLS["train"]) == 2
    for key, names in STUDENTS.items():
        for name in names:
            line = f"{key}  phis {name}  not on Drive, fitted now or copied from another folder"
            assert (line in said) == (name not in SUMMARY_ONLY), line
    for name in (DINO, PE, RADAR):            # once in the measurement, once in the long run
        assert said.count(f"phis {name}: fitted on") == 2, name
    assert "phis siglip2" not in said
    for name, source, copies in ((DINO, "F", "NC"), (PE, "F", "NC"), (RADAR, "F", "N")):
        for key in copies:
            assert f"phis {name}: copied {OUT[source] / f'phis_{name}.pt'} to {OUT[key] / f'phis_{name}.pt'}" in said
    assert f"phis {RADAR}: copied {OUT['F'] / f'phis_{RADAR}.pt'} to {OUT['C']}" not in said
    for key in KEYS:
        assert f"{key}  step 12 of 12" in said and f"student_{key} step 12: held-out loss" in said
        result = one[f"result_{key.lower()}"]
        assert result["step"] == 12 and result["mode"] == "joint" and "held_out" in result
        assert result["final"] == OUT[key] / "final.pt" and not {"student", "heads", "phis"} & set(result), \
            "the models stay in the notebook, on the GPU"
        assert T.load_checkpoint(OUT[key] / "last_prev.pt")["step"] == 10       # the checkpoint before last.pt
        final = T.load_checkpoint(OUT[key] / "final.pt")
        assert final["step"] == 12 and T.summary_only_of(final) == [n for n in STUDENTS[key] if n in SUMMARY_ONLY]
        assert list(final["teacher_dims"]) == STUDENTS[key], final["teacher_dims"]
        held = log_rows(OUT[key] / "held_out.csv")
        assert list(held["step"]) == [5, 10, 12] and held["loss"].notna().all()
        assert files_of(OUT[key]) == PHIS[key] | {"last.pt", "last_prev.pt", "final.pt", "train_log.csv",
                                                  "held_out.csv"}, files_of(OUT[key])
        assert list(log_rows(OUT[key] / "train_log.csv").columns) == log_columns(key), key
    for name in (DINO, PE, RADAR):            # one PHI-S file per teacher, the same bytes in every folder
        assert len({(OUT[key] / f"phis_{name}.pt").read_bytes() for key in KEYS if name in STUDENTS[key]}) == 1, name
    logs = {key: log_rows(OUT[key] / "train_log.csv") for key in KEYS}
    assert all(list(log["step"]) == list(range(1, 13)) and (log["gpu_gb"] == 0).all() for log in logs.values())
    for column in ("step_s", "data_s", "teacher_s", "img_s"):     # one step holds the three students
        assert (logs["F"][column] == logs["N"][column]).all() and (logs["F"][column] == logs["C"][column]).all()
    assert all((log["student_s"] > 0).all() for log in logs.values())
    WEIGHTS.update({key: weights_of(OUT[key] / "final.pt") for key in KEYS})
    assert not same_weights(WEIGHTS["F"], WEIGHTS["N"]) and not same_weights(WEIGHTS["F"], WEIGHTS["C"])
    for key in KEYS:
        student, stats, cfg = T.load_student(OUT[key] / "final.pt", summary_only=SUMMARY_ONLY)
        assert stats == STATS and cfg.steps == 12 and student.chans == {"s1": 2, "s2": 12, "aerial": 4}
    # the summary - the held-out losses side by side, - where a teacher is not that student's
    final_paths = {key: str(OUT[key] / "final.pt") for key in KEYS}
    assert one["final_paths"] == final_paths and said.count("held-out loss, 3 rows, the last ones") == 3
    table = one["held_table"]
    assert list(table.columns) == list(KEYS)
    assert list(table.index) == ["step", "loss", *log_columns("F")[3:-7], f"{SN}/summary"], list(table.index)
    for key in KEYS:
        last = log_rows(OUT[key] / "held_out.csv").iloc[-1]
        for part in table.index:
            if part in last.index:
                assert table.loc[part, key] == last[part], (key, part)
            else:
                assert np.isnan(table.loc[part, key]), (key, part)
    assert "held-out losses at the last held-out step, per teacher part (rows) and student (columns)" in said
    on_drive = pd.read_csv(RUNS / "held_out_last.csv")
    assert list(on_drive.columns) == ["part", *KEYS] and list(on_drive["part"]) == list(table.index)
    assert "notebook 03 reads" in said and "teacher_s" in said and "student_s" in said
    assert re.search(r"^the run so far, 12 steps, \d\.\d{3} s per step, [\d,]+ images per second, GPU memory peak "
                     r"0\.0 GB$", said, re.M), said
    assert set(one["TIMES"]) == set(STEPS)
    assert files_of(RUNS) == {*FOLDERS, "runtime.json", "measurement.json", "final_paths.json", "times.json",
                              "held_out_last.csv"}, files_of(RUNS)
    for path in RUNS.rglob("*"):
        if path.suffix in (".csv", ".json"):
            assert FAKE_TOKEN not in path.read_text(encoding="utf-8"), path
    assert files_of(CONTENT) == {"drive", "sar-transfer-distillation"} and files_of(WORK) == {"hub"}
    assert files_of(MYDRIVE) == {FAKE_DRIVE_DIR, "sar-transfer-distillation", "model_cache"}
    print("\nfirst run done, every cell ran, the three students trained in one loop and are on Drive")

    # ---- second run, as on a new runtime. Drive holds the three finished students. No data is loaded, no
    # teacher built, nothing measured or trained ---------------------------------------------------------
    shutil.rmtree(CONTENT / "sar-transfer-distillation")
    before_calls = {k: list(v) for k, v in CALLS.items()}
    downloads = len(HUB_CALLS["downloads"])
    stamps = {**{f"{key}/{k}": v for key in KEYS for k, v in stamps_of(OUT[key]).items()},
              "measurement.json": (RUNS / "measurement.json").stat().st_mtime_ns}
    two: dict = {"__name__": "__main__"}
    since = len(PATCHED)
    patch(D, "ChunkReader", not_again("pairs loaded again"))
    said = run_cells(ALL, two, "second run")
    unpatch(since)
    assert CALLS == {**before_calls, "env": before_calls["env"] * 2}, f"work was repeated, {CALLS}"
    assert len(HUB_CALLS["downloads"]) == downloads, "the data was fetched again"
    changed = [name for name, stamp in stamps.items()
               if (RUNS / re.sub(r"^([FNC])/", r"student_\1/", name)).stat().st_mtime_ns != stamp]
    assert not changed, f"written again in the second run, {changed}"
    assert "every student is finished on Drive (final.pt), the data is not loaded" in said and two["data"] is None
    assert "every student is finished on Drive, the teachers are not built" in said
    assert two["teachers"] is None and two["targets"] is None
    assert "every student is finished on Drive, no measurement is needed" in said and two["measured"] is None
    assert said.count("final.pt found on Drive, not trained again") == 3
    assert "every student is finished on Drive, nothing to train" in said
    assert (two["result_f"], two["result_n"], two["result_c"]) == (None, None, None) and two["results"] == {}
    assert said.count("found on Drive") >= 8 and "fitted now" not in said and "students:" not in said
    assert two["final_paths"] == final_paths and two["held_table"].equals(one["held_table"])
    assert files_of(WORK) == set(), "the runtime disk holds something"
    print("\nsecond run done, the three students came from Drive, nothing was loaded, built, measured or trained")

    # ---- third run, the same session. A joint run cut at step 10, then C's last.pt deleted, so C lags at
    # step 5. The data cell first with a Hub that fails (the archives on Drive are taken), then with nothing on
    # Drive either (it stops). Then all students must go on from step 5 (F and N from their last_prev.pt) and
    # give the weights of the first run. The long run has begun, so the measurement is not taken again
    cut_joint_run(one, 10)
    (OUT["C"] / "last.pt").unlink()
    hub_fake.fail = True
    said = run_cells([DATA], two, "third run, the Hub fails")
    assert "pilot data from the archives on Drive, not from Hugging Face (FileNotFoundError, stand-in" in said
    assert "<token>" in said and (two["SOURCE_MT"], two["SOURCE_AERIAL"]) == (DRIVE_MT, DRIVE_AERIAL)
    assert len(HUB_CALLS["downloads"]) == downloads and "chunk 0  8 pairs in chunk_0.tar, read in place" in said
    # Drive had no infos tables (part three's first data notebook), they are built once from the tars there
    assert "chunk 0  infos built from chunk_0.tar, 8 pairs" in said and "chunk 1  infos built from chunk_1.tar, 8 pairs" in said
    assert (DRIVE_MT / "chunk_0_infos.csv").is_file() and (DRIVE_MT / "chunk_1_infos.csv").is_file()
    assert all(D.split_location(p)[0].parent == DRIVE_MT for p in two["lists"]["train"]), "the archives on Drive are read in place"
    for name in ("s1", "s2", "aerial", "held_s1", "held_s2", "held_aerial"):     # the same arrays from Drive
        assert np.array_equal(getattr(two["data"], name), getattr(data, name)), name
    os.replace(DRIVE_MT, PILOT / "majortom_aside")
    printed = must_stop(DATA, two, "the Hub fails and Drive has no archives", AssertionError,
                        "pilot/majortom of DATA_DRIVE_DIR is not on Drive either (the pair archives)")
    os.replace(PILOT / "majortom_aside", DRIVE_MT)
    assert "data" not in two, "the stopped cell freed the arrays first"
    said = run_cells([DATA], two, "data cell from Drive again")
    assert "pilot data from the archives on Drive" in said and two["data"].s2.shape == (12, 12, 384, 384)
    assert "infos built" not in said, "the infos tables on Drive were built again"
    # the Hub answers but two archives do not arrive - chunk_1 and the held-out aerial set. The small files and
    # the other two archives come from the Hub, these two from Drive in place, the arrays are those of the first run
    hub_fake.fail = False
    HUB_REFUSE.update({"majortom/chunk_1.tar", "aerial/distil_heldout.tar"})
    said = run_cells([DATA], two, "two archives do not arrive")
    HUB_REFUSE.clear()
    hub_fake.fail = True
    assert "pilot data from the private Hugging Face repo of DATA_HF_REPO" in said
    assert "the archives that did not arrive are read from Drive (OSError" in said and "did not arrive whole" in said
    assert two["ARRIVED"] == {"majortom/chunk_0.tar", "aerial/distil_train.tar"}
    assert (HUB_DIR / "majortom" / "chunk_0.tar").is_file() and not (HUB_DIR / "majortom" / "chunk_1.tar").exists()
    assert not (HUB_DIR / "aerial" / "distil_heldout.tar").exists()
    assert "chunk 0  8 pairs read from chunk_0.tar" in said and "chunk 1  8 pairs read from chunk_1.tar" in said
    assert "chunk 1  infos built" not in said, "chunk_1 was not read from Drive, where its infos table is"
    for name in ("s1", "s2", "aerial", "held_s1", "held_s2", "held_aerial"):
        assert np.array_equal(getattr(two["data"], name), getattr(data, name)), name
    measured_stamp = (RUNS / "measurement.json").stat().st_mtime_ns
    said = run_cells([TEACHERS, MEASURE_CELL, STUDENTS_CELL, SUMMARY], two, "third run, C lags behind F and N")
    assert CALLS["build_teacher"][len(before_calls["build_teacher"]):] == TEACHER_NAMES
    assert [c[0] for c in CALLS["train"][2:]] == ["runs_12"], \
        f"the measurement was taken again although a student has a checkpoint, {CALLS['train'][2:]}"
    assert "a student has a checkpoint on Drive, the long run has begun" in said and "measurement is not repeated" in said
    assert two["measured"]["steps"] == 12 and "(joint), of it the data path" in said and "share 5.0%" in said
    assert (RUNS / "measurement.json").stat().st_mtime_ns == measured_stamp
    assert "F  last.pt found on Drive, the run goes on from it" in said
    assert "N  last.pt found on Drive, the run goes on from it" in said
    assert "C  last_prev.pt found on Drive, the run goes on from it" in said
    assert f"C  phis {DINO}  found on Drive" in said and "fitted now" not in said and "copied" not in said
    assert joint_line(70.0) in said
    for key in ("F", "N"):
        assert f"resumed from {OUT[key] / 'last_prev.pt'} at step 5 (its newest checkpoint is at step 10, the steps " \
               "from 5 on are computed again" in said, said
    assert f"resumed from {OUT['C'] / 'last_prev.pt'} at step 5 (the previous checkpoint, last.pt is missing)" in said
    for key in KEYS:
        assert f"{key}  step 12 of 12" in said
        assert list(log_rows(OUT[key] / "train_log.csv")["step"]) == list(range(1, 13))
        assert list(log_rows(OUT[key] / "held_out.csv")["step"]) == [5, 10, 12]
        assert (T.load_checkpoint(OUT[key] / "last.pt")["step"], T.load_checkpoint(OUT[key] / "last_prev.pt")["step"]) == (12, 10)
    same_as_first("third run")
    assert two["final_paths"] == final_paths
    print("\nthird run done, the data came from Drive, the students went on from the smaller step and equal the first run")

    # ---- checkpoints on Drive but no measurement file (deleted for a new one). Measured again ----
    (RUNS / "measurement.json").unlink()
    cut_joint_run(two, 5)
    said = run_cells([MEASURE_CELL, STUDENTS_CELL], two, "checkpoints without a measurement file")
    assert [c[0] for c in CALLS["train"][3:]] == ["measure", "runs_12"], CALLS["train"][3:]
    assert "12 steps measured (joint)" in said and (RUNS / "measurement.json").is_file()
    for key in KEYS:
        assert f"resumed from {OUT[key] / 'last.pt'} at step 5" in said and f"{key}  step 12 of 12" in said
    assert "computed again" not in said
    same_as_first("measured again")

    # ---- the fallback. Too little free GPU memory, the three students train one after the other through
    # train_student, the measurement says so and projects the joint run, the weights are the same
    FREE["gb"] = 30.0
    (RUNS / "measurement.json").unlink()
    unlink("final.pt", "last.pt", "last_prev.pt")
    said = run_cells([MEASURE_CELL, STUDENTS_CELL], two, "fallback, 30 GB free")
    fallback_line = (f"students: {', '.join(FOLDERS)} one after the other (train_student), only 30.0 GB of GPU memory "
                     f"free, under {T.JOINT_MIN_GB}")
    assert said.count(fallback_line) == 2 and "in one loop" not in said, said
    for k, folder in enumerate(FOLDERS, 1):
        assert f"{folder}: student {k} of 3" in said
    assert two["measured"]["mode"] == "sequential" and two["measured"]["hours_projected"] == two["measured"]["hours_sequential"]
    assert two["measured"]["hours_joint"] < two["measured"]["hours_sequential"]
    assert abs(two["measured"]["data_share"] - 0.05) < 1e-9
    assert "12 steps measured (sequential)" in said
    assert re.search(r"^seconds per step   \d\.\d{3} for the 3 students one after the other \(\d\.\d{3} \+ \d\.\d{3} \+ "
                     r"\d\.\d{3}\)$", said, re.M), said
    assert re.search(r"^projected          joint \d+\.\d\d h \(estimated\), one after the other \d+\.\d\d h \(measured\)",
                     said, re.M), said
    assert "split of a step    data" in said and "the long run can start" in said
    assert "step 12: held-out loss" in said and "student_F step 12" not in said   # train_student's own lines
    assert f"phis {DINO}: loaded" in said and "fitted now" not in said
    for key in KEYS:
        assert f"{key}  step 12 of 12" in said and two[f"result_{key.lower()}"]["mode"] == "sequential"
        assert list(log_rows(OUT[key] / "train_log.csv").columns) == log_columns(key)
    same_as_first("the fallback")
    assert not (WORK / "measure").exists()
    FREE["gb"] = 70.0
    print("\nthe fallback trains one student after the other and gives the same weights")

    # ---- the measurement with a data path that is too slow. The notebook stops with the advice
    unlink("final.pt", "last.pt")
    (RUNS / "measurement.json").unlink()      # the fallback measurement, deleted so that it is measured again
    MEASURE["data_share"] = 0.5
    printed = must_stop(MEASURE_CELL, two, "slow data path", RuntimeError, "the data path is too slow")
    assert "ADVICE  the data path takes 50% of a step, more than 15%" in printed
    assert "Nothing was changed automatically" in printed
    assert abs(json.loads((RUNS / "measurement.json").read_text(encoding="utf-8"))["data_share"] - 0.5) < 1e-9
    assert not (WORK / "measure").exists()
    MEASURE["data_share"] = 0.05
    # only last_prev.pt is left of every student and the measurement file is on Drive, so the long run has begun.
    # The measure cell must read the file and not measure again (a runtime that died while it wrote last.pt
    # leaves last_prev.pt alone, the students cell goes on from it)
    for out in OUT.values():
        assert (out / "last_prev.pt").is_file() and not (out / "last.pt").exists()
    runs_before = len(CALLS["train"])
    said = run_cells([MEASURE_CELL], two, "last_prev.pt alone, the measurement file on Drive")
    assert len(CALLS["train"]) == runs_before, "the measurement was taken again although last_prev.pt is on Drive"
    assert "a student has a checkpoint on Drive, the long run has begun" in said and "share 50.0%" in said
    (RUNS / "measurement.json").unlink()      # deleted for a new one, as the intro says. Measured again
    said = run_cells([MEASURE_CELL, STUDENTS_CELL], two, "data path fine again")
    assert "the long run can start" in said and "12 steps measured (joint)" in said
    for key in KEYS:
        assert f"{key}  last_prev.pt found on Drive, the run goes on from it" in said and f"{key}  step 12 of 12" in said
        assert f"resumed from {OUT[key] / 'last_prev.pt'} at step 10 (the previous checkpoint, last.pt is missing)" in said
    same_as_first("resumed from last_prev.pt")

    # ---- the data cell with the statistics missing, a cut archive, a cell without a pair file and a cut
    # aerial archive. F's final.pt is deleted first, else the cell loads nothing (every student finished)
    (OUT["F"] / "final.pt").unlink()
    good = (DRIVE_MT / "channel_stats.json").read_bytes()
    (DRIVE_MT / "channel_stats.json").unlink()
    must_stop(DATA, two, "statistics missing", AssertionError, "channel_stats.json is missing in the source")
    (DRIVE_MT / "channel_stats.json").write_bytes(good)
    whole = (DRIVE_MT / "chunk_1.tar").read_bytes()
    (DRIVE_MT / "chunk_1.tar").write_bytes(whole[:len(whole) // 2])
    must_stop(DATA, two, "cut archive", AssertionError, "chunk_1.tar is missing or cut")
    (DRIVE_MT / "chunk_1.tar").write_bytes(whole)
    (DRIVE_MT / D.HELD_OUT_FILE).write_text("nobody_here\n", encoding="utf-8")
    must_stop(DATA, two, "a cell without a pair file", AssertionError, "have no pair file")
    (DRIVE_MT / D.HELD_OUT_FILE).write_text("".join(f"{c}\n" for c in HELD_CELLS), encoding="utf-8")
    run_cells([DATA], two, "data cell whole again")
    assert two["SOURCE_AERIAL"] == DRIVE_AERIAL
    whole_aerial = (DRIVE_AERIAL / "distil_heldout.tar").read_bytes()
    (DRIVE_AERIAL / "distil_heldout.tar").write_bytes(whole_aerial[:len(whole_aerial) // 2])
    must_stop(DATA, two, "cut aerial archive", (RuntimeError, AssertionError), "distil_heldout.tar")
    (DRIVE_AERIAL / "distil_heldout.tar").write_bytes(whole_aerial)
    run_cells([DATA], two, "data cell whole again, the aerial archive restored")

    # ---- the data cell a second time in the same session (the intro says to run from the top after a crash).
    # The arrays of the first load must be gone BEFORE the loader allocates the new ones, else two copies of
    # about 88 GB sit in RAM at once and the 179 GB runtime dies
    old_s2, old_aerial = weakref.ref(two["data"].s2), weakref.ref(two["sets"]["distil_train"]["images"])
    assert old_s2() is not None and old_aerial() is not None

    class ReaderAfterFree(REAL["ChunkReader"]):
        def __init__(self, *args, **kwargs):
            assert old_s2() is None and old_aerial() is None, \
                "the arrays of the earlier load are still in RAM when the new load starts"
            super().__init__(*args, **kwargs)

    since = len(PATCHED)
    patch(D, "ChunkReader", ReaderAfterFree)
    said = run_cells([DATA], two, "data cell again in the same session")
    unpatch(since)
    assert "chunk 0  8 pairs in chunk_0.tar, read in place" in said and two["data"].s2.shape == (12, 12, 384, 384)
    assert old_s2() is None and two["data"].stats == STATS
    print("\nthe data cell run twice in one session frees the first arrays before it loads again")

    # ---- the data cell on a runtime with too little RAM (the standard one). It stops before anything is made
    RAM.update(total=89.6, free=5.0)
    since = len(PATCHED)
    patch(D, "ChunkReader", not_again("allocated before the check"))
    printed = must_stop(DATA, two, "too little RAM", AssertionError, "Pick the A100 runtime with High-RAM")
    unpatch(since)
    assert f"the arrays will take {NEED_GB:.1f} GB of RAM (16 pairs, {N_PATCHES} aerial patches), 5.0 GB are free" in printed
    assert "data" not in two and "ram_train" not in two
    RAM.update(total=179.0, free=170.0)
    run_cells([DATA], two, "data cell with High-RAM again")

    # ---- one student whose final.pt is missing (a runtime that died between the final writes). It alone goes
    # on from its last.pt at the last step and writes final.pt, the others are not touched
    stamps = {key: (OUT[key] / "final.pt").stat().st_mtime_ns for key in ("N", "C")}
    said = run_cells([TEACHERS, MEASURE_CELL, STUDENTS_CELL, SUMMARY], two, "F without final.pt")
    assert CALLS["train"][-1] == ("runs_12", ["student_F"], True, None), CALLS["train"][-1]
    assert "a student has a checkpoint on Drive, the long run has begun" in said
    assert "F  last.pt found on Drive, the run goes on from it" in said
    assert "N  final.pt found on Drive, not trained again" in said and "C  final.pt found on Drive, not trained again" in said
    assert "students: student_F in one loop" in said and f"resumed from {OUT['F'] / 'last.pt'} at step 12" in said
    assert all((OUT[key] / "final.pt").stat().st_mtime_ns == stamp for key, stamp in stamps.items())
    assert two["result_n"] is None and two["result_c"] is None and two["result_f"]["step"] == 12
    same_as_first("F alone")
    assert two["final_paths"] == final_paths

    # ---- the setup cell on a runtime without Drive. It stops before the token is asked for --
    WRONG["drive not mounted"] = True
    asked = len(CALLS["env"])
    must_stop(SETUP, {"__name__": "__main__"}, "no Drive", AssertionError, "Google Drive is not mounted")
    assert CALLS["env"][asked:] == ["model_cache_on_drive"], CALLS["env"][asked:]
    WRONG["drive not mounted"] = False
    print("\nwithout Drive the setup cell stops at once and asks for no token")

    assert not CALLS["network"], f"the dry run asked for the network, {CALLS['network']}"
    assert set(HUB_CALLS["tokens"]) == {FAKE_TOKEN}, "another token reached the Hub"
finally:
    unpatch()
    for k in HAD_DATA:                     # the settings of the fake .env leave with the test
        os.environ.pop(k, None)

assert not PATCHED and T.train_students is REAL["train_students"] and T.TrainConfig is REAL["TrainConfig"]
assert T.free_gpu_gb is REAL["free_gpu_gb"] and T.free_gpu_gb("cpu") is None
assert TEACH.build_teacher is REAL["build_teacher"] and subprocess.run is REAL["run"]
assert D.load_pairs_to_ram is REAL["load_pairs_to_ram"] and Z.runtime_report is REAL["runtime_report"]
assert D.ChunkReader is REAL["ChunkReader"] and D.open_pair is REAL["open_pair"]
assert TEACH.warm_weights is REAL["warm_weights"]
assert not [t for t in threading.enumerate() if t.name == "ChunkReader"], "a reader thread is still running"
assert H.hub_api is REAL["hub_api"] and H.hub_download is REAL["hub_download"] and E.load_env is REAL["load_env"]
assert socket.getaddrinfo is REAL["lookup"] and not torch.cuda.is_available()
assert Z.WORK_FOLDERS == ("/content", "/content/drive/MyDrive")
assert T.TrainConfig().steps == 6250
for key in ("HF_HOME", "HF_TOKEN", "HF_HUB_DISABLE_SYMLINKS_WARNING", "DATA_HF_REPO", "DATA_DRIVE_DIR"):
    assert os.environ.get(key) == at_start.get(key), f"a cell changed {key}"
for k, v in HAD_DATA.items():
    if v is not None:
        os.environ[k] = v
assert Path("/content").exists() == HAD_CONTENT, "a cell wrote to the folder /content"

del one, two, data
shutil.rmtree(TMP, ignore_errors=True)
print("test_notebook02: all checks passed")
