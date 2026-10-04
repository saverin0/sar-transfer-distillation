"""Dry run of notebooks/03_score.ipynb on this PC, no network, no GPU, no real weights.

Every code cell but the mount cell, the sync cell and the flush cell is executed in order and
unchanged. What a cell would reach on Colab is replaced from OUTSIDE, by patching names of the
modules

    folders /content and Drive    stage0.WORK_FOLDERS, a temp folder on D that holds what
                                  notebook 02 writes (three tiny students F, N and C trained
                                  here in one loop with the real train_students, a radar and
                                  two summary-only toy teachers), the pilot data of part three
                                  (the aerial archives), part three's results folder (made-up
                                  summaries of terrafm-b-s1, students M and S, dinov3-aerial,
                                  and its own stage A table made from them), part two's results
                                  folder (the control summary made by the same tiny control
                                  run, made-up summaries of the satellite DINOv3 and C-RADIOv4-H,
                                  and a decoy run of the DINOv3 at another layer that must never
                                  mix in), and the Drive copies of the four SAR datasets under
                                  MyDrive/datasets
    .env file and token           env.load_env reads a fake .env with the four settings
                                  DATA_HF_REPO, DATA_DRIVE_DIR, REFERENCE_DRIVE_DIR and
                                  DATASET_HF_OWNER (the real parser, made-up values; the third
                                  run reads one without DATASET_HF_OWNER), env.require gives a
                                  made-up token and the folders of that .env
    the model cache               env.model_cache_on_drive, a folder of the temp Drive
    the GPU check                 three names of torch.cuda, only while the setup cell runs
    the dataset copies on the Hub huggingface_hub.snapshot_download, so the REAL hub.fetch runs,
                                  copies the files its allow_patterns ask for from a local
                                  folder; one dataset is not there, its Drive copy is taken
                                  through the real prepared.copy_prepared (the failing item),
                                  and in the first run snow is held back until the first SAR
                                  call starts (the slow item)
    the background fetcher        the REAL score.Prefetch of the prefetch cell, its two lane
                                  threads; the read-ahead lane reads the tiny control encoder
                                  and the three student files for real
    the private pilot repo        hub.hub_api and hub.hub_download over a dict that holds the
                                  aerial files of the synthetic Drive, so the real fetch_pilot
                                  runs. In the third run it fails, the archives on Drive are read
    the reference encoder         models.encoders.ENCODERS entry dinov3-l-sat points at a tiny
                                  DINOv3 with 22 layers saved here (the control's layer 21). The
                                  saved encoders of part two and three are never built here -
                                  terrafm.build and the dinov3-aerial entry fail when called
    score.score_all               the real function on the CPU with one decoder seed and 10
                                  decoder steps. The notebook asks for cuda and 3 seeds
    score.score_parallel          the same, with 3 workers whose processes are a stand-in context
                                  that runs each worker in this process (score._context), so the
                                  orchestration, the skips, the logs and the printed lines are
                                  the real ones. Once, the aerial cell with two runs to do starts
                                  two REAL processes (spawn), and their summaries must equal the
                                  ones made in this process byte for byte

A connection to another machine stops the test. The made-up token must not show up in anything
the cells print or save. sys.stdout and sys.stderr are one object for the whole dry run that
hands what a cell prints to that cell's buffer, so the streams the fetcher wraps in the prefetch
cell stay in place for the next cells, as in a notebook.

Runs. The first one from nothing. The control runs while snow is still held back, so it waits
for forest and the DINOv3 weights only; the first SAR call takes the three datasets on the disk,
snow comes in a second call; the data lane is ready in its order; every wait and the waited
line are printed. The second one in the same session, every run is skipped and nothing is
fetched but forest, which is on the disk. The third one as on a new runtime with a .env without
DATASET_HF_OWNER, the pilot repo out of reach, one dataset folder that holds only its manifest and
the summaries of two runs gone - only forest, snow and the aerial sets are fetched, from Drive
and from the archives, the two runs are made again byte for byte, the others skipped. Then the
control with a reference off by 0.01 in a probe score (passes), 0.03 (stops), a decoder seed mean
off by 0.005 (passes) and 0.03 (stops), a count off by one pixel, a row lacking and the file
missing (stop). Then the table cell with part three's own table off in one number (stops), the
SAR cell with one summary gone (names it, goes on), the aerial cell with two real processes, a
dataset on neither the Hub nor Drive (named, its run MISSING, the other lane goes on), the
verdicts cell on made-up rows that give every word and one row missing, and
last the setup cell without a student, without a saved run of part two and without Drive (stop).

STAND-INS ONLY, so NOT tested against the real thing by this file - the mount cell, the sync
cell and the flush cell as cells (the packed source is unpacked and compared, the flush cell
is compared with the shared flush cell, neither is run), the real .env file and the
token, the model cache on Drive, every download from Hugging Face, the real weights, datasets,
students and saved results and their sizes, every CUDA call, how far a CUDA run is from part
two's saved numbers, Drive as a file system, and the library versions of Colab.
"""
import _setup

import contextlib
import fnmatch
import io
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from pathlib import Path

os.environ["IPYTHONDIR"] = str(_setup.TMP_BASE / "ipython")

import huggingface_hub  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sartransfer import env as E  # noqa: E402
from sartransfer import sync as SYNC  # noqa: E402
from sartransfer.distil import aerial as A  # noqa: E402
from sartransfer.distil import score as SC  # noqa: E402
from sartransfer.distil import stage0 as Z  # noqa: E402
from sartransfer.distil import train as T  # noqa: E402
from sartransfer.distil.teachers import dinov3_sat as DS  # noqa: E402
from sartransfer.distil.teachers import terrafm as TF  # noqa: E402
from sartransfer.distil.teachers.base import TeacherOut  # noqa: E402
from sartransfer.domains import hub as H  # noqa: E402
from sartransfer.domains import prepared as PR  # noqa: E402
from sartransfer.domains import run as RUN  # noqa: E402
from sartransfer.models import encoders as ENC  # noqa: E402

NOTEBOOK = _setup.REPO / "notebooks" / "03_score.ipynb"
TMP = _setup.TMP_BASE / "notebook03"
for _ in range(40):                    # Windows may take a moment to let go of a folder
    shutil.rmtree(TMP, ignore_errors=True)
    if not TMP.exists():
        break
    time.sleep(0.25)
TMP.mkdir(parents=True)                # files of an older run would change what is tested
CONTENT = TMP / "content"                                  # stands for /content
MYDRIVE = CONTENT / "drive" / "MyDrive"                    # stands for /content/drive/MyDrive
FAKE_REPO = "made-up-owner/made-up-pilot-data"             # DATA_HF_REPO of the fake .env
FAKE_DRIVE_DIR = "made-up-data-folder"                     # DATA_DRIVE_DIR, part three's data and results
FAKE_REFERENCE_DIR = "made-up-reference-folder"            # REFERENCE_DRIVE_DIR, part two's results
FAKE_OWNER = "made-up-dataset-owner"                       # DATASET_HF_OWNER, the SAR dataset copies
PREP = CONTENT / "sar-transfer-distillation" / "prepared"         # the notebook derives these from the settings
HUB_DIR = CONTENT / "sar-transfer-distillation" / "hub"
RES = MYDRIVE / "sar-transfer-distillation" / "results"
RUNS = MYDRIVE / "sar-transfer-distillation" / "runs_6250"        # the folder of notebook 02, named by the steps
REF = MYDRIVE / FAKE_REFERENCE_DIR / "results"
PART3 = MYDRIVE / FAKE_DRIVE_DIR / "results"
DRIVE_AERIAL = MYDRIVE / FAKE_DRIVE_DIR / "pilot" / "aerial"
DRIVE_DATASETS = MYDRIVE / "datasets"
HUB = TMP / "hub"                                          # what the dataset repos on the Hub hold
STAGING = TMP / "staging"
for folder in (REF, PART3, RUNS, DRIVE_AERIAL, DRIVE_DATASETS, HUB, STAGING):
    folder.mkdir(parents=True)
FAKE_TOKEN = "MADE-UP-TOKEN-OF-THE-DRY-RUN"
SETTINGS = {"DATA_HF_REPO": FAKE_REPO, "DATA_DRIVE_DIR": FAKE_DRIVE_DIR, "REFERENCE_DRIVE_DIR": FAKE_REFERENCE_DIR,
            "DATASET_HF_OWNER": FAKE_OWNER}
FAKE_ENV, ENV_NO_OWNER = TMP / "fake.env", TMP / "fake_no_owner.env"
FAKE_ENV.write_text("".join(f"{k}={v}\n" for k, v in SETTINGS.items()), encoding="utf-8")
ENV_NO_OWNER.write_text("".join(f"{k}={v}\n" for k, v in SETTINGS.items() if k != "DATASET_HF_OWNER"),
                        encoding="utf-8")
HAD_ENV = {k: os.environ.pop(k, None) for k in (*SETTINGS, "HF_WRITE_TOKEN")}   # the fake .env sets them
HAD_CONTENT = Path("/content").exists()         # the folder that a cell must not write to
SAR = ("glacial_lakes", "snow", "forty_v1", "glavitu_alp")
ON_DRIVE_ONLY = "glavitu_alp"                   # not on the Hub, taken from its Drive copy
CONTROL_RUN = "dom_forty_v1__dinov3-l-sat__L21__vv_vh_diff"
STUDENTS = ("student-f", "student-n", "student-c")
WORDS = ("beats", "matches", "mixed", "does not reach")

# ---- the file -----------------------------------------------------------------------------
NAMES = ["intro", "mount", "sync", "setup", "prefetch", "control", "sar-runs", "aerial-runs", "table", "verdicts",
         "flush"]
(INTRO, MOUNT, SYNC_CELL, SETUP, PREFETCH, CONTROL, SAR_RUNS, AERIAL_RUNS, TABLE, VERDICTS, FLUSH) = range(len(NAMES))
STEPS = ["setup", "prefetch", "control", "sar_runs", "aerial_runs", "table", "verdicts", "waited_for_data"]
text = NOTEBOOK.read_text(encoding="utf-8")
nb = json.loads(text)
assert (nb["nbformat"], nb["nbformat_minor"]) == (4, 5)
assert nb["metadata"]["kernelspec"]["name"] == "python3"
cells = nb["cells"]
assert [c["cell_type"] for c in cells] == ["markdown"] + ["code"] * 10
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

# the text a reader sees. No colon at all, and the facts the task asks for
intro = SOURCE[INTRO]
assert intro.startswith("# ") and ":" not in intro, "a colon in the markdown cell"
for words in ("students F, N and C of notebook 02", "run step of part two, unchanged", "",
              "validation splits", "test splits wait", "control", "layer 21", "1e-6", "0.02", "bfloat16",
              "mean over the seeds with the tie rule", "Sentinel-1 mode", "FLAIR-HUB", "19 classes",
              # the saved rows, never run again
              "Saved rows, never run again", "C-RADIOv4-H at layer 32", "TerraFM-B reading Sentinel-1",
              "students M and S", "DINOv3 SAT reading the aerial RGB", "checked against part three's own stage A table",
              "summary file of its own run", "Nothing is written into the folders of part two or part three",
              # the table and the verdicts, written by the code
              "IoU of the positive class", "forest and the aerial set by mIoU", "written by the code, not by hand",
              "more than 0.01 AND by more than twice the larger seed SD", "gap is more than 0.02",
              "4 SAR validation decoder rows", "the aerial row is reported and not counted",
              "4 wins and no loss = beats", "no loss and fewer than 4 wins = matches",
              "1 or more losses and more wins than losses = mixed", "otherwise does not reach",
              "F against the frozen satellite DINOv3, the headline", "F against C", "N against F",
              "F and N against their radar teacher", "student M of the part three pilot, informational",
              # the .env settings, the data sources
              "DATASET_HF_OWNER", "MyDrive/datasets", "DATA_HF_REPO and DATA_DRIVE_DIR", "REFERENCE_DRIVE_DIR",
              "never names another project's folders itself", "train and val, and every split of the forest dataset",
              "archives on Drive when the fetch fails", "which source was used", "read straight out of the archives",
              "never printed",
              # the data in the background, the order and why, who waits for what
              "## Data in the background", "starts right after the setup cell", "two lanes of one thread each",
              "no cell waits for data it does not need yet", "the item needed first gets the whole bandwidth",
              "8 files at once from the Hub or 4 from Drive", "Forest first, the control needs it",
              "Then the alpine glaciers, small (about 1.2 GB of train and val) but with the longest runs",
              "the GPU has work for longer than the rest takes to arrive",
              "lakes (about 5.5 GB) before snow (about 7 GB)", "the aerial sets last",
              "the read-ahead lane reads the DINOv3 weights of the control (1.2 GB) and the three student files",
              "with FAST_DECODER it first unpacks the compile cache of an earlier session",
              "the control waits for forest and the DINOv3 weights only",
              "The decoder trains with part two's step, as every saved row", "is off",
              "moved 0.020 down in the first run of this notebook and 0.013 up in notebook 03a",
              "115 ms at 992 px and 32 ms at 512 px",
              "start as soon as it is on the runtime disk", "the ones still arriving into the next",
              "falls back to the copy on Drive without stopping the others", "its runs are MISSING",
              "is not fetched, forest always", "shows in the cell that waits for it",
              "the seconds the GPU work waited for data against the data one after the other",
              # runtime, time, results, how to run
              "The A100 with High-RAM, as part three's notebook 03", "VS Code", "About 1 hour 40 minutes, an estimate",
              "16 scoring runs in 121 minutes", "3 students x 5 datasets plus the control",
              "side by side, one process each on the same GPU", "snow and the alpine glaciers gain little",
              "The data no longer comes before the runs", "About 14 GB of SAR data and 2 GB of aerial archives",
              "part three's data cells took 2.4 minutes from the Hub", "the GPU waits only at the start",
              "60 GB of GPU memory and 100 GB of RAM", "fewer students run at once",
              "results/logs/score__student-f.log", "the fetcher of the first run stops before the new one starts",
              "MyDrive/sar-transfer-distillation/results/", "verdicts.json", "stage_a_rows.csv", "skipped", "FAILED",
              "END_RUNTIME = False", "run the mount cell again"):
    assert words in intro, f"the first cell does not say {words!r}"
# no name of another repository or of its folders in any cell (the sync cell holds base64 only)
FORBIDDEN = _setup.private_names()            # empty in a public copy
for k, source in enumerate(SOURCE):
    if k != SYNC_CELL:
        hits = [w for w in FORBIDDEN if w in source]
        assert not hits, f"cell {k} ({NAMES[k]}) names another repository or its folder, {hits}"
for k in range(MOUNT, len(cells)):
    first = SOURCE[k].splitlines()[0]
    assert first.startswith("# ") and ":" not in first, f"cell {k} starts with {first!r}"
    comments = [ln for ln in SOURCE[k].splitlines() if ln.lstrip().startswith("#") and k != SYNC_CELL]
    assert not [ln for ln in comments if ":" in ln], f"cell {k} has a colon in a comment"

assert "google.colab" in SOURCE[MOUNT] and 'drive.mount("/content/drive")' in SOURCE[MOUNT]
# the last cell is the shared flush cell, byte for byte - flush Drive, then release the machine
assert SOURCE[FLUSH] == _setup.FLUSH_SOURCE, "the last cell differs from the shared flush cell"
assert "drive.flush_and_unmount()" in SOURCE[FLUSH] and "END_RUNTIME = True" in SOURCE[FLUSH]
assert SOURCE[FLUSH].index("drive.flush_and_unmount()") < SOURCE[FLUSH].index("print(") \
    < SOURCE[FLUSH].index("runtime.unassign()"), "flush, then the message, then the release"
assert "if END_RUNTIME" in SOURCE[FLUSH] and "mount cell" in SOURCE[FLUSH]
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
    assert "print(HF_TOKEN" not in source and "{HF_TOKEN" not in source and "DATA_TOKEN}" not in source
    assert re.search(r'print\(took\(start, "[a-z_]+"\)\)$', source.rstrip()), f"cell {k} does not print its time"
    assert "start = time.perf_counter()" in source
    CODE[k] = compile(source, f"03_score cell {k} ({NAMES[k]})", "exec")
for words in ("env.load_env()", 'env.require("HF_TOKEN"', "env.model_cache_on_drive()",
              'DATA_DIR = env.require("DATA_DRIVE_DIR"', 'REFERENCE_DIR = env.require("REFERENCE_DRIVE_DIR"',
              'DATASET_OWNER = os.environ.get("DATASET_HF_OWNER", "").strip()',
              "assert torch.cuda.is_available()", "stage0.runtime_report(", 'os.environ.get("HF_WRITE_TOKEN") or HF_TOKEN',
              'SPLITS = ("val",)', 'CONTROL_DATASET, CONTROL_ENCODER = "forty_v1", "dinov3-l-sat"',
              "SCORE_TOLERANCE = 0.02", "COUNT_TOLERANCE = 1e-6", "DECODER_MEAN_TOLERANCE = 0.01",
              'REF = MYDRIVE / REFERENCE_DIR / "results"', 'PART3 = MYDRIVE / DATA_DIR / "results"',
              'DRIVE_AERIAL = MYDRIVE / DATA_DIR / "pilot" / "aerial"', 'DRIVE_DATASETS = MYDRIVE / "datasets"',
              'RES = MYDRIVE / "sar-transfer-distillation" / "results"', 'f"runs_{train.TrainConfig().steps}"',
              "train.FINAL", '"student-f": "student_F"', '"student-n": "student_N"', '"student-c": "student_C"',
              "run_step.PRE_REGISTERED_LAYERS[e]", "def missing_runs(", "score.POSITIVE_CLASS_DATASETS"):
    assert words in SOURCE[SETUP], f"the setup cell has no {words}"
for k, call in ((PREFETCH, "hub.fetch(name, DATASET_OWNER, PREP, token=DATA_TOKEN, splits=NEEDED[name])"),
                (PREFETCH, "DRIVE_DATASETS / hub.repo_name(name)"),
                (PREFETCH, "prepared.copy_prepared(copy, PREP, name, splits=NEEDED[name])"),
                (PREFETCH, 'None if name == CONTROL_DATASET else ("train", *SPLITS)'), (PREFETCH, "def on_disk("),
                (PREFETCH, "hub.fetch_pilot(HUB_DIR, token=HF_TOKEN,"),
                (PREFETCH, 'want=("aerial/manifest.json", "aerial/probe.tar", "aerial/val.tar")'),
                (PREFETCH, "folder, source = DRIVE_AERIAL"), (PREFETCH, "dict(enumerate(aerial.CLASSES))"),
                (PREFETCH, 'aerial.load_set_from_tar(archive, name, manifest["sets"][name]["files"])'),
                (PREFETCH, 'DATA_ORDER = ("forty_v1", "glavitu_alp", "glacial_lakes", "snow")'),
                (PREFETCH, 'score.Prefetch(secrets=(HF_TOKEN, os.environ.get("HF_WRITE_TOKEN")))'),
                (PREFETCH, 'prefetch.add(name, lambda name=name: fetch_sar(name), lane="data")'),
                (PREFETCH, 'prefetch.add(AERIAL_DATASET, fetch_aerial, lane="data")'),
                (PREFETCH, 'prefetch.add("DINOv3 weights", read_control_weights, lane="read-ahead")'),
                (PREFETCH, 'prefetch.add("student weights", read_students, lane="read-ahead")'),
                (PREFETCH, 'score.model_files(ENCODERS[CONTROL_ENCODER].model_id, cache_dir=MODEL_CACHE / "hub")'),
                (PREFETCH, "score.read_ahead(CHECKPOINTS.values())"), (PREFETCH, "prefetch.start()"),
                (PREFETCH, "prefetch.close()"), (PREFETCH, "disable_progress_bars()"),
                (PREFETCH, "missing_runs([score.run_name(d, e, LAST, S1_MODE) for e in SAR_ENCODERS])"),
                (PREFETCH, "AERIAL_TODO = bool(missing_runs(AERIAL_RUNS))"),
                (CONTROL, 'prefetch.wait(CONTROL_DATASET, by="control")'),
                (CONTROL, 'prefetch.wait("DINOv3 weights", by="control")'),
                (CONTROL, "run_step.PRE_REGISTERED_LAYERS[CONTROL_ENCODER]"),
                (CONTROL, "score.score_all(PREP, [CONTROL_ENCODER], [CONTROL_DATASET], RES, token=HF_TOKEN, device=DEVICE,"),
                (SETUP, "FAST_DECODER = False"), (SETUP, 'COMPILE_CACHE = CONTENT / "sar-transfer-distillation" / "compile_cache"'),
                (SETUP, 'COMPILE_ARCHIVE = MYDRIVE / "sar-transfer-distillation" / "compile_cache.tar"'),
                (PREFETCH, "score.restore_compile_cache(COMPILE_ARCHIVE, COMPILE_CACHE)"),
                (PREFETCH, 'prefetch.add("compile cache", restore_cache, lane="read-ahead")'),
                (PREFETCH, "if FAST_DECODER and (CONTROL_TODO or SAR_TODO or AERIAL_TODO)"),
                (CONTROL, 'prefetch.wait("compile cache", by="control")'),
                (VERDICTS, "score.save_compile_cache(COMPILE_CACHE, COMPILE_ARCHIVE)"),
                (CONTROL, "raise RuntimeError"), (CONTROL, "DECODER_MEAN_TOLERANCE"),
                (SAR_RUNS, 'prefetch.wait("student weights", by="SAR runs")'),
                (SAR_RUNS, 'prefetch.first_ready(left, by="SAR runs")'), (SAR_RUNS, 'prefetch.record(d)["ok"]'),
                (SAR_RUNS, "checkpoints=CHECKPOINTS"), (SAR_RUNS, "splits=SPLITS"), (SAR_RUNS, "missing_runs(SAR_RUNS)"),
                (SAR_RUNS, "score.score_parallel(PREP, list(SAR_ENCODERS), sar, RES, token=HF_TOKEN,"),
                (SAR_RUNS, "score.gain_line(times_sar, time.perf_counter() - start)"),
                (AERIAL_RUNS, 'prefetch.wait("student weights", by="aerial runs")'),
                (AERIAL_RUNS, 'prefetch.wait(AERIAL_DATASET, by="aerial runs")'),
                (AERIAL_RUNS, "checkpoints=CHECKPOINTS"), (AERIAL_RUNS, "missing_runs(AERIAL_RUNS)"),
                (AERIAL_RUNS, "score.score_parallel(PREP, list(AERIAL_ENCODERS), [AERIAL_DATASET], RES,"),
                (AERIAL_RUNS, "score.gain_line(times_aerial, time.perf_counter() - start)"),
                (AERIAL_RUNS, "prefetch.close()"), (AERIAL_RUNS, "prefetch.summary()"),
                (AERIAL_RUNS, 'TIMES["waited_for_data"] = sum(prefetch.waited.values())'),
                (AERIAL_RUNS, "print(prefetch.waited_line())"),
                (TABLE, "score.read_runs(folder, runs, splits=SPLITS)"), (TABLE, 'PART3 / "stage_a_table.csv"'),
                (TABLE, "score.check_against("), (TABLE, "score.reading_score("),
                (VERDICTS, "score.stage_a_verdicts(table, split=SPLITS[0])"), (VERDICTS, "score.verdict_lines(verdicts)"),
                (VERDICTS, "stage0.save_report(RES, verdicts=verdicts, stage_a_rows=compared_rows")):
    assert call in SOURCE[k], f"cell {k} ({NAMES[k]}) has no {call}"
assert "21" not in SOURCE[CONTROL] and "1e-6" not in SOURCE[CONTROL], "the control layer or tolerance is retyped"
assert "extractall" not in SOURCE[PREFETCH], "the prefetch cell unpacks an archive"
# the order of the data lane and why, the threads and why, written next to them
for words in ("The order of the data lane. Forest first, the control needs it", "Then the alpine glaciers",
              "the longest runs", "the GPU has work for longer than the rest takes to arrive",
              "Then lakes (about 5.5 GB) before snow (about 7 GB, the biggest)", "cannot make a run wait",
              "The aerial sets last", "Two lanes, one thread each", "the item needed first gets the whole bandwidth",
              "one thread per file, the reason is written there"):
    flat = re.sub(r"\n\s*# ?", " ", SOURCE[PREFETCH])          # the comments as one text
    assert words in flat, f"the prefetch cell does not say {words!r}"
for k in (CONTROL, SAR_RUNS, AERIAL_RUNS):            # each consumer names its own items only
    waits = re.findall(r'prefetch\.(?:wait|first_ready)\(([^,]+),', SOURCE[k])
    assert waits, k
assert re.findall(r'prefetch\.wait\(([^,]+),', SOURCE[CONTROL]) == ["CONTROL_DATASET", '"compile cache"',
                                                                  '"DINOv3 weights"']
# every scoring call runs the fast decoder path with the one cache folder, the cache goes to Drive after the results
for k in (CONTROL, SAR_RUNS, AERIAL_RUNS):
    flat = re.sub(r"\s+", " ", SOURCE[k])
    assert flat.count("fast_decoder=FAST_DECODER, compile_cache=COMPILE_CACHE") == 1, NAMES[k]
assert SOURCE[VERDICTS].index("stage0.save_report(") < SOURCE[VERDICTS].index("score.save_compile_cache(")
assert re.findall(r'prefetch\.(?:wait|first_ready)\(([^,]+),', SOURCE[SAR_RUNS]) == ['"student weights"', "left"]
assert re.findall(r'prefetch\.wait\(([^,]+),', SOURCE[AERIAL_RUNS]) == ['"student weights"', "AERIAL_DATASET"]
for k in (SAR_RUNS, AERIAL_RUNS):           # the saved encoders are never run again
    assert not any(e in SOURCE[k] for e in ("terrafm", "dinov3-aerial", "cradio", "student-m")), k
    assert "score_all" not in SOURCE[k] and "workers=" not in SOURCE[k], k     # side by side, the rule of score.py
assert "score_parallel" not in SOURCE[CONTROL], "the control is one run, sequential"
for number in ("0.01", "0.02", "2.0", "4 wins"):       # the rule lives in score.py, not in the cell
    assert f"= {number}" not in SOURCE[VERDICTS]
print(f"the file is a notebook of {len(cells)} cells without outputs, the sync cell is up to date")

# ---- the tiny reference encoder of the control --------------------------------------------------
from transformers import DINOv3ViTConfig, DINOv3ViTModel  # noqa: E402

DINO_DIR = TMP / "dinov3_tiny_22"
torch.manual_seed(7)
with contextlib.redirect_stdout(io.StringIO()):
    DINOv3ViTModel(DINOv3ViTConfig(**{**DS.TINY_CONFIG, "num_hidden_layers": 22})).save_pretrained(DINO_DIR)
assert RUN.PRE_REGISTERED_LAYERS["dinov3-l-sat"] == 21 <= 22
TINY_ENCODERS = {
    "dinov3-l-sat": ENC.EncoderSpec(str(DINO_DIR), "dinov3-l-sat", "dinov3", batch=4),
    # part three's rows are saved, so this one must never be built; a missing folder fails at once
    "dinov3-aerial": ENC.EncoderSpec(str(TMP / "never-built"), "dinov3-aerial", "dinov3", batch=4, in_channels=(4,))}


def never_build(**kw):
    raise AssertionError("TerraFM-B was built, its rows are saved rows of part three")


# ---- the three students, trained here in one loop with the real train_students (3 steps, CPU) --
PX = 32
STATS = {"s1": {"channels": ["VV", "VH"], "unit": "dB", "mean": [-14.0, -16.0], "sd": [5.0, 4.0]},
         "s2": {"channels": ["B01"] * 12, "unit": "DN", "mean": [4000.0] * 12, "sd": [2300.0] * 12},
         "aerial": {"channels": ["R", "G", "B", "NIR"], "unit": "8 bit", "mean": [100.0, 110.0, 90.0, 120.0],
                    "sd": [50.0, 45.0, 40.0, 55.0]}}


class ToyTeacher:
    """A frozen function of its input, in the shape of a teacher."""

    def __init__(self, dim, grid, in_chans, scale, seed):
        gen = torch.Generator().manual_seed(seed)
        self.dim, self.grid, self.scale = dim, grid, scale
        self.w_sum = torch.randn(in_chans, dim, generator=gen) / in_chans ** 0.5
        self.w_spa = torch.randn(in_chans, dim, generator=gen) / in_chans ** 0.5

    @torch.inference_mode()
    def __call__(self, x):
        x = x.float() / self.scale
        pooled = F.adaptive_avg_pool2d(x, self.grid).permute(0, 2, 3, 1)
        return TeacherOut(torch.tanh(x.mean((2, 3)) @ self.w_sum), torch.tanh(pooled @ self.w_spa))


gt = np.random.default_rng(3)
student_cfg = T.TrainConfig(steps=3, batch=8, n_aerial=2, crop=PX, warmup=1, checkpoint_every=3,
                            held_out_every=3, phis_samples=4, bf16=False,
                            student=dict(img=PX, patch=16, dim=32, depth=2, heads=4))
student_data = T.TrainData(gt.normal(-15.0, 5.0, (40, 2, 48, 48)).astype(np.float16),
                           gt.integers(1, 8000, (40, 12, 48, 48), dtype=np.uint16),
                           gt.integers(0, 256, (12, 4, PX, PX), dtype=np.uint8), STATS)
pool = [T.Target("sg", 8, 2, ToyTeacher(8, 2, 12, 1000.0, 20), ToyTeacher(8, 2, 4, 255.0, 21)),
        T.Target("sn", 8, 2, ToyTeacher(8, 2, 12, 1000.0, 22), ToyTeacher(8, 2, 4, 255.0, 23)),
        T.Target("ta", 8, 2, ToyTeacher(8, 2, 12, 1000.0, 10), ToyTeacher(8, 2, 4, 255.0, 11)),
        T.Target("rd", 8, 2, None, radar=ToyTeacher(8, 2, 2, 10.0, 13))]
SPECS = [("student_F", ["sg", "ta", "rd"]), ("student_N", ["sn", "ta", "rd"]), ("student_C", ["sg", "ta"])]
with contextlib.redirect_stdout(io.StringIO()):
    T.train_students(SPECS, student_cfg, student_data, pool, RUNS, device="cpu", joint=True,
                     summary_only=("sg", "sn"))
CHECKPOINTS = {f"student-{folder[-1].lower()}": RUNS / folder / T.FINAL for folder, _ in SPECS}
assert all(p.is_file() for p in CHECKPOINTS.values()) and list(CHECKPOINTS) == list(STUDENTS)
assert T.load_checkpoint(CHECKPOINTS["student-f"])["summary_only"] == ["sg"]   # the new checkpoint record

# ---- the aerial probe and validation sets, as notebook 01 of part three left them on Drive -----
AERIAL_SETS = {"probe": [("D004", "2021", 6), ("D017", "2018", 6)],
               "val": [("D012", "2019", 3), ("D036", "2020", 3)]}
manifest = {"repo": A.HF_REPO, "revision": A.HF_REVISION, "out_px": PX, "classes": list(A.CLASSES), "sets": {}}
for name, parts in AERIAL_SETS.items():
    files = []
    for j, (dept, year, n) in enumerate(parts):
        rng = np.random.default_rng(200 + j + len(name))
        file = f"{name}__{dept}-{year}.npz"
        labels = rng.integers(0, A.N_CLASSES, (n, PX, PX), dtype=np.uint8)
        images = np.clip(labels[:, None] * 12 + rng.normal(0, 10, (n, 4, PX, PX)), 0, 255).astype(np.uint8)
        np.savez(STAGING / file, images=images, labels=labels,
                 ids=np.array([f"{dept}-{year}_AA-S1-1_{i}-1" for i in range(n)]), department=dept, year=year)
        files.append(file)
    with tarfile.open(DRIVE_AERIAL / f"{name}.tar", "w") as tar:
        for file in files:
            tar.add(STAGING / file, arcname=file)
    manifest["sets"][name] = {"count": sum(p[2] for p in parts), "files": files,
                              "departments": sorted({p[0] for p in parts}), "class_shares": None}
(DRIVE_AERIAL / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


class FakeHub:
    """list_repo_tree over a dict of bytes. `fail` = the repo cannot be reached."""

    class Entry:
        def __init__(self, path, size):
            self.path, self.size = path, size

    def __init__(self, store):
        self.store, self.fail = store, False

    def list_repo_tree(self, rid, repo_type, recursive):
        assert repo_type == "dataset" and recursive is True and rid == FAKE_REPO
        if self.fail:
            raise FileNotFoundError(f"stand-in, {rid} is not on the Hub, the token was {FAKE_TOKEN}")
        return [self.Entry(p, len(b)) for p, b in self.store.items()]


HUB_STORE = {f"aerial/{p.name}": p.read_bytes() for p in DRIVE_AERIAL.iterdir() if p.is_file()}
HUB_STORE["majortom/channel_stats.json"] = b"{}"           # of the pilot too, never asked for here
hub_fake = FakeHub(HUB_STORE)
HUB_CALLS = {"tokens": [], "downloads": []}


def fake_hub_api(token):
    HUB_CALLS["tokens"].append(token)
    assert token == FAKE_TOKEN, "the fetch used another token than the read token"
    return hub_fake


def fake_hub_download(rid, name, dest, token):
    assert token == FAKE_TOKEN and rid == FAKE_REPO
    HUB_CALLS["downloads"].append(name)
    target = Path(dest) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(HUB_STORE[name])
    return target


# ---- the four SAR datasets in the prepared format of part two, tiny. Three on the Hub, all four on Drive
SPLITS_OF = {"glacial_lakes": ("train", "val", "test", "test_challenge", "test_region_SEE"),
             "snow": ("train", "val", "test", "test_spatial", "test_temporal"),
             "forty_v1": ("train", "val", "test"), "glavitu_alp": ("train", "val", "test")}
CLASSES_OF = {"glacial_lakes": {0: "background", 1: "lake"}, "snow": {0: "no_snow", 1: "snow"},
              "forty_v1": {0: "other", 1: "forest"}, "glavitu_alp": {0: "background", 1: "glacier"}}


def toy_sar(root: Path, name: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    w = PR.PreparedWriter(root, name, 64, CLASSES_OF[name], source={"url": "synthetic"}, pixel_spacing_m=10.0)
    for split in SPLITS_OF[name]:
        for i in range(24 if split == "train" else 6):
            yy, xx = np.mgrid[:64, :64]
            cy, cx, r = rng.integers(16, 48, 2).tolist() + [rng.integers(8, 20)]
            blob = ((yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2)
            vv = np.where(blob, -22.0, -9.0) + rng.normal(0, 1.5, (64, 64))
            vh = vv - 7 + rng.normal(0, 1.0, (64, 64))
            img = np.stack([vv, vh]).astype(np.float32)
            img[:, :4, :4] = np.nan
            lab = blob.astype(np.uint8)
            lab[:4, :4] = 255
            w.add(split, img, lab, region=f"R{i % 3}", date="2020-08-01")
    w.close()


for j, name in enumerate(SAR):
    toy_sar(HUB if name != ON_DRIVE_ONLY else STAGING / "sar", name, 20 + j)
    source = HUB / name if name != ON_DRIVE_ONLY else STAGING / "sar" / name
    shutil.copytree(source, DRIVE_DATASETS / H.repo_name(name))       # the Drive copies, the same files
    (DRIVE_DATASETS / H.repo_name(name) / "README.md").write_text("card\n", encoding="utf-8")
assert not (HUB / ON_DRIVE_ONLY).exists() and (DRIVE_DATASETS / "sar-transfer-glavitu-alp" / "manifest.json").is_file()

# ---- part two's results folder. The control summary from the same tiny run, the other rows made up
SCORE_KW = dict(seeds=(0,), steps=10, batch=4)     # what the score_all stand-in adds on the CPU
saved_encoders = {k: ENC.ENCODERS[k] for k in TINY_ENCODERS}
ENC.ENCODERS.update(TINY_ENCODERS)
scratch_prep, scratch_res = TMP / "scratch_prep", TMP / "scratch_res"
shutil.copytree(HUB / "forty_v1", scratch_prep / "forty_v1")
with contextlib.redirect_stdout(io.StringIO()):
    control_rows = SC.score_all(scratch_prep, ["dinov3-l-sat"], ["forty_v1"], scratch_res, token=None,
                                device="cpu", **SCORE_KW)
ENC.ENCODERS.update(saved_encoders)
assert set(control_rows.run) == {CONTROL_RUN} and set(control_rows.split) == {"val", "test"}
shutil.copyfile(scratch_res / f"{CONTROL_RUN}__summary.csv", REF / f"{CONTROL_RUN}__summary.csv")
GOOD_REFERENCE = (REF / f"{CONTROL_RUN}__summary.csv").read_bytes()


def made_up(folder: Path, dataset: str, encoder: str, layer, mode: str, splits, decoder: float, probe: float,
            sd: float = 0.004) -> str:
    """A summary file of the run step with made-up numbers, majority, probe and three decoder seeds."""
    run = SC.run_name(dataset, encoder, layer, mode)
    rows = []
    for head, seed, value in (("majority", -1, 0.3), ("probe", -1, probe), ("decoder", 0, decoder - sd),
                              ("decoder", 1, decoder), ("decoder", 2, decoder + sd)):
        for split in splits:
            row = {"run": run, "dataset": dataset, "encoder": encoder, "layer": layer, "mode": mode, "units": "dB",
                   "head": head, "seed": seed, "split": split, "mIoU": value, "accuracy": 0.8, "n_pixels": 1000}
            if dataset != "flair_hub":
                row["IoU_pos"] = value - 0.1
            rows.append(row)
    pd.DataFrame(rows).to_csv(folder / f"{run}__summary.csv", index=False)
    return run


rng = np.random.default_rng(11)
PART2_RUNS, PART3_RUNS = [CONTROL_RUN], []
for dataset in SAR:
    if dataset != "forty_v1":
        PART2_RUNS.append(made_up(REF, dataset, "dinov3-l-sat", 21, "vv_vh_diff", ("val", "test"), *rng.uniform(0.4, 0.7, 2)))
    PART2_RUNS.append(made_up(REF, dataset, "cradio-v4-h", 32, "vv_vh_diff", ("val", "test"), *rng.uniform(0.4, 0.7, 2)))
    for encoder in ("terrafm-b-s1", "student-m", "student-s"):
        PART3_RUNS.append(made_up(PART3, dataset, encoder, "last", "co_cross", ("val",), *rng.uniform(0.3, 0.6, 2)))
for encoder in ("student-m", "student-s", "dinov3-aerial"):
    PART3_RUNS.append(made_up(PART3, "flair_hub", encoder, "last", "aerial_rgbn", ("val",), *rng.uniform(0.3, 0.5, 2)))
# a decoy in part two's folder, the DINOv3 at another layer - read by run name, it never mixes in
DECOY = made_up(REF, "glacial_lakes", "dinov3-l-sat", 24, "vv_vh_diff", ("val", "test"), 0.99, 0.99)
# part three's control repeat sits in its folder too, its rows are not part three's own rows
shutil.copyfile(scratch_res / f"{CONTROL_RUN}__summary.csv", PART3 / f"{CONTROL_RUN}__summary.csv")
# part three's own table, as its notebook 03 wrote it (score_table of its results next to part two's)
OWN_TABLE = SC.score_table(PART3, reference=REF)
OWN_TABLE.to_csv(PART3 / "stage_a_table.csv", index=False)
GOOD_OWN = (PART3 / "stage_a_table.csv").read_bytes()
print(f"synthetic Drive of notebook 02, of part three and part two in {MYDRIVE}, {len(SAR)} tiny SAR datasets")


def snapshot(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


SAVED_BEFORE = {"part two": snapshot(REF.parent), "part three": snapshot(PART3.parent)}

# ---- the stand-ins, patched in from outside --------------------------------------------------
CALLS = {"snapshot": [], "score": [], "network": [], "env": []}
WRONG = {"drive not mounted": False, "env file": FAKE_ENV}
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
        "load_env": E.load_env, "score_all": SC.score_all, "build": TF.build, "hub_api": H.hub_api,
        "hub_download": H.hub_download, "snapshot": huggingface_hub.snapshot_download,
        "score_parallel": SC.score_parallel, "context": SC._context}
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
    """The real parser on the fake .env of the dry run (the four settings, no token)."""
    CALLS["env"].append("load_env")
    for key in SETTINGS:                        # a new runtime, nothing from an earlier run
        os.environ.pop(key, None)
    print("stand-in, the fake .env of the dry run is read, not the one on Drive")
    return REAL["load_env"](WRONG["env file"], verbose=False)


def made_up_token(key: str, why: str = "") -> str:
    CALLS["env"].append(f"require {key}")
    if key in ("DATA_DRIVE_DIR", "REFERENCE_DRIVE_DIR"):     # from the fake .env, nothing is asked for
        return os.environ[key]
    assert key == "HF_TOKEN", key
    return FAKE_TOKEN


def cache_in_temp(*args, **kwargs) -> Path | None:
    CALLS["env"].append("model_cache_on_drive")
    if WRONG["drive not mounted"]:
        return None
    root = MYDRIVE / "model_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


# The slow item of the first run. snow's fetch is held back until the first SAR call starts; the control
# stand-in notes what snow was doing while the control ran and waits until lakes and the alpine glaciers are
# through, so the first SAR call takes those three and snow comes in a second call
HOLD = {"on": False, "snow": threading.Event(), "seen": {}, "timed out": False}
NOW: dict = {"names": None}                     # the names of the cells running now
HOLD_SECONDS = 60                               # a hold that is never let go ends after this, the test then fails


def item_status(name: str) -> str:
    return NOW["names"]["prefetch"].record(name)["status"]


def fake_snapshot(repo_id, repo_type=None, local_dir=None, token=None, allow_patterns=None, max_workers=8, **kw):
    """snapshot_download from a local folder, the files its allow_patterns ask for. A dataset that is
    not there fails as the Hub would. snow waits while HOLD is on."""
    CALLS["snapshot"].append((repo_id, None if allow_patterns is None else tuple(allow_patterns), max_workers))
    assert repo_type == "dataset" and token == FAKE_TOKEN and not kw, (repo_type, kw)
    owner, name = repo_id.split("/")
    assert owner == FAKE_OWNER, owner
    if HOLD["on"] and name == H.repo_name("snow") and not HOLD["snow"].wait(HOLD_SECONDS):
        HOLD["timed out"] = True
    src = HUB / name[len("sar-transfer-"):].replace("-", "_")
    if not src.is_dir():
        raise FileNotFoundError(f"stand-in, {repo_id} is not on the Hub, token {token}")
    out = Path(local_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if allow_patterns is None or any(fnmatch.fnmatchcase(p.name, pat) for pat in allow_patterns):
            shutil.copyfile(p, out / p.name)
    return str(out)


def score_on_cpu(prepared_root, encoders, datasets, res_dir, token, checkpoints=None, device="cuda",
                 splits=None, **run_kw):
    """The real score_all on the CPU, one decoder seed, 10 decoder steps. The notebook asks for cuda."""
    assert device == "cuda" and token == FAKE_TOKEN, device
    assert (run_kw.pop("fast_decoder") is False and Path(run_kw.pop("compile_cache")).name == "compile_cache"
            and not run_kw), run_kw
    CALLS["score"].append((tuple(encoders), tuple(datasets), splits, None if checkpoints is None
                           else {k: str(v) for k, v in checkpoints.items()}))
    if HOLD["on"]:                               # the control runs while snow is still arriving
        HOLD["seen"]["control"] = {n: item_status(n) for n in NOW["names"]["prefetch"].names}
        clock = time.perf_counter()
        while any(item_status(n) in SC.BUSY for n in ("glacial_lakes", "glavitu_alp")):
            assert time.perf_counter() - clock < HOLD_SECONDS, "lakes and the alpine glaciers never arrived"
            time.sleep(0.05)
    return REAL["score_all"](prepared_root, encoders, datasets, res_dir, token=None, checkpoints=checkpoints,
                             device="cpu", splits=splits, **SCORE_KW)


WORKERS = {"n": 3, "started": []}            # what the score_parallel stand-in asks for, the workers started


def parallel_on_cpu(prepared_root, encoders, datasets, res_dir, token, checkpoints=None, device="cuda",
                    splits=None, **run_kw):
    """The real score_parallel on the CPU, one decoder seed, 10 decoder steps, WORKERS["n"] at once. The
    notebook asks for cuda and leaves the worker count to score.py. The made-up token goes on to the workers,
    so a printed or logged token would show."""
    assert device == "cuda" and token == FAKE_TOKEN, device
    assert (run_kw.pop("fast_decoder") is False and Path(run_kw.pop("compile_cache")).name == "compile_cache"
            and not run_kw), run_kw
    CALLS["score"].append((tuple(encoders), tuple(datasets), splits, None if checkpoints is None
                           else {k: str(v) for k, v in checkpoints.items()}))
    if HOLD["on"] and not HOLD["snow"].is_set():  # the first SAR call starts while snow is still arriving
        HOLD["seen"]["first SAR call"] = (tuple(datasets), item_status("snow"))
        HOLD["snow"].set()
    return REAL["score_parallel"](prepared_root, encoders, datasets, res_dir, token=token, checkpoints=checkpoints,
                                  device="cpu", splits=splits, workers=WORKERS["n"], **SCORE_KW)


class OwnThreadOnly:
    """contextlib for score.py while its workers run in this process. A real worker is a process of its own,
    so its redirect of sys.stdout and sys.stderr into its log never reaches the threads of the fetcher here;
    this redirect sends only the writes of the thread that made it to the log, the rest goes on as before."""

    class Router:
        def __init__(self, base, target, owner):
            self.base, self.target, self.owner = base, target, owner

        def write(self, text):
            return (self.target if threading.get_ident() == self.owner else self.base).write(text)

        def flush(self):
            (self.target if threading.get_ident() == self.owner else self.base).flush()

    @staticmethod
    @contextlib.contextmanager
    def _redirect(name, target):
        before = getattr(sys, name)
        setattr(sys, name, OwnThreadOnly.Router(before, target, threading.get_ident()))
        try:
            yield target
        finally:
            setattr(sys, name, before)

    def redirect_stdout(self, target):
        return self._redirect("stdout", target)

    def redirect_stderr(self, target):
        return self._redirect("stderr", target)

    def __getattr__(self, name):
        return getattr(contextlib, name)


class InProcess:
    """A stand-in multiprocessing context for score.py. Process.start() runs the worker here, to its end."""

    Queue = queue.Queue

    class Process:
        def __init__(self, target, name, args):
            self.target, self.args, self.exitcode = target, args, None

        def start(self):
            WORKERS["started"].append(self.args[0])
            self.target(*self.args)
            self.exitcode = 0

        def is_alive(self):
            return False

        def join(self):
            pass

        def terminate(self):
            raise AssertionError("nothing is left to stop")


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


class CellOutput:
    """sys.stdout and sys.stderr of the whole dry run, one object. What is written while a cell runs goes to
    that cell's buffer, everything else to the console. The fetcher wraps this object in the prefetch cell and
    its wrapper stays in place for the next cells, as the streams of a notebook kernel do."""

    def __init__(self, console):
        self.console, self.buffer = console, None

    def write(self, text):
        return (self.console if self.buffer is None else self.buffer).write(text)

    def flush(self):
        (self.console if self.buffer is None else self.buffer).flush()


CELL_OUT = CellOutput(sys.stdout)
REAL_STREAMS = (sys.stdout, sys.stderr)


@contextlib.contextmanager
def into(buffer):
    """What the cells print goes to `buffer` while this runs."""
    CELL_OUT.buffer = buffer
    try:
        yield
    finally:
        CELL_OUT.buffer = None


def run_cells(which, names: dict, title: str) -> str:
    """Execute these code cells in `names`, unchanged. Returns what they printed."""
    printed = []
    NOW["names"] = names
    for k in which:
        out = io.StringIO()
        begin = time.perf_counter()
        try:
            with into(out):
                with gpu_stand_in() if k == SETUP else contextlib.nullcontext():
                    exec(CODE[k], names)
        except BaseException:
            print(out.getvalue())
            traceback.print_exc()
            raise SystemExit(f"{title}, cell {k} ({NAMES[k]}) raised") from None
        said = out.getvalue()
        print(f"\n==== {title}, cell {k} ({NAMES[k]}), {time.perf_counter() - begin:.1f} s")
        print("\n".join("    | " + line for line in said.splitlines() if "\r" not in line))
        assert FAKE_TOKEN not in said, f"cell {k} printed the token"
        assert "cell time" in said.splitlines()[-1], f"cell {k} did not print its time"
        printed.append(said)
    return "\n".join(printed)


def must_stop(k: int, names: dict, title: str, kind, says: str) -> str:
    """Cell `k` must raise `kind` with `says` in the message. Returns what it printed."""
    out = io.StringIO()
    NOW["names"] = names
    try:
        with into(out):
            with gpu_stand_in() if k == SETUP else contextlib.nullcontext():
                exec(CODE[k], names)
    except kind as err:
        assert says in str(err), f"{title}, cell {k} stopped with {err}"
        print(f"\n==== {title}, cell {k} ({NAMES[k]}) stopped as it must, {says!r}")
        assert FAKE_TOKEN not in out.getvalue()
        return out.getvalue()
    raise SystemExit(f"{title}, cell {k} ({NAMES[k]}) ran to its end")


def summaries() -> dict:
    return {p.name: p.stat().st_mtime_ns for p in RES.glob("dom_*__summary.csv")}


def lane_threads() -> list:
    """The threads of a fetcher that are still alive."""
    return [t for t in threading.enumerate() if t.name.startswith("prefetch ") and t.is_alive()]


STUDENT_RUNS = {f"dom_{d}__{e}__Llast__co_cross" for d in SAR for e in STUDENTS}
AERIAL_RUNS = {f"dom_flair_hub__{e}__Llast__aerial_rgbn" for e in STUDENTS}
ALL = list(range(SETUP, FLUSH))
at_start = dict(os.environ)
try:
    patch(subprocess, "run", run_but_pip)
    patch(socket.socket, "connect", connect_here_only)
    patch(socket, "getaddrinfo", lookup_here_only)
    patch(Z, "WORK_FOLDERS", (str(CONTENT), str(MYDRIVE)))
    patch(E, "load_env", fake_env)
    patch(E, "require", made_up_token)
    patch(E, "model_cache_on_drive", cache_in_temp)
    patch(huggingface_hub, "snapshot_download", fake_snapshot)
    patch(H, "hub_api", fake_hub_api)
    patch(H, "hub_download", fake_hub_download)
    patch(SC, "score_all", score_on_cpu)
    patch(SC, "score_parallel", parallel_on_cpu)
    patch(SC, "_context", lambda: InProcess)
    patch(SC, "contextlib", OwnThreadOnly())          # the redirect of an in-process worker, as in its own process
    patch(TF, "build", never_build)
    ENC.ENCODERS.update(TINY_ENCODERS)
    sys.stdout = sys.stderr = CELL_OUT

    # ---- first run, from nothing, snow held back until the first SAR call ---------------------------
    one: dict = {"__name__": "__main__"}
    HOLD["on"] = True
    try:
        said = run_cells(ALL, one, "first run")
    finally:
        HOLD["on"] = False
        HOLD["snow"].set()
    assert CALLS["env"] == ["model_cache_on_drive", "load_env", "require HF_TOKEN", "require DATA_DRIVE_DIR",
                            "require REFERENCE_DRIVE_DIR"], CALLS["env"]
    assert (one["PREP"], one["RES"], one["REF"], one["PART3"], one["RUNS"]) == (PREP, RES, REF, PART3, RUNS)
    assert (one["DRIVE_AERIAL"], one["DRIVE_DATASETS"], one["HUB_DIR"]) == (DRIVE_AERIAL, DRIVE_DATASETS, HUB_DIR)
    assert one["DEVICE"] == "cuda" and one["HF_TOKEN"] == FAKE_TOKEN and one["DATASET_OWNER"] == FAKE_OWNER
    assert one["CHECKPOINTS"] == CHECKPOINTS and one["SPLITS"] == ("val",)
    assert sorted(one["SAVED_RUNS"]["part two"]) == sorted(PART2_RUNS) and DECOY not in one["SAVED_RUNS"]["part two"]
    assert sorted(one["SAVED_RUNS"]["part three"]) == sorted(PART3_RUNS)
    assert set(one["SAR_RUNS"]) == STUDENT_RUNS and set(one["AERIAL_RUNS"]) == AERIAL_RUNS
    assert "stand-in GPU of the dry run, 40.0 GB, free 30.0 GB" in said
    assert "saved rows of part two   8 runs in" in said and "saved rows of part three 15 runs in" in said
    # the fetcher - two lanes, the data in the order of the plan, the weights of the control and the students
    fetch = one["prefetch"]
    DATA_LANE = ["forty_v1", "glavitu_alp", "glacial_lakes", "snow", "flair_hub"]
    assert fetch.lanes() == {"data": DATA_LANE, "read-ahead": ["DINOv3 weights", "student weights"]}, fetch.lanes()
    assert "data       lane, one item after the other   forty_v1, glavitu_alp, glacial_lakes, snow, flair_hub" in said
    assert "read-ahead lane, one item after the other   DINOv3 weights, student weights" in said
    # FAST_DECODER is off - no compile cache restored, none saved
    assert "compile cache" not in fetch.names and "compile cache" not in said
    assert not (MYDRIVE / "sar-transfer-distillation" / "compile_cache.tar").exists()
    record = {name: fetch.record(name) for name in fetch.names}
    assert all(r["ok"] and r["status"] == "ready" for r in record.values()), record
    ready = [record[name]["ready_at"] for name in DATA_LANE]
    assert ready == sorted(ready), f"the data lane was not ready in its order, {ready}"
    assert [record[n]["source"] for n in DATA_LANE] == ["Hugging Face", "Drive", "Hugging Face", "Hugging Face",
                                                        "the private Hugging Face repo of DATA_HF_REPO"]
    assert record["DINOv3 weights"]["source"] == "the model cache on Drive"
    assert record["DINOv3 weights"]["bytes"] == sum((DINO_DIR / f).stat().st_size
                                                    for f in ("config.json", "model.safetensors"))
    assert record["student weights"]["bytes"] == sum(p.stat().st_size for p in CHECKPOINTS.values())
    assert record["glacial_lakes"]["bytes"] == sum((PREP / "glacial_lakes" / f).stat().st_size
                                                   for f in PR.split_files(("train", "val")))
    assert record["forty_v1"]["bytes"] == sum((PREP / "forty_v1" / f).stat().st_size
                                              for f in PR.split_files(("test", "train", "val")))
    assert record["flair_hub"]["bytes"] == sum((DRIVE_AERIAL / f"{n}.tar").stat().st_size for n in ("probe", "val"))
    # the slow item. The control ran while snow was still arriving, so it waited for forest and its weights only;
    # the first SAR call took the three datasets on the disk while snow was still arriving, snow came next
    assert not HOLD["timed out"], "snow was never let go, a cell waited for it before the first SAR call"
    assert HOLD["seen"]["control"]["snow"] in SC.BUSY, HOLD["seen"]      # not through
    assert HOLD["seen"]["control"]["forty_v1"] == "ready" and HOLD["seen"]["control"]["DINOv3 weights"] == "ready"
    assert HOLD["seen"]["first SAR call"] == (("glacial_lakes", "forty_v1", "glavitu_alp"), "running"), HOLD["seen"]
    assert record["snow"]["ready_at"] > record["glavitu_alp"]["ready_at"]
    for line in (r"^forty_v1  ready after [\d.]+ s, from Hugging Face, [\d,]+ MB in [\d.]+ s, [\d,]+ MB per second$",
                 r"^control waited [\d.]+ s for forty_v1$", r"^control waited [\d.]+ s for DINOv3 weights$",
                 r"^DINOv3 weights  ready after [\d.]+ s, from the model cache on Drive, [\d,]+ MB in [\d.]+ s, "
                 r"[\d,]+ MB per second$",
                 r"^SAR runs waited [\d.]+ s for student weights$",
                 r"^SAR runs waited [\d.]+ s, on the runtime now glacial_lakes, forty_v1, glavitu_alp$",
                 r"^SAR runs waited [\d.]+ s, on the runtime now snow$",
                 r"^aerial runs waited [\d.]+ s for student weights$", r"^aerial runs waited [\d.]+ s for flair_hub$",
                 r"^glavitu_alp  ready after [\d.]+ s, from Drive, ", r"^snow  ready after [\d.]+ s, from Hugging Face, ",
                 r"^the GPU work waited [\d.]+ s for data \(control [\d.]+ s, SAR runs [\d.]+ s, aerial runs [\d.]+ s\) "
                 r"- one after the other before the first run, as the old data cells did, the 7 items take [\d.]+ s "
                 r"\(their own times added up, timed side by side\), -?[\d.]+ s saved$"):
        assert re.search(line, said, re.M), line
    assert said.index("control waited") < said.index("control run  ") < said.index("on the runtime now snow")
    waited = fetch.waited
    assert set(waited) == {"control", "SAR runs", "aerial runs"} and waited["control"] < 30, waited
    assert abs(one["TIMES"]["waited_for_data"] - sum(waited.values())) < 1e-9
    assert record["snow"]["seconds"] > waited["control"], "the control waited as long as snow was held back"
    table = fetch.summary()
    assert list(table["item"]) == fetch.names and set(table["status"]) == {"ready"}
    # the SAR datasets - three from the Hub (the real hub.fetch), only the splits read here; one from Drive
    asked = {repo.split("/")[1]: patterns for repo, patterns, _ in CALLS["snapshot"]}
    assert list(asked) == [H.repo_name(n) for n in DATA_LANE[:4]], asked           # in the order of the lane
    assert all(workers == H.DATASET_FETCH_WORKERS == 8 for *_, workers in CALLS["snapshot"])
    assert asked["sar-transfer-glacial-lakes"] == ("manifest.json", *PR.split_files(("train", "val")))
    assert asked["sar-transfer-forty-v1"] is None, "the control dataset is fetched whole, every split"
    for name in SAR:
        assert (f"{name}  from Hugging Face in" in said) == (name != ON_DRIVE_ONLY)
        assert (PREP / name / "manifest.json").is_file() and (PREP / name / "val_images.npy").is_file()
        assert (PREP / name / "test_images.npy").is_file() == (name == "forty_v1"), "a test split was taken"
        assert not list((PREP / name).glob("*.part")) and not (PREP / name / "README.md").exists()
    assert f"{ON_DRIVE_ONLY}  not on Hugging Face (FileNotFoundError, stand-in" in said and "<token>" in said
    assert f"{ON_DRIVE_ONLY}  from Drive in" in said
    assert re.search(r"glavitu_alp: copied 6 of 6 files of the splits \['train', 'val'\], [\d,]+ MB in \d+ s, "
                     r"[\d,]+ MB per second with 4 threads, 0 were there already", said), said
    assert re.search(r"snow: 2 splits ready in .*, [\d,]+ MB in \d+ s, [\d,]+ MB per second with 8 threads", said)
    assert "NOT AVAILABLE" not in said and "not fetched" not in said and "nothing to run" not in said
    assert re.search(r"^glacial_lakes  splits \{.*\}, 64 px, units dB, free disk \d+ GB$", said, re.M)
    # the aerial sets came from the pilot repo (the stand-in), with the read token, read in place
    assert set(HUB_CALLS["downloads"]) == {"aerial/manifest.json", "aerial/probe.tar", "aerial/val.tar"}
    assert HUB_CALLS["tokens"] == [FAKE_TOKEN]
    assert "aerial sets from the private Hugging Face repo of DATA_HF_REPO, 3 files" in said
    assert re.search(r"^val    read in place from val\.tar  \(6, 4, 32, 32\), 2 departments, \d+ MB in \d+\.\d s, "
                     r"[\d,]+ MB per second$", said, re.M), said
    man = PR.load_manifest(PREP, "flair_hub")
    assert man["splits"] == {"train": 12, "val": 6} and len(man["classes"]) == 19
    # the control, the run of part two repeated and equal (the same CPU code)
    assert one["control_run"] == CONTROL_RUN and one["reference_file"] == REF / f"{CONTROL_RUN}__summary.csv"
    assert CALLS["score"][0] == (("dinov3-l-sat",), ("forty_v1",), None, None)
    assert (f"the control agrees with part two, {len(control_rows)} rows, deterministic score gaps at most "
            "0.00e+00 (allowed 0.02), decoder seed-mean gaps at most 0.00e+00 (tie rule), count gaps at most "
            "0.00e+00 (allowed 1e-06)") in said
    assert one["fine"] is True and "WARNING" not in said and "part four" in one["compared"].columns
    assert set(pd.read_csv(RES / f"{CONTROL_RUN}__summary.csv").split) == {"val", "test"}
    # the student runs, validation only, from their checkpoints; nothing of the saved rows is run
    STUDENT_FILES = {k: str(v) for k, v in CHECKPOINTS.items()}
    CONTROL_CALL = (("dinov3-l-sat",), ("forty_v1",), None, None)
    assert CALLS["score"] == [CONTROL_CALL,
                              (STUDENTS, ("glacial_lakes", "forty_v1", "glavitu_alp"), ("val",), STUDENT_FILES),
                              (STUDENTS, ("snow",), ("val",), STUDENT_FILES),
                              (STUDENTS, ("flair_hub",), ("val",), STUDENT_FILES)], CALLS["score"]
    assert set(one["rows_sar"].run) == STUDENT_RUNS and set(one["rows_aerial"].run) == AERIAL_RUNS
    assert len(one["times_sar"]) == 12 and set(one["times_sar"]["status"]) == {"done"}
    assert "FAILED" not in said and "MISSING" not in said
    # side by side - one worker per student (here in this process), a line per finished run, the gain line, logs
    assert WORKERS["started"] == [*STUDENTS] * 3, WORKERS["started"]
    assert said.count("cpu - 3 at once (given)") == 3
    assert "wave 1 - glacial_lakes, forty_v1, glavitu_alp, 3 encoders, 3 at once" in said
    assert "wave 1 - snow, 3 encoders, 3 at once" in said and "wave 1 - flair_hub, 3 encoders, 3 at once" in said
    for run in STUDENT_RUNS | AERIAL_RUNS:
        dataset, encoder = run.split("__")[0][len("dom_"):], run.split("__")[1]
        assert f"  {encoder} x {dataset} - done in " in said, run
    assert re.search(r"^12 runs in [\d.]+ min wall against [\d.]+ min of the runs added up \(each timed while it "
                     r"shared the GPU\), gain [\d.]+ x - one at a time part three took 108\.6 min for these datasets "
                     r"on an A100 with 80 GB, [\d.]+ x that$", said, re.M), "the gain line of the SAR cell"
    assert re.search(r"^3 runs in .*, gain [\d.]+ x - one at a time part three took 9\.0 min", said, re.M)
    for name in STUDENTS:
        log = (RES / "logs" / f"score__{name}.log").read_text(encoding="utf-8")
        assert f"===== dom_glacial_lakes__{name}__Llast__co_cross" in log and f"===== dom_flair_hub__{name}" in log
        assert FAKE_TOKEN not in log, f"the log of {name} holds the token"
        assert "splits ready" not in log and "read in place" not in log, "a line of the fetcher went to a log"
    assert f"0 of {len(STUDENT_RUNS)} runs without a summary on Drive" in said
    assert f"0 of {len(AERIAL_RUNS)} runs without a summary on Drive" in said
    assert {p.name.split("__summary")[0] for p in RES.glob("dom_*__summary.csv")} == STUDENT_RUNS | AERIAL_RUNS | {CONTROL_RUN}
    for run in STUDENT_RUNS | AERIAL_RUNS:
        info = json.loads((RES / f"{run}__info.json").read_text(encoding="utf-8"))
        assert info["scored_splits"] == ["val"], run
    info = json.loads((RES / "dom_flair_hub__student-n__Llast__aerial_rgbn__info.json").read_text(encoding="utf-8"))
    assert info["input_stats"]["nir"] == [120.0, 55.0]               # the checkpoint's aerial statistics
    # the table - every dataset, head and model on val, the saved rows by their own run files
    table = one["table"]
    assert set(table["split"]) == {"val"} and len(table) == 3 * (8 + 15 + 15)
    by_source = {s: set(g["encoder"]) for s, g in table.groupby("source")}
    assert by_source == {"part two": {"dinov3-l-sat", "cradio-v4-h"},
                         "part three": {"terrafm-b-s1", "student-m", "student-s", "dinov3-aerial"},
                         "part four": set(STUDENTS)}
    row = table[(table["dataset"] == "glacial_lakes") & (table["encoder"] == "dinov3-l-sat") & (table["head"] == "decoder")]
    l21 = pd.read_csv(REF / f"{SC.run_name('glacial_lakes', 'dinov3-l-sat', 21, 'vv_vh_diff')}__summary.csv")
    l21 = l21[(l21["head"] == "decoder") & (l21["split"] == "val")]
    assert len(row) == 1 and abs(float(row["mIoU"].iloc[0]) - l21["mIoU"].mean()) < 1e-12, "the decoy mixed in"
    assert int(row["seeds"].iloc[0]) == 3 and abs(float(row["mIoU_sd"].iloc[0]) - 0.004) < 1e-9
    assert set(table.loc[table["dataset"].isin(["glacial_lakes", "snow", "glavitu_alp"]), "metric"]) == {"IoU_pos"}
    assert set(table.loc[table["dataset"].isin(["forty_v1", "flair_hub"]), "metric"]) == {"mIoU"}
    pos_rows = table[table["metric"] == "IoU_pos"]
    pd.testing.assert_series_equal(pos_rows["score"], pos_rows["IoU_pos"], check_names=False)
    pd.testing.assert_series_equal(pos_rows["score_sd"], pos_rows["IoU_pos_sd"], check_names=False)
    m_rows = table[table["metric"] == "mIoU"]
    pd.testing.assert_series_equal(m_rows["score"], m_rows["mIoU"], check_names=False)
    pd.testing.assert_series_equal(m_rows["score_sd"], m_rows["mIoU_sd"], check_names=False)
    assert "part three's 45 rows agree with its own table, stage_a_table.csv" in said
    pos, side = one["pos"], one["side"]
    assert list(pos.columns) == ["dinov3-l-sat", "cradio-v4-h", "terrafm-b-s1", "dinov3-aerial", "student-m",
                                 "student-s", *STUDENTS]
    assert set(pos.index.get_level_values("head")) == {"probe", "decoder"} and pos.index.equals(side.index)
    for name in ("stage_a_table.csv", "stage_a_val_table.csv", "stage_a_val_table_pos.csv"):
        assert (RES / name).is_file(), name
    # the verdicts, written by the code, saved as JSON and as one row per compared row
    verdicts = json.loads((RES / "verdicts.json").read_text(encoding="utf-8"))
    assert list(verdicts["questions"]) == ["1", "2", "3", "4", "5"] and verdicts["split"] == "val"
    again = SC.stage_a_verdicts(table)
    for q, spec in verdicts["questions"].items():
        assert spec["answer"] == again["questions"][q]["answer"], q
        assert all(word in WORDS for word in spec["answer"].values()), spec["answer"]
    assert list(verdicts["questions"]["1"]["answer"]) == ["student-f"]
    assert [(c["model"], c["reference"]) for c in verdicts["questions"]["1"]["comparisons"]] == \
        [(s, "dinov3-l-sat") for s in STUDENTS]
    assert list(verdicts["questions"]["4"]["answer"]) == ["student-f", "student-n"]
    assert verdicts["questions"]["5"]["informational"] is True
    q1f = verdicts["questions"]["1"]["comparisons"][0]
    assert sum(q1f["decoder"].values()) == 4 and q1f["aerial_reference"] == "dinov3-aerial"
    assert [r["role"] for r in q1f["rows"]] == ["word", "tally"] * 4 + ["reported"] * 2
    assert verdicts["questions"]["4"]["comparisons"][0]["aerial"] is None, "TerraFM-B has no aerial row"
    rows_csv = pd.read_csv(RES / "stage_a_rows.csv")
    assert len(rows_csv) == 10 * 3 + 10 + 10 + 8 * 2 + 10 * 3 and set(rows_csv["outcome"]) <= {"win", "loss", "tie"}
    assert "question 1 - student F against the frozen satellite DINOv3" in said
    assert "  answer - the word of student-f against dinov3-l-sat is " in said
    assert set(one["TIMES"]) == set(STEPS)
    for path in RES.glob("*"):
        if path.suffix in (".csv", ".json"):
            assert FAKE_TOKEN not in path.read_text(encoding="utf-8"), path
    assert {p.name for p in CONTENT.iterdir()} == {"drive", "sar-transfer-distillation"}
    assert not lane_threads() and sys.stdout is CELL_OUT and sys.stderr is CELL_OUT, "the fetcher is not through"
    print("\nfirst run done, every cell ran, the control passed, the students scored val only, the verdicts written")

    # ---- second run, the same session. Everything is on disk and on Drive, nothing is run again, and nothing
    # is fetched but forest, which the control reads and which is on the disk
    before_calls = {k: list(v) for k, v in CALLS.items()}
    stamps = summaries()
    two: dict = {"__name__": "__main__"}
    said = run_cells(ALL, two, "second run")
    assert CALLS["snapshot"] == before_calls["snapshot"], "a dataset was fetched again"
    assert CALLS["score"] == before_calls["score"] + [CONTROL_CALL] and summaries() == stamps, "a run was repeated"
    assert two["prefetch"].lanes() == {"data": ["forty_v1"]} and two["SAR_TODO"] == [] and not two["AERIAL_TODO"]
    assert re.search(r"^forty_v1  ready after [\d.]+ s, on the runtime disk already$", said, re.M)
    assert said.count("every student run has its summary on Drive, not fetched") == len(SAR)   # 3 + aerial,
    assert "forty_v1  every student run has its summary on Drive, not fetched" not in said   # the control reads it
    assert said.count("every student run has its summary on Drive, nothing to run") == len(SAR) + 1
    assert said.count("skip dom_") == 1, "the control is skipped by its run step, the students before any fetch"
    assert "wave 1 -" not in said and len(WORKERS["started"]) == 3 * len(STUDENTS), "a worker started for nothing"
    assert said.count("no run was scored here") == 2
    assert re.search(r"^the GPU work waited [\d.]+ s for data \(control [\d.]+ s\) - .*, the 1 item takes ", said, re.M)
    pd.testing.assert_frame_equal(two["table"], one["table"], obj="table")
    assert json.dumps(two["verdicts"], sort_keys=True) == json.dumps(one["verdicts"], sort_keys=True)
    assert not lane_threads()
    print("\nsecond run done, every run was skipped, only forest was looked at")

    # ---- third run, a new runtime with a .env without DATASET_HF_OWNER, the pilot repo out of reach, one
    # dataset folder that holds only its manifest and the summaries of two runs gone. Only what those two runs
    # need is fetched, from Drive and from the archives, the two runs are made again, the others are skipped
    shutil.rmtree(CONTENT / "sar-transfer-distillation")
    (PREP / "snow").mkdir(parents=True)
    shutil.copyfile(HUB / "snow" / "manifest.json", PREP / "snow" / "manifest.json")
    hub_fake.fail, WRONG["env file"] = True, ENV_NO_OWNER
    downloads = len(HUB_CALLS["downloads"])
    again = {run: RES / f"{run}__summary.csv" for run in ("dom_snow__student-c__Llast__co_cross",
                                                          "dom_flair_hub__student-c__Llast__aerial_rgbn")}
    kept = {run: path.read_bytes() for run, path in again.items()}
    for path in again.values():
        path.unlink()
    three: dict = {"__name__": "__main__"}
    said = run_cells(ALL, three, "third run")
    hub_fake.fail, WRONG["env file"] = False, FAKE_ENV
    for run, path in again.items():
        assert path.read_bytes() == kept[run], f"{run} differs when it is made again"
    assert three["DATASET_OWNER"] == "" and "DATASET_HF_OWNER is not set, the SAR datasets come from their copies on Drive" in said
    assert CALLS["snapshot"] == before_calls["snapshot"], "the Hub was asked without an owner"
    assert three["prefetch"].lanes() == {"data": ["forty_v1", "snow", "flair_hub"],
                                         "read-ahead": ["student weights"]}
    assert three["SAR_TODO"] == ["snow"] and three["AERIAL_TODO"] is True
    assert "snow  incomplete on disk, taken again" in said and three["on_disk"]("snow")
    assert "forty_v1  from Drive in" in said and "snow  from Drive in" in said and "from Hugging Face in" not in said
    assert not (PREP / "glacial_lakes").exists() and not (PREP / "glavitu_alp").exists(), "a finished dataset was fetched"
    for name in ("glacial_lakes", "glavitu_alp"):
        assert f"{name}  every student run has its summary on Drive, not fetched" in said
        assert f"{name}  every student run has its summary on Drive, nothing to run" in said
    assert not (PREP / "snow" / "test_images.npy").exists() and (PREP / "forty_v1" / "test_images.npy").is_file()
    assert "aerial sets from the archives on Drive, not from Hugging Face (FileNotFoundError, stand-in" in said
    assert three["prefetch"].record("flair_hub")["source"] == "the archives on Drive"
    assert len(HUB_CALLS["downloads"]) == downloads
    assert CALLS["score"][len(before_calls["score"]) + 1:] == [
        CONTROL_CALL, (STUDENTS, ("snow",), ("val",), STUDENT_FILES),
        (STUDENTS, ("flair_hub",), ("val",), STUDENT_FILES)], CALLS["score"]
    assert said.count("skip dom_") == 1 + 2 + 2 and WORKERS["started"][9:] == ["student-c", "student-c"]
    assert "skip dom_snow__student-f__Llast__co_cross - its summary holds" in said
    assert "  student-c x snow - done in " in said and "  student-c x flair_hub - done in " in said
    pd.testing.assert_frame_equal(three["table"], one["table"], obj="table")
    stamps = summaries()
    print("\nthird run done, a new runtime fetched only what two runs needed, from Drive, and made them again")

    # ---- the control. A probe score off by 0.01 passes, off by 0.03 stops. A decoder score is judged by
    # the seed mean with the tie rule, off by 0.005 passes, off by 0.03 stops (one seed, no spread to hide
    # in). A count off by one pixel, a row lacking or the file missing must stop it
    reference = REF / f"{CONTROL_RUN}__summary.csv"
    calls_before = len(CALLS["score"])
    rows = pd.read_csv(reference)
    probe_val = rows.index[(rows["head"] == "probe") & (rows["split"] == "val")]
    decoder_val = rows.index[(rows["head"] == "decoder") & (rows["split"] == "val")]
    assert len(probe_val) == 1 and len(decoder_val) == 1
    rows.loc[probe_val, "mIoU"] += 0.01
    rows.to_csv(reference, index=False)
    said = run_cells([CONTROL], three, "probe reference off by 0.01")
    assert "the control agrees with part two" in said and abs(three["worst"]["score"] - 0.01) < 1e-6
    rows.loc[probe_val, "mIoU"] += 0.02
    rows.to_csv(reference, index=False)
    printed = must_stop(CONTROL, three, "probe reference off by 0.03", RuntimeError, "the control run differs from part two")
    assert "WARNING  THE CONTROL RUN DOES NOT MATCH" in printed and "1 numbers are off" in printed
    rows = pd.read_csv(io.BytesIO(GOOD_REFERENCE))
    rows.loc[decoder_val, "mIoU"] += 0.005
    rows.to_csv(reference, index=False)
    said = run_cells([CONTROL], three, "decoder reference off by 0.005")
    assert "decoder seed-mean gaps at most 5.00e-03 (tie rule)" in said and three["n_mean_off"] == 0
    rows.loc[decoder_val, "mIoU"] += 0.025
    rows.to_csv(reference, index=False)
    printed = must_stop(CONTROL, three, "decoder reference off by 0.03", RuntimeError, "the control run differs from part two")
    assert "1 decoder seed means are off" in printed and "numbers are off" not in printed
    rows = pd.read_csv(io.BytesIO(GOOD_REFERENCE))
    rows.loc[rows.index[0], "n_pixels"] += 1
    rows.to_csv(reference, index=False)
    printed = must_stop(CONTROL, three, "count off by one pixel", RuntimeError, "the control run differs from part two")
    assert "n_pixels" in printed.split("numbers are off")[1]
    pd.read_csv(io.BytesIO(GOOD_REFERENCE)).iloc[1:].to_csv(reference, index=False)
    assert "the rows differ" in must_stop(CONTROL, three, "reference lacks a row", RuntimeError, "differs from part two")
    reference.unlink()
    must_stop(CONTROL, three, "reference missing", AssertionError, "the control needs the saved rows of part two")
    reference.write_bytes(GOOD_REFERENCE)
    said = run_cells([CONTROL], three, "reference restored")
    assert "the control agrees with part two" in said and summaries() == stamps
    assert CALLS["score"][calls_before:] == [CONTROL_CALL] * 7, "the control ran again"
    assert said.count("control waited ") == 1 and "for forty_v1" in said and "DINOv3 weights" not in said

    # ---- the table cell with part three's own table off in one number. It must stop and name it
    own = pd.read_csv(io.BytesIO(GOOD_OWN))
    hit = own.index[(own["source"] == "part three") & (own["encoder"] == "terrafm-b-s1")
                    & (own["dataset"] == "snow") & (own["head"] == "decoder")]
    own.loc[hit, "IoU_pos"] += 0.001
    own.to_csv(PART3 / "stage_a_table.csv", index=False)
    printed = must_stop(TABLE, three, "part three's table off", RuntimeError, "differ from part three's own table")
    assert "terrafm-b-s1" in printed and "IoU_pos" in printed
    (PART3 / "stage_a_table.csv").write_bytes(GOOD_OWN)
    said = run_cells([TABLE], three, "part three's table restored")
    pd.testing.assert_frame_equal(three["table"], one["table"], obj="table")

    # ---- the prefetch and SAR cells again in this session, with one summary gone and every run patched away.
    # The fetcher of the run before is stopped first, snow is on the disk; the cell names the missing run, goes on
    gone = RES / "dom_snow__student-c__Llast__co_cross__summary.csv"
    aside = TMP / "summary_put_aside.csv"
    os.replace(gone, aside)
    since, old_fetch = len(PATCHED), three["prefetch"]
    patch(SC, "score_parallel", lambda *args, **kwargs: (pd.DataFrame(), pd.DataFrame(columns=SC.TIMES_COLUMNS)))
    said = run_cells([PREFETCH, SAR_RUNS], three, "one summary missing")
    unpatch(since)
    assert three["prefetch"] is not old_fetch and three["prefetch"].names == ["forty_v1", "snow", "student weights"]
    assert re.search(r"^snow  ready after [\d.]+ s, on the runtime disk already$", said, re.M)
    assert "1 of 12 runs without a summary on Drive" in said and "no run was scored here" in said
    assert "  MISSING dom_snow__student-c__Llast__co_cross, see FAILED or NOT AVAILABLE above" in said
    os.replace(aside, gone)
    assert summaries() == stamps
    print("\nthe SAR cell names a run without a summary and does not stop")

    # ---- the aerial cell with two runs to do and the REAL spawn context. Two processes, the same files as the
    # runs made in this process in the first run, byte for byte; the third student is skipped before any start
    redo = [f"dom_flair_hub__{e}__Llast__aerial_rgbn" for e in ("student-f", "student-n")]
    kept = {run: (RES / f"{run}__summary.csv").read_bytes() for run in redo}
    for run in redo:
        os.replace(RES / f"{run}__summary.csv", TMP / f"{run}__summary_aside.csv")
    since, started = len(PATCHED), len(WORKERS["started"])
    patch(SC, "_context", REAL["context"])
    WORKERS["n"] = 2
    try:
        said = run_cells([PREFETCH, NAMES.index("aerial-runs")], three, "two real processes")   # AERIAL_RUNS is a set here
    finally:
        unpatch(since)
        WORKERS["n"] = 3
    assert len(WORKERS["started"]) == started, "the stand-in context was used"
    assert three["prefetch"].lanes() == {"data": ["forty_v1", "flair_hub"],
                                         "read-ahead": ["student weights"]}
    assert re.search(r"^flair_hub  ready after [\d.]+ s, on the runtime disk already$", said, re.M)
    for run in redo:
        assert (RES / f"{run}__summary.csv").read_bytes() == kept[run], f"{run} differs in a real process"
        (TMP / f"{run}__summary_aside.csv").unlink()
    assert "skip dom_flair_hub__student-c__Llast__aerial_rgbn - its summary holds" in said
    assert "cpu - 2 at once (given)" in said and "wave 1 - flair_hub, 2 encoders, 2 at once" in said
    for name in ("student-f", "student-n"):
        assert f"started {name} on flair_hub" in said and f"  {name} x flair_hub - done in " in said
    assert re.search(r"^2 runs in .* part three took 6\.0 min", said, re.M) and "FAILED" not in said
    assert "0 of 3 runs without a summary on Drive" in said
    pids = set()
    for name in ("student-f", "student-n"):
        log = (RES / "logs" / f"score__{name}.log").read_text(encoding="utf-8")
        pids.add(int(re.findall(r"^===== \S+, process (\d+), datasets \['flair_hub'\]", log, re.M)[-1]))
        assert FAKE_TOKEN not in log
    assert len(pids) == 2 and os.getpid() not in pids, f"not two processes of their own, {pids}"
    print("\nthe aerial cell ran two students in two real processes, the files equal the ones made here")
    stamps = summaries()                              # the two summaries were written again, the same bytes

    # ---- a dataset on neither the Hub nor Drive. Its Hub fetch fails, it has no Drive copy, the SAR cell names
    # it and its run is MISSING, no process starts for it, the other lane goes on. The token stays hidden
    lakes_run = "dom_glacial_lakes__student-f__Llast__co_cross"
    moved = [(RES / f"{lakes_run}__summary.csv", TMP / "lakes_summary_aside.csv"),
             (HUB / "glacial_lakes", TMP / "lakes_hub_aside"),
             (DRIVE_DATASETS / H.repo_name("glacial_lakes"), TMP / "lakes_drive_aside")]
    for here, there in moved:
        os.replace(here, there)
    three["DATASET_OWNER"], calls_before, snapshots_before = FAKE_OWNER, len(CALLS["score"]), len(CALLS["snapshot"])
    try:
        said = run_cells([PREFETCH, SAR_RUNS], three, "a dataset on neither the Hub nor Drive")
    finally:
        three["DATASET_OWNER"] = ""
        for here, there in moved:
            os.replace(there, here)
    lakes = three["prefetch"].record("glacial_lakes")
    assert lakes["status"] == "failed" and not lakes["ok"] and FAKE_TOKEN not in lakes["error"]
    assert [c[0] for c in CALLS["snapshot"][snapshots_before:]] == [f"{FAKE_OWNER}/{H.repo_name('glacial_lakes')}"]
    assert "glacial_lakes  not on Hugging Face (FileNotFoundError, stand-in" in said and "<token>" in said
    assert re.search(r"^glacial_lakes  NOT AVAILABLE after [\d.]+ s, FileNotFoundError, neither on Hugging Face nor "
                     r"on Drive under .*sar-transfer-glacial-lakes$", said, re.M), said
    assert "glacial_lakes  NOT AVAILABLE, its runs are skipped" in said
    assert len(CALLS["score"]) == calls_before, "a process started for a dataset that is not there"
    assert "1 of 12 runs without a summary on Drive" in said
    assert f"  MISSING {lakes_run}, see FAILED or NOT AVAILABLE above" in said
    assert three["prefetch"].record("student weights")["ok"], "the failure stopped the other lane"
    assert summaries() == stamps
    print("\na dataset on neither the Hub nor Drive is named, its run is MISSING, the rest goes on")

    # ---- the verdicts cell on made-up rows that give every word. Lakes, snow and alpine are read by the
    # IoU of the positive class (their mIoU is the same for every model, so reading mIoU would give ties
    # only), forest by mIoU (its IoU_pos is the same for every model). The SD that does not belong to the
    # score is huge, so taking the wrong one would turn every decoder row into a tie
    DEC = {  # decoder score per model and dataset, SD 0.002 unless named in SD
        "dinov3-l-sat": {"glacial_lakes": 0.30, "snow": 0.70, "forty_v1": 0.40, "glavitu_alp": 0.55},
        "student-f": {"glacial_lakes": 0.35, "snow": 0.75, "forty_v1": 0.45, "glavitu_alp": 0.60},
        "student-n": {"glacial_lakes": 0.31, "snow": 0.73, "forty_v1": 0.43, "glavitu_alp": 0.58},
        "student-c": {"glacial_lakes": 0.40, "snow": 0.60, "forty_v1": 0.45, "glavitu_alp": 0.555},
        "terrafm-b-s1": {"glacial_lakes": 0.50, "snow": 0.80, "forty_v1": 0.50, "glavitu_alp": 0.70},
        "student-m": {"glacial_lakes": 0.20, "snow": 0.60, "forty_v1": 0.30}}          # alpine missing
    SD = {("student-n", "snow"): 0.02}
    PROBE = {"dinov3-l-sat": {"glacial_lakes": 0.28, "snow": 0.62, "forty_v1": 0.40, "glavitu_alp": 0.36},
             "student-f": {"glacial_lakes": 0.25, "snow": 0.65, "forty_v1": 0.41, "glavitu_alp": 0.38}}
    AERIAL = {"dinov3-aerial": (0.42, 0.44), "student-f": (0.30, 0.44), "student-n": (0.30, 0.30),
              "student-c": (0.45, 0.30), "student-m": (0.30, 0.30)}
    made = []
    for model, scores in DEC.items():
        for dataset, value in scores.items():
            sd = SD.get((model, dataset), 0.002)
            positive = dataset != "forty_v1"
            for head, v, s, seeds in (("decoder", value, sd, 3),
                                      ("probe", PROBE.get(model, {}).get(dataset, value - 0.05), float("nan"), 1)):
                made.append({"dataset": dataset, "encoder": model, "head": head, "split": "val", "seeds": seeds,
                             "mIoU": v if not positive else 0.5, "mIoU_sd": s if not positive else 0.3,
                             "IoU_pos": v if positive else 0.9, "IoU_pos_sd": s if positive else 0.3,
                             "accuracy": 0.8})
    for model, (decoder, probe) in AERIAL.items():
        for head, v, s, seeds in (("decoder", decoder, 0.002, 3), ("probe", probe, float("nan"), 1)):
            made.append({"dataset": "flair_hub", "encoder": model, "head": head, "split": "val", "seeds": seeds,
                         "mIoU": v, "mIoU_sd": s, "IoU_pos": float("nan"), "IoU_pos_sd": float("nan"),
                         "accuracy": 0.8})
    MADE_RES = TMP / "made_up_results"
    MADE_RES.mkdir()
    made_names = {**three, "table": pd.DataFrame(made), "RES": MADE_RES}
    said = run_cells([VERDICTS], made_names, "made-up rows")
    words = json.loads((MADE_RES / "verdicts.json").read_text(encoding="utf-8"))["questions"]
    got = {q: [(c["model"], c["reference"], c["word"], c["decoder"]["win"], c["decoder"]["loss"], c["decoder"]["tie"])
               for c in spec["comparisons"]] for q, spec in words.items()}
    assert got == {
        "1": [("student-f", "dinov3-l-sat", "beats", 4, 0, 0),           # every gap 0.05
              ("student-n", "dinov3-l-sat", "matches", 2, 0, 2),         # lakes gap 0.01 exactly, snow 0.03 < 2 x 0.02
              ("student-c", "dinov3-l-sat", "mixed", 2, 1, 1)],          # alpine gap 0.005
        "2": [("student-f", "student-c", "mixed", 2, 1, 1)],
        "3": [("student-n", "student-f", "does not reach", 0, 3, 1)],    # snow -0.02 within 2 x 0.02
        "4": [("student-f", "terrafm-b-s1", "does not reach", 0, 4, 0),
              ("student-n", "terrafm-b-s1", "does not reach", 0, 4, 0)],
        "5": [("student-f", "student-m", None, 3, 0, 0), ("student-n", "student-m", None, 3, 0, 0),
              ("student-c", "student-m", None, 2, 0, 1)]}, got
    assert {q: spec["answer"] for q, spec in words.items()} == {
        "1": {"student-f": "beats"}, "2": {"student-f": "mixed"}, "3": {"student-n": "does not reach"},
        "4": {"student-f": "does not reach", "student-n": "does not reach"},
        "5": {"student-f": None, "student-n": None, "student-c": None}}
    f1 = words["1"]["comparisons"][0]
    assert f1["probe"] == {"win": 1, "loss": 1, "tie": 2, "missing": 0}, "probe tallied, the margin 0.02 exactly is a tie"
    assert f1["aerial"] == {"decoder": "loss", "probe": "tie"}, "the aerial loss is reported and does not touch beats"
    assert words["5"]["comparisons"][0]["decoder"]["missing"] == 1
    assert words["2"]["comparisons"][0]["aerial"] == {"decoder": "loss", "probe": "win"}   # C's own aerial row
    assert "student-f against dinov3-l-sat - decoder 4 wins, 0 losses, 0 ties of 4 SAR rows, BEATS" in said
    assert "student-n against dinov3-l-sat - decoder 2 wins, 0 losses, 2 ties of 4 SAR rows, MATCHES" in said
    assert "student-c against dinov3-l-sat - decoder 2 wins, 1 loss, 1 tie of 4 SAR rows, MIXED" in said
    assert "decoder 0 wins, 4 losses, 0 ties of 4 SAR rows, DOES NOT REACH" in said
    assert "decoder 3 wins, 0 losses, 0 ties, 1 row missing of 4 SAR rows, NOT DECIDED, A COUNTED ROW IS MISSING" in said
    assert ("BEATS | probe 1 win, 1 loss, 2 ties | aerial against dinov3-aerial decoder loss, probe tie "
            "(reported, not counted)") in said
    assert "  answer - the word of student-f against dinov3-l-sat is beats" in said
    assert "  answer - the word of student-c against student-m is not decided, a counted row is missing" in said
    made_rows = pd.read_csv(MADE_RES / "stage_a_rows.csv")
    exact = made_rows[(made_rows["model"] == "student-n") & (made_rows["reference"] == "dinov3-l-sat")
                      & (made_rows["dataset"] == "glacial_lakes") & (made_rows["head"] == "decoder")]
    assert exact["outcome"].tolist() == ["tie"] and abs(float(exact["needed"].iloc[0]) - 0.01) < 1e-12
    wide = made_rows[(made_rows["model"] == "student-n") & (made_rows["reference"] == "dinov3-l-sat")
                     & (made_rows["dataset"] == "snow") & (made_rows["head"] == "decoder")]
    assert abs(float(wide["needed"].iloc[0]) - 0.04) < 1e-12 and wide["outcome"].tolist() == ["tie"]
    print ("\nthe verdicts cell gives every word of on made-up rows, and none with a row missing")

    # ---- the setup cell without a student, without a saved run of part two, and without Drive
    final_c = CHECKPOINTS["student-c"]
    kept = final_c.read_bytes()
    final_c.unlink()
    must_stop(SETUP, {"__name__": "__main__"}, "student missing", AssertionError, "run notebook 02 first")
    final_c.write_bytes(kept)
    cradio = REF / f"{SC.run_name('snow', 'cradio-v4-h', 32, 'vv_vh_diff')}__summary.csv"
    kept = cradio.read_bytes()
    cradio.unlink()
    must_stop(SETUP, {"__name__": "__main__"}, "saved run missing", AssertionError, "the saved rows of part two are not all in")
    cradio.write_bytes(kept)
    WRONG["drive not mounted"] = True
    asked = len(CALLS["env"])
    must_stop(SETUP, {"__name__": "__main__"}, "no Drive", AssertionError, "Google Drive is not mounted")
    assert CALLS["env"][asked:] == ["model_cache_on_drive"], CALLS["env"][asked:]
    WRONG["drive not mounted"] = False
    print("\nthe setup cell stops without a student, without a saved run and without Drive")

    assert not CALLS["network"], f"the dry run asked for the network, {CALLS['network']}"
    assert set(HUB_CALLS["tokens"]) == {FAKE_TOKEN}, "another token reached the pilot repo"
    assert {"part two": snapshot(REF.parent), "part three": snapshot(PART3.parent)} == SAVED_BEFORE, \
        "a cell wrote into the folders of part two or part three"
finally:
    HOLD["snow"].set()                            # a held fetch must not outlive the test
    for names in (globals().get(n) for n in ("one", "two", "three")):
        if isinstance(names, dict) and names.get("prefetch") is not None:
            names["prefetch"].close()
    sys.stdout, sys.stderr = REAL_STREAMS
    unpatch()
    ENC.ENCODERS.update(saved_encoders)
    for name in STUDENTS:
        ENC.ENCODERS.pop(name, None)
        RUN.PRE_REGISTERED_LAYERS.pop(name, None)
    for key in SETTINGS:
        os.environ.pop(key, None)
    for key, value in HAD_ENV.items():
        if value is not None:
            os.environ[key] = value

assert not lane_threads(), "a thread of a fetcher is left"
assert not PATCHED and SC.score_all is REAL["score_all"] and TF.build is REAL["build"]
assert SC.score_parallel is REAL["score_parallel"] and SC._context is REAL["context"]
assert H.hub_api is REAL["hub_api"] and H.hub_download is REAL["hub_download"]
assert huggingface_hub.snapshot_download is REAL["snapshot"] and E.load_env is REAL["load_env"]
assert subprocess.run is REAL["run"] and socket.getaddrinfo is REAL["lookup"] and not torch.cuda.is_available()
assert ENC.ENCODERS["dinov3-l-sat"].model_id == "facebook/dinov3-vitl16-pretrain-sat493m"
for key in ("HF_HOME", "HF_TOKEN", "HF_HUB_DISABLE_SYMLINKS_WARNING"):
    assert os.environ.get(key) == at_start.get(key), f"a cell changed {key}"
assert all(os.environ.get(k) == v for k, v in HAD_ENV.items()), "the settings of the fake .env stayed"
assert Path("/content").exists() == HAD_CONTENT, "a cell wrote to the folder /content"

del one, two, three, made_names
shutil.rmtree(TMP, ignore_errors=True)
print("flush cell compared with the shared flush cell and not executed")
print("test_notebook03: all checks passed")
