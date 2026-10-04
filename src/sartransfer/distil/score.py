"""Scoring of stage A - part two's run step over the students, and the verdicts.

    register_student   one trained student (checkpoint file) as an encoder named `name`
    score_all          every encoder x every prepared dataset, skip-if-done, as part two's run notebook
    score_parallel     the same runs with one process per encoder side by side on one GPU (part four)
    run_need           the estimated GPU memory and RAM of one run, from the sizes the run step makes
    gain_line          the wall time of a parallel call against its runs added up, as a line to print
    restore_compile_cache  the compile cache folder of the fast decoder path from its one archive on Drive
    save_compile_cache     that folder as one archive on Drive, at the end of a session
    score_table        the new rows next to part two's saved rows (the table of part three)
    run_name           the name part two's run step gives one dataset, encoder, layer and mode
    read_runs          the rows of exactly these runs of a results folder, aggregated as the table
    check_against      the numbers of some rows that a saved table holds otherwise
    reading_score      the score a row is read by, the IoU of the positive class or mIoU
    judge              one row, model against reference, the rule
    stage_a_word       the word of a model from its wins and losses on the counted rows
    stage_a_verdicts   questions 1 to 5, every row, the tallies, the words
    verdict_lines      the verdicts as lines to print
    verdict_rows       the verdicts as one table row per compared row

Encoders. The students (`checkpoints`, name -> file) are registered as encoders of kind
"student" and scored on their final output (`models.LAST`); "terrafm-b-s1" and
"dinov3-aerial" are in `models.encoders.ENCODERS`. The references dinov3-l-sat and
cradio-v4-h are not re-run here; their part two rows come in through `score_table` (part
three) or `read_runs` (part four, which reads part three's rows the same way).
The control (dinov3-l-sat on one dataset, mode vv_vh_diff, layer 21)
runs through `score_all` too, with the run name of part two.

Modes. A dataset with 2 channels (the SAR datasets) is read in mode "co_cross" by an
encoder that accepts 2 channels and in part two's "vv_vh_diff" by one that accepts 3; one
with 4 channels (aerial) in mode "aerial_rgbn". An encoder that accepts neither is skipped
for that dataset with one printed line.

Side by side (part four, 2026-10-04). `score_parallel` runs what `score_all` runs, with one
process per encoder (multiprocessing, start method spawn) on the same GPU. Each process gets
the encoder's spec, loads the encoder once, runs part two's run step unchanged over its
datasets and writes the run files as `score_all` does, so every number is what a run on its
own gives (the same seeds; CUDA kernels are not bit-reproducible across sessions anyway).
A run whose summary is done is skipped before any process starts. How many processes run at
once follows from the free GPU memory and RAM against `run_need` of each dataset; a dataset
that allows fewer runs in a later wave. A failing run is reported with its encoder and
dataset and the other processes go on.

Saved rows (part four). `read_runs` reads the summary files of the named runs only, so a
results folder that also holds another layer, another mode or a scene-disjoint run of the
same encoder never mixes into a row (`domain_table` groups by dataset and encoder and would
mix them). The aggregation is the one of `domain_table` (mean and SD over the seeds).

Verdicts (part four, FIXED before any score)
- A row is read by `reading_score` - lakes, alpine glaciers and snow by the IoU of the
  positive class, forest and the aerial set by mIoU, as part three's table.
- Rule per row. Decoder - a win when the model's mean over the seeds is higher by more than
  DECODER_MARGIN (0.01) AND by more than DECODER_SD_TIMES (2) x the larger seed SD of the two
  rows, a loss the same way round, otherwise a tie. A row with one seed has no SD, it counts
  as 0. Probe - a win or a loss when the gap is more than PROBE_MARGIN (0.02), otherwise a
  tie. A gap equal to its margin up to float rounding (FLOAT_SLACK) is not more than it.
- The word of a model counts its 4 SAR validation decoder rows only (role "word") - 4 wins
  (STAGE_A_BEATS) and no loss = beats; no loss and fewer wins = matches; 1 or more losses and
  more wins than losses = mixed; otherwise = does not reach. Its 4 SAR probe rows are tallied
  next to the word (role "tally") and the aerial rows are reported (role "reported"); neither
  changes the word. A model with a counted row missing gets no word (None, not decided).
- QUESTIONS holds questions 1 to 5 with the pairs compared and the models whose words answer
  them. The aerial row of a reference is its own, or the row of the same model reading the
  aerial RGB (`AERIAL_COUNTERPART`, the satellite DINOv3 -> dinov3-aerial), or none.
"""

from __future__ import annotations

import contextlib
import gc
import inspect
import multiprocessing
import os
import queue as queue_module
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ..domains.prepared import load_manifest
from ..domains.run import PATCH, PRE_REGISTERED_LAYERS, domain_table, run_domain
from ..domains.s1input import MODE_CHANNELS
from ..models import LAST
from ..models.encoders import ENCODERS, EncoderSpec, student_spec

MODES_BY_CHANNELS = {2: ("co_cross", "vv_vh_diff"), 4: ("aerial_rgbn",)}   # dataset channels -> modes tried
KEYS = ["dataset", "encoder", "head", "split"]


def register_student(name: str, checkpoint: str | Path, batch: int = 32) -> EncoderSpec:
    """Make a trained student an encoder `run_domain` knows, scored on its final output."""
    if name in ENCODERS and ENCODERS[name].kind != "student":
        raise ValueError(f"{name!r} is already the encoder {ENCODERS[name]}")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"{name}: checkpoint {checkpoint} not found")
    ENCODERS[name] = student_spec(name, checkpoint, batch=batch)
    PRE_REGISTERED_LAYERS[name] = LAST
    return ENCODERS[name]


def mode_for(spec: EncoderSpec, n_channels: int) -> str | None:
    """The s1input mode `spec` reads a dataset with `n_channels` stored channels in, None if it cannot."""
    for mode in MODES_BY_CHANNELS.get(n_channels, ()):
        if MODE_CHANNELS[mode] in spec.in_channels:
            return mode
    return None


def score_all(prepared_root: str | Path, encoders: list[str], datasets: list[str], res_dir: str | Path,
              token: str | None, checkpoints: dict[str, str | Path] | None = None, device: str = "cuda",
              splits: tuple[str, ...] | None = None, fast_decoder: bool = False,
              compile_cache: str | Path | None = None, **run_kw) -> pd.DataFrame:
    """Loop encoders x datasets, each encoder loaded once; `run_domain` skips a pair already in res_dir.

    checkpoints  student name -> checkpoint file, registered before the loop
    splits       the splits to score, passed to `run_domain`. None = every non-train split (part two),
                 a tuple keeps only the named ones (stage A scores ("val",))
    fast_decoder True = run_domain's fast decoder path (part four, the measured choice of eager or
                 compiled, same recipe), False = part two's step
    compile_cache  with fast_decoder, the folder of the compile caches (run.use_compile_cache)
    run_kw       passed to `run_domain` (seeds, steps, batch, force, ...)
    Returns the summary rows of the pairs run or skipped in this call.
    """
    run_kw = _fast_kw(run_kw, fast_decoder, compile_cache)
    for name, path in (checkpoints or {}).items():
        register_student(name, path)
    unknown = [e for e in encoders if e not in ENCODERS]
    if unknown:
        raise KeyError(f"unknown encoders {unknown}, known: {sorted(ENCODERS)}")
    channels = {d: len(load_manifest(prepared_root, d)["channels"]) for d in datasets}
    frames = []
    for encoder in encoders:
        frames += _score_encoder(prepared_root, encoder, datasets, channels, res_dir, token, device, splits,
                                 **run_kw)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fast_kw(run_kw: dict, fast_decoder: bool, compile_cache) -> dict:
    """run_kw with run_domain's fast decoder path and its cache folder when asked for, run_kw as it is otherwise."""
    if not fast_decoder:
        return run_kw
    return {**run_kw, "fast_decoder": True, "compile_cache": None if compile_cache is None else str(compile_cache)}


def _score_encoder(prepared_root, encoder: str, datasets: list[str], channels: dict, res_dir, token, device: str,
                   splits, report=None, **run_kw) -> list[pd.DataFrame]:
    """One encoder over the datasets, loaded once (the loop body of `score_all`).

    report  None in `score_all`. In a worker of `score_parallel` it gets one dict per dataset
            run (encoder, dataset, seconds, the summary frame or the error, the peak GPU memory),
            and the GPU cache is emptied after every run so the other processes can use it.
    """
    import torch

    spec = ENCODERS[encoder]
    enc_cache, frames = {}, []                           # load each encoder once, for all datasets
    on_cuda = str(device).startswith("cuda")
    try:
        for dataset in datasets:
            mode = mode_for(spec, channels[dataset])
            if mode is None:
                print(f"skip {dataset} x {encoder}: the dataset has {channels[dataset]} channels, "
                      f"the encoder reads {spec.in_channels}")
                continue
            if report is not None and on_cuda:
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            frame, error = None, None
            try:
                frame = run_domain(prepared_root, dataset, encoder, res_dir, token=token, mode=mode,
                                   device=device, enc_cache=enc_cache, splits=splits, **run_kw)
                frames.append(frame)
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"FAILED {dataset} {encoder}: {error}")
            seconds = time.perf_counter() - t0
            print(f"{dataset} x {encoder}: {seconds / 60:.0f} min\n")
            if report is not None:
                peak = torch.cuda.max_memory_reserved() / 1e9 if on_cuda else None
                gc.collect()
                if on_cuda:
                    torch.cuda.empty_cache()                 # what this run cached goes back to the other workers
                report({"kind": "run", "encoder": encoder, "dataset": dataset, "seconds": seconds,
                        "frame": frame, "error": error, "peak_gb": peak})
    finally:
        enc_cache.clear()
        gc.collect()
        if on_cuda:
            torch.cuda.empty_cache()
    return frames


# ---- part four - one process per encoder, side by side on one GPU --------------------------------

START_METHOD = "spawn"            # a CUDA context cannot be forked, spawn starts a fresh interpreter that imports
                                  # the package from sys.path (on Colab the sync cell's folder) and nothing else
CONTEXT_GB = 1.0                  # per process, the CUDA context and the cuBLAS and cuDNN workspaces (an estimate)
ENCODER_GB = 1.5                  # the encoder's weights on the GPU, a ViT-L in float32 is 1.2 GB (an estimate)
FEATURE_DIM = 1024                # the feature width the need is computed for, the widest encoder scored here
                                  # (DINOv3 ViT-L); the students are 768 wide, so the estimate errs high for them
# Bytes per pixel of a batch of the decoder (models.decoder.GridDecoder), the peak of the live tensors, measured
# 2026-10-04 on the CPU with a dispatch-mode counter on every op output - float32 1,317 (training step, forward
# and backward) and 804 (inference), bfloat16 autocast on the CPU 723 and 402. CUDA autocast keeps GroupNorm in
# float32, so the A100 sits between the two, the float32 numbers are kept.
DECODER_TRAIN_BYTES = 1320
DECODER_EVAL_BYTES = 810
DECODER_EVAL_BATCH = 32           # the batch of run.py's _decoder_conf
EXTRACT_BYTES = 32                # a ViT forward in inference, bytes per token and feature channel (an estimate)
GPU_MARGIN_GB = 2.0               # kept free on the GPU, the allocator's rounding and what the notebook holds
HOST_BASE_GB = 3.0                # RAM per process, Python, torch, the CUDA host side, small arrays (an estimate)
POLL_SECONDS = 2.0                # how often the parent looks for a worker that ended without saying so
# Minutes of one run of a student, one run at a time, measured in part three on an A100 with 80 GB (the
# total_min of its __info.json files, students M and S) - the yardstick of `gain_line`
PART3_RUN_MINUTES = {"glacial_lakes": 4.1, "snow": 8.4, "forty_v1": 2.0, "glavitu_alp": 21.7, "flair_hub": 3.0}
TIMES_COLUMNS = ["encoder", "dataset", "run", "status", "minutes", "peak_gpu_gb", "need_gpu_gb", "need_ram_gb",
                 "wave", "workers", "error"]


def run_need(prepared_root: str | Path, dataset: str, splits=None, feature_dim: int = FEATURE_DIM,
             encoder_batch: int = 32, max_train_chips: int = 4000, probe_max_patches: int = 2_000_000,
             batch: int = 8) -> dict:
    """The estimated GPU memory and RAM of one run of part two's run step on `dataset`, in GB.

    From the manifest (chip size, chips per split) and the sizes `domains.run.run_domain` makes.
    On the GPU, one after another - the encoder forward of the feature extraction; the probe's
    training patches (float16, all on the GPU); the decoder's training tiles (`TrainSet`, float16
    features plus the 8-bit image and labels of up to `max_train_chips` chips) with one training
    batch; the decoder's inference batch of 32 chips. The features of all chips stay in RAM.
    Returns {"gpu_gb", "ram_gb", "parts"} with the GPU parts in GB. An estimate - the measured
    peak of every run is printed by `score_parallel` next to it.
    """
    man = load_manifest(prepared_root, dataset)
    size = int(man["chip_size"])
    grid = size // PATCH
    per_chip = grid * grid * feature_dim * 2                       # float16 features of one chip
    counts = {str(s): int(n) for s, n in man["splits"].items()}
    evals = [s for s in counts if s != "train"] if splits is None else [str(s) for s in splits if str(s) in counts]
    n_train = counts.get("train", 0)
    n_decoder = min(n_train, max_train_chips)
    n_eval = sum(counts[s] for s in evals)
    probe_bytes = min(probe_max_patches, n_train * grid * grid) * (feature_dim * 2 + 8)
    ebatch = min(512, encoder_batch * max(1, (512 // size) ** 2))  # run.py's encoder batch
    parts = {"extract": ebatch * (grid * grid + 5) * feature_dim * EXTRACT_BYTES / 1e9,
             "probe": probe_bytes / 1e9,
             "train_tiles": n_decoder * (per_chip + 2 * size * size) / 1e9,
             "decoder_train": batch * size * size * DECODER_TRAIN_BYTES / 1e9,
             "decoder_eval": min(DECODER_EVAL_BATCH, max([counts[s] for s in evals], default=0))
             * size * size * DECODER_EVAL_BYTES / 1e9}
    gpu = CONTEXT_GB + ENCODER_GB + max(parts["extract"], parts["probe"] + max(
        parts["train_tiles"] + parts["decoder_train"], parts["decoder_eval"]))
    # RAM - the features of every split read, the probe's patch selection or the decoder's chip pick on top,
    # the labels and the 8-bit skip images
    ram = HOST_BASE_GB + ((n_train + n_eval) * per_chip + max(n_train * per_chip + probe_bytes,
                                                               2 * n_decoder * per_chip)
                          + 2 * size * size * (n_train + n_eval)) / 1e9
    return {"gpu_gb": gpu, "ram_gb": ram, "parts": parts}


def _context():
    """The multiprocessing context of the workers (a test puts an in-process stand-in here)."""
    return multiprocessing.get_context(START_METHOD)


@contextlib.contextmanager
def _main_hidden():
    """While a worker starts, the caller's main module is hidden from spawn.

    spawn imports the parent's main script again in every worker; a notebook has none, a script
    (a test) without a main guard would run again from its top. The worker needs nothing of it.
    """
    main = sys.modules.get("__main__")
    saved_file = main.__dict__.pop("__file__", None) if main is not None else None
    saved_spec = getattr(main, "__spec__", None)
    if saved_spec is not None:
        main.__spec__ = None
    try:
        yield
    finally:
        if saved_file is not None:
            main.__file__ = saved_file
        if saved_spec is not None:
            main.__spec__ = saved_spec


def _worker(encoder: str, spec: EncoderSpec, layer, datasets: list[str], channels: dict, prepared_root, res_dir,
            token, device: str, splits, run_kw: dict, log_path: str, events) -> None:
    """One process of `score_parallel` - one encoder over its datasets, the output of the run step in a log file."""
    with open(log_path, "a", encoding="utf-8", buffering=1) as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        error = None
        try:
            ENCODERS[encoder] = spec                        # the parent's spec, a student or a test's encoder too
            if layer is not None:
                PRE_REGISTERED_LAYERS[encoder] = layer
            print(f"===== {encoder}, process {os.getpid()}, datasets {datasets}, {time.strftime('%Y-%m-%d %H.%M.%S')}")
            _score_encoder(prepared_root, encoder, datasets, channels, res_dir, token, device, splits,
                           report=events.put, **run_kw)
        except BaseException as e:                          # anything outside a run, the parent names it
            error = f"{type(e).__name__}: {str(e)[:300]}"
            traceback.print_exc()
        finally:
            events.put({"kind": "end", "encoder": encoder, "error": error})


def _finished(prepared_root, dataset: str, encoder: str, mode: str, res_dir, splits, kw: dict):
    """(run name, its summary) when `run_domain` would skip the run, (run name, None) when it would run it.

    The rule of run.py - the summary holds the probe and every decoder seed on every scored split.
    The run name is None when the encoder has no layer, the worker then reports the error of run.py.
    """
    layer = kw.get("layer")
    if layer is None:
        layer = PRE_REGISTERED_LAYERS.get(encoder)
    if layer is None:
        return None, None
    sharing = tuple(kw.get("exclude_train_sharing") or ())
    run = run_name(dataset, encoder, layer, mode) + ("__disjoint_" + "_".join(sharing) if sharing else "")
    path = Path(res_dir) / f"{run}__summary.csv"
    if kw.get("force") or not path.is_file():
        return run, None
    all_splits = list(load_manifest(prepared_root, dataset)["splits"])
    evals = [s for s in all_splits if s != "train"] if splits is None else [str(s) for s in splits]
    s = pd.read_csv(path)
    want = {("probe", -1)} | {("decoder", sd) for sd in kw["seeds"]}
    if want <= set(zip(s["head"], s["seed"])) and set(evals) <= set(s["split"]):
        return run, s
    return run, None


def _hide(text, token) -> str:
    return str(text).replace(token, "<token>") if token else str(text)


def _tail(path: Path, lines: int = 15) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def score_parallel(prepared_root: str | Path, encoders: list[str], datasets: list[str], res_dir: str | Path,
                   token: str | None, checkpoints: dict[str, str | Path] | None = None, device: str = "cuda",
                   splits: tuple[str, ...] | None = None, workers: int | None = None,
                   log_dir: str | Path | None = None, fast_decoder: bool = False,
                   compile_cache: str | Path | None = None, **run_kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`score_all` with one process per encoder side by side, the same runs, files and numbers.

    checkpoints, splits, fast_decoder, compile_cache, run_kw   as `score_all`; every worker gets the
              same cache folder, so a shape one process compiled is read back by the processes after it
    workers   processes at once. None = per dataset, min(encoders with a run to do, what the
              free GPU memory allows, what the free RAM allows), each from `run_need` with
              GPU_MARGIN_GB kept free; datasets that allow fewer run in a later wave, each wave
              starts its processes again (a student loads in seconds). Off the GPU None = 1
    log_dir   where each process writes everything the run step prints, score__<encoder>.log,
              appended. None = res_dir / "logs"

    Before any process starts every pair is checked - an encoder that cannot read a dataset and a
    run whose summary is done (the skip rule of run.py) are skipped with one printed line. Each
    finished run prints one line with its minutes and, on the GPU, its peak memory next to the
    estimate. A run that fails, and a process that ends without finishing its datasets, is
    printed with its encoder and dataset and the error; the other processes go on. Stopping the
    cell stops the processes. Call it from a notebook or a script; workers never import the
    caller's main module.

    Returns (the summary rows of the runs done or skipped, in the order `score_all` gives them;
    one row per encoder and dataset with TIMES_COLUMNS - status done, skipped, cannot read,
    failed or not run, minutes, peak_gpu_gb, the estimates, the wave, its workers, the error).
    """
    import torch

    run_kw = _fast_kw(run_kw, fast_decoder, compile_cache)
    for name, path in (checkpoints or {}).items():
        register_student(name, path)
    unknown = [e for e in encoders if e not in ENCODERS]
    if unknown:
        raise KeyError(f"unknown encoders {unknown}, known: {sorted(ENCODERS)}")
    if workers is not None and workers < 1:
        raise ValueError(f"workers {workers}, at least 1")
    channels = {d: len(load_manifest(prepared_root, d)["channels"]) for d in datasets}
    kw = {k: p.default for k, p in inspect.signature(run_domain).parameters.items()
          if p.default is not inspect.Parameter.empty}
    kw.update(run_kw)
    log_dir = Path(res_dir) / "logs" if log_dir is None else Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    frames, times, pending = {}, {}, {}

    def record(encoder, dataset, **values):
        row = times.setdefault((encoder, dataset), dict.fromkeys(TIMES_COLUMNS))
        row.update(encoder=encoder, dataset=dataset, **values)

    for encoder in encoders:
        spec = ENCODERS[encoder]
        for dataset in datasets:
            mode = mode_for(spec, channels[dataset])
            if mode is None:
                print(f"skip {dataset} x {encoder} - the dataset has {channels[dataset]} channels, "
                      f"the encoder reads {spec.in_channels}")
                record(encoder, dataset, status="cannot read")
                continue
            run, done = _finished(prepared_root, dataset, encoder, mode, res_dir, splits, kw)
            record(encoder, dataset, run=run, status="not run")
            if done is not None:
                print(f"skip {run} - its summary holds the probe and the decoder seeds {list(kw['seeds'])}")
                frames[(encoder, dataset)] = done
                record(encoder, dataset, status="skipped", minutes=0.0)
                continue
            pending.setdefault(encoder, []).append(dataset)

    # the worker rule, per dataset with a run to do
    todo = [d for d in datasets if any(d in ds for ds in pending.values())]
    on_cuda = str(device).startswith("cuda")
    allowed = {}
    if todo and on_cuda:
        import psutil

        free_gpu = torch.cuda.mem_get_info()[0] / 1e9
        free_ram = psutil.virtual_memory().available / 1e9
        usable = free_gpu - GPU_MARGIN_GB
        ebatch = max(ENCODERS[e].batch for e in pending)
        for d in todo:
            n = sum(d in ds for ds in pending.values())
            need = run_need(prepared_root, d, splits, encoder_batch=ebatch, max_train_chips=kw["max_train_chips"],
                            probe_max_patches=kw["probe_max_patches"], batch=kw["batch"])
            by_gpu, by_ram = int(usable // need["gpu_gb"]), int(free_ram // need["ram_gb"])
            allowed[d] = max(1, min(n, by_gpu, by_ram)) if workers is None else min(workers, n)
            for e in pending:
                if d in pending[e]:
                    record(e, d, need_gpu_gb=need["gpu_gb"], need_ram_gb=need["ram_gb"])
            why = ("given" if workers is not None else
                   f"min({n} encoders, {by_gpu} by GPU memory, {by_ram} by RAM)")
            print(f"{d} - one run needs about {need['gpu_gb']:.1f} GB of GPU memory and {need['ram_gb']:.1f} GB "
                  f"of RAM (estimates), free {free_gpu:.1f} GB on the GPU ({GPU_MARGIN_GB:g} kept) and "
                  f"{free_ram:.1f} GB of RAM, {allowed[d]} at once = {why}")
    else:
        for d in todo:
            allowed[d] = min(workers or 1, sum(d in ds for ds in pending.values()))
        if todo:
            print(f"{device} - {allowed[todo[0]]} at once ({'given' if workers else 'no GPU, one after another'})")

    waves = {}
    for d in todo:                                       # the most at once first, datasets in the given order
        waves.setdefault(allowed[d], []).append(d)
    for number, (count, wave) in enumerate(sorted(waves.items(), key=lambda kv: -kv[0]), 1):
        jobs = [(e, [d for d in ds if d in wave]) for e, ds in pending.items()]
        jobs = [(e, ds) for e, ds in jobs if ds]
        print(f"wave {number} - {', '.join(wave)}, {len(jobs)} encoders, {count} at once, "
              f"logs in {log_dir}")
        for e, ds in jobs:
            for d in ds:
                record(e, d, wave=number, workers=count)
        _run_wave(jobs, count, prepared_root, res_dir, token, device, splits, run_kw, channels, log_dir,
                  frames, record, times)

    order = [(e, d) for e in encoders for d in datasets if (e, d) in frames]
    rows = pd.concat([frames[k] for k in order], ignore_index=True) if order else pd.DataFrame()
    table = pd.DataFrame([times[(e, d)] for e in encoders for d in datasets if (e, d) in times],
                         columns=TIMES_COLUMNS)
    return rows, table


def _run_wave(jobs, count, prepared_root, res_dir, token, device, splits, run_kw, channels, log_dir,
              frames, record, times) -> None:
    """Start up to `count` workers at once over `jobs` (encoder, datasets) and collect what they report."""
    ctx = _context()
    events = ctx.Queue()
    waiting, running = list(jobs), {}                    # encoder -> (process, datasets, datasets reported)

    def handle(event) -> None:
        encoder = event["encoder"]
        if event["kind"] == "run":
            dataset, minutes = event["dataset"], event["seconds"] / 60
            running[encoder][2].add(dataset)
            peak = event["peak_gb"]
            need = times[(encoder, dataset)]["need_gpu_gb"]
            memory = ("" if peak is None else f", peak GPU {peak:.1f} GB reserved"
                      + ("" if need is None else f" (estimate {need:.1f} GB)"))
            if event["error"] is None:
                frames[(encoder, dataset)] = event["frame"]
                record(encoder, dataset, status="done", minutes=minutes, peak_gpu_gb=peak)
                print(f"  {encoder} x {dataset} - done in {minutes:.1f} min{memory}")
            else:
                error = _hide(event["error"], token)
                record(encoder, dataset, status="failed", minutes=minutes, peak_gpu_gb=peak, error=error)
                print(f"  {encoder} x {dataset} - FAILED after {minutes:.1f} min, {error}")
            return
        proc, datasets, reported = running.pop(encoder)      # "end", the worker is through
        proc.join()
        left = [d for d in datasets if d not in reported]
        if event["error"] is not None or left:
            error = _hide(event["error"] or "the worker ended early", token)
            for d in left:
                record(encoder, d, status="failed", error=error)
            print(f"  {encoder} - FAILED outside a run, {error}, not run {left}, see {log_dir / f'score__{encoder}.log'}")

    def next_event(wait: float):
        try:
            return events.get(timeout=wait)
        except queue_module.Empty:
            return None

    try:
        while waiting or running:
            while waiting and len(running) < count:
                encoder, datasets = waiting.pop(0)
                log = log_dir / f"score__{encoder}.log"
                proc = ctx.Process(target=_worker, name=f"score {encoder}",
                                   args=(encoder, ENCODERS[encoder], PRE_REGISTERED_LAYERS.get(encoder), datasets,
                                         channels, prepared_root, res_dir, token, device, splits, run_kw, str(log),
                                         events))
                running[encoder] = (proc, datasets, set())
                with _main_hidden():
                    proc.start()
                print(f"  started {encoder} on {', '.join(datasets)}")
            event = next_event(POLL_SECONDS)
            if event is not None:
                handle(event)
                continue
            for encoder in [e for e, (p, _, _) in running.items() if not p.is_alive()]:
                while (event := next_event(0.5)) is not None:      # what it sent just before it ended
                    handle(event)
                if encoder not in running:
                    continue
                proc, datasets, reported = running.pop(encoder)
                proc.join()
                left = [d for d in datasets if d not in reported]
                error = f"its process ended with exit code {proc.exitcode}"
                for d in left:
                    record(encoder, d, status="failed", error=error)
                print(f"  {encoder} - FAILED, {error}, not run {left}, the last lines of its log")
                for line in _tail(log_dir / f"score__{encoder}.log"):
                    print(f"    | {_hide(line, token)}")
    finally:
        for encoder, (proc, datasets, reported) in running.items():   # the cell was stopped, or the parent failed
            proc.terminate()
            proc.join()
            print(f"  {encoder} stopped, not run {[d for d in datasets if d not in reported]}")


def gain_line(times: pd.DataFrame, wall_seconds: float, one_at_a_time: dict | None = None) -> str:
    """The wall time of a `score_parallel` call against its runs added up, one line, no colon.

    times          the second frame `score_parallel` returns
    one_at_a_time  minutes of one run per dataset when runs go one after another (default
                   PART3_RUN_MINUTES, measured in part three), the honest yardstick - each run's own
                   minutes are timed while it shares the GPU, so their sum is longer than alone
    """
    one_at_a_time = PART3_RUN_MINUTES if one_at_a_time is None else one_at_a_time
    done = times[times["status"] == "done"] if len(times) and "status" in times else times.iloc[0:0]
    if not len(done):
        return f"no run was scored here, {wall_seconds / 60:.1f} min wall"
    wall = wall_seconds / 60
    added = float(done["minutes"].sum())
    line = (f"{len(done)} runs in {wall:.1f} min wall against {added:.1f} min of the runs added up (each timed "
            f"while it shared the GPU), gain {added / max(wall, 1e-9):.1f} x")
    known = [one_at_a_time.get(d) for d in done["dataset"]]
    if all(m is not None for m in known):
        alone = float(sum(known))
        line += (f" - one at a time part three took {alone:.1f} min for these datasets on an A100 with 80 GB, "
                 f"{alone / max(wall, 1e-9):.1f} x that")
    return line


# ---- part four - the compile cache of the fast decoder path as one archive on Drive --------------

CACHE_SKIP_SUFFIXES = (".lock",)  # Inductor's lock files are never archived


def _cache_files(folder: Path) -> list[tuple[str, int]]:
    """(path under the folder with /, size) of every file there, sorted, lock files left out."""
    if not folder.is_dir():
        return []
    return sorted((p.relative_to(folder).as_posix(), p.stat().st_size) for p in folder.rglob("*")
                  if p.is_file() and not p.name.endswith(CACHE_SKIP_SUFFIXES))


def save_compile_cache(folder: str | Path, archive: str | Path) -> dict:
    """The compile cache folder (run.use_compile_cache) as ONE archive, tar without compression, for Drive.

    Many small files are slow on Drive, one archive is one stream. Call it when no process compiles,
    at the end of the notebook. Skipped when the archive is there and its record (<archive>.json, the
    names and sizes of the files it holds) matches the folder. Written as <archive>.part first and
    renamed, so a stopped save never leaves a cut archive. Prints one line. Returns {"status" -
    "saved", "unchanged" or "empty"; "files", "mb", "seconds"}.
    """
    import hashlib
    import json
    import tarfile

    folder, archive = Path(folder), Path(archive)
    record = archive.with_name(archive.name + ".json")
    t0 = time.perf_counter()
    files = _cache_files(folder)
    mb = sum(size for _, size in files) / 1e6
    info = {"status": "empty", "files": len(files), "mb": mb, "seconds": 0.0}
    if not files:
        print(f"compile cache folder {folder.name} holds no file, nothing saved")
        return info
    fingerprint = hashlib.sha256("".join(f"{name} {size}\n" for name, size in files).encode()).hexdigest()
    try:
        old = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    if (archive.is_file() and old.get("fingerprint") == fingerprint
            and old.get("archive_bytes") == archive.stat().st_size):
        info["status"] = "unchanged"
        print(f"compile cache unchanged - {len(files)} files, {mb:.1f} MB, {archive.name} kept")
        return info
    archive.parent.mkdir(parents=True, exist_ok=True)
    part = archive.with_name(archive.name + ".part")
    with tarfile.open(part, "w") as tar:
        for name, _ in files:
            tar.add(folder / name, arcname=name, recursive=False)
    os.replace(part, archive)
    record.write_text(json.dumps({"fingerprint": fingerprint, "files": len(files), "mb": mb,
                                  "archive_bytes": archive.stat().st_size,
                                  "saved": time.strftime("%Y-%m-%d %H.%M.%S")}, indent=1), encoding="utf-8")
    info.update(status="saved", seconds=time.perf_counter() - t0)
    print(f"compile cache saved - {len(files)} files, {mb:.1f} MB in {info['seconds']:.1f} s, "
          f"{mb / max(info['seconds'], 1e-9):.0f} MB per second, one archive {archive.name}")
    return info


def restore_compile_cache(archive: str | Path, folder: str | Path) -> dict:
    """The archive of `save_compile_cache` unpacked into the compile cache folder, at the start of a session.

    Nothing is unpacked when the folder already holds files (this session's cache is the newer one) or
    when there is no archive yet (the first compile of each shape then fills the folder). Only plain
    files under the folder are written (the tar "data" filter). Prints one line. Returns {"status" -
    "unpacked", "kept" or "none"; "files", "mb", "seconds"}.
    """
    import tarfile

    archive, folder = Path(archive), Path(folder)
    t0 = time.perf_counter()
    have = _cache_files(folder)
    if have:
        print(f"compile cache folder {folder.name} already holds {len(have)} files, {archive.name} not unpacked")
        return {"status": "kept", "files": len(have), "mb": sum(s for _, s in have) / 1e6, "seconds": 0.0}
    folder.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        print(f"no compile cache archive {archive.name} yet, the first compile of each shape fills {folder.name}")
        return {"status": "none", "files": 0, "mb": 0.0, "seconds": 0.0}
    with tarfile.open(archive, "r") as tar:
        try:
            tar.extractall(folder, filter="data")
        except TypeError:                                # a Python without extraction filters
            members = [m for m in tar.getmembers() if m.isfile() and not Path(m.name).is_absolute()
                       and ".." not in Path(m.name).parts]
            tar.extractall(folder, members=members)
    got = _cache_files(folder)
    mb, seconds = sum(s for _, s in got) / 1e6, time.perf_counter() - t0
    print(f"compile cache restored - {len(got)} files, {mb:.1f} MB from {archive.name} in {seconds:.1f} s, "
          f"{mb / max(seconds, 1e-9):.0f} MB per second")
    return {"status": "unpacked", "files": len(got), "mb": mb, "seconds": seconds}


def _aggregate(s: pd.DataFrame) -> pd.DataFrame:
    """Summary rows (one per head, seed, split) -> the rows of `domain_table`, same aggregation."""
    agg = (s.groupby(KEYS).agg(seeds=("seed", "nunique"), mIoU=("mIoU", "mean"), mIoU_sd=("mIoU", "std"),
                               accuracy=("accuracy", "mean")).reset_index())
    if "IoU_pos" in s:
        pos = s.groupby(KEYS).agg(IoU_pos=("IoU_pos", "mean"), IoU_pos_sd=("IoU_pos", "std")).reset_index()
        agg = agg.merge(pos, on=KEYS, how="left")
    return agg


def score_table(res_dir: str | Path, reference: str | Path | None = None) -> pd.DataFrame:
    """`domain_table` of res_dir, with part two's saved rows in front when `reference` is given.

    reference  a folder with part two's dom_*__summary.csv files, or one CSV file holding either
               those rows (per head, seed and split) or an already aggregated table. A missing
               path is reported and the part two rows are left out.
    The column "source" says "part two" or "part three".
    """
    new = domain_table(res_dir)
    new.insert(0, "source", "part three")
    if reference is None:
        return new
    p = Path(reference)
    if not p.exists():
        print(f"reference {p} not found, part two rows left out")
        return new
    if p.is_dir():
        old = domain_table(p)
    else:
        df = pd.read_csv(p)
        old = _aggregate(df) if "seed" in df.columns else df
    old.insert(0, "source", "part two")
    out = pd.concat([old, new], ignore_index=True)
    return out.sort_values(["dataset", "split", "encoder", "head", "source"]).reset_index(drop=True)


# ---- part four - the saved rows, the reading score, the rule per row and the words -----------------

SAR_DATASETS = ("glacial_lakes", "snow", "forty_v1", "glavitu_alp")   # the 4 rows of the word
AERIAL_DATASET = "flair_hub"                                          # reported next to them
POSITIVE_CLASS_DATASETS = ("glacial_lakes", "snow", "glavitu_alp")    # read by the IoU of the positive
                                                                      # class, the others by mIoU
HEADS = ("decoder", "probe")
OUTCOMES = ("win", "loss", "tie", "missing")
# FIXED - the rule per row and the words of stage A
DECODER_MARGIN = 0.01             # decoder, the mean higher by more than this ...
DECODER_SD_TIMES = 2.0            # ... AND by more than this many times the larger seed SD
PROBE_MARGIN = 0.02               # probe, a gap of more than this
STAGE_A_BEATS = 4                 # wins of the 4 SAR validation decoder rows, with no loss, for "beats"
FLOAT_SLACK = 1e-9                # a gap equal to its margin up to float rounding is not more than it
# the aerial row of a reference - the same model reading the aerial RGB, or None when it has none
AERIAL_COUNTERPART = {"dinov3-l-sat": "dinov3-aerial", "cradio-v4-h": None, "terrafm-b-s1": None}

# FIXED - what each question compares, (model, reference), and whose word answers it
QUESTIONS = {
    "1": {"asks": "student F against the frozen satellite DINOv3, the headline (N and C next to it)",
          "pairs": (("student-f", "dinov3-l-sat"), ("student-n", "dinov3-l-sat"), ("student-c", "dinov3-l-sat")),
          "answer": ("student-f",), "informational": False},
    "2": {"asks": "does a radar-reading teacher help, student F against student C",
          "pairs": (("student-f", "student-c"),), "answer": ("student-f",), "informational": False},
    "3": {"asks": "SigLIP2 NaFlex against SigLIP2 FixRes as the first teacher, student N against student F",
          "pairs": (("student-n", "student-f"),), "answer": ("student-n",), "informational": False},
    "4": {"asks": "students F and N against their radar teacher, TerraFM-B reading Sentinel-1 (saved rows)",
          "pairs": (("student-f", "terrafm-b-s1"), ("student-n", "terrafm-b-s1")),
          "answer": ("student-f", "student-n"), "informational": False},
    "5": {"asks": "students F, N and C against student M of the part three pilot (saved rows)",
          "pairs": (("student-f", "student-m"), ("student-n", "student-m"), ("student-c", "student-m")),
          "answer": ("student-f", "student-n", "student-c"), "informational": True},
}

RULE = {
    "decoder": (f"win when the mean over the seeds is higher by more than {DECODER_MARGIN} AND by more than "
                f"{DECODER_SD_TIMES:g} x the larger seed SD of the two rows, loss the same way round, else tie "
                "(one seed = no SD, counted as 0)"),
    "probe": f"win or loss when the gap is more than {PROBE_MARGIN}, else tie",
    "word": (f"over the {len(SAR_DATASETS)} SAR validation decoder rows - {STAGE_A_BEATS} wins and no loss = "
             f"beats; no loss and fewer than {STAGE_A_BEATS} wins = matches; 1 or more losses and more wins "
             "than losses = mixed; otherwise = does not reach. A counted row missing = not decided"),
    "read_by": (f"IoU of the positive class for {list(POSITIVE_CLASS_DATASETS)}, mIoU for the others "
                f"(forest and {AERIAL_DATASET})"),
    "not_counted": f"the probe rows are tallied, the {AERIAL_DATASET} rows reported, neither changes the word",
}


def run_name(dataset: str, encoder: str, layer, mode: str) -> str:
    """The name part two's run step gives a run, dom_<dataset>__<encoder>__L<layer>__<mode>."""
    return f"dom_{dataset}__{encoder}__L{layer}__{mode}"


def seed_mean_check(reference: pd.DataFrame, other: pd.DataFrame, tolerance: float = 0.01) -> pd.DataFrame:
    """The decoder rows of two summaries of one run, every score column and split, by the tie rule.

    The rule of notebook 03's control - per column and split the mean over the
    seeds of each side, off when the gap is larger than `tolerance` AND larger than twice the larger
    seed SD (one seed = no spread; both sides not known = no gap, one side not known = off). A score
    column is a float column both summaries hold, counts (integer columns) and the keys are left out.
    Next to it, informational, over the seeds both sides ran - the mean of other - reference per seed
    and how many of them are lower in other. Returns one row per column and split - column, split,
    reference_mean, reference_sd, other_mean, other_sd, gap, allowed, off, seeds, paired_diff,
    seeds_lower.
    """
    keys = ["head", "seed", "split"]
    a = reference[reference["head"] == "decoder"].set_index(keys).droplevel("head").sort_index()
    b = other[other["head"] == "decoder"].set_index(keys).droplevel("head").sort_index()
    if a.empty or b.empty:
        raise ValueError("both summaries need decoder rows")
    scores = [c for c in a.columns if c in b.columns and pd.api.types.is_float_dtype(a[c])
              and pd.api.types.is_float_dtype(b[c])]
    rows = []
    for split in sorted(set(a.index.get_level_values("split")) | set(b.index.get_level_values("split"))):
        sa = a[a.index.get_level_values("split") == split].droplevel("split")      # index seed
        sb = b[b.index.get_level_values("split") == split].droplevel("split")
        shared = sorted(set(sa.index) & set(sb.index))
        for c in scores:
            va, vb = sa[c].dropna(), sb[c].dropna()
            ma, mb = (float(v.mean()) if len(v) else float("nan") for v in (va, vb))
            sd_a, sd_b = (float(v.std(ddof=1)) if len(v) > 1 else 0.0 for v in (va, vb))
            gap = 0.0 if not len(va) and not len(vb) else abs(ma - mb)     # nan when one side is not known
            allowed = max(tolerance, 2 * max(sd_a, sd_b))
            diff = (sb.loc[shared, c] - sa.loc[shared, c]).dropna()
            rows.append({"column": c, "split": split, "reference_mean": ma, "reference_sd": sd_a,
                         "other_mean": mb, "other_sd": sd_b, "gap": gap, "allowed": allowed,
                         "off": bool(gap != gap or gap > allowed), "seeds": len(diff),
                         "paired_diff": float(diff.mean()) if len(diff) else float("nan"),
                         "seeds_lower": int((diff < 0).sum())})
    return pd.DataFrame(rows)


def read_runs(res_dir: str | Path, runs, splits=("val",)) -> tuple[pd.DataFrame, list[str]]:
    """The rows of exactly these runs of a results folder, and the runs without a summary there.

    res_dir  a results folder of the run step (part two's, part three's or this one)
    runs     run names (`run_name`). Only their <run>__summary.csv files are read
    splits   the splits kept, None = every split in the files
    Returns (one row per dataset, encoder, head and split with its run name, aggregated as
    `domain_table` does - seeds, mIoU, mIoU_sd, accuracy, IoU_pos, IoU_pos_sd where the task is
    binary; the names of the runs whose summary file is missing). Two runs of one dataset and
    encoder (another layer or mode) are a ValueError, they would be mixed into one row.
    """
    frames, missing = [], []
    for run in runs:
        path = Path(res_dir) / f"{run}__summary.csv"
        if not path.is_file():
            missing.append(run)
            continue
        s = pd.read_csv(path)
        frames.append(s if splits is None else s[s["split"].isin([str(x) for x in splits])])
    if not frames:
        return pd.DataFrame(columns=["run", *KEYS, "seeds", "mIoU", "mIoU_sd", "accuracy"]), missing
    s = pd.concat(frames, ignore_index=True)
    per_pair = s.groupby(["dataset", "encoder"])["run"].nunique()
    if (per_pair > 1).any():
        raise ValueError(f"more than one run of {list(per_pair[per_pair > 1].index)}, they would be mixed")
    agg = _aggregate(s)
    names = s.groupby(KEYS)["run"].first().reset_index()
    out = names.merge(agg, on=KEYS, how="right")
    return out.sort_values(KEYS).reset_index(drop=True), missing


def check_against(rows: pd.DataFrame, table: pd.DataFrame,
                  columns=("seeds", "mIoU", "mIoU_sd", "IoU_pos", "IoU_pos_sd"),
                  tolerance: float = 1e-9) -> pd.DataFrame:
    """The numbers of `rows` that `table` holds otherwise, one line per number. Empty = the same.

    Both have one row per dataset, encoder, head and split (`read_runs`, `domain_table`).
    A number differs when the two are further apart than `tolerance`, or one is nan and the
    other not. A row of `rows` that `table` lacks is one line with the column "row". `table`
    may hold more rows, they are not looked at. Duplicate keys in `table` are a ValueError.
    """
    t = table.set_index(KEYS)
    if not t.index.is_unique:
        raise ValueError("the saved table holds a dataset, encoder, head and split twice")
    out = []
    for _, r in rows.iterrows():
        key = tuple(r[k] for k in KEYS)
        if key not in t.index:
            out.append({**dict(zip(KEYS, key)), "column": "row", "here": np.nan, "saved": np.nan})
            continue
        for c in columns:
            if c not in rows.columns:
                continue
            a = float(r[c])
            b = float(t.loc[key, c]) if c in t.columns else np.nan
            same = (a != a and b != b) or (a == a and b == b and abs(a - b) <= tolerance)
            if not same:
                out.append({**dict(zip(KEYS, key)), "column": c, "here": a, "saved": b})
    return pd.DataFrame(out, columns=[*KEYS, "column", "here", "saved"])


def reading_score(table: pd.DataFrame, positive=POSITIVE_CLASS_DATASETS) -> pd.DataFrame:
    """The table with "metric", "score" and "score_sd" - the IoU of the positive class (and its SD
    over the seeds) for the datasets in `positive`, mIoU (and its SD) for the others."""
    out = table.copy()
    pos = out["dataset"].isin(list(positive)).to_numpy()
    none = pd.Series(np.nan, index=out.index)
    out["metric"] = np.where(pos, "IoU_pos", "mIoU")
    out["score"] = np.where(pos, out.get("IoU_pos", none), out["mIoU"]).astype(float)
    out["score_sd"] = np.where(pos, out.get("IoU_pos_sd", none), out.get("mIoU_sd", none)).astype(float)
    return out


def _missing(x) -> bool:
    return x is None or x != x


def judge(head: str, mean, sd, ref_mean, ref_sd) -> tuple[str, float, float]:
    """One row, the model against the reference, by the rule.

    Returns (outcome, gap, needed). outcome is "win", "loss", "tie", or "missing" when either
    mean is missing; gap = mean - ref_mean; needed = the margin the gap must pass, for the
    decoder the larger of DECODER_MARGIN and DECODER_SD_TIMES x the larger seed SD (a missing
    SD, one seed, counts as 0), for the probe PROBE_MARGIN.
    """
    if head not in HEADS:
        raise ValueError(f"head {head!r}, the rule has {HEADS}")
    if _missing(mean) or _missing(ref_mean):
        return "missing", float("nan"), float("nan")
    gap = float(mean) - float(ref_mean)
    if head == "decoder":
        spread = max(0.0 if _missing(sd) else float(sd), 0.0 if _missing(ref_sd) else float(ref_sd))
        needed = max(DECODER_MARGIN, DECODER_SD_TIMES * spread)
    else:
        needed = PROBE_MARGIN
    if gap > needed + FLOAT_SLACK:
        return "win", gap, needed
    if -gap > needed + FLOAT_SLACK:
        return "loss", gap, needed
    return "tie", gap, needed


def stage_a_word(wins: int, losses: int, beats_at: int = STAGE_A_BEATS) -> str:
    """The word from the wins and losses on the counted rows."""
    if losses == 0:
        return "beats" if wins >= beats_at else "matches"
    return "mixed" if wins > losses else "does not reach"


def _value(scored: pd.DataFrame, dataset: str, encoder: str, head: str, split: str) -> tuple[float, float, int]:
    """(score, score_sd, seeds) of one row of a `reading_score` table, (nan, nan, 0) when it has none."""
    hit = scored[(scored["dataset"] == dataset) & (scored["encoder"] == encoder) & (scored["head"] == head)
                 & (scored["split"] == split)]
    if len(hit) > 1:
        raise ValueError(f"{len(hit)} rows of {dataset} {encoder} {head} {split}, the table must hold one")
    if not len(hit):
        return float("nan"), float("nan"), 0
    r = hit.iloc[0]
    return float(r["score"]), float(r["score_sd"]), 0 if _missing(r.get("seeds")) else int(r["seeds"])


def compare(table: pd.DataFrame, model: str, reference: str, split: str = "val",
            sar=SAR_DATASETS, aerial: str | None = AERIAL_DATASET) -> dict:
    """Every row of `model` against `reference`, the tallies and the word.

    The SAR decoder rows give the word (role "word"), the SAR probe rows are tallied ("tally"),
    the aerial rows are judged by the same rule and reported ("reported") against the aerial
    counterpart of the reference (`AERIAL_COUNTERPART`); none = no aerial row.
    """
    scored = table if "score" in table.columns else reading_score(table)
    aerial_ref = AERIAL_COUNTERPART.get(reference, reference) if aerial else None
    rows = []
    for dataset in (*sar, *([aerial] if aerial_ref else [])):
        ref = reference if dataset in sar else aerial_ref
        for head in HEADS:
            a, a_sd, a_n = _value(scored, dataset, model, head, split)
            b, b_sd, b_n = _value(scored, dataset, ref, head, split)
            outcome, gap, needed = judge(head, a, a_sd, b, b_sd)
            role = ("word" if head == "decoder" else "tally") if dataset in sar else "reported"
            rows.append({"dataset": dataset, "head": head, "role": role,
                         "metric": "IoU_pos" if dataset in POSITIVE_CLASS_DATASETS else "mIoU",
                         "model": model, "model_score": a, "model_sd": a_sd, "model_seeds": a_n,
                         "reference": ref, "reference_score": b, "reference_sd": b_sd, "reference_seeds": b_n,
                         "gap": gap, "needed": needed, "outcome": outcome})
    tally = {head: {o: sum(1 for r in rows if r["head"] == head and r["role"] != "reported" and r["outcome"] == o)
                    for o in OUTCOMES} for head in HEADS}
    dec = tally["decoder"]
    word = stage_a_word(dec["win"], dec["loss"]) if dec["missing"] == 0 else None
    reported = {r["head"]: r["outcome"] for r in rows if r["role"] == "reported"} or None
    return {"model": model, "reference": reference, "aerial_reference": aerial_ref, "word": word,
            "decoder": dec, "probe": tally["probe"], "aerial": reported, "rows": rows}


def stage_a_verdicts(table: pd.DataFrame, questions: dict | None = None, split: str = "val") -> dict:
    """Questions 1 to 5 (`QUESTIONS`) on the rows of `table` (one row per dataset, encoder, head
    and split, `read_runs`). Per question what it asks, every comparison (`compare`) and the
    answer, the words of the models named in its "answer". Everything is plain JSON."""
    questions = QUESTIONS if questions is None else questions
    scored = reading_score(table)
    out = {"split": split, "rule": dict(RULE), "questions": {}}
    for q, spec in questions.items():
        comparisons = [compare(scored, model, ref, split) for model, ref in spec["pairs"]]
        out["questions"][str(q)] = {
            "asks": spec["asks"], "informational": bool(spec.get("informational", False)),
            "comparisons": comparisons,
            "answer": {c["model"]: c["word"] for c in comparisons if c["model"] in spec["answer"]}}
    return out


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _tally(t: dict) -> str:
    text = ", ".join(_count(t[k], k, plural) for k, plural in (("win", "wins"), ("loss", "losses"), ("tie", "ties")))
    return text + (f", {_count(t['missing'], 'row', 'rows')} missing" if t["missing"] else "")


def verdict_lines(verdicts: dict) -> list[str]:
    """The verdicts as lines to print, one block per question, no colon in the text."""
    lines = []
    for q, spec in verdicts["questions"].items():
        lines.append(f"question {q} - {spec['asks']}" + (", informational" if spec["informational"] else ""))
        for c in spec["comparisons"]:
            n = sum(c["decoder"].values())
            word = c["word"] or "not decided, a counted row is missing"
            aerial = ("no aerial row" if not c["aerial"] else
                      f"aerial against {c['aerial_reference']} decoder {c['aerial'].get('decoder')}, probe "
                      f"{c['aerial'].get('probe')} (reported, not counted)")
            lines.append(f"  {c['model']} against {c['reference']} - decoder {_tally(c['decoder'])} of {n} SAR "
                         f"rows, {word.upper()} | probe {_tally(c['probe'])} | {aerial}")
        for model, word in spec["answer"].items():
            ref = next(c["reference"] for c in spec["comparisons"] if c["model"] == model)
            lines.append(f"  answer - the word of {model} against {ref} is "
                         f"{word or 'not decided, a counted row is missing'}")
    return lines


def verdict_rows(verdicts: dict) -> pd.DataFrame:
    """Every compared row of every question, one table row each, with the word of its comparison."""
    rows = [{"question": q, "word": c["word"], **r} for q, spec in verdicts["questions"].items()
            for c in spec["comparisons"] for r in c["rows"]]
    return pd.DataFrame(rows)


# ---- part four - every download started at the start, each consumer waits for its own item ---------
#
#   Prefetch      items fetched or read in the background from the start, one thread per lane, the items
#                 of a lane one after the other; a consumer waits for its own items only
#   read_ahead    files read once from start to end, one thread per file, into the page cache
#   model_files   where the files AutoModel.from_pretrained(model_id) reads are on this disk, no network
#
# Notebook 03 (2026-10-04) starts one Prefetch right after its setup cell - a data lane (the SAR datasets
# and the aerial sets, one at a time) and a read-ahead lane (the weight files of the control and of the
# students over the Drive mount). The control, the SAR runs and the aerial runs each wait for their own
# items, so the data arrives while the GPU works instead of before it.

# Threads of `read_ahead`, one per file. A read over the Drive mount is bound per stream and by the latency of
# each request, not by the bytes - an earlier run read its weight files one after the other at 23 to 65 MB/s each
# (2026-10-03), so files read side by side add up, and a second thread on one file is not the plan. 5 at most,
# as teachers.WARM_WORKERS; notebook 03 reads 2 files (the DINOv3 config and weights) and 3 (the students) at
# once. Not measured on their own, the item line of `Prefetch` prints the MB per second
READ_AHEAD_WORKERS = 5
READ_BLOCK = 64 << 20             # bytes per read call, 64 MB per thread stays small next to the RAM
PREFETCH_POLL = 1.0               # seconds between two looks of a waiting consumer, an interrupt gets through
BUSY = ("waiting", "running")     # the states of an item that is not through yet


def model_files(model_id: str, names=("config.json", "model.safetensors"), cache_dir=None) -> list[Path]:
    """The files `AutoModel.from_pretrained(model_id)` reads, where they are on this disk, without the network.

    model_id   a local folder (its files) or a Hugging Face repository id (its files in the Hugging Face
               cache at the revision main, the one from_pretrained takes without a revision)
    cache_dir  the hub cache folder, None = the one huggingface_hub uses
    A file that is not there is left out, so an empty list means nothing to read ahead.
    """
    folder = Path(model_id)
    if folder.is_dir():
        return [folder / name for name in names if (folder / name).is_file()]
    from huggingface_hub import try_to_load_from_cache

    found = []
    for name in names:
        path = try_to_load_from_cache(model_id, name, cache_dir=None if cache_dir is None else str(cache_dir))
        if isinstance(path, str) and os.path.isfile(path):
            found.append(Path(path))
    return found


def read_ahead(paths, workers: int = READ_AHEAD_WORKERS, block: int = READ_BLOCK) -> dict:
    """Read files once from start to end so that whoever reads them next finds the bytes in the page cache.

    One thread per file, at most `workers` at once (READ_AHEAD_WORKERS, the reason is written there), plain
    reads in blocks of `block` bytes into one buffer per thread, nothing is kept. Returns {"bytes", "seconds",
    "files"}. A file that is missing or cannot be read raises its OSError once the other files are read; no
    file at all is a FileNotFoundError.
    """
    from concurrent.futures import ThreadPoolExecutor

    paths = [Path(p) for p in paths]
    if not paths:
        raise FileNotFoundError("no file to read ahead")

    def one(path: Path) -> int:
        buffer = memoryview(bytearray(max(1, min(int(block), path.stat().st_size))))
        total = 0
        with open(path, "rb", buffering=0) as f:
            while n := f.readinto(buffer):
                total += n
        return total

    clock = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(paths)))) as pool:
        jobs = [pool.submit(one, p) for p in paths]
    sizes = [job.result() for job in jobs]                 # the first error, after every file had its turn
    return {"bytes": sum(sizes), "seconds": time.perf_counter() - clock, "files": len(paths)}


class _ThreadLines:
    """sys.stdout or sys.stderr while a `Prefetch` runs. What one of its lane threads writes goes to the
    lines of the item that thread works on, everything else to the stream that was there before."""

    def __init__(self, base, sinks: dict):
        import threading

        self.base, self.sinks, self._ident = base, sinks, threading.get_ident   # sinks, thread id -> texts

    def write(self, text):
        sink = self.sinks.get(self._ident())
        if sink is None:
            return self.base.write(text)
        sink.append(str(text))
        return len(text)

    def flush(self):
        if self._ident() not in self.sinks:
            self.base.flush()

    def __getattr__(self, name):                       # encoding, isatty, fileno and the rest, as the stream has them
        return getattr(self.base, name)


class Prefetch:
    """Items fetched or read in the background from the start, each consumer waits for its own items only.

        fetch = Prefetch(secrets=(token,))
        fetch.add("forty_v1", job, lane="data")       job() runs in the thread of its lane and returns a
                                                      dict with "source" and "bytes" (both optional), or raises
        fetch.start()                                 one thread per lane, the items of a lane one after the
                                                      other in the order they were added
        fetch.wait("forty_v1", by="control")          blocks until that item is through, returns its record
        fetch.first_ready(names, by="SAR runs")       blocks until one of them is through, returns every one
                                                      of them that is, in the order of `names`
        fetch.close()                                 no item starts any more, the running ones end first

    Lanes. Items that share a pipe go into one lane, so the one needed first gets the whole bandwidth
    instead of sharing it with the next ones; items on another pipe go into another lane and run at the same
    time. A job may run threads of its own.
    Output. What a job prints, to stdout or stderr, is kept with its item and not printed into whatever cell
    runs at that moment (sys.stdout and sys.stderr are wrapped from `start` until the last lane ends, the
    other threads write through; code that swaps sys.stdout itself while a lane runs, contextlib's
    redirect_stdout, takes the lines of the lanes for that time - the cells of notebook 03 do not, the
    workers of score_parallel redirect in their own processes). The first wait that reaches an item prints
    those lines, then one line with
    the seconds from the start to when it was ready, its source, its MB in its own seconds and the MB per
    second, or NOT AVAILABLE with the error. Every wait prints one line with the seconds it blocked, and
    `waited` adds them up by consumer.
    Errors. A job that raises fails its own item, the next items of its lane go on. The error is kept as
    text, every one of `secrets` replaced by <token>, as in every line printed here.
    `record(name)` is a copy of the record of an item - name, lane, status (waiting, running, ready, failed,
    stopped), ok, source, bytes, seconds (its own), mb_s, started and ready_at (seconds since `start`),
    error. `summary()` gives one row per item, `waited_line()` the waited seconds against the seconds of the
    items added up, the time they took one after the other.
    """

    def __init__(self, secrets=()):
        import threading

        self.secrets = tuple(s for s in secrets if s)
        self.waited: dict[str, float] = {}
        self._items: dict[str, dict] = {}
        self._lanes: dict[str, list[str]] = {}
        self._jobs: dict[str, object] = {}
        self._cond = threading.Condition()
        self._threads: list = []
        self._sinks: dict[int, list] = {}
        self._streams = None
        self._live = 0
        self._stopping = False
        self._clock = None

    def __contains__(self, name) -> bool:
        return name in self._items

    @property
    def names(self) -> list[str]:
        return list(self._items)

    def lanes(self) -> dict[str, list[str]]:
        """The item names of every lane, in the order they run."""
        return {lane: list(names) for lane, names in self._lanes.items()}

    def add(self, name: str, job, lane: str = "data") -> None:
        if self._clock is not None:
            raise RuntimeError("the fetch has started, every item is added before start()")
        if name in self._items:
            raise ValueError(f"{name} is added twice")
        self._jobs[name] = job
        self._items[name] = {"name": name, "lane": lane, "status": "waiting", "ok": False, "source": None,
                             "bytes": 0, "seconds": None, "mb_s": None, "started": None, "ready_at": None,
                             "error": None, "out": [], "shown": False}
        self._lanes.setdefault(lane, []).append(name)

    def start(self) -> "Prefetch":
        import threading

        if self._clock is not None:
            raise RuntimeError("the fetch was started already")
        self._clock = time.perf_counter()
        self._streams = (_ThreadLines(sys.stdout, self._sinks), _ThreadLines(sys.stderr, self._sinks))
        sys.stdout, sys.stderr = self._streams
        self._threads = [threading.Thread(target=self._run, args=(names,), name=f"prefetch {lane}", daemon=True)
                         for lane, names in self._lanes.items()]
        self._live = len(self._threads)
        for thread in self._threads:
            thread.start()
        if not self._threads:
            self._give_back()
        return self

    def _hide(self, text) -> str:
        out = str(text)
        for secret in self.secrets:
            out = out.replace(secret, "<token>")
        return out

    def _give_back(self) -> None:
        """The streams of before, when they are still the wrapped ones."""
        if self._streams is None:
            return
        out, err = self._streams
        if sys.stdout is out:
            sys.stdout = out.base
        if sys.stderr is err:
            sys.stderr = err.base

    def _run(self, names: list[str]) -> None:
        import threading

        me = threading.get_ident()
        try:
            for name in names:
                item = self._items[name]
                with self._cond:
                    if self._stopping:
                        item.update(status="stopped", error="the fetch was stopped before this item started")
                        self._cond.notify_all()
                        continue
                    item.update(status="running", started=time.perf_counter() - self._clock)
                self._sinks[me] = item["out"]
                begin = time.perf_counter()
                result, error = {}, None
                try:
                    got = self._jobs[name]()
                    result = got if isinstance(got, dict) else {}
                except BaseException as e:                     # this item fails, the lane goes on
                    error = f"{type(e).__name__}, {e}"
                finally:
                    self._sinks.pop(me, None)
                seconds = time.perf_counter() - begin
                with self._cond:
                    item.update(seconds=seconds, ready_at=time.perf_counter() - self._clock)
                    if error is None:
                        size = int(result.get("bytes") or 0)
                        item.update(status="ready", ok=True, source=result.get("source"), bytes=size,
                                    mb_s=size / 1e6 / max(seconds, 1e-9) if size else None)
                    else:
                        item.update(status="failed", error=self._hide(error))
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._live -= 1
                if self._live == 0:
                    self._give_back()
                self._cond.notify_all()

    def _item(self, name: str) -> dict:
        if name not in self._items:
            raise KeyError(f"{name} is not an item of this fetch, the items are {self.names}")
        if self._clock is None:
            raise RuntimeError("the fetch has not started, call start() first")
        return self._items[name]

    def record(self, name: str) -> dict:
        """A copy of the record of an item (see the class)."""
        with self._cond:
            return {k: v for k, v in self._items[name].items() if k not in ("out", "shown")}

    def item_line(self, name: str) -> str:
        """One line about an item, when it was ready and how fast, or why it is not there."""
        r = self.record(name)
        if r["status"] == "ready":
            if r["bytes"]:
                return (f"{name}  ready after {r['ready_at']:.1f} s, from {r['source']}, {r['bytes'] / 1e6:,.0f} MB "
                        f"in {r['seconds']:.1f} s, {r['mb_s']:,.0f} MB per second")
            return f"{name}  ready after {r['ready_at']:.1f} s, {r['source'] or 'nothing fetched'}"
        if r["status"] in BUSY:
            return f"{name}  {r['status']}"
        after = "" if r["seconds"] is None else f" after {r['seconds']:.1f} s"
        return f"{name}  NOT AVAILABLE{after}, {str(r['error'])[:300]}"

    def _show(self, item: dict) -> None:
        if item["shown"]:
            return
        item["shown"] = True
        for line in "".join(item["out"]).split("\n"):
            line = line.rsplit("\r", 1)[-1]                   # a progress bar, its last state
            if line.strip():
                print(self._hide(line))
        print(self.item_line(item["name"]))

    def _block(self, items: list[dict]) -> float:
        begin = time.perf_counter()
        with self._cond:
            while all(item["status"] in BUSY for item in items):
                self._cond.wait(PREFETCH_POLL)
        return time.perf_counter() - begin

    def wait(self, name: str, by: str) -> dict:
        """Block until `name` is through, print its lines once and the seconds `by` waited, return its record."""
        item = self._item(name)
        waited = self._block([item])
        self.waited[by] = self.waited.get(by, 0.0) + waited
        self._show(item)
        print(f"{by} waited {waited:.1f} s for {name}")
        return self.record(name)

    def first_ready(self, names, by: str) -> list[str]:
        """Block until one of `names` is through, print the lines of every one that is and the seconds `by`
        waited, return those names in the order of `names`."""
        items = [self._item(name) for name in names]
        if not items:
            raise ValueError("no item to wait for")
        waited = self._block(items)
        self.waited[by] = self.waited.get(by, 0.0) + waited
        with self._cond:
            ready = [item["name"] for item in items if item["status"] not in BUSY]
        for name in ready:
            self._show(self._items[name])
        print(f"{by} waited {waited:.1f} s, on the runtime now {', '.join(ready)}")
        return ready

    def close(self) -> None:
        """No item starts any more (those left are "stopped"), the running ones end first, the streams of
        before come back. A second close does nothing more."""
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        for thread in self._threads:
            thread.join()
        self._give_back()

    def summary(self) -> pd.DataFrame:
        """One row per item in the order they were added - lane, status, source, MB, its own seconds, MB per
        second, seconds from the start to its start and to when it was ready, the error."""
        rows = []
        for name in self.names:
            r = self.record(name)
            rows.append({"item": name, "lane": r["lane"], "status": r["status"], "source": r["source"],
                         "MB": r["bytes"] / 1e6, "seconds": r["seconds"], "MB_per_s": r["mb_s"],
                         "started_s": r["started"], "ready_s": r["ready_at"], "error": r["error"]})
        return pd.DataFrame(rows, columns=["item", "lane", "status", "source", "MB", "seconds", "MB_per_s",
                                           "started_s", "ready_s", "error"])

    def waited_line(self, who: str = "the GPU work") -> str:
        """The seconds the consumers waited against the seconds of the items added up, one line, no colon."""
        waited = sum(self.waited.values())
        parts = ", ".join(f"{by} {seconds:.1f} s" for by, seconds in self.waited.items()) or "no consumer"
        done = [self.record(name) for name in self.names]
        own = sum(r["seconds"] for r in done if r["seconds"] is not None)
        return (f"{who} waited {waited:.1f} s for data ({parts}) - one after the other before the first run, "
                f"as the old data cells did, the {len(done)} {'item takes' if len(done) == 1 else 'items take'} "
                f"{own:.1f} s (their own times added up, "
                f"timed side by side), {own - waited:.1f} s saved")
