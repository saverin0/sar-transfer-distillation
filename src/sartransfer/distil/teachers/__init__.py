"""The frozen teachers and one function to build them.

    build_teacher("dinov3-l-sat", device="cuda", token=...)       real weights
    build_teacher("terrafm-b", device="cpu", tiny=True)           small random model, for tests
    build_teacher("siglip2-so400m-naflex", device="cuda")         part four, see TEACHERS
    warm_weights(names)                                           the weight files read once ahead

Every teacher module offers the same function:

    build(*, rule=None, device="cuda", dtype=torch.bfloat16, token=None, weights_dir=None,
          random_init=False, tiny=False) -> Teacher

    rule          input scaling rule, None = the module's DEFAULT_RULE
    weights_dir   folder that already holds the weights (no download), else the HF cache
    random_init   the real architecture with random weights, no download (timing, dry runs)
    tiny          a very small model of the same kind with random weights (offline tests)

and weight_files(**fixed) -> ((repository, pinned revision, file name), ...), the files its
build reads from the Hugging Face cache, which `warm_weights` reads ahead.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module
from typing import Iterable

from .base import Teacher, TeacherOut, band_index

# name -> (module, fixed keyword arguments of its build). Part four - the
# two SigLIP2 variants share one module, TerraFM-B reading Sentinel-1 shares the TerraFM module.
TEACHERS = {
    "dinov3-l-sat": ("dinov3_sat", {}),
    "terrafm-b": ("terrafm", {}),
    "terrafm-b-radar": ("terrafm", {"variant": "radar"}),
    "siglip2-so400m-256": ("siglip2", {"variant": "fixres"}),
    "siglip2-so400m-naflex": ("siglip2", {"variant": "naflex"}),
    "pe-spatial-l": ("pe_spatial", {}),
}

# Threads of `warm_weights`, one per weight file. A read over the Drive mount is bound per stream
# and by the latency of each request, not by the bytes - the builds of an earlier run read their
# files one after the other at 23 to 65 MB/s each (2026-10-03, A100 runtime, 279 s for the five
# teachers, about 11 GB). So files read side by side add up, while a second thread on one file
# is not the plan and a thread without a file of its own does nothing. 5 = the five teachers of
# part four, one big file each (their small config files ride along). Not measured yet,
# warm_weights prints the MB per second of every file and the notebook the total
WARM_WORKERS = 5
# Bytes per read call of `warm_weights`. Large reads are few calls through the Drive mount; 64 MB
# per thread stays small next to the RAM of the runtime, a smaller file gets a buffer of its size
WARM_BLOCK = 64 << 20
MB = 1e6


def _module(name: str):
    if name not in TEACHERS:
        raise KeyError(f"unknown teacher {name!r}, known: {sorted(TEACHERS)}")
    module, fixed = TEACHERS[name]
    return import_module(f"{__name__}.{module}"), fixed


def build_teacher(name: str, **kwargs) -> Teacher:
    module, fixed = _module(name)
    return module.build(**fixed, **kwargs)


def weight_files(name: str) -> tuple[tuple[str, str, str], ...]:
    """(repository, pinned revision, file name) of every file the build of teacher `name` reads
    from the Hugging Face cache, as its module lists them."""
    module, fixed = _module(name)
    return tuple(tuple(entry) for entry in module.weight_files(**fixed))


def _read_through(path: str | os.PathLike, block: int = WARM_BLOCK) -> int:
    """Read a file once from the start to the end in blocks of `block` bytes, plain reads into
    one buffer. Returns the bytes read."""
    size = os.path.getsize(path)
    buffer = memoryview(bytearray(max(1, min(int(block), size))))
    total = 0
    with open(path, "rb", buffering=0) as f:
        while n := f.readinto(buffer):
            total += n
    return total


def warm_weights(names: Iterable[str], workers: int = WARM_WORKERS, cache_dir: str | os.PathLike | None = None,
                 block: int = WARM_BLOCK) -> dict:
    """Read the weight files of the teachers `names` once, side by side, before they are built.

    Each file is looked up in the local Hugging Face cache at the pinned revision and file name
    of its module (`weight_files`) with huggingface_hub.try_to_load_from_cache, which never uses
    the network. A file that is not in the cache is reported and left alone, its build fetches
    it as before. A file two teachers share is read once. The files are read through in blocks
    of `block` bytes by plain reads, one thread per file, at most `workers` at once
    (`WARM_WORKERS`, the reason is written there), the largest first so the longest read starts
    at once. Nothing of torch or transformers runs in the threads. Afterwards the builds find
    the bytes in the page cache instead of reading them over the Drive mount one file after the
    other. A file that cannot be read (OSError) is reported, the build reads it itself.

    cache_dir   the hub cache folder, None = the one huggingface_hub uses (HF_HOME/hub)

    Prints one line per file. Returns a dict with
        files    one dict per file that was read, largest first - teachers, repo, revision,
                 file, path, bytes, seconds, mb_s
        missing  one dict per file that is not in the cache - teachers, repo, revision, file
        failed   one dict per file whose read stopped - the same keys and error
        bytes, seconds, mb_s, workers   the bytes read, the wall time, the MB per second of all
                 files together and the threads used
    """
    from huggingface_hub import try_to_load_from_cache

    users: dict[tuple[str, str, str], list[str]] = {}
    for name in names:
        for key in weight_files(name):
            users.setdefault(key, []).append(name)
    found, missing = [], []
    for (repo, revision, file), teachers in users.items():
        entry = {"teachers": teachers, "repo": repo, "revision": revision, "file": file}
        path = try_to_load_from_cache(repo, file, cache_dir=cache_dir, revision=revision)
        if isinstance(path, str) and os.path.isfile(path):
            found.append({**entry, "path": path, "bytes": os.path.getsize(path)})
        else:
            missing.append(entry)
            print(f"warm  {', '.join(teachers):<44} {file:<26} not in the cache at revision {revision[:8]}, "
                  "its build fetches it")
    found.sort(key=lambda e: -e["bytes"])
    threads = max(1, min(int(workers), len(found))) if found else 0

    def one(entry: dict) -> dict:
        begin = time.perf_counter()
        try:
            entry["bytes"] = _read_through(entry["path"], block)
        except OSError as err:
            entry["error"] = f"{type(err).__name__}, {err}"
        entry["seconds"] = time.perf_counter() - begin
        entry["mb_s"] = entry["bytes"] / MB / max(entry["seconds"], 1e-9)
        return entry

    start = time.perf_counter()
    failed = []
    if found:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            for job in as_completed([pool.submit(one, entry) for entry in found]):
                e = job.result()
                if "error" in e:
                    failed.append(e)
                    print(f"warm  {', '.join(e['teachers']):<44} {e['file']:<26} NOT read ahead ({e['error']}), "
                          "its build reads it itself")
                    continue
                print(f"warm  {', '.join(e['teachers']):<44} {e['file']:<26} {e['bytes'] / MB:9,.1f} MB in "
                      f"{e['seconds']:6.1f} s, {e['mb_s']:,.0f} MB per second")
    wall = time.perf_counter() - start
    read = [e for e in found if "error" not in e]
    total = sum(e["bytes"] for e in read)
    return {"files": read, "missing": missing, "failed": failed, "bytes": total, "seconds": wall,
            "mb_s": total / MB / max(wall, 1e-9), "workers": threads}


__all__ = ["TEACHERS", "WARM_WORKERS", "Teacher", "TeacherOut", "band_index", "build_teacher", "warm_weights",
           "weight_files"]
