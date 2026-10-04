"""Private Hugging Face copies of the data, part two's converted SAR datasets and the paired Sentinel-1 and Sentinel-2 pilot data.

Part two uploaded each prepared dataset folder (manifest.json, <split>_images.npy,
<split>_labels.npy, <split>_meta.csv) into a PRIVATE dataset repo
<owner>/sar-transfer-<dataset with '_' as '-'>. `fetch` downloads one back into a local
folder (e.g. /content on Colab), as part two's run notebook does, since part four only the
splits asked for. The upload side of those stays in part two. The owner of these copies
comes from DATASET_HF_OWNER of the .env file (part four); the same folders are on Drive too,
MyDrive/datasets/<repo_name>, which `prepared.copy_prepared` copies when the Hub has none.

The pilot data of notebook 01 goes to Drive AND to ONE private dataset repo of the user. Its
full id, owner/name, is NOT written here but read from DATA_HF_REPO of the .env file by
`data_repo_id` (2026-10-03), so that this repository names no
other one. The next notebooks fetch the data from the Hub first and fall back to Drive.
`upload_pilot` writes a README card with the
licences and the attribution of the sources, creates the repo PRIVATE, refuses a public one
as part two's upload did, uploads every pilot file that is not on the Hub with the same size
yet, `UPLOAD_WORKERS` files at once, and checks every size afterwards. `fetch_pilot` downloads
the files into a folder, `FETCH_WORKERS` at once, and skips complete ones; the small files come
first and the pair archives in chunk order (`fetch_order`), and every file that is whole on the
disk is handed to a callback (`on_file`), so the data notebook reads an archive into RAM while
the next ones arrive (pilotdata.ChunkReader). `list_pilot` lists
the files with their sizes. The files are the ones notebook 01 writes under majortom/ and
aerial/, named by `pilot_patterns`.

Why an own thread pool. huggingface_hub's `run_as_future` runs its jobs one after the other
("queued to preserve order but are not ran in parallel", its docstring says to use an own
ThreadPoolExecutor). `upload_folder` streams a folder in one or a few commits and `create_commit`
takes `num_threads`, but neither skips a file by its size on the Hub and neither is the
`upload_file` the tests stand in for, so each file is one `upload_file` call in a pool.
`hf_hub_download` fetches one file per call, so the fetch is a pool of these calls too. Both
print the MB per second per file and in all. A file that fails does not stop the others, it is
named in the error at the end, and a re-run skips the files that arrived whole.

The tokens come from the environment (HF_TOKEN to read, HF_WRITE_TOKEN to write, loaded from
.env by env.load_env). They go to huggingface_hub only, which sends them to huggingface.co,
and are never printed or written anywhere.
"""

from __future__ import annotations

import datetime as dt
import fnmatch
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

from .prepared import load_manifest

MB = 1e6
GB = 1e9
# Files sent or fetched at once. The pilot is about 20 files of 60 GB, most of them archives of
# 2 to 6 GB, so the wall is the throughput of one connection, not the latency of a request (a
# range request answered after about 4 s on 2026-10-01, whatever its size). One upload_file is
# one commit whose file goes up in chunks, one hf_hub_download streams one file to the disk.
# 4 at once keeps the disk writes and the memory of the transfers small; how much it gains over
# one at a time is not measured, both print the MB per second. The upload of 2026-09-30 sent the
# files one after the other and its rate was not measured. Part three's data notebook (A100
# runtime) fetched its 46 files 4 at once in 2.8 min, about 70 MB/s per file.
UPLOAD_WORKERS = 4
FETCH_WORKERS = 4
# Files fetched at once when only the small files of the pilot are asked for (the chunk JSONs, the
# infos tables, the two lists, the statistics, the aerial manifest, about 31 files of a few kB to a
# few MB). Each costs a request or two whose answer takes about 1 to 4 s whatever its size (see
# above), so the wall is the latency, not the bytes - 32 at once take them all in one round, 4 at
# once would wait about 8 rounds. About 60 requests, far under the limit of the Hub with a token
# (5,000 requests per 5 minutes, pilotdata.WORKERS). Not measured, the fetch prints its rate
SMALL_FETCH_WORKERS = 32
MAJORTOM_S2 = "Major-TOM/Core-S2L2A-249k"      # majortom.S2_REPO
MAJORTOM_S1 = "Major-TOM/Core-S1RTC"           # majortom.S1_REPO


def repo_name(dataset: str) -> str:
    """The name of a converted dataset's private copy, sar-transfer-<dataset with '_' as '-'>.

    The repo on the Hub is <owner>/<this name>; the copy on Drive (part four) is the folder
    MyDrive/datasets/<this name>, the same files."""
    return f"sar-transfer-{dataset.replace('_', '-')}"


def repo_id(dataset: str, owner: str) -> str:
    return f"{owner}/{repo_name(dataset)}"


# Files of one dataset fetched at once by `fetch`. The manifest and three files per split, 7 for
# train and val (stage A), 10 for a dataset with three splits; 8 is huggingface_hub's own default
# for snapshot_download, so every file of train and val comes at once. Not measured on its own,
# `fetch` prints the MB per second of the dataset.
DATASET_FETCH_WORKERS = 8


def fetch(dataset: str, owner: str, dest: str | Path, token: str, splits=None,
          workers: int = DATASET_FETCH_WORKERS) -> Path:
    """Download one converted dataset from its private repo into dest/<dataset>.

    splits   the splits to fetch with the manifest, their images, labels and meta files. None =
             every split (part two). Part four's stage A reads train and val only, and every split
             of the control dataset, so the test splits of the others stay on the Hub.
    workers  files at once (`DATASET_FETCH_WORKERS`, the reason is written there)
    Every asked split must be in the manifest (ValueError) and on disk after the download
    (FileNotFoundError). Prints the MB of those splits and the MB per second.
    """
    from huggingface_hub import snapshot_download

    from .prepared import split_files

    out = Path(dest) / dataset
    allow = None if splits is None else ["manifest.json", *split_files(splits)]
    clock = time.perf_counter()
    snapshot_download(repo_id(dataset, owner), repo_type="dataset", local_dir=str(out), token=token,
                      allow_patterns=allow, max_workers=max(1, int(workers)))
    seconds = time.perf_counter() - clock
    man = load_manifest(dest, dataset)
    want = list(man["splits"]) if splits is None else [str(s) for s in splits]
    unknown = [s for s in want if s not in man["splits"]]
    if unknown:
        raise ValueError(f"{dataset}: splits {unknown} are not in the manifest, {list(man['splits'])}")
    for name in split_files(want):
        if not (out / name).is_file():
            raise FileNotFoundError(f"{out}: {name} missing after download")
    mb = sum((out / name).stat().st_size for name in split_files(want)) / MB
    print(f"{dataset}: {len(want)} splits ready in {out}, {want}, {mb:,.0f} MB in {seconds:.0f} s, "
          f"{mb / max(seconds, 1e-9):,.0f} MB per second with {max(1, int(workers))} threads")
    return out


# ------------------------------------------------------------- the pilot data

def data_repo_id() -> str:
    """The private dataset repo of the pilot data, the full "owner/name" from DATA_HF_REPO.

    The value comes from the .env file through env.load_env (or the environment). Unset or not
    of the form owner/name = ValueError. Nothing is guessed and no default is written here.
    """
    rid = os.environ.get("DATA_HF_REPO", "").strip()
    if not rid or rid.count("/") != 1 or not all(rid.split("/")):
        raise ValueError("DATA_HF_REPO is not set or is not owner/name. Put the full id of the private "
                         "dataset repo of the pilot data into the .env file (see "
                         ".env.example) and call env.load_env() first")
    return rid


def pilot_repo_id(owner: str | None = None) -> str:
    """The pilot repo id, `data_repo_id()`. An owner is no longer accepted (2026-10-03)."""
    if owner is not None:
        raise TypeError("pilot_repo_id takes no owner any more, the full id comes from DATA_HF_REPO "
                        "of the .env file")
    return data_repo_id()


def hub_api(token: str | None):
    """An HfApi that holds the token. Tests replace this with a stand-in."""
    from huggingface_hub import HfApi

    return HfApi(token=token)


def hub_download(rid: str, name: str, dest: Path, token: str | None) -> Path:
    """One file of the dataset repo `rid` to dest/<name>. Tests replace this with a stand-in."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(rid, name, repo_type="dataset", local_dir=str(dest), token=token))


def pilot_patterns() -> tuple[str, ...]:
    """The files of the pilot, as notebook 01 writes them under majortom/ and aerial/, plus the card."""
    from ..distil import aerial, pilotdata

    return ("majortom/chunk_*.tar", "majortom/chunk_*.json", "majortom/chunk_*_infos.csv",
            f"majortom/{pilotdata.TRAIN_FILE}", f"majortom/{pilotdata.HELD_OUT_FILE}",
            "majortom/channel_stats.json", "majortom/pilot_sample.csv",
            *(f"aerial/{name}.tar" for name in aerial.SETS), "aerial/manifest.json", "aerial/plan.csv",
            "README.md")


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def pilot_files(pilot_dir: str | Path) -> dict[str, int]:
    """The pilot files under `pilot_dir` with their bytes, by relative path with '/'.

    Every chunk JSON needs its tar with the bytes the JSON says (ValueError when cut), and
    the lists, the statistics, the sample, the aerial archives, manifest and plan must be
    there (FileNotFoundError). The card README.md is taken when it is there.
    """
    d = Path(pilot_dir)
    patterns = pilot_patterns()
    files = {p.relative_to(d).as_posix(): p.stat().st_size for p in sorted(d.rglob("*"))
             if p.is_file() and _matches(p.relative_to(d).as_posix(), patterns)}
    infos = [name for name in files if fnmatch.fnmatchcase(name, "majortom/chunk_*.json")]
    if not infos:
        raise FileNotFoundError(f"{d}: no chunk archive (majortom/chunk_<k>.json), run notebook 01 first")
    for info_name in infos:
        info = json.loads((d / info_name).read_text(encoding="utf-8"))
        tar_name = info_name[:-len(".json")] + ".tar"
        if tar_name not in files:
            raise FileNotFoundError(f"{d}: {tar_name} is missing next to {info_name}, run notebook 01 again")
        if files[tar_name] != info.get("bytes"):
            raise ValueError(f"{d}: {tar_name} has {files[tar_name]} bytes, its JSON says "
                             f"{info.get('bytes')}, the archive is cut, run notebook 01 again")
    alone = [name for name in files if fnmatch.fnmatchcase(name, "majortom/chunk_*.tar")
             and name[:-len(".tar")] + ".json" not in files]
    if alone:
        raise FileNotFoundError(f"{d}: {alone} have no JSON next to them, run notebook 01 again")
    needed = [p for p in patterns if "*" not in p and p != "README.md"]
    missing = [name for name in needed if name not in files]
    if missing:
        raise FileNotFoundError(f"{d}: pilot files missing, {missing}, run notebook 01 first")
    return files


def pilot_card(rid: str, pilot_dir: str | Path, files: dict[str, int], date: str | None = None) -> str:
    """The dataset card, a private working copy with the sources, their licences and attribution."""
    from ..distil import aerial

    d = Path(pilot_dir)
    date = date or dt.date.today().isoformat()
    chunks = sorted(name for name in files if fnmatch.fnmatchcase(name, "majortom/chunk_*.json"))
    pairs = sum(int(json.loads((d / name).read_text(encoding="utf-8")).get("pairs", 0)) for name in chunks)
    manifest = json.loads((d / "aerial" / "manifest.json").read_text(encoding="utf-8"))
    sets = "\n".join(f"| {name} | {entry['count']:,} |" for name, entry in manifest["sets"].items())
    listing = "\n".join(f"| {name} | {size:,} |" for name, size in sorted(files.items()) if name != "README.md")
    return f"""---
license:
- cc-by-sa-4.0
- etalab-2.0
tags: [sentinel-1, sentinel-2, aerial, distillation, sar-transfer]
---
# {rid}, the paired Sentinel-1 and Sentinel-2 pilot data

Private working copy, written by the code of the project on {date}. It is NOT an original
dataset. It holds {pairs:,} paired Major-TOM cells in {len(chunks)} archives (Sentinel-1 VV and VH in
dB as float16, the 12 Sentinel-2 L2A bands at native resolution as uint16, the cloud mask, crops
of 384 px at 10 m, the rules of the plan applied once when a pair was made), the two lists of
cells (training and held out), the channel statistics, the sample table, and the FLAIR-HUB aerial
sets (RGB + NIR, 8 bit, patches halved to {aerial.OUT_PX} px at {aerial.PIXEL_M} m) with their manifest and plan.

| aerial set | patches |
|---|---|
{sets}

## Sources, licences and attribution

**Major-TOM (ESA).** {MAJORTOM_S2} and {MAJORTOM_S1} on Hugging Face,
https://huggingface.co/datasets/{MAJORTOM_S2} and https://huggingface.co/datasets/{MAJORTOM_S1},
licence CC-BY-SA-4.0. ESA Major-TOM (Francis and Czerkawski, Major TOM - Expandable Datasets
for Earth Observation, IGARSS 2024, arXiv 2402.12095, verified 2026-09-30). Contains modified
Copernicus Sentinel data. Read from the
branch main (the code pins no revision of these two sets), fetched on {date}. This copy keeps
that licence and its attribution terms.

**FLAIR-HUB (IGN France).** {aerial.ATTRIBUTION} Pinned revision {aerial.HF_REVISION}
(updated {aerial.HF_UPDATED}), licence Etalab Open Licence 2.0. This copy keeps that licence
and its attribution terms.

## Files

| file | bytes |
|---|---|
{listing}
"""


def _remote_sizes(api, rid: str) -> dict[str, int]:
    """The files of the dataset repo with their bytes (folders have no size and are left out)."""
    return {f.path: int(f.size) for f in api.list_repo_tree(rid, repo_type="dataset", recursive=True)
            if getattr(f, "size", None) is not None}


def _transfer(todo: dict[str, int], one: Callable[[str], None], workers: int, verb: str,
              on_done: Callable[[str], None] | None = None) -> tuple[dict[str, str], float]:
    """`one(name)` over the files of `todo` (name -> bytes) in a pool of `workers` threads, started
    in the order of `todo`.

    One line per file as it finishes, with its MB per second over the time of its own transfer
    (not the time it waited for a free thread), then one line for all. A file that fails does
    not stop the others, its error is kept; returns (errors by name, seconds). `on_done(name)`
    is called in the calling thread for every file that finished without an error, right after
    its line. When the loop stops early (an error of on_done, an interrupt) the files that have
    not started are cancelled.
    """
    errors: dict[str, str] = {}
    took: dict[str, float] = {}

    def timed(name: str) -> None:
        begin = time.perf_counter()
        try:
            one(name)
        finally:
            took[name] = time.perf_counter() - begin

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        jobs = {pool.submit(timed, name): name for name in todo}
        try:
            for job in as_completed(jobs):
                name = jobs[job]
                seconds = took.get(name, 0.0)
                err = job.exception()
                if err is not None:
                    errors[name] = f"{type(err).__name__}, {err}"
                    print(f"  {name:<36} {todo[name] / MB:10.1f} MB  FAILED after {seconds:.0f} s, "
                          f"{type(err).__name__}")
                    continue
                print(f"  {name:<36} {todo[name] / MB:10.1f} MB  {verb} in {seconds:.0f} s, "
                      f"{todo[name] / MB / max(seconds, 1e-9):,.0f} MB per second")
                if on_done is not None:
                    on_done(name)
        except BaseException:
            for job in jobs:
                job.cancel()
            raise
    wall = time.perf_counter() - start
    done = sum(size for name, size in todo.items() if name not in errors)
    print(f"{verb} {len(todo) - len(errors)} of {len(todo)} files, {done / MB:,.0f} MB in {wall:,.0f} s, "
          f"{done / MB / max(wall, 1e-9):,.0f} MB per second with {max(1, int(workers))} threads")
    return errors, wall


def upload_pilot(pilot_dir: str | Path, token: str | None = None, dry_run: bool = False,
                 date: str | None = None, workers: int = UPLOAD_WORKERS) -> str:
    """Upload the pilot files to the private repo `data_repo_id()`, skipping files on the Hub with the same size.

    Writes README.md into `pilot_dir` first. The repo is created private, a public one is
    refused. The files go up `workers` at once (`UPLOAD_WORKERS`, the reason is written there),
    each one its own upload_file call and commit, the MB per second printed per file and in
    all. A file that fails does not stop the others, it is named in the IOError at the end.
    After the upload every file must be on the Hub with its local size, else IOError.
    Returns the repo id. `dry_run` lists the files and touches no repo.
    """
    d = Path(pilot_dir)
    files = pilot_files(d)
    rid = pilot_repo_id()
    (d / "README.md").write_text(pilot_card(rid, d, files, date), encoding="utf-8")
    files["README.md"] = (d / "README.md").stat().st_size
    total = sum(files.values())
    print(f"{rid} (private), {len(files)} files, {total / GB:.2f} GB")
    if dry_run:
        for name, size in files.items():
            print(f"  {name:<36} {size / MB:10.1f} MB")
        return rid

    api = hub_api(token)
    api.create_repo(rid, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(rid, repo_type="dataset").private:
        raise RuntimeError(f"{rid} exists and is PUBLIC, refusing to upload into it")
    remote = _remote_sizes(api, rid)
    todo = {}
    for name, size in files.items():
        if remote.get(name) == size:
            print(f"  {name:<36} {size / MB:10.1f} MB  on the Hub with the same size, skipped")
        else:
            todo[name] = size
    skipped = len(files) - len(todo)

    def one(name: str) -> None:
        api.upload_file(path_or_fileobj=str(d / name), path_in_repo=name, repo_id=rid, repo_type="dataset",
                        commit_message=f"pilot, {name}")

    errors = {}
    if todo:
        errors, _ = _transfer(todo, one, workers, "uploaded")
    remote = _remote_sizes(api, rid)
    bad = [name for name, size in files.items() if remote.get(name) != size]
    if errors or bad:
        raise IOError(f"{rid} - after the upload these files are missing or differ in size, {bad}"
                      + (f", these failed, {errors}" if errors else "")
                      + ". Run the upload again, files on the Hub with the same size are skipped")
    print(f"uploaded and checked, {len(files)} files, {skipped} skipped, sizes equal")
    return rid


def fetch_order(name: str) -> tuple:
    """Sort key of the fetch - the small files first, then the pair archives by chunk number
    (chunk_2 before chunk_10), then the other archives (the aerial sets), each group by name.

    The data notebook reads a pair archive as soon as it is on the disk (`on_file` of
    `fetch_pilot`), so the archives are started in the order it reads them, and small files
    asked for together with archives are there before the first archive."""
    chunk = re.fullmatch(r"majortom/chunk_(\d+)\.tar", name)
    if chunk:
        return (1, int(chunk.group(1)), name)
    return (2 if name.endswith(".tar") else 0, 0, name)


def fetch_pilot(dest: str | Path, token: str | None = None, want: Iterable[str] | None = None,
                workers: int = FETCH_WORKERS,
                on_file: Callable[[str, Path], None] | None = None) -> dict[str, Path]:
    """Download the pilot files of the private repo `data_repo_id()` into `dest` (majortom/ and aerial/ as on Drive).

    want    patterns of the files to fetch, None = every pilot file. Only pilot files are
            ever fetched, whatever the patterns say.
    workers files fetched at once (`FETCH_WORKERS`, the reason is written there;
            `SMALL_FETCH_WORKERS` for the small files alone), each one its own hf_hub_download
            call, started in `fetch_order`, the MB per second printed per file and in all
    on_file called as on_file(name, path) in the calling thread for every file that is whole on
            the disk - first for the files that were there already, then for each file right
            after it arrived, while the others are still being fetched. It must return quickly
            (the reader of the data notebook only queues the archive). An error it raises stops
            the fetch, the files not started yet are cancelled
    A file already in `dest` with the size of the Hub is skipped. A file that fails or arrives
    with other bytes does not stop the others, it is named in the IOError at the end, and a
    re-run skips the files that arrived whole. A repo that is missing or holds none of the
    wanted files raises, the notebooks then fall back to Drive. Returns the local paths by
    relative name, in name order whatever the order the files arrived in.
    """
    rid = pilot_repo_id()
    api = hub_api(token)
    remote = _remote_sizes(api, rid)
    patterns = tuple(want) if want is not None else pilot_patterns()
    chosen = {name: size for name, size in remote.items()
              if _matches(name, patterns) and _matches(name, pilot_patterns())}
    if not chosen:
        raise FileNotFoundError(f"{rid} holds no file of {list(patterns)}")
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    todo, ready = {}, []
    for name in sorted(chosen, key=fetch_order):
        target = out / name
        if target.is_file() and target.stat().st_size == chosen[name]:
            ready.append(name)
        else:
            todo[name] = chosen[name]
    skipped = len(ready)
    if on_file is not None:                 # the files there already, before the transfers start
        for name in ready:
            on_file(name, out / name)

    def one(name: str) -> None:
        got = hub_download(rid, name, out, token)
        if not got.is_file() or got.stat().st_size != chosen[name]:
            raise IOError(f"{name} has not the bytes of the Hub after the download")

    errors = {}
    if todo:
        done = None if on_file is None else (lambda name: on_file(name, out / name))
        errors, _ = _transfer(todo, one, workers, "fetched", on_done=done)
    if errors:
        raise IOError(f"{rid} - these files did not arrive whole, {errors}. Run the fetch again, "
                      "files on the disk with the size of the Hub are skipped")
    paths = {name: out / name for name in sorted(chosen)}
    print(f"{rid}, {len(paths)} files in {out}, {skipped} were there already")
    return paths


def list_pilot(token: str | None = None) -> dict[str, int]:
    """The files of the private repo `data_repo_id()` with their bytes, by path."""
    return _remote_sizes(hub_api(token), pilot_repo_id())
