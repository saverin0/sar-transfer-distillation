"""The prepared-data format shared by every domain, copied from part two.

Each converter turns one public dataset into this layout ONCE, on a CPU runtime.
It is stored on Drive as a few large files per split, so a GPU session copies a
handful of big files instead of opening thousands of small ones:

    <root>/<dataset>/
        manifest.json        dataset, source, licence, classes, ignore_index,
                             channels, units, dtype, pixel_spacing_m, chip_size,
                             splits {name: n_chips}, converter notes
        <split>_images.npy   SAR (part two): float16 (N, 2, H, W). Channel 0 = co-
                             polarisation (VV or HH), channel 1 = cross-polarisation
                             (VH or HV). Units as in the manifest, dB where the source
                             allows. NaN where the image has no data.
                             Aerial (part three): uint8 (N, 4, H, W), channels R, G, B,
                             NIR, 8 bit, no NaN (no-data is a label of 255).
        <split>_labels.npy   uint8 (N, H, W): class id, 255 = ignore
        <split>_meta.csv     one row per chip: chip_id, region, date, source
                             file and any other columns the converter adds

"train" is the only split training may use; every other split is scored.
Nothing here downloads anything or knows about a particular dataset.

Part three (2026-09-30) adds the keyword `dtype` to `PreparedWriter` (default float16,
the aerial domain writes uint8) and takes the channel count from `channels`. With the
defaults every check and every file is as in part two.

Part four (2026-10-04) adds `split_files` and `copy_prepared`, which copies a prepared
folder (the copies on Drive) to the runtime disk, only the splits that are read, several
files at once, each one complete or not there.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

IGNORE = 255
CHUNK = 512                      # chips per temporary part file, at most ...
FLUSH_MB = 256.0                 # ... and at most this many MB of buffered chips per split


class PreparedWriter:
    """Collect chips per split and write the prepared layout on `close()`."""

    def __init__(self, root: str | Path, dataset: str, chip_size: int, classes: dict[int, str],
                 source: dict, channels: tuple[str, ...] = ("VV", "VH"), units: str = "dB",
                 pixel_spacing_m: float | None = None, notes: str = "", flush_mb: float = FLUSH_MB,
                 dtype=np.float16):
        if chip_size % 16:
            raise ValueError(f"chip_size {chip_size} must be a multiple of the 16 px patch")
        if IGNORE in classes or not classes or max(classes) >= IGNORE:
            raise ValueError(f"class ids must be 0..254, got {sorted(classes)}")
        self.dir = Path(root) / dataset
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dataset, self.chip_size = dataset, chip_size
        self.classes = {int(k): str(v) for k, v in classes.items()}
        self.source, self.channels, self.units = source, tuple(channels), units
        self.dtype = np.dtype(dtype)
        self.n_ch = len(self.channels)
        self.pixel_spacing_m, self.notes = pixel_spacing_m, notes
        self._buf: dict[str, dict[str, list]] = {}
        self._parts: dict[str, int] = {}
        self._n: dict[str, int] = {}
        self._meta: dict[str, list] = {}
        self._bytes: dict[str, int] = {}
        self.flush_bytes = int(flush_mb * 1e6)

    def add(self, split: str, image: np.ndarray, label: np.ndarray, **meta) -> None:
        """image (C, H, W) float, NaN = no data (uint8 for the aerial domain); label (H, W) class ids, 255 = ignore."""
        s = self.chip_size
        if image.shape != (self.n_ch, s, s):
            raise ValueError(f"image {image.shape}, expected ({self.n_ch}, {s}, {s})")
        if label.shape != (s, s):
            raise ValueError(f"label {label.shape}, expected ({s}, {s})")
        if np.isinf(image).any():
            raise ValueError("image has inf values; use NaN for no data")
        vals = np.unique(label)
        bad = [int(v) for v in vals if int(v) != IGNORE and int(v) not in self.classes]
        if bad:
            raise ValueError(f"label values {bad} not in classes {sorted(self.classes)} or {IGNORE}")
        img16 = image.astype(self.dtype)
        if np.isinf(img16).any():
            raise ValueError("image values exceed the float16 range (|x| >= 65504); check the units")
        b = self._buf.setdefault(split, {"img": [], "lab": []})
        b["img"].append(img16)
        b["lab"].append(label.astype(np.uint8))
        self._meta.setdefault(split, []).append({"chip_id": f"{split}_{self._n.get(split, 0):07d}", **meta})
        self._n[split] = self._n.get(split, 0) + 1
        self._bytes[split] = self._bytes.get(split, 0) + img16.nbytes + label.size
        if len(b["img"]) >= CHUNK or self._bytes[split] >= self.flush_bytes:
            self._flush(split)

    def flush(self, split: str) -> None:
        """Write the buffered chips of `split` to a part file now (bounded RAM)."""
        self._flush(split)

    def abort(self) -> None:
        """Delete the part files of a conversion that stopped before close()."""
        for p in self.dir.glob(".*.part*.npy"):
            p.unlink()
        self._buf.clear()

    def _flush(self, split: str) -> None:
        b = self._buf.get(split)
        if not b or not b["img"]:
            return
        k = self._parts.get(split, 0)
        np.save(self.dir / f".{split}_images.part{k:05d}.npy", np.stack(b["img"]))
        np.save(self.dir / f".{split}_labels.part{k:05d}.npy", np.stack(b["lab"]))
        self._parts[split] = k + 1
        b["img"].clear()
        b["lab"].clear()
        self._bytes[split] = 0

    def close(self, stats: dict | None = None) -> dict:
        """Concatenate the parts into one .npy per array and split, write the manifest.

        `stats`: optional converter statistics (JSON-serialisable), stored in the manifest."""
        s = self.chip_size
        for split in list(self._n):
            self._flush(split)
            n = self._n[split]
            imgs = np.lib.format.open_memmap(self.dir / f"{split}_images.npy", mode="w+",
                                             dtype=self.dtype, shape=(n, self.n_ch, s, s))
            labs = np.lib.format.open_memmap(self.dir / f"{split}_labels.npy", mode="w+",
                                             dtype=np.uint8, shape=(n, s, s))
            i = 0
            for k in range(self._parts[split]):
                pi, pl = self.dir / f".{split}_images.part{k:05d}.npy", self.dir / f".{split}_labels.part{k:05d}.npy"
                a, lab = np.load(pi), np.load(pl)
                imgs[i:i + len(a)] = a
                labs[i:i + len(lab)] = lab
                i += len(a)
                pi.unlink()
                pl.unlink()
            assert i == n, (split, i, n)
            imgs.flush()
            labs.flush()
            del imgs, labs
            pd.DataFrame(self._meta[split]).to_csv(self.dir / f"{split}_meta.csv", index=False)
        manifest = {"dataset": self.dataset, "format_version": 1,
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "source": self.source, "classes": self.classes, "ignore_index": IGNORE,
                    "channels": list(self.channels), "units": self.units, "dtype": self.dtype.name,
                    "pixel_spacing_m": self.pixel_spacing_m, "chip_size": s,
                    "splits": {k: int(v) for k, v in sorted(self._n.items())}, "notes": self.notes}
        if stats is not None:
            manifest["stats"] = stats
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        return manifest


def load_manifest(root: str | Path, dataset: str) -> dict:
    return json.loads((Path(root) / dataset / "manifest.json").read_text(encoding="utf-8"))


SPLIT_FILES = ("images.npy", "labels.npy", "meta.csv")       # the files of one split, <split>_<kind>
# Files copied at once by `copy_prepared`. A dataset is the manifest and three files per split, the
# images file of a split up to several GB; the copy reads from Google Drive, where one stream is one
# file. 4 at once - not measured yet, the copy prints its MB per second per file and in all, so the
# first real run shows whether more or fewer pay off.
COPY_WORKERS = 4


def split_files(splits) -> list[str]:
    """The file names of these splits in a prepared dataset folder, <split>_images.npy and so on."""
    return [f"{s}_{kind}" for s in splits for kind in SPLIT_FILES]


def copy_prepared(source: str | Path, root: str | Path, dataset: str, splits=None,
                  workers: int = COPY_WORKERS) -> Path:
    """Copy a prepared dataset folder into root/<dataset> - its manifest and the files of `splits`.

    source   a folder that holds manifest.json and the split files, as the private copies of the
             datasets hold them (on Drive under MyDrive/datasets, part four)
    splits   the splits to copy, None = every split of the manifest
    workers  files copied at once (`COPY_WORKERS`, the reason is written there)
    Every file is checked in `source` before anything is copied (FileNotFoundError, a split not in
    the manifest is a ValueError). A file already in root/<dataset> with the size of the source is
    skipped. Each file is written under a temporary name and renamed when it is complete, so a copy
    that was cut never looks like a whole file; the manifest is written last. One line per file
    with its MB per second, then one for all. Returns root/<dataset>.
    """
    import os
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    src, out = Path(source), Path(root) / dataset
    man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    want = list(man["splits"]) if splits is None else [str(s) for s in splits]
    unknown = [s for s in want if s not in man["splits"]]
    if unknown:
        raise ValueError(f"{dataset}: splits {unknown} are not in the manifest, {list(man['splits'])}")
    names = split_files(want)
    absent = [n for n in names if not (src / n).is_file()]
    if absent:
        raise FileNotFoundError(f"{src}: {absent} missing, the copy of {dataset} is not complete")
    out.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names
            if not ((out / n).is_file() and (out / n).stat().st_size == (src / n).stat().st_size)]

    def one(name: str) -> tuple[str, float]:
        clock = time.perf_counter()
        part = out / f"{name}.part"
        shutil.copyfile(src / name, part)
        os.replace(part, out / name)
        return name, time.perf_counter() - clock

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for name, seconds in pool.map(one, todo):
            mb = (out / name).stat().st_size / 1e6
            print(f"  {dataset}/{name:<24} {mb:10.1f} MB in {seconds:.0f} s, "
                  f"{mb / max(seconds, 1e-9):,.0f} MB per second")
    wall = time.perf_counter() - start
    total = sum((out / n).stat().st_size for n in todo) / 1e6
    shutil.copyfile(src / "manifest.json", out / "manifest.json.part")
    os.replace(out / "manifest.json.part", out / "manifest.json")
    print(f"{dataset}: copied {len(todo)} of {len(names)} files of the splits {want}, {total:,.0f} MB in "
          f"{wall:.0f} s, {total / max(wall, 1e-9):,.0f} MB per second with {max(1, int(workers))} threads, "
          f"{len(names) - len(todo)} were there already")
    return out


def load_split(root: str | Path, dataset: str, split: str, mmap: bool = True) -> dict:
    """{"images" (N,C,H,W) f16 or u8, "labels" (N,H,W) u8, "meta" DataFrame}; memory-mapped by default."""
    d = Path(root) / dataset
    mode = "r" if mmap else None
    return {"images": np.load(d / f"{split}_images.npy", mmap_mode=mode),
            "labels": np.load(d / f"{split}_labels.npy", mmap_mode=mode),
            "meta": read_meta(d / f"{split}_meta.csv")}


def read_meta(path: str | Path) -> pd.DataFrame:
    """Meta CSV with strings kept as written, so region 'NA' (North Asia) and empty dates
    do not turn into NaN (pandas' default)."""
    return pd.read_csv(path, keep_default_na=False, na_values=[], dtype={"region": str, "date": str})


def validate(root: str | Path, dataset: str, sample: int = 200, seed: int = 0) -> pd.DataFrame:
    """Print and return one row per split with chips, label shares, NaN share, value range (sample)."""
    man = load_manifest(root, dataset)
    rng = np.random.default_rng(seed)
    rows = []
    for split, n in man["splits"].items():
        d = load_split(root, dataset, split)
        assert len(d["images"]) == len(d["labels"]) == len(d["meta"]) == n, (split, n)
        idx = np.sort(rng.choice(n, min(sample, n), replace=False))
        img, lab = np.asarray(d["images"][idx], np.float32), np.asarray(d["labels"][idx])
        counts = np.bincount(lab.ravel(), minlength=256)
        tot = counts.sum()
        row = {"split": split, "chips": n, "nan_share": float(np.isnan(img).mean()),
               "ignore_share": float(counts[IGNORE] / tot)}
        for ch, name in enumerate(man["channels"]):
            v = img[:, ch][np.isfinite(img[:, ch])]
            row[f"{name}_p1"] = float(np.percentile(v, 1)) if v.size else np.nan
            row[f"{name}_p99"] = float(np.percentile(v, 99)) if v.size else np.nan
        for cid, name in man["classes"].items():
            row[f"share_{name}"] = float(counts[int(cid)] / tot)
        rows.append(row)
    out = pd.DataFrame(rows)
    print(f"{dataset}: {man['chip_size']} px chips, units {man['units']}, classes {man['classes']}")
    print(out.round(4).to_string(index=False))
    return out
