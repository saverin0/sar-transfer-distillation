"""The aerial domain of part three in the prepared format of part two.

The FLAIR-HUB patches come from the data module of part three (`distil.aerial`). Its
`build_sets` writes one .npz per set AND department-year, and its `load_set(out_dir, name)`
joins them into the arrays of one set

    images        uint8 (N, 4, 256, 256), channels R, G, B, NIR, 8 bit, 0.4 m
    labels        uint8 (N, 256, 256), the 19 land cover class codes, halved by majority
    ids           (N,), the patch ids
    departments   (N,), the department of every patch

`aerial_to_prepared` takes such arrays per set (or one .npz per set holding images, labels
and ids) and writes them as one prepared dataset (manifest.json, <split>_images.npy uint8,
<split>_labels.npy, <split>_meta.csv), so `run.run_domain` scores it like the SAR datasets -
"train" is trained on, every other set is scored, mIoU over the classes plus the IoU per
class. The class ids of the result are 0 .. k-1 in the order of the sorted source codes;
`classes` (source code -> name) comes from the data module, nothing is guessed here.
A label equal to `nodata` becomes 255 (ignored); any other code outside `classes` stops the
conversion. The manifest keeps the code table under "stats". The region of a patch (the
per-region tables of the run) is what `region_of` gives, else its department when the
arrays carry departments, else "all".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .prepared import IGNORE, PreparedWriter

AERIAL_CHANNELS = ("R", "G", "B", "NIR")
UNITS = "8 bit"


def aerial_to_prepared(sets: Mapping[str, str | Path | Mapping[str, Any]], out_root: str | Path,
                       classes: Mapping[int, str], dataset: str = "flair_hub", nodata: int | None = None,
                       pixel_spacing_m: float = 0.4, source: dict | None = None,
                       region_of: Callable[[str], str] | None = None, notes: str = "") -> dict:
    """Set name ("train" and the scored sets) -> its arrays or its .npz file. Returns the manifest.

    A set given as arrays is what `distil.aerial.load_set` returns (images, labels, ids,
    departments); a set given as a file is one .npz with images, labels and ids.
    region_of   patch id -> region name for the per-region tables. None = the department of
                the patch when the arrays carry departments, else one region "all".
    """
    if "train" not in sets:
        raise ValueError(f"no 'train' set in {sorted(sets)}")
    codes = sorted(int(c) for c in classes)
    if nodata is not None and int(nodata) in codes:
        raise ValueError(f"nodata code {nodata} is also a class in {codes}")
    if any(c < 0 or c > 255 for c in codes):
        raise ValueError(f"class codes must be 0..255, got {codes}")
    lut = np.full(256, IGNORE, dtype=np.uint8)          # source code -> class id, 255 elsewhere
    lut[codes] = np.arange(len(codes), dtype=np.uint8)
    new_classes = {i: str(classes[c]) for i, c in enumerate(codes)}
    allowed = set(codes) | ({int(nodata)} if nodata is not None else set())

    writer, size = None, None
    for split, given in sets.items():
        if isinstance(given, (str, os.PathLike)):
            with np.load(given) as z:
                images, labels, ids = z["images"], z["labels"], z["ids"]
            departments = None
        else:
            images, labels, ids = given["images"], given["labels"], given["ids"]
            departments = given.get("departments")
        if labels is None:
            raise ValueError(f"{split}: the set has no labels, an unlabelled set cannot be scored")
        if departments is not None and len(departments) != len(images):
            raise ValueError(f"{split}: {len(departments)} departments for {len(images)} chips")
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1] != len(AERIAL_CHANNELS):
            raise ValueError(f"{split}: images must be uint8 (N, {len(AERIAL_CHANNELS)}, H, W), "
                             f"got {images.dtype} {images.shape}")
        if labels.dtype != np.uint8 or labels.shape != images.shape[:1] + images.shape[2:]:
            raise ValueError(f"{split}: labels must be uint8 (N, H, W) matching the images, "
                             f"got {labels.dtype} {labels.shape}")
        if images.shape[2] != images.shape[3] or len(ids) != len(images):
            raise ValueError(f"{split}: chips must be square and ids must match, got {images.shape}, {len(ids)} ids")
        stray = sorted(int(v) for v in np.unique(labels) if int(v) not in allowed)
        if stray:
            raise ValueError(f"{split}: label codes {stray} are neither a class {codes} nor nodata {nodata}")
        if writer is None:
            size = int(images.shape[-1])
            writer = PreparedWriter(out_root, dataset, size, new_classes, source=source or {},
                                    channels=AERIAL_CHANNELS, units=UNITS, pixel_spacing_m=pixel_spacing_m,
                                    notes=notes, dtype=np.uint8)
        elif int(images.shape[-1]) != size:
            raise ValueError(f"{split}: chip size {images.shape[-1]} differs from train's {size}")
        for i in range(len(images)):
            pid = str(ids[i])
            region = region_of(pid) if region_of else (str(departments[i]) if departments is not None else "all")
            writer.add(split, images[i], lut[labels[i]], region=region, date="", source_id=pid)
        writer.flush(split)
        print(f"  {split}: {len(images)} chips of {size} px")
    man = writer.close(stats={"source_class_codes": {str(i): c for i, c in enumerate(codes)},
                              "nodata_code": None if nodata is None else int(nodata)})
    print(f"{dataset}: {man['splits']} written to {writer.dir}")
    return man
