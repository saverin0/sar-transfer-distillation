"""FLAIR-HUB aerial sets, on synthetic zips with the real layout and a server on this machine."""
import _setup

import gc
import http.server
import json
import os
import shutil
import struct
import tarfile
import threading
import time
import tracemalloc
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

from sartransfer.distil import aerial as A
from sartransfer.distil import majortom as M

DIR = _setup.TMP_BASE / "aerial"
shutil.rmtree(DIR, ignore_errors=True)
(DIR / "data").mkdir(parents=True)
M.BACKOFF_S = 0.01
HF0 = M.HF
PX = 32                                  # small patches for speed, one check runs at 512
A.PATCH_PX, A.OUT_PX = PX, PX // 2
FAKE = "not-a-real-token"


def fails(kind, fn, *args, **kwargs) -> str:
    """The call must raise `kind`. Returns the error text."""
    try:
        out = fn(*args, **kwargs)
        if hasattr(out, "__next__"):
            list(out)
    except kind as err:
        return str(err)
    raise SystemExit(f"{getattr(fn, '__name__', fn)}{args[1:]} did not raise {kind.__name__}")


# ------------------------------------------------------------------ synthetic data, real layout

def tif(array, left=900000.0, top=6300000.0, res=0.2, crs="EPSG:2154") -> bytes:
    """One GeoTIFF as bytes, as in the zips (uint8, Lambert-93, 0.2 m, LZW)."""
    array = array if array.ndim == 3 else array[None]
    with rasterio.MemoryFile() as memory:
        with memory.open(driver="GTiff", count=array.shape[0], height=array.shape[1],
                         width=array.shape[2], dtype=array.dtype, crs=crs,
                         transform=Affine(res, 0.0, left, 0.0, -res, top), compress="lzw") as dst:
            dst.write(array)
        return memory.read()


def patch(seed: int, px: int = PX) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (4, px, px), dtype=np.uint8)
    label = rng.integers(0, A.N_CLASSES, (px, px), dtype=np.uint8)
    label[: px // 2] = rng.integers(0, 4, (px // 2, px), dtype=np.uint8)      # many ties here
    return image, label


def to_zip64(data: bytes) -> bytes:
    """The same zip with a zip64 end of central directory (as a zip over 4 GB has one)."""
    assert data[-22:-18] == b"PK\x05\x06" and data[-2:] == b"\0\0"          # no comment
    _, _, _, n_disk, n_all, cd_size, cd_start, _ = struct.unpack("<4sHHHHIIH", data[-22:])
    record = struct.pack("<4sQHHIIQQQQ", b"PK\x06\x06", 44, 45, 45, 0, 0, n_disk, n_all, cd_size, cd_start)
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, cd_start + cd_size, 1)
    end = struct.pack("<4sHHHHIIH", b"PK\x05\x06", 0, 0, n_disk, n_all, 0xFFFFFFFF, 0xFFFFFFFF, 0)
    return data[:-22] + record + locator + end


# (department, year) -> ROI -> number of patches. D045 is small (fewer than its share).
# D012's RGBI zip stores its members (method 0), D036's RGBI zip is zip64, the first member
# of D004's RGBI zip has a 20 byte zip64 extra field in its local header only.
FAKE_ZIPS = {("D004", "2021"): {"AA-S1-32": 6, "AB-S1-7": 6}, ("D017", "2018"): {"UU-S1-3": 6},
             ("D017", "2021"): {"UU-S1-3": 6}, ("D045", "2020"): {"CC-S1-1": 4},
             ("D012", "2019"): {"AF-S1-27": 8}, ("D036", "2020"): {"ZZ-S1-9": 8}}
TRUTH = {}                               # patch id -> (image, label)
NO_LABEL = "D045-2020_CC-S1-1_2-1"       # one patch without a label
seed = 0
for (dept, year), rois in FAKE_ZIPS.items():
    members = {"AERIAL_RGBI": {}, "AERIAL_LABEL-COSIA": {}}
    for roi, n in rois.items():
        for k in range(n):
            patch_id = f"{dept}-{year}_{roi}_{k // 3 + 1}-{k % 3 + 1}"
            image, label = patch(seed)
            seed += 1
            TRUTH[patch_id] = (image, label)
            left, top = 900000.0 + 100 * seed, 6300000.0 - 100 * seed
            for modality, array in (("AERIAL_RGBI", image), ("AERIAL_LABEL-COSIA", label)):
                if patch_id == NO_LABEL and modality != "AERIAL_RGBI":
                    continue
                stem = f"{dept}-{year}_{modality}"
                # the real image files store the pixel size as 0.19999999999999998, the labels as 0.2
                res = 0.19999999999999998 if modality == "AERIAL_RGBI" else 0.2
                members[modality][A.member_name(stem, patch_id)] = tif(array, left, top, res)
    for modality, files in members.items():
        path = DIR / "data" / f"{dept}-{year}_{modality}.zip"
        method = zipfile.ZIP_STORED if (dept, modality) == ("D012", "AERIAL_RGBI") else zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(path, "w", method) as z:
            for i, (name, data) in enumerate(files.items()):
                if i == 0 and (dept, modality) == ("D004", "AERIAL_RGBI"):
                    info = zipfile.ZipInfo(name)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    with z.open(info, "w", force_zip64=True) as f:
                        f.write(data)
                else:
                    z.writestr(name, data)
        if (dept, modality) == ("D036", "AERIAL_RGBI"):
            path.write_bytes(to_zip64(path.read_bytes()))
LISTING = pd.DataFrame([{"department": d, "year": y, "modality": m, "path": f"data/{d}-{y}_{m}.zip",
                         "bytes": os.path.getsize(DIR / "data" / f"{d}-{y}_{m}.zip")}
                        for d, y in FAKE_ZIPS for m in A.MODALITIES]).sort_values("path", ignore_index=True)
TRAIN_IDS = sorted(p for p in TRUTH if p.partition("-")[0] in A.TRAIN_VAL_DEPARTMENTS)
TEST_IDS = sorted(p for p in TRUTH if p.partition("-")[0] in A.TEST_DEPARTMENTS)
assert len(TRUTH) == 44 and len(TRAIN_IDS) == 28 and len(TEST_IDS) == 16
assert A.member_name("data/D004-2021_AERIAL_RGBI.zip", "D004-2021_AA-S1-32_1-1") == \
    "D004-2021_AERIAL_RGBI/AA-S1-32/D004-2021_AERIAL_RGBI_AA-S1-32_1-1.tif"
assert A.member_name(DIR / "data" / "D004-2021_AERIAL_LABEL-COSIA.zip", "D004-2021_AA-S1-32_1-1") == \
    "D004-2021_AERIAL_LABEL-COSIA/AA-S1-32/D004-2021_AERIAL_LABEL-COSIA_AA-S1-32_1-1.tif"
assert "not a patch of" in fails(ValueError, A.member_name, "D004-2021_AERIAL_RGBI.zip", "D005-2018_AA-S1-32_1-1")
assert "not a patch id" in fails(ValueError, A.member_name, "D004-2021_AERIAL_RGBI.zip", "AA-S1-32_1-1")

# the patch index as the central directories give it, built here with zipfile alone
INFOS = {}                               # zip stem -> {member name: ZipInfo}
for d, y in FAKE_ZIPS:
    for m in A.MODALITIES:
        with zipfile.ZipFile(DIR / "data" / f"{d}-{y}_{m}.zip") as z:
            INFOS[f"{d}-{y}_{m}"] = {i.filename: i for i in z.infolist()}
assert len(INFOS["D036-2020_AERIAL_RGBI"]) == 8                  # zipfile reads the zip64 form
assert all(i.compress_type == 0 for i in INFOS["D012-2019_AERIAL_RGBI"].values())
assert all(i.compress_type == 8 for i in INFOS["D004-2021_AERIAL_RGBI"].values())


def fields(stem: str, patch_id: str) -> dict:
    info = INFOS[stem].get(A.member_name(stem, patch_id))
    kind = "rgbi" if stem.endswith("RGBI") else "label"
    if info is None:
        return {f"{kind}_zip": "", **{f"{kind}_{f}": "" for f in A.MEMBER_FIELDS}}
    return {f"{kind}_zip": f"data/{stem}.zip", f"{kind}_offset": str(info.header_offset),
            f"{kind}_size": str(info.compress_size), f"{kind}_method": str(info.compress_type),
            f"{kind}_crc": str(info.CRC)}


PATCHES = pd.DataFrame([{"department": p.partition("-")[0], "year": p[5:9], "patch_id": p,
                         **fields(f"{p[:9]}_AERIAL_RGBI", p), **fields(f"{p[:9]}_AERIAL_LABEL-COSIA", p)}
                        for p in sorted(TRUTH)])[list(A.PATCH_COLUMNS)]
assert (PATCHES["label_zip"] == "").sum() == 1 and PATCHES.loc[PATCHES["label_zip"] == "", "patch_id"].iloc[0] == NO_LABEL
assert (PATCHES.loc[PATCHES["label_zip"] == "", "label_offset"] == "").all()


def row_of(patch_id: str) -> pd.Series:
    return PATCHES.set_index("patch_id").loc[patch_id]


def id_of(info: zipfile.ZipInfo) -> str:
    stem, _, rest = info.filename.rpartition("/")[2].partition("_AERIAL_")
    return f"{stem}_{rest.partition('_')[2][:-4]}"


# ------------------------------------------------------------------ constants against the card

assert len(A.TRAIN_VAL_DEPARTMENTS) == 54 and len(A.TEST_DEPARTMENTS) == 16
assert not set(A.TRAIN_VAL_DEPARTMENTS) & set(A.TEST_DEPARTMENTS)
assert len(A.CLASSES) == A.N_CLASSES == 19 and A.CLASSES[0] == "building" and A.CLASSES[18] == "undefined"
assert A.IGNORE is None and A.PATCH_PX == PX and A.OUT_PX == PX // 2
assert (A.PIXEL_M, A.SOURCE_PIXEL_M, A.BANDS) == (0.4, 0.2, ("R", "G", "B", "NIR"))
assert len(A.HF_REVISION) == 40 and A.HF_REPO == "IGNF/FLAIR-HUB"
assert sorted(A.SETS) == sorted(["distil_train", "distil_heldout", "probe", "val", "test"])
assert A.SET_SIZES == {"distil": 20000, "distil_heldout": 1000, "probe": 4000, "val": 2000, "test": 2000}
assert A.SET_SEEDS == {"distil": 0, "probe": 0, "val": 0, "test": 1}
assert A.PLAN_COLUMNS == ("set", *A.PATCH_COLUMNS) and len(A.PATCH_COLUMNS) == 13
assert A.RANGE_COST["requests"] == A.RANGE_COST["rgbi_members"] + A.RANGE_COST["label_members"]

# ------------------------------------------------------------------ halving

x = np.array([[1, 2, 0, 0], [3, 4, 0, 1], [255, 255, 10, 20], [255, 254, 30, 40]], dtype=np.uint8)
assert A.halve_image(x).tolist() == [[3, 0], [255, 25]]           # 2.5 -> 3, 0.25 -> 0, 254.75 -> 255
assert A.halve_image(x).dtype == np.uint8
four = np.stack([x, x + 1, x // 2, x], axis=0)
assert A.halve_image(four).shape == (4, 2, 2) and (A.halve_image(four)[0] == A.halve_image(x)).all()
assert A.halve_image(four[None]).shape == (1, 4, 2, 2)
assert "even sides" in fails(ValueError, A.halve_image, x[:3])
assert "uint8" in fails(ValueError, A.halve_image, x.astype(np.uint16))

blocks = {(3, 3, 5, 5): 3, (5, 5, 3, 3): 3, (7, 7, 7, 1): 7, (1, 7, 7, 7): 7, (0, 1, 2, 3): 0,
          (9, 2, 9, 9): 9, (18, 18, 0, 0): 0, (5, 2, 2, 5): 2, (4, 4, 4, 4): 4, (2, 18, 18, 1): 18,
          (17, 16, 16, 17): 16}
y = np.zeros((2, 2 * len(blocks)), dtype=np.uint8)
for i, (a, b, c, d) in enumerate(blocks):
    y[0, 2 * i], y[0, 2 * i + 1], y[1, 2 * i], y[1, 2 * i + 1] = a, b, c, d
assert A.halve_label(y).tolist() == [list(blocks.values())] and A.halve_label(y).dtype == np.uint8


def slow_halve_label(label):
    out = np.zeros((label.shape[0] // 2, label.shape[1] // 2), dtype=np.uint8)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            counts = Counter(label[2 * i:2 * i + 2, 2 * j:2 * j + 2].ravel().tolist())
            top = max(counts.values())
            out[i, j] = min(code for code, n in counts.items() if n == top)
    return out


def slow_halve_image(image):
    out = np.zeros((image.shape[0], image.shape[1] // 2, image.shape[2] // 2), dtype=np.uint8)
    for b in range(image.shape[0]):
        for i in range(out.shape[1]):
            for j in range(out.shape[2]):
                total = int(image[b, 2 * i:2 * i + 2, 2 * j:2 * j + 2].sum())
                out[b, i, j] = (total + 2) // 4
    return out


for patch_id in list(TRUTH)[:4]:
    image, label = TRUTH[patch_id]
    assert (A.halve_label(label) == slow_halve_label(label)).all()
    assert (A.halve_image(image) == slow_halve_image(image)).all()
assert A.halve_label(np.stack([label, label])).shape == (2, PX // 2, PX // 2)
assert "even sides" in fails(ValueError, A.halve_label, label[:, :-1])

# ------------------------------------------------------------------ iter_patches on local zips

rgbi = DIR / "data" / "D004-2021_AERIAL_RGBI.zip"
labels = DIR / "data" / "D004-2021_AERIAL_LABEL-COSIA.zip"
got = list(A.iter_patches(rgbi, labels))
assert [g[0] for g in got] == sorted(p for p in TRUTH if p.startswith("D004-2021_"))   # name order
for patch_id, image, label in got:
    assert image.dtype == np.uint8 and image.shape == (4, PX, PX) and (image == TRUTH[patch_id][0]).all()
    assert label.dtype == np.uint8 and label.shape == (PX, PX) and (label == TRUTH[patch_id][1]).all()
ids = ["D004-2021_AB-S1-7_2-1", "D004-2021_AA-S1-32_1-2"]
got = list(A.iter_patches(rgbi, None, ids))
assert [g[0] for g in got] == ids and all(g[2] is None for g in got)          # the asked order
assert all((g[1] == TRUTH[g[0]][0]).all() for g in got)
before = dict(M.STATS)
list(A.iter_patches(rgbi, labels, ids))
assert M.STATS["decoded"] - before["decoded"] == 4                              # counted in majortom
for stem in ("D012-2019", "D036-2020"):                                         # stored members, zip64
    read = {p: i for p, i, _ in A.iter_patches(DIR / "data" / f"{stem}_AERIAL_RGBI.zip",
                                               DIR / "data" / f"{stem}_AERIAL_LABEL-COSIA.zip")}
    assert len(read) == 8 and all((read[p] == TRUTH[p][0]).all() for p in read)
small = DIR / "data" / "D045-2020_AERIAL_RGBI.zip"
assert "no member" in fails(KeyError, A.iter_patches, small, DIR / "data" / "D045-2020_AERIAL_LABEL-COSIA.zip",
                            [NO_LABEL])
assert "no member" in fails(KeyError, A.iter_patches, rgbi, None, ["D004-2021_AA-S1-32_9-9"])

# damaged and wrong zips stop with an error that names the file
(DIR / "text.zip").write_bytes(b"this is not a zip file " * 100)
text = fails(ValueError, A.iter_patches, DIR / "text.zip")
assert "text.zip" in text and "not a zip" in text
data = bytearray(rgbi.read_bytes())
first = zipfile.ZipFile(rgbi).infolist()[0]
start = first.header_offset + 30 + len(first.filename) + 200
data[start:start + 50] = bytes(50)                                              # a member is damaged
(DIR / "damaged").mkdir()
(DIR / "damaged" / rgbi.name).write_bytes(data)
text = fails(ValueError, A.iter_patches, DIR / "damaged" / rgbi.name, None, ["D004-2021_AA-S1-32_1-1"])
assert f"damaged{os.sep}{rgbi.name}" in text and ("is damaged" in text or "not an image" in text), text
(DIR / "cut").mkdir()
(DIR / "cut" / rgbi.name).write_bytes(rgbi.read_bytes()[:-3000])               # the end is missing
assert f"cut{os.sep}{rgbi.name}" in fails(ValueError, A.iter_patches, DIR / "cut" / rgbi.name)


def one_zip(name, modality, items) -> str:
    """A zip named D004-2021_<modality>.zip under DIR/name with the given (patch_id, bytes)."""
    (DIR / name).mkdir(exist_ok=True)
    path = DIR / name / f"D004-2021_{modality}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for patch_id, data in items:
            z.writestr(A.member_name(path, patch_id), data)
    return str(path)


image, label = TRUTH["D004-2021_AA-S1-32_1-1"]
pid = "D004-2021_AA-S1-32_1-1"
AT = (900100.0, 6299900.0)                          # the corner of that patch in the zip (seed 1)
BAD = {
    "image has 3 bands": (one_zip("b1", "AERIAL_RGBI", [(pid, tif(image[:3]))]), None, "expected uint8 (4"),
    "image is 16 px": (one_zip("b2", "AERIAL_RGBI", [(pid, tif(image[:, :16, :16]))]), None, "expected uint8 (4"),
    "image is uint16": (one_zip("b3", "AERIAL_RGBI", [(pid, tif(image.astype(np.uint16)))]), None, "uint16"),
    "label off the grid": (str(rgbi), one_zip("b4", "AERIAL_LABEL-COSIA", [(pid, tif(label, left=900100.4, top=AT[1]))]),
                           "not on the grid"),
    "label in another crs": (str(rgbi), one_zip("b5", "AERIAL_LABEL-COSIA", [(pid, tif(label, *AT, crs="EPSG:32631"))]),
                             "not on the grid"),
    "label code 19": (str(rgbi), one_zip("b6", "AERIAL_LABEL-COSIA", [(pid, tif(np.full_like(label, 19), *AT))]),
                      "19 is not one of the 19 classes"),
    "label has 2 bands": (str(rgbi), one_zip("b7", "AERIAL_LABEL-COSIA", [(pid, tif(np.stack([label, label]), *AT))]),
                          "expected uint8 (32, 32)"),
    "member is not an image": (one_zip("b8", "AERIAL_RGBI", [(pid, b"not a picture")]), None, "not an image"),
    "member of another form": (one_zip("b9", "AERIAL_RGBI", [(pid, tif(image))]), None, "is not"),
}
with zipfile.ZipFile(BAD["member of another form"][0], "a") as z:
    z.writestr("D004-2021_AERIAL_RGBI/notes.txt", b"hello")
for what, (rgbi_path, label_path, expected) in BAD.items():
    ids = None if what == "member of another form" else [pid]
    text = fails(ValueError, A.iter_patches, rgbi_path, label_path, ids)
    assert expected in text, (what, text)
    assert (label_path or rgbi_path).rpartition(os.sep)[2] in text, (what, text)
# a label a rounding error away is on the grid, one a tenth of a pixel away is not
near = one_zip("near", "AERIAL_LABEL-COSIA", [(pid, tif(label, AT[0] + 1e-7, AT[1] - 1e-7, 0.2000000001))])
assert (next(A.iter_patches(rgbi, near, [pid]))[2] == label).all()
off = one_zip("off", "AERIAL_LABEL-COSIA", [(pid, tif(label, AT[0] + 0.02, AT[1]))])
assert "not on the grid" in fails(ValueError, A.iter_patches, rgbi, off, [pid])
# a real-size patch passes the checks and halves to (4, 256, 256)
A.PATCH_PX, A.OUT_PX = 512, 256
big_image, big_label = patch(999, 512)
big = one_zip("big", "AERIAL_RGBI", [(pid, tif(big_image))])
big_labels = one_zip("big", "AERIAL_LABEL-COSIA", [(pid, tif(big_label))])
(got_id, got_image, got_label), = A.iter_patches(big, big_labels)
assert got_image.shape == (4, 512, 512) and got_label.shape == (512, 512)
assert A.halve_image(got_image).shape == (4, 256, 256) and A.halve_label(got_label).shape == (256, 256)
assert (A.halve_label(got_label)[:8, :8] == slow_halve_label(got_label[:16, :16])).all()
A.PATCH_PX, A.OUT_PX = PX, PX // 2

# ------------------------------------------------------------------ the spread rule

assert A._spread(20, {"a": 12, "b": 12, "c": 4}) == {"a": 8, "b": 8, "c": 4}
assert A._spread(7, {"a": 10, "b": 10, "c": 10}) == {"a": 3, "b": 2, "c": 2}
assert A._spread(6, {"a": 10, "b": 10, "c": 10}) == {"a": 2, "b": 2, "c": 2}
assert A._spread(5, {"a": 1, "b": 1, "c": 10}) == {"a": 1, "b": 1, "c": 3}
assert A._spread(0, {"a": 3}) == {"a": 0}
assert A._spread(3, {"a": 3}) == {"a": 3}
assert A._spread(4, {"a": 0, "b": 4}) == {"a": 0, "b": 4}
assert "hold 3" in fails(ValueError, A._spread, 5, {"a": 2, "b": 1, "c": 0})

# ------------------------------------------------------------------ plan_sets on the index

SIZES = {"distil": 20, "distil_heldout": 4, "probe": 8, "val": 6, "test": 6}
plan = A.plan_sets(PATCHES, SIZES)
assert list(plan.columns) == list(A.PLAN_COLUMNS)
assert plan["set"].value_counts().to_dict() == {"distil_train": 16, "distil_heldout": 4, "probe": 8,
                                                "val": 6, "test": 6}
assert plan["set"].drop_duplicates().tolist() == list(A.SETS)                 # sets in order
by = {name: group for name, group in plan.groupby("set")}
distil = pd.concat([by["distil_train"], by["distil_heldout"]])
assert distil["department"].value_counts().to_dict() == {"D004": 8, "D017": 8, "D045": 4}   # D045 gives all
assert by["distil_heldout"]["department"].value_counts().to_dict() == {"D004": 2, "D017": 1, "D045": 1}
assert not set(by["distil_train"]["patch_id"]) & set(by["distil_heldout"]["patch_id"])
assert by["probe"]["department"].value_counts().to_dict() == {"D004": 3, "D017": 3, "D045": 2}
assert by["val"]["department"].value_counts().to_dict() == {"D012": 3, "D036": 3}
assert by["test"]["department"].value_counts().to_dict() == {"D012": 3, "D036": 3}
assert not set(by["val"]["patch_id"]) & set(by["test"]["patch_id"])           # disjoint
assert NO_LABEL not in set(by["probe"]["patch_id"]) and NO_LABEL in set(distil["patch_id"]) | set(TRAIN_IDS)
for name in A.LABELLED:
    assert (by[name]["label_zip"] != "").all() and (by[name]["label_offset"] != "").all()
for name in ("val", "test"):
    assert set(by[name]["patch_id"]) <= set(TEST_IDS)
for name in ("distil_train", "distil_heldout", "probe"):
    assert set(by[name]["patch_id"]) <= set(TRAIN_IDS)
assert (plan["rgbi_zip"] == "data/" + plan["patch_id"].str[:9] + "_AERIAL_RGBI.zip").all()
# the plan holds patches that exist with their member fields, and the two years of D017 are one pool
assert set(plan["patch_id"]) <= set(TRUTH)
assert plan.drop(columns="set").drop_duplicates().set_index("patch_id").equals(
    PATCHES.set_index("patch_id").loc[plan["patch_id"].drop_duplicates()])
assert {p[:9] for p in distil[distil["department"] == "D017"]["patch_id"]} == {"D017-2018", "D017-2021"}
# deterministic, and another seed gives another draw
again = A.plan_sets(PATCHES, SIZES)
assert again.equals(plan) and A.plan_sets(PATCHES.sample(frac=1, random_state=3), SIZES).equals(plan)
other = A.plan_sets(PATCHES, SIZES, {"distil": 5})
assert other[other["set"] == "val"].equals(plan[plan["set"] == "val"])
assert not other[other["set"] == "distil_train"]["patch_id"].tolist() == by["distil_train"]["patch_id"].tolist()
# the csv round trip is exact and the plan is checked when read
A.write_plan(plan, DIR / "plan.csv")
assert A.read_plan(DIR / "plan.csv").equals(plan) and not list(DIR.glob("*.part"))
assert "unknown sets" in fails(ValueError, A._checked_plan, plan.assign(set="other"))
assert "no label zip" in fails(ValueError, A._checked_plan, plan.assign(label_zip=""))
assert "twice" in fails(ValueError, A._checked_plan, pd.concat([plan, plan]))
assert "member field is missing" in fails(ValueError, A._checked_plan, plan.assign(rgbi_size=""))
assert "do not go with" in fails(ValueError, A._checked_plan, plan.assign(label_crc=""))
assert "whole number" in fails(ValueError, A._checked_plan, plan.assign(rgbi_offset="x"))
fails(KeyError, A._checked_plan, plan.drop(columns="year"))
# not enough patches, unknown departments, a patch listed twice
assert "distil" in fails(ValueError, A.plan_sets, PATCHES)                       # the real sizes
assert "test:" in fails(ValueError, A.plan_sets, PATCHES, {**SIZES, "test": 11})
assert "not on the card" in fails(ValueError, A.plan_sets, PATCHES.assign(department="D999"), SIZES)
assert "twice" in fails(ValueError, A.plan_sets, pd.concat([PATCHES, PATCHES]), SIZES)
assert "held out" in fails(ValueError, A.plan_sets, PATCHES, {**SIZES, "distil_heldout": 21})

# ------------------------------------------------------------------ a server on this machine

HITS = []              # (path, headers) of every request to the first server
AWAY = []              # the same for the second server, which stands for another host
PLAN = {}              # path -> answers to give before the file is served ("503", "429", "cut")
LISTING_STATE = {"empty": False}


class Handler(http.server.BaseHTTPRequestHandler):
    seen = HITS

    def log_message(self, *args):
        pass

    def plain(self, code: int, headers: dict | None = None, body: bytes = b"") -> None:
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def listing(self, path: str, query: str) -> None:
        if LISTING_STATE["empty"]:
            return self.plain(200, body=b"[]")
        entries = [{"type": "file", "path": p, "size": int(b), "oid": "x"}
                   for p, b in zip(LISTING["path"], LISTING["bytes"])]
        others = [{"type": "directory", "path": "data/old"},
                  {"type": "file", "path": "data/GLOBAL_ALL_MTD.zip", "size": 5},
                  {"type": "file", "path": "data/D004-2021_SPOT_RGBI.zip", "size": 7},
                  {"type": "file", "path": "data/D004-2021_AERIAL_RGBI.txt", "size": 9}]
        if query == "":
            link = f'<{BASE}{path}?cursor=2>; rel="next"'
            self.plain(200, {"Link": link}, json.dumps(entries[5:] + others).encode())
        else:
            self.plain(200, body=json.dumps(entries[:5]).encode())

    def serve(self, name: str, follow_range: bool = True, cut: bool = False, short: bool = False) -> None:
        file = DIR / "data" / name
        if not file.is_file():
            return self.plain(404)
        data = file.read_bytes()
        asked = self.headers.get("Range")
        if asked and follow_range:
            a, _, b = asked.removeprefix("bytes=").partition("-")
            start = max(0, len(data) - int(b)) if a == "" else int(a)
            end = len(data) if a == "" or b == "" else min(int(b) + 1, len(data))
            body, code = data[start:end], 206
            headers = {"Content-Range": f"bytes {start}-{end - 1}/{len(data)}"}
        else:
            body, code, headers = data, 200, {}
        if short:                                  # ten bytes short, and the answer says so
            body = body[:-10]
        self.send_response(code)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body[:len(body) // 2] if cut else body)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        self.seen.append((path, dict(self.headers)))
        kind, _, rest = path.lstrip("/").partition("/")
        name = rest.rpartition("/")[2]
        todo = PLAN.get(path) or []
        step = todo.pop(0) if todo else ""
        if "/api/datasets/" in path:
            self.listing(path, query)
        elif kind == "down" or step == "503":
            self.plain(503)
        elif step == "429":
            self.plain(429, {"RateLimit": '"resolvers";r=0;t=1'})
        elif kind == "moved":                  # as the Hub does, the file is on another host
            self.plain(302, {"Location": f"{ELSEWHERE}/files/{name}"})
        elif kind == "missing":
            self.plain(404)
        elif kind == "secret":
            self.plain(401)
        elif kind == "nosize":
            self.plain(206, body=b"PK\x05\x06" + bytes(18))
        else:
            self.serve(name, follow_range=kind != "norange", cut=step == "cut", short=kind == "short")


class Away(Handler):
    seen = AWAY


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{server.server_address[1]}"
away = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Away)
threading.Thread(target=away.serve_forever, daemon=True).start()
ELSEWHERE = f"http://127.0.0.1:{away.server_address[1]}"
# From here on a netrc file names a login for this machine. No request may carry it.
(DIR / "netrc").write_text("machine 127.0.0.1 login somebody password not-a-real-password\n")
os.environ["NETRC"] = str(DIR / "netrc")


def hits(path: str = "", where=HITS) -> list:
    return [h for p, h in where if not path or p == path]


def paths(where=HITS) -> list:
    return [p for p, _ in where]


def zip_hits(where=HITS) -> Counter:
    return Counter(p.rpartition("/")[2] for p in paths(where) if p.endswith(".zip"))


# ------------------------------------------------------------------ list_zips

M.HF = f"{BASE}/hub"
A._LISTINGS.clear()
HITS.clear()
listed = A.list_zips()
assert listed.equals(LISTING) and list(listed.columns) == list(A.LISTING_COLUMNS)
assert listed["bytes"].dtype == np.int64 and listed["year"].tolist()[0] == "2021"
api = f"/hub/api/datasets/{A.HF_REPO}/tree/{A.HF_REVISION}/data"
assert paths() == [api, api]                                    # two pages, the revision pinned
assert not any("Authorization" in h for h in hits())            # no token, netrc not read
HITS.clear()
assert A.list_zips().equals(LISTING) and HITS == []             # kept in memory
cache = DIR / "cache" / "zips.csv"
assert A.list_zips(cache=cache).equals(LISTING) and cache.is_file() and not list(cache.parent.glob("*.part"))
A._LISTINGS.clear()
from_cache = A.list_zips(cache=cache)
assert from_cache.equals(LISTING) and HITS == []                # the cache is returned as it is
assert from_cache["bytes"].dtype == np.int64
LISTING_STATE["empty"] = True
fails(FileNotFoundError, A.list_zips)
LISTING_STATE["empty"] = False
HITS.clear()
assert A.list_zips(token=FAKE).equals(LISTING) and all(h.get("Authorization") == f"Bearer {FAKE}" for h in hits())

# ------------------------------------------------------------------ list_patches, only ranges

HITS.clear()
before = dict(M.STATS)
patches = A.list_patches(LISTING, workers=3, progress=True)
assert list(patches.columns) == list(A.PATCH_COLUMNS) and len(patches) == len(TRUTH)
assert patches.equals(PATCHES)                                  # offsets, sizes, methods, crcs as zipfile reads them
assert len(hits()) == 12 and all(h["Range"] == f"bytes=-{A.TAIL_GUESS}" for h in hits())
assert M.STATS["bytes"] - before["bytes"] == LISTING["bytes"].sum()     # small files, fetched whole
assert not any("Authorization" in h for h in hits())
HITS.clear()
A.TAIL_GUESS = 300                                                      # two requests per zip now
assert A.list_patches(LISTING, workers=2, progress=False).equals(patches)
assert len(hits()) == 24 and sum(h["Range"] == "bytes=-300" for h in hits()) == 12
A.TAIL_GUESS = 1 << 21
cache = DIR / "cache" / "patches.csv"
HITS.clear()
assert A.list_patches(LISTING, cache=cache, progress=False).equals(patches) and cache.is_file()
assert len(hits()) == 12
HITS.clear()
assert A.list_patches(LISTING, cache=cache, progress=False).equals(patches) and HITS == []
assert A.list_patches(LISTING, token=FAKE, progress=False).equals(patches)
assert all(h.get("Authorization") == f"Bearer {FAKE}" for h in hits())
assert "no aerial zip" in fails(ValueError, A.list_patches, LISTING[LISTING["modality"] == "x"])
fails(KeyError, A.list_patches, LISTING.drop(columns="modality"))
# a zip that is not a zip, a size that differs from the listing, a member of another form
(DIR / "data" / "D005-2018_AERIAL_RGBI.zip").write_bytes(b"not a zip " * 100)
odd = pd.DataFrame([{"department": "D005", "year": "2018", "modality": "AERIAL_RGBI",
                     "path": "data/D005-2018_AERIAL_RGBI.zip", "bytes": 1000}])
assert "not a zip" in fails(ValueError, A.list_patches, odd, progress=False)
shutil.copy(rgbi, DIR / "data" / "D005-2018_AERIAL_RGBI.zip")           # holds D004 members
assert "is not D005-2018" in fails(ValueError, A.list_patches, odd.assign(bytes=os.path.getsize(rgbi)),
                                   progress=False)
assert "the listing says 1000" in fails(OSError, A.list_patches, odd, progress=False)
(DIR / "data" / "D005-2018_AERIAL_RGBI.zip").unlink()
assert "Content-Range" in fails(OSError, A._central_directory, f"{BASE}/nosize/x.zip")
# the zip64 form is read through its zip64 record
infos, size = A._central_directory(A._zip_url("data/D036-2020_AERIAL_RGBI.zip"))
assert size == os.path.getsize(DIR / "data" / "D036-2020_AERIAL_RGBI.zip")
assert [(i.filename, i.header_offset, i.compress_size, i.CRC) for i in infos] == \
    [(i.filename, i.header_offset, i.compress_size, i.CRC) for i in INFOS["D036-2020_AERIAL_RGBI"].values()]

# ------------------------------------------------------------------ fetch_member, fetch_patch

def fetch(patch_id: str, which: str = "rgbi", **more):
    r = row_of(patch_id)
    return A.fetch_patch(r[f"{which}_zip"], patch_id, r[f"{which}_offset"], r[f"{which}_size"],
                         r[f"{which}_method"], r[f"{which}_crc"], listing=LISTING, **more)


members = list(INFOS["D004-2021_AERIAL_RGBI"].values())
assert members[0].header_offset == 0 and members[-1].header_offset > members[5].header_offset
for info in (members[0], members[5], members[-1]):               # the first, a middle, the last
    patch_id = id_of(info)
    HITS.clear()
    before = dict(M.STATS)
    array, grid = fetch(patch_id)
    assert (array == TRUTH[patch_id][0]).all() and array.shape == (4, PX, PX)
    assert len(hits()) == 1 and M.STATS["row_requests"] - before["row_requests"] == 1
    want = f"bytes={info.header_offset}-{info.header_offset + 30 + len(info.filename) + A.LOCAL_SLACK + info.compress_size - 1}"
    assert hits()[0]["Range"] == want, (hits()[0]["Range"], want)
    assert M.STATS["row_bytes"] - before["row_bytes"] == 30 + len(info.filename) + A.LOCAL_SLACK + info.compress_size
    assert hits()[0]["Accept-Encoding"] == "identity" and "Authorization" not in hits()[0]
# the first member's local header carries a 20 byte extra field. With no slack it takes a second
# request for exactly the missing 20 bytes; with the slack it is one request
A.LOCAL_SLACK = 0
first_id = id_of(members[0])
HITS.clear()
array, _ = fetch(first_id)
assert (array == TRUTH[first_id][0]).all() and len(hits()) == 2
n = len(members[0].filename)
assert hits()[0]["Range"] == f"bytes=0-{30 + n + members[0].compress_size - 1}"
assert hits()[1]["Range"] == f"bytes={30 + n + members[0].compress_size}-{30 + n + 20 + members[0].compress_size - 1}"
HITS.clear()
array, _ = fetch(id_of(members[1]))
assert (array == TRUTH[id_of(members[1])][0]).all() and len(hits()) == 1   # no extra field, no slack needed
A.LOCAL_SLACK = 10 ** 6                                          # a range never reaches past the zip
HITS.clear()
array, _ = fetch(first_id)
assert (array == TRUTH[first_id][0]).all() and hits()[0]["Range"] == f"bytes=0-{os.path.getsize(rgbi) - 1}"
A.LOCAL_SLACK = 128
# a stored member, a zip64 zip, a label member
stored_id = "D012-2019_AF-S1-27_1-1"
assert row_of(stored_id)["rgbi_method"] == "0"
array, grid = fetch(stored_id)
assert (array == TRUTH[stored_id][0]).all()
z64_id = "D036-2020_ZZ-S1-9_2-2"
array, grid = fetch(z64_id)
assert (array == TRUTH[z64_id][0]).all()
label_array, label_grid = fetch(z64_id, "label")
assert (label_array == TRUTH[z64_id][1]).all() and label_array.shape == (PX, PX)
assert label_grid[1] == grid[1] and max(abs(a - b) for a, b in zip(label_grid[0], grid[0])) < 1e-9
# wrong fields are caught, nothing is guessed
r = row_of(first_id)
url = A._zip_url(r["rgbi_zip"])
name = A.member_name(r["rgbi_zip"], first_id)
good = dict(url=url, offset=r["rgbi_offset"], size=r["rgbi_size"], name=name, method=r["rgbi_method"], crc=r["rgbi_crc"])
assert A.fetch_member(**good) == zipfile.ZipFile(rgbi).read(name)
assert "holds" in fails(ValueError, A.fetch_member, **{**good, "offset": members[1].header_offset})
assert "no local file header" in fails(ValueError, A.fetch_member, **{**good, "offset": 7})
assert "CRC-32 differs" in fails(ValueError, A.fetch_member, **{**good, "crc": 12345})
assert "damaged" in fails(ValueError, A.fetch_member, **{**good, "size": 10})
assert "neither stored nor deflated" in fails(ValueError, A.fetch_member, **{**good, "method": 9})
assert "not a patch of" in fails(ValueError, A.fetch_patch, r["rgbi_zip"], stored_id, 0, 1, 8, 0, listing=LISTING)
# a cut answer is tried again and then fails, a wrong-size answer fails at once, both say why
M.HF = f"{BASE}/flaky"
flaky_path = f"/flaky/datasets/{A.HF_REPO}/resolve/{A.HF_REVISION}/{r['rgbi_zip']}"
PLAN[flaky_path] = ["cut"] * M.RETRIES
HITS.clear()
before = dict(M.STATS)
text = fails(ConnectionError, fetch, first_id)
assert f"{M.RETRIES} tries" in text and len(hits()) == M.RETRIES, text
assert M.STATS["retries"] - before["retries"] == M.RETRIES - 1
PLAN[flaky_path] = ["503", "cut", "429"]
HITS.clear()
start = time.perf_counter()
array, _ = fetch(first_id)
assert (array == TRUTH[first_id][0]).all() and len(hits()) == 4 and time.perf_counter() - start >= 1.0
M.HF = f"{BASE}/short"
HITS.clear()
text = fails(OSError, fetch, first_id)
assert "got" in text and "bytes" in text and len(hits()) == 1
M.HF = f"{BASE}/hub"
# a token goes to the Hub only, never to the host it sends us to, never into an error
HITS.clear()
array, _ = fetch(first_id, token=FAKE)
assert hits()[0]["Authorization"] == f"Bearer {FAKE}"
M.HF = f"{BASE}/moved"
HITS.clear()
AWAY.clear()
array, _ = fetch(first_id, token=FAKE)
assert (array == TRUTH[first_id][0]).all()
assert len(hits()) == 1 and hits()[0]["Authorization"] == f"Bearer {FAKE}"
assert len(AWAY) == 1 and "Authorization" not in AWAY[0][1] and AWAY[0][1]["Range"].startswith("bytes=0-")
for kind, error in (("missing", FileNotFoundError), ("secret", PermissionError), ("down", ConnectionError)):
    M.HF = f"{BASE}/{kind}"
    assert FAKE not in fails(error, fetch, first_id, token=FAKE)
M.HF = f"{BASE}/hub"

# ------------------------------------------------------------------ download_zip

work = DIR / "work"
HITS.clear()
before = dict(M.STATS)
path = "data/D045-2020_AERIAL_RGBI.zip"
got = A.download_zip(path, work, listing=LISTING)
assert got == work / "D045-2020_AERIAL_RGBI.zip" and got.read_bytes() == (DIR / "data" / got.name).read_bytes()
assert len(hits()) == 1 and hits()[0]["Range"] == "bytes=0-" and hits()[0]["Accept-Encoding"] == "identity"
assert not any("Authorization" in h for h in hits()) and not list(work.glob("*.part"))
assert M.STATS["bytes"] - before["bytes"] == got.stat().st_size
assert M.STATS["requests"] - before["requests"] == 1
HITS.clear()
assert A.download_zip(path, work, listing=LISTING) == got and HITS == []       # complete, skipped
got.write_bytes(b"short")
assert "remove it first" in fails(OSError, A.download_zip, path, work, listing=LISTING)
got.unlink()
assert "not in the listing" in fails(KeyError, A.download_zip, "data/x.zip", work, listing=LISTING)
# a stopped download goes on where it stopped
whole = (DIR / "data" / "D045-2020_AERIAL_RGBI.zip").read_bytes()
part = work / "D045-2020_AERIAL_RGBI.zip.part"
part.write_bytes(whole[:1000])
HITS.clear()
before = dict(M.STATS)
assert A.download_zip(path, work, listing=LISTING).read_bytes() == whole
assert len(hits()) == 1 and hits()[0]["Range"] == "bytes=1000-" and not part.exists()
assert M.STATS["bytes"] - before["bytes"] == len(whole) - 1000
got.unlink()
part.write_bytes(whole + b"more")                                  # longer than the file, thrown away
HITS.clear()
assert A.download_zip(path, work, listing=LISTING).read_bytes() == whole and hits()[0]["Range"] == "bytes=0-"
got.unlink()
# the listing is fetched when none is given
A._LISTINGS.clear()
HITS.clear()
assert A.download_zip(path, work).read_bytes() == whole and paths()[:2] == [api, api]
got.unlink()
# trouble is tried again, a cut answer goes on from the bytes that came (in whole chunks)
M.HF = f"{BASE}/flaky"
A.CHUNK = 1024
url_path = f"/flaky/datasets/{A.HF_REPO}/resolve/{A.HF_REVISION}/{path}"
PLAN[url_path] = ["503", "cut", "429"]
HITS.clear()
before = dict(M.STATS)
start = time.perf_counter()
assert A.download_zip(path, work, listing=LISTING).read_bytes() == whole
assert time.perf_counter() - start >= 1.0                            # the 429 asked for one second
ranges = [h["Range"] for h in hits()]
assert len(ranges) == 4 and ranges[:2] == ["bytes=0-", "bytes=0-"] and ranges[2] == ranges[3] != "bytes=0-"
assert M.STATS["retries"] - before["retries"] == 2 and M.STATS["wait_seconds"] - before["wait_seconds"] >= 1.0
assert ranges[2] == f"bytes={len(whole) // 2 // 1024 * 1024}-"
A.CHUNK = 1 << 22
got.unlink()
M.HF = f"{BASE}/down"
HITS.clear()
text = fails(ConnectionError, A.download_zip, path, work, listing=LISTING)
assert len(hits()) == M.RETRIES and "HTTP 503" in text and f"{M.RETRIES} tries" in text
assert not (work / "D045-2020_AERIAL_RGBI.zip").exists()
M.HF = f"{BASE}/norange"                                             # the range is ignored, 200
HITS.clear()
part.write_bytes(whole[:1000])
assert A.download_zip(path, work, listing=LISTING).read_bytes() == whole and len(hits()) == 1
got.unlink()
for kind, error in (("missing", FileNotFoundError), ("secret", PermissionError)):
    M.HF = f"{BASE}/{kind}"
    HITS.clear()
    fails(error, A.download_zip, path, work, listing=LISTING)
    assert len(hits()) == 1, kind
M.HF = f"{BASE}/hub"
assert "the listing says" in fails(OSError, A.download_zip, path, work,
                                   listing=LISTING.assign(bytes=LISTING["bytes"] + 1))
assert not list(work.glob("*.part"))
# a token goes to addresses under HF only, never to the host the Hub sends us to, never into an error
HITS.clear()
assert A.download_zip(path, work, token=FAKE, listing=LISTING).read_bytes() == whole
assert len(hits()) == 1 and hits()[0]["Authorization"] == f"Bearer {FAKE}"
got.unlink()
HITS.clear()
real_url = A._zip_url
A._zip_url = lambda p: f"{BASE}/hubx/{p}"                            # only starts like HF
assert A.download_zip(path, work, token=FAKE, listing=LISTING).read_bytes() == whole
assert len(hits()) == 1 and "Authorization" not in hits()[0]
A._zip_url = real_url
got.unlink()
M.HF = f"{BASE}/moved"
HITS.clear()
AWAY.clear()
assert A.download_zip(path, work, token=FAKE, listing=LISTING).read_bytes() == whole
assert len(hits()) == 1 and hits()[0]["Authorization"] == f"Bearer {FAKE}"
assert len(AWAY) == 1 and "Authorization" not in AWAY[0][1] and AWAY[0][1]["Range"] == "bytes=0-"
got.unlink()
AWAY.clear()
assert A.list_patches(LISTING, token=FAKE, progress=False).equals(patches)
assert len(AWAY) == 12 and not any("Authorization" in h for _, h in AWAY)
for kind, error in (("missing", FileNotFoundError), ("secret", PermissionError), ("down", ConnectionError)):
    M.HF = f"{BASE}/{kind}"
    assert FAKE not in fails(error, A.download_zip, path, work, FAKE, LISTING)
    assert FAKE not in fails(error, A._central_directory, A._zip_url(path), FAKE)

# ------------------------------------------------------------------ build_sets by ranges, load_set

M.HF = f"{BASE}/hub"
out = DIR / "out"
HITS.clear()
before = dict(M.STATS)
manifest = A.build_sets(plan, work, out, listing=LISTING, workers=3)
assert (out / "manifest.json").is_file() and json.loads((out / "manifest.json").read_text()) == manifest
assert manifest["revision"] == A.HF_REVISION and manifest["repo"] == A.HF_REPO
assert manifest["classes"] == list(A.CLASSES) and manifest["pixel_m"] == 0.4
assert sorted(manifest["sets"]) == sorted(A.SETS)
assert {n: s["count"] for n, s in manifest["sets"].items()} == plan["set"].value_counts().to_dict()
for name, group in plan.groupby("set"):
    zips = sorted(set(zip(group["department"], group["year"])))
    assert manifest["sets"][name]["files"] == [f"{name}__{d}-{y}.npz" for d, y in zips]
    assert manifest["sets"][name]["departments"] == sorted(set(group["department"]))
    shares = manifest["sets"][name]["class_shares"]
    assert (shares is None) == (name not in A.LABELLED)
    if shares is not None:
        assert len(shares) == 19 and abs(sum(shares) - 1) < 1e-4
assert sorted(p.name for p in out.iterdir()) == sorted(
    ["manifest.json"] + [f for s in manifest["sets"].values() for f in s["files"]])
# only range requests, one per member: every distinct planned patch of a zip, and the label
# members of the patches in labelled sets. No whole zip, nothing in work_dir
assert all("Range" in h and not h["Range"].endswith("-") for h in hits())
labelled = plan[plan["set"].isin(A.LABELLED)]
expected = Counter()
for r in plan.drop_duplicates("patch_id").itertuples():
    expected[r.rgbi_zip.rpartition("/")[2]] += 1
for r in labelled.drop_duplicates("patch_id").itertuples():
    expected[r.label_zip.rpartition("/")[2]] += 1
assert zip_hits() == expected, (zip_hits(), expected)
assert M.STATS["row_requests"] - before["row_requests"] == sum(expected.values()) == len(hits())
assert not work.exists() or not list(work.iterdir())
for name in A.SETS:
    loaded = A.load_set(out, name)
    group = plan[plan["set"] == name]
    assert sorted(loaded["ids"]) == sorted(group["patch_id"])       # every planned patch, once
    assert loaded["departments"].tolist() == [p.partition("-")[0] for p in loaded["ids"]]
    for file in manifest["sets"][name]["files"]:                     # per file, the order of the plan
        stem = file[len(name) + 2:-4]
        with np.load(out / file) as z:
            assert z["ids"].tolist() == group.loc[group["patch_id"].str.startswith(stem), "patch_id"].tolist()
            assert str(z["department"]) + "-" + str(z["year"]) == stem
            assert sorted(z.files) == sorted(["images", "ids", "department", "year"] + (["labels"] if name in A.LABELLED else []))
    assert loaded["images"].shape == (len(group), 4, PX // 2, PX // 2) and loaded["images"].dtype == np.uint8
    for i, patch_id in enumerate(loaded["ids"]):
        assert (loaded["images"][i] == A.halve_image(TRUTH[patch_id][0])).all()
    if name in A.LABELLED:
        assert loaded["labels"].shape == (len(group), PX // 2, PX // 2) and loaded["labels"].dtype == np.uint8
        for i, patch_id in enumerate(loaded["ids"]):
            assert (loaded["labels"][i] == A.halve_label(TRUTH[patch_id][1])).all()
    else:
        assert loaded["labels"] is None
assert "no set" in fails(KeyError, A.load_set, out, "other")
# a leftover file of another build under the right name with the right ids (another pixel size,
# or no labels for a labelled set) stops the build and names the file; nothing is rebuilt over it
stale = out / "probe__D004-2021.npz"
good_bytes = stale.read_bytes()
probe_d004 = plan.loc[(plan["set"] == "probe") & (plan["department"] == "D004"), "patch_id"].tolist()
A._save_npz(stale, {"images": np.zeros((len(probe_d004), 4, 8, 8), np.uint8), "labels": np.zeros((len(probe_d004), 8, 8), np.uint8),
                    "ids": np.array(probe_d004), "department": "D004", "year": "2021"})
text = fails(RuntimeError, A.build_sets, plan, work, out, listing=LISTING, progress=False)
assert stale.name in text and "another build" in text and f"expected uint8 ({len(probe_d004)}, 4, {PX // 2}, {PX // 2})" in text, text
A._save_npz(stale, {"images": np.zeros((len(probe_d004), 4, PX // 2, PX // 2), np.uint8),
                    "ids": np.array(probe_d004), "department": "D004", "year": "2021"})
text = fails(RuntimeError, A.build_sets, plan, work, out, listing=LISTING, progress=False)
assert stale.name in text and "no labels" in text, text
stale.write_bytes(good_bytes)
assert A.build_sets(plan, work, out, listing=LISTING, progress=False) == manifest
# load_set_from_tar gives the arrays of load_set straight out of a set's archive (packed as notebook 01
# packs it), nothing is unpacked. A member missing, a cut tar and a member of another build are named
tars = DIR / "tars"
tars.mkdir()
for name in A.SETS:
    files = manifest["sets"][name]["files"]
    with tarfile.open(tars / f"{name}.tar", "w") as tar:
        for file in files:
            tar.add(out / file, arcname=file)
    from_tar = A.load_set_from_tar(tars / f"{name}.tar", name, files)
    from_files = A.load_set(out, name)
    assert set(from_tar) == set(from_files) == {"images", "labels", "ids", "departments"}
    assert (from_tar["images"] == from_files["images"]).all() and from_tar["images"].dtype == np.uint8
    assert from_tar["ids"].tolist() == from_files["ids"].tolist()
    assert from_tar["departments"].tolist() == from_files["departments"].tolist()
    if name in A.LABELLED:
        assert (from_tar["labels"] == from_files["labels"]).all() and from_tar["labels"].dtype == np.uint8
    else:
        assert from_tar["labels"] is None and from_files["labels"] is None
assert sorted(p.name for p in tars.iterdir()) == sorted(f"{n}.tar" for n in A.SETS), "something was unpacked"
probe_files = manifest["sets"]["probe"]["files"]
with open(out / probe_files[0], "rb") as f:                              # headers from an open file object
    assert A._npz_headers(f, ("ids", "images"), "x")["images"] == A._npz_header(out / probe_files[0], "images")
text = fails(FileNotFoundError, A.load_set_from_tar, tars / "val.tar", "val",
             [*manifest["sets"]["val"]["files"], "val__D999-2020.npz"])
assert "does not hold ['val__D999-2020.npz']" in text and "notebook 01" in text
probe_tar = (tars / "probe.tar").read_bytes()
(tars / "probe.tar").write_bytes(probe_tar[:len(probe_tar) // 2])         # cut, its end is gone
try:
    A.load_set_from_tar(tars / "probe.tar", "probe", probe_files)
except (tarfile.ReadError, FileNotFoundError) as err:
    print(f"a cut tar stops load_set_from_tar with {type(err).__name__}")
else:
    raise SystemExit("a cut tar did not stop load_set_from_tar")
(tars / "probe.tar").write_bytes(probe_tar)
wrong = tars / "wrong"
wrong.mkdir()
A._save_npz(wrong / probe_files[0], {"images": np.zeros((len(probe_d004), 4, 8, 8), np.uint8),
                                     "labels": np.zeros((len(probe_d004), 8, 8), np.uint8),
                                     "ids": np.array(probe_d004), "department": "D004", "year": "2021"})
with tarfile.open(tars / "probe_bad.tar", "w") as tar:
    tar.add(wrong / probe_files[0], arcname=probe_files[0])
    for file in probe_files[1:]:
        tar.add(out / file, arcname=file)
text = fails(RuntimeError, A.load_set_from_tar, tars / "probe_bad.tar", "probe", probe_files)
assert "files of two builds" in text and f"probe_bad.tar::{probe_files[1]}" in text, text
print("load_set_from_tar: the arrays of load_set straight from the tar, nothing unpacked")
# load_set makes each array once at full size and copies the files in, so the peak is the set
# plus one file, not twice the set (5 GB for the 19,000 distillation patches); rows keep file order
mix = DIR / "mix"
mix.mkdir()
n_files, per, side = 8, 75, 64
rng = np.random.default_rng(5)
mix_files = [f"probe__D{k:03d}-2020.npz" for k in range(n_files)]
for k, file in enumerate(mix_files):
    A._save_npz(mix / file, {"images": rng.integers(0, 256, (per, 4, side, side), dtype=np.uint8),
                             "labels": rng.integers(0, 19, (per, side, side), dtype=np.uint8),
                             "ids": np.array([f"D{k:03d}-2020_AA-S1-1_{i}-1" for i in range(per)]),
                             "department": f"D{k:03d}", "year": "2020"})
(mix / "manifest.json").write_text(json.dumps({"sets": {"probe": {"files": mix_files}}}))
gc.collect()
tracemalloc.start()
loaded = A.load_set(mix, "probe")
peak = tracemalloc.get_traced_memory()[1]
tracemalloc.stop()
data = loaded["images"].nbytes + loaded["labels"].nbytes
assert loaded["images"].shape == (n_files * per, 4, side, side) and loaded["labels"].shape == (n_files * per, side, side)
assert peak < 1.5 * data, (peak, data)                                       # 2.0 x the data before the fix
assert loaded["ids"].tolist() == [f"D{k:03d}-2020_AA-S1-1_{i}-1" for k in range(n_files) for i in range(per)]
assert loaded["departments"].tolist() == [f"D{k:03d}" for k in range(n_files) for _ in range(per)]
with np.load(mix / mix_files[2]) as z:
    assert (loaded["images"][2 * per:3 * per] == z["images"]).all() and (loaded["labels"][2 * per:3 * per] == z["labels"]).all()
A._save_npz(mix / mix_files[3], {"images": np.zeros((per, 4, 32, 32), np.uint8), "labels": np.zeros((per, 32, 32), np.uint8),
                                 "ids": np.array([f"D003-2020_AA-S1-1_{i}-1" for i in range(per)]), "department": "D003", "year": "2020"})
assert "files of two builds" in fails(RuntimeError, A.load_set, mix, "probe")
# a second build fetches nothing, reads no patch, and gives the same manifest
HITS.clear()
real_fetch, real_iter, real_download = A.fetch_member, A.iter_patches, A.download_zip
A.fetch_member = A.iter_patches = A.download_zip = lambda *a, **k: fails(SystemExit, lambda: None)
again = A.build_sets(plan, work, out, listing=LISTING)
A.fetch_member, A.iter_patches, A.download_zip = real_fetch, real_iter, real_download
assert again == manifest and HITS == []
# a stopped build goes on: two files removed, only their members are fetched again
(out / "val__D012-2019.npz").unlink()
(out / "distil_train__D045-2020.npz").unlink()
HITS.clear()
assert A.build_sets(plan, work, out, listing=LISTING) == manifest
val_d012 = plan[(plan["set"] == "val") & (plan["department"] == "D012")]
train_d045 = plan[(plan["set"] == "distil_train") & (plan["department"] == "D045")]
assert zip_hits() == {"D012-2019_AERIAL_RGBI.zip": len(val_d012), "D012-2019_AERIAL_LABEL-COSIA.zip": len(val_d012),
                      "D045-2020_AERIAL_RGBI.zip": len(train_d045)}
# a file from another plan is caught (val of D012 built with the patches test has there)
(out / "val__D012-2019.npz").unlink()
swapped = plan.copy()
is_val = (swapped["set"] == "val") & (swapped["department"] == "D012")
is_test = (swapped["set"] == "test") & (swapped["department"] == "D012")
swapped.loc[is_val, list(A.PATCH_COLUMNS)] = swapped.loc[is_test, list(A.PATCH_COLUMNS)].to_numpy()
A.build_sets(swapped, work, out, listing=LISTING, progress=False)
assert "other patches than the plan" in fails(RuntimeError, A.build_sets, plan, work, out, listing=LISTING)
(out / "val__D012-2019.npz").unlink()
(out / "val__D036-2020.npz").unlink()
A.build_sets(plan, work, out, listing=LISTING, progress=False)
# a wrong member field in the plan is caught during the build, and no file is left half done
(out / "probe__D004-2021.npz").unlink()
wrong = plan.copy()
i = wrong[(wrong["set"] == "probe") & (wrong["department"] == "D004")].index[0]
wrong.loc[wrong["patch_id"] == wrong.loc[i, "patch_id"], "rgbi_crc"] = "1"
assert "CRC-32 differs" in fails(ValueError, A.build_sets, wrong, work, out, listing=LISTING, progress=False)
assert not (out / "probe__D004-2021.npz").exists() and not list(out.glob("*.part"))
mixed = plan.copy()
mixed.loc[mixed[mixed["department"] == "D004"].index[0], "rgbi_zip"] = "data/D004-2021_AERIAL_RGBI_x.zip"
assert "two different entries" in fails(ValueError, A.build_sets, mixed, work, out, listing=LISTING)
A.build_sets(plan, work, out, listing=LISTING, progress=False)
assert "how is" in fails(ValueError, A.build_sets, plan, work, out, listing=LISTING, how="other")
# a token goes along to the Hub, and a build from the csv plan is the same
HITS.clear()
shutil.rmtree(out)
assert A.build_sets(A.read_plan(DIR / "plan.csv"), work, out, token=FAKE, listing=LISTING, progress=False) == manifest
assert all(h.get("Authorization") == f"Bearer {FAKE}" for h in hits()) and len(hits()) == sum(expected.values())

# ------------------------------------------------------------------ build_sets from whole zips, the same bytes

out_zips = DIR / "out_zips"
HITS.clear()
before = dict(M.STATS)
from_zips = A.build_sets(plan, work, out_zips, listing=LISTING, how="zips")
assert from_zips == manifest
needed = {f"{d}-{y}_AERIAL_RGBI.zip" for d, y in set(zip(plan["department"], plan["year"]))}
needed |= {f"{d}-{y}_AERIAL_LABEL-COSIA.zip" for d, y in set(zip(labelled["department"], labelled["year"]))}
assert zip_hits() == {name: 1 for name in needed}                    # each zip fetched once, whole
assert all(h["Range"] == "bytes=0-" for h in hits())
assert M.STATS["row_requests"] - before["row_requests"] == 0
assert not list(work.iterdir())                                      # zips deleted after use
assert sorted(p.name for p in out_zips.iterdir()) == sorted(p.name for p in out.iterdir())
for file in sorted(p.name for p in out.iterdir()):
    assert (out / file).read_bytes() == (out_zips / file).read_bytes(), file    # byte-identical
HITS.clear()
shutil.rmtree(out_zips)
kept = A.build_sets(plan, work, out_zips, token=FAKE, listing=LISTING, how="zips", keep_zips=True, progress=False)
assert kept == manifest and sorted(p.name for p in work.iterdir()) == sorted(needed)
assert all(h.get("Authorization") == f"Bearer {FAKE}" for h in hits())
HITS.clear()
shutil.rmtree(out_zips)
assert A.build_sets(plan, work, out_zips, listing=LISTING, how="zips", progress=False) == manifest and HITS == []

# ------------------------------------------------------------------ licence text

text = A.print_attribution()
assert text == A.ATTRIBUTION and "Etalab Open Licence 2.0" in text and "IGN" in text
assert "https://huggingface.co/datasets/IGNF/FLAIR-HUB" in text and A.HF_REVISION[:12] in text
assert A.HF_UPDATED in text and "10.1016/j.isprsjprs.2026.04.017" in text and "0.4 m" in text

server.shutdown()
away.shutdown()
del os.environ["NETRC"]
M.HF = HF0

# ------------------------------------------------------------------ the real listing, only when allowed

if _setup.NET:
    M.BACKOFF_S = 1.0
    A.PATCH_PX, A.OUT_PX = 512, 256
    A._LISTINGS.clear()
    real = A.list_zips()
    assert len(real) == 148 and real["modality"].value_counts().to_dict() == {m: 74 for m in A.MODALITIES}
    assert set(real["department"]) == set(A.TRAIN_VAL_DEPARTMENTS) | set(A.TEST_DEPARTMENTS)
    assert real["path"].str.fullmatch(A.ZIP_NAME.pattern).all()
    rgbi_gb = real.loc[real["modality"] == "AERIAL_RGBI", "bytes"].sum() / 1e9
    assert 241 < rgbi_gb < 242, rgbi_gb
    two_years = real[real["modality"] == "AERIAL_RGBI"].groupby("department").size()
    assert sorted(two_years[two_years > 1].index) == ["D017", "D033", "D044", "D080"]
    assert not set(two_years[two_years > 1].index) & set(A.TEST_DEPARTMENTS)
    small = real[real["department"] == "D080"]
    small = small[small["year"] == "2017"]
    found = A.list_patches(small, workers=2)
    assert len(found) == 100 and (found["label_zip"] != "").all()
    assert found["patch_id"].iloc[0] == "D080-2017_UU-S1-25_1-1"
    assert (found["rgbi_method"] == "8").all() and (found["label_method"] == "8").all()
    infos, size = A._central_directory(A._zip_url("data/D032-2019_AERIAL_RGBI.zip"))     # zip64
    assert len(infos) == 14300 and size == 13950098401
    assert infos[0].filename == "D032-2019_AERIAL_RGBI/AA-S1-V17/D032-2019_AERIAL_RGBI_AA-S1-V17_1-1.tif"
    # three real patches by range, image and label, the CRC-32 checked inside
    before = dict(M.STATS)
    for r in found.iloc[[0, 49, 99]].itertuples():
        image, grid = A.fetch_patch(r.rgbi_zip, r.patch_id, r.rgbi_offset, r.rgbi_size, r.rgbi_method,
                                    r.rgbi_crc, listing=real)
        label, label_grid = A.fetch_patch(r.label_zip, r.patch_id, r.label_offset, r.label_size,
                                          r.label_method, r.label_crc, listing=real)
        assert image.shape == (4, 512, 512) and image.dtype == np.uint8
        assert label.shape == (512, 512) and label.max() <= 18
        A._check_label(label, label_grid, grid, r.patch_id)          # within GRID_TOL_M, as in the build
        assert grid[1] == "EPSG:2154" and abs(grid[0][0] - 0.2) < 1e-9
    paid = {k: M.STATS[k] - before[k] for k in M.STATS}
    assert paid["row_requests"] == 6 and paid["decoded"] == 6
    print(f"network checks done, the listing has {len(real)} aerial zips, RGBI {rgbi_gb:.1f} GB, "
          f"3 real patches by range in 6 requests, {paid['row_bytes'] / 1e6:.2f} MB")
    A.PATCH_PX, A.OUT_PX = PX, PX // 2
else:
    print("network checks skipped (SARTRANSFER_TEST_NET is not 1)")

shutil.rmtree(DIR, ignore_errors=True)
print("test_aerial: all checks passed")
