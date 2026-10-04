"""The coherence test - domains/coherence.py, run_domain's extra_skip, notebooks 04a and 04b.

Small tests only - tiny data, a stand-in encoder, no network. The bucket is a local folder of real GeoTIFF tiles in the
layout of the public data set (one-degree tiles of 1200 x 1200 px at 3 arcsec, EPSG:4326, named by the upper
left corner, uint8 coherence x 100 and uint16 backscatter, 0 = no data).

1. tile names and the pixel positions of a chip under both readings
2. sample - exact bilinear values of a linear field across a tile border, no data, a missing tile
3. chip_coherence - the mean over the seasons with data, a season missing
4. placement_check - the reading the chips were made with wins, the shifted floor is lower
5. build - the arrays, the chips without coherence, the record
6. run_domain extra_skip - the run name, the two-channel image skip, the checks before anything runs
7. notebooks 04a and 04b - the text, the rule, the sync cell, the flush cell, and dry runs (the coherence made
   and saved, made again on a second run only where missing, the words helps, no gain and not known)
"""
import _setup  # noqa: F401

import contextlib
import io
import json
import math
import os
import re
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.transform import from_origin

from _setup import TMP_BASE  # noqa: E402

from sartransfer import env as E  # noqa: E402
from sartransfer import sync as SYNC  # noqa: E402
from sartransfer.distil import stage0 as Z  # noqa: E402
from sartransfer.domains import coherence as C, hub as H, prepared as P, run as R  # noqa: E402
from sartransfer.models import encoders as encmod  # noqa: E402

TMP = TMP_BASE / "coherence"
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True)
FAKE_TOKEN = "hf_standin_coherence_0123456789"
PATCHED = []
TILES_2X2 = [(48, -1), (48, 0), (49, -1), (49, 0)]       # covers 47 to 49 N and 1 W to 1 E


def patch(obj, name, value):
    PATCHED.append((obj, name, getattr(obj, name)))
    setattr(obj, name, value)


def unpatch_all():
    while PATCHED:
        obj, name, value = PATCHED.pop()
        setattr(obj, name, value)


def fails(fn, exc, text=""):
    try:
        fn()
    except exc as e:
        assert text in str(e), f"expected {text!r} in {e!r}"
        return
    raise SystemExit(f"no {exc.__name__} was raised")


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        out = fn(*args, **kwargs)
    return out, buf.getvalue()


def write_bucket(root: Path, layers: dict, tiles=TILES_2X2, skip=()):
    """Tiles of the public layout under root. layers - {kind - fn(global rows, global cols) -> array}."""
    for top, left in tiles:
        name = C.tile_name(top, left)
        gr = (90 - top) * C.TILE_PX + np.arange(C.TILE_PX)[:, None]
        gc = (left + 180) * C.TILE_PX + np.arange(C.TILE_PX)[None, :]
        for kind, fn in layers.items():
            if (name, kind) in skip:
                continue
            arr = fn(gr, gc)
            path = root / name / f"{name}_{kind}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(path, "w", driver="GTiff", width=C.TILE_PX, height=C.TILE_PX, count=1,
                               dtype=arr.dtype, crs="EPSG:4326", nodata=0, compress="lzw",
                               transform=from_origin(left, top, 1 / C.TILE_PX, 1 / C.TILE_PX)) as dst:
                dst.write(arr, 1)


# ---- 1. names and positions -------------------------------------------------------------------------
assert C.tile_of(47.5, -0.5) == (48, -1) and C.tile_name(48, -1) == "N48W001"
assert C.tile_of(-25.07, 113.9) == (-25, 113) and C.tile_name(-25, 113) == "S25E113"
assert C.tile_name(*C.tile_of(-0.5, -0.5)) == "N00W001" and C.tile_name(*C.tile_of(0.5, 0.5)) == "N01E000"
lat, lon = C.chip_lat_lon(47.5, -0.5, "centre")
assert lat.shape == lon.shape == (128, 128)
assert abs(lat.mean() - 47.5) < 1e-9 and abs(lon.mean() + 0.5) < 1e-9, "the centre reading is not centred"
assert abs((lat[0, 0] - lat[1, 0]) * C.M_PER_DEG - 10.0) < 1e-6 and lat[0, 0] > lat[-1, 0], "north up, 10 m rows"
assert abs((lon[0, 1] - lon[0, 0]) * C.M_PER_DEG * math.cos(math.radians(47.5)) - 10.0) < 1e-6
lat_c, lon_c = C.chip_lat_lon(47.5, -0.5, "corner")
assert abs(lat_c[0, 0] - (47.5 - 5 / C.M_PER_DEG)) < 1e-12 and lat_c.max() < 47.5 and lon_c.min() > -0.5
lat_s, _ = C.chip_lat_lon(47.5, -0.5, "centre", shift_m=(1300.0, 0.0))
assert abs((lat_s - lat).mean() * C.M_PER_DEG - 1300.0) < 1e-6
fails(lambda: C.chip_lat_lon(0, 0, "middle"), ValueError, "anchor must be")
print("names and positions - tiles by their upper left corner, chips north up at 10 m, centre and corner")

# ---- 2. sample --------------------------------------------------------------------------------------
G0, C0 = int((90 - 48.0) * 1200), int((-0.3 + 180) * 1200)       # a chip on the border of the N49 and N48 tiles
RAMP = TMP / "ramp"


def ramp(gr, gc):
    return np.clip(60 + (gr - G0) + (gc - C0), 1, 250).astype(np.uint8)


write_bucket(RAMP, {"summer_vv_COH12": ramp})
tiles = C.Tiles(RAMP)
lat, lon = C.chip_lat_lon(48.0, -0.3, "centre")
got = C.sample(tiles, "summer_vv_COH12", lat, lon)
fr, fc = (90 - lat) * 1200 - 0.5, (lon + 180) * 1200 - 0.5
assert np.allclose(got, 60 + (fr - G0) + (fc - C0), atol=1e-3), "bilinear of a linear field is not exact"
assert tiles.reads == 2, f"a chip on the border reads both tiles, {tiles.reads}"
# no data - a block of zeros in the upper tile, neighbours of no data drop their weight
NODATA = TMP / "nodata"
write_bucket(NODATA, {"summer_vv_COH12": lambda gr, gc: np.where(gr < G0, 0, ramp(gr, gc)).astype(np.uint8)})
got = C.sample(C.Tiles(NODATA), "summer_vv_COH12", lat, lon)
assert np.isnan(got[fr < G0 - 1]).all(), "rows whose four cells are all no data must be nan"
assert np.isfinite(got[fr >= G0]).all(), "rows with a valid cell below them keep a value"
assert np.allclose(got[fr >= G0], (60 + (fr - G0) + (fc - C0))[fr >= G0], atol=1e-3), "valid rows stay exact"
# a missing tile is no data
MISSING = TMP / "missing"
write_bucket(MISSING, {"summer_vv_COH12": ramp}, skip={("N49W001", "summer_vv_COH12")})
got = C.sample(C.Tiles(MISSING), "summer_vv_COH12", lat, lon)
assert np.isnan(got[fr < G0 - 1]).all() and np.isfinite(got[fr > G0]).all()
print("sample - exact bilinear across a tile border, no data and a missing tile give nan")

# ---- 3. chip_coherence ------------------------------------------------------------------------------
SEASONS = TMP / "seasons"
write_bucket(SEASONS, {f"{s}_vv_COH12": (lambda off: lambda gr, gc: np.clip(ramp(gr, gc).astype(int) + off, 1, 250)
                                         .astype(np.uint8))(4 * k) for k, s in enumerate(C.SEASONS)},
             skip={("N49W001", "fall_vv_COH12"), ("N48W001", "fall_vv_COH12")})
coh = C.chip_coherence(C.Tiles(SEASONS), 48.0, -0.3, "centre")
base = 60 + (fr - G0) + (fc - C0)
assert np.allclose(coh, (base + 4) / 100.0, atol=1e-4), "the mean of winter, spring, summer (fall missing)"
u8 = C.to_u8(np.array([[0.0, 0.5, 1.0, np.nan, 1.7]]))
assert u8.dtype == np.uint8 and u8.tolist() == [[0, 128, 255, 0, 255]]
print("chip_coherence - the mean over the seasons with data, divided by 100; to_u8 0 to 255, no data 0")

# ---- 4. placement_check -----------------------------------------------------------------------------
rng = np.random.default_rng(0)
noise = rng.normal(0, 1, (2400, 2400))
for _ in range(1):                                   # a field with structure at about 600 m, one box mean of 7 cells,
    k = 3                                            # so a reading half a chip off sees another pattern
    c = np.pad(noise, ((k, k), (k, k)), mode="reflect").cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    noise = (c[2 * k + 1:, 2 * k + 1:] - c[:-2 * k - 1, 2 * k + 1:] - c[2 * k + 1:, :-2 * k - 1]
             + c[:-2 * k - 1, :-2 * k - 1])[:2400, :2400] / (2 * k + 1) ** 2
field = (noise - noise.mean()) / noise.std()
R0, Q0 = (90 - 49) * 1200, (-1 + 180) * 1200


def amp(gr, gc):
    return (3000 + 1500 * field[gr - R0, gc - Q0]).clip(100, 60000).astype(np.uint16)


def coh_field(off):
    return lambda gr, gc: (40 + 15 * field[gr - R0, gc - Q0] + off).clip(1, 100).astype(np.uint8)


FIELD = TMP / "field"
write_bucket(FIELD, {"summer_vv_AMP": amp, **{f"{s}_vv_COH12": coh_field(2 * k) for k, s in enumerate(C.SEASONS)}})
field_tiles = C.Tiles(FIELD)
lats = rng.uniform(47.15, 48.85, 24)
lons = rng.uniform(-0.85, 0.85, 24)


def chips_made(anchor):
    """The chips' VV in dB as the field gives it under one reading, with noise."""
    out = []
    for la, lo in zip(lats, lons):
        la_g, lo_g = C.chip_lat_lon(la, lo, anchor)
        out.append(10 * np.log10(C.sample(field_tiles, "summer_vv_AMP", la_g, lo_g)) + rng.normal(0, 0.3, (128, 128)))
    return np.stack(out).astype(np.float32)


for truth in ("centre", "corner"):
    vv = chips_made(truth)
    vv[0, :5, :5] = np.nan                           # missing pixels, as real chips have
    check, said = quiet(C.placement_check, field_tiles, vv, lats, lons, n=20, seed=0, workers=4)
    other = "corner" if truth == "centre" else "centre"
    m = check["medians"]
    assert check["anchor"] == truth and check["chips"] == 20, check
    assert m[truth] > 0.9 and m[truth] > m[other] + 0.2 and m[truth] > m["centre shifted 1.3 km"] + 0.2, m
print(f"placement_check - picks the reading the chips were made with (last medians {json.dumps({k: round(v, 2) for k, v in m.items()})})")

# ---- 5. build ---------------------------------------------------------------------------------------
build_lats, build_lons = np.array([47.5, 48.5, 47.9999]), np.array([-0.5, 0.5, 0.2])
arrays, record = quiet(C.build, field_tiles, build_lats, build_lons, "centre", workers=2, every=2)[0]
assert arrays.shape == (3, 128, 128) and arrays.dtype == np.uint8
for k in range(3):
    want = C.to_u8(C.chip_coherence(field_tiles, build_lats[k], build_lons[k], "centre"))
    assert np.array_equal(arrays[k], want), k
assert record["chips"] == 3 and record["chips_without_coherence"] == 0 and record["tile_windows_read"] >= 12
(_, record_out), said = quiet(C.build, C.Tiles(TMP / "empty_bucket"), np.array([10.5]), np.array([10.5]), "centre",
                              workers=1, every=1)
assert record_out["chips_without_coherence"] == 1 and record_out["pixels_without_coherence_share"] == 1.0
assert "1 of 1 chips" in said and "chips per second" in said
print("build - the arrays of chip_coherence, chips without coherence counted, the rate printed")

# ---- 6. run_domain extra_skip -----------------------------------------------------------------------
CLASSES = {0: "natural_forest", 1: "water", 2: "built"}


def toy(root: Path, name: str, place=None) -> Path:
    """A tiny prepared forest dataset, 64 px chips, train and val, a latitude and longitude per chip."""
    w = P.PreparedWriter(root, name, 64, CLASSES, source={"url": "synthetic"}, pixel_spacing_m=10.0)
    for split, n in (("train", 12), ("val", 4)):
        for i in range(n):
            yy, xx = np.mgrid[:64, :64]
            water = (yy - 32) ** 2 + (xx - int(rng.integers(16, 48))) ** 2 < 200
            built = (yy < 20) & (xx > 40)
            img = np.stack([np.where(water, -22.0, np.where(built, -4.0, -9.0)) + rng.normal(0, 1.5, (64, 64)),
                            np.where(water, -29.0, -16.0) + rng.normal(0, 1.0, (64, 64))]).astype(np.float32)
            lab = np.where(water, 1, np.where(built, 2, 0)).astype(np.uint8)
            lab[:3, :3] = 255
            la, lo = (47.2 + 0.1 * i, -0.8 + 0.1 * i) if place is None else place(split, i)
            w.add(split, img, lab, region=f"r{i % 2}", date="", lat=round(la, 6), lon=round(lo, 6))
    w.close()
    return root / name


class FakeEncoder:
    patch, dim, _n_prefix, n_layers = 16, 8, 0, 4
    spec = encmod.EncoderSpec("fake/model", "fake", "fake", batch=4)
    encoded = []

    def __init__(self, *a, **k):
        pass

    def encode(self, tiles_in, batch=None, layer=None):
        FakeEncoder.encoded.append(len(tiles_in))
        n, _, size, _ = tiles_in.shape
        gg = size // 16
        b = tiles_in[:, :2].reshape(n, 2, gg, 16, gg, 16)
        m, s = b.mean((3, 5)), b.std((3, 5))
        return np.concatenate([m, s, m * 0.5, -m], 1).transpose(0, 2, 3, 1).astype(np.float16)


HAD_ENV = {k: os.environ.get(k) for k in ("DATASET_HF_OWNER", "HF_WRITE_TOKEN")}
PREP = TMP / "prepared"
toy(PREP, "forty_v1")
patch(encmod, "FrozenEncoder", FakeEncoder)
encmod.ENCODERS["fake"] = FakeEncoder.spec
extra = {s: rng.integers(0, 256, (n, 64, 64)).astype(np.uint8) for s, n in (("train", 12), ("val", 4))}
KW = dict(token=None, layer=1, seeds=(0, 1), steps=30, batch=3, device="cpu", splits=("val",))
cache = {}
quiet(R.run_domain, PREP, "forty_v1", "fake", TMP / "res", feature_cache=cache, **KW)
_, said = quiet(R.run_domain, PREP, "forty_v1", "fake", TMP / "res", feature_cache=cache,
                extra_skip=("coherence", extra), **KW)
RUNX = "dom_forty_v1__fake__L1__vv_vh_diff__skip_coherence"
assert "reused" in said and json.loads((TMP / "res" / f"{RUNX}__info.json").read_text(encoding="utf-8"))[
    "extra_skip"] == "coherence"
headx = torch.load(TMP / "res" / f"{RUNX}__decoder__s0.pt")
head1 = torch.load(TMP / "res" / "dom_forty_v1__fake__L1__vv_vh_diff__decoder__s0.pt")
assert tuple(headx["stem.blocks.0.0.weight"].shape) == (16, 2, 3, 3), "the image skip has two channels"
assert tuple(head1["stem.blocks.0.0.weight"].shape) == (16, 1, 3, 3), "the plain run kept one channel"
rows = pd.read_csv(TMP / "res" / f"{RUNX}__summary.csv")
assert set(rows["head"]) == {"majority", "probe", "decoder"} and (rows["run"] == RUNX).all()
bad = {"train": extra["train"][:5], "val": extra["val"]}
fails(lambda: R.run_domain(PREP, "forty_v1", "fake", TMP / "res_bad", extra_skip=("coherence", bad), **KW),
      ValueError, "must be uint8 (12, 64, 64)")
fails(lambda: R.run_domain(PREP, "forty_v1", "fake", TMP / "res_bad", extra_skip=("coherence", extra),
                           exclude_train_sharing=("region",), **KW), ValueError, "cannot go with exclude_train_sharing")
assert not (TMP / "res_bad" / f"{RUNX}__summary.csv").exists()
print("run_domain extra_skip - __skip_coherence, a two-channel image skip, wrong arrays refused before running")

# ---- 7. notebooks 04a and 04b -----------------------------------------------------------------------
FORBIDDEN = _setup.private_names()            # empty in a public copy
nb03 = json.loads((_setup.REPO / "notebooks" / "03_score.ipynb").read_text(encoding="utf-8"))
SOURCES = {}
for name, ids, words in (
        ("04a_coherence.ipynb", ["intro", "mount", "sync", "setup", "placement", "coherence", "flush"],
         (" (written down 2026-10-04 before it was built)", "No GPU is needed",
          "centre or the upper left corner", "For 300 training chips (seed 0)", "shifted by 1.3 km",
          "No label is used", "12-day VV coherence of the four seasons of December 2019 to November 2020",
          "Chips without coherence get 0 and are counted", "Kellndorfer et al., Scientific Data 9, 73, 2022",
          "CC BY 4.0", "never whole tiles", "CPU runtime with High-RAM", "chips per second",
          "MyDrive/sar-transfer-distillation/coherence/forty_v1/", "is not made again", "END_RUNTIME = False")),
        ("04b_coherence_score.ipynb", ["intro", "mount", "sync", "setup", "runs", "verdict", "flush"],
         (" (written down 2026-10-04 before it was built)", "layer 21",
          "the seeds 0, 1 and 2", "a second channel of the decoder's image skip", "extracted once for both",
          "The probe is the same in both", "The decoder mIoU on val",
          "larger than 0.01 and larger than twice the larger seed SD", "date-matched coherence at 20 m for lakes and snow",
          "the coherence question stops here", "it does not decide", "The L4 GPU", "About 10 minutes, an estimate",
          "MyDrive/sar-transfer-distillation/coherence_score/", "Run notebook 04a first", "END_RUNTIME = False"))):
    nb = json.loads((_setup.REPO / "notebooks" / name).read_text(encoding="utf-8"))
    got_ids = [c.get("id") for c in nb["cells"]]
    assert got_ids == ids, (name, got_ids)
    src = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    SOURCES[name] = src
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"), f"{name} holds outputs"
    assert src["intro"].startswith("# ") and ":" not in src["intro"], f"a colon in the intro of {name}"
    for w in words:
        assert w in src["intro"], f"{name} does not say {w!r}"
    for cid, s in src.items():
        if cid != "sync":
            assert not [f for f in FORBIDDEN if f in s], f"{name} cell {cid} names another repository"
    for cid in ids[1:]:
        first = src[cid].splitlines()[0]
        assert first.startswith("# ") and ":" not in first, (name, cid, first)
        if cid != "sync":
            assert not [ln for ln in src[cid].splitlines() if ln.lstrip().startswith("#") and ":" in ln], (name, cid)
    assert src["flush"] == _setup.FLUSH_SOURCE, f"the last cell of {name} differs from the shared flush cell"
    assert src["mount"] == "".join(next(c for c in nb03["cells"] if c.get("id") == "mount")["source"])
    assert SYNC.cell_index(nb) == ids.index("sync")
    _setup.check_sync_cell(nb, SYNC)                    # packed and current, or cleared in a published copy
A, B = SOURCES["04a_coherence.ipynb"], SOURCES["04b_coherence_score.ipynb"]
for cell, s in ((A["setup"], "CHECK_CHIPS, CHECK_SEED = 300, 0"), (A["setup"], 'SPLITS = ("train", "val")'),
                (A["placement"], "coherence.placement_check(TILES, vv_db"), (A["coherence"], "coherence.build(TILES"),
                (B["setup"], 'ENCODER, MODE = "dinov3-l-sat", "vv_vh_diff"'), (B["setup"], "SEEDS = (0, 1, 2)"),
                (B["setup"], "LAYER = run_step.PRE_REGISTERED_LAYERS[ENCODER]"), (B["setup"], 'SPLITS = ("val",)'),
                (B["runs"], '"with coherence": ("coherence", COHERENCE)'), (B["runs"], "extra_skip=extra"),
                (B["verdict"], 'score.judge("decoder", new.mean(), new.std(ddof=1), ref.mean(), ref.std(ddof=1))')):
    assert s in cell, s
print("notebooks 04a and 04b - the text, the rule, no other repository, sync cells current, the shared flush cell")

# the dry runs
CONTENT = TMP / "content"
MYDRIVE = CONTENT / "drive" / "MyDrive"
NB_PREP = CONTENT / "sar-transfer-distillation" / "prepared"
COH_OUT = MYDRIVE / "sar-transfer-distillation" / "coherence" / "forty_v1"
SCORE_OUT = MYDRIVE / "sar-transfer-distillation" / "coherence_score"


def place(split, i):                                     # the toy chips inside the fake bucket, 0.13 degree apart
    k = i + (0 if split == "train" else 12)
    return 47.2 + 0.1 * (k % 16), -0.8 + 0.1 * (k // 2)


drive_copy = toy(TMP / "drive_source", "forty_v1", place=place)
# the toy chips' VV made from the field under the centre reading, so the placement check has something to find
images = np.load(drive_copy / "train_images.npy")
meta = P.read_meta(drive_copy / "train_meta.csv")
for k, (la, lo) in enumerate(zip(meta["lat"], meta["lon"])):
    la_g, lo_g = C.chip_lat_lon(la, lo, "centre", size=64)
    images[k, 0] = (10 * np.log10(C.sample(field_tiles, "summer_vv_AMP", la_g, lo_g))).astype(images.dtype)
np.save(drive_copy / "train_images.npy", images)
shutil.copytree(drive_copy, MYDRIVE / "datasets" / H.repo_name("forty_v1"))
CALLS = {"fetch": [], "run": [], "build": []}
REAL_RUN, REAL_BUILD, REAL_PLACE = R.run_domain, C.build, C.placement_check


class LocalTiles(C.Tiles):
    def __init__(self, root=None):
        assert root is None, "the notebook reads the public bucket"
        super().__init__(FIELD)


def chip_lat_lon_64(lat0, lon0, anchor="centre", size=C.CHIP_PX, pixel_m=C.PIXEL_M, shift_m=(0.0, 0.0)):
    return REAL_LAT_LON(lat0, lon0, anchor, size=64, pixel_m=pixel_m, shift_m=shift_m)


def build_counted(tiles_in, lats_in, lons_in, anchor, **kw):
    CALLS["build"].append((len(lats_in), anchor))
    out, rec = REAL_BUILD(tiles_in, lats_in, lons_in, anchor, workers=2, every=0)
    return out[:, :64, :64].copy(), rec                  # the toy chips are 64 px


def hub_down(dataset, owner, dest, token, splits=None, **kw):
    CALLS["fetch"].append((dataset, owner, splits))
    raise OSError(f"stand-in, the Hub refused the token {token}")


def run_on_cpu(prepared_root, dataset, encoder, res_dir, token, **kw):
    assert kw.pop("device") == "cuda" and token == FAKE_TOKEN and kw["seeds"] == (0, 1, 2) and kw["layer"] == 21, kw
    assert encoder == "dinov3-l-sat" and kw["mode"] == "vv_vh_diff" and kw["splits"] == ("val",)
    CALLS["run"].append(None if kw["extra_skip"] is None else kw["extra_skip"][0])
    return REAL_RUN(prepared_root, dataset, encoder, res_dir, token=None, device="cpu", steps=30, batch=3, **kw)


REAL_LAT_LON = C.chip_lat_lon
patch(C, "chip_lat_lon", chip_lat_lon_64)
patch(C, "CHIP_PX", 64)
patch(C, "Tiles", LocalTiles)
patch(C, "build", build_counted)
patch(Z, "WORK_FOLDERS", (str(CONTENT), str(MYDRIVE)))
patch(E, "model_cache_on_drive", lambda *a, **k: MYDRIVE / "hf_cache")
patch(E, "load_env", lambda *a, **k: {})
patch(E, "require", lambda name, why="": FAKE_TOKEN)
patch(H, "fetch", hub_down)
patch(R, "run_domain", run_on_cpu)
patch(torch.cuda, "is_available", lambda: True)
patch(torch.cuda, "get_device_name", lambda device=None: "stand-in L4")
os.environ["DATASET_HF_OWNER"] = "owner-stand-in"
os.environ.pop("HF_WRITE_TOKEN", None)


def run_cells(source: dict, which, names: dict, title: str) -> str:
    printed = []
    for name in which:
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                exec(source[name], names)
        except BaseException:
            print(out.getvalue())
            traceback.print_exc()
            raise SystemExit(f"{title}, cell {name} raised") from None
        said = out.getvalue()
        print(f"\n==== {title}, cell {name}")
        print("\n".join("    | " + line for line in said.splitlines()[-8:]))
        assert FAKE_TOKEN not in said, f"cell {name} printed the token"
        assert "cell time" in said.splitlines()[-1], f"cell {name} did not print its time"
        printed.append(said)
    return "\n".join(printed)


def set_forest(run: str, values):
    rows_ = pd.read_csv(SCORE_OUT / f"{run}__summary.csv")
    pick = (rows_["head"] == "decoder") & (rows_["split"] == "val")
    rows_.loc[pick, "mIoU"] = values
    rows_.to_csv(SCORE_OUT / f"{run}__summary.csv", index=False)


try:
    a1 = {}
    said = run_cells(A, ["setup", "placement", "coherence"], a1, "04a first run")
    assert CALLS["fetch"] == [("forty_v1", "owner-stand-in", ("train", "val"))]
    assert "forty_v1  not on Hugging Face (OSError, stand-in, the Hub refused the token <token>)" in said
    assert re.search(r"^forty_v1  from Drive in [\d.]+ min$", said, re.M)
    assert "the chip latitude and longitude are read as the centre, from 12 training chips" in said, said
    assert CALLS["build"] == [(12, "centre"), (4, "centre")], CALLS["build"]
    rec = json.loads((COH_OUT / "coherence.json").read_text(encoding="utf-8"))
    assert rec["anchor"] == "centre" and rec["placement"]["anchor"] == "centre" and set(rec["splits"]) == {"train", "val"}
    assert rec["source"]["licence"] == "CC BY 4.0" and rec["layer"] == "vv_COH12" and set(rec["times"]) == {"setup", "placement"}
    for split, n in (("train", 12), ("val", 4)):
        arr = np.load(COH_OUT / f"{split}_coherence.npy")
        assert arr.shape == (n, 64, 64) and arr.dtype == np.uint8 and arr.max() > 0, split
    a2 = {}
    CALLS["build"].clear()
    (COH_OUT / "val_coherence.npy").unlink()             # a cut run, val is made again, train is not
    said = run_cells(A, ["setup", "placement", "coherence"], a2, "04a second run")
    assert "the placement of an earlier run, from coherence.json" in said and "train on Drive already, 12 chips" in said
    assert CALLS["build"] == [(4, "centre")], CALLS["build"]
    print("04a dry run - Drive copy after the Hub failed, centre found, both splits saved, a second run makes only val")

    b1 = {}
    FakeEncoder.encoded.clear()
    said = run_cells(B, ["setup", "runs", "verdict"], b1, "04b first run")
    assert "forty_v1  on the runtime disk already" in said and "coherence train (12, 64, 64), 0 chips without" in said
    assert CALLS["run"] == [None, "coherence"] and len(FakeEncoder.encoded) == 2, (CALLS["run"], FakeEncoder.encoded)
    report = json.loads((SCORE_OUT / "coherence_score.json").read_text(encoding="utf-8"))
    assert report["verdict"] in ("helps", "no gain") and report["coherence"]["anchor"] == "centre"
    assert report["gpu"] == "stand-in L4" and report["seeds"] == [0, 1, 2]
    plain, withc = "dom_forty_v1__dinov3-l-sat__L21__vv_vh_diff", "dom_forty_v1__dinov3-l-sat__L21__vv_vh_diff__skip_coherence"
    kept = {r: (SCORE_OUT / f"{r}__summary.csv").read_bytes() for r in (plain, withc)}
    set_forest(plain, [0.30, 0.31, 0.32])
    set_forest(withc, [0.39, 0.39, 0.39])
    said = run_cells(B, ["verdict"], b1, "coherence higher")
    assert "COHERENCE HELPS - with coherence beats as it is on the forest mIoU by 0.0800" in said, said
    set_forest(withc, [0.30, 0.31, 0.32])
    said = run_cells(B, ["verdict"], b1, "coherence the same")
    assert "NO GAIN - with coherence against as it is on the forest mIoU is a tie (gap 0.0000" in said, said
    assert "The coherence question stops here" in said
    rows_ = pd.read_csv(SCORE_OUT / f"{withc}__summary.csv")
    rows_[rows_["head"] != "decoder"].to_csv(SCORE_OUT / f"{withc}__summary.csv", index=False)
    said = run_cells(B, ["verdict"], b1, "decoder rows missing")
    assert "NOT KNOWN - a decoder row is missing" in said
    for r, b in kept.items():
        (SCORE_OUT / f"{r}__summary.csv").write_bytes(b)
    b2 = {}
    FakeEncoder.encoded.clear()
    said = run_cells(B, ["setup", "runs", "verdict"], b2, "04b second run")
    assert said.count("skip dom_") == 2 and FakeEncoder.encoded == []
    print("04b dry run - both settings with one extraction, helps, no gain, not known, a second run trains nothing")
finally:
    unpatch_all()
    encmod.ENCODERS.pop("fake", None)
    for k, v in HAD_ENV.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
print("test_coherence: all checks passed")
