"""Scoring code of part four - part two's step under its own tests, the new modes and encoders, the aerial domain."""
import _setup  # noqa: F401

import ast
import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from _setup import REPO, TMP_BASE  # noqa: E402

TMP = TMP_BASE / "score"
TMP.mkdir(parents=True, exist_ok=True)

from sartransfer.domains import aerial as A, hub as H, prepared as P, run as R, s1input as S  # noqa: E402
from sartransfer.models import LAST, encoders as encmod  # noqa: E402
from sartransfer.distil import score as SC  # noqa: E402
from sartransfer.distil.teachers import dinov3_sat as D, terrafm as T  # noqa: E402

torch.manual_seed(0)


def fails(fn, exc, text=""):
    """`fn` must raise `exc` with `text` in its message."""
    try:
        fn()
    except exc as e:
        assert text in str(e), f"expected {text!r} in the message {e!r}"
        return
    raise SystemExit(f"no {exc.__name__} was raised")


# ---- 1. part two's test_contract on the copied prepared and s1input ------------------------------
rng = np.random.default_rng(0)
root = Path(tempfile.mkdtemp(dir=TMP))
w = P.PreparedWriter(root, "toy", 32, {0: "background", 1: "lake"}, source={"url": "synthetic"}, pixel_spacing_m=10.0)
P.CHUNK = 7                                               # force several part files
for split, n in (("train", 20), ("test", 9)):
    for i in range(n):
        img = rng.normal(-15, 4, (2, 32, 32)).astype(np.float32)
        img[1] -= 7
        img[:, :3, :3] = np.nan
        lab = (rng.random((32, 32)) < 0.3).astype(np.uint8)
        lab[:3, :3] = 255
        w.add(split, img, lab, region=f"r{i % 3}", date="2020-07-01")
man = w.close()
assert man["splits"] == {"test": 9, "train": 20}, man
assert not list((root / "toy").glob(".*part*")), "part files left behind"
d = P.load_split(root, "toy", "train")
assert d["images"].shape == (20, 2, 32, 32) and d["images"].dtype == np.float16 and d["labels"].dtype == np.uint8
assert list(d["meta"].chip_id[:2]) == ["train_0000000", "train_0000001"] and set(d["meta"].region) == {"r0", "r1", "r2"}
v = P.validate(root, "toy")
assert set(v.split) == {"train", "test"} and (v.nan_share > 0).all()
for bad in (dict(image=np.zeros((2, 16, 16)), label=np.zeros((32, 32))),
            dict(image=np.zeros((2, 32, 32)), label=np.full((32, 32), 7)),
            dict(image=np.full((2, 32, 32), np.inf), label=np.zeros((32, 32)))):
    try:
        P.PreparedWriter(root, "toy2", 32, {0: "a", 1: "b"}, source={}).add("train", **bad)
        raise SystemExit(f"accepted {bad}")
    except ValueError:
        pass
st = S.channel_stats(d["images"])
x = S.encoder_input(d["images"][:4], st)
assert x.shape == (4, 3, 32, 32) and x.dtype == np.float32 and np.isfinite(x).all() and (x[:, :, :3, :3] == 0).all()
g = S.encoder_input(d["images"][:4], st, "vv_grey")
assert np.array_equal(g[:, 0], g[:, 2])
sk = S.skip_image(d["images"][:4], st)
assert sk.shape == (4, 1, 32, 32) and sk.min() >= 0 and sk.max() <= 1
print("contract verified: writer parts + close, reader mmap, validate, bad input refused, S1 input modes, skip image")
P.CHUNK = 512                                             # as in a run of its own
assert man["dtype"] == "float16"                          # the SAR default of the extended writer

# ---- 2. part two's test_run on the copied run step (fake encoder, CPU) ----------------------------
RealFrozen = encmod.FrozenEncoder
rng = np.random.default_rng(0)
root2, res2 = Path(tempfile.mkdtemp(dir=TMP)), Path(tempfile.mkdtemp(dir=TMP))
w = P.PreparedWriter(root2, "toy_lakes", 64, {0: "background", 1: "lake"}, source={"url": "synthetic"}, pixel_spacing_m=10.0)
for split, n in (("train", 40), ("val", 8), ("test", 8), ("test_challenge", 5)):
    for i in range(n):
        yy, xx = np.mgrid[:64, :64]
        cy, cx, r = rng.integers(16, 48, 2).tolist() + [rng.integers(8, 20)]
        lake = ((yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2)
        vv = np.where(lake, -22.0, -9.0) + rng.normal(0, 1.5, (64, 64))
        vh = vv - 7 + rng.normal(0, 1.0, (64, 64))
        img = np.stack([vv, vh]).astype(np.float32)
        img[:, :4, :4] = np.nan
        lab = lake.astype(np.uint8)
        lab[:4, :4] = 255
        w.add(split, img, lab, region=f"RGI{i % 3}", date="2020-08-01")
w.close()


class FakeEncoder:
    patch, dim, _n_prefix, n_layers = 16, 8, 0, 4
    spec = encmod.EncoderSpec("fake/model", "fake", "fake", batch=4)
    def __init__(self, *a, **k): pass
    def encode(self, tiles, batch=None, layer=None):
        n, _, t, _ = tiles.shape
        g = t // 16
        b = tiles[:, :2].reshape(n, 2, g, 16, g, 16)
        m, s = b.mean((3, 5)), b.std((3, 5))
        return np.concatenate([m, s, m * 0.5, -m], 1).transpose(0, 2, 3, 1).astype(np.float16)   # standardised input -> O(1) features


encmod.ENCODERS["fake"] = FakeEncoder.spec
encmod.FrozenEncoder = FakeEncoder

out = R.run_domain(root2, "toy_lakes", "fake", res2, token=None, layer=1, seeds=(0, 1), steps=40, batch=4, device="cpu")
heads = sorted(set(zip(out["head"], out["seed"])))
assert heads == [("decoder", 0), ("decoder", 1), ("majority", -1), ("probe", -1)], heads
assert set(out.split) == {"val", "test", "test_challenge"}, set(out.split)
assert {"IoU_pos", "F1_pos", "mIoU", "accuracy"} <= set(out.columns)
probe = out[out["head"] == "probe"]
assert probe.patch_mIoU.notna().all() and (probe.mIoU > out[out["head"] == "majority"].mIoU.values).all(), probe
run = "dom_toy_lakes__fake__L1__vv_vh_diff"
for f in ("__summary.csv", "__probe.pt", "__decoder__s0.pt", "__decoder__s1.pt", "__info.json",
          "__probe__regions.csv", "__decoder__s1__regions.csv", "__majority__regions.csv"):
    assert (res2 / f"{run}{f}").exists(), f
reg = pd.read_csv(res2 / f"{run}__probe__regions.csv")
assert set(reg.region) == {"RGI0", "RGI1", "RGI2"} and set(reg.split) == {"val", "test", "test_challenge"}
info = json.loads((res2 / f"{run}__info.json").read_text())
assert info["train_chips_decoder"] == 40 and set(info["stats"]) == {"co", "cross", "diff"}
assert info["input_stats"] == info["stats"]                # no encoder statistics: the training split's, as in part two
again = R.run_domain(root2, "toy_lakes", "fake", res2, token=None, layer=1, seeds=(0, 1))
assert len(again) == len(out)                                       # skipped, summary returned
tab = R.domain_table(res2)
assert set(tab["head"]) == {"decoder", "majority", "probe"} and (tab[tab["head"] == "decoder"].seeds == 2).all(), tab
print(tab.round(3).to_string(index=False))

# per-chip counting (the scoring path since 2026-09-28) equals the reference CPU count, for split and region sums
for k in (2, 3, 8):
    t = rng.integers(0, k, (7, 40, 40)).astype(np.uint8)
    t[rng.random(t.shape) < 0.3] = 255
    p = rng.integers(0, k, t.shape).astype(np.uint8)
    cc = R._chip_conf(torch.from_numpy(p), torch.from_numpy(t), k)
    assert cc.shape == (7, k, k) and (cc.sum(0) == R._confusion(p, t, k)).all()
    assert (cc[2:5].sum(0) == R._confusion(p[2:5], t[2:5], k)).all()
# enc_cache: one encoder load shared by two runs (forced rerun of the same data counts as a second dataset)
loads = []
FakeEncoder.__init__ = lambda self, *a, **k: loads.append(1)
cache, res2b = {}, Path(tempfile.mkdtemp(dir=TMP))
for _ in range(2):
    R.run_domain(root2, "toy_lakes", "fake", res2b, token=None, layer=1, seeds=(0,), steps=10, batch=4, device="cpu",
                 force=True, enc_cache=cache)
assert len(loads) == 1 and "fake" in cache, (len(loads), list(cache))
print("run_domain verified: probe + decoder seeds + majority on every eval split, region tables, info, skip, domain_table")
encmod.FrozenEncoder = RealFrozen
assert R.PRE_REGISTERED_LAYERS["dinov3-l-sat"] == 21 and R.PRE_REGISTERED_LAYERS["cradio-v4-h"] == 32
assert R.PRE_REGISTERED_LAYERS["terrafm-b-s1"] == LAST and R.PRE_REGISTERED_LAYERS["dinov3-aerial"] == LAST
assert H.repo_id("glacial_lakes", "owner") == "owner/sar-transfer-glacial-lakes"

# ---- 3. the new modes co_cross and aerial_rgbn, values by hand ----------------------------------
img = np.array([[[[-10.0, -12.0], [np.nan, -8.0]], [[-17.0, -19.0], [-15.0, np.nan]]]], np.float32)   # (1, 2, 2, 2)
st = {"co": (-10.0, 2.0), "cross": (-17.0, 4.0), "diff": (7.0, 1.0)}
cc = S.encoder_input(img, st, "co_cross")
assert cc.shape == (1, 2, 2, 2) and cc.dtype == np.float32
assert np.allclose(cc[0, 0], [[0.0, -1.0], [0.0, 1.0]]) and np.allclose(cc[0, 1], [[0.0, -0.5], [0.5, 0.0]])
old = S.encoder_input(img, st)                    # the default mode is untouched: the same two channels plus the difference
assert old.shape == (1, 3, 2, 2) and np.array_equal(old[:, :2], cc) and old[0, 2, 0, 0] == 0.0 and old[0, 2, 0, 1] == 0.0
ae = np.zeros((2, 4, 16, 16), np.uint8)
ae[0] = np.array([10, 20, 30, 40], np.uint8)[:, None, None]
ae[1] = 255
astats = S.aerial_stats([10, 20, 30, 40], [5, 10, 15, 20])
assert list(astats) == list(S.AERIAL_KEYS) and astats["nir"] == (40.0, 20.0)
za = S.encoder_input(ae, astats, "aerial_rgbn")
assert za.shape == (2, 4, 16, 16) and za.dtype == np.float32 and np.allclose(za[0], 0.0)
assert np.allclose(za[1, :, 0, 0], [(255 - 10) / 5, (255 - 20) / 10, (255 - 30) / 15, (255 - 40) / 20])
# DINOv3's aerial statistics equal 8 bit / 255 and then its mean and std
dn = S.encoder_input(ae, encmod.DINOV3_AERIAL_STATS, "aerial_rgbn")
want = ((ae[:, :3].astype(np.float32) / 255.0 - np.array(D.MEAN, np.float32)[None, :, None, None])
        / np.array(D.STD, np.float32)[None, :, None, None])
assert np.allclose(dn[:, :3], want, atol=1e-5)
for bad in ((img, st, "aerial_rgbn"), (ae, ast, "co_cross"), (img, st, "rgb")):
    fails(lambda bad=bad: S.encoder_input(*bad), ValueError)
assert S.MODE_CHANNELS == {"vv_vh_diff": 3, "vv_grey": 3, "co_cross": 2, "aerial_rgbn": 4}
for bad_stats in (([1, 2, 3], [1, 1, 1]), ([1, 2, 3, 4], [1, 0, 1, 1]), ([1, 2, 3, np.nan], [1, 1, 1, 1])):
    fails(lambda bad_stats=bad_stats: S.aerial_stats(*bad_stats), ValueError)
sk = S.skip_image(ae, {"co": (10.0, 5.0)})        # the skip image is channel 0 standardised, for any channel count
assert sk.shape == (2, 1, 16, 16) and sk[0].max() == 0.5 and sk[1].max() == 1.0
print("modes verified: co_cross and aerial_rgbn by hand, DINOv3 aerial statistics = 8 bit / 255 then mean/std")

# ---- 4. the prepared writer with uint8 aerial chips ---------------------------------------------
root4 = Path(tempfile.mkdtemp(dir=TMP))
w = P.PreparedWriter(root4, "toy_air", 32, {0: "a", 1: "b"}, source={}, channels=A.AERIAL_CHANNELS, units=A.UNITS,
                     pixel_spacing_m=0.4, dtype=np.uint8)
for i in range(5):
    w.add("train", rng.integers(0, 256, (4, 32, 32), dtype=np.uint8), (rng.random((32, 32)) < 0.5).astype(np.uint8),
          region="all", date="")
man4 = w.close()
assert man4["dtype"] == "uint8" and man4["channels"] == ["R", "G", "B", "NIR"] and man4["splits"] == {"train": 5}
d4 = P.load_split(root4, "toy_air", "train")
assert d4["images"].shape == (5, 4, 32, 32) and d4["images"].dtype == np.uint8 and d4["labels"].dtype == np.uint8
v4 = P.validate(root4, "toy_air")
assert v4.nan_share.iloc[0] == 0 and {"R_p1", "NIR_p99"} <= set(v4.columns)
w2 = P.PreparedWriter(root4, "toy_air2", 32, {0: "a"}, source={}, channels=A.AERIAL_CHANNELS, dtype=np.uint8)
fails(lambda: w2.add("train", np.zeros((2, 32, 32)), np.zeros((32, 32), np.uint8)), ValueError, "expected (4, 32, 32)")
print("prepared verified: uint8 chips with 4 channels, manifest dtype, SAR default unchanged")

# ---- 4b. the student encoder against a REAL checkpoint of distil.train (3 steps on the CPU) ------
from sartransfer.distil.student import Student  # noqa: E402
from sartransfer.distil.teachers.base import TeacherOut  # noqa: E402
from sartransfer.distil.train import Target, TrainConfig, TrainData, train_student  # noqa: E402
from sartransfer.distil.train import load_student as real_load_student  # noqa: E402

SIDE, CROP = 48, 32
gt = np.random.default_rng(3)
REAL_STATS = {"s1": {"channels": ["VV", "VH"], "unit": "dB", "mean": [-14.0, -16.0], "sd": [5.0, 4.0]},
              "s2": {"mean": [4000.0] * 12, "sd": [2300.0] * 12},
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
        pooled = nn.functional.adaptive_avg_pool2d(x, self.grid).permute(0, 2, 3, 1)
        return TeacherOut(torch.tanh(x.mean((2, 3)) @ self.w_sum), torch.tanh(pooled @ self.w_spa))


real_dir = TMP / "real_student"
shutil.rmtree(real_dir, ignore_errors=True)
real_cfg = TrainConfig(steps=3, batch=8, n_aerial=2, crop=CROP, warmup=1, checkpoint_every=3, held_out_every=3,
                       phis_samples=4, bf16=False, student=dict(img=CROP, patch=16, dim=32, depth=2, heads=4))
real_data = TrainData(gt.normal(-15.0, 5.0, (40, 2, SIDE, SIDE)).astype(np.float16),
                      gt.integers(1, 8000, (40, 12, SIDE, SIDE), dtype=np.uint16),
                      gt.integers(0, 256, (12, 4, CROP, CROP), dtype=np.uint8), REAL_STATS)
with contextlib.redirect_stdout(io.StringIO()):
    real_res = train_student(real_cfg, real_data, [Target("ta", 8, 2, ToyTeacher(8, 2, 12, 1000.0, 10),
                                                           ToyTeacher(8, 2, 4, 255.0, 11))], real_dir, device="cpu")
real_ckpt = real_dir / "final.pt"
assert real_res["step"] == 3 and real_ckpt.is_file()
spec4 = encmod.student_spec("real-student", real_ckpt, batch=4)
enc = encmod.FrozenEncoder(spec4, token=None, device="cpu")
ref, ref_stats, ref_cfg = real_load_student(real_ckpt, "cpu")
assert isinstance(enc.model, Student) and ref_stats == REAL_STATS and ref_cfg == real_cfg and enc.config == real_cfg
assert enc.dim == 32 and enc.patch == 16 and enc.n_layers == 2
assert enc.input_stats == {"aerial_rgbn": S.aerial_stats(REAL_STATS["aerial"]["mean"], REAL_STATS["aerial"]["sd"])}
for kw, tiles in (("s1", gt.normal(0, 1, (5, 2, 64, 64))), ("aerial", gt.normal(0, 1, (5, 4, 32, 32))),
                  ("s1", gt.normal(0, 1, (3, 2, 128, 128))), ("aerial", gt.normal(0, 1, (2, 4, 96, 96)))):
    tiles = tiles.astype(np.float32)
    f = enc.encode(tiles, layer=LAST, batch=2)                       # grid tile / 16, position embedding interpolated
    with torch.inference_mode():
        _, sp = ref(**{kw: torch.from_numpy(tiles)})
        raw = ref.embed(**{kw: torch.from_numpy(tiles)}).permute(0, 2, 3, 1)
    assert f.shape == tuple(sp.shape) and np.array_equal(f, sp.numpy().astype(np.float16)), (kw, tiles.shape)
    assert not np.allclose(f.astype(np.float32), raw.numpy(), atol=1e-3)   # through the trunk, not the embedding
# run_domain feeds the student in co_cross with the DATASET training-split statistics
fed = {}


class Recording(encmod.FrozenEncoder):
    def encode(self, tiles, batch=None, layer=None):
        fed.setdefault(tiles.shape[1], []).append(np.array(tiles, copy=True))
        return super().encode(tiles, batch=batch, layer=layer)


encmod.FrozenEncoder = Recording
SC.register_student("real-student", real_ckpt, batch=4)
res4 = Path(tempfile.mkdtemp(dir=TMP))
with contextlib.redirect_stdout(io.StringIO()):
    R.run_domain(root2, "toy_lakes", "real-student", res4, token=None, seeds=(0,), mode="co_cross", steps=10, batch=4,
                 device="cpu")
lakes = {s: P.load_split(root2, "toy_lakes", s) for s in ("test", "test_challenge", "train", "val")}   # manifest order
tr_stats = S.channel_stats(lakes["train"]["images"])
want = np.concatenate([S.encoder_input(lakes[s]["images"], tr_stats, "co_cross") for s in lakes])
assert np.array_equal(np.concatenate(fed[2]), want), "co_cross input is not the training-split standardisation"
ck_s1 = {"co": (-14.0, 5.0), "cross": (-16.0, 4.0)}
assert not np.allclose(want, np.concatenate([S.encoder_input(lakes[s]["images"], ck_s1, "co_cross") for s in lakes]),
                       atol=1e-3)
encmod.FrozenEncoder = RealFrozen
print("student encoder verified on a real checkpoint: features == Student.forward spatial in s1 and aerial mode, "
      "co_cross fed with the dataset's training-split statistics, aerial statistics from the checkpoint")

# ---- 5. the student encoder with a stub student -------------------------------------------------


class StubStudent(nn.Module):
    """The attributes and the forward of distil.student.Student with the aerial input, tiny."""

    def __init__(self, dim=8):
        super().__init__()
        self.patch, self.dim, self.img = 16, dim, 256
        self.chans = {"s1": 2, "s2": 12, "aerial": 4}
        self.embed_s1 = nn.Conv2d(2, dim, 16, stride=16)
        self.embed_aerial = nn.Conv2d(4, dim, 16, stride=16)
        self.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.calls = []

    def forward(self, s1=None, s2=None, aerial=None):
        if s2 is not None:
            raise ValueError("the stub reads s1 or aerial")
        x = self.embed_s1(s1) if s1 is not None else self.embed_aerial(aerial)
        self.calls.append("s1" if s1 is not None else "aerial")
        return x.mean((2, 3)), x.permute(0, 2, 3, 1)                # summary (B, dim), spatial (B, g, g, dim)


STUB_STATS = {"s1": {"channels": ["VV", "VH"], "unit": "dB", "mean": [-12.0, -19.0], "sd": [5.0, 6.0]},
              "aerial": {"channels": ["R", "G", "B", "NIR"], "unit": "8 bit",
                         "mean": [100.0, 110.0, 90.0, 120.0], "sd": [50.0, 45.0, 40.0, 55.0]}}
loaded = []


def load_student(path, device):
    """The signature of distil.train.load_student, written by another agent at the same time."""
    loaded.append((str(path), device))
    torch.manual_seed(0)
    return StubStudent().to(device), STUB_STATS, {"depth": 2, "dim": 8}


train_stub = types.ModuleType("sartransfer.distil.train")
train_stub.load_student = load_student
sys.modules["sartransfer.distil.train"] = train_stub
ckpt = TMP / "stub_student.pt"
ckpt.write_bytes(b"stub")
spec5 = encmod.student_spec("stub-student", ckpt, batch=4)
assert spec5.kind == "student" and spec5.in_channels == (2, 4) and spec5.model_id == str(ckpt) and spec5.input_stats == {}
enc = encmod.FrozenEncoder(spec5, token=None, device="cpu")
assert loaded == [(str(ckpt), "cpu")] and enc.dtype == torch.float32 and enc.patch == 16 and enc.dim == 8
assert enc.input_stats == {"aerial_rgbn": S.aerial_stats(STUB_STATS["aerial"]["mean"], STUB_STATS["aerial"]["sd"])}
assert spec5.input_stats == {}                      # the spec is not written into
s1 = rng.normal(0, 1, (3, 2, 64, 64)).astype(np.float32)
f = enc.encode(s1, layer=LAST)
assert f.shape == (3, 4, 4, 8) and f.dtype == np.float16 and enc.model.calls == ["s1"] and enc._n_prefix == 0
fa = enc.encode(rng.normal(0, 1, (5, 4, 32, 32)).astype(np.float32), batch=2)
assert fa.shape == (5, 2, 2, 8) and enc.model.calls == ["s1", "aerial", "aerial", "aerial"]
with torch.inference_mode():                         # the grid is the student's own spatial output, row by row
    _, sp = enc.model(s1=torch.from_numpy(s1))
assert np.array_equal(f, sp.numpy().astype(np.float16))
assert enc.n_layers == 2 and "student" in repr(enc)
for bad in (np.zeros((1, 3, 32, 32), np.float32), np.zeros((1, 12, 32, 32), np.float32)):
    fails(lambda bad=bad: enc.encode(bad), ValueError, "expected (N, 2 or 4, H, W)")
fails(lambda: enc.encode(s1, layer=1), NotImplementedError, "use layer 'last'")
fails(lambda: encmod._mean_sd({"s1": {}}, "aerial"), ValueError, "no 'aerial' entry")
assert encmod._mean_sd({"aerial": {"mean": [1], "std": [2]}}, "aerial") == ([1], [2])
fails(lambda: encmod.FrozenEncoder(encmod.EncoderSpec("x", "x", "unknown"), device="cpu"), ValueError, "unknown encoder kind")
print("student encoder verified: load_student stub, mode by channel count, aerial statistics from the checkpoint, grid")

# ---- 6. TerraFM-B reading Sentinel-1, tiny vendored model ---------------------------------------
orig_build = T.build


def tiny_build(**kw):
    return orig_build(**{**kw, "tiny": True, "weights_dir": None})


T.build = tiny_build
try:
    spec6 = encmod.ENCODERS["terrafm-b-s1"]
    assert spec6.kind == "terrafm-s1" and spec6.in_channels == (2,) and spec6.batch == 16
    assert spec6.input_stats == {}, "TerraFM-B is fed with the training-split statistics, not its own (2026-09-30)"
    assert encmod.TERRAFM_S1_STATS == {"co": (T.PR_MEAN_S1[0], T.PR_STD_S1[0]),
                                       "cross": (T.PR_MEAN_S1[1], T.PR_STD_S1[1])}   # kept as the record only
    enc = encmod.FrozenEncoder(spec6, token=None, device="cpu")
    assert enc.patch == 16 and enc.dim == 24 and enc.weights == "random" and enc.n_layers == 2
    assert enc.input_stats == {}
    used, seen = [], []

    def record(name):
        def hook(m, i, o):                           # returns None, so the output stays as it is
            used.append(name)
            seen.append(i[0].clone())
        return hook

    hooks = [getattr(enc.model.patch_embed, n).register_forward_hook(record(n))
             for n in ("conv2d_s2_l2a", "conv2d_s2_l1c", "conv2d_s1")]
    db = rng.normal(-15, 5, (3, 2, 64, 64)).astype(np.float32)               # dB tiles (VV, VH)
    db[1, :, :5, :5] = np.nan                                                # some no-data
    train_stats = S.channel_stats(db)                                        # the dataset's training-split statistics
    assert abs(train_stats["co"][0] - (-15)) < 0.5 and abs(train_stats["co"][0] - T.PR_MEAN_S1[0]) > 1.0
    x6 = S.encoder_input(db, train_stats, "co_cross")                        # what run_domain feeds (no input_stats)
    f = enc.encode(x6, layer=LAST)
    assert f.shape == (3, 4, 4, 24) and f.dtype == np.float16 and used == ["conv2d_s1"] and enc._n_prefix == 0
    z = np.stack([(db[:, 0] - train_stats["co"][0]) / train_stats["co"][1],
                  (db[:, 1] - train_stats["cross"][0]) / train_stats["cross"][1]], axis=1)
    z = np.where(np.isfinite(z), z, 0.0).astype(np.float32)                  # training-split standardisation, by hand
    minmax = z / (2 * T.PR_MINMAX_SCALE) + 0.5                               # the min-max mapping, by hand
    assert np.allclose(seen[0].numpy(), minmax, atol=1e-5), "the fed tiles differ from standardised / 4 + 0.5"
    one = ((db[0, 1, 0, 0] - train_stats["cross"][0]) / train_stats["cross"][1]) / (2 * T.PR_MINMAX_SCALE) + 0.5
    assert abs(float(seen[0][0, 1, 0, 0]) - one) < 1e-5
    assert np.allclose(seen[0].numpy()[1, :, :5, :5], 0.5)                   # no-data = the training mean = 0.5
    with_pr = (db - (np.array(T.PR_MEAN_S1, np.float32)[None, :, None, None] - 2 * np.array(T.PR_STD_S1, np.float32)[None, :, None, None])) / (4 * np.array(T.PR_STD_S1, np.float32)[None, :, None, None])
    assert not np.allclose(np.nan_to_num(with_pr[0]), seen[0].numpy()[0], atol=1e-3), "the pull request's statistics are still in use"
    nan_tile = np.full((1, 2, 32, 32), np.nan, np.float32)                     # no data = the mean = 0.5 under min-max
    enc.encode(S.encoder_input(nan_tile, train_stats, "co_cross"))
    assert np.allclose(seen[-1].numpy(), 0.5)
    for h in hooks:
        h.remove()
    with torch.inference_mode():                     # the vendored model's own call on the min-max input, last block, normed
        (ref,) = enc.model.get_intermediate_layers(torch.from_numpy(minmax), n=1, norm=True)
    assert np.allclose(f.astype(np.float32), ref.reshape(3, 4, 4, 24).numpy(), atol=1e-2)
    assert enc.encode(np.ascontiguousarray(x6[:, :, :32, :32])).shape == (3, 2, 2, 24)    # grid = tile / 16 at any size
    fails(lambda: enc.encode(np.zeros((1, 3, 32, 32), np.float32)), ValueError, "expected (N, 2, H, W)")
    fails(lambda: enc.encode(x6, layer=2), NotImplementedError)
finally:
    T.build = orig_build
print("terrafm-s1 verified: conv2d_s1 used, min-max rule by hand, no-data = 0.5, last block after the norm, grid tile / 16")

# ---- 7. DINOv3 (tiny) reading aerial R, G, B, NIR, the NIR channel dropped ----------------------
folder = TMP / "dinov3_tiny"
with contextlib.redirect_stdout(io.StringIO()):
    D.build(tiny=True, device="cpu", dtype=torch.float32).model.save_pretrained(folder)
assert encmod.ENCODERS["dinov3-aerial"].in_channels == (4,) and encmod.ENCODERS["dinov3-l-sat"].in_channels == (3,)
assert encmod.ENCODERS["dinov3-aerial"].model_id == encmod.ENCODERS["dinov3-l-sat"].model_id
spec7 = encmod.EncoderSpec(str(folder), "dinov3-tiny", "dinov3", batch=4, in_channels=(3, 4),
                           input_stats=dict(encmod.ENCODERS["dinov3-aerial"].input_stats))
enc = encmod.FrozenEncoder(spec7, token=None, device="cpu")
assert enc.dim == 32 and enc.patch == 16 and enc.n_layers == 2 and enc.dtype == torch.float32
rgbn = S.encoder_input(rng.integers(0, 256, (2, 4, 32, 32), dtype=np.uint8), spec7.input_stats["aerial_rgbn"], "aerial_rgbn")
f4 = enc.encode(rgbn, layer=LAST)
f3 = enc.encode(np.ascontiguousarray(rgbn[:, :3]))
assert f4.shape == (2, 2, 2, 32) and np.array_equal(f4, f3) and enc._n_prefix == 5
assert np.allclose(enc.encode(rgbn, layer=2).astype(np.float32), f4.astype(np.float32), atol=1e-2)   # block 2 = final output
fails(lambda: enc.encode(np.zeros((1, 2, 32, 32), np.float32)), ValueError, "expected (N, 3 or 4, H, W)")
print("dinov3 aerial verified: 4-channel tiles give the features of their RGB, prefix 5")

# ---- 8. the aerial converter, then run_domain and score_all on the CPU --------------------------


def toy_npz(path, n, size=32, seed=0):
    """Chips with two brightness regions; source codes 1, 2, 3 and 0 = no data in one corner."""
    r = np.random.default_rng(seed)
    images = np.zeros((n, 4, size, size), np.uint8)
    labels = np.zeros((n, size, size), np.uint8)
    for i in range(n):
        a, b = r.choice([1, 2, 3], 2, replace=False)
        cut = int(r.integers(8, size - 8))
        lab = np.full((size, size), a, np.uint8)
        lab[:, cut:] = b
        level = np.select([lab == 1, lab == 2, lab == 3], [40.0, 120.0, 210.0]).astype(np.float32)
        images[i] = np.clip(level[None] + r.normal(0, 8, (4, size, size)), 0, 255).astype(np.uint8)
        lab[:4, :4] = 0
        labels[i] = lab
    ids = np.array([f"D{i % 3:03d}_patch{i:04d}" for i in range(n)])
    np.savez(path, images=images, labels=labels, ids=ids)


npz_dir = TMP / "npz"
npz_dir.mkdir(exist_ok=True)
files = {}
for split, n, seed in (("train", 24, 1), ("val", 6, 2), ("test", 6, 3)):
    files[split] = npz_dir / f"{split}.npz"
    toy_npz(files[split], n, seed=seed)
root8 = Path(tempfile.mkdtemp(dir=TMP))
classes = {1: "dark", 2: "mid", 3: "bright"}
man8 = A.aerial_to_prepared(files, root8, classes, dataset="toy_aerial", nodata=0,
                            region_of=lambda pid: pid.split("_")[0], source={"name": "synthetic"})
assert man8["classes"] == {0: "dark", 1: "mid", 2: "bright"} and man8["chip_size"] == 32
assert man8["dtype"] == "uint8" and man8["units"] == "8 bit" and man8["splits"] == {"test": 6, "train": 24, "val": 6}
assert man8["stats"] == {"source_class_codes": {"0": 1, "1": 2, "2": 3}, "nodata_code": 0}
assert man8["channels"] == ["R", "G", "B", "NIR"] and man8["pixel_spacing_m"] == 0.4
d8 = P.load_split(root8, "toy_aerial", "train")
assert d8["images"].dtype == np.uint8 and d8["images"].shape == (24, 4, 32, 32)
assert (d8["labels"][:, :4, :4] == 255).all() and set(np.unique(d8["labels"]).tolist()) == {0, 1, 2, 255}
assert set(d8["meta"].region) == {"D000", "D001", "D002"} and d8["meta"].source_id[0] == "D000_patch0000"
with np.load(files["train"]) as z:                  # class ids follow the sorted source codes
    assert np.array_equal(d8["labels"][0][4:, 4:], z["labels"][0][4:, 4:] - 1)
    assert np.array_equal(d8["images"][3], z["images"][3])
fails(lambda: A.aerial_to_prepared(files, Path(tempfile.mkdtemp(dir=TMP)), {0: "x", 1: "y"}, dataset="bad", nodata=0),
      ValueError, "also a class")
fails(lambda: A.aerial_to_prepared(files, Path(tempfile.mkdtemp(dir=TMP)), {1: "dark", 2: "mid"}, dataset="bad", nodata=0),
      ValueError, "neither a class")
fails(lambda: A.aerial_to_prepared({"val": files["val"]}, root8, classes, dataset="bad"), ValueError, "no 'train' set")
print("aerial converter verified: uint8 prepared dataset, codes to 0..k-1, nodata to 255, regions, stray codes refused")

# the layout distil.aerial.build_sets writes (one file per set and department-year, the department per file), read
# through load_set, with the real class table - 19 codes, no nodata code; the region is the department
from sartransfer.distil import aerial as AER  # noqa: E402

CLASSES19 = dict(enumerate(AER.CLASSES))
assert len(CLASSES19) == 19 and AER.IGNORE is None and AER.N_CLASSES == 19
built = TMP / "built"
shutil.rmtree(built, ignore_errors=True)
built.mkdir()
sets = {}
with np.load(files["train"]) as z:
    im_all, lb_all = z["images"], z["labels"]
for name, n_per, start in (("probe", 6, 0), ("val", 3, 12)):
    fl = []
    for j, dept in enumerate(("D012", "D015")):
        pick = slice(start + j * n_per, start + (j + 1) * n_per)
        ids = np.array([f"{dept}-2019_AF-S1-27_{i}-{i}" for i in range(n_per)])
        fn = AER._file_name(name, dept, "2019")
        AER._save_npz(built / fn, {"images": im_all[pick], "ids": ids, "department": dept, "year": "2019",
                                   "labels": lb_all[pick]})
        fl.append(fn)
    sets[name] = {"count": 2 * n_per, "files": fl}
(built / "manifest.json").write_text(json.dumps({"sets": sets}))
probe_set, val_set = AER.load_set(built, "probe"), AER.load_set(built, "val")
assert probe_set["images"].shape == (12, 4, 32, 32) and list(probe_set["departments"]) == ["D012"] * 6 + ["D015"] * 6
root8b = Path(tempfile.mkdtemp(dir=TMP))
man8b = A.aerial_to_prepared({"train": probe_set, "val": val_set}, root8b, CLASSES19, dataset="flair_sets")
assert man8b["classes"] == CLASSES19 and man8b["splits"] == {"train": 12, "val": 6}
assert man8b["stats"] == {"source_class_codes": {str(i): i for i in range(19)}, "nodata_code": None}
d8b = P.load_split(root8b, "flair_sets", "train")
assert np.array_equal(d8b["images"], probe_set["images"]) and np.array_equal(d8b["labels"], probe_set["labels"])
assert list(d8b["meta"].region) == list(probe_set["departments"]) and list(d8b["meta"].source_id) == list(probe_set["ids"])
fails(lambda: A.aerial_to_prepared({"train": {"images": probe_set["images"], "labels": None, "ids": probe_set["ids"]}},
                                   Path(tempfile.mkdtemp(dir=TMP)), CLASSES19, dataset="bad"), ValueError, "no labels")
fails(lambda: A.aerial_to_prepared({"train": dict(probe_set, departments=probe_set["departments"][:3])},
                                   Path(tempfile.mkdtemp(dir=TMP)), CLASSES19, dataset="bad"), ValueError, "departments")
print("aerial converter verified on load_set's arrays: several files per set, region = department, the 19 codes kept")

fails(lambda: SC.register_student("dinov3-l-sat", ckpt), ValueError, "already the encoder")
fails(lambda: SC.register_student("nobody", TMP / "missing.pt"), FileNotFoundError)
SC.register_student("stub-student", ckpt, batch=4)
assert R.PRE_REGISTERED_LAYERS["stub-student"] == LAST and encmod.ENCODERS["stub-student"].kind == "student"
assert SC.mode_for(encmod.ENCODERS["stub-student"], 2) == "co_cross" and SC.mode_for(encmod.ENCODERS["stub-student"], 4) == "aerial_rgbn"
assert SC.mode_for(encmod.ENCODERS["dinov3-l-sat"], 2) == "vv_vh_diff" and SC.mode_for(encmod.ENCODERS["dinov3-l-sat"], 4) is None
assert SC.mode_for(encmod.ENCODERS["terrafm-b-s1"], 2) == "co_cross" and SC.mode_for(encmod.ENCODERS["terrafm-b-s1"], 4) is None
assert SC.mode_for(encmod.ENCODERS["dinov3-aerial"], 4) == "aerial_rgbn" and SC.mode_for(encmod.ENCODERS["dinov3-aerial"], 2) is None

res8 = Path(tempfile.mkdtemp(dir=TMP))
out = R.run_domain(root8, "toy_aerial", "stub-student", res8, token=None, seeds=(0,), mode="aerial_rgbn",
                   steps=40, batch=4, device="cpu")
run8 = "dom_toy_aerial__stub-student__Llast__aerial_rgbn"
assert set(out.run) == {run8} and set(out.split) == {"val", "test"} and set(out["head"]) == {"majority", "probe", "decoder"}
assert {"IoU_dark", "IoU_mid", "IoU_bright", "mIoU", "patch_mIoU"} <= set(out.columns) and "IoU_pos" not in out.columns
assert (out.layer == LAST).all() and (out.units == "8 bit").all() and (out["mode"] == "aerial_rgbn").all()
probe, majority = out[out["head"] == "probe"], out[out["head"] == "majority"]
assert (probe.mIoU.values > majority.mIoU.values).all(), (probe.mIoU.values, majority.mIoU.values)
info = json.loads((res8 / f"{run8}__info.json").read_text())
assert set(info["stats"]) == {"co", "cross", "diff"}
assert info["input_stats"] == {k: list(v) for k, v in S.aerial_stats(STUB_STATS["aerial"]["mean"], STUB_STATS["aerial"]["sd"]).items()}
reg = pd.read_csv(res8 / f"{run8}__probe__regions.csv")
assert set(reg.region) == {"D000", "D001", "D002"} and "IoU_bright" in reg.columns
print("run_domain on the aerial domain verified: student in aerial mode, 3-class mIoU + per-class IoU, checkpoint statistics")

# the 19-class set - mIoU over the classes and one IoU column per class, regions = departments
with contextlib.redirect_stdout(io.StringIO()):
    out19 = R.run_domain(root8b, "flair_sets", "stub-student", res8, token=None, seeds=(0,), mode="aerial_rgbn",
                         steps=40, batch=4, device="cpu")
assert [c for c in out19.columns if c.startswith("IoU_")] == [f"IoU_{n}" for n in AER.CLASSES]
assert "IoU_pos" not in out19.columns and out19.mIoU.notna().all() and set(out19.split) == {"val"}
reg19 = pd.read_csv(res8 / "dom_flair_sets__stub-student__Llast__aerial_rgbn__probe__regions.csv")
assert set(reg19.region) == {"D012", "D015"} and "IoU_undefined" in reg19.columns
print("run_domain on the 19-class set verified: mIoU + 19 IoU columns, regions = departments")

# score_all over both kinds of dataset, every encoder loaded once, pairs an encoder cannot read skipped
shutil.copytree(root2 / "toy_lakes", root8 / "toy_lakes")
encmod.ENCODERS["terrafm-tiny"] = encmod.EncoderSpec("tiny", "terrafm-tiny", "terrafm-s1", batch=4, in_channels=(2,),
                                                     input_stats=dict(encmod.ENCODERS["terrafm-b-s1"].input_stats))
encmod.ENCODERS["dinov3-tiny-aerial"] = encmod.EncoderSpec(str(folder), "dinov3-tiny-aerial", "dinov3", batch=4,
                                                           in_channels=(4,),
                                                           input_stats=dict(encmod.ENCODERS["dinov3-aerial"].input_stats))
R.PRE_REGISTERED_LAYERS["terrafm-tiny"] = LAST
R.PRE_REGISTERED_LAYERS["dinov3-tiny-aerial"] = LAST
ENCS, DATA = ["stub-student", "terrafm-tiny", "dinov3-tiny-aerial"], ["toy_lakes", "toy_aerial"]
res9 = Path(tempfile.mkdtemp(dir=TMP))
n_loaded = len(loaded)
T.build = tiny_build
try:
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        summ = SC.score_all(root8, ENCS, DATA, res9, token=None, checkpoints={"stub-student": ckpt}, device="cpu",
                            seeds=(0,), steps=10, batch=4)
finally:
    T.build = orig_build
text = buf.getvalue()
print(text[-1500:])
want_runs = {"dom_toy_lakes__stub-student__Llast__co_cross", "dom_toy_aerial__stub-student__Llast__aerial_rgbn",
             "dom_toy_lakes__terrafm-tiny__Llast__co_cross", "dom_toy_aerial__dinov3-tiny-aerial__Llast__aerial_rgbn"}
assert set(summ.run) == want_runs, set(summ.run)
assert "skip toy_aerial x terrafm-tiny" in text and "skip toy_lakes x dinov3-tiny-aerial" in text and "FAILED" not in text
assert len(loaded) == n_loaded + 1                  # the student was loaded once for both datasets
for r in want_runs:
    assert (res9 / f"{r}__summary.csv").exists() and (res9 / f"{r}__decoder__s0.pt").exists(), r
i1 = json.loads((res9 / "dom_toy_lakes__stub-student__Llast__co_cross__info.json").read_text())
i2 = json.loads((res9 / "dom_toy_lakes__terrafm-tiny__Llast__co_cross__info.json").read_text())
assert i1["input_stats"] == i1["stats"]             # the student in Sentinel-1 mode: the training split's statistics
assert i2["input_stats"] == i2["stats"] == i1["stats"]   # TerraFM-B too (2026-09-30), the pull request's numbers unused
assert i2["input_stats"]["co"] != [T.PR_MEAN_S1[0], T.PR_STD_S1[0]]
with contextlib.redirect_stdout(io.StringIO()) as buf2:         # second call: everything already there, nothing loaded
    again = SC.score_all(root8, ENCS, DATA, res9, token=None, device="cpu", seeds=(0,))
assert buf2.getvalue().count("skip dom_") == 4 and len(again) == len(summ) and len(loaded) == n_loaded + 1
fails(lambda: SC.score_all(root8, ["nobody"], DATA, res9, token=None, device="cpu"), KeyError, "unknown encoders")
# a failing encoder never stops the loop - the failure is printed, the other pairs are done
encmod.ENCODERS["broken"] = encmod.EncoderSpec("nowhere", "broken", "no-such-kind", in_channels=(2, 4))
R.PRE_REGISTERED_LAYERS["broken"] = LAST
res10 = Path(tempfile.mkdtemp(dir=TMP))
with contextlib.redirect_stdout(io.StringIO()) as buf3:
    summ3 = SC.score_all(root8, ["broken", "stub-student"], DATA, res10, token=None, device="cpu", seeds=(0,), steps=10,
                         batch=4)
text3 = buf3.getvalue()
assert text3.count("FAILED toy_lakes broken: ValueError") == 1 and text3.count("FAILED toy_aerial broken: ValueError") == 1
assert set(summ3.run) == {"dom_toy_lakes__stub-student__Llast__co_cross", "dom_toy_aerial__stub-student__Llast__aerial_rgbn"}
assert not list(res10.glob("dom_*broken*"))
del encmod.ENCODERS["broken"], R.PRE_REGISTERED_LAYERS["broken"]

tab = SC.score_table(res9, reference=res2)          # part two's rows = the toy lakes run of section 2, from its folder
assert set(tab.source) == {"part two", "part three"} and {"toy_lakes", "toy_aerial"} <= set(tab.dataset)
two = tab[tab.source == "part two"]
assert set(two.encoder) == {"fake"} and two.seeds.max() == 2 and set(two["head"]) == {"decoder", "majority", "probe"}
assert set(tab[tab.source == "part three"].encoder) == set(ENCS)
one_csv = TMP / "reference.csv"                     # the same rows from one CSV of summary rows
pd.concat([pd.read_csv(f) for f in sorted(res2.glob("dom_*__summary.csv"))]).to_csv(one_csv, index=False)
tab2 = SC.score_table(res9, reference=one_csv)
pd.testing.assert_frame_equal(tab, tab2)
agg_csv = TMP / "reference_table.csv"               # or from an already aggregated table
R.domain_table(res2).to_csv(agg_csv, index=False)
pd.testing.assert_frame_equal(tab, SC.score_table(res9, reference=agg_csv))
assert len(SC.score_table(res9, reference=TMP / "missing.csv")) == len(tab[tab.source == "part three"])
assert len(SC.score_table(res9)) == len(tab[tab.source == "part three"])
print(tab.round(3).to_string(index=False))
print("score_all and score_table verified: loop, mode per dataset, skip, skip-if-done, part two rows joined")

# ---- 8b. splits, stage A scores the validation split only --------------------
res11 = Path(tempfile.mkdtemp(dir=TMP))
with contextlib.redirect_stdout(io.StringIO()) as buf:
    only_val = SC.score_all(root8, ["stub-student"], ["toy_lakes"], res11, token=None, device="cpu", seeds=(0,),
                            steps=10, batch=4, splits=("val",))
run11 = "dom_toy_lakes__stub-student__Llast__co_cross"      # the run name does not change
assert set(only_val.run) == {run11} and set(only_val.split) == {"val"}
assert set(only_val["head"]) == {"majority", "probe", "decoder"}
assert "scored splits ['val'], the other splits of the manifest are not touched" in buf.getvalue()
info11 = json.loads((res11 / f"{run11}__info.json").read_text())
assert info11["scored_splits"] == ["val"] and info11["splits"] == {"test": 8, "test_challenge": 5, "train": 40, "val": 8}
assert {k for k in info11["times"] if k.startswith("extract_")} == {"extract_train_min", "extract_val_min"}
assert set(pd.read_csv(res11 / f"{run11}__probe__regions.csv").split) == {"val"}
assert set(pd.read_csv(res11 / f"{run11}__summary.csv").split) == {"val"}
with contextlib.redirect_stdout(io.StringIO()) as buf:     # the same splits again: skipped
    again11 = SC.score_all(root8, ["stub-student"], ["toy_lakes"], res11, token=None, device="cpu", seeds=(0,),
                           splits=("val",))
assert "skip dom_" in buf.getvalue() and set(again11.split) == {"val"}
with contextlib.redirect_stdout(io.StringIO()) as buf:     # a split that is not there yet: run again
    more11 = SC.score_all(root8, ["stub-student"], ["toy_lakes"], res11, token=None, device="cpu", seeds=(0,),
                          steps=10, batch=4, splits=("val", "test"))
assert "skip dom_" not in buf.getvalue() and set(more11.split) == {"val", "test"}
assert json.loads((res11 / f"{run11}__info.json").read_text())["scored_splits"] == ["val", "test"]
info_none = json.loads((res9 / f"{run11}__info.json").read_text())      # the default, every non-train split
assert info_none["scored_splits"] == ["test", "test_challenge", "val"]
for bad in (("train",), ("nothing",), (), ("val", "train")):
    fails(lambda bad=bad: R.run_domain(root8, "toy_lakes", "stub-student", res11, token=None, mode="co_cross",
                                       device="cpu", seeds=(0,), splits=bad), ValueError, "not scored splits")
print("splits verified: val only, info records scored_splits, skip only when every asked split is there")

# ---- 8c. score_parallel, one process per encoder side by side (part four, 2026-10-04) -------------
import queue  # noqa: E402

import sartransfer.distil as distil_pkg  # noqa: E402

# Section 5 put a stub of distil.train into sys.modules. A spawned worker imports the real one, so the
# parent's score_all must load the same real student - the real module is put back for this section.
stub_train = sys.modules["sartransfer.distil.train"]
assert distil_pkg.train is not stub_train and distil_pkg.train.load_student is real_load_student
sys.modules["sartransfer.distil.train"] = distil_pkg.train
encmod.ENCODERS["broken"] = encmod.EncoderSpec("nowhere", "broken", "no-such-kind", in_channels=(2, 4))
R.PRE_REGISTERED_LAYERS["broken"] = LAST
KW8 = dict(seeds=(0,), steps=10, batch=4)
MADE_UP = "MADE-UP-TOKEN-OF-THE-PARALLEL-TEST"
REAL_CONTEXT, REAL_POLL, REAL_WAVE = SC._context, SC.POLL_SECONDS, SC._run_wave


class InProcess:
    """A stand-in multiprocessing context. Process.start() runs the worker right here, to its end."""

    Queue = queue.Queue
    started = []

    class Process:
        def __init__(self, target, name, args):
            self.target, self.args, self.exitcode = target, args, None

        def start(self):
            InProcess.started.append(self.args[0])
            self.target(*self.args)
            self.exitcode = 0

        def is_alive(self):
            return False

        def join(self):
            pass

        def terminate(self):
            raise AssertionError("nothing is left to stop")


class Dies(InProcess):
    """A context whose processes end at once with exit code 3, before they report anything."""

    class Process(InProcess.Process):
        def start(self):
            InProcess.started.append(self.args[0])
            self.exitcode = 3


try:
    # run_need by hand - toy_lakes, 64 px chips (4 x 4 grid), train 40, val 8, test 8, test_challenge 5
    need = SC.run_need(root8, "toy_lakes")
    parts = need["parts"]
    assert abs(parts["train_tiles"] - 40 * (16 * 1024 * 2 + 2 * 64 * 64) / 1e9) < 1e-12
    assert abs(parts["probe"] - 40 * 16 * (1024 * 2 + 8) / 1e9) < 1e-12           # 640 patches, under the cap
    assert abs(parts["decoder_train"] - 8 * 64 * 64 * SC.DECODER_TRAIN_BYTES / 1e9) < 1e-12
    assert abs(parts["decoder_eval"] - 8 * 64 * 64 * SC.DECODER_EVAL_BYTES / 1e9) < 1e-12   # 8 chips, under 32
    assert abs(parts["extract"] - 512 * (16 + 5) * 1024 * SC.EXTRACT_BYTES / 1e9) < 1e-12   # run.py's batch 512
    assert abs(need["gpu_gb"] - (SC.CONTEXT_GB + SC.ENCODER_GB + parts["extract"])) < 1e-12
    assert need["ram_gb"] > SC.HOST_BASE_GB
    assert SC.run_need(root8, "toy_lakes", splits=("test_challenge",))["parts"]["decoder_eval"] == \
        5 * 64 * 64 * SC.DECODER_EVAL_BYTES / 1e9
    small = SC.run_need(root8, "toy_lakes", max_train_chips=10, probe_max_patches=100, batch=2)["parts"]
    assert abs(small["train_tiles"] - parts["train_tiles"] / 4) < 1e-12
    assert abs(small["probe"] - 100 * 2056 / 1e9) < 1e-12 and abs(small["decoder_train"] - parts["decoder_train"] / 4) < 1e-12

    # the worker rule on a stand-in GPU - per dataset, the waves with the most at once first
    planned = []
    SC._run_wave = lambda jobs, count, *a: planned.append((jobs, count))
    SC.register_student("real-student-2", real_ckpt)
    lakes_gb = SC.run_need(root8, "toy_lakes")["gpu_gb"]
    aerial_gb = SC.run_need(root8, "toy_aerial")["gpu_gb"]
    free = SC.GPU_MARGIN_GB + 2 * aerial_gb + 0.01                     # 2 aerial runs fit, 2 lakes runs do not
    assert 2 * lakes_gb > free - SC.GPU_MARGIN_GB
    import psutil  # noqa: E402

    had_mem, had_vm = torch.cuda.mem_get_info, psutil.virtual_memory
    torch.cuda.mem_get_info = lambda device=None: (int(free * 1e9), 80 * 10 ** 9)
    psutil.virtual_memory = lambda: types.SimpleNamespace(available=500e9, total=600e9)
    try:
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            SC.score_parallel(root8, ["real-student", "real-student-2"], DATA, Path(tempfile.mkdtemp(dir=TMP)),
                              token=None, device="cuda", **KW8)
    finally:
        torch.cuda.mem_get_info, psutil.virtual_memory = had_mem, had_vm
    text = buf.getvalue()
    assert planned == [([("real-student", ["toy_aerial"]), ("real-student-2", ["toy_aerial"])], 2),
                       ([("real-student", ["toy_lakes"]), ("real-student-2", ["toy_lakes"])], 1)], planned
    assert re.search(r"toy_lakes - one run needs about [\d.]+ GB of GPU memory and [\d.]+ GB of RAM \(estimates\), "
                     r"free [\d.]+ GB on the GPU \(2 kept\) and 500\.0 GB of RAM, 1 at once = min\(2 encoders, 1 by "
                     r"GPU memory, \d+ by RAM\)", text), text
    assert "toy_aerial - " in text and "2 at once = min(2 encoders, 2 by GPU memory" in text
    assert text.index("wave 1 - toy_aerial, 2 encoders, 2 at once") < text.index("wave 2 - toy_lakes, 2 encoders, 1 at once")
    assert ":" not in text.replace(str(TMP), "")
    SC._run_wave = REAL_WAVE
    del encmod.ENCODERS["real-student-2"], R.PRE_REGISTERED_LAYERS["real-student-2"]

    # a process that ends without a word - both its runs failed, named with the encoder, its log tail printed
    SC._context, SC.POLL_SECONDS = (lambda: Dies), 0.05
    res_dies = Path(tempfile.mkdtemp(dir=TMP))
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        rows_d, times_d = SC.score_parallel(root8, ["real-student"], DATA, res_dies, token=None, device="cpu",
                                            workers=2, **KW8)
    text = buf.getvalue()
    assert rows_d.empty and list(times_d["status"]) == ["failed", "failed"], times_d
    assert set(times_d["error"]) == {"its process ended with exit code 3"}
    assert "real-student - FAILED, its process ended with exit code 3, not run ['toy_lakes', 'toy_aerial']" in text
    assert "cpu - 1 at once (given)" in text, "one encoder, so 1 at once although 2 were given"

    # two real processes (spawn) side by side on the CPU - the same numbers as score_all, the error of one
    # worker reported with its encoder and dataset while the other finishes
    SC._context, SC.POLL_SECONDS = REAL_CONTEXT, REAL_POLL
    res_seq, res_par = Path(tempfile.mkdtemp(dir=TMP)), Path(tempfile.mkdtemp(dir=TMP))
    with contextlib.redirect_stdout(io.StringIO()):
        seq = SC.score_all(root8, ["real-student"], DATA, res_seq, token=None,
                           checkpoints={"real-student": real_ckpt}, device="cpu", **KW8)
    clock = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        par, times_p = SC.score_parallel(root8, ["real-student", "broken"], DATA, res_par, token=MADE_UP,
                                         checkpoints={"real-student": real_ckpt}, device="cpu", workers=2, **KW8)
    wall = time.perf_counter() - clock
    text = buf.getvalue()
    print(text)
    print(SC.gain_line(times_p, wall))
    pd.testing.assert_frame_equal(par, seq)                 # the rows of the student, the same numbers and order
    for r in sorted(set(par["run"])):
        for suffix in ("__summary.csv", "__probe__regions.csv", "__decoder__s0__regions.csv"):
            assert (res_par / f"{r}{suffix}").read_bytes() == (res_seq / f"{r}{suffix}").read_bytes(), (r, suffix)
        for head in ("__probe.pt", "__decoder__s0.pt"):
            a, b = torch.load(res_par / f"{r}{head}"), torch.load(res_seq / f"{r}{head}")
            assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), (r, head)
    assert [tuple(x) for x in times_p[["encoder", "dataset", "status"]].to_numpy()] == [
        ("real-student", "toy_lakes", "done"), ("real-student", "toy_aerial", "done"),
        ("broken", "toy_lakes", "failed"), ("broken", "toy_aerial", "failed")], times_p
    assert set(times_p["workers"]) == {2} and set(times_p["wave"]) == {1} and times_p["minutes"].notna().all()
    assert times_p["peak_gpu_gb"].isna().all() and times_p["need_gpu_gb"].isna().all()   # no GPU here
    assert all(e.startswith("ValueError: unknown encoder kind 'no-such-kind'")
               for e in times_p.loc[times_p["encoder"] == "broken", "error"])
    for d in DATA:
        assert f"  broken x {d} - FAILED after " in text and f"  real-student x {d} - done in " in text
    assert "started real-student on toy_lakes, toy_aerial" in text and "started broken on toy_lakes, toy_aerial" in text
    assert "cpu - 2 at once (given)" in text and "wave 1 - toy_lakes, toy_aerial, 2 encoders, 2 at once" in text
    logs = {e: (res_par / "logs" / f"score__{e}.log").read_text(encoding="utf-8") for e in ("real-student", "broken")}
    assert "===== dom_toy_lakes__real-student__Llast__co_cross" in logs["real-student"]
    assert "FAILED toy_lakes broken: ValueError" in logs["broken"] and "FAILED toy_aerial broken" in logs["broken"]
    pids = {e: int(re.search(r"^===== \S+, process (\d+),", log).group(1)) for e, log in logs.items()}
    assert len(set(pids.values())) == 2 and os.getpid() not in pids.values(), pids      # two real processes
    assert MADE_UP not in text and not any(MADE_UP in log for log in logs.values())
    assert not list(res_par.glob("dom_*broken*"))
    # skips kept - a second call starts nothing, the summaries come back as score_all gives them
    InProcess.started.clear()
    SC._context = lambda: InProcess
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        again_p, times_again = SC.score_parallel(root8, ["real-student"], DATA, res_par, token=None, device="cpu",
                                                 checkpoints={"real-student": real_ckpt}, workers=2, **KW8)
    with contextlib.redirect_stdout(io.StringIO()):
        again_s = SC.score_all(root8, ["real-student"], DATA, res_par, token=None, device="cpu", **KW8)
    assert InProcess.started == [] and "wave" not in buf.getvalue() and buf.getvalue().count("skip dom_") == 2
    pd.testing.assert_frame_equal(again_p, again_s)
    assert list(times_again["status"]) == ["skipped", "skipped"] and list(times_again["minutes"]) == [0.0, 0.0]
    assert SC.gain_line(times_again, 6.0) == "no run was scored here, 0.1 min wall"
    # one summary gone - only that run is done again, the other is read; an encoder that cannot read a
    # dataset is named and never started there
    os.remove(res_par / "dom_toy_aerial__real-student__Llast__aerial_rgbn__summary.csv")
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        one_p, times_one = SC.score_parallel(root8, ["real-student", "dinov3-tiny-aerial"], DATA, res_par,
                                             token=None, device="cpu", workers=2, **KW8)
    text = buf.getvalue()
    assert InProcess.started == ["real-student", "dinov3-tiny-aerial"], InProcess.started
    assert "skip dom_toy_lakes__real-student__Llast__co_cross - its summary holds" in text
    assert "skip toy_lakes x dinov3-tiny-aerial - the dataset has 2 channels, the encoder reads (4,)" in text
    assert list(times_one["status"]) == ["skipped", "done", "cannot read", "done"], times_one
    pd.testing.assert_frame_equal(one_p[one_p["encoder"] == "real-student"].reset_index(drop=True),
                                  again_s, check_dtype=False)
    line = SC.gain_line(times_one, 60.0, {"toy_aerial": 1.0})
    assert line.startswith("2 runs in 1.0 min wall against ") and line.endswith(
        "2.0 min for these datasets on an A100 with 80 GB, 2.0 x that"), line
    assert ":" not in line and "part three" not in SC.gain_line(times_one, 60.0, {})
    fails(lambda: SC.score_parallel(root8, ["real-student"], DATA, res_par, token=None, device="cpu", workers=0),
          ValueError, "at least 1")
    fails(lambda: SC.score_parallel(root8, ["nobody"], DATA, res_par, token=None, device="cpu"), KeyError,
          "unknown encoders")
finally:
    SC._context, SC.POLL_SECONDS, SC._run_wave = REAL_CONTEXT, REAL_POLL, REAL_WAVE
    sys.modules["sartransfer.distil.train"] = stub_train
    del encmod.ENCODERS["broken"], R.PRE_REGISTERED_LAYERS["broken"]
print("score_parallel verified: run_need by hand, the worker rule and its waves, a process that dies, two real "
      "processes with the numbers of score_all and one failing worker, skips before any process, one run redone")

# ---- 9. no name of another repository in the copied or new files --------------------------------
src = REPO / "src" / "sartransfer"
checked = [src / p for p in ("probe.py", "models/__init__.py", "models/decoder.py", "models/encoders.py",
                             "data/__init__.py", "data/targets.py", "domains/__init__.py", "domains/prepared.py",
                             "domains/s1input.py", "domains/hub.py", "domains/run.py", "domains/aerial.py",
                             "distil/score.py")] + [Path(__file__)]
forbidden = list(_setup.private_names())      # empty in a public copy
for f in checked:
    assert f.is_file(), f
    hits = [pat for pat in forbidden if pat in f.read_text(encoding="utf-8")]
    assert not hits, f"{f} names another repository: {hits}"
print(f"no other repository named in {len(checked)} files")
# no colon in a docstring title (the first line of any docstring) of these files, no colon in titles
for f in checked:
    tree = ast.parse(f.read_text(encoding="utf-8"))
    nodes = [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    titles = [(getattr(n, "name", "module"), (ast.get_docstring(n) or "").strip().splitlines()[:1]) for n in nodes]
    bad = [(name, t[0]) for name, t in titles if t and ":" in t[0]]
    assert not bad, f"{f}: colon in a docstring title {bad}"
print(f"no colon in a docstring title of {len(checked)} files")

# ---- 10. the private pilot repo of part three (hub.py), with a stand-in for huggingface_hub ------
import fnmatch  # noqa: E402
import tarfile  # noqa: E402

from sartransfer.distil import aerial as AER, pilotdata as PD  # noqa: E402

HUB_TOKEN = "MADE-UP-WRITE-TOKEN-OF-THE-TEST"
pilot = Path(tempfile.mkdtemp(dir=TMP)) / "pilot"
(pilot / "majortom").mkdir(parents=True)
(pilot / "aerial").mkdir()
for k, n_pairs in ((0, 3), (1, 2)):
    tar_path = pilot / "majortom" / f"chunk_{k}.tar"
    with tarfile.open(tar_path, "w") as tar:
        for i in range(n_pairs):
            member = pilot / f"cell_{k}_{i}.npz"
            member.write_bytes(bytes(200 * (i + 1)))
            tar.add(member, arcname=f"chunk_{k}/cell_{k}_{i}.npz")
            member.unlink()
    (pilot / "majortom" / f"chunk_{k}.json").write_text(json.dumps(
        {"chunk": k, "files": [f"cell_{k}_{i}.npz" for i in range(n_pairs)], "bytes": tar_path.stat().st_size,
         "pairs": n_pairs}), encoding="utf-8")
for name, text in ((PD.TRAIN_FILE, "a\nb\nc\n"), (PD.HELD_OUT_FILE, "d\ne\n"), ("channel_stats.json", "{}"),
                   ("pilot_sample.csv", "grid_cell,chunk\na,0\n"), ("fetch_skips.csv", "x\n"),
                   ("fetch_timing.csv", "y\n")):
    (pilot / "majortom" / name).write_text(text, encoding="utf-8")
for name in AER.SETS:
    (pilot / "aerial" / f"{name}.tar").write_bytes(bytes(1000 + len(name)))
(pilot / "aerial" / "manifest.json").write_text(json.dumps(
    {"sets": {name: {"count": 10 + i, "files": []} for i, name in enumerate(AER.SETS)}}), encoding="utf-8")
for name in ("plan.csv", "zips.csv", "patches.csv"):
    (pilot / "aerial" / name).write_text("set\n", encoding="utf-8")
WANTED_PILOT = {"majortom/chunk_0.tar", "majortom/chunk_0.json", "majortom/chunk_1.tar", "majortom/chunk_1.json",
                f"majortom/{PD.TRAIN_FILE}", f"majortom/{PD.HELD_OUT_FILE}", "majortom/channel_stats.json",
                "majortom/pilot_sample.csv", *(f"aerial/{n}.tar" for n in AER.SETS), "aerial/manifest.json",
                "aerial/plan.csv"}
N_PILOT_FILES = len(WANTED_PILOT) + 1                             # plus the card README.md
files = H.pilot_files(pilot)
assert set(files) == WANTED_PILOT, set(files) ^ WANTED_PILOT
assert all(files[n] == (pilot / n).stat().st_size for n in files)
# the repo id comes from DATA_HF_REPO of the .env (part four), a made-up one here
PILOT_RID = "made-up-owner/made-up-pilot-data"
HAD_DATA_REPO = os.environ.get("DATA_HF_REPO")
os.environ["DATA_HF_REPO"] = PILOT_RID
assert H.pilot_repo_id() == PILOT_RID == H.data_repo_id()
fails(lambda: H.pilot_repo_id("x"), TypeError, "no owner")
good_tar = (pilot / "majortom" / "chunk_1.tar").read_bytes()
(pilot / "majortom" / "chunk_1.tar").write_bytes(good_tar[:-10])
fails(lambda: H.pilot_files(pilot), ValueError, "chunk_1.tar has")
(pilot / "majortom" / "chunk_1.tar").write_bytes(good_tar)
(pilot / "majortom" / PD.HELD_OUT_FILE).rename(pilot / "held_aside.txt")
fails(lambda: H.pilot_files(pilot), FileNotFoundError, f"majortom/{PD.HELD_OUT_FILE}")
(pilot / "held_aside.txt").rename(pilot / "majortom" / PD.HELD_OUT_FILE)
(pilot / "majortom" / "chunk_1.json").rename(pilot / "info_aside.json")
fails(lambda: H.pilot_files(pilot), FileNotFoundError, "['majortom/chunk_1.tar'] have no JSON")
(pilot / "info_aside.json").rename(pilot / "majortom" / "chunk_1.json")
(pilot / "majortom" / "chunk_1.tar").rename(pilot / "tar_aside.tar")
fails(lambda: H.pilot_files(pilot), FileNotFoundError, "chunk_1.tar is missing")
(pilot / "tar_aside.tar").rename(pilot / "majortom" / "chunk_1.tar")

# the card, written by the dry run
with contextlib.redirect_stdout(io.StringIO()) as buf:
    rid = H.upload_pilot(pilot, token="unused", dry_run=True, date="2026-09-30")
assert rid == PILOT_RID and f"(private), {N_PILOT_FILES} files" in buf.getvalue()
card = (pilot / "README.md").read_text(encoding="utf-8")
head = card.split("\n---\n", 1)[1].splitlines()[0]
assert head.startswith(f"# {PILOT_RID}, the paired Sentinel-1 and Sentinel-2 pilot data") and ":" not in head
assert "part three" not in card
for words in ("license:\n- cc-by-sa-4.0\n- etalab-2.0", "Major-TOM/Core-S2L2A-249k", "Major-TOM/Core-S1RTC",
              "CC-BY-SA-4.0", "ESA", "Etalab Open Licence 2.0", AER.ATTRIBUTION, AER.HF_REVISION, "2026-09-30",
              "5 paired Major-TOM cells in 2 archives", "| distil_train | 10 |", "| val | 13 |",
              "| majortom/chunk_0.tar |", "Private working copy", "NOT an original"):
    assert words in card, f"the card lacks {words!r}"
assert "unused" not in card and "README.md |" not in card
assert card.count("aerial/") == len(AER.SETS) + 2 and "fetch_skips" not in card


class FakeHub:
    """create_repo, repo_info, list_repo_tree, upload_file over a dict of bytes."""

    class Entry:
        def __init__(self, path, size):
            self.path, self.size = path, size

    class Folder:
        def __init__(self, path):
            self.path = path

    def __init__(self):
        self.store, self.private, self.created, self.uploads, self.drop = {}, True, [], [], set()
        self.missing = False
        self.slow, self.threads, self.raise_on = 0.0, set(), set()     # for the parallel path

    def create_repo(self, rid, repo_type, private, exist_ok):
        assert (repo_type, private, exist_ok) == ("dataset", True, True)
        self.created.append(rid)

    def repo_info(self, rid, repo_type):
        assert repo_type == "dataset"
        return types.SimpleNamespace(private=self.private)

    def list_repo_tree(self, rid, repo_type, recursive):
        assert repo_type == "dataset" and recursive is True
        if self.missing:
            raise FileNotFoundError(f"stand-in, {rid} is not on the Hub, the token was {HUB_TOKEN}")
        folders = {p.rpartition("/")[0] for p in self.store if "/" in p}
        return [self.Entry(p, len(b)) for p, b in self.store.items()] + [self.Folder(f) for f in folders]

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        assert repo_type == "dataset" and commit_message == f"pilot, {path_in_repo}"
        self.threads.add(threading.get_ident())
        time.sleep(self.slow)                                          # so the threads overlap
        if path_in_repo in self.raise_on:
            raise OSError(f"stand-in, {path_in_repo} was refused")
        data = Path(path_or_fileobj).read_bytes()
        self.store[path_in_repo] = data[:-1] if path_in_repo in self.drop else data
        self.uploads.append(path_in_repo)


fake = FakeHub()
seen_tokens = []


def fake_api(token):
    seen_tokens.append(token)
    assert token == HUB_TOKEN, "another token reached the Hub"
    return fake


DOWNLOAD = {"slow": 0.0, "threads": set(), "raise_on": set(), "names": []}   # for the parallel path


def error_of(fn, kind) -> str:
    """The text of the `kind` that `fn` raises."""
    try:
        fn()
    except kind as err:
        return str(err)
    raise AssertionError(f"no {kind.__name__} was raised")


def fake_download(rid, name, dest, token):
    assert token == HUB_TOKEN and rid == PILOT_RID
    DOWNLOAD["threads"].add(threading.get_ident())
    DOWNLOAD["names"].append(name)
    time.sleep(DOWNLOAD["slow"])
    if name in DOWNLOAD["raise_on"]:
        raise OSError(f"stand-in, {name} was refused")
    target = Path(dest) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(fake.store[name])
    return target


real_api, real_download = H.hub_api, H.hub_download
H.hub_api, H.hub_download = fake_api, fake_download
try:
    fake.private = False                                           # a public repo is refused
    fails(lambda: H.upload_pilot(pilot, token=HUB_TOKEN, date="2026-09-30"), RuntimeError, "PUBLIC")
    assert fake.created == [PILOT_RID] and fake.uploads == []
    fake.private = True
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        H.upload_pilot(pilot, token=HUB_TOKEN, date="2026-09-30")
    assert set(fake.uploads) == WANTED_PILOT | {"README.md"} and len(fake.uploads) == N_PILOT_FILES
    assert f"uploaded and checked, {N_PILOT_FILES} files, 0 skipped, sizes equal" in buf.getvalue()
    assert buf.getvalue().count("uploaded in") == N_PILOT_FILES and HUB_TOKEN not in buf.getvalue()
    assert buf.getvalue().count("MB per second") == N_PILOT_FILES + 1, buf.getvalue()
    assert re.search(rf"uploaded {N_PILOT_FILES} of {N_PILOT_FILES} files, [\d,]+ MB in \d+ s, [\d,]+ MB per second "
                     "with 4 threads", buf.getvalue()), buf.getvalue()
    fake.uploads.clear()
    with contextlib.redirect_stdout(io.StringIO()) as buf:         # the same again, all skipped
        H.upload_pilot(pilot, token=HUB_TOKEN, date="2026-09-30")
    assert fake.uploads == [] and f"{N_PILOT_FILES} skipped" in buf.getvalue()
    assert buf.getvalue().count("on the Hub with the same size, skipped") == N_PILOT_FILES
    # one file grew - from 6 to 10 bytes, so its entry in the card gains a digit on every platform (with
    # 8 bytes the card kept its size on Linux, where write_text writes LF, and was skipped)
    (pilot / "majortom" / PD.TRAIN_FILE).write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        H.upload_pilot(pilot, token=HUB_TOKEN, date="2026-09-30")
    assert sorted(fake.uploads) == sorted([f"majortom/{PD.TRAIN_FILE}", "README.md"]), fake.uploads   # the card lists the bytes
    assert f"{N_PILOT_FILES - 2} skipped" in buf.getvalue()
    fake.uploads.clear()
    fake.drop.add("aerial/plan.csv")                               # a file that arrives short
    (pilot / "aerial" / "plan.csv").write_text("set,x\n", encoding="utf-8")
    fails(lambda: H.upload_pilot(pilot, token=HUB_TOKEN, date="2026-09-30"), IOError, "differ in size, ['aerial/plan.csv']")
    fake.drop.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        H.upload_pilot(pilot, token=HUB_TOKEN, date="2026-09-30")
    assert H.list_pilot(token=HUB_TOKEN) == {n: len(b) for n, b in fake.store.items()}
    assert set(H.list_pilot(token=HUB_TOKEN)) == WANTED_PILOT | {"README.md"}

    # the parallel path. The files go up in threads in any order, all present with their sizes (a slow
    # stand-in shows the overlap, 21 uploads of 0.05 s in well under 21 x 0.05 s). A file that fails is
    # named at the end, does not hang and does not stop the others, and the next run sends only it
    # The time is the waiting only - the same call without the delay first (the reads and the hashing, which a
    # busy machine slows down), subtracted from the call with it; three tries against a passing load spike
    for attempt in range(3):
        times = {}
        for slow in (0.0, 0.05):
            fake.store.clear()
            fake.uploads.clear()
            fake.threads.clear()
            fake.slow = slow
            clock = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                H.upload_pilot(pilot, token=HUB_TOKEN, date="2026-09-30")
            times[slow] = time.perf_counter() - clock
        waiting = times[0.05] - times[0.0]
        if waiting < 0.75 * N_PILOT_FILES * fake.slow:
            break
    assert waiting < 0.75 * N_PILOT_FILES * fake.slow, (
        f"{N_PILOT_FILES} uploads of {fake.slow} s waited {waiting:.2f} s ({times[0.05]:.2f} s with the delay, "
        f"{times[0.0]:.2f} s without)")
    assert len(fake.threads) >= 2 and sorted(fake.uploads) == sorted(WANTED_PILOT | {"README.md"})
    assert {n: len(b) for n, b in fake.store.items()} == {n: (pilot / n).stat().st_size for n in fake.store}
    assert f"uploaded and checked, {N_PILOT_FILES} files, 0 skipped, sizes equal" in buf.getvalue()
    assert "with 4 threads" in buf.getvalue() and HUB_TOKEN not in buf.getvalue()
    fake.slow = 0.0
    fake.store.clear()
    fake.uploads.clear()
    fake.raise_on.add("aerial/val.tar")
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        text = error_of(lambda: H.upload_pilot(pilot, token=HUB_TOKEN, date="2026-09-30"), IOError)
    assert "these failed, {'aerial/val.tar': 'OSError, stand-in, aerial/val.tar was refused'}" in text, text
    assert "differ in size, ['aerial/val.tar']" in text and "same size are skipped" in text
    assert "FAILED after" in buf.getvalue() and f"uploaded {N_PILOT_FILES - 1} of {N_PILOT_FILES} files" in buf.getvalue()
    assert set(fake.uploads) == (WANTED_PILOT | {"README.md"}) - {"aerial/val.tar"}, "the other files went up"
    fake.raise_on.clear()
    fake.uploads.clear()
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        H.upload_pilot(pilot, token=HUB_TOKEN, date="2026-09-30")
    assert fake.uploads == ["aerial/val.tar"] and f"{N_PILOT_FILES - 1} skipped" in buf.getvalue()
    assert H.list_pilot(token=HUB_TOKEN) == {n: len(b) for n, b in fake.store.items()}

    # the fetch, into another folder, then again (skipped), with patterns, with a foreign file on the Hub
    dest = pilot.parent / "fetched"
    fake.store["notes.txt"] = b"never fetched"
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        got = H.fetch_pilot(dest, token=HUB_TOKEN)
    assert set(got) == WANTED_PILOT | {"README.md"} and "notes.txt" not in got
    assert all(p == dest / n and p.read_bytes() == fake.store[n] for n, p in got.items())
    assert f"{N_PILOT_FILES + 1} files" not in buf.getvalue() and f"{N_PILOT_FILES} files in" in buf.getvalue()
    assert "0 were there already" in buf.getvalue()
    assert list(got) == sorted(got) and buf.getvalue().count("MB per second") == N_PILOT_FILES + 1
    assert re.search(rf"fetched {N_PILOT_FILES} of {N_PILOT_FILES} files, [\d,]+ MB in \d+ s, [\d,]+ MB per second "
                     "with 4 threads", buf.getvalue()), buf.getvalue()
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        again = H.fetch_pilot(dest, token=HUB_TOKEN)
    assert again == got and f"{N_PILOT_FILES} were there already" in buf.getvalue()
    assert "fetched in" not in buf.getvalue() and "MB per second" not in buf.getvalue()
    (dest / "aerial" / "val.tar").write_bytes(b"cut")                # a cut file is fetched again
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        H.fetch_pilot(dest, token=HUB_TOKEN, want=("aerial/*", "notes.txt"))
    assert (dest / "aerial" / "val.tar").read_bytes() == fake.store["aerial/val.tar"]
    assert buf.getvalue().count("fetched in") == 1 and buf.getvalue().count("MB per second") == 2
    assert f"{len(AER.SETS) + 2} files in" in buf.getvalue()
    assert not (dest / "notes.txt").exists()
    # the parallel fetch. The files come down in threads in any order, the paths come back in name
    # order. A file that fails is named at the end, does not hang, the others are on the disk, and the
    # next run fetches only the missing one
    dest2 = pilot.parent / "fetched2"
    # the waiting only, as for the uploads - the same fetch without the delay first, three tries
    for attempt in range(3):
        times = {}
        for slow in (0.0, 0.05):
            shutil.rmtree(dest2, ignore_errors=True)
            DOWNLOAD["threads"].clear()
            DOWNLOAD["names"].clear()
            DOWNLOAD["slow"] = slow
            clock = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                got2 = H.fetch_pilot(dest2, token=HUB_TOKEN)
            times[slow] = time.perf_counter() - clock
        waiting = times[0.05] - times[0.0]
        if waiting < 0.75 * N_PILOT_FILES * DOWNLOAD["slow"]:
            break
    assert waiting < 0.75 * N_PILOT_FILES * DOWNLOAD["slow"], (
        f"{N_PILOT_FILES} downloads of {DOWNLOAD['slow']} s waited {waiting:.2f} s ({times[0.05]:.2f} s with the "
        f"delay, {times[0.0]:.2f} s without)")
    assert len(DOWNLOAD["threads"]) >= 2 and sorted(DOWNLOAD["names"]) == sorted(got2) and list(got2) == sorted(got2)
    assert all(p == dest2 / n and p.read_bytes() == fake.store[n] for n, p in got2.items())
    assert "with 4 threads" in buf.getvalue() and HUB_TOKEN not in buf.getvalue()
    DOWNLOAD["slow"] = 0.0
    shutil.rmtree(dest2)
    DOWNLOAD["raise_on"].add("aerial/probe.tar")
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        text = error_of(lambda: H.fetch_pilot(dest2, token=HUB_TOKEN), IOError)
    assert "these files did not arrive whole, {'aerial/probe.tar': 'OSError, stand-in, aerial/probe.tar was refused'}" in text, text
    assert "FAILED after" in buf.getvalue() and f"fetched {N_PILOT_FILES - 1} of {N_PILOT_FILES} files" in buf.getvalue()
    assert not (dest2 / "aerial" / "probe.tar").exists()
    assert all((dest2 / n).read_bytes() == fake.store[n] for n in (WANTED_PILOT | {"README.md"}) - {"aerial/probe.tar"})
    DOWNLOAD["raise_on"].clear()
    DOWNLOAD["names"].clear()
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        got2 = H.fetch_pilot(dest2, token=HUB_TOKEN)
    assert DOWNLOAD["names"] == ["aerial/probe.tar"] and f"{N_PILOT_FILES - 1} were there already" in buf.getvalue()
    assert set(got2) == WANTED_PILOT | {"README.md"} and (dest2 / "aerial" / "probe.tar").read_bytes() == fake.store["aerial/probe.tar"]
    fails(lambda: H.fetch_pilot(dest, token=HUB_TOKEN, want=("nothing/*",)), FileNotFoundError, "holds no file of")
    fake.missing = True
    fails(lambda: H.fetch_pilot(dest, token=HUB_TOKEN), FileNotFoundError, "is not on the Hub")
    fake.missing = False
    fake.store["aerial/plan.csv"] = fake.store["aerial/plan.csv"] + b"x"      # the Hub's copy changed
    (dest / "aerial" / "plan.csv").unlink()
    H.hub_download = lambda rid, name, d, token: (fake_download(rid, name, d, token),
                                                  (Path(d) / name).write_bytes(b""))[0]
    fails(lambda: H.fetch_pilot(dest, token=HUB_TOKEN, want=("aerial/plan.csv",)), IOError, "has not the bytes")
    H.hub_download = fake_download

    # the order of the fetch and the hand-over of every file that is whole on the disk (part four, the data
    # notebook reads a pair archive while the next ones arrive). The small files first, then the pair archives
    # by chunk number (2 before 10), then the other archives. The files there already are handed over first,
    # then every file as it arrives, in the calling thread, with its path, whole
    assert sorted(["majortom/chunk_10.tar", "aerial/val.tar", "majortom/chunk_2.tar", "majortom/chunk_2.json",
                   "aerial/manifest.json"], key=H.fetch_order) == [
        "aerial/manifest.json", "majortom/chunk_2.json", "majortom/chunk_2.tar", "majortom/chunk_10.tar",
        "aerial/val.tar"]
    assert (H.FETCH_WORKERS, H.SMALL_FETCH_WORKERS) == (4, 32)
    before_small = Path(H.__file__).read_text(encoding="utf-8").split("SMALL_FETCH_WORKERS = 32")[0].rsplit(
        "FETCH_WORKERS = 4", 1)[1]
    assert all(w in before_small for w in ("latency", "one round", "5,000 requests per 5 minutes", "Not measured")), \
        "the reason for 32 threads is not next to the number"
    dest3 = pilot.parent / "fetched3"
    (dest3 / "majortom").mkdir(parents=True)
    (dest3 / "majortom" / "chunk_1.json").write_bytes(fake.store["majortom/chunk_1.json"])       # there already
    DOWNLOAD["names"].clear()
    handed = []

    def hand_over(name, path):
        handed.append((name, path, threading.get_ident(),
                       path.is_file() and path.stat().st_size == len(fake.store[name])))

    with contextlib.redirect_stdout(io.StringIO()) as buf:
        got3 = H.fetch_pilot(dest3, token=HUB_TOKEN, workers=1, on_file=hand_over)
    started = list(DOWNLOAD["names"])
    assert started == sorted(started, key=H.fetch_order) and "majortom/chunk_1.json" not in started
    assert started.index("majortom/chunk_0.tar") < started.index("majortom/chunk_1.tar") < started.index("aerial/val.tar")
    assert max(started.index(n) for n in started if not n.endswith(".tar")) < min(
        started.index(n) for n in started if n.endswith(".tar")), "an archive was started before a small file"
    assert [h[0] for h in handed] == ["majortom/chunk_1.json", *started], handed     # one thread, in the started order
    assert {h[0] for h in handed} == set(got3) and all(h[1] == dest3 / h[0] and h[3] for h in handed)
    assert {h[2] for h in handed} == {threading.get_ident()}, "a file was handed over in another thread"
    assert list(got3) == sorted(got3) and "1 were there already" in buf.getvalue()
    # nothing to fetch, every file handed over all the same
    handed.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        H.fetch_pilot(dest3, token=HUB_TOKEN, want=("majortom/*.json",), on_file=hand_over)
    assert [h[0] for h in handed] == ["majortom/channel_stats.json", "majortom/chunk_0.json", "majortom/chunk_1.json"]
    # a hand-over that fails stops the fetch, the files not started yet are cancelled
    DOWNLOAD["names"].clear()
    DOWNLOAD["slow"] = 0.05

    def refuse(name, path):
        raise RuntimeError("stand-in, the reader refused")

    with contextlib.redirect_stdout(io.StringIO()):
        fails(lambda: H.fetch_pilot(pilot.parent / "fetched4", token=HUB_TOKEN, workers=1, on_file=refuse),
              RuntimeError, "stand-in, the reader refused")
    DOWNLOAD["slow"] = 0.0
    assert 1 <= len(DOWNLOAD["names"]) <= 2 < N_PILOT_FILES, f"the fetch went on, {DOWNLOAD['names']}"
    assert set(seen_tokens) == {HUB_TOKEN}
finally:
    H.hub_api, H.hub_download = real_api, real_download
    os.environ.pop("DATA_HF_REPO", None)
    if HAD_DATA_REPO is not None:
        os.environ["DATA_HF_REPO"] = HAD_DATA_REPO
assert H.hub_api is real_api and not fnmatch.fnmatch("majortom/fetch_skips.csv", "majortom/chunk_*.json")
print("pilot repo verified, files, card, public refused, skip on size, size check, fetch skip, listing, "
      "the parallel upload and fetch, a failing file named")

# ---- 11. part four - only the splits read, from the Hub (fetch) or from Drive (copy_prepared) ------
import huggingface_hub  # noqa: E402

assert H.repo_name("glacial_lakes") == "sar-transfer-glacial-lakes" and H.repo_id("forty_v1", "o") == "o/sar-transfer-forty-v1"
assert P.split_files(("train", "val")) == ["train_images.npy", "train_labels.npy", "train_meta.csv",
                                           "val_images.npy", "val_labels.npy", "val_meta.csv"]
copy_src = root / "toy"                                     # section 1's dataset, splits train and test
copy_dst = Path(tempfile.mkdtemp(dir=TMP))
with contextlib.redirect_stdout(io.StringIO()) as buf:
    out = P.copy_prepared(copy_src, copy_dst, "toy", splits=("train",), workers=2)
assert out == copy_dst / "toy" and sorted(p.name for p in out.iterdir()) == ["manifest.json", *P.split_files(("train",))]
assert (out / "train_images.npy").read_bytes() == (copy_src / "train_images.npy").read_bytes()
assert re.search(r"toy: copied 3 of 3 files of the splits \['train'\], [\d,]+ MB in \d+ s, [\d,]+ MB per second "
                 r"with 2 threads, 0 were there already", buf.getvalue()), buf.getvalue()
assert buf.getvalue().count("MB per second") == 4
with contextlib.redirect_stdout(io.StringIO()) as buf:     # again, every file there with its size is skipped
    P.copy_prepared(copy_src, copy_dst, "toy")
assert "copied 3 of 6 files of the splits ['test', 'train'], " in buf.getvalue() and "3 were there already" in buf.getvalue()
(out / "test_meta.csv").write_text("cut", encoding="utf-8")   # a file of another size is copied again
with contextlib.redirect_stdout(io.StringIO()) as buf:
    P.copy_prepared(copy_src, copy_dst, "toy")
assert "copied 1 of 6 files" in buf.getvalue() and not list(out.glob("*.part"))
assert P.load_split(copy_dst, "toy", "test")["meta"].equals(P.load_split(root, "toy", "test")["meta"])
fails(lambda: P.copy_prepared(copy_src, copy_dst, "toy", splits=("val",)), ValueError, "not in the manifest")
broken = Path(tempfile.mkdtemp(dir=TMP)) / "toy"
shutil.copytree(copy_src, broken)
(broken / "test_labels.npy").unlink()
fresh = Path(tempfile.mkdtemp(dir=TMP))
fails(lambda: P.copy_prepared(broken, fresh, "toy"), FileNotFoundError, "test_labels.npy")
assert not (fresh / "toy").exists(), "a copy began before every file was checked"

SNAP = {"calls": []}


def fake_snapshot(repo_id, repo_type=None, local_dir=None, token=None, allow_patterns=None, max_workers=8):
    SNAP["calls"].append((repo_id, allow_patterns, max_workers, token))
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    for f in copy_src.iterdir():
        if allow_patterns is None or any(fnmatch.fnmatchcase(f.name, p) for p in allow_patterns):
            shutil.copyfile(f, Path(local_dir) / f.name)
    return local_dir


real_snapshot = huggingface_hub.snapshot_download
huggingface_hub.snapshot_download = fake_snapshot
try:
    dest = Path(tempfile.mkdtemp(dir=TMP))
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        H.fetch("toy", "an-owner", dest, token="t", splits=("train",))
    assert SNAP["calls"][-1] == ("an-owner/sar-transfer-toy", ["manifest.json", *P.split_files(("train",))], 8, "t")
    assert sorted(p.name for p in (dest / "toy").iterdir()) == ["manifest.json", *P.split_files(("train",))]
    assert re.search(r"toy: 1 splits ready in .*, \['train'\], [\d,]+ MB in \d+ s, [\d,]+ MB per second with 8 threads",
                     buf.getvalue()), buf.getvalue()
    with contextlib.redirect_stdout(io.StringIO()):
        H.fetch("toy", "an-owner", dest, token="t", workers=3)    # every split, as part two fetched
    assert SNAP["calls"][-1][1:3] == (None, 3) and (dest / "toy" / "test_images.npy").is_file()
    fails(lambda: H.fetch("toy", "an-owner", Path(tempfile.mkdtemp(dir=TMP)), token="t", splits=("val",)),
          ValueError, "not in the manifest")
finally:
    huggingface_hub.snapshot_download = real_snapshot
print("part four data, repo_name, split_files, copy_prepared (splits, skip on size, a cut file again, checks "
      "first), fetch with splits through allow_patterns")

# ---- 12. part four - the saved rows by run name, the reading score, the rule per row, the words ----
res12 = Path(tempfile.mkdtemp(dir=TMP))


def summary12(folder, dataset, encoder, layer, mode, decoder, sd, probe, splits=("val", "test")):
    run = SC.run_name(dataset, encoder, layer, mode)
    rows = [{"run": run, "dataset": dataset, "encoder": encoder, "head": h, "seed": s, "split": sp,
             "mIoU": v, "accuracy": 0.9, "IoU_pos": v - 0.1}
            for h, s, v in (("majority", -1, 0.3), ("probe", -1, probe), ("decoder", 0, decoder - sd),
                            ("decoder", 1, decoder), ("decoder", 2, decoder + sd)) for sp in splits]
    pd.DataFrame(rows).to_csv(Path(folder) / f"{run}__summary.csv", index=False)
    return run


assert SC.run_name("snow", "dinov3-l-sat", 21, "vv_vh_diff") == "dom_snow__dinov3-l-sat__L21__vv_vh_diff"
assert SC.run_name("snow", "student-f", LAST, "co_cross") == "dom_snow__student-f__Llast__co_cross"
r21 = summary12(res12, "snow", "dinov3-l-sat", 21, "vv_vh_diff", 0.6, 0.01, 0.5)
summary12(res12, "snow", "dinov3-l-sat", 24, "vv_vh_diff", 0.9, 0.01, 0.9)          # another layer, a decoy
rf = summary12(res12, "snow", "student-f", LAST, "co_cross", 0.7, 0.02, 0.55, splits=("val",))
rows12, gone12 = SC.read_runs(res12, [r21, rf, "dom_lakes__nobody__L1__co_cross"])
assert gone12 == ["dom_lakes__nobody__L1__co_cross"] and set(rows12["split"]) == {"val"} and len(rows12) == 6
dec = rows12[(rows12["encoder"] == "dinov3-l-sat") & (rows12["head"] == "decoder")].iloc[0]
assert dec["run"] == r21 and dec["seeds"] == 3 and abs(dec["mIoU"] - 0.6) < 1e-12 and abs(dec["mIoU_sd"] - 0.01) < 1e-12
assert abs(dec["IoU_pos"] - 0.5) < 1e-12 and abs(dec["IoU_pos_sd"] - 0.01) < 1e-12
assert len(SC.read_runs(res12, [r21], splits=None)[0]) == 6 and SC.read_runs(res12, [])[1] == []
fails(lambda: SC.read_runs(res12, [r21, SC.run_name("snow", "dinov3-l-sat", 24, "vv_vh_diff")]), ValueError,
      "more than one run")
mixed_in = R.domain_table(res12)                                     # what grouping by encoder would do
assert abs(float(mixed_in[(mixed_in.encoder == "dinov3-l-sat") & (mixed_in["head"] == "decoder")
                          & (mixed_in.split == "val")]["mIoU"].iloc[0]) - 0.75) < 1e-12
# check_against - the same rows agree, a changed number, a nan against a number and a lacking row are named
assert SC.check_against(rows12, rows12).empty
other = rows12.copy()
other.loc[other.index[1], "mIoU"] += 1e-6
other.loc[other.index[2], "IoU_pos_sd"] = 0.5
diff = SC.check_against(rows12, other.iloc[:-1])
assert list(diff["column"]) == ["mIoU", "IoU_pos_sd", "row"], diff
assert SC.check_against(rows12, other.iloc[:1], tolerance=1).shape[0] == 5
fails(lambda: SC.check_against(rows12, pd.concat([rows12, rows12])), ValueError, "twice")
# reading_score - IoU of the positive class for lakes, snow and alpine, mIoU for the others
sc = SC.reading_score(pd.DataFrame({"dataset": ["snow", "forty_v1", "flair_hub"], "mIoU": [0.5, 0.6, 0.7],
                                    "mIoU_sd": [0.05, 0.06, 0.07], "IoU_pos": [0.4, 0.3, float("nan")],
                                    "IoU_pos_sd": [0.04, 0.03, float("nan")]}))
assert list(sc["metric"]) == ["IoU_pos", "mIoU", "mIoU"] and list(sc["score"]) == [0.4, 0.6, 0.7]
assert list(sc["score_sd"]) == [0.04, 0.06, 0.07]
assert list(SC.reading_score(pd.DataFrame({"dataset": ["flair_hub"], "mIoU": [0.7]}))["score"]) == [0.7]
# judge - the rule per row
assert (SC.DECODER_MARGIN, SC.DECODER_SD_TIMES, SC.PROBE_MARGIN, SC.STAGE_A_BEATS) == (0.01, 2.0, 0.02, 4)
assert SC.judge("decoder", 0.55, 0.002, 0.50, 0.002)[0] == "win" and SC.judge("decoder", 0.50, 0.002, 0.55, 0.0)[0] == "loss"
assert SC.judge("decoder", 0.31, 0.0, 0.30, 0.0)[0] == "tie", "a gap of exactly 0.01 is not more than 0.01"
assert SC.judge("decoder", 0.32, 0.0, 0.30, 0.0)[0] == "win"
outcome, gap, needed = SC.judge("decoder", 0.33, 0.015, 0.30, 0.001)
assert outcome == "tie" and abs(needed - 0.03) < 1e-12, "0.03 is not more than 2 x 0.015"
assert SC.judge("decoder", 0.33, float("nan"), 0.30, None)[0] == "win", "one seed has no SD, counted as 0"
assert SC.judge("decoder", 0.27, 0.0, 0.30, 0.016)[0] == "tie" and SC.judge("decoder", 0.26, 0.0, 0.30, 0.016)[0] == "loss"
assert SC.judge("probe", 0.52, None, 0.50, None)[0] == "tie" and SC.judge("probe", 0.53, None, 0.50, None)[0] == "win"
assert SC.judge("probe", 0.47, 0.5, 0.50, 0.5)[0] == "loss", "the probe ignores any SD"
assert SC.judge("probe", float("nan"), None, 0.5, None)[0] == "missing" and SC.judge("decoder", 0.5, 0, None, 0)[0] == "missing"
fails(lambda: SC.judge("majority", 0.5, 0, 0.4, 0), ValueError, "the rule has")
# the words of stage A, over the 4 SAR validation decoder rows
assert [SC.stage_a_word(w, n) for w, n in ((4, 0), (3, 0), (0, 0), (2, 1), (3, 1), (1, 1), (0, 4), (1, 2))] == \
    ["beats", "matches", "matches", "mixed", "mixed", "does not reach", "does not reach", "does not reach"]
assert SC.stage_a_word(6, 0, beats_at=6) == "beats" and SC.stage_a_word(5, 0, beats_at=6) == "matches"   # stage B
assert set(SC.QUESTIONS) == {"1", "2", "3", "4", "5"} and SC.QUESTIONS["1"]["answer"] == ("student-f",)
assert SC.QUESTIONS["5"]["informational"] and SC.AERIAL_COUNTERPART["dinov3-l-sat"] == "dinov3-aerial"
print("part four verdict code, run_name, read_runs by run name (a decoy layer never mixes in), check_against, "
      "reading_score, judge at the margins, the four words")

print("test_score: all checks passed")
