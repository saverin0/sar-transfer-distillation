"""PHI-S, the batcher, the schedule, one tiny training run with checkpoints and a resume."""
import _setup  # noqa: F401

import contextlib
import csv
import io
import math
import shutil
import threading
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sartransfer.distil import phis as P
from sartransfer.distil import train as T
from sartransfer.distil.phis import PhiS, hadamard
from sartransfer.distil.student import MAJORTOM_MODES, Student
from sartransfer.distil.teachers import TeacherOut, build_teacher
from sartransfer.distil.student import distil_loss
from sartransfer.distil.train import (Batcher, Normaliser, Target, TrainConfig, TrainData,
                                      as_targets, batch_loss, fit_phis,
                                      held_out_loss, load_checkpoint, load_student, lr_at,
                                      param_groups, target_of, train_student, train_students)

TMP = _setup.TMP_BASE / "train"
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True)


def raises(exc, fn, *args, says: str = "", **kwargs) -> None:
    name = getattr(fn, "__name__", type(fn).__name__)
    try:
        fn(*args, **kwargs)
    except exc as err:
        if says not in str(err):
            raise SystemExit(f"wrong message from {name}, wanted {says!r}, got {str(err)!r}")
        return
    raise SystemExit(f"bad input accepted by {name}")


# ---- Hadamard matrices, Sylvester and Paley ------------------------------------------------------
for n in (1, 2, 4, 8, 12, 16, 20, 24, 36, 40, 768, 1024, 1152):
    h = hadamard(n)
    assert h.shape == (n, n) and h.dtype == torch.float64
    assert torch.equal(h.abs(), torch.ones(n, n, dtype=torch.float64))
    assert torch.allclose(h @ h.T, n * torch.eye(n, dtype=torch.float64))
assert torch.equal(hadamard(2), torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64))
assert P.is_prime(383) and P.is_prime(1151) and not P.is_prime(767) and not P.is_prime(1)
for n in (0, 3, 6, 10, 18):
    raises(ValueError, hadamard, n)
for n in (52, 100):                    # 4k, but no rule here builds it (52 = 2 x 26, 26 = 25 + 1)
    raises(ValueError, hadamard, n, says=str(n))

# ---- PHI-S, unit variance per channel, exact inverse, one rotation and one scalar ---------------
torch.manual_seed(0)
d = 24
mix = torch.randn(d, d) * torch.linspace(0.1, 10.0, d)          # channels of very different size
x = torch.randn(5000, d) @ mix + torch.linspace(-3.0, 3.0, d)
phis = PhiS.fit(x, name="toy")
assert phis.dim == d and phis.n_fit == 5000 and phis.name == "toy"
assert phis.mean.dtype == phis.rotation.dtype == torch.float32
y = phis(x)
assert y.shape == x.shape and y.dtype == torch.float32
assert torch.allclose(y.mean(0), torch.zeros(d), atol=1e-4)
var = y.var(0, unbiased=True)
assert torch.allclose(var, torch.ones(d), atol=1e-3), var
assert torch.allclose(phis.check(x), torch.ones(d), atol=1e-3)
assert torch.allclose(phis.inverse(y), x, atol=1e-3 * float(x.abs().max()))
assert torch.allclose(phis.rotation @ phis.rotation.T, torch.eye(d), atol=1e-5)   # a rotation
cov = torch.cov(x.T.double())
assert abs(phis.alpha - math.sqrt(d / float(cov.trace()))) < 1e-6 * phis.alpha
assert torch.allclose(phis.eigenvalues.double(), torch.linalg.eigvalsh(cov).flip(0).float().double(),
                      rtol=1e-4, atol=1e-4)
assert phis.eigenvalues[0] >= phis.eigenvalues[-1]
# the standardised space keeps distances up to the scalar (an isometry times alpha)
a, b = torch.randn(7, d), torch.randn(7, d)
assert torch.allclose((phis(a) - phis(b)).norm(dim=1), phis.alpha * (a - b).norm(dim=1), rtol=1e-4)
# on a grid (B, g, g, C) and in float64 too
grid = torch.randn(2, 3, 3, d) @ mix
assert phis(grid).shape == (2, 3, 3, d)
assert torch.allclose(phis.inverse(phis(grid)), grid, atol=1e-3)
assert phis(grid.double()).dtype == torch.float64
# other data does not have unit variance, the check says so
raises(ValueError, phis.check, 3.0 * x, says="not the fitted")
raises(ValueError, phis, torch.randn(4, d + 1))
raises(ValueError, PhiS.fit, torch.randn(1, d))
raises(ValueError, PhiS.fit, torch.zeros(10, d), says="no variance")
raises(ValueError, PhiS.fit, torch.randint(0, 5, (10, d)))
raises(ValueError, PhiS.fit, torch.randn(10, d, 2))
# saved and loaded, the same numbers, the file is written through a temporary name
path = TMP / "phis_toy.pt"
phis.save(path)
assert path.is_file() and not path.with_name("phis_toy.pt.tmp").exists()
back = PhiS.load(path)
assert torch.equal(back.mean, phis.mean) and torch.equal(back.rotation, phis.rotation)
assert back.alpha == phis.alpha and back.name == "toy" and back.n_fit == 5000
assert torch.equal(back(x), y) and torch.equal(back.eigenvalues, phis.eigenvalues)
torch.save({"kind": "other"}, TMP / "not_phis.pt")
raises(ValueError, PhiS.load, TMP / "not_phis.pt")
# a width that is not a power of 2 uses a Paley matrix and still gives unit variance
z = torch.randn(3000, 40) * torch.linspace(1.0, 5.0, 40)
assert torch.allclose(PhiS.fit(z).check(z), torch.ones(40), atol=2e-3)
print("phis: unit variance, exact inverse, save and load ok")

# ---- the schedule, from the config without running -------------------------------------------
cfg = TrainConfig()
assert (cfg.steps, cfg.batch, cfg.n_aerial, cfg.n_majortom, cfg.crop) == (6250, 128, 32, 96, 256)
assert (cfg.lr, cfg.weight_decay, cfg.betas, cfg.warmup, cfg.clip) == (5e-4, 0.05, (0.9, 0.999),
                                                                        500, 1.0)
assert (cfg.seed, cfg.bf16, cfg.checkpoint_every, cfg.held_out_every, cfg.phis_samples) == \
    (0, True, 1000, 1000, 2048)
assert cfg.progress_every == 100
assert (T._duration(3900), T._duration(3300), T._duration(40), T._duration(-1)) == ("1 h 05 min", "55 min", "40 s", "0 s")
assert cfg.student == {} and Student(**cfg.student).n_params() == 88_794_624
assert cfg.steps == 20 * 40_000 // 128        # 20 epochs over the 40,000 pilot samples (30 = 9,375)
assert abs(lr_at(0, cfg) - 1e-6) < 1e-12                       # lr / warmup at the first step
assert abs(lr_at(499, cfg) - 5e-4) < 1e-12 and abs(lr_at(500, cfg) - 5e-4) < 1e-12
assert lr_at(6250, cfg) == 0.0 and lr_at(9375, cfg) == 0.0 and lr_at(20_000, cfg) == 0.0
assert 0 < lr_at(6249, cfg) < 1e-9
lrs = [lr_at(s, cfg) for s in range(cfg.steps + 1)]
assert all(b > a for a, b in zip(lrs[:499], lrs[1:500]))       # rising through the warm-up
assert lrs[499] == lrs[500] == 5e-4                            # the peak, then the cosine
assert all(b < a for a, b in zip(lrs[500:], lrs[501:]))        # falling after it
assert abs(lr_at(500 + (cfg.steps - 500) // 2, cfg) - 2.5e-4) < 2e-7   # half way, half the peak
cfg2 = TrainConfig.from_dict({"steps": 10, "batch": 4, "n_aerial": 1, "crop": 32, "lr": 1e-3,
                              "weight_decay": 0.0, "betas": [0.8, 0.9], "warmup": 2, "clip": 0.5,
                              "seed": 3, "bf16": False, "checkpoint_every": 5, "held_out_every": 5,
                              "phis_samples": 4, "student": None})
assert cfg2.betas == (0.8, 0.9) and cfg2.student == {} and cfg2 != cfg
assert TrainConfig.from_dict(T.asdict(cfg)) == cfg
raises(ValueError, TrainConfig(n_aerial=128).check)
raises(ValueError, TrainConfig(warmup=0).check)

# ---- toy data in the stored dtypes ----------------------------------------------------------------
SIDE, CROP, N_MT, N_AER = 48, 32, 200, 60


def toy_arrays(n_mt: int, n_aer: int, seed: int) -> tuple:
    g = np.random.default_rng(seed)
    s1 = (g.normal(-15.0, 5.0, (n_mt, 2, SIDE, SIDE))).astype(np.float16)
    s2 = g.integers(1, 8000, (n_mt, 12, SIDE, SIDE), dtype=np.uint16)
    aerial = g.integers(0, 256, (n_aer, 4, CROP, CROP), dtype=np.uint8)
    return s1, s2, aerial


s1, s2, aerial = toy_arrays(N_MT, N_AER, 0)
h_s1, h_s2, h_aerial = toy_arrays(20, 10, 1)
stats = {"s1": {"mean": [-15.0, -15.0], "sd": [5.0, 5.0]},
         "s2": {"mean": [4000.0] * 12, "sd": [2300.0] * 12},
         "aerial": {"mean": [127.5] * 4, "sd": [74.0] * 4}}
data = TrainData(s1, s2, aerial, stats, h_s1, h_s2, h_aerial)
data.check(CROP)
assert data.has_held_out and not TrainData(s1, s2, aerial, stats).has_held_out
raises(ValueError, TrainData(s1.astype(np.float32), s2, aerial, stats).check, CROP, says="float16")
raises(ValueError, TrainData(s1, s2.astype(np.int32), aerial, stats).check, CROP, says="uint16")
raises(ValueError, TrainData(s1, s2, aerial[:, :3], stats).check, CROP, says="(n, 4")
raises(ValueError, TrainData(s1[:10], s2, aerial, stats).check, CROP, says="not pairs")
raises(ValueError, TrainData(s1, s2, aerial, stats).check, 64, says="crop")
raises(ValueError, TrainData(s1, s2, aerial, {"s1": stats["s1"]}).check, CROP, says="stats")
bad = dict(stats, s2={"mean": [1.0] * 12, "sd": [0.0] * 12})
raises(ValueError, TrainData(s1, s2, aerial, bad).check, CROP, says="above 0")
norm = Normaliser(stats, "cpu")
assert torch.allclose(norm("s1", torch.full((1, 2, 4, 4), -10.0)), torch.ones(1, 2, 4, 4))

# ---- the batcher, 96 + 32, modes per sample, crops and flips ------------------------------------
big = TrainConfig(crop=CROP, student=dict(img=CROP, patch=16, dim=32, depth=2, heads=4))
bat = Batcher(s1, s2, aerial, big, "cpu")
seen_modes = set()
for step in range(3):
    b = bat.batch(step)
    assert b.n == 128 and b.n_majortom == 96 and len(b.modes) == 128
    assert all(m in MAJORTOM_MODES for m in b.modes[:96]) and b.modes[96:] == ["aerial"] * 32
    assert len(set(b.modes[:96])) > 1                          # drawn per sample, not per batch
    seen_modes |= set(b.modes[:96])
    assert b.s1.shape == (96, 2, CROP, CROP) and b.s2.shape == (96, 12, CROP, CROP)
    assert b.aerial.shape == (32, 4, CROP, CROP)
    assert b.s1.dtype == b.s2.dtype == b.aerial.dtype == torch.float32
    assert float(b.s2.min()) >= 1 and float(b.s2.max()) < 8000 and float(b.aerial.max()) <= 255
    assert b.index["corner"].shape == (96, 2) and b.index["corner"].max() <= SIDE - CROP
    assert b.index["flip_majortom"].shape == (96, 2) and b.index["flip_aerial"].shape == (32, 2)
    inputs = b.student_inputs(norm)
    assert inputs["modes"] == b.modes and inputs["s2"].shape == (96, 12, CROP, CROP)
    assert torch.allclose(inputs["s2"], (b.s2 - 4000.0) / 2300.0)
    assert abs(float(inputs["aerial"].mean())) < 0.1 and abs(float(inputs["s1"].mean())) < 0.1
assert seen_modes == set(MAJORTOM_MODES)
counts = {m: 0 for m in MAJORTOM_MODES}
for step in range(30):
    for m in bat.batch(step).modes[:96]:
        counts[m] += 1
assert all(0.28 < c / (30 * 96) < 0.39 for c in counts.values()), counts   # equal chance
# the sampling rule, every sample once per pass, the batch of a step depends on the step alone
seq = np.concatenate([bat.stream_index("majortom", s) for s in range(25)])   # 2,400 = 12 passes
assert len(seq) == 12 * N_MT
for p in range(12):
    assert np.array_equal(np.sort(seq[p * N_MT:(p + 1) * N_MT]), np.arange(N_MT)), p
assert not np.array_equal(seq[:N_MT], seq[N_MT:2 * N_MT])                 # a new permutation
seq_a = np.concatenate([bat.stream_index("aerial", s) for s in range(15)])   # 480 = 8 passes
for p in range(8):
    assert np.array_equal(np.sort(seq_a[p * N_AER:(p + 1) * N_AER]), np.arange(N_AER)), p
again = Batcher(s1, s2, aerial, big, "cpu")
for step in (0, 7, 23):
    x, y = bat.batch(step), again.batch(step)
    assert x.modes == y.modes and torch.equal(x.s1, y.s1) and torch.equal(x.s2, y.s2)
    assert torch.equal(x.aerial, y.aerial)
    assert np.array_equal(x.index["majortom"], y.index["majortom"])
other = Batcher(s1, s2, aerial, big, "cpu", seed=1)
assert not np.array_equal(other.batch(0).index["majortom"], bat.batch(0).index["majortom"])
raises(ValueError, bat.batch, -1)
raises(ValueError, bat.fixed)
# a marked pixel lands at the same place in s1, s2, the student input and the teacher input
mark = 17
s1m, s2m = s1.copy(), s2.copy()
s1m[mark, 0, 20, 30] = 100.0
s2m[mark, 5, 20, 30] = 65000
batm = Batcher(s1m, s2m, aerial, big, "cpu")
found = 0
for step in range(3):
    b = batm.batch(step)
    for k in np.flatnonzero(b.index["majortom"] == mark):
        r, c = b.index["corner"][k]
        h, v = b.index["flip_majortom"][k]
        row, col = 20 - r, 30 - c
        row = CROP - 1 - row if v else row
        col = CROP - 1 - col if h else col
        inputs = b.student_inputs(norm)
        for name, img in (("s1 teacher side", b.s1[k, 0]), ("s2 teacher side", b.s2[k, 5]),
                          ("s1 student", inputs["s1"][k, 0]), ("s2 student", inputs["s2"][k, 5])):
            at = divmod(int(img.argmax()), CROP)
            assert at == (row, col), (name, at, (row, col), (r, c, h, v))
        found += 1
assert found == 1, found
# flips are applied, the corner is used
b0 = batm.batch(0)
k = int(np.flatnonzero(b0.index["flip_majortom"][:, 0])[0])
i, (r, c) = b0.index["majortom"][k], b0.index["corner"][k]
raw = torch.from_numpy(s1m[i, :, r:r + CROP, c:c + CROP].astype(np.float32))
if b0.index["flip_majortom"][k, 1]:
    raw = raw.flip(-2)
assert torch.equal(b0.s1[k], raw.flip(-1))
# fixed batches, centre crop, no flips, modes in turn
fb = bat.fixed(range(5), range(3))
assert fb.modes == ["s1", "s2", "both", "s1", "s2", "aerial", "aerial", "aerial"]
assert (fb.index["corner"] == (SIDE - CROP) // 2).all() and not fb.index["flip_majortom"].any()
assert torch.equal(fb.s2[1], torch.from_numpy(s2[1, :, 8:40, 8:40].astype(np.float32)))
assert bat.fixed(majortom=range(2)).aerial is None and bat.fixed(aerial=range(2)).s2 is None
# the stats are applied per channel in the stored order (VV, VH), S2_BANDS, (R, G, B, NIR)
per_ch = {"s1": {"mean": [-12.0, -19.0], "sd": [5.0, 6.0]},
          "s2": {"mean": [100.0 * (i + 1) for i in range(12)], "sd": [10.0 * (i + 1) for i in range(12)]},
          "aerial": {"mean": [100.0, 110.0, 120.0, 130.0], "sd": [50.0, 55.0, 60.0, 65.0]}}
b0, in0 = bat.batch(0), bat.batch(0).student_inputs(Normaliser(per_ch, "cpu"))
for key, x in (("s1", b0.s1), ("s2", b0.s2), ("aerial", b0.aerial)):
    for i in range(x.shape[1]):
        want = (x[:, i] - per_ch[key]["mean"][i]) / per_ch[key]["sd"][i]
        assert torch.allclose(in0[key][:, i], want), (key, i)
# uint16 goes to the device as int16 (CUDA support for uint16 is limited), the values come back
u16 = np.array([[0, 1, 32767, 32768, 65000, 65535]], dtype=np.uint16)
assert Batcher._host(u16).dtype == torch.int16 and Batcher._host(s1[:1]).dtype == torch.float16
assert Batcher._host(aerial[:1]).dtype == torch.uint8
assert torch.equal(bat._to_device(u16), torch.tensor([[0.0, 1.0, 32767.0, 32768.0, 65000.0, 65535.0]]))
assert torch.equal(bat._to_device(u16[:, ::2]), torch.tensor([[0.0, 32767.0, 65000.0]]))
assert bat._to_device(s1[:1]).dtype == torch.float32
assert max(float(batm.batch(s).s2.max()) for s in range(3)) == 65000.0     # the marked pixel
print("batcher: 96 + 32, modes per sample, sampling rule, marked pixel, channel order, uint16 ok")


# ---- toy teachers, frozen functions of the input, like the real ones ---------------------------
class ToyTeacher:
    def __init__(self, name: str, dim: int, grid: int, in_chans: int, scale: float, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.name, self.dim, self.grid, self.scale = name, dim, grid, scale
        self.w_sum = torch.randn(in_chans, dim, generator=g) / in_chans ** 0.5
        self.w_spa = torch.randn(in_chans, dim, generator=g) / in_chans ** 0.5
        self.calls = 0

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> TeacherOut:
        assert x.ndim == 4 and x.shape[1] == self.w_sum.shape[0] and torch.is_floating_point(x)
        self.calls += 1
        x = x.float() / self.scale
        pooled = F.adaptive_avg_pool2d(x, self.grid).permute(0, 2, 3, 1)
        return TeacherOut(torch.tanh(x.mean((2, 3)) @ self.w_sum), torch.tanh(pooled @ self.w_spa))


ta = ToyTeacher("ta", 8, 2, 12, 1000.0, 10)
ta_aerial = ToyTeacher("ta", 8, 2, 4, 255.0, 11)
tb = ToyTeacher("tb", 24, 3, 12, 1000.0, 12)
targets = [Target("ta", 8, 2, ta, ta_aerial), Target("tb", 24, 3, tb)]
assert [t.name for t in as_targets(targets)] == ["ta", "tb"]
conv = as_targets([tb])
assert conv[0].name == "tb" and conv[0].dim == 24 and conv[0].grid == 3 and conv[0].aerial is None
assert conv[0].s2 is tb
raises(ValueError, as_targets, [])
raises(ValueError, as_targets, [tb, tb], says="unique")
b = bat.batch(0)
(out_a, rows_a), (out_b, rows_b) = target_of(targets[0], b), target_of(targets[1], b)
assert out_a.summary.shape == (128, 8) and out_a.spatial.shape == (128, 2, 2, 8)
assert out_b.summary.shape == (96, 24) and out_b.spatial.shape == (96, 3, 3, 24)
assert rows_a == slice(0, 128) and rows_b == slice(0, 96)       # the first rows, a slice as in part three
assert target_of(targets[1], bat.fixed(aerial=range(2))) is None
out_aer, rows_aer = target_of(targets[0], bat.fixed(aerial=range(2)))
assert out_aer.summary.shape == (2, 8) and rows_aer == slice(0, 2)

# PHI-S fitted through the batcher on the first n samples that reach the teacher
small = TrainConfig(steps=12, batch=8, n_aerial=2, crop=CROP, warmup=3, checkpoint_every=5,
                    held_out_every=5, phis_samples=20, bf16=False,
                    student=dict(img=CROP, patch=16, dim=32, depth=2, heads=4))
small.check()
sb = Batcher(s1, s2, aerial, small, "cpu")
ta.calls = 0
ph = fit_phis(lambda bb: target_of(targets[0], bb), sb, n=20, name="ta")
assert ph.dim == 8 and ph.n_fit == 20 * 4 and ta.calls == 3                 # 8 per batch, 3 batches
feats = torch.cat([target_of(targets[0], sb.batch(s))[0].spatial for s in range(3)])[:20]
assert torch.allclose(ph.check(feats.reshape(-1, 8)), torch.ones(8), atol=1e-3)
raises(RuntimeError, fit_phis, lambda bb: None, sb, n=4)
ph_b = fit_phis(lambda bb: target_of(targets[1], bb), sb, n=20, name="tb")

# batch_loss by hand - the teacher without aerial gets exactly the Major-TOM rows (the first ones)
torch.manual_seed(0)
st0, hd0 = Student(**small.student), T.Heads({"ta": 8, "tb": 24}, student_dim=32)
bb = sb.batch(0)
auto0 = T._autocast(torch.device("cpu"), small)
loss0, parts0 = batch_loss(st0, hd0, targets, bb, norm, {"ta": ph, "tb": ph_b}, auto0)
assert (parts0["ta/rows"], parts0["tb/rows"]) == (8, 6)
out0 = hd0(*st0.forward_mixed(bb.student_inputs(norm)))
want0, want_parts0 = distil_loss({"ta": out0["ta"], "tb": (out0["tb"][0][:6], out0["tb"][1][:6])},
                                 {"ta": target_of(targets[0], bb)[0], "tb": tb(bb.s2)},
                                 standardise={"ta": ph, "tb": ph_b})
assert abs(float(loss0.detach()) - float(want0.detach())) < 1e-6
assert all(abs(want_parts0[k] - parts0[k]) < 1e-6 for k in want_parts0), (want_parts0, parts0)
_, last_rows = distil_loss({"tb": (out0["tb"][0][-6:], out0["tb"][1][-6:])}, {"tb": tb(bb.s2)},
                           standardise={"tb": ph_b})
assert abs(last_rows["tb/spatial"] - parts0["tb/spatial"]) > 1e-4       # the last rows are wrong
# weight decay on the weights of rank 2 and up only, every parameter in exactly one group
g_decay, g_none = param_groups(st0, hd0, 0.05)
assert (g_decay["weight_decay"], g_none["weight_decay"]) == (0.05, 0.0)
named = list(st0.named_parameters()) + [("heads." + k, v) for k, v in hd0.named_parameters()]
ids_decay, ids_none = {id(p) for p in g_decay["params"]}, {id(p) for p in g_none["params"]}
assert not ids_decay & ids_none and len(ids_decay | ids_none) == len(named)
for name, p in named:
    assert (id(p) in ids_none) == (p.ndim <= 1 or name in ("pos_embed", "cls_token")), name
assert id(st0.pos_embed) in ids_none and id(st0.cls_token) in ids_none
assert id(st0.blocks[0].norm1.weight) in ids_none and id(st0.blocks[0].qkv.bias) in ids_none
assert id(st0.blocks[0].qkv.weight) in ids_decay and id(st0.embed_s2.weight) in ids_decay
assert id(hd0.summary["ta"][1].weight) in ids_none and id(hd0.summary["ta"][0].weight) in ids_decay
print("batch_loss rows and param groups ok")

# ---- one run of 12 steps, checkpoints every 5 steps -------------------------------------------
dir_a = TMP / "run_a"
res_a = train_student(small, data, targets, dir_a, device="cpu")
assert res_a["step"] == 12 and (dir_a / "last.pt").is_file() and (dir_a / "final.pt").is_file()
assert (dir_a / "phis_ta.pt").is_file() and (dir_a / "phis_tb.pt").is_file()
assert not list(dir_a.glob("*.tmp"))
with open(dir_a / "train_log.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
assert list(rows[0]) == ["step", "lr", "loss", "ta/summary", "ta/spatial", "tb/summary",
                         "tb/spatial", "grad_norm", "step_s", "data_s", "teacher_s", "student_s",
                         "img_s", "gpu_gb"]
assert T.LOG_END == ("grad_norm", "step_s", "data_s", "teacher_s", "student_s", "img_s", "gpu_gb")
assert [int(r["step"]) for r in rows] == list(range(1, 13))
for r in rows:
    assert all(math.isfinite(float(v)) for v in r.values())
    assert float(r["step_s"]) > float(r["data_s"]) > 0 and float(r["img_s"]) > 0
    assert float(r["gpu_gb"]) == 0.0 and float(r["grad_norm"]) > 0
    # the split of a step: the teacher forwards and the student from its forward to its update
    assert float(r["teacher_s"]) >= 0 and float(r["student_s"]) > 0
    assert float(r["data_s"]) + float(r["teacher_s"]) + float(r["student_s"]) <= float(r["step_s"]) + 2e-4
    assert abs(float(r["loss"]) - sum(float(r[k]) for k in ("ta/summary", "ta/spatial",
                                                             "tb/summary", "tb/spatial"))) < 1e-5
assert all(abs(float(r["lr"]) - lr_at(int(r["step"]) - 1, small)) < 1e-6 * small.lr
           for r in rows)                                          # logged with 6 digits
assert float(rows[-1]["loss"]) < float(rows[0]["loss"])
with open(dir_a / "held_out.csv", newline="", encoding="utf-8") as f:
    held_rows = list(csv.DictReader(f))
assert list(held_rows[0]) == ["step", "loss", "ta/summary", "ta/spatial", "tb/summary", "tb/spatial"]
assert [int(r["step"]) for r in held_rows] == [5, 10, 12]
assert all(math.isfinite(float(v)) and float(v) > 0 for r in held_rows for v in r.values())
ho = res_a["held_out"]
assert abs(ho["loss"] - float(held_rows[-1]["loss"])) < 1e-5
ho_again = held_out_loss(res_a["student"], res_a["heads"], targets,
                         Batcher(h_s1, h_s2, h_aerial, small, "cpu"), norm, res_a["phis"], small,
                         T._autocast(torch.device("cpu"), small))
assert ho_again == ho and res_a["student"].training                     # eval mode is left again
# the checkpoint holds everything, and load_student gives the trained model back
ck = load_checkpoint(dir_a / "last.pt")
assert ck["step"] == 12 and ck["stats"] == stats and TrainConfig.from_dict(ck["config"]) == small
assert ck["teacher_dims"] == {"ta": 8, "tb": 24} and ck["optimizer"] is not None
assert ck["scheduler"]["last_epoch"] == 12 and "torch" in ck["rng"]
assert load_checkpoint(dir_a / "final.pt")["optimizer"] is None
raises(ValueError, load_checkpoint, TMP / "not_phis.pt")
for name in ("final.pt", "last.pt"):
    loaded, loaded_stats, loaded_cfg = load_student(dir_a / name)
    assert not loaded.training and loaded_stats == stats and loaded_cfg == small
    assert loaded.n_params() == res_a["student"].n_params()
    inputs = bat.fixed(range(4), range(2)).student_inputs(norm)
    with torch.no_grad():
        got, want = loaded.forward_mixed(inputs), res_a["student"].forward_mixed(inputs)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
    for k, v in loaded.state_dict().items():
        assert torch.equal(v, res_a["student"].state_dict()[k]), k

# ---- cut after step 10, resume, the same final weights as the run in one go --------------------
dir_b = TMP / "run_b"
res_b1 = train_student(small, data, targets, dir_b, device="cpu", stop_after=10)
assert res_b1["step"] == 10 and load_checkpoint(dir_b / "last.pt")["step"] == 10
assert not (dir_b / "final.pt").exists()
with open(dir_b / "train_log.csv", newline="", encoding="utf-8") as f:
    assert len(list(csv.DictReader(f))) == 10
for name in ("phis_ta.pt", "phis_tb.pt"):
    assert torch.equal(PhiS.load(dir_b / name).rotation, PhiS.load(dir_a / name).rotation)
raises(ValueError, train_student, TrainConfig(**{**T.asdict(small), "lr": 1e-3}), data, targets,
       dir_b, device="cpu", says="another config")
res_b2 = train_student(small, data, targets, dir_b, device="cpu")
assert res_b2["step"] == 12 and (dir_b / "final.pt").is_file()
for module in ("student", "heads"):
    for k, v in res_b2[module].state_dict().items():
        assert torch.equal(v, res_a[module].state_dict()[k]), (module, k)
with open(dir_b / "train_log.csv", newline="", encoding="utf-8") as f:
    rows_b = list(csv.DictReader(f))
assert [int(r["step"]) for r in rows_b] == list(range(1, 13))
assert [r["loss"] for r in rows_b] == [r["loss"] for r in rows]
with open(dir_b / "held_out.csv", newline="", encoding="utf-8") as f:
    assert [int(r["step"]) for r in csv.DictReader(f)] == [5, 10, 12]
# nothing left to do = nothing changes
res_b3 = train_student(small, data, targets, dir_b, device="cpu")
assert res_b3["step"] == 12
assert all(torch.equal(v, res_a["student"].state_dict()[k])
           for k, v in res_b3["student"].state_dict().items())
# resume=False starts over and gives the run in one go again
res_c = train_student(small, data, targets, dir_b, device="cpu", resume=False)
assert all(torch.equal(v, res_a["student"].state_dict()[k])
           for k, v in res_c["student"].state_dict().items())
# cut one step after the last checkpoint, the log rows after it are dropped on the resume,
# and so is a torn last line (a run killed while writing - "1" is the start of "12,...")
dir_f = TMP / "run_f"
train_student(small, data, targets, dir_f, device="cpu", stop_after=11)
with open(dir_f / "train_log.csv", newline="", encoding="utf-8") as f:
    assert [int(r["step"]) for r in csv.DictReader(f)] == list(range(1, 12))
for name, torn in (("train_log.csv", "1"), ("held_out.csv", "1,2.9")):
    with open(dir_f / name, "a", newline="", encoding="utf-8") as f:
        f.write(torn)
res_f = train_student(small, data, targets, dir_f, device="cpu")
with open(dir_f / "held_out.csv", newline="", encoding="utf-8") as f:
    held_f = list(csv.reader(f))
assert [r[0] for r in held_f] == ["step", "5", "10", "12"] and all(len(r) == 6 for r in held_f)
with open(dir_f / "train_log.csv", newline="", encoding="utf-8") as f:
    rows_f = list(csv.DictReader(f))
assert [int(r["step"]) for r in rows_f] == list(range(1, 13))
assert [r["loss"] for r in rows_f] == [r["loss"] for r in rows]
assert all(len(r) == 7 + len(T.LOG_END) and None not in r.values() for r in rows_f)
assert all(torch.equal(v, res_a["student"].state_dict()[k])
           for k, v in res_f["student"].state_dict().items())
# another seed and another cut point - checkpoints every 4, cut at 6, resumed from 4
cfg5 = TrainConfig(**{**T.asdict(small), "seed": 5, "checkpoint_every": 4, "held_out_every": 4})
res_5 = train_student(cfg5, data, targets, TMP / "run_5", device="cpu")
train_student(cfg5, data, targets, TMP / "run_5cut", device="cpu", stop_after=6)
assert load_checkpoint(TMP / "run_5cut" / "last.pt")["step"] == 4
res_5cut = train_student(cfg5, data, targets, TMP / "run_5cut", device="cpu")
assert res_5cut["step"] == 12
for module in ("student", "heads"):
    for k, v in res_5cut[module].state_dict().items():
        assert torch.equal(v, res_5[module].state_dict()[k]), (module, k)
for name, drop in (("train_log.csv", set(T.LOG_END) - {"grad_norm"}), ("held_out.csv", set())):
    logs = []
    for d in ("run_5", "run_5cut"):
        with open(TMP / d / name, newline="", encoding="utf-8") as f:
            logs.append([{k: v for k, v in r.items() if k not in drop} for r in csv.DictReader(f)])
    assert logs[0] == logs[1] and len(logs[0]) > 2, name          # the timing columns differ
assert not torch.equal(res_5["student"].pos_embed, res_a["student"].pos_embed)
# gradient clipping is applied every step, on every parameter, at cfg.clip
clip_calls = []
_clip = torch.nn.utils.clip_grad_norm_


def _clip_spy(parameters, max_norm, *a, **kw):
    parameters = list(parameters)
    total = _clip(parameters, max_norm, *a, **kw)
    after = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in parameters if p.grad is not None))
    clip_calls.append((float(total), float(max_norm), float(after), len(parameters)))
    return total


torch.nn.utils.clip_grad_norm_ = _clip_spy
try:
    tight = TrainConfig(**{**T.asdict(small), "steps": 3, "clip": 0.05})
    res_clip = train_student(tight, TrainData(s1, s2, aerial, stats), targets, TMP / "run_clip",
                             device="cpu")
finally:
    torch.nn.utils.clip_grad_norm_ = _clip
n_all = len(list(res_clip["student"].parameters())) + len(list(res_clip["heads"].parameters()))
assert len(clip_calls) == 3, clip_calls
for total, max_norm, after, n in clip_calls:
    assert max_norm == 0.05 and n == n_all and total > 0.05, (total, max_norm, n)
    assert after <= 0.05 * (1 + 1e-5), (total, after)
with open(TMP / "run_clip" / "train_log.csv", newline="", encoding="utf-8") as f:
    logged = [float(r["grad_norm"]) for r in csv.DictReader(f)]
assert all(abs(a - c[0]) < 1e-3 for a, c in zip(logged, clip_calls))     # the norm before clipping
# another seed, other weights
other = TrainConfig(**{**T.asdict(small), "seed": 1})
res_d = train_student(other, data, targets, TMP / "run_d", device="cpu")
assert not torch.equal(res_d["student"].pos_embed, res_a["student"].pos_embed)
print("training: 12 steps, checkpoints, resume from step 10 = run in one go, load_student ok")


def same_state(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def printed(fn, *args, **kwargs) -> tuple:
    """What `fn` returns and what it printed."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = fn(*args, **kwargs)
    return result, out.getvalue()


def log_of(folder: Path) -> list:
    with open(folder / "train_log.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prefetch_threads() -> list:
    return [t for t in threading.enumerate() if t.name == T.Prefetch.THREAD_NAME]


def snapshot(*folders: Path) -> dict:
    """Every file of the folders with its bytes, to show that a refusal changed nothing."""
    return {(f.name, p.name): p.read_bytes() for f in folders for p in f.iterdir()}


rows_a = log_of(dir_a)

# ---- the previous checkpoint. last_prev.pt is the checkpoint before last.pt, the fallback ------
assert T.PREV == "last_prev.pt"
assert (load_checkpoint(dir_a / "last.pt")["step"], load_checkpoint(dir_a / "last_prev.pt")["step"]) == (12, 10)


def copy_run(name: str, drop=(), cut=()) -> Path:
    """dir_a copied without final.pt, some files dropped, some cut to their first half (a
    runtime that died while the file was written)."""
    folder = TMP / name
    shutil.copytree(dir_a, folder)
    (folder / "final.pt").unlink()
    for file in drop:
        (folder / file).unlink()
    for file in cut:
        whole = (folder / file).read_bytes()
        (folder / file).write_bytes(whole[:len(whole) // 2])
    return folder


# last.pt cut short: the previous one is used with a printed line, the log rows after its step
# are dropped, and the end equals the run in one go
dir_g = copy_run("run_g", cut=("last.pt",))
res_g, said_g = printed(train_student, small, data, targets, dir_g, device="cpu")
assert "last.pt cannot be loaded" in said_g and "the previous checkpoint is tried" in said_g, said_g
assert f"resumed from {dir_g / 'last_prev.pt'} at step 10" in said_g
assert res_g["step"] == 12 and same_state(res_g["student"], res_a["student"])
assert [r["loss"] for r in log_of(dir_g)] == [r["loss"] for r in rows_a]
# the cut last.pt was moved aside, so the rotation kept the good previous checkpoint
assert "last.pt moved to last.pt.broken" in said_g and (dir_g / "last.pt.broken").is_file()
assert (load_checkpoint(dir_g / "last.pt")["step"], load_checkpoint(dir_g / "last_prev.pt")["step"]) == (12, 10)
assert not list(dir_g.glob("*.tmp"))
# last.pt missing (a runtime that died between the rename and the new write): the previous one
dir_h = copy_run("run_h", drop=("last.pt",))
res_h, said_h = printed(train_student, small, data, targets, dir_h, device="cpu")
assert "last.pt is missing" in said_h and f"resumed from {dir_h / 'last_prev.pt'} at step 10" in said_h
assert res_h["step"] == 12 and same_state(res_h["student"], res_a["student"])
# both there and neither loads: the run stops, nothing is overwritten
dir_i = copy_run("run_i", cut=("last.pt", "last_prev.pt"))
before_i = {p.name: p.read_bytes() for p in dir_i.iterdir()}
_, said_i = printed(raises, RuntimeError, train_student, small, data, targets, dir_i, device="cpu",
                    says="can be loaded")
assert "the previous checkpoint is tried" in said_i and "no other checkpoint is there" in said_i, said_i
assert {p.name: p.read_bytes() for p in dir_i.iterdir()} == before_i
# only a cut last_prev.pt: the same stop. resume=False ignores the checkpoints and starts over
dir_j = copy_run("run_j", drop=("last.pt",), cut=("last_prev.pt",))
raises(RuntimeError, train_student, small, data, targets, dir_j, device="cpu",
       says="can be loaded (['last_prev.pt']). Delete them to start over")
res_j = train_student(small, data, targets, dir_j, device="cpu", resume=False)
assert same_state(res_j["student"], res_a["student"])
# the rotation at every checkpoint: last.pt becomes last_prev.pt, the older one goes, no .tmp stays
dir_k = TMP / "run_k"
train_student(small, data, targets, dir_k, device="cpu", stop_after=5)
assert load_checkpoint(dir_k / "last.pt")["step"] == 5 and not (dir_k / "last_prev.pt").exists()
train_student(small, data, targets, dir_k, device="cpu", stop_after=10)
assert (load_checkpoint(dir_k / "last.pt")["step"], load_checkpoint(dir_k / "last_prev.pt")["step"]) == (10, 5)
res_k = train_student(small, data, targets, dir_k, device="cpu")
assert (load_checkpoint(dir_k / "last.pt")["step"], load_checkpoint(dir_k / "last_prev.pt")["step"]) == (12, 10)
assert not list(dir_k.glob("*.tmp")) and same_state(res_k["student"], res_a["student"])
assert not prefetch_threads()
print("previous checkpoint: rotation, fallback from a cut or a missing last.pt, stop when none loads")

# ---- the progress line: every progress_every steps and at the end, step, loss, s/step, time left
paced = TrainConfig(**{**T.asdict(small), "progress_every": 4})
_, said_p = printed(train_student, paced, data, targets, TMP / "paced", device="cpu")
lines_p = [ln for ln in said_p.splitlines() if ln.startswith("step ") and " s/step" in ln]
assert [ln.split("  ")[0] for ln in lines_p] == ["step 4 of 12", "step 8 of 12", "step 12 of 12"], lines_p
assert all(" loss " in ln and " s/step" in ln for ln in lines_p), lines_p
assert all(ln.endswith(" left") and "about " in ln for ln in lines_p[:-1]), lines_p
assert lines_p[-1].endswith("  finished"), lines_p
steps_g = [ln for ln in said_g.splitlines() if ln.startswith("step ") and " s/step" in ln]
assert steps_g == [ln for ln in said_g.splitlines() if ln.startswith("step 12 of 12  loss ")]
assert len(steps_g) == 1                                                         # default 100, 12 steps
raises(ValueError, TrainConfig(**{**T.asdict(small), "progress_every": 0}).check)
print("progress line: every progress_every steps and at the end, one line with the default")

# ---- the prefetch thread. The same numbers with and without it, no thread left, its error surfaces
dir_p = TMP / "run_p"
res_p = train_student(small, data, targets, dir_p, device="cpu", prefetch=False)
assert same_state(res_p["student"], res_a["student"]) and same_state(res_p["heads"], res_a["heads"])
timing = set(T.LOG_END) - {"grad_norm"}


def no_timing(log_rows: list) -> list:
    return [{k: v for k, v in r.items() if k not in timing} for r in log_rows]


assert no_timing(log_of(dir_p)) == no_timing(rows_a)
assert (dir_p / "held_out.csv").read_bytes() == (dir_a / "held_out.csv").read_bytes()
assert not prefetch_threads()
# cut with the thread, resumed without it (and the other way round): still the run in one go
dir_r = TMP / "run_r"
train_student(small, data, targets, dir_r, device="cpu", stop_after=10, prefetch=True)
res_r = train_student(small, data, targets, dir_r, device="cpu", prefetch=False)
assert same_state(res_r["student"], res_a["student"])
dir_t = TMP / "run_t"                                   # (run_s belongs to the student S test below)
train_student(small, data, targets, dir_t, device="cpu", stop_after=7, prefetch=False)
res_t = train_student(small, data, targets, dir_t, device="cpu", prefetch=True)
assert same_state(res_t["student"], res_a["student"]) and not prefetch_threads()
# host + to_device = batch. The host batch carries the stored dtypes, nothing is pinned on the CPU
hb = bat.host(3)
assert hb.step == 3 and hb.s2.dtype == torch.int16 and hb.s1.dtype == torch.float16
assert hb.aerial.dtype == torch.uint8 and not hb.s2.is_pinned() and bat.pin is False
b3, b3_again = bat.to_device(hb), bat.batch(3)
assert b3.modes == b3_again.modes and torch.equal(b3.s2, b3_again.s2) and torch.equal(b3.s1, b3_again.s1)
assert torch.equal(b3.aerial, b3_again.aerial) and float(b3.s2.max()) == float(b3_again.s2.max())
# the prefetch alone: the steps in order, the thread gone after close, none without the thread
feed = T.Prefetch(bat, 2, 6)
assert feed.threaded and len(prefetch_threads()) == 1
for s in range(2, 6):
    got = feed.get(s)
    assert got.step == s and torch.equal(got.s2, bat.host(s).s2)
raises(ValueError, feed.get, 9, says="next")
feed.close()
assert not prefetch_threads() and not feed.threaded
early = T.Prefetch(bat, 0, 50)                          # closed while it is one step ahead
assert early.get(0).step == 0
early.close()
assert not prefetch_threads()
plain = T.Prefetch(bat, 0, 3, enabled=False)
assert not plain.threaded and plain.get(0).step == 0 and not prefetch_threads()
assert not T.Prefetch(bat, 5, 5).threaded               # nothing to build, no thread
# an error in the thread is raised again in the loop (the same error), the thread is gone, the
# log holds the steps that ran, and the run resumes from its checkpoint once the cause is fixed
real_host = Batcher.host


def host_that_breaks(self, step):
    if step == 7:
        raise RuntimeError("boom at step 7")
    return real_host(self, step)


dir_q = TMP / "run_q"
Batcher.host = host_that_breaks
try:
    raises(RuntimeError, train_student, small, data, targets, dir_q, device="cpu", says="boom at step 7")
    assert not prefetch_threads()
    assert [int(r["step"]) for r in log_of(dir_q)] == list(range(1, 8))     # steps 1 to 7 ran
    assert load_checkpoint(dir_q / "last.pt")["step"] == 5
    raises(RuntimeError, train_student, small, data, targets, dir_q, device="cpu", prefetch=False,
           says="boom at step 7")
    assert not prefetch_threads()
finally:
    Batcher.host = real_host
res_q = train_student(small, data, targets, dir_q, device="cpu")
assert same_state(res_q["student"], res_a["student"]) and not prefetch_threads()
print("prefetch: same numbers with and without the thread, resume unchanged, error surfaces, no thread left")

# ---- student S, one teacher, no aerial target = no aerial rows in the loss -------------------
res_s = train_student(small, data, [Target("tb", 24, 3, tb)], TMP / "run_s", device="cpu")
assert list(res_s["heads"].teacher_dims) == ["tb"]
with open(TMP / "run_s" / "train_log.csv", newline="", encoding="utf-8") as f:
    assert list(csv.DictReader(f))[0].keys() >= {"tb/summary", "tb/spatial"}
raises(ValueError, train_student, small, TrainData(s1, s2, aerial, stats), [], TMP / "run_e",
       device="cpu")


# ---- a loss that is not finite stops the run, the last checkpoint stays ------------------------
class PoisonTeacher(ToyTeacher):
    """A ToyTeacher whose spatial features are NaN from its `poison_from`-th call on."""

    poison_from = 10 ** 9

    def __call__(self, x):
        out = super().__call__(x)
        if self.calls >= self.poison_from:
            return TeacherOut(out.summary, torch.full_like(out.spatial, float("nan")))
        return out


poison = PoisonTeacher("tb", 24, 3, 12, 1000.0, 12)
poison_target = [Target("tb", 24, 3, poison)]
# checkpoints every 2 steps, no held-out loss inside the run, so the teacher is called once per
# step after the PHI-S fit. The fit is done first, in a run of one step
stop_cfg = TrainConfig(**{**T.asdict(small), "checkpoint_every": 2, "held_out_every": 100})
dir_nan = TMP / "run_nan"
train_student(stop_cfg, data, poison_target, dir_nan, device="cpu", stop_after=1)
assert (dir_nan / "phis_tb.pt").is_file() and not (dir_nan / "last.pt").exists()
poison.calls, poison.poison_from = 0, 4                           # the 4th step is poisoned
try:
    train_student(stop_cfg, data, poison_target, dir_nan, device="cpu")
except RuntimeError as err:
    text = str(err)
else:
    raise SystemExit("a NaN loss did not stop the run")
assert "step 4: the loss is not finite" in text and "loss nan" in text, text
assert "last.pt at step 2" in text and str(dir_nan / "train_log.csv") in text, text
assert load_checkpoint(dir_nan / "last.pt")["step"] == 2 and not (dir_nan / "final.pt").exists()
with open(dir_nan / "train_log.csv", newline="", encoding="utf-8") as f:
    rows_nan = list(csv.DictReader(f))
assert [int(r["step"]) for r in rows_nan] == [1, 2, 3, 4]         # the bad step is in the log
assert all(math.isfinite(float(r["loss"])) for r in rows_nan[:3])
assert math.isnan(float(rows_nan[3]["loss"])) and math.isnan(float(rows_nan[3]["tb/spatial"]))
# the message before any checkpoint exists (a fresh folder, the PHI-S fit takes 4 calls first)
poison.calls, poison.poison_from = 0, 5
try:
    train_student(stop_cfg, data, poison_target, TMP / "run_nan_first", device="cpu")
except RuntimeError as err:
    assert "step 1:" in str(err) and "no checkpoint was written in this run" in str(err), err
else:
    raise SystemExit("a NaN loss at the first step did not stop the run")
assert not (TMP / "run_nan_first" / "last.pt").exists()
# the resume after the stop goes on from step 2 with a healthy teacher and gives the run in one go
poison.calls, poison.poison_from = 0, 10 ** 9
res_after = train_student(stop_cfg, data, poison_target, dir_nan, device="cpu")
assert res_after["step"] == 12 and (dir_nan / "final.pt").is_file()
res_whole = train_student(stop_cfg, data, poison_target, TMP / "run_nan_whole", device="cpu")
for module in ("student", "heads"):
    for k, v in res_after[module].state_dict().items():
        assert torch.equal(v, res_whole[module].state_dict()[k]), (module, k)
with open(dir_nan / "train_log.csv", newline="", encoding="utf-8") as f:
    rows_after = list(csv.DictReader(f))
assert [int(r["step"]) for r in rows_after] == list(range(1, 13))
assert all(math.isfinite(float(r["loss"])) for r in rows_after), "the NaN row was not dropped"
# stop_on_nonfinite=False lets the run go on, as before (a fresh folder, 4 fit calls come first)
poison.calls, poison.poison_from = 0, 8
res_on = train_student(stop_cfg, data, poison_target, TMP / "run_nan_on", device="cpu",
                       stop_on_nonfinite=False)
assert res_on["step"] == 12 and (TMP / "run_nan_on" / "final.pt").is_file()
with open(TMP / "run_nan_on" / "train_log.csv", newline="", encoding="utf-8") as f:
    rows_on = list(csv.DictReader(f))
assert len(rows_on) == 12 and math.isnan(float(rows_on[3]["loss"]))
print("a loss that is not finite stops the run, the last checkpoint stays, the resume is clean")

# ---- DINOv3 SAT reading aerial RGB as 8 bit / 255 through its own reader (teachers/base.py) ---
dino = build_teacher("dinov3-l-sat", tiny=True, device="cpu", dtype=torch.float32)
assert dino.reads_aerial and torch.equal(torch.tensor(dino.aerial_mean), dino.mean.flatten())
read = dino.aerial
torch.manual_seed(1)
aer = torch.randint(0, 256, (2, 4, 256, 256)).float()
out = read(aer)
assert out.summary.shape == (2, dino.dim) and out.spatial.shape == (2, 16, 16, dino.dim)
assert out.summary.dtype == out.spatial.dtype == torch.float32
with torch.inference_mode():
    want = dino._forward((aer[:, :3] / 255.0 - dino.mean) / dino.std)
assert torch.equal(out.summary, want.summary) and torch.equal(out.spatial, want.spatial)
nir = aer.clone()
nir[:, 3] = 0.0
assert torch.equal(read(nir).spatial, out.spatial)                     # NIR is not read
assert not torch.equal(read(aer.flip(-1)).spatial, out.spatial)
small_px = read(torch.randint(0, 256, (1, 4, 64, 64)).float())          # resized to 256
assert small_px.spatial.shape == (1, 16, 16, dino.dim)
assert torch.equal(read(aer[:, :3]).spatial, out.spatial)                # RGB alone is fine too
raises(ValueError, read, aer[:, :2])
raises(ValueError, read, aer.to(torch.uint8))
auto = as_targets([dino])[0]
assert auto.name == "dinov3-l-sat" and auto.aerial == dino.aerial and auto.s2 is dino
no_aerial = as_targets([build_teacher("terrafm-b", tiny=True, device="cpu", dtype=torch.float32)])[0]
assert no_aerial.aerial is None, "TerraFM-B gives no aerial target"
assert auto.dim == dino.dim and auto.grid == 16
mixed = Batcher(s1, s2, aerial, TrainConfig(crop=CROP, batch=4, n_aerial=1), "cpu").fixed(
    range(3), range(1))
dino_out, dino_rows = target_of(auto, mixed)
assert dino_rows == slice(0, 4)
assert dino_out.summary.shape == (4, dino.dim) and dino_out.spatial.shape == (4, 16, 16, dino.dim)
assert torch.equal(dino_out.spatial[3], read(mixed.aerial).spatial[0])
assert torch.equal(dino_out.spatial[:3], dino(mixed.s2).spatial)
print("aerial reader of the teacher, through the loop, ok")

# ---- make_optimizer. Fused AdamW on CUDA only, the groups and their decay rules unchanged ------
recorded = []
_AdamW = torch.optim.AdamW


class _AdamWRecorder(_AdamW):
    """Records the keyword arguments. The parameters here are on the CPU, so fused is dropped."""

    def __init__(self, params, **kw):
        recorded.append(dict(kw))
        kw.pop("fused", None)
        super().__init__(params, **kw)


torch.optim.AdamW = _AdamWRecorder
try:
    opt_cpu = T.make_optimizer(param_groups(st0, hd0, 0.05), small, "cpu")
    opt_gpu = T.make_optimizer(param_groups(st0, hd0, 0.05), small, torch.device("cuda"))
finally:
    torch.optim.AdamW = _AdamW
assert len(recorded) == 2 and "fused" not in recorded[0] and recorded[1]["fused"] is True, recorded
for kw in recorded:
    assert (kw["lr"], kw["betas"], kw["weight_decay"]) == (small.lr, tuple(small.betas), small.weight_decay)
for opt in (opt_cpu, opt_gpu):
    assert [g["weight_decay"] for g in opt.param_groups] == [0.05, 0.0]
    assert sum(len(g["params"]) for g in opt.param_groups) == n_all
assert isinstance(T.make_optimizer(param_groups(st0, hd0, 0.05), small, "cpu"), _AdamW)
assert not load_checkpoint(dir_a / "last.pt")["optimizer"]["param_groups"][0].get("fused")
print("make_optimizer: plain AdamW on the CPU, fused=True on CUDA, the groups unchanged")

# ---- decide_joint. The memory rule with a patched reader, under and over 36 GB ----------------
real_free = T.free_gpu_gb
assert T.JOINT_MIN_GB == 36
assert real_free("cpu") is None and real_free(torch.device("cpu")) is None
assert real_free("cuda") is None                                  # no CUDA on this PC
for gb, want in ((30.0, False), (35.9, False), (36.0, True), (42.0, True), (84.0, True)):
    T.free_gpu_gb = lambda device=None, gb=gb: gb
    try:
        use, why = T.decide_joint("auto", torch.device("cuda"))
    finally:
        T.free_gpu_gb = real_free
    assert use is want and f"{gb:.1f} GB" in why and "36" in why, (gb, use, why)
assert T.decide_joint("auto", torch.device("cpu")) == (False, "no CUDA device")
assert T.decide_joint("auto", "cpu") == (False, "no CUDA device")
assert T.decide_joint(True, "cpu") == (True, "joint=True given")
assert T.decide_joint(False, torch.device("cuda")) == (False, "joint=False given")
raises(ValueError, T.decide_joint, "yes", "cpu", says="auto")
raises(ValueError, T.decide_joint, 1, "cpu", says="auto")
assert T.free_gpu_gb is real_free
print("decide_joint: joint at 36 GB and above, sequential under it, off CUDA and when forced")

# ---- train_students. One loop for both, bit for bit the two separate runs ---------------------
specs = [("student_M", ["ta", "tb"]), ("student_S", ["ta"])]
res_single_s = train_student(small, data, [targets[0]], TMP / "single_S", device="cpu")   # S alone
root_j = TMP / "joint"
res_j, said_j = printed(train_students, specs, small, data, targets, root_j, device="cpu", joint=True)
assert list(res_j) == ["student_M", "student_S"]
assert "students: student_M, student_S in one loop, the teachers run once per step, joint=True given" in said_j
assert said_j.count("phis ta: fitted on") == 1 and said_j.count("phis tb: fitted on") == 1   # once
assert f"phis ta: copied {root_j / 'student_M' / 'phis_ta.pt'} to {root_j / 'student_S' / 'phis_ta.pt'}" in said_j
assert "student_M step 12: held-out loss" in said_j and "student_S step 12: held-out loss" in said_j
assert said_j.count("step 12 of 12  loss student_M ") == 1 and "  student_S " in said_j      # one line for both
for name, ref, folder in (("student_M", res_a, dir_a), ("student_S", res_single_s, TMP / "single_S")):
    r = res_j[name]
    assert r["step"] == 12 and r["mode"] == "joint" and r["out_dir"] == root_j / name
    assert r["final"] == root_j / name / "final.pt" and r["last"] == root_j / name / "last.pt"
    assert same_state(r["student"], ref["student"]) and same_state(r["heads"], ref["heads"]), name
    assert r["held_out"] == ref["held_out"], name                      # the same floats
    assert no_timing(log_of(root_j / name)) == no_timing(log_of(folder)), name
    assert (root_j / name / "held_out.csv").read_bytes() == (folder / "held_out.csv").read_bytes()
    for file in ("last.pt", "final.pt"):                              # the same checkpoint content
        ck, ck_ref = load_checkpoint(root_j / name / file), load_checkpoint(folder / file)
        assert ck["step"] == ck_ref["step"] == 12 and ck["teacher_dims"] == ck_ref["teacher_dims"]
        assert torch.equal(ck["rng"]["torch"], ck_ref["rng"]["torch"])
        assert all(torch.equal(v, ck_ref["student"][k]) for k, v in ck["student"].items())
    assert load_checkpoint(root_j / name / "last_prev.pt")["step"] == 10
    assert not list((root_j / name).glob("*.tmp"))
    loaded, loaded_stats, loaded_cfg = load_student(root_j / name / "final.pt")
    assert same_state(loaded, ref["student"]) and loaded_stats == stats and loaded_cfg == small
assert {p.name for p in (root_j / "student_M").iterdir()} == {"phis_ta.pt", "phis_tb.pt", "last.pt", "last_prev.pt",
                                                             "final.pt", "train_log.csv", "held_out.csv"}
assert {p.name for p in (root_j / "student_S").iterdir()} == {"phis_ta.pt", "last.pt", "last_prev.pt", "final.pt",
                                                             "train_log.csv", "held_out.csv"}
# PHI-S in both folders, equal, and the fit of the single run
assert (root_j / "student_M" / "phis_ta.pt").read_bytes() == (root_j / "student_S" / "phis_ta.pt").read_bytes()
for name in ("phis_ta.pt", "phis_tb.pt"):
    assert torch.equal(PhiS.load(root_j / "student_M" / name).rotation, PhiS.load(dir_a / name).rotation)
assert res_j["student_S"]["phis"].keys() == {"ta"} and res_j["student_M"]["phis"].keys() == {"ta", "tb"}
# the new columns, and the shared ones equal in both logs of one step
log_jm, log_js = log_of(root_j / "student_M"), log_of(root_j / "student_S")
assert list(log_js[0]) == ["step", "lr", "loss", "ta/summary", "ta/spatial"] + list(T.LOG_END)
for r in log_jm + log_js:
    for col in ("teacher_s", "student_s", "data_s", "step_s"):
        assert math.isfinite(float(r[col])) and float(r[col]) >= 0, (col, r)
    assert float(r["student_s"]) > 0
for col in ("step_s", "data_s", "teacher_s", "gpu_gb", "img_s"):
    assert [r[col] for r in log_jm] == [r[col] for r in log_js], col
for m, s in zip(log_jm, log_js):                                      # both updates fit in the step
    assert float(m["data_s"]) + float(m["teacher_s"]) + float(m["student_s"]) + float(s["student_s"]) \
        <= float(m["step_s"]) + 2e-4
assert not prefetch_threads()
# joint=False gives the same, through train_student; "auto" on the CPU takes that path too
res_f, said_f = printed(train_students, specs, small, data, targets, TMP / "joint_false", device="cpu", joint=False)
assert "one after the other (train_student), joint=False given" in said_f
assert "student_M: student 1 of 2" in said_f and "student_S: student 2 of 2" in said_f
assert said_f.count("phis ta: fitted on") == 1 and said_f.count("phis ta: loaded") == 2
res_auto, said_auto = printed(train_students, specs, small, data, targets, TMP / "joint_auto", device="cpu")
assert "one after the other (train_student), no CUDA device" in said_auto
for res, root in ((res_f, TMP / "joint_false"), (res_auto, TMP / "joint_auto")):
    for name, ref in (("student_M", res_a), ("student_S", res_single_s)):
        assert res[name]["mode"] == "sequential" and res[name]["step"] == 12
        assert same_state(res[name]["student"], ref["student"]) and same_state(res[name]["heads"], ref["heads"])
        assert no_timing(log_of(root / name)) == no_timing(log_of(root_j / name))
        assert res[name]["held_out"] == ref["held_out"]
    assert (root / "student_M" / "phis_ta.pt").read_bytes() == (root / "student_S" / "phis_ta.pt").read_bytes()
# the memory rule inside train_students, with the patched reader (the loop then runs on the CPU)
for gb, mode in ((30.0, "sequential"), (70.0, "joint")):
    T.free_gpu_gb = lambda device=None, gb=gb: gb
    try:
        res_gb, said_gb = printed(train_students, specs, small, data, targets, TMP / f"joint_{int(gb)}gb",
                                  device="cpu", stop_after=2)
    finally:
        T.free_gpu_gb = real_free
    assert res_gb["student_M"]["mode"] == mode and f"{gb:.1f} GB of GPU memory free" in said_gb, (gb, said_gb)
assert "at least 36 needed" in said_gb and "under 36" not in said_gb
# a cut at an arbitrary step and the resume, both at the same step
root_c = TMP / "joint_cut"
train_students(specs, small, data, targets, root_c, device="cpu", joint=True, stop_after=7)
for name in ("student_M", "student_S"):
    assert load_checkpoint(root_c / name / "last.pt")["step"] == 5 and not (root_c / name / "final.pt").exists()
    assert len(log_of(root_c / name)) == 7
res_c, said_c = printed(train_students, specs, small, data, targets, root_c, device="cpu", joint=True)
for name, ref in (("student_M", res_a), ("student_S", res_single_s)):
    assert f"resumed from {root_c / name / 'last.pt'} at step 5" in said_c
    assert same_state(res_c[name]["student"], ref["student"]) and same_state(res_c[name]["heads"], ref["heads"])
    assert no_timing(log_of(root_c / name)) == no_timing(log_of(root_j / name))
    with open(root_c / name / "held_out.csv", newline="", encoding="utf-8") as f:
        assert [int(r["step"]) for r in csv.DictReader(f)] == [5, 10, 12]
assert "computed again" not in said_c and "fitted now" not in said_c
# the two at different steps: cut at 10, S's last.pt deleted, so S lags at 5 and both go on from 5
root_l = TMP / "joint_lag"
train_students(specs, small, data, targets, root_l, device="cpu", joint=True, stop_after=10)
(root_l / "student_S" / "last.pt").unlink()
res_l, said_l = printed(train_students, specs, small, data, targets, root_l, device="cpu", joint=True)
assert f"resumed from {root_l / 'student_M' / 'last_prev.pt'} at step 5 (its newest checkpoint is at step 10, " \
       "the steps from 5 on are computed again so that all students take the same steps)" in said_l, said_l
assert f"resumed from {root_l / 'student_S' / 'last_prev.pt'} at step 5 (the previous checkpoint, last.pt is missing)" in said_l
assert "student_M: last.pt at step 10 deleted, last_prev.pt at step 5 is now last.pt" in said_l, said_l
for name, ref in (("student_M", res_a), ("student_S", res_single_s)):
    assert same_state(res_l[name]["student"], ref["student"]) and same_state(res_l[name]["heads"], ref["heads"])
    assert no_timing(log_of(root_l / name)) == no_timing(log_of(root_j / name))
    assert (load_checkpoint(root_l / name / "last.pt")["step"], load_checkpoint(root_l / name / "last_prev.pt")["step"]) == (12, 10)
# the student ahead went back to last_prev.pt: its newer last.pt is gone and last_prev.pt became
# last.pt, so the next checkpoint rotates two different steps and a second crash in the same
# window (S's last.pt lost again) still finds the common step instead of starting over
root_d = TMP / "joint_double"
train_students(specs, small, data, targets, root_d, device="cpu", joint=True, stop_after=10)   # both {10, 5}
(root_d / "student_S" / "last.pt").unlink()                                                     # S {5}
_, said_d = printed(train_students, specs, small, data, targets, root_d, device="cpu", joint=True, stop_after=10)
assert "student_M: last.pt at step 10 deleted, last_prev.pt at step 5 is now last.pt" in said_d, said_d
for name in ("student_M", "student_S"):
    assert (load_checkpoint(root_d / name / "last.pt")["step"], load_checkpoint(root_d / name / "last_prev.pt")["step"]) == (10, 5)
(root_d / "student_S" / "last.pt").unlink()
res_d, said_d2 = printed(train_students, specs, small, data, targets, root_d, device="cpu", joint=True)
assert f"resumed from {root_d / 'student_M' / 'last_prev.pt'} at step 5" in said_d2, said_d2
assert f"resumed from {root_d / 'student_S' / 'last_prev.pt'} at step 5" in said_d2
for name, ref in (("student_M", res_a), ("student_S", res_single_s)):
    assert same_state(res_d[name]["student"], ref["student"]) and same_state(res_d[name]["heads"], ref["heads"])
    assert no_timing(log_of(root_d / name)) == no_timing(log_of(root_j / name))
# S has no checkpoint at all (a runtime that died between the two writes of the first
# checkpoint, or a folder cleared by hand): the run stops with the steps per student, nothing
# is touched; joint=False lets M go on from 10 and S from 0, the same run in one go
root_n = TMP / "joint_none"
train_students(specs, small, data, targets, root_n, device="cpu", joint=True, stop_after=10)
for file in ("last.pt", "last_prev.pt", "phis_ta.pt"):
    (root_n / "student_S" / file).unlink()
before_n = {p.name: p.read_bytes() for p in (root_n / "student_M").iterdir()}
try:
    said_n = printed(train_students, specs, small, data, targets, root_n, device="cpu", joint=True)
except RuntimeError as err:
    text_n = str(err)
else:
    raise SystemExit("the joint loop did not stop without a common step")
assert "no step every student has a checkpoint for (student_M at the steps [5, 10], student_S without a " \
       "checkpoint), nothing is overwritten" in text_n, text_n
assert "joint=False" in text_n and "delete last.pt and last_prev.pt" in text_n, text_n
assert {p.name: p.read_bytes() for p in (root_n / "student_M").iterdir()} == before_n
assert (root_n / "student_S" / "phis_ta.pt").is_file() and not prefetch_threads()      # shared before the stop
res_n, said_n = printed(train_students, specs, small, data, targets, root_n, device="cpu", joint=False)
assert f"resumed from {root_n / 'student_M' / 'last.pt'} at step 10" in said_n and "starts over" not in said_n
for name, ref in (("student_M", res_a), ("student_S", res_single_s)):
    assert same_state(res_n[name]["student"], ref["student"]) and no_timing(log_of(root_n / name)) == no_timing(log_of(root_j / name))
# M finished (last.pt 12, final.pt) while S sits at 5: the same stop, final.pt is kept, and
# joint=False finishes S alone
root_m = TMP / "joint_m_done"
train_students(specs, small, data, targets, root_m, device="cpu", joint=True, stop_after=7)    # both at 5
shutil.copytree(root_m / "student_S", TMP / "joint_s_at_5")
train_students(specs, small, data, targets, root_m, device="cpu", joint=True)                  # both done
shutil.rmtree(root_m / "student_S")
shutil.copytree(TMP / "joint_s_at_5", root_m / "student_S")
before_m = {p.name: p.read_bytes() for p in (root_m / "student_M").iterdir()}
try:
    train_students(specs, small, data, targets, root_m, device="cpu", joint=True)
except RuntimeError as err:
    assert "student_M at the steps [10, 12], student_S at the steps [5]" in str(err), err
else:
    raise SystemExit("a finished student was trained again from step 0")
assert {p.name: p.read_bytes() for p in (root_m / "student_M").iterdir()} == before_m
res_m = train_students(specs, small, data, targets, root_m, device="cpu", joint=False)
for name, ref in (("student_M", res_a), ("student_S", res_single_s)):
    assert same_state(res_m[name]["student"], ref["student"]) and res_m[name]["step"] == 12
# a cut last.pt of M: the previous one is used with the printed line, the cut file is moved aside
root_b = TMP / "joint_broken"
train_students(specs, small, data, targets, root_b, device="cpu", joint=True, stop_after=10)
whole = (root_b / "student_M" / "last.pt").read_bytes()
(root_b / "student_M" / "last.pt").write_bytes(whole[:len(whole) // 2])
res_b, said_b = printed(train_students, specs, small, data, targets, root_b, device="cpu", joint=True)
assert f"{root_b / 'student_M' / 'last.pt'} cannot be loaded" in said_b and "the previous checkpoint is tried" in said_b
assert "last.pt moved to last.pt.broken" in said_b and (root_b / "student_M" / "last.pt.broken").is_file()
assert f"resumed from {root_b / 'student_M' / 'last_prev.pt'} at step 5" in said_b
assert f"resumed from {root_b / 'student_S' / 'last_prev.pt'} at step 5 (its newest checkpoint is at step 10" in said_b
for name, ref in (("student_M", res_a), ("student_S", res_single_s)):
    assert same_state(res_b[name]["student"], ref["student"])
    assert (load_checkpoint(root_b / name / "last.pt")["step"], load_checkpoint(root_b / name / "last_prev.pt")["step"]) == (12, 10)
    assert not list((root_b / name).glob("*.tmp"))
# both checkpoints of M cut: the run stops, nothing is overwritten
root_x = TMP / "joint_nothing_loads"
train_students(specs, small, data, targets, root_x, device="cpu", joint=True, stop_after=10)
for file in ("last.pt", "last_prev.pt"):
    whole = (root_x / "student_M" / file).read_bytes()
    (root_x / "student_M" / file).write_bytes(whole[:len(whole) // 2])
before_x = {p.name: p.read_bytes() for p in (root_x / "student_M").iterdir()}
raises(RuntimeError, train_students, specs, small, data, targets, root_x, device="cpu", joint=True,
       says="can be loaded")
assert {p.name: p.read_bytes() for p in (root_x / "student_M").iterdir()} == before_x
raises(ValueError, train_students, specs, TrainConfig(**{**T.asdict(small), "lr": 1e-3}), data, targets, root_l,
       device="cpu", joint=True, says="another config")
# without the prefetch thread the same numbers, no thread left either way
res_np = train_students(specs, small, data, targets, TMP / "joint_noprefetch", device="cpu", joint=True, prefetch=False)
assert same_state(res_np["student_M"]["student"], res_a["student"]) and not prefetch_threads()
assert no_timing(log_of(TMP / "joint_noprefetch" / "student_S")) == no_timing(log_of(root_j / "student_S"))
# bad specs
raises(ValueError, train_students, [("student_M", ["zz"])], small, data, targets, TMP / "joint_bad", device="cpu",
       says="unknown")
raises(ValueError, train_students, [("a", ["ta"]), ("a", ["tb"])], small, data, targets, TMP / "joint_bad",
       device="cpu", says="unique")
raises(ValueError, train_students, [("a", ["ta", "ta"])], small, data, targets, TMP / "joint_bad", device="cpu",
       says="twice")
raises(ValueError, train_students, [], small, data, targets, TMP / "joint_bad", device="cpu", says="no student")
raises(ValueError, train_students, [("a", [])], small, data, targets, TMP / "joint_bad", device="cpu")
raises(ValueError, train_students, [("..", ["ta"])], small, data, targets, TMP / "joint_bad", device="cpu",
       says="folder name")
raises(ValueError, train_students, specs, small, data, targets, TMP / "joint_bad", device="cpu", joint="maybe",
       says="auto")
assert not (TMP / "joint_bad").exists()
print("train_students: bit for bit the two single runs, resume at one and at two steps, fallback, sequential ok")

# ---- the non-finite stop and the fallback checkpoint in the joint loop ------------------------
poison_j = PoisonTeacher("tb", 24, 3, 12, 1000.0, 12)
targets_j = [targets[0], Target("tb", 24, 3, poison_j)]
root_p = TMP / "joint_nan"
train_students(specs, stop_cfg, data, targets_j, root_p, device="cpu", joint=True, stop_after=1)
assert (root_p / "student_M" / "phis_tb.pt").is_file() and not (root_p / "student_M" / "last.pt").exists()
poison_j.calls, poison_j.poison_from = 0, 4                           # the 4th step is poisoned
try:
    train_students(specs, stop_cfg, data, targets_j, root_p, device="cpu", joint=True)
except RuntimeError as err:
    text = str(err)
else:
    raise SystemExit("a NaN loss did not stop the joint loop")
assert "student_M step 4: the loss is not finite" in text and "loss nan" in text, text
assert "last.pt at step 2" in text and str(root_p / "student_M" / "train_log.csv") in text, text
assert not prefetch_threads()
for name in ("student_M", "student_S"):
    assert load_checkpoint(root_p / name / "last.pt")["step"] == 2 and not (root_p / name / "final.pt").exists()
    assert [int(r["step"]) for r in log_of(root_p / name)] == [1, 2, 3, 4]
assert math.isnan(float(log_of(root_p / "student_M")[3]["loss"]))
assert math.isfinite(float(log_of(root_p / "student_S")[3]["loss"]))   # S has no tb, its row is fine
poison_j.calls, poison_j.poison_from = 0, 10 ** 9
res_pj = train_students(specs, stop_cfg, data, targets_j, root_p, device="cpu", joint=True)
res_pw = train_students(specs, stop_cfg, data, targets_j, TMP / "joint_nan_whole", device="cpu", joint=True)
for name in ("student_M", "student_S"):
    assert res_pj[name]["step"] == 12 and (root_p / name / "final.pt").is_file()
    assert same_state(res_pj[name]["student"], res_pw[name]["student"]) and same_state(res_pj[name]["heads"], res_pw[name]["heads"])
    assert [int(r["step"]) for r in log_of(root_p / name)] == list(range(1, 13))
    assert all(math.isfinite(float(r["loss"])) for r in log_of(root_p / name))
# stop_on_nonfinite=False lets the joint loop go on
poison_j.calls, poison_j.poison_from = 0, 8
res_on_j = train_students(specs, stop_cfg, data, targets_j, TMP / "joint_nan_on", device="cpu", joint=True,
                          stop_on_nonfinite=False)
assert res_on_j["student_M"]["step"] == 12 and math.isnan(float(log_of(TMP / "joint_nan_on" / "student_M")[7]["loss"]))
assert not prefetch_threads()
print("joint loop: a loss that is not finite stops it naming the student, the resume is clean")

# ==== part four. A radar teacher covers the rows of the modes s1 and both ====
assert T.S1_MODES == ("s1", "both")
# the rows are found on the host, in the host half and in the batch, and interleave
for step in range(5):
    hb_r = bat.host(step)
    b = bat.to_device(hb_r)
    want = [i for i, m in enumerate(b.modes) if m in ("s1", "both")]
    assert b.s1_rows.dtype == torch.int64 and b.s1_rows.tolist() == want == hb_r.s1_rows.tolist()
    assert want != list(range(len(want))) and 40 < len(want) < 90        # about 2/3 of 96, not a prefix
fb_r = bat.fixed(range(5), range(3))                                     # modes s1, s2, both, s1, s2
assert fb_r.s1_rows.tolist() == [0, 2, 3] and bat.fixed(aerial=range(2)).s1_rows.numel() == 0
# a toy radar teacher reads (VV, VH) in dB only, its assert stops a Sentinel-2 input
rd = ToyTeacher("rd", 12, 2, 2, 10.0, 13)                              # width 4k, as PHI-S needs
rd_t = Target("rd", 12, 2, None, radar=rd)
raises(ValueError, as_targets, [Target("x", 4, 2, None)], says="one of the two")
raises(ValueError, as_targets, [Target("x", 4, 2, tb, radar=rd)], says="one of the two")
b = bat.batch(0)
out_rd, rows_rd = target_of(rd_t, b)
assert torch.equal(rows_rd, b.s1_rows) and out_rd.summary.shape == (len(rows_rd), 12)
assert out_rd.spatial.shape == (len(rows_rd), 2, 2, 12)
ref_rd = rd(b.s1[rows_rd])                                               # the crop the student sees
assert torch.equal(out_rd.summary, ref_rd.summary) and torch.equal(out_rd.spatial, ref_rd.spatial)
assert target_of(rd_t, bat.fixed(aerial=range(2))) is None               # no target for aerial rows
hand = T.Batch(b.modes, b.s1, b.s2, b.aerial, b.index)                  # built by hand, no s1_rows
raises(ValueError, target_of, rd_t, hand, says="no s1_rows")             # stops instead of skipping
assert target_of(targets[0], hand)[1] == slice(0, 128)                   # a Sentinel-2 teacher needs none
three = T.targets_of([targets[0], rd_t, targets[1]], b)
assert list(three) == ["ta", "rd", "tb"] and three["ta"][1] == slice(0, 128) and three["tb"][1] == slice(0, 96)
# PHI-S on the first n COVERED samples
k_fit, got = 0, 0
while got < 20:
    got += len(sb.batch(k_fit).s1_rows)
    k_fit += 1
rd.calls = 0
ph_rd = fit_phis(lambda x: target_of(rd_t, x), sb, n=20, name="rd")
assert rd.calls == k_fit and ph_rd.n_fit == 20 * 4
first = torch.cat([rd(sb.batch(s).s1[sb.batch(s).s1_rows]).spatial for s in range(k_fit)])[:20]
ref_ph = PhiS.fit(first.reshape(-1, 12))
assert torch.equal(ph_rd.mean, ref_ph.mean) and torch.equal(ph_rd.rotation, ref_ph.rotation)
# the loss by hand on a batch whose covered rows interleave - the head outputs at those rows
step_r = next(s for s in range(50) if len(sb.batch(s).s1_rows) >= 2
              and sb.batch(s).s1_rows.tolist() != list(range(len(sb.batch(s).s1_rows))))
bb_r = sb.batch(step_r)
rows_r = bb_r.s1_rows
torch.manual_seed(0)
st_r, hd_r = Student(**small.student), T.make_heads([targets[0], rd_t], 32)
phis_r = {"ta": ph, "rd": ph_rd}
loss_r, parts_r = batch_loss(st_r, hd_r, [targets[0], rd_t], bb_r, norm, phis_r, auto0)
assert parts_r["rd/rows"] == len(rows_r) and parts_r["ta/rows"] == 8
with torch.no_grad():
    o_r = hd_r(*st_r.forward_mixed(bb_r.student_inputs(norm)))
t_rd = rd(bb_r.s1[rows_r])
want_r, want_parts_r = distil_loss({"ta": o_r["ta"], "rd": (o_r["rd"][0][rows_r], o_r["rd"][1][rows_r])},
                                   {"ta": target_of(targets[0], bb_r)[0], "rd": t_rd}, standardise=phis_r)
assert abs(float(loss_r.detach()) - float(want_r.detach())) < 1e-6
assert all(abs(want_parts_r[k] - parts_r[k]) < 1e-6 for k in want_parts_r), (want_parts_r, parts_r)
# both parts are means over the covered rows only, row by row by hand (grid 2 on both sides)
per_row = 1.0 - F.cosine_similarity(o_r["rd"][0][rows_r], t_rd.summary, dim=-1)
assert abs(parts_r["rd/summary"] - float(per_row.mean())) < 1e-6
per_row_spa = ((o_r["rd"][1][rows_r] - ph_rd(t_rd.spatial)) ** 2).mean((1, 2, 3))
assert abs(parts_r["rd/spatial"] - float(per_row_spa.mean())) < 1e-5
# the prefix cut of part three would read other rows and give another number
_, prefix = distil_loss({"rd": (o_r["rd"][0][:len(rows_r)], o_r["rd"][1][:len(rows_r)])}, {"rd": t_rd},
                        standardise={"rd": ph_rd})
assert abs(prefix["rd/summary"] - parts_r["rd/summary"]) > 1e-4
# a batch without an s1 or both row - the radar teacher is not called, adds nothing, its parts are nan
hb_none = sb._build_host(None, np.arange(6), np.arange(2), ["s2"] * 6, np.full((6, 2), (SIDE - CROP) // 2),
                         torch.zeros(6, 2, dtype=torch.bool), torch.zeros(2, 2, dtype=torch.bool))
b_none = sb.to_device(hb_none)
assert b_none.s1_rows.numel() == 0 and target_of(rd_t, b_none) is None
rd.calls = 0
loss_n, parts_n = batch_loss(st_r, hd_r, [targets[0], rd_t], b_none, norm, phis_r, auto0)
assert rd.calls == 0 and parts_n["rd/rows"] == 0
assert math.isnan(parts_n["rd/summary"]) and math.isnan(parts_n["rd/spatial"])
loss_ta, _ = batch_loss(st_r, hd_r, [targets[0]], b_none, norm, phis_r, auto0)
assert float(loss_n.detach()) == float(loss_ta.detach())
loss_n.backward()                                                        # the radar heads get no gradient
assert all(p.grad is None for h in (hd_r.summary["rd"], hd_r.spatial["rd"]) for p in h.parameters())
assert hd_r.summary["ta"][0].weight.grad is not None
raises(ValueError, batch_loss, st_r, hd_r, [rd_t], b_none, norm, phis_r, auto0, says="no teacher has a target")
# the held-out loss of the radar teacher is the mean over the covered held-out rows, by hand
held_r = Batcher(h_s1, h_s2, h_aerial, small, "cpu")
ho_r = held_out_loss(st_r, hd_r, [targets[0], rd_t], held_r, norm, phis_r, small, auto0)
assert list(ho_r) == ["ta/summary", "ta/spatial", "rd/summary", "rd/spatial", "loss"] and st_r.training
num, den = 0.0, 0
with torch.no_grad():
    for a in range(0, 20, 6):                                            # the Major-TOM chunks
        hbat = held_r.fixed(range(a, min(a + 6, 20)))
        o_h = hd_r(*st_r.forward_mixed(hbat.student_inputs(norm)))
        d = 1.0 - F.cosine_similarity(o_h["rd"][0][hbat.s1_rows], rd(hbat.s1[hbat.s1_rows]).summary, dim=-1)
        num, den = num + float(d.sum()), den + len(hbat.s1_rows)
assert den == 13 and abs(ho_r["rd/summary"] - num / den) < 1e-6         # 4 + 4 + 4 + 1 covered rows
print("radar teacher: interleaved rows, gather, means over the covered rows, nan when none, PHI-S, held-out")

# the real wrapper, TerraFM-B radar (tiny), through as_targets and the loop
radar_teacher = build_teacher("terrafm-b-radar", tiny=True, device="cpu", dtype=torch.float32)
auto_rd = as_targets([radar_teacher])[0]
assert auto_rd.name == "terrafm-b-radar" and auto_rd.s2 is None and auto_rd.aerial is None
assert auto_rd.radar == radar_teacher.radar and (auto_rd.dim, auto_rd.grid) == (radar_teacher.dim, 2)
out_tr, rows_tr = target_of(auto_rd, bb_r)
assert torch.equal(rows_tr, rows_r)
assert torch.equal(out_tr.spatial, radar_teacher.radar(bb_r.s1[rows_r]).spatial)
raises(TypeError, radar_teacher, bb_r.s2, says="does not read Sentinel-2")    # never called so by the loop
short = TrainConfig(**{**T.asdict(small), "steps": 3, "checkpoint_every": 3, "held_out_every": 3})
res_tr = train_student(short, data, [targets[0], radar_teacher], TMP / "run_radar", device="cpu")
log_tr = log_of(TMP / "run_radar")
assert list(log_tr[0])[3:7] == ["ta/summary", "ta/spatial", "terrafm-b-radar/summary", "terrafm-b-radar/spatial"]
for r in log_tr:
    covered = len(sb.batch(int(r["step"]) - 1).s1_rows) > 0
    assert math.isfinite(float(r["terrafm-b-radar/spatial"])) == covered and math.isfinite(float(r["loss"]))
assert (TMP / "run_radar" / "phis_terrafm-b-radar.pt").is_file() and res_tr["step"] == 3
print("radar teacher: the tiny TerraFM-B radar wrapper through as_targets and train_student")

# ==== part four. Summary-only teachers ==================================
sg, sg_aer = ToyTeacher("sg", 12, 2, 12, 1000.0, 20), ToyTeacher("sg", 12, 2, 4, 255.0, 21)
sn, sn_aer = ToyTeacher("sn", 12, 2, 12, 1000.0, 22), ToyTeacher("sn", 12, 2, 4, 255.0, 23)
pool = [Target("sg", 12, 2, sg, sg_aer), Target("sn", 12, 2, sn, sn_aer), targets[0], rd_t]
marked = as_targets(pool, summary_only=("sg", "sn"))
assert [t.summary_only for t in marked] == [True, True, False, False]
assert not any(t.summary_only for t in pool)                             # the given Targets stay as they are
assert as_targets(marked)[0].summary_only                                # a marked Target stays marked
raises(ValueError, as_targets, pool, summary_only=("siglip",), says="match no teacher")
assert T.part_columns(marked) == ["sg/summary", "sn/summary", "ta/summary", "ta/spatial", "rd/summary",
                                  "rd/spatial"]
by = {t.name: t for t in marked}
# target_of keeps the summary and drops the spatial output of a summary-only teacher
b_so = sb.batch(0)
(out_sg, rows_sg), full_sg = target_of(by["sg"], b_so), target_of(pool[0], b_so)[0]
assert rows_sg == slice(0, 8) and torch.equal(out_sg.summary, full_sg.summary)
assert out_sg.spatial.shape == (0, 2, 2, 12) and full_sg.spatial.shape == (8, 2, 2, 12)
raises(ValueError, fit_phis, lambda bb: target_of(by["sg"], bb), sb, n=4, says="summary only")
# one student with a summary-only teacher
dir_so = TMP / "run_so"
res_so = train_student(small, data, [pool[0], targets[0]], dir_so, device="cpu", summary_only=("sg",))
hd_so = res_so["heads"]
assert hd_so.summary_only == ("sg",) and "sg" in hd_so.summary and "sg" not in hd_so.spatial
assert not (dir_so / "phis_sg.pt").exists() and (dir_so / "phis_ta.pt").is_file() and res_so["phis"].keys() == {"ta"}
rows_so = log_of(dir_so)
assert list(rows_so[0]) == ["step", "lr", "loss", "sg/summary", "ta/summary", "ta/spatial"] + list(T.LOG_END)
for r in rows_so:
    assert abs(float(r["loss"]) - sum(float(r[k]) for k in ("sg/summary", "ta/summary", "ta/spatial"))) < 1e-5
with open(dir_so / "held_out.csv", newline="", encoding="utf-8") as f:
    held_so = list(csv.DictReader(f))
assert list(held_so[0]) == ["step", "loss", "sg/summary", "ta/summary", "ta/spatial"] and len(held_so) == 3
assert set(res_so["held_out"]) == {"sg/summary", "ta/summary", "ta/spatial", "loss"}
ck_so = load_checkpoint(dir_so / "final.pt")
assert ck_so["summary_only"] == ["sg"] and ck_so["teacher_dims"] == {"sg": 12, "ta": 8}
assert load_checkpoint(dir_so / "last.pt")["summary_only"] == ["sg"]
assert not any(k.startswith("spatial.sg.") for k in ck_so["heads"]) and any(k.startswith("summary.sg.") for k in ck_so["heads"])
assert load_checkpoint(dir_a / "last.pt")["summary_only"] == [] and T.summary_only_of({}) == []
# the trunk loads as before (scoring), and load_student checks the set when it is given
loaded_so = load_student(dir_so / "final.pt")[0]
assert same_state(loaded_so, res_so["student"])
assert same_state(load_student(dir_so / "final.pt", summary_only=["sg"])[0], res_so["student"])
raises(ValueError, load_student, dir_so / "final.pt", summary_only=[], says="summary-only teachers ['sg'], the call has []")
# the pilot's whole tuple is fine for every student, only the names among its teachers count
assert same_state(load_student(dir_so / "final.pt", summary_only=("sg", "sn"))[0], res_so["student"])
raises(ValueError, load_student, dir_so / "final.pt", summary_only=["sn"], says="the call has []")
from sartransfer.models import LAST, encoders as encmod  # noqa: E402
enc_so = encmod.FrozenEncoder(encmod.student_spec("so", dir_so / "final.pt", batch=4), token=None, device="cpu")
tiles = np.random.default_rng(5).normal(0, 1, (3, 2, 64, 64)).astype(np.float32)
f_so = enc_so.encode(tiles, layer=LAST)
with torch.inference_mode():
    _, sp_so = loaded_so(s1=torch.from_numpy(tiles))
assert f_so.shape == (3, 4, 4, 32) and np.array_equal(f_so, sp_so.numpy().astype(np.float16))
# a resume with another summary-only set stops, both ways round, nothing is loaded
shutil.copytree(dir_so, TMP / "run_so_mismatch")
before_mm = snapshot(TMP / "run_so_mismatch")
sg.calls = sg_aer.calls = 0
raises(ValueError, train_student, small, data, [pool[0], targets[0]], TMP / "run_so_mismatch", device="cpu",
       says="summary-only teachers ['sg'], this run has [] - the heads differ")
# refused before the PHI-S fit sg would now need - no teacher call, no file written
assert snapshot(TMP / "run_so_mismatch") == before_mm and sg.calls == sg_aer.calls == 0
dir_other = copy_run("run_so_other")
before_other = snapshot(dir_other)
raises(ValueError, train_student, small, data, targets, dir_other, device="cpu", summary_only=("tb",),
       says="summary-only teachers [], this run has ['tb']")
assert snapshot(dir_other) == before_other
print("summary-only teacher: no spatial head, no PHI-S, summary column only, checkpoint record, load and resume checks")

# ==== part four. Three students F, N, C in one loop, a radar and summary-only teachers ===========
specs3 = [("student_F", ["sg", "ta", "rd"]), ("student_N", ["sn", "ta", "rd"]), ("student_C", ["sg", "ta"])]
sep = {name: train_student(small, data, [by[n] for n in names], TMP / "sep3" / name, device="cpu")
       for name, names in specs3}
counters = (sg, sg_aer, sn, sn_aer, ta, ta_aerial, rd)
for c in counters:
    c.calls = 0
root3 = TMP / "joint3"
res3, said3 = printed(train_students, specs3, small, data, pool, root3, device="cpu", joint=True,
                      summary_only=("sg", "sn"))
# every teacher once per step and once per held-out batch (3 passes, 4 Major-TOM and 5 aerial batches),
# PHI-S once for ta (3 batches) and rd (k_fit batches), none for sg and sn
covered_steps = sum(len(sb.batch(s).s1_rows) > 0 for s in range(12))
assert (sg.calls, sg_aer.calls, sn.calls, sn_aer.calls) == (12 + 3 * 4, 12 + 3 * 5, 12 + 3 * 4, 12 + 3 * 5)
assert (ta.calls, ta_aerial.calls) == (3 + 12 + 3 * 4, 3 + 12 + 3 * 5)
assert rd.calls == k_fit + covered_steps + 3 * 4
assert said3.count("phis rd: fitted on") == 1 and said3.count("phis ta: fitted on") == 1 and "phis sg" not in said3
for name, names in specs3:
    r, ref, folder = res3[name], sep[name], TMP / "sep3" / name
    assert r["mode"] == "joint" and r["step"] == 12
    assert same_state(r["student"], ref["student"]) and same_state(r["heads"], ref["heads"]), name
    assert r["held_out"] == ref["held_out"], name
    assert no_timing(log_of(root3 / name)) == no_timing(log_of(folder)), name
    assert (root3 / name / "held_out.csv").read_bytes() == (folder / "held_out.csv").read_bytes(), name
    assert load_checkpoint(root3 / name / "final.pt")["summary_only"] == [n for n in names if n in ("sg", "sn")]
    assert {p.name for p in (root3 / name).glob("phis_*.pt")} == {f"phis_{n}.pt" for n in names if n not in ("sg", "sn")}
assert list(log_of(root3 / "student_F")[0])[3:8] == ["sg/summary", "ta/summary", "ta/spatial", "rd/summary", "rd/spatial"]
assert list(log_of(root3 / "student_C")[0])[3:6] == ["sg/summary", "ta/summary", "ta/spatial"]
assert [math.isnan(float(r["rd/summary"])) for r in log_of(root3 / "student_F")] == \
    [len(sb.batch(s).s1_rows) == 0 for s in range(12)]
assert (root3 / "student_F" / "phis_rd.pt").read_bytes() == (root3 / "student_N" / "phis_rd.pt").read_bytes()
# held_out_losses - the teachers once per held-out batch for all students, the numbers of one pass each
held3 = Batcher(h_s1, h_s2, h_aerial, small, "cpu")
models3 = [(res3[n]["student"], res3[n]["heads"], [by[x] for x in names], res3[n]["phis"]) for n, names in specs3]
for c in counters:
    c.calls = 0
hos3 = T.held_out_losses(models3, held3, norm, small, auto0)
assert (sg.calls, sg_aer.calls, sn.calls, ta.calls, ta_aerial.calls, rd.calls) == (4, 5, 4, 4, 5, 4)
for c in counters:
    c.calls = 0
alone = [held_out_loss(m[0], m[1], m[2], held3, norm, m[3], small, auto0) for m in models3]
assert (sg.calls, ta.calls, rd.calls) == (2 * 4, 3 * 4, 2 * 4)          # one pass per student runs them again
assert hos3 == alone == [res3[n]["held_out"] for n, _ in specs3]
assert all(m[0].training and m[1].training for m in models3)
# cut at 7, resumed from the common step 5 - the run in one go
root3c = TMP / "joint3_cut"
train_students(specs3, small, data, pool, root3c, device="cpu", joint=True, stop_after=7, summary_only=("sg", "sn"))
res3c, said3c = printed(train_students, specs3, small, data, pool, root3c, device="cpu", joint=True,
                        summary_only=("sg", "sn"))
for name, _ in specs3:
    assert f"resumed from {root3c / name / 'last.pt'} at step 5" in said3c
    assert same_state(res3c[name]["student"], sep[name]["student"]) and same_state(res3c[name]["heads"], sep[name]["heads"])
    assert no_timing(log_of(root3c / name)) == no_timing(log_of(TMP / "sep3" / name))
assert not prefetch_threads()
# one after the other gives the same, and a resume with another summary-only set stops
res3s = train_students(specs3, small, data, pool, TMP / "joint3_seq", device="cpu", joint=False,
                       summary_only=("sg", "sn"))
for name, _ in specs3:
    assert res3s[name]["mode"] == "sequential" and same_state(res3s[name]["student"], sep[name]["student"])
before3 = snapshot(*(root3c / name for name, _ in specs3))
sg.calls = sg_aer.calls = 0
raises(ValueError, train_students, specs3, small, data, pool, root3c, device="cpu", joint=True,
       says="this run has [] - the heads differ")
assert snapshot(*(root3c / name for name, _ in specs3)) == before3 and sg.calls == sg_aer.calls == 0
# PHI-S once per step for all students that share it - 4 steps, no held-out pass in them
calls_phis = []
_phis_call = PhiS.__call__


def _phis_spy(self, x):
    calls_phis.append(self.name)
    return _phis_call(self, x)


PhiS.__call__ = _phis_spy
try:
    train_students(specs3, stop_cfg, data, pool, TMP / "joint3_phis", device="cpu", joint=True, stop_after=4,
                   summary_only=("sg", "sn"))
finally:
    PhiS.__call__ = _phis_call
covered4 = sum(len(sb.batch(s).s1_rows) > 0 for s in range(4))
# one call in each fit (its variance check), then one per step - not 3 for ta and 2 for rd as per student
assert calls_phis.count("ta") == 1 + 4 and calls_phis.count("rd") == 1 + covered4, calls_phis
assert set(calls_phis) == {"ta", "rd"}
print("train_students F, N, C: bit for bit the single runs, teachers and PHI-S once per step, held-out once, resume ok")

# ---- folders with different PHI-S files for one teacher - that teacher is standardised per student
root_dp = TMP / "joint_diff_phis"
for name in ("student_M", "student_S"):
    (root_dp / name).mkdir(parents=True)
shutil.copyfile(dir_a / "phis_ta.pt", root_dp / "student_M" / "phis_ta.pt")    # the fit of the single run
odd = fit_phis(lambda bb: target_of(targets[0], bb), sb, n=4, name="ta")         # another fit, 4 samples
odd.save(root_dp / "student_S" / "phis_ta.pt")
dir_odd = TMP / "single_S_odd"
dir_odd.mkdir()
odd.save(dir_odd / "phis_ta.pt")
res_odd = train_student(small, data, [targets[0]], dir_odd, device="cpu")
res_dp = train_students(specs, small, data, targets, root_dp, device="cpu", joint=True)
assert set(T._shared_phis([res_dp["student_M"]["phis"], res_dp["student_S"]["phis"]])) == {"tb"}
for name, ref, folder in (("student_M", res_a, dir_a), ("student_S", res_odd, dir_odd)):
    assert same_state(res_dp[name]["student"], ref["student"]) and same_state(res_dp[name]["heads"], ref["heads"])
    assert no_timing(log_of(root_dp / name)) == no_timing(log_of(folder)), name
    assert (root_dp / name / "held_out.csv").read_bytes() == (folder / "held_out.csv").read_bytes(), name
assert not same_state(res_dp["student_S"]["student"], res_single_s["student"])  # the other PHI-S did count
print("different PHI-S files of one teacher - standardised per student, the single runs bit for bit")

# ---- a resume checks the teachers - names, widths and their order - and a refusal touches nothing
tc = ToyTeacher("tc", 8, 2, 12, 1000.0, 14)
pair = [targets[0], Target("tc", 8, 2, tc)]                          # ta and tc, both 8 wide
dir_ord = TMP / "run_order"
train_student(small, data, pair, dir_ord, device="cpu", stop_after=7)
before_ord = snapshot(dir_ord)
raises(ValueError, train_student, small, data, pair[::-1], dir_ord, device="cpu",
       says="written for the teachers [('ta', 8), ('tc', 8)], this run has [('tc', 8), ('ta', 8)]")
raises(ValueError, train_student, small, data, [targets[0]], dir_ord, device="cpu", says="this run has [('ta', 8)]")
assert snapshot(dir_ord) == before_ord
root_o = TMP / "joint_order"
train_students([("student_M", ["ta", "tc"]), ("student_S", ["ta"])], small, data, pair, root_o, device="cpu",
               joint=True, stop_after=7)
before_o = snapshot(root_o / "student_M", root_o / "student_S")
raises(ValueError, train_students, [("student_M", ["tc", "ta"]), ("student_S", ["ta"])], small, data, pair, root_o,
       device="cpu", joint=True, says="(name and width, in this order)")
assert snapshot(root_o / "student_M", root_o / "student_S") == before_o
# a teacher added to the student behind - M at {10, 5}, S at {5}, S's spec gets tb
root_ts = TMP / "joint_teacher_set"
train_students(specs, small, data, targets, root_ts, device="cpu", joint=True, stop_after=10)
(root_ts / "student_S" / "last.pt").unlink()
grown = [("student_M", ["ta", "tb"]), ("student_S", ["ta", "tb"])]
folders_ts = (root_ts / "student_M", root_ts / "student_S")
before_ts = snapshot(*folders_ts)
for c in (ta, ta_aerial, tb):
    c.calls = 0
raises(ValueError, train_students, grown, small, data, targets, root_ts, device="cpu", joint=True,
       says="this run has [('ta', 8), ('tb', 24)]")
assert snapshot(*folders_ts) == before_ts and ta.calls == ta_aerial.calls == tb.calls == 0    # no PHI-S fit of tb
# with S's PHI-S file there the resume itself refuses, after it read M (whose last.pt is cut), before any change
shutil.copyfile(root_ts / "student_M" / "phis_tb.pt", root_ts / "student_S" / "phis_tb.pt")
whole = (root_ts / "student_M" / "last.pt").read_bytes()
(root_ts / "student_M" / "last.pt").write_bytes(whole[:len(whole) // 2])
before_ts = snapshot(*folders_ts)
_, said_ts = printed(raises, ValueError, train_students, grown, small, data, targets, root_ts, device="cpu",
                     joint=True, says="this run has [('ta', 8), ('tb', 24)]")
assert "last.pt cannot be loaded" in said_ts and "moved to" not in said_ts and "resumed" not in said_ts, said_ts
assert snapshot(*folders_ts) == before_ts       # M's cut last.pt not moved, no log cut, no last.pt deleted
assert not prefetch_threads()
print("resume checks - teacher order and set refused before any PHI-S fit, load or file change")

notes = " ".join(T.__doc__.split())
assert "bit for bit on the CPU" in notes and "not bit for bit" in notes     # the CUDA limit is named
assert "SMALLER step" in notes and "JOINT_MIN_GB" in notes and "teachers run ONCE" in notes
assert "teachers ONCE per held-out batch" in notes and "nan" in notes and "summary_only" in notes
assert {f.name for f in fields(TrainConfig)} == {"steps", "batch", "n_aerial", "crop", "lr",
                                                 "weight_decay", "betas", "warmup", "clip", "seed",
                                                 "bf16", "checkpoint_every", "held_out_every",
                                                 "progress_every", "phis_samples", "student"}
print("test_train: all checks passed")
