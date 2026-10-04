"""The fast decoder path of part four (domains/run.py) and the compile cache helpers (distil/score.py), on the CPU.

Small tests only - tiny models, no real weights, and no Inductor (the test machine has no C++ compiler). torch.compile is
replaced by stand-ins, and in two places by the real torch.compile with backend "aot_eager" (dynamo and
AOTAutograd, no code generation).

1. part two's step bit for bit - _train_decoder with path None, "eager" and "compiled" (a stand-in that
   calls the same function) against a verbatim copy of the code before the change; aot_eager close
2. run_domain without fast_decoder - the same files and printed lines as with that copy, no new info keys
3. the rule (_decide) on numbers, every branch
4. decoder_speed_check with stand-ins - faster and agreeing, faster but disagreeing, slower, compile raising
   (at the compile, at the first call, after it), the real torch.compile with aot_eager, random state and
   cudnn.benchmark left as they were
5. decoder_choice - once per shape and process, one line with both ms, a failing check gives eager
6. run_domain with fast_decoder - compiled path with the files of part two's step, the new info keys next
   to the old ones, the cache folder in the environment; a compiled step that fails in the run; a compile
   that raises
7. score_all and score_parallel - fast_decoder and the cache folder reach every run and every worker;
   without them nothing new is passed
8. the archive round trip - saved, unchanged, saved again, restored, kept, none, a path outside refused
"""
import _setup  # noqa: F401

import contextlib
import functools
import io
import json
import os
import queue
import re
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _setup import TMP_BASE  # noqa: E402

from sartransfer.distil import score as SC  # noqa: E402
from sartransfer.domains import prepared as P, run as R  # noqa: E402
from sartransfer.models import LAST, encoders as encmod  # noqa: E402
from sartransfer.models.decoder import GridDecoder  # noqa: E402

TMP = TMP_BASE / "fast_decoder"
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True)
torch.manual_seed(0)
REAL = {"compile": R._compile, "step": R._step_loss, "check": R.decoder_speed_check, "run_domain": SC.run_domain,
        "context": SC._context}
ENV_KEYS = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_FX_GRAPH_CACHE", "TORCHINDUCTOR_AUTOGRAD_CACHE")
HAD_ENV = {k: os.environ.get(k) for k in ENV_KEYS}


def fails(fn, exc, text=""):
    try:
        fn()
    except exc as e:
        assert text in str(e), f"expected {text!r} in {e!r}"
        return
    raise SystemExit(f"no {exc.__name__} was raised")


def quiet(fn, *args, **kwargs):
    """fn's result and what it printed."""
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        out = fn(*args, **kwargs)
    return out, buf.getvalue()


def same_weights(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def minutes_out(text: str) -> str:
    return re.sub(r"[\d.]+ min", "X min", text)


def part_two_train_decoder(head, feats, skip_u8, labels, k, device, steps, batch, lr, seed, log_every=1000):
    """run._train_decoder as it was before part four's change (2026-10-04), copied verbatim."""
    import torch
    import torch.nn as nn

    IGNORE, TrainSet = R.IGNORE, R.TrainSet
    rng = np.random.default_rng(seed)
    counts = np.zeros(k, np.float64)
    for lab in labels:
        counts += np.bincount(lab[lab != IGNORE].ravel(), minlength=k)[:k]
    w = torch.tensor(counts.sum() / (k * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)
    head = head.to(device)
    ts = TrainSet(list(feats), list(skip_u8), list(labels), device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    t0, running, n = time.perf_counter(), torch.zeros((), device=device), 0
    for it in range(1, steps + 1):
        head.train()
        idx = rng.choice(len(feats), batch, replace=len(feats) < batch)
        f, raw, y, _ = ts.batch(idx)
        if rng.random() < 0.5:
            f, raw, y = f.flip(-1), raw.flip(-1), y.flip(-1)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = loss_fn(head(f, raw.float() / 255.0).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.detach()
        n += 1
        if it % log_every == 0 or it == steps:
            print(f"    step {it:>5}  loss {float(running) / n:.4f}  {(time.perf_counter() - t0) / 60:.1f} min")
            running, n = torch.zeros((), device=device), 0
    return head


def wrapping(fn):
    """A compile stand-in - another function object that computes exactly what fn computes."""
    return lambda *a, **k: fn(*a, **k)


# ---- 1. part two's step bit for bit ---------------------------------------------------------------
rng = np.random.default_rng(0)
N_CHIPS, TILE, WIDTH = 7, 32, 8
feats = rng.normal(0, 1, (N_CHIPS, TILE // 16, TILE // 16, WIDTH)).astype(np.float16)
skip = rng.integers(0, 256, (N_CHIPS, TILE, TILE), dtype=np.uint8)
labels = rng.integers(0, 3, (N_CHIPS, TILE, TILE)).astype(np.uint8)
labels[:, :3] = 255


def train(fn, k=3, **kw):
    torch.manual_seed(0)
    head = GridDecoder(WIDTH, n_classes=k)
    out, text = quiet(fn, head, feats, skip, labels, k, "cpu", 12, 3, 1e-3, 5, log_every=4, **kw)
    return out.state_dict(), minutes_out(text)


ref_weights, ref_text = train(part_two_train_decoder)
for path in (None, "eager"):
    report = {}
    weights, text = train(R._train_decoder, path=path, report=report)
    assert same_weights(weights, ref_weights) and text == ref_text, path
    assert report["path"] == "eager" and report["step_ms"] > 0, report
R._compile = wrapping
report = {}
weights, text = train(R._train_decoder, path="compiled", report=report)
assert same_weights(weights, ref_weights) and text == ref_text and report["path"] == "compiled", report
torch._dynamo.reset()
R._compile = lambda fn: torch.compile(fn, backend="aot_eager", dynamic=False)
weights, _ = train(R._train_decoder, path="compiled", report=report)
# AOTAutograd's backward sums some gradients in another order, float32 rounding that AdamW's normalisation
# enlarges for gradients near 0; one AdamW step moves a weight by up to lr = 1e-3, 12 steps by up to 1.2e-2.
# How large the rounding comes out depends on the CPU and the torch build (2.7e-4 on a GitHub ubuntu runner
# with torch 2.14.1+cpu, smaller elsewhere), so the bound is one step's move, lr - a compiled path that
# computed something else would drift by that much on every step and far exceed it over 12
aot_diff = max(float((weights[k] - ref_weights[k]).abs().max()) for k in weights)
assert report["path"] == "compiled" and aot_diff < 1e-3, aot_diff
R._compile = REAL["compile"]
assert torch.backends.cudnn.benchmark is False, "cudnn.benchmark was not put back"
print(f"part two's step verified - path None, eager and a compiled stand-in equal the old code bit for bit, "
      f"aot_eager after 12 steps within {aot_diff:.1e} of it")

# ---- 2. run_domain without fast_decoder, the files of the old step --------------------------------
PREP = TMP / "prepared"
w = P.PreparedWriter(PREP, "toy", 64, {0: "background", 1: "lake"}, source={"url": "synthetic"}, pixel_spacing_m=10.0)
for split, n in (("train", 12), ("val", 4)):
    for i in range(n):
        yy, xx = np.mgrid[:64, :64]
        lake = (yy - 32) ** 2 + (xx - int(rng.integers(16, 48))) ** 2 < 200
        img = np.stack([np.where(lake, -22.0, -9.0) + rng.normal(0, 1.5, (64, 64)),
                        np.where(lake, -29.0, -16.0) + rng.normal(0, 1.0, (64, 64))]).astype(np.float32)
        lab = lake.astype(np.uint8)
        lab[:3, :3] = 255
        w.add(split, img, lab, region=f"r{i % 2}", date="2020-08-01")
w.close()


class FakeEncoder:
    patch, dim, _n_prefix, n_layers = 16, 8, 0, 4
    spec = encmod.EncoderSpec("fake/model", "fake", "fake", batch=4)

    def __init__(self, *a, **k):
        pass

    def encode(self, tiles, batch=None, layer=None):
        n, _, t, _ = tiles.shape
        g = t // 16
        b = tiles[:, :2].reshape(n, 2, g, 16, g, 16)
        m, s = b.mean((3, 5)), b.std((3, 5))
        return np.concatenate([m, s, m * 0.5, -m], 1).transpose(0, 2, 3, 1).astype(np.float16)


RealFrozen = encmod.FrozenEncoder
encmod.ENCODERS["fake"] = FakeEncoder.spec
encmod.FrozenEncoder = FakeEncoder
RUN = "dom_toy__fake__L1__vv_vh_diff"
KW = dict(token=None, layer=1, seeds=(0, 1), steps=30, batch=3, device="cpu")   # 20 steps divide by 0 in OneCycleLR


def run_files(res: Path) -> dict:
    return {suffix: (res / f"{RUN}{suffix}").read_bytes() for suffix in
            ("__summary.csv", "__probe__regions.csv", "__decoder__s0__regions.csv", "__decoder__s1__regions.csv")}


def heads(res: Path) -> dict:
    return {s: torch.load(res / f"{RUN}__decoder__s{s}.pt") for s in (0, 1)}


def info(res: Path) -> dict:
    return json.loads((res / f"{RUN}__info.json").read_text(encoding="utf-8"))


res_old, res_new = TMP / "res_old", TMP / "res_new"
real_train = R._train_decoder


def old_step(*args, path=None, report=None, **kwargs):
    assert path is None, "the default run asked for a fast path"
    return part_two_train_decoder(*args, **kwargs)


R._train_decoder = old_step
_, text_old = quiet(R.run_domain, PREP, "toy", "fake", res_old, **KW)
R._train_decoder = real_train
_, text_new = quiet(R.run_domain, PREP, "toy", "fake", res_new, **KW)
assert run_files(res_new) == run_files(res_old) and minutes_out(text_new) == minutes_out(text_old)
assert all(same_weights(heads(res_new)[s], heads(res_old)[s]) for s in (0, 1))
OLD_TIMES = set(info(res_old)["times"])
assert set(info(res_new)["times"]) == OLD_TIMES and not any(k.startswith("decoder_check") for k in OLD_TIMES)
assert "decoder step" not in text_new and R._CHOICES == {}, "the default run measured a speed"
print("run_domain without fast_decoder verified - the files, heads, info keys and printed lines of the old step")

# ---- 3. the rule ----------------------------------------------------------------------------------
BASE = {"error": None, "eager_ms": 100.0, "compiled_ms": 50.0, "loss_gap": 1e-5, "loss_tolerance": 3e-4,
        "grad_gap": 0.005, "grad_tolerance": 0.03}
assert R._decide(BASE) == ("compiled", "compiled is faster and agrees")
assert R._decide({**BASE, "compiled_ms": 94.0})[0] == "compiled"
assert R._decide({**BASE, "compiled_ms": 96.0}) == ("eager", "compiled is not 5 percent faster")
assert R._decide({**BASE, "compiled_ms": 150.0})[0] == "eager"
for bad in ({"loss_gap": 4e-4}, {"grad_gap": 0.04}, {"loss_gap": float("nan")}, {"grad_gap": None}):
    assert R._decide({**BASE, **bad}) == ("eager", "compiled disagrees with eager"), bad
assert R._decide({**BASE, "error": "RuntimeError stand-in"}) == ("eager", "compile failed")
assert R._decide({**BASE, "compiled_ms": None})[0] == "eager" and R._decide({**BASE, "eager_ms": None})[0] == "eager"
assert (R.LOSS_FLOOR, R.GRAD_FLOOR, R.MIN_GAIN) == (2e-4, 1e-2, 0.05)
print("the rule verified - faster and agreeing, not faster, disagreeing loss or gradient, nan, compile failed")

# ---- 4. decoder_speed_check with stand-ins -------------------------------------------------------
SLOW = 0.1                                   # seconds added to a step of the path that must lose
CHECK = dict(n_classes=3, batch=2, warm=1, steps=3, rounds=3)


def slowed(fn):
    def step(*a, **k):
        time.sleep(SLOW)
        return fn(*a, **k)
    return step


def check(**kw):
    state = torch.get_rng_state()
    r = R.decoder_speed_check(TILE, WIDTH, "cpu", **{**CHECK, **kw})
    assert torch.equal(state, torch.get_rng_state()), "the check moved the global random state"
    assert torch.backends.cudnn.benchmark is False
    assert (r["path"], r["why"]) == R._decide(r), r
    return r


plain_step = REAL["step"]
try:
    R._step_loss = slowed(plain_step)                                  # eager slow, compiled fast and the same
    R._compile = lambda fn: wrapping(plain_step)
    r = check()
    assert r["path"] == "compiled" and r["compiled_ms"] < r["eager_ms"], r
    assert r["loss_gap"] == 0.0 and r["grad_gap"] == 0.0 and r["error"] is None and r["compile_s"] is not None
    assert r["loss_tolerance"] == R.LOSS_FLOOR and r["grad_tolerance"] == R.GRAD_FLOOR, "on the CPU bf16 is off"
    R._compile = lambda fn: (lambda *a, **k: plain_step(*a, **k) * 1.05)  # fast but wrong by 5 percent
    r = check()
    assert r["path"] == "eager" and r["why"] == "compiled disagrees with eager" and r["compiled_ms"] < r["eager_ms"]
    assert abs(r["loss_gap"] - 0.05) < 1e-6 and abs(r["grad_gap"] - 0.05) < 1e-6, r
    R._step_loss = plain_step                                          # compiled slow
    R._compile = lambda fn: slowed(plain_step)
    r = check()
    assert r["path"] == "eager" and r["why"] == "compiled is not 5 percent faster" and r["loss_gap"] == 0.0, r

    def no_compiler(fn):
        raise RuntimeError("stand-in: no C++ compiler")

    R._compile = no_compiler
    r = check()
    assert r["path"] == "eager" and r["why"] == "compile failed" and r["compiled_ms"] is None and r["eager_ms"] > 0
    assert r["error"] == "RuntimeError stand-in - no C++ compiler" and r["loss_gap"] is None
    line = R._choice_line(r)
    assert "compiled failed (RuntimeError stand-in - no C++ compiler)" in line and ":" not in line, line

    def raises_at(n):
        """A compile stand-in whose compiled function raises from its n-th call on."""
        def make(fn):
            calls = []

            def step(*a, **k):
                calls.append(1)
                if len(calls) >= n:
                    raise ValueError(f"stand-in, call {len(calls)}")
                return plain_step(*a, **k)
            return step
        return make

    for n, first in ((1, True), (3, False)):                          # at the first call, after it
        R._compile = raises_at(n)
        r = check()
        assert r["path"] == "eager" and r["why"] == "compile failed" and r["compiled_ms"] is None, (n, r)
        assert r["error"].startswith("ValueError stand-in, call") and (r["loss_gap"] is None) == first, (n, r)
    torch._dynamo.reset()                                              # the real torch.compile, no code generation
    R._compile = lambda fn: torch.compile(fn, backend="aot_eager", dynamic=False)
    r = check()
    assert r["error"] is None and r["compiled_ms"] > 0 and r["compile_s"] > 0, r
    assert r["loss_gap"] <= r["loss_tolerance"] and r["grad_gap"] <= r["grad_tolerance"], r
    aot_line = R._choice_line(r)
finally:
    R._step_loss, R._compile = plain_step, REAL["compile"]
assert ":" not in aot_line and "eager " in aot_line and "compiled " in aot_line, aot_line
print(aot_line)
print("decoder_speed_check verified - faster and agreeing, faster but 5 percent off, slower, compile raising at the "
      "compile, at the first call and after it, the real torch.compile (aot_eager), random state kept")

# ---- 5. decoder_choice, once per shape and process ------------------------------------------------
asked = []


def counted(tile, n_feat, device, n_classes=2, batch=8):
    asked.append((tile, n_feat, n_classes, batch, device))
    out = {"tile": tile, "n_feat": n_feat, "n_classes": n_classes, "batch": batch, "device": device,
           **BASE, "eager_ms": 10.0, "compiled_ms": 5.0, "compile_s": 42.0}
    out["path"], out["why"] = R._decide(out)
    return out


R._CHOICES.clear()
R.decoder_speed_check = counted
try:
    first, text1 = quiet(R.decoder_choice, 992, 768, 2, "cpu", 8)
    again, text2 = quiet(R.decoder_choice, 992, 768, 2, "cpu", 8)
    other, text3 = quiet(R.decoder_choice, 992, 1024, 2, "cpu", 8)
    assert asked == [(992, 768, 2, 8, "cpu"), (992, 1024, 2, 8, "cpu")] and again is first and text2 == ""
    assert text1.strip() == ("decoder step 992 px, 768 features, 2 classes, batch 8 - eager 10.0 ms, compiled "
                             "5.0 ms (first call 42 s), loss gap 1.0e-05 (tolerance 3.0e-04), gradient gap 5.0e-03 "
                             "(tolerance 3.0e-02) - compiled, compiled is faster and agrees"), text1
    assert text1.count("\n") == 1 and text3.count("\n") == 1 and first["path"] == "compiled"

    def out_of_memory(*a, **k):
        raise MemoryError("stand-in: out of memory")

    R.decoder_speed_check = out_of_memory
    failed, text4 = quiet(R.decoder_choice, 128, 768, 2, "cpu", 8)
    assert failed["path"] == "eager" and failed["why"] == "the speed check failed", failed
    assert "the check failed (MemoryError stand-in - out of memory) - eager, the speed check failed" in text4, text4
    assert ":" not in text1 + text3 + text4
finally:
    R.decoder_speed_check = REAL["check"]
    R._CHOICES.clear()
print("decoder_choice verified - one check and one line per shape, kept for the process, a failing check gives eager")

# ---- 6. run_domain with fast_decoder ---------------------------------------------------------------
NEW_TIMES = {"decoder_path", "decoder_step_ms", "decoder_s0_step_ms", "decoder_s1_step_ms", "decoder_check_eager_ms",
             "decoder_check_compiled_ms", "decoder_check_compile_s", "decoder_check_why"}
compiles = []


def counted_wrapping(fn):
    compiles.append(fn)
    return wrapping(fn)


def fast_run(name, compile_with, cache=None):
    """run_domain with fast_decoder on a stand-in check that picks the compiled path."""
    R._CHOICES.clear()
    R.decoder_speed_check = counted
    R._compile = compile_with
    try:
        _, text = quiet(R.run_domain, PREP, "toy", "fake", TMP / name, fast_decoder=True, compile_cache=cache, **KW)
    finally:
        R.decoder_speed_check, R._compile = REAL["check"], REAL["compile"]
        R._CHOICES.clear()
    return TMP / name, text


cache_a = TMP / "compile_cache_a"
try:
    res_fast, text = fast_run("res_fast", counted_wrapping, cache_a)
    assert len(compiles) == 2 and all(fn is REAL["step"] for fn in compiles), "one compile per seed, of the step"
    assert run_files(res_fast) == run_files(res_old), "the compiled path changed a number"
    assert all(same_weights(heads(res_fast)[s], heads(res_old)[s]) for s in (0, 1))
    times = info(res_fast)["times"]
    assert set(times) == OLD_TIMES | NEW_TIMES, set(times) ^ (OLD_TIMES | NEW_TIMES)
    assert times["decoder_path"] == "compiled" and times["decoder_check_why"] == "compiled is faster and agrees"
    assert (times["decoder_check_eager_ms"], times["decoder_check_compiled_ms"], times["decoder_check_compile_s"]) \
        == (10.0, 5.0, 42.0)
    assert times["decoder_s0_step_ms"] > 0 and abs(times["decoder_step_ms"] - (times["decoder_s0_step_ms"]
                                                                               + times["decoder_s1_step_ms"]) / 2) < 1e-9
    assert {k: v for k, v in info(res_fast).items() if k != "times"} == \
        {k: v for k, v in info(res_old).items() if k != "times"}
    new_lines = [ln for ln in text.splitlines() if "decoder step" in ln or " path, " in ln]
    assert len(new_lines) == 3 and "decoder step 64 px, 8 features, 2 classes, batch 3 - eager 10.0 ms" in new_lines[0]
    assert re.fullmatch(r"  decoder seed 1 - compiled path, [\d.]+ ms per training step", new_lines[2]), new_lines
    assert not [ln for ln in new_lines if ":" in ln]
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(cache_a.resolve()) and cache_a.is_dir()
    assert os.environ["TRITON_CACHE_DIR"] == str(cache_a.resolve() / "triton")
    assert os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1" == os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"]
    import torch._functorch.config as functorch_config  # noqa: E402
    import torch._inductor.config as inductor_config  # noqa: E402

    assert inductor_config.fx_graph_cache is True and functorch_config.enable_autograd_cache is True
    from torch._inductor.runtime.cache_dir_utils import cache_dir  # noqa: E402

    assert cache_dir() == str(cache_a.resolve()), "Inductor does not read the folder"

    def fails_from_step(n):
        """A compile stand-in whose step raises at its n-th call (a fresh count per compile, so per seed)."""
        def make(fn):
            calls = []

            def step(*a, **k):
                calls.append(1)
                if len(calls) == n:
                    raise RuntimeError("stand-in: a kernel failed")
                return fn(*a, **k)
            return step
        return make

    res_mid, text = fast_run("res_mid", fails_from_step(5))
    assert run_files(res_mid) == run_files(res_old) and all(same_weights(heads(res_mid)[s], heads(res_old)[s])
                                                            for s in (0, 1)), "the step done again changed a number"
    assert text.count("compiled step 5 failed (RuntimeError stand-in - a kernel failed), this step and the rest run "
                      "eager") == 2, text
    assert info(res_mid)["times"]["decoder_path"] == "compiled, then eager from step 5"

    def no_compiler(fn):
        raise RuntimeError("stand-in: no C++ compiler")

    res_nc, text = fast_run("res_nc", no_compiler)
    assert run_files(res_nc) == run_files(res_old) and info(res_nc)["times"]["decoder_path"] == "eager"
    assert text.count("torch.compile failed (RuntimeError stand-in - no C++ compiler), the decoder trains eager") == 2
finally:
    for key, value in HAD_ENV.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
print("run_domain with fast_decoder verified - the files of part two's step on the compiled path, the new info keys "
      "next to the old ones, the cache folder in the environment, a failing step done again eager, no compiler")

# ---- 7. score_all and score_parallel pass the switch and the cache folder -------------------------
runs = []


@functools.wraps(R.run_domain)
def recorded(prepared_root, dataset, encoder, res_dir, token, **kw):
    runs.append({"encoder": encoder, "dataset": dataset, "pid": os.getpid(), **kw})
    return pd.DataFrame([{"run": f"{dataset}-{encoder}", "dataset": dataset, "encoder": encoder}])


class InProcess:
    """A stand-in multiprocessing context. Process.start() runs the worker here, to its end, and keeps its args."""

    Queue = queue.Queue
    started = []

    class Process:
        def __init__(self, target, name, args):
            self.target, self.args, self.exitcode = target, args, None

        def start(self):
            InProcess.started.append(self.args)
            self.target(*self.args)
            self.exitcode = 0

        def is_alive(self):
            return False

        def join(self):
            pass

        def terminate(self):
            raise AssertionError("nothing is left to stop")


ENCS = ["fast-a", "fast-b"]
for e in ENCS:
    encmod.ENCODERS[e] = encmod.EncoderSpec("fake/model", e, "fake", batch=4, in_channels=(2,))
    R.PRE_REGISTERED_LAYERS[e] = LAST
cache_b = TMP / "compile_cache_b"
SC.run_domain, SC._context = recorded, (lambda: InProcess)
try:
    quiet(SC.score_all, PREP, ENCS, ["toy"], TMP / "res7", token=None, device="cpu", fast_decoder=True,
          compile_cache=cache_b, seeds=(0,))
    assert [(r["encoder"], r["fast_decoder"], r["compile_cache"], r["seeds"]) for r in runs] == \
        [("fast-a", True, str(cache_b), (0,)), ("fast-b", True, str(cache_b), (0,))], runs
    runs.clear()
    quiet(SC.score_all, PREP, ENCS, ["toy"], TMP / "res7", token=None, device="cpu", seeds=(0,))
    quiet(SC.score_all, PREP, ENCS, ["toy"], TMP / "res7", token=None, device="cpu", compile_cache=cache_b, seeds=(0,))
    assert len(runs) == 4 and not any("fast_decoder" in r or "compile_cache" in r for r in runs), runs
    runs.clear()
    (rows, table), text = quiet(SC.score_parallel, PREP, ENCS, ["toy"], TMP / "res7p", token=None, device="cpu",
                                workers=2, fast_decoder=True, compile_cache=cache_b, seeds=(0,))
    assert [a[0] for a in InProcess.started] == ENCS and list(table["status"]) == ["done", "done"], (text, table)
    assert all(a[10] == {"seeds": (0,), "fast_decoder": True, "compile_cache": str(cache_b)} for a in InProcess.started)
    assert sorted(r["encoder"] for r in runs) == ENCS and all(r["compile_cache"] == str(cache_b) and r["fast_decoder"]
                                                              for r in runs), runs
    runs.clear()
    InProcess.started.clear()
    quiet(SC.score_parallel, PREP, ENCS, ["toy"], TMP / "res7q", token=None, device="cpu", workers=2, seeds=(0,))
    assert all(a[10] == {"seeds": (0,)} for a in InProcess.started) and len(runs) == 2
    assert not any("fast_decoder" in r or "compile_cache" in r for r in runs)
finally:
    SC.run_domain, SC._context = REAL["run_domain"], REAL["context"]
    for e in ENCS:
        del encmod.ENCODERS[e], R.PRE_REGISTERED_LAYERS[e]
    encmod.FrozenEncoder = RealFrozen
print("score_all and score_parallel verified - the switch and the cache folder reach every run and every worker, "
      "nothing new without them")

# ---- 8. the compile cache as one archive ----------------------------------------------------------
folder, archive = TMP / "cache_folder", TMP / "drive" / "compile_cache.tar"
(folder / "fxgraph" / "ab").mkdir(parents=True)
(folder / "triton" / "0").mkdir(parents=True)
(folder / "fxgraph" / "ab" / "entry").write_bytes(rng.integers(0, 256, 5000, dtype=np.uint8).tobytes())
(folder / "triton" / "0" / "kernel.cubin").write_bytes(bytes(range(256)) * 3)
(folder / "empty.json").write_bytes(b"")
(folder / "locks").mkdir()
(folder / "locks" / "graph.lock").write_bytes(b"held")
saved, text = quiet(SC.save_compile_cache, folder, archive)
assert saved["status"] == "saved" and saved["files"] == 3 and archive.is_file(), saved
assert not archive.with_name(archive.name + ".part").exists() and archive.with_name(archive.name + ".json").is_file()
assert re.fullmatch(r"compile cache saved - 3 files, 0\.0 MB in [\d.]+ s, \d+ MB per second, one archive "
                    r"compile_cache\.tar\n", text), text
with tarfile.open(archive) as tar:
    assert sorted(tar.getnames()) == ["empty.json", "fxgraph/ab/entry", "triton/0/kernel.cubin"], tar.getnames()
again, text = quiet(SC.save_compile_cache, folder, archive)
assert again["status"] == "unchanged" and text == "compile cache unchanged - 3 files, 0.0 MB, compile_cache.tar kept\n"
(folder / "fxgraph" / "ab" / "second").write_bytes(b"new graph")
assert quiet(SC.save_compile_cache, folder, archive)[0]["status"] == "saved"
fresh = TMP / "fresh_runtime" / "compile_cache"
restored, text = quiet(SC.restore_compile_cache, archive, fresh)
assert restored["status"] == "unpacked" and restored["files"] == 4, restored
assert re.fullmatch(r"compile cache restored - 4 files, 0\.0 MB from compile_cache\.tar in [\d.]+ s, \d+ MB per "
                    r"second\n", text), text
for name in ("fxgraph/ab/entry", "fxgraph/ab/second", "triton/0/kernel.cubin", "empty.json"):
    assert (fresh / name).read_bytes() == (folder / name).read_bytes(), name
assert not (fresh / "locks").exists(), "a lock file was archived"
kept, text = quiet(SC.restore_compile_cache, archive, fresh)
assert kept["status"] == "kept" and text == "compile cache folder compile_cache already holds 4 files, " \
                                            "compile_cache.tar not unpacked\n", text
none, text = quiet(SC.restore_compile_cache, TMP / "drive" / "missing.tar", TMP / "new_folder")
assert none["status"] == "none" and (TMP / "new_folder").is_dir()
assert text == "no compile cache archive missing.tar yet, the first compile of each shape fills new_folder\n", text
empty, text = quiet(SC.save_compile_cache, TMP / "nothing_here", TMP / "drive" / "nothing.tar")
assert empty["status"] == "empty" and not (TMP / "drive" / "nothing.tar").exists()
evil = TMP / "drive" / "evil.tar"                       # a member that points outside the folder is never written
with tarfile.open(evil, "w") as tar:
    member = tarfile.TarInfo("../outside.txt")
    member.size = 4
    tar.addfile(member, io.BytesIO(b"evil"))
target = TMP / "evil_target" / "cache"
try:
    quiet(SC.restore_compile_cache, evil, target)
except tarfile.TarError:
    pass
assert not (TMP / "evil_target" / "outside.txt").exists()
print("compile cache archive verified - one tar, lock files left out, unchanged skipped, restored byte for byte, "
      "a filled folder kept, no archive, an empty folder, a path outside refused")

print("test_fast_decoder: all checks passed")
