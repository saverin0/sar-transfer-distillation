"""Frozen features -> linear probe + small decoder -> metrics, for any prepared domain.

Copied from part two, whose run step is used unchanged.

PRE-REGISTERED 2026-09-26, before any new-domain number (no tuning per domain):
- encoders and layers from the glacier study: dinov3-l-sat layer 21,
  cradio-v4-h layer 32 (chosen on the glacier validation glacier, not here);
- input: s1input mode "vv_vh_diff", channel statistics from the training split;
- head 1, linear probe: one linear layer on the 16 px patch features, patch
  label = majority vote of its valid pixels, balanced cross-entropy, AdamW
  lr 1e-3, batch 65,536 patches (the glacier probe's recipe), trained for at
  least 30 epochs AND at least ~1,000 optimiser steps -- the glacier probe's
  2.1 M patches gave ~1,000 steps in 30 epochs; small domains would otherwise
  get only a few dozen steps; pixel predictions = bilinear upsampling of the
  patch logits;
- head 2, small decoder: row 3 of the glacier study (GridDecoder with image
  skip = s1input.skip_image), 4,000 steps, batch 8, AdamW lr 1e-3 + OneCycle,
  balanced cross-entropy ignoring 255, horizontal flips, seeds 0, 1, 2, at most
  4,000 training chips (uniform sample, seed 0);
- nothing is selected on any split, so every non-train split (validation and
  tests) is scored in the same run; validation is reported as a check only;
- metrics per split: pixel confusion -> per-class IoU, mIoU, accuracy; for
  binary tasks also IoU, F1, precision and recall of class 1; per region
  (meta "region") pooled; a majority-class baseline from the training pixels.
- one no-data rule for every domain (added 2026-09-26, still before any
  number, because the converters differed): a pixel whose co- OR cross-pol
  value is missing (NaN) is ignored (label 255) in training and scoring;
- every result row carries the manifest's SAR units; units other than dB are
  reported, not converted (the third input channel is then not a log ratio);
- optional, off by default: `exclude_train_sharing` drops training chips that
  share a value of the given meta columns (e.g. "s2_scene") with any scored
  split, for scene-disjoint scoring.

Speed (2026-09-28, before any number; results unchanged): the pixel confusion is
counted per chip on the GPU right where the predictions are made
(`_chip_conf`); split and region totals are sums of those integer counts, so
they equal the earlier CPU `_confusion` exactly (measured on the PC: the CPU
count took about 16 min per encoder for snow's 3.5 billion test pixels).
`enc_cache` (a dict the caller keeps) loads each encoder once for all datasets
instead of once per dataset (1-2 min per load from the Drive model cache).

Part three, added without changing the
numbers of the modes and encoders above:
- encoders scored on their final output carry the layer label `models.LAST`:
  terrafm-b-s1, dinov3-aerial, and the students that `distil.score` registers
  with their checkpoints;
- modes "co_cross" (2 channels) and "aerial_rgbn" (4 channels) of s1input. An
  encoder with its own `input_stats` for the mode is fed with those instead of
  the training-split statistics (the decoder's skip image keeps the training-
  split statistics); `__info.json` records both, "stats" and "input_stats";
- the encoder is built on `device` (part two built it on the GPU, its default);
- `TrainSet` is part two's heads.TrainSet, copied unchanged; part two's pip
  check (ensure_packages, for kinds not used here) is dropped;
- the keyword `splits` of `run_domain` (2026-09-30, stage A scores the validation
  splits only). None = every non-train split, as before. A tuple
  keeps only the named splits, features are extracted for those and train alone, the
  run name stays the same, and `__info.json` records them as "scored_splits". A summary
  on disk counts as done only when it holds every asked split.

Part four (2026-10-04), a faster decoder step, off unless asked. `run_domain(fast_decoder=True)`
trains the decoder with the same recipe (GridDecoder, 4,000 steps, batch 8, AdamW lr 1e-3 with
OneCycle, balanced cross-entropy, flips, the seeds, bf16 autocast); only the kernels that run it
change. Without the argument every step is part two's, bit for bit (tested on the CPU against a
copy of the code before the change).
- Why. Part three on an A100-80GB, per student - alpine glaciers 21.7 min, 21.2 of them the three
  decoders (992 px tiles), snow 8.4 (6.5). A training step took 9, 11, 32 and 106 ms at 128, 256,
  512 and 992 px (batch 8). Arithmetic of the 992 px step, eager under CUDA bf16 autocast, every
  op reading its inputs and writing its outputs once - the convolutions store bfloat16, autocast
  casts every conv input to bfloat16 and every GroupNorm input to float32, so GroupNorm,
  LeakyReLU, the bilinear upsampling and the concatenation work on float32 maps of 16 to 64
  channels at full and half size. With the backward (GroupNorm's reads its input and gradient
  twice, max pooling keeps int64 indices, the upsampling's zero-fills and scatters) that is about
  85 GB per step for 7.9 M pixels (10,800 bytes per pixel, 54 GB of it at full size), 42 ms at the
  A100's 2.0 TB/s. The convolutions are 37,000 multiply-adds per pixel forward, about 1.75 TFLOP
  per step with the backward, 6 ms at the bf16 peak of 312 TFLOPS, 11 ms at half of it. So about
  50 ms of the 106 are the floor of these eager kernels; the rest is cuDNN's layout changes for
  NCHW bfloat16, the scattered adds of the upsampling backward and GroupNorm kernels below peak
  bandwidth. At 128 and 256 px the same bytes take 0.7 and 2.8 ms of the 9 and 11 measured, those
  steps are bound by kernel launches. torch.compile (Inductor) fuses the casts, GroupNorm,
  LeakyReLU, the concatenation and the loss into fewer passes and launches, which works on both
  limits; how much it gains is measured per shape, not assumed.
- How. `decoder_speed_check` times warm steps (forward, backward, AdamW) of eager and of
  torch.compile(dynamic=False) on a synthetic batch of the run's shapes (tile, feature width,
  classes, batch, the layout of TrainSet.batch), in interleaved rounds with the median of each
  (other runs may share the GPU), cudnn.benchmark on for both. It also takes one step of each
  from the same weights and batch, and one eager step in float32 (the A100 runs its convolutions
  in TF32, 10 bits against bfloat16's 7, so the gap is bfloat16's). Rule (`_decide`) - compiled
  only when its first call worked, its loss is within 2 x |eager - float32| + LOSS_FLOOR of
  eager's and its gradient (every parameter, one vector) within 2 x |eager - float32| +
  GRAD_FLOOR (relative), AND it takes at most (1 - MIN_GAIN) of eager's ms per step; otherwise
  eager. The tolerance is how far bfloat16 autocast itself moves the eager step from float32,
  twice (each path rounds at its own places, Inductor keeps float32 inside a fused kernel), plus a
  floor for a reference that comes out small by chance. Measured on the CPU (2026-10-04, 128 and
  256 px, 2 and 19 classes, CPU autocast also rounds GroupNorm) - bfloat16 against float32 moved
  the loss by 3e-6 to 8e-5 and the gradient by 0.7 to 3.4 percent; wrong graphs (no image skip,
  nearest upsampling, LeakyReLU slope 0.2, flipped features) moved the loss by 5e-4 to 3e-2 and
  the gradient by 5 to 11,600 percent, a GroupNorm eps of 1e-3 the gradient by 1.8 to 4.7
  percent. So the gradient carries the check (the loss of a fresh decoder on random labels stays
  near log k whatever the graph). `decoder_choice` keeps the result per tile, width, classes,
  batch and device in the process and prints one line with both ms; run_domain writes the path
  and its ms into __info.json "times" - decoder_path, decoder_step_ms and decoder_s<seed>_step_ms
  (measured in the run, every step with its data), decoder_check_eager_ms,
  decoder_check_compiled_ms, decoder_check_compile_s (the first compiled call) and
  decoder_check_why.
- Never stops a run. A compile that raises (no compiler, an error) gives eager with a printed
  line; a compiled step that raises in the run is done again eager and the rest of the seed stays
  eager (no gradient is applied before the step ends, so the math is unchanged).
- Left as it was. The eval pass `_decoder_conf` stays eager - its last batch has another size,
  and part three's numbers put the 4,000 training steps at about all of the decoder minutes (snow
  4,000 x 32 ms = 2.1 of 2.2 min per seed). channels_last is not set by hand - on CUDA, eager
  GroupNorm makes its input contiguous (ATen group_norm), so a channels_last eager decoder adds
  copies around every GroupNorm, and Inductor's layout optimisation
  (torch._inductor.config.layout_optimization, on by default) already gives the convolutions of
  a compiled graph channels_last where it pays. models/decoder.py is unchanged.
- Compile cache. `use_compile_cache(folder)` points TORCHINDUCTOR_CACHE_DIR (the FX graph and
  AOTAutograd caches, the Triton kernels bundled into them) and TRITON_CACHE_DIR at one folder;
  run_domain calls it with `compile_cache`, so every worker of score.score_parallel uses the same
  folder and a shape compiled once is read back by the processes after it.
  score.restore_compile_cache and score.save_compile_cache move the folder to and from Drive as
  one archive.

Outputs in res_dir, all prefixed dom_<dataset>__<encoder>__L<layer>__<mode>:
    __summary.csv                  one row per head / seed / split
    __<head>[__s<seed>]__regions.csv   per split and region
    __probe.pt, __decoder__s<seed>.pt  trained heads
    __info.json                    channel stats, chip counts, timings
"""

from __future__ import annotations

import contextlib
import copy
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..models import LAST
from .prepared import IGNORE, load_manifest, load_split

PATCH = 16
PROBE_STEPS = 1000               # optimiser steps of the glacier probe (2.1 M patches, 30 epochs)
PRE_REGISTERED_LAYERS = {"dinov3-l-sat": 21, "cradio-v4-h": 32,          # part two, unchanged
                         "terrafm-b-s1": LAST, "dinov3-aerial": LAST}     # part three, final output


# ------------------------------------------------------------------ metrics

def seg_metrics(conf: np.ndarray, names: list[str]) -> dict:
    """Pixel confusion (rows truth, cols prediction) -> IoU per class, mIoU, accuracy (+ binary extras)."""
    tp = np.diag(conf).astype(np.float64)
    fp, fn = conf.sum(0) - tp, conf.sum(1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    out = {f"IoU_{n}": float(v) for n, v in zip(names, iou)}
    out["mIoU"] = float(np.nanmean(iou)) if np.isfinite(iou).any() else np.nan
    out["accuracy"] = float(tp.sum() / max(conf.sum(), 1))
    out["n_pixels"] = int(conf.sum())
    if len(names) == 2:
        p = tp[1] / max(tp[1] + fp[1], 1)
        r = tp[1] / max(tp[1] + fn[1], 1)
        out.update(IoU_pos=float(iou[1]), F1_pos=float(2 * p * r / max(p + r, 1e-12)),
                   precision_pos=float(p), recall_pos=float(r))
    return out


def _confusion(pred: np.ndarray, truth: np.ndarray, k: int) -> np.ndarray:
    """Reference CPU count (rows truth, cols prediction, 255 left out); scoring uses _chip_conf."""
    ok = truth != IGNORE
    return np.bincount(truth[ok].astype(np.int64) * k + pred[ok].astype(np.int64),
                       minlength=k * k).reshape(k, k)


def _chip_conf(pred, truth, k: int) -> np.ndarray:
    """(B, H, W) predicted and true class ids (torch tensors, one device) -> (B, k, k) int64 confusion per chip.

    Rows truth, cols prediction, label 255 left out; summed over chips it equals _confusion()."""
    import torch

    b = pred.shape[0]
    chip = torch.arange(b, device=pred.device).view(b, 1, 1).expand_as(truth)
    ok = truth != IGNORE
    code = (chip * (k * k) + truth.long() * k + pred.long())[ok]
    return torch.bincount(code, minlength=b * k * k).view(b, k, k).cpu().numpy()


def _labels_on(labels: np.ndarray, i: int, batch: int, device: str):
    import torch

    return torch.from_numpy(np.ascontiguousarray(labels[i:i + batch])).to(device)


def _majority_conf(labels: np.ndarray, k: int, maj: int, device: str, batch: int = 64) -> np.ndarray:
    """Per-chip confusion of the constant prediction `maj`."""
    import torch

    out = []
    for i in range(0, len(labels), batch):
        truth = _labels_on(labels, i, batch, device)
        out.append(_chip_conf(torch.full_like(truth, maj), truth, k))
    return np.concatenate(out)


# ------------------------------------------------------------------ features

def _encode_split(enc, images, stats: dict, mode: str, layer, batch: int) -> np.ndarray:
    """(N, 2, H, W) SAR or (N, 4, H, W) aerial -> (N, g, g, D) fp16 features at `layer`."""
    from .s1input import encoder_input

    out = []
    for i in range(0, len(images), batch):
        out.append(enc.encode(encoder_input(images[i:i + batch], stats, mode), batch=batch, layer=layer))
    return np.concatenate(out)


def _patch_targets(labels: np.ndarray, k: int, patch: int = PATCH) -> tuple[np.ndarray, np.ndarray]:
    from ..data.targets import patch_labels

    labs, keeps = [], []
    for lab in labels:
        lab = np.asarray(lab)
        pl, keep = patch_labels(np.where(lab == IGNORE, 0, lab), lab != IGNORE, patch=patch, n_classes=k)
        labs.append(pl)
        keeps.append(keep)
    return np.stack(labs), np.stack(keeps)


# ------------------------------------------------------------------ heads

class TrainSet:
    """Training tiles as tensors, built once and reused (part two's heads.TrainSet, unchanged).

    Speed-up 2026-09-25: the per-step numpy stack + fp32 conversion + host->GPU
    copy of 8 tiles took ~40-50 ms of a ~70 ms step (measured on a desktop CPU),
    while the decoder itself needs ~2-5 ms on an A100. Here the tiles live on the
    GPU when they fit (fp16 features, uint8 image and labels), else in host
    memory with only the fp16 batch copied per step. `batch(idx)` returns the
    same values as the old per-step path (fp16 -> fp32 is exact).
    """

    def __init__(self, feats: list, raws: list, labels: list, device: str,
                 extra: list | None = None, reserve_gb: float = 20.0):
        import torch

        n, (g1, g2, d), t = len(feats), feats[0].shape, raws[0].shape[-1]
        c = 1 if raws[0].ndim == 2 else raws[0].shape[0]     # image skip channels, 1 = part two's grey tile
        need = n * (g1 * g2 * d * 2 + (c + 1) * t * t)
        fits = device == "cuda" and torch.cuda.mem_get_info()[0] > need + reserve_gb * 1e9
        self.device, self.store = device, (device if device != "cuda" or fits else "cpu")
        self.feats = torch.empty((n, g1, g2, d), dtype=torch.float16, device=self.store)
        self.raw = torch.empty((n, t, t) if c == 1 else (n, c, t, t), dtype=torch.uint8, device=self.store)
        self.lab = torch.empty((n, t, t), dtype=torch.uint8, device=self.store)
        for i in range(n):
            self.feats[i] = torch.from_numpy(np.ascontiguousarray(feats[i], dtype=np.float16))
            self.raw[i] = torch.from_numpy(np.ascontiguousarray(raws[i]))
            self.lab[i] = torch.from_numpy(np.ascontiguousarray(labels[i]))
        self.extra = (None if extra is None else
                      torch.from_numpy(np.stack(extra).astype(np.float32)).to(device))
        print(f"    training tiles held on {self.store} ({need / 1e9:.1f} GB)")

    def batch(self, idx):
        import torch

        ix = torch.as_tensor(np.asarray(idx), device=self.store)
        f = self.feats.index_select(0, ix).to(self.device, non_blocking=True).float().permute(0, 3, 1, 2)
        raw = self.raw.index_select(0, ix).to(self.device, non_blocking=True)
        raw = raw[:, None] if raw.ndim == 3 else raw
        y = self.lab.index_select(0, ix).to(self.device, non_blocking=True).long()
        ex = None if self.extra is None else self.extra.index_select(0, ix.to(self.extra.device))
        return f, raw, y, ex


# ------------------------------------------------------------------ the fast decoder path (part four)

# The rule of `_decide` (module docstring, part four). Relative gaps of the compiled step against the eager one.
LOSS_FLOOR = 2e-4        # added to 2 x |eager - float32| of the loss; below the 5e-4 of the mildest wrong graph
GRAD_FLOOR = 1e-2        # added to 2 x |eager - float32| of the gradient; bfloat16 alone moved it 0.7 to 3.4 percent
MIN_GAIN = 0.05          # compiled must take at most 95 percent of eager's ms per step - a smaller gain cannot be
                         # told from the noise of a GPU shared with other runs, and eager is part two's path
CHECK_WARM, CHECK_STEPS, CHECK_ROUNDS = 3, 8, 5   # warm steps, then rounds of eager and compiled steps in turn
RECOMPILE_LIMIT = 64     # graphs kept per compiled function, one per shape (dynamo keeps 8 by default)
_CHOICES: dict = {}      # (tile, width, classes, batch, device) -> the result of decoder_speed_check, this process


def _step_loss(head, loss_fn, f, raw, y, device: str, bf16: bool = True):
    """The loss of one training step of part two's recipe - called as it is on the eager path, compiled on the fast one."""
    import torch

    with torch.autocast(device, dtype=torch.bfloat16, enabled=bf16 and device == "cuda"):
        return loss_fn(head(f, raw.float() / 255.0).float(), y)


def _compile(fn):
    """torch.compile of the training step, shapes fixed per dataset (dynamic=False). A test puts a stand-in here."""
    import torch

    config = torch._dynamo.config
    for name in ("recompile_limit", "cache_size_limit"):        # the newer name first
        if hasattr(config, name):
            setattr(config, name, max(int(getattr(config, name)), RECOMPILE_LIMIT))
            break
    return torch.compile(fn, dynamic=False)


def _plain(error, limit: int = 160) -> str:
    """An exception or text as one short line without a colon (no colon in what is printed)."""
    text = f"{type(error).__name__} {error}" if isinstance(error, BaseException) else str(error)
    return " ".join(text.replace(":", " -").split())[:limit]


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize()


@contextlib.contextmanager
def _cudnn_benchmark(on: bool):
    """cudnn.benchmark on inside the block when `on`, the earlier value back after it; nothing at all otherwise."""
    if not on:
        yield
        return
    import torch

    before = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = before


def use_compile_cache(folder: str | Path) -> Path:
    """Point the compile caches of this process at `folder`, before its first compile.

    TORCHINDUCTOR_CACHE_DIR = folder (Inductor keeps the FX graph cache and the AOTAutograd cache
    under it, the Triton kernels bundled into the graph entries), TRITON_CACHE_DIR = folder/triton,
    and both local caches switched on. Processes started later inherit the environment; each
    worker of score.score_parallel calls this itself through run_domain's `compile_cache`.
    """
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    changed = os.environ.get("TORCHINDUCTOR_CACHE_DIR") != str(folder)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(folder)
    os.environ["TRITON_CACHE_DIR"] = str(folder / "triton")
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    try:
        import torch._functorch.config as functorch_config
        import torch._inductor.config as inductor_config

        inductor_config.fx_graph_cache = True
        functorch_config.enable_autograd_cache = True
        if changed:
            from torch._inductor.utils import clear_caches

            clear_caches()                            # paths an earlier compile of this process kept
    except Exception as e:                            # an older or a CPU-only torch - the variables still hold
        print(f"  compile cache settings not all set ({_plain(e)})")
    return folder


def _one_step(step, head, loss_fn, f, raw, y, device: str, **kw):
    """(loss, the gradient of every parameter as one float32 vector) of one step from the head's weights, no update."""
    import torch

    head.train()
    head.zero_grad(set_to_none=True)
    loss = step(head, loss_fn, f, raw, y, device, **kw)
    loss.backward()
    grad = torch.cat([p.grad.detach().flatten().float() for p in head.parameters()])
    head.zero_grad(set_to_none=True)
    return float(loss.detach()), grad


def _timed(step, head, opt, loss_fn, f, raw, y, device: str, n: int) -> float:
    """Milliseconds per step of `n` training steps (forward, backward, AdamW), the device synchronised around them."""
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n):
        head.train()
        loss = step(head, loss_fn, f, raw, y, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    _sync(device)
    return (time.perf_counter() - t0) / max(n, 1) * 1000


def _decide(r: dict) -> tuple[str, str]:
    """(path, why) from a decoder_speed_check result, by the rule of the module docstring (part four)."""
    if r.get("error"):
        return "eager", "compile failed"
    loss_ok = r.get("loss_gap") is not None and r["loss_gap"] <= r["loss_tolerance"]
    grad_ok = r.get("grad_gap") is not None and r["grad_gap"] <= r["grad_tolerance"]
    if not (loss_ok and grad_ok):
        return "eager", "compiled disagrees with eager"
    if not (r.get("compiled_ms") is not None and r.get("eager_ms") is not None
            and r["compiled_ms"] <= (1 - MIN_GAIN) * r["eager_ms"]):
        return "eager", f"compiled is not {MIN_GAIN * 100:.0f} percent faster"
    return "compiled", "compiled is faster and agrees"


def decoder_speed_check(tile: int, n_feat: int, device: str, n_classes: int = 2, batch: int = 8,
                        warm: int = CHECK_WARM, steps: int = CHECK_STEPS, rounds: int = CHECK_ROUNDS) -> dict:
    """Milliseconds per decoder training step, eager and compiled, on a synthetic batch of the real shapes.

    tile, n_feat, n_classes, batch   the run's chip size, feature width, classes and decoder batch
    warm, steps, rounds              warm steps of each path, then `rounds` rounds of `steps` eager
                                     steps and `steps` compiled steps in turn; the median per path
    Returns {"tile", "n_feat", "n_classes", "batch", "device", "eager_ms", "compiled_ms" (None when
    the compile failed), "compile_s" (the first compiled call with its backward), "loss_gap",
    "loss_tolerance", "grad_gap", "grad_tolerance" (relative, the rule of the module docstring),
    "error" (None or one line), "path" ("compiled" or "eager"), "why"}. The global torch random
    state is left as it was; nothing of the run is touched.
    """
    import torch
    import torch.nn as nn

    from ..models.decoder import GridDecoder

    g = int(tile) // PATCH
    gen = torch.Generator().manual_seed(0)
    # the layouts of TrainSet.batch - features stored (B, g, g, D) and permuted, the image with a unit channel axis
    f = torch.randn(batch, g, g, n_feat, generator=gen).to(device).permute(0, 3, 1, 2)
    raw = torch.randint(0, 256, (batch, tile, tile), generator=gen, dtype=torch.uint8).to(device)[:, None]
    lab = torch.randint(0, n_classes, (batch, tile, tile), generator=gen, dtype=torch.uint8)
    lab[:, :max(1, tile // 16)] = IGNORE                               # some no-data rows, as real labels have
    y = lab.to(device).long()
    counts = torch.bincount(lab[lab != IGNORE].long(), minlength=n_classes)[:n_classes].double()
    w = (counts.sum() / (n_classes * counts.clamp(min=1))).float().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        eager_head = GridDecoder(n_feat, n_classes=n_classes).to(device)
    fast_head = copy.deepcopy(eager_head)
    out = {"tile": int(tile), "n_feat": int(n_feat), "n_classes": int(n_classes), "batch": int(batch),
           "device": str(device), "eager_ms": None, "compiled_ms": None, "compile_s": None, "loss_gap": None,
           "loss_tolerance": None, "grad_gap": None, "grad_tolerance": None, "error": None}
    batch_args = (loss_fn, f, raw, y, device)

    def rel(a, b) -> float:
        if isinstance(a, float):
            return abs(a - b) / max(abs(b), 1e-12)
        return float((a - b).norm() / b.norm().clamp_min(1e-12))

    try:
        with _cudnn_benchmark(True):
            loss_e, grad_e = _one_step(_step_loss, eager_head, *batch_args)
            loss_32, grad_32 = _one_step(_step_loss, eager_head, *batch_args, bf16=False)
            compiled = None
            try:
                compiled = _compile(_step_loss)
                t0 = time.perf_counter()
                loss_c, grad_c = _one_step(compiled, fast_head, *batch_args)
                _sync(device)
                out["compile_s"] = time.perf_counter() - t0
                out.update(loss_gap=rel(loss_c, loss_e), loss_tolerance=2 * rel(loss_32, loss_e) + LOSS_FLOOR,
                           grad_gap=rel(grad_c, grad_e), grad_tolerance=2 * rel(grad_32, grad_e) + GRAD_FLOOR)
            except Exception as e:
                compiled, out["error"] = None, _plain(e)
            opt_e = torch.optim.AdamW(eager_head.parameters(), lr=1e-3, weight_decay=1e-4)
            opt_c = torch.optim.AdamW(fast_head.parameters(), lr=1e-3, weight_decay=1e-4)
            _timed(_step_loss, eager_head, opt_e, *batch_args, warm)
            eager_ms, compiled_ms = [], []
            try:
                if compiled is not None:
                    _timed(compiled, fast_head, opt_c, *batch_args, warm)
                for _ in range(rounds):
                    eager_ms.append(_timed(_step_loss, eager_head, opt_e, *batch_args, steps))
                    if compiled is not None:
                        compiled_ms.append(_timed(compiled, fast_head, opt_c, *batch_args, steps))
            except Exception as e:                           # a compiled step that fails after its first call
                compiled, compiled_ms, out["error"] = None, [], _plain(e)
                eager_ms = eager_ms or [_timed(_step_loss, eager_head, opt_e, *batch_args, steps)]
            out["eager_ms"] = float(np.median(eager_ms))
            out["compiled_ms"] = float(np.median(compiled_ms)) if compiled_ms else None
    finally:
        del eager_head, fast_head, f, raw, y
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()                         # what the check cached goes back to the other runs
    out["path"], out["why"] = _decide(out)
    return out


def _choice_line(r: dict) -> str:
    """The one printed line of a choice, both ms, no colon."""
    def ms(name):
        return "not measured" if r.get(f"{name}_ms") is None else f"{r[f'{name}_ms']:.1f} ms"

    if r.get("why") == "the speed check failed":
        compiled = f"the check failed ({r['error']})"
    elif r.get("error") and r.get("compiled_ms") is None:
        compiled = f"compiled failed ({r['error']})"
    else:
        compiled = f"compiled {ms('compiled')}" + ("" if r.get("compile_s") is None else
                                                   f" (first call {r['compile_s']:.0f} s)")
    agree = ("" if r.get("loss_gap") is None else
             f", loss gap {r['loss_gap']:.1e} (tolerance {r['loss_tolerance']:.1e}), gradient gap "
             f"{r['grad_gap']:.1e} (tolerance {r['grad_tolerance']:.1e})")
    return (f"  decoder step {r['tile']} px, {r['n_feat']} features, {r['n_classes']} classes, batch {r['batch']} - "
            f"eager {ms('eager')}, {compiled}{agree} - {r['path']}, {r['why']}")


def decoder_choice(tile: int, n_feat: int, n_classes: int, device: str, batch: int = 8) -> dict:
    """The path of the decoder's training step for these shapes - `decoder_speed_check` once per process, then kept.

    Prints one line per new choice. A check that fails itself (out of memory, anything) gives eager,
    printed, and is kept too - a run never stops because of the speed path.
    """
    key = (int(tile), int(n_feat), int(n_classes), int(batch), str(device))
    if key not in _CHOICES:
        try:
            r = decoder_speed_check(tile, n_feat, device, n_classes=n_classes, batch=batch)
        except Exception as e:
            r = {"tile": key[0], "n_feat": key[1], "n_classes": key[2], "batch": key[3], "device": key[4],
                 "eager_ms": None, "compiled_ms": None, "compile_s": None, "loss_gap": None, "error": _plain(e),
                 "path": "eager", "why": "the speed check failed"}
        _CHOICES[key] = r
        print(_choice_line(r))
    return _CHOICES[key]


def _train_decoder(head, feats: np.ndarray, skip_u8: np.ndarray, labels: np.ndarray, k: int, device: str,
                   steps: int, batch: int, lr: float, seed: int, log_every: int = 1000, path: str | None = None,
                   report: dict | None = None):
    """Row 3's recipe (part two's heads._train_head) with generic class ids.

    path    None = part two's step, unchanged. "eager" = the same step with cudnn.benchmark on;
            "compiled" = the step through `_compile` with cudnn.benchmark on - a compile that raises
            gives eager, a compiled step that raises is done again eager and the rest stays eager,
            each with a printed line (part four, the module docstring)
    report  a dict that gets "path" (the path the steps ran, "compiled, then eager from step N"
            after a failure) and "step_ms" (wall time of the training loop per step, data included)
    """
    import torch
    import torch.nn as nn

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
    step, used = _step_loss, "eager"
    if path == "compiled":
        try:
            step, used = _compile(_step_loss), "compiled"
        except Exception as e:
            print(f"    torch.compile failed ({_plain(e)}), the decoder trains eager")
    with _cudnn_benchmark(path is not None):
        t0, running, n = time.perf_counter(), torch.zeros((), device=device), 0
        for it in range(1, steps + 1):
            head.train()
            idx = rng.choice(len(feats), batch, replace=len(feats) < batch)
            f, raw, y, _ = ts.batch(idx)
            if rng.random() < 0.5:
                f, raw, y = f.flip(-1), raw.flip(-1), y.flip(-1)
            if step is not _step_loss:
                try:
                    loss = step(head, loss_fn, f, raw, y, device)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                except Exception as e:                  # nothing applied yet, the same step again eager
                    print(f"    compiled step {it} failed ({_plain(e)}), this step and the rest run eager")
                    step, used = _step_loss, f"compiled, then eager from step {it}"
            if step is _step_loss:
                loss = _step_loss(head, loss_fn, f, raw, y, device)
                opt.zero_grad(set_to_none=True)
                loss.backward()
            opt.step()
            sched.step()
            running += loss.detach()
            n += 1
            if it % log_every == 0 or it == steps:
                print(f"    step {it:>5}  loss {float(running) / n:.4f}  {(time.perf_counter() - t0) / 60:.1f} min")
                running, n = torch.zeros((), device=device), 0
        if report is not None:                          # the last print synchronised the device
            report.update(path=used, step_ms=(time.perf_counter() - t0) / max(steps, 1) * 1000)
    return head


def _decoder_conf(head, feats: np.ndarray, skip_u8: np.ndarray, labels: np.ndarray, k: int, device: str,
                  batch: int = 32) -> np.ndarray:
    """Per-chip pixel confusion (N, k, k) of the decoder's argmax, counted on `device`."""
    import torch

    head.eval()
    out = []
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for i in range(0, len(feats), batch):
            f = torch.from_numpy(feats[i:i + batch]).to(device).float().permute(0, 3, 1, 2)
            img = torch.from_numpy(skip_u8[i:i + batch])
            img = (img[:, None] if img.ndim == 3 else img).to(device).float() / 255.0
            pred = head(f, img).float().argmax(1)
            out.append(_chip_conf(pred, _labels_on(labels, i, batch, device), k))
    return np.concatenate(out)


def _probe_conf(probe, feats: np.ndarray, labels: np.ndarray, k: int, size: int, device: str,
                batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Patch argmax (N, g, g) and the per-chip pixel confusion (N, k, k) of the bilinearly upsampled logits."""
    import torch
    import torch.nn.functional as F

    W = probe.weight.detach().float().to(device)
    b = probe.bias.detach().float().to(device)
    pa, cc = [], []
    with torch.inference_mode():
        for i in range(0, len(feats), batch):
            f = torch.from_numpy(feats[i:i + batch]).to(device).float()           # (B, g, g, D)
            lg = (f @ W.T + b).permute(0, 3, 1, 2)                                 # (B, K, g, g)
            pa.append(lg.argmax(1).to(torch.uint8).cpu().numpy())
            up = F.interpolate(lg, size=(size, size), mode="bilinear", align_corners=False)
            cc.append(_chip_conf(up.argmax(1), _labels_on(labels, i, batch, device), k))
    return np.concatenate(pa), np.concatenate(cc)


# ------------------------------------------------------------------ driver

def run_domain(prepared_root: str | Path, dataset: str, encoder: str, res_dir: str | Path,
               token: str | None, layer: int | str | None = None, seeds=(0, 1, 2), mode: str = "vv_vh_diff",
               max_train_chips: int = 4000, probe_max_patches: int = 2_000_000, steps: int = 4000,
               batch: int = 8, lr: float = 1e-3, device: str = "cuda", force: bool = False,
               exclude_train_sharing: tuple[str, ...] = (), enc_cache: dict | None = None,
               splits: tuple[str, ...] | None = None, fast_decoder: bool = False,
               compile_cache: str | Path | None = None, decoder_path: str | None = None,
               upsampler=None, feature_cache: dict | None = None,
               extra_skip: tuple[str, dict] | None = None) -> pd.DataFrame:
    """Everything for one prepared dataset and one encoder; see the module docstring.

    enc_cache: a dict kept by the caller; the encoder is loaded into it on first use and reused by later
    calls (the caller frees it). None loads the encoder for this call only and frees it after the features.
    splits: the splits to score. None = every split of the manifest but "train" (part two). A tuple keeps
    only the named ones (they must be scored splits of the manifest, "train" is never one); the run name
    does not change and __info.json records them as "scored_splits".
    fast_decoder: False = part two's decoder step, unchanged. True = the measured choice of part four
    (`decoder_choice`, eager or compiled, cudnn.benchmark on), same recipe and math; __info.json "times"
    then also holds decoder_path, decoder_step_ms, decoder_s<seed>_step_ms and decoder_check_*.
    compile_cache: with fast_decoder, the folder of the compile caches (`use_compile_cache`); None leaves
    Inductor's default folder. Without fast_decoder it is not used.
    decoder_path: None = as fast_decoder says. "eager" or "compiled" = that path for every seed, with
    fast_decoder's settings otherwise (the check of the compiled step against part two's, notebook 03a);
    the speed is still measured, and __info.json "times" says the path was set and what was measured.
    upsampler: None = part two's 16 px grid. One of `domains.upsample` (part four, the cheap test of the 8 px
    grid) = the features of every split made `upsampler.factor` times finer right after
    extraction, the probe on patches that much smaller and the decoder entering there (GridDecoder entry);
    the run name ends in __up<grid px>_<upsampler.name> and __info.json records "upsampler" and "grid_px".
    feature_cache: a dict kept by the caller; the extracted features (before any upsampler) of each split are
    kept in it and reused by later calls on the same dataset, encoder, layer and mode. None = not kept.
    extra_skip: None = part two's grey image skip. (name, {split - (N, H, W) uint8}) (part four, section
    15) = a second channel of the decoder's image skip for every split read, in the chip order of the
    prepared split; the run name ends in __skip_<name> and __info.json records "extra_skip".
    """
    if decoder_path not in (None, "eager", "compiled"):
        raise ValueError(f"decoder_path must be None, 'eager' or 'compiled', got {decoder_path!r}")
    import torch

    from ..models.decoder import GridDecoder
    from ..models.encoders import ENCODERS, FrozenEncoder
    from ..probe import train_linear_probe
    from .s1input import channel_stats, skip_image

    layer = PRE_REGISTERED_LAYERS[encoder] if layer is None else layer
    man = load_manifest(prepared_root, dataset)
    classes = {int(k): v for k, v in man["classes"].items()}
    k = len(classes)
    if sorted(classes) != list(range(k)):
        raise ValueError(f"class ids must be 0..{k - 1}, got {sorted(classes)}")
    names = [classes[i] for i in range(k)]
    all_splits = list(man["splits"])
    if "train" not in all_splits:
        raise ValueError(f"{dataset}: no train split in {all_splits}")
    if splits is None:
        evals = [s for s in all_splits if s != "train"]
    else:
        evals = [str(s) for s in splits]
        bad = [s for s in evals if s not in all_splits or s == "train"]
        if bad or not evals:
            raise ValueError(f"{dataset}: splits {bad or list(splits)} are not scored splits of "
                             f"{all_splits} (train is never scored)")
    splits = [s for s in all_splits if s == "train" or s in evals]     # manifest order, as before
    run = f"dom_{dataset}__{encoder}__L{layer}__{mode}" + ("__disjoint_" + "_".join(exclude_train_sharing)
                                                            if exclude_train_sharing else "")
    patch = PATCH if upsampler is None else PATCH // int(upsampler.factor)
    if upsampler is not None:
        run += f"__up{patch}_{upsampler.name}"
    if extra_skip is not None:
        if exclude_train_sharing:
            raise ValueError("extra_skip is in the order of the whole split, it cannot go with exclude_train_sharing")
        run += f"__skip_{extra_skip[0]}"
    units = str(man.get("units", "dB"))
    if units != "dB":
        print(f"  NOTE: SAR units are {units!r}, not dB; results are reported with that label")
    res_dir = Path(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    spath = res_dir / f"{run}__summary.csv"
    if spath.exists() and not force:
        s = pd.read_csv(spath)
        want = {("probe", -1)} | {("decoder", sd) for sd in seeds}
        if want <= set(zip(s["head"], s["seed"])) and set(evals) <= set(s["split"]):
            print(f"skip {run}: summary already has probe + decoder seeds {list(seeds)} on {evals}")
            return s
    print(f"\n===== {run}: splits {man['splits']} | classes {names} | chip {man['chip_size']} px =====")
    if len(evals) < len(all_splits) - 1:
        print(f"  scored splits {evals}, the other splits of the manifest are not touched")
    t_all = time.perf_counter()
    data = {s: load_split(prepared_root, dataset, s) for s in splits}
    if extra_skip is not None:                          # part four, checked before anything is computed
        for s in splits:
            got = extra_skip[1].get(s)
            want = (len(data[s]["images"]), *data[s]["images"].shape[-2:])
            if got is None or tuple(np.shape(got)) != want or np.asarray(got).dtype != np.uint8:
                raise ValueError(f"extra_skip {extra_skip[0]} for {s} must be uint8 {want}, got "
                                 f"{None if got is None else (tuple(np.shape(got)), np.asarray(got).dtype)}")
    if exclude_train_sharing:
        shared = set()
        for s in evals:
            for c in exclude_train_sharing:
                shared |= {(c, v) for v in data[s]["meta"][c].astype(str) if v != ""}
        tm = data["train"]["meta"]
        keep = ~np.array([any((c, str(tm[c].iloc[i])) in shared for c in exclude_train_sharing)
                          for i in range(len(tm))])
        print(f"  scene-disjoint: {int((~keep).sum())} of {len(tm)} training chips share "
              f"{exclude_train_sharing} with a scored split -> dropped")
        idx = np.flatnonzero(keep)
        data["train"] = {"images": data["train"]["images"][idx], "labels": data["train"]["labels"][idx],
                         "meta": tm.iloc[idx].reset_index(drop=True)}
    stats = channel_stats(data["train"]["images"])
    size = int(man["chip_size"])

    # ---- frozen features, once per split
    spec = ENCODERS[encoder]
    feats, times = {}, {}
    enc = None if enc_cache is None else enc_cache.get(encoder)
    if enc is None:
        t0 = time.perf_counter()
        enc = FrozenEncoder(spec, token=token, device=device)
        if enc_cache is not None:
            enc_cache[encoder] = enc
        times["encoder_load_min"] = (time.perf_counter() - t0) / 60
        print(f"  encoder loaded in {times['encoder_load_min']:.1f} min")
    else:
        print("  encoder reused (already loaded)")
    # part three: an encoder with statistics of its own for this mode is fed with those
    in_stats = getattr(enc, "input_stats", {}).get(mode, stats)
    if in_stats is not stats:
        print(f"  encoder input standardised with the encoder's own statistics for mode {mode!r}")
    ebatch = int(min(512, spec.batch * max(1, (512 // size) ** 2)))
    try:
        for s in splits:
            key = (dataset, encoder, str(layer), mode, s, tuple(exclude_train_sharing))
            if feature_cache is not None and key in feature_cache:
                feats[s] = feature_cache[key]
                print(f"  features {s}: {feats[s].shape} reused")
                continue
            t0 = time.perf_counter()
            feats[s] = _encode_split(enc, data[s]["images"], in_stats, mode, layer, ebatch)
            times[f"extract_{s}_min"] = (time.perf_counter() - t0) / 60
            print(f"  features {s}: {feats[s].shape} in {times[f'extract_{s}_min']:.1f} min")
            if feature_cache is not None:
                feature_cache[key] = feats[s]
    finally:
        del enc                                         # with enc_cache the caller's dict keeps it
        if enc_cache is None:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    if upsampler is not None:                           # part four, the 8 px grid
        for s in splits:
            t0 = time.perf_counter()
            feats[s] = upsampler(feats[s], data[s]["images"], stats)
            times[f"upsample_{s}_min"] = (time.perf_counter() - t0) / 60
            print(f"  {upsampler.name} {s}: {feats[s].shape} in {times[f'upsample_{s}_min']:.1f} min")
    D = feats["train"].shape[-1]
    labels = {}
    for s in splits:                      # one no-data rule: either channel missing -> ignore
        lab = np.array(data[s]["labels"])
        n_before = int((lab != IGNORE).sum())
        for i in range(0, len(lab), 256):
            miss = np.isnan(np.asarray(data[s]["images"][i:i + 256], np.float32)).any(axis=1)
            lab[i:i + 256][miss] = IGNORE
        labels[s] = lab
        print(f"  {s}: {n_before - int((lab != IGNORE).sum()):,} labelled pixels ignored for missing SAR")
    region = {s: data[s]["meta"]["region"].astype(str).to_numpy() for s in splits}

    rows, region_rows = [], []

    def score(head_name: str, seed: int, s: str, cc: np.ndarray) -> None:
        """cc is the (N, k, k) pixel confusion per chip of split s; split and region totals are its sums."""
        rows.append({"run": run, "dataset": dataset, "encoder": encoder, "layer": layer, "mode": mode,
                     "units": units, "head": head_name, "seed": seed, "split": s, **seg_metrics(cc.sum(0), names)})
        for r in np.unique(region[s]):
            m = region[s] == r
            region_rows.append({"head": head_name, "seed": seed, "split": s, "region": r, "chips": int(m.sum()),
                                **seg_metrics(cc[m].sum(0), names)})

    # ---- majority baseline (most frequent training pixel class everywhere)
    tr_counts = np.bincount(labels["train"][labels["train"] != IGNORE].ravel(), minlength=k)[:k]
    maj = int(tr_counts.argmax())
    for s in evals:
        score("majority", -1, s, _majority_conf(labels[s], k, maj, device))

    # ---- head 1: linear probe on patch features
    t0 = time.perf_counter()
    plab, pkeep = _patch_targets(labels["train"], k, patch)
    # The kept patches in the order of the boolean mask, the sample drawn from them as before, and only the
    # sampled ones copied (part four, 2026-10-04) - the same X and y as feats["train"][pkeep][sel], without
    # first copying every kept patch (31 GB for the lakes at 8 px)
    kept = np.flatnonzero(pkeep)
    if len(kept) > probe_max_patches:
        kept = kept[np.sort(np.random.default_rng(0).choice(len(kept), probe_max_patches, replace=False))]
    X, y = feats["train"].reshape(-1, feats["train"].shape[-1])[kept], plab.reshape(-1)[kept]
    per_epoch = -(-len(y) // 65536)
    epochs = max(30, -(-PROBE_STEPS // per_epoch))
    probe = train_linear_probe(np.ascontiguousarray(X), y.astype(np.int64), balanced=True, n_classes=k,
                               epochs=epochs, lr=1e-3, batch=65536, device=device, verbose=False)
    torch.save(probe.state_dict(), res_dir / f"{run}__probe.pt")
    for s in evals:
        pa, cc = _probe_conf(probe, feats[s], labels[s], k, size, device)
        score("probe", -1, s, cc)
        el, ek = _patch_targets(labels[s], k, patch)
        pconf = np.bincount(el[ek].astype(np.int64) * k + pa[ek].astype(np.int64), minlength=k * k).reshape(k, k)
        rows[-1]["patch_mIoU"] = seg_metrics(pconf, names)["mIoU"]
    times["probe_min"] = (time.perf_counter() - t0) / 60
    print(f"  probe: {len(y):,} patches, {epochs} epochs x {per_epoch} steps, {times['probe_min']:.1f} min")
    del X, y

    # ---- head 2: small decoder, several seeds, same capped uniform chip sample
    n_tr = len(feats["train"])
    pick = (np.sort(np.random.default_rng(0).choice(n_tr, max_train_chips, replace=False))
            if n_tr > max_train_chips else np.arange(n_tr))
    to_u8 = lambda im: np.round(skip_image(im, stats)[:, 0] * 255).astype(np.uint8)  # noqa: E731
    skip = {s: np.concatenate([to_u8(data[s]["images"][i:i + 256]) for i in range(0, len(data[s]["images"]), 256)])
            for s in splits}
    if extra_skip is not None:                          # part four, the second channel of the image skip
        skip = {s: np.stack([skip[s], np.asarray(extra_skip[1][s])], 1) for s in splits}
    choice, used = None, {}
    if fast_decoder or decoder_path is not None:        # part four, the measured choice of the decoder step
        if compile_cache is not None:
            use_compile_cache(compile_cache)
        choice = decoder_choice(size, D, k, device, batch)
        if decoder_path is not None:                    # set by the caller, the measured choice kept as it was
            choice = {**choice, "path": decoder_path,
                      "why": f"set to {decoder_path} by the caller, the measured choice was {choice['path']}"}
            print(f"  decoder path {choice['why']}")
    for sd in seeds:
        t0 = time.perf_counter()
        torch.manual_seed(sd)
        head_kw = ({} if upsampler is None else {"entry": patch}) | ({} if extra_skip is None else {"in_ch": 2})
        head = GridDecoder(D, n_classes=k, **head_kw)
        report = {}
        head = _train_decoder(head, feats["train"][pick], skip["train"][pick], labels["train"][pick], k,
                              device, steps, batch, lr, sd, path=None if choice is None else choice["path"],
                              report=report)
        torch.save(head.state_dict(), res_dir / f"{run}__decoder__s{sd}.pt")
        for s in evals:
            score("decoder", sd, s, _decoder_conf(head, feats[s], skip[s], labels[s], k, device))
        times[f"decoder_s{sd}_min"] = (time.perf_counter() - t0) / 60
        print(f"  decoder seed {sd}: {times[f'decoder_s{sd}_min']:.1f} min")
        if choice is not None:
            used[sd] = report["path"]
            times[f"decoder_s{sd}_step_ms"] = report["step_ms"]
            print(f"  decoder seed {sd} - {report['path']} path, {report['step_ms']:.1f} ms per training step")
        del head
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    if choice is not None:
        paths = sorted(set(used.values()))
        times.update({
            "decoder_path": paths[0] if len(paths) == 1 else ", ".join(f"s{sd} {p}" for sd, p in used.items()),
            "decoder_step_ms": float(np.mean([times[f"decoder_s{sd}_step_ms"] for sd in seeds])) if seeds else None,
            "decoder_check_eager_ms": choice.get("eager_ms"), "decoder_check_compiled_ms": choice.get("compiled_ms"),
            "decoder_check_compile_s": choice.get("compile_s"), "decoder_check_why": choice.get("why")})

    out = pd.DataFrame(rows)
    out.to_csv(spath, index=False)
    reg = pd.DataFrame(region_rows)
    for (h, sd), g in reg.groupby(["head", "seed"]):
        g.to_csv(res_dir / f"{run}__{h}{'' if sd < 0 else f'__s{sd}'}__regions.csv", index=False)
    times["total_min"] = (time.perf_counter() - t_all) / 60
    info = {"stats": stats, "input_stats": in_stats, "splits": man["splits"], "scored_splits": evals,
            "train_chips_decoder": int(len(pick)),
            "probe_patches": int(min(probe_max_patches, int(pkeep.sum()))), "classes": names,
            "chip_size": size, "times": times, "encoder_batch": ebatch}
    if upsampler is not None:
        info.update(upsampler=upsampler.name, grid_px=patch)
    if extra_skip is not None:
        info.update(extra_skip=extra_skip[0])
    (res_dir / f"{run}__info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    cols = ["head", "seed", "split", "mIoU", "accuracy"] + (["IoU_pos", "F1_pos"] if k == 2 else [])
    print(out[cols].round(3).to_string(index=False))
    print(f"  total {times['total_min']:.1f} min")
    return out


def domain_table(res_dir: str | Path) -> pd.DataFrame:
    """All dom_*__summary.csv files -> one row per dataset / encoder / head / split, decoder as mean +- SD over seeds."""
    import glob

    files = sorted(glob.glob(str(Path(res_dir) / "dom_*__summary.csv")))
    if not files:
        raise FileNotFoundError(f"no dom_*__summary.csv in {res_dir}")
    s = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    agg = (s.groupby(["dataset", "encoder", "head", "split"])
             .agg(seeds=("seed", "nunique"), mIoU=("mIoU", "mean"), mIoU_sd=("mIoU", "std"),
                  accuracy=("accuracy", "mean"))
             .reset_index())
    if "IoU_pos" in s:
        pos = s.groupby(["dataset", "encoder", "head", "split"]).agg(IoU_pos=("IoU_pos", "mean"),
                                                                     IoU_pos_sd=("IoU_pos", "std")).reset_index()
        agg = agg.merge(pos, on=["dataset", "encoder", "head", "split"], how="left")
    return agg
