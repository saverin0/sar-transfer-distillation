"""Stage A training of the students (decided 2026-09-30; both students in
one loop decided 2026-10-01).

One call trains one student from the arrays in RAM

    train_student(config, data, teachers, out_dir)        student M = both teachers
    train_student(config, data, [dinov3], out_dir_s)      student S = DINOv3 SAT only

and one call trains both in ONE loop, the teachers running once per step (`train_students`)

    train_students([("student_M", ["dinov3-l-sat", "terrafm-b"]), ("student_S", ["dinov3-l-sat"])],
                   config, data, teachers, out_root)       folders out_root/student_M, student_S

`data` is a `TrainData` with the arrays as `pilotdata.load_pairs_to_ram` builds them
(that module is not imported here, the arrays are passed in). `teachers` are the frozen
teachers of `teachers/` (name, dim, grid, callable on Sentinel-2 digital numbers) or
`Target` objects, which is what the tests pass.

What happens in `train_student`, in this order
1. Student and heads are built under the seed. AdamW (peak lr 5e-4, weight decay 0.05 on the
   weights of rank 2 and up, none on biases, norms, class token and position embedding, as
   timm does; betas 0.9, 0.999; `fused=True` on CUDA, see `make_optimizer`), 500 warm-up
   steps then cosine to 0 (`lr_at`), gradient clipping at 1.0, bfloat16 autocast on CUDA,
   batch 128 = 96 Major-TOM + 32 aerial, 6,250 steps = 20 epochs over 40,000 samples
   (fixed 30 epochs = 9,375 steps for the pilot; 20 was decided on
   2026-10-01 after the first run measured 0.644 s per step).
2. PHI-S per teacher on its spatial features of the first `phis_samples` (2,048) training
   samples that reach it, checked for unit variance, saved to <out_dir>/phis_<name>.pt and
   loaded from there on a resume. Summary targets stay raw (cosine loss). A resume that has
   a PHI-S file to fit or copy checks the checkpoints first (`_check_before_phis`), so a
   resume that will be refused runs no teacher and writes no file.
3. The loop. Every step is logged to <out_dir>/train_log.csv (step, lr, loss, the parts per
   teacher, grad_norm, step_s, data_s, teacher_s, student_s, img_s, gpu_gb; the four
   timings are the whole step, the wait for the batch plus its copy, the teacher forwards,
   and the student from its forward to its update, each after torch.cuda.synchronize on
   CUDA). Every `checkpoint_every` steps <out_dir>/last.pt is written through a temporary
   name (student, heads, optimiser, scheduler, step, RNG states, config, stats) and the
   held-out loss goes to <out_dir>/held_out.csv. The checkpoint before it is kept as
   <out_dir>/last_prev.pt (`PREV`): the older last_prev.pt is deleted, last.pt is renamed to
   last_prev.pt, then the new last.pt is written, so no file is ever renamed onto an
   existing one and at every moment one complete checkpoint exists. With resume=True a run
   continues from last.pt, or from last_prev.pt with a printed line when last.pt is missing
   or cannot be loaded (a runtime that died while it was written); when a checkpoint is
   there and none loads, the run stops instead of starting over. A checkpoint written for
   another config, other teachers (names, widths and their order) or other summary-only
   teachers stops the run before anything is loaded (`check_resumable`); the order counts
   because AdamW keeps the moments of the head parameters by position. The log rows after the
   resumed step are dropped, and because every batch is a function of the step alone
   (`Batcher`) the result equals an uninterrupted run - bit for bit on the CPU (tested). On
   CUDA the backward kernels of the bilinear interpolation, the row gather of
   `forward_mixed` and the attention add in a non-deterministic order, so there the two runs
   agree in distribution, not bit for bit. At the end <out_dir>/final.pt holds the student,
   heads, config and stats without the optimiser; `load_student` reads it (or last.pt) for
   scoring. A loss or gradient norm that is not finite stops the run (`stop_on_nonfinite`,
   RuntimeError) before the weights take it; the last checkpoint stays. Every
   `progress_every` (100) steps and at the end a line is printed with the step, the loss,
   the seconds per step since the last line and the time left at the pace of the run so
   far (wall time, so the held-out losses and the checkpoints count), `_progress`.
4. The batch of step t + 1 is built on the CPU while step t runs (`Prefetch`, one thread, one
   batch ahead, `prefetch=True`). `Batcher.host` does the draws, the crops and the host
   tensors (pinned on CUDA), `Batcher.to_device` the copy, float32 and the flips, and
   `batch` = `to_device(host)`, so the thread changes no number: with the thread and without
   it the run is bit for bit the same on the CPU (tested). An error in the thread is raised
   again in the loop, the thread is stopped in a finally. data_s in the log is what a step
   waited for its batch plus the copy to the device.

What `train_students` does on top (one loop for all students, the rules above unchanged)
- Per step the batch is built ONCE (the batch the single loop builds for that step, so every
  student sees exactly what it would see alone), the teachers run ONCE (`targets_of`, each
  teacher on the rows it reads, see Targets below), then every student in the order of
  `specs` takes its forward, heads, loss on ITS teachers, backward, clipping, optimiser and
  scheduler step, its own log row, its own held-out loss and its own checkpoints in its own
  folder, the same files and formats as `train_student` writes. Each student is built with
  the seed exactly as `train_student` builds it, so its weights, logs (all columns but the
  timings) and held-out values are bit-identical on the CPU to a separate `train_student`
  run (tested). step_s, data_s, teacher_s and gpu_gb are the same in every log (one step
  holds all students), student_s is each student's own forward-to-update time, img_s is
  images per step over the whole step.
- PHI-S runs once per step too. Every spatial target is standardised once right after the
  teachers (`standardised`, counted in teacher_s) and fed to every student, when all its
  students hold the same PHI-S (the case of `share_phis`); a teacher whose folders hold
  different PHI-S files is standardised per student as before. The same operations on the
  same tensor, so the same numbers (tested); part three standardised once per student, about
  3.5 ms per 1024-wide teacher and student on the A100 by arithmetic (review 2026-10-04).
- The held-out pass runs the teachers ONCE per held-out batch too and feeds every student
  from them (`held_out_losses`, part four); part three ran the teachers
  again for every student there. The numbers are the same as one `held_out_loss` per
  student (tested on the CPU).
- PHI-S is fitted once per teacher (`share_phis`): a file that one student folder holds is
  copied to the folders that lack it, a teacher without any file is fitted once and saved
  into every folder that needs it. A folder that already holds a file keeps it.
- Resume. Each student takes its own checkpoint. When the students are at different steps
  (a runtime that died between the two saves of a checkpoint step) both go on from the
  SMALLER step: the student ahead loads its last_prev.pt at that step and computes the
  steps in between again, which gives the same numbers on the CPU and the same in
  distribution on CUDA, and keeps both students on the same steps. Its newer last.pt is
  deleted and its last_prev.pt becomes its last.pt, so the rotation keeps two different
  steps and a second such crash still finds the common step. In general both start from
  the largest step every student has a checkpoint for, from step 0 when no student has one
  (`_resume_jointly`). When some student has a checkpoint but no step is shared (a student
  finished alone, a folder cleared by hand, a runtime that died between the two writes of
  the FIRST checkpoint) the run stops with the steps per student and nothing is touched -
  joint=False lets every student go on from its own checkpoint, deleting last.pt and
  last_prev.pt of the folders starts them over together. The fallback to last_prev.pt, the
  stop when no checkpoint loads, the checks of `check_resumable` and the stop on a loss that
  is not finite work as in `train_student`. Every check and every load of every student
  comes before the first file is changed (a log cut, a cut last.pt moved aside, a newer
  last.pt deleted), so a refusal leaves all folders as they were.
- GPU memory. joint="auto" takes the joint loop when the device is CUDA and at least
  `JOINT_MIN_GB` of GPU memory are free at the start (`decide_joint`, the arithmetic is next
  to the constant), else the students train one after the other through `train_student`,
  with the same results; joint=True or False forces it. The choice is printed.

Targets and the rows they cover (part four). The rows of a batch are the
Major-TOM samples first, then the aerial ones.
- A teacher that reads Sentinel-2 (slots 1 to 3, and both teachers of part three) reads the
  Sentinel-2 image of a pair in every Major-TOM mode (s1, s2, both). A teacher that reads
  aerial RGB brings its own reader (`reads_aerial`, teachers/base.py), as DINOv3 SAT,
  SigLIP2 and PE-Spatial do. Its rows are the first n rows of the batch.
- A teacher that reads Sentinel-1 (`reads_radar` and not `reads_s2`, TerraFM-B radar in slot
  4) reads the Sentinel-1 crop of the rows whose mode is s1 or both (`S1_MODES`) - the crop
  and flips the student sees, in dB, the teacher standardises it with its own rule - and
  gives no target for the rows of the modes s2 and aerial. The modes are drawn per sample,
  so its rows are an interleaved subset of the Major-TOM rows. `Batch.s1_rows` holds them,
  found on the host from the modes while the batch is built and copied with it, so the
  step finds them without a device sync. A batch with Sentinel-1 rows and no s1_rows (built
  by hand, not by `Batcher`) stops a radar teacher with a ValueError instead of skipping it.
- `target_of` returns the teacher output and the batch rows it belongs to. The rows are a
  slice when they are the first rows of the batch (every teacher but the radar one), so the
  head outputs are cut exactly as part three cut them ([:n], a view, the same numbers), and
  an index tensor on the device otherwise (a gather). `loss_of` takes the head outputs at
  those rows, so the summary and the spatial loss of a teacher are the means over its
  covered rows only (weight 1 = one unit per covered row).
- A teacher that covers no row of a batch (a batch without an s1 or both row for the radar
  teacher) adds nothing to that step. Its columns in train_log.csv show nan for that step
  (my choice - a 0 would read as a perfect match). In the held-out loss it is left out of
  that batch, so each held-out part is the mean over the rows that teacher covered.
- PHI-S is fitted on the spatial targets of the first `phis_samples` COVERED samples.

Summary-only teachers (SigLIP2 gives its summary only).
`as_targets(teachers, summary_only=names)` (or train_student / train_students with
summary_only=names) marks them. For such a teacher the student's `Heads` hold a summary
head and no spatial head, `distil_loss` computes the summary part only, no PHI-S is fitted,
and train_log.csv and held_out.csv have the column <name>/summary and no <name>/spatial.
The teacher still runs as it is (the model computes its spatial output anyway); `target_of`
drops that output (an empty tensor in its place), so it is neither joined nor kept while the
students run. Checkpoints record the names ("summary_only"); a resume with another set stops
with a message instead of loading heads of another shape, and `load_student` with
summary_only=names checks the same for the names among that student's teachers.

Stats. `data.stats` is the per channel mean and SD of the training samples,
{"s1": {"mean": [2], "sd": [2]}, "s2": {"mean": [12], "sd": [12]}, "aerial": {"mean": [4],
"sd": [4]}}. The student sees (x - mean) / sd, the teachers see the raw values.
"""

from __future__ import annotations

import contextlib
import csv
import math
import os
import queue
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence, Union

import numpy as np
import torch

from .phis import PhiS
from .student import MAJORTOM_MODES, MODE_KEYS, Heads, Student, _plain, distil_loss
from .teachers.base import TeacherOut

FORMAT = 1                                  # version of the checkpoint layout
# the Major-TOM modes whose student input holds the Sentinel-1 crop, s1 and both. The radar
# teacher gives targets for these rows only
S1_MODES = tuple(m for m in MAJORTOM_MODES if "s1" in MODE_KEYS[m])
CHANNELS = {"s1": 2, "s2": 12, "aerial": 4}
STREAMS = ("majortom", "aerial")
LOG_END = ("grad_norm", "step_s", "data_s", "teacher_s", "student_s", "img_s", "gpu_gb")
LAST, FINAL, LOG, HELD = "last.pt", "final.pt", "train_log.csv", "held_out.csv"
PREV = "last_prev.pt"                       # the checkpoint before last.pt, the fallback
PREFETCH_WAIT_S = 0.05                      # how often the prefetch thread looks for a stop

# seeds of the generators, all offsets from config.seed * SEED_SPAN
SEED_SPAN = 1_000_000
SEED_PERM = {"majortom": 0, "aerial": 100_000}      # + pass number
SEED_STEP = 200_000                                 # + step number

# Free GPU memory (GB = 10^9 bytes, as stage0.runtime_report counts) the joint loop asks for
# at the start. Part four, stage 0 on the A100 - the expected peak of
# the joint loop with five teachers and three students is 31 GB, by arithmetic on measured
# parts: teacher weights in float32 6.3 GB, the largest teacher forward 3.5 GB, one student
# step at batch 128 17.9 GB (student, heads, optimiser states and the activations of one
# forward), the two other students' weights, gradients and optimiser states 3.5 GB. Only one
# student's activations are alive at a time, because each student's backward frees them
# before the next student runs its forward (`loss_of` returns the scalar loss and floats,
# nothing else holds the graph). The 5 GB up to 36 are headroom for the allocator (it keeps
# freed blocks, so the peak of the process lies above that sum), the held-out pass and the
# checkpoint writes. The 80 GB A100 reports about 84 GB free and trains jointly; the 40 GB one
# reports about 42 GB free and trains jointly too, with less headroom. The measure cell of
# notebook 02 prints the real gpu_gb. History - part three set 36 on 2026-10-01 for two
# teachers and two students (about 21 GB by the same arithmetic, first a 60 GB rule).
JOINT_MIN_GB = 36


@dataclass
class TrainConfig:
    """The fixed values, with one change for the pilot - 20 epochs
    (6,250 steps) instead of 30 (9,375), decided 2026-10-01 after the first run measured
    0.644 s per step. `student` holds keyword arguments of `Student` ({} =
    ViT-B/16 at 256 px), the tests put a tiny model there. `progress_every` only says how
    often the progress line is printed."""

    steps: int = 6250             # 20 epochs over 40,000 samples at batch 128
    batch: int = 128
    n_aerial: int = 32            # aerial samples per batch, the rest are Major-TOM pairs
    crop: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    warmup: int = 500
    clip: float = 1.0
    seed: int = 0
    bf16: bool = True             # bfloat16 autocast on CUDA, nothing on the CPU
    checkpoint_every: int = 1000
    held_out_every: int = 1000
    progress_every: int = 100     # a line with step, loss, s/step and time left every so many steps
    phis_samples: int = 2048
    student: dict = field(default_factory=dict)

    @property
    def n_majortom(self) -> int:
        return self.batch - self.n_aerial

    def check(self) -> None:
        if self.steps < 1 or self.batch < 1 or not 0 <= self.n_aerial < self.batch:
            raise ValueError(f"config: steps {self.steps}, batch {self.batch}, n_aerial "
                             f"{self.n_aerial} do not fit")
        if self.warmup < 1 or self.crop < 1 or self.checkpoint_every < 1 or self.held_out_every < 1                 or self.progress_every < 1:
            raise ValueError("config: warmup, crop, checkpoint_every, held_out_every and "
                             "progress_every must be at least 1")
        if self.phis_samples < 2:
            raise ValueError("config: phis_samples must be at least 2")

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        d = dict(d)
        d["betas"] = tuple(d["betas"])
        d["student"] = dict(d.get("student") or {})
        return cls(**d)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Learning rate of step `step` (0-based). Linear warm-up from lr / warmup at step 0 to
    the peak at step `warmup`, then cosine to 0 at step `steps`."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    if step >= cfg.steps:
        return 0.0
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * p))


# ---- data --------------------------------------------------------------------------------------

@dataclass
class TrainData:
    """The arrays in RAM. Training arrays are required, held-out ones are for the held-out loss."""

    s1: Any                       # (N, 2, side, side) float16, dB, (VV, VH)
    s2: Any                       # (N, 12, side, side) uint16, digital numbers on the 10 m grid
    aerial: Any                   # (M, 4, crop, crop) uint8, (R, G, B, NIR)
    stats: dict                   # per channel mean and SD, layout in the module docstring
    held_s1: Any = None
    held_s2: Any = None
    held_aerial: Any = None

    def check(self, crop: int) -> None:
        check_arrays(self.s1, self.s2, self.aerial, crop, "training")
        if any(x is not None for x in (self.held_s1, self.held_s2, self.held_aerial)):
            check_arrays(self.held_s1, self.held_s2, self.held_aerial, crop, "held-out")
        for key, c in CHANNELS.items():
            part = self.stats.get(key) if isinstance(self.stats, dict) else None
            if not isinstance(part, dict) or len(part.get("mean", ())) != c or \
                    len(part.get("sd", ())) != c:
                raise ValueError(f"stats[{key!r}] needs 'mean' and 'sd' with {c} values each")
            if any(not (float(v) > 0) for v in part["sd"]):
                raise ValueError(f"stats[{key!r}]['sd'] must be above 0")

    @property
    def has_held_out(self) -> bool:
        return self.held_s1 is not None and self.held_s2 is not None and \
            self.held_aerial is not None


def check_arrays(s1, s2, aerial, crop: int, what: str) -> None:
    want = {"s1": (s1, np.float16, 2), "s2": (s2, np.uint16, 12), "aerial": (aerial, np.uint8, 4)}
    for key, (x, dtype, c) in want.items():
        if x is None:
            raise ValueError(f"{what} {key} is missing")
        if x.ndim != 4 or x.shape[1] != c or x.shape[2] != x.shape[3]:
            raise ValueError(f"{what} {key}: expected (n, {c}, side, side), got {tuple(x.shape)}")
        if np.dtype(x.dtype) != dtype:
            raise ValueError(f"{what} {key}: expected {np.dtype(dtype).name}, got {x.dtype}")
        if x.shape[0] < 1:
            raise ValueError(f"{what} {key} is empty")
    if s1.shape[0] != s2.shape[0] or s1.shape[2] != s2.shape[2]:
        raise ValueError(f"{what} s1 {tuple(s1.shape)} and s2 {tuple(s2.shape)} are not pairs")
    if s2.shape[2] < crop:
        raise ValueError(f"{what} pairs are {s2.shape[2]} px, the crop is {crop} px")
    if aerial.shape[2] != crop:
        raise ValueError(f"{what} aerial is {aerial.shape[2]} px, the crop is {crop} px")


class Normaliser:
    """(x - mean) / sd per channel with the training statistics, on `device`."""

    def __init__(self, stats: dict, device):
        self.mean, self.sd = {}, {}
        for key, c in CHANNELS.items():
            m = torch.tensor([float(v) for v in stats[key]["mean"]], dtype=torch.float32)
            s = torch.tensor([float(v) for v in stats[key]["sd"]], dtype=torch.float32)
            if m.numel() != c or s.numel() != c or not (s > 0).all():
                raise ValueError(f"stats[{key!r}] needs {c} means and {c} SDs above 0")
            self.mean[key] = m.view(1, c, 1, 1).to(device)
            self.sd[key] = s.view(1, c, 1, 1).to(device)

    def __call__(self, key: str, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean[key]) / self.sd[key]


@dataclass
class Batch:
    """One batch of raw values on the device, Major-TOM samples first, then the aerial ones.

    s1 (n_mt, 2, crop, crop) float32 dB, s2 (n_mt, 12, crop, crop) float32 digital numbers,
    aerial (n_aer, 4, crop, crop) float32 8-bit values, None when that stream has no rows.
    The same crop and flips were applied to s1 and s2 of a pair, so the teachers (which get
    s2 and aerial as they are here) and the student (which gets them standardised) see the
    same pixels. `index` records what was drawn.

    s1_rows  (k,) int64 on the device, the batch rows whose mode is s1 or both (`S1_MODES`) in
    increasing order - the rows a radar teacher covers. They are also positions in s1, since
    the Major-TOM rows come first. Found on the host while the batch is built.
    """

    modes: list[str]
    s1: torch.Tensor | None
    s2: torch.Tensor | None
    aerial: torch.Tensor | None
    index: dict[str, Any]
    s1_rows: torch.Tensor | None = None

    @property
    def n(self) -> int:
        return len(self.modes)

    @property
    def n_majortom(self) -> int:
        return 0 if self.s2 is None else int(self.s2.shape[0])

    def student_inputs(self, norm: Normaliser) -> dict[str, Any]:
        """The mixed batch of `Student.forward_mixed`, standardised float32."""
        out: dict[str, Any] = {"modes": self.modes}
        for key in CHANNELS:
            x = getattr(self, key)
            if x is not None:
                out[key] = norm(key, x)
        return out


@dataclass
class HostBatch:
    """One batch as the CPU builds it, before the device (`Batcher.host`). `Batcher.to_device`
    turns it into a `Batch`. s1 float16, s2 int16 with the uint16 bits, aerial uint8, all CPU
    tensors (pinned on CUDA); the flips are drawn here and applied on the device. s1_rows is
    the int64 tensor of `Batch.s1_rows`, on the host."""

    step: int | None
    modes: list[str]
    s1: torch.Tensor | None
    s2: torch.Tensor | None
    aerial: torch.Tensor | None
    flip_mt: torch.Tensor
    flip_aer: torch.Tensor
    index: dict[str, Any]
    s1_rows: torch.Tensor | None = None


class Batcher:
    """Draws the batches from arrays in RAM, on `device`.

    Sampling rule (my choice, simple and deterministic)
    - Two streams, Major-TOM pairs and aerial patches. Each stream is a sequence of
      permutations of its samples, one after the other (pass 0, pass 1, ...), each drawn
      from a generator seeded with (seed, stream, pass). Step t takes the elements
      [n t, n t + n) of that sequence, n = 96 for Major-TOM and 32 for aerial. So every
      sample of a stream is seen once per pass, and the batch of a step is a function of
      the step alone, which makes a resumed run equal to an uninterrupted one. With 18,000
      training pairs a pass is 187.5 steps, with 19,000 aerial patches 593.75 steps; the
      6,250 steps are 33.3 passes over the pairs and 10.5 over the aerial patches.
    - Per Major-TOM sample the mode is drawn with equal chance among s1, s2, both, the crop
      corner uniformly in 0 ... side - crop on both axes, a horizontal and a vertical flip
      with chance 1/2 each. An aerial sample gets the two flips only. All from a generator
      seeded with (seed, step).
    - Fixed batches (`fixed`, the held-out loss) take the given rows in order, the centre
      crop, no flips, and the modes s1, s2, both in turn.
    The stored dtypes are moved to the device and turned into float32 there. The uint16
    digital numbers travel as int16 with the same bits (`_host`), because CUDA support for
    uint16 is limited, and the values above 32767 are put back on the device.

    Two halves. `host(step)` is the CPU part (the draws, the crops, the host tensors, pinned
    when `pin`, by default on CUDA) and makes no device call, so `Prefetch` can run it in a
    thread one step ahead. `to_device` is the rest (copy, float32, the uint16 fix, the
    flips). `batch(step)` is the two in a row.
    """

    def __init__(self, s1, s2, aerial, cfg: TrainConfig, device, seed: int | None = None,
                 pin: bool | None = None):
        check_arrays(s1, s2, aerial, cfg.crop, "batcher")
        self.s1, self.s2, self.aerial = np.asarray(s1), np.asarray(s2), np.asarray(aerial)
        self.cfg = cfg
        self.device = torch.device(device)
        self.pin = (self.device.type == "cuda") if pin is None else bool(pin)
        self.seed = cfg.seed if seed is None else int(seed)
        self.crop = cfg.crop
        self.side = int(self.s2.shape[2])
        self.sizes = {"majortom": int(self.s2.shape[0]), "aerial": int(self.aerial.shape[0])}
        self.per_step = {"majortom": cfg.n_majortom, "aerial": cfg.n_aerial}
        self._perms: dict[str, dict[int, np.ndarray]] = {s: {} for s in STREAMS}

    def _perm(self, stream: str, p: int) -> np.ndarray:
        cache = self._perms[stream]
        if p not in cache:
            g = torch.Generator().manual_seed(self.seed * SEED_SPAN + SEED_PERM[stream] + p)
            cache[p] = torch.randperm(self.sizes[stream], generator=g).numpy()
            for old in [k for k in cache if k < p - 1]:
                del cache[old]
        return cache[p]

    def stream_index(self, stream: str, step: int) -> np.ndarray:
        """The sample indices of `stream` at `step`, from the sequence of permutations."""
        n, size = self.per_step[stream], self.sizes[stream]
        out = np.empty(n, dtype=np.int64)
        filled = 0
        while filled < n:
            p, off = divmod(step * n + filled, size)
            take = min(size - off, n - filled)
            out[filled:filled + take] = self._perm(stream, p)[off:off + take]
            filled += take
        return out

    def batch(self, step: int) -> Batch:
        return self.to_device(self.host(step))

    def host(self, step: int) -> HostBatch:
        """The CPU half of `batch(step)` - the draws of the step, the crops, the host tensors."""
        if step < 0:
            raise ValueError(f"batcher: step {step} is below 0")
        idx_mt, idx_aer = self.stream_index("majortom", step), self.stream_index("aerial", step)
        n_mt, n_aer = len(idx_mt), len(idx_aer)
        g = torch.Generator().manual_seed(self.seed * SEED_SPAN + SEED_STEP + step)
        modes = torch.randint(0, len(MAJORTOM_MODES), (n_mt,), generator=g)
        corner = torch.randint(0, self.side - self.crop + 1, (n_mt, 2), generator=g)
        flip_mt = torch.rand(n_mt, 2, generator=g) < 0.5
        flip_aer = torch.rand(n_aer, 2, generator=g) < 0.5
        return self._build_host(step, idx_mt, idx_aer, [MAJORTOM_MODES[int(m)] for m in modes],
                                corner.numpy(), flip_mt, flip_aer)

    def fixed(self, majortom: Sequence[int] = (), aerial: Sequence[int] = ()) -> Batch:
        """Given rows, centre crop, no flips, modes s1, s2, both in turn."""
        idx_mt, idx_aer = np.asarray(majortom, dtype=np.int64), np.asarray(aerial, dtype=np.int64)
        n_mt, n_aer = len(idx_mt), len(idx_aer)
        if n_mt + n_aer == 0:
            raise ValueError("batcher: no rows given")
        corner = np.full((n_mt, 2), (self.side - self.crop) // 2, dtype=np.int64)
        return self.to_device(self._build_host(
            None, idx_mt, idx_aer, [MAJORTOM_MODES[i % len(MAJORTOM_MODES)] for i in range(n_mt)],
            corner, torch.zeros(n_mt, 2, dtype=torch.bool), torch.zeros(n_aer, 2, dtype=torch.bool)))

    def _build_host(self, step, idx_mt, idx_aer, modes_mt, corner, flip_mt, flip_aer) -> HostBatch:
        c = self.crop
        s1 = s2 = aerial = None
        if len(idx_mt):
            s1_np = np.empty((len(idx_mt), 2, c, c), dtype=self.s1.dtype)
            s2_np = np.empty((len(idx_mt), 12, c, c), dtype=self.s2.dtype)
            for k, (i, (r, col)) in enumerate(zip(idx_mt, corner)):
                s1_np[k] = self.s1[i, :, r:r + c, col:col + c]
                s2_np[k] = self.s2[i, :, r:r + c, col:col + c]
            s1, s2 = self._pinned(self._host(s1_np)), self._pinned(self._host(s2_np))
        if len(idx_aer):
            aerial = self._pinned(self._host(self.aerial[idx_aer]))
        index = {"majortom": idx_mt, "aerial": idx_aer, "corner": corner,
                 "flip_majortom": flip_mt.numpy(), "flip_aerial": flip_aer.numpy()}
        s1_rows = torch.from_numpy(np.flatnonzero(
            np.array([m in S1_MODES for m in modes_mt], dtype=bool)).astype(np.int64))
        return HostBatch(step, list(modes_mt) + ["aerial"] * len(idx_aer), s1, s2, aerial,
                         flip_mt, flip_aer, index, self._pinned(s1_rows))

    def to_device(self, hb: HostBatch) -> Batch:
        """The device half of `batch` - the copy, float32, the uint16 fix and the flips."""
        s1 = s2 = aerial = None
        if hb.s1 is not None:
            s1 = self._flip(self._to_device(hb.s1), hb.flip_mt)
            s2 = self._flip(self._to_device(hb.s2), hb.flip_mt)
        if hb.aerial is not None:
            aerial = self._flip(self._to_device(hb.aerial), hb.flip_aer)
        s1_rows = None if hb.s1_rows is None else hb.s1_rows.to(self.device, non_blocking=True)
        return Batch(list(hb.modes), s1, s2, aerial, hb.index, s1_rows)

    @staticmethod
    def _host(x: np.ndarray) -> torch.Tensor:
        """The array as a CPU tensor of a dtype every device converts. uint16 becomes int16
        with the same bits (a value above 32767 turns negative, `_to_device` adds 65536)."""
        if x.dtype == np.uint16:
            return torch.from_numpy(np.ascontiguousarray(x).view(np.int16))
        return torch.from_numpy(x)

    def _pinned(self, t: torch.Tensor) -> torch.Tensor:
        """Page-locked host memory, so the copy to CUDA is fast and does not hold the CPU."""
        return t.pin_memory() if self.pin else t

    def _to_device(self, x) -> torch.Tensor:
        """A numpy array or a host tensor of `_host` to float32 on the device. An int16 tensor
        carries uint16 bits (the convention of `_host`), its values above 32767 come back."""
        t = self._host(x) if isinstance(x, np.ndarray) else x
        bits16 = t.dtype == torch.int16
        t = t.to(self.device, non_blocking=True).to(torch.float32, copy=True)
        if bits16:
            t = torch.where(t < 0, t + 65536.0, t)
        return t

    def _flip(self, x: torch.Tensor, flips: torch.Tensor) -> torch.Tensor:
        """flips (n, 2) bool, column 0 horizontal (last axis), column 1 vertical."""
        if not flips.any():
            return x
        h = flips[:, 0].to(x.device).view(-1, 1, 1, 1)
        v = flips[:, 1].to(x.device).view(-1, 1, 1, 1)
        x = torch.where(h, x.flip(-1), x)
        return torch.where(v, x.flip(-2), x)


class Prefetch:
    """The host batches of the steps `start` ... `stop - 1`, built one ahead in a thread.

    `get(step)` gives the `HostBatch` of `step` (the steps are asked for in order, from
    `start`) and raises again what the thread raised. `close()` stops the thread and waits
    for it; `train_student` calls it in a finally, so a stop, a break or an error in the
    loop leaves no thread behind. With enabled=False (or nothing to build) `get` builds the
    batch in the calling thread and there is no thread at all. The queue holds one batch,
    so the thread is at most one step ahead of the loop.
    """

    THREAD_NAME = "batch-prefetch"

    def __init__(self, batcher: Batcher, start: int, stop: int, enabled: bool = True):
        self.batcher, self.start, self.stop = batcher, int(start), int(stop)
        self.next = self.start
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None
        if enabled and self.start < self.stop:
            self._thread = threading.Thread(target=self._run, name=self.THREAD_NAME, daemon=True)
            self._thread.start()

    @property
    def threaded(self) -> bool:
        return self._thread is not None

    def _put(self, item) -> bool:
        """Into the queue, unless a stop comes first."""
        while not self._halt.is_set():
            try:
                self._queue.put(item, timeout=PREFETCH_WAIT_S)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        try:
            for step in range(self.start, self.stop):
                if self._halt.is_set():
                    return
                if not self._put((step, self.batcher.host(step))):
                    return
        except BaseException as err:                 # handed to the loop, raised again there
            self._put((None, err))

    def get(self, step: int) -> HostBatch:
        if step != self.next:
            raise ValueError(f"prefetch: step {step} asked, {self.next} is next")
        self.next = step + 1
        if self._thread is None:
            return self.batcher.host(step)
        while True:
            try:
                got, item = self._queue.get(timeout=PREFETCH_WAIT_S)
            except queue.Empty:
                if not self._thread.is_alive() and self._queue.empty():
                    raise RuntimeError(f"prefetch: the thread ended before step {step}") from None
                continue
            if got is None:
                raise item
            if got != step:
                raise RuntimeError(f"prefetch: got step {got}, asked for {step}")
            return item

    def close(self) -> None:
        self._halt.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        while True:                                  # what was built and never asked for
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


# ---- targets -----------------------------------------------------------------------------------

@dataclass
class Target:
    """A frozen teacher as a source of targets. It reads Sentinel-2 or Sentinel-1, one of the two.

    s2            maps Sentinel-2 digital numbers (B, 12, H, W) float32 to a TeacherOut, for
                  every Major-TOM row. None for a teacher that does not read Sentinel-2
    aerial        maps aerial 8-bit values (B, 4, H, W) float32 to a TeacherOut, or None when
                  the teacher gives no aerial target
    radar         maps the Sentinel-1 crop in dB (B, 2, H, W) float32 to a TeacherOut, for the
                  rows whose mode is s1 or both only (`S1_MODES`). None for every teacher but
                  the radar one
    summary_only  True = only the summary target is used, no spatial head, no PHI-S
    """

    name: str
    dim: int
    grid: int
    s2: Callable[[torch.Tensor], TeacherOut] | None
    aerial: Callable[[torch.Tensor], TeacherOut] | None = None
    radar: Callable[[torch.Tensor], TeacherOut] | None = None
    summary_only: bool = False


# the batch rows a target belongs to - a slice for the first rows of the batch, else an int64
# index tensor on the device (`target_of`)
Rows = Union[slice, torch.Tensor]


def as_targets(teachers: Sequence[Any], summary_only: Sequence[str] = ()) -> list[Target]:
    """Teacher objects (name, dim, grid, callable) or Targets to Targets.

    A teacher that reads aerial RGB (`reads_aerial`, teachers/base.py) brings its own `aerial`
    reader - 8 bit / 255, its input size, its mean and std, NIR dropped; every other teacher
    gives no aerial target. A teacher that reads Sentinel-1 (`reads_radar` and not `reads_s2`)
    gets its `radar` reader in place of the Sentinel-2 call. Part four - DINOv3 SAT, SigLIP2
    and PE-Spatial read Sentinel-2 and aerial, TerraFM-B radar reads Sentinel-1 only.

    summary_only  names of teachers whose spatial target is not used, for example
    ("siglip2-so400m-256", "siglip2-so400m-naflex"). Every name must be one of the teachers
    given. A Target already marked stays marked."""
    out = []
    for t in teachers:
        if isinstance(t, Target):
            out.append(t)
        else:
            reads_s2 = getattr(t, "reads_s2", True)
            radar = t.radar if getattr(t, "reads_radar", False) and not reads_s2 else None
            aerial = t.aerial if getattr(t, "reads_aerial", False) else None
            out.append(Target(str(t.name), int(t.dim), int(t.grid), t if reads_s2 else None,
                              aerial, radar))
    names = [t.name for t in out]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"teachers: names must be given and unique, got {names}")
    unknown = sorted(set(summary_only) - set(names))
    if unknown:
        raise ValueError(f"teachers - summary-only names {unknown} match no teacher, "
                         f"teachers {names}")
    out = [replace(t, summary_only=True) if t.name in summary_only and not t.summary_only else t
           for t in out]
    for t in out:
        if (t.s2 is None) == (t.radar is None):
            raise ValueError(f"teacher {t.name} - a target reads Sentinel-2 (s2) or Sentinel-1 "
                             "(radar), one of the two")
    return out


def target_of(target: Target, batch: Batch) -> tuple[TeacherOut, Rows] | None:
    """The targets of `target` for `batch` and the batch rows they belong to, or None when the
    teacher covers no row of the batch.

    Rows, in this order - the Major-TOM rows (all of them for a teacher that reads Sentinel-2,
    the rows of `batch.s1_rows` for a radar teacher), then the aerial rows when the teacher
    reads aerial. The rows come back as a slice when they are the first rows of the batch (so
    a cut of the head outputs is the [:n] of part three, a view), else as an int64 index
    tensor on the device. The TeacherOut has one row per covered batch row, in that order.
    For a summary-only teacher its spatial is an empty tensor (0 rows), the model's spatial
    output is dropped here, so it is neither joined nor kept while the students run.
    ValueError for a radar teacher on a batch with Sentinel-1 rows but no s1_rows (a Batch
    built by hand) - it would otherwise get no target on any step without a word."""
    outs, rows = [], []
    n_mt = batch.n_majortom
    if target.s2 is not None and batch.s2 is not None:
        outs.append(target.s2(batch.s2))
        rows.append(slice(0, n_mt))
    if target.radar is not None and batch.s1 is not None:
        if batch.s1_rows is None:
            raise ValueError(f"teacher {target.name} reads Sentinel-1 and the batch has no s1_rows "
                             "- Batcher finds them from the modes, a Batch built by hand needs them")
        if len(batch.s1_rows):                  # the shape only, no device sync
            outs.append(target.radar(batch.s1.index_select(0, batch.s1_rows)))
            rows.append(batch.s1_rows)
    if target.aerial is not None and batch.aerial is not None:
        outs.append(target.aerial(batch.aerial))
        rows.append(slice(n_mt, n_mt + int(batch.aerial.shape[0])))
    if not outs:
        return None
    if all(isinstance(r, slice) for r in rows):           # first rows, then the rows after them
        joined: Rows = slice(rows[0].start, rows[-1].stop)
    else:
        joined = torch.cat([r if isinstance(r, torch.Tensor) else
                            torch.arange(r.start, r.stop, device=batch.s1_rows.device)
                            for r in rows])
    if target.summary_only:                     # its spatial target is not used
        spatial = outs[0].spatial
        empty = spatial.new_empty((0,) + tuple(spatial.shape[1:]))
        summary = outs[0].summary if len(outs) == 1 else torch.cat([p.summary for p in outs])
        return TeacherOut(summary, empty), joined
    if len(outs) == 1:
        return outs[0], joined
    return (TeacherOut(torch.cat([p.summary for p in outs]), torch.cat([p.spatial for p in outs])),
            joined)


def targets_of(targets: Sequence[Target], batch: Batch) -> dict[str, tuple[TeacherOut, Rows]]:
    """The targets of every teacher in `targets` for `batch` with their rows, by name
    (`target_of`), without the teachers that cover no row of the batch. The teachers run once
    here, so the result can feed several students."""
    out = {}
    for t in targets:
        got = target_of(t, batch)
        if got is not None:
            out[t.name] = got
    return out


def take_rows(x: torch.Tensor, rows: Rows) -> torch.Tensor:
    """The rows `rows` of x (a slice is a view, an index tensor gathers on the device)."""
    return x[rows] if isinstance(rows, slice) else x.index_select(0, rows)


def fit_phis(teacher_fn: Callable[[Batch], Any], batcher: Batcher, n: int = 2048,
             name: str = "") -> PhiS:
    """PHI-S of the spatial features of the first `n` training samples that reach the teacher,
    every patch vector of them. `teacher_fn` maps a Batch to what `target_of` gives - a
    (TeacherOut, rows) pair or None - or to a TeacherOut. The rows of the TeacherOut are the
    covered samples in batch order, so for the radar teacher the first `n` samples whose mode
    is s1 or both are taken."""
    feats, count, step = [], 0, 0
    while count < n:
        out = teacher_fn(batcher.batch(step))
        step += 1
        if out is None:
            if step > 10_000:
                raise RuntimeError("fit_phis: no batch reaches the teacher")
            continue
        if isinstance(out, tuple):
            out = out[0]
        if out.spatial.shape[0] == 0 and out.summary.shape[0] > 0:     # would loop for ever
            raise ValueError(f"fit_phis {name} - the teacher gives no spatial target (summary "
                             "only), there is nothing to fit")
        take = min(int(out.spatial.shape[0]), n - count)
        feats.append(out.spatial[:take].reshape(-1, out.spatial.shape[-1]).float())
        count += take
    x = torch.cat(feats)
    phis = PhiS.fit(x, name=name)
    var = phis.check(x)
    print(f"phis {name}: fitted on {count} samples ({x.shape[0]} patch vectors, width "
          f"{x.shape[1]}), alpha {phis.alpha:.4g}, channel variance {float(var.min()):.4f} "
          f"to {float(var.max()):.4f}")
    return phis


def phis_path(out_dir: str | os.PathLike, name: str) -> Path:
    """The PHI-S file of teacher `name` in a student folder."""
    return Path(out_dir) / f"phis_{name}.pt"


def share_phis(needs: Sequence[tuple[Path, Sequence[Target]]], batcher: Batcher,
               cfg: TrainConfig) -> None:
    """One PHI-S file per teacher in every student folder that needs it, fitted ONCE.

    needs  (folder, the targets of the student in that folder), one entry per student.
    A teacher whose file one folder holds gets that file copied to the folders that lack it
    (through a temporary name, so a cut copy leaves no half file). A teacher without any
    file is fitted once (`fit_phis`) and saved into every folder that needs it. A folder
    that already holds a file keeps it, so an earlier fit is never replaced. A summary-only
    teacher gets no file, its spatial target is not used.
    """
    teachers: dict[str, Target] = {}
    for _, targets in needs:
        for t in targets:
            if not t.summary_only:
                teachers.setdefault(t.name, t)
    for name, t in teachers.items():
        folders = [Path(folder) for folder, targets in needs if any(x.name == name for x in targets)]
        have = [f for f in folders if phis_path(f, name).is_file()]
        lack = [f for f in folders if not phis_path(f, name).is_file()]
        if not lack:
            continue
        if have:
            source = phis_path(have[0], name)
        else:
            phis = fit_phis(lambda b, t=t: target_of(t, b), batcher, cfg.phis_samples, name=name)
            source = phis_path(lack[0], name)
            phis.save(source)
            lack = lack[1:]
        for folder in lack:
            dest = phis_path(folder, name)
            tmp = dest.with_name(dest.name + ".tmp")
            shutil.copyfile(source, tmp)
            os.replace(tmp, dest)
            print(f"phis {name}: copied {source} to {dest}")


# ---- the loss of one batch ---------------------------------------------------------------------

def part_columns(targets: Sequence[Target]) -> list[str]:
    """The loss columns of a student's logs, per teacher in order - <name>/summary and
    <name>/spatial, <name>/summary alone for a summary-only teacher."""
    return [f"{t.name}/{k}" for t in targets
            for k in (("summary",) if t.summary_only else ("summary", "spatial"))]


def make_heads(targets: Sequence[Target], student_dim: int) -> Heads:
    """The heads of a student with these teachers, no spatial head for a summary-only one."""
    return Heads({t.name: t.dim for t in targets}, student_dim=student_dim,
                 summary_only=[t.name for t in targets if t.summary_only])


def loss_of(student: Student, heads: Heads, targets: Sequence[Target],
            targets_out: dict[str, tuple[TeacherOut, Rows]], batch: Batch, norm: Normaliser,
            phis: dict[str, PhiS], autocast) -> tuple[torch.Tensor, dict]:
    """Student forward on the mixed batch, heads, `distil_loss` against the teacher outputs
    `targets_out` of this batch (`targets_of`, the outputs with their rows), for the teachers
    in `targets`, in that order. The head outputs are taken at the rows of each teacher, so
    its parts are means over its covered rows. Returns the total, the parts and the rows per
    teacher ("<name>/rows"). A teacher of `targets` without an entry in `targets_out` (it
    covers no row of the batch) adds nothing; its parts are nan and its rows 0."""
    with autocast():
        summary, spatial = student.forward_mixed(batch.student_inputs(norm))
        out = heads(summary, spatial)
    pred, tgt, rows, absent = {}, {}, {}, {}
    for t in targets:
        got = targets_out.get(t.name)
        if got is None:
            for col in part_columns([t]):
                absent[col] = float("nan")
            rows[f"{t.name}/rows"] = 0
            continue
        target, at = got
        p_sum, p_spa = out[t.name]
        pred[t.name] = (take_rows(p_sum, at), None if p_spa is None else take_rows(p_spa, at))
        tgt[t.name] = target
        rows[f"{t.name}/rows"] = int(target.summary.shape[0])
    if not pred:
        raise ValueError("batch_loss: no teacher has a target for this batch")
    loss, parts = distil_loss(pred, tgt, standardise={k: phis[k] for k in pred if k in phis})
    parts.update(absent)
    parts.update(rows)
    return loss, parts


def batch_loss(student: Student, heads: Heads, targets: Sequence[Target], batch: Batch,
               norm: Normaliser, phis: dict[str, PhiS], autocast) -> tuple[torch.Tensor, dict]:
    """The teachers on the batch (`targets_of`), then `loss_of`."""
    return loss_of(student, heads, targets, targets_of(targets, batch), batch, norm, phis, autocast)


def standardised(targets_out: dict[str, tuple[TeacherOut, Rows]],
                 phis: dict[str, PhiS]) -> dict[str, tuple[TeacherOut, Rows]]:
    """`targets_out` with the spatial target of every teacher in `phis` put through its PHI-S
    once, in float32 - the operations `distil_loss` applies, on the same tensor, so the same
    numbers. `loss_of` then gets no PHI-S for these teachers. The joint loop calls it once per
    step, so a teacher that three students share is standardised once, not three times."""
    out = dict(targets_out)
    for name, fn in phis.items():
        if name in out:
            target, rows = out[name]
            out[name] = (TeacherOut(target.summary, fn(_plain(target.spatial, target.spatial))), rows)
    return out


def _shared_phis(per_student: Sequence[dict[str, PhiS]]) -> dict[str, PhiS]:
    """The PHI-S of every teacher whose transform is the same in every student that has it
    (what `share_phis` gives, one file copied into every folder). A teacher whose folders hold
    different files (a folder that kept an older fit) is left out, its students standardise
    with their own as before."""
    shared = {}
    for name in dict.fromkeys(n for phis in per_student for n in phis):      # first seen order
        fns = [phis[name] for phis in per_student if name in phis]
        a = fns[0]
        if all(b.alpha == a.alpha and torch.equal(b.mean, a.mean) and
               torch.equal(b.rotation, a.rotation) for b in fns[1:]):
            shared[name] = a
    return shared


def held_out_losses(models: Sequence[tuple[Student, Heads, Sequence[Target], dict[str, PhiS]]],
                    held: Batcher, norm: Normaliser, cfg: TrainConfig,
                    autocast) -> list[dict[str, float]]:
    """`held_out_loss` for several students in ONE pass. models = [(student, heads, its
    targets, its PHI-S)]. Every teacher of every student runs once per held-out batch
    (`targets_of`) and its targets feed every student that has it, so a teacher shared by
    three students runs once, not three times. A student none of whose teachers covers a
    batch skips it. Returns one dict per student, in order, the numbers of `held_out_loss`."""
    teachers: dict[str, Target] = {}                # every teacher once, in the order given
    for _, _, targets, _ in models:
        for t in targets:
            teachers.setdefault(t.name, t)
    used = list(teachers.values())
    chunks = [(range(a, min(a + cfg.n_majortom, held.sizes["majortom"])), ())
              for a in range(0, held.sizes["majortom"], cfg.n_majortom)]
    if any(t.aerial is not None for t in used) and cfg.n_aerial:
        chunks += [((), range(a, min(a + cfg.n_aerial, held.sizes["aerial"])))
                   for a in range(0, held.sizes["aerial"], cfg.n_aerial)]
    sums: list[dict[str, float]] = [{} for _ in models]
    counts: list[dict[str, int]] = [{} for _ in models]
    for student, heads, _, _ in models:
        student.eval()
        heads.eval()
    try:
        with torch.no_grad():
            for mt, aer in chunks:
                batch = held.fixed(mt, aer)
                outs = targets_of(used, batch)      # the teachers run once for all students
                for k, (student, heads, targets, phis) in enumerate(models):
                    if not any(t.name in outs for t in targets):
                        continue
                    _, parts = loss_of(student, heads, targets, outs, batch, norm, phis, autocast)
                    for key, v in parts.items():
                        if key.endswith("/rows"):
                            continue
                        n = parts[key.rsplit("/", 1)[0] + "/rows"]
                        if n == 0:                  # a teacher that covers no row of this batch
                            continue
                        sums[k][key] = sums[k].get(key, 0.0) + v * n
                        counts[k][key] = counts[k].get(key, 0) + n
    finally:
        for student, heads, _, _ in models:
            student.train()
            heads.train()
    results = []
    for (_, _, targets, _), s, c in zip(models, sums, counts):
        out = {key: s[key] / c[key] for key in part_columns(targets) if key in s}   # log order
        out["loss"] = sum(out.values())
        results.append(out)
    return results


def held_out_loss(student: Student, heads: Heads, targets: Sequence[Target], held: Batcher,
                  norm: Normaliser, phis: dict[str, PhiS], cfg: TrainConfig,
                  autocast) -> dict[str, float]:
    """Mean loss parts over the held-out pairs (batches of n_majortom, fixed crops and modes)
    and the held-out aerial patches (batches of n_aerial), no gradient. Every part is the mean
    over the samples its teacher covered (for the radar teacher the rows of the modes s1 and
    both), "loss" is the sum of the parts. A part whose teacher covered no held-out row is
    missing (the held-out file shows nan)."""
    return held_out_losses([(student, heads, targets, phis)], held, norm, cfg, autocast)[0]


# ---- checkpoints -------------------------------------------------------------------------------

def _rng_state() -> dict:
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path: str | os.PathLike, student: Student, heads: Heads, step: int,
                    cfg: TrainConfig, stats: dict, optimizer=None, scheduler=None,
                    keep_previous: str | None = None) -> Path:
    """Write a checkpoint through a temporary name. Without optimizer it is a final file.

    keep_previous  a file name next to `path` (`PREV`). The file at `path`, if any, is moved
    to that name before the new one is written, and the older file of that name is deleted
    first. So the new file is renamed onto nothing, and one complete checkpoint exists at
    every moment (a run that dies mid-write leaves the moved one)."""
    path = Path(path)
    if keep_previous and path.is_file():
        previous = path.with_name(keep_previous)
        previous.unlink(missing_ok=True)
        os.replace(path, previous)
    state = {"format": FORMAT, "step": int(step), "config": asdict(cfg), "stats": stats,
             "teacher_dims": dict(heads.teacher_dims),
             "summary_only": list(heads.summary_only),
             "student": {k: v.detach().cpu() for k, v in student.state_dict().items()},
             "heads": {k: v.detach().cpu() for k, v in heads.state_dict().items()},
             "optimizer": None if optimizer is None else optimizer.state_dict(),
             "scheduler": None if scheduler is None else scheduler.state_dict(),
             "rng": _rng_state()}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | os.PathLike) -> dict:
    """The checkpoint dict, every tensor on the CPU. Only tensors and plain values are read."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != FORMAT or "student" not in state:
        raise ValueError(f"{path} is not a checkpoint of format {FORMAT}")
    return state


def summary_only_of(state: dict) -> list[str]:
    """The summary-only teachers a checkpoint records, [] for a checkpoint written before
    part four recorded them (every teacher had both heads then)."""
    return [str(n) for n in state.get("summary_only", [])]


def check_summary_only(state: dict, want: Sequence[str], path, who: str) -> None:
    """ValueError when the checkpoint at `path` records other summary-only teachers than `want`
    (`who` names the side that wants them, "this run" or "the call")."""
    have = summary_only_of(state)
    if sorted(have) != sorted(want):
        raise ValueError(f"{path} was written with the summary-only teachers {sorted(have)}, "
                         f"{who} has {sorted(want)} - the heads differ, nothing is loaded")


def check_resumable(state: dict, cfg: TrainConfig, teacher_dims: dict[str, int],
                    summary_only: Sequence[str], path) -> None:
    """ValueError when the checkpoint at `path` cannot go on in this run - another config,
    other teachers (names, widths or their order) or other summary-only teachers. Called
    before anything is loaded. The order counts - AdamW keeps the moments of the head
    parameters by position, so two teachers of the same width swapped would load each
    other's moments without an error, and the log columns would be under the old header."""
    if TrainConfig.from_dict(state["config"]) != cfg:
        raise ValueError(f"{path} was written with another config, not resumed")
    have = [(str(n), int(d)) for n, d in state.get("teacher_dims", {}).items()]
    want = [(str(n), int(d)) for n, d in teacher_dims.items()]
    if have != want:
        raise ValueError(f"{path} was written for the teachers {have}, this run has {want} "
                         "(name and width, in this order) - the heads and their optimiser "
                         "states differ, nothing is loaded")
    check_summary_only(state, list(summary_only), path, "this run")


def _check_before_phis(needs: Sequence[tuple[Path, Sequence[Target]]], cfg: TrainConfig) -> None:
    """On a resume, before any PHI-S file is fitted or copied - the newest checkpoint that loads
    in each folder (last.pt, else last_prev.pt) goes through `check_resumable`, so a resume
    that will be refused runs no teacher and writes no file first. Changes nothing. Reads the
    checkpoints only when some folder lacks a PHI-S file; on a normal resume every file is
    there, nothing is fitted, and the resume reads each checkpoint once, checking it there.
    A file that does not load is left to the resume, which reports it."""
    if all(phis_path(folder, t.name).is_file()
           for folder, targets in needs for t in targets if not t.summary_only):
        return
    for folder, targets in needs:
        for candidate in (Path(folder) / LAST, Path(folder) / PREV):
            if not candidate.is_file():
                continue
            try:
                state = load_checkpoint(candidate)
            except Exception:                   # a cut file, the resume reports it
                continue
            check_resumable(state, cfg, {t.name: int(t.dim) for t in targets},
                            [t.name for t in targets if t.summary_only], candidate)
            break


def load_student(path: str | os.PathLike, device="cpu",
                 summary_only: Sequence[str] | None = None) -> tuple[Student, dict, TrainConfig]:
    """The student of a checkpoint (final.pt or last.pt) in eval mode, its stats and config.

    The trunk only, no heads, so a student whose heads lack spatial parts loads the same way.
    summary_only  the summary-only teachers the caller expects, for example the pilot's
    tuple for every student. Only the names among this student's teachers count (student C
    has no NaFlex teacher); a checkpoint that records another set among them stops with a
    ValueError. None = not checked."""
    state = load_checkpoint(path)
    if summary_only is not None:
        own = set(state.get("teacher_dims", {}))
        check_summary_only(state, [n for n in summary_only if n in own], path, "the call")
    cfg = TrainConfig.from_dict(state["config"])
    student = Student(**cfg.student)
    student.load_state_dict(state["student"], strict=True)
    student = student.to(device).eval()
    return student, state["stats"], cfg


# ---- the run -----------------------------------------------------------------------------------

def param_groups(student: Student, heads: Heads, weight_decay: float) -> list[dict]:
    """Weight decay on the weights of rank 2 and up, none on biases, norms, the class token
    and the position embedding (the rule of timm's optimizer factory)."""
    decay, no_decay = [], []
    for module in (student, heads):
        for name, p in module.named_parameters():
            if p.ndim <= 1 or name in ("pos_embed", "cls_token"):
                no_decay.append(p)
            else:
                decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def make_optimizer(groups: list[dict], cfg: TrainConfig, device) -> torch.optim.AdamW:
    """AdamW on the groups of `param_groups`, with the same decay rules on every device.

    On CUDA it is built with fused=True: the update of all parameters then runs in a few
    kernels instead of one small kernel per parameter and per moment, which is faster for
    the student's 89 M parameters in about 150 tensors. The fused path needs CUDA tensors,
    so the CPU gets the plain AdamW. The numbers of the update are the same up to the float
    rounding of the kernel.
    """
    fused = {"fused": True} if torch.device(device).type == "cuda" else {}
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas),
                             weight_decay=cfg.weight_decay, **fused)


def free_gpu_gb(device) -> float | None:
    """Free memory of the CUDA device in GB (10^9 bytes, as stage0.runtime_report counts).
    None when `device` is not a CUDA device or CUDA is not available."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    free, _ = torch.cuda.mem_get_info(device)
    return free / 1e9


def decide_joint(joint, device) -> tuple[bool, str]:
    """Whether `train_students` runs one loop for all students, and why (printed).

    joint  True or False forces it. "auto" takes the joint loop when `device` is a CUDA
           device with at least `JOINT_MIN_GB` of free memory (`free_gpu_gb`), else the
           students train one after the other.
    """
    if joint is True or joint is False:
        return bool(joint), f"joint={joint} given"
    if joint != "auto":
        raise ValueError(f"joint must be True, False or 'auto', got {joint!r}")
    free = free_gpu_gb(device)
    if free is None:
        return False, "no CUDA device"
    if free >= JOINT_MIN_GB:
        return True, f"{free:.1f} GB of GPU memory free, at least {JOINT_MIN_GB} needed"
    return False, f"only {free:.1f} GB of GPU memory free, under {JOINT_MIN_GB}"


def _duration(seconds: float) -> str:
    """Seconds as text, "1 h 05 min", "55 min" or "40 s"."""
    s = max(int(round(seconds)), 0)
    if s >= 3600:
        return f"{s // 3600} h {(s % 3600) // 60:02d} min"
    if s >= 60:
        return f"{s // 60} min"
    return f"{s} s"


def _progress(done: int, cfg: TrainConfig, losses: str, since_s: float, n_since: int,
              wall_s: float, n_wall: int) -> str:
    """The progress line, every `progress_every` steps and at the end - the step, the losses,
    the seconds per step over the steps since the last line (the step alone) and the time
    left at the pace of this run so far (wall time, the held-out losses and the checkpoints
    included)."""
    text = f"step {done:,} of {cfg.steps:,}  loss {losses}  {since_s / max(n_since, 1):.3f} s/step"
    if done >= cfg.steps:
        return text + "  finished"
    return text + f"  about {_duration((cfg.steps - done) * wall_s / max(n_wall, 1))} left"


def _autocast(device: torch.device, cfg: TrainConfig):
    if device.type == "cuda" and cfg.bf16:
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext


def _sync(device: torch.device) -> None:
    """Wait for the device, so a timing taken after it is the time of the work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _keep_log_rows(path: Path, step: int) -> None:
    """Drop the rows after `step` from a CSV log (a run that was cut after its last checkpoint).

    A row that is not complete (a run killed while its line was written leaves a torn last
    line) is dropped too. Such a row belongs to a step after the checkpoint, its cells can
    only mislead."""
    if not path.is_file():
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    width = len(rows[0])
    kept = [rows[0]] + [r for r in rows[1:]
                        if len(r) == width and r[0].isdigit() and int(r[0]) <= step]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(kept)


def train_student(config: TrainConfig, data: TrainData, teachers: Sequence[Any],
                  out_dir: str | os.PathLike, resume: bool = True, device=None,
                  stop_after: int | None = None, stop_on_nonfinite: bool = True,
                  prefetch: bool = True, summary_only: Sequence[str] = ()) -> dict[str, Any]:
    """Train one student, see the module docstring. `stop_after` (tests) cuts the loop after
    that many steps without a checkpoint of its own. Returns the student, the heads, the step
    reached, the PHI-S transforms and the paths.

    summary_only  names of teachers whose spatial target is not used (`as_targets`),
    among THIS student's teachers - a name that matches none of them stops with a ValueError
    before any work (more likely a typo than a teacher of another student; `train_students`
    takes the names for all its students).

    stop_on_nonfinite  a loss or a gradient norm that is not finite (NaN or infinite) stops
    the run with a RuntimeError that names the step and the values. The step's row goes into
    the log first, the weights are not updated with it, and no checkpoint is written, so the
    last checkpoint on disk stays as it was. False lets the run go on as before.
    prefetch  build the batch of the next step in a thread while the current one runs
    (`Prefetch`). False builds every batch in the loop. The numbers are the same either way."""
    cfg = config
    cfg.check()
    data.check(cfg.crop)
    device = torch.device(device if device is not None else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = as_targets(teachers, summary_only)
    autocast = _autocast(device, cfg)

    torch.manual_seed(cfg.seed)
    student = Student(**cfg.student).to(device)
    heads = make_heads(targets, student.dim).to(device)
    optimizer = make_optimizer(param_groups(student, heads, cfg.weight_decay), cfg, device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_at(s, cfg) / cfg.lr)
    norm = Normaliser(data.stats, device)
    batcher = Batcher(data.s1, data.s2, data.aerial, cfg, device)
    held = (Batcher(data.held_s1, data.held_s2, data.held_aerial, cfg, device)
            if data.has_held_out else None)

    if resume:                                  # a resume that will be refused fits no PHI-S first
        _check_before_phis([(out_dir, targets)], cfg)
    phis: dict[str, PhiS] = {}
    for t in targets:
        if t.summary_only:                      # its spatial target is not used
            continue
        path = phis_path(out_dir, t.name)
        if path.is_file():
            phis[t.name] = PhiS.load(path, device)
            print(f"phis {t.name}: loaded {path}")
        else:
            phis[t.name] = fit_phis(lambda b, t=t: target_of(t, b), batcher, cfg.phis_samples,
                                    name=t.name)
            phis[t.name].save(path)

    start = 0
    last, previous = out_dir / LAST, out_dir / PREV
    checkpoint_step: int | None = None          # the step of the last checkpoint written or resumed
    if resume:
        found = [p for p in (last, previous) if p.is_file()]
        state = None
        for candidate in found:                 # last.pt first, then the one before it
            try:
                state = load_checkpoint(candidate)
            except Exception as err:            # a file cut short by a runtime that died mid-write
                print(f"{candidate} cannot be loaded ({type(err).__name__}: {err}), "
                      + ("the previous checkpoint is tried"
                         if candidate == last and previous in found else "no other checkpoint is there"))
                continue
            check_resumable(state, cfg, heads.teacher_dims, heads.summary_only, candidate)
            student.load_state_dict(state["student"], strict=True)
            heads.load_state_dict(state["heads"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            _set_rng_state(state["rng"])
            start = checkpoint_step = int(state["step"])
            _keep_log_rows(out_dir / LOG, start)
            _keep_log_rows(out_dir / HELD, start)
            if candidate == previous and last in found:     # last.pt is there but does not load
                broken = last.with_name(last.name + ".broken")
                broken.unlink(missing_ok=True)
                os.replace(last, broken)                     # so the rotation keeps the good one
                print(f"{last.name} moved to {broken.name}, the next checkpoint keeps {previous.name}")
            print(f"resumed from {candidate} at step {start}"
                  + (" (the previous checkpoint, last.pt is missing)"
                     if candidate == previous and last not in found else ""))
            break
        if found and state is None:
            raise RuntimeError(f"no checkpoint in {out_dir} can be loaded ({[p.name for p in found]}). "
                               "Delete them to start over, nothing is overwritten")

    parts_columns = part_columns(targets)
    columns = ["step", "lr", "loss"] + parts_columns + list(LOG_END)
    held_columns = ["step", "loss"] + parts_columns
    new_log = start == 0 or not (out_dir / LOG).is_file()
    log = open(out_dir / LOG, "a" if not new_log else "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log)
    if new_log:
        log_writer.writerow(columns)
    held_file = open(out_dir / HELD, "a" if not new_log else "w", newline="", encoding="utf-8")
    held_writer = csv.writer(held_file)
    if new_log:
        held_writer.writerow(held_columns)

    params = [p for g in optimizer.param_groups for p in g["params"]]
    student.train()
    heads.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    done = start
    result: dict[str, Any] = {}
    feed = Prefetch(batcher, start, cfg.steps, enabled=prefetch)
    wall0, since, last_line = time.perf_counter(), 0.0, start    # for the progress line
    try:
        for step in range(start, cfg.steps):
            t0 = time.perf_counter()
            batch = batcher.to_device(feed.get(step))
            _sync(device)
            t_data = time.perf_counter() - t0
            t1 = time.perf_counter()
            targets_out = targets_of(targets, batch)
            _sync(device)
            t_teacher = time.perf_counter() - t1
            t2 = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            loss, parts = loss_of(student, heads, targets, targets_out, batch, norm, phis, autocast)
            loss_value = float(loss.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.clip))
            finite = math.isfinite(loss_value) and math.isfinite(grad_norm)
            if finite or not stop_on_nonfinite:
                optimizer.step()
                scheduler.step()
            _sync(device)
            t_student = time.perf_counter() - t2
            t_step = time.perf_counter() - t0
            done = step + 1
            gpu_gb = (torch.cuda.max_memory_allocated(device) / 2 ** 30
                      if device.type == "cuda" else 0.0)
            row = [done, f"{lr:.6e}", f"{loss_value:.6f}"]
            row += [f"{parts[c]:.6f}" for c in parts_columns]
            row += [f"{grad_norm:.4f}", f"{t_step:.4f}", f"{t_data:.4f}", f"{t_teacher:.4f}",
                    f"{t_student:.4f}", f"{batch.n / t_step:.1f}", f"{gpu_gb:.2f}"]
            log_writer.writerow(row)
            log.flush()
            if not finite and stop_on_nonfinite:
                kept = (f"the last checkpoint stays, {last} at step {checkpoint_step}"
                        if checkpoint_step is not None else
                        "no checkpoint was written in this run, none is written now")
                raise RuntimeError(f"step {done}: the loss is not finite (loss {loss_value}, "
                                   f"gradient norm {grad_norm}), the run stopped before the "
                                   f"weights were updated with it. {kept}. The row of this step "
                                   f"is in {out_dir / LOG}")
            since += t_step
            if done % cfg.progress_every == 0 or done == cfg.steps:
                print(_progress(done, cfg, f"{loss_value:.4f}", since, done - last_line,
                                time.perf_counter() - wall0, done - start), flush=True)
                since, last_line = 0.0, done
            if done % cfg.held_out_every == 0 or done == cfg.steps:
                if held is not None:
                    ho = held_out_loss(student, heads, targets, held, norm, phis, cfg, autocast)
                    held_writer.writerow([done] + [f"{ho.get(c, float('nan')):.6f}"
                                                   for c in held_columns[1:]])
                    held_file.flush()
                    result["held_out"] = ho
                    print(f"step {done}: held-out loss {ho['loss']:.4f}")
            if done % cfg.checkpoint_every == 0 or done == cfg.steps:
                save_checkpoint(last, student, heads, done, cfg, data.stats, optimizer, scheduler,
                                keep_previous=PREV)
                checkpoint_step = done
            if stop_after is not None and done >= stop_after:
                break
    finally:
        feed.close()                            # no thread is left, whatever ended the loop
        log.close()
        held_file.close()
    if done == cfg.steps:
        save_checkpoint(out_dir / FINAL, student, heads, done, cfg, data.stats)
    result.update({"student": student, "heads": heads, "step": done, "phis": phis,
                   "out_dir": out_dir, "log": out_dir / LOG, "held_out_log": out_dir / HELD,
                   "last": last, "final": out_dir / FINAL})
    return result


# ---- several students in one loop --------------------------------------------------------------

class _Trainee:
    """One student of the joint loop, built exactly as `train_student` builds its student."""

    def __init__(self, name: str, out_dir: Path, targets: Sequence[Target], cfg: TrainConfig,
                 device: torch.device):
        self.name, self.out_dir, self.targets = name, out_dir, list(targets)
        torch.manual_seed(cfg.seed)
        self.student = Student(**cfg.student).to(device)
        self.heads = make_heads(self.targets, self.student.dim).to(device)
        self.optimizer = make_optimizer(param_groups(self.student, self.heads, cfg.weight_decay),
                                        cfg, device)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lambda s: lr_at(s, cfg) / cfg.lr)
        self.params = [p for g in self.optimizer.param_groups for p in g["params"]]
        self.rng = _rng_state()                 # what a checkpoint of this student records
        self.phis: dict[str, PhiS] = {}
        self.own_phis: dict[str, PhiS] = {}     # the PHI-S not applied once per step (`_shared_phis`)
        self.parts_columns = part_columns(self.targets)
        self.columns = ["step", "lr", "loss"] + self.parts_columns + list(LOG_END)
        self.held_columns = ["step", "loss"] + self.parts_columns
        self.start = 0
        self.checkpoint_step: int | None = None
        self.log = self.log_writer = self.held_file = self.held_writer = None
        self.row: tuple | None = None           # lr, loss, parts, grad_norm, finite of the step
        self.result: dict[str, Any] = {}

    def load(self, state: dict) -> None:
        self.student.load_state_dict(state["student"], strict=True)
        self.heads.load_state_dict(state["heads"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.rng = state["rng"]
        _set_rng_state(self.rng)
        self.start = self.checkpoint_step = int(state["step"])

    def checkpoint(self, path: Path, step: int, cfg: TrainConfig, stats: dict,
                   final: bool = False) -> None:
        """A checkpoint with this student's RNG state, as `train_student` would write it."""
        _set_rng_state(self.rng)
        if final:
            save_checkpoint(path, self.student, self.heads, step, cfg, stats)
        else:
            save_checkpoint(path, self.student, self.heads, step, cfg, stats, self.optimizer,
                            self.scheduler, keep_previous=PREV)
            self.checkpoint_step = step


def _check_specs(specs: Sequence[tuple[str, Sequence[str]]], known: Sequence[str]) -> list:
    out = []
    for item in specs:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"specs: every entry is (folder name, teacher names), got {item!r}")
        name, names = str(item[0]), [str(n) for n in item[1]]
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError(f"specs: {name!r} is not a folder name")
        unknown = [n for n in names if n not in known]
        if not names or unknown:
            raise ValueError(f"specs: {name} names the teachers {names}, unknown {unknown}, "
                             f"known {list(known)}")
        if len(set(names)) != len(names):
            raise ValueError(f"specs: {name} names a teacher twice, {names}")
        out.append((name, names))
    if not out:
        raise ValueError("specs: no student given")
    if len({name for name, _ in out}) != len(out):
        raise ValueError(f"specs: folder names must be unique, got {[n for n, _ in out]}")
    return out


def _resume_jointly(trainees: Sequence[_Trainee], cfg: TrainConfig) -> int:
    """Load the checkpoints of the joint loop, see the module docstring. Returns the step
    every student goes on from - the largest step every student has a checkpoint for, 0 when
    no student has one. The log rows after it are dropped. A student that goes back to its
    last_prev.pt gets that file as its last.pt (the newer one is deleted). When some student
    has a checkpoint but no step is shared, RuntimeError, nothing is touched. Three phases -
    read and check every file (`check_resumable`), load every student, then change files, so
    a refusal in any of the first two leaves every folder as it was."""
    loaded: dict[str, dict[int, tuple[Path, dict]]] = {}
    files: dict[str, list[Path]] = {}
    broken: set[str] = set()                    # students whose last.pt does not load, moved aside last

    def read(st: _Trainee, candidate: Path, then: str) -> dict | None:
        try:
            state = load_checkpoint(candidate)
        except Exception as err:                # a file cut short by a runtime that died mid-write
            print(f"{candidate} cannot be loaded ({type(err).__name__}: {err}), {then}")
            return None
        check_resumable(state, cfg, st.heads.teacher_dims, st.heads.summary_only, candidate)
        return state

    for st in trainees:                         # the newest loadable checkpoint of every student
        last, previous = st.out_dir / LAST, st.out_dir / PREV
        found = [p for p in (last, previous) if p.is_file()]
        files[st.name], loaded[st.name] = found, {}
        for candidate in found:                 # last.pt first, then the one before it
            state = read(st, candidate, "the previous checkpoint is tried"
                         if candidate == last and previous in found else "no other checkpoint is there")
            if state is None:
                continue
            loaded[st.name][int(state["step"])] = (candidate, state)
            if candidate == previous and last in found:     # last.pt is there but does not load
                broken.add(st.name)
            break
        if found and not loaded[st.name]:
            raise RuntimeError(f"no checkpoint in {st.out_dir} can be loaded ({[p.name for p in found]}). "
                               "Delete them to start over, nothing is overwritten")
    newest = {name: max(states) if states else 0 for name, states in loaded.items()}
    lowest = min(newest.values())
    for st in trainees:                         # a student ahead of another: its older checkpoint too
        previous = st.out_dir / PREV
        if newest[st.name] > lowest and previous in files[st.name] and \
                all(path != previous for path, _ in loaded[st.name].values()):
            state = read(st, previous, "only its newest checkpoint counts")
            if state is not None:
                loaded[st.name][int(state["step"])] = (previous, state)
    common = max(set.intersection(*[set(states) | {0} for states in loaded.values()]))
    if common == 0:
        if not any(loaded.values()):
            return 0
        have = ", ".join(f"{st.name} " + (f"at the steps {sorted(loaded[st.name])}"
                                           if loaded[st.name] else "without a checkpoint")
                         for st in trainees)
        raise RuntimeError(
            f"no step every student has a checkpoint for ({have}), nothing is overwritten. "
            "Either train the students one after the other with joint=False (each goes on from "
            f"its own checkpoint), or delete {LAST} and {PREV} of the folder(s) to start them "
            "over from step 0 together")
    for st in trainees:                         # every student loads before any file is changed
        st.load(loaded[st.name][common][1])
    for st in trainees:                         # then the files
        path = loaded[st.name][common][0]
        last, previous = st.out_dir / LAST, st.out_dir / PREV
        if st.name in broken:
            moved = last.with_name(last.name + ".broken")
            moved.unlink(missing_ok=True)
            os.replace(last, moved)                          # so the rotation keeps the good one
            print(f"{last.name} moved to {moved.name}, the next checkpoint keeps {previous.name}")
        _keep_log_rows(st.out_dir / LOG, common)
        _keep_log_rows(st.out_dir / HELD, common)
        note = ""
        if path == previous and last not in files[st.name]:
            note = " (the previous checkpoint, last.pt is missing)"
        if newest[st.name] > common:
            note += (f" (its newest checkpoint is at step {newest[st.name]}, the steps from "
                     f"{common} on are computed again so that all students take the same steps)")
        print(f"resumed from {path} at step {common}{note}")
        if path == previous and last.is_file():  # the newer last.pt is superseded by the recomputation
            last.unlink()                        # so the rotation keeps two different steps
            os.replace(previous, last)
            print(f"{st.name}: {LAST} at step {newest[st.name]} deleted, {PREV} at step {common} "
                  f"is now {LAST}")
    return common


def train_students(specs: Sequence[tuple[str, Sequence[str]]], config: TrainConfig,
                   data: TrainData, teachers: Sequence[Any], out_root: str | os.PathLike,
                   resume: bool = True, device=None, stop_after: int | None = None,
                   prefetch: bool = True, joint: bool | str = "auto",
                   stop_on_nonfinite: bool = True,
                   summary_only: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    """Train several students on the same batches, see the module docstring.

    specs     [(folder name under out_root, the names of its teachers)], in training order,
              for example the pilot of part four
              [("student_F", ["siglip2-so400m-256", "dinov3-l-sat", "pe-spatial-l", "terrafm-b-radar"]),
               ("student_N", ["siglip2-so400m-naflex", "dinov3-l-sat", "pe-spatial-l", "terrafm-b-radar"]),
               ("student_C", ["siglip2-so400m-256", "dinov3-l-sat", "pe-spatial-l"])]
    teachers  all teachers (or Targets) the specs name
    joint     "auto" (the memory rule of `decide_joint`), True or False
    summary_only  names of teachers whose spatial target is not used, for the pilot
              ("siglip2-so400m-256", "siglip2-so400m-naflex"); the same for every student
    The other arguments are those of `train_student`. Returns {folder name: the result dict
    of `train_student`, plus "mode" = "joint" or "sequential"}.

    One loop (joint). The batch of a step is built once and the teachers run once, then
    every student takes its own update on it; each student has its own files in its own
    folder, bit-identical on the CPU to a `train_student` run. The held-out pass runs the
    teachers once per held-out batch for all students (`held_out_losses`). The resume rule
    of the module docstring applies: students at different steps go on from the smaller
    step, the one ahead computes the steps in between again.
    One after the other (not joint). `train_student` per spec, in order, after the PHI-S
    files were shared (`share_phis`), so the statistics are fitted once here too.
    """
    cfg = config
    cfg.check()
    data.check(cfg.crop)
    device = torch.device(device if device is not None else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    out_root = Path(out_root)
    targets_all = as_targets(teachers, summary_only)
    by_name = {t.name: t for t in targets_all}
    specs = _check_specs(specs, list(by_name))
    chosen = {name: [by_name[n] for n in names] for name, names in specs}
    use_joint, reason = decide_joint(joint, device)
    for name in chosen:
        (out_root / name).mkdir(parents=True, exist_ok=True)
    print(f"students: {', '.join(chosen)} "
          + ("in one loop, the teachers run once per step" if use_joint
             else "one after the other (train_student)") + f", {reason}")

    needs = [(out_root / name, chosen[name]) for name in chosen]
    if resume:                                  # a resume that will be refused fits no PHI-S first
        _check_before_phis(needs, cfg)
    batcher = Batcher(data.s1, data.s2, data.aerial, cfg, device)
    share_phis(needs, batcher, cfg)

    if not use_joint:
        results = {}
        for k, (name, _) in enumerate(specs, 1):
            print(f"{name}: student {k} of {len(specs)}")
            results[name] = train_student(cfg, data, chosen[name], out_root / name, resume=resume,
                                          device=device, stop_after=stop_after,
                                          stop_on_nonfinite=stop_on_nonfinite, prefetch=prefetch)
            results[name]["mode"] = "sequential"
        return results

    autocast = _autocast(device, cfg)
    norm = Normaliser(data.stats, device)
    held = (Batcher(data.held_s1, data.held_s2, data.held_aerial, cfg, device)
            if data.has_held_out else None)
    trainees = []
    for name, _ in specs:
        st = _Trainee(name, out_root / name, chosen[name], cfg, device)
        for t in st.targets:
            if t.summary_only:                  # no PHI-S, its spatial target is not used
                continue
            path = phis_path(st.out_dir, t.name)
            st.phis[t.name] = PhiS.load(path, device)
            print(f"phis {t.name}: loaded {path}")
        trainees.append(st)
    start = _resume_jointly(trainees, cfg) if resume else 0
    shared = _shared_phis([st.phis for st in trainees])     # applied once per step (`standardised`)
    for st in trainees:
        st.own_phis = {k: v for k, v in st.phis.items() if k not in shared}
    used: dict[str, Target] = {}                # every teacher once, in the order of the specs
    for st in trainees:
        for t in st.targets:
            used.setdefault(t.name, t)
    teachers_used = list(used.values())

    for st in trainees:
        new_log = start == 0 or not (st.out_dir / LOG).is_file()
        st.log = open(st.out_dir / LOG, "a" if not new_log else "w", newline="", encoding="utf-8")
        st.log_writer = csv.writer(st.log)
        if new_log:
            st.log_writer.writerow(st.columns)
        st.held_file = open(st.out_dir / HELD, "a" if not new_log else "w", newline="",
                            encoding="utf-8")
        st.held_writer = csv.writer(st.held_file)
        if new_log:
            st.held_writer.writerow(st.held_columns)
        st.student.train()
        st.heads.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    done = start
    feed = Prefetch(batcher, start, cfg.steps, enabled=prefetch)
    wall0, since, last_line = time.perf_counter(), 0.0, start    # for the progress line
    try:
        for step in range(start, cfg.steps):
            t0 = time.perf_counter()
            batch = batcher.to_device(feed.get(step))           # built once
            _sync(device)
            t_data = time.perf_counter() - t0
            t1 = time.perf_counter()
            targets_out = standardised(targets_of(teachers_used, batch), shared)   # teachers and PHI-S once
            _sync(device)
            t_teacher = time.perf_counter() - t1
            t_student = {}
            for st in trainees:                                 # every student in order
                t2 = time.perf_counter()
                lr = st.optimizer.param_groups[0]["lr"]
                loss, parts = loss_of(st.student, st.heads, st.targets, targets_out, batch, norm,
                                      st.own_phis, autocast)
                loss_value = float(loss.detach())
                st.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(st.params, cfg.clip))
                finite = math.isfinite(loss_value) and math.isfinite(grad_norm)
                if finite or not stop_on_nonfinite:
                    st.optimizer.step()
                    st.scheduler.step()
                _sync(device)
                t_student[st.name] = time.perf_counter() - t2
                st.row = (lr, loss_value, parts, grad_norm, finite)
            t_step = time.perf_counter() - t0
            done = step + 1
            gpu_gb = (torch.cuda.max_memory_allocated(device) / 2 ** 30
                      if device.type == "cuda" else 0.0)
            for st in trainees:
                lr, loss_value, parts, grad_norm, finite = st.row
                row = [done, f"{lr:.6e}", f"{loss_value:.6f}"]
                row += [f"{parts[c]:.6f}" for c in st.parts_columns]
                row += [f"{grad_norm:.4f}", f"{t_step:.4f}", f"{t_data:.4f}", f"{t_teacher:.4f}",
                        f"{t_student[st.name]:.4f}", f"{batch.n / t_step:.1f}", f"{gpu_gb:.2f}"]
                st.log_writer.writerow(row)
                st.log.flush()
            for st in trainees:
                lr, loss_value, parts, grad_norm, finite = st.row
                if not finite and stop_on_nonfinite:
                    kept = (f"the last checkpoint stays, {st.out_dir / LAST} at step {st.checkpoint_step}"
                            if st.checkpoint_step is not None else
                            "no checkpoint was written in this run, none is written now")
                    raise RuntimeError(f"{st.name} step {done}: the loss is not finite (loss "
                                       f"{loss_value}, gradient norm {grad_norm}), the run stopped "
                                       f"before the weights were updated with it. {kept}. The row "
                                       f"of this step is in {st.out_dir / LOG}")
            since += t_step
            if done % cfg.progress_every == 0 or done == cfg.steps:
                losses = "  ".join(f"{st.name} {st.row[1]:.4f}" for st in trainees)
                print(_progress(done, cfg, losses, since, done - last_line,
                                time.perf_counter() - wall0, done - start), flush=True)
                since, last_line = 0.0, done
            held_now = (done % cfg.held_out_every == 0 or done == cfg.steps) and held is not None
            if held_now:                        # one pass, the teachers once per held-out batch
                hos = held_out_losses([(st.student, st.heads, st.targets, st.phis)
                                       for st in trainees], held, norm, cfg, autocast)
            for k, st in enumerate(trainees):
                if held_now:
                    ho = hos[k]
                    st.held_writer.writerow([done] + [f"{ho.get(c, float('nan')):.6f}"
                                                      for c in st.held_columns[1:]])
                    st.held_file.flush()
                    st.result["held_out"] = ho
                    print(f"{st.name} step {done}: held-out loss {ho['loss']:.4f}")
                if done % cfg.checkpoint_every == 0 or done == cfg.steps:
                    st.checkpoint(st.out_dir / LAST, done, cfg, data.stats)
            if stop_after is not None and done >= stop_after:
                break
    finally:
        feed.close()                            # no thread is left, whatever ended the loop
        for st in trainees:
            if st.log is not None:
                st.log.close()
            if st.held_file is not None:
                st.held_file.close()
    results = {}
    for st in trainees:
        if done == cfg.steps:
            st.checkpoint(st.out_dir / FINAL, done, cfg, data.stats, final=True)
        st.result.update({"student": st.student, "heads": st.heads, "step": done, "phis": st.phis,
                          "out_dir": st.out_dir, "log": st.out_dir / LOG,
                          "held_out_log": st.out_dir / HELD, "last": st.out_dir / LAST,
                          "final": st.out_dir / FINAL, "mode": "joint"})
        results[st.name] = st.result
    return results
