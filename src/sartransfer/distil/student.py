"""The student, its heads and the distillation loss.

The model and the loss. PHI-S is in `phis.py` and reaches `distil_loss` through
`standardise`, the training loop is in `train.py`. Torch only.

Student inputs, all float and already standardised with our own training statistics

    s1      (B, 2, H, W)     Sentinel-1 in dB, channel order (VV, VH)
    s2      (B, 12, H, W)    Sentinel-2, band order `records.S2_BANDS`, all bands on the 10 m grid
    aerial  (B, 4, H, W)     aerial imagery, channel order (R, G, B, NIR)

H = W and a multiple of the patch size. The model is trained at 256 px (16 x 16 grid).
One input layer per input. Modes `MODES` - s1, s2, both (the two embeddings added) and
aerial (alone, it is a separate stream). `forward` runs one mode, `forward_mixed` runs a
batch whose mode is drawn per sample (Major-TOM + 32 aerial).

Student outputs, both taken after the final LayerNorm

    summary  (B, dim)          the class token
    spatial  (B, g, g, dim)    the patch tokens, g = H // patch, indexed [row, column]

Sources, read on 2026-09-29
- ViT-Base, 12 layers, width 768, MLP 3072, 12 heads, 86M parameters, LayerNorm before every
  block, GELU. arXiv 2010.11929, table 1 and section 3.1.
- Heads and summary loss. AM-RADIO, arXiv 2312.06709, sections 3.2 and 3.4.
- MSE for the spatial features. PHI-S, arXiv 2410.01680, section 2.1.

Where this file differs from those pages
- Grids. matches the two grids at the SMALLER one. AM-RADIO (section 4)
  interpolates to the larger one. This file follows.
- Head activation. The AM-RADIO paper says GELU. The released RADIO code (radio/adaptor_mlp.py
  and radio/adaptor_module_factory.py, branch main) has two head classes. MLP (version "v1")
  uses ReLU. MLP2 (version "v2") uses GELU, and without inner layers it is Linear - LayerNorm -
  GELU - Linear, the layout of this file. Which version a released model uses is stored with
  its weights and was not read. This file follows the paper and.
- Size. The default student has 88,794,624 parameters, not 86 M, because it has three input
  layers with 2, 12 and 4 channels where ViT-Base has one with 3.

Our own choices, no page fixes them for this model
- `LN_EPS` and `INIT_STD`, the usual values in ViT code.
- The class token has no position embedding of its own (it is learned anyway).
- Interpolation is plain, with align_corners False and no antialias filter. Going down from
  27 to 16 it reads the teacher grid at the centre of every student patch (the two nearest
  rows and columns). It is not a mean over the patch, a teacher row counts between 0.31 and
  0.97 (measured).
- The heads start from the PyTorch default, the trunk from a truncated normal.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .teachers.base import TeacherOut

MODES = ("s1", "s2", "both", "aerial")
MAJORTOM_MODES = ("s1", "s2", "both")          # the modes a Major-TOM pair can take
MODE_KEYS = {"s1": ("s1",), "s2": ("s2",), "both": ("s1", "s2"), "aerial": ("aerial",)}

LN_EPS = 1e-6            # epsilon of every LayerNorm
INIT_STD = 0.02          # SD of the truncated normal that starts the trunk weights
MLP_RATIO = 4            # width of the MLP inside a block, in units of the model width


def mode_rows(modes: Sequence[str]) -> dict[str, list[int]]:
    """Batch positions of every mode, from the "modes" of a mixed batch. Unknown names stop."""
    rows: dict[str, list[int]] = {m: [] for m in MODES}
    for i, m in enumerate(modes):
        if m not in rows:
            raise ValueError(f"unknown mode {m!r} at position {i}, known: {MODES}")
        rows[m].append(i)
    return rows


def mode_inputs(batch: Mapping[str, Any], mode: str) -> dict[str, torch.Tensor]:
    """Keyword inputs of `Student.forward` for one mode.

    A batch of ONE mode holds "s1", "s2" or "aerial", and every row of it is used.
    A MIXED batch also holds "modes", one mode name per sample in batch order. Its "s1" and
    "s2" hold the rows of the samples with a Major-TOM mode (s1, s2, both) in that order, its
    "aerial" the rows of the aerial samples in that order. Then only the rows of `mode` are
    returned (the aerial rows as they are, the Major-TOM rows picked out).
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, known: {MODES}")
    keys = MODE_KEYS[mode]
    if "modes" not in batch:
        return {k: batch[k] for k in keys}
    modes = list(batch["modes"])
    mode_rows(modes)
    if mode == "aerial":
        return {"aerial": batch["aerial"]}
    stream = [m for m in modes if m != "aerial"]
    rows = [i for i, m in enumerate(stream) if m == mode]
    return {k: batch[k][rows] for k in keys}


class Block(nn.Module):
    """One pre-norm layer on tokens (B, N, dim). Attention, then an MLP, each with a skip."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, eps=LN_EPS)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=LN_EPS)
        self.fc1 = nn.Linear(dim, MLP_RATIO * dim)
        self.fc2 = nn.Linear(MLP_RATIO * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)            # each (B, heads, N, d / heads)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class Student(nn.Module):
    """ViT with one input layer per input. `img` is the training size in pixels."""

    def __init__(self, img: int = 256, patch: int = 16, dim: int = 768, depth: int = 12,
                 heads: int = 12, s1_chans: int = 2, s2_chans: int = 12, aerial_chans: int = 4):
        super().__init__()
        if img <= 0 or img % patch:
            raise ValueError(f"img {img} is not a multiple of patch {patch}")
        if dim % heads:
            raise ValueError(f"dim {dim} is not a multiple of heads {heads}")
        self.img, self.patch, self.dim = img, patch, dim
        self.grid = img // patch
        self.chans = {"s1": s1_chans, "s2": s2_chans, "aerial": aerial_chans}
        self.embed_s1 = nn.Conv2d(s1_chans, dim, patch, stride=patch)
        self.embed_s2 = nn.Conv2d(s2_chans, dim, patch, stride=patch)
        self.embed_aerial = nn.Conv2d(aerial_chans, dim, patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, self.grid, self.grid))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=LN_EPS)

        nn.init.trunc_normal_(self.cls_token, std=INIT_STD)
        nn.init.trunc_normal_(self.pos_embed, std=INIT_STD)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=INIT_STD)
                nn.init.zeros_(m.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def check_input(self, name: str, x: torch.Tensor) -> None:
        c, p = self.chans[name], self.patch
        if x.ndim != 4 or x.shape[1] != c:
            raise ValueError(f"student {name}: expected (B, {c}, H, W), got {tuple(x.shape)}")
        if not torch.is_floating_point(x):
            raise ValueError(f"student {name}: expected a float tensor, got {x.dtype}")
        h, w = x.shape[2:]
        if h != w or h == 0 or h % p:
            raise ValueError(f"student {name}: H and W must be equal and a multiple of {p}, "
                             f"got {h} x {w}")

    def embed(self, s1: torch.Tensor | None = None, s2: torch.Tensor | None = None,
              aerial: torch.Tensor | None = None) -> torch.Tensor:
        """Patch embedding (B, dim, g, g). With s1 and s2 the two embeddings are added.

        `aerial` is a stream of its own and cannot be given together with s1 or s2.
        """
        given = {n: x for n, x in (("s1", s1), ("s2", s2), ("aerial", aerial)) if x is not None}
        if not given:
            raise ValueError("student: give s1, s2, both, or aerial")
        if aerial is not None and len(given) > 1:
            raise ValueError("student: aerial goes alone, not together with s1 or s2")
        for name, x in given.items():
            self.check_input(name, x)
        sizes = {(x.shape[0], x.shape[2]) for x in given.values()}     # H = W is checked above
        if len(sizes) > 1:
            raise ValueError("student: " + " and ".join(f"{n} {tuple(x.shape)}" for n, x in
                                                        given.items()) + " differ in batch or "
                             "image size")
        out = None
        for name, x in given.items():
            e = getattr(self, f"embed_{name}")(x)
            out = e if out is None else out + e
        return out

    def pos(self, g: int) -> torch.Tensor:
        """Position embedding (1, g * g, dim). Bicubic when g is not the training grid."""
        pos = self.pos_embed
        if g != self.grid:
            pos = F.interpolate(pos.float(), size=(g, g), mode="bicubic", align_corners=False)
        return pos.flatten(2).transpose(1, 2)

    def forward(self, s1: torch.Tensor | None = None, s2: torch.Tensor | None = None,
                aerial: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.trunk(self.embed(s1, s2, aerial))

    def forward_mixed(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """`forward` for a mixed batch (layout in `mode_inputs`), every sample in its own mode.

        The samples are embedded grouped by mode, then all run through the trunk together.
        summary (B, dim) and spatial (B, g, g, dim) come back in batch order.
        """
        modes = list(batch["modes"])
        if not modes:
            raise ValueError("student: the mixed batch is empty")
        rows = mode_rows(modes)
        n_aerial = len(rows["aerial"])
        n_majortom = len(modes) - n_aerial
        for key in sorted({k for m in MODES if rows[m] for k in MODE_KEYS[m]}):
            want = n_aerial if key == "aerial" else n_majortom
            if key not in batch or batch[key].shape[0] != want:
                got = tuple(batch[key].shape) if key in batch else None
                raise ValueError(f"student: mixed batch needs {key!r} with {want} rows, got {got}")
        parts, order, used = [], [], []
        for mode in MODES:
            if rows[mode]:
                parts.append(self.embed(**mode_inputs(batch, mode)))
                order += rows[mode]
                used.append(mode)
        if len({tuple(p.shape[1:]) for p in parts}) > 1:
            raise ValueError("student: the modes of a mixed batch must share one image size, got "
                             + ", ".join(f"{m} {p.shape[2] * self.patch} px"
                                         for m, p in zip(used, parts)))
        x = torch.cat(parts)
        back = torch.empty(len(order), dtype=torch.long, device=x.device)
        back[torch.tensor(order, device=x.device)] = torch.arange(len(order), device=x.device)
        return self.trunk(x[back])

    def trunk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """From the patch embedding (B, dim, g, g) to summary (B, dim) and spatial (B, g, g, dim)."""
        b, _, g, _ = x.shape
        x = x.flatten(2).transpose(1, 2)                              # (B, g * g, dim), row by row
        x = x + self.pos(g).to(x.dtype)
        x = torch.cat([self.cls_token.to(x.dtype).expand(b, -1, -1), x], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:].reshape(b, g, g, self.dim)


def tiny_student(**kw) -> Student:
    """A very small student with random weights, for tests. Same 16 x 16 training grid."""
    return Student(**{"img": 256, "patch": 16, "dim": 32, "depth": 2, "heads": 4, **kw})


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.LayerNorm(d_hidden, eps=LN_EPS),
                         nn.GELU(), nn.Linear(d_hidden, d_out))


class Heads(nn.Module):
    """Per teacher a summary head and a spatial head, each Linear - LayerNorm - GELU - Linear.

    Input width = the student's, hidden width = the largest teacher width, output width =
    that teacher's. Both heads work on the last axis, so the grid stays (B, g, g, .).

    summary_only  names of teachers whose spatial target is not used (SigLIP2 in part four). Such a teacher gets a summary head and NO spatial head, and
    `forward` gives None in place of its spatial output. Its width still counts for the
    hidden width, it is a teacher of the student.
    """

    def __init__(self, teacher_dims: dict[str, int], student_dim: int = 768,
                 summary_only: Iterable[str] = ()):
        super().__init__()
        if not teacher_dims:
            raise ValueError("heads: no teacher given")
        for name, d in teacher_dims.items():
            if not name or "." in name:
                raise ValueError(f"heads: teacher name {name!r} is empty or holds a dot")
            if int(d) != d or d <= 0:
                raise ValueError(f"heads: width of {name!r} must be a whole number above 0, "
                                 f"got {d}")
        only = set(summary_only)
        unknown = sorted(only - set(teacher_dims))
        if unknown:
            raise ValueError(f"heads - summary-only names {unknown} match no teacher, "
                             f"teachers {list(teacher_dims)}")
        self.teacher_dims = {n: int(d) for n, d in teacher_dims.items()}
        self.summary_only = tuple(n for n in self.teacher_dims if n in only)     # teacher order
        self.hidden = max(self.teacher_dims.values())
        self.summary = nn.ModuleDict(
            {n: _mlp(student_dim, self.hidden, d) for n, d in self.teacher_dims.items()})
        self.spatial = nn.ModuleDict(
            {n: _mlp(student_dim, self.hidden, d) for n, d in self.teacher_dims.items()
             if n not in only})

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, summary: torch.Tensor,
                spatial: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
        return {n: (self.summary[n](summary),
                    self.spatial[n](spatial) if n in self.spatial else None)
                for n in self.teacher_dims}


def resize_grid(x: torch.Tensor, g: int) -> torch.Tensor:
    """(B, h, w, d) -> (B, g, g, d), bilinear. A grid of that size is returned as it is."""
    if x.shape[1] == g and x.shape[2] == g:
        return x
    x = F.interpolate(x.permute(0, 3, 1, 2), size=(g, g), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1)


def match_grids(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two square grids (B, g, g, d) brought to the smaller of the two grid sizes."""
    g = min(a.shape[1], b.shape[1])
    return resize_grid(a, g), resize_grid(b, g)


def _plain(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """A teacher tensor as float32 on the device of `like`, usable in a backward pass.

    Teachers run under `torch.inference_mode()`, and such tensors cannot be saved for backward.
    """
    t = t.to(like.device, torch.float32)
    return t.clone() if t.is_inference() else t


def distil_loss(pred: dict[str, tuple[torch.Tensor, torch.Tensor | None]],
                target: dict[str, TeacherOut],
                standardise: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
                ) -> tuple[torch.Tensor, dict[str, float]]:
    """Sum over the teachers of summary loss + spatial loss, every teacher with weight 1.

    pred         name -> (summary (B, dim_t), spatial (B, g_s, g_s, dim_t)), the output of `Heads`.
                 spatial None = a summary-only teacher (`Heads(summary_only=...)`), only its
                 summary part is computed and its spatial target is not read.
    target       name -> TeacherOut with summary (B, dim_t) and spatial (B, g_t, g_t, dim_t)
    standardise  name -> function for that teacher's spatial target (B, g_t, g_t, dim_t), applied
                 on the teacher's own grid. None or no entry = the target is used as it is.

    summary loss = 1 - cosine similarity, mean over the batch
    spatial loss = mean squared error, after both grids are brought to the smaller one

    B is the rows the teacher has a target for. The training loop gives here the head outputs
    at exactly those rows (`train.loss_of`), so both parts are means over those rows only.

    The loss is computed in float32. Returns the total and the parts as floats, with the
    keys "<name>/summary" and "<name>/spatial" ("<name>/summary" alone for a summary-only
    teacher). `pred` and `target` must hold the same names.
    """
    if set(pred) != set(target):
        raise ValueError(f"loss: heads {sorted(pred)} and teachers {sorted(target)} differ")
    unknown = set(standardise or {}) - set(pred)
    if unknown:
        raise ValueError(f"loss: standardise names {sorted(unknown)} match no teacher")

    total, parts = None, {}
    for name, (p_sum, p_spa) in pred.items():
        t_sum = _plain(target[name].summary, p_sum)
        fn = (standardise or {}).get(name)
        if p_spa is None:
            if fn is not None:
                raise ValueError(f"loss {name} - a summary-only teacher has no spatial target "
                                 "to standardise")
        else:
            t_spa = _plain(target[name].spatial, p_spa)
            if fn is not None:
                t_spa = fn(t_spa)
        if p_sum.ndim != 2 or p_sum.shape != t_sum.shape:
            raise ValueError(f"loss {name}: summary {tuple(p_sum.shape)} against teacher "
                             f"{tuple(t_sum.shape)}")
        if p_spa is not None:
            for what, x in (("head", p_spa), ("teacher", t_spa)):
                if x.ndim != 4 or x.shape[1] != x.shape[2]:
                    raise ValueError(f"loss {name}: {what} grid must be (B, g, g, dim), "
                                     f"got {tuple(x.shape)}")
            if (p_spa.shape[0], p_spa.shape[3]) != (t_spa.shape[0], t_spa.shape[3]):
                raise ValueError(f"loss {name}: spatial {tuple(p_spa.shape)} against teacher "
                                 f"{tuple(t_spa.shape)}, batch or width differ")

        # The order of the operations and of the sums is the one of part three, so a teacher
        # with both parts gives the same numbers as before.
        l_sum = (1.0 - F.cosine_similarity(p_sum.float(), t_sum.float(), dim=-1)).mean()
        l_spa = None
        if p_spa is not None:
            a, b = match_grids(p_spa.float(), t_spa.float())
            l_spa = F.mse_loss(a, b)
        parts[f"{name}/summary"] = float(l_sum.detach())
        if l_spa is not None:
            parts[f"{name}/spatial"] = float(l_spa.detach())
        if total is None:
            total = l_sum if l_spa is None else l_sum + l_spa
        else:
            total = total + l_sum if l_spa is None else total + l_sum + l_spa
    if total is None:
        raise ValueError("loss: no teacher given")
    return total, parts
