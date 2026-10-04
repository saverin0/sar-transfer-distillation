"""Frozen encoders behind one interface, tiles in and a (N, gh, gw, dim) patch grid out.

Copied from part two (the `dinov3` kind and the interface, unchanged), with the encoders of
part three added. Every encoder receives the tiles the way
`domains.s1input.encoder_input` makes them and returns one vector per 16 px patch, so the
probe and the decoder of `domains.run` see the grid tile / 16 from every encoder.

Kinds
- `dinov3` (Hugging Face transformers), as in part two. ViT-L/16 at 512 px returns 1029
  tokens, 32x32 patches + 5 prefix (1 CLS + 4 registers). The prefix count is inferred from
  token count vs grid size, never hard-coded. Given 4-channel tiles (mode aerial_rgbn) it
  reads the first three, R, G, B, and drops the NIR channel ("dinov3-aerial").
- `student` (part three). The trained student of `distil.student`, loaded through
  `distil.train.load_student(path, device) -> (model, stats, config)`. It reads 2 channels
  (Sentinel-1 mode) or 4 channels (aerial mode); the input is picked by the channel count
  of the tiles among `model.chans`. Features are the patch tokens after the final norm, the
  last layer of the trunk. Its aerial statistics (`stats["aerial"]`, mean and sd per channel
  in 8-bit units, as `distil.pilotdata.channel_stats` writes them) become `input_stats`.
- `terrafm-s1` (part three). The vendored TerraFM-B of `distil.teachers.terrafm` reading
  Sentinel-1 through its own input layer conv2d_s1, which the vendored file picks for
  2-channel input. The tiles arrive standardised with the TRAINING-SPLIT statistics of the
  dataset (mode co_cross, as for every encoder of part two's protocol and for the students,
  no `input_stats`); the min-max mapping of the pull request, (x - (mean - 2 SD)) / (4 SD),
  equals standardised / 4 + 0.5 and is applied here, no clipping, so the model still sees
  its own input range. The pull request's own Sentinel-1 mean and SD stay in
  `TERRAFM_S1_STATS` as the record of what the model was trained with and are not used for
  scoring (decided 2026-09-30, a clarification, never fixed them for scoring). Why the
  dataset statistics: the glacial lakes are stored as a per-chip 0-1 stretch and the snow
  data as positive uncalibrated dB, where the pull request's dB numbers would put the input
  far outside 0 to 1; on calibrated dB the two statistics differ by under 0.25 dB (stage 0
  measured -12.98 / -19.74 dB against the pull request's -12.76 / -19.64). Patch tokens of
  the last block after the final norm, as `get_intermediate_layers(n=1, norm=True)` gives
  them. `model_id` may be a local folder holding TerraFM-B.pth (no download), else the file
  comes from the Hugging Face cache at the pinned revision.

Input statistics. An encoder that standardises its input with its own numbers instead of a
dataset's training split carries them in `input_stats`, keyed by the s1input mode;
`domains.run.run_domain` uses them in place of the training-split statistics for that mode.
Only dinov3-aerial and the students' aerial mode do so.

Layer. The references of part two are scored at a pre-registered block (dinov3-l-sat 21).
The encoders of part three are scored on their final output, layer label `LAST`.

Grid. Every encoder reads the tile at its own size, so the grid is tile / 16 for every tile
size, as in part two (the models interpolate their position embeddings themselves). TerraFM-B
was trained at 224 px (14 x 14 grid). Resizing a tile to 224 px would give 14 x 14 grids the
probe cannot use (patch labels need a whole number of pixels per patch) and the decoder
cannot use (its image stem is built for stride 16), so the tile is not resized.

Dropped from part two: the kinds radio, terramind and hivit, the remote code hash check and
the checkpoint inspection. Nothing here needs them; the C-RADIO numbers are the saved ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import LAST
from ..distil.teachers.dinov3_sat import MEAN as DINOV3_MEAN, STD as DINOV3_STD
from ..distil.teachers.terrafm import PR_MEAN_S1, PR_MINMAX_SCALE, PR_STD_S1
from ..domains.s1input import aerial_stats


@dataclass
class EncoderSpec:
    model_id: str                   # Hugging Face id or a local path (student: its checkpoint file)
    name: str
    kind: str                       # dinov3 | student | terrafm-s1
    batch: int = 32                 # tiles per forward pass at 512 px on an A100-40GB
    in_channels: tuple[int, ...] = (3,)               # channel counts the encoder accepts
    input_stats: dict = field(default_factory=dict)   # s1input mode -> statistics of the encoder's own


# DINOv3 SAT reads aerial RGB as 8 bit / 255 and then its mean and std.
# As statistics of the 8-bit values that is mean 255 * MEAN and sd 255 * STD. NIR is dropped.
DINOV3_AERIAL_STATS = aerial_stats([255.0 * m for m in DINOV3_MEAN] + [0.0],
                                   [255.0 * s for s in DINOV3_STD] + [255.0])
# The pull request's Sentinel-1 (VV, VH) mean and SD in dB, the record of what TerraFM-B was trained
# with. Not used for scoring: terrafm-b-s1 is fed with the training-split statistics of each dataset
# and then the min-max mapping, see the docstring.
TERRAFM_S1_STATS = {"co": (float(PR_MEAN_S1[0]), float(PR_STD_S1[0])),
                    "cross": (float(PR_MEAN_S1[1]), float(PR_STD_S1[1]))}

ENCODERS = {
    "dinov3-l-sat": EncoderSpec("facebook/dinov3-vitl16-pretrain-sat493m", "dinov3-l-sat", "dinov3"),
    "dinov3-aerial": EncoderSpec("facebook/dinov3-vitl16-pretrain-sat493m", "dinov3-aerial", "dinov3",
                                 in_channels=(4,), input_stats={"aerial_rgbn": DINOV3_AERIAL_STATS}),
    "terrafm-b-s1": EncoderSpec("MBZUAI/TerraFM", "terrafm-b-s1", "terrafm-s1", batch=16, in_channels=(2,)),
}


def student_spec(name: str, checkpoint: str | Path, batch: int = 32) -> EncoderSpec:
    """The spec of one trained student (a checkpoint file), Sentinel-1 mode and aerial mode."""
    return EncoderSpec(str(checkpoint), name, "student", batch=batch, in_channels=(2, 4))


def _mean_sd(stats: dict, key: str) -> tuple[list, list]:
    """(mean, sd) lists of `stats[key]`, in the layout of `distil.pilotdata.channel_stats`."""
    try:
        s = stats[key]
        mean = s["mean"]
        sd = s["sd"] if "sd" in s else s["std"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"student statistics have no {key!r} entry with mean and sd ({e!r})") from e
    return list(mean), list(sd)


class FrozenEncoder:
    """A frozen backbone that maps normalised tiles to a (gh, gw, dim) patch grid."""

    def __init__(self, spec: EncoderSpec, token: str | None = None,
                 dtype: torch.dtype | None = None, device: str = "cuda"):
        self.spec = spec
        self.device = device
        self.on_cuda = str(device).startswith("cuda")
        # part two: bfloat16. Kept on the GPU; on the CPU (tests) float32.
        self.dtype = (torch.bfloat16 if self.on_cuda else torch.float32) if dtype is None else dtype
        self.patch = 16
        self.dim = 0
        self._n_prefix: int | None = None
        self.input_stats = dict(spec.input_stats)

        if spec.kind == "dinov3":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(spec.model_id, token=token, dtype=self.dtype)
            cfg = self.model.config
            self.patch = int(getattr(cfg, "patch_size", 16))
            self.dim = int(getattr(cfg, "hidden_size", 0))
        elif spec.kind == "student":
            from ..distil.train import load_student
            self.model, stats, self.config = load_student(spec.model_id, device)
            self.patch, self.dim = int(self.model.patch), int(self.model.dim)
            self.input_stats["aerial_rgbn"] = aerial_stats(*_mean_sd(stats, "aerial"))
        elif spec.kind == "terrafm-s1":
            from ..distil.teachers import terrafm
            weights_dir = spec.model_id if Path(spec.model_id).is_dir() else None
            teacher = terrafm.build(device=device, dtype=self.dtype, weights_dir=weights_dir)
            self.model, self.weights = teacher.model, teacher.weights
            self.patch = int(self.model.patch_embed.patch_size)
            self.dim = int(self.model.embed_dim)
            print(f"{spec.name}: Sentinel-1 input layer conv2d_s1 (picked by the 2-channel input), "
                  f"min-max mapping on the training-split standardisation, scale {PR_MINMAX_SCALE}")
        else:
            raise ValueError(f"unknown encoder kind {spec.kind!r}")

        self.model = self.model.to(device).eval()
        print(f"{spec.name}: {sum(q.numel() for q in self.model.parameters()) / 1e6:.1f} M parameters loaded")
        if spec.kind == "dinov3":   # the kernel, so the Colab log shows the scoring reads the same one as the teacher
            print(f"{spec.name}: attention {getattr(self.model.config, '_attn_implementation', 'unknown')}")
        # Frozen. Nothing trainable sits in front during extraction, so
        # inference_mode below is safe.
        self.model.requires_grad_(False)

    def __repr__(self) -> str:
        return (f"FrozenEncoder({self.spec.name}, kind={self.spec.kind}, patch={self.patch}, "
                f"dim={self.dim}, n_prefix={self._n_prefix})")

    def _check(self, tiles: np.ndarray) -> None:
        if tiles.ndim != 4 or tiles.shape[1] not in self.spec.in_channels:
            want = " or ".join(str(c) for c in self.spec.in_channels)
            raise ValueError(f"expected (N, {want}, H, W), got {tiles.shape}")

    def _grid(self, tokens: torch.Tensor, tile: int) -> torch.Tensor:
        """(B, T, D) -> (B, gh, gw, D), dropping any prefix tokens."""
        gh = gw = tile // self.patch
        n_patch = gh * gw
        n_prefix = tokens.shape[1] - n_patch
        if n_prefix < 0 or n_prefix > 16:
            raise RuntimeError(
                f"{tokens.shape[1]} tokens for a {tile}px tile at patch {self.patch}: "
                f"expected {n_patch} patch tokens plus a small prefix, got prefix {n_prefix}")
        if self._n_prefix is None:
            self._n_prefix = n_prefix
        elif self._n_prefix != n_prefix:
            raise RuntimeError(f"prefix count changed: {self._n_prefix} -> {n_prefix}")
        self.dim = int(tokens.shape[-1])
        return tokens[:, n_prefix:, :].reshape(tokens.shape[0], gh, gw, -1)

    def _student_input(self, c: int) -> str:
        """The student input with `c` channels ("s1" for 2, "aerial" for 4), read from `model.chans`."""
        names = [n for n, k in self.model.chans.items() if int(k) == c]
        if len(names) != 1:
            raise ValueError(f"{self.spec.name}: {c} channels fit the student inputs {names}, "
                             f"expected exactly one of {dict(self.model.chans)}")
        return names[0]

    def _tokens(self, b: torch.Tensor) -> torch.Tensor:
        """One batch (B, C, H, W) -> token sequence (B, T, D)."""
        kind = self.spec.kind
        if kind == "dinov3":
            if b.shape[1] == 4:                          # aerial R, G, B, NIR -> R, G, B
                b = b[:, :3]
            return self.model(pixel_values=b.to(self.dtype)).last_hidden_state
        with torch.autocast("cuda" if self.on_cuda else "cpu", dtype=self.dtype, enabled=self.on_cuda):
            if kind == "student":
                _, spatial = self.model(**{self._student_input(b.shape[1]): b})
                return spatial.reshape(b.shape[0], -1, spatial.shape[-1])   # (B, g, g, D) -> (B, g g, D)
            if kind == "terrafm-s1":
                x = b / (2.0 * PR_MINMAX_SCALE) + 0.5    # standardised -> (x - (mean - 2 SD)) / (4 SD)
                (tokens,) = self.model.get_intermediate_layers(x, n=1, norm=True)
                return tokens                            # patch tokens only, no prefix
        raise ValueError(kind)

    @property
    def n_layers(self) -> int:
        """Number of transformer blocks."""
        kind = self.spec.kind
        if kind == "dinov3":
            return int(self.model.config.num_hidden_layers)
        if kind in ("student", "terrafm-s1"):
            return len(self.model.blocks)
        raise NotImplementedError(f"layer sweep not supported for {kind}")

    def _layer_tokens(self, b: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
        """Token sequences after block k (1-based), final norm applied, for each k.

        dinov3: HF `hidden_states[k]` is the output of block k (index 0 = the
                embeddings); the model's own backbone variant applies `norm` to
                intermediates, so the same is done here. k = n_layers equals the
                usual `last_hidden_state`.
        student, terrafm-s1: scored on their final output only (layer LAST).
        """
        kind = self.spec.kind
        if kind == "dinov3":
            if b.shape[1] == 4:
                b = b[:, :3]
            out = self.model(pixel_values=b.to(self.dtype), output_hidden_states=True)
            hs = out.hidden_states
            return {k: self.model.norm(hs[k]) for k in layers}
        raise NotImplementedError(f"intermediate layers not supported for {kind}, use layer {LAST!r}")

    @torch.inference_mode()
    def encode_layers(self, tiles: np.ndarray, layers: list[int],
                      batch: int | None = None) -> dict[int, np.ndarray]:
        """Like `encode`, for several layers at once - a dict from layer to (N, gh, gw, dim) fp16."""
        self._check(tiles)
        batch = batch or self.spec.batch
        tile = tiles.shape[-1]
        out: dict[int, list] = {k: [] for k in layers}
        x = torch.from_numpy(tiles)
        for i in range(0, len(x), batch):
            b = x[i:i + batch].to(self.device, non_blocking=True)
            for k, tok in self._layer_tokens(b, layers).items():
                out[k].append(to_host_fp16(self._grid(tok, tile)))
        return {k: np.concatenate(v) for k, v in out.items()}

    @torch.inference_mode()
    def encode(self, tiles: np.ndarray, batch: int | None = None,
               layer: int | str | tuple[int, ...] | None = None) -> np.ndarray:
        """tiles (N, C, tile, tile) float32 -> features (N, gh, gw, dim) float16.

        `layer` (1-based block index) taps an intermediate layer, final norm
        applied, as chosen by the layer sweep. None or LAST = the model's final
        output. A tuple of layers returns them concatenated on the channel axis
        (N, gh, gw, len(layers) * dim), in the given order.
        """
        if layer == LAST:
            layer = None
        self._check(tiles)
        if isinstance(layer, (tuple, list)):
            d = self.encode_layers(tiles, list(layer), batch=batch)
            return np.concatenate([d[k] for k in layer], axis=-1)
        if layer is not None:
            return self.encode_layers(tiles, [layer], batch=batch)[layer]
        batch = batch or self.spec.batch
        tile = tiles.shape[-1]
        out = []
        x = torch.from_numpy(tiles)
        for i in range(0, len(x), batch):
            b = x[i:i + batch].to(self.device, non_blocking=True)
            tok = self._tokens(b)
            out.append(to_host_fp16(self._grid(tok, tile)))
        return np.concatenate(out) if out else np.empty((0,), dtype=np.float16)


def to_host_fp16(t: torch.Tensor) -> np.ndarray:
    """Features -> float16 numpy, cast on the device first (speed-up 2026-09-27).

    Copies half the bytes and skips numpy's float16 conversion on the CPU. Same
    values as the old `t.float().cpu().numpy().astype(np.float16)`: bf16 -> fp32
    is exact, and both paths round fp32 -> fp16 to nearest-even.
    """
    return t.to(torch.float16).cpu().numpy()
