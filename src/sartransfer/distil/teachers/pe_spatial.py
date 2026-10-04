"""Teacher Perception Encoder PE-Spatial-L14-448, frozen, reading Sentinel-2 RGB and aerial RGB.

Every fact below was read on its primary page on 2026-10-03. "Official code" means
github.com/facebookresearch/perception_models at commit 3e352cca660658d4b5c90f42a7808b11469e4c66
(head of main that day), "card" means huggingface.co/facebook/PE-Spatial-L14-448 at revision
a09b69ff95973d70edd96e6fa4be216d8c6caa70 (last change 2025-07-12, not gated).

- Model. The vision tower of the Perception Encoder (arXiv 2504.13181), spatial variant. The
  card's file list holds PE-Spatial-L14-448.pt (1,215,952,299 bytes) and config.yaml, and that
  config.yaml is an EMPTY file (0 bytes, the empty git blob e69de29b). The architecture
  therefore comes from the official code, core/vision_encoder/config.py, entry
  "PE-Spatial-L14-448" = the "PE-Core-L14-336" entry with image_size 448, pool_type "none",
  use_ln_post False and output_dim None. That is patch 14, width 1024, 24 layers, 16 heads,
  MLP ratio 4.0 (4096 wide), a class token (use_cls_token True), a learned absolute position
  embedding (use_abs_posemb True) AND 2-D RoPE (use_rope2d True), a LayerNorm before the
  blocks (ln_pre), NO LayerNorm after them (ln_post is Identity), no LayerScale, no
  attention pool and no projection. `FULL_CONFIG` is that entry, typed in full, and `build`
  compares it with the vendored config.py.
- Weights. 293 float32 tensors, 303,964,160 values, the keys of VisionTransformer without a
  prefix or a wrapper (read from the file with torch.load, weights_only=True, mmap=True).
  The card's table says 0.32B for the L/14 vision tower; that table is the PE core table,
  whose tower also carries the attention pool and the projection. The spatial file holds
  0.30B. The measured number is the one checked here (`N_PARAMS`, `N_TENSORS`), and the
  test derives it from the config by arithmetic.
- Training. PE spatial "takes the strong spatial performance from the intermediate layers of
  PE core and aligns it to the end using a simple frozen teacher self-distillation loss and
  further refines with a novel SAM 2.1 mask-based learning strategy" (apps/pe/README.md of
  the official code, section "Perception Encoder: Spatial"). Released 2025-07-11 with the
  other small spatial models (root README.md, the Jul-11-25 line).
- Input. core/vision_encoder/transforms.py, get_image_transform - resize to (image_size,
  image_size), bilinear ("We used bilinear during training", the "Squash" branch), convert to
  RGB, ToTensor (0-1), then Normalize with mean (0.5, 0.5, 0.5) and std (0.5, 0.5, 0.5)
  (line 25). So the model sees RGB in -1 to 1. `MEAN` and `STD`.
- Output. The official README (apps/pe/README.md, section 3) reads the spatial tower with
  model.forward_features(image, strip_cls_token=True) and gets (1, 1024, 1024) at 448 px,
  that is 32 x 32 patch tokens. Here forward_features(x, norm=True) is called; for this
  config ln_post is Identity, so norm changes nothing and the tokens are the output of the
  last block. Token order - the class token first, then the patches row by row (the conv1
  output permuted (0, 2, 3, 1) and flattened, pe.py forward_features). summary = the class
  token, because the config has one (a model without one would get the mean of the patch
  tokens, `summary_from` says which); spatial = the patch tokens as (B, grid, grid, 1024).
- 224 px input (grid 16 x 16, the student's grid). The tower was trained at 448 px (grid
  32 x 32). The official code takes any input size by itself (pe.py, "Pretrain image size
  only; you can pass in any image size") - the absolute position embedding, stored for
  32 x 32 plus the class row, is interpolated to the input grid on every forward
  (_sample_abs_posemb, bilinear, align_corners False, the class row kept as is), and the
  2-D RoPE table is built for the input grid (rope.py, Rope2D.update_grid, positions 1 to
  grid per axis, the class token at 0; so RoPE follows the positions by itself). Nothing of
  that is changed here. The model is built at image_size 448, so the file's tensors load
  unchanged, and the authors' own interpolation runs at 224 px - bilinear, theirs, not a
  bicubic one of ours. Whether the features at 224 px follow the scene is measured, not
  assumed - `grid_check` compares the 224 px grid with the 448 px grid of the same picture
  pooled 2 x 2 (mean cosine over the 256 patches). The fall-back the plan
  names is build(..., input_size=448), grid 32 x 32, no interpolation, four times the tokens.
- Attention. SelfAttention runs torch.nn.functional.scaled_dot_product_attention (pe.py),
  the fused kernel. RoPE is applied in float32 (the autocast decorators in rope.py).
- Speed. The forward is one batched pass. The position embedding
  interpolation is one small F.interpolate per call (1 x 1024 x 32 x 32 to 16 x 16), the
  RoPE table is built once per grid size and cached in Rope2D. No per-sample Python loop.
- Code and weights (vendored with its licence, nothing fetched from another repository while running). The vision tower code is vendored (vendor/pe, LICENSE and
  NOTICE.md there) and imports torch and the standard library only. The file is downloaded
  once from the card at the pinned revision (public, no token), checked by size and SHA-256,
  loaded with torch.load(weights_only=True) and a strict key check (no missing, no
  unexpected, no wrong-shape key). No torch.hub, no pip install, no network at import.
- Licence. apache-2.0 in the card front matter and in the Hugging Face tag; LICENSE.PE of the
  official code is the Apache License 2.0 text; apps/pe/README.md says "All checkpoints
  released on this page, unless otherwise specified, are released with the Apache 2.0
  license. The code itself is licensed under the parent license of this repository."
  Exact words, URLs and the date in docs/licences/pe-2026-10-03.md.

The Sentinel-2 input rule is the DINOv3 thumbnail rule of part three ("the DINOv3 thumbnail rule of part three is kept as is"), `scale_rgb` of dinov3_sat, followed
by PE's own normalisation. Aerial RGB (8 bit / 255, NIR dropped) gets PE's normalisation
through the base class (`aerial_mean`, `aerial_std`).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .base import Teacher, TeacherOut, band_index
from .dinov3_sat import RULE_SOURCES as _DINOV3_RULE_SOURCES
from .dinov3_sat import file_sha256, scale_rgb
from .vendor.pe import PE_VISION_CONFIG, VisionTransformer

# Read from the Hugging Face API (models/facebook/PE-Spatial-L14-448, blobs) on 2026-10-03.
HF_ID = "facebook/PE-Spatial-L14-448"
HF_REVISION = "a09b69ff95973d70edd96e6fa4be216d8c6caa70"     # last change 2025-07-12
WEIGHTS_FILE = "PE-Spatial-L14-448.pt"
WEIGHTS_BYTES = 1_215_952_299
# The LFS record of the file list. The file downloaded on 2026-10-03 has the same SHA-256.
WEIGHTS_SHA256 = "47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1"
N_PARAMS = 303_964_160            # measured in the file and in the built model, 2026-10-03
N_TENSORS = 293

CODE_COMMIT = "3e352cca660658d4b5c90f42a7808b11469e4c66"
CONFIG_NAME = "PE-Spatial-L14-448"

# core/vision_encoder/transforms.py line 25 of the official code,
# T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) on ToTensor output (RGB in 0-1).
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

# core/vision_encoder/config.py, PE_VISION_CONFIG["PE-Spatial-L14-448"], every field.
FULL_CONFIG = dict(patch_size=14, width=1024, layers=24, heads=16, mlp_ratio=4.0, output_dim=None,
                   ls_init_value=None, drop_path=0.0, image_size=448, use_abs_posemb=True,
                   use_cls_token=True, use_rope2d=True, pool_type="none", attn_pooler_heads=8,
                   use_ln_pre=True, use_ln_post=False)
# The same kind of model, very small. image_size stays 448, so a 224 px input runs the same
# position embedding interpolation as the real model. Only for offline tests.
TINY_CONFIG = dict(FULL_CONFIG, width=32, layers=2, heads=2)

_CODE = (f"core/vision_encoder/transforms.py line 25 of github.com/facebookresearch/"
         f"perception_models at commit {CODE_COMMIT}, read 2026-10-03")
RULES = ("majortom_thumbnail",)
RULE_SOURCES = {
    "majortom_thumbnail": (
        f"{_DINOV3_RULE_SOURCES['majortom_thumbnail']}. The RGB picture in 0-1 then gets PE's own "
        f"normalisation, (x - 0.5) / 0.5 per channel, {_CODE}. The picture is resized to the "
        "input size with bilinear interpolation, as get_image_transform does"),
}
DEFAULT_RULE = "majortom_thumbnail"

LICENCE = ("Apache-2.0 - 'license: apache-2.0' in the model card front matter and the Hugging "
           "Face tag, the Apache License 2.0 text in LICENSE.PE of the official code, and "
           "apps/pe/README.md 'All checkpoints released on this page, unless otherwise "
           "specified, are released with the Apache 2.0 license.' (read 2026-10-03, "
           "docs/licences/pe-2026-10-03.md)")


def weight_files() -> tuple[tuple[str, str, str], ...]:
    """(repository, pinned revision, file name) of the file the build reads from the Hub cache,
    for teachers.warm_weights."""
    return ((HF_ID, HF_REVISION, WEIGHTS_FILE),)


def check_weights_file(path: str | Path, n_bytes: int = WEIGHTS_BYTES,
                       sha256: str = WEIGHTS_SHA256) -> None:
    """Stop unless the file has the pinned size (bytes) and the pinned SHA-256."""
    size = Path(path).stat().st_size
    if size != n_bytes:
        raise RuntimeError(f"pe-spatial-l: {path} has {size} bytes, expected {n_bytes}")
    found = file_sha256(path)
    if found != sha256:
        raise RuntimeError(f"pe-spatial-l: {path} has SHA-256 {found}, expected {sha256}")


def load_weights(model: torch.nn.Module, path: str | Path) -> None:
    """Load a state dict into `model`. Strict: prints and fails on every key that does not fit.

    The file is read with weights_only=True (tensors only, nothing that runs code) in one
    plain read, 1.2 GB in RAM next to the model for a moment. Not mmap - measured on
    2026-10-03 on the local machine (Windows, torch 2.14 CPU), load_state_dict out of an
    mmap'd file took 96 s against 9.6 s for the plain read plus 1.8 s for the copy.
    The released file holds the keys of the model as they are, no "module." or "visual."
    prefix and no "state_dict" wrapper, so nothing is renamed - a file with other keys fails.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError(f"pe-spatial-l: {path} holds a {type(state).__name__}, "
                           "expected a state dict")
    want = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    have = {k: tuple(v.shape) if torch.is_tensor(v) else None for k, v in state.items()}
    missing = sorted(set(want) - set(have))
    unexpected = sorted(set(have) - set(want))
    shape = sorted(k for k in set(want) & set(have) if want[k] != have[k])
    if missing or unexpected or shape:
        print(f"pe-spatial-l: key check FAILED for {path}")
        print(f"  missing in the file ({len(missing)}) {missing}")
        print(f"  unexpected in the file ({len(unexpected)}) {unexpected}")
        print(f"  shape differs ({len(shape)}) {[(k, have[k], want[k]) for k in shape]}")
        raise RuntimeError(f"pe-spatial-l: {len(missing)} missing, {len(unexpected)} unexpected, "
                           f"{len(shape)} wrong-shape keys in {path}, nothing was loaded")
    model.load_state_dict(state, strict=True)
    print(f"pe-spatial-l: loaded {len(want)} tensors from {path}, no key missing, none unexpected")


def make_model(name: str = CONFIG_NAME) -> VisionTransformer:
    """The architecture of one named config of the vendored config.py, random weights."""
    return VisionTransformer.from_config(name)


def check_model(model: VisionTransformer, config: dict | None = None,
                n_params: int | None = N_PARAMS) -> None:
    """Stop if the model is not the verified architecture (config values, parameter count)."""
    config = FULL_CONFIG if config is None else config
    vendored = dict(vars(PE_VISION_CONFIG[CONFIG_NAME]))
    if vendored != FULL_CONFIG:
        diff = {k: (vendored.get(k), v) for k, v in FULL_CONFIG.items() if vendored.get(k) != v}
        raise ValueError(f"the vendored config {CONFIG_NAME} differs from FULL_CONFIG, "
                         f"(vendored, expected) = {diff}")
    block0 = model.transformer.resblocks[0]
    got = dict(patch_size=model.patch_size, width=model.width,
               layers=len(model.transformer.resblocks), heads=model.heads,
               mlp_ratio=block0.mlp.c_fc.out_features / model.width, output_dim=model.proj_dim,
               ls_init_value=getattr(model.transformer.resblocks[0].ls_1, "init_values", None),
               drop_path=0.0, image_size=model.image_size, use_abs_posemb=model.use_abs_posemb,
               use_cls_token=model.use_cls_token, use_rope2d=model.use_rope2d,
               pool_type=model.pool_type, attn_pooler_heads=config["attn_pooler_heads"],
               use_ln_pre=isinstance(model.ln_pre, torch.nn.LayerNorm),
               use_ln_post=isinstance(model.ln_post, torch.nn.LayerNorm))
    if got != config:
        diff = {k: (got[k], v) for k, v in config.items() if got[k] != v}
        raise ValueError(f"model differs from the verified config, (got, expected) = {diff}")
    if model.attn_pool is not None or hasattr(model, "proj"):
        raise ValueError("model has an attention pool or a projection, the spatial config has none")
    if model.use_abs_posemb and model.posemb_grid_size != model.image_size // model.patch_size:
        raise ValueError(f"position embedding grid {model.posemb_grid_size} is not "
                         f"{model.image_size} // {model.patch_size}")
    n = sum(p.numel() for p in model.parameters())
    if n_params is not None and n != n_params:
        raise ValueError(f"model has {n:,} parameters, expected {n_params:,}")


class PESpatial(Teacher):
    """summary = the class token, spatial = the patch tokens, from forward_features(norm=True)."""

    name = "pe-spatial-l"
    dim = 1024
    patch = 14
    input_size = 224
    bands = ("B04", "B03", "B02")          # red, green, blue
    rules = RULES
    rule_sources = RULE_SOURCES
    licence = LICENCE
    reads_aerial = True
    aerial_mean = MEAN
    aerial_std = STD
    weights = "unknown"        # build() sets it, "random" or the SHA-256 of the loaded file

    def __init__(self, model: VisionTransformer, rule: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16, input_size: int = 224):
        super().__init__(model, rule, device=device, dtype=dtype)
        self.dim = int(model.width)                # differs from 1024 only for a tiny model
        self.patch = int(model.patch_size)
        self.input_size = int(input_size)
        if self.input_size % self.patch:
            raise ValueError(f"{self.name}: input {self.input_size} px is no multiple of "
                             f"patch {self.patch}")
        self.rgb = band_index(self.bands)
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self.summary_from = "class token" if model.use_cls_token else "mean of the patch tokens"
        # the grid the stored position embedding is for, None without an absolute embedding
        self.posemb_grid = int(model.posemb_grid_size) if model.use_abs_posemb else None

    def prepare(self, s2: torch.Tensor, size: int | None = None) -> torch.Tensor:
        """(B, 12, H, W) digital numbers to (B, 3, size, size) normalised RGB, size = input_size.

        The picture is made first and resized after, as get_image_transform resizes a
        picture. `size` serves `grid_check`, which needs the same crop at twice the size.
        """
        size = self.input_size if size is None else int(size)
        dn = s2[:, self.rgb]
        if not torch.isfinite(dn).all():
            raise ValueError(f"{self.name}: the RGB bands hold NaN or infinite values")
        x = scale_rgb(dn, self.rule)
        if tuple(x.shape[-2:]) != (size, size):
            x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True,
                              align_corners=False)
        return (x - self.mean) / self.std

    def _tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """summary (B, dim) and the patch grid (B, H // patch, W // patch, dim) of any input."""
        tokens = self.model.forward_features(x, norm=True)       # (B, prefix + gh * gw, dim)
        b, n, d = tokens.shape
        gh, gw = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        prefix = 1 if self.model.use_cls_token else 0
        if n != prefix + gh * gw:
            raise RuntimeError(f"{self.name}: {n} tokens for a {gh} x {gw} grid, expected "
                               f"{prefix} prefix token(s) + {gh * gw}")
        patches = tokens[:, prefix:]
        summary = tokens[:, 0] if prefix else patches.mean(dim=1)
        return summary, patches.reshape(b, gh, gw, d)

    def _forward(self, x: torch.Tensor) -> TeacherOut:
        summary, spatial = self._tokens(x)
        return TeacherOut(summary, spatial)


def _grid(teacher: PESpatial, x: torch.Tensor) -> torch.Tensor:
    """The patch grid of `x` as float32, under the teacher's autocast on CUDA."""
    if teacher.device.startswith("cuda") and teacher.dtype != torch.float32:
        with torch.autocast("cuda", dtype=teacher.dtype):
            return teacher._tokens(x)[1].float()
    return teacher._tokens(x)[1].float()


@torch.inference_mode()
def grid_check(teacher: PESpatial, x448: torch.Tensor, centred: bool = False) -> float:
    """Mean cosine between the grid at the teacher's input size and the grid at twice that
    size pooled 2 x 2, over all patches and rows. The number says whether the features at
    224 px follow the scene as they do at the 448 px training size.

    `x448` is the model input at twice the teacher's input size, (B, 3, 448, 448) for the
    default teacher - what `teacher.prepare(s2, size=448)` or a 448 px aerial prepare gives.
    The input at the teacher's size is made from it by the teacher's own resampling
    (bilinear with antialias), so both grids see the same crop. 448 px gives a 32 x 32 grid,
    pooled 2 x 2 (mean) to 16 x 16, compared patch by patch with the 16 x 16 grid of the
    224 px input. No threshold here, the notebook decides by eye.

    `centred` first takes the mean token over the patches of each picture out of both grids.
    Measured on 2026-10-03 with the real weights on the CPU - the plain cosine is 0.91 on a
    structured synthetic scene and 0.90 on white noise, so a component that every patch
    token shares carries most of it; the centred cosine is the one that says whether the
    spatial pattern at 224 px is the pattern at 448 px (tests/test_teacher_pe.py prints both).
    """
    size = teacher.input_size
    if x448.ndim != 4 or x448.shape[1] != 3 or tuple(x448.shape[-2:]) != (2 * size, 2 * size):
        raise ValueError(f"{teacher.name} grid_check: expected (B, 3, {2 * size}, {2 * size}), "
                         f"got {tuple(x448.shape)}")
    x448 = x448.to(teacher.device, non_blocking=True).float()
    x = F.interpolate(x448, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    big = _grid(teacher, x448)                                               # (B, 2g, 2g, d)
    pooled = F.avg_pool2d(big.permute(0, 3, 1, 2), kernel_size=2).permute(0, 2, 3, 1)
    small = _grid(teacher, x)                                                # (B, g, g, d)
    if centred:
        pooled = pooled - pooled.mean(dim=(1, 2), keepdim=True)
        small = small - small.mean(dim=(1, 2), keepdim=True)
    return float(F.cosine_similarity(pooled, small, dim=-1).mean())


def build(*, rule: str | None = None, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
          token: str | None = None, weights_dir: str | Path | None = None,
          random_init: bool = False, tiny: bool = False, input_size: int = 224) -> Teacher:
    """Build the frozen teacher. See `teachers/__init__.py` for the common arguments.

    token         not used. The repository is public and the download is anonymous.
    weights_dir   folder that holds PE-Spatial-L14-448.pt. Nothing is downloaded when it is
                  given. Without it the file comes from the Hugging Face cache, pinned by
                  revision.
    input_size    224 (grid 16 x 16, the default) or 448 (grid 32 x 32, the training size,
                  the fall-back). Any multiple of 14 is accepted.
    The weights file is checked by size and SHA-256 before the model is built. With
    `random_init` or `tiny` no file is loaded, a line is printed that says so, and the
    teacher's attribute `weights` is "random". Else it is the SHA-256 of the loaded file.
    """
    rule = DEFAULT_RULE if rule is None else rule
    if rule not in RULES:                   # before any download
        raise ValueError(f"{PESpatial.name}: unknown rule {rule!r}, known: {RULES}")
    path = None
    if not tiny and not random_init:        # found and checked before the model is built
        if weights_dir is not None:
            path = Path(weights_dir) / WEIGHTS_FILE
            if not path.is_file():
                raise FileNotFoundError(f"{PESpatial.name}: {path} not found, nothing is "
                                        "downloaded when weights_dir is given")
        else:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(HF_ID, WEIGHTS_FILE, revision=HF_REVISION, token=False))
        check_weights_file(path)
    if tiny:
        model = VisionTransformer(**TINY_CONFIG)
    else:
        model = make_model()
        check_model(model)
    if path is not None:
        load_weights(model, path)
    teacher = PESpatial(model, rule, device=device, dtype=dtype, input_size=input_size)
    # the file passed check_weights_file, so its SHA-256 is the pinned one
    teacher.weights = "random" if path is None else WEIGHTS_SHA256
    if path is None:
        unused = "" if weights_dir is None else f" The given weights_dir {weights_dir} is not used."
        print(f"{teacher.name}: RANDOM weights, no file loaded.{unused}")
    g = teacher.grid
    if teacher.posemb_grid is None:
        pos = "no absolute position embedding"
    elif teacher.posemb_grid == g:
        pos = f"position embedding stored for {g} x {g}, no interpolation"
    else:
        pos = (f"position embedding stored for {teacher.posemb_grid} x {teacher.posemb_grid}, "
               f"interpolated to {g} x {g} by the authors' code (bilinear)")
    print(f"{teacher.name}: rule {rule}, input {teacher.input_size} px, grid {g} x {g}, "
          f"summary = {teacher.summary_from}, {pos}")
    return teacher
