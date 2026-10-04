"""Teachers SigLIP2 so400m patch16, frozen, reading Sentinel-2 RGB or aerial RGB.

Slot 1 in two variants, one module. Both are the vision tower of a
SigLIP 2 model (paper arXiv 2502.14786, named on both model cards), loaded through the
transformers library (5.17.0 here). Every fact below was read on its primary page on
2026-10-03 without a token. The public model information of Hugging Face
(huggingface.co/api/models/<id>/revision/<revision>?blobs=true) gives the revision, the
file sizes, the SHA-256 of the LFS files and the parameter count of the checkpoint.

- "fixres", teacher name "siglip2-so400m-256". google/siglip2-so400m-patch16-256, model type
  siglip, not gated. Revision e8708ab72d125807e45b36fb7d4e0aacbb59f379 (last change
  2025-02-21). Files - config.json 537 bytes, preprocessor_config.json 394 bytes,
  model.safetensors 4,542,792,928 bytes (SHA-256 810ddc85ac019e6b2738b9130ae3602e41eb093a
  848a297a0ad296ca1f39dc67, line break ours). The vision tower runs as `SiglipVisionModel`.
  config.json, vision_config - hidden_size 1152, image_size 256, intermediate_size 4304,
  num_attention_heads 16, num_hidden_layers 27. patch_size is not in the file, so it is the
  default 16 of SiglipVisionConfig (transformers 5.17.0, as the repository name says).
- "naflex", teacher name "siglip2-so400m-naflex". google/siglip2-so400m-patch16-naflex, model
  type siglip2, not gated. Revision cc24074f717b612951c2dead130904ab9b65a81e (last change
  2025-02-21). Files - config.json 590 bytes, preprocessor_config.json 393 bytes,
  model.safetensors 4,542,792,928 bytes (SHA-256 11a61a2068800d5f4f35cb041c1fea25de86ce87
  725e4c97a9ed046d0b22c076, line break ours). The vision tower runs as `Siglip2VisionModel`.
  config.json, vision_config - hidden_size 1152, intermediate_size 4304, num_attention_heads
  16, num_hidden_layers 27. patch_size and num_patches are not in the file, so they are the
  defaults 16 and 256 of Siglip2VisionConfig (transformers 5.17.0). The learned position
  embedding has 256 entries, a 16 x 16 grid, and `max_num_patches` of the processor is 256.
- Parameters. Both checkpoints hold 1,135,670,962 float32 parameters (safetensors metadata of
  the Hub). The vision tower has 427,888,064 of them, computed from the config (the patch
  embedding, conv or linear, has the same count) and checked on the loaded weights by
  `check_model`. With the text tower (707,782,896 by the same arithmetic from the text config
  of the file) and the two scalars logit_scale and logit_bias the sum is the Hub total.
- Input. Both preprocessor_config.json files rescale by 1/255 and normalise with image_mean
  (0.5, 0.5, 0.5) and image_std (0.5, 0.5, 0.5), so a picture in 0-1 becomes -1 to 1. The
  FixRes one (SiglipImageProcessor) resizes to 256 x 256 with resample 2 (bilinear); the
  NaFlex one (Siglip2ImageProcessorFast, patch_size 16, max_num_patches 256) resizes to the
  largest grid of at most 256 patches that keeps the aspect ratio, 16 x 16 for a square. The
  library processors are not used here. The input is a float tensor that is already 256 px,
  so only the normalisation is applied (`MEAN`, `STD`); the Sentinel-2 digital numbers become
  a picture in 0-1 first under the thumbnail rule of DINOv3 (`scale_rgb` of dinov3_sat).
- Output, FixRes (modeling_siglip.py, SiglipVisionModel.forward). A conv with kernel and
  stride 16 gives (B, dim, 16, 16), flattened with .flatten(2).transpose(1, 2), so token i is
  the patch in row i // 16 and column i % 16 (row major), the position embedding is read in
  the same order. 27 encoder layers, then post_layernorm give `last_hidden_state`
  (B, 256, 1152). `pooler_output` (B, 1152) is `model.head`, a
  SiglipMultiheadAttentionPoolingHead - a learned probe attends over all tokens
  (torch.nn.MultiheadAttention), then LayerNorm and an MLP with a residual. summary =
  pooler_output, spatial = last_hidden_state as (B, 16, 16, dim).
- Output, NaFlex (modeling_siglip2.py, Siglip2VisionModel.forward). The input is already
  patchified - pixel_values (B, N, 3 * 16 * 16), pixel_attention_mask (B, N) and
  spatial_shapes (B, 2). `patchify` here is the batched form of the library's
  convert_image_to_patches (image_processing_siglip2.py) - reshape (C, h, p, w, p), permute
  (1, 3, 2, 4, 0), reshape (h * w, p * p * C) - so patch i is row i // w, column i % w, row
  major again. Siglip2VisionEmbeddings applies a Linear to the patch vectors and adds the
  position embedding reshaped to (16, 16, dim) and resized by bilinear antialiased
  interpolation to each row's spatial shape (resize_positional_embeddings, a Python loop over
  the batch rows, each reading spatial_shapes[i].tolist()), flattened (dim, h * w) in the
  same row-major order. At (16, 16) the resize is the identity (measured, difference 0, in
  tests/test_teacher_siglip2.py). The pooling head takes the mask too; with a mask of ones
  it equals the head without a mask (measured there). spatial_shapes is kept on the CPU so
  the per-row .tolist() of the library costs no device sync; the mask is int32 as the
  library processor makes it.
- Loading. The checkpoint is the whole model (vision tower, text tower, logit scale and
  bias). `SiglipVisionModel.from_pretrained` and `Siglip2VisionModel.from_pretrained` on
  that folder read the vision tower by themselves (base_model_prefix "vision_model",
  base_config_key "vision_config", transformers 5.17.0) and report the rest as unexpected
  keys. `load_folder` stops on a missing or mismatched vision tensor and on any unexpected
  key that is not the text tower or one of the two scalars (`UNEXPECTED_OK`).
- Attention kernel. sdpa, as the DINOv3 teacher (`ATTENTION`); the library default on
  5.17.0 for both model types. On the tiny model sdpa and eager agree within 1e-6 (test).
- Licence. apache-2.0 in the front matter of README.md of both repositories, no LICENSE file
  in either, docs/licences/siglip2-2026-10-03.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from .base import Teacher, TeacherOut, band_index
from .dinov3_sat import RULE_SOURCES as _DINOV3_RULE_SOURCES
from .dinov3_sat import attention_of, scale_rgb

WEIGHTS_FILE = "model.safetensors"
# The files of a variant that the build takes from the Hub cache (snapshot_download) and that
# teachers.warm_weights reads ahead, one list for both (`weight_files`)
FILES = ("config.json", WEIGHTS_FILE, "preprocessor_config.json")
WEIGHTS_BYTES = 4_542_792_928      # the same size in both repositories
N_PARAMS = 427_888_064             # the vision tower, see the docstring
N_PARAMS_CHECKPOINT = 1_135_670_962

MEAN = (0.5, 0.5, 0.5)             # applied to RGB in 0-1, both preprocessor_config.json files
STD = (0.5, 0.5, 0.5)
ATTENTION = "sdpa"

# Every value of config.json (vision_config) that sets the architecture or the forward pass,
# plus the defaults of the config class that the file leaves out (see the docstring).
_SHARED = dict(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
               num_attention_heads=16, patch_size=16, num_channels=3,
               hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6)
FULL_CONFIG = {
    "fixres": dict(_SHARED, image_size=256),
    "naflex": dict(_SHARED, num_patches=256),
}
# The same kind of model, very small. Only for offline tests.
_TINY = dict(hidden_size=32, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2)
TINY_CONFIG = {variant: dict(config, **_TINY) for variant, config in FULL_CONFIG.items()}

RULES = ("majortom_thumbnail",)
RULE_SOURCES = {"majortom_thumbnail": _DINOV3_RULE_SOURCES["majortom_thumbnail"]}
DEFAULT_RULE = "majortom_thumbnail"

LICENCE = ("Apache-2.0, read 2026-10-03 as license apache-2.0 in the front matter of README.md "
           "of both model repositories, which hold no LICENSE file. The words are copied in "
           "docs/licences/siglip2-2026-10-03.md")

# Unexpected keys of the checkpoint that may stay unused - the text tower and the two scalars
# of the image-text loss. Anything else stops the load.
UNEXPECTED_OK = ("text_model.", "logit_scale", "logit_bias")


def unexpected_ok(key: str) -> bool:
    return key.startswith(UNEXPECTED_OK[0]) or key in UNEXPECTED_OK[1:]


def patchify(x: torch.Tensor, patch: int) -> torch.Tensor:
    """(B, C, H, W) to (B, h * w, patch * patch * C), batched, in the order of the library's
    convert_image_to_patches (one image - reshape (C, h, p, w, p), permute (1, 3, 2, 4, 0),
    reshape (h * w, p * p * C)). Patch i covers row i // w and column i % w."""
    b, c, height, width = x.shape
    if height % patch or width % patch:
        raise ValueError(f"patchify - {height} x {width} px is no multiple of patch {patch}")
    h, w = height // patch, width // patch
    x = x.reshape(b, c, h, patch, w, patch).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(b, h * w, patch * patch * c)


def with_attention(make: Callable[..., object], attention: str = ATTENTION):
    """`make(attn_implementation=attention)`, or `make()` with a printed line when the
    installed transformers refuses the argument (TypeError) or that kernel (ValueError)."""
    try:
        return make(attn_implementation=attention)
    except (TypeError, ValueError) as err:
        why = str(err)[:120].replace(":", " -")
        print(f"siglip2 - attention {attention!r} is not accepted by this transformers "
              f"({type(err).__name__}, {why}), its default kernel is used")
        return make()


def model_classes(variant: str):
    """(model class, config class) of the vision tower for a variant."""
    if variant == "fixres":
        from transformers import SiglipVisionConfig, SiglipVisionModel
        return SiglipVisionModel, SiglipVisionConfig
    if variant == "naflex":
        from transformers import Siglip2VisionConfig, Siglip2VisionModel
        return Siglip2VisionModel, Siglip2VisionConfig
    raise ValueError(f"siglip2 - unknown variant {variant!r}, known {tuple(FULL_CONFIG)}")


def load_folder(variant: str, folder: str | Path,
                n_bytes: int | None = WEIGHTS_BYTES) -> torch.nn.Module:
    """The vision tower from a local folder with config.json and model.safetensors of the
    whole model. No network.

    The weights file must have the pinned size (`n_bytes`, None skips that), the download
    itself is checked by the Hub client against the pinned revision. Every vision tensor must
    come from the file (transformers would fill a missing one with random numbers and only
    print a report), and the unused tensors may only be the ones of `UNEXPECTED_OK`.
    """
    model_class, _ = model_classes(variant)
    path = Path(folder) / WEIGHTS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    size = path.stat().st_size
    if n_bytes is not None and size != n_bytes:
        raise ValueError(f"{path} has {size:,} bytes, the pinned file has {n_bytes:,}")

    def make(**kw):
        return model_class.from_pretrained(str(folder), local_files_only=True,
                                           output_loading_info=True, dtype=torch.float32, **kw)

    # transformers prints a report of the unused text-tower tensors at every load. They are
    # expected and checked below, so the report is held back while loading.
    from transformers import logging as hf_logging
    level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        model, info = with_attention(make)
    finally:
        hf_logging.set_verbosity(level)
    bad = {}
    for kind, keys in info.items():
        keys = [str(k) for k in keys]
        if kind == "unexpected_keys":
            keys = [k for k in keys if not unexpected_ok(k)]
        if keys:
            bad[kind] = sorted(keys)
    if bad:
        raise ValueError(f"{path} and the vision tower of this transformers version do not hold "
                         f"the same tensors, nothing is used. {bad}")
    return model


def check_model(model: torch.nn.Module, variant: str, config: dict | None = None,
                n_params: int | None = N_PARAMS) -> None:
    """Stop if the model is not the verified architecture (config values, parameter count)."""
    config = FULL_CONFIG[variant] if config is None else config
    got = {k: getattr(model.config, k, None) for k in config}
    if got != config:
        diff = {k: (got[k], v) for k, v in config.items() if got[k] != v}
        raise ValueError(f"config differs from the verified one, (got, expected) = {diff}")
    n = sum(p.numel() for p in model.parameters())
    if n_params is not None and n != n_params:
        raise ValueError(f"model has {n:,} parameters, expected {n_params:,}")


class SigLIP2(Teacher):
    """Shared part of both variants. summary = pooler_output of the attention pooling head
    (`model.head`), spatial = last_hidden_state as (B, 16, 16, dim), row major."""

    variant = ""
    hf_id = ""
    hf_revision = ""
    weights_sha256 = ""
    dim = 1152
    patch = 16
    input_size = 256
    bands = ("B04", "B03", "B02")          # red, green, blue
    rules = RULES
    rule_sources = RULE_SOURCES
    licence = LICENCE
    reads_aerial = True
    aerial_mean = MEAN
    aerial_std = STD

    def __init__(self, model: torch.nn.Module, rule: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__(model, rule, device=device, dtype=dtype)
        self.dim = int(model.config.hidden_size)       # differs from 1152 only for a tiny model
        patch = model.config.patch_size
        if not isinstance(patch, int):
            raise ValueError(f"{self.name} - patch_size {patch!r} is not one integer")
        self.patch = patch
        if self.input_size % self.patch:
            raise ValueError(f"{self.name} - input {self.input_size} px is no multiple of "
                             f"patch {self.patch}")
        if not getattr(model, "use_head", False) or not hasattr(model, "head"):
            raise ValueError(f"{self.name} - the model has no pooling head, so no summary")
        self.check_config(model.config)
        self.rgb = band_index(self.bands)
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self.attention = attention_of(model)           # the kernel the model runs

    def check_config(self, config) -> None:
        raise NotImplementedError

    def prepare(self, s2: torch.Tensor) -> torch.Tensor:
        """(B, 12, H, W) digital numbers to (B, 3, 256, 256) in -1 to 1.

        The picture is made first (thumbnail rule, 0-1) and resized after, as the processor
        resizes a picture. Values stay float, they are not rounded to 8 bit.
        """
        dn = s2[:, self.rgb]
        if not torch.isfinite(dn).all():
            raise ValueError(f"{self.name} - the RGB bands hold NaN or infinite values")
        x = scale_rgb(dn, self.rule)
        size = (self.input_size, self.input_size)
        if tuple(x.shape[-2:]) != size:
            x = F.interpolate(x, size=size, mode="bilinear", antialias=True, align_corners=False)
        return (x - self.mean) / self.std

    def _grid_out(self, out) -> TeacherOut:
        tokens = out.last_hidden_state                 # (B, g * g, dim), after post_layernorm
        b, n, d = tokens.shape
        g = self.grid
        if n != g * g:
            raise RuntimeError(f"{self.name} - {n} tokens, expected {g * g} for a {g} x {g} grid")
        if out.pooler_output is None:
            raise RuntimeError(f"{self.name} - the model gave no pooler_output")
        return TeacherOut(out.pooler_output, tokens.reshape(b, g, g, d))


class SigLIP2FixRes(SigLIP2):
    """google/siglip2-so400m-patch16-256 through SiglipVisionModel, pixel_values (B, 3, 256, 256)."""

    variant = "fixres"
    name = "siglip2-so400m-256"
    hf_id = "google/siglip2-so400m-patch16-256"
    hf_revision = "e8708ab72d125807e45b36fb7d4e0aacbb59f379"      # last change 2025-02-21
    weights_sha256 = "810ddc85ac019e6b2738b9130ae3602e41eb093a848a297a0ad296ca1f39dc67"

    def check_config(self, config) -> None:
        # the position embedding has (image_size // patch) ** 2 entries and is read as is
        if int(config.image_size) != self.input_size:
            raise ValueError(f"{self.name} - the model was built for {config.image_size} px, "
                             f"the input is {self.input_size} px")

    def _forward(self, x: torch.Tensor) -> TeacherOut:
        return self._grid_out(self.model(pixel_values=x))


class SigLIP2NaFlex(SigLIP2):
    """google/siglip2-so400m-patch16-naflex through Siglip2VisionModel, one 256 px square, so
    256 patches of 3 * 16 * 16 values, a mask of ones and spatial shapes (16, 16)."""

    variant = "naflex"
    name = "siglip2-so400m-naflex"
    hf_id = "google/siglip2-so400m-patch16-naflex"
    hf_revision = "cc24074f717b612951c2dead130904ab9b65a81e"      # last change 2025-02-21
    weights_sha256 = "11a61a2068800d5f4f35cb041c1fea25de86ce87725e4c97a9ed046d0b22c076"

    def check_config(self, config) -> None:
        # num_patches is the length of the position embedding and the most patches one row
        # may hold (max_num_patches of the processor)
        n = self.grid * self.grid
        if int(config.num_patches) < n:
            raise ValueError(f"{self.name} - num_patches {config.num_patches} is below the "
                             f"{n} patches of a {self.grid} x {self.grid} grid")

    def _forward(self, x: torch.Tensor) -> TeacherOut:
        b = x.shape[0]
        g = self.grid
        pixel_values = patchify(x, self.patch)                             # (B, g * g, p * p * 3)
        mask = torch.ones(b, g * g, dtype=torch.int32, device=x.device)    # every patch is real
        shapes = torch.full((b, 2), g, dtype=torch.int64)                  # CPU, see the docstring
        out = self.model(pixel_values=pixel_values, pixel_attention_mask=mask, spatial_shapes=shapes)
        return self._grid_out(out)


CLASSES = {"fixres": SigLIP2FixRes, "naflex": SigLIP2NaFlex}


def weight_files(*, variant: str) -> tuple[tuple[str, str, str], ...]:
    """(repository, pinned revision, file name) of every file the build of `variant` reads from
    the Hub cache, `FILES` of its repository."""
    if variant not in CLASSES:
        raise ValueError(f"siglip2 - unknown variant {variant!r}, known {tuple(CLASSES)}")
    cls = CLASSES[variant]
    return tuple((cls.hf_id, cls.hf_revision, name) for name in FILES)


def build(*, variant: str, rule: str | None = None, device: str = "cuda",
          dtype: torch.dtype = torch.bfloat16, token: str | None = None,
          weights_dir: str | Path | None = None, random_init: bool = False,
          tiny: bool = False) -> Teacher:
    """Build one variant. See `teachers/__init__.py` for the other arguments.

    Real weights come from `weights_dir` (a folder with config.json and model.safetensors of
    the whole model, no network) or from the public Hugging Face repository at the pinned
    revision through the Hub cache. The token is passed on and never kept. `tiny` wins over
    `random_init`, and both win over `weights_dir`, which is then not read.
    """
    if variant not in CLASSES:                                    # before any import or download
        raise ValueError(f"siglip2 - unknown variant {variant!r}, known {tuple(CLASSES)}")
    cls = CLASSES[variant]
    rule = DEFAULT_RULE if rule is None else rule
    if rule not in RULES:
        raise ValueError(f"{cls.name} - unknown rule {rule!r}, known {RULES}")
    model_class, config_class = model_classes(variant)

    if tiny:
        model = with_attention(lambda **kw: model_class(config_class(**TINY_CONFIG[variant], **kw)))
    elif random_init:
        model = with_attention(lambda **kw: model_class(config_class(**FULL_CONFIG[variant], **kw)))
        check_model(model, variant)
    else:
        if weights_dir is None:
            from huggingface_hub import snapshot_download
            weights_dir = snapshot_download(cls.hf_id, revision=cls.hf_revision, token=token,
                                            allow_patterns=list(FILES))
        model = load_folder(variant, weights_dir)
        check_model(model, variant)
    teacher = cls(model, rule, device=device, dtype=dtype)
    print(f"{teacher.name} - attention {teacher.attention}")
    return teacher
