"""Teacher DINOv3 ViT-L/16 SAT-493M, frozen, reading Sentinel-2 RGB.

Every fact below was read on its primary page on 2026-09-29. "Official code" means
github.com/facebookresearch/dinov3 at commit 6876159a11b4df116f30f667f8c9888617df0751.

- Model. ViT-L/16 with patch 16, embedding dimension 1024, 4 register tokens, 16 heads, MLP,
  RoPE (MODEL_CARD.md). 24 layers, MLP ratio 4 (dinov3/hub/backbones.py, dinov3_vitl16).
  303,129,600 parameters (file information of the Hugging Face model repository).
- Training data. 493 million RGB images of 512 x 512 px from Maxar at 0.6 m (MODEL_CARD.md).
- Input. RGB in 0-1, then mean (0.430, 0.411, 0.296) and std (0.213, 0.156, 0.143)
  (README.md, "For models using the SAT-493M weights").
- The model repository is gated, no file of it can be downloaded without a token. Its file
  listing is public and gives the git blob id and the size of every file. A text with the
  same blob id is the file, so two small files could be rebuilt and are known that way
  (the test does the rebuild).
  - config.json, 745 bytes. Every value that sets the architecture or the forward pass is
    in `FULL_CONFIG`. They agree with the conversion script of transformers and with the
    official code (norm eps 1e-5, RoPE base 100, no bias on the key projection).
  - preprocessor_config.json, 585 bytes. It holds the satellite mean and std above, a
    rescale by 1/255 and a bilinear resize to 224 px. The conversion script builds the
    processor with the ImageNet mean and std for every model, so the file was corrected
    after the conversion. It is a second source for `MEAN` and `STD`.
- The Hugging Face image processor is not used. It resizes to 224 px and expects a picture,
  the input here is a float tensor that is already 256 px.
- Output. `last_hidden_state` holds all tokens after the final norm, in the order class token,
  register tokens, patch tokens row by row (transformers, modeling_dinov3_vit.py). The official
  satellite ViT-L has a second norm, `local_cls_norm`. It is used on local crops in training
  only and the conversion drops it, so one norm serves every token at inference.
- With real weights `check_model` compares the loaded config with `FULL_CONFIG` and counts
  the parameters, and `load_folder` stops when a tensor of the file did not reach the model.
- Licence. DINOv3 License, last updated August 19, 2025. LICENSE.md of the official code and
  LICENSE.md of the model repository are the same file (same git blob id).
- Attention kernel (2026-10-01, user). The model is built with attn_implementation="sdpa"
  (`ATTENTION`), the fused kernel of torch.nn.functional.scaled_dot_product_attention, not
  the plain matmul and softmax ("eager"). transformers picks sdpa by itself where a model
  supports it (4.56 and 5.x, checked on 5.17.0), so the argument pins that choice, and the
  teacher records what the model runs (`attention`, printed by `build`). A transformers that
  refuses the argument or the kernel gets its default with a printed line (`with_attention`).
  The outputs under sdpa and under eager agree within 1e-4 on the tiny model and within 1e-3
  with the real weights in float32 on the CPU (tests/test_teacher_dinov3.py, which prints the
  measured difference; on 2026-10-01 it was 1.4e-6 on the tiny model and 7.9e-6 with the real
  weights on a 256 px input, the CPU forward 1.22 s under sdpa against 1.16 s under eager per
  image). CPU timings say nothing about the A100, where the gain is in the fused bf16 kernel.
  The scoring notebook loads the same model through AutoModel.from_pretrained without the
  argument (models/encoders.py, as part two did) and gets the library default, sdpa on 5.17.0
  with the same numbers (checked in the test), so the pin changes nothing the scoring sees.

The open point is how Sentinel-2 digital numbers become RGB in 0-1. No primary page fixes it.

- The DINOv3 paper (arXiv 2508.10104v1, section 8.3 and appendix D.13) says its GEO-Bench runs
  use "RGB-only input". It gives no scaling for Sentinel-2.
- The official code holds no GEO-Bench and no Sentinel-2 evaluation.
- Issues 9 and 310 of the official code. The answers come from the accounts JohnMBrandt and
  se-yi, names that match two authors of the paper. Issue 9 says data that is not 8 bit
  "should have just been converted to uint8 by scaling to 0-255", and that the writer does not
  know the details. Issue 310 says the GEO-Bench band statistics can be used. Both point to
  statistics of each dataset, so they give no number for Major-TOM. Such a rule is not
  offered here, because its numbers would be our own.

So the scaling is a named rule, see `RULES`, `RULE_SOURCES` and `DEFAULT_RULE`. Stage 0 draws
the features under every rule (check 5), then one rule is fixed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from .base import Teacher, TeacherOut, band_index

HF_ID = "facebook/dinov3-vitl16-pretrain-sat493m"
# Revision, size and SHA-256 are in the public model information of Hugging Face
# (huggingface.co/api/models/<id>/revision/<revision>?blobs=true), read without a token.
# The file itself, its headers and the hash in the tree listing need a token.
HF_REVISION = "f692fa42da72c6797b67cd73494a168d1120d3ee"     # last change 2025-08-19
WEIGHTS_FILE = "model.safetensors"                            # 1,212,559,808 bytes
# The files the build takes from the Hub cache (snapshot_download) and that teachers.warm_weights
# reads ahead, one list for both (`weight_files`)
FILES = ("config.json", WEIGHTS_FILE)
WEIGHTS_SHA256 = "4e6356d992c1301b5e7c275f465e47296c5c4ad17052e262b29fc21e82ccc698"
N_PARAMS = 303_129_600

MEAN = (0.430, 0.411, 0.296)       # applied to RGB in 0-1
STD = (0.213, 0.156, 0.143)
QUANTIFICATION = 10000.0           # digital numbers per unit of reflectance
ATTENTION = "sdpa"                 # the attention kernel asked of transformers, see the docstring

# Every value of config.json that sets the architecture or the forward pass at inference.
# The first 8 are set by transformers, convert_dinov3_vit_to_hf.py, branch "vitl16_sat493m"
# (commit 80a79d1bef2b). The others are what the rebuilt config.json holds (see the docstring).
FULL_CONFIG = dict(patch_size=16, hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
                   num_attention_heads=16, num_register_tokens=4, use_gated_mlp=False,
                   hidden_act="gelu", num_channels=3, layer_norm_eps=1e-5, rope_theta=100.0,
                   query_bias=True, key_bias=False, value_bias=True, proj_bias=True,
                   mlp_bias=True)
# The same kind of model, very small. Only for offline tests.
TINY_CONFIG = dict(FULL_CONFIG, hidden_size=32, intermediate_size=128, num_hidden_layers=2,
                   num_attention_heads=2)

_MAJORTOM = "official Major-TOM code, github.com/ESA-PhiLab/Major-TOM at commit f7078cd0e9d8"

RULES = ("reflectance", "majortom_embedder", "majortom_thumbnail")
RULE_SOURCES = {
    "reflectance": (
        "digital number / 10000, cut to 0-1, no brightening. ESA gives the formula "
        "L2A_SR = (L2A_DN + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE on "
        "sentiwiki.copernicus.eu/web/s2-products and leaves the number to the product metadata. "
        f"10000 is the number the {_MAJORTOM} divides by in MajorTOM/extras/thumbnail_s2.py "
        "and in MajorTOM/embedder/models/DINOv2_S2RGB.py and SigLIP_S2RGB.py"),
    "majortom_embedder": (
        "2.5 * digital number / 10000, cut to 0-1. Function normalize in "
        "MajorTOM/embedder/models/DINOv2_S2RGB.py and SigLIP_S2RGB.py, "
        f"{_MAJORTOM}. Major-TOM feeds Sentinel-2 RGB to two photo models this way"),
    "majortom_thumbnail": (
        "1.3 * (digital number / 10000) ** 0.6, cut to 0-1. Function s2l2a_thumbnail "
        "(gain 1.3, gamma 0.6) in MajorTOM/extras/thumbnail_s2.py, "
        f"{_MAJORTOM}. The file says the thumbnails of the Core dataset were computed this way"),
}

# Reason for the default. The model was trained on RGB pictures (Maxar imagery, no page gives
# the bit depth or how they were stretched), and its own mean (0.430, 0.411, 0.296) says
# those pictures are bright. A reflectance of 0.05, 0.10 or 0.20 becomes 0.05, 0.10, 0.20
# under "reflectance", 0.13, 0.25, 0.50 under "majortom_embedder" and 0.22, 0.33, 0.49 under
# "majortom_thumbnail". The thumbnail rule lands nearest to the model's mean, and it is the
# rule the authors of the data use to turn a cell into an RGB picture.
# This is an argument, not a measurement. It is a starting value until stage 0 has drawn the
# features under every rule and compared the input statistics on real cells.
DEFAULT_RULE = "majortom_thumbnail"

LICENCE = ("DINOv3 License, last updated August 19, 2025, read 2026-09-29 in LICENSE.md of the "
           "official code and of the model repository. Derivative works may be shared only "
           "under the same terms and with a copy of the licence. Distillation is not mentioned")


def weight_files() -> tuple[tuple[str, str, str], ...]:
    """(repository, pinned revision, file name) of every file the build reads from the Hub cache."""
    return tuple((HF_ID, HF_REVISION, name) for name in FILES)


def scale_rgb(dn: torch.Tensor, rule: str) -> torch.Tensor:
    """Digital numbers (..., H, W), offset already removed, to values in 0-1 under one rule.

    Negative numbers count as 0. The sources read uint16 and never meet them, here they would
    turn into NaN under the power of the thumbnail rule.
    """
    r = dn.clamp_min(0) / QUANTIFICATION
    if rule == "reflectance":
        out = r
    elif rule == "majortom_embedder":
        out = 2.5 * r
    elif rule == "majortom_thumbnail":
        out = 1.3 * r ** 0.6
    else:
        raise ValueError(f"unknown rule {rule!r}, known: {RULES}")
    return out.clamp(0, 1)


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file as lower-case hex, read in blocks of 8 MB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def with_attention(make: Callable[..., object], attention: str = ATTENTION):
    """`make(attn_implementation=attention)`, or `make()` with a printed line when the
    installed transformers refuses the argument (TypeError) or that kernel for this model
    (ValueError). `make` builds or loads the model."""
    try:
        return make(attn_implementation=attention)
    except (TypeError, ValueError) as err:
        print(f"{DINOv3Sat.name}: attention {attention!r} is not accepted by this transformers "
              f"({type(err).__name__}, {str(err)[:120]}), its default kernel is used")
        return make()


def attention_of(model: torch.nn.Module) -> str:
    """The attention kernel the model runs, as its config records it ("default" when the
    installed transformers does not record one)."""
    return str(getattr(model.config, "_attn_implementation", None) or "default")


def load_folder(folder: str | Path, sha256: str = WEIGHTS_SHA256,
                attention: str | None = ATTENTION) -> torch.nn.Module:
    """The model from a local folder with config.json and model.safetensors. No network.

    The weights file must have the pinned SHA-256, else nothing is loaded. Every tensor of
    the file must reach the model and every tensor of the model must come from the file.
    transformers fills a tensor it does not find with random numbers and only prints a
    report, so a library version with other tensor names would give a random teacher.
    `attention` is asked of transformers (`with_attention`), None asks for nothing.
    """
    path = Path(folder) / WEIGHTS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    got = file_sha256(path)
    if got != sha256:
        raise ValueError(f"{path} has SHA-256 {got}, expected {sha256}")
    from transformers import AutoModel

    def make(**kw):
        return AutoModel.from_pretrained(str(folder), local_files_only=True,
                                         output_loading_info=True, **kw)

    model, info = make() if attention is None else with_attention(make, attention)
    bad = {k: sorted(str(item) for item in v) for k, v in info.items() if v}
    if bad:
        raise ValueError(f"{path} and the model of this transformers version do not hold the "
                         f"same tensors, nothing is used. {bad}")
    return model


def check_model(model: torch.nn.Module, config: dict | None = None,
                n_params: int | None = N_PARAMS) -> None:
    """Stop if the model is not the verified architecture (config values, parameter count)."""
    config = FULL_CONFIG if config is None else config
    got = {k: getattr(model.config, k, None) for k in config}
    if got != config:
        diff = {k: (got[k], v) for k, v in config.items() if got[k] != v}
        raise ValueError(f"config differs from the verified one, (got, expected) = {diff}")
    n = sum(p.numel() for p in model.parameters())
    if n_params is not None and n != n_params:
        raise ValueError(f"model has {n:,} parameters, expected {n_params:,}")


class DINOv3Sat(Teacher):
    """summary = class token, spatial = patch tokens, both after the final norm."""

    name = "dinov3-l-sat"
    dim = 1024
    patch = 16
    input_size = 256
    bands = ("B04", "B03", "B02")          # red, green, blue
    rules = RULES
    rule_sources = RULE_SOURCES
    licence = LICENCE
    reads_aerial = True                    # aerial RGB as 8 bit / 255, then MEAN and STD
    aerial_mean = MEAN
    aerial_std = STD

    def __init__(self, model: torch.nn.Module, rule: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__(model, rule, device=device, dtype=dtype)
        self.dim = int(model.config.hidden_size)       # differs from 1024 only for a tiny model
        self.patch = int(model.config.patch_size)
        if self.input_size % self.patch:
            raise ValueError(f"{self.name}: input {self.input_size} px is no multiple of "
                             f"patch {self.patch}")
        self.rgb = band_index(self.bands)
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self.attention = attention_of(model)           # the kernel the model runs

    def prepare(self, s2: torch.Tensor) -> torch.Tensor:
        """(B, 12, H, W) digital numbers to (B, 3, 256, 256) normalised RGB.

        The picture is made first and resized after, as the README resizes a picture.
        Values stay float, they are not rounded to 8 bit.
        """
        dn = s2[:, self.rgb]
        if not torch.isfinite(dn).all():
            raise ValueError(f"{self.name}: the RGB bands hold NaN or infinite values")
        x = scale_rgb(dn, self.rule)
        size = (self.input_size, self.input_size)
        if tuple(x.shape[-2:]) != size:
            x = F.interpolate(x, size=size, mode="bilinear", antialias=True, align_corners=False)
        return (x - self.mean) / self.std

    def _forward(self, x: torch.Tensor) -> TeacherOut:
        tokens = self.model(pixel_values=x).last_hidden_state     # (B, prefix + g * g, dim)
        b, n, d = tokens.shape
        g = self.grid
        prefix = n - g * g                                        # inferred from the count
        expected = 1 + int(self.model.config.num_register_tokens)
        if prefix != expected:
            raise RuntimeError(f"{self.name}: {n} tokens for a {g} x {g} grid leave {prefix} "
                               f"prefix tokens, the config says {expected} (1 class + registers)")
        return TeacherOut(tokens[:, 0], tokens[:, prefix:].reshape(b, g, g, d))


def build(*, rule: str | None = None, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
          token: str | None = None, weights_dir: str | Path | None = None,
          random_init: bool = False, tiny: bool = False) -> Teacher:
    """Build the teacher. See `teachers/__init__.py` for the arguments.

    Real weights come from `weights_dir` (a folder with config.json and model.safetensors, no
    network) or from the gated Hugging Face repository at the pinned revision, which needs a
    token with access. The token is passed on and never kept. `tiny` wins over `random_init`,
    and both win over `weights_dir`, which is then not read.
    """
    rule = DEFAULT_RULE if rule is None else rule
    if rule not in RULES:                                         # before any download
        raise ValueError(f"{DINOv3Sat.name}: unknown rule {rule!r}, known: {RULES}")
    try:
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
    except ImportError as e:
        raise ImportError("DINOv3 needs transformers 4.56.0 or newer") from e

    if tiny:
        model = with_attention(lambda **kw: DINOv3ViTModel(DINOv3ViTConfig(**TINY_CONFIG, **kw)))
    elif random_init:
        model = with_attention(lambda **kw: DINOv3ViTModel(DINOv3ViTConfig(**FULL_CONFIG, **kw)))
        check_model(model)
    else:
        if weights_dir is None:
            from huggingface_hub import snapshot_download
            weights_dir = snapshot_download(HF_ID, revision=HF_REVISION, token=token,
                                            allow_patterns=list(FILES))
        model = load_folder(weights_dir)
        check_model(model)
    teacher = DINOv3Sat(model, rule, device=device, dtype=dtype)
    print(f"{teacher.name}: attention {teacher.attention}")
    return teacher
