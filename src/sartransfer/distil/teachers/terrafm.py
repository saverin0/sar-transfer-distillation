"""TerraFM-B as a frozen teacher, reading Sentinel-2 (slot of part three) or Sentinel-1 (slot 4 of part four).

Two teachers share this module, the model and the weight loading. `TerraFMTeacher`
("terrafm-b") reads the 12 Sentinel-2 bands through one of the two Sentinel-2 input layers.
`TerraFMRadar` ("terrafm-b-radar", added 2026-10-03, slot 4) reads the
Sentinel-1 crop in dB through the input layer conv2d_s1, which the vendored PatchEmbed picks
by itself for a 2-channel input. `build(variant=None)` gives the first, `build(variant="radar")`
the second.

What is known and where it was read, all on 2026-09-29:

- Model file and weights come from the Hugging Face repository MBZUAI/TerraFM at HF_REVISION.
  Only TerraFM-B.pth is released. ViT-B/16, 768-d, 224 px, three input layers
  (Sentinel-2 L2A with 12 bands, Sentinel-2 L1C with 12 bands, Sentinel-1 with 2 channels).
- The released file sends every 12-band image to the L1C input layer. The vendored copy has
  ONE marked change, so this wrapper picks the layer (vendor/terrafm/NOTICE.md).
- Band order and scaling are NOT on the model card, in the files of the code repository or
  in the paper. Two written sources exist, both from the first author.
  1. His open pull request 947 to terratorch, a SECONDARY source. Its numbers are typed in
     below, each with its file and line.
  2. His answer in issue 3 of the code repository mbzuai-oryx/TerraFM (see ISSUE 3 below).
     The evaluation "followed the DOFA evaluation setup", and the pull request is named as
     the implementation of TerraFM in TerraTorch.
  No page says which scaling was used in pre-training, and none says that the scalings of
  the DOFA evaluation code were kept for TerraFM.

A rule is an input layer and a scaling. Stage 0 draws the features under every rule, the
choice is then fixed. Nothing is clipped under any rule.

    l2a_minmax        L2A layer, (x - (mean - 2 std)) / (4 std)      the default
    l1c_minmax        L1C layer, same numbers                        what the released file does
    l2a_standardise   L2A layer, (x - mean) / std                    control, see ISSUE 3
    l1c_standardise   L1C layer, (x - mean) / std                    control, see ISSUE 3

The radar teacher has one rule, the same mapping the scoring encoder "terrafm-s1" of
models/encoders.py applies (fixed there on 2026-09-30)

    pr947_s1_minmax   S1 layer, (s1 - mean) / std per channel in dB, then x / (2 scale) + 0.5,
                      which equals (s1 - (mean - 2 std)) / (4 std), no clipping, then 256 -> 224 px

Output: summary = class token, spatial = patch tokens, both from the last layer after the
final norm. spatial is (B, 14, 14, 768), rows first, as in the authors' `extract_feature`.
"""

from __future__ import annotations

import hashlib
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from .base import Teacher, TeacherOut, band_index
from .vendor.terrafm import TerraFM, terrafm_base

# Read from the Hugging Face API (models/MBZUAI/TerraFM, blobs) on 2026-09-29.
HF_REPO = "MBZUAI/TerraFM"
HF_REVISION = "3631173fec2c3a445a71e15985e0f3d5aaf4ecb3"
WEIGHTS_FILE = "TerraFM-B.pth"
WEIGHTS_BYTES = 451_862_010
WEIGHTS_SHA256 = "8bdb6a4cfd707a09d79f4790119daa9d76bb316b6ea756c6eb305f99d1797a06"

# ---------------------------------------------------------------------------------------
# Numbers of pull request 947 to terratorch (open, not merged), typed from its head commit
# 43ed76239163f8a4d8319e0ce4667a2d6c5bbdbc, branch feature/terrafm-backbone of
# msohaildanish/terratorch, read 2026-09-29.
#   https://github.com/msohaildanish/terratorch/blob/43ed76239163f8a4d8319e0ce4667a2d6c5bbdbc/terratorch/models/backbones/terrafm.py
# Units are digital numbers for Sentinel-2 and dB for Sentinel-1.
# The pull request does not say which data the numbers were computed on.
# ---------------------------------------------------------------------------------------
PR947_COMMIT = "43ed76239163f8a4d8319e0ce4667a2d6c5bbdbc"

# Band order. examples/confs/terrafm_test.yaml lines 18-30 (dataset_bands) and 102-114
# (backbone_bands) list B01 ... B12 without B10. It is the order of records.S2_BANDS.
# That file calls itself an L1C example and sets backbone_is_l2a false, yet it holds the
# L2A mean and std. The pull request is not consistent here. Typed as found.
PR_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")

# terratorch/models/backbones/terrafm.py line 23, TerraFM_MEAN_S2_L2A
PR_MEAN_S2_L2A = (1033.9414, 1135.3391, 1382.2553, 1526.9789, 1883.3911, 2516.2726,
                  2782.7811, 2829.3833, 2977.2936, 3004.2905, 2832.1983, 2186.6507)
# line 24, TerraFM_STD_S2_L2A. The first value (B01) is far larger than the others.
# Typed as found.
PR_STD_S2_L2A = (2761.4226, 807.365, 919.6919, 1221.4198, 1265.8146, 1323.0764,
                 1428.7845, 1459.0343, 1497.9797, 1577.071, 1738.6128, 1620.8854)
# line 25, named TerraFM_MEAN_S3_L1C there. Not used by any rule here, our data is L2A.
PR_MEAN_S2_L1C = (2153.94, 1937.191, 1885.8236, 1906.7263, 2096.8369, 2692.8083,
                  3015.8569, 2930.3957, 3243.99, 1389.849, 2915.8289, 2186.0514)
# line 26, named TerraFM_STD_S3_L1C there
PR_STD_S2_L1C = (581.6321, 619.9756, 714.7784, 1018.3657, 1021.4312, 1116.4055,
                 1265.8559, 1234.6123, 1374.9582, 608.362, 1547.9093, 1336.0316)
# lines 21 and 22, TerraFM_MEAN_S1 and TerraFM_STD_S1. The channel order is not stated there,
# the values fit (VV, VH). Used by the radar rule of TerraFMRadar (2026-10-03), not by the
# Sentinel-2 rules.
PR_MEAN_S1 = (-12.7572, -19.6423)
PR_STD_S1 = (5.8491, 5.8516)

# Min-max rule. terratorch/datasets/transforms.py lines 288-371, class MinMaxNormalize,
# min = mean - scale * std (line 315), div = 2 * scale * std (line 316), result
# (x - min) / div (line 339), no clipping. The example configuration sets scale 2.0.
PR_MINMAX_SCALE = 2.0

# rule name -> (input layer, scaling). Every rule uses the L2A numbers, Major-TOM is L2A.
RULES = {
    "l2a_minmax": ("l2a", "minmax"),
    "l1c_minmax": ("l1c", "minmax"),
    "l2a_standardise": ("l2a", "standardise"),
    "l1c_standardise": ("l1c", "standardise"),
}

# Why this default. Major-TOM Core-S2L2A is L2A and the model has an input layer trained on
# L2A. The pull request makes the L2A layer its own default (is_l2a=True, line 563) and its
# docs/guide/data.md names min-max scaling between mean - 2 std and mean + 2 std for TerraFM.
# It is a starting point from a secondary source, not a settled fact. Stage 0 draws all rules.
DEFAULT_RULE = "l2a_minmax"

# The one rule of the radar teacher (part four, 2026-10-03). The Sentinel-1 numbers of the pull
# request and its min-max scale, applied as the scoring encoder "terrafm-s1" applies them.
RADAR_RULE = "pr947_s1_minmax"

# ---------------------------------------------------------------------------------------
# ISSUE 3 of the official code repository mbzuai-oryx/TerraFM, "Training code availability",
# opened 2026-01-30, still open. One answer, by msohaildanish (collaborator of the
# repository, first author of the paper) on 2026-02-28, read 2026-09-29. Exact words
#   "For evaluation and benchmarking, we followed the DOFA evaluation setup"
#       followed by the link https://github.com/xiong-zhitong/DOFA-pytorch
#   "TerraFM is also integrated into TerraTorch, and the implementation can be found here
#    (currently as an open PR)"
#       followed by the link https://github.com/terrastackai/terratorch/pull/947
#       (the old address of pull request 947, it now leads to torchgeo/terratorch)
# What the DOFA evaluation code does, read 2026-09-29 at commit
# b915a2f6d2983c04fd08a270a09e5032e9eb91a9 (branch geofm, the newest commit, 2025-02-18)
#   src/datasets/geobench_wrapper.py   K.augmentation.Normalize(mean, std) with the mean and
#       std of each GEO-Bench dataset (lines 18, 25, 37, 57, 83). This is standardisation.
#   src/datasets/benv2_wrapper.py      (x - min) / (max - min) with fixed min and max values
#       per band (lines 214 to 265).
# That repository holds no TerraFM file (its last push is older than the TerraFM release).
# NOT stated on any page - that these scalings were kept for TerraFM, with which numbers,
# and which scaling was used in pre-training.
# ---------------------------------------------------------------------------------------
_ISSUE3_URL = "https://github.com/mbzuai-oryx/TerraFM/issues/3#issuecomment-3976648098"
_DOFA_COMMIT = "b915a2f6d2983c04fd08a270a09e5032e9eb91a9"

_PR = (f"terratorch pull request 947, commit {PR947_COMMIT}, read 2026-09-29, secondary source. "
       "The first author names this pull request as the implementation of TerraFM in TerraTorch "
       f"in issue 3 of mbzuai-oryx/TerraFM ({_ISSUE3_URL}, answer of 2026-02-28, "
       "read 2026-09-29)")
_LAYER_SOURCE = {
    "l2a": "Input layer conv2d_s2_l2a, picked through the one marked change in "
           "vendor/terrafm/terrafm.py. The pull request uses this layer by default.",
    "l1c": "Input layer conv2d_s2_l1c, the layer the released terrafm.py uses for every "
           "12-band image.",
}
_SCALING_SOURCE = {
    "minmax": "Scaling (x - (mean - 2 std)) / (4 std) without clipping, MinMaxNormalize with "
              "scale 2.0 in terratorch/datasets/transforms.py and "
              "examples/confs/terrafm_test.yaml.",
    "standardise": "Scaling (x - mean) / std, a control. It is not in the pull request. In issue "
                   "3 the first author says the evaluation followed the DOFA evaluation setup. "
                   f"In xiong-zhitong/DOFA-pytorch (commit {_DOFA_COMMIT}) "
                   "src/datasets/geobench_wrapper.py standardises with the mean and std of each "
                   "dataset, and src/datasets/benv2_wrapper.py scales with fixed min and max "
                   "values. No page says that these scalings were kept for TerraFM or which "
                   "scaling was used in pre-training. This rule takes the mean and std of the "
                   "pull request, not those of a dataset.",
}
_RADAR_SOURCE = (
    "Input layer conv2d_s1, which the vendored PatchEmbed picks by itself for a 2-channel input "
    "(no marked change is needed for it). Scaling per channel in dB, (s1 - mean) / std and then "
    f"x / (2 * {PR_MINMAX_SCALE}) + 0.5, which equals (x - (mean - 2 std)) / (4 std) without "
    "clipping, the same mapping the scoring encoder terrafm-s1 of models/encoders.py applies on "
    "its standardised tiles. Sentinel-1 mean and std (VV, VH) from "
    "terratorch/models/backbones/terrafm.py lines 21 and 22, the scale from MinMaxNormalize in "
    f"terratorch/datasets/transforms.py. {_PR}. Added on 2026-10-03 (slot 4).")


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def check_weights_file(path: str | Path, n_bytes: int = WEIGHTS_BYTES,
                       sha256: str = WEIGHTS_SHA256) -> None:
    """Stop unless the file has the pinned size (bytes) and the pinned SHA-256."""
    size = Path(path).stat().st_size
    if size != n_bytes:
        raise RuntimeError(f"terrafm-b: {path} has {size} bytes, expected {n_bytes}")
    found = sha256_of(path)
    if found != sha256:
        raise RuntimeError(f"terrafm-b: {path} has SHA-256 {found}, expected {sha256}")


def load_weights(model: torch.nn.Module, path: str | Path) -> None:
    """Load a state dict into `model`. Strict: prints and fails on every key that does not fit.

    Expected to be unused: nothing. The key list of the released TerraFM-B.pth was read on
    2026-09-29. It holds exactly the 164 tensors of the encoder (112,951,296 values), no head
    and no decoder, so no key may be left over and none may be missing. One rule uses one of
    the three input layers. The two others are loaded and checked too, only not used.
    The file is read with weights_only=True, so it can hold tensors and nothing that runs code.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError(f"terrafm-b: {path} holds a {type(state).__name__}, "
                           "expected a state dict")
    want = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    have = {k: tuple(v.shape) if torch.is_tensor(v) else None for k, v in state.items()}
    missing = sorted(set(want) - set(have))
    unexpected = sorted(set(have) - set(want))
    shape = sorted(k for k in set(want) & set(have) if want[k] != have[k])
    if missing or unexpected or shape:
        print(f"terrafm-b: key check FAILED for {path}")
        print(f"  missing in the file ({len(missing)}) {missing}")
        print(f"  unexpected in the file ({len(unexpected)}) {unexpected}")
        print(f"  shape differs ({len(shape)}) {[(k, have[k], want[k]) for k in shape]}")
        raise RuntimeError(f"terrafm-b: {len(missing)} missing, {len(unexpected)} unexpected, "
                           f"{len(shape)} wrong-shape keys in {path}, nothing was loaded")
    model.load_state_dict(state, strict=True)
    print(f"terrafm-b: loaded {len(want)} tensors from {path}, no key missing, none unexpected")


class TerraFMTeacher(Teacher):
    name = "terrafm-b"
    dim = 768
    patch = 16
    input_size = 224
    bands = PR_BANDS
    rules = tuple(RULES)
    rule_sources = {
        name: f"{_LAYER_SOURCE[layer]} {_SCALING_SOURCE[scaling]} L2A mean and std from "
              f"terratorch/models/backbones/terrafm.py lines 23 and 24. {_PR}."
        for name, (layer, scaling) in RULES.items()
    }
    licence = ("Apache-2.0 in the model card front matter, MIT in the author's answer to issue 5 "
               "of mbzuai-oryx/TerraFM, no LICENSE file in the code repository "
               "(read 2026-09-29, docs/licences/terrafm-2026-09-29.md)")
    weights = "unknown"        # build() sets it, "random" or the SHA-256 of the loaded file

    def __init__(self, model: TerraFM, rule: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__(model, rule, device, dtype)
        self.layer, scaling = RULES[rule]
        # sizes come from the model, so a tiny test model reports its own
        self.dim = model.embed_dim
        self.patch = model.patch_embed.patch_size
        self.input_size = model.patch_embed.img_size
        self.pick = band_index(self.bands)
        mean = torch.tensor(PR_MEAN_S2_L2A, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        std = torch.tensor(PR_STD_S2_L2A, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        if scaling == "minmax":
            self.shift, self.scale = mean - PR_MINMAX_SCALE * std, 2.0 * PR_MINMAX_SCALE * std
        else:
            self.shift, self.scale = mean, std

    def prepare(self, s2: torch.Tensor) -> torch.Tensor:
        """(B, 12, H, W) digital numbers -> (B, 12, input_size, input_size), scaled, not clipped."""
        x = (s2[:, self.pick] - self.shift) / self.scale
        size = (self.input_size, self.input_size)
        if tuple(x.shape[-2:]) != size:
            x = F.interpolate(x, size=size, mode="bilinear", antialias=True, align_corners=False)
        return x

    def _forward(self, x: torch.Tensor) -> TeacherOut:
        # read by the changed line of the vendored file. Set on every call, so two teachers
        # with different rules can share one model.
        self.model.is_l2a = self.layer == "l2a"
        ((tokens, cls),) = self.model.get_intermediate_layers(x, n=1, return_class_token=True,
                                                              norm=True)
        return TeacherOut(cls, tokens.reshape(x.shape[0], self.grid, self.grid, self.dim))


class TerraFMRadar(Teacher):
    """TerraFM-B reading the Sentinel-1 crop through its own input layer conv2d_s1 (slot 4).

    Input (B, 2, H, W), VV and VH in dB, through `radar`. It reads no Sentinel-2 crop and gives
    no aerial target (the radar teacher covers the rows of the modes s1 and
    both only). The one rule maps the dB values as the scoring encoder "terrafm-s1" does, so a
    student target and a scored feature come from the same input range.
    """

    name = "terrafm-b-radar"
    dim = 768
    patch = 16
    input_size = 224
    bands = ()                             # no Sentinel-2 band is read
    rules = (RADAR_RULE,)
    rule_sources = {RADAR_RULE: _RADAR_SOURCE}
    licence = TerraFMTeacher.licence
    reads_s2 = False
    reads_aerial = False
    reads_radar = True
    weights = "unknown"        # build() sets it, "random" or the SHA-256 of the loaded file

    def __init__(self, model: TerraFM, rule: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__(model, rule, device, dtype)
        self.layer = "s1"
        # sizes come from the model, so a tiny test model reports its own
        self.dim = model.embed_dim
        self.patch = model.patch_embed.patch_size
        self.input_size = model.patch_embed.img_size
        self.mean = torch.tensor(PR_MEAN_S1, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(PR_STD_S1, dtype=torch.float32, device=device).view(1, -1, 1, 1)

    def prepare_radar(self, s1: torch.Tensor) -> torch.Tensor:
        """(B, 2, H, W) dB -> (B, 2, input_size, input_size), standardised then min-max mapped, not clipped."""
        z = (s1 - self.mean) / self.std                       # standardised, as the scoring tiles arrive
        x = z / (2.0 * PR_MINMAX_SCALE) + 0.5                 # the mapping of models/encoders.py
        size = (self.input_size, self.input_size)
        if tuple(x.shape[-2:]) != size:
            x = F.interpolate(x, size=size, mode="bilinear", antialias=True, align_corners=False)
        return x

    def _forward(self, x: torch.Tensor) -> TeacherOut:
        # the vendored PatchEmbed sends a 2-channel input through conv2d_s1, whatever is_l2a says
        ((tokens, cls),) = self.model.get_intermediate_layers(x, n=1, return_class_token=True,
                                                              norm=True)
        return TeacherOut(cls, tokens.reshape(x.shape[0], self.grid, self.grid, self.dim))


# variant -> (teacher class, its default rule). None keeps the Sentinel-2 teacher as it was.
VARIANTS = {None: (TerraFMTeacher, DEFAULT_RULE), "radar": (TerraFMRadar, RADAR_RULE)}


def weight_files(*, variant: str | None = None) -> tuple[tuple[str, str, str], ...]:
    """(repository, pinned revision, file name) of the file the build reads from the Hub cache,
    for teachers.warm_weights. Both variants read the same file."""
    if variant not in VARIANTS:
        raise ValueError(f"terrafm-b: unknown variant {variant!r}, known: None (Sentinel-2) and 'radar'")
    return ((HF_REPO, HF_REVISION, WEIGHTS_FILE),)


def build(*, variant: str | None = None, rule: str | None = None, device: str = "cuda",
          dtype: torch.dtype = torch.bfloat16, token: str | None = None,
          weights_dir: str | Path | None = None, random_init: bool = False,
          tiny: bool = False) -> Teacher:
    """Build the frozen teacher.

    variant       None = TerraFMTeacher reading Sentinel-2 ("terrafm-b"), "radar" = TerraFMRadar
                  reading Sentinel-1 ("terrafm-b-radar"). Same weights, same checks.
    token         not used. The repository is public and the download is anonymous.
    weights_dir   folder that holds TerraFM-B.pth. Nothing is downloaded when it is given.
                  Without it the file comes from the Hugging Face cache, pinned by revision.
    The weights file is checked by size and SHA-256 before it is loaded.
    With `random_init` or `tiny` no file is loaded, a line is printed that says so, and the
    teacher's attribute `weights` is "random". Else it is the SHA-256 of the loaded file.
    """
    if variant not in VARIANTS:     # before any download
        raise ValueError(f"terrafm-b: unknown variant {variant!r}, known: None (Sentinel-2) and 'radar'")
    cls, default_rule = VARIANTS[variant]
    rule = default_rule if rule is None else rule
    if rule not in cls.rules:       # before any download
        raise ValueError(f"{cls.name}: unknown rule {rule!r}, known: {cls.rules}")
    path = None
    if not tiny and not random_init:        # found and checked before the model is built
        if weights_dir is not None:
            path = Path(weights_dir) / WEIGHTS_FILE
            if not path.is_file():
                raise FileNotFoundError(f"terrafm-b: {path} not found, nothing is downloaded "
                                        "when weights_dir is given")
        else:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(HF_REPO, WEIGHTS_FILE, revision=HF_REVISION,
                                        token=False))
        check_weights_file(path)
    if tiny:
        model = TerraFM(img_size=[32], patch_size=16, embed_dim=24, depth=2, num_heads=2,
                        mlp_ratio=2, qkv_bias=True,
                        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))
    else:
        model = terrafm_base()
    if path is not None:
        load_weights(model, path)
    teacher = cls(model, rule, device=device, dtype=dtype)
    # the file passed check_weights_file, so its SHA-256 is the pinned one
    teacher.weights = "random" if path is None else WEIGHTS_SHA256
    if path is None:
        unused = "" if weights_dir is None else f" The given weights_dir {weights_dir} is not used."
        print(f"{teacher.name}: RANDOM weights, no file loaded.{unused}")
    if variant == "radar":
        print(f"{teacher.name}: rule {rule}, input layer conv2d_s1, picked by the vendored code for "
              "the 2-channel input. The two Sentinel-2 input layers are part of the model but are "
              "not used.")
    else:
        print(f"{teacher.name}: rule {rule}, input layer conv2d_s2_{teacher.layer}. The two other "
              "input layers are part of the model but are not used.")
    return teacher
