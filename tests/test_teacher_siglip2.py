"""Teachers SigLIP2 FixRes and NaFlex on tiny random models, patchification, outputs, guards,
and the real weights when they are in the Hub cache.

Run as a script - python tests/test_teacher_siglip2.py. HF_HOME names the model cache (the
project's weights/hf_home); without it an empty folder on D: is used and the real-weight
checks print a skip line. No network is used.
"""
import _setup

import contextlib
import gc
import hashlib
import io
import json
import os
import shutil
import time
from pathlib import Path

# No network. The cache is HF_HOME as given, else an empty folder on D:, set before transformers.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", str(_setup.TMP_BASE / "siglip2_hf_home"))
HF_HOME = Path(os.environ["HF_HOME"])
HF_HOME.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import Siglip2Config, Siglip2Model, Siglip2TextConfig  # noqa: E402
from transformers import Siglip2VisionConfig, Siglip2VisionModel  # noqa: E402
from transformers import SiglipConfig, SiglipModel, SiglipTextConfig, SiglipTextModel  # noqa: E402
from transformers import SiglipVisionConfig, SiglipVisionModel  # noqa: E402
from transformers import logging as hf_logging  # noqa: E402
from transformers.models.siglip2.image_processing_siglip2 import convert_image_to_patches  # noqa: E402

from sartransfer.distil.teachers import TEACHERS, Teacher, build_teacher  # noqa: E402
from sartransfer.distil.teachers import dinov3_sat as D  # noqa: E402
from sartransfer.distil.teachers import siglip2 as S  # noqa: E402

torch.manual_seed(0)
CPU = dict(device="cpu", dtype=torch.float32)
VARIANTS = ("fixres", "naflex")
NAMES = {"fixres": "siglip2-so400m-256", "naflex": "siglip2-so400m-naflex"}


def raises(kind, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except kind:
        return True
    return False


def printed(fn, *args, **kw):
    """What `fn` returns and what it printed."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = fn(*args, **kw)
    return result, out.getvalue()


def dn_batch(b=2, size=256, low=0, high=9000):
    """Random digital numbers (b, 12, size, size) float32."""
    return torch.randint(low, high, (b, 12, size, size)).float()


def aerial_batch(b=2, size=256):
    """Random aerial 8-bit values as float (b, 4, size, size), RGB and NIR."""
    return torch.randint(0, 256, (b, 4, size, size)).float()


def finite(out):
    return bool(torch.isfinite(out.summary).all()) and bool(torch.isfinite(out.spatial).all())


# ---- facts typed into the module -----------------------------------------------------------
assert TEACHERS["siglip2-so400m-256"] == ("siglip2", {"variant": "fixres"})
assert TEACHERS["siglip2-so400m-naflex"] == ("siglip2", {"variant": "naflex"})
assert set(S.CLASSES) == set(VARIANTS) == set(S.FULL_CONFIG) == set(S.TINY_CONFIG)
FR, NF = S.SigLIP2FixRes, S.SigLIP2NaFlex
assert issubclass(FR, S.SigLIP2) and issubclass(NF, S.SigLIP2) and issubclass(S.SigLIP2, Teacher)
assert (FR.variant, FR.name, FR.hf_id) == ("fixres", NAMES["fixres"], "google/siglip2-so400m-patch16-256")
assert (NF.variant, NF.name, NF.hf_id) == ("naflex", NAMES["naflex"], "google/siglip2-so400m-patch16-naflex")
# revisions, file size and SHA-256 as the public model information gives them (2026-10-03)
assert FR.hf_revision == "e8708ab72d125807e45b36fb7d4e0aacbb59f379"
assert NF.hf_revision == "cc24074f717b612951c2dead130904ab9b65a81e"
assert FR.weights_sha256 == "810ddc85ac019e6b2738b9130ae3602e41eb093a848a297a0ad296ca1f39dc67"
assert NF.weights_sha256 == "11a61a2068800d5f4f35cb041c1fea25de86ce87725e4c97a9ed046d0b22c076"
assert S.WEIGHTS_FILE == "model.safetensors" and S.WEIGHTS_BYTES == 4_542_792_928
assert S.N_PARAMS == 427_888_064 and S.N_PARAMS_CHECKPOINT == 1_135_670_962
assert S.MEAN == (0.5, 0.5, 0.5) and S.STD == (0.5, 0.5, 0.5)
for C in (FR, NF):
    assert (C.dim, C.patch, C.input_size) == (1152, 16, 256) and C.bands == ("B04", "B03", "B02")
    assert C.reads_s2 and C.reads_aerial and not C.reads_radar
    assert C.aerial_mean == S.MEAN and C.aerial_std == S.STD
    assert C.rules == S.RULES == ("majortom_thumbnail",) and C.rule_sources is S.RULE_SOURCES
    assert C.licence is S.LICENCE
assert S.DEFAULT_RULE == "majortom_thumbnail" and S.DEFAULT_RULE in S.RULES
assert S.RULE_SOURCES["majortom_thumbnail"] == D.RULE_SOURCES["majortom_thumbnail"]
for text in (*S.RULE_SOURCES.values(), S.LICENCE):
    assert len(text) > 40 and ":" not in text, text
assert S.LICENCE.startswith("Apache-2.0") and "read 2026-10-03" in S.LICENCE
assert "siglip2-2026-10-03.md" in S.LICENCE
assert S.UNEXPECTED_OK == ("text_model.", "logit_scale", "logit_bias")
assert S.unexpected_ok("text_model.encoder.layers.0.mlp.fc1.weight") and S.unexpected_ok("logit_bias")
assert not S.unexpected_ok("vision_model.head.probe") and not S.unexpected_ok("logit_scale2")
assert S.ATTENTION == "sdpa" and "2026-10-03" in S.__doc__ and "row major" in S.__doc__
SHARED = dict(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27, num_attention_heads=16,
              patch_size=16, num_channels=3, hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6)
assert S.FULL_CONFIG == {"fixres": dict(SHARED, image_size=256), "naflex": dict(SHARED, num_patches=256)}
for variant in VARIANTS:
    assert S.TINY_CONFIG[variant] == dict(S.FULL_CONFIG[variant], hidden_size=32, intermediate_size=128,
                                          num_hidden_layers=2, num_attention_heads=2)
# the defaults of the config classes that config.json leaves out (transformers 5.17.0)
assert SiglipVisionConfig().patch_size == 16 and SiglipVisionConfig().hidden_act == "gelu_pytorch_tanh"
assert Siglip2VisionConfig().patch_size == 16 and Siglip2VisionConfig().num_patches == 256
assert SiglipVisionConfig().layer_norm_eps == 1e-6 and Siglip2VisionConfig().layer_norm_eps == 1e-6
print("facts - ids, revisions, size, parameter counts, mean and std, configs typed as read on 2026-10-03")


# ---- the parameter count of the vision tower and of the whole checkpoint --------------------
def layer_params(d, m):
    """One encoder layer - attention (4 projections with bias), 2 LayerNorms, MLP fc1 and fc2."""
    return (4 * d * d + 4 * d) + 4 * d + (d * m + m) + (m * d + d)


def vision_params(c, grid_positions):
    d, m, n = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"]
    embed = (c["num_channels"] * c["patch_size"] ** 2 * d + d) + grid_positions * d
    head = d + (3 * d * d + 3 * d) + (d * d + d) + 2 * d + (d * m + m) + (m * d + d)
    return embed + n * layer_params(d, m) + 2 * d + head


def text_params(d, m, n, vocab, positions):
    return vocab * d + positions * d + n * layer_params(d, m) + 2 * d + (d * d + d)


def positions_of(variant, c):
    return (c["image_size"] // c["patch_size"]) ** 2 if variant == "fixres" else c["num_patches"]


# the formula holds on real modules (tiny), then gives the typed count on the full config
# a tiny text tower (token ids inside its 100-word vocabulary, else transformers warns)
TEXT_TINY = dict(hidden_size=32, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
                 vocab_size=100, max_position_embeddings=16, pad_token_id=0, bos_token_id=1,
                 eos_token_id=2)
tiny_text = SiglipTextModel(SiglipTextConfig(**TEXT_TINY, projection_size=32))
assert text_params(32, 128, 2, 100, 16) == sum(p.numel() for p in tiny_text.parameters())
del tiny_text
for variant in VARIANTS:
    c = S.FULL_CONFIG[variant]
    assert vision_params(c, positions_of(variant, c)) == S.N_PARAMS == 427_888_064, variant
# text_config of both config.json files - hidden 1152, intermediate 4304, 16 heads, 27 layers,
# vocab 256000, projection 1152; max_position_embeddings is not in the file, default 64
TEXT = text_params(1152, 4304, 27, 256000, 64)
assert TEXT == 707_782_896 and S.N_PARAMS + TEXT + 2 == S.N_PARAMS_CHECKPOINT
print(f"parameters - vision tower {S.N_PARAMS:,} by the formula, with the text tower {TEXT:,} and "
      f"two scalars {S.N_PARAMS_CHECKPOINT:,}, the total of the Hub metadata")

# ---- tiny teachers of both variants ---------------------------------------------------------
tiny = {}
for variant in VARIANTS:
    t, said = printed(S.build, variant=variant, tiny=True, **CPU)
    assert f"{NAMES[variant]} - attention sdpa" in said, said
    assert isinstance(t, S.CLASSES[variant]) and t.rule == S.DEFAULT_RULE and t.name == NAMES[variant]
    assert (t.dim, t.patch, t.input_size, t.grid) == (32, 16, 256, 16)
    assert type(t.model).__name__ == {"fixres": "SiglipVisionModel", "naflex": "Siglip2VisionModel"}[variant]
    assert t.model.config.num_hidden_layers == 2 and t.model.use_head and hasattr(t.model, "head")
    assert not t.model.training and all(not p.requires_grad for p in t.model.parameters())
    assert t.attention == "sdpa" and t.reads_s2 and t.reads_aerial and not t.reads_radar
    assert isinstance(build_teacher(NAMES[variant], tiny=True, **CPU), S.CLASSES[variant])
    assert vision_params(S.TINY_CONFIG[variant], 256) == t.n_params()
    with torch.no_grad():                   # a final norm that is far from doing nothing
        t.model.post_layernorm.weight.uniform_(0.5, 2.0)
        t.model.post_layernorm.bias.uniform_(-1.0, 1.0)
    tiny[variant] = t
print("tiny - both variants build, dim 32, grid 16, frozen, sdpa recorded, the parameter formula holds")

# ---- prepare picks (B04, B03, B02), thumbnail rule, then (x - 0.5) / 0.5 --------------------
s2 = torch.zeros(2, 12, 256, 256)
for i in range(12):
    s2[:, i] = 1000.0 * (i + 1)             # B04 = 4000, B03 = 3000, B02 = 2000
for variant, t in tiny.items():
    x = t.prepare(s2)
    assert x.shape == (2, 3, 256, 256) and x.dtype == torch.float32
    want = [(float(D.scale_rgb(torch.tensor([v]), "majortom_thumbnail")) - 0.5) / 0.5 for v in (4000.0, 3000.0, 2000.0)]
    for ch in range(3):
        assert torch.allclose(x[:, ch], torch.full((2, 256, 256), want[ch]), atol=1e-6), (variant, ch)
    assert float(t.prepare(torch.zeros(1, 12, 256, 256)).max()) == -1.0                 # DN 0 -> -1
    assert float(t.prepare(torch.full((1, 12, 256, 256), 10000.0)).min()) == 1.0       # DN 10000 -> 1
    r = dn_batch()
    direct = (D.scale_rgb(r[:, [3, 2, 1]], "majortom_thumbnail") - 0.5) / 0.5
    assert torch.equal(t.prepare(r), direct)
    for shape in ((128, 128), (384, 384), (128, 192)):
        y = t.prepare(torch.full((1, 12, *shape), 1500.0))
        assert y.shape == (1, 3, 256, 256)
        assert torch.allclose(y, t.prepare(torch.full((1, 12, 256, 256), 1500.0)), atol=1e-5)
    for bad in (float("nan"), float("inf")):
        broken = r.clone()
        broken[0, 3, 5, 5] = bad
        assert raises(ValueError, t, broken)
        other = r.clone()
        other[0, 7, 5, 5] = bad
        t(other)
print("prepare - RGB bands, thumbnail rule, -1 to 1, resize to 256 px, NaN in RGB stops")

# ---- outputs on a random 12-band digital-number tensor --------------------------------------
r = dn_batch()
outs = {}
for variant, t in tiny.items():
    out = t(r)
    assert out.summary.shape == (2, 32) and out.spatial.shape == (2, 16, 16, 32)
    assert out.summary.dtype == torch.float32 and out.spatial.dtype == torch.float32
    assert finite(out) and not out.summary.requires_grad and not out.spatial.requires_grad
    again = t(r)
    assert torch.equal(out.summary, again.summary) and torch.equal(out.spatial, again.spatial)
    outs[variant] = out
    print(f"{NAMES[variant]} tiny - DN input gives summary {tuple(out.summary.shape)} and spatial "
          f"{tuple(out.spatial.shape)}, finite, deterministic")

# summary is the pooling head on last_hidden_state, spatial is last_hidden_state row by row
t = tiny["fixres"]
with torch.inference_mode():
    ref = t.model(pixel_values=t.prepare(r))
    head = t.model.head(ref.last_hidden_state)
assert ref.last_hidden_state.shape == (2, 256, 32)
assert torch.allclose(outs["fixres"].summary, ref.pooler_output, atol=1e-6)
assert torch.allclose(outs["fixres"].summary, head, atol=1e-6)
assert torch.allclose(outs["fixres"].spatial.reshape(2, 256, 32), ref.last_hidden_state, atol=1e-6)
t = tiny["naflex"]
pv = S.patchify(t.prepare(r), 16)
mask = torch.ones(2, 256, dtype=torch.int32)
shapes = torch.tensor([[16, 16], [16, 16]])
with torch.inference_mode():
    ref = t.model(pixel_values=pv, pixel_attention_mask=mask, spatial_shapes=shapes)
    head = t.model.head(ref.last_hidden_state, mask)
    head_no_mask = t.model.head(ref.last_hidden_state, None)
assert ref.last_hidden_state.shape == (2, 256, 32)
assert torch.allclose(outs["naflex"].summary, ref.pooler_output, atol=1e-6)
assert torch.allclose(outs["naflex"].summary, head, atol=1e-6)
assert torch.allclose(outs["naflex"].spatial.reshape(2, 256, 32), ref.last_hidden_state, atol=1e-6)
d_mask = float((head - head_no_mask).abs().max())
assert d_mask == 0.0, d_mask
print(f"outputs - summary is model.head(last_hidden_state), spatial is last_hidden_state as 16 x 16; "
      f"the NaFlex head with a mask of ones against no mask differs by {d_mask:.1e}")

# bands outside RGB change nothing, red and blue swapped change the features
for variant, t in tiny.items():
    no_rgb = r.clone()
    no_rgb[:, [0, 4, 5, 6, 7, 8, 9, 10, 11]] = 0.0
    assert torch.equal(t(no_rgb).spatial, outs[variant].spatial)
    swapped = r.clone()
    swapped[:, [3, 1]] = r[:, [1, 3]]
    assert not torch.allclose(t(swapped).spatial, outs[variant].spatial, atol=1e-3)

# ---- patchification equals the library's convert_image_to_patches ---------------------------
x = torch.randn(2, 3, 256, 256)
mine = S.patchify(x, 16)
assert mine.shape == (2, 256, 768)
for i in range(2):
    lib = convert_image_to_patches(x[i], 16)
    assert lib.shape == (256, 768) and torch.equal(mine[i], lib), i
for patch, shape in ((8, (1, 3, 64, 48)), (16, (3, 1, 32, 32))):
    y = torch.randn(*shape)
    assert all(torch.equal(S.patchify(y, patch)[i], convert_image_to_patches(y[i], patch)) for i in range(shape[0]))
assert raises(ValueError, S.patchify, torch.randn(1, 3, 250, 256), 16)
# a patch at row 1, column 2 lands at index 1 * 16 + 2 (row major)
one = torch.zeros(1, 3, 256, 256)
one[:, :, 16:32, 32:48] = 1.0
assert S.patchify(one, 16).abs().sum(-1).nonzero()[:, 1].tolist() == [18]
# the library processor on a random 256 px 8-bit picture gives the same patches, a 16 x 16
# shape and a mask of ones
img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
try:
    from transformers import Siglip2ImageProcessor
    proc = Siglip2ImageProcessor(max_num_patches=256, patch_size=16, image_mean=list(S.MEAN),
                                 image_std=list(S.STD))
    got = proc(images=[img], return_tensors="pt")
except ImportError as e:                    # the fast processor needs torchvision
    got = None
    print(f"library processor - skipped ({type(e).__name__})")
if got is not None:
    assert got["spatial_shapes"].tolist() == [[16, 16]]
    assert got["pixel_attention_mask"].shape == (1, 256) and bool((got["pixel_attention_mask"] == 1).all())
    assert got["pixel_attention_mask"].dtype == torch.int32
    norm = (torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 - 0.5) / 0.5
    d_proc = float((S.patchify(norm[None], 16)[0] - got["pixel_values"][0]).abs().max())
    assert d_proc < 1e-6, d_proc
    print(f"patchify - equals convert_image_to_patches per sample, and the library processor on a "
          f"256 px picture (spatial shape 16 x 16, mask of ones, max difference {d_proc:.1e})")
else:
    print("patchify - equals convert_image_to_patches per sample")

# ---- NaFlex position embedding - 256 entries, the resize to (16, 16) is the identity --------
emb = tiny["naflex"].model.embeddings
assert emb.num_patches == 256 and emb.position_embedding_size == 16
pe = emb.position_embedding.weight.reshape(16, 16, -1)
with torch.inference_mode():
    resized = emb.resize_positional_embeddings(pe, shapes, max_length=256)
d_pe = float((resized[0] - emb.position_embedding.weight).abs().max())
assert d_pe == 0.0 and torch.equal(resized[0], resized[1])
assert torch.equal(resized[0].reshape(16, 16, -1), pe)            # row major, height first
print(f"NaFlex position embedding - 256 entries as 16 x 16, resized to (16, 16) differs by {d_pe:.1e}")

# ---- patch tokens run row by row (0 layers) - one patch moves one cell of the grid ---------
for variant in VARIANTS:
    model_class, config_class = S.model_classes(variant)
    flat0 = S.CLASSES[variant](model_class(config_class(**dict(S.TINY_CONFIG[variant], num_hidden_layers=0))),
                               S.DEFAULT_RULE, **CPU)
    moved = r.clone()
    moved[:, 3, 16:32, 32:48] += 500.0      # B04, patch in row 1 and column 2
    p, q = flat0(r), flat0(moved)
    cells = (p.spatial != q.spatial).any(-1)
    assert cells.sum().item() == 2 and bool(cells[:, 1, 2].all()), (variant, cells.nonzero())
    assert not torch.equal(p.summary, q.summary)     # the pooling head sees every token
print("token order - a change in one patch moves the cell (1, 2) only, both variants")

# ---- aerial RGB (NIR dropped) ----------------------------------------------------------------
a = aerial_batch()
for variant, t in tiny.items():
    out = t.aerial(a)
    assert out.summary.shape == (2, 32) and out.spatial.shape == (2, 16, 16, 32) and finite(out)
    x = t.prepare_aerial(a)
    assert x.shape == (2, 3, 256, 256)
    assert torch.allclose(x, (a[:, :3] / 255.0 - 0.5) / 0.5, atol=1e-6)
    nir = a.clone()
    nir[:, 3] = 0.0
    assert torch.equal(t.aerial(nir).spatial, out.spatial)
    assert torch.equal(t.aerial(a[:, :3]).spatial, out.spatial)
    assert torch.allclose(t.aerial(torch.full((1, 4, 256, 256), 127.5)).spatial,
                          t._run(torch.zeros(1, 3, 256, 256)).spatial, atol=1e-6)
    small = t.aerial(aerial_batch(size=128))
    assert small.spatial.shape == (2, 16, 16, 32) and finite(small)
    assert raises(ValueError, t.aerial, a * 2)                       # above 255
    assert raises(ValueError, t.aerial, a.to(torch.uint8))           # not float
    assert raises(TypeError, t.radar, torch.zeros(1, 2, 256, 256))   # no radar reader
    print(f"{NAMES[variant]} tiny - aerial (B, 4, 256, 256) 8-bit values give summary "
          f"{tuple(out.summary.shape)} and spatial {tuple(out.spatial.shape)}, finite, NIR dropped")

# ---- guards of build and of the classes -----------------------------------------------------
assert raises(ValueError, S.build, variant="fixres", tiny=True, rule="reflectance", **CPU)
assert raises(ValueError, S.build, variant="naflex", rule="reflectance", **CPU)     # before any download
assert raises(ValueError, S.build, variant="so400m", tiny=True, **CPU)
assert raises(ValueError, S.build, variant="so400m", **CPU)
assert raises(ValueError, S.model_classes, "so400m")
assert raises(ValueError, FR, tiny["fixres"].model, "reflectance", **CPU)
assert raises(KeyError, build_teacher, "siglip2-so400m-512", tiny=True, **CPU)
assert S.build(variant="naflex", tiny=True, random_init=True, **CPU).dim == 32       # tiny wins
secret = "not-a-real-token-0123456789"
held = S.build(variant="fixres", tiny=True, token=secret, **CPU)
assert secret not in repr(vars(held)) and secret not in repr(held.model.config.to_dict())
# a FixRes model built for another size, a NaFlex model with too few positions
assert raises(ValueError, FR, SiglipVisionModel(SiglipVisionConfig(**dict(S.TINY_CONFIG["fixres"], image_size=224))),
              S.DEFAULT_RULE, **CPU)
assert raises(ValueError, NF, Siglip2VisionModel(Siglip2VisionConfig(**dict(S.TINY_CONFIG["naflex"], num_patches=64))),
              S.DEFAULT_RULE, **CPU)
NF(Siglip2VisionModel(Siglip2VisionConfig(**dict(S.TINY_CONFIG["naflex"], num_patches=1024))), S.DEFAULT_RULE, **CPU)(r)
# the attention kernel - sdpa against eager on the tiny models
for variant, t in tiny.items():
    model_class, config_class = S.model_classes(variant)
    eager_model = model_class(config_class(**S.TINY_CONFIG[variant], attn_implementation="eager"))
    eager_model.load_state_dict(t.model.state_dict())
    eager = S.CLASSES[variant](eager_model, t.rule, **CPU)
    assert eager.attention == "eager"
    oe = eager(r)
    d_sum = float((oe.summary - outs[variant].summary).abs().max())
    d_spa = float((oe.spatial - outs[variant].spatial).abs().max())
    assert d_sum < 1e-4 and d_spa < 1e-4, (variant, d_sum, d_spa)
    print(f"{NAMES[variant]} tiny - sdpa against eager, max abs difference summary {d_sum:.1e}, spatial {d_spa:.1e}")


def refuse_argument(**kw):
    if "attn_implementation" in kw:
        raise TypeError("__init__() got an unexpected keyword argument 'attn_implementation'")
    return SiglipVisionModel(SiglipVisionConfig(**S.TINY_CONFIG["fixres"]))


fell, said_fell = printed(S.with_attention, refuse_argument)
assert "siglip2 - attention 'sdpa' is not accepted by this transformers (TypeError" in said_fell, said_fell
assert ":" not in said_fell and isinstance(fell, SiglipVisionModel)
print("guards - unknown rule and variant stop before any download, tiny wins, token not kept, "
      "wrong image size or too few positions stop, the attention fallback prints")

# ---- the vision tower out of a tiny checkpoint of the whole model ---------------------------
hf_logging.set_verbosity_error()             # the load reports of the folders below are not shown
full_models = {
    "fixres": SiglipModel(SiglipConfig(text_config=SiglipTextConfig(**TEXT_TINY).to_dict(),
                                       vision_config=SiglipVisionConfig(**S.TINY_CONFIG["fixres"]).to_dict())),
    "naflex": Siglip2Model(Siglip2Config(text_config=Siglip2TextConfig(**TEXT_TINY).to_dict(),
                                         vision_config=Siglip2VisionConfig(**S.TINY_CONFIG["naflex"]).to_dict())),
}
for variant, full in full_models.items():
    full.eval()
    folder = _setup.TMP_BASE / f"siglip2_tiny_full_{variant}"
    if folder.exists():
        shutil.rmtree(folder)
    full.save_pretrained(str(folder))
    state = load_file(str(folder / S.WEIGHTS_FILE))
    assert {k for k in state if not k.startswith(("vision_model.", "text_model."))} == {"logit_scale", "logit_bias"}
    n_vision = sum(k.startswith("vision_model.") for k in state)
    model = S.load_folder(variant, folder, n_bytes=None)
    assert type(model) is type(full.vision_model) and len(model.state_dict()) == n_vision
    assert next(model.parameters()).dtype == torch.float32 and S.attention_of(model) == "sdpa"
    loaded = S.CLASSES[variant](model, S.DEFAULT_RULE, **CPU)
    lo = loaded(r)
    tower = S.CLASSES[variant](full.vision_model, S.DEFAULT_RULE, **CPU)(r)
    assert torch.equal(lo.summary, tower.summary) and torch.equal(lo.spatial, tower.spatial)
    S.check_model(model, variant, config=S.TINY_CONFIG[variant], n_params=loaded.n_params())
    assert raises(ValueError, S.check_model, model, variant)                  # not the so400m
    assert raises(ValueError, S.check_model, model, variant, config=S.TINY_CONFIG[variant],
                  n_params=loaded.n_params() + 1)
    # the size is pinned unless it is switched off
    assert raises(ValueError, S.load_folder, variant, folder)
    assert raises(ValueError, S.build, variant=variant, weights_dir=folder, **CPU)
    empty = _setup.TMP_BASE / f"siglip2_no_weights_{variant}"
    empty.mkdir(exist_ok=True)
    assert raises(FileNotFoundError, S.load_folder, variant, empty, n_bytes=None)
    assert raises(FileNotFoundError, S.build, variant=variant, weights_dir=empty, **CPU)

    # a file that does not hold the tower stops - transformers would fill the gap randomly
    def folder_with(name, tensors):
        dest = _setup.TMP_BASE / f"siglip2_tiny_{variant}_{name}"
        dest.mkdir(exist_ok=True)
        shutil.copy(folder / "config.json", dest / "config.json")
        save_file(tensors, str(dest / S.WEIGHTS_FILE), metadata={"format": "pt"})
        return dest

    def load_error(where):
        try:
            S.load_folder(variant, where, n_bytes=None)
        except ValueError as e:
            return str(e)
        return ""

    again = S.CLASSES[variant](S.load_folder(variant, folder_with("same", state), n_bytes=None),
                               S.DEFAULT_RULE, **CPU)(r)
    assert torch.equal(again.spatial, lo.spatial)
    cases = {
        "no_post_norm": {k: v for k, v in state.items() if "post_layernorm" not in k},
        "no_head": {k: v for k, v in state.items() if ".head." not in k},
        "other_names": {k.replace("encoder.layers.", "blocks."): v for k, v in state.items()},
        "one_more": dict(state, **{"vision_model.local_norm.weight": torch.ones(32)}),
        "stray": dict(state, **{"logit_scale2": torch.ones(1)}),
    }
    said = {name: load_error(folder_with(name, tensors)) for name, tensors in cases.items()}
    for name, text in said.items():
        assert "do not hold the same tensors" in text, (variant, name, text)
    assert "missing_keys" in said["no_post_norm"] and "post_layernorm" in said["no_post_norm"]
    assert "missing_keys" in said["no_head"] and "head.probe" in said["no_head"]
    assert "unexpected_keys" in said["one_more"] and "local_norm" in said["one_more"]
    assert "unexpected_keys" in said["stray"] and "logit_scale2" in said["stray"]
    # the text tower and the two scalars alone never appear in an error
    only_tower = {k: v for k, v in state.items() if k.startswith("vision_model.")}
    S.load_folder(variant, folder_with("tower_only", only_tower), n_bytes=None)
    print(f"{NAMES[variant]} tiny - the tower loads out of the whole checkpoint with the same numbers "
          f"({n_vision} vision tensors, {len(state) - n_vision} unused text and logit tensors), a "
          f"missing or stray tensor stops")
hf_logging.set_verbosity_warning()
del full_models

# ---- the real weights in the Hub cache (skipped with a printed line when absent) -----------
# Without them the full architecture is built with random weights once, which checks the
# typed parameter count the same way (about 20 to 30 s on this CPU), and build() without
# weights and without network must stop - random weights never stand in.
def snapshot_of(cls):
    return HF_HOME / "hub" / f"models--{cls.hf_id.replace('/', '--')}" / "snapshots" / cls.hf_revision


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


for variant in VARIANTS:
    cls = S.CLASSES[variant]
    snap = snapshot_of(cls)
    weights = snap / S.WEIGHTS_FILE
    if not weights.is_file():
        print(f"{cls.name} real weights - skipped, {weights} is not in the cache")
        try:
            S.build(variant=variant, **CPU)
            raise SystemExit(f"{cls.name} - build returned a teacher although no weights could be read")
        except SystemExit:
            raise
        except Exception as e:
            print(f"{cls.name} - no weights, no network, build stops with {type(e).__name__}")
        clock = time.perf_counter()
        big, _ = printed(S.build, variant=variant, random_init=True, **CPU)
        n = big.n_params()
        assert n == S.N_PARAMS and (big.dim, big.grid) == (1152, 16)
        print(f"{cls.name} random_init - {n:,} parameters in {time.perf_counter() - clock:.1f} s")
        del big
        gc.collect()
        continue
    size = weights.stat().st_size
    assert size == S.WEIGHTS_BYTES, (size, S.WEIGHTS_BYTES)
    blob = weights.resolve().name            # the Hub names the blob by the LFS SHA-256
    if blob != weights.name:
        assert blob == cls.weights_sha256, (blob, cls.weights_sha256)
    clock = time.perf_counter()
    digest = sha256_of(weights)
    assert digest == cls.weights_sha256, (digest, cls.weights_sha256)
    print(f"{cls.name} real weights - {size:,} bytes, SHA-256 of the bytes as pinned, hashed in "
          f"{time.perf_counter() - clock:.0f} s")
    pre = snap / "preprocessor_config.json"
    if pre.is_file():
        p = json.loads(pre.read_text(encoding="utf-8"))
        assert tuple(p["image_mean"]) == S.MEAN and tuple(p["image_std"]) == S.STD
        assert p["do_rescale"] and p["rescale_factor"] == 1 / 255 and p["do_normalize"]
        if variant == "naflex":
            assert p["max_num_patches"] == 256 and p["patch_size"] == 16
        else:
            assert p["size"] == {"height": 256, "width": 256}
        print(f"{cls.name} real weights - preprocessor_config.json holds mean {p['image_mean']} and "
              f"std {p['image_std']}, rescale 1/255")
    # every tensor of the file is a vision tensor that reached the model, or one of UNEXPECTED_OK
    with safe_open(str(weights), framework="pt") as f:
        file_keys = set(f.keys())
    clock = time.perf_counter()
    teacher, said = printed(S.build, variant=variant, **CPU)         # through the offline cache
    load_s = time.perf_counter() - clock
    assert f"{cls.name} - attention sdpa" in said, said
    assert teacher.n_params() == S.N_PARAMS and (teacher.dim, teacher.grid) == (1152, 16)
    S.check_model(teacher.model, variant)
    model_keys = {"vision_model." + k for k in teacher.model.state_dict()}
    assert model_keys <= file_keys, sorted(model_keys - file_keys)[:5]
    rest = file_keys - model_keys
    assert all(S.unexpected_ok(k) for k in rest), sorted(k for k in rest if not S.unexpected_ok(k))[:5]
    print(f"{cls.name} real weights - loaded in {load_s:.1f} s, {teacher.n_params():,} parameters, no "
          f"missing vision key, {len(model_keys)} vision tensors read, {len(rest)} left unused "
          f"(text tower, logit_scale, logit_bias)")
    torch.manual_seed(0)
    x_real = dn_batch(b=1)                   # one 256 px random thumbnail of digital numbers
    clock = time.perf_counter()
    o = teacher(x_real)
    fwd_s = time.perf_counter() - clock
    assert o.summary.shape == (1, 1152) and o.spatial.shape == (1, 16, 16, 1152) and finite(o)
    oa = teacher.aerial(aerial_batch(b=1))
    assert oa.summary.shape == (1, 1152) and oa.spatial.shape == (1, 16, 16, 1152) and finite(oa)
    print(f"{cls.name} real weights - one CPU forward at batch 1 of a 256 px thumbnail in {fwd_s:.2f} s, "
          f"summary (1, 1152) and spatial (1, 16, 16, 1152) finite, summary norm {float(o.summary.norm()):.2f}, "
          f"aerial forward finite too. The CPU timing says nothing about the A100")
    del teacher, o, oa
    gc.collect()

print("test_teacher_siglip2 - all checks passed")
