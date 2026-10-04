"""Teacher DINOv3 on tiny random models, scaling rules, normalisation, token layout, guards."""
import _setup

import contextlib
import gc
import hashlib
import io
import json
import os
import shutil
import time

# No network and an empty model cache on D:, set before transformers is imported.
HF_HOME = _setup.TMP_BASE / "dinov3_hf_home"
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from transformers import DINOv3ViTConfig, DINOv3ViTModel  # noqa: E402
from transformers import logging as hf_logging  # noqa: E402

from sartransfer.distil.teachers import TEACHERS, Teacher, build_teacher  # noqa: E402
from sartransfer.distil.teachers import dinov3_sat as D  # noqa: E402

torch.manual_seed(0)
CPU = dict(device="cpu", dtype=torch.float32)


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


# ---- facts typed into the module -----------------------------------------------------------
assert TEACHERS["dinov3-l-sat"] == ("dinov3_sat", {})
assert D.HF_ID == "facebook/dinov3-vitl16-pretrain-sat493m"
# revision, file name, SHA-256 and parameter count as the public model information gives them
assert D.HF_REVISION == "f692fa42da72c6797b67cd73494a168d1120d3ee"
assert D.WEIGHTS_FILE == "model.safetensors"
assert D.WEIGHTS_SHA256 == "4e6356d992c1301b5e7c275f465e47296c5c4ad17052e262b29fc21e82ccc698"
assert D.N_PARAMS == 303_129_600
assert D.MEAN == (0.430, 0.411, 0.296) and D.STD == (0.213, 0.156, 0.143)
C = D.DINOv3Sat
assert issubclass(C, Teacher) and C.name == "dinov3-l-sat"
assert (C.dim, C.patch, C.input_size) == (1024, 16, 256)
assert C.bands == ("B04", "B03", "B02")
assert D.FULL_CONFIG == dict(patch_size=16, hidden_size=1024, intermediate_size=4096,
                             num_hidden_layers=24, num_attention_heads=16, num_register_tokens=4,
                             use_gated_mlp=False, hidden_act="gelu", num_channels=3,
                             layer_norm_eps=1e-5, rope_theta=100.0, query_bias=True,
                             key_bias=False, value_bias=True, proj_bias=True, mlp_bias=True)
assert C.dim == D.FULL_CONFIG["hidden_size"] and C.patch == D.FULL_CONFIG["patch_size"]
assert D.TINY_CONFIG == dict(D.FULL_CONFIG, hidden_size=32, intermediate_size=128,
                             num_hidden_layers=2, num_attention_heads=2)

# ---- the two small files of the gated repository, rebuilt ----------------------------------
# The files cannot be downloaded without a token. The public file listing gives the git blob
# id of each (SHA-1 over "blob <size>", a zero byte and the bytes). A text with the same id
# is the file, so the typed numbers below are the numbers of the repository.
CONFIG_BLOB = "ab2a8b183fd4fd5213e9898a7f499c7c857b1a86"          # config.json, 745 bytes
PROCESSOR_BLOB = "3f77f75dd0ce822970aa276898bec394f3ef924f"       # preprocessor_config.json, 585
CONFIG_TEXT = """{
  "architectures": [
    "DINOv3ViTModel"
  ],
  "attention_dropout": 0.0,
  "drop_path_rate": 0.0,
  "hidden_act": "gelu",
  "hidden_size": 1024,
  "image_size": 224,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "key_bias": false,
  "layer_norm_eps": 1e-05,
  "layerscale_value": 1.0,
  "mlp_bias": true,
  "model_type": "dinov3_vit",
  "num_attention_heads": 16,
  "num_channels": 3,
  "num_hidden_layers": 24,
  "num_register_tokens": 4,
  "patch_size": 16,
  "pos_embed_jitter": null,
  "pos_embed_rescale": 2.0,
  "pos_embed_shift": null,
  "proj_bias": true,
  "query_bias": true,
  "rope_theta": 100.0,
  "torch_dtype": "float32",
  "transformers_version": "4.56.0.dev0",
  "use_gated_mlp": false,
  "value_bias": true
}
"""
PROCESSOR_TEXT = """{
  "crop_size": null,
  "data_format": "channels_first",
  "default_to_square": true,
  "device": null,
  "disable_grouping": null,
  "do_center_crop": null,
  "do_convert_rgb": null,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.430,
    0.411,
    0.296
  ],
  "image_processor_type": "DINOv3ViTImageProcessorFast",
  "image_std": [
    0.213,
    0.156,
    0.143
  ],
  "input_data_format": null,
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "return_tensors": null,
  "size": {
    "height": 224,
    "width": 224
  }
}
"""


def git_blob_id(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


assert git_blob_id(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"      # the empty file of git
assert len(CONFIG_TEXT.encode()) == 745 and git_blob_id(CONFIG_TEXT.encode()) == CONFIG_BLOB
assert len(PROCESSOR_TEXT.encode()) == 585
assert git_blob_id(PROCESSOR_TEXT.encode()) == PROCESSOR_BLOB
repo_config, repo_processor = json.loads(CONFIG_TEXT), json.loads(PROCESSOR_TEXT)
for key, value in D.FULL_CONFIG.items():
    assert repo_config[key] == value and type(repo_config[key]) is type(value), key
assert repo_config["architectures"] == ["DINOv3ViTModel"]
assert tuple(repo_processor["image_mean"]) == D.MEAN
assert tuple(repo_processor["image_std"]) == D.STD
assert repo_processor["do_rescale"] and repo_processor["rescale_factor"] == 1 / 255
# the other values of config.json are names, or they act in training or at the start only
# (measured further down on a tiny model)
NO_EFFECT = dict(attention_dropout=0.5, drop_path_rate=0.5, image_size=64, initializer_range=0.5,
                 layerscale_value=0.5, pos_embed_jitter=2.0, pos_embed_rescale=None,
                 pos_embed_shift=0.5)
NAMES = {"architectures", "model_type", "torch_dtype", "transformers_version"}
assert set(repo_config) == set(D.FULL_CONFIG) | set(NO_EFFECT) | NAMES

assert 2 <= len(D.RULES) <= 4 and len(set(D.RULES)) == len(D.RULES)
assert C.rules == D.RULES and D.DEFAULT_RULE in D.RULES
assert set(D.RULE_SOURCES) == set(D.RULES) and C.rule_sources is D.RULE_SOURCES
for text in (*D.RULE_SOURCES.values(), C.licence):
    assert len(text) > 40 and ":" not in text, text
assert all("commit" in D.RULE_SOURCES[r] for r in ("majortom_embedder", "majortom_thumbnail"))
# every text stands alone in a table, so it names its files and points to no other text
for text in D.RULE_SOURCES.values():
    assert "below" not in text and "above" not in text, text
for name in ("thumbnail_s2.py", "DINOv2_S2RGB.py", "SigLIP_S2RGB.py", "commit f7078cd0e9d8"):
    assert name in D.RULE_SOURCES["reflectance"], name
assert C.licence.startswith("DINOv3 License") and "read 2026-09-29" in C.licence

# ---- scaling rules against the formulas of their sources ------------------------------------
dn = torch.tensor([0.0, 1.0, 500.0, 1000.0, 2000.0, 3999.0, 4000.0, 5000.0, 6458.0, 9999.0,
                   10000.0, 20000.0, 65535.0])
a = dn.double().numpy()
source = {
    "reflectance": (a / 10000).clip(0, 1),
    "majortom_embedder": (2.5 * (a / 1e4)).clip(0, 1),                 # DINOv2_S2RGB.normalize
    "majortom_thumbnail": (1.3 * ((a / 10_000) ** 0.6)).clip(0, 1),    # s2l2a_thumbnail
}
assert set(source) == set(D.RULES)
for rule in D.RULES:
    got = D.scale_rgb(dn, rule)
    assert got.dtype == torch.float32 and got.shape == dn.shape
    assert np.allclose(got.numpy(), source[rule], atol=1e-6), rule
    assert float(got.min()) >= 0.0 and float(got.max()) <= 1.0
    assert float(got[-1]) == 1.0 and float(got[0]) == 0.0
at_1000 = {r: float(D.scale_rgb(torch.tensor([1000.0]), r)) for r in D.RULES}
assert abs(at_1000["reflectance"] - 0.1) < 1e-6
assert abs(at_1000["majortom_embedder"] - 0.25) < 1e-6
assert abs(at_1000["majortom_thumbnail"] - 0.326545) < 1e-5
assert float(D.scale_rgb(torch.tensor([4000.0]), "majortom_embedder")) == 1.0
assert float(D.scale_rgb(torch.tensor([3999.0]), "majortom_embedder")) < 1.0

# the thumbnail as the source makes it (uint8), at most one grey level apart
thumb = (source["majortom_thumbnail"] * 255).astype(np.uint8).astype(int)
ours = (D.scale_rgb(dn, "majortom_thumbnail") * 255).to(torch.uint8).numpy().astype(int)
assert np.abs(thumb - ours).max() <= 1

# negative numbers count as 0 and never give NaN
neg = torch.tensor([-1.0, -1000.0])
for rule in D.RULES:
    assert torch.equal(D.scale_rgb(neg, rule), torch.zeros(2)), rule
assert raises(ValueError, D.scale_rgb, dn, "percentile")

# the numbers in the written reason for DEFAULT_RULE
said = {"reflectance": (0.05, 0.10, 0.20), "majortom_embedder": (0.13, 0.25, 0.50),
        "majortom_thumbnail": (0.22, 0.33, 0.49)}
for rule, values in said.items():
    got = D.scale_rgb(torch.tensor([500.0, 1000.0, 2000.0]), rule)
    assert np.allclose(got.numpy(), values, atol=0.0051), (rule, got)
mean = torch.tensor(D.MEAN)
gap = {r: float((D.scale_rgb(torch.tensor([500.0, 1000.0]), r)[:, None] - mean).abs().mean())
       for r in D.RULES}
assert min(gap, key=gap.get) == D.DEFAULT_RULE == "majortom_thumbnail", gap

# ---- tiny teacher ---------------------------------------------------------------------------
t = D.build(tiny=True, **CPU)
assert isinstance(t, C) and t.rule == D.DEFAULT_RULE
assert (t.dim, t.patch, t.input_size, t.grid) == (32, 16, 256, 16)
assert type(t.model).__name__ == "DINOv3ViTModel"
assert t.model.config.num_register_tokens == 4 and t.model.config.num_hidden_layers == 2
assert not t.model.training and all(not p.requires_grad for p in t.model.parameters())
assert isinstance(build_teacher("dinov3-l-sat", tiny=True, **CPU), C)
with torch.no_grad():                       # a final norm that is far from doing nothing
    t.model.norm.weight.uniform_(0.5, 2.0)
    t.model.norm.bias.uniform_(-1.0, 1.0)

# prepare picks (B04, B03, B02) and applies the mean and std of the README
s2 = torch.zeros(2, 12, 256, 256)
for i in range(12):
    s2[:, i] = 1000.0 * (i + 1)             # B04 = 4000, B03 = 3000, B02 = 2000
tr = D.build(tiny=True, rule="reflectance", **CPU)
x = tr.prepare(s2)
assert x.shape == (2, 3, 256, 256) and x.dtype == torch.float32
want = [(0.4 - 0.430) / 0.213, (0.3 - 0.411) / 0.156, (0.2 - 0.296) / 0.143]
for c in range(3):
    assert torch.allclose(x[:, c], torch.full((2, 256, 256), want[c]), atol=1e-6), c
at_mean = torch.zeros(1, 12, 256, 256)
at_mean[:, [3, 2, 1]] = (torch.tensor(D.MEAN) * 10000).view(1, 3, 1, 1)
assert float(tr.prepare(at_mean).abs().max()) < 1e-6

# at 256 px nothing is resampled, other sizes are brought to 256 px
r = dn_batch()
direct = (D.scale_rgb(r[:, [3, 2, 1]], "reflectance") - tr.mean) / tr.std
assert torch.equal(tr.prepare(r), direct)
for shape in ((128, 128), (384, 384), (128, 192)):
    flat = torch.full((1, 12, *shape), 1500.0)
    y = tr.prepare(flat)
    assert y.shape == (1, 3, 256, 256)
    assert torch.allclose(y, tr.prepare(torch.full((1, 12, 256, 256), 1500.0)), atol=1e-5)
small = tr.prepare(dn_batch(size=128))
assert small.shape == (2, 3, 256, 256) and float(small.std()) > 0.1

# NaN or infinity in the RGB bands stops, in another band it does not matter
for bad in (float("nan"), float("inf")):
    broken = r.clone()
    broken[0, 3, 5, 5] = bad
    assert raises(ValueError, t, broken)
    other = r.clone()
    other[0, 7, 5, 5] = bad
    t(other)

# ---- outputs --------------------------------------------------------------------------------
out = t(r)
assert out.summary.shape == (2, 32) and out.spatial.shape == (2, 16, 16, 32)
assert out.summary.dtype == torch.float32 and out.spatial.dtype == torch.float32
assert not out.summary.requires_grad and not out.spatial.requires_grad
again = t(r)
assert torch.equal(out.summary, again.summary) and torch.equal(out.spatial, again.spatial)

with torch.inference_mode():
    ref = t.model(pixel_values=t.prepare(r), output_hidden_states=True)
    before_norm = ref.hidden_states[-1]
    after_norm = t.model.norm(before_norm)
assert ref.last_hidden_state.shape == (2, 1 + 4 + 256, 32)
assert torch.allclose(out.summary, after_norm[:, 0], atol=1e-6)
assert torch.allclose(out.spatial.reshape(2, 256, 32), after_norm[:, 5:], atol=1e-6)
assert torch.allclose(out.summary, ref.pooler_output, atol=1e-6)
assert not torch.allclose(out.spatial.reshape(2, 256, 32), before_norm[:, 5:], atol=1e-2)
assert not torch.allclose(out.spatial.reshape(2, 256, 32), after_norm[:, 4:-1], atol=1e-2)

# every rule gives other features, and bands outside RGB change nothing
feats = {rule: D.build(tiny=True, rule=rule, **CPU) for rule in D.RULES}
for rule, teacher in feats.items():
    teacher.model.load_state_dict(t.model.state_dict())
    assert teacher.rule == rule
outs = {rule: teacher(r) for rule, teacher in feats.items()}
for i, p in enumerate(D.RULES):
    for q in D.RULES[i + 1:]:
        assert not torch.allclose(outs[p].spatial, outs[q].spatial, atol=1e-3), (p, q)
no_rgb = r.clone()
no_rgb[:, [0, 4, 5, 6, 7, 8, 9, 10, 11]] = 0.0
assert torch.equal(t(no_rgb).spatial, out.spatial)
swapped = r.clone()
swapped[:, [3, 1]] = r[:, [1, 3]]           # red and blue exchanged
assert not torch.allclose(t(swapped).spatial, out.spatial, atol=1e-3)

# ---- prefix tokens are inferred from the count and checked ---------------------------------
two = C(DINOv3ViTModel(DINOv3ViTConfig(**dict(D.TINY_CONFIG, num_register_tokens=2))),
        "reflectance", **CPU)
o2 = two(r)
with torch.inference_mode():
    tok = two.model(pixel_values=two.prepare(r)).last_hidden_state
assert tok.shape[1] == 1 + 2 + 256
assert torch.allclose(o2.spatial.reshape(2, 256, 32), tok[:, 3:], atol=1e-6)
two.model.config.num_register_tokens = 3    # the config now disagrees with the token count
assert raises(RuntimeError, two, r)
two.model.config.num_register_tokens = 2
two(r)

# ---- patch tokens run row by row: a change in one patch moves one cell of the grid ----------
flat0 = C(DINOv3ViTModel(DINOv3ViTConfig(**dict(D.TINY_CONFIG, num_hidden_layers=0))),
          "reflectance", **CPU)
moved = r.clone()
moved[:, 3, 16:32, 32:48] += 500.0          # B04, patch in row 1 and column 2
p, q = flat0(r), flat0(moved)
cells = (p.spatial != q.spatial).any(-1)
assert cells.sum().item() == 2 and bool(cells[:, 1, 2].all()), cells.nonzero()
assert torch.equal(p.summary, q.summary)

# ---- the full config gives the published parameter count -----------------------------------
def count(c, channels=3):
    d, m, n = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"]
    embed = channels * c["patch_size"] ** 2 * d + d + 2 * d + c["num_register_tokens"] * d
    layer = (4 * d * d + 3 * d) + (2 * d * m + m + d) + 4 * d + 2 * d
    return embed + n * layer + 2 * d


assert count(D.TINY_CONFIG) == t.n_params()            # the formula holds on a real module
assert count(dict(D.TINY_CONFIG, num_register_tokens=2)) == two.n_params()
assert count(D.FULL_CONFIG) == D.N_PARAMS == 303_129_600

D.check_model(t.model, config=D.TINY_CONFIG, n_params=t.n_params())
D.check_model(t.model, config=D.TINY_CONFIG, n_params=None)
assert raises(ValueError, D.check_model, t.model)                       # not the ViT-L
assert raises(ValueError, D.check_model, t.model, config=D.TINY_CONFIG, n_params=t.n_params() + 1)
assert raises(ValueError, D.check_model, two.model, config=D.TINY_CONFIG, n_params=None)

# ---- guards of build ------------------------------------------------------------------------
assert raises(ValueError, D.build, tiny=True, rule="percentile", **CPU)
assert raises(ValueError, D.build, rule="percentile", **CPU)            # before any download
assert raises(ValueError, C, t.model, "percentile", **CPU)
assert D.build(tiny=True, random_init=True, **CPU).dim == 32            # tiny wins

secret = "not-a-real-token-0123456789"
held = D.build(tiny=True, token=secret, **CPU)
assert secret not in repr(vars(held)) and secret not in repr(held.model.config.to_dict())

# ---- weights from a local folder, checked by SHA-256 ---------------------------------------
folder = _setup.TMP_BASE / "dinov3_tiny_weights"
t.model.save_pretrained(str(folder))
weights = folder / D.WEIGHTS_FILE
digest = hashlib.sha256(weights.read_bytes()).hexdigest()
assert D.file_sha256(weights) == digest and digest != D.WEIGHTS_SHA256

loaded = C(D.load_folder(folder, sha256=digest), D.DEFAULT_RULE, **CPU)
lo = loaded(r)
assert torch.equal(lo.summary, out.summary) and torch.equal(lo.spatial, out.spatial)

assert raises(ValueError, D.load_folder, folder)                        # not the pinned file
one_off = digest[:-1] + ("0" if digest[-1] != "0" else "1")
assert raises(ValueError, D.load_folder, folder, sha256=one_off)
assert raises(ValueError, D.build, weights_dir=folder, **CPU)
empty = _setup.TMP_BASE / "dinov3_no_weights"
empty.mkdir(exist_ok=True)
assert raises(FileNotFoundError, D.load_folder, empty)
assert raises(FileNotFoundError, D.build, weights_dir=empty, **CPU)

# ---- a file and a model that do not hold the same tensors stop the load ---------------------
# transformers fills a tensor it does not find with random numbers and goes on. The SHA-256
# of the file is right in every case here, so only the check of the load itself can stop it.
state = load_file(str(weights))
assert len(state) == len(list(t.model.parameters())) and "norm.weight" in state


def folder_with(name, tensors, config=None):
    """A weights folder with these tensors and this config, and the SHA-256 of its file."""
    dest = _setup.TMP_BASE / f"dinov3_tiny_{name}"
    dest.mkdir(exist_ok=True)
    if config is None:
        shutil.copy(folder / "config.json", dest / "config.json")
    else:
        (dest / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(tensors, str(dest / D.WEIGHTS_FILE), metadata={"format": "pt"})
    return dest, hashlib.sha256((dest / D.WEIGHTS_FILE).read_bytes()).hexdigest()


def load_error(where, sha):
    try:
        D.load_folder(where, sha256=sha)
    except ValueError as e:
        return str(e)
    return ""


same, sha = folder_with("same", state)
again = C(D.load_folder(same, sha256=sha), D.DEFAULT_RULE, **CPU)(r)
assert torch.equal(again.spatial, out.spatial)                          # the helper itself is fine
cases = {
    "no_final_norm": {k: v for k, v in state.items() if not k.startswith("norm.")},
    "no_registers": {k: v for k, v in state.items() if "register_tokens" not in k},
    "other_names": {k.replace("layer.", "blocks."): v for k, v in state.items()},
    "one_more": dict(state, **{"local_cls_norm.weight": torch.ones(32)}),
}
hf_logging.set_verbosity_error()             # the load reports of the broken files are not shown
said = {name: load_error(*folder_with(name, tensors)) for name, tensors in cases.items()}
hf_logging.set_verbosity_warning()
for name, text in said.items():
    assert "do not hold the same tensors" in text, (name, text)
assert "norm.weight" in said["no_final_norm"] and "missing_keys" in said["no_final_norm"]
assert "local_cls_norm.weight" in said["one_more"] and "unexpected_keys" in said["one_more"]

# ---- config.json next to the right weights, with one value changed -------------------------
tiny_config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
for key in (*D.FULL_CONFIG, *NO_EFFECT):
    assert key in tiny_config, key
CHANGED = dict(layer_norm_eps=1e-2, rope_theta=10.0, hidden_act="relu")
for key, value in CHANGED.items():           # the features change, so check_model must stop
    where, sha = folder_with(f"cfg_{key}", state, dict(tiny_config, **{key: value}))
    m = D.load_folder(where, sha256=sha)
    assert m.config.to_dict()[key] == value
    assert not torch.equal(C(m, D.DEFAULT_RULE, **CPU)(r).spatial, out.spatial), key
    assert raises(ValueError, D.check_model, m, config=D.TINY_CONFIG, n_params=None), key
for key, value in NO_EFFECT.items():         # not one number changes, check_model lets it pass
    where, sha = folder_with(f"cfg_{key}", state, dict(tiny_config, **{key: value}))
    m = D.load_folder(where, sha256=sha)
    D.check_model(m, config=D.TINY_CONFIG, n_params=t.n_params())
    assert torch.equal(C(m, D.DEFAULT_RULE, **CPU)(r).spatial, out.spatial), key

# without weights and without network nothing is built, random weights never stand in
assert not any(HF_HOME.rglob("*.safetensors"))
try:
    D.build(**CPU)
    raise SystemExit("build returned a teacher although no weights could be read")
except SystemExit:
    raise
except Exception as e:
    print(f"no weights, no network: {type(e).__name__}")

# ---- the attention kernel. sdpa is asked for and recorded, its outputs equal eager's ---------
assert D.ATTENTION == "sdpa" and t.attention == "sdpa" and D.attention_of(t.model) == "sdpa"
assert "sdpa" in D.__doc__ and "with_attention" in D.__doc__ and "CPU timings say nothing" in D.__doc__
eager_model = DINOv3ViTModel(DINOv3ViTConfig(**D.TINY_CONFIG, attn_implementation="eager"))
eager_model.load_state_dict(t.model.state_dict())
eager = C(eager_model, t.rule, **CPU)
assert eager.attention == "eager" and D.attention_of(eager_model) == "eager"
oe = eager(r)
assert oe.summary.shape == out.summary.shape and oe.spatial.shape == out.spatial.shape
diff_tiny = (float((oe.summary - out.summary).abs().max()), float((oe.spatial - out.spatial).abs().max()))
assert diff_tiny[0] < 1e-4 and diff_tiny[1] < 1e-4, diff_tiny
with torch.inference_mode():                 # the prefix tokens and the token count are the same
    tok_e = eager.model(pixel_values=eager.prepare(r)).last_hidden_state
    tok_s = t.model(pixel_values=t.prepare(r)).last_hidden_state
assert tok_e.shape == tok_s.shape == (2, 1 + 4 + 256, 32)
assert torch.allclose(tok_e[:, :5], tok_s[:, :5], atol=1e-4)
print(f"tiny model: sdpa against eager, max abs difference summary {diff_tiny[0]:.1e}, spatial {diff_tiny[1]:.1e}")
# with_attention asks for the kernel and falls back with a printed line when it is refused
built = D.with_attention(lambda **kw: DINOv3ViTModel(DINOv3ViTConfig(**D.TINY_CONFIG, **kw)))
assert D.attention_of(built) == "sdpa"


def refuse_argument(**kw):
    if "attn_implementation" in kw:
        raise TypeError("__init__() got an unexpected keyword argument 'attn_implementation'")
    return DINOv3ViTModel(DINOv3ViTConfig(**D.TINY_CONFIG))


fell, said_fell = printed(D.with_attention, refuse_argument)
assert "dinov3-l-sat: attention 'sdpa' is not accepted by this transformers (TypeError" in said_fell, said_fell
assert "its default kernel is used" in said_fell and isinstance(fell, DINOv3ViTModel)
fell2, said_fell2 = printed(D.with_attention,
                            lambda **kw: DINOv3ViTModel(DINOv3ViTConfig(**D.TINY_CONFIG, **kw)), "nonsense")
assert "attention 'nonsense' is not accepted" in said_fell2 and "ValueError" in said_fell2
assert isinstance(fell2, DINOv3ViTModel)
_, said_tiny = printed(D.with_attention, lambda **kw: DINOv3ViTModel(DINOv3ViTConfig(**D.TINY_CONFIG, **kw)))
assert said_tiny == ""                      # nothing printed when the kernel is taken
# load_folder asks for sdpa, records it, takes eager on request, and falls back when
# from_pretrained refuses the argument (patched here)
import transformers  # noqa: E402

own = transformers.AutoModel.__dict__.get("from_pretrained")
real_from_pretrained = transformers.AutoModel.from_pretrained


def refusing_from_pretrained(*args, **kw):
    if "attn_implementation" in kw:
        raise TypeError("from_pretrained() got an unexpected keyword argument 'attn_implementation'")
    return real_from_pretrained(*args, **kw)


transformers.AutoModel.from_pretrained = refusing_from_pretrained
try:
    fb, said_fb = printed(D.load_folder, folder, sha256=digest)
finally:
    if own is None:
        del transformers.AutoModel.from_pretrained
    else:
        transformers.AutoModel.from_pretrained = own
assert transformers.AutoModel.from_pretrained == real_from_pretrained
assert "attention 'sdpa' is not accepted by this transformers (TypeError" in said_fb, said_fb
assert torch.equal(C(fb, D.DEFAULT_RULE, **CPU)(r).spatial, out.spatial)
assert D.attention_of(D.load_folder(folder, sha256=digest)) == "sdpa"
le = D.load_folder(folder, sha256=digest, attention="eager")
assert D.attention_of(le) == "eager"
assert torch.allclose(C(le, D.DEFAULT_RULE, **CPU)(r).spatial, out.spatial, atol=1e-4)
assert isinstance(D.load_folder(folder, sha256=digest, attention=None), DINOv3ViTModel)
_, said_build = printed(D.build, tiny=True, **CPU)
assert "dinov3-l-sat: attention sdpa" in said_build, said_build
# the scoring notebook loads the same model through AutoModel.from_pretrained WITHOUT the argument
# (models/encoders.py, FrozenEncoder, copied from part two) and gets the library default. That default
# must be the pinned kernel and give the same numbers, so the pin changes nothing the scoring sees
plain = transformers.AutoModel.from_pretrained(str(folder))
assert D.attention_of(plain) == D.ATTENTION, (D.attention_of(plain), transformers.__version__)
po = C(plain, D.DEFAULT_RULE, **CPU)(r)
assert torch.equal(po.summary, out.summary) and torch.equal(po.spatial, out.spatial)
print(f"the library default of transformers {transformers.__version__} without the argument is "
      f"{D.attention_of(plain)}, the same numbers as the pinned kernel")
print("attention: sdpa asked, recorded and equal to eager on the tiny model, the fallback prints")

# ---- the real weights, sdpa against eager on the CPU (only with SARTRANSFER_TEST_WEIGHTS) ----
real = None
for given in _setup.WEIGHTS:
    for candidate in (given, given / "dinov3-l-sat"):
        if (candidate / D.WEIGHTS_FILE).is_file() and (candidate / "config.json").is_file():
            real = candidate
            break
    if real is not None:
        break
if real is None:
    print("real weights: skipped, SARTRANSFER_TEST_WEIGHTS names no folder with model.safetensors")
else:
    torch.manual_seed(0)
    x_real = dn_batch(b=1)                   # one 256 px random input of digital numbers
    outs, secs = {}, {}
    for kernel in ("sdpa", "eager"):
        clock = time.perf_counter()
        model = D.load_folder(real, attention=kernel)
        D.check_model(model)
        load_s = time.perf_counter() - clock
        teacher = C(model, D.DEFAULT_RULE, **CPU)
        assert teacher.attention == kernel and (teacher.dim, teacher.grid) == (1024, 16)
        teacher(x_real)                      # a warm-up forward, not timed
        clock = time.perf_counter()
        for _ in range(3):
            o = teacher(x_real)
        secs[kernel] = (time.perf_counter() - clock) / 3
        outs[kernel] = o
        assert o.summary.shape == (1, 1024) and o.spatial.shape == (1, 16, 16, 1024)
        print(f"real weights {kernel}: loaded in {load_s:.1f} s (SHA-256 checked), forward "
              f"{secs[kernel]:.2f} s per 256 px image on the CPU")
        del model, teacher, o
        gc.collect()
    d_sum = float((outs["sdpa"].summary - outs["eager"].summary).abs().max())
    d_spa = float((outs["sdpa"].spatial - outs["eager"].spatial).abs().max())
    scale = float(outs["eager"].spatial.abs().max())
    print(f"real weights: sdpa against eager, max abs difference summary {d_sum:.2e}, spatial "
          f"{d_spa:.2e} (values up to {scale:.1f}), float32 on the CPU, sdpa {secs['sdpa']:.2f} s "
          f"against eager {secs['eager']:.2f} s per image. The CPU timings say nothing about the GPU")
    assert d_sum < 1e-3 and d_spa < 1e-3, (d_sum, d_spa)

print("test_teacher_dinov3: all checks passed")
