"""TerraFM-B teacher - vendored file, typed numbers, rules, input layer, outputs, strict loading."""
import _setup

import contextlib
import copy
import hashlib
import inspect
import io
import pickle
import re
from pathlib import Path

import huggingface_hub
import torch
import torch.nn.functional as F

from sartransfer.distil import records as R
from sartransfer.distil.teachers import Teacher, build_teacher
from sartransfer.distil.teachers import terrafm as T
from sartransfer.distil.teachers.vendor import terrafm as V

torch.manual_seed(0)
TMP = _setup.TMP_BASE / "teacher_terrafm"
TMP.mkdir(parents=True, exist_ok=True)


def fails(fn, exc, text=""):
    """`fn` must raise `exc` with `text` in its message."""
    try:
        fn()
    except exc as e:
        assert text in str(e), f"expected {text!r} in the message {e!r}"
        return
    raise SystemExit(f"no {exc.__name__} was raised")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def lf(path) -> bytes:
    """File content with LF line ends (a checkout on Windows may have turned them into CRLF)."""
    data = path.read_bytes()
    if b"\r\n" in data:
        print(f"note: {path.name} has CRLF line ends here, it is hashed with LF line ends")
    return data.replace(b"\r\n", b"\n")


def tiny(rule=None) -> Teacher:
    return T.build(rule=rule, device="cpu", dtype=torch.float32, tiny=True)


def small224(rule: str) -> Teacher:
    """The real input size and grid (224 px, 14 x 14) with a very small width."""
    model = V.TerraFM(img_size=[224], patch_size=16, embed_dim=24, depth=2, num_heads=2,
                      mlp_ratio=2, qkv_bias=True)
    return T.TerraFMTeacher(model, rule, device="cpu", dtype=torch.float32)


def same(model, state) -> bool:
    """The model holds exactly the tensors of `state`."""
    now = model.state_dict()
    return set(now) == set(state) and all(torch.equal(now[k], v) for k, v in state.items())


def printed(fn):
    """What `fn` returns and what it printed."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = fn()
    print(out.getvalue(), end="")
    return result, out.getvalue()


def layers_used(teacher: Teacher, s2: torch.Tensor):
    """Names of the input layers that ran during one call, and the output of the call."""
    calls, embed = [], teacher.model.patch_embed
    hooks = [getattr(embed, n).register_forward_hook(lambda m, i, o, n=n: calls.append(n))
             for n in ("conv2d_s2_l2a", "conv2d_s2_l1c", "conv2d_s1")]
    out = teacher(s2)
    for h in hooks:
        h.remove()
    return calls, out


# ---- 1. the vendored file: one marked change, everything else as released ----------------
ORIGINAL_SHA256 = "f2d2a8bf03549261c74ed58754502768e532cd8e7b963cc2197c48bb92f4a0ad"
VENDOR = _setup.SRC / "sartransfer" / "distil" / "teachers" / "vendor" / "terrafm"
NOTE = (rb"# >>> sar-transfer part three, note on this copy \(begin\)\n.*?"
        rb"# <<< sar-transfer part three, note on this copy \(end\)\n")
LINE = (rb"( *)# >>> sar-transfer part three, changed line \(begin\)\n *# original - (.*?)\n.*?"
        rb"# <<< sar-transfer part three, changed line \(end\)\n")

code = lf(VENDOR / "terrafm.py")
notice = (VENDOR / "NOTICE.md").read_text(encoding="utf-8")
assert code.startswith(b"# >>> sar-transfer part three, note on this copy (begin)\n# CHANGED FILE")
assert len(re.findall(NOTE, code, flags=re.S)) == 1 and len(re.findall(LINE, code, flags=re.S)) == 1
assert code.count(b"sar-transfer part three") == 4, "a marked place that the test does not know"
back = re.sub(NOTE, b"", code, count=1, flags=re.S)
back = re.sub(LINE, rb"\1\2\n", back, count=1, flags=re.S)
assert sha(back) == ORIGINAL_SHA256, "the vendored file differs from the original outside the marks"
assert sha(code) != ORIGINAL_SHA256
for must in (sha(code), ORIGINAL_SHA256, sha(lf(VENDOR / "LICENSE")), T.HF_REVISION,
             T.WEIGHTS_SHA256, f"{T.WEIGHTS_BYTES:,}"):
    assert must in notice, f"NOTICE.md does not list {must}"
assert lf(VENDOR / "LICENSE").lstrip().startswith(b"Apache License")
assert b"Version 2.0, January 2004" in lf(VENDOR / "LICENSE")
# the folder holds the Apache 2.0 text and no MIT text, and NOTICE.md says exactly that
held = b"".join(lf(p) for p in sorted(VENDOR.iterdir()) if p.is_file() and p.name != "NOTICE.md")
assert b"Permission is hereby granted" not in held, "an MIT text is here now, update NOTICE.md"
assert "either licence" not in notice
assert "holds no MIT text" in notice and "must be added" in notice

# the licence record: dated, the three statements with their exact words, no colon in a heading
named = re.search(r"docs/licences/terrafm-\d{4}-\d{2}-\d{2}\.md", T.TerraFMTeacher.licence)
assert named and (_setup.REPO / named.group(0)).is_file()
record = (_setup.REPO / named.group(0)).read_text(encoding="utf-8")
for must in ("https://huggingface.co/MBZUAI/TerraFM", '"license: apache-2.0"',
             "https://github.com/mbzuai-oryx/TerraFM", '"license": null',
             "https://github.com/mbzuai-oryx/TerraFM/issues/5",
             '"We are following the MIT Open Source License for both the code and model '
             'checkpoints. We will update the repository with the official LICENSE file shortly."'):
    assert must in record, f"the licence record does not hold {must}"
# the answer in issue 3, the authors' own words on the evaluation code and the pull request
ISSUE3 = "https://github.com/mbzuai-oryx/TerraFM/issues/3#issuecomment-3976648098"
for must in (ISSUE3, "https://github.com/xiong-zhitong/DOFA-pytorch",
             '"For evaluation and benchmarking, we followed the DOFA evaluation setup"',
             '"TerraFM is also integrated into TerraTorch, and the implementation can be found '
             'here (currently as an open PR)"'):
    assert must in record, f"the licence record does not hold {must}"
assert record.count("date read") >= 4
for text in (notice, record):
    headings = [ln for ln in text.splitlines() if ln.startswith("#")]
    assert headings and not any(":" in ln for ln in headings)
for must in ("Apache-2.0", "MIT", "no LICENSE file", "2026-09-29"):
    assert must in T.TerraFMTeacher.licence

# ---- 2. the numbers typed from the pull request ------------------------------------------
assert T.PR_BANDS == R.S2_BANDS and len(T.PR_BANDS) == 12 and "B10" not in T.PR_BANDS
assert T.PR947_COMMIT == "43ed76239163f8a4d8319e0ce4667a2d6c5bbdbc"
assert T.PR_MINMAX_SCALE == 2.0
# typed from the pull request file itself (terratorch/models/backbones/terrafm.py, lines 21
# to 26) on 2026-09-29 and compared value by value, so one wrong digit fails offline too
TYPED = {
    "TerraFM_MEAN_S2_L2A": (T.PR_MEAN_S2_L2A, (
        1033.9414, 1135.3391, 1382.2553, 1526.9789, 1883.3911, 2516.2726, 2782.7811, 2829.3833,
        2977.2936, 3004.2905, 2832.1983, 2186.6507)),
    "TerraFM_STD_S2_L2A": (T.PR_STD_S2_L2A, (
        2761.4226, 807.365, 919.6919, 1221.4198, 1265.8146, 1323.0764, 1428.7845, 1459.0343,
        1497.9797, 1577.071, 1738.6128, 1620.8854)),
    "TerraFM_MEAN_S3_L1C": (T.PR_MEAN_S2_L1C, (
        2153.94, 1937.191, 1885.8236, 1906.7263, 2096.8369, 2692.8083, 3015.8569, 2930.3957,
        3243.99, 1389.849, 2915.8289, 2186.0514)),
    "TerraFM_STD_S3_L1C": (T.PR_STD_S2_L1C, (
        581.6321, 619.9756, 714.7784, 1018.3657, 1021.4312, 1116.4055, 1265.8559, 1234.6123,
        1374.9582, 608.362, 1547.9093, 1336.0316)),
    "TerraFM_MEAN_S1": (T.PR_MEAN_S1, (
        -12.7572, -19.6423)),
    "TerraFM_STD_S1": (T.PR_STD_S1, (
        5.8491, 5.8516)),
}
for name, (typed, exact) in TYPED.items():
    assert isinstance(typed, tuple) and typed == exact, name
    assert len(typed) == (2 if name.endswith("_S1") else 12), name
    if "STD" in name:
        assert all(v > 0 for v in typed), name

if _setup.NET:
    import requests
    base = f"https://raw.githubusercontent.com/msohaildanish/terratorch/{T.PR947_COMMIT}/"
    got = {}
    for f in ("terratorch/models/backbones/terrafm.py", "examples/confs/terrafm_test.yaml",
              "terratorch/datasets/transforms.py"):
        r = requests.get(base + f, timeout=60)
        r.raise_for_status()
        got[f] = r.text
    for name, (typed, *_) in TYPED.items():
        m = re.search(rf"^{name}\s*=\s*\[(.*?)\]", got["terratorch/models/backbones/terrafm.py"],
                      re.M)
        assert tuple(float(v) for v in m.group(1).split(",")) == typed, name
    conf = got["examples/confs/terrafm_test.yaml"]
    for key in ("dataset_bands", "backbone_bands"):
        m = re.search(rf"{key}:\n((?:\s+- \w+\n)+)", conf)
        assert tuple(re.findall(r"- (\w+)", m.group(1))) == T.PR_BANDS, key
    assert conf.count("scale: 2.0") == 4                  # train, val, test and predict
    rule = got["terratorch/datasets/transforms.py"]
    rule = rule[rule.index("class MinMaxNormalize"):]
    rule = rule[:re.search(r"\n(def|class) ", rule).start()]          # the class, nothing after it
    assert "def apply" in rule and "self._min = self._means - self.scale * self._stds" in rule
    assert "self._div = np.maximum(2.0 * self.scale * self._stds" in rule
    assert "return (tensor - min_value) / div_value" in rule
    assert "clip" not in rule and "clamp" not in rule
    print("network check done: typed numbers, band order and min-max rule equal the pull request")
else:
    print("SKIPPED (SARTRANSFER_TEST_NET is not 1): typed numbers against the pull request files")

# ---- 3. rules and scaling ----------------------------------------------------------------
assert T.DEFAULT_RULE == "l2a_minmax"
assert T.RULES == {"l2a_minmax": ("l2a", "minmax"), "l1c_minmax": ("l1c", "minmax"),
                   "l2a_standardise": ("l2a", "standardise"),
                   "l1c_standardise": ("l1c", "standardise")}
C = T.TerraFMTeacher
assert (C.name, C.dim, C.patch, C.input_size) == ("terrafm-b", 768, 16, 224)
assert C.input_size // C.patch == 14
assert C.bands == R.S2_BANDS and C.rules == tuple(T.RULES) and issubclass(C, Teacher)
assert set(C.rule_sources) == set(T.RULES)
for name, (layer, scaling) in T.RULES.items():
    text = C.rule_sources[name]
    assert "pull request 947" in text and T.PR947_COMMIT in text and "secondary source" in text
    assert f"conv2d_s2_{layer}" in text
    assert ("4 std" in text) == (scaling == "minmax")
    assert ("a control" in text) == (scaling != "minmax")
    # issue 3 of the official repository is named, and nothing claims that no page exists
    assert ISSUE3 in text and "No page names it" not in text
    assert ("DOFA evaluation setup" in text and "geobench_wrapper.py" in text
            and "b915a2f6d2983c04fd08a270a09e5032e9eb91a9" in text) == (scaling != "minmax")
assert "only written source" not in T.__doc__ and "The only" not in T.__doc__
assert "issue 3" in T.__doc__ and "DOFA evaluation setup" in T.__doc__
assert C.weights == "unknown"                   # a teacher made by hand, build() sets the mark

mean = torch.tensor(T.PR_MEAN_S2_L2A).view(1, 12, 1, 1)
std = torch.tensor(T.PR_STD_S2_L2A).view(1, 12, 1, 1)


def flat(k: float) -> torch.Tensor:
    """(2, 12, 40, 40), every band at its mean + k * std."""
    return (mean + k * std).expand(2, 12, 40, 40).clone()


WANT = {"minmax": {0: 0.5, -2: 0.0, 2: 1.0, 6: 2.0, -6: -1.0},        # 2.0 and -1.0 = not clipped
        "standardise": {0: 0.0, 1: 1.0, -3: -3.0, 9: 9.0}}
teachers = {name: tiny(name) for name in T.RULES}
assert tiny().rule == T.DEFAULT_RULE
for name, (layer, scaling) in T.RULES.items():
    t = teachers[name]
    assert (t.rule, t.layer, t.dim, t.patch, t.input_size, t.grid) == (name, layer, 24, 16, 32, 2)
    for k, want in WANT[scaling].items():
        x = t.prepare(flat(k))
        assert x.shape == (2, 12, 32, 32) and x.dtype == torch.float32
        assert torch.allclose(x, torch.full_like(x, want), atol=1e-4), (name, k)
# one value by hand. B02 at 0 -> (0 - (1135.3391 - 2 * 807.365)) / (4 * 807.365)
x = teachers["l2a_minmax"].prepare(torch.zeros(1, 12, 32, 32))
assert abs(float(x[0, 1, 0, 0]) - 0.148443) < 1e-5
x = teachers["l2a_standardise"].prepare(torch.zeros(1, 12, 32, 32))
assert abs(float(x[0, 1, 0, 0]) - (-1135.3391 / 807.365)) < 1e-5
# the layer does not change the scaling
s2 = torch.rand(2, 12, 64, 64) * 4000
assert torch.equal(teachers["l2a_minmax"].prepare(s2), teachers["l1c_minmax"].prepare(s2))
assert torch.equal(teachers["l2a_standardise"].prepare(s2), teachers["l1c_standardise"].prepare(s2))
assert not torch.allclose(teachers["l2a_minmax"].prepare(s2),
                          teachers["l2a_standardise"].prepare(s2))

# resize 256 -> 224, bilinear WITH antialias
t224 = small224("l2a_minmax")
assert (t224.input_size, t224.patch, t224.grid) == (224, 16, 14)
s2 = torch.rand(2, 12, 256, 256) * 4000
scaled = (s2 - (mean - 2 * std)) / (4 * std)
x = t224.prepare(s2)
assert x.shape == (2, 12, 224, 224)
KW = dict(size=(224, 224), mode="bilinear", align_corners=False)
with_aa = F.interpolate(scaled, antialias=True, **KW)
without = F.interpolate(scaled, antialias=False, **KW)
assert torch.allclose(x, with_aa, atol=1e-5) and not torch.allclose(x, without, atol=1e-3)
assert torch.allclose(t224.prepare(s2[..., :224, :224]), scaled[..., :224, :224], atol=1e-6)

# ---- 4. the input layer ------------------------------------------------------------------
s2 = torch.rand(2, 12, 64, 64) * 4000
for name, (layer, _) in T.RULES.items():
    calls, _ = layers_used(teachers[name], s2)
    assert calls == [f"conv2d_s2_{layer}"], (name, calls)

# same weights, two rules that share ONE model: each call uses its own layer
model = tiny().model
a = T.TerraFMTeacher(model, "l2a_minmax", device="cpu", dtype=torch.float32)
b = T.TerraFMTeacher(model, "l1c_minmax", device="cpu", dtype=torch.float32)
for t, layer in ((a, "l2a"), (b, "l1c"), (a, "l2a"), (a, "l2a"), (b, "l1c")):
    calls, _ = layers_used(t, s2)
    assert calls == [f"conv2d_s2_{layer}"], calls
assert not torch.allclose(a(s2).summary, b(s2).summary, atol=1e-4)
assert not torch.allclose(a(s2).spatial, b(s2).spatial, atol=1e-4)

# a model that was never touched by the wrapper behaves as released: L1C layer
fresh = V.TerraFM(img_size=[32], patch_size=16, embed_dim=24, depth=1, num_heads=2).eval()
assert not hasattr(fresh, "is_l2a")
x = torch.randn(2, 12, 32, 32)
with torch.inference_mode():
    released = fresh(x)
    fresh.is_l2a = False
    assert torch.equal(fresh(x), released)
    fresh.is_l2a = True
    assert not torch.allclose(fresh(x), released, atol=1e-4)
    tokens = fresh.patch_embed(x, is_l2a=True)
    assert torch.allclose(fresh.prepare_tokens(x)[:, 1:], tokens + fresh.pos_embed[:, 1:],
                          atol=1e-6)

# ---- 5. outputs --------------------------------------------------------------------------
for t, (b, g, d) in ((tiny(), (3, 2, 24)), (t224, (2, 14, 24))):
    s2 = torch.rand(b, 12, 256, 256) * 4000
    out = t(s2)
    assert out.summary.shape == (b, d) and out.spatial.shape == (b, g, g, d)
    assert out.summary.dtype == torch.float32 and out.spatial.dtype == torch.float32
    assert not t.model.training and not any(p.requires_grad for p in t.model.parameters())
    x = t.prepare(s2)
    last = len(t.model.blocks) - 1
    with torch.inference_mode():
        # summary = class token after the final norm = what the authors' forward() returns
        assert torch.allclose(out.summary, t.model(x), atol=1e-6)
        # spatial = the authors' own patch grid of the last layer (extract_feature applies the norm)
        theirs = t.model.extract_feature(x, out_indices=[last])[0]            # (B, dim, g, g)
        assert torch.allclose(out.spatial, theirs.permute(0, 2, 3, 1), atol=1e-6)
        (raw,) = t.model.get_intermediate_layers(x, n=1, norm=False)          # (B, g * g, dim)
        assert torch.allclose(out.spatial.reshape(b, g * g, d), t.model.norm(raw), atol=1e-6)
        assert not torch.allclose(out.spatial.reshape(b, g * g, d), raw, atol=1e-3)

# rows first: a change inside the patch of row 3, column 9 moves the token at [3, 9] most
s2 = torch.rand(1, 12, 224, 224) * 4000
moved = s2.clone()
moved[:, :, 3 * 16:4 * 16, 9 * 16:10 * 16] += 3000.0
diff = (t224(moved).spatial - t224(s2).spatial).norm(dim=-1)[0]               # (14, 14)
assert divmod(int(diff.argmax()), 14) == (3, 9), divmod(int(diff.argmax()), 14)

# ---- 6. guards ---------------------------------------------------------------------------
fails(lambda: tiny("l2a"), ValueError, "unknown rule")
fails(lambda: T.TerraFMTeacher(model, "other", device="cpu"), ValueError, "unknown rule")
t = tiny()
fails(lambda: t(torch.zeros(2, 3, 64, 64)), ValueError)
fails(lambda: t(torch.zeros(2, 13, 64, 64)), ValueError)
fails(lambda: t(torch.zeros(12, 64, 64)), ValueError)
fails(lambda: t(torch.zeros(2, 12, 64, 64, dtype=torch.int16)), ValueError)

# ---- 7. the build function ---------------------------------------------------------------
ONLY = inspect.Parameter.KEYWORD_ONLY
assert [(p.name, p.default, p.kind) for p in inspect.signature(T.build).parameters.values()] == [
    ("variant", None, ONLY), ("rule", None, ONLY), ("device", "cuda", ONLY),
    ("dtype", torch.bfloat16, ONLY), ("token", None, ONLY), ("weights_dir", None, ONLY),
    ("random_init", False, ONLY), ("tiny", False, ONLY)]
t = build_teacher("terrafm-b", device="cpu", dtype=torch.float32, tiny=True)
assert isinstance(t, C) and t.rule == T.DEFAULT_RULE
assert (T.HF_REPO, T.WEIGHTS_FILE) == ("MBZUAI/TerraFM", "TerraFM-B.pth")
assert T.WEIGHTS_BYTES == 451862010
assert T.HF_REVISION == "3631173fec2c3a445a71e15985e0f3d5aaf4ecb3"
assert T.WEIGHTS_SHA256 == "8bdb6a4cfd707a09d79f4790119daa9d76bb316b6ea756c6eb305f99d1797a06"


class Stop(Exception):
    pass


seen = {}


def no_download(*args, **kwargs):
    raise AssertionError("a download was attempted")


def record_download(repo, name, **kwargs):
    seen.update(repo=repo, name=name, **kwargs)
    raise Stop


# Stand-ins, so every line of build() runs offline with small things. `small_base` takes the
# place of the real architecture, the file in `standin` the place of TerraFM-B.pth.
made, checked = [], []
released = copy.deepcopy(tiny().model.state_dict())        # what the stand-in file holds
standin = TMP / "standin"
standin.mkdir(exist_ok=True)
torch.save(released, standin / T.WEIGHTS_FILE)


def small_base():
    made.append(V.TerraFM(img_size=[32], patch_size=16, embed_dim=24, depth=2, num_heads=2,
                          mlp_ratio=2, qkv_bias=True))
    return made[-1]


def no_check(*args, **kwargs):
    raise AssertionError("a file was checked although no file is used")


def check_standin(path):
    """The real check, against the size and the hash of the stand-in file."""
    checked.append(Path(path))
    file = standin / T.WEIGHTS_FILE
    real_check(path, n_bytes=file.stat().st_size, sha256=T.sha256_of(file))


real_check, real_base = T.check_weights_file, T.terrafm_base
real_download = huggingface_hub.hf_hub_download
try:
    huggingface_hub.hf_hub_download = no_download
    # a token is accepted and dropped
    t = T.build(token="not-a-real-token", device="cpu", dtype=torch.float32, tiny=True)
    assert "not-a-real-token" not in repr(vars(t))
    # weights_dir given = no download, the file must be there and must be the pinned one
    empty = TMP / "empty"
    empty.mkdir(exist_ok=True)
    fails(lambda: T.build(device="cpu", weights_dir=empty), FileNotFoundError,
          "nothing is downloaded")
    wrong = TMP / "wrong"
    wrong.mkdir(exist_ok=True)
    (wrong / T.WEIGHTS_FILE).write_bytes(b"not the weights")
    fails(lambda: T.build(device="cpu", weights_dir=wrong), RuntimeError, "expected 451862010")
    # an unknown rule stops before anything else
    fails(lambda: T.build(rule="other", device="cpu"), ValueError, "unknown rule")
    # no weights_dir = the pinned revision, without a token, even when one was passed
    huggingface_hub.hf_hub_download = record_download
    fails(lambda: T.build(device="cpu", token="not-a-real-token"), Stop)
    assert seen == {"repo": "MBZUAI/TerraFM", "name": "TerraFM-B.pth",
                    "revision": T.HF_REVISION, "token": False}, seen

    # random_init = the real architecture from terrafm_base(), no download, no file checked,
    # and the teacher and the output say that the weights are random
    huggingface_hub.hf_hub_download = no_download
    T.terrafm_base, T.check_weights_file = small_base, no_check
    t, text = printed(lambda: T.build(device="cpu", dtype=torch.float32, random_init=True))
    assert len(made) == 1 and t.model is made[0], "random_init did not use terrafm_base()"
    assert t.weights == "random" and "RANDOM weights, no file loaded" in text
    assert not same(t.model, released)
    # tiny = random weights too. A given weights_dir is not used, and the output says so
    t, text = printed(lambda: T.build(device="cpu", dtype=torch.float32, tiny=True,
                                      weights_dir=TMP / "not-there"))
    assert len(made) == 1 and t.weights == "random"
    assert "RANDOM weights, no file loaded" in text and "not-there is not used" in text
    _, text = printed(tiny)
    assert "RANDOM weights, no file loaded" in text and "is not used" not in text

    # weights_dir = that file is checked and then LOADED into terrafm_base()
    T.check_weights_file = check_standin
    t, text = printed(lambda: T.build(device="cpu", dtype=torch.float32, weights_dir=standin))
    assert checked == [standin / T.WEIGHTS_FILE], checked
    assert len(made) == 2 and t.model is made[1], "the file was not loaded into terrafm_base()"
    assert same(t.model, released), "build() did not load the weights of the file"
    assert t.weights == T.WEIGHTS_SHA256 and "RANDOM" not in text
    # no weights_dir = the file from the Hugging Face cache is checked and loaded the same way
    huggingface_hub.hf_hub_download = lambda repo, name, **kw: str(standin / T.WEIGHTS_FILE)
    checked.clear()
    t, text = printed(lambda: T.build(device="cpu", dtype=torch.float32))
    assert checked == [standin / T.WEIGHTS_FILE], checked
    assert len(made) == 3 and t.model is made[2] and same(t.model, released)
    assert t.weights == T.WEIGHTS_SHA256 and "RANDOM" not in text
    # and with the pinned size and hash a file that is not TerraFM-B.pth is refused there too
    T.check_weights_file = real_check
    fails(lambda: T.build(device="cpu"), RuntimeError, "expected 451862010")
    assert len(made) == 3, "the model was built before the file was checked"
finally:
    huggingface_hub.hf_hub_download = real_download
    T.check_weights_file, T.terrafm_base = real_check, real_base

# ---- 8. size, hash and strict key check --------------------------------------------------
small = TMP / "small.bin"
small.write_bytes(b"abc")
assert T.sha256_of(small) == sha(b"abc")
T.check_weights_file(small, n_bytes=3, sha256=sha(b"abc"))
fails(lambda: T.check_weights_file(small, n_bytes=4, sha256=sha(b"abc")), RuntimeError, "bytes")
fails(lambda: T.check_weights_file(small, n_bytes=3, sha256="0" * 64), RuntimeError, "SHA-256")
fails(lambda: T.check_weights_file(small), RuntimeError, "expected 451862010")


def saved(name: str, state):
    torch.save(state, TMP / name)
    return TMP / name


source, target, other = tiny().model, tiny().model, tiny().model
good = copy.deepcopy(source.state_dict())
assert not same(target, good)
T.load_weights(target, saved("good.pth", good))
assert same(target, good)
s2 = torch.rand(2, 12, 64, 64) * 4000
loaded = T.TerraFMTeacher(target, T.DEFAULT_RULE, device="cpu", dtype=torch.float32)
first = T.TerraFMTeacher(source, T.DEFAULT_RULE, device="cpu", dtype=torch.float32)
assert torch.equal(loaded(s2).spatial, first(s2).spatial)

bad = copy.deepcopy(other.state_dict())          # differs from what `target` holds now
CASES = {
    "missing": ({k: v for k, v in bad.items() if k != "norm.weight"},
                "1 missing, 0 unexpected, 0 wrong-shape"),
    "missing_layer": ({k: v for k, v in bad.items() if "conv2d_s2_l2a" not in k},
                      "2 missing, 0 unexpected"),
    "head": ({**bad, "head.weight": torch.zeros(3, 24)}, "0 missing, 1 unexpected"),
    "decoder": ({**bad, "decoder.0.weight": torch.zeros(3, 24), "decoder.0.bias": torch.zeros(3)},
                "0 missing, 2 unexpected"),
    "shape": ({**bad, "pos_embed": torch.zeros(1, 10, 24)}, "0 unexpected, 1 wrong-shape"),
    "wrapped": ({"state_dict": bad}, f"{len(bad)} missing, 1 unexpected"),
    "prefixed": ({f"module.{k}": v for k, v in bad.items()},
                 f"{len(bad)} missing, {len(bad)} unexpected"),
}
for name, (state, text) in CASES.items():
    path = saved(f"{name}.pth", state)
    fails(lambda: T.load_weights(target, path), RuntimeError, text)
    assert same(target, good), f"{name}: the model was changed by a file that failed the check"
fails(lambda: T.load_weights(target, saved("tensor.pth", torch.zeros(3))), RuntimeError,
      "expected a state dict")


class Thing:
    pass


# weights_only=True: a file that holds more than tensors is refused by torch itself
fails(lambda: T.load_weights(target, saved("object.pth", {"x": Thing()})), pickle.UnpicklingError)
assert same(target, good)

# ---- 9. the radar teacher of part four, TerraFM-B reading Sentinel-1 (slot 4)
RADAR = "pr947_s1_minmax"
RC = T.TerraFMRadar
assert T.RADAR_RULE == RADAR and issubclass(RC, Teacher)
assert (RC.name, RC.dim, RC.patch, RC.input_size) == ("terrafm-b-radar", 768, 16, 224)
assert RC.input_size // RC.patch == 14 and RC.bands == ()
assert (RC.reads_s2, RC.reads_radar, RC.reads_aerial) == (False, True, False)
assert (C.reads_s2, C.reads_radar, C.reads_aerial) == (True, False, False)     # the Sentinel-2 one as before
assert RC.rules == (RADAR,) and set(RC.rule_sources) == {RADAR} and RC.licence == C.licence
assert RC.weights == "unknown"
text = RC.rule_sources[RADAR]
for must in ("conv2d_s1", "lines 21 and 22", "4 std", "no", "clipping", "terrafm-s1", "pull request 947",
             T.PR947_COMMIT, "secondary source", ISSUE3, "2026-10-03"):
    assert must in text, f"the radar rule source does not say {must!r}"
assert "pr947_s1_minmax" in T.__doc__ and "conv2d_s1" in T.__doc__
assert T.VARIANTS == {None: (C, T.DEFAULT_RULE), "radar": (RC, RADAR)}


def tiny_radar(rule=None) -> Teacher:
    return T.build(variant="radar", rule=rule, device="cpu", dtype=torch.float32, tiny=True)


def small224_radar() -> Teacher:
    model = V.TerraFM(img_size=[224], patch_size=16, embed_dim=24, depth=2, num_heads=2,
                      mlp_ratio=2, qkv_bias=True)
    return RC(model, RADAR, device="cpu", dtype=torch.float32)


def radar_layers_used(teacher: Teacher, s1: torch.Tensor):
    """Names of the input layers that ran during one radar call, and the output of the call."""
    calls, embed = [], teacher.model.patch_embed
    hooks = [getattr(embed, n).register_forward_hook(lambda m, i, o, n=n: calls.append(n))
             for n in ("conv2d_s2_l2a", "conv2d_s2_l1c", "conv2d_s1")]
    out = teacher.radar(s1)
    for h in hooks:
        h.remove()
    return calls, out


# build. variant None is the Sentinel-2 teacher exactly as before, "radar" the new one, both
# through the registry too; an unknown variant and an unknown rule stop before any download
r, text = printed(tiny_radar)
assert isinstance(r, RC) and r.rule == RADAR and r.weights == "random"
assert (r.dim, r.patch, r.input_size, r.grid, r.layer) == (24, 16, 32, 2, "s1")
assert "terrafm-b-radar: RANDOM weights, no file loaded" in text and "input layer conv2d_s1" in text
assert "conv2d_s2" not in text.split("input layer")[1].split(".")[0]
assert isinstance(build_teacher("terrafm-b-radar", device="cpu", dtype=torch.float32, tiny=True), RC)
assert type(build_teacher("terrafm-b", device="cpu", dtype=torch.float32, tiny=True)) is C
assert T.build(device="cpu", dtype=torch.float32, tiny=True).rule == T.DEFAULT_RULE
huggingface_hub.hf_hub_download = no_download
try:
    fails(lambda: T.build(variant="optical", device="cpu"), ValueError, "unknown variant")
    fails(lambda: T.build(variant="radar", rule="l2a_minmax", device="cpu"), ValueError, "unknown rule")
    fails(lambda: T.build(variant="radar", device="cpu", weights_dir=empty), FileNotFoundError,
          "nothing is downloaded")
finally:
    huggingface_hub.hf_hub_download = real_download
fails(lambda: tiny_radar("l2a_minmax"), ValueError, "unknown rule")
fails(lambda: RC(r.model, "l2a_minmax", device="cpu"), ValueError, "unknown rule")
fails(lambda: build_teacher("terrafm-b-radar", rule="l2a_minmax", device="cpu", tiny=True), ValueError,
      "unknown rule")

# the scaling. (s1 - mean) / std in dB, then x / (2 * scale) + 0.5 = (s1 - (mean - 2 std)) / (4 std)
mean_s1 = torch.tensor(T.PR_MEAN_S1).view(1, 2, 1, 1)
std_s1 = torch.tensor(T.PR_STD_S1).view(1, 2, 1, 1)
for k, want in {0: 0.5, -2: 0.0, 2: 1.0, 6: 2.0, -6: -1.0}.items():        # 2.0 and -1.0 = not clipped
    x = r.prepare_radar((mean_s1 + k * std_s1).expand(2, 2, 40, 40).clone())
    assert x.shape == (2, 2, 32, 32) and x.dtype == torch.float32
    assert torch.allclose(x, torch.full_like(x, want), atol=1e-5), k
# one value by hand. VV at 0 dB -> (0 - (-12.7572 - 2 * 5.8491)) / (4 * 5.8491)
x = r.prepare_radar(torch.zeros(1, 2, 32, 32))
assert abs(float(x[0, 0, 0, 0]) - (12.7572 + 2 * 5.8491) / (4 * 5.8491)) < 1e-5
assert abs(float(x[0, 1, 0, 0]) - (19.6423 + 2 * 5.8516) / (4 * 5.8516)) < 1e-5
# the same mapping the scoring encoder applies to a standardised tile, standardised / 4 + 0.5
torch.manual_seed(3)
z = torch.randn(2, 2, 32, 32)
assert torch.allclose(r.prepare_radar(z * std_s1 + mean_s1), z / (2 * T.PR_MINMAX_SCALE) + 0.5, atol=1e-5)
# resize 256 -> 224, bilinear WITH antialias
r224 = small224_radar()
assert (r224.input_size, r224.patch, r224.grid) == (224, 16, 14)
z = torch.randn(2, 2, 256, 256)
s1 = z * std_s1 + mean_s1
x = r224.prepare_radar(s1)
assert x.shape == (2, 2, 224, 224)
with_aa = F.interpolate(z / 4 + 0.5, antialias=True, **KW)
without = F.interpolate(z / 4 + 0.5, antialias=False, **KW)
assert torch.allclose(x, with_aa, atol=1e-5) and not torch.allclose(x, without, atol=1e-3)
assert torch.allclose(r224.prepare_radar(s1[..., :224, :224]), (z / 4 + 0.5)[..., :224, :224], atol=1e-5)

# the input layer. The 2-channel input goes through conv2d_s1 inside the vendored PatchEmbed,
# also when the Sentinel-2 teacher that shares the SAME model has just used its own layer
s1 = torch.randn(2, 2, 64, 64) * 5.0 - 15.0
s2 = torch.rand(2, 12, 64, 64) * 4000
calls, _ = radar_layers_used(r, s1)
assert calls == ["conv2d_s1"], calls
shared = T.TerraFMTeacher(r.model, "l2a_minmax", device="cpu", dtype=torch.float32)
assert layers_used(shared, s2)[0] == ["conv2d_s2_l2a"]
assert radar_layers_used(r, s1)[0] == ["conv2d_s1"]
assert layers_used(T.TerraFMTeacher(r.model, "l1c_minmax", device="cpu", dtype=torch.float32), s2)[0] == \
    ["conv2d_s2_l1c"]
assert radar_layers_used(r, s1)[0] == ["conv2d_s1"]

# outputs. The shapes on a random dB tensor, float32, frozen, equal to the authors' own functions
for t, (b, g, d) in ((r, (3, 2, 24)), (r224, (2, 14, 24))):
    s1 = torch.randn(b, 2, 256, 256) * 5.0 - 15.0
    out = t.radar(s1)
    assert out.summary.shape == (b, d) and out.spatial.shape == (b, g, g, d)
    assert out.summary.dtype == torch.float32 and out.spatial.dtype == torch.float32
    assert bool(torch.isfinite(out.summary).all()) and bool(torch.isfinite(out.spatial).all())
    assert not t.model.training and not any(p.requires_grad for p in t.model.parameters())
    x = t.prepare_radar(s1)
    last = len(t.model.blocks) - 1
    with torch.inference_mode():
        assert torch.allclose(out.summary, t.model(x), atol=1e-6)
        theirs = t.model.extract_feature(x, out_indices=[last])[0]            # (B, dim, g, g)
        assert torch.allclose(out.spatial, theirs.permute(0, 2, 3, 1), atol=1e-6)
        (raw,) = t.model.get_intermediate_layers(x, n=1, norm=False)
        assert torch.allclose(out.spatial.reshape(b, g * g, d), t.model.norm(raw), atol=1e-6)
# rows first: a change inside the patch of row 3, column 9 moves the token at [3, 9] most
s1 = torch.randn(1, 2, 224, 224) * 5.0 - 15.0
moved = s1.clone()
moved[:, :, 3 * 16:4 * 16, 9 * 16:10 * 16] += 20.0
diff = (r224.radar(moved).spatial - r224.radar(s1).spatial).norm(dim=-1)[0]   # (14, 14)
assert divmod(int(diff.argmax()), 14) == (3, 9), divmod(int(diff.argmax()), 14)

# guards. Sentinel-2 and aerial are refused with a TypeError, a bad radar tensor with a ValueError
fails(lambda: r(torch.rand(2, 12, 64, 64) * 4000), TypeError, "does not read Sentinel-2")
fails(lambda: r.aerial(torch.rand(2, 4, 64, 64) * 255), TypeError, "no aerial target")
fails(lambda: r.prepare(torch.rand(2, 12, 64, 64) * 4000), NotImplementedError)
fails(lambda: r.radar(torch.zeros(2, 12, 64, 64)), ValueError, "expected a float tensor (B, 2, H, W)")
fails(lambda: r.radar(torch.zeros(2, 64, 64)), ValueError)
fails(lambda: r.radar(torch.zeros(2, 2, 64, 64, dtype=torch.int16)), ValueError)
fails(lambda: shared.radar(torch.zeros(2, 2, 64, 64)), TypeError, "does not read Sentinel-1")

# ---- 10. the real weights, only when they are on this machine ------------------------------
# SARTRANSFER_TEST_WEIGHTS, or the gitignored folder weights/ of this repository (the same file as
# part three holds, 2026-10-03)
found = [p for d in (*_setup.WEIGHTS, _setup.REPO / "weights") for p in d.rglob(T.WEIGHTS_FILE)]
if found:
    t = T.build(device="cpu", dtype=torch.float32, weights_dir=found[0].parent)
    assert (t.dim, t.patch, t.input_size, t.grid) == (768, 16, 224, 14)
    assert t.n_params() == 112_951_296 and len(t.model.state_dict()) == 164
    # the teacher holds the values of the file, not only its shapes
    assert same(t.model, torch.load(found[0], map_location="cpu", weights_only=True))
    assert t.weights == T.WEIGHTS_SHA256
    # LayerNorm eps of the released file. It is not stored in the weights file
    assert t.model.norm.eps == 1e-6
    assert all(b.norm1.eps == 1e-6 and b.norm2.eps == 1e-6 for b in t.model.blocks)
    out = t(torch.rand(1, 12, 256, 256) * 4000)
    assert out.summary.shape == (1, 768) and out.spatial.shape == (1, 14, 14, 768)
    assert bool(torch.isfinite(out.summary).all()) and bool(torch.isfinite(out.spatial).all())
    print(f"real weights check done: {found[0]}")

    # the radar teacher with the same file, a CPU forward at batch 1
    radar, text = printed(lambda: T.build(variant="radar", device="cpu", dtype=torch.float32,
                                          weights_dir=found[0].parent))
    assert "RANDOM" not in text and "loaded 164 tensors" in text and "input layer conv2d_s1" in text
    assert (radar.dim, radar.patch, radar.input_size, radar.grid) == (768, 16, 224, 14)
    assert radar.n_params() == 112_951_296 and radar.weights == T.WEIGHTS_SHA256
    assert same(radar.model, t.model.state_dict())
    torch.manual_seed(2)
    out = radar.radar(torch.randn(1, 2, 256, 256) * 5.0 - 15.0)
    assert out.summary.shape == (1, 768) and out.spatial.shape == (1, 14, 14, 768)
    assert bool(torch.isfinite(out.summary).all()) and bool(torch.isfinite(out.spatial).all())
    del t

    # PARITY with the scoring path "terrafm-s1" of models/encoders.py, built as the scoring code
    # builds it (a local folder as model_id, no download). The encoder takes tiles standardised with
    # the training-split statistics and applies standardised / 4 + 0.5 itself; it does not resize,
    # so the tile is 224 px here. One standardised tensor z goes to both. `_tokens` is the float32
    # token sequence the encoder grids, before the float16 of `encode`.
    import dataclasses
    from sartransfer.models import encoders as ENC
    spec = dataclasses.replace(ENC.ENCODERS["terrafm-b-s1"], model_id=str(found[0].parent))
    enc, text = printed(lambda: ENC.FrozenEncoder(spec, token=None, device="cpu"))
    assert "min-max mapping" in text and enc.dim == 768 and enc.weights == T.WEIGHTS_SHA256
    z = torch.randn(1, 2, 224, 224)
    with torch.inference_mode():
        theirs = enc._tokens(z)                                        # (1, 196, 768) float32
    assert theirs.shape == (1, 196, 768) and theirs.dtype == torch.float32
    # (a) the same standardised tensor into the teacher's forward (its prepare is the encoder's mapping)
    mine = radar._run(z / (2 * T.PR_MINMAX_SCALE) + 0.5).spatial.reshape(1, 196, 768)
    gap_same = float((mine - theirs).abs().max())
    assert gap_same < 1e-5, gap_same
    # (b) through `radar` from the dB values that standardise to z. The float32 round trip of the dB
    # values moves the input by about 1e-7 and the tokens by about 1.5e-5 (measured 2026-10-03)
    s1 = z * std_s1 + mean_s1
    assert torch.allclose(radar.prepare_radar(s1), z / (2 * T.PR_MINMAX_SCALE) + 0.5, atol=1e-6)
    mine_db = radar.radar(s1).spatial.reshape(1, 196, 768)
    gap_db = float((mine_db - theirs).abs().max())
    assert gap_db < 1e-4, gap_db
    # (c) the public path of the encoder, (1, 14, 14, 768) float16, the same grid up to float16 rounding
    grid = enc.encode(z.numpy(), batch=1)
    assert grid.shape == (1, 14, 14, 768) and grid.dtype.name == "float16"
    assert torch.allclose(torch.from_numpy(grid).float(), mine.reshape(1, 14, 14, 768), atol=1e-2)
    print(f"parity with the scoring path terrafm-s1 done, largest difference {gap_same:.1e} on the same "
          f"standardised tensor, {gap_db:.1e} through the dB values")
else:
    print("SKIPPED (no TerraFM-B.pth under SARTRANSFER_TEST_WEIGHTS or weights/): size, hash, keys, one "
          "forward pass of the real weights, the radar teacher and its parity with the scoring path")

print("test_teacher_terrafm: all checks passed")
