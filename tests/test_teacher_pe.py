"""PE-Spatial-L14-448 teacher - vendored files, typed facts, rules, outputs, 224 px grid, strict
loading. Run with  python tests/test_teacher_pe.py  (env as in tests/_setup.py)."""
import _setup

import ast
import sys

# The vendored package first, before anything that could pull einops, timm or the hub in.
from sartransfer.distil.teachers.vendor import pe as V  # noqa: E402

# (numpy is not in this list - torch itself imports it; the import-line check below covers it)
for _name in ("einops", "timm", "torchvision", "huggingface_hub", "transformers"):
    assert _name not in sys.modules, f"importing the vendored package imported {_name}"
print("vendored package imported without einops, timm, torchvision, huggingface_hub, transformers")

import contextlib  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import io  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import huggingface_hub  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sartransfer.distil import records as R  # noqa: E402
from sartransfer.distil.teachers import Teacher, build_teacher  # noqa: E402
from sartransfer.distil.teachers import dinov3_sat as D  # noqa: E402
from sartransfer.distil.teachers import pe_spatial as P  # noqa: E402

torch.manual_seed(0)
TMP = _setup.TMP_BASE / "teacher_pe"
TMP.mkdir(parents=True, exist_ok=True)
TAG = "sar-transfer part four"


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


def tiny(rule=None, input_size=224) -> Teacher:
    return P.build(rule=rule, device="cpu", dtype=torch.float32, tiny=True, input_size=input_size)


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


def scene(seed: int, size: int = 256, noise: float = 0.0) -> torch.Tensor:
    """(1, 12, size, size) digital numbers with smooth blobs and two squares, so the picture
    has structure that a feature grid can follow. `noise` adds that much white noise in DN."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij")
    bands = []
    for _ in range(12):
        img = torch.full((size, size), 600.0)
        for _ in range(4):
            cx, cy, s, a = torch.rand(4, generator=g).tolist()
            img += 2500 * a * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (0.02 + 0.08 * s))
        img[size // 8: size // 3, size // 2: size // 2 + size // 4] += 1500
        img[size // 2: size // 2 + size // 5, size // 10: size // 10 + size // 6] *= 0.3
        bands.append(img)
    x = torch.stack(bands)[None]
    if noise:
        x = x + noise * torch.randn(x.shape, generator=g)
    return x.clamp(0, 10000)


# ---- 1. the vendored files: torch and the standard library only, marks, hashes, licence ---
VENDOR = _setup.SRC / "sartransfer" / "distil" / "teachers" / "vendor" / "pe"
COPIED = ("pe.py", "rope.py", "config.py")
notice = (VENDOR / "NOTICE.md").read_text(encoding="utf-8")
stdlib = set(sys.stdlib_module_names)
for name in COPIED + ("__init__.py",):
    code = lf(VENDOR / name).decode("utf-8")
    roots = set()
    for node in ast.walk(ast.parse(code)):           # every import, at any depth
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots.add((node.module or "").split(".")[0])
    outside = sorted(r for r in roots if r != "torch" and r not in stdlib)
    assert not outside, f"{name} imports {outside}"
    # no leftover use of the dropped packages in a code line (a bare `repeat(` is einops, a
    # `.repeat(` is the tensor method and stays)
    live = [ln for ln in code.split("\n") if not ln.lstrip().startswith("#")]
    leftover = re.compile(r"(?<![\w.])(np\.|rearrange\(|repeat\(|DropPath\(|hf_hub_download\(|"
                          r"core\.vision_encoder|einops\.|timm\.)")
    for ln in live:
        assert not leftover.search(ln), f"{name} still uses a dropped package: {ln.strip()!r}"
    print(f"{name}: imports {sorted(roots)}, all torch or standard library")

for name in COPIED:
    code = lf(VENDOR / name).decode("utf-8")
    assert code.startswith(f"# >>> {TAG}, note on this copy (begin)\n# CHANGED FILE"), name
    begins = re.findall(rf"# >>> {TAG}, (\w[\w ]*?) \(begin\)", code)
    ends = re.findall(rf"# <<< {TAG}, (\w[\w ]*?) \(end\)", code)
    assert begins == ends and begins.count("note on this copy") == 1, (name, begins, ends)
    assert set(begins) <= {"note on this copy", "changed lines", "removed block"}, begins
    kinds = {k: begins.count(k) for k in set(begins)}
    for must in (sha(code.encode("utf-8")), f"{len(code.encode('utf-8')):,}"):
        assert must in notice, f"NOTICE.md does not list {must} for {name}"
    print(f"{name}: {kinds}, hash listed in NOTICE.md")
assert {"changed lines": 6, "removed block": 2} == {
    k: lf(VENDOR / "pe.py").decode().count(f"# >>> {TAG}, {k} (begin)")
    for k in ("changed lines", "removed block")}
assert lf(VENDOR / "rope.py").decode().count(f"# >>> {TAG}, changed lines (begin)") == 10
lic = lf(VENDOR / "LICENSE")
assert lic.lstrip().startswith(b"Apache License") and b"Version 2.0, January 2004" in lic
assert sha(lic) == "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
ORIGINALS = {            # as released at the pinned commit, read 2026-10-03
    "pe.py": "36b62204c4cf12a709946facce2df8980e0880bf06deb6e49a2aa288158c3d53",
    "rope.py": "85a16113138f26389ea35fccd313f2aca12945c7ed2eb24e5d52dbe680dad121",
    "config.py": "37c10ea7b68889241c8d9ebce27c62577c7379a33c444f9b0129bc90946aaff5",
}
for must in (*ORIGINALS.values(), sha(lic), P.CODE_COMMIT, P.HF_REVISION, P.WEIGHTS_SHA256,
             f"{P.WEIGHTS_BYTES:,}", f"{P.N_PARAMS:,}", str(P.N_TENSORS)):
    assert must in notice, f"NOTICE.md does not list {must}"

# the licence record: dated, the exact words, the URLs, no colon in any heading
named = re.search(r"docs/licences/pe-\d{4}-\d{2}-\d{2}\.md", P.PESpatial.licence)
assert named and (_setup.REPO / named.group(0)).is_file()
record = (_setup.REPO / named.group(0)).read_text(encoding="utf-8")
for must in ("https://huggingface.co/facebook/PE-Spatial-L14-448", '"license: apache-2.0"',
             "https://github.com/facebookresearch/perception_models", '"spdx_id": "Apache-2.0"',
             "LICENSE.PE", "LICENSE.PLM", "FAIR Noncommercial Research License",
             '"All checkpoints released on this page, unless otherwise specified, are released '
             'with the [Apache 2.0 license](https://opensource.org/license/apache-2-0). The code '
             'itself is licensed under the parent license of this repository."',
             P.CODE_COMMIT, P.HF_REVISION, P.WEIGHTS_SHA256):
    assert must in record, f"the licence record does not hold {must}"
assert record.count("date read") >= 2
for text in (notice, record):
    headings = [ln for ln in text.splitlines() if ln.startswith("#")]
    assert headings and not any(":" in ln for ln in headings), "a colon in a heading"
for must in ("Apache-2.0", "2026-10-03", "apache-2.0", "LICENSE.PE"):
    assert must in P.PESpatial.licence
print("NOTICE.md and the licence record hold the hashes, the words and the URLs, no colon in a heading")


def rebuild(copy_text: str):
    """The released file from the copy - marks out, original lines back, removed ranges as
    placeholders. Returns (lines, removed ranges)."""
    out, removed, i, lines = [], [], 0, copy_text.split("\n")
    while i < len(lines):
        ln = lines[i]
        m = re.match(rf"^(\s*)# >>> {TAG}, (note on this copy|changed lines|removed block) "
                     rf"\(begin\)(.*)$", ln)
        if not m:
            out.append(ln)
            i += 1
            continue
        base, kind, rest = m.groups()
        j = next(k for k in range(i + 1, len(lines))
                 if lines[k].strip() == f"# <<< {TAG}, {kind} (end)")
        if kind == "changed lines":
            for inner in lines[i + 1: j]:
                o = re.match(r"^(\s*)# original -( (.*))?$", inner)
                if o:
                    out.append("" if o.group(2) is None else base + o.group(3))
        elif kind == "removed block":
            a, b = map(int, re.search(r"original lines (\d+)-(\d+)", rest).groups())
            removed.append((a, b))
            out.append(f"<<removed {a}-{b}>>")
        i = j + 1
    return out, removed


if _setup.NET:
    import requests
    base = f"https://raw.githubusercontent.com/facebookresearch/perception_models/{P.CODE_COMMIT}/"
    for name in COPIED:
        r = requests.get(base + "core/vision_encoder/" + name, timeout=60)
        r.raise_for_status()
        orig = r.content
        assert sha(orig) == ORIGINALS[name], f"{name} at the pinned commit has another hash now"
        orig_lines = orig.decode("utf-8").split("\n")
        got, removed = rebuild(lf(VENDOR / name).decode("utf-8").rstrip("\n"))
        want = list(orig_lines)
        for a, b in sorted(removed, reverse=True):          # 1-based, both ends included
            want[a - 1: b] = [f"<<removed {a}-{b}>>"]
        if want and want[-1] == "" and got and got[-1] != "":
            want = want[:-1]                                # pe.py as released has no final newline
        if got and got[-1] == "" and want and want[-1] != "":
            got = got[:-1]
        first = next((k for k, (x, y) in enumerate(zip(got, want)) if x != y), None)
        where = (first, got[first:first + 2], want[first:first + 2]) if first is not None else None
        assert first is None and len(got) == len(want), (name, where, len(got), len(want))
        print(f"network check done: {name} rebuilt from the marks equals the released file "
              f"({len(removed)} removed block(s) {removed})")
    r = requests.get(base + "LICENSE.PE", timeout=60)
    r.raise_for_status()
    assert r.content == lic, "LICENSE.PE at the pinned commit differs from the vendored LICENSE"
    print("network check done: LICENSE equals LICENSE.PE at the pinned commit")
else:
    print("SKIPPED (SARTRANSFER_TEST_NET is not 1): the released files against the marks")

# ---- 2. the facts typed in the module ------------------------------------------------------
assert (P.HF_ID, P.WEIGHTS_FILE) == ("facebook/PE-Spatial-L14-448", "PE-Spatial-L14-448.pt")
assert P.HF_REVISION == "a09b69ff95973d70edd96e6fa4be216d8c6caa70"
assert P.WEIGHTS_BYTES == 1215952299
assert P.WEIGHTS_SHA256 == "47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1"
assert P.CODE_COMMIT == "3e352cca660658d4b5c90f42a7808b11469e4c66"
assert P.MEAN == (0.5, 0.5, 0.5) and P.STD == (0.5, 0.5, 0.5)
# the config is the vendored entry, field by field, and that entry is Core-L14-336 with the
# four replacements that config.py writes
assert P.FULL_CONFIG == asdict(V.PE_VISION_CONFIG["PE-Spatial-L14-448"])
core = asdict(V.PE_VISION_CONFIG["PE-Core-L14-336"])
assert P.FULL_CONFIG == dict(core, image_size=448, pool_type="none", use_ln_post=False, output_dim=None)
assert (core["use_cls_token"], core["use_abs_posemb"], core["use_rope2d"]) == (True, True, True)
assert P.TINY_CONFIG == dict(P.FULL_CONFIG, width=32, layers=2, heads=2)
# the parameter count from the config by arithmetic (no ln_post, no pool, no projection)
w, L, p, mlp = 1024, 24, 14, 4096
block = (3 * w * w + 3 * w) + (w * w + w) + 4 * w + (w * mlp + mlp) + (mlp * w + w)
assert block == 12_596_224
assert 3 * w * p * p + w + (1 + 32 * 32) * w + 2 * w + L * block == P.N_PARAMS == 303_964_160
assert 5 + 12 * L == P.N_TENSORS == 293
print(f"typed facts agree with the vendored config; {P.N_PARAMS:,} parameters by arithmetic")

C = P.PESpatial
assert (C.name, C.dim, C.patch, C.input_size) == ("pe-spatial-l", 1024, 14, 224)
assert C.input_size // C.patch == 16 and C.input_size % C.patch == 0
assert C.bands == ("B04", "B03", "B02") and issubclass(C, Teacher)
assert C.rules == ("majortom_thumbnail",) == P.RULES and P.DEFAULT_RULE == "majortom_thumbnail"
assert (C.reads_s2, C.reads_aerial, C.reads_radar) == (True, True, False)
assert (C.aerial_mean, C.aerial_std) == (P.MEAN, P.STD)
assert C.weights == "unknown"
src = C.rule_sources["majortom_thumbnail"]
assert src.startswith(D.RULE_SOURCES["majortom_thumbnail"]), "the DINOv3 thumbnail rule, as is"
for must in ("gain 1.3, gamma 0.6", "(x - 0.5) / 0.5", "transforms.py line 25", P.CODE_COMMIT,
             "2026-10-03", "bilinear"):
    assert must in src, must
assert "SAM 2.1" in P.__doc__ and "ln_post is Identity" in P.__doc__ and "bilinear" in P.__doc__
print("class attributes and the rule source as specified")

# ---- 3. the tiny model: prepare, outputs, class token, grid order, 224 px handling --------
t, text = printed(tiny)
assert "RANDOM weights" in text and "interpolated to 16 x 16 by the authors' code (bilinear)" in text
assert "summary = class token" in text and "grid 16 x 16" in text
assert (t.dim, t.patch, t.input_size, t.grid, t.posemb_grid) == (32, 14, 224, 16, 32)
assert t.summary_from == "class token" and t.weights == "random" and t.rule == "majortom_thumbnail"
assert t.rgb == [R.S2_BANDS.index(b) for b in ("B04", "B03", "B02")] == [3, 2, 1]
assert not t.model.training and not any(q.requires_grad for q in t.model.parameters())

# prepare = thumbnail rule, resize, PE normalisation. One value by hand -
# DN 1000 -> 0.1 -> 1.3 * 0.1 ** 0.6 = 0.326545 -> (0.326545 - 0.5) / 0.5 = -0.346910
x = t.prepare(torch.full((1, 12, 256, 256), 1000.0))
assert x.shape == (1, 3, 224, 224) and x.dtype == torch.float32
assert torch.allclose(x, torch.full_like(x, 1.3 * 0.1 ** 0.6 * 2 - 1), atol=1e-5)
assert abs(float(x[0, 0, 0, 0]) - (-0.346910)) < 1e-5
s2 = torch.rand(2, 12, 256, 256) * 10000
want = D.scale_rgb(s2[:, [3, 2, 1]], "majortom_thumbnail")
KW = dict(size=(224, 224), mode="bilinear", align_corners=False)
with_aa = (F.interpolate(want, antialias=True, **KW) - 0.5) / 0.5
without = (F.interpolate(want, antialias=False, **KW) - 0.5) / 0.5
assert torch.allclose(t.prepare(s2), with_aa, atol=1e-5)
assert not torch.allclose(t.prepare(s2), without, atol=1e-3)
assert torch.allclose(t.prepare(s2[..., :224, :224]), (want[..., :224, :224] - 0.5) / 0.5, atol=1e-6)
assert t.prepare(s2, size=448).shape == (2, 3, 448, 448)
assert float(t.prepare(s2).min()) >= -1 - 1e-5 and float(t.prepare(s2).max()) <= 1 + 1e-5
fails(lambda: t.prepare(torch.full((1, 12, 32, 32), float("nan"))), ValueError, "NaN")

# a random 12-band digital-number tensor -> summary (B, dim), spatial (B, 16, 16, dim), finite
out = t(s2)
assert out.summary.shape == (2, 32) and out.spatial.shape == (2, 16, 16, 32)
assert out.summary.dtype == torch.float32 and out.spatial.dtype == torch.float32
assert bool(torch.isfinite(out.summary).all()) and bool(torch.isfinite(out.spatial).all())
print(f"tiny on 12-band DN (2, 12, 256, 256): summary {tuple(out.summary.shape)}, spatial "
      f"{tuple(out.spatial.shape)}, finite")
# a random aerial tensor of 8-bit values (B, 4, 256, 256) -> same shapes, NIR dropped
aerial = torch.randint(0, 256, (2, 4, 256, 256)).float()
outa = t.aerial(aerial)
assert outa.summary.shape == (2, 32) and outa.spatial.shape == (2, 16, 16, 32)
assert bool(torch.isfinite(outa.summary).all()) and bool(torch.isfinite(outa.spatial).all())
xa = t.prepare_aerial(aerial)
assert xa.shape == (2, 3, 224, 224) and float(xa.min()) >= -1 - 1e-5 and float(xa.max()) <= 1 + 1e-5
assert torch.allclose(xa, (F.interpolate(aerial[:, :3] / 255, antialias=True, **KW) - 0.5) / 0.5, atol=1e-6)
nir = aerial.clone()
nir[:, 3] = 0
assert torch.equal(t.aerial(nir).spatial, outa.spatial), "the NIR channel changed the output"
assert torch.allclose(t.aerial(aerial[:, :3]).spatial, outa.spatial, atol=1e-6)
fails(lambda: t.aerial(aerial * 2), ValueError, "0 to 255")
fails(lambda: t.aerial(aerial.to(torch.uint8)), ValueError, "8-bit")
fails(lambda: t.aerial(torch.zeros(2, 5, 64, 64)), ValueError, "3 or 4")
print(f"tiny on aerial 8-bit values (2, 4, 256, 256): summary {tuple(outa.summary.shape)}, spatial "
      f"{tuple(outa.spatial.shape)}, finite, NIR dropped")

# summary = the class token, spatial = the patch tokens, from forward_features(norm=True);
# ln_post is Identity for this config, so norm=True equals norm=False
x = t.prepare(s2)
with torch.inference_mode():
    tokens = t.model.forward_features(x, norm=True)
    assert tokens.shape == (2, 1 + 256, 32)
    assert torch.allclose(out.summary, tokens[:, 0], atol=1e-6)
    assert torch.allclose(out.spatial, tokens[:, 1:].reshape(2, 16, 16, 32), atol=1e-6)
    assert isinstance(t.model.ln_post, torch.nn.Identity)
    assert torch.equal(tokens, t.model.forward_features(x, norm=False))
    theirs = t.model.forward_features(x, strip_cls_token=True)           # README section 3
    assert torch.allclose(out.spatial.reshape(2, 256, 32), theirs, atol=1e-6)
    assert torch.equal(t(s2).spatial, out.spatial), "two calls differ"

# rows first: a change inside the patch of row 3, column 9 moves the token at [3, 9] most
moved = x.clone()
moved[:, :, 3 * 14:4 * 14, 9 * 14:10 * 14] += 3.0
with torch.inference_mode():
    diff = (t._tokens(moved)[1] - t._tokens(x)[1]).norm(dim=-1)[0]          # (16, 16)
assert divmod(int(diff.argmax()), 16) == (3, 9), divmod(int(diff.argmax()), 16)

# 224 px handling: the stored embedding is for 32 x 32 (+ the class row) and is interpolated
# to 16 x 16 on the fly by the authors' code, bilinear, align_corners False, class row as is
pe = t.model.positional_embedding
assert pe.shape == (1 + 32 * 32, 32)
with torch.inference_mode():
    sampled = t.model._sample_abs_posemb(16, 16)
    byhand = F.interpolate(pe[1:].reshape(1, 32, 32, 32).permute(0, 3, 1, 2), size=(16, 16),
                           mode="bilinear", align_corners=False).permute(0, 2, 3, 1).reshape(256, 32)
    assert sampled.shape == (1, 257, 32)
    assert torch.equal(sampled[0, 0], pe[0]) and torch.allclose(sampled[0, 1:], byhand, atol=1e-6)
    assert t.model._sample_abs_posemb(32, 32).shape == (1, 1025, 32)
    assert torch.equal(t.model._sample_abs_posemb(32, 32)[0], pe)
    # RoPE follows the grid: positions 1..16 per axis at 224 px, 1..32 at 448 px, class at 0
    t.model.forward_features(x)
    freq = t.model.rope.freq
    assert freq.shape == (1, 257, 16) and bool((freq[0, 0] == 0).all())
    t.model.forward_features(torch.randn(1, 3, 448, 448))
    assert t.model.rope.freq.shape == (1, 1025, 16)
# a model built for 224 px needs no interpolation
native = V.VisionTransformer(**dict(P.TINY_CONFIG, image_size=224)).eval()
with torch.inference_mode():
    assert torch.equal(native._sample_abs_posemb(16, 16)[0], native.positional_embedding)
tn, text = printed(lambda: P.PESpatial(native, "majortom_thumbnail", device="cpu", dtype=torch.float32))
assert tn.posemb_grid == 16 and tn(s2).spatial.shape == (2, 16, 16, 32)
print("224 px: position embedding 32 x 32 -> 16 x 16 by the authors' bilinear interpolation, "
      "class row kept, RoPE table (257, 16) at 224 px and (1025, 16) at 448 px")

# the fall-back: input_size=448, grid 32 x 32, no interpolation
t448, text = printed(lambda: tiny(input_size=448))
assert "no interpolation" in text and "grid 32 x 32" in text
assert (t448.input_size, t448.grid) == (448, 32)
o = t448(s2[:1])
assert o.summary.shape == (1, 32) and o.spatial.shape == (1, 32, 32, 32)
assert t448.aerial(aerial[:1]).spatial.shape == (1, 32, 32, 32)
fails(lambda: tiny(input_size=200), ValueError, "no multiple of patch 14")

# grid_check runs on the tiny model and gives a number in -1..1 (random weights, no meaning)
x448 = t.prepare(scene(1), size=448)
assert x448.shape == (1, 3, 448, 448)
cos = P.grid_check(t, x448)
assert isinstance(cos, float) and -1.0 <= cos <= 1.0
print(f"grid_check on the tiny random model: mean cosine {cos:.4f} (no meaning, random weights)")
# it compares the 2 x 2 pooled 448 grid with the 224 grid of the same picture, by hand
with torch.inference_mode():
    big = t._tokens(x448)[1]
    pooled = F.avg_pool2d(big.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
    small = t._tokens(F.interpolate(x448, size=(224, 224), mode="bilinear", antialias=True,
                                    align_corners=False))[1]
    assert abs(float(F.cosine_similarity(pooled, small, dim=-1).mean()) - cos) < 1e-5
    # centred = the mean token over the patches of each picture taken out of both grids
    cen = P.grid_check(t, x448, centred=True)
    byhand = F.cosine_similarity(pooled - pooled.mean(dim=(1, 2), keepdim=True),
                                 small - small.mean(dim=(1, 2), keepdim=True), dim=-1).mean()
    assert isinstance(cen, float) and abs(float(byhand) - cen) < 1e-5 and cen != cos
print(f"grid_check on the tiny random model, centred: {cen:.4f} (no meaning, random weights)")
fails(lambda: P.grid_check(t, t.prepare(scene(1))), ValueError, "expected (B, 3, 448, 448)")
fails(lambda: P.grid_check(t448, x448), ValueError, "expected (B, 3, 896, 896)")

# ---- 4. guards ---------------------------------------------------------------------------
fails(lambda: tiny("reflectance"), ValueError, "unknown rule")
fails(lambda: P.PESpatial(t.model, "other", device="cpu"), ValueError, "unknown rule")
fails(lambda: t(torch.zeros(2, 3, 64, 64)), ValueError)
fails(lambda: t(torch.zeros(2, 13, 64, 64)), ValueError)
fails(lambda: t(torch.zeros(12, 64, 64)), ValueError)
fails(lambda: t(torch.zeros(2, 12, 64, 64, dtype=torch.int16)), ValueError)
fails(lambda: t.radar(torch.zeros(2, 2, 64, 64)), TypeError, "does not read Sentinel-1")
for _name in ("einops", "timm", "torchvision"):
    assert _name not in sys.modules, f"{_name} was imported by now"
print("guards: unknown rule, wrong shapes, no radar")

# ---- 5. the build function -----------------------------------------------------------------
ONLY = inspect.Parameter.KEYWORD_ONLY
assert [(q.name, q.default, q.kind) for q in inspect.signature(P.build).parameters.values()] == [
    ("rule", None, ONLY), ("device", "cuda", ONLY), ("dtype", torch.bfloat16, ONLY),
    ("token", None, ONLY), ("weights_dir", None, ONLY), ("random_init", False, ONLY),
    ("tiny", False, ONLY), ("input_size", 224, ONLY)]
t, _ = printed(lambda: build_teacher("pe-spatial-l", device="cpu", dtype=torch.float32, tiny=True))
assert isinstance(t, C) and t.rule == P.DEFAULT_RULE


class Stop(Exception):
    pass


seen = {}


def no_download(*args, **kwargs):
    raise AssertionError("a download was attempted")


def record_download(repo, name, **kwargs):
    seen.update(repo=repo, name=name, **kwargs)
    raise Stop


# Stand-ins, so every line of build() runs offline with small things. `small_model` takes
# the place of the real architecture, the file in `standin` the place of the weights file.
made, checked = [], []
released = copy.deepcopy(t.model.state_dict())        # what the stand-in file holds
standin = TMP / "standin"
standin.mkdir(exist_ok=True)
torch.save(released, standin / P.WEIGHTS_FILE)


def small_model(name=P.CONFIG_NAME):
    assert name == P.CONFIG_NAME
    made.append(V.VisionTransformer(**P.TINY_CONFIG))
    return made[-1]


def no_check(*args, **kwargs):
    raise AssertionError("a file was checked although no file is used")


def check_standin(path):
    """The real check, against the size and the hash of the stand-in file."""
    checked.append(Path(path))
    file = standin / P.WEIGHTS_FILE
    real_check(path, n_bytes=file.stat().st_size, sha256=D.file_sha256(file))


def small_ok(model, config=None, n_params=None):
    assert isinstance(model, V.VisionTransformer)


real_check, real_make, real_model_check = P.check_weights_file, P.make_model, P.check_model
real_download = huggingface_hub.hf_hub_download
try:
    huggingface_hub.hf_hub_download = no_download
    # a token is accepted and dropped
    t, _ = printed(lambda: P.build(token="not-a-real-token", device="cpu", dtype=torch.float32, tiny=True))
    assert "not-a-real-token" not in repr(vars(t))
    # weights_dir given = no download, the file must be there and must be the pinned one
    empty = TMP / "empty"
    empty.mkdir(exist_ok=True)
    fails(lambda: P.build(device="cpu", weights_dir=empty), FileNotFoundError, "nothing is downloaded")
    wrong = TMP / "wrong"
    wrong.mkdir(exist_ok=True)
    (wrong / P.WEIGHTS_FILE).write_bytes(b"not the weights")
    fails(lambda: P.build(device="cpu", weights_dir=wrong), RuntimeError, "expected 1215952299")
    # an unknown rule stops before anything else
    fails(lambda: P.build(rule="other", device="cpu"), ValueError, "unknown rule")
    # no weights_dir = the pinned revision, without a token, even when one was passed
    huggingface_hub.hf_hub_download = record_download
    fails(lambda: P.build(device="cpu", token="not-a-real-token"), Stop)
    assert seen == {"repo": "facebook/PE-Spatial-L14-448", "name": "PE-Spatial-L14-448.pt",
                    "revision": P.HF_REVISION, "token": False}, seen

    # random_init = the real architecture from make_model(), no download, no file checked
    huggingface_hub.hf_hub_download = no_download
    P.make_model, P.check_weights_file, P.check_model = small_model, no_check, small_ok
    t, text = printed(lambda: P.build(device="cpu", dtype=torch.float32, random_init=True))
    assert len(made) == 1 and t.model is made[0], "random_init did not use make_model()"
    assert t.weights == "random" and "RANDOM weights, no file loaded" in text
    assert not same(t.model, released)
    # tiny = random weights too. A given weights_dir is not used, and the output says so
    t, text = printed(lambda: P.build(device="cpu", dtype=torch.float32, tiny=True,
                                      weights_dir=TMP / "not-there"))
    assert len(made) == 1 and t.weights == "random"
    assert "RANDOM weights, no file loaded" in text and "not-there is not used" in text
    _, text = printed(tiny)
    assert "RANDOM weights, no file loaded" in text and "is not used" not in text

    # weights_dir = that file is checked and then LOADED into make_model()
    P.check_weights_file = check_standin
    t, text = printed(lambda: P.build(device="cpu", dtype=torch.float32, weights_dir=standin))
    assert checked == [standin / P.WEIGHTS_FILE], checked
    assert len(made) == 2 and t.model is made[1], "the file was not loaded into make_model()"
    assert same(t.model, released), "build() did not load the weights of the file"
    assert t.weights == P.WEIGHTS_SHA256 and "RANDOM" not in text and "loaded 29 tensors" in text
    # no weights_dir = the file from the Hugging Face cache is checked and loaded the same way
    huggingface_hub.hf_hub_download = lambda repo, name, **kw: str(standin / P.WEIGHTS_FILE)
    checked.clear()
    t, text = printed(lambda: P.build(device="cpu", dtype=torch.float32, input_size=448))
    assert checked == [standin / P.WEIGHTS_FILE], checked
    assert len(made) == 3 and t.model is made[2] and same(t.model, released)
    assert t.weights == P.WEIGHTS_SHA256 and "RANDOM" not in text and t.grid == 32
    # and with the pinned size and hash a file that is not the weights file is refused
    P.check_weights_file = real_check
    fails(lambda: P.build(device="cpu"), RuntimeError, "expected 1215952299")
    assert len(made) == 3, "the model was built before the file was checked"
finally:
    huggingface_hub.hf_hub_download = real_download
    P.check_weights_file, P.make_model, P.check_model = real_check, real_make, real_model_check
print("build(): token dropped, pinned revision without a token, weights_dir honoured, file checked "
      "before the model is built, random_init and tiny say RANDOM")

# ---- 6. size, hash and the strict key check ------------------------------------------------
small = TMP / "small.bin"
small.write_bytes(b"abc")
P.check_weights_file(small, n_bytes=3, sha256=sha(b"abc"))
fails(lambda: P.check_weights_file(small, n_bytes=4, sha256=sha(b"abc")), RuntimeError, "bytes")
fails(lambda: P.check_weights_file(small, n_bytes=3, sha256="0" * 64), RuntimeError, "SHA-256")
fails(lambda: P.check_weights_file(small), RuntimeError, "expected 1215952299")


def saved(name: str, state):
    torch.save(state, TMP / name)
    return TMP / name


source = printed(tiny)[0].model
target = printed(tiny)[0].model
other = printed(tiny)[0].model
good = copy.deepcopy(source.state_dict())
assert not same(target, good)
_, text = printed(lambda: P.load_weights(target, saved("good.pth", good)))
assert same(target, good) and "loaded 29 tensors" in text and "none unexpected" in text
loaded = P.PESpatial(target, P.DEFAULT_RULE, device="cpu", dtype=torch.float32)
first = P.PESpatial(source, P.DEFAULT_RULE, device="cpu", dtype=torch.float32)
assert torch.equal(loaded(s2).spatial, first(s2).spatial)

bad = copy.deepcopy(other.state_dict())          # differs from what `target` holds now
CASES = {
    "missing": ({k: v for k, v in bad.items() if k != "ln_pre.weight"},
                "1 missing, 0 unexpected, 0 wrong-shape"),
    "missing_block": ({k: v for k, v in bad.items() if "resblocks.1." not in k},
                      "12 missing, 0 unexpected"),
    "ln_post": ({**bad, "ln_post.weight": torch.ones(32), "ln_post.bias": torch.zeros(32)},
                "0 missing, 2 unexpected"),
    "proj": ({**bad, "proj": torch.zeros(32, 32)}, "0 missing, 1 unexpected"),
    "shape": ({**bad, "positional_embedding": torch.zeros(257, 32)}, "0 unexpected, 1 wrong-shape"),
    "wrapped": ({"state_dict": bad}, f"{len(bad)} missing, 1 unexpected"),
    "visual": ({f"visual.{k}": v for k, v in bad.items()}, f"{len(bad)} missing, {len(bad)} unexpected"),
    "module": ({f"module.{k}": v for k, v in bad.items()}, f"{len(bad)} missing, {len(bad)} unexpected"),
}
for name, (state, text) in CASES.items():
    path = saved(f"{name}.pth", state)
    _, printed_text = printed(lambda: fails(lambda: P.load_weights(target, path), RuntimeError, text))
    assert "key check FAILED" in printed_text
    assert same(target, good), f"{name}: the model was changed by a file that failed the check"
fails(lambda: P.load_weights(target, saved("tensor.pth", torch.zeros(3))), RuntimeError,
      "expected a state dict")


class Thing:
    pass


# weights_only=True: a file that holds more than tensors is refused by torch itself
fails(lambda: P.load_weights(target, saved("object.pth", {"x": Thing()})), pickle.UnpicklingError)
assert same(target, good)
print("strict loading: missing, unexpected (ln_post, proj), wrong shape, wrapped and prefixed files "
      "are refused and leave the model as it was; weights_only refuses an object")

# ---- 7. the real weights, only when they are on this machine --------------------------------
places = list(_setup.WEIGHTS) + [_setup.REPO / "weights" / "pe"]
found = [p for d in places if d.is_dir() for p in d.rglob(P.WEIGHTS_FILE)]
if found:
    folder = found[0].parent
    t0 = time.perf_counter()
    t = P.build(device="cpu", dtype=torch.float32, weights_dir=folder)
    print(f"real weights: build (hash + architecture + strict load) {time.perf_counter() - t0:.1f} s "
          f"from {found[0]}")
    assert (t.dim, t.patch, t.input_size, t.grid, t.posemb_grid) == (1024, 14, 224, 16, 32)
    assert t.n_params() == P.N_PARAMS and len(t.model.state_dict()) == P.N_TENSORS
    assert t.weights == P.WEIGHTS_SHA256 and t.summary_from == "class token"
    # the teacher holds the values of the file, not only its shapes (a plain read, 1.2 GB; a
    # comparison through an mmap read took 96 s on this machine, see load_weights)
    file_state = torch.load(found[0], map_location="cpu", weights_only=True)
    assert len(file_state) == P.N_TENSORS and all(v.dtype == torch.float32 for v in file_state.values())
    assert sum(v.numel() for v in file_state.values()) == P.N_PARAMS
    assert same(t.model, file_state)
    del file_state
    # LayerNorm eps of the released code (norm_layer = partial(nn.LayerNorm, eps=1e-5))
    assert t.model.ln_pre.eps == 1e-5
    assert all(b.ln_1.eps == 1e-5 and b.ln_2.eps == 1e-5 for b in t.model.transformer.resblocks)
    assert isinstance(t.model.ln_post, torch.nn.Identity)
    print(f"real weights: {P.N_TENSORS} tensors, {t.n_params():,} parameters, values equal the file, "
          "LayerNorm eps 1e-5, ln_post Identity")
    # one CPU forward at batch 1, 224 px
    crop = scene(7)
    t0 = time.perf_counter()
    out = t(crop)
    dt = time.perf_counter() - t0
    assert out.summary.shape == (1, 1024) and out.spatial.shape == (1, 16, 16, 1024)
    assert bool(torch.isfinite(out.summary).all()) and bool(torch.isfinite(out.spatial).all())
    print(f"real weights: one CPU forward at batch 1, 224 px, {dt:.2f} s, summary (1, 1024), "
          f"spatial (1, 16, 16, 1024), finite, summary norm {float(out.summary.norm()):.1f}, "
          f"patch norm mean {float(out.spatial.norm(dim=-1).mean()):.1f}")
    outa = t.aerial(torch.randint(0, 256, (1, 4, 256, 256)).float())
    assert outa.spatial.shape == (1, 16, 16, 1024) and bool(torch.isfinite(outa.spatial).all())
    # the 224 px check: 224 grid against the 448 grid pooled 2 x 2
    for label, s in (("structured scene", scene(7)), ("structured scene + noise", scene(7, noise=300.0)),
                     ("white noise", torch.rand(1, 12, 256, 256) * 10000)):
        x448 = t.prepare(s, size=448)
        t0 = time.perf_counter()
        cos = P.grid_check(t, x448)
        dt = time.perf_counter() - t0
        cen = P.grid_check(t, x448, centred=True)
        print(f"grid_check (224 px grid vs 448 px grid pooled 2 x 2), {label}: mean cosine "
              f"{cos:.4f}, centred {cen:.4f}, {dt:.1f} s for both forwards")
        assert -1.0 <= cos <= 1.0 and -1.0 <= cen <= 1.0
    # the same crop by both roads - prepare at 224 and the 448 input downsampled - agree
    x224 = F.interpolate(t.prepare(scene(7), size=448), size=(224, 224), mode="bilinear",
                         antialias=True, align_corners=False)
    print(f"real weights: prepare at 224 vs the 448 input downsampled, max abs difference "
          f"{float((x224 - t.prepare(scene(7))).abs().max()):.4f} (two resamplings of one crop)")
else:
    print(f"SKIPPED (no {P.WEIGHTS_FILE} under SARTRANSFER_TEST_WEIGHTS or weights/pe): size, hash, "
          "keys, one forward pass and grid_check of the real weights")

print("test_teacher_pe: all checks passed")
