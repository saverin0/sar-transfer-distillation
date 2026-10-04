"""The cheap test of the 8 px grid and notebook 03b, on the CPU.

Small tests only - tiny data, a stand-in encoder, a randomly initialised AnyUp, no real weights.

1. GridDecoder - entry 16 the same module and output as a verbatim copy of the old class, entry 8 the same
   decoder without its first stage, any other entry refused
2. _patch_targets with 8 px patches, the default still 16 px
3. the upsamplers - bilinear as torch interpolates, AnyUp guided by the skip image at the chip's full size,
   the output at the 8 px grid, float16
4. load_anyup - a missing file and a file of another size or hash refused, a matching file loaded
5. run_domain with an upsampler - the run name, the 8 px probe and decoder, the info keys, the feature cache
   that extracts once for several settings
6. notebook 03b - the text, the rule, no other repository, the sync cell current, the shared flush
   cell, and a dry run - the Drive copy after a failing Hub fetch, the three settings per dataset with
   one extraction, the words for helps, no gain and not known, a second run that trains nothing
"""
import _setup  # noqa: F401

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from _setup import TMP_BASE  # noqa: E402

from sartransfer import env as E  # noqa: E402
from sartransfer import sync as SYNC  # noqa: E402
from sartransfer.distil import stage0 as Z  # noqa: E402
from sartransfer.domains import hub as H, prepared as P, run as R, upsample as U  # noqa: E402
from sartransfer.domains.s1input import channel_stats, skip_image  # noqa: E402
from sartransfer.models import anyup_loader as AL, encoders as encmod  # noqa: E402
from sartransfer.models.anyup import AnyUp as AnyUpModel  # noqa: E402
from sartransfer.models.decoder import GridDecoder, ImageStem, _conv  # noqa: E402

TMP = TMP_BASE / "grid_check"
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True)
FAKE_TOKEN = "hf_standin_grid_check_0123456789"
PATCHED = []


def patch(obj, name, value):
    PATCHED.append((obj, name, getattr(obj, name)))
    setattr(obj, name, value)


def unpatch_all():
    while PATCHED:
        obj, name, value = PATCHED.pop()
        setattr(obj, name, value)


def fails(fn, exc, text=""):
    try:
        fn()
    except exc as e:
        assert text in str(e), f"expected {text!r} in {e!r}"
        return
    raise SystemExit(f"no {exc.__name__} was raised")


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        out = fn(*args, **kwargs)
    return out, buf.getvalue()


# ---- 1. GridDecoder --------------------------------------------------------------------------------
class OldGridDecoder(nn.Module):
    """models/decoder.GridDecoder as it was before part four's entry argument (2026-10-04), copied verbatim."""

    def __init__(self, feat_dim: int, n_classes: int = 4, width: int = 96):
        super().__init__()
        self.stem = ImageStem()
        self.reduce = nn.Conv2d(feat_dim, width, 1)
        ch = ImageStem.CH
        self.up = nn.ModuleList([
            _conv(width + ch[4], width),          # 1/16: grid + s16
            _conv(width + ch[3], width),          # 1/8
            _conv(width + ch[2], 64),             # 1/4
            _conv(64 + ch[1], 48),                # 1/2
            _conv(48 + ch[0], 32),                # 1/1
        ])
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, feats, img):
        maps = self.stem(img)
        x = self.reduce(feats)
        x = self.up[0](torch.cat([x, maps[4]], 1))
        for i, m in zip(range(1, 5), (maps[3], maps[2], maps[1], maps[0])):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.up[i](torch.cat([x, m], 1))
        return self.head(x)


torch.manual_seed(0)
old = OldGridDecoder(12, n_classes=3)
torch.manual_seed(0)
new = GridDecoder(12, n_classes=3)
assert list(old.state_dict()) == list(new.state_dict()), "the 16 px decoder has other parameters"
assert all(torch.equal(a, b) for a, b in zip(old.state_dict().values(), new.state_dict().values())), "other weights"
g = torch.Generator().manual_seed(1)
f16, img = torch.randn(2, 12, 4, 4, generator=g), torch.rand(2, 1, 64, 64, generator=g)
assert torch.equal(old(f16, img), new(f16, img)), "the 16 px decoder computes something else"
torch.manual_seed(0)
eight = GridDecoder(12, n_classes=3, entry=8)
out = eight(torch.randn(2, 12, 8, 8, generator=g), img)
assert out.shape == (2, 3, 64, 64) and eight.entry == 8 and len(eight.up) == 4 and len(eight.stem.blocks) == 4
assert [tuple(m[0].weight.shape) for m in eight.up] == [tuple(m[0].weight.shape) for m in new.up[1:]], \
    "the 8 px decoder is not the 16 px one without its first stage"
n_old, n_eight = (sum(p.numel() for p in m.parameters()) for m in (new, eight))
assert n_eight < n_old, (n_old, n_eight)
fails(lambda: GridDecoder(12, entry=4), ValueError, "entry must be 16 or 8")
print(f"GridDecoder - entry 16 is the old head bit for bit, entry 8 drops the first stage ({n_old:,} -> {n_eight:,} "
      "parameters)")

# ---- 2. patch targets ------------------------------------------------------------------------------
from sartransfer.data.targets import patch_labels  # noqa: E402

labs = np.random.default_rng(0).integers(0, 2, (3, 32, 32)).astype(np.uint8)
labs[:, :2] = 255
pl8, keep8 = R._patch_targets(labs, 2, 8)
assert pl8.shape == (3, 4, 4) and keep8.shape == (3, 4, 4)
want = patch_labels(np.where(labs[1] == 255, 0, labs[1]), labs[1] != 255, patch=8, n_classes=2)
assert np.array_equal(pl8[1], want[0]) and np.array_equal(keep8[1], want[1])
pl16, _ = R._patch_targets(labs, 2)
assert pl16.shape == (3, 2, 2), "the default is not 16 px"
print("_patch_targets - 8 px patches as patch_labels gives them, the default 16 px")

# ---- 3. the upsamplers -----------------------------------------------------------------------------
rng = np.random.default_rng(0)
feats = rng.normal(0, 1, (5, 4, 4, 8)).astype(np.float16)
images = np.stack([rng.normal(-12, 3, (2, 64, 64)) for _ in range(5)]).astype(np.float32)
images[0, :, :4, :4] = np.nan
stats = channel_stats(images)
bil = U.Bilinear(device="cpu", batch=2)(feats, images, stats)
want = F.interpolate(torch.from_numpy(feats).float().permute(0, 3, 1, 2), scale_factor=2, mode="bilinear",
                     align_corners=False).permute(0, 2, 3, 1).half().numpy()
assert bil.dtype == np.float16 and bil.shape == (5, 8, 8, 8) and np.array_equal(bil, want)
assert U.Bilinear.name == "bilinear" and U.AnyUp.name == "anyup" and U.Bilinear.factor == U.AnyUp.factor == 2

torch.manual_seed(0)
random_anyup = AnyUpModel()
seen = []
real_forward = random_anyup.forward


def watched(image, features, output_size=None, **kw):
    seen.append((tuple(image.shape), tuple(features.shape), tuple(output_size), image.dtype))
    return real_forward(image, features, output_size=output_size, **kw)


random_anyup.forward = watched
up = U.AnyUp("unused", device="cpu", batch=2, model=random_anyup)
got = up(feats, images, stats)
assert got.dtype == np.float16 and got.shape == (5, 8, 8, 8) and np.isfinite(got.astype(np.float32)).all()
assert seen[0] == ((2, 3, 64, 64), (2, 8, 4, 4), (8, 8), torch.float32), seen[0]
assert len(seen) == 3, "batches of 2 over 5 chips"
# the guide is the decoder's skip image as grey 0-255, ImageNet-normalised RGB
grey = np.round(skip_image(images[:2], stats)[:, :1] * 255).astype(np.uint8)
guide = AL.guide_image(torch.from_numpy(grey))
captured = []
random_anyup.forward = lambda image, features, output_size=None, **kw: (captured.append(image), real_forward(
    image, features, output_size=output_size, **kw))[1]
U.AnyUp("unused", device="cpu", batch=2, model=random_anyup)(feats[:2], images[:2], stats)
assert torch.allclose(captured[0], guide), "the guide is not the skip image"
assert torch.allclose(guide[:, 0], guide[:, 0]) and not torch.allclose(guide[:, 0], guide[:, 1]), "RGB normalised"
random_anyup.forward = real_forward
print("upsamplers - bilinear as torch, AnyUp guided by the full-size skip image, 8 px grid, float16, batched")

# ---- 4. load_anyup ---------------------------------------------------------------------------------
fails(lambda: AL.load_anyup(TMP / "missing.pth", "cpu"), FileNotFoundError, "AnyUp weights")
bad = TMP / "bad.pth"
bad.write_bytes(b"x" * 100)
fails(lambda: AL.load_anyup(bad, "cpu"), RuntimeError, "are not those of the released weights")
good = TMP / AL.WEIGHTS_NAME
torch.save(random_anyup.state_dict(), good)
data = good.read_bytes()
patch(AL, "WEIGHTS_BYTES", len(data))
patch(AL, "WEIGHTS_SHA256", hashlib.sha256(data).hexdigest())
loaded = AL.load_anyup(good, "cpu")
assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), random_anyup.state_dict().values()))
unpatch_all()
print("load_anyup - missing and other files refused, a matching file loaded frozen")

# ---- 5. run_domain with an upsampler ---------------------------------------------------------------
CLASSES = {"glacial_lakes": {0: "background", 1: "lake"}, "forty_v1": {0: "natural_forest", 1: "water", 2: "built"}}


def toy(root: Path, name: str) -> Path:
    """A tiny prepared dataset, 64 px chips, train and val."""
    k = len(CLASSES[name])
    w = P.PreparedWriter(root, name, 64, CLASSES[name], source={"url": "synthetic"}, pixel_spacing_m=10.0)
    for split, n in (("train", 12), ("val", 4)):
        for i in range(n):
            yy, xx = np.mgrid[:64, :64]
            water = (yy - 32) ** 2 + (xx - int(rng.integers(16, 48))) ** 2 < 200
            built = (yy < 20) & (xx > 40)
            img = np.stack([np.where(water, -22.0, np.where(built, -4.0, -9.0)) + rng.normal(0, 1.5, (64, 64)),
                            np.where(water, -29.0, -16.0) + rng.normal(0, 1.0, (64, 64))]).astype(np.float32)
            lab = (np.where(water, 1, np.where(built, 2, 0)) if k == 3 else water).astype(np.uint8)
            lab[:3, :3] = 255
            w.add(split, img, lab, region=f"r{i % 2}", date="2020-08-01")
    w.close()
    return root / name


ENCODED = []


class FakeEncoder:
    patch, dim, _n_prefix, n_layers = 16, 8, 0, 4
    spec = encmod.EncoderSpec("fake/model", "fake", "fake", batch=4)

    def __init__(self, *a, **k):
        pass

    def encode(self, tiles, batch=None, layer=None):
        ENCODED.append(len(tiles))
        n, _, size, _ = tiles.shape
        gg = size // 16
        b = tiles[:, :2].reshape(n, 2, gg, 16, gg, 16)
        m, s = b.mean((3, 5)), b.std((3, 5))
        return np.concatenate([m, s, m * 0.5, -m], 1).transpose(0, 2, 3, 1).astype(np.float16)


HAD_ENV = {k: os.environ.get(k) for k in ("DATASET_HF_OWNER", "HF_WRITE_TOKEN")}
PREP = TMP / "prepared"
toy(PREP, "glacial_lakes")
patch(encmod, "FrozenEncoder", FakeEncoder)
encmod.ENCODERS["fake"] = FakeEncoder.spec
KW = dict(token=None, layer=1, seeds=(0, 1), steps=30, batch=3, device="cpu", splits=("val",))
cache = {}
_, said16 = quiet(R.run_domain, PREP, "glacial_lakes", "fake", TMP / "res", feature_cache=cache, **KW)
first = list(ENCODED)
_, said8 = quiet(R.run_domain, PREP, "glacial_lakes", "fake", TMP / "res", feature_cache=cache,
                 upsampler=U.Bilinear(device="cpu"), **KW)
assert ENCODED == first, "the features were extracted again"
assert "  features train: (12, 4, 4, 8) reused" in said8 and "  bilinear train: (12, 8, 8, 8) in" in said8, said8
RUN8 = "dom_glacial_lakes__fake__L1__vv_vh_diff__up8_bilinear"
info8 = json.loads((TMP / "res" / f"{RUN8}__info.json").read_text(encoding="utf-8"))
assert info8["upsampler"] == "bilinear" and info8["grid_px"] == 8 and "upsample_train_min" in info8["times"]
info16 = json.loads((TMP / "res" / "dom_glacial_lakes__fake__L1__vv_vh_diff__info.json").read_text(encoding="utf-8"))
assert "upsampler" not in info16 and "grid_px" not in info16 and "upsample_train_min" not in info16["times"]
head8 = torch.load(TMP / "res" / f"{RUN8}__decoder__s0.pt")
assert "up.3.0.weight" in head8 and "up.4.0.weight" not in head8 and "stem.blocks.4.0.weight" not in head8
rows8 = pd.read_csv(TMP / "res" / f"{RUN8}__summary.csv")
assert set(rows8["head"]) == {"majority", "probe", "decoder"} and set(rows8["split"]) == {"val"}
assert np.isfinite(rows8.loc[rows8["head"] == "probe", "patch_mIoU"]).all()
assert (rows8["run"] == RUN8).all()
_, said_again = quiet(R.run_domain, PREP, "glacial_lakes", "fake", TMP / "res", feature_cache=cache,
                      upsampler=U.Bilinear(device="cpu"), **KW)
assert said_again.startswith(f"skip {RUN8}"), said_again
print("run_domain upsampler - run name __up8_bilinear, 8 px probe and decoder, info keys, one extraction for both")

# ---- 6. notebook 03b -------------------------------------------------------------------------------
NOTEBOOK = _setup.REPO / "notebooks" / "03b_grid_check.ipynb"
nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
NAMES = [c.get("id") for c in nb["cells"]]
assert NAMES == ["intro", "mount", "sync", "setup", "runs", "verdict", "flush"], NAMES
SOURCE = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"), "the notebook holds outputs"
intro = SOURCE["intro"]
assert intro.startswith("# ") and ":" not in intro, "a colon in the markdown cell"
for words in ("lose to the frozen satellite DINOv3 by 0.03 to 0.06", "10 to 13 A100 hours", "about half an A100 hour",
              "written down before it runs (2026-10-04)", "layer 21", "glacial lakes and forest",
              "the seeds 0, 1 and 2", "16 px, the run step as it is", "AnyUp 8 px", "at the chip's full size",
              "the decoder entering at 8 px", "bilinear 8 px", "the control", "extracted once for the three settings",
              "The decoder IoU of the lake class on val", "larger than 0.01 and larger than twice the larger seed SD",
              "a cut-down 8 px run is proposed with its compute", "dropped and part four is written up",
              "do not decide", "In part one", "An A100, High-RAM is not needed", "83.5 GB of RAM",
              "The peak RAM is about 65 GB by arithmetic", "About 30 minutes, an estimate",
              "MyDrive/sar-transfer-distillation/weights", "checked by size and hash and never downloaded", "never printed",
              "MyDrive/sar-transfer-distillation/grid_check/", "grid_check.json", "is not run again", "END_RUNTIME = False"):
    assert words in intro, f"the first cell does not say {words!r}"
FORBIDDEN = _setup.private_names()            # empty in a public copy
for name, source in SOURCE.items():
    if name != "sync":
        assert not [w for w in FORBIDDEN if w in source], f"cell {name} names another repository or its folder"
for name in NAMES[1:]:
    first_line = SOURCE[name].splitlines()[0]
    assert first_line.startswith("# ") and ":" not in first_line, f"cell {name} starts with {first_line!r}"
    if name != "sync":
        assert not [ln for ln in SOURCE[name].splitlines() if ln.lstrip().startswith("#") and ":" in ln], name
nb03 = json.loads((_setup.REPO / "notebooks" / "03_score.ipynb").read_text(encoding="utf-8"))
assert SOURCE["flush"] == _setup.FLUSH_SOURCE, "the last cell differs from the shared flush cell"
assert SOURCE["mount"] == "".join(next(c for c in nb03["cells"] if c.get("id") == "mount")["source"])
assert SYNC.cell_index(nb) == NAMES.index("sync")
SYNC_STATE = _setup.check_sync_cell(nb, SYNC)            # packed and current, or cleared in a published copy
for cell, words in (("setup", 'DATASETS = ("glacial_lakes", "forty_v1")'), ("setup", 'DECIDES = "glacial_lakes"'),
                    ("setup", 'METRIC = {"glacial_lakes": "IoU_pos", "forty_v1": "mIoU"}'),
                    ("setup", 'ENCODER, MODE = "dinov3-l-sat", "vv_vh_diff"'), ("setup", "SEEDS = (0, 1, 2)"),
                    ("setup", "LAYER = run_step.PRE_REGISTERED_LAYERS[ENCODER]"), ("setup", 'SPLITS = ("val",)'),
                    ("setup", 'WEIGHTS = MYDRIVE / "sar-transfer-distillation" / "weights" / anyup_loader.WEIGHTS_NAME'),
                    ("runs", '"AnyUp 8 px": upsample.AnyUp(WEIGHTS, DEVICE)'),
                    ("runs", '"bilinear 8 px": upsample.Bilinear(DEVICE)'), ("runs", '{"16 px": None'),
                    ("runs", "enc_cache=enc_cache, feature_cache=feature_cache"),
                    ("verdict", 'score.judge(head, row["mean"], row["sd"], ref["mean"], ref["sd"])')):
    assert words in SOURCE[cell], f"cell {cell} has no {words}"
print("notebook 03b - the text, the rule, no other repository, the sync cell current, the shared flush cell")

# the dry run
CONTENT = TMP / "content"
MYDRIVE = CONTENT / "drive" / "MyDrive"
OUT = MYDRIVE / "sar-transfer-distillation" / "grid_check"
NB_PREP = CONTENT / "sar-transfer-distillation" / "prepared"
WEIGHTS = MYDRIVE / "sar-transfer-distillation" / "weights" / AL.WEIGHTS_NAME
WEIGHTS.parent.mkdir(parents=True)
WEIGHTS.write_bytes(b"stand-in")                       # the cell checks it is there, the stand-in AnyUp loads nothing
lakes_copy = toy(TMP / "drive_source", "glacial_lakes")
shutil.copytree(lakes_copy, MYDRIVE / "datasets" / H.repo_name("glacial_lakes"))
toy(NB_PREP, "forty_v1")                               # forest is on the runtime disk already
CALLS = {"fetch": [], "run": []}
REAL_RUN = R.run_domain
REAL_ANYUP, REAL_BILINEAR = U.AnyUp, U.Bilinear


def hub_down(dataset, owner, dest, token, splits=None, **kw):
    CALLS["fetch"].append((dataset, owner, splits))
    assert token == FAKE_TOKEN
    raise OSError(f"stand-in, the Hub refused the token {token}")


def run_on_cpu(prepared_root, dataset, encoder, res_dir, token, **kw):
    """The real run_domain on the CPU, 30 decoder steps, batch 3, the stand-in encoder. The notebook asks for cuda."""
    assert kw.pop("device") == "cuda" and token == FAKE_TOKEN and kw["seeds"] == (0, 1, 2) and kw["layer"] == 21, kw
    assert encoder == "dinov3-l-sat" and kw["mode"] == "vv_vh_diff" and kw["splits"] == ("val",)
    CALLS["run"].append((dataset, None if kw["upsampler"] is None else kw["upsampler"].name))
    return REAL_RUN(prepared_root, dataset, encoder, res_dir, token=None, device="cpu", steps=30, batch=3, **kw)


class AnyUpOnCPU(REAL_ANYUP):
    def __init__(self, weights_path, device="cuda", batch=32, model=None):
        assert Path(weights_path) == WEIGHTS and device == "cuda", (weights_path, device)
        torch.manual_seed(0)
        super().__init__(weights_path, device="cpu", batch=batch, model=AnyUpModel())


class BilinearOnCPU(REAL_BILINEAR):
    def __init__(self, device="cuda", batch=64):
        assert device == "cuda"
        super().__init__(device="cpu", batch=batch)


patch(Z, "WORK_FOLDERS", (str(CONTENT), str(MYDRIVE)))
patch(E, "model_cache_on_drive", lambda *a, **k: MYDRIVE / "hf_cache")
patch(E, "load_env", lambda *a, **k: {})
patch(E, "require", lambda name, why="": FAKE_TOKEN)
patch(H, "fetch", hub_down)
patch(R, "run_domain", run_on_cpu)
patch(U, "AnyUp", AnyUpOnCPU)
patch(U, "Bilinear", BilinearOnCPU)
patch(torch.cuda, "is_available", lambda: True)
patch(torch.cuda, "get_device_name", lambda device=None: "stand-in A100")
os.environ["DATASET_HF_OWNER"] = "owner-stand-in"
os.environ.pop("HF_WRITE_TOKEN", None)
CODE = {name: SOURCE[name] for name in ("setup", "runs", "verdict")}


def run_cells(which, names: dict, title: str) -> str:
    printed = []
    for name in which:
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                exec(CODE[name], names)
        except BaseException:
            print(out.getvalue())
            traceback.print_exc()
            raise SystemExit(f"{title}, cell {name} raised") from None
        said = out.getvalue()
        print(f"\n==== {title}, cell {name}")
        print("\n".join("    | " + line for line in said.splitlines()[-10:]))
        assert FAKE_TOKEN not in said, f"cell {name} printed the token"
        assert "cell time" in said.splitlines()[-1], f"cell {name} did not print its time"
        printed.append(said)
    return "\n".join(printed)


def set_lake(run: str, values):
    """The lake decoder IoU of the three seeds on val in one setting's summary set to `values`."""
    rows = pd.read_csv(OUT / f"{run}__summary.csv")
    pick = (rows["head"] == "decoder") & (rows["split"] == "val")
    rows.loc[pick, "IoU_pos"] = values
    rows.to_csv(OUT / f"{run}__summary.csv", index=False)


try:
    one = {}
    ENCODED.clear()
    said = run_cells(["setup", "runs", "verdict"], one, "first run")
    assert CALLS["fetch"] == [("glacial_lakes", "owner-stand-in", ("train", "val"))], CALLS["fetch"]
    assert "glacial_lakes  not on Hugging Face (OSError, stand-in, the Hub refused the token <token>)" in said
    assert re.search(r"^glacial_lakes  from Drive in [\d.]+ min$", said, re.M)
    assert "forty_v1  on the runtime disk already" in said and f"AnyUp weights  {WEIGHTS}" in said
    assert CALLS["run"] == [(d, s) for d in ("glacial_lakes", "forty_v1") for s in (None, "anyup", "bilinear")]
    assert said.count("  encoder loaded in") == 1, "the encoder was loaded twice"
    assert len(ENCODED) == 4, f"each split of each dataset is extracted once, {ENCODED}"
    assert len(re.findall(r"features \w+: \(.*\) reused", said)) == 8, "the 8 px settings did not reuse the features"
    report = json.loads((OUT / "grid_check.json").read_text(encoding="utf-8"))
    assert report["verdict"] in ("helps", "no gain") and len(report["settings"]) == 12 and len(report["compared"]) == 8
    assert [c["decides"] for c in report["compared"]].count(True) == 1
    assert report["seeds"] == [0, 1, 2] and report["layer"] == 21 and report["gpu"] == "stand-in A100"
    assert set(report["times"]) == {"setup", "runs"}, report["times"]
    print("first run - Drive copy after the Hub failed, three settings per dataset, one extraction, the json")

    anyup_run = "dom_glacial_lakes__dinov3-l-sat__L21__vv_vh_diff__up8_anyup"
    run16 = "dom_glacial_lakes__dinov3-l-sat__L21__vv_vh_diff"
    kept = {r: (OUT / f"{r}__summary.csv").read_bytes() for r in (anyup_run, run16)}
    set_lake(run16, [0.30, 0.31, 0.32])                # mean 0.31, SD 0.01, so a gap must pass 0.02
    set_lake(anyup_run, [0.39, 0.39, 0.39])
    said = run_cells(["verdict"], one, "AnyUp higher")
    assert "THE FINER GRID HELPS - AnyUp 8 px beats 16 px on the lake IoU by 0.0800" in said, said
    assert json.loads((OUT / "grid_check.json").read_text(encoding="utf-8"))["verdict"] == "helps"
    set_lake(anyup_run, [0.30, 0.31, 0.32])
    said = run_cells(["verdict"], one, "AnyUp the same")
    assert "NO GAIN - AnyUp 8 px against 16 px on the lake IoU is a tie (gap 0.0000" in said, said
    assert "The 8 px run and our own upsampler are dropped, part four is written up" in said
    set_lake(anyup_run, [0.23, 0.23, 0.23])
    said = run_cells(["verdict"], one, "AnyUp lower")
    assert "NO GAIN - AnyUp 8 px against 16 px on the lake IoU is a loss (gap -0.0800" in said, said
    rows = pd.read_csv(OUT / f"{anyup_run}__summary.csv")
    rows[rows["head"] != "decoder"].to_csv(OUT / f"{anyup_run}__summary.csv", index=False)
    said = run_cells(["verdict"], one, "AnyUp decoder rows missing")
    assert "NOT KNOWN - a lake decoder row is missing" in said
    assert json.loads((OUT / "grid_check.json").read_text(encoding="utf-8"))["verdict"] == "not known"
    for r, b in kept.items():
        (OUT / f"{r}__summary.csv").write_bytes(b)
    print("verdict words - helps, no gain for a tie and a loss, not known when the deciding row is missing")

    two = {}
    ENCODED.clear()
    CALLS["fetch"].clear()
    said = run_cells(["setup", "runs", "verdict"], two, "second run")
    assert CALLS["fetch"] == [] and said.count("on the runtime disk already") == 2
    assert said.count("skip dom_") == 6 and ENCODED == [] and not re.search(r"decoder seed \d+: ", said)
    print("second run - both datasets on disk, all six settings on Drive, nothing extracted or trained")
finally:
    unpatch_all()
    encmod.ENCODERS.pop("fake", None)
    for k, v in HAD_ENV.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
print("test_grid_check: all checks passed")
