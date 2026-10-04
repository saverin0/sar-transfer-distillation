"""The figures of the README (figures.py) and notebook 05, on the CPU.

Small tests only - tiny chips, a stand-in encoder, randomly initialised decoder heads, no real weights.

1. pick_chips - the labelled share, the most even mix, another region for the second chip, ties to the lower index
2. radar_rgb, label_rgb, chip_score - the colours and the score of one chip, by hand
3. predict - the class map of a tiny decoder, the same as its argmax by hand
4. prediction_frames, feature_frames, save_gif - frames of one size, the GIF read back with its durations
5. results_chart, architecture_picture - the files
6. notebook 05 - the text, no colon, the flush cell, the sync cell, and a dry run - the Drive copies of the
   validation splits, the picked chips in chips.json, the three models, the GIFs and pictures on Drive
"""
import _setup  # noqa: F401

import contextlib
import io
import json
import os
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from _setup import TMP_BASE  # noqa: E402

from sartransfer import env as E  # noqa: E402
from sartransfer import figures as FG  # noqa: E402
from sartransfer import sync as SYNC  # noqa: E402
from sartransfer.distil import stage0 as Z  # noqa: E402
from sartransfer.domains import hub as H, prepared as P  # noqa: E402
from sartransfer.domains.s1input import channel_stats  # noqa: E402
from sartransfer.models import encoders as encmod  # noqa: E402
from sartransfer.models.decoder import GridDecoder  # noqa: E402

TMP = TMP_BASE / "figures"
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True)
FAKE_TOKEN = "hf_standin_figures_0123456789"
PATCHED = []
rng = np.random.default_rng(0)


def patch(obj, name, value):
    PATCHED.append((obj, name, getattr(obj, name)))
    setattr(obj, name, value)


def unpatch_all():
    while PATCHED:
        obj, name, value = PATCHED.pop()
        setattr(obj, name, value)


# ---- 1. pick_chips -------------------------------------------------------------------------------
labels = np.zeros((5, 8, 8), np.uint8)
labels[0, :4] = 1                        # half and half, region a
labels[1, :, :] = 0                      # one class only
labels[2, :4] = 1
labels[2, :2] = 255                      # half and half but 25 percent unlabelled
labels[3, :4] = 1                        # half and half, region a again
labels[4, :3] = 1                        # 3 of 8 rows, region b
regions = np.array(["a", "a", "a", "a", "b"])
assert FG.pick_chips(labels, regions, 2, n=2) == [0, 4], FG.pick_chips(labels, regions, 2, n=2)
assert FG.pick_chips(labels, np.array(["a"] * 5), 2, n=2) == [0, 3], "one region only, the next best chip"
assert FG.pick_chips(labels, regions, 2, n=1, min_labelled=0.7) == [0]
shares, labelled = FG.class_shares(labels[2], 2)
assert labelled == 0.75 and np.allclose(shares, [2 / 3, 1 / 3])
padded = labels.copy()                    # every chip with a padded border, as the alpine tiles - at most 0.875
padded[:, :, 0] = 255
assert FG.pick_chips(padded, regions, 2, n=2) == [0, 4], "padded tiles, 90 percent of the best chip's share"
assert FG.pick_chips(padded, regions, 2, n=2, min_labelled=0.95) == [0, 4]
print("pick_chips - labelled share, the most even mix, another region next, ties to the lower index, padded tiles")

# ---- 2. colours and scores -----------------------------------------------------------------------
radar = np.stack([np.linspace(-25, -5, 64).reshape(8, 8), np.full((8, 8), -18.0)]).astype(np.float32)
radar[0, 0, 0] = np.nan
rgb = FG.radar_rgb(radar)
assert rgb.shape == (8, 8, 3) and rgb.dtype == np.uint8 and tuple(rgb[0, 0]) == (0, rgb[0, 0, 1], 0)
assert rgb[..., 0].max() == 255 and rgb[..., 1].max() == 0, "a flat VH channel stays dark"
lab = np.array([[0, 1], [255, 1]], np.uint8)
assert FG.label_rgb(lab, FG.PALETTES["glacial_lakes"]).tolist() == [[[58, 58, 58], [47, 127, 216]], [[0, 0, 0], [47, 127, 216]]]
pred = np.array([[1, 1], [0, 0]], np.uint8)
assert abs(FG.chip_score(pred, lab, 2) - 1 / 3) < 1e-9, "the lake IoU on the labelled pixels, 1 of 3"
assert abs(FG.chip_score(np.array([[0, 2], [1, 1]]), np.array([[0, 1], [1, 1]]), 3) - (1 + 2 / 3 + 0) / 3) < 1e-9
assert np.isnan(FG.chip_score(np.zeros((2, 2)), np.full((2, 2), 255), 2))
print("radar_rgb, label_rgb, chip_score - by hand")

# ---- 3. predict ------------------------------------------------------------------------------------
torch.manual_seed(0)
head = GridDecoder(8, n_classes=3)
feats = rng.normal(0, 1, (3, 4, 4, 8)).astype(np.float16)
skip = rng.integers(0, 256, (3, 64, 64), dtype=np.uint8)
got = FG.predict(head, feats, skip, device="cpu", batch=2)
with torch.inference_mode():
    want = head(torch.from_numpy(feats).float().permute(0, 3, 1, 2), torch.from_numpy(skip)[:, None].float() / 255).argmax(1)
assert got.dtype == np.uint8 and got.shape == (3, 64, 64) and np.array_equal(got, want.numpy())
print("predict - the argmax of the decoder, batched")

# ---- 4. frames and GIFs ----------------------------------------------------------------------------
palette = FG.PALETTES["forty_v1"]
names = ["natural_forest", "planted_forest", "tree_crops", "other_vegetation", "built", "water", "ice", "bare_ground"]
models = ["Frozen DINOv3 SAT", "Student F", "Student C (no radar teacher)"]
chips = [{"radar": FG.radar_rgb(rng.normal(-12, 3, (2, 32, 32))), "truth": FG.label_rgb(rng.integers(0, 8, (32, 32)), palette),
          "pred": {m: FG.label_rgb(rng.integers(0, 8, (32, 32)), palette) for m in models},
          "score": {m: 0.3 + 0.1 * k for k, m in enumerate(models)}, "label": f"region {r}"} for r in ("x", "y")]
frames, seconds = FG.prediction_frames("Forest types", chips, models, palette, names, "mIoU")
assert len(frames) == 2 + len(models) + 1 and len(seconds) == len(frames) and seconds[-1] == FG.LAST_SECONDS
assert len({f.shape for f in frames}) == 1 and frames[0].shape == (675, 1200, 3)
gif = FG.save_gif(frames, TMP / "pred.gif", seconds)
with Image.open(gif) as im:
    durations = []
    for k in range(im.n_frames):
        im.seek(k)
        durations.append(im.info["duration"])
assert im.n_frames == len(frames) and durations == [int(s * 1000) for s in seconds], durations
rows = [{"title": t, "radar": chips[0]["radar"], "truth": chips[0]["truth"], "features": chips[0]["pred"]}
        for t in ("Glacial lakes", "Snow", "Forest types", "Alpine glaciers")]
frames, seconds = FG.feature_frames(rows, models)
assert len(frames) == 2 + len(models) + 1 and len({f.shape for f in frames}) == 1
try:
    FG.save_gif([frames[0], frames[0][:10]], TMP / "bad.gif", [1, 1])
except ValueError as err:
    assert "differ in size" in str(err)
else:
    raise SystemExit("frames of two sizes must be refused")
print(f"frames and GIFs - {im.n_frames} frames of one size, the durations read back")

# ---- 5. pictures -----------------------------------------------------------------------------------
table = pd.DataFrame([{"dataset": d, "model": m, "mean": 0.3 + 0.05 * k, "sd": 0.01}
                      for d in ("glacial_lakes", "snow") for k, m in enumerate(["A", "B"])])
png = FG.results_chart(table, TMP / "results.png", ["A", "B"], ["glacial_lakes", "snow"], {"A": "#e8a33d", "B": "#3b6fb6"})
arch = FG.architecture_picture(TMP / "architecture.png")
assert png.stat().st_size > 10_000 and arch.stat().st_size > 10_000
print("results_chart and architecture_picture - the files")

# ---- 6. notebook 05 ----------------------------------------------------------------------------------
NOTEBOOK = _setup.REPO / "notebooks" / "05_figures.ipynb"
nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
NAMES = [c.get("id") for c in nb["cells"]]
assert NAMES == ["intro", "mount", "sync", "setup", "chips", "predictions", "gifs", "pictures", "flush"], NAMES
SOURCE = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"), "the notebook holds outputs"
intro = SOURCE["intro"]
assert intro.startswith("# ") and ":" not in intro.replace("MyDrive/", "")
for words in ("Nothing is trained or tuned here", "a rule on the labels alone", "at least 90 percent",
              "the most even class mix", "from another region", "The L4 GPU", "About 10 minutes, an estimate",
              "MyDrive/sar-transfer-distillation/figures/", "chips.json", "END_RUNTIME = False"):
    assert words in intro, f"the first cell does not say {words!r}"
for name in NAMES[1:]:
    first = SOURCE[name].splitlines()[0]
    assert first.startswith("# ") and ":" not in first, (name, first)
    if name != "sync":
        assert not [ln for ln in SOURCE[name].splitlines() if ln.lstrip().startswith("#") and ":" in ln], name
assert SOURCE["flush"] == _setup.FLUSH_SOURCE, "the last cell differs from the shared flush cell"
_setup.check_sync_cell(nb, SYNC)
print("notebook 05 - the text, no colon, the shared flush cell, the sync cell")

# the dry run
CONTENT = TMP / "content"
MYDRIVE = CONTENT / "drive" / "MyDrive"
OUT = MYDRIVE / "sar-transfer-distillation" / "figures"
RES = MYDRIVE / "sar-transfer-distillation" / "results"
REF = MYDRIVE / "part-two-results" / "results"
CLASSES = {"glacial_lakes": {0: "background", 1: "lake"}, "snow": {0: "no_snow", 1: "snow"},
           "forty_v1": {0: "natural_forest", 1: "water", 2: "built"}, "glavitu_alp": {0: "other", 1: "glacier"}}


def toy(root: Path, name: str) -> Path:
    k = len(CLASSES[name])
    w = P.PreparedWriter(root, name, 64, CLASSES[name], source={"url": "synthetic"}, pixel_spacing_m=10.0)
    for split, n in (("train", 4), ("val", 6)):
        for i in range(n):
            yy, xx = np.mgrid[:64, :64]
            blob = (yy - 32) ** 2 + (xx - int(rng.integers(16, 48))) ** 2 < int(rng.integers(150, 600))
            img = np.stack([np.where(blob, -22.0, -9.0) + rng.normal(0, 1.5, (64, 64)),
                            np.where(blob, -29.0, -16.0) + rng.normal(0, 1.0, (64, 64))]).astype(np.float32)
            lab = (blob.astype(np.uint8) if k == 2 else np.where(blob, 1, np.where(yy < 20, 2, 0))).astype(np.uint8)
            w.add(split, img, lab, region=f"r{i % 3}", date="")
    w.close()
    return root / name


class FakeEncoder:
    patch, dim, _n_prefix, n_layers = 16, 8, 0, 4
    spec = encmod.EncoderSpec("fake/model", "fake", "fake", batch=4)

    def __init__(self, *a, **k):
        pass

    def encode(self, tiles, batch=None, layer=None):
        n, _, size, _ = tiles.shape
        g = size // 16
        b = tiles[:, :2].reshape(n, 2, g, 16, g, 16)
        m, s = b.mean((3, 5)), b.std((3, 5))
        return np.concatenate([m, s, m * 0.5, -m], 1).transpose(0, 2, 3, 1).astype(np.float16)


DATASETS = ("glacial_lakes", "snow", "forty_v1", "glavitu_alp")
for name in DATASETS:
    src = toy(TMP / "drive_source", name)
    shutil.copytree(src, MYDRIVE / "datasets" / H.repo_name(name))
    stats = channel_stats(np.load(src / "train_images.npy"))
    for folder, encoder, layer, mode in ((REF, "dinov3-l-sat", 21, "vv_vh_diff"), (RES, "student-f", "last", "co_cross"),
                                         (RES, "student-c", "last", "co_cross")):
        run = f"dom_{name}__{encoder}__L{layer}__{mode}"
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(GridDecoder(8, n_classes=len(CLASSES[name])).state_dict(), folder / f"{run}__decoder__s0.pt")
        (folder / f"{run}__info.json").write_text(json.dumps({"stats": stats}), encoding="utf-8")
rows_csv = []
for name in DATASETS:
    for model, ref in (("student-f", "dinov3-l-sat"), ("student-c", "terrafm-b-s1"), ("student-n", "student-m")):
        rows_csv.append({"question": 1, "model": model, "reference": ref, "dataset": name, "head": "decoder",
                         "role": "word", "metric": "x", "model_score": 0.3, "model_sd": 0.01,
                         "reference_score": 0.35, "reference_sd": 0.02, "gap": -0.05, "needed": 0.01, "outcome": "loss"})
pd.DataFrame(rows_csv).to_csv(RES / "stage_a_rows.csv", index=False)
RUNS = MYDRIVE / "sar-transfer-distillation" / "runs_6250"
for folder in ("student_F", "student_C"):
    (RUNS / folder).mkdir(parents=True)
    (RUNS / folder / "final.pt").write_bytes(b"stand-in")
CALLS = {"fetch": []}
HAD_ENV = {k: os.environ.get(k) for k in ("DATASET_HF_OWNER", "HF_WRITE_TOKEN")}


def hub_down(dataset, owner, dest, token, splits=None, **kw):
    CALLS["fetch"].append((dataset, splits))
    raise OSError(f"stand-in, the Hub refused the token {token}")


def predict_on_cpu(head, feats, skip, device="cuda", batch=4):
    assert device == "cuda"
    return REAL_PREDICT(head, feats, skip, device="cpu", batch=batch)


REAL_PREDICT = FG.predict
patch(Z, "WORK_FOLDERS", (str(CONTENT), str(MYDRIVE)))
patch(E, "model_cache_on_drive", lambda *a, **k: MYDRIVE / "model_cache")
patch(E, "load_env", lambda *a, **k: {})
patch(E, "require", lambda name, why="": FAKE_TOKEN if name == "HF_TOKEN" else "part-two-results")
patch(H, "fetch", hub_down)
patch(FG, "predict", predict_on_cpu)
patch(encmod, "FrozenEncoder", FakeEncoder)
patch(torch.cuda, "is_available", lambda: True)
patch(torch.cuda, "get_device_name", lambda device=None: "stand-in L4")
os.environ["DATASET_HF_OWNER"] = "owner-stand-in"
os.environ.pop("HF_WRITE_TOKEN", None)
try:
    names_ns, printed = {}, []
    for cell in ("setup", "chips", "predictions", "gifs", "pictures"):
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                exec(SOURCE[cell], names_ns)
        except BaseException:
            print(out.getvalue())
            traceback.print_exc()
            raise SystemExit(f"notebook 05, cell {cell} raised") from None
        said = out.getvalue()
        print(f"\n==== notebook 05, cell {cell}")
        print("\n".join("    | " + line for line in said.splitlines()[-8:]))
        assert FAKE_TOKEN not in said and "cell time" in said.splitlines()[-1], cell
        printed.append(said)
    said = "\n".join(printed)
    assert [d for d, _ in CALLS["fetch"]] == list(DATASETS) and all(s == ("val",) for _, s in CALLS["fetch"])
    assert said.count("from Drive") == 4 and "not on Hugging Face (OSError, stand-in, the Hub refused the token <token>)" in said
    picked = json.loads((OUT / "chips.json").read_text(encoding="utf-8"))
    assert set(picked) == set(DATASETS) and all(len(v) == 2 for v in picked.values())
    for name in DATASETS:
        regions = [names_ns["DATA"][name]["regions"][i] for i in range(2)]
        assert regions[0] != regions[1], f"{name} took two chips of one region"
    for name in DATASETS:
        with Image.open(OUT / f"predictions_{name}.gif") as im:
            assert im.n_frames == 2 + 3 + 1, (name, im.n_frames)
    with Image.open(OUT / "features.gif") as im:
        assert im.n_frames == 2 + 3 + 1
    assert (OUT / "results.png").stat().st_size > 10_000 and (OUT / "architecture.png").stat().st_size > 10_000
    assert said.count("per chip") == 12, "three models on four datasets"
    print("notebook 05 dry run - Drive copies of the validation splits, two chips of two regions per dataset, "
          "the three models, four prediction GIFs, the feature GIF, the chart and the picture")
finally:
    unpatch_all()
    for k, v in HAD_ENV.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
print("test_figures: all checks passed")
