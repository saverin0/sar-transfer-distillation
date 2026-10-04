"""Notebook 03a and what it uses, on the CPU (the check of the compiled decoder step).

Small tests only - tiny data, a stand-in encoder, no real weights and no Inductor (torch.compile replaced by a stand-in
that calls the same function, as test_fast_decoder does).

1. score.seed_mean_check on made-up summaries - the control's rule, nan, counts left out, the paired
   differences; on the numbers of the failed control it finds the one mean that was off
2. run_domain with decoder_path - "compiled" and "eager" forced over the measured choice, recorded in the info,
   the numbers of part two's step; a wrong value refused before anything is written
3. notebook 03a - the text (no colon, the rule, no other repository), the sync cell current, the flush cell
   the shared one, then a dry run of its cells - the Drive copy after a failing Hub fetch, both sides,
   the verdict words for agrees, differs and not tested, the speeds with one size failing, a second run that
   fetches and trains nothing, the token never printed
"""
import _setup  # noqa: F401

import contextlib
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

from _setup import TMP_BASE  # noqa: E402

from sartransfer import env as E  # noqa: E402
from sartransfer import sync as SYNC  # noqa: E402
from sartransfer.distil import score as SC  # noqa: E402
from sartransfer.distil import stage0 as Z  # noqa: E402
from sartransfer.domains import hub as H, prepared as P, run as R  # noqa: E402
from sartransfer.models import encoders as encmod  # noqa: E402

TMP = TMP_BASE / "decoder_check"
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True)
FAKE_TOKEN = "hf_standin_decoder_check_0123456789"
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


# ---- 1. seed_mean_check ---------------------------------------------------------------------------
def summary(values: dict, seeds=(0, 1, 2), splits=("val",)) -> pd.DataFrame:
    """A summary of the run step - decoder rows with these per seed values, a probe row far off, counts."""
    rows = []
    for split in splits:
        for i, sd in enumerate(seeds):
            rows.append({"run": "r", "head": "decoder", "seed": sd, "split": split, "layer": 21, "n_pixels": 1000 + i,
                         **{c: float(v[i]) for c, v in values.items()}})
        rows.append({"run": "r", "head": "probe", "seed": -1, "split": split, "layer": 21, "n_pixels": 5,
                     **{c: 9.0 for c in values}})
    return pd.DataFrame(rows)


# the failed control of 2026-10-04, water IoU on val, part two against the compiled step (rounded to 3 digits)
ref = summary({"IoU_water": [0.695, 0.706, 0.700], "mIoU": [0.372, 0.393, 0.367]})
got = summary({"IoU_water": [0.676, 0.678, 0.687], "mIoU": [0.366, 0.385, 0.385]})
t = SC.seed_mean_check(ref, got)
assert list(t.columns) == ["column", "split", "reference_mean", "reference_sd", "other_mean", "other_sd", "gap",
                           "allowed", "off", "seeds", "paired_diff", "seeds_lower"], list(t.columns)
assert set(t["column"]) == {"IoU_water", "mIoU"}, "a count or a key was compared"
water = t[t["column"] == "IoU_water"].iloc[0]
assert water["off"] and abs(water["gap"] - 0.02) < 1e-3 and abs(water["allowed"] - 0.0117) < 2e-4, water.to_dict()
assert water["seeds"] == 3 and water["seeds_lower"] == 3 and abs(water["paired_diff"] + 0.02) < 1e-3
assert not t[t["column"] == "mIoU"].iloc[0]["off"]
# the rule, gap against the larger of 0.01 and twice the larger SD
same = SC.seed_mean_check(ref, ref)
assert not same["off"].any() and (same["gap"] == 0).all() and (same["seeds_lower"] == 0).all()
small = SC.seed_mean_check(summary({"x": [0.5, 0.501, 0.5]}), summary({"x": [0.508, 0.509, 0.508]}))
assert not small.iloc[0]["off"] and small.iloc[0]["allowed"] == 0.01, "a gap under 0.01 is a tie"
wide = SC.seed_mean_check(summary({"x": [0.48, 0.50, 0.52]}), summary({"x": [0.51, 0.53, 0.55]}))
assert not wide.iloc[0]["off"] and abs(wide.iloc[0]["allowed"] - 0.04) < 1e-9, "twice the seed SD"
tight = SC.seed_mean_check(summary({"x": [0.498, 0.500, 0.502]}), summary({"x": [0.528, 0.530, 0.532]}))
assert tight.iloc[0]["off"], "a gap of 0.03 with SD 0.002 is off"
one = SC.seed_mean_check(summary({"x": [0.5]}, seeds=(0,)), summary({"x": [0.509]}, seeds=(0,)))
assert one.iloc[0]["reference_sd"] == 0 and not one.iloc[0]["off"], "one seed has no spread, 0.01 decides"
# nan - not known on both sides is no gap, on one side is off
nan3 = [float("nan")] * 3
both = SC.seed_mean_check(summary({"x": [0.5] * 3, "patch": nan3}), summary({"x": [0.5] * 3, "patch": nan3}))
assert not both.set_index("column").loc["patch", "off"] and both.set_index("column").loc["patch", "gap"] == 0
half = SC.seed_mean_check(summary({"x": [0.5] * 3, "patch": nan3}), summary({"x": [0.5] * 3, "patch": [0.4] * 3}))
assert half.set_index("column").loc["patch", "off"], "a column known on one side only is off"
# paired differences over the shared seeds only, rows per split
more = SC.seed_mean_check(summary({"x": [0.5, 0.6, 0.7]}, splits=("val", "test")),
                          summary({"x": [0.51, 0.58, 0.7, 0.1]}, seeds=(0, 1, 2, 3), splits=("val", "test")))
assert list(more["split"]) == ["test", "val"] and (more["seeds"] == 3).all() and (more["seeds_lower"] == 1).all()
assert abs(more.iloc[0]["paired_diff"] - (0.01 - 0.02 + 0.0) / 3) < 1e-9
assert abs(more.iloc[0]["other_mean"] - (0.51 + 0.58 + 0.7 + 0.1) / 4) < 1e-9, "the mean takes every seed of a side"
fails(lambda: SC.seed_mean_check(ref[ref["head"] == "probe"], got), ValueError, "decoder rows")
print("seed_mean_check - the control's rule, nan, counts left out, paired differences; the failed control's "
      f"water mean off (gap {water['gap']:.4f}, allowed {water['allowed']:.4f})")

# ---- 2. run_domain with decoder_path ---------------------------------------------------------------
rng = np.random.default_rng(0)
CLASSES = {0: "natural_forest", 1: "water", 2: "built"}


def toy(root: Path, name: str) -> Path:
    """A tiny prepared dataset of three classes, 64 px chips, train val and test."""
    w = P.PreparedWriter(root, name, 64, CLASSES, source={"url": "synthetic"}, pixel_spacing_m=10.0)
    for split, n in (("train", 12), ("val", 4), ("test", 4)):
        for i in range(n):
            yy, xx = np.mgrid[:64, :64]
            water_px = (yy - 32) ** 2 + (xx - int(rng.integers(16, 48))) ** 2 < 200
            built = (yy < 20) & (xx > 40)
            img = np.stack([np.where(water_px, -22.0, np.where(built, -4.0, -9.0)) + rng.normal(0, 1.5, (64, 64)),
                            np.where(water_px, -29.0, -16.0) + rng.normal(0, 1.0, (64, 64))]).astype(np.float32)
            lab = np.where(water_px, 1, np.where(built, 2, 0)).astype(np.uint8)
            lab[:3, :3] = 255
            w.add(split, img, lab, region=f"r{i % 2}", date="2020-08-01")
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


def wrapping(fn):
    """A compile stand-in - another function object that computes exactly what fn computes."""
    return lambda *a, **k: fn(*a, **k)


def measured(tile, n_feat, device, n_classes=2, batch=8, **kw):
    """decoder_speed_check's stand-in - the compiled step agrees but is not 5 percent faster, so eager is chosen."""
    return {"tile": tile, "n_feat": n_feat, "n_classes": n_classes, "batch": batch, "device": device,
            "eager_ms": 10.0, "compiled_ms": 9.8, "compile_s": 3.0, "loss_gap": 1e-6, "loss_tolerance": 1e-4,
            "grad_gap": 1e-3, "grad_tolerance": 1e-2, "error": None, "path": "eager",
            "why": "compiled is not 5 percent faster"}


ENV_KEYS = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_FX_GRAPH_CACHE", "TORCHINDUCTOR_AUTOGRAD_CACHE",
            "DATASET_HF_OWNER", "HF_WRITE_TOKEN")
HAD_ENV = {k: os.environ.get(k) for k in ENV_KEYS}
PREP = TMP / "prepared"
toy(PREP, "toy")
patch(encmod, "FrozenEncoder", FakeEncoder)
encmod.ENCODERS["fake"] = FakeEncoder.spec
patch(R, "_compile", wrapping)
patch(R, "decoder_speed_check", measured)
R._CHOICES.clear()
KW = dict(token=None, layer=1, seeds=(0, 1), steps=30, batch=3, device="cpu")
RUN = "dom_toy__fake__L1__vv_vh_diff"


def info(res: Path) -> dict:
    return json.loads((res / f"{RUN}__info.json").read_text(encoding="utf-8"))["times"]


quiet(R.run_domain, PREP, "toy", "fake", TMP / "res_default", **KW)
_, said = quiet(R.run_domain, PREP, "toy", "fake", TMP / "res_compiled", decoder_path="compiled",
                compile_cache=TMP / "cache", **KW)
times = info(TMP / "res_compiled")
assert times["decoder_path"] == "compiled", times
assert times["decoder_check_why"] == "set to compiled by the caller, the measured choice was eager", times
assert "  decoder path set to compiled by the caller, the measured choice was eager\n" in said, said
assert "decoder seed 0 - compiled path" in said and "decoder seed 1 - compiled path" in said
assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str((TMP / "cache").resolve()), "the cache folder was not used"
assert R._CHOICES[(64, 8, 3, 3, "cpu")]["path"] == "eager", "the measured choice kept for other runs was changed"
default_rows = pd.read_csv(TMP / "res_default" / f"{RUN}__summary.csv")
compiled_rows = pd.read_csv(TMP / "res_compiled" / f"{RUN}__summary.csv")
pd.testing.assert_frame_equal(default_rows, compiled_rows)       # the stand-in computes part two's step
assert "decoder_path" not in info(TMP / "res_default"), "the default run recorded a path"
quiet(R.run_domain, PREP, "toy", "fake", TMP / "res_eager", decoder_path="eager", **KW)
assert info(TMP / "res_eager")["decoder_path"] == "eager"
assert info(TMP / "res_eager")["decoder_check_why"] == "set to eager by the caller, the measured choice was eager"
fails(lambda: R.run_domain(PREP, "toy", "fake", TMP / "res_wrong", decoder_path="fast", **KW), ValueError,
      "decoder_path must be")
assert not (TMP / "res_wrong").exists(), "a refused run wrote a folder"
print("run_domain decoder_path - compiled and eager forced over the measured choice, recorded, part two's numbers")

# ---- 3. notebook 03a -------------------------------------------------------------------------------
NOTEBOOK = _setup.REPO / "notebooks" / "03a_decoder_check.ipynb"
nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
NAMES = [c.get("id") for c in nb["cells"]]
assert NAMES == ["intro", "mount", "sync", "setup", "runs", "verdict", "speed", "flush"], NAMES
SOURCE = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"), "the notebook holds outputs"
intro = SOURCE["intro"]
assert intro.startswith("# ") and ":" not in intro, "a colon in the markdown cell"
for words in ("0.680 against 0.700, allowed 0.012", "Part three's control with part two's step had 0.704",
              "written down before it runs", "the new seeds 3 to 8",
              "once with part two's decoder step and once with the compiled step for every seed",
              "128, 256, 512 and 992 px", "larger than 0.01 and larger than twice the larger seed SD",
              "notebook 03 runs its control with part two's step and the students with the measured choice",
              "notebook 03 runs part two's step everywhere", "Not tested, nothing is decided", "do not decide",
              "An A100", "Standard RAM is enough", "About 15 minutes, an estimate", "DATASET_HF_OWNER",
              "MyDrive/datasets", "never printed", "MyDrive/sar-transfer-distillation/decoder_check/", "decoder_check.json",
              "is not run again", "END_RUNTIME = False"):
    assert words in intro, f"the first cell does not say {words!r}"
FORBIDDEN = _setup.private_names()            # empty in a public copy
for name, source in SOURCE.items():
    if name != "sync":
        assert not [w for w in FORBIDDEN if w in source], f"cell {name} names another repository or its folder"
for name in NAMES[1:]:
    first = SOURCE[name].splitlines()[0]
    assert first.startswith("# ") and ":" not in first, f"cell {name} starts with {first!r}"
    if name != "sync":
        assert not [ln for ln in SOURCE[name].splitlines() if ln.lstrip().startswith("#") and ":" in ln], name
nb03 = json.loads((_setup.REPO / "notebooks" / "03_score.ipynb").read_text(encoding="utf-8"))
assert SOURCE["flush"] == _setup.FLUSH_SOURCE, "the last cell differs from the shared flush cell"
assert SOURCE["mount"] == "".join(next(c for c in nb03["cells"] if c.get("id") == "mount")["source"])
assert SYNC.cell_index(nb) == NAMES.index("sync")
SYNC_STATE = _setup.check_sync_cell(nb, SYNC)            # packed and current, or cleared in a published copy
for cell, words in (("setup", "SEEDS = (3, 4, 5, 6, 7, 8)"), ("setup", "TOLERANCE = 0.01"),
                    ("setup", 'DATASET, ENCODER, MODE = "forty_v1", "dinov3-l-sat", "vv_vh_diff"'),
                    ("setup", "LAYER = run_step.PRE_REGISTERED_LAYERS[ENCODER]"),
                    ("setup", '"compiled": {"decoder_path": "compiled", "compile_cache": CACHE}'),
                    ("setup", 'ARM_SETTINGS = {"eager": {}'),
                    ("runs", "enc_cache=enc_cache, **ARM_SETTINGS[arm]"),
                    ("verdict", 'score.seed_mean_check(summaries["eager"], summaries["compiled"], TOLERANCE)'),
                    ("speed", "run_step.decoder_speed_check(tile, STUDENT_WIDTH, DEVICE)")):
    assert words in SOURCE[cell], f"cell {cell} has no {words}"
assert "21" not in SOURCE["setup"].split("LAYER =")[1].split("#")[0], "the layer is retyped"
print("notebook 03a - the text, the rule, no other repository, the sync cell current, the shared flush cell")

# the dry run
CONTENT = TMP / "content"
MYDRIVE = CONTENT / "drive" / "MyDrive"
OUT = MYDRIVE / "sar-transfer-distillation" / "decoder_check"
NB_PREP = CONTENT / "sar-transfer-distillation" / "prepared"
drive_copy = toy(TMP / "drive_source", "forty_v1")
shutil.copytree(drive_copy, MYDRIVE / "datasets" / H.repo_name("forty_v1"))
CALLS = {"fetch": [], "run": [], "speed": []}


def hub_down(dataset, owner, dest, token, splits=None, **kw):
    CALLS["fetch"].append((dataset, owner, str(dest), splits))
    assert token == FAKE_TOKEN
    raise OSError(f"stand-in, the Hub refused the token {token}")


def run_on_cpu(prepared_root, dataset, encoder, res_dir, token, **kw):
    """The real run_domain on the CPU, 30 decoder steps, batch 3. The notebook asks for cuda."""
    assert kw.pop("device") == "cuda" and token == FAKE_TOKEN and kw["seeds"] == (3, 4, 5, 6, 7, 8), kw
    CALLS["run"].append((dataset, encoder, Path(res_dir).name, kw.get("decoder_path")))
    return REAL_RUN(prepared_root, dataset, encoder, res_dir, token=None, device="cpu", steps=30, batch=3, **kw)


def speed_stand_in(tile, n_feat, device, n_classes=2, batch=8, **kw):
    CALLS["speed"].append((tile, n_feat, device))
    if tile == 992:
        raise RuntimeError(f"stand-in CUDA out of memory, token {FAKE_TOKEN}")
    return {**measured(tile, n_feat, device), "eager_ms": tile / 10, "compiled_ms": tile / 12, "compile_s": 30.0}


def gpu_name(device=None):
    return "stand-in A100"


REAL_RUN = R.run_domain
patch(Z, "WORK_FOLDERS", (str(CONTENT), str(MYDRIVE)))
patch(E, "model_cache_on_drive", lambda *a, **k: MYDRIVE / "hf_cache")
patch(E, "load_env", lambda *a, **k: {})
patch(E, "require", lambda name, why="": FAKE_TOKEN if name == "HF_TOKEN" else SystemExit(f"asked for {name}"))
patch(H, "fetch", hub_down)
patch(R, "run_domain", run_on_cpu)
patch(torch.cuda, "is_available", lambda: True)
patch(torch.cuda, "get_device_name", gpu_name)
os.environ["DATASET_HF_OWNER"] = "owner-stand-in"
os.environ.pop("HF_WRITE_TOKEN", None)
R._CHOICES.clear()
CODE = {name: SOURCE[name] for name in ("setup", "runs", "verdict", "speed")}


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
        print("\n".join("    | " + line for line in said.splitlines()[-12:]))
        assert FAKE_TOKEN not in said, f"cell {name} printed the token"
        assert "cell time" in said.splitlines()[-1], f"cell {name} did not print its time"
        printed.append(said)
    return "\n".join(printed)


try:
    one = {}
    said = run_cells(["setup", "runs", "verdict"], one, "first run")
    patch(R, "decoder_speed_check", speed_stand_in)
    said += run_cells(["speed"], one, "first run")
    patch(R, "decoder_speed_check", measured)
    # the data - the Hub fetch failed (token hidden), the copy on Drive came instead
    assert CALLS["fetch"] == [("forty_v1", "owner-stand-in", str(NB_PREP), None)], CALLS["fetch"]
    assert "forty_v1  not on Hugging Face (OSError, stand-in, the Hub refused the token <token>)" in said
    assert re.search(r"^forty_v1  from Drive in [\d.]+ min$", said, re.M)
    assert all((NB_PREP / "forty_v1" / f).is_file() for f in P.split_files(("train", "val", "test")))
    assert "GPU     stand-in A100" in said and f"results go to   {OUT}" in said
    # both sides, the seeds 3 to 8, part two's step first, the compiled one forced
    run = "dom_forty_v1__dinov3-l-sat__L21__vv_vh_diff"
    assert CALLS["run"] == [("forty_v1", "dinov3-l-sat", "eager", None),
                            ("forty_v1", "dinov3-l-sat", "compiled", "compiled")], CALLS["run"]
    assert said.count("  encoder loaded in") == 1 and "encoder reused (already loaded)" in said, "loaded twice"
    for arm in ("eager", "compiled"):
        rows = pd.read_csv(OUT / arm / f"{run}__summary.csv")
        assert sorted(set(rows.loc[rows["head"] == "decoder", "seed"])) == [3, 4, 5, 6, 7, 8], arm
        assert set(rows["split"]) == {"val", "test"}, "the control scores val and test"
    assert json.loads((OUT / "compiled" / f"{run}__info.json").read_text(encoding="utf-8"))["times"][
        "decoder_path"] == "compiled"
    assert "decoder_path" not in json.loads((OUT / "eager" / f"{run}__info.json").read_text(encoding="utf-8"))["times"]
    # the verdict, the same numbers (the compile stand-in computes the same step) - agrees
    report = json.loads((OUT / "decoder_check.json").read_text(encoding="utf-8"))
    n_means = len(one["table"])
    assert report["verdict"] == "agrees" and report["off"] == 0 and report["means"] == n_means, report["verdict"]
    assert n_means == 2 * len([c for c in pd.read_csv(OUT / "eager" / f"{run}__summary.csv").columns
                               if c in one["table"]["column"].tolist()]), "a column or split was not compared"
    assert {"IoU_water", "mIoU", "accuracy"} <= set(one["table"]["column"]) and "n_pixels" not in set(one["table"]["column"])
    assert f"AGREES - none of {n_means} means is off. Notebook 03 runs its control with part two's step" in said
    assert report["seeds"] == [3, 4, 5, 6, 7, 8] and report["tolerance"] == 0.01 and report["gpu"] == "stand-in A100"
    assert report["compiled_side"]["decoder_path"] == "compiled"
    # the speeds, 992 px failing on its own, the token hidden, the rest measured
    assert CALLS["speed"] == [(t, 768, "cuda") for t in (128, 256, 512, 992)], CALLS["speed"]
    assert re.search(r"^ 128 px  eager 12\.8 ms, compiled 10\.7 ms, 17 percent less, first compile 30 s, ", said, re.M)
    assert " 992 px  eager not measured, compiled not measured (RuntimeError, stand-in CUDA out of memory, token <token>)" in said
    report = json.loads((OUT / "decoder_check.json").read_text(encoding="utf-8"))
    assert [r["tile"] for r in report["speed"]] == [128, 256, 512, 992] and report["verdict"] == "agrees"
    assert set(report["times"]) == {"setup", "runs", "verdict"}, report["times"]
    print("first run - Drive copy after the Hub failed, both sides, agrees, the speeds with 992 px failing alone")

    # differs - the compiled side's water IoU lower on val by 0.05 in every seed
    summary_file = OUT / "compiled" / f"{run}__summary.csv"
    kept = summary_file.read_bytes()
    rows = pd.read_csv(summary_file)
    lower = (rows["head"] == "decoder") & (rows["split"] == "val")
    rows.loc[lower, "IoU_water"] = rows.loc[lower, "IoU_water"] - 0.05
    rows.to_csv(summary_file, index=False)
    said = run_cells(["verdict"], one, "water moved")
    assert f"DIFFERS - 1 of {n_means} means are off. The compiled step moves the numbers, notebook 03 runs part two's " \
           "step everywhere" in said, said
    assert re.search(r"^off  IoU_water val, gap 0\.0500, allowed 0\.\d{4}$", said, re.M)
    assert json.loads((OUT / "decoder_check.json").read_text(encoding="utf-8"))["verdict"] == "differs"
    summary_file.write_bytes(kept)
    # not tested - the compiled side ran eager (a compile that failed)
    info_file = OUT / "compiled" / f"{run}__info.json"
    kept_info = info_file.read_bytes()
    data = json.loads(kept_info)
    data["times"]["decoder_path"] = "eager"
    info_file.write_text(json.dumps(data), encoding="utf-8")
    said = run_cells(["verdict"], one, "compile failed")
    assert "NOT TESTED - the compiled side ran eager, see its lines above. Nothing is decided" in said
    assert json.loads((OUT / "decoder_check.json").read_text(encoding="utf-8"))["verdict"] == "not tested"
    info_file.write_bytes(kept_info)
    print("verdict words - differs with the off mean named, not tested when the compiled side ran eager")

    # a second run on the same runtime - nothing fetched, nothing trained
    CALLS["fetch"].clear()
    two = {}
    said = run_cells(["setup", "runs", "verdict"], two, "second run")
    assert CALLS["fetch"] == [] and "forty_v1  on the runtime disk already" in said
    assert said.count(f"skip {run}: summary already has probe + decoder seeds [3, 4, 5, 6, 7, 8]") == 2, said
    assert not re.search(r"decoder seed \d+: ", said) and "AGREES" in said, "a decoder was trained again"
    print("second run - the data on disk and both sides on Drive, nothing fetched or trained, the same verdict")
finally:
    unpatch_all()
    encmod.ENCODERS.pop("fake", None)
    for k, v in HAD_ENV.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
print("test_decoder_check: all checks passed")
