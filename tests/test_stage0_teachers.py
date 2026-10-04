"""stage0 helpers of part four - timing and pictures for teachers that read radar or aerial,
and the scene check (one PCA per image, the input descriptors per patch, the scene number).

Tiny teachers on the CPU. The radar teacher is TerraFM-B through its Sentinel-1 input layer,
the aerial reader is the one every RGB teacher gets from teachers/base.py.
"""
import _setup  # noqa: F401

import numpy as np
import torch

from sartransfer.distil import stage0
from sartransfer.distil.teachers import build_teacher

TMP = _setup.TMP_BASE / "stage0_teachers"
TMP.mkdir(parents=True, exist_ok=True)

# ---- random inputs per modality --------------------------------------------------------------
for modality, channels in (("s2", 12), ("radar", 2), ("aerial", 4)):
    x = stage0.random_input(modality, 3, 32, "cpu")
    assert x.shape == (3, channels, 32, 32) and x.dtype == torch.float32, (modality, x.shape)
s2 = stage0.random_input("s2", 2, 32, "cpu")
assert 0 <= s2.min() and s2.max() <= stage0.DN_MAX
radar = stage0.random_input("radar", 2, 32, "cpu")
assert -60 < radar.mean() < 10, "dB around -15"
aerial = stage0.random_input("aerial", 2, 32, "cpu")
assert aerial.min() >= 0 and aerial.max() <= 255 and torch.equal(aerial, aerial.round())
_setup.raises(ValueError, stage0.random_input, "sonar", 1, 32, "cpu") if hasattr(_setup, "raises") else None
try:
    stage0.random_input("sonar", 1, 32, "cpu")
except ValueError as err:
    assert "sonar" in str(err)
else:
    raise SystemExit("an unknown modality must raise")
print("random_input ok for s2, radar, aerial")

# ---- which reader a teacher offers ------------------------------------------------------------
dino = build_teacher("dinov3-l-sat", tiny=True, device="cpu", dtype=torch.float32)
terra = build_teacher("terrafm-b-radar", tiny=True, device="cpu", dtype=torch.float32)
assert stage0.teacher_call(dino, "s2") is dino and stage0.teacher_call(dino, "aerial") == dino.aerial
assert stage0.teacher_call(terra, "radar") == terra.radar
for teacher, modality in ((dino, "radar"), (terra, "s2"), (terra, "aerial")):
    try:
        stage0.teacher_call(teacher, modality)
    except TypeError as err:
        assert teacher.name in str(err)
    else:
        raise SystemExit(f"{teacher.name} must refuse {modality}")
print("teacher_call picks the reader and refuses the rest")

# ---- timing rows carry the modality ----------------------------------------------------------
size = 64
rows = stage0.time_teacher(dino, batch_sizes=(2, 3), n_batches=1, size=size, modality="aerial")
assert list(rows["modality"]) == ["aerial", "aerial"] and list(rows["batch_size"]) == [2, 3]
assert (rows["images_per_s"] > 0).all() and rows["teacher"].iloc[0] == dino.name
rows = stage0.time_teacher(terra, batch_sizes=(2,), n_batches=1, size=256, modality="radar")
assert rows["modality"].iloc[0] == "radar" and rows["images_per_s"].iloc[0] > 0
assert rows["grid"].iloc[0] == terra.grid and rows["dim"].iloc[0] == terra.dim
default = stage0.time_teacher(dino, batch_sizes=(2,), n_batches=1, size=size)
assert default["modality"].iloc[0] == "s2", "the default modality is Sentinel-2, as in part three"
print("time_teacher rows per modality ok")

# ---- previews ---------------------------------------------------------------------------------
pre = stage0.preview(radar, "radar")
assert pre.shape == (2, 32, 32, 3) and pre.dtype == np.uint8
flat = torch.full((1, 2, 8, 8), -15.0)
assert stage0.preview(flat, "radar").max() == 0, "a flat radar image stretches to 0"
pre = stage0.preview(aerial, "aerial")
assert pre.shape == (2, 32, 32, 3) and np.array_equal(pre, aerial[:, :3].permute(0, 2, 3, 1).numpy().astype(np.uint8))
assert stage0.preview(s2, "s2").shape == (2, 32, 32, 3)
print("previews ok")

# ---- pictures per modality --------------------------------------------------------------------
png = stage0.feature_pictures("tiny dino", lambda rule: dino, stage0.random_input("aerial", 2, 64, "cpu"),
                              [dino.rule], TMP / "aerial.png", modality="aerial")
assert png.is_file() and png.stat().st_size > 1000
png = stage0.feature_pictures("tiny terrafm", lambda rule: terra, stage0.random_input("radar", 2, 256, "cpu"),
                              [terra.rule], TMP / "radar.png", modality="radar")
assert png.is_file() and png.stat().st_size > 1000
png = stage0.feature_pictures("tiny dino", lambda rule: dino, stage0.random_input("s2", 2, 64, "cpu"),
                              [dino.rule], TMP / "s2.png")
assert png.is_file()
try:
    stage0.feature_pictures("tiny dino", lambda rule: dino, radar, [dino.rule], TMP / "bad.png", modality="s2")
except ValueError as err:
    assert "expected" in str(err)
else:
    raise SystemExit("a radar batch under the s2 modality must raise")
print("feature_pictures draws s2, radar and aerial inputs")

# ---- one PCA per image ------------------------------------------------------------------------
g = torch.Generator().manual_seed(0)
features = torch.randn(3, 6, 6, 8, generator=g)
rgb = stage0.pca_rgb_per_image(features)
assert rgb.shape == (3, 6, 6, 3) and rgb.dtype == np.uint8
assert np.array_equal(stage0.pca_rgb_per_image(features.numpy()), rgb), "an array gives the same colours"
assert all(len(np.unique(image.reshape(-1, 3), axis=0)) > 1 for image in rgb), "an image of one colour"
other = features.clone()
other[1] = torch.randn(6, 6, 8, generator=g) * 10 + 3              # image 1 changes, 0 and 2 do not
again = stage0.pca_rgb_per_image(other)
assert np.array_equal(again[0], rgb[0]) and np.array_equal(again[2], rgb[2]), "the colours of one image depend on another"
assert not np.array_equal(again[1], rgb[1])
broken = features.clone()
broken[2, 3, 3, 0] = float("nan")
try:
    stage0.pca_rgb_per_image(broken)
except ValueError as err:
    assert "not finite" in str(err)
else:
    raise SystemExit("features that are not finite must raise")
print("pca_rgb_per_image - one PCA per image, an image does not change the colours of another")

# ---- what a patch of the input looks like ---------------------------------------------------
for modality, channels, n in (("s2", 12, 3), ("radar", 2, 2), ("aerial", 4, 3)):
    x = stage0.random_input(modality, 2, 32, "cpu")
    assert stage0.patch_descriptors(x, modality, 4).shape == (2, 4, 4, 2 * n), modality
small = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4) ** 1.5 - 40.0        # radar, 4 x 4
got = stage0.patch_descriptors(small, "radar", 2).numpy()
blocks = small.numpy().reshape(1, 2, 2, 2, 2, 2).transpose(0, 2, 4, 1, 3, 5).reshape(1, 2, 2, 2, 4)
want = np.concatenate([blocks.mean(-1), blocks.std(-1)], -1)                           # by hand
assert np.allclose(got, want, atol=1e-3), (got, want)
bands = torch.arange(12, dtype=torch.float32).view(1, 12, 1, 1).expand(1, 12, 4, 4) * 100.0
got = stage0.patch_descriptors(bands, "s2", 2)
assert torch.equal(got[0, 0, 0], torch.tensor([*(100.0 * i for i in stage0.band_index(stage0.PREVIEW_BANDS)), 0, 0, 0]))
try:
    stage0.patch_descriptors(small, "sonar", 2)
except ValueError as err:
    assert "sonar" in str(err)
else:
    raise SystemExit("an unknown modality must raise")
print("patch_descriptors - mean and SD per patch and channel, as by hand")

# ---- the scene number -----------------------------------------------------------------------
g = torch.Generator().manual_seed(0)
fields = torch.rand(4, 2, 8, 8, generator=g) * 10 - 20                   # a scene of 8 x 8 fields
scene = torch.nn.functional.interpolate(fields, size=(128, 128), mode="nearest") + torch.randn(4, 2, 128, 128, generator=g)
descriptors = stage0.patch_descriptors(scene, "radar", 16)
z = (descriptors - descriptors.mean((1, 2), keepdim=True)) / descriptors.std((1, 2), keepdim=True)
made = stage0.scene_correlation(z @ torch.randn(4, 64, generator=g), descriptors)    # features made from the input
noise = stage0.scene_correlation(torch.randn(4, 16, 16, 64, generator=g), descriptors)
pattern = stage0.scene_correlation(torch.randn(1, 16, 16, 64, generator=g).expand(4, -1, -1, -1), descriptors)
assert len(made) == len(noise) == len(pattern) == 4
assert min(made) > 0.65 and np.mean(made) > 0.7, made
assert max(abs(v) for v in noise) < 0.1, noise
assert max(abs(v) for v in pattern) < 0.1, f"a pattern of its own, the same in every image, {pattern}"
try:
    stage0.scene_correlation(torch.randn(4, 14, 14, 64), descriptors)
except ValueError as err:
    assert "grid" in str(err)
else:
    raise SystemExit("features and descriptors on two grids must raise")
print(f"scene_correlation - made from the input {min(made):.2f} to {max(made):.2f}, noise "
      f"{min(noise):+.2f} to {max(noise):+.2f}, a fixed pattern {min(pattern):+.2f} to {max(pattern):+.2f}")

# ---- one picture and one number per image ---------------------------------------------------
for teacher, modality, batch in ((dino, "s2", stage0.random_input("s2", 3, 64, "cpu")),
                                 (dino, "aerial", stage0.random_input("aerial", 2, 64, "cpu")),
                                 (terra, "radar", stage0.random_input("radar", 2, 64, "cpu"))):
    out_png = TMP / f"scene_{teacher.name}_{modality}.png"
    out_png.unlink(missing_ok=True)
    png, numbers = stage0.scene_pictures(teacher, batch, modality, out_png)
    assert png == out_png and png.is_file() and png.stat().st_size > 1000, (teacher.name, modality)
    assert len(numbers) == len(batch) and all(-1.0 <= v <= 1.0 for v in numbers), (teacher.name, modality, numbers)
print("scene_pictures - a file and one number per image for the tiny DINOv3 (s2, aerial) and the tiny radar teacher")
print("test_stage0_teachers: all checks passed")
