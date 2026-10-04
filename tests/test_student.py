"""The student, its heads and the loss - shapes, guards, gradients, one training step."""
import _setup  # noqa: F401

import torch
import torch.nn.functional as F

from sartransfer.distil import student as S
from sartransfer.distil.student import (MAJORTOM_MODES, MODE_KEYS, MODES, Heads, Student,
                                        distil_loss, mode_inputs, mode_rows, tiny_student)
from sartransfer.distil.teachers import TeacherOut

try:                                  # timm is only the reference of the last check
    import timm
    from timm.models.vision_transformer import VisionTransformer
except Exception as err:
    timm = VisionTransformer = None
    print(f"timm not importable ({type(err).__name__}), the reference check will be skipped")


def short(v) -> str:
    if isinstance(v, torch.Tensor):
        return f"tensor{tuple(v.shape)}"
    return f"dict{sorted(v)}" if isinstance(v, dict) else repr(v)[:60]


def raises(exc, fn, *args, says: str = "", **kwargs) -> None:
    """`fn` must stop with `exc`, and with `says` in the message when that is given."""
    name = getattr(fn, "__name__", type(fn).__name__)
    try:
        fn(*args, **kwargs)
    except exc as err:
        if says not in str(err):
            raise SystemExit(f"wrong message from {name}, wanted {says!r}, got {str(err)!r}")
        return
    given = [short(a) for a in args] + [f"{k}={short(v)}" for k, v in kwargs.items()]
    raise SystemExit(f"bad input accepted by {name}, given {', '.join(given)}")


def bchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2)


def images(px: int, b: int = 2) -> dict:
    return {"s1": torch.randn(b, 2, px, px), "s2": torch.randn(b, 12, px, px),
            "aerial": torch.randn(b, 4, px, px)}


# ---- modes ------------------------------------------------------------------------------
assert MODES == ("s1", "s2", "both", "aerial") and MAJORTOM_MODES == ("s1", "s2", "both")
assert MODE_KEYS == {"s1": ("s1",), "s2": ("s2",), "both": ("s1", "s2"), "aerial": ("aerial",)}
batch = images(64)
assert list(mode_inputs(batch, "s1")) == ["s1"] and list(mode_inputs(batch, "s2")) == ["s2"]
assert list(mode_inputs(batch, "both")) == ["s1", "s2"]
assert list(mode_inputs(batch, "aerial")) == ["aerial"]
assert all(mode_inputs(batch, "both")[k] is batch[k] for k in ("s1", "s2"))
assert mode_inputs(batch, "aerial")["aerial"] is batch["aerial"]
raises(ValueError, mode_inputs, batch, "s3")
raises(KeyError, mode_inputs, {"s1": batch["s1"]}, "both")
assert mode_rows(["s2", "aerial", "s1", "both", "aerial", "s1"]) == \
    {"s1": [2, 5], "s2": [0], "both": [3], "aerial": [1, 4]}
assert mode_rows([]) == {m: [] for m in MODES}
raises(ValueError, mode_rows, ["s1", "s3"], says="position 1")

# ---- shapes, four modes, three sizes ------------------------------------------------------
torch.manual_seed(0)
st = tiny_student().eval()
assert isinstance(st, Student)
assert (st.img, st.patch, st.grid, st.dim) == (256, 16, 16, 32)
for px in (128, 256, 384):
    b = images(px)
    outs = {}
    for mode in MODES:
        with torch.no_grad():
            summary, spatial = st(**mode_inputs(b, mode))
        assert summary.shape == (2, 32), (mode, px, summary.shape)
        assert spatial.shape == (2, px // 16, px // 16, 32), (mode, px, spatial.shape)
        assert summary.dtype == spatial.dtype == torch.float32
        assert torch.isfinite(summary).all() and torch.isfinite(spatial).all()
        outs[mode] = (summary, spatial)
    for i in (0, 1):      # the four modes are four different results
        for m1 in MODES:
            for m2 in MODES:
                if m1 < m2:
                    assert not torch.equal(outs[m1][i], outs[m2][i]), (m1, m2)
with torch.no_grad():     # positional call = (s1, s2, aerial)
    assert torch.equal(st(b["s1"], b["s2"])[1], outs["both"][1])
    assert torch.equal(st(None, None, b["aerial"])[1], outs["aerial"][1])

# ---- guards of the student ------------------------------------------------------------------
x1, x2 = batch["s1"], batch["s2"]
raises(ValueError, st)                                                  # no input
raises(ValueError, st, s1=None, s2=None)
raises(ValueError, st, s1=torch.randn(2, 2, 64, 80))                    # not square
raises(ValueError, st, s2=torch.randn(2, 12, 80, 64))
raises(ValueError, st, s1=torch.randn(2, 2, 72, 72))                    # not a multiple of 16
raises(ValueError, st, s1=torch.randn(2, 3, 64, 64))                    # wrong channel count
raises(ValueError, st, s2=x1)                                           # radar given as optical
raises(ValueError, st, s1=x2)                                           # optical given as radar
raises(ValueError, st, s1=torch.randn(2, 64, 64))                       # no channel axis
raises(ValueError, st, s1=torch.randn(2, 2, 64), says="expected (B, 2, H, W)")
raises(ValueError, st, s2=torch.randn(1, 12, 64, 64, 1), says="expected (B, 12, H, W)")
raises(ValueError, st, s2=torch.zeros(2, 12, 64, 64, dtype=torch.int16))  # raw numbers, not float
raises(ValueError, st, s1=x1, s2=torch.randn(2, 12, 128, 128))          # image sizes differ
raises(ValueError, st, s1=x1, s2=torch.randn(3, 12, 64, 64))            # batch sizes differ
raises(ValueError, st, s1=x1, s2=torch.randn(2, 12, 64, 80))            # second input checked too
raises(ValueError, st, aerial=torch.randn(2, 3, 64, 64))                # aerial is RGB + NIR
raises(ValueError, st, aerial=torch.zeros(2, 4, 64, 64, dtype=torch.uint8))
raises(ValueError, st, s1=x1, aerial=batch["aerial"], says="aerial goes alone")
raises(ValueError, st, s2=x2, aerial=batch["aerial"], says="aerial goes alone")
raises(ValueError, st, s1=x1, s2=x2, aerial=batch["aerial"], says="aerial goes alone")
raises(ValueError, Student, img=250)
raises(ValueError, Student, dim=100, heads=12)

# ---- with both inputs the two embeddings are added ---------------------------------------------
with torch.no_grad():
    e1, e2, e12 = st.embed(s1=x1), st.embed(s2=x2), st.embed(s1=x1, s2=x2)
    assert e12.shape == (2, 32, 4, 4)
    assert torch.equal(e12, e1 + e2)
    assert not torch.equal(e12, e1) and not torch.equal(e12, e2)
    for silent, other in (("s2", "s1"), ("s1", "s2")):    # a silent input layer changes nothing
        m = tiny_student().eval()
        layer = getattr(m, f"embed_{silent}")
        layer.weight.zero_()
        layer.bias.zero_()
        both, single = m(s1=x1, s2=x2), m(**{other: batch[other]})
        assert torch.allclose(both[0], single[0], atol=1e-6)
        assert torch.allclose(both[1], single[1], atol=1e-6)

# ---- mixed batch, every sample in its own mode, results in batch order ---------------------------
torch.manual_seed(9)
modes = ["s2", "aerial", "s1", "both", "aerial", "s1"]           # Major-TOM rows: s2, s1, both, s1
mixed = {"modes": modes, "s1": torch.randn(4, 2, 64, 64), "s2": torch.randn(4, 12, 64, 64),
         "aerial": torch.randn(2, 4, 64, 64)}
picked = mode_inputs(mixed, "s1")
assert list(picked) == ["s1"] and torch.equal(picked["s1"], mixed["s1"][[1, 3]])
picked = mode_inputs(mixed, "both")
assert list(picked) == ["s1", "s2"]
assert torch.equal(picked["s1"], mixed["s1"][[2]]) and torch.equal(picked["s2"], mixed["s2"][[2]])
assert torch.equal(mode_inputs(mixed, "s2")["s2"], mixed["s2"][[0]])
assert mode_inputs(mixed, "aerial")["aerial"] is mixed["aerial"]
assert mode_inputs(mixed, "s1")["s1"].shape[0] == 2 and mode_inputs(dict(mixed, modes=["aerial"] * 2),
                                                                     "s1")["s1"].shape[0] == 0
with torch.no_grad():
    summary, spatial = st.forward_mixed(mixed)
    assert summary.shape == (6, 32) and spatial.shape == (6, 4, 4, 32)
    want = {0: st(s2=mixed["s2"][[0]]), 1: st(aerial=mixed["aerial"][[0]]),
            2: st(s1=mixed["s1"][[1]]), 3: st(s1=mixed["s1"][[2]], s2=mixed["s2"][[2]]),
            4: st(aerial=mixed["aerial"][[1]]), 5: st(s1=mixed["s1"][[3]])}
    for i, (w_sum, w_spa) in want.items():
        assert torch.allclose(summary[i], w_sum[0], atol=1e-5), i
        assert torch.allclose(spatial[i], w_spa[0], atol=1e-5), i
    # one mode only = plain forward
    one = {"modes": ["aerial", "aerial"], "aerial": mixed["aerial"]}
    assert torch.allclose(st.forward_mixed(one)[1], st(aerial=mixed["aerial"])[1], atol=1e-5)
    one = {"modes": ["both", "both", "both", "both"], "s1": mixed["s1"], "s2": mixed["s2"]}
    assert torch.allclose(st.forward_mixed(one)[1], st(s1=mixed["s1"], s2=mixed["s2"])[1], atol=1e-5)
    # keys that no mode uses may be missing
    no_s1 = {"modes": ["s2", "aerial"], "s2": mixed["s2"][[0]], "aerial": mixed["aerial"][[0]]}
    assert torch.allclose(st.forward_mixed(no_s1)[0][0], want[0][0][0], atol=1e-5)
raises(ValueError, st.forward_mixed, {"modes": []}, says="empty")
raises(ValueError, st.forward_mixed, dict(mixed, modes=modes[:5]), says="rows")
raises(ValueError, st.forward_mixed, dict(mixed, s1=mixed["s1"][:3]), says="rows")
raises(ValueError, st.forward_mixed, {"modes": ["s1"], "s2": mixed["s2"][[0]]}, says="'s1'")
raises(ValueError, st.forward_mixed, dict(mixed, modes=["s2", "aerial", "s1", "both", "sar", "s1"]))
raises(ValueError, st.forward_mixed, dict(mixed, aerial=torch.randn(2, 4, 128, 128)),
       says="one image size")

# ---- layout, spatial[row, column] is the patch at that place -----------------------------------
flat = tiny_student(depth=0).eval()                   # no blocks = tokens do not mix
img_a = torch.randn(1, 12, 64, 64)
img_b = img_a.clone()
img_b[:, :, 16:32, 48:64] += 1.0                      # the patch in row 1, column 3
with torch.no_grad():
    (sum_a, spa_a), (sum_b, spa_b) = flat(s2=img_a), flat(s2=img_b)
    change = (spa_a - spa_b).abs().sum(-1)[0]         # (4, 4)
    assert change[1, 3] > 0 and change.sum() == change[1, 3], change
    # the summary is the class token after the final LayerNorm
    assert torch.equal(sum_a, sum_b)
    assert torch.allclose(sum_a[0], flat.norm(flat.cls_token)[0, 0], atol=1e-6)
    # the final LayerNorm is applied to the patch tokens too
    tokens = flat.embed(s2=img_a).flatten(2).transpose(1, 2) + flat.pos(4)
    assert torch.allclose(spa_a.reshape(1, 16, 32), flat.norm(tokens), atol=1e-6)
    assert not torch.allclose(spa_a.reshape(1, 16, 32), tokens, atol=1e-3)
    # with blocks the summary does see the image
    assert not torch.equal(st(s2=img_a)[0], st(s2=img_b)[0])

# ---- position embedding, learned for 16 x 16, bicubic for other grids --------------------------
assert st.pos_embed.shape == (1, 32, 16, 16) and st.pos_embed.requires_grad
assert st.cls_token.shape == (1, 1, 32) and st.cls_token.requires_grad
with torch.no_grad():
    assert torch.equal(st.pos(16), st.pos_embed.flatten(2).transpose(1, 2))
    for g in (8, 24):
        cubic = F.interpolate(st.pos_embed, size=(g, g), mode="bicubic", align_corners=False)
        linear = F.interpolate(st.pos_embed, size=(g, g), mode="bilinear", align_corners=False)
        assert st.pos(g).shape == (1, g * g, 32)
        assert torch.equal(st.pos(g), cubic.flatten(2).transpose(1, 2))
        assert not torch.allclose(st.pos(g), linear.flatten(2).transpose(1, 2), atol=1e-4)
    # a ramp over the rows stays a ramp over the rows (no axis is swapped on the way)
    ramp = tiny_student(img=64, depth=0)
    ramp.pos_embed.copy_(torch.arange(4.0).view(1, 1, 4, 1).expand(1, 32, 4, 4))
    p8 = ramp.pos(8).reshape(8, 8, 32)
    assert torch.allclose(p8, p8[:, :1].expand(8, 8, 32)) and (p8[1:, 0, 0] >= p8[:-1, 0, 0]).all()
    assert p8[-1, 0, 0] > p8[0, 0, 0]
    # the position embedding is used
    before = flat(s2=img_a)[1]
    flat.pos_embed.zero_()
    assert not torch.allclose(before, flat(s2=img_a)[1], atol=1e-4)

# ---- the default model ------------------------------------------------------------------
full = Student()
n_full = full.n_params()
print(f"default student: {n_full:,} parameters")
assert (full.img, full.patch, full.grid, full.dim, len(full.blocks)) == (256, 16, 16, 768, 12)
assert all(blk.heads == 12 and blk.fc1.out_features == 4 * 768 for blk in full.blocks)
assert full.embed_s1.in_channels == 2 and full.embed_s2.in_channels == 12
assert full.embed_aerial.in_channels == 4 and full.chans == {"s1": 2, "s2": 12, "aerial": 4}
for layer in (full.embed_s1, full.embed_s2, full.embed_aerial):
    assert layer.kernel_size == layer.stride == (16, 16) and layer.out_channels == 768
assert n_full == 88_794_624
assert abs(n_full - 86e6) / 86e6 < 0.04                              # close to 86 M
n_in = sum(p.numel() for m in (full.embed_s1, full.embed_s2, full.embed_aerial)
           for p in m.parameters())
assert n_in == 393_984 + 2_360_064 + 787_200 and n_full - n_in == 85_253_376
assert "88,794,624" in " ".join(S.__doc__.split())
del full

# ---- heads ------------------------------------------------------------------------------
# three slots, the hidden width is the widest teacher (for student C)
dims = {"dinov3-l-sat": 1024, "terrafm-b": 768, "pe-spatial-l": 1024}
heads = Heads(dims)
print(f"heads for the three teachers: {heads.n_params():,} parameters")
assert heads.hidden == 1024 and list(heads.summary) == list(heads.spatial) == list(dims)
for name, d in dims.items():
    for head in (heads.summary[name], heads.spatial[name]):
        assert [type(m).__name__ for m in head] == ["Linear", "LayerNorm", "GELU", "Linear"]
        assert (head[0].in_features, head[0].out_features) == (768, 1024)
        assert head[1].normalized_shape == (1024,)
        assert (head[3].in_features, head[3].out_features) == (1024, d)
    assert heads.summary[name][0].weight is not heads.spatial[name][0].weight
with torch.no_grad():
    out = heads(torch.randn(2, 768), torch.randn(2, 16, 16, 768))
assert list(out) == list(dims)
for name, d in dims.items():
    assert out[name][0].shape == (2, d) and out[name][1].shape == (2, 16, 16, d)
del heads, out
small = Heads({"a": 8, "b": 24}, student_dim=32)
assert small.hidden == 24 and small.summary["a"][0].in_features == 32
with torch.no_grad():      # the spatial head works per patch, one patch in, the same vector out
    grid = torch.randn(1, 4, 4, 32)
    assert torch.allclose(small.spatial["a"](grid)[0, 2, 1], small.spatial["a"](grid[0, 2, 1]),
                          atol=1e-6)
raises(ValueError, Heads, {}, says="no teacher")
raises(ValueError, Heads, {"a.b": 8})
raises(ValueError, Heads, {"": 8})
raises(ValueError, Heads, {"a": 0})
raises(ValueError, Heads, {"a": 7.5})

# ---- grid matching, 16 x 16 student against 14, 16 and 27 -------------------------------------
torch.manual_seed(1)
s_grid = torch.randn(2, 16, 16, 8)
for g_t, g in ((14, 14), (16, 16), (27, 16)):
    t_grid = torch.randn(2, g_t, g_t, 8)
    a, b = S.match_grids(s_grid, t_grid)
    assert a.shape == b.shape == (2, g, g, 8), (g_t, a.shape, b.shape)
    a2, b2 = S.match_grids(t_grid, s_grid)                       # the order does not matter
    assert torch.equal(a2, b) and torch.equal(b2, a)
    assert (a is s_grid) == (g_t >= 16) and (b is t_grid) == (g_t <= 16)   # right size = untouched
    moved, source = (a, s_grid) if g_t < 16 else (b, t_grid)
    if g_t != 16:
        ref = F.interpolate(bchw(source), size=(g, g), mode="bilinear", align_corners=False)
        near = F.interpolate(bchw(source), size=(g, g), mode="nearest")
        cubic = F.interpolate(bchw(source), size=(g, g), mode="bicubic", align_corners=False)
        assert torch.equal(bchw(moved), ref)
        assert not torch.allclose(bchw(moved), near, atol=1e-3)
        assert not torch.allclose(bchw(moved), cubic, atol=1e-3)
rows = torch.arange(27.0).view(1, 27, 1, 1) + 100.0 * torch.arange(3.0).view(1, 1, 1, 3)
rows16 = S.resize_grid(rows.expand(1, 27, 27, 3), 16)            # rows stay rows
assert rows16.shape == (1, 16, 16, 3)
assert torch.allclose(rows16, rows16[:, :, :1].expand(1, 16, 16, 3), atol=1e-4)
assert (rows16[0, 1:, 0, 0] > rows16[0, :-1, 0, 0]).all()
assert torch.allclose(rows16[..., 2] - rows16[..., 0], torch.full((1, 16, 16), 200.0), atol=1e-3)
const = S.resize_grid(torch.full((1, 16, 16, 3), 2.5), 14)
assert torch.allclose(const, torch.full((1, 14, 14, 3), 2.5), atol=1e-6)
# no shift, a ramp is read at the centre of every patch of the new grid (closed form, no torch)
for g_from, g_to in ((27, 16), (16, 14)):
    ramp = torch.arange(float(g_from)).view(1, g_from, 1, 1).expand(1, g_from, g_from, 1)
    centre = (torch.arange(float(g_to)) + 0.5) * g_from / g_to - 0.5
    assert torch.allclose(S.resize_grid(ramp, g_to)[0, :, 0, 0], centre, atol=1e-5)
    assert torch.allclose(S.resize_grid(ramp.transpose(1, 2), g_to)[0, 0, :, 0], centre, atol=1e-5)
# the numbers in the module docstring - from 27 to 16 a teacher row counts between 0.31 and 0.97
eye = torch.eye(27).view(1, 27, 1, 27).expand(1, 27, 27, 27)     # channel r is 1 on row r
counts = S.resize_grid(eye, 16)[0, :, 0, :].sum(0)               # one number per teacher row
assert abs(float(counts.min()) - 0.3125) < 1e-4 and abs(float(counts.max()) - 0.96875) < 1e-4
assert abs(float(counts.sum()) - 16.0) < 1e-4
assert "between 0.31 and 0.97" in " ".join(S.__doc__.split())

# ---- loss ---------------------------------------------------------------------------------
toy_dims = {"t14": 24, "t16": 16, "t27": 40}
toy_grid = {"t14": 14, "t16": 16, "t27": 27}


def toy_targets(b: int = 2) -> dict:
    return {n: TeacherOut(torch.randn(b, d), torch.randn(b, toy_grid[n], toy_grid[n], d))
            for n, d in toy_dims.items()}


torch.manual_seed(2)
target = toy_targets()
# a perfect prediction on each of the three grids gives 0
free = torch.randn(2, 16, 16, 24)
target["t14"] = TeacherOut(target["t14"].summary, S.resize_grid(free, 14))
perfect = {
    "t14": (3.0 * target["t14"].summary, free),                  # cosine ignores the length
    "t16": (target["t16"].summary.clone(), target["t16"].spatial.clone()),
    "t27": (0.5 * target["t27"].summary, S.resize_grid(target["t27"].spatial, 16)),
}
loss, parts = distil_loss(perfect, target)
assert sorted(parts) == sorted(f"{n}/{k}" for n in toy_dims for k in ("summary", "spatial"))
assert all(isinstance(v, float) for v in parts.values())
assert loss.ndim == 0 and loss.dtype == torch.float32
assert abs(float(loss)) < 1e-5 and all(abs(v) < 1e-5 for v in parts.values()), parts

# known values - opposite summary = 2, summary at a right angle = 1,
# zeros against a constant c = c * c
one = {"t": TeacherOut(torch.tensor([[1.0, 0.0], [0.0, 2.0]]), torch.full((2, 27, 27, 2), 3.0))}
_, p_opp = distil_loss({"t": (-one["t"].summary, torch.zeros(2, 16, 16, 2))}, one)
assert abs(p_opp["t/summary"] - 2.0) < 1e-6 and abs(p_opp["t/spatial"] - 9.0) < 1e-5, p_opp
_, p_right = distil_loss({"t": (torch.tensor([[0.0, 5.0], [7.0, 0.0]]),
                                torch.zeros(2, 16, 16, 2))}, one)
assert abs(p_right["t/summary"] - 1.0) < 1e-6, p_right
# mean over the batch, one sample equal and one opposite = (0 + 2) / 2
_, p_half = distil_loss({"t": (one["t"].summary * torch.tensor([[1.0], [-1.0]]),
                               torch.zeros(2, 16, 16, 2))}, one)
assert abs(p_half["t/summary"] - 1.0) < 1e-6, p_half

# every teacher has weight 1, so the total is the plain sum of all parts
torch.manual_seed(3)
target = toy_targets()
pred = {n: (torch.randn(2, d), torch.randn(2, 16, 16, d)) for n, d in toy_dims.items()}
loss, parts = distil_loss(pred, target)
assert abs(float(loss) - sum(parts.values())) < 1e-5 and all(v > 0 for v in parts.values())
for n in toy_dims:        # each part is the loss of that teacher alone
    _, alone = distil_loss({n: pred[n]}, {n: target[n]})
    assert abs(alone[f"{n}/summary"] - parts[f"{n}/summary"]) < 1e-6
    assert abs(alone[f"{n}/spatial"] - parts[f"{n}/spatial"]) < 1e-6
# the smaller grid is used, t27 is compared at 16 x 16 and t14 at 14 x 14
ref27 = F.mse_loss(pred["t27"][1], S.resize_grid(target["t27"].spatial, 16))
ref14 = F.mse_loss(S.resize_grid(pred["t14"][1], 14), target["t14"].spatial)
big27 = F.mse_loss(S.resize_grid(pred["t27"][1], 27), target["t27"].spatial)
assert abs(parts["t27/spatial"] - float(ref27)) < 1e-6
assert abs(parts["t14/spatial"] - float(ref14)) < 1e-6
assert abs(parts["t27/spatial"] - float(big27)) > 1e-3        # the larger grid gives another number
# predictions in bfloat16, the loss is still computed in float32 and the target is not rounded
pred16 = {n: (s.to(torch.bfloat16), p.to(torch.bfloat16)) for n, (s, p) in pred.items()}
loss16, parts16 = distil_loss(pred16, target)
_, parts32 = distil_loss({n: (s.float(), p.float()) for n, (s, p) in pred16.items()}, target)
assert loss16.dtype == torch.float32 and parts16 == parts32, (loss16.dtype, parts16, parts32)

# standardise is applied to the spatial target of the named teacher, on the teacher's own grid
seen = []


def double(t: torch.Tensor) -> torch.Tensor:
    seen.append(tuple(t.shape))
    return 2.0 * t


pred2 = dict(pred)
pred2["t27"] = (pred["t27"][0], S.resize_grid(2.0 * target["t27"].spatial, 16))
_, plain = distil_loss(pred2, target)
_, stand = distil_loss(pred2, target, standardise={"t27": double})
assert seen == [(2, 27, 27, 40)]
assert plain["t27/spatial"] > 0.1 and abs(stand["t27/spatial"]) < 1e-6
for key in parts:         # nothing else moves, the summary target is not touched
    if key != "t27/spatial":
        assert stand[key] == plain[key], key
_, none_entry = distil_loss(pred2, target, standardise={"t27": None})
assert none_entry == plain

# guards of the loss
raises(ValueError, distil_loss, {}, {})
raises(ValueError, distil_loss, pred, {n: target[n] for n in ("t14", "t16")})
raises(ValueError, distil_loss, {n: pred[n] for n in ("t14", "t16")}, target)
raises(ValueError, distil_loss, pred, target, standardise={"t99": double})
ok_t = {"t": TeacherOut(torch.randn(2, 8), torch.randn(2, 14, 14, 8))}
distil_loss({"t": (torch.randn(2, 8), torch.randn(2, 16, 16, 8))}, ok_t)
raises(ValueError, distil_loss, {"t": (torch.randn(2, 9), torch.randn(2, 16, 16, 8))}, ok_t)
raises(ValueError, distil_loss, {"t": (torch.randn(3, 8), torch.randn(2, 16, 16, 8))}, ok_t)
raises(ValueError, distil_loss, {"t": (torch.randn(2, 8), torch.randn(2, 16, 16, 9))}, ok_t)
raises(ValueError, distil_loss, {"t": (torch.randn(2, 8), torch.randn(3, 16, 16, 8))}, ok_t)
raises(ValueError, distil_loss, {"t": (torch.randn(2, 8), torch.randn(2, 16, 12, 8))}, ok_t)
raises(ValueError, distil_loss, {"t": (torch.randn(2, 8), torch.randn(2, 256, 8))}, ok_t)
raises(ValueError, distil_loss, {"t": (torch.randn(2, 8), torch.randn(2, 8, 16, 16))}, ok_t)
raises(ValueError, distil_loss, {"t": (torch.randn(2, 8), torch.randn(2, 16, 16, 8))},
       {"t": TeacherOut(torch.randn(2, 8), torch.randn(2, 14, 12, 8))})

# ---- summary-only teachers (SigLIP2 gives its summary only) ----------
dims_so = {"sg": 12, "da": 8, "rd": 10}
h_so = Heads(dims_so, student_dim=32, summary_only=["sg"])
h_full = Heads(dims_so, student_dim=32)
assert h_so.summary_only == ("sg",) and h_full.summary_only == ()
assert h_so.hidden == h_full.hidden == 12                     # its width still counts for the hidden width
assert list(h_so.summary) == ["sg", "da", "rd"] and list(h_so.spatial) == ["da", "rd"]
one_spatial = sum(p.numel() for p in h_full.spatial["sg"].parameters())
assert one_spatial == (32 * 12 + 12) + 2 * 12 + (12 * 12 + 12)
assert h_full.n_params() - h_so.n_params() == one_spatial
assert not any(k.startswith("spatial.sg.") for k in h_so.state_dict())
assert Heads(dims_so, student_dim=32, summary_only=("rd", "sg")).summary_only == ("sg", "rd")   # teacher order
with torch.no_grad():
    o_so = h_so(torch.randn(2, 32), torch.randn(2, 4, 4, 32))
assert o_so["sg"][1] is None and o_so["sg"][0].shape == (2, 12) and o_so["da"][1].shape == (2, 4, 4, 8)
raises(ValueError, Heads, {"a": 8}, summary_only=["b"], says="match no teacher")
raises(RuntimeError, h_so.load_state_dict, h_full.state_dict())          # other heads, strict
# the loss of a summary-only teacher is its summary part alone
torch.manual_seed(10)
t_so = {"sg": TeacherOut(torch.randn(2, 12), torch.randn(2, 16, 16, 12)),
        "da": TeacherOut(torch.randn(2, 8), torch.randn(2, 14, 14, 8))}
p_so = {"sg": (torch.randn(2, 12), None), "da": (torch.randn(2, 8), torch.randn(2, 16, 16, 8))}
loss_so, parts_so = distil_loss(p_so, t_so)
assert list(parts_so) == ["sg/summary", "da/summary", "da/spatial"]
assert abs(float(loss_so) - sum(parts_so.values())) < 1e-6
_, sg_alone = distil_loss({"sg": p_so["sg"]}, {"sg": t_so["sg"]})
_, sg_with_spatial = distil_loss({"sg": (p_so["sg"][0], torch.randn(2, 16, 16, 12))}, {"sg": t_so["sg"]})
assert sg_alone == {"sg/summary": sg_with_spatial["sg/summary"]} and sg_alone["sg/summary"] == parts_so["sg/summary"]
broken = {"sg": TeacherOut(t_so["sg"].summary, torch.full((3, 5, 7, 1), float("nan")))}
assert distil_loss({"sg": p_so["sg"]}, broken)[1] == sg_alone           # its spatial target is not read
raises(ValueError, distil_loss, {"sg": p_so["sg"]}, {"sg": t_so["sg"]}, standardise={"sg": double},
       says="summary-only")
assert distil_loss({"sg": p_so["sg"]}, {"sg": t_so["sg"]}, standardise={"sg": None})[1] == sg_alone
# a training step reaches every head there is, and the trunk through the summary-only head too
torch.manual_seed(11)
st_so, hd_so = tiny_student(), Heads({"sg": 12, "t16": 16}, student_dim=32, summary_only=["sg"])
frz = {"sg": TeacherOut(torch.randn(2, 12), torch.randn(2, 16, 16, 12)),
       "t16": TeacherOut(torch.randn(2, 16), torch.randn(2, 16, 16, 16))}
loss_st, parts_st = distil_loss(hd_so(*st_so(**mode_inputs(images(256), "both"))), frz)
loss_st.backward()
assert list(parts_st) == ["sg/summary", "t16/summary", "t16/spatial"]
assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in hd_so.parameters())
assert float(st_so.cls_token.grad.abs().sum()) > 0
print("summary-only heads and loss ok")

# ---- teacher outputs come from inference mode and must still work in a backward pass ----------
torch.manual_seed(4)
with torch.inference_mode():
    frozen = toy_targets()
assert all(t.summary.is_inference() and t.spatial.is_inference() for t in frozen.values())
batch = images(256)
toy_student, toy_heads = tiny_student(), Heads(toy_dims, student_dim=32)
loss, parts = distil_loss(toy_heads(*toy_student(**mode_inputs(batch, "both"))), frozen)
loss.backward()
assert torch.isfinite(loss) and toy_student.pos_embed.grad is not None
assert all(t.spatial.is_inference() for t in frozen.values())       # the targets are not changed

# ---- gradients reach the input layer of the inputs that were given, and only those -----------
def check_grads(toy_student, toy_heads, used: set, where) -> None:
    for name in ("s1", "s2", "aerial"):
        layer = getattr(toy_student, f"embed_{name}")
        if name in used:
            for p in (layer.weight, layer.bias):
                assert p.grad is not None and float(p.grad.abs().sum()) > 0, (where, name)
        else:
            assert layer.weight.grad is None and layer.bias.grad is None, (where, name)
    rest = [(k, p) for k, p in toy_student.named_parameters() if not k.startswith("embed_")]
    rest += [(f"heads.{k}", p) for k, p in toy_heads.named_parameters()]
    for k, p in rest:
        assert p.grad is not None, (where, k)
        assert torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0, (where, k)


for mode in MODES:
    torch.manual_seed(5)
    toy_student, toy_heads = tiny_student(), Heads(toy_dims, student_dim=32)
    loss, _ = distil_loss(toy_heads(*toy_student(**mode_inputs(batch, mode))), frozen)
    loss.backward()
    check_grads(toy_student, toy_heads, set(MODE_KEYS[mode]), mode)
# the same through forward_mixed, only the layers of the modes in the batch get a gradient
mixed_cases = {("s1", "aerial"): ["s1", "aerial"], ("s1", "s2", "aerial"): ["both", "aerial"],
               ("s1", "s2"): ["both", "s2"], ("aerial",): ["aerial", "aerial"],
               ("s2",): ["s2", "s2"]}
for used, modes in mixed_cases.items():
    torch.manual_seed(5)
    toy_student, toy_heads = tiny_student(), Heads(toy_dims, student_dim=32)
    n_mt = sum(m != "aerial" for m in modes)
    mixed = {"modes": modes}
    if n_mt:
        mixed["s1"], mixed["s2"] = batch["s1"][:n_mt], batch["s2"][:n_mt]
    if n_mt < 2:
        mixed["aerial"] = batch["aerial"][:2 - n_mt]
    loss, _ = distil_loss(toy_heads(*toy_student.forward_mixed(mixed)), frozen)
    loss.backward()
    check_grads(toy_student, toy_heads, set(used), modes)

# ---- one optimiser step lowers the loss on a fixed batch ----------------------------------------
for mode in MODES:
    torch.manual_seed(6)
    toy_student, toy_heads = tiny_student(), Heads(toy_dims, student_dim=32)
    opt = torch.optim.AdamW(list(toy_student.parameters()) + list(toy_heads.parameters()), lr=1e-3)
    first, _ = distil_loss(toy_heads(*toy_student(**mode_inputs(batch, mode))), frozen)
    opt.zero_grad()
    first.backward()
    opt.step()
    with torch.no_grad():
        second, _ = distil_loss(toy_heads(*toy_student(**mode_inputs(batch, mode))), frozen)
    first, second = float(first.detach()), float(second)
    print(f"one AdamW step, mode {mode}, loss {first:.4f} -> {second:.4f}")
    assert second < first, (mode, first, second)

# ---- the loss is float32 also when the model runs in bfloat16 ----------------------------------
torch.manual_seed(7)
toy_student, toy_heads = tiny_student(), Heads(toy_dims, student_dim=32)
with torch.autocast("cpu", dtype=torch.bfloat16):
    summary, spatial = toy_student(**mode_inputs(batch, "both"))
    loss, parts = distil_loss(toy_heads(summary, spatial), frozen)
assert spatial.dtype == torch.bfloat16, spatial.dtype
assert loss.dtype == torch.float32 and torch.isfinite(loss)
loss.backward()
assert toy_student.embed_s1.weight.grad.dtype == torch.float32
with torch.no_grad():
    exact, _ = distil_loss(toy_heads(*toy_student(**mode_inputs(batch, "both"))), frozen)
loss, exact = float(loss.detach()), float(exact)
assert abs(loss - exact) < 0.05 * exact, (loss, exact)
for px in (128, 384):     # the same at sizes where the position embedding is resized
    other = images(px)
    toy_student.zero_grad()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        pred_bf = toy_heads(*toy_student(**mode_inputs(other, "both")))
        inside, parts_in = distil_loss(pred_bf, frozen)
    _, parts_out = distil_loss(pred_bf, frozen)                 # same predictions, no autocast
    assert pred_bf["t27"][1].shape == (2, px // 16, px // 16, 40), (px, pred_bf["t27"][1].shape)
    assert pred_bf["t27"][1].dtype == torch.bfloat16
    assert inside.dtype == torch.float32 and torch.isfinite(inside), (px, inside)
    assert parts_in == parts_out, (px, parts_in, parts_out)     # autocast does not reach the loss
    inside.backward()
    grad = toy_student.pos_embed.grad
    assert torch.isfinite(grad).all() and float(grad.abs().sum()) > 0, px

# ---- the notes on the sources name both head classes of the released RADIO code -----------------
notes = " ".join(S.__doc__.split())
assert 'MLP (version "v1") uses ReLU' in notes and 'MLP2 (version "v2") uses GELU' in notes
assert "was not read" in notes and "the released RADIO code (class MLP" not in notes


# ---- determinism under a fixed seed ---------------------------------------------------------
def run(seed: int):
    torch.manual_seed(seed)
    m, h = tiny_student(), Heads(toy_dims, student_dim=32)
    x = images(256)
    summary, spatial = m(**mode_inputs(x, "both"))
    loss, parts = distil_loss(h(summary, spatial), frozen)
    loss.backward()
    return summary.detach(), spatial.detach(), m.embed_s2.weight.grad.clone(), parts


r0, r0_again, r1 = run(0), run(0), run(1)
assert all(torch.equal(a, b) for a, b in zip(r0[:3], r0_again[:3])) and r0[3] == r0_again[3]
assert not torch.equal(r0[0], r1[0]) and not torch.equal(r0[1], r1[1]) and r0[3] != r1[3]

# ---- the trunk against the ViT of timm, same weights, same output -----------------------------
if VisionTransformer is not None:
    torch.manual_seed(8)
    ours = tiny_student(img=64, depth=3).eval()
    for p in ours.parameters():                     # biases and LayerNorms away from 0 and 1
        p.data.add_(0.05 * torch.randn_like(p))
    ref = VisionTransformer(img_size=64, patch_size=16, in_chans=12, num_classes=0, global_pool="",
                            embed_dim=32, depth=3, num_heads=4, mlp_ratio=4.0, qkv_bias=True,
                            class_token=True).eval()
    pos = ours.pos_embed.flatten(2).transpose(1, 2)
    state = {"cls_token": ours.cls_token,
             "pos_embed": torch.cat([torch.zeros(1, 1, 32), pos], dim=1),
             "patch_embed.proj.weight": ours.embed_s2.weight,
             "patch_embed.proj.bias": ours.embed_s2.bias,
             "norm.weight": ours.norm.weight, "norm.bias": ours.norm.bias}
    names = {"norm1": "norm1", "qkv": "attn.qkv", "proj": "attn.proj", "norm2": "norm2",
             "fc1": "mlp.fc1", "fc2": "mlp.fc2"}
    for i, blk in enumerate(ours.blocks):
        for mine, theirs in names.items():
            for end in ("weight", "bias"):
                state[f"blocks.{i}.{theirs}.{end}"] = getattr(getattr(blk, mine), end)
    ref.load_state_dict({k: v.detach().clone() for k, v in state.items()}, strict=True)
    x = torch.randn(2, 12, 64, 64)
    with torch.no_grad():
        want = ref.forward_features(x)              # (B, 1 + 16, 32), after the final norm
        summary, spatial = ours(s2=x)
    assert want.shape == (2, 17, 32)
    assert torch.allclose(summary, want[:, 0], atol=1e-5), (summary - want[:, 0]).abs().max()
    assert torch.allclose(spatial.reshape(2, 16, 32), want[:, 1:], atol=1e-5)
    n_ref = sum(p.numel() for p in ref.parameters())
    n_ours = ours.n_params() - sum(p.numel() for m in (ours.embed_s1, ours.embed_aerial)
                                   for p in m.parameters())
    assert n_ours == n_ref - 32, (n_ours, n_ref)    # timm has a position for the class token
    print(f"timm {timm.__version__} reference check ok, largest difference "
          f"{float((spatial.reshape(2, 16, 32) - want[:, 1:]).abs().max()):.2e}")

print("test_student: all checks passed")
