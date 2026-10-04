"""Shared records and the teacher interface: shapes, names, guards."""
import _setup  # noqa: F401

import numpy as np
import torch

from sartransfer.distil import records as R
from sartransfer.distil.teachers import TEACHERS, Teacher, TeacherOut, band_index

assert len(R.S2_BANDS) == 12 and "B10" not in R.S2_BANDS
assert set(R.S2_RES_M) == set(R.S2_BANDS)
assert [R.CROP_PX * 10 // R.S2_RES_M[b] for b in ("B02", "B05", "B01")] == [384, 192, 64]
assert band_index(("B04", "B03", "B02")) == [3, 2, 1]
assert sorted(TEACHERS) == ["dinov3-l-sat", "pe-spatial-l", "siglip2-so400m-256", "siglip2-so400m-naflex",
                            "terrafm-b", "terrafm-b-radar"]        # (part four)

try:
    R.Skip("x", "not a reason")
    raise SystemExit("Skip accepted an unknown reason")
except ValueError:
    pass
assert R.Skip("x", "gap_gt_7d").reason == "gap_gt_7d"


class Toy(Teacher):
    name, dim, patch, input_size = "toy", 8, 16, 64
    bands, rules = ("B04", "B03", "B02"), ("unit",)

    def prepare(self, s2):
        return s2[:, band_index(self.bands)] / 10000.0

    def _forward(self, x):
        b = x.shape[0]
        return TeacherOut(torch.zeros(b, 8), torch.zeros(b, 4, 4, 8))


t = Toy(torch.nn.Linear(1, 1), "unit", device="cpu", dtype=torch.float32)
out = t(torch.zeros(2, 12, 64, 64))
assert out.summary.shape == (2, 8) and out.spatial.shape == (2, 4, 4, 8)
for bad in (torch.zeros(2, 3, 64, 64), torch.zeros(2, 12, 64, 64, dtype=torch.int16)):
    try:
        t(bad)
        raise SystemExit("bad input accepted")
    except ValueError:
        pass
try:
    Toy(torch.nn.Linear(1, 1), "other", device="cpu")
    raise SystemExit("unknown rule accepted")
except ValueError:
    pass
assert np.dtype(np.float16).itemsize == 2
print("test_records: all checks passed")
