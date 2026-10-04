"""Shared data records. Every module of `distil` passes these around, nothing else.

Units and layouts are fixed here, once:

- Sentinel-2 values are the digital numbers stored by Major-TOM (uint16, L2A). Whether an
  image carries the +1000 offset follows its processing baseline, measured in stage 0 and
  decided on 2026-09-30. `pairing.remove_offset_by_baseline` takes it off and
  `pairing.make_pair` stores every pair on one scale unless told not to (offset=None).
  A pixel of 0 is no-data in every band, a crop with one is skipped (`s2_nodata_in_crop`).
- Sentinel-2 bands keep their native resolution (10, 20 or 60 m) until `pairing.s2_to_grid`
  brings them to the 10 m grid.
- Sentinel-1 is linear power in a cell (as stored) and dB in a pair (10 * log10).
- Band order is always `S2_BANDS`; Sentinel-1 channel order is always (VV, VH).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

S2_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
S2_RES_M = {"B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20, "B07": 20,
            "B08": 10, "B8A": 20, "B09": 60, "B11": 20, "B12": 20}
S1_CHANNELS = ("VV", "VH")

CROP_PX = 384            # side of the Sentinel-2 crop in the 249k subset, at 10 m
S1_CELL_PX = 1068        # side of a regular Core-S1RTC cell, at 10 m
PIXEL_M = 10.0

# Every reason a cell can be left out. The counts per reason are reported.
SKIP_REASONS = (
    "no_s1_for_cell",        # the grid cell has no Sentinel-1 row
    "gap_gt_7d",             # Sentinel-1 and Sentinel-2 dates more than 7 days apart
    "s1_not_1068_10m",       # Sentinel-1 cell is not 1,068 px at 10 m
    "crs_differs",           # the two images are not in the same coordinate system
    "crop_outside_s1",       # the Sentinel-2 crop is not fully inside the Sentinel-1 cell
    "s1_nodata_in_crop",     # no-data, non-finite or non-positive Sentinel-1 pixels in the crop
    "s2_nodata_in_crop",     # a 0 pixel (the L2A no-data value) in any Sentinel-2 band of the crop
    "cloud_gt_10pct",        # more than 10 % cloud in the crop
    "s2_band_shape",         # a Sentinel-2 band does not have the expected native size
    "read_error",            # the row could not be read or decoded
)

# The cloud mask codes that count as cloud, decided 2026-09-30 after the stage 0 run. The
# numbers are the order of the classes on the Major-TOM card (clear, thin, thick, shadow),
# no page gives them. Stage 0 saw the codes 0, 1, 2, 3, code 0 covers 97.6 % of the pixels.
# `pairing.make_pair` applies the cloud rule only when it gets codes (None = not applied).
CLOUD_CODES = (1, 2, 3)


@dataclass
class S2Cell:
    grid_cell: str
    product_id: str
    datetime: Any                         # pandas.Timestamp, UTC
    bands: dict[str, np.ndarray]          # name -> (h, w) uint16 at native resolution
    transforms: dict[str, tuple]          # name -> affine as a 6-tuple (a, b, c, d, e, f)
    crs: str                              # e.g. "EPSG:32633"
    cloud_mask: np.ndarray | None         # (384, 384) uint8 at 10 m, codes as stored
    bounds: tuple[float, float, float, float]   # (left, bottom, right, top) of the 10 m crop


@dataclass
class S1Cell:
    grid_cell: str
    product_id: str
    datetime: Any                         # pandas.Timestamp, UTC
    vv: np.ndarray                        # (h, w) float32, linear power
    vh: np.ndarray                        # (h, w) float32, linear power
    transform: tuple                      # affine as a 6-tuple
    crs: str
    nodata: float | None
    orbit_state: str | None = None


@dataclass
class Pair:
    grid_cell: str
    s1_db: np.ndarray                     # (2, 384, 384) float16, dB, (VV, VH)
    s2: dict[str, np.ndarray]             # name -> uint16 at native resolution (384, 192 or 64 px),
                                          # offset taken off when meta["offset_removed"] is True
    cloud_mask: np.ndarray | None         # (384, 384) uint8
    s2_datetime: Any
    s1_datetime: Any
    gap_days: float
    crs: str
    bounds: tuple[float, float, float, float]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Skip:
    grid_cell: str
    reason: str                           # one of SKIP_REASONS
    detail: str = ""

    def __post_init__(self) -> None:
        if self.reason not in SKIP_REASONS:
            raise ValueError(f"unknown skip reason {self.reason!r}")
