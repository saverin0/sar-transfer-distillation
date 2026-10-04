"""One fixed rule per mode, the same for every domain, from the stored channels to the encoder input.

Copied from part two. Fixed there on 2026-09-26 before any new-domain number (no tuning per
domain):

    mode "vv_vh_diff" (default): three channels (co-pol, cross-pol, co - cross),
        each standardised with mean/std from the TRAINING split. In dB the
        third channel is the log polarisation ratio, the usual third channel
        for dual-pol Sentinel-1.
    mode "vv_grey": co-pol only, standardised and copied three times -- the
        glacier study's input (one grey channel), kept for comparison.

Part three adds two modes without the difference
channel, for encoders that read two or four channels as they are:

    mode "co_cross": two channels (co-pol, cross-pol), each standardised with
        mean/std. The statistics come from the training split of the dataset,
        for the student and for TerraFM-B alike (TerraFM-B then maps them into
        its own min-max range inside the encoder, see models.encoders).
    mode "aerial_rgbn": four channels (R, G, B, NIR, 8 bit), each standardised
        with the statistics passed in, keys `AERIAL_KEYS` (the student's own
        aerial statistics from its checkpoint; for DINOv3 255 x its RGB mean
        and std, which is 8 bit / 255 and then mean/std).

No-data pixels (NaN) become 0 after standardisation, i.e. the training mean.
The decoder's image skip is the standardised channel 0 (co-pol, or red for the
aerial domain) mapped to [0, 1] by clip((z + 3) / 6, 0, 1), with the training-split
statistics in every mode. It does not depend on the units.
"""

from __future__ import annotations

import numpy as np

MODES = ("vv_vh_diff", "vv_grey", "co_cross", "aerial_rgbn")
MODE_CHANNELS = {"vv_vh_diff": 3, "vv_grey": 3, "co_cross": 2, "aerial_rgbn": 4}   # encoder input channels
AERIAL_KEYS = ("red", "green", "blue", "nir")


def channel_stats(images: np.ndarray, max_chips: int = 2000, seed: int = 0) -> dict:
    """Mean/std of co, cross and co - cross over finite pixels of up to `max_chips` chips."""
    rng = np.random.default_rng(seed)
    n = len(images)
    idx = np.sort(rng.choice(n, min(max_chips, n), replace=False))
    x = np.asarray(images[idx], np.float32)
    co, cross = x[:, 0], x[:, 1]
    out = {}
    for name, v in (("co", co), ("cross", cross), ("diff", co - cross)):
        f = v[np.isfinite(v)]
        if f.size == 0:
            out[name] = (0.0, 1.0)                       # channel absent everywhere
        else:
            out[name] = (float(f.mean()), float(max(f.std(), 1e-6)))
    return out


def aerial_stats(mean, sd) -> dict:
    """Per-channel (mean, sd) of R, G, B, NIR in 8-bit units -> the statistics `encoder_input` reads."""
    mean, sd = [float(v) for v in mean], [float(v) for v in sd]
    if len(mean) != len(AERIAL_KEYS) or len(sd) != len(AERIAL_KEYS):
        raise ValueError(f"aerial statistics need {len(AERIAL_KEYS)} means and SDs, got {len(mean)} and {len(sd)}")
    if not all(np.isfinite(mean)) or not all(np.isfinite(sd)) or min(sd) <= 0:
        raise ValueError(f"aerial statistics must be finite with SD > 0, got mean {mean}, sd {sd}")
    return {k: (m, s) for k, m, s in zip(AERIAL_KEYS, mean, sd)}


def _z(v: np.ndarray, ms: tuple[float, float]) -> np.ndarray:
    z = (v - ms[0]) / ms[1]
    return np.where(np.isfinite(z), z, 0.0).astype(np.float32)


def encoder_input(images: np.ndarray, stats: dict, mode: str = "vv_vh_diff") -> np.ndarray:
    """(N, 2, H, W) SAR or (N, 4, H, W) aerial -> (N, MODE_CHANNELS[mode], H, W) float32 standardised encoder input."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    x = np.asarray(images, np.float32)
    if mode == "aerial_rgbn":
        if x.ndim != 4 or x.shape[1] != len(AERIAL_KEYS):
            raise ValueError(f"mode {mode!r} expects (N, {len(AERIAL_KEYS)}, H, W), got {x.shape}")
        return np.stack([_z(x[:, i], stats[k]) for i, k in enumerate(AERIAL_KEYS)], axis=1)
    co, cross = x[:, 0], x[:, 1]
    if mode == "co_cross":
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError(f"mode {mode!r} expects (N, 2, H, W), got {x.shape}")
        return np.stack([_z(co, stats["co"]), _z(cross, stats["cross"])], axis=1)
    if mode == "vv_grey":
        z = _z(co, stats["co"])
        return np.stack([z, z, z], axis=1)
    return np.stack([_z(co, stats["co"]), _z(cross, stats["cross"]), _z(co - cross, stats["diff"])], axis=1)


def skip_image(images: np.ndarray, stats: dict) -> np.ndarray:
    """(N, C, H, W) chips -> (N, 1, H, W) float32 in [0, 1] for the decoder's image branch (channel 0)."""
    z = _z(np.asarray(images[:, 0], np.float32), stats["co"])
    return np.clip((z + 3.0) / 6.0, 0.0, 1.0)[:, None]
