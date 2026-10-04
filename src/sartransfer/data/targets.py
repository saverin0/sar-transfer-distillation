"""Pixel labels to one label per encoder patch, copied from part two (only what `domains.run` uses).

The probe reads one vector per 16 x 16 px patch, so each patch needs a single label, the
majority vote of its valid pixels. A patch without a valid pixel is marked IGNORE.
"""

from __future__ import annotations

import numpy as np

IGNORE = 255                                   # target value for excluded patches


def patch_labels(classes: np.ndarray, valid: np.ndarray, patch: int = 16,
                 min_purity: float = 0.0, n_classes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Reduce pixel labels to one label per patch by majority vote.

    classes, valid : (H, W), H and W already multiples of `patch`
    min_purity     : 0.0 keeps every patch that has any valid pixel (majority
                     vote). Raise it to drop mixed patches -- e.g. 0.75 keeps
                     only patches where one class holds 75 % of valid pixels.
                     Note that discarding mixed patches removes exactly the zone
                     boundaries the task is about, so the default keeps them.

    Returns (labels, patch_valid), both (H//patch, W//patch).
    """
    h, w = classes.shape
    if h % patch or w % patch:
        raise ValueError(f"shape {classes.shape} is not a multiple of patch {patch}")
    gh, gw = h // patch, w // patch

    # (gh, gw, patch*patch) blocks
    blocks = (classes.reshape(gh, patch, gw, patch).transpose(0, 2, 1, 3)
              .reshape(gh, gw, patch * patch))
    vblocks = (valid.reshape(gh, patch, gw, patch).transpose(0, 2, 1, 3)
               .reshape(gh, gw, patch * patch))

    counts = np.zeros((gh, gw, n_classes), dtype=np.int32)
    for cls in range(n_classes):
        counts[..., cls] = ((blocks == cls) & vblocks).sum(axis=-1)

    n_valid = counts.sum(axis=-1)
    labels = counts.argmax(axis=-1).astype(np.uint8)
    keep = n_valid > 0
    if min_purity > 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            purity = counts.max(axis=-1) / np.maximum(n_valid, 1)
        keep &= purity >= min_purity

    labels[~keep] = IGNORE
    return labels, keep
