"""The vendored AnyUp (models/anyup, see its NOTICE.md) with the authors' weights, checked by size and hash.

Part four, the cheap test of the 8 px grid (notebook 03b). The weights file is kept
on Drive; it is never downloaded here (nothing fetched from another repository while running) - a file that is missing or differs is an error.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .anyup import AnyUp

WEIGHTS_NAME = "anyup_multi_backbone.pth"
WEIGHTS_SHA256 = "b6cc407da8986c7e5c9098e61f7531767a9aca8fff20a1bc6c99d488e61aac59"   # the release file, checked 2026-09-25
WEIGHTS_BYTES = 3_541_624
# AnyUp's guide image is ImageNet-normalised RGB (its README)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_anyup(weights_path: str | Path, device: str = "cuda") -> AnyUp:
    """AnyUp with the released weights, frozen, in eval mode. Refuses a file of another size or hash."""
    p = Path(weights_path)
    if not p.is_file():
        raise FileNotFoundError(f"{p} is not there, the AnyUp weights ({WEIGHTS_NAME}, {WEIGHTS_BYTES:,} bytes)")
    raw = p.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != WEIGHTS_BYTES or digest != WEIGHTS_SHA256:
        raise RuntimeError(f"{p} - size {len(raw):,} and sha256 {digest[:12]} are not those of the released "
                           f"weights ({WEIGHTS_BYTES:,}, {WEIGHTS_SHA256[:12]})")
    state = torch.load(p, map_location="cpu", weights_only=True)      # never executes code
    model = AnyUp()
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model.to(device).eval()


def guide_image(grey: torch.Tensor) -> torch.Tensor:
    """(N, 1, H, W) grey 0-255 (uint8 or float) -> (N, 3, H, W) ImageNet-normalised RGB float."""
    x = grey.float() / 255.0
    x = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std
