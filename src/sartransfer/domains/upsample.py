"""Patch features twice as fine, for the cheap test of the 8 px grid (notebook 03b).

`run_domain(upsampler=...)` calls one of these on the features of every split right after they are
extracted. Each is called as up(feats, images, stats) - feats (N, g, g, D) float16 on the 16 px grid,
images (N, C, H, W) as stored in the prepared split, stats those of the run's training split - and
returns (N, 2g, 2g, D) float16. `name` goes into the run name, `factor` is 2.

Bilinear   torch's bilinear interpolation (align_corners False), the control.
AnyUp      the vendored AnyUp (models/anyup, CC BY 4.0) with the released weights, guided by the
           decoder's skip image (the standardised co-pol as grey 0-255, `s1input.skip_image`) at the
           chip's full size, the output at the 8 px grid; bf16 autocast on CUDA, as part two ran it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FACTOR = 2


class Bilinear:
    """Bilinear 2x, the control of the 8 px test."""

    name, factor = "bilinear", FACTOR

    def __init__(self, device: str = "cuda", batch: int = 64):
        self.device, self.batch = device, batch

    def __call__(self, feats: np.ndarray, images: np.ndarray, stats: dict) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        n, g1, g2, d = feats.shape
        out = np.empty((n, g1 * FACTOR, g2 * FACTOR, d), np.float16)
        with torch.inference_mode():
            for i in range(0, n, self.batch):
                f = torch.from_numpy(np.ascontiguousarray(feats[i:i + self.batch])).to(self.device).float()
                up = F.interpolate(f.permute(0, 3, 1, 2), scale_factor=FACTOR, mode="bilinear", align_corners=False)
                out[i:i + self.batch] = up.permute(0, 2, 3, 1).half().cpu().numpy()
        return out


class AnyUp:
    """AnyUp 2x guided by the decoder's skip image at full size."""

    name, factor = "anyup", FACTOR

    def __init__(self, weights_path: str | Path, device: str = "cuda", batch: int = 32, model=None):
        from ..models.anyup_loader import load_anyup

        self.device, self.batch = device, batch
        self.model = load_anyup(weights_path, device) if model is None else model.to(device).eval()

    def __call__(self, feats: np.ndarray, images: np.ndarray, stats: dict) -> np.ndarray:
        import torch

        from ..models.anyup_loader import guide_image
        from .s1input import skip_image

        n, g1, g2, d = feats.shape
        out = np.empty((n, g1 * FACTOR, g2 * FACTOR, d), np.float16)
        on_cuda = str(self.device).startswith("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=on_cuda):
            for i in range(0, n, self.batch):
                grey = np.round(skip_image(np.asarray(images[i:i + self.batch], np.float32), stats)[:, :1] * 255)
                guide = guide_image(torch.from_numpy(grey.astype(np.uint8)).to(self.device))
                f = torch.from_numpy(np.ascontiguousarray(feats[i:i + self.batch])).to(self.device).float()
                up = self.model(guide, f.permute(0, 3, 1, 2), output_size=(g1 * FACTOR, g2 * FACTOR))
                out[i:i + self.batch] = up.float().permute(0, 2, 3, 1).half().cpu().numpy()
        return out
