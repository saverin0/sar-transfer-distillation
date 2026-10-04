"""The small decoder head of part two (row 3 of the glacier study), copied unchanged.

The head gets the raw grey tile as a second input ("image skip"), because the 16 px
feature grid alone cannot hold pixel-level evidence.

GridDecoder  frozen (D, gh, gw) grid -> 1x1 reduce -> four x2 upsampling stages, each
             concatenating a same-scale map from a tiny image encoder -> class logits at
             tile resolution. 0.54 M trainable parameters at D = 1024.

Only what `domains.run` uses is here (the AnyUp heads and the head loader stay in part two).

Part four (2026-10-04) - `entry=8` for the cheap test of the 8 px grid: features on
an 8 px grid enter at 1/8 and the head is row 3 without its first stage (the stem builds 4 levels, the
last four stages are the same convolutions). `entry=16`, the default, builds the head exactly as before.
`in_ch=2` - the image skip with a second channel (the coherence); the default 1 is
part two's grey tile.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(8, cout), nn.LeakyReLU(0.1, inplace=True))


class ImageStem(nn.Module):
    """Tiny image encoder - maps at strides 1, 2, 4, 8, 16 (channels 16, 24, 32, 48, 64).

    `levels` builds only the first maps a head uses (GridDecoder 5); the maps it does
    build are unchanged.
    """

    CH = (16, 24, 32, 48, 64)

    def __init__(self, in_ch: int = 1, levels: int = 5):
        super().__init__()
        if not 1 <= levels <= len(self.CH):
            raise ValueError(f"levels must be 1-{len(self.CH)}, got {levels}")
        self.blocks = nn.ModuleList()
        c_prev = in_ch
        for c in self.CH[:levels]:
            self.blocks.append(_conv(c_prev, c))
            c_prev = c

    def forward(self, img: torch.Tensor) -> list[torch.Tensor]:
        maps, x = [], img
        for i, blk in enumerate(self.blocks):
            if i > 0:
                x = F.max_pool2d(x, 2)
            x = blk(x)
            maps.append(x)
        return maps                              # [s1, s2, s4, s8, s16][:levels]


class GridDecoder(nn.Module):
    def __init__(self, feat_dim: int, n_classes: int = 4, width: int = 96, entry: int = 16, in_ch: int = 1):
        super().__init__()
        if entry not in (16, 8):
            raise ValueError(f"entry must be 16 or 8 px, got {entry}")
        self.entry = entry
        self.stem = ImageStem(in_ch=in_ch) if entry == 16 else ImageStem(in_ch=in_ch, levels=4)
        self.reduce = nn.Conv2d(feat_dim, width, 1)
        ch = ImageStem.CH
        stages = [
            _conv(width + ch[4], width),          # 1/16: grid + s16
            _conv(width + ch[3], width),          # 1/8
            _conv(width + ch[2], 64),             # 1/4
            _conv(64 + ch[1], 48),                # 1/2
            _conv(48 + ch[0], 32),                # 1/1
        ] if entry == 16 else [
            _conv(width + ch[3], width),          # 1/8: grid + s8
            _conv(width + ch[2], 64),             # 1/4
            _conv(64 + ch[1], 48),                # 1/2
            _conv(48 + ch[0], 32),                # 1/1
        ]
        self.up = nn.ModuleList(stages)
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, feats: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """feats (B, D, gh, gw) float on the `entry` px grid; img (B, in_ch, H, W) in [0, 1]."""
        maps = self.stem(img)
        x = self.reduce(feats)
        x = self.up[0](torch.cat([x, maps[-1]], 1))
        for i, m in zip(range(1, len(self.up)), reversed(maps[:-1])):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.up[i](torch.cat([x, m], 1))
        return self.head(x)
