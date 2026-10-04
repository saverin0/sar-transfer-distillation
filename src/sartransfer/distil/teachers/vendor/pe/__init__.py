"""Perception Encoder vision tower, copied from github.com/facebookresearch/perception_models
(core/vision_encoder at commit 3e352cca660658d4b5c90f42a7808b11469e4c66, read 2026-10-03).

`pe.py`, `rope.py` and `config.py` are CHANGED copies. They import torch and the standard
library only; the text tower, the CLIP wrapper and every download are gone, and the einops
and timm calls are written in torch. Each change is marked in place. Source, hashes, changes
and the licence are in NOTICE.md, the licence text (LICENSE.PE of the repository) is in LICENSE.
"""

from .config import PE_VISION_CONFIG, PEConfig
from .pe import VisionTransformer

__all__ = ["PE_VISION_CONFIG", "PEConfig", "VisionTransformer"]
