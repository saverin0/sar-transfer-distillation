"""One interface for every frozen teacher.

Every teacher receives the SAME input: `s2`, a float32 tensor (B, 12, H, W) holding the
Sentinel-2 L2A digital numbers of `records.S2_BANDS`, all bands already on the 10 m grid,
the +1000 offset already handled. H = W = the student's crop (256 px).

`prepare` turns that into what the model was trained on: it picks the bands the model
reads, applies ONE named scaling rule and resizes to `input_size`. The candidate rules are
listed in `rules`; stage 0 draws the features under each of them, one is then fixed in
. A rule is never guessed silently: every rule names its source in `rule_sources`.

`forward_features` returns a summary vector and the patch grid, both float32:

    summary  (B, dim)
    spatial  (B, grid, grid, dim)        grid = input_size // patch

Part four adds two more ways in, both optional per teacher
- `aerial(x)` for a teacher that reads RGB pictures (`reads_aerial`): x is a float tensor
  (B, 4, H, W) or (B, 3, H, W) of 8-bit values, the first three channels are RGB; they are
  scaled by 1/255, resized to `input_size` and normalised with `aerial_mean` and
  `aerial_std`, then `_forward` runs as for Sentinel-2. DINOv3, SigLIP2 and PE read aerial.
- `radar(s1)` for a teacher that reads Sentinel-1 itself (`reads_radar`): s1 is a float
  tensor (B, 2, H, W) of VV and VH in dB; `prepare_radar` turns it into the model input.
  TerraFM-B through its Sentinel-1 input layer is the one radar teacher of the pilot.
A radar teacher does not read Sentinel-2 (`reads_s2` is False), so calling it with s2 fails.

Teachers are frozen and run under `torch.inference_mode()`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..records import S2_BANDS


@dataclass
class TeacherOut:
    summary: torch.Tensor      # (B, dim) float32
    spatial: torch.Tensor      # (B, grid, grid, dim) float32


def band_index(names: tuple[str, ...] | list[str]) -> list[int]:
    """Positions of `names` inside the 12-band input."""
    return [S2_BANDS.index(n) for n in names]


class Teacher:
    """Base class. A subclass sets the attributes and writes `prepare` and `_forward`."""

    name: str = ""
    dim: int = 0
    patch: int = 16
    input_size: int = 256
    bands: tuple[str, ...] = ()            # the Sentinel-2 bands the model reads, in its order
    rules: tuple[str, ...] = ()            # names of the candidate input scaling rules
    rule_sources: dict[str, str] = {}      # rule name -> where the numbers come from
    licence: str = ""                      # as found, with the date it was read
    reads_s2: bool = True                  # reads the Sentinel-2 crop (slots 1 to 3)
    reads_aerial: bool = False             # reads aerial RGB through `aerial`
    reads_radar: bool = False              # reads the Sentinel-1 crop through `radar`
    aerial_mean: tuple[float, float, float] | None = None    # RGB in 0-1, for `aerial`
    aerial_std: tuple[float, float, float] | None = None

    def __init__(self, model: torch.nn.Module, rule: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        if rule not in self.rules:
            raise ValueError(f"{self.name}: unknown rule {rule!r}, known: {self.rules}")
        self.rule = rule
        self.device = device
        self.dtype = dtype
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)

    @property
    def grid(self) -> int:
        return self.input_size // self.patch

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def prepare(self, s2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _forward(self, x: torch.Tensor) -> TeacherOut:
        raise NotImplementedError

    def check_input(self, s2: torch.Tensor) -> None:
        if s2.ndim != 4 or s2.shape[1] != len(S2_BANDS):
            raise ValueError(f"{self.name}: expected (B, {len(S2_BANDS)}, H, W), got {tuple(s2.shape)}")
        if not torch.is_floating_point(s2):
            raise ValueError(f"{self.name}: expected a float tensor of digital numbers, got {s2.dtype}")

    def _run(self, x: torch.Tensor) -> TeacherOut:
        """`_forward` under autocast on CUDA, the shapes checked, float32 out."""
        if self.device.startswith("cuda") and self.dtype != torch.float32:
            with torch.autocast("cuda", dtype=self.dtype):
                out = self._forward(x)
        else:
            out = self._forward(x)
        g, d = self.grid, self.dim
        if out.summary.shape[1:] != (d,) or out.spatial.shape[1:] != (g, g, d):
            raise RuntimeError(f"{self.name}: got summary {tuple(out.summary.shape)} and spatial "
                               f"{tuple(out.spatial.shape)}, expected (B, {d}) and (B, {g}, {g}, {d})")
        return TeacherOut(out.summary.float(), out.spatial.float())

    @torch.inference_mode()
    def __call__(self, s2: torch.Tensor) -> TeacherOut:
        if not self.reads_s2:
            raise TypeError(f"{self.name} does not read Sentinel-2, call radar(s1) instead")
        self.check_input(s2)
        return self._run(self.prepare(s2.to(self.device, non_blocking=True).float()))

    def prepare_aerial(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3 or 4, H, W) float 8-bit values to the model's normalised RGB at `input_size`."""
        if self.aerial_mean is None or self.aerial_std is None:
            raise NotImplementedError(f"{self.name}: aerial_mean and aerial_std are not set")
        if x.ndim != 4 or x.shape[1] not in (3, 4):
            raise ValueError(f"{self.name} aerial: expected (B, 3 or 4, H, W), got {tuple(x.shape)}")
        if not torch.is_floating_point(x) or x.min() < 0 or x.max() > 255:
            raise ValueError(f"{self.name} aerial: expected float 8-bit values in 0 to 255")
        rgb = x[:, :3].to(self.device, non_blocking=True).float() / 255.0
        size = (self.input_size, self.input_size)
        if tuple(rgb.shape[-2:]) != size:
            rgb = torch.nn.functional.interpolate(rgb, size=size, mode="bilinear", antialias=True,
                                                  align_corners=False)
        mean = torch.tensor(self.aerial_mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.aerial_std, device=self.device).view(1, 3, 1, 1)
        return (rgb - mean) / std

    @torch.inference_mode()
    def aerial(self, x: torch.Tensor) -> TeacherOut:
        """The teacher on aerial RGB 8-bit values (B, 4 or 3, H, W), NIR dropped."""
        if not self.reads_aerial:
            raise TypeError(f"{self.name} gives no aerial target")
        return self._run(self.prepare_aerial(x))

    def prepare_radar(self, s1: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def radar(self, s1: torch.Tensor) -> TeacherOut:
        """The teacher on the Sentinel-1 crop (B, 2, H, W), VV and VH in dB, float."""
        if not self.reads_radar:
            raise TypeError(f"{self.name} does not read Sentinel-1")
        if s1.ndim != 4 or s1.shape[1] != 2 or not torch.is_floating_point(s1):
            raise ValueError(f"{self.name} radar: expected a float tensor (B, 2, H, W) in dB, got "
                             f"{tuple(s1.shape)} {s1.dtype}")
        return self._run(self.prepare_radar(s1.to(self.device, non_blocking=True).float()))
