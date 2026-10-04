# AnyUp -- vendored copy

Source - https://github.com/wimmerth/anyup, commit `351807a9c4287368732cc247f26c7c81c9139af4`, fetched 2026-09-25.
Licence - Creative Commons Attribution 4.0 International (see LICENSE in this folder).
Paper - T. Wimmer et al., "AnyUp, Universal Feature Upsampling", ICLR 2026, arXiv 2510.12764.

Only the inference code is kept (model + layers + two utils). Training code,
the NATTEN attention variant, backbones, dataloaders and visualisation are not
included. Files are unmodified. Weights (`anyup_multi_backbone.pth`, 3.5 MB,
released by the authors at
https://github.com/wimmerth/anyup/releases/download/checkpoint_v2/anyup_multi_backbone.pth)
are kept as a file on Drive, checked by size and full SHA-256 and loaded from there
(models/anyup_loader.py) -- never downloaded by the code, no torch.hub.

Changes - none to the code. This folder exists so the project has no runtime
dependency on the upstream repository.
