# Perception Encoder vision tower, vendored copy

Copied and checked on 2026-10-03. The model is from the paper "Perception Encoder - The best
visual embeddings are not at the output of the network" by Daniel Bolya and others,
arXiv 2504.13181, Meta.

## Source

| what | value |
|---|---|
| repository | GitHub, facebookresearch/perception_models |
| folder | core/vision_encoder |
| pinned commit | 3e352cca660658d4b5c90f42a7808b11469e4c66 (head of main on 2026-10-03, committed 2025-12-18, "Updated README for PE-AV to add new indicators.") |
| files | https://github.com/facebookresearch/perception_models/blob/3e352cca660658d4b5c90f42a7808b11469e4c66/core/vision_encoder/pe.py and rope.py and config.py, LICENSE.PE at the repository root |
| read as | the raw files, https://raw.githubusercontent.com/facebookresearch/perception_models/3e352cca660658d4b5c90f42a7808b11469e4c66/core/vision_encoder/pe.py and so on |

The three copied files are the whole of what builds the vision transformer and runs its
forward. `tokenizer.py`, `transforms.py` (torchvision and the tokenizer, the picture and text
transforms) and `bpe_simple_vocab_16e6.txt.gz` of the same folder are not copied; the picture
normalisation of `transforms.py` is typed into the teacher module with its line.

## Files in this folder

| file | bytes | SHA-256 | state |
|---|---|---|---|
| pe.py as released (not stored here) | 24,000 | 36b62204c4cf12a709946facce2df8980e0880bf06deb6e49a2aa288158c3d53 | original |
| pe.py in this folder | 19,650 | f8beda5f7e028a8886d9ad0eebc2a37f210f999cf84157389e83e203a76337b1 | CHANGED |
| rope.py as released (not stored here) | 10,507 | 85a16113138f26389ea35fccd313f2aca12945c7ed2eb24e5d52dbe680dad121 | original |
| rope.py in this folder | 13,240 | e6113145c0b54af6c86ca79997dd5c5084a6cf85b30fedc171d72634b2490388 | CHANGED |
| config.py as released (not stored here) | 5,175 | 37c10ea7b68889241c8d9ebce27c62577c7379a33c444f9b0129bc90946aaff5 | original |
| config.py in this folder | 5,766 | dfd808836c6a4a3e224ec597058aad4f33eb222ef79f9486936ac9296dd0641a | CHANGED |
| LICENSE | 11,357 | c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4 | LICENSE.PE of the repository, Apache License 2.0, unchanged |
| `__init__.py` | - | - | ours |
| NOTICE.md | - | - | ours |

The hashes are for files with LF line ends, as released. The copies import torch and the
Python standard library only. `tests/test_teacher_pe.py` reads the import lines, checks the
hashes above and, with network, rebuilds the originals from the marks.

## How a change is marked

Every change stands between a begin mark and an end mark that name "sar-transfer part four".

- `note on this copy` - the note at the top of each file.
- `changed lines` - the original lines are kept as comments, one per line, each starting with
  `# original - ` at the indentation of the block (the rest of the line follows as released,
  so a whitespace-only original line leaves trailing spaces after the prefix, and an empty
  original line is `# original -` with nothing after it). The new lines follow the comments.
- `removed block` - the two marks name the original line numbers that are gone (1-based,
  both ends included) and what stood there. The text itself is not kept.

Taking the notes out, putting the original lines back and the removed ranges back in gives
the released files, except that pe.py as released has no final newline and the copy has one.

## Changes

pe.py, original line numbers

| lines | kind | what and why |
|---|---|---|
| 10-14 | changed | `import numpy as np`, `from einops import rearrange` and `from timm.layers import DropPath` are gone, the torch imports between them stay. numpy was used by the CLIP wrapper only, einops and timm are replaced below. The copy needs torch and the standard library only |
| 21-22 | changed | `from core.vision_encoder.rope import Rope2D` is `from .rope import Rope2D`, and the config import takes `PEConfig` and `PE_VISION_CONFIG` from `.config` only (no text config, no `fetch_pe_checkpoint`) |
| 138-140 | changed | `rearrange(q, "b s (h d) -> b h s d", h=...)` for q, k and v is `q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)` |
| 148 | changed | `rearrange(attn, "b h s d -> b s (h d)")` is `attn.transpose(1, 2).reshape(batch, seq, embed_dim)` |
| 186-187 | changed | `DropPath(drop_path) if drop_path > 0.0 else nn.Identity()` for the two drop paths is `nn.Identity()`, and a `drop_path > 0.0` raises NotImplementedError. Drop path (stochastic depth) acts in training only; every released config has drop_path 0.0 (`PEConfig` default), and this copy runs frozen models |
| 407-425 | removed | `VisionTransformer.load_ckpt`, which loads with `strict=False` and prints the key report. The teacher module loads with its own strict check |
| 434-452 | changed | `VisionTransformer.from_config(name, pretrained, checkpoint_path, **kwdargs)` is `from_config(name, **kwdargs)` - the same config lookup and `cls(**args)`, without the `pretrained` download through `fetch_pe_checkpoint` |
| 552-761 | removed | `TextTransformer` and `CLIP`, the text tower and the CLIP wrapper |
| end | added | a final newline (the released file ends without one) |

rope.py, original line numbers. Each einops call is written as the torch call with the same
result.

| line | original | now |
|---|---|---|
| 5 | `from einops import rearrange, repeat` | gone |
| 33 | `x = rearrange(x, "... (d r) -> ... d r", r=2)` | `x = x.unflatten(-1, (-1, 2))` |
| 36 | `return rearrange(x, "... d r -> ... (d r)")` | `return x.flatten(-2)` |
| 71 | `rotations = rearrange(rotations, "... r f -> ... (r f)")` | `rotations = rotations.flatten(-2)` |
| 73 | `rotations = repeat(rotations, "... n -> ... (n r)", r=2)` | `rotations = rotations.repeat_interleave(2, dim=-1)` |
| 186 | `freqs = rearrange(freqs, "n d -> n 1 d")` | `freqs = freqs.unsqueeze(1)` |
| 218 | `freqs = rearrange(freqs, "n d -> n 1 d")` | `freqs = freqs.unsqueeze(1)` |
| 219 | `scale = rearrange(scale, "n d -> n 1 d")` | `scale = scale.unsqueeze(1)` |
| 244 | `scale = self.scale ** rearrange(power, "n -> n 1")` | `scale = self.scale ** power.unsqueeze(1)` |
| 292 | `freqs = repeat(freqs, "... n -> ... (n r)", r=2)` | `freqs = freqs.repeat_interleave(2, dim=-1)` |

Ten changed lines in all. Only lines 33, 36 and 292 run for the spatial teacher
(`rotate_half` and `RotaryEmbedding.forward`, called through `Rope2D`); the others are in
methods that this project does not call and were changed so that the file imports nothing
but torch. The first version of this copy missed line 218 (the same text as line 186, the
generator matched the first one only); `tests/test_teacher_pe.py` found the leftover
`rearrange` call on 2026-10-03 and the line was changed the same way.

config.py, original line numbers

| lines | kind | what and why |
|---|---|---|
| 12 | changed | `from huggingface_hub import hf_hub_download` is gone |
| 16-26 | removed | `fetch_pe_checkpoint`, the checkpoint download |

## Left as found

- pe.py imports `copy`, `math` and `random` and `xavier_normal_` without using them in the
  kept part. `from torch.utils.checkpoint import checkpoint` serves gradient checkpointing,
  a training path.
- `PEConfig.image_size` is written `image_size: int = 224,` with a trailing comma, so its
  default is the tuple `(224,)`. Every named config sets image_size, the default is never used.
- `PETextConfig` and `PE_TEXT_CONFIG` stay in config.py. They are data for the text tower,
  whose code is not here.
- `Rope2D` is a plain class, not an `nn.Module`. Its `RotaryEmbedding` holds one parameter,
  `freqs` (16 values from theta 10000, requires_grad False), that is NOT in the state dict and
  is NOT moved by `model.to(device)`; `Rope2D.update_grid` moves it on the first forward and
  whenever the input grid changes, and the frequency table is cached per grid size.
- `RotaryEmbedding.forward` and `apply_rotary_emb` carry `@autocast("cuda", enabled=False)`
  from torch.amp, so RoPE runs in float32 under bfloat16 autocast. On a CPU-only torch the
  decorator is silent (checked with torch 2.14.0+cpu).
- `VisionTransformer.__init__` draws random values for every parameter (`init_tensors`), so
  building the real architecture takes a moment before any file is loaded.
- `VisionTransformer._sample_abs_posemb` interpolates the absolute position embedding
  (bilinear, align_corners False, the class row kept as is) whenever the input grid differs
  from the grid the model was built for. The teacher module relies on it for 224 px inputs.
- pe.py and rope.py carry no copyright line; config.py carries
  `# Copyright (c) Meta Platforms, Inc. and affiliates.`; the appendix of LICENSE.PE is the
  unfilled template (`Copyright [yyyy] [name of copyright owner]`).

## Weights

The weights are not in this folder and not in the repository.

| what | value |
|---|---|
| repository | Hugging Face, facebook/PE-Spatial-L14-448, revision a09b69ff95973d70edd96e6fa4be216d8c6caa70 (last change 2025-07-12), not gated |
| file | PE-Spatial-L14-448.pt |
| bytes | 1,215,952,299 |
| SHA-256 | 47fc1657db08e44f8202b4c1190680a86bbb18a9e2f4252a2f62d4a2d4ba06b1 (the LFS record of the file list and the hash of the downloaded file agree) |
| content | 293 float32 tensors, 303,964,160 values, the names and shapes of `VisionTransformer.from_config("PE-Spatial-L14-448")`, no prefix, no wrapper, no head |
| config.yaml of that repository | an empty file (0 bytes, the empty git blob e69de29b). The architecture comes from config.py here |

## Licence

| where | statement |
|---|---|
| Hugging Face model card, front matter | `license: apache-2.0` |
| GitHub repository, LICENSE.PE | the Apache License 2.0 text (this folder's LICENSE) |
| GitHub repository, apps/pe/README.md | "All checkpoints released on this page, unless otherwise specified, are released with the Apache 2.0 license. The code itself is licensed under the parent license of this repository." |

The exact words, the URLs and the date are in `docs/licences/pe-2026-10-03.md`. Apache 2.0
asks to keep the licence text and to mark changed files. This folder does both.
This is a reading of the pages, not legal advice.
