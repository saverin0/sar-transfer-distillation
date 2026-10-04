# TerraFM model code, vendored copy

Copied and checked on 2026-09-29. The model is from the TerraFM paper by Muhammad Sohail
Danish and others, arXiv 2506.06281.

## Source

| what | value |
|---|---|
| repository | Hugging Face, MBZUAI/TerraFM |
| file | terrafm.py |
| pinned revision | 3631173fec2c3a445a71e15985e0f3d5aaf4ecb3 |
| URL | https://huggingface.co/MBZUAI/TerraFM/blob/3631173fec2c3a445a71e15985e0f3d5aaf4ecb3/terrafm.py |
| git blob id | 438cf763ad46ef089fd73f0e35dd7c1610876ead |
| same file elsewhere | GitHub mbzuai-oryx/TerraFM, branch master, commit e186ad85be486862cda99e992342e9afb175be02, same SHA-256 |

## Files in this folder

| file | bytes | SHA-256 | state |
|---|---|---|---|
| terrafm.py as released (not stored here) | 14,175 | f2d2a8bf03549261c74ed58754502768e532cd8e7b963cc2197c48bb92f4a0ad | original |
| terrafm.py in this folder | 15,113 | ccc0fcc19d370baa2b1794cbeddfcb03b47441991139e666c943cebb0136977e | CHANGED |
| LICENSE | 11,358 | cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30 | Apache License 2.0, text from https://www.apache.org/licenses/LICENSE-2.0.txt, unchanged |
| `__init__.py` | - | - | ours |
| NOTICE.md | - | - | ours |

The hashes are for files with LF line ends, as released.

## Changes

1. A note of 9 comment lines at the top of terrafm.py. It says that the file is changed.
   It stands between the marks `note on this copy (begin)` and `note on this copy (end)`.
2. ONE changed line in `TerraFM.prepare_tokens`, between the marks `changed line (begin)`
   and `changed line (end)`. The original line is kept there as a comment.

| | line |
|---|---|
| original | `x = self.patch_embed(x)  # patch linear embedding` |
| now | `x = self.patch_embed(x, is_l2a=getattr(self, "is_l2a", False))  # patch linear embedding` |

Why. `PatchEmbed.forward` has the switch `is_l2a` with the default False, and the released
`prepare_tokens` never passes it. So the released file sends every 12-band image through the
L1C input layer `conv2d_s2_l1c`. Our data is L2A. With the change the wrapper sets
`model.is_l2a` and so picks the input layer. A model without that attribute behaves as released.

Everything else is byte-identical. Taking the note out and putting the original line back
gives the SHA-256 of the original. `tests/test_teacher_terrafm.py` checks this.

## Left as found

- `warnings` is used in `_no_grad_trunc_normal_` but is never imported. That line runs only
  for a mean far outside the cut range, and the model never asks for one.
- `import random` is not used.
- `terrafm_large` is in the file, but no TerraFM-L weights are released.
- `B, nc, w, h = x.shape` names the two image sides the other way round. This has no effect
  on square images.

## Weights

The weights are not in this folder and not in the repository.

| what | value |
|---|---|
| file | TerraFM-B.pth, same repository and revision |
| bytes | 451,862,010 |
| SHA-256 | 8bdb6a4cfd707a09d79f4790119daa9d76bb316b6ea756c6eb305f99d1797a06 |
| content | 164 float32 tensors, 112,951,296 values, the names and shapes of `terrafm_base()`. No head, no decoder |

The content was read on 2026-09-29 from the list of names and shapes inside the file
(2 MB read over HTTP). The tensor data itself was not downloaded for this check.

## Licence choice

Three licence statements exist for TerraFM and they do not agree. The exact words, the URLs
and the date are in `docs/licences/terrafm-2026-09-29.md`.

| where | statement |
|---|---|
| Hugging Face model card, front matter | apache-2.0 |
| GitHub repository mbzuai-oryx/TerraFM | no licence file, no licence declared |
| author's answer in issue 5 of that repository | MIT for code and checkpoints, licence file promised, not there yet |

LICENSE in this folder is the Apache License 2.0. The reasons

1. The file was copied from the Hugging Face repository, and the card of that repository
   declares apache-2.0. It is the only licence statement attached to the place we copied from.
2. The file says at its top that it includes code copied and adapted from DINO. DINO
   (GitHub facebookresearch/dino, LICENSE at commit 7c446df5b9f45747937fb0d72314eb9f7b66930a)
   is under the Apache License 2.0. Its source files carry the line
   "Copyright (c) Facebook, Inc. and its affiliates." The TerraFM file does not repeat that
   line, so it is written down here.
3. The MIT statement is a comment in an issue. No MIT licence file with a copyright line
   exists, so there is no official MIT text that could be copied.
4. Apache 2.0 asks to keep the licence text and to mark changed files. This folder does
   both, so it keeps the Apache 2.0 terms.

What is missing for MIT. MIT asks that the copyright notice and the MIT text go with every
copy. This folder holds no MIT text and no copyright line of the TerraFM authors, because
none is published yet. When the authors publish their MIT file, that text with their
copyright line must be added here before anything is shared.

This is a reading of the pages, not legal advice. The three statements are read again before
anything is published.
