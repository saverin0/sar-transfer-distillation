"""Linear probe on frozen patch features, copied from part two (only `train_linear_probe`).

One linear layer on the patch vectors, so the head adds as little capacity as possible and
the number measures the encoder, not the head. `domains.run` calls it with `balanced=True`
and `verbose=False`; the verbose line prints the class weights by class id.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def train_linear_probe(X: np.ndarray, y: np.ndarray, *, balanced: bool = False,
                       n_classes: int = 3,
                       epochs: int = 30, lr: float = 1e-3, batch: int = 65536,
                       device: str = "cuda", seed: int = 0, verbose: bool = True) -> nn.Linear:
    """Fit a single linear layer on patch vectors.

    `balanced` weights the loss by inverse class frequency.
    """
    torch.manual_seed(seed)
    n, dim = X.shape
    probe = nn.Linear(dim, n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    if balanced:
        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        w = counts.sum() / (n_classes * np.maximum(counts, 1))
        weight = torch.tensor(w, dtype=torch.float32, device=device)
        if verbose:
            print("class weights:", dict(enumerate(w.round(3))))
    else:
        weight = None
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.astype(np.int64))
    # Speed-up (2026-09-24): put the whole training set on the GPU once instead of
    # gathering every batch on the CPU. ~2 M x 1280 patches fit easily in 80 GB.
    # The shuffle order is still drawn on the CPU, so batches are identical to before.
    on_gpu = str(device).startswith("cuda")
    if on_gpu:
        try:
            Xt, yt = Xt.to(device), yt.to(device)
        except torch.cuda.OutOfMemoryError:
            on_gpu = False
            torch.cuda.empty_cache()
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            if on_gpu:
                idx = idx.to(device)
            xb = Xt[idx].to(device, non_blocking=True).float()
            yb = yt[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(probe(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:>3}  loss {total/n:.4f}")
    return probe
