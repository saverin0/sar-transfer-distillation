"""PHI-S standardisation of teacher features (arXiv 2410.01680).

Read on 2026-09-30 in the HTML of arXiv 2410.01680, section 2.2.5 (PCA-Hadamard Isotropic
Standardization) and appendix A.1.1. With Sigma = U Lambda U^T the covariance of the teacher
features, H the Hadamard matrix of the feature width C with entries +-1 / sqrt(C), and
phi = sqrt(mean of the eigenvalues), the transform and its inverse are

    y' = alpha * H U^T (y - mu)          alpha = 1 / phi           (equation 25)
    y  = U H^T (y' / alpha) + mu

Every channel of y' has variance 1 (the rows of H all have entries of the same size, so
each output channel gets the mean of the eigenvalues, then alpha scales it to 1), and the
whole transform is one rotation and one scalar, so the MSE in the standardised space is
the MSE of the teacher space up to that scalar. Summary vectors are not standardised here,
they meet a cosine loss (student.py).

Hadamard matrix. Sylvester's doubling for a power of 2, Paley's constructions for the other
widths, as in appendix A.1.1 of the paper (768 = S(2) x P1(384), 1024 = S(1024), 1152 there
as S(32) x P2(36)). This file builds the matrix by the first rule that fits - Paley I when
C - 1 is a prime of the form 4k + 3, Paley II when C / 2 - 1 is a prime of the form 4k + 1,
else doubling - and checks H H^T = C I every time. Any Hadamard matrix gives the property
above, the construction is not part of the result. Prime powers that are not primes are
not handled, no teacher here needs one.

Fitting. Mean and covariance in float64 over the given feature vectors (N, C), the
covariance with N - 1. The eigenvectors come from `torch.linalg.eigh`, sorted by falling
eigenvalue. The fitted transform is kept in float32.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch

FORMAT = 1                 # version of the saved file
CHUNK = 65_536             # rows per pass over the features when fitting


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, math.isqrt(n) + 1):
        if n % d == 0:
            return False
    return True


def _jacobsthal(q: int) -> torch.Tensor:
    """(q, q) with Q[i, j] = chi(j - i), chi the quadratic character of the prime field q."""
    squares = {(a * a) % q for a in range(1, q)}
    chi = torch.tensor([0] + [1 if a in squares else -1 for a in range(1, q)], dtype=torch.float64)
    i = torch.arange(q)
    return chi[(i.view(1, q) - i.view(q, 1)) % q]


def _paley1(q: int) -> torch.Tensor:
    """Order q + 1 for a prime q = 4k + 3. H = I + S with S = [[0, 1], [-1, Q]] skew-symmetric."""
    s = torch.zeros(q + 1, q + 1, dtype=torch.float64)
    s[0, 1:] = 1.0
    s[1:, 0] = -1.0
    s[1:, 1:] = _jacobsthal(q)
    return torch.eye(q + 1, dtype=torch.float64) + s


def _paley2(q: int) -> torch.Tensor:
    """Order 2 (q + 1) for a prime q = 4k + 1. C = [[0, 1], [1, Q]] is a symmetric conference
    matrix, H = C x [[1, 1], [1, -1]] + I x [[1, -1], [-1, -1]]."""
    c = torch.zeros(q + 1, q + 1, dtype=torch.float64)
    c[0, 1:] = 1.0
    c[1:, 0] = 1.0
    c[1:, 1:] = _jacobsthal(q)
    a = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    b = torch.tensor([[1.0, -1.0], [-1.0, -1.0]], dtype=torch.float64)
    return torch.kron(c, a) + torch.kron(torch.eye(q + 1, dtype=torch.float64), b)


def hadamard(n: int) -> torch.Tensor:
    """(n, n) float64 with entries +-1 and H H^T = n I. ValueError when no rule here builds it."""
    if n < 1:
        raise ValueError(f"hadamard: order {n} is below 1")
    if n == 1:
        h = torch.ones(1, 1, dtype=torch.float64)
    elif n == 2:
        h = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    elif n % 4:
        raise ValueError(f"hadamard: no Hadamard matrix of order {n}, it must be 1, 2 or 4k")
    elif is_prime(n - 1) and (n - 1) % 4 == 3:
        h = _paley1(n - 1)
    elif is_prime(n // 2 - 1) and (n // 2 - 1) % 4 == 1:
        h = _paley2(n // 2 - 1)
    else:
        try:
            h = torch.kron(hadamard(2), hadamard(n // 2))
        except ValueError as e:
            raise ValueError(f"hadamard: no rule here builds order {n} ({e})") from e
    if not torch.equal(h.abs(), torch.ones(n, n, dtype=torch.float64)) or \
            not torch.allclose(h @ h.T, n * torch.eye(n, dtype=torch.float64)):
        raise RuntimeError(f"hadamard: the matrix of order {n} failed its check")
    return h


class PhiS:
    """The fitted transform of one teacher. Callable on (..., C), `inverse` goes back."""

    def __init__(self, mean: torch.Tensor, rotation: torch.Tensor, alpha: float,
                 eigenvalues: torch.Tensor | None = None, name: str = "", n_fit: int = 0):
        c = mean.numel()
        if mean.ndim != 1 or rotation.shape != (c, c):
            raise ValueError(f"phis: mean {tuple(mean.shape)} and rotation {tuple(rotation.shape)} "
                             "do not fit")
        if not (alpha > 0 and math.isfinite(alpha)):
            raise ValueError(f"phis: alpha must be a finite number above 0, got {alpha}")
        self.mean = mean.detach().to(torch.float32)
        self.rotation = rotation.detach().to(torch.float32)         # R = H U^T, orthogonal
        self.alpha = float(alpha)
        self.eigenvalues = None if eigenvalues is None else eigenvalues.detach().to(torch.float32)
        self.name = name
        self.n_fit = int(n_fit)

    @property
    def dim(self) -> int:
        return self.mean.numel()

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def to(self, device) -> "PhiS":
        self.mean = self.mean.to(device)
        self.rotation = self.rotation.to(device)
        if self.eigenvalues is not None:
            self.eigenvalues = self.eigenvalues.to(device)
        return self

    @classmethod
    def fit(cls, features: torch.Tensor, name: str = "") -> "PhiS":
        """Fit on feature vectors (N, C), any float dtype, on their device. N must be above 1."""
        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError(f"phis: expected (N, C) with N above 1, got {tuple(features.shape)}")
        if not torch.is_floating_point(features):
            raise ValueError(f"phis: expected float features, got {features.dtype}")
        n, c = features.shape
        mean = torch.zeros(c, dtype=torch.float64, device=features.device)
        for chunk in features.split(CHUNK):
            mean += chunk.to(torch.float64).sum(0)
        mean /= n
        cov = torch.zeros(c, c, dtype=torch.float64, device=features.device)
        for chunk in features.split(CHUNK):
            centred = chunk.to(torch.float64) - mean
            cov += centred.T @ centred
        cov /= n - 1
        if not torch.isfinite(cov).all():
            raise ValueError("phis: the features hold NaN or infinite values")
        evals, evecs = torch.linalg.eigh(cov)                 # ascending
        evals, evecs = evals.flip(0), evecs.flip(1)           # falling, U[:, i] to evals[i]
        trace = float(evals.sum())
        if trace <= 0:
            raise ValueError("phis: the features have no variance")
        h = hadamard(c).to(features.device) / math.sqrt(c)
        rotation = h @ evecs.T
        alpha = math.sqrt(c / trace)
        return cls(mean.float(), rotation.float(), alpha, evals.float(), name=name, n_fit=n)

    def _check(self, x: torch.Tensor) -> None:
        if x.shape[-1] != self.dim:
            raise ValueError(f"phis {self.name or ''}: expected (..., {self.dim}), got {tuple(x.shape)}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Teacher features (..., C) to the standardised space, in the dtype of `x`."""
        self._check(x)
        d = x.dtype if torch.is_floating_point(x) else torch.float32
        return ((x.to(d) - self.mean.to(d)) @ self.rotation.to(d).T) * self.alpha

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Standardised features (..., C) back to the teacher space."""
        self._check(y)
        d = y.dtype if torch.is_floating_point(y) else torch.float32
        return (y.to(d) / self.alpha) @ self.rotation.to(d) + self.mean.to(d)

    def check(self, features: torch.Tensor, tol: float = 0.01) -> torch.Tensor:
        """Variance per channel of the transformed features (N, C), all within `tol` of 1, else
        ValueError. On the features that were fitted this is a check of the construction."""
        if features.ndim != 2:
            raise ValueError(f"phis: expected (N, C), got {tuple(features.shape)}")
        var = torch.zeros(self.dim, dtype=torch.float64, device=features.device)
        mean = torch.zeros(self.dim, dtype=torch.float64, device=features.device)
        for chunk in features.split(CHUNK):
            y = self(chunk.to(torch.float64))
            mean += y.sum(0)
            var += (y * y).sum(0)
        n = features.shape[0]
        mean /= n
        var = (var - n * mean * mean) / (n - 1)
        off = float((var - 1).abs().max())
        if not off <= tol:
            raise ValueError(f"phis {self.name or ''}: a channel variance is off by {off:.4f} "
                             f"(tolerance {tol}), the features are not the fitted ones")
        return var.float()

    def state(self) -> dict:
        return {"format": FORMAT, "kind": "phis", "name": self.name, "dim": self.dim,
                "n_fit": self.n_fit, "alpha": self.alpha, "mean": self.mean.cpu(),
                "rotation": self.rotation.cpu(),
                "eigenvalues": None if self.eigenvalues is None else self.eigenvalues.cpu()}

    def save(self, path: str | os.PathLike) -> None:
        """torch file, written through a temporary name so a cut write leaves no half file."""
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(self.state(), tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | os.PathLike, device="cpu") -> "PhiS":
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or state.get("kind") != "phis":
            raise ValueError(f"{path} is not a saved PhiS")
        if state.get("format") != FORMAT:
            raise ValueError(f"{path} has format {state.get('format')}, this code reads {FORMAT}")
        out = cls(state["mean"], state["rotation"], state["alpha"], state["eigenvalues"],
                  name=state["name"], n_fit=state["n_fit"])
        return out.to(device)

    def __repr__(self) -> str:
        return f"PhiS({self.name!r}, dim={self.dim}, alpha={self.alpha:.4g}, n_fit={self.n_fit})"
