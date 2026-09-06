"""Memory-polynomial (MP) and generalised memory-polynomial (GMP) models fitted by least squares.

The classical DPD baselines of the OpenDPD benchmark (``benchmark/benchmark_volterra.py``) as first-class
models of the compute core: the same basis definitions, a column-normalised least-squares solve with an
explicit singular-value cutoff (truncated SVD), and a torch module that applies the fitted coefficients with
the segment semantics every other model is evaluated under (delays reset at the start of each segment).

Fitting is deterministic: no seed, no epochs. Its stability is reported, not assumed (rank, condition number,
cutoff, residual) so a broken baseline cannot manufacture a gain for a learned model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

POLYNOMIAL_KEYS = ("mp_ls", "gmp_ls")
MEMORY_BUDGET_BYTES = 3 << 30        # the full basis is held in memory for the solve (complex128)


def _delay(x: np.ndarray, delay: int) -> np.ndarray:
    """Shift a 1-D complex signal by ``delay`` samples (positive = past), zero-filling the boundary."""
    out = np.zeros_like(x)
    n = len(x)
    if 0 <= delay < n:
        out[delay:] = x[:n - delay]
    elif -n < delay < 0:
        out[:n + delay] = x[-delay:]
    return out


def mp_basis(x: np.ndarray, K: int, Q: int) -> np.ndarray:
    """Columns ``x(n-q) |x(n-q)|^k`` for k in 0..K-1, q in 0..Q-1 (column index k*Q + q)."""
    x = np.asarray(x, dtype=np.complex128)
    phi = np.empty((len(x), K * Q), dtype=np.complex128)
    for k in range(K):
        for q in range(Q):
            d = _delay(x, q)
            phi[:, k * Q + q] = d * np.abs(d) ** k
    return phi


def gmp_basis(x: np.ndarray, Ka: int, La: int, Kb: int, Lb: int, Mb: int, Kc: int, Lc: int, Mc: int) -> np.ndarray:
    """Aligned (``x(n-l)|x(n-l)|^k``), lagging (``x(n-l)|x(n-l-m)|^k``) and leading (``x(n-l)|x(n-l+m)|^k``) terms."""
    x = np.asarray(x, dtype=np.complex128)
    columns = []
    for k in range(Ka):
        for l in range(La):
            d = _delay(x, l)
            columns.append(d * np.abs(d) ** k)
    for k in range(1, Kb + 1):
        for l in range(Lb):
            for m in range(1, Mb + 1):
                columns.append(_delay(x, l) * np.abs(_delay(x, l + m)) ** k)
    for k in range(1, Kc + 1):
        for l in range(Lc):
            for m in range(1, Mc + 1):
                columns.append(_delay(x, l) * np.abs(_delay(x, l - m)) ** k)
    return np.stack(columns, axis=1)


def coefficient_count(key: str, params: Mapping[str, object]) -> int:
    p = {k: int(v) for k, v in params.items() if k != "rcond"}
    if key == "mp_ls":
        return p["K"] * p["Q"]
    if key == "gmp_ls":
        return p["Ka"] * p["La"] + p["Kb"] * p["Lb"] * p["Mb"] + p["Kc"] * p["Lc"] * p["Mc"]
    raise KeyError(key)


def lookahead_samples(key: str, params: Mapping[str, object]) -> int:
    """Future samples a model reads: only the leading GMP envelope terms do."""
    return int(params.get("Mc", 0)) if key == "gmp_ls" and int(params.get("Kc", 0)) > 0 else 0


def context_samples(key: str, params: Mapping[str, object]) -> int:
    """Past samples a model reads (its memory): what a split guard must cover."""
    p = {k: int(v) for k, v in params.items() if k != "rcond"}
    if key == "mp_ls":
        return p["Q"]
    if key == "gmp_ls":
        return max(p["La"], p["Lb"] + p["Mb"] if p["Kb"] > 0 else 0, p["Lc"])
    raise KeyError(key)


def basis(key: str, params: Mapping[str, object], x: np.ndarray) -> np.ndarray:
    p = {k: int(v) for k, v in params.items() if k != "rcond"}
    if key == "mp_ls":
        return mp_basis(x, p["K"], p["Q"])
    if key == "gmp_ls":
        return gmp_basis(x, **p)
    raise KeyError(key)


def segmented_basis(key: str, params: Mapping[str, object], x: np.ndarray, segment_length: int) -> np.ndarray:
    """The basis with delays reset at every segment start: how the models are evaluated (IQSegmentDataset)."""
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    x = np.asarray(x, dtype=np.complex128)
    blocks = [basis(key, params, x[s:s + segment_length]) for s in range(0, len(x), segment_length)]
    return np.concatenate(blocks, axis=0)


@dataclass(frozen=True)
class FitDiagnostics:
    """What the least-squares solve did; recorded with every baseline result."""

    solver: str                       # column-normalised least squares, singular-value cutoff = rcond
    n_observations: int
    n_coefficients: int
    rank: int                         # singular directions retained
    rcond: float                      # cutoff relative to the largest singular value (0 = machine precision)
    condition_number: float           # s_max / s_min of the normalised basis (all directions)
    retained_condition_number: float  # s_max / smallest retained singular value
    column_norm_ratio: float          # max / min column norm before normalisation
    train_nmse_db: float              # residual on the fitted split

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def fit_least_squares(phi: np.ndarray, target: np.ndarray, rcond: float = 0.0) -> Tuple[np.ndarray, FitDiagnostics]:
    """Solve ``phi w ~= target`` after L2 column scaling. ``rcond`` truncates singular values below
    ``rcond * s_max`` (NumPy ``lstsq`` semantics); 0 means machine precision."""
    phi = np.asarray(phi, dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128)
    if phi.ndim != 2 or target.ndim != 1 or phi.shape[0] != target.shape[0]:
        raise ValueError("basis must be (n, p) and target (n,) with matching n")
    if not (np.all(np.isfinite(phi)) and np.all(np.isfinite(target))):
        raise ValueError("least-squares input and target must be finite")
    n, p = phi.shape
    if n < p:
        raise ValueError(f"least-squares system is underdetermined: {n} observations for {p} coefficients")
    if not 0.0 <= rcond < 1.0:
        raise ValueError("rcond must be in [0, 1)")
    norms = np.linalg.norm(phi, axis=0)
    if np.any(norms == 0):
        raise ValueError(f"basis contains {int(np.sum(norms == 0))} all-zero columns")
    cutoff = rcond if rcond > 0 else None
    w_n, _, rank, s = np.linalg.lstsq(phi / norms, target, rcond=cutoff)
    w = w_n / norms
    residual = target - phi @ w
    energy = float(np.sum(np.abs(target) ** 2))
    nmse = 10 * np.log10(float(np.sum(np.abs(residual) ** 2)) / energy) if energy > 0 else float("nan")
    s_max = float(s[0])
    kept = s[:rank] if rank > 0 else s[:1]
    diag = FitDiagnostics(solver="column-normalised least squares (SVD, numpy.linalg.lstsq)",
                          n_observations=int(n), n_coefficients=int(p), rank=int(rank), rcond=float(rcond),
                          condition_number=float(s_max / s[-1]) if s[-1] > 0 else float("inf"),
                          retained_condition_number=float(s_max / kept[-1]) if kept[-1] > 0 else float("inf"),
                          column_norm_ratio=float(norms.max() / norms.min()), train_nmse_db=float(nmse))
    return w, diag


def to_complex(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq)
    return iq[..., 0].astype(np.complex128) + 1j * iq[..., 1].astype(np.complex128)


def basis_bytes(key: str, params: Mapping[str, object], n_samples: int) -> int:
    return int(n_samples) * coefficient_count(key, params) * 16


class PolynomialModel:
    """Torch module applying fitted coefficients; built lazily so the core stays importable without torch."""

    def __new__(cls, key: str, params: Mapping[str, object], coefficients: Optional[np.ndarray] = None):
        import torch
        from torch import nn

        class _Polynomial(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.key = key
                self.params = {k: (float(v) if k == "rcond" else int(v)) for k, v in params.items()}
                p = coefficient_count(key, self.params)
                init = torch.zeros(p, dtype=torch.complex128) if coefficients is None \
                    else torch.as_tensor(np.asarray(coefficients, dtype=np.complex128))
                if init.numel() != p:
                    raise ValueError(f"{p} coefficients expected, got {init.numel()}")
                self.register_buffer("coefficients", init)

            @property
            def n_real_parameters(self) -> int:
                return 2 * int(self.coefficients.numel())

            def forward(self, x, h_0=None):
                # x: (batch, frame, 2) float; every frame is one segment (delays reset), like the legacy loaders
                xc = torch.complex(x[..., 0].to(torch.float64), x[..., 1].to(torch.float64)).cpu().numpy()
                out = np.empty_like(xc)
                w = self.coefficients.cpu().numpy()
                for i in range(xc.shape[0]):
                    out[i] = basis(self.key, self.params, xc[i]) @ w
                y = torch.from_numpy(np.stack([out.real, out.imag], axis=-1)).to(dtype=x.dtype, device=x.device)
                return y

        return _Polynomial()
