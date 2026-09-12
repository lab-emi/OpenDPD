"""Synthetic PA data fixtures with *known* impairments.

Used by the test suite (tiny), by the performance protocol (standard/stress,
generated on demand) and by Dataset Doctor detector tests. Everything here is
clearly synthetic: a memory-polynomial PA with configurable delay, complex
gain, clipping and outliers. It never stands in for a measured PA in results.

    python -m tests.fixtures.synthetic --tier standard-1e6 --out /tmp/std
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

TIERS = {
    # name: paired complex samples
    "tiny": 20_000,
    "standard-1e6": 1_000_000,
    "standard-1e7": 10_000_000,
    "stress-1e8": 100_000_000,
}


@dataclass(frozen=True)
class Impairments:
    """Ground-truth impairments injected into a synthetic capture."""

    delay_samples: int = 0          # integer delay of output vs. input
    gain: complex = 1.0 + 0j        # linear complex gain applied to output
    clip_level: float | None = None  # |y| saturation level; None = no clipping
    n_outliers: int = 0             # isolated samples replaced by large spikes
    nan_indices: tuple[int, ...] = ()  # positions set to NaN in the output


def ofdm_like_input(n: int, fs: float, bandwidth: float, seed: int,
                    block: int = 4096) -> np.ndarray:
    """Band-limited complex Gaussian signal built block-wise in the frequency
    domain (memory-bounded; suitable for 1e8-sample stress tiers)."""
    rng = np.random.default_rng(seed)
    out = np.empty(n, dtype=np.complex64)
    freqs = np.fft.fftfreq(block, d=1 / fs)
    mask = np.abs(freqs) <= bandwidth / 2
    written = 0
    while written < n:
        spectrum = np.zeros(block, dtype=np.complex128)
        spectrum[mask] = rng.normal(size=mask.sum()) + 1j * rng.normal(size=mask.sum())
        chunk = np.fft.ifft(spectrum)
        chunk /= np.sqrt(np.mean(np.abs(chunk) ** 2)) * 4.0  # ~ -12 dBFS rms
        take = min(block, n - written)
        out[written:written + take] = chunk[:take]
        written += take
    return out


def memory_polynomial_pa(x: np.ndarray, memory: int = 3) -> np.ndarray:
    """Mild compressive PA with short memory (fixed coefficients)."""
    y = np.zeros_like(x, dtype=np.complex128)
    coeffs = {
        (1, 0): 1.0, (1, 1): 0.05 - 0.02j, (1, 2): -0.01j,
        (3, 0): -0.35 + 0.08j, (3, 1): -0.04, (5, 0): 0.05 - 0.01j,
    }
    xc = x.astype(np.complex128)
    for (order, lag), c in coeffs.items():
        if lag >= memory:
            continue
        shifted = np.roll(xc, lag)
        shifted[:lag] = 0
        y += c * shifted * np.abs(shifted) ** (order - 1)
    return y


def synthesize(n: int, seed: int = 0, fs: float = 800e6, bandwidth: float = 200e6,
               impairments: Impairments = Impairments()) -> tuple[np.ndarray, np.ndarray]:
    """Return (input_iq, output_iq) float32 arrays of shape (n, 2)."""
    x = ofdm_like_input(n, fs, bandwidth, seed)
    y = memory_polynomial_pa(x) * impairments.gain
    if impairments.delay_samples:
        y = np.roll(y, impairments.delay_samples)
        y[: impairments.delay_samples] = 0
    if impairments.clip_level is not None:
        mag = np.abs(y)
        over = mag > impairments.clip_level
        y[over] = y[over] / mag[over] * impairments.clip_level
    if impairments.n_outliers:
        rng = np.random.default_rng(seed + 1)
        idx = rng.choice(n, size=impairments.n_outliers, replace=False)
        y[idx] = 50.0 * np.exp(1j * rng.uniform(0, 2 * np.pi, size=idx.size))
    for i in impairments.nan_indices:
        y[i] = np.nan
    return (np.stack([x.real, x.imag], -1).astype(np.float32),
            np.stack([y.real, y.imag], -1).astype(np.float32))


def write_dataset(out_dir: Path, n: int, seed: int, fs: float, bandwidth: float,
                  impairments: Impairments, fmt: str = "npy") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y = synthesize(n, seed, fs, bandwidth, impairments)
    if fmt == "npy":
        np.save(out_dir / "input_iq.npy", x)
        np.save(out_dir / "output_iq.npy", y)
    elif fmt == "csv":
        import pandas as pd
        pd.DataFrame({"I_in": x[:, 0], "Q_in": x[:, 1], "I_out": y[:, 0], "Q_out": y[:, 1]}).to_csv(
            out_dir / "data.csv", index=False)
    else:
        raise ValueError(fmt)
    (out_dir / "fixture.json").write_text(json.dumps({
        "synthetic": True, "n_samples": n, "seed": seed, "sample_rate_hz": fs,
        "bandwidth_hz": bandwidth, "format": fmt,
        "impairments": {k: (str(v) if isinstance(v, complex) else v)
                        for k, v in asdict(impairments).items()},
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=sorted(TIERS), default="tiny")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", choices=["npy", "csv"], default="npy")
    args = parser.parse_args()
    write_dataset(args.out, TIERS[args.tier], args.seed, 800e6, 200e6, Impairments(), args.format)
    print(f"wrote {args.tier} fixture ({TIERS[args.tier]:,} samples) to {args.out}")


if __name__ == "__main__":
    main()
