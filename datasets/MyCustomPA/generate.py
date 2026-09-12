"""Reproduce MyCustomPA: dummy dataset for tutorial purpose.

No measured data are used. Run from the repository root:
    python -m datasets.MyCustomPA.generate
"""

import json
from pathlib import Path

import numpy as np

from opendpd.core.splits import DEFAULT_RATIOS, contiguous_boundaries


def generate():
    rng = np.random.default_rng(20260912)
    nfft, frames, active = 2560, 40, 64
    levels = np.arange(-7, 8, 2)
    grid = (levels[:, None] + 1j * levels[None, :]).ravel() / np.sqrt(42)
    spectrum = np.zeros((frames, nfft), dtype=np.complex128)
    bins = np.r_[np.arange(nfft // 2 - active // 2, nfft // 2),
                 np.arange(nfft // 2 + 1, nfft // 2 + active // 2 + 1)]
    for frame in range(frames):
        spectrum[frame, bins] = rng.permutation(grid)
    x = (np.fft.ifft(np.fft.ifftshift(spectrum, axes=1), axis=1) * nfft / np.sqrt(active) * 0.2).ravel()
    # Illustrative memoryless compression and amplitude-dependent phase.
    drive = np.abs(x) / 0.5
    y = 1.8 * x / np.sqrt(1 + drive ** 2) * np.exp(0.2j * drive ** 2)
    directory = Path(__file__).resolve().parent
    np.savetxt(directory / "data.csv", np.column_stack((x.real, x.imag, y.real, y.imag)),
               delimiter=",", header="I_in,Q_in,I_out,Q_out", comments="", fmt="%.10g")
    boundaries = contiguous_boundaries(len(x), DEFAULT_RATIOS, guard_samples=0)
    spec = {
        "description": "Synthetic 64QAM PA example — dummy dataset for tutorial purpose. Generated independently; not a physical PA measurement or benchmark.",
        "origin": "synthetic", "tutorial": True, "dataset_format": "single_csv", "csv_filename": "data.csv",
        "split_ratios": dict(DEFAULT_RATIOS),
        "split_indices": {"train_end": boundaries["train"][1], "val_end": boundaries["val"][1]},
        "input_signal_fs": 80e6, "bw_main_ch": 2e6, "bw_sub_ch": 2e6, "n_sub_ch": 1,
        "nperseg": nfft, "n_active": active, "modulation": "64QAM",
        "generator": "datasets.MyCustomPA.generate", "generator_seed": 20260912,
    }
    (directory / "spec.json").write_text(json.dumps(spec, indent=4) + "\n", encoding="utf-8")


if __name__ == "__main__":
    generate()
