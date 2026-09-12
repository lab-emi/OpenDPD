"""Regenerate the frozen legacy metric references (protected path).

Run only when a maintainer has approved a change of the *legacy-opendpd-v1*
metric semantics; the output is compared bit-for-bit (within tolerance) by
``test_legacy_metrics_golden.py``. The stimulus is a deterministic multi-tone
signal so the fixture does not depend on any random number generator stream.

    python tests/golden/generate_legacy_metric_goldens.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from utils.metrics import ACLR, EVM, NMSE
from utils.util import set_target_gain

GOLDEN_PATH = Path(__file__).with_name("legacy_metrics_v1.json")

PROFILES = {
    # Mirrors datasets/DPA_200MHz/spec.json
    "dpa_200mhz": dict(fs=800e6, nperseg=2560, bw_main_ch=200e6, n_sub_ch=10, n_segments=4),
    # Mirrors datasets/APA_200MHz/spec.json
    "apa_200mhz": dict(fs=983.04e6, nperseg=19662, bw_main_ch=200e6, n_sub_ch=5, n_segments=2),
}


def deterministic_multitone(n_segments: int, nperseg: int, fs: float, bw: float,
                            n_tones: int = 48) -> np.ndarray:
    """Complex multi-tone stimulus with index-derived phases (no RNG)."""
    t = np.arange(nperseg) / fs
    tones = np.linspace(-0.45 * bw, 0.45 * bw, n_tones)
    signal = np.zeros((n_segments, nperseg), dtype=np.complex128)
    for s in range(n_segments):
        for k, f in enumerate(tones):
            phase = 2 * np.pi * ((k * 0.6180339887 + s * 0.3247) % 1.0)
            signal[s] += np.exp(1j * (2 * np.pi * f * t + phase))
    # RMS 0.25 so the cubic term below stays a mild compression, not a wreck
    return signal / (4.0 * np.sqrt(n_tones))


def to_iq(signal: np.ndarray) -> np.ndarray:
    return np.stack([signal.real, signal.imag], axis=-1)


def stimulus(profile: dict):
    x = deterministic_multitone(profile["n_segments"], profile["nperseg"],
                                profile["fs"], profile["bw_main_ch"])
    gain = 1.5 * np.exp(1j * 0.1)
    ground_truth = gain * x                      # ideal linear PA
    prediction = gain * (x - 0.05 * np.abs(x) ** 2 * x)  # mildly compressive PA-like output
    return to_iq(prediction), to_iq(ground_truth), to_iq(x)


def compute(profile: dict) -> dict:
    prediction, ground_truth, pa_input = stimulus(profile)
    aclr_left, aclr_right = ACLR(prediction, fs=profile["fs"], nperseg=profile["nperseg"],
                                 bw_main_ch=profile["bw_main_ch"], n_sub_ch=profile["n_sub_ch"])
    return {
        "NMSE": float(NMSE(prediction, ground_truth)),
        "EVM": float(EVM(prediction, ground_truth, sample_rate=int(profile["fs"]),
                         bw_main_ch=profile["bw_main_ch"], n_sub_ch=profile["n_sub_ch"],
                         nperseg=profile["nperseg"])),
        "ACLR_L": float(aclr_left),
        "ACLR_R": float(aclr_right),
        "ACLR_AVG": float((aclr_left + aclr_right) / 2),
        "target_gain": float(set_target_gain(pa_input.reshape(-1, 2), ground_truth.reshape(-1, 2))),
    }


def main() -> None:
    payload = {
        "metric_profile": "legacy-opendpd-v1",
        "note": "Frozen reference values of utils.metrics on a deterministic stimulus. "
                "Changing these numbers is a scientific-semantics change (AGENTS.md §3).",
        "profiles": {name: {"config": cfg, "expected": compute(cfg)} for name, cfg in PROFILES.items()},
    }
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
