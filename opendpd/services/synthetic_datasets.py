"""Deterministic data for exercising the research UI, never RF evidence."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from opendpd.schemas.conditions import Condition, ConditionSet
from opendpd.schemas.dataset import DatasetOrigin, SignalSpec
from opendpd.schemas.dataset_catalog import SyntheticSuite, SyntheticSuiteRequest
from opendpd.schemas.importing import CsvOptions
from opendpd.services.datasets import import_dataset
from opendpd.services.workspace import Workspace, WorkspaceError, sha256_file

GENERATOR = "synthetic-memory-pa-v1"
DRIVES = (0.18, 0.26, 0.34)
LIMITATIONS = [
    "SYNTHETIC: generated OFDM-like IQ and an illustrative nonlinear memory PA; no instrument was used.",
    "Three normalized drive settings and separate random realizations exercise conditions/repeats; these are not independent hardware captures.",
    "No calibrated dBm, temperature, DC power, efficiency, receiver uncertainty or hardware cost is supplied.",
    "64QAM subcarriers have no standard-specific framing/receiver; this fixture does not validate standard EVM.",
]


def capture(request: SyntheticSuiteRequest, condition: int, repeat: int):
    """Independent input/noise streams per realization, no slices reused as captures."""
    if condition not in range(3) or repeat not in range(request.repeats):
        raise ValueError("unknown synthetic condition/repeat")
    seeds = np.random.SeedSequence([request.seed, condition, repeat]).spawn(2)
    signal_rng, noise_rng = (np.random.Generator(np.random.PCG64(s)) for s in seeds)
    nfft, active = 512, 128
    frames = request.samples_per_capture // nfft
    grid = np.zeros((frames, nfft), dtype=np.complex128)
    bins = np.r_[np.arange(1, active // 2 + 1), np.arange(nfft - active // 2, nfft)]
    levels = np.arange(-7, 8, 2)
    grid[:, bins] = (signal_rng.choice(levels, (frames, active)) + 1j * signal_rng.choice(levels, (frames, active))) / np.sqrt(42)
    x = (np.fft.ifft(grid, axis=1) * nfft / np.sqrt(active) * DRIVES[condition]).ravel()
    # Causal three-tap complex polynomial memory, zero history at the capture start.
    y = np.zeros_like(x)
    for lag, coefficient in enumerate((1.6 + 0.02j, 0.12 - 0.06j, -0.04 + 0.02j)):
        z = x[:len(x) - lag] if lag else x
        y[lag:] += coefficient * z * (1 - .7 * np.abs(z)**2 + .12 * np.abs(z)**4)
    y += .001 * (noise_rng.standard_normal(len(x)) + 1j * noise_rng.standard_normal(len(x))) / np.sqrt(2)
    provenance = {
        "generator": GENERATOR, "seed": request.seed, "condition_index": condition,
        "repeat_index": repeat, "samples": len(x), "normalized_rms_drive": DRIVES[condition],
        "rng": "NumPy PCG64 / SeedSequence([seed, condition, repeat]).spawn(2)",
        "nfft": nfft, "active_subcarriers": active, "delay_samples": 0,
        "memory_taps": [[1.6, .02], [.12, -.06], [-.04, .02]],
        "polynomial": "sum(c[k]*x[n-k]*(1-0.7*abs(x[n-k])**2+0.12*abs(x[n-k])**4), k=0..2)",
        "complex_noise_rms": .001, "initial_history": "zero", "physical_measurement": False,
    }
    return np.column_stack((x.real, x.imag, y.real, y.imag)).astype(np.float32), provenance


def generate_suite(ws: Workspace, request: SyntheticSuiteRequest) -> SyntheticSuite:
    identifiers = [f"{request.prefix}-d{c}-r{r}" for c in range(3) for r in range(request.repeats)]
    # Refuse collisions before writing any member. Repeated identical requests are idempotent.
    existing = []
    for index, identifier in enumerate(identifiers):
        if (ws.dataset_dir(identifier) / "manifest.json").exists():
            previous = ws.get_dataset(identifier)
            _, expected = capture(request, index // request.repeats, index % request.repeats)
            if previous.simulation != expected:
                raise WorkspaceError("A synthetic dataset ID already contains different data. Choose another prefix.")
            for ref in previous.files:
                path = ws.dataset_dir(identifier) / ref.path
                if not path.is_file() or path.is_symlink() or sha256_file(path) != ref.sha256:
                    raise WorkspaceError("An existing synthetic source has changed; choose another prefix after reviewing it.")
            existing.append(previous)
    by_id = {d.dataset_id: d for d in existing}
    signal = SignalSpec(sample_rate_hz=80e6, bandwidth_hz=20e6, sub_channel_bandwidth_hz=20e6,
                        n_sub_ch=1, nperseg=512, modulation="64QAM OFDM-like (synthetic)", amplitude_units="normalized")
    for condition in range(3):
        for repeat in range(request.repeats):
            identifier = f"{request.prefix}-d{condition}-r{repeat}"
            if identifier in by_id:
                continue
            data, provenance = capture(request, condition, repeat)
            with tempfile.TemporaryDirectory(prefix="opendpd-synthetic-") as temporary:
                source = Path(temporary) / "data.csv"
                np.savetxt(source, data, fmt="%.9g", delimiter=",", header="I_in,Q_in,I_out,Q_out", comments="")
                manifest = import_dataset(ws, source, dataset_id=identifier,
                    display_name=f"Synthetic PA · drive {DRIVES[condition]:.2f} · realization {repeat + 1}",
                    origin=DatasetOrigin.synthetic, signal=signal, guard_samples=512,
                    notes=" ".join(LIMITATIONS), csv_options=CsvOptions())
            manifest = manifest.model_copy(update={"simulation": provenance,
                "source": manifest.source.model_copy(update={"original_path": None})})
            ws.save_dataset(manifest)
            by_id[identifier] = manifest
    card = ConditionSet(set_id=f"{request.prefix}-conditions", device="Synthetic memory PA (no physical DUT)",
        dimension="mode", conditions=[Condition(condition_id=f"drive-{c}",
            dataset_id=f"{request.prefix}-d{c}-r0", role="source" if c == 0 else "target",
            capture_batch=f"synthetic-{request.seed}-d{c}-r0", values={"mode": f"normalized-drive-{drive}"},
            notes="Synthetic realization, not a physical capture. Remaining realizations are separate datasets.")
            for c, drive in enumerate(DRIVES)])
    card.card_sha256 = card.compute_sha256()
    return SyntheticSuite(request=request, datasets=[by_id[i] for i in identifiers], condition_set=card, limitations=LIMITATIONS)
