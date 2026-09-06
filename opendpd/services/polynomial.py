"""Least-squares baselines as runs.

PA modeling fits ``Phi(x) w ~= y`` on the train split; DPD uses indirect learning (ILA): the postdistorter
``Phi(y / G) w ~= x`` is identified on the *measured* train split and copied to the predistorter. The PA
surrogate is used for evaluation only: no gradient ever flows through it. The fit writes a checkpoint and
the legacy-shaped logs so the rest of the workflow (artifacts, evaluation under every profile, packages,
reports, comparison) treats a baseline exactly like a learned model.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from opendpd.core.polynomial import (
    MEMORY_BUDGET_BYTES,
    FitDiagnostics,
    PolynomialModel,
    basis,
    basis_bytes,
    coefficient_count,
    fit_least_squares,
    segmented_basis,
    to_complex,
)
from opendpd.core.registry import get_model
from opendpd.schemas import ModelSpec, ResolvedExperimentConfig, TaskType
from opendpd.services.workspace import Workspace, write_json_atomic

FIT_FILE = "fit.json"
LEGACY_METRICS = ("NMSE", "EVM", "ACLR_L", "ACLR_R", "ACLR_AVG")
ILA_STATEMENT = ("indirect learning: the predistorter was identified as a postdistorter on the measured train split "
                 "(Phi(y / G) w ~= x) and copied; the PA surrogate is used for evaluation only, no gradient path "
                 "runs through it")


def is_least_squares(key: str) -> bool:
    return get_model(key).training_method == "least_squares"


def polynomial_module(spec: ModelSpec):
    """An un-fitted torch module for ``spec`` (coefficients come from the checkpoint's state_dict)."""
    return PolynomialModel(spec.key, spec.parameters)


def model_id(role: str, resolved: ResolvedExperimentConfig) -> str:
    """Legacy-shaped id; ``_P_<n>`` carries the real parameter count like the trainer's checkpoints."""
    params = resolved.model.parameters
    tag = "_".join(f"{k.upper()}_{int(v)}" for k, v in params.items() if k != "rcond")
    n_real = 2 * coefficient_count(resolved.model.key, params)
    return f"{role.upper()}_S_{resolved.training.seed}_M_{resolved.model.key.upper()}_{tag}_F_{resolved.training.frame_length}_P_{n_real}"


@dataclass
class FitProject:
    """The attributes ``collect_artifacts`` reads from a legacy Project."""

    path_save_file_best: str
    path_log_file_hist: str
    path_log_file_best: str
    target_gain: float
    diagnostics: FitDiagnostics


def _apply(model, iq: np.ndarray, nperseg: int) -> np.ndarray:
    """Apply a fitted module segment by segment (delays reset), like IQSegmentDataset evaluation."""
    import torch
    n = len(iq)
    pad = (-n) % nperseg
    padded = np.concatenate([iq, np.zeros((pad, 2), dtype=iq.dtype)]) if pad else iq
    with torch.inference_mode():
        out = model(torch.from_numpy(padded.astype(np.float32)).reshape(-1, nperseg, 2)).numpy()
    return out.reshape(-1, 2)[:n]


def _write_logs(hist: Path, best: Path, row: Dict[str, float]) -> None:
    for path in (hist, best):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)


def fit_run(ws: Workspace, run_dir: Path, resolved: ResolvedExperimentConfig, ns,
            on_epoch: Optional[Callable[[Dict], None]] = None) -> FitProject:
    """Fit the baseline for a train_pa / train_dpd run in ``run_dir`` (the current directory)."""
    import torch
    from modules.data_collector import load_dataset
    from utils.util import set_target_gain
    from opendpd.core.metrics import evaluate

    dataset = ws.get_dataset(resolved.dataset.id)
    nperseg = int(dataset.signal.nperseg)
    x_tr, y_tr, x_va, y_va, x_te, y_te = load_dataset(dataset_path=ns.dataset_path)
    key, params = resolved.model.key, resolved.model.parameters
    rcond = float(params.get("rcond", 0.0))
    gain = float(set_target_gain(x_tr, y_tr))
    need = basis_bytes(key, params, len(x_tr))
    if need > MEMORY_BUDGET_BYTES:
        raise MemoryError(f"the {key} basis for {len(x_tr)} training samples needs {need / 2 ** 30:.1f} GiB "
                          f"(budget {MEMORY_BUDGET_BYTES / 2 ** 30:.0f} GiB); reduce the memory depths / orders, "
                          "or use benchmark/benchmark_volterra.py on a GPU")
    role = "pa" if resolved.task == TaskType.train_pa else "dpd"
    xc, yc = to_complex(x_tr), to_complex(y_tr)
    if role == "pa":
        phi, target = segmented_basis(key, params, xc, nperseg), yc
    else:
        phi, target = segmented_basis(key, params, yc / gain, nperseg), xc      # ILA: postdistorter on measured data
    w, diag = fit_least_squares(phi, target, rcond)
    del phi
    model = PolynomialModel(key, params, w)

    mid = model_id(role, resolved)
    save = Path("save") / ns.dataset_name / resolved.task.value / f"{mid}.pt"
    hist = Path("log") / ns.dataset_name / resolved.task.value / "history" / f"{mid}.csv"
    best = Path("log") / ns.dataset_name / resolved.task.value / "best" / f"{mid}.csv"
    save.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save)

    row: Dict[str, float] = {"EPOCH": 0, "N_EPOCH": 1, "TRAIN_LOSS": diag.train_nmse_db}
    if role == "pa":
        for split, xs, ys in (("VAL", x_va, y_va), ("TEST", x_te, y_te)):
            pred = _apply(model, xs, nperseg)
            scores = {m.name: m.value for m in evaluate("legacy-opendpd-v1", pred, ys, dataset.signal)}
            for name in LEGACY_METRICS:
                if scores.get(name) is not None:
                    row[f"{split}_{name}"] = float(scores[name])
    else:
        # identification residual on the other splits; the formal DPD scores come from the evaluation stage
        for split, xs, ys in (("VAL", x_va, y_va), ("TEST", x_te, y_te)):
            phi_s = segmented_basis(key, params, to_complex(ys) / gain, nperseg)
            res = to_complex(xs) - phi_s @ w
            row[f"{split}_ID_NMSE"] = float(10 * np.log10(np.sum(np.abs(res) ** 2) / np.sum(np.abs(to_complex(xs)) ** 2)))
    _write_logs(hist, best, row)
    write_json_atomic(run_dir / FIT_FILE, {
        "schema_version": 1, "model": resolved.model.model_dump(mode="json"), "role": role,
        "method": "direct least squares on the train split" if role == "pa" else ILA_STATEMENT,
        "segment_length": nperseg, "reference_gain": gain if role == "dpd" else None,
        "n_real_parameters": 2 * int(w.size), "diagnostics": diag.to_dict(), "checkpoint": str(save),
    })
    if on_epoch is not None:
        on_epoch(row)
    return FitProject(str(save), str(hist), str(best), gain, diag)


def apply_run(ws: Workspace, run_dir: Path, resolved: ResolvedExperimentConfig, ns) -> None:
    """run_dpd with a least-squares DPD: export u = DPD(x) for the whole test split in one pass."""
    import pandas as pd
    import torch
    from modules.data_collector import load_dataset
    from opendpd.services.experiments import load_artifacts
    from opendpd.services.workspace import sha256_file

    ref = resolved.dpd_reference
    manifest = load_artifacts(ws, ref.run_id)
    artifact = next(a for a in manifest.artifacts if a.artifact_id == ref.checkpoint_artifact_id)
    path = ws.run_dir(ref.run_id) / artifact.file.path
    if sha256_file(path) != ref.checkpoint_sha256:
        raise FileNotFoundError(f"DPD checkpoint of run {ref.run_id} does not match its recorded hash")
    model = polynomial_module(resolved.model)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    x_test = load_dataset(dataset_path=ns.dataset_path)[4]
    u = basis(resolved.model.key, resolved.model.parameters, to_complex(x_test)) @ model.coefficients.numpy()
    out = Path("dpd_out")
    out.mkdir(exist_ok=True)
    frame = pd.DataFrame({"I": x_test[:, 0].astype(np.float32), "Q": x_test[:, 1].astype(np.float32),
                          "I_dpd": u.real.astype(np.float32), "Q_dpd": u.imag.astype(np.float32)})
    frame.to_csv(out / f"{Path(artifact.file.path).stem}.csv", index=False)


def fit_limitations(ws: Workspace, run_id: str, resolved: ResolvedExperimentConfig) -> List[str]:
    """What a result must say about a least-squares baseline (read from the fit record of the fitting run)."""
    from opendpd.services.workspace import read_json

    source = resolved.dpd_reference.run_id if resolved.task == TaskType.run_dpd else run_id
    path = ws.run_dir(source) / FIT_FILE
    if not path.exists():
        return ["least-squares baseline without a fit record"]
    fit = read_json(path)
    d = fit["diagnostics"]
    out = [f"least-squares fit: rank {d['rank']} of {d['n_coefficients']} coefficients retained "
           f"(cutoff rcond={d['rcond']:g}), condition number {d['condition_number']:.3g}, "
           f"train residual {d['train_nmse_db']:.2f} dB; deterministic, no seed or epochs"]
    if fit["role"] == "dpd":
        out.append(ILA_STATEMENT)
    return out
