"""Formal evaluation of a finished run: best checkpoint over the test split, scored by a metric profile.

The worker calls ``evaluate_all`` once at the end of a run (predictions are
computed once, every registered profile is scored); ``evaluate_run`` lets the
CLI and the Python API re-score a run later. Nothing here runs inside the
HTTP server process.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from opendpd.core.metrics import evaluate as score, get_profile, list_profiles
from opendpd.schemas import ArtifactKind, ArtifactManifest, EvaluationResult, ResolvedExperimentConfig, RunStatus, TaskType
from opendpd.services.experiments import (
    RESULT_FILE,
    RESULTS_DIR,
    _prepare_inputs,
    build_result,
    load_artifacts,
    load_resolved,
    load_run,
)
from opendpd.services.legacy_adapter import build_namespace, run_in_directory
from opendpd.services.workspace import Workspace, WorkspaceError, write_json_atomic


class Predictions:
    """Best-checkpoint output over the test split, as the trainer's test evaluation produces it."""

    def __init__(self, prediction: np.ndarray, ground_truth: np.ndarray, n_valid: int, target_gain: Optional[float]):
        self.prediction = prediction          # (n_segments, nperseg, 2) float32
        self.ground_truth = ground_truth      # same shape; linear target for DPD tasks
        self.n_valid = n_valid                # real samples; the rest is zero padding
        self.target_gain = target_gain


def _n_test_samples(ws: Workspace, resolved: ResolvedExperimentConfig) -> Optional[int]:
    manifest = ws.get_dataset(resolved.dataset.id)
    version = manifest.version(resolved.dataset.preprocessing_version)
    split = version.split if version is not None else manifest.split
    if split.boundaries and "test" in split.boundaries:
        start, end = split.boundaries["test"]
        return int(end - start)
    return None


def _build_net(proj, task: TaskType, input_size: int):
    """Mirror of steps/train_pa.py and steps/train_dpd.py model construction."""
    import torch
    import models as model
    from quant import get_quant_model
    from utils.util import count_net_params

    pa = model.CoreModel(input_size=input_size, hidden_size=proj.PA_hidden_size, num_layers=proj.PA_num_layers,
                         backbone_type=proj.PA_backbone, window_size=proj.window_size,
                         num_dvr_units=proj.num_dvr_units, thx=proj.thx, thh=proj.thh)
    if task == TaskType.train_pa:
        return pa
    pa_id = proj.gen_pa_model_id(count_net_params(pa))
    pa.load_state_dict(torch.load(os.path.join("save", proj.dataset_name, "train_pa", pa_id + ".pt"),
                                  map_location="cpu", weights_only=True))
    dpd = model.CoreModel(input_size=input_size, hidden_size=proj.DPD_hidden_size, num_layers=proj.DPD_num_layers,
                          backbone_type=proj.DPD_backbone, window_size=proj.window_size,
                          num_dvr_units=proj.num_dvr_units, thx=proj.thx, thh=proj.thh)
    return model.CascadedModel(dpd_model=get_quant_model(proj, dpd), pa_model=pa)


def predict_test_split(ws: Workspace, run_id: str, resolved: ResolvedExperimentConfig,
                       manifest: ArtifactManifest) -> Predictions:
    import torch
    from modules.train_funcs import net_eval
    from project import Project

    dataset = ws.get_dataset(resolved.dataset.id)
    run_dir = ws.run_dir(run_id)
    ns = build_namespace(resolved, dataset_dir=ws.dataset_version_dir(dataset.dataset_id, resolved.dataset.preprocessing_version),
                         dataset_name=dataset.dataset_id)
    ns.plot = False
    checkpoint = run_dir / manifest.by_kind(ArtifactKind.checkpoint)[0].file.path
    with run_in_directory(run_dir):
        if resolved.task == TaskType.train_dpd:
            _prepare_inputs(ws, run_dir, resolved, ns)      # PA surrogate where the legacy code expects it
        proj = Project(args=ns)
        proj.set_device()
        (_, _, test_loader), input_size = proj.build_dataloaders()
        net = _build_net(proj, resolved.task, input_size)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        (net.dpd_model if resolved.task == TaskType.train_dpd else net).load_state_dict(state)
        net = net.to(proj.device)
        _, prediction, ground_truth = net_eval(log={}, net=net, dataloader=test_loader,
                                               criterion=proj.build_criterion(), device=proj.device)
    n_valid = _n_test_samples(ws, resolved) or int(prediction.shape[0] * prediction.shape[1])
    gain = getattr(proj, "target_gain", None)
    return Predictions(prediction, ground_truth, n_valid, float(gain) if gain is not None else None)


def result_for(ws: Workspace, run_id: str, resolved: ResolvedExperimentConfig, manifest: ArtifactManifest,
               predictions: Predictions, profile_id: str) -> EvaluationResult:
    dataset = ws.get_dataset(resolved.dataset.id)
    profile = get_profile(profile_id)
    metrics = score(profile_id, predictions.prediction, predictions.ground_truth, dataset.signal,
                    valid_samples=predictions.n_valid)
    return build_result(ws, run_id, resolved, manifest, profile=profile, metrics=metrics,
                        n_valid=predictions.n_valid, target_gain=predictions.target_gain)


def evaluate_all(ws: Workspace, run_id: str, resolved: ResolvedExperimentConfig,
                 manifest: ArtifactManifest) -> Dict[str, EvaluationResult]:
    """Score every registered profile from one prediction pass; the configured profile is the primary result."""
    predictions = predict_test_split(ws, run_id, resolved, manifest)
    results = {p.profile_id: result_for(ws, run_id, resolved, manifest, predictions, p.profile_id)
               for p in list_profiles()}
    run_dir = ws.run_dir(run_id)
    (run_dir / RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    for profile_id, result in results.items():
        write_json_atomic(run_dir / RESULTS_DIR / f"{profile_id}.json", result)
    write_json_atomic(run_dir / RESULT_FILE, results[resolved.evaluation.profile_id])
    return results


def evaluate_run(ws: Workspace, run_id: str, profile_id: str, *, store: bool = False) -> EvaluationResult:
    """Re-score a succeeded run under ``profile_id`` from its stored best checkpoint."""
    record = load_run(ws, run_id)
    if record.status != RunStatus.succeeded or record.task == TaskType.run_dpd:
        raise WorkspaceError(f"run '{run_id}' has no evaluable result (task {record.task.value}, status {record.status.value})")
    resolved = load_resolved(ws, run_id)
    manifest = load_artifacts(ws, run_id)
    if manifest is None or not manifest.complete:
        raise WorkspaceError(f"run '{run_id}' has no complete artifact manifest")
    get_profile(profile_id)
    predictions = predict_test_split(ws, run_id, resolved, manifest)
    result = result_for(ws, run_id, resolved, manifest, predictions, profile_id)
    if store:
        (ws.run_dir(run_id) / RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        write_json_atomic(ws.run_dir(run_id) / RESULTS_DIR / f"{profile_id}.json", result)
    return result


def available_profiles(ws: Workspace, run_id: str) -> List[str]:
    """Profile ids with a stored result for this run (primary first)."""
    run_dir = ws.run_dir(run_id)
    primary: List[str] = []
    if (run_dir / RESULT_FILE).exists():
        primary.append(EvaluationResult.model_validate_json((run_dir / RESULT_FILE).read_text()).metric_profile_id)
    others = sorted(p.stem for p in (run_dir / RESULTS_DIR).glob("*.json")) if (run_dir / RESULTS_DIR).exists() else []
    return primary + [o for o in others if o not in primary]
