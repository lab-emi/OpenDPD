"""Formal evaluation of a finished run: best checkpoint over the test split, scored by a metric profile.

The worker calls ``evaluate_all`` once at the end of a run (predictions are
computed once, every registered profile is scored); ``evaluate_run`` lets the
CLI and the Python API re-score a run later. Nothing here runs inside the
HTTP server process.

For DPD tasks the evaluated chain is made explicit: ``x`` (target input),
``u = DPD(x)`` (pre-distorted PA input) and ``y = PA(u)`` where the PA is a
learned surrogate. Two baselines are scored under the *same* linear target:
the surrogate driven by ``x`` directly and the measured PA output of the test
split. Nothing is normalised separately.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from opendpd import __version__
from opendpd.core.metrics import evaluate as score, get_profile, list_profiles
from opendpd.schemas import (
    ArtifactKind,
    ArtifactManifest,
    BaselineScore,
    ComparisonPair,
    ComparisonReport,
    DatasetManifest,
    EvaluationResult,
    ExecutionEvidence,
    ResolvedExperimentConfig,
    RunStatus,
    ScalingInfo,
    SignalStage,
    SurrogateCoverage,
    TaskType,
)
from opendpd.services.experiments import (
    PLOTS_DIR,
    RESULT_FILE,
    RESULTS_DIR,
    _prepare_inputs,
    build_result,
    load_artifacts,
    load_resolved,
    load_result,
    load_run,
)
from opendpd.services.streaming import is_streaming, segments, stream_outputs
from opendpd.services.legacy_adapter import build_namespace, load_checkpoint, run_in_directory
from opendpd.services.workspace import Workspace, WorkspaceError, sha256_file, write_json_atomic

class Predictions:
    """Best-checkpoint output over the test split, as the trainer's test evaluation produces it.
    Arrays are ``(n_segments, nperseg, 2)`` float32; only the first ``n_valid`` samples are real."""

    def __init__(self, prediction: np.ndarray, ground_truth: np.ndarray, n_valid: int, target_gain: Optional[float],
                 *, x: Optional[np.ndarray] = None, u: Optional[np.ndarray] = None,
                 surrogate_without_dpd: Optional[np.ndarray] = None, measured: Optional[np.ndarray] = None,
                 fitted_peak_abs: Optional[float] = None, execution: Optional[ExecutionEvidence] = None):
        self.prediction = prediction          # PA model output (train_pa) or y = PA_surrogate(u) (DPD tasks)
        self.ground_truth = ground_truth      # measured y (train_pa) or the linear target gain * x (DPD tasks)
        self.n_valid = n_valid
        self.target_gain = target_gain
        self.x = x                            # the model input over the test split
        self.u = u                            # DPD tasks: the pre-distorted PA input
        self.surrogate_without_dpd = surrogate_without_dpd   # DPD tasks: PA_surrogate(x)
        self.measured = measured              # measured PA output of the test split
        self.fitted_peak_abs = fitted_peak_abs   # DPD tasks: max |x| the surrogate was trained on
        self.execution = execution            # streaming variants: how the signal was consumed (S18)


def _amplitude(iq: np.ndarray) -> np.ndarray:
    return np.hypot(iq[..., 0], iq[..., 1])


def _valid(segments: np.ndarray, n_valid: int) -> np.ndarray:
    return segments.reshape(-1, segments.shape[-1])[:n_valid]


def _stats(segments: np.ndarray, n_valid: int):
    amp = _amplitude(_valid(segments, n_valid))
    return float(np.max(amp)) if amp.size else 0.0, float(np.sqrt(np.mean(amp ** 2))) if amp.size else 0.0


def _n_test_samples(ws: Workspace, resolved: ResolvedExperimentConfig) -> Optional[int]:
    manifest = ws.get_dataset(resolved.dataset.id)
    version = manifest.version(resolved.dataset.preprocessing_version)
    split = version.split if version is not None else manifest.split
    if split.boundaries and "test" in split.boundaries:
        start, end = split.boundaries["test"]
        return int(end - start)
    return None


def _input_scaling(manifest: DatasetManifest, version_name: str) -> str:
    """Human-readable statement of how the input amplitudes were scaled before training."""
    version = manifest.version(version_name)
    params = version.params if version is not None else None
    if params is None or params.normalize == "none":
        return "none: amplitudes as imported"
    fit = f" fitted on train samples [{version.fit_range[0]}, {version.fit_range[1]})" if version.fit_range else ""
    return f"{params.normalize} normalisation{fit} (preprocess-v1, version {version_name})"


def _fitted_peak(ws: Workspace, pa_run_id: str) -> float:
    """Largest input amplitude in the training split the PA surrogate was fitted on."""
    from modules.data_collector import load_dataset

    pa = load_resolved(ws, pa_run_id)
    x_train, *_ = load_dataset(dataset_path=str(ws.dataset_version_dir(pa.dataset.id, pa.dataset.preprocessing_version)))
    return float(np.max(_amplitude(np.asarray(x_train, dtype=np.float32))))


def _build_net(proj, resolved: ResolvedExperimentConfig, input_size: int):
    """Mirror of steps/train_pa.py and steps/train_dpd.py model construction. Least-squares baselines are torch
    modules of the compute core (no legacy backbone); a DPD is always cascaded with a gradient-trained PA."""
    import models as model
    from quant import get_quant_model
    from utils.util import count_net_params
    from opendpd.services.polynomial import is_least_squares, polynomial_module

    def core(hidden: int, layers: int, backbone: str):
        return model.CoreModel(input_size=input_size, hidden_size=hidden, num_layers=layers, backbone_type=backbone,
                               window_size=proj.window_size, num_dvr_units=proj.num_dvr_units, thx=proj.thx, thh=proj.thh)

    least_squares = is_least_squares(resolved.model.key)
    if resolved.task in (TaskType.train_pa, TaskType.evaluate_pa):
        return polynomial_module(resolved.model) if least_squares \
            else core(proj.PA_hidden_size, proj.PA_num_layers, proj.PA_backbone)
    pa = core(proj.PA_hidden_size, proj.PA_num_layers, proj.PA_backbone)
    pa_id = proj.gen_pa_model_id(count_net_params(pa))
    pa.load_state_dict(load_checkpoint(os.path.join("save", proj.dataset_name, "train_pa", pa_id + ".pt")))
    dpd = polynomial_module(resolved.model) if least_squares \
        else get_quant_model(proj, core(proj.DPD_hidden_size, proj.DPD_num_layers, proj.DPD_backbone))
    return model.CascadedModel(dpd_model=dpd, pa_model=pa)


def _checkpoint_path(ws: Workspace, run_id: str, resolved: ResolvedExperimentConfig, manifest: ArtifactManifest) -> Path:
    """The weights under evaluation: this run's best checkpoint, or the referenced run's for run_dpd (the DPD)
    and evaluate_pa (the PA), hash verified."""
    if resolved.task not in (TaskType.run_dpd, TaskType.evaluate_pa):
        return ws.run_dir(run_id) / manifest.by_kind(ArtifactKind.checkpoint)[0].file.path
    ref = resolved.dpd_reference if resolved.task == TaskType.run_dpd else resolved.pa_reference
    ref_manifest = load_artifacts(ws, ref.run_id)
    artifact = next((a for a in (ref_manifest.artifacts if ref_manifest else [])
                     if a.artifact_id == ref.checkpoint_artifact_id), None)
    path = ws.run_dir(ref.run_id) / artifact.file.path if artifact else None
    if path is None or not path.exists() or sha256_file(path) != ref.checkpoint_sha256:
        raise FileNotFoundError(f"checkpoint of run {ref.run_id} is missing or its hash changed")
    return path


def predict_test_split(ws: Workspace, run_id: str, resolved: ResolvedExperimentConfig,
                       manifest: ArtifactManifest) -> Predictions:
    import torch
    from modules.data_collector import IQSegmentDataset, load_dataset
    from modules.train_funcs import net_eval
    from project import Project

    dataset = ws.get_dataset(resolved.dataset.id)
    run_dir = ws.run_dir(run_id)
    ns = build_namespace(resolved, dataset_dir=ws.dataset_version_dir(dataset.dataset_id, resolved.dataset.preprocessing_version),
                         dataset_name=dataset.dataset_id)
    ns.plot = False
    checkpoint = _checkpoint_path(ws, run_id, resolved, manifest)
    dpd_task = resolved.task in (TaskType.train_dpd, TaskType.run_dpd)
    extra: Dict[str, object] = {}
    with run_in_directory(run_dir):
        if dpd_task:
            _prepare_inputs(ws, run_dir, resolved, ns)      # PA surrogate (and DPD) where the legacy code expects them
        proj = Project(args=ns)
        proj.set_device()
        (_, _, test_loader), input_size = proj.build_dataloaders()
        net = _build_net(proj, resolved, input_size)
        state = load_checkpoint(checkpoint)
        (net.dpd_model if dpd_task else net).load_state_dict(state)
        net = net.to(proj.device)
        _, prediction, ground_truth = net_eval(log={}, net=net, dataloader=test_loader,
                                               criterion=proj.build_criterion(), device=proj.device)
        xs, us, y0s = [], [], []
        net.eval()
        with torch.inference_mode():
            for features, _ in test_loader:
                features = features.to(proj.device)
                xs.append(features.cpu())
                if dpd_task:
                    us.append(net.dpd_model(features).cpu())
                    y0s.append(net.pa_model(features).cpu())
        extra = dict(x=torch.cat(xs).numpy(), measured=ground_truth)
        x_test, y_test = load_dataset(dataset_path=ns.dataset_path)[4:6]
        nperseg = proj.args.nperseg
        if is_streaming(resolved.model.key):
            # the variant consumes the continuous test split chunk by chunk; the PA surrogate of a DPD task stays
            # the offline module (it is not the model under evaluation)
            evaluated = net.dpd_model if dpd_task else net
            y_stream, extra["execution"] = stream_outputs(
                evaluated.cpu(), resolved.model.key, np.asarray(x_test, dtype=np.float32),
                chunk_samples=resolved.evaluation.chunk_samples, sample_rate_hz=dataset.signal.sample_rate_hz)
            streamed = segments(y_stream, nperseg)
            if dpd_task:
                with torch.inference_mode():
                    prediction = net.pa_model.cpu()(torch.from_numpy(streamed)).numpy()
                us = [torch.from_numpy(streamed)]
            else:
                prediction = streamed
        if dpd_task:
            # The reference is the linear target gain * x for every DPD task (the legacy loader only builds it
            # for train_dpd; a run_dpd would otherwise be scored against the measured PA output).
            ground_truth = IQSegmentDataset(x_test, proj.target_gain * x_test, nperseg=nperseg).targets.numpy()
            extra.update(u=torch.cat(us).numpy(), surrogate_without_dpd=torch.cat(y0s).numpy(),
                         measured=IQSegmentDataset(x_test, y_test, nperseg=nperseg).targets.numpy(),
                         fitted_peak_abs=_fitted_peak(ws, resolved.pa_reference.run_id))
    n_valid = _n_test_samples(ws, resolved) or int(prediction.shape[0] * prediction.shape[1])
    gain = getattr(proj, "target_gain", None)
    return Predictions(prediction, ground_truth, n_valid, float(gain) if gain is not None else None, **extra)


def _surrogate_evidence(ws: Workspace, run_id: str, resolved: ResolvedExperimentConfig, manifest: ArtifactManifest,
                        dataset: DatasetManifest, predictions: Predictions, profile_id: str) -> Dict[str, object]:
    """Signal chain, baselines, amplitude coverage and scaling for a DPD result."""
    n = predictions.n_valid
    pa = resolved.pa_reference
    if resolved.task == TaskType.run_dpd:
        dpd_run, dpd_sha = resolved.dpd_reference.run_id, resolved.dpd_reference.checkpoint_sha256
    else:
        dpd_run, dpd_sha = run_id, manifest.by_kind(ArtifactKind.checkpoint)[0].file.sha256
    x_peak, x_rms = _stats(predictions.x, n)
    u_peak, u_rms = _stats(predictions.u, n)
    y_peak, y_rms = _stats(predictions.prediction, n)
    version = resolved.dataset.preprocessing_version
    chain = [
        SignalStage(symbol="x", role="target input: the PA output should equal reference_gain * x",
                    source=f"dataset {dataset.dataset_id} version {version}, test split",
                    n_samples=n, peak_abs=x_peak, rms=x_rms),
        SignalStage(symbol="u", role="pre-distorted PA input, u = DPD(x)",
                    source=f"DPD {resolved.model.key} weights {dpd_sha[:12]} from run {dpd_run}",
                    n_samples=n, peak_abs=u_peak, rms=u_rms,
                    artifact_id="dpd-output" if resolved.task == TaskType.run_dpd else None),
        SignalStage(symbol="y", role="PA output, y = PA(u)",
                    source=f"PA surrogate {pa.model.key} weights {pa.checkpoint_sha256[:12]} from run {pa.run_id}; "
                           "simulated, not measured",
                    simulated=True, n_samples=n, peak_abs=y_peak, rms=y_rms),
    ]
    reference = predictions.ground_truth
    baselines = [
        BaselineScore(kind="surrogate_without_dpd",
                      description=f"PA surrogate {pa.run_id} driven by x directly (no DPD), scored against the same "
                                  "linear target reference_gain * x",
                      metrics=score(profile_id, predictions.surrogate_without_dpd, reference, dataset.signal, valid_samples=n)),
        BaselineScore(kind="measured_without_dpd",
                      description="measured PA output of the test split (no DPD), scored against the same linear target",
                      metrics=score(profile_id, predictions.measured, reference, dataset.signal, valid_samples=n)),
    ]
    fitted = float(predictions.fitted_peak_abs or 0.0)
    above = float(np.mean(_amplitude(_valid(predictions.u, n)) > fitted)) if fitted > 0 and n else 0.0
    if above > 0:
        note = (f"{above:.2%} of the pre-distorted samples exceed the largest input amplitude the surrogate was fitted "
                f"on ({fitted:.4g}); the surrogate extrapolates there and y is unverified for those samples. "
                "Staying inside the range would not prove the surrogate accurate either.")
    else:
        note = (f"every pre-distorted sample stays within the largest input amplitude the surrogate was fitted on "
                f"({fitted:.4g}); this rules out amplitude extrapolation only and does not prove the surrogate accurate.")
    coverage = SurrogateCoverage(fitted_peak_abs=fitted, u_peak_abs=u_peak, fraction_above_fitted_peak=above, note=note)
    scaling = ScalingInfo(amplitude_units=dataset.signal.amplitude_units, input_scaling=_input_scaling(dataset, version),
                          reference_gain=predictions.target_gain, physical_calibration=False)
    return dict(signal_chain=chain, baselines=baselines, surrogate_coverage=coverage, scaling=scaling)


def result_for(ws: Workspace, run_id: str, resolved: ResolvedExperimentConfig, manifest: ArtifactManifest,
               predictions: Predictions, profile_id: str) -> EvaluationResult:
    dataset = ws.get_dataset(resolved.dataset.id)
    profile = get_profile(profile_id)
    metrics = score(profile_id, predictions.prediction, predictions.ground_truth, dataset.signal,
                    valid_samples=predictions.n_valid)
    evidence = {} if resolved.task in (TaskType.train_pa, TaskType.evaluate_pa) \
        else _surrogate_evidence(ws, run_id, resolved, manifest, dataset, predictions, profile_id)
    return build_result(ws, run_id, resolved, manifest, profile=profile, metrics=metrics,
                        n_valid=predictions.n_valid, target_gain=predictions.target_gain, execution=predictions.execution,
                        **evidence)


def write_plots(ws: Workspace, run_id: str, resolved: ResolvedExperimentConfig, predictions: Predictions) -> List[Path]:
    """plots-v1 derived data next to the result: spectrum, time excerpt, AM-AM / AM-PM (display only)."""
    from opendpd.core import plots

    dataset = ws.get_dataset(resolved.dataset.id)
    n = predictions.n_valid
    if resolved.task in (TaskType.train_pa, TaskType.evaluate_pa):
        signals = {"PA input x": predictions.x, "measured PA output": predictions.ground_truth,
                   "PA model output": predictions.prediction}
        roles = {"PA input x": "input", "measured PA output": "reference", "PA model output": "primary"}
        outputs = {"measured PA output": predictions.ground_truth, "PA model output": predictions.prediction}
    else:
        signals = {"target input x": predictions.x, "u = DPD(x)": predictions.u,
                   "linear target gain*x": predictions.ground_truth,
                   "with DPD: PA_surrogate(u)": predictions.prediction,
                   "surrogate without DPD": predictions.surrogate_without_dpd,
                   "measured PA without DPD": predictions.measured}
        roles = {"target input x": "input", "u = DPD(x)": "predistorted", "linear target gain*x": "reference",
                 "with DPD: PA_surrogate(u)": "primary", "surrogate without DPD": "baseline",
                 "measured PA without DPD": "baseline"}
        outputs = {k: signals[k] for k in ("with DPD: PA_surrogate(u)", "surrogate without DPD", "measured PA without DPD")}
    sig = dataset.signal
    out_dir = ws.run_dir(run_id) / PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, data in (
        ("spectrum", plots.spectrum(signals, roles, sample_rate_hz=sig.sample_rate_hz, nperseg=sig.nperseg,
                                    bandwidth_hz=sig.bandwidth_hz, valid_samples=n)),
        ("time", plots.time_excerpt(signals, roles, valid_samples=n)),
        ("amam", plots.am_am_pm(predictions.x, outputs, roles, valid_samples=n)),
    ):
        write_json_atomic(out_dir / f"{name}.json", data)
        written.append(out_dir / f"{name}.json")
    return written


def evaluate_all(ws: Workspace, run_id: str, resolved: ResolvedExperimentConfig,
                 manifest: ArtifactManifest) -> Dict[str, EvaluationResult]:
    """Score every registered profile from one prediction pass; the configured profile is the primary result."""
    predictions = predict_test_split(ws, run_id, resolved, manifest)
    results = {p.profile_id: result_for(ws, run_id, resolved, manifest, predictions, p.profile_id)
               for p in list_profiles()}
    store_results(ws.run_dir(run_id), results, resolved.evaluation.profile_id)
    write_plots(ws, run_id, resolved, predictions)
    return results


def store_results(run_dir: Path, results: Dict[str, EvaluationResult], primary_profile: str) -> None:
    """One file per profile under results/ plus the primary result as result.json."""
    (run_dir / RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    for profile_id, result in results.items():
        write_json_atomic(run_dir / RESULTS_DIR / f"{profile_id}.json", result)
    write_json_atomic(run_dir / RESULT_FILE, results[primary_profile])


def compare_results(ws: Workspace, run_ids: List[str], profile_id: Optional[str] = None) -> ComparisonReport:
    """Stored results side by side. Every pair is checked with ``incompatibilities``; results under different
    protocols are shown with the reasons and never ranked."""
    from opendpd.core.metrics import incompatibilities

    results = []
    for run_id in run_ids:
        result = load_result(ws, run_id, profile_id)
        if result is None:
            raise WorkspaceError(f"run '{run_id}' has no stored result" + (f" under profile '{profile_id}'" if profile_id else ""))
        results.append(result)
    pairs = [ComparisonPair(a=a.run_id or a.result_id, b=b.run_id or b.result_id, incompatibilities=incompatibilities(a, b))
             for i, a in enumerate(results) for b in results[i + 1:]]
    comparable = all(not p.incompatibilities for p in pairs)
    note = ("all results were produced under the same protocol (profile, evidence, data, split, reference); "
            "they may be ranked" if comparable else
            "results differ in protocol; they are shown side by side with the differences and must not be ranked")
    return ComparisonReport(results=results, pairs=pairs, comparable=comparable, note=note)


def comparison_csv(report: ComparisonReport) -> str:
    """Metric rows by result columns, preceded by the provenance every number is bound to. No recomputation."""
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    cols = report.results
    w.writerow(["field"] + [r.run_id or r.result_id for r in cols])
    w.writerow(["result_id"] + [r.result_id for r in cols])
    w.writerow(["evidence_type"] + [r.evidence_type.value for r in cols])
    w.writerow(["metric_profile"] + [f"{r.metric_profile_id} v{r.metric_profile_version}" for r in cols])
    w.writerow(["dataset"] + [f"{r.dataset.dataset_id} {r.dataset.preprocessing_version} {r.dataset.split_version}" for r in cols])
    w.writerow(["reference"] + [r.reference.kind for r in cols])
    w.writerow(["models"] + ["; ".join(f"{m.role}:{m.model.key}:{(m.weights_sha256 or '')[:12]}" for m in r.models) for r in cols])
    w.writerow(["comparable"] + [str(report.comparable).lower()] * len(cols))
    names = []
    for r in cols:
        for m in r.metrics:
            if m.name not in names:
                names.append(m.name)
    for name in names:
        row = [name]
        for r in cols:
            m = next((x for x in r.metrics if x.name == name), None)
            row.append("" if m is None else (f"{m.value}" if m.value is not None else f"{m.status.value}: {m.reason}"))
        w.writerow(row)
    for pair in report.pairs:
        if pair.incompatibilities:
            w.writerow([f"incompatible {pair.a} vs {pair.b}", "; ".join(pair.incompatibilities)])
    return buf.getvalue()


def evaluate_run(ws: Workspace, run_id: str, profile_id: str) -> EvaluationResult:
    """Re-score a succeeded run under ``profile_id`` from its stored checkpoint (nothing is written)."""
    record = load_run(ws, run_id)
    if record.status != RunStatus.succeeded:
        raise WorkspaceError(f"run '{run_id}' has no evaluable result (status {record.status.value})")
    resolved = load_resolved(ws, run_id)
    manifest = load_artifacts(ws, run_id)
    if manifest is None or not manifest.complete:
        raise WorkspaceError(f"run '{run_id}' has no complete artifact manifest")
    get_profile(profile_id)
    if resolved.task == TaskType.evaluate_measured:
        from opendpd.services import measurements

        signals = measurements.load_signals(ws, run_id, resolved)
        return measurements.result_for_measured(ws, run_id, resolved, manifest, signals, profile_id)
    predictions = predict_test_split(ws, run_id, resolved, manifest)
    return result_for(ws, run_id, resolved, manifest, predictions, profile_id)


def available_profiles(ws: Workspace, run_id: str) -> List[str]:
    """Profile ids with a stored result for this run (primary first)."""
    run_dir = ws.run_dir(run_id)
    primary: List[str] = []
    if (run_dir / RESULT_FILE).exists():
        primary.append(EvaluationResult.model_validate_json((run_dir / RESULT_FILE).read_text()).metric_profile_id)
    others = sorted(p.stem for p in (run_dir / RESULTS_DIR).glob("*.json")) if (run_dir / RESULTS_DIR).exists() else []
    return primary + [o for o in others if o not in primary]


def write_dpd_output_metadata(ws: Workspace, run_id: str, resolved: ResolvedExperimentConfig, ns) -> List[Path]:
    """Sidecar for every exported pre-distorted signal: what the columns are, their order, dtype, scaling
    and the exact checkpoints that produced them. Written by the worker next to the CSV."""
    import pandas as pd
    from modules.data_collector import load_dataset
    from utils.util import set_target_gain

    run_dir = ws.run_dir(run_id)
    dataset = ws.get_dataset(resolved.dataset.id)
    x_train, y_train, *_ = load_dataset(dataset_path=ns.dataset_path)
    gain = float(set_target_gain(x_train, y_train))
    written: List[Path] = []
    for csv in sorted((run_dir / "dpd_out").rglob("*.csv")):
        frame = pd.read_csv(csv)
        x = frame[["I", "Q"]].to_numpy(dtype=np.float32)
        u = frame[["I_dpd", "Q_dpd"]].to_numpy(dtype=np.float32)
        meta = {
            "schema_version": 1,
            "signal_role": "pa_input_predistorted",
            "statement": "u = DPD(x) is the pre-distorted PA *input* for the test split. It is not a PA output and "
                         "its existence does not show that the PA was linearised.",
            "columns": {"I": "x: target input, I", "Q": "x: target input, Q",
                        "I_dpd": "u = DPD(x): pre-distorted PA input, I", "Q_dpd": "u = DPD(x): pre-distorted PA input, Q"},
            "dtype": "float32",
            "n_samples": int(len(frame)),
            "sample_order": f"test split of dataset {dataset.dataset_id} version {resolved.dataset.preprocessing_version} "
                            f"({resolved.dataset.split_version}), contiguous, original order",
            "semantics": "offline: the DPD ran once over the whole test split with its state carried across samples",
            "amplitude_units": dataset.signal.amplitude_units,
            "input_scaling": _input_scaling(dataset, resolved.dataset.preprocessing_version),
            "reference_gain": gain,
            "reference_gain_rule": "max|y_train| / max|x_train| (legacy utils.util.set_target_gain)",
            "peak_abs_x": float(np.max(_amplitude(x))) if len(x) else 0.0,
            "peak_abs_u": float(np.max(_amplitude(u))) if len(u) else 0.0,
            "physical_calibration": False,
            "dataset": {"id": dataset.dataset_id, "preprocessing_version": resolved.dataset.preprocessing_version,
                        "split_version": resolved.dataset.split_version, "raw_sha256": dataset.raw_sha256},
            "dpd": {"run_id": resolved.dpd_reference.run_id, "model": resolved.model.model_dump(mode="json"),
                    "checkpoint_sha256": resolved.dpd_reference.checkpoint_sha256},
            "pa_surrogate": {"run_id": resolved.pa_reference.run_id, "model": resolved.pa_reference.model.model_dump(mode="json"),
                             "checkpoint_sha256": resolved.pa_reference.checkpoint_sha256},
            "generated_by": {"run_id": run_id, "opendpd_version": __version__},
        }
        target = csv.with_suffix(".meta.json")
        write_json_atomic(target, meta)
        written.append(target)
    return written
