"""Deployment packages under ``fixed-point-v1`` (plan S19): quantise a finished GRU run, produce the golden vectors,
verify the C99 backend bit for bit against the software reference, report the float-to-fixed loss and the
resources with their labels, and zip everything with a hash per file."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from opendpd.core.fixed_point import FixedGRU, QuantisedGRU, quantise, to_fixed, to_float
from opendpd.core.metrics import evaluate as score
from opendpd.core.registry import get_model
from opendpd.core.streaming import DEFAULT_CHUNK_SAMPLES, run_stream
from opendpd.export import c_backend
from opendpd.schemas import ArtifactKind, RunStatus, TaskType
from opendpd.schemas.fixed_point import (
    SPEC_ID,
    DeploymentManifest,
    FixedPointReport,
    FixedPointSpec,
    GoldenCase,
    MeasuredExecution,
    MetricDelta,
    ResourceEstimate,
)
from opendpd.services.benchmark import _machine
from opendpd.services.evaluation import TrainedModel, trained_model
from opendpd.services.experiments import load_artifacts, load_run
from opendpd.services.streaming import segments, streaming_model
from opendpd.services.workspace import Workspace, WorkspaceError, sha256_bytes, software_provenance

LONG_SEQUENCE = 65536
EXECUTION_ASSUMPTIONS = [
    "one sample per step, sequential; the state (spec.h) is carried across samples and chunks (gru_stream semantics)",
    "two's-complement integers; right shifts are floors implemented with division (no arithmetic-shift assumption)",
    "dot products are exact in the accumulator (int64 in the references); the bound is checked, never wrapped",
    "no parallelism, pipelining or sparsity is assumed; the C reference is single-threaded",
    "the input is already in spec.x: analogue front-end, AGC and I/Q calibration are outside this package",
]


def support(model_key: str) -> Optional[str]:
    """None when the model can be exported under fixed-point-v1, otherwise the reason it cannot."""
    model = get_model(model_key)
    if SPEC_ID in model.export_formats:
        return None
    return (f"model '{model_key}' has no fixed-point specification; {SPEC_ID} covers one-layer gru weights executed as "
            "gru_stream (see `opendpd models`: export_formats)")


# --- golden vectors ------------------------------------------------------------------------------------

def golden_inputs(x_test: np.ndarray, spec: FixedPointSpec, seed: int = 0) -> List[Tuple[str, str, np.ndarray, Tuple[int, ...]]]:
    """(case id, description, quantised inputs, resets) for every case the protocol requires."""
    rng = np.random.default_rng(seed)
    x_q = to_fixed(x_test, spec.x)
    hi = spec.x.max_int
    n = x_q.shape[0]
    block = x_q[:256]
    long = np.concatenate([x_q] * (-(-LONG_SEQUENCE // n)))[:LONG_SEQUENCE] if n < LONG_SEQUENCE else x_q[:LONG_SEQUENCE]
    alternating = np.tile(np.array([[hi, -hi], [-hi, hi]], dtype=np.int64), (128, 1))
    return [
        ("normal", "the first 4096 samples of the test split", x_q[:4096], ()),
        ("extreme", "full-scale inputs: constant +max, constant -max, alternating", np.concatenate(
            [np.full((128, 2), hi, dtype=np.int64), np.full((128, 2), -hi, dtype=np.int64), alternating]), ()),
        ("saturation", "random full-scale signs: pre-activations beyond the table ranges, state at its bounds",
         rng.choice(np.array([-hi, hi], dtype=np.int64), size=(1024, 2)), ()),
        ("all_zero", "zero input: the state stays at zero and the output is the bias path", np.zeros((256, 2), dtype=np.int64), ()),
        ("state_reset", "one 256-sample block four times with a reset before each; the four outputs are identical",
         np.concatenate([block] * 4), (0, 256, 512, 768)),
        ("long_sequence", f"{LONG_SEQUENCE} samples of the test split (tiled when shorter): no drift, no overflow", long, ()),
    ]


def _i16(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype="<i2").tobytes()


def write_golden(q: QuantisedGRU, cases, out: Path) -> List[GoldenCase]:
    """Run the software reference over every case and write inputs, outputs, final state and state trace."""
    index: List[GoldenCase] = []
    for case_id, description, x_q, resets in cases:
        y_q, traces = FixedGRU(q).run(x_q, resets, trace=True)
        if case_id == "state_reset":
            block = x_q.shape[0] // 4
            for k in range(1, 4):
                if not np.array_equal(y_q[:block], y_q[k * block:(k + 1) * block]):
                    raise RuntimeError("the reference is not reset-exact: outputs after a reset differ from the first block")
        d = out / case_id
        d.mkdir(parents=True, exist_ok=True)
        files = {"x.i16": _i16(x_q), "y.i16": _i16(y_q), "h_final.i16": _i16(traces["h"][-1]), "h_trace.i16": _i16(traces["h"])}
        for name, data in files.items():
            (d / name).write_bytes(data)
        meta = {"case_id": case_id, "description": description, "n_samples": int(x_q.shape[0]), "resets_at": list(resets),
                "layout": {"x.i16": "n x inputs int16 little-endian, spec.x", "y.i16": "n x outputs int16, spec.y",
                           "h_final.i16": "hidden int16, spec.h", "h_trace.i16": "n x hidden int16: the state after every sample"}}
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        index.append(GoldenCase(case_id=case_id, description=description, n_samples=int(x_q.shape[0]), resets_at=list(resets),
                                input_sha256=sha256_bytes(files["x.i16"]), output_sha256=sha256_bytes(files["y.i16"]),
                                state_sha256=sha256_bytes(files["h_final.i16"]), trace_sha256=sha256_bytes(files["h_trace.i16"])))
    return index


# --- quality loss ------------------------------------------------------------------------------------------

def quality_loss(tm: TrainedModel, q: QuantisedGRU) -> Tuple[List[MetricDelta], str]:
    """The run's profile scored on the float streaming model and on the fixed-point reference over the test split
    (the fixed reference sees quantised inputs, as a deployment would)."""
    import torch

    spec = q.spec
    profile_id = tm.resolved.evaluation.profile_id
    x = tm.x_test
    float_out = run_stream(streaming_model("gru_stream", tm.evaluated), x, DEFAULT_CHUNK_SAMPLES)
    fixed_q, _ = FixedGRU(q).run(to_fixed(x, spec.x))
    fixed_out = to_float(fixed_q, spec.y).astype(np.float32)
    n = x.shape[0]
    if tm.resolved.task == TaskType.train_pa:
        reference = segments(tm.y_test, tm.nperseg)
        preds = {"float": segments(float_out, tm.nperseg), "fixed": segments(fixed_out, tm.nperseg)}
    else:
        pa = tm.net.pa_model.cpu()
        with torch.inference_mode():
            preds = {k: pa(torch.from_numpy(segments(u, tm.nperseg))).numpy() for k, u in (("float", float_out), ("fixed", fixed_out))}
        reference = segments((tm.target_gain or 1.0) * x, tm.nperseg)
    scored = {k: {m.name: m for m in score(profile_id, v, reference, tm.dataset.signal, valid_samples=n)} for k, v in preds.items()}
    deltas = []
    for name, m in scored["float"].items():
        f, g = m.value, scored["fixed"][name].value
        deltas.append(MetricDelta(name=name, unit=m.unit, float_value=f, fixed_value=g,
                                  delta=None if f is None or g is None else float(g - f)))
    return deltas, profile_id


# --- the package -------------------------------------------------------------------------------------------

def _weights_json(q: QuantisedGRU) -> Dict[str, object]:
    return {"spec_id": SPEC_ID, "hidden": q.hidden, "inputs": q.inputs, "outputs": q.outputs,
            "fractions": {"w_ih": q.f_ih, "w_hh": q.f_hh, "w_out": q.f_out, "bias": q.spec.pre.frac},
            "w_ih": q.w_ih.tolist(), "w_hh": q.w_hh.tolist(), "w_out": q.w_out.tolist(),
            "b_ih": q.b_ih.tolist(), "b_hh": q.b_hh.tolist(), "b_out": q.b_out.tolist(),
            "sigmoid_table": q.sigmoid_table.tolist(), "tanh_table": q.tanh_table.tolist(),
            "gate_order": ["r", "z", "n"]}


def export_deployment(ws: Workspace, run_id: str, out: Path, *, spec: Optional[FixedPointSpec] = None) -> DeploymentManifest:
    """Build ``out`` (a zip) for a succeeded train_pa / train_dpd run of a supported model. Raises WorkspaceError
    when the run or the model does not qualify; a verification failure is recorded, never hidden."""
    spec = spec or FixedPointSpec()
    record = load_run(ws, run_id)
    if record.status != RunStatus.succeeded or record.task not in (TaskType.train_pa, TaskType.train_dpd):
        raise WorkspaceError(f"run '{run_id}' is a {record.status.value} {record.task.value} run; a deployment package "
                             "starts from a succeeded train_pa or train_dpd run")
    reason = support(record.model_key or "")
    if reason:
        raise WorkspaceError(reason)
    tm = trained_model(ws, run_id)
    try:
        q = quantise({k: v.detach().cpu().numpy() for k, v in tm.evaluated.state_dict().items()}, spec)
    except ValueError as err:
        raise WorkspaceError(f"run '{run_id}': {err}") from None
    manifest = load_artifacts(ws, run_id)
    checkpoints = manifest.by_kind(ArtifactKind.checkpoint) if manifest else []
    checkpoint = checkpoints[0] if checkpoints else None

    work = Path(tempfile.mkdtemp(prefix="opendpd-deploy-"))
    build = Path(tempfile.mkdtemp(prefix="opendpd-deploy-build-"))
    try:
        cases = golden_inputs(tm.x_test, spec)
        golden = write_golden(q, cases, work / "golden")
        verification, seconds = c_backend.verify(q, [(c[0], c[2], c[3]) for c in cases], build)
        (work / "c").mkdir()
        for name in c_backend.SOURCES:                          # the sources only; binaries and replays stay out
            shutil.copyfile(build / name, work / "c" / name)
        deltas, profile_id = quality_loss(tm, q)
        storage = q.storage_bytes()
        measured = None
        if verification.status == "bit_exact" and seconds.get("long_sequence"):
            measured = MeasuredExecution(what="the compiled C99 reference replaying the long_sequence golden vector, single thread",
                                         samples_per_second=LONG_SEQUENCE / seconds["long_sequence"], machine=_machine())
        report = FixedPointReport(
            quality_loss=deltas, metric_profile_id=profile_id,
            resources=ResourceEstimate(mac_per_sample=q.mac_per_sample, table_lookups_per_sample=q.table_lookups_per_sample, **storage),
            measured_execution=measured, synthesis_estimate=None, measured_power=None,
            execution_assumptions=list(EXECUTION_ASSUMPTIONS))
        (work / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        (work / "weights.json").write_text(json.dumps(_weights_json(q)), encoding="utf-8")
        files = {str(p.relative_to(work)).replace("\\", "/"): sha256_bytes(p.read_bytes()) for p in sorted(work.rglob("*")) if p.is_file()}
        deployment = DeploymentManifest(
            spec=spec, run_id=run_id, model_key=record.model_key or "gru", weights_sha256=checkpoint.file.sha256 if checkpoint else None,
            hidden_size=q.hidden, tensors=q.tensors, golden=golden, verification=verification, report=report, files=files,
            software=software_provenance())
        (work / "README.md").write_text(report_markdown(deployment), encoding="utf-8")
        deployment = deployment.model_copy(update={"files": {**files, "README.md": sha256_bytes((work / "README.md").read_bytes())}})
        (work / "manifest.json").write_text(deployment.model_dump_json(indent=2), encoding="utf-8")
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(work.rglob("*")):
                if p.is_file():
                    zf.write(p, str(p.relative_to(work)).replace("\\", "/"))
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(build, ignore_errors=True)
    return deployment


def read_manifest(path: Path) -> DeploymentManifest:
    with zipfile.ZipFile(path) as zf:
        return DeploymentManifest.model_validate_json(zf.read("manifest.json"))


def _num(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def report_markdown(m: DeploymentManifest) -> str:
    r, v = m.report, m.verification
    out = [f"# Deployment package `{m.spec.spec_id}` for run {m.run_id}", "",
           f"Model {m.model_key} (hidden {m.hidden_size}) executed as {m.spec.model_key}; weights {(m.weights_sha256 or '')[:12]}.",
           f"Software: opendpd {m.software.opendpd_version}, torch {m.software.torch_version}.", "",
           "## Verification", "",
           f"Backend `{v.backend}`: **{v.status}** ({v.detail})" + (f"; compiler `{v.compiler}`" if v.compiler else "") + ".",
           "", "## Golden vectors", "", "| case | samples | resets | description | output sha256 |", "|---|---:|---|---|---|"]
    out += [f"| {g.case_id} | {g.n_samples} | {g.resets_at or '-'} | {g.description} | {g.output_sha256[:12]} |" for g in m.golden]
    out += ["", f"## Float to fixed: quality loss (profile {r.metric_profile_id}, test split)", "",
            "| metric | float | fixed | fixed - float |", "|---|---:|---:|---:|"]
    out += [f"| {d.name} ({d.unit}) | {_num(d.float_value)} | {_num(d.fixed_value)} | {_num(d.delta)} |" for d in r.quality_loss]
    res = r.resources
    out += ["", "## Resources", "", f"Label **{res.label}** (from the specification and the shapes; nothing measured):", "",
            f"- MAC per sample: {res.mac_per_sample}; table lookups per sample: {res.table_lookups_per_sample}",
            f"- weights {res.weight_bytes} bytes, biases {res.bias_bytes} bytes, state {res.state_bytes} bytes, tables {res.table_bytes} bytes",
            f"- sparsity: {res.sparsity}", ""]
    if r.measured_execution:
        me = r.measured_execution
        out += [f"Label **{me.label}**: {me.samples_per_second:,.0f} samples/s, {me.what} ({me.machine.get('cpu', '?')}). "
                "A property of this build on this machine, not of a deployment.", ""]
    out += [f"Label **synthesis_estimate**: {r.synthesis_estimate or 'not available (nothing was synthesised)'}.",
            f"Label **measured_power**: {r.measured_power or 'not available (nothing was measured); energy is never inferred from MAC or parameter counts'}.",
            "", "## Execution assumptions", ""] + [f"- {a}" for a in r.execution_assumptions]
    out += ["", "## Weight formats", "", "| tensor | shape | bits | fraction | max abs (float) | saturated |", "|---|---|---:|---:|---:|---:|"]
    out += [f"| {t.name} | {t.shape} | {t.bits} | {t.frac} | {t.max_abs_float:.4g} | {t.saturated} |" for t in m.tensors]
    out += ["", "Files and hashes are listed in `manifest.json`; `c/` holds the generated reference and the harness "
            "(`cc -std=c99 -O2 -o harness gru_fixed.c harness.c`; `./harness golden/<case>/x.i16 <n> y.out h.out [reset ...]`).", ""]
    return "\n".join(out)
