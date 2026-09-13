"""Read stored RF metrics alongside implementation-specific, sourced costs."""
from __future__ import annotations

from pathlib import Path

from opendpd.core.metrics import get_profile
from opendpd.schemas import ArtifactKind, ProfileValidation
from opendpd.schemas.common import FileRef, MetricStatus
from opendpd.schemas.hardware import (CostAttachment, CostValues, HardwareCostDraft, HardwareCostEntry,
                                     HardwareCostReport, PrecisionModule)
from opendpd.services import experiments
from opendpd.services.review import object_sha
from opendpd.services.workspace import WorkspaceError, read_json, sha256_bytes, sha256_file, write_json_atomic

NOTE = ('Costs retain their source categories. CPU reference timing, FPGA synthesis, board measurements, ASIC synthesis, '
        'post-layout simulation and chip measurements are different evidence. No power is inferred from parameter count or execution time.')


def directory(ws):
    result = ws.root / 'hardware-costs'
    result.mkdir(exist_ok=True)
    return result


def result_of(ws, run_id, profile_id):
    try:
        profile = get_profile(profile_id)
    except KeyError as exc:
        raise WorkspaceError(str(exc)) from None
    if profile.validation == ProfileValidation.pending_cross_validation:
        raise WorkspaceError('This metric profile is pending independent validation; it is unavailable for hardware trade-offs.')
    result = experiments.load_result(ws, run_id, profile_id)
    if result is None:
        raise WorkspaceError('The selected run has no saved result under this metric profile.')
    model = next((m for m in result.models if m.role == ('dpd' if any(x.role == 'dpd' for x in result.models) else 'pa')), None)
    if model is None or not model.weights_sha256:
        raise WorkspaceError('The result has no full model checkpoint hash for binding cost evidence.')
    return result, model


def _checkpoint(ws, result, model):
    run_id = model.run_id or result.run_id
    manifest = experiments.load_artifacts(ws, run_id)
    checkpoint = next((a for a in manifest.by_kind(ArtifactKind.checkpoint) if a.file.sha256 == model.weights_sha256), None) if manifest else None
    if checkpoint is None:
        raise WorkspaceError('The cost source checkpoint is unavailable.')
    path = ws.run_dir(run_id) / checkpoint.file.path
    if not path.is_file() or sha256_file(path) != model.weights_sha256:
        raise WorkspaceError('The cost source checkpoint changed or is missing.')
    return path, FileRef(path=f'runs/{run_id}/{checkpoint.file.path}', sha256=model.weights_sha256, size_bytes=path.stat().st_size)


def automatic(ws, run_id, profile_id):
    import torch
    result, model = result_of(ws, run_id, profile_id)
    path, source = _checkpoint(ws, result, model)
    tensors = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(tensors, dict) or not all(isinstance(t, torch.Tensor) for t in tensors.values()):
        raise WorkspaceError('The checkpoint is not a tensor state dictionary.')
    resolved = experiments.load_resolved(ws, model.run_id or run_id)
    q = resolved.quantization
    constants = sum(t.numel() * t.element_size() for t in tensors.values())
    # These models execute each affine matrix once per sample. Nonlinear and
    # elementwise work is deliberately itemised as uncounted, not called free.
    affine = model.model.key in {'gru', 'gru_stream', 'qgru', 'qgru_amp1'}
    macs = sum(t.numel() for key, t in tensors.items() if t.ndim == 2 and ('weight' in key) and 'quantizer' not in key) if affine else None
    parameters = model.model.parameters
    state = int(parameters.get('hidden_size', 0)) * int(parameters.get('num_layers', 1)) * 4 if affine else None
    precision = [PrecisionModule(module='checkpoint tensors', format=', '.join(sorted({str(t.dtype).replace('torch.', '') for t in tensors.values()}))),
                 PrecisionModule(module='features / nonlinearities', format='FP32 software operations; no LUT/approximation hardware specified' if affine else 'operator precision not audited; tensor dtype alone does not establish feature arithmetic')]
    if q and q.enabled:
        precision += [PrecisionModule(module='QAT weight / activation simulation', format=f'{q.n_bits_w} / {q.n_bits_a} bits, stored and executed as float tensors'),
                      PrecisionModule(module='hardware accumulator', format='unspecified; QAT does not define an integer accumulator')]
    lookahead = result.execution.lookahead_samples if result.execution else model.lookahead_samples
    fs = result.evaluated_signal.sample_rate_hz if result.evaluated_signal else None
    notes = ['Model constants include every serialized tensor (weights, biases, quantizer scales/buffers); archive overhead, code and runtime allocations are excluded.',
             'Stored parameter count is recorded separately; sparse/effective counts never replace stored tensors.',
             'MAC/sample counts only dense affine matrices for GRU/QGRU. Bias, gating, feature extraction and nonlinear functions are additional work.',
             'No dynamic skipping is assumed or measured. Missing operator counts remain unavailable.',
             'Lookahead/Fs is an algorithmic information bound; it is not warm-up, queue time, throughput or end-to-end latency.']
    if not affine:
        notes.append(f'No reviewed per-sample operation counter for {model.model.key}; storage remains exact for the tensor checkpoint.')
    dataset = ws.get_dataset(result.dataset.dataset_id)
    entry = HardwareCostEntry(entry_id='', run_id=run_id, profile_id=profile_id, title=f'{model.model.key}: stored tensors / affine operation count',
        source_kind='operation_count', source_type='checkpoint_shapes', source_file=source,
        result_sha256=object_sha(result), weights_sha256=model.weights_sha256,
        values=CostValues(constant_bytes=constants, state_bytes=state, mac_per_sample=macs), precision=precision,
        target='PyTorch reference / shape-based operation estimate', activity='dense affine execution; other operators not counted',
        boundary='model constants and recurrent state only; excludes RF chain, host runtime allocation and transport buffers',
        batch_size=1, metrics=result.metrics, metric_basis='stored result under its original execution semantics',
        rf_evidence_type=result.evidence_type.value, execution_semantics=model.execution_semantics,
        synthetic=result.is_mock or dataset.origin.value == 'synthetic', stored_parameter_count=model.n_parameters,
        stored_tensor_elements=sum(t.numel() for t in tensors.values()), lookahead_samples=lookahead,
        lookahead_lower_bound_s=(lookahead / fs if lookahead is not None and fs else None),
        warmup_samples=result.execution.warmup_samples if result.execution else None, limitations=notes)
    entry.entry_id = 'cost-' + object_sha(entry.model_dump(mode='json', exclude={'entry_id', 'created_at'}))
    return entry


def save_entry(ws, entry):
    entry.entry_id = 'cost-' + object_sha(entry.model_dump(mode='json', exclude={'entry_id', 'created_at', 'stale'}))
    path = directory(ws) / (entry.entry_id + '.json')
    if not path.exists():
        write_json_atomic(path, entry)
    return HardwareCostEntry.model_validate(read_json(path))


def attachment(ws, filename, data):
    suffix = Path(filename).suffix.lower()
    if not data or len(data) > 5 * 1024 * 1024 or suffix not in {'.pdf', '.json', '.txt', '.rpt', '.csv'}:
        raise WorkspaceError('Attach a non-empty PDF, JSON, text, report or CSV file of at most 5 MiB.')
    if suffix == '.pdf':
        if not data.startswith(b'%PDF-'):
            raise WorkspaceError('The report is not a PDF file.')
    else:
        try:
            text = data.decode('utf-8-sig')
            if '\x00' in text:
                raise ValueError()
            if suffix == '.json':
                import json
                json.loads(text)
        except (ValueError, UnicodeError):
            raise WorkspaceError('The report must contain valid UTF-8 text or JSON.') from None
    digest = sha256_bytes(data)
    folder = directory(ws) / 'attachments'
    folder.mkdir(exist_ok=True)
    path = folder / f'{digest}{suffix}'
    if not path.exists():
        path.write_bytes(data)
    return CostAttachment(sha256=digest, size_bytes=len(data), filename=path.name)


def reported(ws, draft: HardwareCostDraft):
    result, model = result_of(ws, draft.run_id, draft.profile_id)
    candidates = list((directory(ws) / 'attachments').glob(f'{draft.report_sha256}.*'))
    source = next((p for p in candidates if p.is_file() and not p.is_symlink() and sha256_file(p) == draft.report_sha256), None)
    if source is None:
        raise WorkspaceError('Upload the supporting report before recording its costs.')
    _checkpoint(ws, result, model)
    return save_entry(ws, HardwareCostEntry(entry_id='', run_id=draft.run_id, profile_id=draft.profile_id, title=draft.title,
        source_kind=draft.source_kind, source_type='user_report', source_file=FileRef(path=source.relative_to(ws.root).as_posix(), sha256=draft.report_sha256, size_bytes=source.stat().st_size),
        result_sha256=object_sha(result), weights_sha256=model.weights_sha256, values=draft.values, precision=draft.precision,
        target=draft.target, process=draft.process, clock_hz=draft.clock_hz, batch_size=draft.batch_size, activity=draft.activity,
        boundary=draft.boundary, metrics=result.metrics, metric_basis='linked saved RF result; performance on the reported implementation is not independently verified',
        rf_evidence_type=result.evidence_type.value, execution_semantics=model.execution_semantics,
        synthetic=draft.synthetic or result.is_mock or ws.get_dataset(result.dataset.dataset_id).origin.value == 'synthetic',
        limitations=['User-transcribed values bound to the full report and checkpoint hashes; Studio has not independently validated the report.', NOTE,
                     'RF metrics come from the linked result; attaching a hardware cost report does not turn surrogate performance into measured hardware performance.', draft.notes]))


def record_deployment(ws, run_id, deployment, archive):
    """Reuse the existing verified report; do not run another cost or metric algorithm."""
    result, model = result_of(ws, run_id, deployment.report.metric_profile_id)
    if deployment.weights_sha256 != model.weights_sha256:
        raise WorkspaceError('Deployment cost report does not match the result weights.')
    # Keep a small immutable copy in the ledger so moving/deleting the ZIP does
    # not make the calculation basis disappear.
    payload = deployment.model_dump_json(indent=2).encode()
    receipt = attachment(ws, 'deployment.json', payload)
    source = FileRef(path=f'hardware-costs/attachments/{receipt.filename}', sha256=receipt.sha256, size_bytes=receipt.size_bytes)
    resources = deployment.report.resources
    fixed_metrics = [result.metric(m.name).model_copy(update={'value': m.fixed_value, 'unit': m.unit, 'status': MetricStatus.ok, 'reason': None})
                     for m in deployment.report.quality_loss if m.fixed_value is not None]
    spec = deployment.spec
    precision = [PrecisionModule(module='weights', format=f'{spec.weight_bits}-bit per-tensor fixed point'),
        PrecisionModule(module='input / hidden state / output', format=f'{spec.x.bits} / {spec.h.bits} / {spec.y.bits} bits'),
        PrecisionModule(module='accumulator / pre-activation', format=f'{spec.accumulator_bits} / {spec.pre.bits} bits'),
        PrecisionModule(module='sigmoid / tanh', format=f'specified lookup tables: {spec.sigmoid.entries} / {spec.tanh.entries} entries')]
    common = dict(entry_id='', run_id=run_id, profile_id=deployment.report.metric_profile_id, source_type='studio_fixed_point_report', source_file=source,
        result_sha256=object_sha(result), weights_sha256=model.weights_sha256, precision=precision,
        metrics=fixed_metrics, metric_basis=f'fixed-point-v1 quality_loss fixed values / gru_stream; C reference verification: {deployment.verification.status}',
        rf_evidence_type=result.evidence_type.value, execution_semantics='streaming_stateful / fixed-point-v1',
        synthetic=result.is_mock or ws.get_dataset(result.dataset.dataset_id).origin.value == 'synthetic',
        target='fixed-point-v1 GRU streaming reference', batch_size=1, activity='sequential, dense MACs; no skipping',
        boundary='digital model only; input already in spec.x, analogue chain and I/Q calibration excluded',
        lookahead_samples=0, lookahead_lower_bound_s=0,
        limitations=[*deployment.report.execution_assumptions, NOTE, f'Deployment archive SHA256: {sha256_file(archive)}',
            'Persistent constants sum weights, biases and LUTs; state is separate. Code, format descriptors and temporary working buffers are excluded.'])
    save_entry(ws, HardwareCostEntry(**common, title='Fixed-point GRU: specification-derived resources', source_kind='operation_count',
        values=CostValues(constant_bytes=resources.weight_bytes + resources.bias_bytes + resources.table_bytes,
            state_bytes=resources.state_bytes, table_bytes=resources.table_bytes, mac_per_sample=resources.mac_per_sample,
            table_lookups_per_sample=resources.table_lookups_per_sample, dynamic_skip_fraction=0)))
    timing = deployment.report.measured_execution
    if timing:
        save_entry(ws, HardwareCostEntry(**{**common, 'target': str(timing.machine), 'boundary': timing.what},
            title='C99 reference: host throughput measurement', source_kind='cpu_reference_timing',
            values=CostValues(throughput_samples_s=timing.samples_per_second)))


def report(ws, run_ids, profile_id):
    if not 1 <= len(set(run_ids)) <= 8:
        raise WorkspaceError('Select one to eight distinct runs for hardware trade-offs.')
    entries, missing = [], {}
    for run_id in dict.fromkeys(run_ids):
        try:
            entries.append(automatic(ws, run_id, profile_id))
        except WorkspaceError as exc:
            missing[run_id] = str(exc)
    current = {e.run_id: e for e in entries}
    for path in sorted(directory(ws).glob('cost-*.json')):
        entry = HardwareCostEntry.model_validate(read_json(path))
        if entry.run_id not in run_ids or entry.profile_id != profile_id:
            continue
        source = ws.root / entry.source_file.path
        entry.stale = (entry.run_id not in current or entry.result_sha256 != current[entry.run_id].result_sha256
                       or entry.weights_sha256 != current[entry.run_id].weights_sha256 or not source.is_file()
                       or sha256_file(source) != entry.source_file.sha256)
        entries.append(entry)
    differences = []
    if len(current) > 1:
        from opendpd.services.evaluation import compare_results
        comparison = compare_results(ws, list(current), profile_id)
        differences += [f'{pair.a} / {pair.b}: {"; ".join(pair.incompatibilities)}' for pair in comparison.pairs if pair.incompatibilities]
    if len({e.execution_semantics for e in entries}) > 1:
        differences.append('Cost entries include different execution semantics (for example offline float and streaming fixed point). Their RF metrics cannot be ranked as one protocol.')
    return HardwareCostReport(entries=entries, missing=missing, comparison_notes=differences, notes=[NOTE,
        'RF comparisons still require matching evidence, profile, dataset/conditions and execution semantics. This plot does not rank unlike experiments.',
        'Energy axes require a supplied power/energy report with a declared implementation boundary. No energy model is inferred.',
        'Synthetic entries and stale source records are explicitly labelled; stale entries cannot supply scatter points.'])
