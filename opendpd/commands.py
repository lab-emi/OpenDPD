"""``opendpd`` sub-commands (Studio entry point).

Heavy imports happen inside each command so ``opendpd --help`` is instant.
Exit codes: 0 ok, 1 run failed, 2 invalid input / configuration, 130 cancelled.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import List, Optional


def _print_json(data) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def cmd_models(args) -> int:
    from opendpd.core.registry import list_models

    models = list_models()
    if args.json:
        _print_json([m.to_dict() for m in models])
        return 0
    print(f"{'key':<16} {'status':<13} {'roles':<8} {'tested on':<10} look-ahead  display name")
    for m in models:
        la = "n/a" if m.lookahead_samples is None else str(m.lookahead_samples)
        print(f"{m.key:<16} {m.status:<13} {'/'.join(m.roles):<8} {','.join(m.devices_tested):<10} {la:<11} {m.display_name}")
    return 0


def cmd_recipes(args) -> int:
    from opendpd.services.recipes import list_recipes

    recipes = list_recipes()
    if args.json:
        _print_json([r.to_dict() for r in recipes])
        return 0
    for r in recipes:
        print(f"{r.recipe_id:<32} [{r.purpose}] {r.title}\n    {r.description}\n    limits: {r.limits}\n    expected: {r.expected_duration}")
    return 0


def _signal_from_args(args):
    from opendpd.schemas import SignalSpec

    fields = {"sample_rate_hz": args.fs, "bandwidth_hz": args.bandwidth, "n_sub_ch": args.n_sub_ch,
              "nperseg": args.nperseg, "sub_channel_bandwidth_hz": args.bw_sub_ch}
    spec = {k: v for k, v in fields.items() if v is not None}
    if args.units:
        spec["amplitude_units"] = args.units
    return SignalSpec(**spec)


def _print_report(report) -> None:
    print(f"{report.doctor_version}  {report.report_id}  blocked={report.evaluation_blocked}")
    for item in report.items:
        conf = f" (confidence {item.confidence:.2f})" if item.confidence is not None else ""
        print(f"  [{item.severity.value:7}] {item.code}: {item.message}{conf}")
        if item.suggestion:
            print(f"            -> {item.suggestion}")


def cmd_datasets(args) -> int:
    from opendpd.services.workspace import Workspace, WorkspaceError

    try:
        ws = Workspace.open_or_create(Path(args.workspace))
        if args.datasets_command == "import-builtin":
            manifest = ws.register_builtin_dataset(args.name, dataset_id=args.id)
            print(f"registered '{manifest.dataset_id}' ({manifest.display_name}) in {ws.root}")
            return 0
        if args.datasets_command == "import":
            from opendpd.schemas import DatasetOrigin
            from opendpd.services import datasets as ds

            mapping = dict(pair.split("=", 1) for pair in (args.map or []))
            manifest = ds.import_dataset(ws, Path(args.path), dataset_id=args.id, display_name=args.name, mapping=mapping,
                                         signal=_signal_from_args(args), origin=DatasetOrigin(args.origin),
                                         guard_samples=args.guard,
                                         waveform=Path(args.waveform) if args.waveform else None)
            if args.json:
                _print_json(manifest.model_dump(mode="json"))
            else:
                print(f"imported '{manifest.dataset_id}' ({manifest.n_samples} samples) from {args.path}")
                if manifest.signal.waveform is not None:
                    b = manifest.signal.waveform
                    print(f"bound to waveform {b.spec.waveform_id} seed {b.spec.seed}: offset {b.input_offset_samples} "
                          f"samples at the waveform clock, correlation {b.correlation:.3f}")
                missing = manifest.missing_metadata()
                if missing:
                    print("metadata still missing for evaluation: " + ", ".join(missing))
            return 0
        if args.datasets_command == "doctor":
            from opendpd.services import datasets as ds

            report = ds.run_doctor(ws, args.id, args.version)
            if args.json:
                _print_json(report.model_dump(mode="json"))
            else:
                _print_report(report)
            return 0 if not report.evaluation_blocked else 1
        if args.datasets_command == "preprocess":
            from opendpd.schemas import PreprocessingParams
            from opendpd.services import datasets as ds

            params = PreprocessingParams(delay_samples=args.delay, gain_db=args.gain_db, phase_deg=args.phase_deg,
                                         interpolate_non_finite=args.interpolate_non_finite,
                                         remove_outliers=args.remove_outliers,
                                         normalize="peak_input" if args.normalize else "none")
            if args.preview:
                preview = ds.preview_preprocess(ws, args.id, params, args.base)
                if args.json:
                    _print_json(preview)
                else:
                    print(f"{preview['n_samples_before']} -> {preview['n_samples_after']} samples; steps: {preview['record']['steps']}")
                    from opendpd.schemas import DiagnosticReport
                    _print_report(DiagnosticReport.model_validate(preview["report_after"]))
                return 0
            version = ds.create_version(ws, args.id, args.version, params, args.base)
            if args.json:
                _print_json(version.model_dump(mode="json"))
            else:
                print(f"created version '{version.version}' of '{args.id}': {version.n_samples} samples, "
                      f"fit range {version.fit_range}, code {version.code_version}")
            return 0
        if args.datasets_command == "add-root":
            ws.add_import_root(args.name, Path(args.path))
            print(f"import roots: {', '.join(f'{k}={v}' for k, v in ws.import_roots().items())}")
            return 0
        datasets = ws.list_datasets()
        if args.json:
            _print_json([d.model_dump(mode="json") for d in datasets])
        elif not datasets:
            print("no datasets registered; try: opendpd datasets import-builtin DPA_200MHz --workspace ...")
        else:
            for d in datasets:
                missing = d.missing_metadata()
                print(f"{d.dataset_id:<24} {d.origin.value:<9} {d.n_samples or '?':>8} samples  "
                      f"{'metadata missing: ' + ','.join(missing) if missing else 'evaluable'}")
        return 0
    except WorkspaceError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


def _load_config(args):
    from opendpd.schemas import ExperimentConfig
    from opendpd.services.recipes import instantiate

    if args.config:
        return ExperimentConfig.model_validate(json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.recipe:
        if not args.dataset:
            raise ValueError("--dataset is required with --recipe")
        return instantiate(args.recipe, args.dataset, pa_run_id=args.pa_run, device=args.device,
                           seed=args.seed, name=args.name)
    raise ValueError("provide --config PATH or --recipe ID --dataset ID")


def cmd_validate(args) -> int:
    from opendpd.services.config import validate

    try:
        config = _load_config(args)
    except Exception as err:  # noqa: BLE001 - user input
        print(f"error: {err}", file=sys.stderr)
        return 2
    report = validate(config)
    if args.json:
        _print_json(report.to_dict())
    else:
        for w in report.warnings:
            print(f"warning: {w.field}: {w.message}")
        for e in report.errors:
            print(f"error: {e.field}: {e.message}" + (f" ({e.hint})" if e.hint else ""))
        if report.ok:
            print(f"ok: resolved config sha256 {report.resolved.resolution.config_sha256}")
    return 0 if report.ok else 2


def cmd_run(args) -> int:
    from opendpd.schemas import RunStatus
    from opendpd.services.config import ConfigError
    from opendpd.services.experiments import create_run, execute_run, load_result
    from opendpd.services.workspace import Workspace, WorkspaceError

    try:
        config = _load_config(args)
        ws = Workspace.open_or_create(Path(args.workspace))
        record = create_run(ws, config, name=args.name, idempotency_key=args.idempotency_key)
    except ConfigError as err:
        for issue in err.issues:
            print(f"error: {issue.field}: {issue.message}" + (f" ({issue.hint})" if issue.hint else ""),
                  file=sys.stderr)
        return 2
    except (WorkspaceError, ValueError, FileNotFoundError, KeyError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    text = sys.stderr if args.json else sys.stdout      # with --json, stdout carries the JSON document only
    if record.status != RunStatus.queued:
        print(f"run {record.run_id} already exists for idempotency key {args.idempotency_key!r} "
              f"(status {record.status.value})", file=text)
    else:
        print(f"run {record.run_id} created in {ws.run_dir(record.run_id)}", file=text)
        with contextlib.redirect_stdout(text):        # the legacy trainer prints progress to stdout
            record = execute_run(ws, record.run_id)
        print(f"run {record.run_id} {record.status.value}", file=text)
    if record.error:
        print(f"  {record.error.code} [{record.error.stage}]: {record.error.message}", file=sys.stderr)
    result = load_result(ws, record.run_id)
    if result is not None:
        _print_result(result, file=text)
    if args.json:
        _print_json({"run": record.model_dump(mode="json"),
                     "result": result.model_dump(mode="json") if result else None})
    return {RunStatus.succeeded: 0, RunStatus.cancelled: 130}.get(record.status, 1)


def _print_result(result, file=None) -> None:
    file = file or sys.stdout
    print(f"  evidence: {result.evidence_type.value}  profile: {result.metric_profile_id} v{result.metric_profile_version}",
          file=file)
    for m in result.metrics:
        shown = f"{m.value:.4f} {m.unit}" if m.value is not None else f"{m.status.value}: {m.reason}"
        print(f"  {m.name:<9} {shown}", file=file)
    for lim in result.limitations:
        print(f"  note: {lim}", file=file)


def cmd_profiles(args) -> int:
    from opendpd.core.metrics import list_profiles

    profiles = list_profiles()
    if args.json:
        _print_json([p.model_dump(mode="json") for p in profiles])
        return 0
    for p in profiles:
        flags = (" (frozen)" if p.frozen else "") + f" [validation: {p.validation.value}]"
        print(f"{p.profile_id} v{p.version}{flags}: {p.description}")
        for m in p.metrics:
            print(f"  {m.name:<9} {m.unit:<4} {m.better.value:<6} {m.formula}")
    return 0


def cmd_waveforms(args) -> int:
    from opendpd.core.waveforms import generate, read_package, write_package
    from opendpd.schemas import WaveformSpec

    if args.waveforms_command == "generate":
        spec = WaveformSpec(seed=args.seed, n_subframes=args.subframes)
        wf = generate(spec)
        path = write_package(wf, Path(args.out))
        if args.json:
            _print_json({"path": str(path), "spec": spec.model_dump(mode="json"), "sha256": wf.sha256(),
                         "n_samples": wf.period})
        else:
            print(f"{spec.waveform_id} seed {spec.seed}, {spec.n_subframes} subframes ({wf.period} samples at "
                  f"{spec.sample_rate_hz / 1e6:.2f} MS/s) written to {path.parent}; package sha256 {wf.sha256()[:12]}")
        return 0
    try:
        spec, digest = read_package(Path(args.path))
    except (OSError, ValueError, KeyError) as err:
        print(f"error: {args.path} is not a readable waveform package: {err}", file=sys.stderr)
        return 2
    regenerated = generate(spec).sha256()
    if args.json:
        _print_json({"spec": spec.model_dump(mode="json"), "sha256": digest, "regenerated_sha256": regenerated,
                     "matches": digest == regenerated, "n_samples": spec.n_samples, "n_symbols": spec.n_symbols})
    else:
        print(f"{spec.waveform_id} v{spec.version}: seed {spec.seed}, {spec.n_subframes} subframes, {spec.n_samples} samples at "
              f"{spec.sample_rate_hz / 1e6:.2f} MS/s, {spec.occupied_subcarriers} subcarriers x {spec.n_symbols} symbols, "
              f"{spec.modulation}, {spec.cyclic_prefix} cyclic prefix")
        print(f"package sha256 {digest[:12]}; regenerated {regenerated[:12]} ({'match' if digest == regenerated else 'MISMATCH'})")
    return 0 if digest == regenerated else 1


def cmd_measurements(args) -> int:
    """Import operator-provided captures of a physical PA driven by a run_dpd export and score them (S16)."""
    import contextlib

    from pydantic import ValidationError

    from opendpd.schemas import CaptureRef, MeasurementConditions, RunStatus
    from opendpd.services.config import ConfigError
    from opendpd.services.experiments import create_run, execute_run, load_result
    from opendpd.services.measurements import measurement_config, stage_capture
    from opendpd.services.workspace import Workspace, WorkspaceError

    try:
        ws = Workspace.open(Path(args.workspace))
        conditions = MeasurementConditions.model_validate(json.loads(Path(args.conditions).read_text()))
        columns = tuple(args.columns.split(",")) if args.columns else None
        with_dpd = CaptureRef(path=stage_capture(ws, Path(args.with_dpd)), columns=columns,
                              declared_output_power_dbm=args.power_with)
        without_dpd = CaptureRef(path=stage_capture(ws, Path(args.without_dpd)), columns=columns,
                                 declared_output_power_dbm=args.power_without) if args.without_dpd else None
        config = measurement_config(ws, args.apply_run, with_dpd=with_dpd, without_dpd=without_dpd, conditions=conditions,
                                    source="mock_adapter" if args.mock else "manual", playback=args.playback,
                                    profile_id=args.profile, name=args.name)
        record = create_run(ws, config)
    except ConfigError as err:
        for issue in err.issues:
            print(f"error: {issue.field}: {issue.message}" + (f" ({issue.hint})" if issue.hint else ""), file=sys.stderr)
        return 2
    except (WorkspaceError, ValidationError, ValueError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    with contextlib.redirect_stdout(sys.stderr):
        record = execute_run(ws, record.run_id)
    result = load_result(ws, record.run_id)
    if args.json:
        _print_json({"run": record.model_dump(mode="json"), "result": result.model_dump(mode="json") if result else None})
    else:
        print(f"run {record.run_id} {record.status.value}")
        if record.error:
            print(f"  {record.error.code} [{record.error.stage}]: {record.error.message}", file=sys.stderr)
        if result is not None:
            m = result.measurement
            print(f"  {m.attestation}")
            for c in m.captures:
                print(f"  {c.role.replace('_', ' ')}: delay {c.delay_samples} samples, correlation {c.correlation:.3f}, "
                      f"gain {c.gain_db:+.2f} dB at {c.gain_phase_deg:+.1f} deg, rms {c.rms:.4g} (capture units)")
            if m.level_difference_db is not None:
                print(f"  output level with DPD relative to without: {m.level_difference_db:+.2f} dB")
            _print_result(result)
            for b in result.baselines:
                print(f"  baseline {b.kind}: " + ", ".join(f"{x.name} {x.value:.4f}" if x.value is not None
                                                            else f"{x.name} {x.status.value}" for x in b.metrics))
    return {RunStatus.succeeded: 0, RunStatus.cancelled: 130}.get(record.status, 1)


def cmd_instruments(args) -> int:
    """Instrument adapters (S16): list them, or run the export → play → capture procedure through one (mock only)."""
    from opendpd.instruments import ADAPTERS, InstrumentError, list_adapters, run_capture_session

    if args.instruments_command == "list":
        infos = list_adapters()
        if args.json:
            _print_json([i.model_dump(mode="json") for i in infos])
        else:
            for i in infos:
                rf = "emits RF (needs OPENDPD_ALLOW_RF_OUTPUT=1 and --arm)" if i.rf_output_capable else "no RF output"
                print(f"{i.adapter_id:<8} {i.kind:<5} {rf}; {i.description}; limits: peak {i.default_limits.max_peak_abs:g}, "
                      f"timeout {i.default_limits.timeout_s:g} s, link {i.default_limits.link_timeout_s:g} s")
        return 0

    from datetime import datetime, timezone

    from opendpd.core.measurement import to_iq
    from opendpd.schemas import ArtifactKind
    from opendpd.services.experiments import load_artifacts, load_resolved
    from opendpd.services.measurements import read_played
    from opendpd.services.workspace import Workspace, WorkspaceError

    try:
        ws = Workspace.open(Path(args.workspace))
        manifest = load_artifacts(ws, args.apply_run)
        played = manifest.by_kind(ArtifactKind.dpd_output) if manifest else []
        if not played:
            raise WorkspaceError(f"run '{args.apply_run}' has no dpd-output export (it must be a succeeded run_dpd run)")
        x, u = read_played(ws.run_dir(args.apply_run) / played[0].file.path)
        dataset = ws.get_dataset(load_resolved(ws, args.apply_run).dataset.id)
    except WorkspaceError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    fs = dataset.signal.sample_rate_hz or 1.0
    adapter = ADAPTERS[args.adapter]()
    out = Path(args.out)
    if not args.arm:
        print(f"RF output stays off: nothing was played. A person arms the session with --arm <operator name>; "
              f"adapter {adapter.info.adapter_id} ({adapter.info.kind}).", file=sys.stderr)
        return 2
    try:
        with_path = run_capture_session(adapter, to_iq(u), fs, operator=args.arm, out=out / "with_dpd.npy",
                                        requested_power_dbm=args.requested_power)
        without_path = run_capture_session(adapter, to_iq(x), fs, operator=args.arm, out=out / "without_dpd.npy",
                                           requested_power_dbm=args.requested_power)
    except InstrumentError as err:
        print(f"aborted (RF output off): {err}", file=sys.stderr)
        return 1
    conditions = {"pa": adapter.info.description if adapter.info.kind == "mock" else "fill in: device under test",
                  "capture_chain": f"{adapter.info.adapter_id} adapter: generator -> PA -> analyser"
                                   + (" (dry run, no RF)" if adapter.info.kind == "mock" else ""),
                  "sample_rate_hz": fs, "drive": "digital full scale 1.0 into the adapter", "calibration": "none",
                  "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "operator": args.arm,
                  "notes": "written by opendpd instruments dry-run; edit before importing a real measurement"}
    (out / "conditions.json").write_text(json.dumps(conditions, indent=2))
    mock_flag = " --mock" if adapter.info.kind == "mock" else ""
    next_command = (f"opendpd measurements import --apply-run {args.apply_run} --with-dpd {with_path} --without-dpd "
                    f"{without_path} --conditions {out / 'conditions.json'}{mock_flag} --workspace {args.workspace}")
    if args.json:
        _print_json({"adapter": adapter.info.model_dump(mode="json"), "with_dpd": str(with_path),
                     "without_dpd": str(without_path), "conditions": str(out / "conditions.json"),
                     "next_command": next_command})
    else:
        print(f"captured through {adapter.info.adapter_id} ({adapter.info.kind}; RF output off again): {with_path}, {without_path}")
        print(f"conditions template: {out / 'conditions.json'}")
        print(f"next: {next_command}")
    return 0


def cmd_evaluate(args) -> int:
    from opendpd.services.evaluation import evaluate_run
    from opendpd.services.workspace import Workspace, WorkspaceError

    import contextlib

    from opendpd.core.metrics import get_profile
    from opendpd.schemas import ProfileValidation

    try:
        if get_profile(args.profile).validation == ProfileValidation.pending_cross_validation:
            print(f"note: profile {args.profile} is pending cross-validation; its numbers are not standard-conformance "
                  "results (docs/protocols/waveform-profiles.md)", file=sys.stderr)
        ws = Workspace.open(Path(args.workspace))
        with contextlib.redirect_stdout(sys.stderr):      # legacy model/loader chatter never pollutes the result
            result = evaluate_run(ws, args.run_id, args.profile)
    except (WorkspaceError, KeyError, FileNotFoundError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.json:
        _print_json(result.model_dump(mode="json"))
    else:
        print(f"run {args.run_id} re-evaluated under {args.profile}")
        _print_result(result)
    return 0


def cmd_apply(args) -> int:
    """Apply a trained DPD to the test split (export u = DPD(x)) and score it through a PA surrogate."""
    import contextlib

    from opendpd.schemas import ArtifactKind, RunStatus
    from opendpd.services.config import ConfigError
    from opendpd.services.experiments import create_run, execute_run, load_artifacts, load_result, load_run
    from opendpd.services.recipes import run_dpd_config
    from opendpd.services.workspace import Workspace, WorkspaceError

    try:
        ws = Workspace.open(Path(args.workspace))
        dpd = load_run(ws, args.dpd_run_id)
        config = run_dpd_config(dpd.dataset_id, args.dpd_run_id, pa_run_id=args.pa, device=args.device)
        record = create_run(ws, config)
    except ConfigError as err:
        for issue in err.issues:
            print(f"error: {issue.field}: {issue.message}" + (f" ({issue.hint})" if issue.hint else ""),
                  file=sys.stderr)
        return 2
    except (WorkspaceError, ValueError, KeyError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    with contextlib.redirect_stdout(sys.stderr):      # legacy step chatter never pollutes the JSON
        record = execute_run(ws, record.run_id)
    manifest = load_artifacts(ws, record.run_id)
    result = load_result(ws, record.run_id)
    if args.json:
        _print_json({"run": record.model_dump(mode="json"),
                     "artifacts": manifest.model_dump(mode="json") if manifest else None,
                     "result": result.model_dump(mode="json") if result else None})
    else:
        print(f"run {record.run_id} {record.status.value}")
        if record.error:
            print(f"  {record.error.code} [{record.error.stage}]: {record.error.message}", file=sys.stderr)
        for artifact in (manifest.by_kind(ArtifactKind.dpd_output) if manifest else []):
            print(f"  u = DPD(x) exported to {ws.run_dir(record.run_id) / artifact.file.path}")
            print("  (a pre-distorted PA *input*; not a PA output and not proof of linearisation)")
        if result is not None:
            _print_result(result)
    return {RunStatus.succeeded: 0, RunStatus.cancelled: 130}.get(record.status, 1)


def cmd_export(args) -> int:
    from opendpd.services.packages import PackageError, export_run
    from opendpd.services.workspace import Workspace, WorkspaceError

    try:
        ws = Workspace.open(Path(args.workspace))
        out = Path(args.out) if args.out else ws.exports_dir / f"{args.run_id}-{args.kind}.zip"
        manifest = export_run(ws, args.run_id, out, kind=args.kind)
    except PackageError as err:
        print(f"error: {err.code}: {err}" + (f" ({err.hint})" if err.hint else ""), file=sys.stderr)
        return 2
    except (WorkspaceError, FileNotFoundError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.json:
        _print_json({"path": str(out), "manifest": manifest.model_dump(mode="json")})
        return 0
    print(f"{manifest.kind} package written to {out} ({len(manifest.files)} files)")
    for line in manifest.redaction:
        print(f"  redacted: {line}")
    for line in manifest.missing:
        print(f"  not included: {line}")
    print(f"  import with: {manifest.reproduction['import']}")
    return 0


def cmd_import(args) -> int:
    from opendpd.services.packages import PackageError, import_package, inspect_package
    from opendpd.services.workspace import Workspace, WorkspaceError

    try:
        ws = Workspace.open_or_create(Path(args.workspace))
        if args.inspect:
            manifest = inspect_package(Path(args.path))
            if args.json:
                _print_json(manifest.model_dump(mode="json"))
            else:
                print(f"valid {manifest.kind} package: run {manifest.run_id} ({manifest.task.value}), "
                      f"{len(manifest.files)} files verified; dataset {manifest.dataset.dataset_id} "
                      f"{'included' if manifest.dataset.included else 'not included'}")
            return 0
        report = import_package(ws, Path(args.path))
    except PackageError as err:
        print(f"error: {err.code}: {err}" + (f" ({err.hint})" if err.hint else ""), file=sys.stderr)
        return 2
    except WorkspaceError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.json:
        _print_json(report.model_dump(mode="json"))
        return 0
    print(f"imported {', '.join(report.imported_runs)} into {ws.root}; dataset {report.dataset_id}: {report.dataset_status}")
    for line in report.missing:
        print(f"  missing: {line}")
    print(f"  {report.note}")
    print(f"  re-evaluate with: {report.evaluate_command}")
    return 0


def cmd_report(args) -> int:
    from opendpd.services.reports import report_html, report_markdown
    from opendpd.services.workspace import Workspace, WorkspaceError

    try:
        ws = Workspace.open(Path(args.workspace))
        body = report_markdown(ws, args.run_id) if args.format == "md" else report_html(ws, args.run_id)
    except (WorkspaceError, FileNotFoundError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.out:
        Path(args.out).write_text(body, encoding="utf-8")
        print(f"report written to {args.out}")
    else:
        print(body)
    return 0


def cmd_gui(args) -> int:
    from opendpd.studio.launcher import default_workspace, launch

    workspace = Path(args.workspace) if args.workspace else default_workspace()
    return launch(workspace, port=args.port, open_in_browser=not args.no_browser)


def cmd_doctor(args) -> int:
    from opendpd.studio.launcher import doctor

    return doctor(Path(args.workspace) if args.workspace else None)


def cmd_benchmark(args) -> int:
    from pydantic import ValidationError

    from opendpd.schemas import RunStatus
    from opendpd.services import benchmark as bm
    from opendpd.services.workspace import Workspace, WorkspaceError

    text = sys.stderr if getattr(args, "json", False) else sys.stdout
    try:
        if args.benchmark_command == "plan":
            seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else bm.DEFAULT_SEEDS
            plan = bm.make_plan(args.dataset, tier=args.tier, seeds=seeds, profile_id=args.profile, device=args.device,
                                preprocessing_version=args.version)
            bm.write_plan(plan, Path(args.out))
            print(f"plan {plan.plan_sha256[:12]} written to {args.out}: {len(plan.entries)} entries x "
                  f"{len(plan.seeds)} seeds, tier {plan.tier}, dataset {plan.dataset.id}")
            return 0
        if args.benchmark_command == "run":
            plan = bm.load_plan(Path(args.plan))
            ws = Workspace.open_or_create(Path(args.workspace))

            def on_run(entry_id: str, seed: int, record) -> None:
                print(f"{entry_id} seed {seed}: {record.run_id} {record.status.value}", file=text)

            with contextlib.redirect_stdout(text):
                runs = bm.run_plan(ws, plan, on_run=on_run)
            failed = sorted(f"{e}/s{s}" for (e, s), r in runs.items() if r.status != RunStatus.succeeded)
            expected = len(plan.entries) * len(plan.seeds)
            if args.json:
                _print_json({"plan_sha256": plan.plan_sha256, "runs": {f"{e}/s{s}": r.run_id for (e, s), r in runs.items()},
                             "failed": failed, "missing": expected - len(runs)})
            else:
                print(f"{len(runs) - len(failed)} of {expected} runs succeeded" + (f"; failed: {', '.join(failed)}" if failed else ""))
            return 0 if not failed and len(runs) == expected else 1
        if args.benchmark_command == "report":
            plan = bm.load_plan(Path(args.plan))
            report = bm.build_report(Workspace.open_or_create(Path(args.workspace)), plan)
            from opendpd.services.workspace import write_json_atomic
            write_json_atomic(Path(args.out), report)
            if args.markdown:
                Path(args.markdown).write_text(bm.report_markdown(report), encoding="utf-8")
            missing = sum(len(e.missing_seeds) for e in report.entries)
            print(f"report {report.report_sha256[:12]} written to {args.out} ({len(report.entries)} entries, "
                  f"{len(report.seeds)} seeds" + (f", {missing} missing runs" if missing else "") + ")")
            return 0
        if args.benchmark_command == "check":
            check = bm.check_regression(bm.load_report(Path(args.report)), bm.load_baseline(Path(args.baseline)))
            if args.json:
                _print_json(check.model_dump(mode="json"))
            else:
                for item in check.items:
                    shown = f"{item.observed:.3f} vs {item.reference:.3f} ± {item.tolerance:.3f}" \
                        if item.observed is not None else "no value"
                    print(f"  {item.status:<9} {item.entry_id}/{item.metric}: {shown}")
                print(check.verdict)
            return 1 if check.blocking else 0
        if args.benchmark_command == "baseline":
            baseline = bm.draft_baseline(bm.load_report(Path(args.report)), tolerance_db=args.tolerance_db)
            from opendpd.services.workspace import write_json_atomic
            write_json_atomic(Path(args.out), baseline)
            print(f"draft baseline written to {args.out}; it blocks nothing until approved_by / approved_on are filled "
                  "by a maintainer")
            return 0
    except (WorkspaceError, ValidationError, ValueError, KeyError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return 2


def cmd_adaptation(args) -> int:
    """conditions-v1 (S17): sealed condition cards, pre-registered adaptation plans and every-cell reports."""
    import contextlib

    from pydantic import ValidationError

    from opendpd.schemas import RunStatus, TargetRule
    from opendpd.services import adaptation as ad
    from opendpd.services.workspace import Workspace, WorkspaceError, write_json_atomic

    text = sys.stderr if getattr(args, "json", False) else sys.stdout

    def card_from(spec: str):
        return ad.builtin_card(spec) if spec in ad.BUILTIN_CARDS and not Path(spec).exists() else ad.load_card(Path(spec))

    try:
        if args.adaptation_command == "card":
            card = card_from(args.card)
            audits = ad.audit_card(Workspace.open_or_create(Path(args.workspace)), card) if args.workspace else []
            if args.out:
                write_json_atomic(Path(args.out), card)
            if args.json:
                _print_json({"card": card.model_dump(mode="json"), "audit": [a.model_dump(mode="json") for a in audits]})
            else:
                print(f"card {card.set_id} ({card.card_sha256[:12]}): {card.device}; dimension {card.dimension}; "
                      f"{len(card.conditions)} conditions" + (f"; written to {args.out}" if args.out else ""))
                for c in card.conditions:
                    print(f"  {c.condition_id:<16} {c.role:<7} dataset {c.dataset_id:<24} batch {c.capture_batch}  {c.values}")
                for a in audits:
                    print(f"  {a.condition_id:<16} registered: origin {a.origin}, {a.n_samples} samples, train split "
                          f"{a.train_samples}, raw {(a.raw_sha256 or '')[:12]}")
            return 0
        if args.adaptation_command == "plan":
            ws = Workspace.open_or_create(Path(args.workspace))
            target = None
            if args.target_metric:
                if args.target_threshold is None:
                    raise WorkspaceError("--target-threshold is required with --target-metric")
                target = TargetRule(metric=args.target_metric, threshold=args.target_threshold, better=args.target_better)
            plan = ad.make_plan(ws, card_from(args.card), pa_recipe=args.pa_recipe, dpd_recipe=args.dpd_recipe,
                                seeds=[int(x) for x in args.seeds.split(",")] if args.seeds else None,
                                budgets=[int(x) for x in args.budgets.split(",")] if args.budgets else None,
                                tasks=args.tasks.split(",") if args.tasks else None, profile_id=args.profile,
                                device=args.device, target=target)
            ad.write_plan(plan, Path(args.out))
            print(f"plan {plan.plan_sha256[:12]} written to {args.out}: {len(ad.cells_of(plan))} cells x "
                  f"{len(plan.seeds)} seeds over {len(plan.condition_set.conditions)} conditions "
                  f"({', '.join(plan.tasks)}; budgets {plan.budgets})")
            return 0
        if args.adaptation_command == "run":
            plan = ad.load_plan(Path(args.plan))
            ws = Workspace.open_or_create(Path(args.workspace))

            def on_cell(key: str, record) -> None:
                print(f"{key}: {record.run_id} {record.status.value}", file=text)

            with contextlib.redirect_stdout(text):
                runs = ad.run_plan(ws, plan, on_cell=on_cell)
            expected = len(ad.cells_of(plan)) * len(plan.seeds)
            failed = sorted(k for k, r in runs.items() if r.status != RunStatus.succeeded)
            if args.json:
                _print_json({"plan_sha256": plan.plan_sha256, "runs": {k: r.run_id for k, r in runs.items()},
                             "failed": failed, "missing": expected - len(runs)})
            else:
                print(f"{len(runs) - len(failed)} of {expected} cells succeeded" + (f"; failed: {', '.join(failed)}" if failed else "")
                      + (f"; {expected - len(runs)} refused (see the report)" if expected > len(runs) else ""))
            return 0 if not failed and len(runs) == expected else 1
        if args.adaptation_command == "report":
            plan = ad.load_plan(Path(args.plan))
            ws = Workspace.open_or_create(Path(args.workspace))
            report = ad.build_report(ws, plan)
            path = ad.store_report(ws, report)
            if args.out:
                write_json_atomic(Path(args.out), report)
            if args.markdown:
                Path(args.markdown).write_text(ad.report_markdown(report), encoding="utf-8")
            if args.json:
                _print_json(report.model_dump(mode="json"))
            else:
                bar = report.evidence_bar
                without = sum(c.status != "ok" for c in report.cells)
                print(f"report {report.report_sha256[:12]} stored at {path} ({len(report.cells)} cells, {without} without a number); "
                      f"evidence bar {'met' if bar.met else 'NOT met'}: {bar.n_conditions} conditions (bar {bar.min_conditions}), "
                      f"independent batches {bar.independent_batches}, measured origin {bar.measured_origin}")
                for lim in report.limitations:
                    print(f"  - {lim}")
            return 0
    except (WorkspaceError, ValidationError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opendpd", description="OpenDPD Studio command line")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("models", help="list registered models and their capabilities")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("recipes", help="list reference recipes")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_recipes)

    p = sub.add_parser("profiles", help="list metric profiles (the definition behind every score)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_profiles)

    p = sub.add_parser("evaluate", help="re-score a succeeded run under a metric profile from its best checkpoint")
    p.add_argument("run_id")
    p.add_argument("--workspace", required=True)
    p.add_argument("--profile", default="general-spectral-v1")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("apply", help="apply a trained DPD to the test split: export u = DPD(x) and score it through a PA surrogate")
    p.add_argument("dpd_run_id", help="a succeeded train_dpd run")
    p.add_argument("--workspace", required=True)
    p.add_argument("--pa", default=None, help="PA run to simulate through (default: the DPD's training surrogate)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser("export", help="write a reproducible experiment package (full: private and complete; share: redacted)")
    p.add_argument("run_id")
    p.add_argument("--workspace", required=True)
    p.add_argument("--out", default=None, help="zip path (default: <workspace>/exports/<run>-<kind>.zip)")
    p.add_argument("--kind", choices=["full", "share"], default="share")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("import", help="verify an experiment package and import it into a workspace")
    p.add_argument("path")
    p.add_argument("--workspace", required=True)
    p.add_argument("--inspect", action="store_true", help="verify only; import nothing")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("report", help="render a report bound to a run's stored result and plot data")
    p.add_argument("run_id")
    p.add_argument("--workspace", required=True)
    p.add_argument("--format", choices=["html", "md"], default="html")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("datasets", help="manage workspace datasets")
    ds = p.add_subparsers(dest="datasets_command", required=True)
    q = ds.add_parser("import-builtin", help="register a packaged dataset in the workspace")
    q.add_argument("name")
    q.add_argument("--workspace", required=True)
    q.add_argument("--id", default=None, help="dataset id (default: slug of the name)")
    q = ds.add_parser("list", help="list workspace datasets")
    q.add_argument("--workspace", required=True)
    q.add_argument("--json", action="store_true")
    q = ds.add_parser("import", help="import a CSV, .npy/.npz or OpenDPD split directory (raw copy + contiguous split)")
    q.add_argument("path")
    q.add_argument("--workspace", required=True)
    q.add_argument("--id", default=None, help="dataset id (default: slug of the file name)")
    q.add_argument("--name", default=None, help="display name")
    q.add_argument("--map", action="append", metavar="LOGICAL=COLUMN",
                   help="column mapping, e.g. --map I_in=tx_i (logical: I_in, Q_in, I_out, Q_out; npz: input, output)")
    q.add_argument("--fs", type=float, default=None, help="sample rate in Hz")
    q.add_argument("--bandwidth", type=float, default=None, help="main channel bandwidth in Hz")
    q.add_argument("--bw-sub-ch", dest="bw_sub_ch", type=float, default=None)
    q.add_argument("--n-sub-ch", dest="n_sub_ch", type=int, default=None)
    q.add_argument("--nperseg", type=int, default=None)
    q.add_argument("--units", choices=["normalized", "volts", "unknown"], default=None)
    q.add_argument("--waveform", default=None, metavar="PACKAGE",
                   help="bind the input column to a reference-waveform package (waveform.json or its directory); "
                        "refused when the input does not correlate with the waveform")
    q.add_argument("--origin", choices=["measured", "synthetic", "unknown"], default="unknown")
    q.add_argument("--guard", type=int, default=256, help="guard samples dropped between splits (>= frame length)")
    q.add_argument("--json", action="store_true")
    q = ds.add_parser("doctor", help="run Dataset Doctor and store the report")
    q.add_argument("id")
    q.add_argument("--workspace", required=True)
    q.add_argument("--version", default="raw-v1")
    q.add_argument("--json", action="store_true")
    q = ds.add_parser("preprocess", help="preview or create a preprocessing version (raw files are never modified)")
    q.add_argument("id")
    q.add_argument("--workspace", required=True)
    q.add_argument("--version", default=None, help="name of the new version (omit with --preview)")
    q.add_argument("--base", default="raw-v1")
    q.add_argument("--delay", type=float, default=0.0, help="output lags input by this many samples")
    q.add_argument("--gain-db", dest="gain_db", type=float, default=0.0)
    q.add_argument("--phase-deg", dest="phase_deg", type=float, default=0.0)
    q.add_argument("--interpolate-non-finite", dest="interpolate_non_finite", action="store_true")
    q.add_argument("--remove-outliers", dest="remove_outliers", action="store_true")
    q.add_argument("--normalize", action="store_true", help="scale by the input peak of the training split")
    q.add_argument("--preview", action="store_true")
    q.add_argument("--json", action="store_true")
    q = ds.add_parser("add-root", help="authorise a directory for imports")
    q.add_argument("name")
    q.add_argument("path")
    q.add_argument("--workspace", required=True)
    p.set_defaults(func=cmd_datasets)

    p = sub.add_parser("waveforms", help="reference waveforms with known symbols (plan S15): generate a package, inspect one")
    wv = p.add_subparsers(dest="waveforms_command", required=True)
    q = wv.add_parser("generate", help="write a reference-waveform package (waveform.json, x.npy to play in a loop, symbols.npy)")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--subframes", type=int, default=10, help="length in 1 ms subframes (10 = one frame)")
    q.add_argument("--out", required=True, help="package directory")
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_waveforms)
    q = wv.add_parser("show", help="print a package's specification and check that it regenerates to the recorded hash")
    q.add_argument("path", help="waveform.json or its directory")
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_waveforms)

    p = sub.add_parser("measurements", help="measured DPD evidence (plan S16): import captures of a physical PA driven by a run_dpd export")
    ms = p.add_subparsers(dest="measurements_command", required=True)
    q = ms.add_parser("import", help="align, score and store operator-provided captures as an evaluate_measured run")
    q.add_argument("--apply-run", required=True, help="the run_dpd run whose dpd-output file was played")
    q.add_argument("--with-dpd", required=True, help="capture of the PA output while u = DPD(x) was played (.csv/.npy/.npz)")
    q.add_argument("--without-dpd", help="capture of the PA output while x was played under the same conditions")
    q.add_argument("--conditions", required=True, help="JSON file with the declared conditions (MeasurementConditions)")
    q.add_argument("--columns", help="I,Q column names or .npz keys when they are not I,Q / I_out,Q_out")
    q.add_argument("--power-with", type=float, help="declared output power with DPD (dBm)")
    q.add_argument("--power-without", type=float, help="declared output power without DPD (dBm)")
    q.add_argument("--playback", choices=["loop", "single"], default="loop")
    q.add_argument("--profile", help="primary metric profile (default legacy-opendpd-v1; every profile is stored)")
    q.add_argument("--mock", action="store_true", help="the captures come from the mock adapter: the result is marked mock")
    q.add_argument("--name")
    q.add_argument("--workspace", required=True)
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_measurements)

    p = sub.add_parser("instruments", help="instrument adapters with fail-closed safety (plan S16); only the mock ships")
    ins = p.add_subparsers(dest="instruments_command", required=True)
    q = ins.add_parser("list", help="registered adapters, their kind, RF capability and default limits")
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_instruments)
    q = ins.add_parser("dry-run", help="export -> arm -> play -> capture -> RF off through an adapter; writes captures + a conditions template")
    q.add_argument("--apply-run", required=True, help="the run_dpd run whose dpd-output file is played")
    q.add_argument("--out", required=True, help="directory for with_dpd.npy, without_dpd.npy, session records and conditions.json")
    q.add_argument("--adapter", choices=["mock"], default="mock")
    q.add_argument("--arm", metavar="OPERATOR", help="the person arming the session; without it nothing is played")
    q.add_argument("--requested-power", type=float, help="requested output power (dBm), checked against the adapter ceiling")
    q.add_argument("--workspace", required=True)
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_instruments)

    p = sub.add_parser("gui", help="start the local Studio service and open the workbench in your browser")
    p.add_argument("--workspace", default=None, help="workspace directory (default: $OPENDPD_WORKSPACE or ~/opendpd-workspace)")
    p.add_argument("--port", type=int, default=None, help=f"loopback port (default: first free from 8765)")
    p.add_argument("--no-browser", dest="no_browser", action="store_true", help="print the URL instead of opening a browser")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("benchmark", help="benchmark-v1: pre-registered plans, per-seed reports, regression checks")
    bm = p.add_subparsers(dest="benchmark_command", required=True)
    q = bm.add_parser("plan", help="write a pre-registered plan: the tier's model matrix x seeds under a fixed budget")
    q.add_argument("--dataset", required=True, help="workspace dataset id")
    q.add_argument("--tier", choices=["cpu_regression", "gpu_full"], default="cpu_regression")
    q.add_argument("--seeds", default=None, help="comma-separated seeds, at least 3 (default 0,1,2)")
    q.add_argument("--profile", default="legacy-opendpd-v1")
    q.add_argument("--device", default="cpu")
    q.add_argument("--version", default="raw-v1", help="dataset preprocessing version")
    q.add_argument("--out", required=True)
    q = bm.add_parser("run", help="execute every (entry, seed) of a plan in this process; finished runs are reused")
    q.add_argument("plan")
    q.add_argument("--workspace", required=True)
    q.add_argument("--json", action="store_true")
    q = bm.add_parser("report", help="assemble the hash-bound per-seed report from the stored results")
    q.add_argument("plan")
    q.add_argument("--workspace", required=True)
    q.add_argument("--out", required=True, help="report JSON")
    q.add_argument("--markdown", default=None, help="also write a Markdown rendering")
    q = bm.add_parser("check", help="compare a report with a regression baseline (exit 1 when an approved band is left)")
    q.add_argument("report")
    q.add_argument("--baseline", required=True)
    q.add_argument("--json", action="store_true")
    q = bm.add_parser("baseline", help="draft a regression baseline from a report; unapproved until a maintainer signs it")
    q.add_argument("report")
    q.add_argument("--out", required=True)
    q.add_argument("--tolerance-db", dest="tolerance_db", type=float, default=0.5)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("adaptation", help="conditions-v1: multi-condition cards, pre-registered adaptation plans, every-cell reports")
    ap = p.add_subparsers(dest="adaptation_command", required=True)
    q = ap.add_parser("card", help="seal a condition card (a JSON file or a built-in id) and audit it against a workspace")
    q.add_argument("card", help="card JSON path, or a built-in id (`opendpd adaptation card apa-200mhz-batches-v1`)")
    q.add_argument("--workspace", default=None, help="audit: every condition must be a registered dataset from its own capture")
    q.add_argument("--out", default=None, help="write the sealed card")
    q.add_argument("--json", action="store_true")
    q = ap.add_parser("plan", help="pre-register a plan: entries x tasks x conditions x budgets x seeds; the hash keys every run")
    q.add_argument("card", help="card JSON path or built-in id")
    q.add_argument("--workspace", required=True)
    q.add_argument("--pa-recipe", dest="pa_recipe", required=True, help="PA recipe id (see `opendpd recipes`)")
    q.add_argument("--dpd-recipe", dest="dpd_recipe", default=None, help="DPD recipe id; its surrogate is the PA of the same condition")
    q.add_argument("--seeds", default=None, help="comma-separated training seeds (default 0)")
    q.add_argument("--budgets", default=None, help="comma-separated few-shot budgets in samples of the target train split (default 2000)")
    q.add_argument("--tasks", default=None, help="comma-separated subset of zero_update,few_shot,full_retrain (default all)")
    q.add_argument("--profile", default="legacy-opendpd-v1")
    q.add_argument("--device", default="cpu")
    q.add_argument("--target-metric", dest="target_metric", default=None, help="what 'reaching the target' means, e.g. NMSE")
    q.add_argument("--target-threshold", dest="target_threshold", type=float, default=None)
    q.add_argument("--target-better", dest="target_better", choices=["lower", "higher"], default="lower")
    q.add_argument("--out", required=True)
    q = ap.add_parser("run", help="execute every cell of a plan in this process; finished runs are reused, refusals are recorded")
    q.add_argument("plan")
    q.add_argument("--workspace", required=True)
    q.add_argument("--json", action="store_true")
    q = ap.add_parser("report", help="assemble the hash-bound report (every cell, failures included) under <workspace>/adaptation/")
    q.add_argument("plan")
    q.add_argument("--workspace", required=True)
    q.add_argument("--out", default=None, help="also write the report JSON here")
    q.add_argument("--markdown", default=None, help="also write a Markdown rendering")
    q.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_adaptation)

    p = sub.add_parser("doctor", help="check that the GUI can start: dependencies, frontend assets, workspace, port")
    p.add_argument("--workspace", default=None)
    p.set_defaults(func=cmd_doctor)

    for name, func, help_text in (("validate", cmd_validate, "validate and resolve an experiment config"),
                                  ("run", cmd_run, "run an experiment in this process (headless)")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", help="experiment config JSON")
        p.add_argument("--recipe", help="reference recipe id (see `opendpd recipes`)")
        p.add_argument("--dataset", help="workspace dataset id (with --recipe)")
        p.add_argument("--pa-run", dest="pa_run", help="PA run id for DPD recipes")
        p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--name", default=None)
        p.add_argument("--json", action="store_true")
        if name == "run":
            p.add_argument("--workspace", required=True)
            p.add_argument("--idempotency-key", dest="idempotency_key", default=None)
        p.set_defaults(func=func)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
