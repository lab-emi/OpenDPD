"""``opendpd`` sub-commands (Studio entry point).

Heavy imports happen inside each command so ``opendpd --help`` is instant.
Exit codes: 0 ok, 1 run failed, 2 invalid input / configuration, 130 cancelled.
"""

from __future__ import annotations

import argparse
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
                                         guard_samples=args.guard)
            if args.json:
                _print_json(manifest.model_dump(mode="json"))
            else:
                print(f"imported '{manifest.dataset_id}' ({manifest.n_samples} samples) from {args.path}")
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
    if record.status != RunStatus.queued:
        print(f"run {record.run_id} already exists for idempotency key {args.idempotency_key!r} "
              f"(status {record.status.value})")
        return 0
    print(f"run {record.run_id} created in {ws.run_dir(record.run_id)}")
    record = execute_run(ws, record.run_id)
    print(f"run {record.run_id} {record.status.value}")
    if record.error:
        print(f"  {record.error.code} [{record.error.stage}]: {record.error.message}", file=sys.stderr)
    result = load_result(ws, record.run_id)
    if result is not None:
        _print_result(result)
    if args.json:
        _print_json({"run": record.model_dump(mode="json"),
                     "result": result.model_dump(mode="json") if result else None})
    return {RunStatus.succeeded: 0, RunStatus.cancelled: 130}.get(record.status, 1)


def _print_result(result) -> None:
    print(f"  evidence: {result.evidence_type.value}  profile: {result.metric_profile_id} v{result.metric_profile_version}")
    for m in result.metrics:
        shown = f"{m.value:.4f} {m.unit}" if m.value is not None else f"{m.status.value}: {m.reason}"
        print(f"  {m.name:<9} {shown}")
    for lim in result.limitations:
        print(f"  note: {lim}")


def cmd_profiles(args) -> int:
    from opendpd.core.metrics import list_profiles

    profiles = list_profiles()
    if args.json:
        _print_json([p.model_dump(mode="json") for p in profiles])
        return 0
    for p in profiles:
        flags = " (frozen)" if p.frozen else ""
        print(f"{p.profile_id} v{p.version}{flags}: {p.description}")
        for m in p.metrics:
            print(f"  {m.name:<9} {m.unit:<4} {m.better.value:<6} {m.formula}")
    return 0


def cmd_evaluate(args) -> int:
    from opendpd.services.evaluation import evaluate_run
    from opendpd.services.workspace import Workspace, WorkspaceError

    import contextlib

    try:
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

    p = sub.add_parser("gui", help="start the local Studio service and open the workbench in your browser")
    p.add_argument("--workspace", default=None, help="workspace directory (default: $OPENDPD_WORKSPACE or ~/opendpd-workspace)")
    p.add_argument("--port", type=int, default=None, help=f"loopback port (default: first free from 8765)")
    p.add_argument("--no-browser", dest="no_browser", action="store_true", help="print the URL instead of opening a browser")
    p.set_defaults(func=cmd_gui)

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
