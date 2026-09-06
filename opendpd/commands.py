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


def cmd_datasets(args) -> int:
    from opendpd.services.workspace import Workspace, WorkspaceError

    try:
        ws = Workspace.open_or_create(Path(args.workspace))
        if args.datasets_command == "import-builtin":
            manifest = ws.register_builtin_dataset(args.name, dataset_id=args.id)
            print(f"registered '{manifest.dataset_id}' ({manifest.display_name}) in {ws.root}")
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
        print(f"  evidence: {result.evidence_type.value}  profile: {result.metric_profile_id}")
        for m in result.metrics:
            shown = f"{m.value:.4f} {m.unit}" if m.value is not None else f"{m.status.value}: {m.reason}"
            print(f"  {m.name:<9} {shown}")
        for lim in result.limitations:
            print(f"  note: {lim}")
    if args.json:
        _print_json({"run": record.model_dump(mode="json"),
                     "result": result.model_dump(mode="json") if result else None})
    return {RunStatus.succeeded: 0, RunStatus.cancelled: 130}.get(record.status, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opendpd", description="OpenDPD Studio command line")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("models", help="list registered models and their capabilities")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("recipes", help="list reference recipes")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_recipes)

    p = sub.add_parser("datasets", help="manage workspace datasets")
    ds = p.add_subparsers(dest="datasets_command", required=True)
    q = ds.add_parser("import-builtin", help="register a packaged dataset in the workspace")
    q.add_argument("name")
    q.add_argument("--workspace", required=True)
    q.add_argument("--id", default=None, help="dataset id (default: slug of the name)")
    q = ds.add_parser("list", help="list workspace datasets")
    q.add_argument("--workspace", required=True)
    q.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_datasets)

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
