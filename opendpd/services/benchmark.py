"""benchmark-v1 as code (plan S12): pre-registered plans, sequential execution under idempotent run keys,
hash-bound per-seed reports, and regression checks against baselines that only a human can approve."""

from __future__ import annotations

import math
import os
import platform
import statistics
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from opendpd.core.metrics import get_profile
from opendpd.schemas import (
    ArtifactKind,
    DatasetRef,
    EvaluationConfig,
    EvidenceType,
    ExecutionConfig,
    ExperimentConfig,
    ModelSpec,
    PAReference,
    RunRecord,
    RunStatus,
    TaskType,
)
from opendpd.schemas.benchmark import (
    COMPUTE_NOTE,
    PROTOCOL_ID,
    SEED_NOTE,
    BaselineBand,
    BenchmarkEntry,
    BenchmarkPlan,
    BenchmarkReport,
    DataAudit,
    EntryResult,
    MetricStats,
    RegressionBaseline,
    RegressionCheck,
    RegressionItem,
    SeedScore,
)
from opendpd.services.experiments import create_run, execute_run, list_runs, load_artifacts, load_result
from opendpd.services.recipes import get_recipe
from opendpd.services.workspace import Workspace, WorkspaceError, read_json, software_provenance, write_json_atomic

DEFAULT_SEEDS = (0, 1, 2)
FIT_FILE = "fit.json"


# --- plans ---------------------------------------------------------------------------

def _from_recipe(entry_id: str, recipe_id: str, pa_entry: Optional[str] = None,
                 parameters: Optional[Dict[str, object]] = None) -> BenchmarkEntry:
    recipe = get_recipe(recipe_id)
    model = recipe.model if parameters is None else ModelSpec(key=recipe.model.key, parameters=parameters)
    return BenchmarkEntry(entry_id=entry_id, task=recipe.task, recipe_id=recipe_id, model=model,
                          training=recipe.training, pa_entry=pa_entry)


def default_entries(tier: str) -> List[BenchmarkEntry]:
    """The fixed matrix of a tier. cpu_regression uses the smoke budgets (regression references, never research
    results); gpu_full uses the research recipes and the benchmark's polynomial configurations."""
    small = {"K": 5, "Q": 20, "rcond": 0.0}
    if tier == "cpu_regression":
        return [_from_recipe("pa-gru", "pa-gru-smoke-v1"),
                _from_recipe("pa-mp-ls", "pa-mp-ls-v1", parameters=small),
                _from_recipe("dpd-gru", "dpd-gru-smoke-v1", pa_entry="pa-gru"),
                _from_recipe("dpd-mp-ila", "dpd-mp-ila-v1", pa_entry="pa-gru", parameters=small)]
    if tier == "gpu_full":
        return [_from_recipe("pa-gru", "pa-gru-research-v1"),
                _from_recipe("pa-mp-ls", "pa-mp-ls-v1"),
                _from_recipe("pa-gmp-ls", "pa-gmp-ls-v1"),
                _from_recipe("dpd-tres-deltagru", "dpd-tres-deltagru-research-v1", pa_entry="pa-gru"),
                _from_recipe("dpd-mp-ila", "dpd-mp-ila-v1", pa_entry="pa-gru"),
                _from_recipe("dpd-gmp-ila", "dpd-gmp-ila-v1", pa_entry="pa-gru")]
    raise ValueError(f"unknown tier '{tier}'")


def make_plan(dataset_id: str, *, tier: str = "cpu_regression", seeds=DEFAULT_SEEDS, profile_id: str = "legacy-opendpd-v1",
              device: str = "cpu", preprocessing_version: str = "raw-v1",
              entries: Optional[List[BenchmarkEntry]] = None) -> BenchmarkPlan:
    plan = BenchmarkPlan(tier=tier, dataset=DatasetRef(id=dataset_id, preprocessing_version=preprocessing_version),
                         metric_profile_id=profile_id, device=device, seeds=list(seeds),
                         entries=entries if entries is not None else default_entries(tier))
    return plan.model_copy(update={"plan_sha256": plan.compute_sha256()})


def write_plan(plan: BenchmarkPlan, path: Path) -> None:
    write_json_atomic(Path(path), plan)


def load_plan(path: Path) -> BenchmarkPlan:
    plan = BenchmarkPlan.model_validate(read_json(Path(path)))
    if plan.plan_sha256 is None or plan.plan_sha256 != plan.compute_sha256():
        raise WorkspaceError(f"plan {path} was edited after registration (its content does not match plan_sha256); "
                             "register a new plan instead of changing one")
    return plan


def run_key(plan: BenchmarkPlan, entry_id: str, seed: int) -> str:
    return f"bench-{plan.plan_sha256[:12]}-{entry_id}-s{seed}"


def entry_config(plan: BenchmarkPlan, entry: BenchmarkEntry, seed: int, pa_run_id: Optional[str]) -> ExperimentConfig:
    evidence = EvidenceType.pa_modeling if entry.task == TaskType.train_pa else EvidenceType.dpd_surrogate
    return ExperimentConfig(
        task=entry.task, recipe_id=entry.recipe_id, name=f"{plan.tier} {entry.entry_id} seed {seed}",
        dataset=plan.dataset, model=entry.model, training=entry.training.model_copy(update={"seed": seed}),
        evaluation=EvaluationConfig(evidence_type=evidence, profile_id=plan.metric_profile_id),
        execution=ExecutionConfig(device=plan.device),
        pa_reference=PAReference(run_id=pa_run_id) if pa_run_id else None,
        notes=f"{PROTOCOL_ID} plan {plan.plan_sha256[:12]} entry {entry.entry_id} seed {seed}",
    )


# --- execution ---------------------------------------------------------------------------

def _ordered(plan: BenchmarkPlan) -> List[BenchmarkEntry]:
    return [e for e in plan.entries if e.task == TaskType.train_pa] + [e for e in plan.entries if e.task != TaskType.train_pa]


def run_plan(ws: Workspace, plan: BenchmarkPlan, *,
             on_run: Optional[Callable[[str, int, RunRecord], None]] = None) -> Dict[Tuple[str, int], RunRecord]:
    """Execute every (entry, seed) in this process. Runs are keyed by the plan hash, so re-running a plan reuses
    finished runs and only fills the gaps; a failed surrogate leaves its DPD entries without a run for that seed."""
    runs: Dict[Tuple[str, int], RunRecord] = {}
    for seed in plan.seeds:
        for entry in _ordered(plan):
            pa_run = runs.get((entry.pa_entry, seed)) if entry.pa_entry else None
            if entry.pa_entry and (pa_run is None or pa_run.status != RunStatus.succeeded):
                continue
            record = create_run(ws, entry_config(plan, entry, seed, pa_run.run_id if pa_run else None),
                                idempotency_key=run_key(plan, entry.entry_id, seed))
            if record.status == RunStatus.queued:
                record = execute_run(ws, record.run_id)
            runs[(entry.entry_id, seed)] = record
            if on_run is not None:
                on_run(entry.entry_id, seed, record)
    return runs


def find_runs(ws: Workspace, plan: BenchmarkPlan) -> Dict[Tuple[str, int], RunRecord]:
    keys = {run_key(plan, e.entry_id, s): (e.entry_id, s) for e in plan.entries for s in plan.seeds}
    return {keys[r.idempotency_key]: r for r in list_runs(ws) if r.idempotency_key in keys}


# --- reports ---------------------------------------------------------------------------

def _machine() -> Dict[str, str]:
    """Hardware and OS identity without a hostname or a path."""
    return {"cpu": platform.processor() or platform.machine(), "cores": str(os.cpu_count() or 0),
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})"}


def stats(values: List[float]) -> MetricStats:
    return MetricStats(n=len(values), mean=float(statistics.fmean(values)),
                       std=float(statistics.stdev(values)) if len(values) >= 2 else None,
                       min=float(min(values)), max=float(max(values)))


def aggregate(seeds: List[SeedScore]) -> Dict[str, MetricStats]:
    """Per metric over the seeds; a metric missing or non-finite on any seed is left out."""
    out: Dict[str, MetricStats] = {}
    for name in sorted({n for s in seeds for n in s.metrics}):
        values = [s.metrics[name] for s in seeds if s.metrics.get(name) is not None and math.isfinite(s.metrics[name])]
        if values:
            out[name] = stats(values)
    return out


def build_report(ws: Workspace, plan: BenchmarkPlan) -> BenchmarkReport:
    """Every number is read from a run's stored result under the plan's profile; nothing is recomputed."""
    import torch  # noqa: F401 - loaded so the software provenance records the torch version the runs used

    runs = find_runs(ws, plan)
    dataset = ws.get_dataset(plan.dataset.id)
    profile = get_profile(plan.metric_profile_id)
    entries: List[EntryResult] = []
    for entry in plan.entries:
        role = "pa" if entry.task == TaskType.train_pa else "dpd"
        seeds: List[SeedScore] = []
        missing: List[int] = []
        meta: Dict[str, object] = {}
        fit = None
        for seed in plan.seeds:
            record = runs.get((entry.entry_id, seed))
            result = load_result(ws, record.run_id, plan.metric_profile_id) \
                if record is not None and record.status == RunStatus.succeeded else None
            if result is None:
                missing.append(seed)
                continue
            manifest = load_artifacts(ws, record.run_id)
            checkpoints = manifest.by_kind(ArtifactKind.checkpoint) if manifest else []
            primary = next(m for m in result.models if m.role == role)
            meta = {"training_path": primary.training_path, "n_parameters": primary.n_parameters,
                    "lookahead_samples": primary.lookahead_samples, "execution_semantics": primary.execution_semantics}
            surrogate = next((m.run_id for m in result.models if m.role == "pa"), None) if role == "dpd" else None
            wall = (record.finished_at - record.started_at).total_seconds() \
                if record.started_at and record.finished_at else None
            if fit is None and (ws.run_dir(record.run_id) / FIT_FILE).exists():
                fit = read_json(ws.run_dir(record.run_id) / FIT_FILE).get("diagnostics")
            seeds.append(SeedScore(seed=seed, run_id=record.run_id, config_sha256=record.config_sha256,
                                   checkpoint_sha256=checkpoints[0].file.sha256 if checkpoints else None,
                                   surrogate_run_id=surrogate, selected_epoch=result.selected_epoch,
                                   metrics={m.name: m.value for m in result.metrics}, wall_clock_s=wall))
        entries.append(EntryResult(entry_id=entry.entry_id, task=entry.task, model=entry.model, fit=fit, seeds=seeds,
                                   missing_seeds=missing, aggregate=aggregate(seeds), **meta))
    audit = DataAudit(dataset_id=dataset.dataset_id, raw_sha256=dataset.raw_sha256,
                      preprocessing_version=plan.dataset.preprocessing_version, split_version=plan.dataset.split_version,
                      guard_samples=dataset.split.guard_samples if dataset.split else None)
    notes = [SEED_NOTE.format(n=len(plan.seeds)), COMPUTE_NOTE, plan.selection_rule, plan.test_set_policy]
    if plan.tier == "cpu_regression":
        notes.append("cpu_regression uses the smoke budgets: its numbers are regression references for this data and "
                     "hardware, never research results")
    report = BenchmarkReport(tier=plan.tier, plan_sha256=plan.plan_sha256, dataset=audit,
                             metric_profile_id=plan.metric_profile_id, metric_profile_version=profile.version,
                             device=plan.device, seeds=list(plan.seeds), entries=entries,
                             software=software_provenance(), machine=_machine(), notes=notes)
    return report.sealed()


def load_report(path: Path) -> BenchmarkReport:
    report = BenchmarkReport.model_validate(read_json(Path(path)))
    if not report.intact:
        raise WorkspaceError(f"report {path} does not match its report_sha256: a number was edited or the file is "
                             "damaged; regenerate it with `opendpd benchmark report`")
    return report


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def report_markdown(report: BenchmarkReport) -> str:
    out = [f"# {PROTOCOL_ID} report ({report.tier})", "",
           f"Plan `{report.plan_sha256[:12]}` · dataset `{report.dataset.dataset_id}` (raw sha256 "
           f"{(report.dataset.raw_sha256 or 'n/a')[:12]}, version {report.dataset.preprocessing_version}, split "
           f"{report.dataset.split_version}, guard {report.dataset.guard_samples}) · profile "
           f"{report.metric_profile_id} v{report.metric_profile_version} · device {report.device} · seeds "
           f"{', '.join(map(str, report.seeds))}", "",
           f"Software: opendpd {report.software.opendpd_version}, python {report.software.python_version}, torch "
           f"{report.software.torch_version}, git {report.software.git_commit}"
           f"{' (dirty)' if report.software.git_dirty else ''} · machine: {report.machine.get('cpu')}, "
           f"{report.machine.get('cores')} cores, {report.machine.get('os')}", ""]
    for entry in report.entries:
        names = sorted(entry.aggregate)
        out += [f"## {entry.entry_id} ({entry.task.value}, {entry.model.key} {entry.model.parameters})", "",
                f"training path {entry.training_path} · {entry.n_parameters} real parameters · look-ahead "
                f"{entry.lookahead_samples} samples · {entry.execution_semantics}"]
        if entry.fit:
            out.append(f"fit: rank {entry.fit.get('rank')} of {entry.fit.get('n_coefficients')}, condition number "
                       f"{entry.fit.get('condition_number'):.3g}, rcond {entry.fit.get('rcond')}")
        out += ["", "| seed | run | selected epoch | " + " | ".join(names) + " | wall clock (s) |",
                "|---|---|---:|" + "---:|" * len(names) + "---:|"]
        for s in entry.seeds:
            out.append(f"| {s.seed} | `{s.run_id}` | {s.selected_epoch if s.selected_epoch is not None else 'n/a'} | "
                       + " | ".join(_fmt(s.metrics.get(n)) for n in names) + f" | {_fmt(s.wall_clock_s)} |")
        agg = entry.aggregate
        out.append("| **mean ± std (min … max)** | | | " + " | ".join(
            f"{agg[n].mean:.2f} ± {_fmt(agg[n].std)} ({agg[n].min:.2f} … {agg[n].max:.2f})" for n in names) + " | |")
        if entry.missing_seeds:
            out.append(f"\nmissing seeds: {', '.join(map(str, entry.missing_seeds))}")
        out.append("")
    out += ["## Data audit", ""] + [f"- {k}: {v}" for k, v in report.dataset.model_dump(mode="json").items()] + [""]
    out += ["## Notes", ""] + [f"- {n}" for n in report.notes] + ["", f"report sha256 `{report.report_sha256}`", ""]
    return "\n".join(out)


# --- regression ---------------------------------------------------------------------------

def draft_baseline(report: BenchmarkReport, *, tolerance_db: float = 0.5, basis: Optional[str] = None) -> RegressionBaseline:
    """Reference = seed mean, tolerance = max(tolerance_db, 2 x sample std). Unapproved until a maintainer signs."""
    profile = get_profile(report.metric_profile_id)
    better = {m.name: m.better.value for m in profile.metrics}
    entries: Dict[str, Dict[str, BaselineBand]] = {}
    for entry in report.entries:
        bands: Dict[str, BaselineBand] = {}
        for name, stats in entry.aggregate.items():
            tol = max(float(tolerance_db), 2.0 * stats.std) if stats.std is not None else float(tolerance_db)
            bands[name] = BaselineBand(reference=stats.mean, tolerance=tol,
                                       worse_is="higher" if better.get(name, "lower") == "lower" else "lower")
        if bands:
            entries[entry.entry_id] = bands
    return RegressionBaseline(
        tier=report.tier, plan_sha256=report.plan_sha256, entries=entries,
        basis=basis or (f"reference = mean over seeds {report.seeds} of report {report.report_sha256[:12]} on "
                        f"{report.machine.get('cpu')} ({report.machine.get('os')}); tolerance = max({tolerance_db} dB, "
                        "2 x sample std); any excursion beyond the band, better or worse, needs a new approval"),
        notes=["approve by filling approved_by and approved_on after reviewing the report; an unapproved baseline "
               "never blocks"])


def load_baseline(path: Path) -> RegressionBaseline:
    return RegressionBaseline.model_validate(read_json(Path(path)))


def check_regression(report: BenchmarkReport, baseline: RegressionBaseline) -> RegressionCheck:
    if not report.intact:
        raise WorkspaceError("the report does not match its report_sha256; regenerate it before checking")
    if baseline.plan_sha256 != report.plan_sha256:
        raise WorkspaceError(f"the baseline belongs to plan {baseline.plan_sha256[:12]}, the report to plan "
                             f"{report.plan_sha256[:12]}: a baseline is approved for one pre-registered plan")
    items: List[RegressionItem] = []
    for entry_id, bands in baseline.entries.items():
        entry = next((e for e in report.entries if e.entry_id == entry_id), None)
        for metric, band in bands.items():
            stats = entry.aggregate.get(metric) if entry is not None else None
            if stats is None:
                items.append(RegressionItem(entry_id=entry_id, metric=metric, reference=band.reference,
                                            tolerance=band.tolerance, worse_is=band.worse_is, status="missing"))
                continue
            delta = stats.mean - band.reference
            beyond = abs(delta) > band.tolerance
            worse = delta > 0 if band.worse_is == "higher" else delta < 0
            status = "within" if not beyond else ("degraded" if worse else "improved")
            items.append(RegressionItem(entry_id=entry_id, metric=metric, reference=band.reference,
                                        tolerance=band.tolerance, worse_is=band.worse_is, observed=stats.mean,
                                        delta=delta, status=status))
    ok = all(i.status == "within" for i in items)
    degraded = [f"{i.entry_id}/{i.metric}" for i in items if i.status == "degraded"]
    improved = [f"{i.entry_id}/{i.metric}" for i in items if i.status == "improved"]
    missing = [f"{i.entry_id}/{i.metric}" for i in items if i.status == "missing"]
    if ok:
        verdict = f"within the approved band on all {len(items)} checks"
    else:
        parts = []
        if degraded:
            parts.append("degraded beyond the band: " + ", ".join(degraded))
        if improved:
            parts.append("better than the band (a protocol or metric change must be ruled out and the baseline "
                         "re-approved): " + ", ".join(improved))
        if missing:
            parts.append("missing: " + ", ".join(missing))
        verdict = "; ".join(parts)
    approved = baseline.approved
    verdict += (" — the baseline is approved" + (": automatic release is blocked" if not ok else "")) if approved \
        else " — advisory only: the baseline is not approved by a maintainer"
    return RegressionCheck(ok=ok, approved=approved, blocking=(not ok) and approved, items=items, verdict=verdict)
