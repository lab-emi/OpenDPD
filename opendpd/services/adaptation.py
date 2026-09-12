"""Multi-condition adaptation (plan S17): ``conditions-v1`` plans, execution and hash-bound reports.

A plan is pre-registered from a sealed condition set (the data card). Every
cell of the matrix is an ordinary run with an idempotency key derived from the
plan hash, so re-running a plan fills gaps and never duplicates work. The
report reads stored results only, keeps failed cells with their reasons, and
states whether the card meets the S17 evidence bar.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from opendpd.core.metrics import get_profile
from opendpd.schemas import (
    AdaptationCell,
    AdaptationEntry,
    AdaptationPlan,
    AdaptationReport,
    ArtifactKind,
    CellAggregate,
    Condition,
    ConditionAudit,
    ConditionSet,
    DatasetRef,
    DPDReference,
    EvaluationConfig,
    EvidenceBar,
    EvidenceType,
    ExecutionConfig,
    ExperimentConfig,
    InitReference,
    PAReference,
    RunRecord,
    RunStatus,
    TargetRule,
    TaskType,
)
from opendpd.schemas.benchmark import MetricStats
from opendpd.schemas.conditions import ALL_TASKS, MIN_CONDITIONS_FOR_EVIDENCE, PROTOCOL_ID, AdaptationTask
from opendpd.services.benchmark import _machine, stats
from opendpd.services.config import ConfigError
from opendpd.services.experiments import create_run, execute_run, list_runs, load_artifacts, load_result
from opendpd.services.recipes import get_recipe
from opendpd.services.workspace import Workspace, WorkspaceError, read_json, software_provenance, write_json_atomic

REPORTS_DIR = "adaptation"

# Built-in cards over packaged datasets. The APA pair is a real "independent capture batch" dimension with two
# conditions: below the S17 bar (three), which every report over it says.
BUILTIN_CARDS: Dict[str, ConditionSet] = {
    "apa-200mhz-batches-v1": ConditionSet(
        set_id="apa-200mhz-batches-v1", device="APA (built-in datasets APA_200MHz / APA_200MHz_b)",
        dimension="capture_batch",
        conditions=[
            Condition(condition_id="batch-a", dataset_id="apa-200mhz", role="source", capture_batch="A",
                      values={"capture_batch": "A"}, notes="built-in APA_200MHz: 5-carrier LTE 20 MHz, 983.04 MS/s"),
            Condition(condition_id="batch-b", dataset_id="apa-200mhz-b", role="target", capture_batch="B",
                      values={"capture_batch": "B"}, notes="built-in APA_200MHz_b: the same set-up, measurement B"),
        ]),
}


def seal_card(card: ConditionSet) -> ConditionSet:
    return card.model_copy(update={"card_sha256": card.compute_sha256()})


def builtin_card(card_id: str) -> ConditionSet:
    if card_id not in BUILTIN_CARDS:
        raise WorkspaceError(f"unknown built-in condition set '{card_id}'; available: {', '.join(sorted(BUILTIN_CARDS))}")
    return seal_card(BUILTIN_CARDS[card_id])


def load_card(path: Path) -> ConditionSet:
    card = ConditionSet.model_validate(read_json(Path(path)))
    if card.card_sha256 is not None and card.card_sha256 != card.compute_sha256():
        raise WorkspaceError(f"{path}: the card was edited after it was sealed (card_sha256 does not match)")
    return seal_card(card)


def _train_split_size(ws: Workspace, dataset_id: str, version: str = "raw-v1") -> Optional[int]:
    manifest = ws.get_dataset(dataset_id)
    dv = manifest.version(version)
    split = dv.split if dv is not None else manifest.split
    bounds = (split.boundaries or {}).get("train") if split is not None else None
    return int(bounds[1] - bounds[0]) if bounds else None


def audit_card(ws: Workspace, card: ConditionSet) -> List[ConditionAudit]:
    """Every condition is a registered dataset from its own capture: distinct raw hashes, or the card is refused."""
    audits: List[ConditionAudit] = []
    seen: Dict[str, str] = {}
    for c in card.conditions:
        try:
            manifest = ws.get_dataset(c.dataset_id)
        except WorkspaceError as err:
            raise WorkspaceError(f"condition '{c.condition_id}': {err}") from None
        if manifest.raw_sha256 in seen:
            raise WorkspaceError(f"conditions '{seen[manifest.raw_sha256]}' and '{c.condition_id}' share the same raw "
                                 "capture (identical hash); one capture split two ways is not two conditions")
        seen[manifest.raw_sha256] = c.condition_id
        audits.append(ConditionAudit(condition_id=c.condition_id, dataset_id=c.dataset_id, origin=manifest.origin.value,
                                     raw_sha256=manifest.raw_sha256, n_samples=manifest.n_samples,
                                     train_samples=_train_split_size(ws, c.dataset_id)))
    return audits


# --- plans ---------------------------------------------------------------------------------

def _entry(entry_id: str, recipe_id: str, pa_entry: Optional[str] = None) -> AdaptationEntry:
    recipe = get_recipe(recipe_id)
    return AdaptationEntry(entry_id=entry_id, task=recipe.task, recipe_id=recipe_id, model=recipe.model,
                           training=recipe.training, pa_entry=pa_entry)


def make_plan(ws: Workspace, card: ConditionSet, *, pa_recipe: str, dpd_recipe: Optional[str] = None,
              seeds: Optional[List[int]] = None, budgets: Optional[List[int]] = None,
              tasks: Optional[List[AdaptationTask]] = None, profile_id: str = "legacy-opendpd-v1",
              device: str = "cpu", target: Optional[TargetRule] = None) -> AdaptationPlan:
    card = seal_card(card)
    audit_card(ws, card)
    entries = [_entry("pa", pa_recipe)]
    if dpd_recipe:
        entries.append(_entry("dpd", dpd_recipe, pa_entry="pa"))
    plan = AdaptationPlan(condition_set=card, entries=entries, tasks=list(tasks or ALL_TASKS),
                          budgets=list(budgets or [2000]), seeds=list(seeds or [0]), metric_profile_id=profile_id,
                          device=device, target=target)
    get_profile(profile_id)
    return plan.model_copy(update={"plan_sha256": plan.compute_sha256()})


def write_plan(plan: AdaptationPlan, path: Path) -> None:
    write_json_atomic(Path(path), plan)


def load_plan(path: Path) -> AdaptationPlan:
    plan = AdaptationPlan.model_validate(read_json(Path(path)))
    if plan.plan_sha256 is None or plan.plan_sha256 != plan.compute_sha256():
        raise WorkspaceError(f"{path}: the plan was edited after it was registered (plan_sha256 does not match)")
    return plan


# --- cells -----------------------------------------------------------------------------------

def cell_key(plan: AdaptationPlan, entry_id: str, task: str, condition_id: str, budget: Optional[int], seed: int) -> str:
    return f"adapt-{plan.plan_sha256[:12]}-{entry_id}-{task}-{condition_id}-b{budget or 0}-s{seed}"


def cells_of(plan: AdaptationPlan) -> List[Tuple[AdaptationEntry, str, Condition, Optional[int]]]:
    """Every (entry, task, condition, budget) in execution order. Full retrains come first because they are the
    source models and the surrogates: a PA entry is retrained on every condition (each DPD cell of a condition goes
    through the PA trained on that condition), a DPD entry on the source always and on the targets only when the
    plan asks for ``full_retrain``. Zero-update and few-shot cells follow on every target; PA entries before DPD."""
    card = plan.condition_set
    out: List[Tuple[AdaptationEntry, str, Condition, Optional[int]]] = []
    for entry in sorted(plan.entries, key=lambda e: e.task != TaskType.train_pa):
        for c in card.conditions:
            if c.role == "source" or entry.task == TaskType.train_pa or "full_retrain" in plan.tasks:
                out.append((entry, "full_retrain", c, None))
        for c in card.targets:
            if "zero_update" in plan.tasks:
                out.append((entry, "zero_update", c, None))
            if "few_shot" in plan.tasks:
                for budget in plan.budgets:
                    out.append((entry, "few_shot", c, budget))
    return out


def _config(plan: AdaptationPlan, entry: AdaptationEntry, task: str, condition: Condition, budget: Optional[int],
            seed: int, runs: Dict[str, RunRecord]) -> ExperimentConfig:
    """The ordinary run behind one cell. Raises KeyError when a run it depends on is missing or failed."""
    card = plan.condition_set
    source = card.source

    def need(entry_id: str, task_: str, cond: Condition) -> str:
        record = runs.get(cell_key(plan, entry_id, task_, cond.condition_id, None, seed))
        if record is None or record.status != RunStatus.succeeded:
            raise KeyError(f"{entry_id} {task_} on {cond.condition_id} (seed {seed}) has no succeeded run")
        return record.run_id

    training = entry.training.model_copy(update={"seed": seed, "train_samples": budget})
    common = dict(recipe_id=entry.recipe_id, execution=ExecutionConfig(device=plan.device),
                  name=f"{PROTOCOL_ID} {entry.entry_id} {task} {condition.condition_id}"
                       + (f" budget {budget}" if budget else "") + f" seed {seed}",
                  notes=f"{PROTOCOL_ID} plan {plan.plan_sha256[:12]} entry {entry.entry_id} task {task} "
                        f"condition {condition.condition_id} budget {budget or 0} seed {seed}")
    dataset = DatasetRef(id=condition.dataset_id)
    if entry.task == TaskType.train_pa:
        if task == "zero_update":
            return ExperimentConfig(task=TaskType.evaluate_pa, dataset=dataset, model=entry.model,
                                    evaluation=EvaluationConfig(evidence_type=EvidenceType.pa_modeling, profile_id=plan.metric_profile_id),
                                    pa_reference=PAReference(run_id=need(entry.entry_id, "full_retrain", source)), **common)
        init = InitReference(run_id=need(entry.entry_id, "full_retrain", source)) if task == "few_shot" else None
        return ExperimentConfig(task=TaskType.train_pa, dataset=dataset, model=entry.model, training=training,
                                evaluation=EvaluationConfig(evidence_type=EvidenceType.pa_modeling, profile_id=plan.metric_profile_id),
                                initialization=init, **common)
    surrogate = PAReference(run_id=need(entry.pa_entry, "full_retrain", condition))
    evaluation = EvaluationConfig(evidence_type=EvidenceType.dpd_surrogate, profile_id=plan.metric_profile_id)
    if task == "zero_update":
        return ExperimentConfig(task=TaskType.run_dpd, dataset=dataset, model=entry.model, evaluation=evaluation,
                                dpd_reference=DPDReference(run_id=need(entry.entry_id, "full_retrain", source), transfer=True),
                                pa_reference=surrogate, **common)
    init = InitReference(run_id=need(entry.entry_id, "full_retrain", source)) if task == "few_shot" else None
    return ExperimentConfig(task=TaskType.train_dpd, dataset=dataset, model=entry.model, training=training,
                            evaluation=evaluation, pa_reference=surrogate, initialization=init, **common)


def refusals_path(ws: Workspace, plan: AdaptationPlan) -> Path:
    return ws.root / REPORTS_DIR / f"{plan.plan_sha256[:12]}.refusals.json"


def run_plan(ws: Workspace, plan: AdaptationPlan, *,
             on_cell: Optional[Callable[[str, RunRecord], None]] = None) -> Dict[str, RunRecord]:
    """Execute every cell in this process; finished runs are reused. A cell whose configuration is refused, or whose
    dependency failed, is recorded next to the plan so the report can show why it has no number."""
    runs: Dict[str, RunRecord] = find_runs(ws, plan)
    refusals: Dict[str, str] = {}
    for seed in plan.seeds:
        for entry, task, condition, budget in cells_of(plan):
            key = cell_key(plan, entry.entry_id, task, condition.condition_id, budget, seed)
            existing = runs.get(key)
            if existing is not None and existing.status == RunStatus.succeeded:
                continue
            try:
                config = _config(plan, entry, task, condition, budget, seed, runs)
                record = create_run(ws, config, idempotency_key=key)
            except KeyError as err:
                refusals[key] = f"dependency missing: {err.args[0]}"
                continue
            except ConfigError as err:
                refusals[key] = "; ".join(f"{i.field}: {i.message}" for i in err.issues)
                continue
            if record.status == RunStatus.queued:
                record = execute_run(ws, record.run_id)
            runs[key] = record
            if on_cell is not None:
                on_cell(key, record)
    path = refusals_path(ws, plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, refusals)
    return runs


def find_runs(ws: Workspace, plan: AdaptationPlan) -> Dict[str, RunRecord]:
    prefix = f"adapt-{plan.plan_sha256[:12]}-"
    return {r.idempotency_key: r for r in list_runs(ws) if r.idempotency_key and r.idempotency_key.startswith(prefix)}


# --- reports ------------------------------------------------------------------------------------

def build_report(ws: Workspace, plan: AdaptationPlan) -> AdaptationReport:
    """Every number is read from a run's stored result under the plan's profile; nothing is recomputed."""
    card = plan.condition_set
    audits = audit_card(ws, card)
    train_sizes = {a.condition_id: a.train_samples or 0 for a in audits}
    runs = find_runs(ws, plan)
    path = refusals_path(ws, plan)
    refusals: Dict[str, str] = read_json(path) if path.exists() else {}
    profile = get_profile(plan.metric_profile_id)
    cells: List[AdaptationCell] = []
    for seed in plan.seeds:
        for entry, task, condition, budget in cells_of(plan):
            key = cell_key(plan, entry.entry_id, task, condition.condition_id, budget, seed)
            record = runs.get(key)
            new_samples = 0 if task == "zero_update" else (budget or train_sizes[condition.condition_id])
            base = dict(entry_id=entry.entry_id, task=task, condition_id=condition.condition_id, budget_samples=budget,
                        seed=seed, new_samples=new_samples)
            if record is None:
                cells.append(AdaptationCell(status="missing", failure=refusals.get(key, "no run for this cell"), **base))
                continue
            result = load_result(ws, record.run_id, plan.metric_profile_id) if record.status == RunStatus.succeeded else None
            wall = (record.finished_at - record.started_at).total_seconds() if record.started_at and record.finished_at else None
            if result is None:
                failure = f"{record.error.code} [{record.error.stage}]: {record.error.message}" if record.error \
                    else f"run {record.status.value}" + (f": {record.status_reason}" if record.status_reason else "")
                cells.append(AdaptationCell(status="failed", failure=failure, run_id=record.run_id, wall_clock_s=wall,
                                            config_sha256=record.config_sha256, **base))
                continue
            manifest = load_artifacts(ws, record.run_id)
            checkpoints = manifest.by_kind(ArtifactKind.checkpoint) if manifest else []
            metrics = {m.name: m.value for m in result.metrics}
            reached = plan.target.reached(metrics.get(plan.target.metric)) if plan.target else None
            cells.append(AdaptationCell(status="ok", run_id=record.run_id, metrics=metrics, wall_clock_s=wall,
                                        config_sha256=record.config_sha256,
                                        checkpoint_sha256=checkpoints[0].file.sha256 if checkpoints else None,
                                        reached_target=reached, **base))
    aggregates: List[CellAggregate] = []
    for entry, task, condition, budget in cells_of(plan):
        group = [c for c in cells if (c.entry_id, c.task, c.condition_id, c.budget_samples)
                 == (entry.entry_id, task, condition.condition_id, budget)]
        ok = [c for c in group if c.status == "ok"]
        metric_stats: Dict[str, MetricStats] = {}
        for name in sorted({n for c in ok for n in c.metrics}):
            values = [c.metrics[name] for c in ok if c.metrics.get(name) is not None and math.isfinite(c.metrics[name])]
            if values:
                metric_stats[name] = stats(values)
        walls = [c.wall_clock_s for c in ok if c.wall_clock_s is not None]
        reached = [c.reached_target for c in ok if c.reached_target is not None]
        aggregates.append(CellAggregate(
            entry_id=entry.entry_id, task=task, condition_id=condition.condition_id, budget_samples=budget,
            n_seeds=len(plan.seeds), n_ok=len(ok), n_failed=len(group) - len(ok), metrics=metric_stats,
            new_samples=group[0].new_samples if group else 0,
            mean_wall_clock_s=float(statistics.fmean(walls)) if walls else None,
            target_reached_fraction=(sum(reached) / len(reached)) if reached else None))
    measured = all(a.origin == "measured" for a in audits)
    bar = EvidenceBar(n_conditions=len(card.conditions), independent_batches=card.independent_batches,
                      measured_origin=measured,
                      met=len(card.conditions) >= MIN_CONDITIONS_FOR_EVIDENCE and card.independent_batches and measured)
    n_failed = sum(c.status != "ok" for c in cells)
    limitations = [f"single device ({card.device}): nothing here generalises to other devices"]
    if not bar.met:
        why = []
        if len(card.conditions) < MIN_CONDITIONS_FOR_EVIDENCE:
            why.append(f"{len(card.conditions)} conditions along '{card.dimension}' (the bar is {MIN_CONDITIONS_FOR_EVIDENCE})")
        if not card.independent_batches:
            why.append("conditions share a capture batch")
        if not measured:
            why.append("not every condition is a measured dataset (synthetic or unknown origin)")
        limitations.append("below the S17 evidence bar: " + "; ".join(why) + "; this report is a rehearsal of the protocol, not evidence")
    if len(plan.seeds) < 3:
        limitations.append(f"{len(plan.seeds)} training seed(s): seed spread is not established (3 or more needed)")
    if n_failed:
        limitations.append(f"{n_failed} of {len(cells)} cells have no number; each states why in the matrix")
    if not plan.target:
        limitations.append("no target rule: the plan does not define what 'reaching the target' means")
    batches = {c.capture_batch for c in card.conditions}
    repeats = (f"training seeds: {len(plan.seeds)}; capture batches: {len(batches)} "
               f"({'one per condition' if card.independent_batches else 'shared between conditions'}); measurement repeats "
               "are conditions of the card and are never counted as seeds")
    report = AdaptationReport(plan_sha256=plan.plan_sha256, condition_set=card, conditions=audits,
                              metric_profile_id=plan.metric_profile_id, metric_profile_version=profile.version,
                              device=plan.device, seeds=list(plan.seeds), budgets=list(plan.budgets), target=plan.target,
                              cells=cells, aggregates=aggregates, evidence_bar=bar, repeats=repeats,
                              limitations=limitations, software=software_provenance(), machine=_machine())
    return report.sealed()


def reports_dir(ws: Workspace) -> Path:
    return ws.root / REPORTS_DIR


def store_report(ws: Workspace, report: AdaptationReport) -> Path:
    path = reports_dir(ws) / f"{report.plan_sha256[:12]}.report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, report)
    return path


def load_report(path: Path) -> AdaptationReport:
    report = AdaptationReport.model_validate(read_json(Path(path)))
    if not report.intact:
        raise WorkspaceError(f"{path}: the report was edited after it was sealed (report_sha256 does not match)")
    return report


def list_reports(ws: Workspace) -> List[AdaptationReport]:
    directory = reports_dir(ws)
    out = []
    for path in sorted(directory.glob("*.report.json")) if directory.exists() else []:
        try:
            out.append(load_report(path))
        except (WorkspaceError, ValueError):
            continue
    return out


def _fmt(agg: Optional[MetricStats]) -> str:
    if agg is None:
        return "n/a"
    return f"{agg.mean:.2f}" + (f" ± {agg.std:.2f}" if agg.std is not None else "") + f" (n={agg.n})"


def _column(task: str, budget: Optional[int]) -> str:
    return f"{task} b{budget}" if budget else task


def report_markdown(report: AdaptationReport) -> str:
    card = report.condition_set
    bar = report.evidence_bar
    out = [f"# Adaptation report `{PROTOCOL_ID}` — plan {report.plan_sha256[:12]}", "",
           f"Device: {card.device}. Varied dimension: `{card.dimension}`. Profile {report.metric_profile_id} "
           f"v{report.metric_profile_version}, device {report.device}, seeds {report.seeds}, budgets {report.budgets}.", "",
           f"Evidence bar: **{'met' if bar.met else 'not met'}** ({bar.n_conditions} conditions, bar {bar.min_conditions}; "
           f"independent batches: {bar.independent_batches}; measured origin: {bar.measured_origin}).", "", report.repeats, "",
           "## Conditions", "", "| id | role | dataset | batch | values | origin | raw sha256 | train samples |",
           "|---|---|---|---|---|---|---|---|"]
    audits = {a.condition_id: a for a in report.conditions}
    for c in card.conditions:
        a = audits[c.condition_id]
        out.append(f"| {c.condition_id} | {c.role} | {c.dataset_id} | {c.capture_batch} | {c.values} | {a.origin} | "
                   f"{(a.raw_sha256 or '')[:12]} | {a.train_samples} |")
    primary = report.target.metric if report.target else None
    for entry_id in dict.fromkeys(a.entry_id for a in report.aggregates):
        aggs = [a for a in report.aggregates if a.entry_id == entry_id]
        names = sorted({n for a in aggs for n in a.metrics})
        metric = primary if primary in names else (names[0] if names else None)
        columns = sorted({(a.task, a.budget_samples) for a in aggs}, key=lambda t: (ALL_TASKS.index(t[0]), t[1] or 0))
        out += ["", f"## Entry `{entry_id}` — {metric or 'no metric'}", "",
                "| condition | " + " | ".join(_column(t, b) for t, b in columns) + " |", "|---|" + "---|" * len(columns)]
        for cond in card.conditions:
            row = [f"{cond.condition_id} ({cond.role})"]
            for task, budget in columns:
                agg = next((a for a in aggs if (a.task, a.budget_samples, a.condition_id) == (task, budget, cond.condition_id)), None)
                if agg is None:
                    row.append("—")
                    continue
                text = _fmt(agg.metrics.get(metric)) if metric else "n/a"
                if agg.n_failed:
                    reasons = sorted({c.failure or "" for c in report.cells if c.status != "ok" and
                                      (c.entry_id, c.task, c.budget_samples, c.condition_id) == (entry_id, task, budget, cond.condition_id)})
                    text += f" **FAILED {agg.n_failed}/{agg.n_seeds}**: " + "; ".join(reasons)
                if agg.target_reached_fraction is not None:
                    text += f"; target reached {agg.target_reached_fraction:.0%}"
                row.append(text)
            out.append("| " + " | ".join(row) + " |")
        out += ["", "| task | condition | new samples | mean wall clock (s) | device |", "|---|---|---:|---:|---|"]
        out += [f"| {_column(a.task, a.budget_samples)} | {a.condition_id} | {a.new_samples} | "
                f"{'n/a' if a.mean_wall_clock_s is None else f'{a.mean_wall_clock_s:.1f}'} | {report.device} |" for a in aggs]
    out += ["", "## Limitations", ""] + [f"- {lim}" for lim in report.limitations]
    out += ["", f"Report sha256 {report.report_sha256}; plan sha256 {report.plan_sha256}; card sha256 {card.card_sha256}.", ""]
    return "\n".join(out)
