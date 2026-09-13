"""Preview and supervise experiment matrices without running training in HTTP handlers.

The condition planner builds the scientific dependencies; Supervisor owns every
worker, resource slot, cancellation and run artifact. This service only keeps a
durable board and dispatches its next eligible ordinary run.
"""

import logging
import math
import threading
import time
from collections import defaultdict
from pydantic import ValidationError

from opendpd.core.metrics import get_profile
from opendpd.core.registry import get_model, validate_parameters, RegistryError
from opendpd.schemas import AdaptationEntry, AdaptationPlan, DatasetRef, ExecutionConfig, EvaluationConfig, PAReference, RunStatus, TaskType, TERMINAL_STATUSES
from opendpd.schemas.common import utcnow
from opendpd.schemas.benchmark import TEST_SET_POLICY
from opendpd.schemas.sweep import SweepDraft, SweepPreview, SweepRecord, SweepCell, SweepAggregate, SweepReport
from opendpd.services import adaptation, benchmark, experiments
from opendpd.services.config import ConfigError, ConfigIssue, validate
from opendpd.services.recipes import instantiate
from opendpd.services.review import object_sha
from opendpd.services.workspace import WorkspaceError, read_json, write_json_atomic

log = logging.getLogger(__name__)


def audit_conditions(ws, card):
    """Keep the existing data audit; require physical dimension values in newly authored cards."""
    allowed = {"capture_batch", "output_power_dbm", "carrier_frequency_hz", "bandwidth_hz", "temperature_c", "supply_v", "mode", "load", "vswr"}
    if card.dimension not in allowed:
        raise WorkspaceError("choose one supported physical dimension or capture_batch")
    for c in card.conditions:
        value = c.values.get(card.dimension)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise WorkspaceError(f"{c.condition_id}: declare the value of {card.dimension}")
        if any(isinstance(v, float) and not math.isfinite(v) for v in c.values.values()):
            raise WorkspaceError(f"{c.condition_id}: physical conditions must be finite")
        if card.dimension == "vswr" and (not isinstance(value, (int, float)) or value < 1 or c.values.get("reflection_phase_deg") is None):
            raise WorkspaceError(f"{c.condition_id}: VSWR needs a value >= 1 and reflection_phase_deg")
    return adaptation.audit_card(ws, card)


def _templates(draft):
    out = {}
    dataset = draft.dataset or DatasetRef(id=draft.condition_set.source.dataset_id)
    for m in draft.methods:
        try:
            config = m.config.model_copy(deep=True) if m.config else instantiate(
                m.recipe_id, dataset.id, pa_run_id=draft.fixed_pa_run_id or ("deferred-pa" if draft.mode == "cross_condition" else None))
        except (KeyError, ValueError) as err:
            raise WorkspaceError(str(err)) from None
        if config.task not in (TaskType.train_pa, TaskType.train_dpd):
            raise WorkspaceError("sweep methods train or fit PA/DPD models; other tasks use their existing workflows")
        if draft.mode == "same_condition" and config.task == TaskType.train_dpd:
            if not draft.fixed_pa_run_id:
                raise WorkspaceError("select a finished PA run shared by every DPD method and seed")
            config.pa_reference = PAReference(run_id=draft.fixed_pa_run_id, seed_policy="fixed_surrogate")
        if config.initialization is not None:
            raise WorkspaceError("copied configurations must clear initialization; cross-condition few-shot tasks declare it in the plan")
        if draft.mode == "cross_condition" and config.quantization and config.quantization.enabled:
            raise WorkspaceError("conditions-v1 does not yet preserve QAT precision/pretraining across adaptation; use a same-condition precision matrix")
        config.dataset = dataset
        config.execution = ExecutionConfig(device=draft.device)
        config.evaluation = EvaluationConfig(profile_id=draft.metric_profile_id, evidence_type=config.evaluation.evidence_type)
        out[m.entry_id] = config
    return out


def preview(ws, draft: SweepDraft):
    try:
        return _preview(ws, draft)
    except ValidationError as err:
        raise ConfigError([ConfigIssue(".".join(map(str, e["loc"])), e["msg"]) for e in err.errors()]) from None
    except (KeyError, ValueError) as err:
        if isinstance(err, ConfigError):
            raise
        raise WorkspaceError(str(err)) from None


def _preview(ws, draft: SweepDraft):
    get_profile(draft.metric_profile_id)
    errors, warnings, sources = [], [], {}
    templates = _templates(draft)
    plan = None
    if draft.mode == "cross_condition":
        card = adaptation.seal_card(draft.condition_set)
        audits = audit_conditions(ws, card)
        if len(audits) < 3 or not card.independent_batches or any(a.origin != "measured" for a in audits):
            warnings.append("Below the conditions-v1 evidence bar: require at least three independent measured conditions; this matrix is a protocol rehearsal.")
        plan = AdaptationPlan(condition_set=card, entries=[AdaptationEntry(
            entry_id=m.entry_id, task=templates[m.entry_id].task, recipe_id=templates[m.entry_id].recipe_id,
            model=templates[m.entry_id].model, training=templates[m.entry_id].training, pa_entry=m.pa_entry)
            for m in draft.methods], tasks=draft.tasks, seeds=draft.seeds, budgets=draft.budgets,
            metric_profile_id=draft.metric_profile_id, device=draft.device)
        plan = plan.model_copy(update={"plan_sha256": plan.compute_sha256()})
        descriptions = [(entry.entry_id, task, c.condition_id, c.dataset_id, budget) for entry, task, c, budget in adaptation.cells_of(plan)]
        warnings.append("Source-model and PA checkpoint binding for dependent cells is validated when their prerequisite run finishes.")
    else:
        descriptions = [(m.entry_id, "full_retrain", draft.dataset.id, draft.dataset.id, templates[m.entry_id].training.train_samples) for m in draft.methods]
    if len(descriptions) * len(draft.seeds) > draft.max_runs:
        errors.append(f"matrix needs {len(descriptions) * len(draft.seeds)} runs; declared limit is {draft.max_runs}")
    if len(descriptions) * len(draft.seeds) > 256:
        raise WorkspaceError(errors[0] + "; reduce the matrix to at most 256 cells")
    sizes = {}
    for dataset_id in sorted({d[3] for d in descriptions}):
        manifest = ws.get_dataset(dataset_id)
        sources[f"dataset:{dataset_id}"] = object_sha(manifest)
        sizes[dataset_id] = adaptation._train_split_size(ws, dataset_id, draft.dataset.preprocessing_version if draft.dataset else "raw-v1")
    if draft.fixed_pa_run_id:
        pa = experiments.load_result(ws, draft.fixed_pa_run_id)
        if pa is None:
            raise WorkspaceError("the fixed PA run has no result")
        sources[f"pa:{draft.fixed_pa_run_id}"] = object_sha(pa)
    for config in templates.values():
        if config.quantization and config.quantization.enabled and config.quantization.pretrained_run_id:
            ref_id = config.quantization.pretrained_run_id
            result = experiments.load_result(ws, ref_id)
            if result is None:
                raise WorkspaceError("the float QAT pretraining run has no result")
            sources[f"pretraining:{ref_id}"] = object_sha(result)
            warnings.append(f"QAT uses float pretraining {ref_id}; its training budget is additional to the matrix sample × epochs. Software fake quantization retains FP32 feature extraction.")
    cells = []
    for seed in draft.seeds:
        for entry_id, task, condition_id, dataset_id, budget in descriptions:
            config = templates[entry_id].model_copy(deep=True)
            config.training.seed = seed
            config.dataset = DatasetRef(id=dataset_id) if plan else draft.dataset
            if plan:
                config.training.train_samples = budget
                check = validate(config)
                # No checkpoint exists yet for these dependencies; everything else must validate now.
                issues = [i for i in check.errors if i.field != "pa_reference"]
                try:
                    validate_parameters(config.model.key, config.model.parameters, "pa" if config.task == TaskType.train_pa else "dpd")
                    if budget is not None and budget < config.training.frame_length and get_model(config.model.key).training_method != "least_squares":
                        errors.append(f"{entry_id}/{condition_id}: sample budget {budget} is shorter than one frame ({config.training.frame_length})")
                except RegistryError as err:
                    errors.append(str(err))
            else:
                check = experiments.validate_experiment(ws, config)
                issues = check.errors
            cell_id = f"cell-{len(cells):04d}"
            errors.extend(f"{cell_id} {entry_id}/{condition_id}/seed {seed}: {i.field}: {i.message}" for i in issues)
            full_size = sizes[dataset_id]
            if budget is not None and full_size is not None and budget > full_size:
                errors.append(f"{cell_id}: requested {budget} training samples; {dataset_id} has {full_size}")
            if task != "zero_update" and full_size is None:
                errors.append(f"{cell_id}: training split sample count is unavailable")
            epochs = config.training.epochs if task != "zero_update" else 0
            cells.append(SweepCell(cell_id=cell_id, entry_id=entry_id, condition_id=condition_id, task=task, seed=seed,
                                   budget_samples=budget, train_samples=0 if task == "zero_update" else budget or full_size, epochs=epochs))
    if len(draft.seeds) < 3:
        warnings.append("Fewer than three training seeds: this is a smoke/rehearsal matrix; seed spread is not established.")
    warnings.extend([benchmark.SEED_NOTE.format(n=len(draft.seeds)), benchmark.COMPUTE_NOTE,
                     "Sample × epochs is a declared training-work budget, not an estimate of MACs, runtime, energy or fitting cost. "
                     "The wall-clock limit includes waiting in the queue. Failed and cancelled attempts remain in the board."])
    content = draft.model_dump(mode="json", exclude={"condition_set": {"created_at", "card_sha256"}})
    sha = object_sha({"draft": content, "templates": {k: v.model_dump(mode="json") for k, v in templates.items()}, "source_hashes": sources})
    return SweepPreview(draft=draft, templates=templates, plan_sha256=sha, adaptation=plan, source_hashes=sources, cells=cells,
                        training_runs=sum(c.epochs > 0 for c in cells), evaluation_runs=sum(c.epochs == 0 for c in cells),
                        sample_epochs=sum(c.epochs * c.train_samples for c in cells) if all(c.train_samples is not None for c in cells) else None,
                        errors=list(dict.fromkeys(errors)), warnings=warnings)


def cell_config(record, cell, runs):
    preview = record.preview
    template = preview.templates[cell.entry_id]
    plan = preview.adaptation
    if plan:
        entry = next(e for e in plan.entries if e.entry_id == cell.entry_id)
        cond = next(c for c in plan.condition_set.conditions if c.condition_id == cell.condition_id)
        mapped = {adaptation.cell_key(plan, c.entry_id, c.task, c.condition_id, c.budget_samples, c.seed): runs[c.run_id]
                  for c in record.cells if c.run_id in runs}
        config = adaptation._config(plan, entry, cell.task, cond, cell.budget_samples, cell.seed, mapped)
        config.quantization = template.quantization
    else:
        config = template.model_copy(deep=True)
        config.training.seed = cell.seed
    config.name = f"{preview.draft.title}: {cell.entry_id} / {cell.condition_id} / {cell.task} / seed {cell.seed}"
    config.notes = (config.notes or "") + f"\nsweep-v1 plan {preview.plan_sha256}, {cell.cell_id}; " + TEST_SET_POLICY
    return config


def report(ws, record):
    groups = defaultdict(list)
    hashes = {}
    warnings = list(record.preview.warnings)
    changed = changed_sources(ws, record.preview)
    if changed:
        warnings.append("Source data or fixed PA changed after registration: " + ", ".join(changed))
    for cell in record.cells:
        result = experiments.load_result(ws, cell.run_id, record.preview.draft.metric_profile_id) if cell.run_id and cell.status == "succeeded" else None
        if result:
            hashes[cell.run_id] = object_sha(result)
        groups[(cell.entry_id, cell.condition_id, cell.task, cell.budget_samples)].append((cell, result))
    aggregates = []
    for (entry, cond, task, budget), rows in groups.items():
        results = [r for c, r in rows if r]
        metrics = defaultdict(list)
        for result in results:
            for m in result.metrics:
                if m.value is not None and math.isfinite(m.value) and m.status.value == "ok":
                    metrics[m.name].append(m.value)
        aggregates.append(SweepAggregate(entry_id=entry, condition_id=cond, task=task, budget_samples=budget,
                                        n_requested_seeds=len(rows), n_succeeded=len(results),
                                        metrics={name: benchmark.stats(values) for name, values in metrics.items()},
                                        run_ids=[r.run_id for r in results]))
    warnings.append("Mean and sample SD describe stored metric values across training seeds, not independent hardware captures or total measurement uncertainty. Metric n may be smaller than the planned seed count.")
    if any(c.status != "succeeded" for c in record.cells):
        warnings.append("Incomplete matrix: inspect every failed, cancelled, blocked and pending cell before drawing conclusions.")
    return SweepReport(sweep_id=record.sweep_id, plan_sha256=record.preview.plan_sha256, profile_id=record.preview.draft.metric_profile_id,
                       aggregates=aggregates, result_hashes=hashes, warnings=warnings)


def changed_sources(ws, candidate):
    changed = []
    for key, digest in candidate.source_hashes.items():
        kind, id = key.split(":", 1)
        try:
            source = ws.get_dataset(id) if kind == "dataset" else experiments.load_result(ws, id)
            if source is None or object_sha(source) != digest:
                changed.append(key)
        except WorkspaceError:
            changed.append(key)
    return changed


class SweepController:
    def __init__(self, ws, supervisor):
        self.ws, self.supervisor = ws, supervisor
        self.root = ws.root / "sweeps"
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._last = {}

    def _save(self, record):
        self.root.mkdir(exist_ok=True)
        write_json_atomic(self.root / f"{record.sweep_id}.json", record)

    def list(self):
        with self._lock:
            return sorted([SweepRecord.model_validate(read_json(p)) for p in self.root.glob("sweep-*.json")], key=lambda r: r.created_at, reverse=True)

    def get(self, sweep_id):
        from pydantic import TypeAdapter
        from opendpd.schemas.common import Slug
        TypeAdapter(Slug).validate_python(sweep_id)
        path = self.root / f"{sweep_id}.json"
        if not path.is_file():
            raise WorkspaceError("sweep not found")
        with self._lock:
            return SweepRecord.model_validate(read_json(path))

    def create(self, draft):
        candidate = preview(self.ws, draft)
        if candidate.errors:
            raise WorkspaceError("; ".join(candidate.errors))
        record = SweepRecord(sweep_id=f"sweep-{candidate.plan_sha256}", preview=candidate, cells=candidate.cells)
        with self._lock:
            if (self.root / f"{record.sweep_id}.json").exists():
                return self.get(record.sweep_id)
            self._save(record)
        return record

    def start_board(self, sweep_id, resume_failed=False):
        with self._lock:
            record = self.get(sweep_id)
            if record.status in ("running", "complete"):
                return record
            fresh = preview(self.ws, record.preview.draft)
            if fresh.plan_sha256 != record.preview.plan_sha256 or fresh.errors:
                raise WorkspaceError("source data, fixed PA or validation changed; preview a new sweep")
            if record.status != "ready" and not resume_failed:
                raise WorkspaceError("use resume_failed to retry unfinished cells; previous attempts are preserved")
            if record.elapsed_seconds >= record.preview.draft.max_wall_clock_seconds:
                raise WorkspaceError("wall-clock budget exhausted; copy the plan with a larger declared budget")
            for c in record.cells:
                run = self.supervisor.store.get_run(c.run_id) if c.run_id else None
                if run and run.status not in TERMINAL_STATUSES:
                    raise WorkspaceError("wait for the previous worker to finish cancellation before resuming")
                if run:
                    c.status = run.status.value
                if c.status in ("failed", "blocked", "cancelled", "interrupted"):
                    c.status, c.run_id, c.reason = "pending", None, None
            record.status, record.started_at, record.reason = "running", record.started_at or utcnow(), None
            self._last[record.sweep_id] = time.monotonic()
            self._save(record)
            return record

    def cancel(self, sweep_id, reason="cancelled by user"):
        with self._lock:
            record = self.get(sweep_id)
            if record.status == "complete":
                return record
            for cell in record.cells:
                if cell.run_id:
                    run = self.supervisor.store.get_run(cell.run_id)
                    if run and run.status not in TERMINAL_STATUSES:
                        cell.status = self.supervisor.cancel(run.run_id).status.value
                elif cell.status == "pending":
                    cell.status, cell.reason = "cancelled", reason
            record.status, record.reason = "cancelled", reason
            self._save(record)
            return record

    def tick(self):
        with self._lock:
            for record in self.list():
                active = False
                runs = {}
                changed = False
                for cell in record.cells:
                    if not cell.run_id:
                        continue
                    run = self.supervisor.store.get_run(cell.run_id)
                    if run is None:
                        cell.status, cell.reason = "interrupted", "run record missing"
                        changed = True
                        continue
                    runs[run.run_id] = run
                    if cell.status != run.status.value:
                        cell.status = run.status.value
                        cell.reason = run.error.message if run.error else run.status_reason
                        changed = True
                    active |= run.status not in TERMINAL_STATUSES
                if record.status != "running":
                    if changed:
                        self._save(record)
                    continue
                now = time.monotonic()
                record.elapsed_seconds += max(0., now - self._last.get(record.sweep_id, now))
                self._last[record.sweep_id] = now
                if record.elapsed_seconds >= record.preview.draft.max_wall_clock_seconds:
                    self._save(record)
                    self.cancel(record.sweep_id, "declared wall-clock budget exhausted")
                    continue
                if not active:
                    changed = changed_sources(self.ws, record.preview)
                    if changed:
                        record.status, record.reason = "interrupted", "sources changed; preview a new sweep: " + ", ".join(changed)
                        self._save(record)
                        continue
                    pending = next((c for c in record.cells if c.status == "pending"), None)
                    if pending is None:
                        record.status = "complete" if all(c.status == "succeeded" for c in record.cells) else "needs_attention"
                    else:
                        try:
                            config = cell_config(record, pending, runs)
                            run = self.supervisor.submit(config, idempotency_key=f"{record.sweep_id}:{pending.cell_id}:attempt-{len(pending.attempts)}",
                                                         parent_run_id=pending.attempts[-1] if pending.attempts else None)
                            pending.run_id, pending.status, pending.config_sha256 = run.run_id, run.status.value, run.config_sha256
                            if run.run_id not in pending.attempts:
                                pending.attempts.append(run.run_id)
                        except KeyError as err:
                            pending.status, pending.reason = "blocked", str(err)
                        except (ConfigError, WorkspaceError, ValueError) as err:
                            pending.status, pending.reason = "failed", str(err)
                self._save(record)

    def start(self):
        # Resuming after a restart is explicit; source changes are re-audited first.
        for record in self.list():
            if record.status == "running":
                record.status, record.reason = "interrupted", "service restarted; resume unfinished cells"
                for c in record.cells:
                    if c.status in ("running", "queued", "cancel_requested"):
                        c.status = "interrupted"
                self._save(record)

        def loop():
            while not self._stop.wait(.25):
                try:
                    self.tick()
                except Exception:
                    log.exception("sweep dispatch failed")

        self._thread = threading.Thread(target=loop, name="opendpd-sweeps", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        with self._lock:
            for record in self.list():
                if record.status == "running":
                    record.status, record.reason = "interrupted", "service stopped; resume unfinished cells"
                    self._save(record)
