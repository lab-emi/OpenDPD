"""``leaderboard-v1`` tooling (plan S20): draft a submission from finished runs, check it (with an optional
recomputation in a fresh workspace), seed a board from a hash-bound benchmark-v1 report, add / review / amend
entries, and render every board to Markdown. Boards are files; nothing here talks to a network."""

from __future__ import annotations

import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from opendpd.core.metrics.registry import get_profile
from opendpd.core.registry import get_model
from opendpd.schemas import ArtifactKind, BenchmarkReport, BetterDirection, MetricStats, RunStatus, SeedScore, TaskType
from opendpd.schemas.leaderboard import (
    MIN_EXTERNAL_ACCEPTED, MIN_INDEPENDENT_RECOMPUTATIONS, MIN_SEEDS, RECOMPUTE_TOLERANCE, TRACK_EVIDENCE, TRACK_GATES,
    TRACK_RANK_METRIC, TRANSITIONS, BoardEntry, CheckItem, ComparabilityKey, DataCard, HistoryEvent, Leaderboard, LicenceCheck,
    MethodCard, ModelCard, PackageRef, Recomputation, ResourceBudget, ResultSummary, Review, SubmissionCard, SubmissionCheck,
    Submitter,
)
from opendpd.services import experiments
from opendpd.services.benchmark import aggregate
from opendpd.services.evaluation import evaluate_run
from opendpd.services.packages import PackageError, export_run, import_package, inspect_package
from opendpd.services.workspace import Workspace, WorkspaceError, read_json, sha256_file, write_json_atomic

TODO = "TODO"    # the draft's placeholder for statements only a person can make
TASK_TRACK: Dict[TaskType, str] = {TaskType.train_pa: "pa_modeling", TaskType.evaluate_pa: "pa_modeling",
                                   TaskType.train_dpd: "dpd_surrogate", TaskType.run_dpd: "dpd_surrogate",
                                   TaskType.evaluate_measured: "dpd_measured"}
STATEMENT_FIELDS = ("method.description", "method.licence", "data.statement", "licence.code", "licence.weights", "licence.data",
                    "licence.statement", "conflict_of_interest", "citation", "isolated_validation")


# --- prepare ---------------------------------------------------------------------------------------------------

def prepare(ws: Workspace, run_ids: List[str], out_dir: Path, *, submission_id: str, submitter: str,
            kind: str = "external") -> Tuple[SubmissionCard, Path]:
    """Export one share package per run and draft the card; the statements a person must make are ``TODO``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds: List[SeedScore] = []
    packages: List[PackageRef] = []
    facts: Optional[Tuple[TaskType, str, str, str]] = None
    model_card = data_card = None
    walls: List[float] = []
    device = "unknown"
    for run_id in run_ids:
        record = experiments.load_run(ws, run_id)
        if record.status != RunStatus.succeeded:
            raise WorkspaceError(f"run '{run_id}' did not succeed ({record.status.value}); a submission lists finished runs only")
        resolved = experiments.load_resolved(ws, run_id)
        result = experiments.load_result(ws, run_id)
        if result is None:
            raise WorkspaceError(f"run '{run_id}' has no stored result")
        these = (resolved.task, resolved.model.key, resolved.dataset.id, result.metric_profile_id)
        if facts is None:
            facts = these
        elif these != facts:
            raise WorkspaceError(f"run '{run_id}' differs from the first run in task, model, dataset or profile "
                                 f"({these} vs {facts}); one submission is one method on one data card")
        role = "pa" if resolved.task in (TaskType.train_pa, TaskType.evaluate_pa) else "dpd"
        primary = next(m for m in result.models if m.role == role)
        manifest = experiments.load_artifacts(ws, run_id)
        checkpoints = manifest.by_kind(ArtifactKind.checkpoint) if manifest else []
        wall = (record.finished_at - record.started_at).total_seconds() if record.started_at and record.finished_at else None
        if wall is not None:
            walls.append(wall)
        device = result.device
        seeds.append(SeedScore(seed=result.seed if result.seed is not None else resolved.training.seed, run_id=run_id,
                               config_sha256=record.config_sha256,
                               checkpoint_sha256=checkpoints[0].file.sha256 if checkpoints else None,
                               surrogate_run_id=next((m.run_id for m in result.models if m.role == "pa"), None) if role == "dpd" else None,
                               selected_epoch=result.selected_epoch, metrics={m.name: m.value for m in result.metrics},
                               wall_clock_s=wall))
        package = out_dir / f"{run_id}-share.zip"
        export_run(ws, run_id, package, kind="share")
        packages.append(PackageRef(path=package.name, sha256=sha256_file(package), run_id=run_id))
        if model_card is None:
            model_card = ModelCard(model_key=primary.model.key, parameters=dict(primary.model.parameters),
                                   n_parameters=primary.n_parameters, lookahead_samples=primary.lookahead_samples,
                                   execution_semantics=primary.execution_semantics, training_path=primary.training_path)
            dataset = ws.get_dataset(resolved.dataset.id)
            builtin = dataset.source.name if dataset.source.kind.value == "builtin" else None
            data_card = DataCard(dataset_id=dataset.dataset_id, raw_sha256=dataset.raw_sha256,
                                 availability="public" if builtin else "private",
                                 statement=f"built-in OpenDPD dataset {builtin}" if builtin else TODO)
    assert facts is not None and model_card is not None and data_card is not None
    task, model_key, _, profile_id = facts
    profile = get_profile(profile_id)
    card = SubmissionCard(
        submission_id=submission_id, track=TASK_TRACK[task], submitter=Submitter(name=submitter, kind=kind),
        method=MethodCard(name=get_model(model_key).display_name, description=TODO, licence=TODO), model=model_card, data=data_card,
        result=ResultSummary(metric_profile_id=profile_id, metric_profile_version=profile.version,
                             split_version=resolved.dataset.split_version, evidence_type=TRACK_EVIDENCE[TASK_TRACK[task]],
                             seeds=seeds, metrics=aggregate(seeds),
                             resources=ResourceBudget(device=device, mean_wall_clock_s=statistics.fmean(walls) if walls else None)),
        packages=packages, licence=LicenceCheck(code=TODO, weights=TODO, data=TODO, redistribution_allowed=False, statement=TODO),
        conflict_of_interest=TODO, citation=TODO, isolated_validation=TODO)
    path = out_dir / "submission.json"
    write_json_atomic(path, card)
    return card, path


def load_card(path: Path) -> SubmissionCard:
    return SubmissionCard.model_validate(read_json(Path(path)))


# --- check ------------------------------------------------------------------------------------------------------

def _field(card: SubmissionCard, dotted: str) -> str:
    obj = card
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return str(obj)


def recompute(card: SubmissionCard, base_dir: Path, *, by: str, kind: str, workspace: Optional[Path] = None) -> Recomputation:
    """Import every package into a fresh workspace and re-score the run under the card's profile."""
    if not card.packages:
        return Recomputation(by=by, kind=kind, within_tolerance=False,
                             note="no package to recompute from: the entry cites a benchmark report, not a share package")
    tmp = None
    if workspace is None:
        tmp = tempfile.mkdtemp(prefix="opendpd-recompute-")
        workspace = Path(tmp)
    ws = Workspace.open_or_create(Path(workspace))
    deltas: List[float] = []
    notes: List[str] = []
    try:
        for ref in card.packages:
            seed = next(s for s in card.result.seeds if s.run_id == ref.run_id)
            report = import_package(ws, Path(base_dir) / ref.path)
            if report.dataset_status == "missing":
                return Recomputation(by=by, kind=kind, within_tolerance=False,
                                     note=f"package {ref.path}: dataset '{report.dataset_id}' is not in the package and not "
                                          f"available here ({'; '.join(report.missing)})")
            result = evaluate_run(ws, ref.run_id, card.result.metric_profile_id)
            got = {m.name: m.value for m in result.metrics}
            for name, expected in seed.metrics.items():
                if expected is None or got.get(name) is None:
                    notes.append(f"{ref.run_id}: {name} has no value on one side")
                    continue
                deltas.append(abs(got[name] - expected))
    except (PackageError, WorkspaceError, FileNotFoundError) as err:
        return Recomputation(by=by, kind=kind, within_tolerance=False, note=f"recomputation failed: {err}")
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    worst = max(deltas) if deltas else None
    within = bool(deltas) and worst <= RECOMPUTE_TOLERANCE and not notes
    note = (f"{len(card.packages)} package(s) imported into a fresh workspace and re-scored under "
            f"{card.result.metric_profile_id}; max |delta| {worst:.3g} dB against the card"
            if worst is not None else "nothing could be compared") + ("; " + "; ".join(notes) if notes else "")
    return Recomputation(by=by, kind=kind, within_tolerance=within, max_abs_delta=worst, note=note)


def check(card: SubmissionCard, base_dir: Path, *, board: Optional[Leaderboard] = None,
          recomputation: Optional[Recomputation] = None) -> SubmissionCheck:
    """The review checklist, the same for every submitter. ``fail`` blocks; ``warn`` is shown on the board."""
    items: List[CheckItem] = []
    todo = [f for f in STATEMENT_FIELDS if _field(card, f).strip() in ("", TODO)]
    items.append(CheckItem(name="statements", status="fail" if todo else "ok",
                           detail=f"still TODO: {', '.join(todo)}" if todo else "every statement is filled in"))
    if card.packages:
        problems: List[str] = []
        for ref in card.packages:
            path = Path(base_dir) / ref.path
            if not path.is_file():
                problems.append(f"{ref.path}: missing")
                continue
            if sha256_file(path) != ref.sha256:
                problems.append(f"{ref.path}: sha256 differs from the card")
                continue
            try:
                manifest = inspect_package(path)
            except PackageError as err:
                problems.append(f"{ref.path}: {err.code}: {err}")
                continue
            seed = next(s for s in card.result.seeds if s.run_id == ref.run_id)
            if manifest.kind != "share":
                problems.append(f"{ref.path}: a {manifest.kind} package is private; submit the share package")
            if manifest.run_id != ref.run_id or manifest.config_sha256 != seed.config_sha256:
                problems.append(f"{ref.path}: run or configuration hash differs from seed {seed.seed}")
            if manifest.dataset.dataset_id != card.data.dataset_id or (card.data.raw_sha256 and manifest.dataset.raw_sha256 != card.data.raw_sha256):
                problems.append(f"{ref.path}: dataset differs from the data card")
            if manifest.metric_profile_id not in (None, card.result.metric_profile_id):
                problems.append(f"{ref.path}: result profile {manifest.metric_profile_id} differs from the card")
        items.append(CheckItem(name="packages", status="fail" if problems else "ok",
                               detail="; ".join(problems) if problems else f"{len(card.packages)} share package(s), hashes and run ids match"))
    else:
        items.append(CheckItem(name="packages", status="ok",
                               detail=f"maintainers' entry copied from benchmark-v1 report {card.benchmark_report_sha256[:12]}; "
                                      "the report's per-seed runs are the traceability"))
    n = len(card.result.seeds)
    items.append(CheckItem(name="seeds", status="ok" if n >= MIN_SEEDS else "warn",
                           detail=f"{n} seed(s)" + ("" if n >= MIN_SEEDS else f": uncertainty is not established below {MIN_SEEDS}")))
    if card.data.availability == "public":
        items.append(CheckItem(name="data", status="ok" if card.data.raw_sha256 else "warn",
                               detail="public data with its raw hash" if card.data.raw_sha256 else "public data without a raw hash cannot be matched"))
    else:
        items.append(CheckItem(name="data", status="warn",
                               detail=f"data is {card.data.availability}: shown as such, not presented as publicly reproducible"))
    items.append(CheckItem(name="licence", status="ok" if card.licence.redistribution_allowed else "warn",
                           detail="redistribution allowed" if card.licence.redistribution_allowed
                           else "no redistribution: the board links the method card, it does not host the weights"))
    items.append(CheckItem(name="isolation", status="ok",
                           detail="statement recorded; an organisational rule the software cannot verify, a reviewer confirms it"))
    if board is not None:
        dup = [e.entry_id for e in board.entries if e.submission.submission_id == card.submission_id
               or {p.sha256 for p in e.submission.packages} & {p.sha256 for p in card.packages}]
        items.append(CheckItem(name="duplicate", status="fail" if dup else "ok",
                               detail=f"already on the board as {', '.join(dup)}" if dup else "not on the board yet"))
    if recomputation is not None:
        items.append(CheckItem(name="recomputation", status="ok" if recomputation.within_tolerance else "fail",
                               detail=recomputation.note))
    return SubmissionCheck(submission_id=card.submission_id, items=items, recomputation=recomputation)


# --- boards -----------------------------------------------------------------------------------------------------

def load_board(path: Path) -> Leaderboard:
    return Leaderboard.model_validate(read_json(Path(path)))


def write_board(board: Leaderboard, path: Path) -> Leaderboard:
    """Seal and write the JSON; the Markdown rendering sits next to it."""
    board = board.sealed()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, board)
    path.with_suffix(".md").write_text(board_markdown(board), encoding="utf-8")
    return board


def seed_board(report: BenchmarkReport, *, board_id: str, version: str, track: str, supersedes: Optional[str] = None) -> Leaderboard:
    """The maintainers' reference entries: every number copied from a hash-bound report, marked as self-reported."""
    if track not in TRACK_EVIDENCE:
        raise ValueError(f"track '{track}' is not open: {TRACK_GATES.get(track, track)}")
    if not report.intact:
        raise ValueError("the benchmark report is not intact (report_sha256 does not match its content)")
    by = f"opendpd leaderboard seed (report {report.report_sha256[:12]})"
    entries: List[BoardEntry] = []
    for entry in report.entries:
        if TASK_TRACK.get(entry.task) != track or not entry.seeds:
            continue
        model = get_model(entry.model.key)
        walls = [s.wall_clock_s for s in entry.seeds if s.wall_clock_s is not None]
        failures = ["not evaluated on any other dataset or operating point"]
        if track == "dpd_surrogate":
            failures.append(report.dataset.statement)
        if report.tier == "cpu_regression":
            failures.append("cpu_regression tier: smoke budgets, regression references for this data and hardware, never research results")
        card = SubmissionCard(
            submission_id=f"{report.report_sha256[:12]}-{entry.entry_id}", track=track,
            submitter=Submitter(name="OpenDPD maintainers", kind="maintainer", affiliation="OpenDPD"),
            method=MethodCard(name=model.display_name, description=f"{model.display_name} ({entry.model.key}) trained by {entry.training_path} "
                              f"under benchmark plan {report.plan_sha256[:12]}", reference="https://arxiv.org/abs/2507.06849",
                              code_url="https://github.com/lab-emi/OpenDPD", licence="Apache-2.0"),
            model=ModelCard(model_key=entry.model.key, parameters=dict(entry.model.parameters), n_parameters=entry.n_parameters,
                            lookahead_samples=entry.lookahead_samples, execution_semantics=entry.execution_semantics,
                            training_path=entry.training_path),
            data=DataCard(dataset_id=report.dataset.dataset_id, raw_sha256=report.dataset.raw_sha256, availability="public",
                          statement="built-in OpenDPD dataset (opendpd datasets import-builtin)"),
            result=ResultSummary(metric_profile_id=report.metric_profile_id, metric_profile_version=report.metric_profile_version,
                                 split_version=report.dataset.split_version, evidence_type=TRACK_EVIDENCE[track],
                                 seeds=list(entry.seeds), metrics=dict(entry.aggregate),
                                 resources=ResourceBudget(device=report.device, mean_wall_clock_s=statistics.fmean(walls) if walls else None)),
            benchmark_report_sha256=report.report_sha256, benchmark_plan_sha256=report.plan_sha256, failure_conditions=failures,
            licence=LicenceCheck(code="Apache-2.0", weights="not redistributed (runs of the maintainers' benchmark workspace)",
                                 data="OpenDPD built-in dataset, as shipped", redistribution_allowed=False,
                                 statement="the report, the plan and the run ids are the artefacts; weights are reproduced by the plan"),
            conflict_of_interest="the maintainers' own methods on their own data; the same review rules apply to every entry",
            citation="docs/community/citation.md", isolated_validation="the maintainers' weekly benchmark, not a submission")
        entries.append(BoardEntry(entry_id=entry.entry_id, submission=card, key=ComparabilityKey.of(card), status="accepted",
                                  evidence_grade="self_reported",
                                  history=[HistoryEvent(action="created", by=by,
                                                        reason="copied from the hash-bound benchmark-v1 report; no independent recomputation")]))
    return Leaderboard(board_id=board_id, version=version, track=track, supersedes=supersedes, entries=entries,
                       history=[HistoryEvent(action="created", by=by, reason=f"version {version} seeded with {len(entries)} reference entries")])


def add_entry(board: Leaderboard, card: SubmissionCard, result: SubmissionCheck, *, by: str) -> Leaderboard:
    if not result.passed:
        failed = [i for i in result.items if i.status == "fail"]
        raise ValueError("the submission does not pass the checklist: " + "; ".join(f"{i.name}: {i.detail}" for i in failed))
    if card.track != board.track:
        raise ValueError(f"a {card.track} submission does not go on the {board.track} board")
    entry = BoardEntry(entry_id=card.submission_id, submission=card, key=ComparabilityKey.of(card),
                       history=[HistoryEvent(action="submitted", by=by,
                                             reason="; ".join(f"{i.name}: {i.detail}" for i in result.items if i.status == "warn") or "checklist passed")])
    return board.model_copy(update={"entries": [*board.entries, entry],
                                    "history": [*board.history, HistoryEvent(action="submitted", by=by, reason=f"entry {entry.entry_id}")]})


def _transition(entry: BoardEntry, status: str) -> None:
    if status not in TRANSITIONS[entry.status]:
        raise ValueError(f"entry '{entry.entry_id}' is {entry.status}; it cannot become {status}")


def review_entry(board: Leaderboard, entry_id: str, review: Review) -> Leaderboard:
    """Append the review; a decision moves the status. Acceptance never rests on a failed recomputation."""
    entry = board.entry(entry_id)
    status = {"accepted": "accepted", "rejected": "rejected", "needs_changes": "under_review"}[review.decision]
    _transition(entry, status)
    if review.decision == "accepted" and review.recomputation is not None and not review.recomputation.within_tolerance:
        raise ValueError("a recomputation outside the tolerance cannot accept the entry")
    grade = entry.evidence_grade
    if review.decision == "accepted":
        grade = "reviewed"
        recomputed = review.recomputation is not None and review.recomputation.within_tolerance
        if recomputed and review.recomputation.by != entry.submission.submitter.name:
            grade = "independently_recomputed"
    event = HistoryEvent(action="reviewed" if status == "under_review" else status, by=review.reviewer, reason=review.notes)
    updated = entry.model_copy(update={"status": status, "evidence_grade": grade, "reviews": [*entry.reviews, review],
                                       "history": [*entry.history, event]})
    return board.model_copy(update={"entries": [updated if e.entry_id == entry_id else e for e in board.entries],
                                    "history": [*board.history, event.model_copy(update={"reason": f"entry {entry_id}: {review.notes}"})]})


def amend_entry(board: Leaderboard, entry_id: str, *, action: str, by: str, reason: str,
                corrected: Optional[SubmissionCard] = None) -> Leaderboard:
    """Retract or correct an accepted entry; what was shown before stays in the history."""
    entry = board.entry(entry_id)
    if action not in ("retract", "correct"):
        raise ValueError("action is retract or correct")
    status = "retracted" if action == "retract" else "corrected"
    _transition(entry, status)
    event = HistoryEvent(action=status, by=by, reason=reason, previous_metrics=dict(entry.submission.result.metrics))
    update: Dict[str, object] = {"status": status, "history": [*entry.history, event]}
    if action == "correct":
        if corrected is None:
            raise ValueError("a correction needs the corrected submission card")
        if corrected.submission_id != entry.submission.submission_id or corrected.submitter != entry.submission.submitter:
            raise ValueError("a correction keeps the submission id and the submitter")
        update["submission"] = corrected
        update["key"] = ComparabilityKey.of(corrected)
    updated = entry.model_copy(update=update)
    return board.model_copy(update={"entries": [updated if e.entry_id == entry_id else e for e in board.entries],
                                    "history": [*board.history, event.model_copy(update={"reason": f"entry {entry_id}: {reason}", "previous_metrics": None})]})


# --- rendering --------------------------------------------------------------------------------------------------

def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _cell(stats: Optional[MetricStats]) -> str:
    if stats is None:
        return "n/a"
    return f"{stats.mean:.2f} ± {stats.std:.2f} (n={stats.n})" if stats.std is not None else f"{stats.mean:.2f} (n={stats.n}, no std)"


def _order(track: str, profile_id: str) -> Tuple[str, bool]:
    name = TRACK_RANK_METRIC[track]
    try:
        better = get_profile(profile_id).metric(name).better
    except (KeyError, ValueError):
        better = BetterDirection.lower
    return name, better == BetterDirection.lower


def board_markdown(board: Leaderboard) -> str:
    ranked = [e for e in board.entries if e.ranked]
    lines = [f"# {board.board_id} {board.version} — {board.label}", "",
             f"Track `{board.track}` ({TRACK_EVIDENCE[board.track].value} evidence), protocol `{board.protocol_id}`"
             + (f", supersedes board `{board.supersedes[:12]}`" if board.supersedes else "") + ".", ""]
    if board.community_bar_met:
        lines.append(f"{board.external_accepted} external submissions accepted and {board.independent_recomputations} of them "
                     "independently recomputed: the community bar is met.")
    else:
        lines.append(f"This is a **reference benchmark**, not a community standard: {board.external_accepted} of the "
                     f"{MIN_EXTERNAL_ACCEPTED} required external submissions are accepted and {board.independent_recomputations} of the "
                     f"{MIN_INDEPENDENT_RECOMPUTATIONS} required independent recomputations exist. Entries by the maintainers are "
                     "self-reported unless their evidence grade says otherwise.")
    lines += ["", "Entries rank only inside one comparability group (data, operating point, metric profile version, split, "
              "execution semantics, resource class). Every number is mean ± sample standard deviation over the listed seeds; "
              "wall clock is a measurement on the stated device, never power. Failure conditions are the submitter's own "
              "statement of where the method fails or was not tried. Evidence grades: self_reported, reviewed, "
              "independently_recomputed.", ""]
    groups: Dict[str, List[BoardEntry]] = {}
    for e in ranked:
        groups.setdefault(e.key.label(), []).append(e)
    if not groups:
        lines += ["No ranked entry.", ""]
    for label, entries in groups.items():
        profile_id = entries[0].submission.result.metric_profile_id
        rank_metric, lower = _order(board.track, profile_id)
        names = sorted({n for e in entries for n in e.submission.result.metrics})
        if rank_metric in names:
            names.remove(rank_metric)
            names.insert(0, rank_metric)

        def sort_key(e: BoardEntry):
            s = e.submission.result.metrics.get(rank_metric)
            return (s is None, (s.mean if lower else -s.mean) if s else 0.0)

        lines += [f"## Group: {label}", "", f"Ordered by {rank_metric} ({'lower' if lower else 'higher'} is better); "
                  "the other columns are not a tie-break.", "",
                  "| # | Method | Model (params) | " + " | ".join(names) + " | Wall clock (s) | Grade | Status | Submitter | Failure conditions | Traceability |",
                  "|---|---|---|" + "---|" * len(names) + "---|---|---|---|---|---|"]
        for i, e in enumerate(sorted(entries, key=sort_key), start=1):
            s = e.submission
            trace = (", ".join(f"`{p.path}` ({p.sha256[:12]})" for p in s.packages) if s.packages
                     else f"benchmark report `{s.benchmark_report_sha256[:12]}`, plan `{(s.benchmark_plan_sha256 or '')[:12]}`")
            trace += "; runs " + ", ".join(f"{x.run_id} ({(x.checkpoint_sha256 or 'no checkpoint')[:12]})" for x in s.result.seeds)
            lines.append(f"| {i} | {s.method.name} | {s.model.model_key} ({s.model.n_parameters if s.model.n_parameters is not None else 'n/a'}) | "
                         + " | ".join(_cell(s.result.metrics.get(n)) for n in names)
                         + f" | {_fmt(s.result.resources.mean_wall_clock_s)} on {s.result.resources.device} | {e.evidence_grade} | {e.status} | "
                         f"{s.submitter.name} ({s.submitter.kind}) | {'; '.join(s.failure_conditions) or 'none stated'} | {trace} |")
        lines.append("")
    others = [e for e in board.entries if not e.ranked]
    if others:
        lines += ["## Not ranked", "", "| Entry | Status | Grade | Submitter | Last event |", "|---|---|---|---|---|"]
        for e in others:
            last = e.history[-1] if e.history else None
            lines.append(f"| {e.entry_id} | {e.status} | {e.evidence_grade} | {e.submission.submitter.name} ({e.submission.submitter.kind}) | "
                         f"{(last.action + ': ' + last.reason) if last else 'none'} |")
        lines.append("")
    lines += ["## History", ""]
    for h in board.history:
        lines.append(f"- {h.date.isoformat()} {h.action} by {h.by}: {h.reason}")
    for e in board.entries:
        for h in e.history:
            extra = f" (previous metrics kept: {', '.join(f'{k} {v.mean:.2f}' for k, v in h.previous_metrics.items())})" if h.previous_metrics else ""
            lines.append(f"- {h.date.isoformat()} entry {e.entry_id} {h.action} by {h.by}: {h.reason}{extra}")
    lines += ["", f"Board hash: `{board.board_sha256 or 'unsealed'}`", ""]
    return "\n".join(lines)
