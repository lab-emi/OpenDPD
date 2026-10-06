"""Arena trust boundary, complete sweep evidence, per-ranking cohort ranks and bounded lifecycle."""

import copy
import json
import os
import threading

import pytest
from pydantic import ValidationError

from opendpd.core import arena as scoring, arena_ops
from opendpd.core.backbone_template import DEFAULT_DEFINITION, TEMPLATE
from opendpd.schemas.arena import (ArenaBackbone, ArenaBoard, ArenaBudgetResult, ArenaMetricSummary,
    ArenaProtocol, ArenaRankEntry, ArenaRanking, ArenaRow, ArenaRule, ArenaSubmission, ArenaSubmissionRequest)
from opendpd.services.arena import ArenaController
from opendpd.services.user_backbones import BackboneController
from opendpd.services.workspace import (Conflict, InvalidInput, Workspace, WorkspaceError,
    read_json, write_json_atomic)

CONDITIONS = ["nominal", "compressed"]
METRICS = dict(nmse_db=-25, aclr_db=-30, baseline_nmse_db=-20, baseline_aclr_db=-25,
               nmse_improvement_db=5, aclr_improvement_db=5)


@pytest.fixture
def protocol():
    return ArenaProtocol(protocol_id="arena-test-v2", protocol_sha256="a" * 64, training_sha256="c" * 64,
        title="Test", description="Frozen test protocol", seeds=[11, 22, 33], budgets=scoring.BUDGETS,
        boards=[ArenaBoard(board_id="dpa-160mhz", title="Synthetic suite", description="Simulation",
            evidence_type="synthetic_simulation", evidence_label="Synthetic PA simulation", dataset="synthetic",
            conditions=CONDITIONS)], rules=[ArenaRule(title="Fixed", description="No tuning")],
        rankings=[ArenaRanking(**ranking) for ranking in scoring.RANKINGS],
        score_formula="sweep mean of quality - cost", training={"epochs": 2},
        scoring={"budgets": scoring.BUDGETS}, cost_model={"operations": "ops = mul + add"})


def sweep_points(backbone="gru", score=5., supplied=None, seeds=3):
    """The registered sweep with its analytic cost; the gated values are declared by each test."""
    points = []
    for entry in scoring.sweep(backbone, supplied):
        parameters = entry["model_parameters"]
        if parameters is None:
            points.append(dict(budget=entry["budget"], available=False, model_parameters=None,
                               reasons=["No configuration of this backbone belongs to this parameter budget"]))
            continue
        cost = arena_ops.count(backbone, parameters)
        points.append(dict(budget=entry["budget"], available=True, model_parameters=parameters,
            parameters=arena_ops.parameter_count(backbone, parameters),
            mul=cost["mul"], add=cost["add"], ops=cost["ops"], nonlinear=cost["nonlinear"],
            qualified=True, quality_db=score, quality_std_db=0., quality_conservative_db=score, score=score,
            metrics=dict(METRICS), expected_cases=len(CONDITIONS) * seeds, completed_cases=len(CONDITIONS) * seeds))
    return points


def values(protocol, **updates):
    """A complete evaluator result as plain data, so that a test can damage any part of it."""
    backbone, score = updates.get("backbone", "gru"), updates.get("score", 5.)
    seeds = updates.get("seeds", protocol.seeds)
    points = updates.get("budgets") or sweep_points(backbone, 5. if score is None else score, seeds=len(seeds))
    available = [point for point in points if point["available"]]
    cases = [{"budget": point["budget"], "condition_id": condition, "seed": seed, "parameters": point["parameters"]}
             for point in available for condition in CONDITIONS for seed in seeds]
    row = dict(entry_id=f"official-{backbone}", board_id="dpa-160mhz", backbone=backbone, display_name="GRU",
        origin="official", status="succeeded", eligible=True, protocol_sha256=protocol.protocol_sha256,
        score=score, rankings={ranking.ranking_id: {"score": score} for ranking in protocol.rankings},
        budgets=points, qualified_budgets=len(available), available_budgets=len(available),
        best_budget=available[-1]["budget"], parameters=available[-1]["parameters"],
        quality_db=5., quality_conservative_db=5., metrics=dict(METRICS),
        ops_per_parameter=sum(point["ops"] / point["parameters"] for point in available) / len(available),
        execution_semantics="offline_overlap_200_100", seeds=seeds, expected_cases=len(cases),
        completed_cases=len(cases), evidence_type="synthetic_simulation", cases=cases,
        provenance={"training_sha256": protocol.training_sha256, "model_parameters": {}})
    row.update(updates)
    return row


def result(protocol, **updates):
    return ArenaRow(**values(protocol, **updates))


def ranking_scores(protocol, default, **scores):
    """One score per protocol ranking; keyword names use ``_`` where a ranking identifier has ``-``."""
    scores = {key.replace("budget_", "budget-"): value for key, value in scores.items()}
    assert set(scores) <= {ranking.ranking_id for ranking in protocol.rankings}
    return {ranking.ranking_id: {"score": scores.get(ranking.ranking_id, default)} for ranking in protocol.rankings}


def request(protocol, **updates):
    return ArenaSubmissionRequest(**{"board_id": "dpa-160mhz", "backbone": "gru",
        "display_name": "My experiment", "accepted_protocol_sha256": protocol.protocol_sha256, **updates})


def controller(tmp_path, protocol, official=None, evaluator=None, enabled=True):
    ws = Workspace.open_or_create(tmp_path / "workspace")
    return ArenaController(ws, BackboneController(ws), enabled=enabled, evaluator=evaluator,
        protocol_provider=lambda: protocol,
        backbone_provider=lambda: [ArenaBackbone(key="gru", display_name="GRU", family="recurrent"),
                                  ArenaBackbone(key="lstm", display_name="LSTM", family="recurrent")],
        official_provider=lambda: official or [], summary_provider=lambda row: {})


def store(arena, digit, protocol, row, *, model_parameters=None, accepted=None, status=None):
    """A durable record exactly as submit() and a finished worker leave it on disk."""
    accepted = accepted or protocol.protocol_sha256
    submission = ArenaSubmission(submission_id="arena-" + digit * 32, protocol_sha256=accepted,
        request=request(protocol, backbone=row.backbone, accepted_protocol_sha256=accepted),
        status=status or row.status, result=row)
    directory = arena.directory(submission.submission_id)
    directory.mkdir()
    write_json_atomic(directory / "request.json", {"board_id": row.board_id, "backbone": row.backbone,
        "protocol_sha256": accepted, "model_parameters": model_parameters or {}, "model_provenance": {}})
    write_json_atomic(directory / "submission.json", submission)
    return submission


def test_request_rejects_client_scores_parameters_and_invalid_custom_reference(protocol):
    for field in ("score", "metrics", "parameters", "model_parameters", "seeds", "run_id",
                  "budgets", "budget", "rankings", "rank", "ops"):
        with pytest.raises(ValidationError):
            request(protocol, **{field: 1})
    assert request(protocol, backbone="user_template").backbone_id is None
    with pytest.raises(ValidationError):
        request(protocol, backbone_id="ub-" + "a" * 64)
    with pytest.raises(ValidationError):
        request(protocol, display_name="\n\t")


def test_protocol_model_and_path_admission(tmp_path, protocol):
    arena = controller(tmp_path, protocol)
    with pytest.raises(Conflict, match="protocol changed"):
        arena.submit(request(protocol, accepted_protocol_sha256="b" * 64))
    with pytest.raises(InvalidInput, match="bundled Arena"):
        arena.submit(request(protocol, backbone="not-a-model"))
    with pytest.raises(InvalidInput):
        arena.get("../submission")
    assert arena.list() == []


def test_ilc_is_rejected_and_stored_ilc_records_never_appear_in_rank(tmp_path, protocol):
    arena = controller(tmp_path, protocol)
    with pytest.raises(InvalidInput, match="bundled Arena"):
        arena.submit(request(protocol, backbone="ilc_dpd"))
    legacy = result(protocol).model_copy(update={"backbone": "ilc_dpd"})
    stored = store(arena, "2", protocol, legacy)
    assert arena.get(stored.submission_id).status == "failed"
    assert arena.leaderboard("dpa-160mhz").rows == []


def test_submission_evaluates_fixed_request_and_keeps_workspace_provenance(tmp_path, protocol):
    seen = []

    def evaluate(directory):
        seen.append(read_json(directory / "request.json"))
        return result(protocol)

    arena = controller(tmp_path, protocol, evaluator=evaluate)
    submitted = arena.submit(request(protocol))
    arena.thread.join(5)
    complete = arena.get(submitted.submission_id)
    assert complete.status == "succeeded"
    assert complete.result.origin == "workspace"
    assert complete.result.entry_id == submitted.submission_id
    assert complete.result.display_name == "My experiment"
    assert complete.result.rank is None       # A rank exists only inside a leaderboard.
    assert [point.budget for point in complete.result.budgets] == protocol.budgets
    assert seen == [{"board_id": "dpa-160mhz", "backbone": "gru",
                     "protocol_sha256": protocol.protocol_sha256, "model_parameters": {}, "model_provenance": {}}]
    row = arena.leaderboard("dpa-160mhz").rows[0]
    assert row.rank == 1 and row.rankings["overall"] == ArenaRankEntry(score=5., rank=1)
    assert {ranking.ranking_id for ranking in protocol.rankings} == set(row.rankings)


def test_only_one_evaluation_can_run_and_stopping_rejects_new_work(tmp_path, protocol):
    entered, release = threading.Event(), threading.Event()

    def evaluate(_directory):
        entered.set()
        release.wait(5)
        return result(protocol)

    arena = controller(tmp_path, protocol, evaluator=evaluate)
    submission = arena.submit(request(protocol))
    assert entered.wait(5)
    try:
        with pytest.raises(Conflict, match="already active"):
            arena.submit(request(protocol))
        arena.stopping.set()
    finally:
        release.set()
        arena.stop()
    assert arena.get(submission.submission_id).status == "interrupted"
    with pytest.raises(Conflict, match="stopping"):
        arena.submit(request(protocol))


def _nothing_available(row):
    for point in row["budgets"]:
        point.update(available=False, model_parameters=None)
    row.update(cases=[], expected_cases=0, completed_cases=0, available_budgets=0)


def _dropped_seed(row):
    """Internally consistent, but the least favourable seed was never reported."""
    row["cases"] = [case for case in row["cases"] if case["seed"] != 33]
    row.update(seeds=[11, 22], expected_cases=len(row["cases"]), completed_cases=len(row["cases"]))


IDENTITY, EVIDENCE, TERMINAL = "different protocol, board or backbone", "evidence label", "terminal result"
INCOMPLETE, UNAUDITED = "Incomplete Arena evaluation", "audited score and operation count"
SWEEP, CASE_IDENTITY = "cover every protocol budget", "canonical budget, condition and seed"


@pytest.mark.parametrize("damage,reason", [
    (lambda row: row.update(protocol_sha256="b" * 64), IDENTITY),
    (lambda row: row.update(backbone="lstm"), IDENTITY),
    (lambda row: row.update(board_id="dpa-200mhz"), IDENTITY),
    (lambda row: row.update(evidence_type="measured_data_simulation"), EVIDENCE),
    (lambda row: row.update(status="running"), TERMINAL),
    (lambda row: row.update(completed_cases=23), INCOMPLETE),
    (lambda row: row.update(expected_cases=23, completed_cases=23), INCOMPLETE),
    (lambda row: row.update(seeds=[11, 22]), INCOMPLETE),
    (_dropped_seed, INCOMPLETE),
    (lambda row: row.update(metrics=None), INCOMPLETE),
    (lambda row: row.update(parameters=None), INCOMPLETE),
    # A rank needs an audited sweep score, a qualified budget and the cost of every evaluated budget.
    (lambda row: row.update(score=None), UNAUDITED),
    (lambda row: row.update(qualified_budgets=0), UNAUDITED),
    (lambda row: row["budgets"][2].update(ops=None), UNAUDITED),
    # The sweep is the protocol's: complete, ordered and with evidence exactly where a model exists.
    (lambda row: row["budgets"].pop(), SWEEP),
    (lambda row: row["budgets"].reverse(), SWEEP),
    (lambda row: row["budgets"][0].update(budget=125), SWEEP),
    (lambda row: row["budgets"][3].update(available=False), INCOMPLETE),
    (_nothing_available, INCOMPLETE),
    (lambda row: row["cases"].pop(), INCOMPLETE),
    (lambda row: row["cases"][0].pop("budget"), CASE_IDENTITY),
    (lambda row: row["cases"][0].update(budget=True), CASE_IDENTITY),
    (lambda row: row["cases"][0].update(seed="11"), CASE_IDENTITY),
    (lambda row: row["cases"][0].update(condition_id=None), CASE_IDENTITY),
    (lambda row: [case.update(budget=250) for case in row["cases"]], INCOMPLETE),
    (lambda row: [row["cases"].remove(case) for case in list(row["cases"]) if case["budget"] == 2000], INCOMPLETE),
])
def test_mismatched_or_incomplete_evaluator_output_never_ranked(tmp_path, protocol, damage, reason):
    raw = values(protocol)
    damage(raw)
    arena = controller(tmp_path, protocol, evaluator=lambda _: raw)
    submission = arena.submit(request(protocol))
    arena.thread.join(5)
    failed = arena.get(submission.submission_id)
    assert failed.status == "failed" and failed.result is None and reason in failed.error
    row = arena.leaderboard("dpa-160mhz").rows[0]
    assert row.rank is None and all(entry == ArenaRankEntry() for entry in row.rankings.values())


def test_undamaged_fixture_is_the_control_for_every_rejection(tmp_path, protocol):
    raw = values(protocol)
    controller(tmp_path, protocol)._validate_complete(ArenaRow(**raw))
    assert len(raw["cases"]) == raw["expected_cases"] == 4 * 2 * 3
    assert {(case["budget"], case["condition_id"], case["seed"]) for case in raw["cases"]} == {
        (budget, condition, seed) for budget in protocol.budgets for condition in CONDITIONS for seed in protocol.seeds}


def test_failed_evaluation_stays_visible_but_cannot_keep_a_score(tmp_path, protocol):
    claimed = result(protocol, status="failed", error="RuntimeError: diverged", score=9., rank=1)
    arena = controller(tmp_path, protocol, evaluator=lambda _: claimed)
    submission = arena.submit(request(protocol))
    arena.thread.join(5)
    stored = arena.get(submission.submission_id)
    assert stored.status == "failed" and stored.error == "RuntimeError: diverged"
    assert stored.result.score is None and not stored.result.eligible and stored.result.rankings == {}
    row = arena.leaderboard("dpa-160mhz").rows[0]
    assert row.entry_id == submission.submission_id and row.rank is None
    with pytest.raises(ValueError, match="cannot carry a score"):
        arena._validate_complete(claimed)


def test_finite_metrics_scores_and_costs_required(protocol):
    with pytest.raises(ValidationError):
        result(protocol, score=float("nan"))
    with pytest.raises(ValidationError):
        result(protocol, quality_conservative_db=float("inf"))
    with pytest.raises(ValidationError):
        result(protocol, ops_per_parameter=-1)
    with pytest.raises(ValidationError):
        result(protocol, best_budget=0)
    for entry in ({"score": float("nan")}, {"score": float("inf")}, {"score": 1., "rank": 0},
                  {"score": 1., "position": 1}):
        with pytest.raises(ValidationError):
            result(protocol, rankings={"overall": entry})
    point = sweep_points()[0]
    for field, value in (("budget", 0), ("parameters", -1), ("mul", -1), ("add", -1), ("ops", -1),
                         ("ops", 1.5), ("parameter_ratio", 0), ("operation_ratio", 0),
                         ("operation_ratio", float("inf")), ("quality_std_db", -.1),
                         ("score", float("nan")), ("parameter_efficiency_db", float("inf")),
                         ("arithmetic_efficiency_db", float("-inf")), ("nonlinear", {"tanh": 1.5})):
        with pytest.raises(ValidationError):
            ArenaBudgetResult(**{**point, field: value})
    metrics = result(protocol).metrics.model_dump()
    for field in ("aer_db", "baseline_aer_db", "aer_improvement_db"):
        with pytest.raises(ValidationError):
            ArenaMetricSummary.model_validate({**metrics, field: float("nan")})
    assert result(protocol).metrics.aer_db is None


@pytest.mark.parametrize("field,value", [("latency_ns_per_sample", 10), ("latency_ratio", 1),
    ("timing_cohort", "cpu-a"), ("score_std", .1), ("macs_per_sample", 100)])
def test_host_timing_is_no_longer_part_of_a_result(protocol, field, value):
    with pytest.raises(ValidationError, match=field):
        result(protocol, **{field: value})
    with pytest.raises(ValidationError):
        ArenaBudgetResult(**{**sweep_points()[0], field: value})
    assert not any("latency" in name or "timing" in name for model in (ArenaRow, ArenaBudgetResult, ArenaProtocol)
                   for name in model.model_fields)


def test_rank_scoped_to_origin_and_semantics_not_hardware_and_excludes_ineligible(tmp_path, protocol):
    rows = [result(protocol, score=5, entry_id="a"),
        result(protocol, score=3, entry_id="b", backbone="lstm"),
        # Operation counts do not depend on the host: another machine is no separate cohort.
        result(protocol, score=8, entry_id="c", provenance={"environment": {"processor": "other", "device": "cuda"}}),
        result(protocol, score=9, entry_id="d", eligible=False, eligibility_reasons=["regression"]),
        result(protocol, score=2, entry_id="e", execution_semantics="streaming_stateful")]
    arena = controller(tmp_path, protocol, official=rows, evaluator=lambda _: result(protocol, score=100))
    submitted = arena.submit(request(protocol))
    arena.thread.join(5)
    board = arena.leaderboard("dpa-160mhz")
    ranks = {row.entry_id: row.rank for row in board.rows}
    assert ranks == {"a": 2, "b": 3, "c": 1, "d": None, "e": 1, submitted.submission_id: 1}
    assert [row.entry_id for row in board.rows] == [submitted.submission_id, "c", "e", "a", "b", "d"]
    unranked = next(row for row in board.rows if row.entry_id == "d")
    assert all(entry == ArenaRankEntry() for entry in unranked.rankings.values())
    assert board.coverage.expected == 2 and board.coverage.missing == []
    assert rows[0].rank is None and rows[0].rankings["overall"].rank is None      # Provider rows are not mutated.


def test_each_ranking_is_ordered_independently_inside_its_origin_and_semantics_cohort(tmp_path, protocol):
    sprinter = result(protocol, entry_id="sprinter", score=5., parameters=900,
        rankings=ranking_scores(protocol, 5., linearization=9., budget_250=None))
    steady = result(protocol, entry_id="steady", backbone="lstm", score=6., parameters=400,
        rankings=ranking_scores(protocol, 6., linearization=7., evm=5.))
    steady.rankings["linearization"].rank = 1             # A claimed position is never trusted.
    stream = result(protocol, entry_id="stream", score=1., execution_semantics="streaming_stateful",
        rankings=ranking_scores(protocol, 1.))
    unqualified = result(protocol, entry_id="unqualified", score=None, eligible=False, qualified_budgets=0,
        rankings=ranking_scores(protocol, 50.))
    local = result(protocol, score=.5, rankings=ranking_scores(protocol, .5, linearization=.25))
    arena = controller(tmp_path, protocol, official=[sprinter, steady, stream, unqualified], evaluator=lambda _: local)
    submitted = arena.submit(request(protocol))
    arena.thread.join(5)
    rows = {row.entry_id: row for row in arena.leaderboard("dpa-160mhz").rows}
    position = lambda entry, ranking: rows[entry].rankings[ranking].rank
    assert (position("steady", "overall"), position("sprinter", "overall")) == (1, 2)
    assert (position("sprinter", "linearization"), position("steady", "linearization")) == (1, 2)
    assert rows["sprinter"].rankings["linearization"].score == 9. and rows["steady"].rankings["overall"].score == 6.
    # Equal EVM scores: the smaller model is listed first.
    assert rows["steady"].rankings["evm"].score == rows["sprinter"].rankings["evm"].score == 5.
    assert (position("steady", "evm"), position("sprinter", "evm")) == (1, 2)
    # No qualified 250-parameter result: absent from that board only, and nobody inherits a gap.
    assert rows["sprinter"].rankings["budget-250"] == ArenaRankEntry() and position("steady", "budget-250") == 1
    assert position("sprinter", "budget-500") == 2
    # Streaming execution and workspace evidence are cohorts of their own, in every ranking.
    for entry in ("stream", submitted.submission_id):
        assert {item.rank for item in rows[entry].rankings.values()} == {1}
    assert rows[submitted.submission_id].rankings["linearization"].score == .25
    assert all(item == ArenaRankEntry() for item in rows["unqualified"].rankings.values())
    # The row position is its Overall FoM position.
    assert {key: row.rank for key, row in rows.items()} == {"steady": 1, "sprinter": 2, "stream": 1,
        "unqualified": None, submitted.submission_id: 1}


def test_bundled_protocol_rescores_raw_sweep_evidence_and_ranks_every_board_without_a_stub(tmp_path):
    """No provider is replaced: registered sweeps, the production scorer and the published rankings."""
    from tests.unit.test_arena_scoring import adjusted, cases
    current, records = scoring.protocol(), scoring.calibration()
    profiles = {"gru": dict(qualities=(12., 12., 12.)), "lstm": dict(qualities=(6., 6., 6.))}

    def evaluate(directory):
        accepted = read_json(directory / "request.json")
        key, board_id = accepted["backbone"], accepted["board_id"]
        raw = cases(records, board_id, backbone=key, **profiles[key])
        for case in raw:       # The sprinter keeps the PA output power at its largest budget only.
            if key == "gru" and case["budget"] != 2000:
                case["judges"][0]["power_error_db"] = .7
        derived = scoring.summarize_cases(key, board_id, raw, scoring.sweep(key))
        claimed = {ranking.ranking_id: {"score": 999., "rank": 1} for ranking in current.rankings}
        claimed["house-special"] = {"score": 999., "rank": 1}              # A ranking the protocol does not publish.
        return dict(entry_id="arena-local-result", board_id=board_id, backbone=key, display_name="worker",
            origin="workspace", status="succeeded", protocol_sha256=accepted["protocol_sha256"],
            evidence_type="measured_data_simulation", execution_semantics="offline_overlap_200_100", cases=raw,
            provenance={"training_sha256": current.training_sha256, "model_parameters": {}},
            **{**derived, "score": 999., "rankings": claimed})

    ws = Workspace.open_or_create(tmp_path / "workspace")
    arena = ArenaController(ws, BackboneController(ws), evaluator=evaluate,
                            protocol_provider=lambda: current, official_provider=lambda: [])
    submitted = {}
    for key in profiles:
        submitted[key] = arena.submit(ArenaSubmissionRequest(board_id="apa-200mhz-b", backbone=key,
            display_name=key, accepted_protocol_sha256=current.protocol_sha256)).submission_id
        arena.thread.join(30)
        assert arena.get(submitted[key]).status == "succeeded", arena.get(submitted[key]).error
        # The production scorer replaces every claimed ranking, so an invented one is not even stored.
        assert set(arena.get(submitted[key]).result.rankings) == {ranking.ranking_id for ranking in current.rankings}
    rows = {row.backbone: row for row in arena.leaderboard("apa-200mhz-b").rows}
    sprinter, steady = rows["gru"], rows["lstm"]
    assert sprinter.entry_id == submitted["gru"] and sprinter.origin == steady.origin == "workspace"
    # One strong budget out of four against a moderate result everywhere.
    assert sprinter.qualified_budgets == 1 and steady.qualified_budgets == 4 and sprinter.best_budget == 2000
    assert sprinter.score == pytest.approx(adjusted("gru", 2000, 12.))
    assert steady.score == pytest.approx(max(adjusted("lstm", budget, 6.) for budget in current.budgets))
    assert sprinter.rankings["linearization"] == ArenaRankEntry(score=12., rank=1)
    assert steady.rankings["linearization"] == ArenaRankEntry(score=6., rank=2)
    assert (steady.rank, sprinter.rank) == (steady.rankings["overall"].rank, sprinter.rankings["overall"].rank) == (1, 2)
    assert sprinter.rankings["budget-2000"].rank == 2 and steady.rankings["budget-2000"].rank == 1
    for budget in (250, 500, 1000):
        assert sprinter.rankings[f"budget-{budget}"] == ArenaRankEntry()
        assert steady.rankings[f"budget-{budget}"].rank == 1
    assert set(sprinter.rankings) == {ranking.ranking_id for ranking in current.rankings}
    assert all(point.ops == point.mul + point.add and point.parameters <= point.budget
               for row in (sprinter, steady) for point in row.budgets)


def test_ranking_without_a_score_receives_no_position(tmp_path, protocol):
    partial = result(protocol, entry_id="partial", rankings={"overall": {"score": 4.}})
    arena = controller(tmp_path, protocol, official=[partial, result(protocol, entry_id="full", backbone="lstm", score=3.)])
    rows = {row.entry_id: row for row in arena.leaderboard("dpa-160mhz").rows}
    assert rows["partial"].rank == 1 and rows["full"].rank == 2
    assert rows["partial"].rankings["linearization"] == ArenaRankEntry()
    assert rows["full"].rankings["linearization"] == ArenaRankEntry(score=3., rank=1)
    assert set(rows["partial"].rankings) == {ranking.ranking_id for ranking in protocol.rankings}


def test_ranking_the_protocol_does_not_publish_never_reaches_a_leaderboard(tmp_path, protocol):
    published = {ranking.ranking_id for ranking in protocol.rankings}
    invented = {"house-special": {"score": 99., "rank": 1}, "Overall": {"score": 99., "rank": 1}}
    shipped = result(protocol, entry_id="shipped", rankings={**ranking_scores(protocol, 5.), **invented})
    impostor = result(protocol, entry_id="impostor", backbone="lstm", score=99., rank=1, rankings=invented)
    arena = controller(tmp_path, protocol, official=[shipped, impostor])
    stored = store(arena, "6", protocol, result(protocol, score=3., rankings={**ranking_scores(protocol, 3.), **invented}))
    # This fixture's scorer keeps the stored rankings, so the claim does arrive at the board builder.
    assert set(arena.get(stored.submission_id).result.rankings) == published | set(invented)
    board = arena.leaderboard("dpa-160mhz")
    rows = {row.entry_id: row for row in board.rows}
    assert all(set(row.rankings) == published for row in rows.values())
    assert "house-special" not in board.model_dump_json()
    assert rows["shipped"].rankings["overall"] == ArenaRankEntry(score=5., rank=1)
    assert rows[stored.submission_id].rankings["overall"] == ArenaRankEntry(score=3., rank=1)
    # A row ranked nowhere but in its own ranking has no position at all.
    assert rows["impostor"].rank is None and all(entry == ArenaRankEntry() for entry in rows["impostor"].rankings.values())
    assert [row.entry_id for row in board.rows][-1] == "impostor"
    assert set(shipped.rankings) == published | set(invented)             # Provider rows are not mutated.


def test_old_protocol_or_untrusted_official_rows_rejected(tmp_path, protocol):
    arena = controller(tmp_path, protocol, official=[result(protocol, origin="workspace")])
    with pytest.raises(WorkspaceError, match="Official Arena results"):
        arena.leaderboard("dpa-160mhz")
    arena.official_provider = lambda: [result(protocol, protocol_sha256="b" * 64)]
    with pytest.raises(WorkspaceError, match="Official Arena results"):
        arena.leaderboard("dpa-160mhz")


def test_duplicate_conditions_do_not_satisfy_complete_coverage(tmp_path, protocol):
    duplicate = result(protocol)
    duplicate.cases[-1] = duplicate.cases[0]
    arena = controller(tmp_path, protocol, evaluator=lambda _: duplicate)
    submission = arena.submit(request(protocol))
    arena.thread.join(5)
    assert arena.get(submission.submission_id).status == "failed"


def test_durable_local_results_recompute_scores_and_cannot_impersonate_official(tmp_path, protocol, monkeypatch):
    arena = controller(tmp_path, protocol)
    arena.summary_provider = None
    seen = []

    def recompute(backbone, board_id, cases, points):
        seen.append((backbone, board_id, cases, points))
        return {"eligible": False, "score": None, "quality_db": -3, "qualified_budgets": 0,
                "rankings": {ranking.ranking_id: {"score": None} for ranking in protocol.rankings},
                "eligibility_reasons": ["Actual observations regress against the baseline"]}

    monkeypatch.setattr(scoring, "summarize_cases", recompute)
    forged = result(protocol, origin="official", score=999, rank=1, quality_db=999,
                    rankings={ranking.ranking_id: {"score": 999, "rank": 1} for ranking in protocol.rankings})
    submission = store(arena, "3", protocol, forged)
    loaded = arena.get(submission.submission_id)
    assert seen == [("gru", "dpa-160mhz", forged.cases, [point.model_dump(mode="json") for point in forged.budgets])]
    assert [(point["budget"], point["model_parameters"]) for point in seen[0][3]] == [
        (point["budget"], point["model_parameters"]) for point in scoring.sweep("gru")]
    assert loaded.result.eligible is False and loaded.result.score is None
    assert loaded.result.quality_db == -3 and loaded.result.rank is None
    assert all(entry == ArenaRankEntry() for entry in loaded.result.rankings.values())
    assert loaded.result.origin == "workspace" and loaded.result.entry_id == submission.submission_id
    assert arena.leaderboard("dpa-160mhz").rows[0].rank is None


def _cheaper_model(row):
    row["budgets"][1]["model_parameters"] = {**row["budgets"][1]["model_parameters"], "hidden_size": 8}


def _skipped_budget(row):
    row["budgets"][3] = dict(budget=2000, available=False, model_parameters=None)
    row["cases"] = [case for case in row["cases"] if case["budget"] != 2000]
    row.update(expected_cases=len(row["cases"]), completed_cases=len(row["cases"]),
               available_budgets=3, qualified_budgets=3, best_budget=1000, parameters=row["budgets"][2]["parameters"])


def _exchanged_presets(row):
    first, second = row["budgets"][0], row["budgets"][1]
    first["model_parameters"], second["model_parameters"] = second["model_parameters"], first["model_parameters"]


@pytest.mark.parametrize("change", [_cheaper_model, _skipped_budget, _exchanged_presets])
def test_stored_result_cannot_choose_a_sweep_other_than_the_one_registered_for_its_request(
        tmp_path, protocol, monkeypatch, change):
    arena = controller(tmp_path, protocol)
    arena.summary_provider = None
    scored = []
    monkeypatch.setattr(scoring, "summarize_cases", lambda *args: scored.append(args) or {})
    honest = store(arena, "6", protocol, result(protocol))
    assert arena.get(honest.submission_id).status == "succeeded" and len(scored) == 1
    raw = values(protocol)
    change(raw)
    arena._validate_complete(ArenaRow(**raw))          # Complete and well formed: only the sweep differs.
    forged = store(arena, "7", protocol, ArenaRow(**raw))
    loaded = arena.get(forged.submission_id)
    assert loaded.status == "failed" and loaded.result is None
    assert "sweep registered for this request" in loaded.error
    assert len(scored) == 1                                             # Rejected before any score is derived.
    ranks = {row.entry_id: row.rank for row in arena.leaderboard("dpa-160mhz").rows}
    assert ranks == {honest.submission_id: 1, forged.submission_id: None}


def test_evaluator_cannot_replace_the_registered_sweep_either(tmp_path, protocol, monkeypatch):
    raw = values(protocol)
    _cheaper_model(raw)
    arena = controller(tmp_path, protocol, evaluator=lambda _: raw)
    arena.summary_provider = None
    monkeypatch.setattr(scoring, "summarize_cases", lambda *args: pytest.fail("an unregistered sweep was scored"))
    submission = arena.submit(request(protocol))
    arena.thread.join(5)
    failed = arena.get(submission.submission_id)
    assert failed.status == "failed" and "sweep registered for this request" in failed.error


def test_template_sweep_is_derived_from_the_accepted_definition_not_from_the_result(tmp_path, protocol, monkeypatch):
    definition = json.loads(DEFAULT_DEFINITION)
    definition["nodes"][0]["features"] = 12          # 602 parameters: evaluated as submitted at the 1,000 budget
    supplied = {"definition": json.dumps(definition)}
    registered, bundled = sweep_points("user_template", supplied=supplied), sweep_points("user_template")
    widths = lambda points: [json.loads(point["model_parameters"]["definition"])["nodes"][0]["features"] for point in points]
    assert widths(registered) == [7, 10, 12, 23] and widths(bundled) == [7, 10, 16, 23]
    arena = controller(tmp_path, protocol)
    arena.summary_provider = None
    monkeypatch.setattr(scoring, "summarize_cases", lambda *args: {})
    accepted = store(arena, "8", protocol, result(protocol, backbone="user_template", budgets=registered),
                     model_parameters=supplied)
    assert arena.get(accepted.submission_id).status == "succeeded"
    swapped = store(arena, "9", protocol, result(protocol, backbone="user_template", budgets=bundled),
                    model_parameters=supplied)
    loaded = arena.get(swapped.submission_id)
    assert loaded.status == "failed" and "sweep registered for this request" in loaded.error
    # The same result is what a request without an uploaded definition registers.
    default = store(arena, "a", protocol, result(protocol, backbone="user_template", budgets=bundled))
    assert arena.get(default.submission_id).status == "succeeded"


def _removed(path):
    path.unlink()


def _symlinked(path):
    """The link leads to the intact request: the indirection itself is what is refused."""
    target = path.with_name("accepted-elsewhere.json")
    path.rename(target)
    path.symlink_to(target)


def _a_directory(path):
    path.unlink()
    path.mkdir()


def _unreadable(path):
    path.chmod(0)
    if os.access(path, os.R_OK):
        pytest.skip("this user reads a file whatever its mode")


DAMAGED_REQUEST = "The accepted Arena request is missing or damaged."


@pytest.mark.parametrize("damage,reason", [
    (_removed, DAMAGED_REQUEST), (_symlinked, DAMAGED_REQUEST), (_a_directory, DAMAGED_REQUEST),
    (lambda path: path.write_text("[]"), DAMAGED_REQUEST),
    (lambda path: path.write_text('[{"model_parameters": {}}]'), DAMAGED_REQUEST),
    (lambda path: path.write_text('"gru"'), DAMAGED_REQUEST),
    (lambda path: path.write_text("null"), DAMAGED_REQUEST),
    (lambda path: path.write_text('{"model_parameters": {'), "Expecting property name"),
    (lambda path: path.write_text('{"model_parameters": {"hidden_size": 8}}'), "server-verified template"),
    (_unreadable, "its stored evidence could not be read"),
], ids=["removed", "symlinked", "a-directory", "a-list", "a-list-of-requests", "a-string", "null", "truncated",
        "a-size-of-its-own", "unreadable"])
def test_record_with_a_damaged_accepted_request_fails_alone_and_the_workspace_stays_usable(
        tmp_path, protocol, monkeypatch, damage, reason):
    arena = controller(tmp_path, protocol)
    arena.summary_provider = None                         # Production path: the sweep is derived from request.json.
    scored = []
    monkeypatch.setattr(scoring, "summarize_cases", lambda *args: scored.append(args[0]) or {})
    healthy = store(arena, "6", protocol, result(protocol, score=4.))
    damaged = store(arena, "7", protocol, result(protocol, backbone="lstm", score=9.))
    assert arena.get(damaged.submission_id).status == "succeeded" and scored == ["lstm"]     # The control: intact, it counts.
    damage(arena.directory(damaged.submission_id) / "request.json")
    loaded = arena.get(damaged.submission_id)
    assert loaded.status == "failed" and loaded.result is None and loaded.request.backbone == "lstm"
    assert loaded.error.startswith("Stored Arena result failed validation: ") and reason in loaded.error
    assert str(tmp_path) not in loaded.error and "Errno" not in loaded.error      # No server path is ever served.
    assert scored == ["lstm"]                             # Evidence of an unverifiable request is never scored.
    # One damaged record is not a damaged workspace: the list, the other record and the board still answer.
    assert {record.submission_id: record.status for record in arena.list()} == {
        healthy.submission_id: "succeeded", damaged.submission_id: "failed"}
    rows = {row.entry_id: row for row in arena.leaderboard("dpa-160mhz").rows}
    assert rows[healthy.submission_id].rankings["overall"] == ArenaRankEntry(score=4., rank=1)
    unranked = rows[damaged.submission_id]
    assert unranked.status == "failed" and unranked.rank is None and unranked.score is None
    assert reason in unranked.error and all(entry == ArenaRankEntry() for entry in unranked.rankings.values())
    assert str(tmp_path) not in unranked.error
    # Studio starts again over this workspace: the constructor reads every record.
    restarted = ArenaController(arena.ws, arena.backbones, protocol_provider=lambda: protocol,
                                backbone_provider=arena.backbone_provider, official_provider=lambda: [])
    assert restarted.summary_provider is None
    assert restarted.get(damaged.submission_id).status == "failed"
    assert restarted.get(healthy.submission_id).status == "succeeded"
    assert restarted.leaderboard("dpa-160mhz").rows[0].entry_id == healthy.submission_id


EARLIER_VERSION = "The stored Arena evaluation is damaged or was written by an earlier Arena version."
V2_ONLY = ("rankings", "budgets", "qualified_budgets", "available_budgets", "best_budget", "ops_per_parameter")


def as_arena_v1_wrote_it(path):
    """One model size, a measured host latency and a timing cohort: a record of the earlier Arena version."""
    stored = read_json(path)
    row = stored["result"]
    for field in V2_ONLY:
        row.pop(field)
    row.update(score=-31.2, score_std=.4, macs_per_sample=2082., latency_ns_per_sample=4000., latency_ratio=1.3,
               timing_cohort="cpu-a", rank=1)
    row["cases"] = [{"condition_id": condition, "seed": seed, "latency_ns_per_sample": 4000.}
                    for condition in CONDITIONS for seed in row["seeds"]]
    row.update(expected_cases=len(row["cases"]), completed_cases=len(row["cases"]))
    path.write_text(json.dumps(stored))
    return stored


@pytest.mark.parametrize("accepted", ["b" * 64, None], ids=["under-its-own-protocol", "claiming-the-current-protocol"])
def test_record_of_the_earlier_arena_version_stays_visible_as_failed_and_is_never_ranked(tmp_path, protocol, accepted):
    sha = accepted or protocol.protocol_sha256
    arena = controller(tmp_path, protocol)
    current = store(arena, "6", protocol, result(protocol, score=4.))
    earlier = store(arena, "5", protocol, result(protocol, backbone="lstm", protocol_sha256=sha, score=50.), accepted=sha)
    path = arena.directory(earlier.submission_id) / "submission.json"
    written = as_arena_v1_wrote_it(path)
    with pytest.raises(ValidationError, match="timing_cohort"):          # The control: today's contract refuses it.
        ArenaSubmission.model_validate(written)
    loaded = arena.get(earlier.submission_id)
    assert (loaded.status, loaded.result, loaded.progress, loaded.error) == ("failed", None, None, EARLIER_VERSION)
    assert (loaded.request.backbone, loaded.request.display_name, loaded.protocol_sha256) == ("lstm", "My experiment", sha)
    assert loaded.created_at == earlier.created_at
    assert {record.submission_id: record.status for record in arena.list()} == {
        current.submission_id: "succeeded", earlier.submission_id: "failed"}
    rows = {row.entry_id: row for row in arena.leaderboard("dpa-160mhz").rows}
    assert rows[current.submission_id].rankings["overall"] == ArenaRankEntry(score=4., rank=1)
    if accepted:                                          # A board lists the evaluations of its own protocol.
        assert set(rows) == {current.submission_id}
    else:
        unranked = rows[earlier.submission_id]
        assert (unranked.status, unranked.rank, unranked.score, unranked.error) == ("failed", None, None, EARLIER_VERSION)
        assert all(entry == ArenaRankEntry() for entry in unranked.rankings.values())
    # Studio starts on this workspace, under the production scorer too, and accepts new work.
    ArenaController(arena.ws, arena.backbones, protocol_provider=lambda: protocol, official_provider=lambda: [],
                    backbone_provider=arena.backbone_provider)
    restarted = controller(tmp_path, protocol, evaluator=lambda _: result(protocol, score=1.))
    assert restarted.get(earlier.submission_id).error == EARLIER_VERSION
    submitted = restarted.submit(request(protocol))
    restarted.thread.join(5)
    assert restarted.get(submitted.submission_id).status == "succeeded" and len(restarted.list()) == 3
    # Reading repaired nothing: what the earlier version measured is still on disk, byte for byte.
    assert read_json(path) == written


def _half_a_result(stored):
    stored["result"].pop("entry_id")


@pytest.mark.parametrize("change", [
    _half_a_result, lambda stored: stored["result"].update(budgets="all of them"),
    lambda stored: stored.update(status="done"), lambda stored: stored.update(error=["not", "text"]),
    lambda stored: stored.update(status="running", progress={"phase": "training", "epoch": 9, "epochs": 1}),
], ids=["half-a-result", "a-malformed-sweep", "an-unknown-status", "a-malformed-error", "an-impossible-progress"])
def test_record_with_damaged_evidence_but_an_intact_identity_is_salvaged_as_failed(tmp_path, protocol, change):
    arena = controller(tmp_path, protocol)
    damaged = store(arena, "7", protocol, result(protocol, score=9.))
    path = arena.directory(damaged.submission_id) / "submission.json"
    stored = read_json(path)
    change(stored)
    path.write_text(json.dumps(stored))
    loaded = arena.get(damaged.submission_id)
    assert (loaded.status, loaded.result, loaded.progress, loaded.error) == ("failed", None, None, EARLIER_VERSION)
    assert loaded.request == damaged.request and loaded.created_at == damaged.created_at
    row = arena.leaderboard("dpa-160mhz").rows[0]
    assert (row.entry_id, row.status, row.rank, row.score) == (damaged.submission_id, "failed", None, None)
    # A restart neither trips over it nor rewrites it as an interrupted evaluation.
    assert controller(tmp_path, protocol).get(damaged.submission_id).status == "failed" and read_json(path) == stored


def _truncated(path):
    path.write_text(path.read_text()[:40])


def _without_its_request(path):
    stored = read_json(path)
    del stored["request"]
    path.write_text(json.dumps(stored))


def _of_another_directory(path):
    stored = read_json(path)
    stored["submission_id"] = "arena-" + "e" * 32
    path.write_text(json.dumps(stored))


def _moved_to_a_folder_of_another_name(path):
    path.parent.rename(path.parent.with_name("arena-copy-of-my-best-run"))


def _a_symlinked_folder(path):
    elsewhere = path.parent.with_name("kept-elsewhere")
    path.parent.rename(elsewhere)
    path.parent.symlink_to(elsewhere, target_is_directory=True)


@pytest.mark.parametrize("damage,refusal", [
    (_truncated, "A stored Arena submission is damaged."),
    (lambda path: path.write_text("[]"), "A stored Arena submission is damaged."),
    (lambda path: path.write_text('"arena"'), "A stored Arena submission is damaged."),
    (lambda path: path.write_text("null"), "A stored Arena submission is damaged."),
    (lambda path: path.write_bytes(b"\xff\xfe{}"), "A stored Arena submission is damaged."),
    (_without_its_request, "A stored Arena submission is damaged."),
    (_unreadable, "A stored Arena submission is damaged."),
    (_of_another_directory, "identity does not match its storage directory"),
    (_a_symlinked_folder, "cannot be a symbolic link"),
    (_moved_to_a_folder_of_another_name, None),
], ids=["truncated", "a-list", "a-string", "null", "not-text", "without-its-request", "unreadable",
        "of-another-directory", "a-symlinked-folder", "a-folder-of-another-name"])
def test_submission_record_that_cannot_be_read_at_all_is_skipped_and_never_closes_the_workspace(
        tmp_path, protocol, caplog, damage, refusal):
    arena = controller(tmp_path, protocol)
    healthy = store(arena, "6", protocol, result(protocol, score=4.))
    broken = store(arena, "7", protocol, result(protocol, backbone="lstm", score=9.))
    damage(arena.directory(broken.submission_id) / "submission.json")
    if refusal:                                           # Asked for by name, it is refused as a workspace problem.
        with pytest.raises(WorkspaceError, match=refusal) as raised:
            arena.get(broken.submission_id)
        assert str(tmp_path) not in str(raised.value)
    with caplog.at_level("WARNING", logger="opendpd.services.arena"):
        listed = arena.list()
    assert [record.submission_id for record in listed] == [healthy.submission_id]
    skipped = [entry.getMessage() for entry in caplog.records if "Skipping Arena submission" in entry.getMessage()]
    assert len(skipped) == 1 and str(tmp_path) not in skipped[0]
    board = arena.leaderboard("dpa-160mhz")
    assert [(row.entry_id, row.rank) for row in board.rows] == [(healthy.submission_id, 1)]
    # Studio starts on this workspace and evaluates new work beside the unreadable record.
    restarted = controller(tmp_path, protocol, evaluator=lambda _: result(protocol, score=1.))
    submitted = restarted.submit(request(protocol))
    restarted.thread.join(5)
    assert {record.submission_id: record.status for record in restarted.list()} == {
        healthy.submission_id: "succeeded", submitted.submission_id: "succeeded"}


@pytest.mark.parametrize("mode", [0o000, 0o444], ids=["no-permission-at-all", "listable-but-not-enterable"])
def test_record_folder_the_server_may_not_enter_is_a_damaged_submission_and_not_a_server_error(tmp_path, protocol, mode):
    arena = controller(tmp_path, protocol)
    healthy = store(arena, "6", protocol, result(protocol, score=4.))
    sealed = store(arena, "7", protocol, result(protocol, backbone="lstm", score=9.))
    folder = arena.directory(sealed.submission_id)
    folder.chmod(mode)
    try:
        if os.access(folder / "submission.json", os.R_OK):
            pytest.skip("this user enters a folder whatever its mode")
        with pytest.raises(WorkspaceError, match="A stored Arena submission is damaged") as raised:
            arena.get(sealed.submission_id)
        assert type(raised.value) is WorkspaceError and str(tmp_path) not in str(raised.value)
        assert [record.submission_id for record in arena.list()] == [healthy.submission_id]
        assert [(row.entry_id, row.rank) for row in arena.leaderboard("dpa-160mhz").rows] == [
            (healthy.submission_id, 1)]
        assert controller(tmp_path, protocol).get(healthy.submission_id).status == "succeeded"
    finally:
        folder.chmod(0o700)
    assert arena.get(sealed.submission_id).status == "succeeded"          # Nothing was lost meanwhile.


def test_damaged_request_of_a_record_without_a_result_or_of_an_earlier_protocol_is_not_an_error(tmp_path, protocol):
    """The accepted request decides the sweep of a current, successful result; nothing else reads it."""
    arena = controller(tmp_path, protocol)
    arena.summary_provider = None
    old = store(arena, "5", protocol, result(protocol, protocol_sha256="b" * 64), accepted="b" * 64)
    failed = store(arena, "8", protocol, result(protocol, status="failed", eligible=False, score=None,
                                                rankings={}, error="RuntimeError: diverged"))
    for record in (old, failed):
        (arena.directory(record.submission_id) / "request.json").unlink()
    assert arena.get(old.submission_id).status == "succeeded" and arena.get(old.submission_id).result.rankings == {}
    loaded = arena.get(failed.submission_id)
    assert loaded.status == "failed" and loaded.result is not None and loaded.result.error == "RuntimeError: diverged"
    assert [row.entry_id for row in arena.leaderboard("dpa-160mhz").rows] == [failed.submission_id]


def test_durable_invalid_case_evidence_fails_closed_without_hiding_submission(tmp_path, protocol, monkeypatch):
    arena = controller(tmp_path, protocol)
    arena.summary_provider = None

    def invalid(*args):
        raise ValueError("Missing required judge")

    monkeypatch.setattr(scoring, "summarize_cases", invalid)
    submission = store(arena, "4", protocol, result(protocol))
    loaded = arena.get(submission.submission_id)
    assert loaded.status == "failed" and loaded.result is None
    assert "Missing required judge" in loaded.error
    row = arena.leaderboard("dpa-160mhz").rows[0]
    assert row.rank is None and row.status == "failed"


def test_old_protocol_local_record_is_not_rescored_under_new_rules(tmp_path, protocol, monkeypatch):
    def rescored(*args):
        pytest.fail("an earlier protocol's evidence was scored under the current rules")

    arena = controller(tmp_path, protocol)
    arena.summary_provider = rescored
    monkeypatch.setattr(scoring, "summarize_cases", rescored)
    claimed = result(protocol, protocol_sha256="b" * 64, rank=1,
                     rankings={ranking.ranking_id: {"score": 7., "rank": 1} for ranking in protocol.rankings})
    submission = store(arena, "5", protocol, claimed, accepted="b" * 64)
    loaded = arena.get(submission.submission_id)
    assert loaded.status == "succeeded" and loaded.result.protocol_sha256 == "b" * 64
    assert loaded.result.eligible is False and loaded.result.score is None
    assert loaded.result.rank is None and loaded.result.rankings == {}
    assert loaded.result.eligibility_reasons == ["This evaluation belongs to an earlier Arena protocol."]
    assert loaded.result.cases == claimed.cases          # The evidence itself stays readable.
    assert arena.leaderboard("dpa-160mhz").rows == []
    arena.summary_provider = None                        # The production path does not rescore it either.
    assert arena.get(submission.submission_id).result.rankings == {}


def test_current_record_beside_an_old_one_is_the_only_ranked_entry(tmp_path, protocol):
    arena = controller(tmp_path, protocol)
    old = store(arena, "5", protocol, result(protocol, protocol_sha256="b" * 64, score=50.), accepted="b" * 64)
    current = store(arena, "6", protocol, result(protocol, score=2.))
    rows = arena.leaderboard("dpa-160mhz").rows
    assert [(row.entry_id, row.rank) for row in rows] == [(current.submission_id, 1)]
    assert {record.submission_id for record in arena.list()} == {old.submission_id, current.submission_id}


def test_deterministic_fits_require_exactly_one_canonical_seed(tmp_path, protocol):
    arena = controller(tmp_path, protocol)
    arena.backbone_provider = lambda: [ArenaBackbone(key="mp_ls", display_name="MP", family="polynomial", deterministic=True)]
    deterministic = result(protocol, backbone="mp_ls", seeds=protocol.seeds[:1])
    assert deterministic.expected_cases == len(deterministic.cases) == 4 * 2
    arena._validate_complete(deterministic)
    with pytest.raises(ValueError, match="Incomplete"):
        arena._validate_complete(result(protocol, backbone="mp_ls"))
    with pytest.raises(ValueError, match="Incomplete"):
        arena._validate_complete(result(protocol, backbone="mp_ls", seeds=protocol.seeds[1:2]))


def test_backbone_without_a_configuration_at_some_budgets_is_complete_with_fewer_cases(tmp_path, protocol):
    arena = controller(tmp_path, protocol)
    partial = result(protocol, backbone="mcldnn")
    assert [point.available for point in partial.budgets] == [False, False, True, True]
    assert partial.expected_cases == 2 * 2 * 3 and {case["budget"] for case in partial.cases} == {1000, 2000}
    arena._validate_complete(partial)
    padded = copy.deepcopy(values(protocol, backbone="mcldnn"))
    padded["cases"] += [dict(case, budget=250) for case in padded["cases"] if case["budget"] == 1000]
    padded.update(expected_cases=len(padded["cases"]), completed_cases=len(padded["cases"]))
    with pytest.raises(ValueError, match="Incomplete"):
        arena._validate_complete(ArenaRow(**padded))


def test_progress_is_bounded_advisory_and_cannot_override_status(tmp_path, protocol):
    arena = controller(tmp_path, protocol)
    submission = ArenaSubmission(submission_id="arena-" + "2" * 32, request=request(protocol),
        protocol_sha256=protocol.protocol_sha256, status="running")
    directory = arena.directory(submission.submission_id)
    directory.mkdir()
    write_json_atomic(directory / "submission.json", submission)
    progress_path = directory / "result.progress.json"
    write_json_atomic(progress_path, {"phase": "training", "epoch": 10, "epochs": 150,
        "completed_cases": 1, "expected_cases": 24, "message": "nominal · ≤250 parameters · seed 22"})
    observed = arena.get(submission.submission_id)
    assert observed.progress.epoch == 10 and observed.progress.expected_cases == 24 and observed.status == "running"
    for contents in ('{"phase":"unfinished"', '{"phase":"x","epoch":9,"epochs":1}',
                     '{"phase":"x","completed_cases":25,"expected_cases":24}',
                     '{"phase":"x","status":"succeeded"}', '{"phase":"' + 'x' * 5000 + '"}'):
        progress_path.write_text(contents)
        observed = arena.get(submission.submission_id)
        assert observed.progress is None and observed.status == "running"


def test_private_template_resolved_and_snapshotted_without_python_execution(tmp_path, protocol):
    captured = []

    def evaluate(directory):
        captured.append(read_json(directory / "request.json"))
        return result(protocol, backbone="user_template")

    arena = controller(tmp_path, protocol, evaluator=evaluate)
    uploaded = arena.backbones.upload("backbone.py", TEMPLATE.encode())
    submission = arena.submit(request(protocol, backbone="user_template", backbone_id=uploaded.backbone_id))
    arena.thread.join(5)
    assert arena.get(submission.submission_id).status == "succeeded"
    assert set(captured[0]["model_parameters"]) == {"definition"}
    assert captured[0]["model_provenance"]["source_sha256"] == uploaded.source_sha256
    source = arena.backbones.directory(uploaded.publication_id) / "package" / "backbone.py"
    source.write_text("raise RuntimeError('must not run')")
    with pytest.raises(WorkspaceError):
        arena.submit(request(protocol, backbone="user_template", backbone_id=uploaded.backbone_id))


def test_bundled_template_without_id_uses_only_server_owned_defaults(tmp_path, protocol):
    captured = []

    def evaluate(directory):
        captured.append(read_json(directory / "request.json"))
        return result(protocol, backbone="user_template")

    arena = controller(tmp_path, protocol, evaluator=evaluate)
    arena.backbone_provider = lambda: [ArenaBackbone(key="user_template", display_name="Bundled example", family="recurrent")]
    submission = arena.submit(request(protocol, backbone="user_template"))
    arena.thread.join(5)
    assert arena.get(submission.submission_id).status == "succeeded"
    assert captured[0]["model_parameters"] == {} and captured[0]["model_provenance"] == {}


def test_oversize_private_template_is_rejected_before_compute(tmp_path, protocol):
    protocol.training["max_parameters"] = 4096
    arena = controller(tmp_path, protocol)
    uploaded = arena.backbones.upload("backbone.py", TEMPLATE.replace('"features": 24', '"features": 64').encode())
    with pytest.raises(InvalidInput, match="at most 4,096"):
        arena.submit(request(protocol, backbone="user_template", backbone_id=uploaded.backbone_id))
    assert arena.list() == [] and arena.thread is None


def test_restart_marks_unfinished_records_interrupted_and_hosted_disabled(tmp_path, protocol):
    arena = controller(tmp_path, protocol)
    submission = ArenaSubmission(submission_id="arena-" + "1" * 32, request=request(protocol),
        protocol_sha256=protocol.protocol_sha256, status="running")
    directory = arena.directory(submission.submission_id)
    directory.mkdir()
    write_json_atomic(directory / "submission.json", submission)
    resumed = controller(tmp_path, protocol, enabled=False)
    assert resumed.get(submission.submission_id).status == "interrupted"
    assert resumed.catalog().submissions_available is False
    with pytest.raises(WorkspaceError, match="disabled"):
        resumed.submit(request(protocol))


def test_catalog_publishes_the_sweep_rankings_and_cost_model(tmp_path, protocol):
    catalog = controller(tmp_path, protocol).catalog()
    assert catalog.protocol.budgets == [250, 500, 1000, 2000] and catalog.submissions_available
    assert [ranking.ranking_id for ranking in catalog.protocol.rankings][:2] == ["overall", "linearization"]
    assert {f"budget-{budget}" for budget in catalog.protocol.budgets} <= {
        ranking.ranking_id for ranking in catalog.protocol.rankings}
    assert catalog.protocol.training_sha256 == "c" * 64 and catalog.protocol.cost_model


def test_public_arena_surface_is_read_only():
    from opendpd.web.policy import allowed
    assert allowed("GET", "/arena")
    assert not allowed("GET", "/arena/boards/dpa-160mhz")
    assert allowed("GET", "/arena/boards/apa-200mhz-b")
    assert not allowed("GET", "/arena/boards/synthetic-suite")
    assert not allowed("GET", "/arena/boards/dpa-200mhz")
    # The explicit route is guarded by opt-in configuration and a matching
    # isolated-worker capability; integration tests exercise those gates.
    assert allowed("POST", "/arena/submissions")
    assert not allowed("GET", "/arena/boards/../runs")


@pytest.mark.parametrize("key", ["deltagru", "deltajanet", "gru", "tres_deltagru", "user_template", "mp_ls"])
def test_local_and_official_dispatch_leave_the_device_to_the_runner(tmp_path, protocol, key):
    # Arena v2 times no host. The runner itself sends the Python Delta cells to the
    # CPU (tests/unit/test_arena_pipeline.py), so no launcher passes a device.
    from benchmark.run_arena_baselines import worker_command
    arena = controller(tmp_path, protocol)
    directory = tmp_path / "a result folder"
    directory.mkdir()
    write_json_atomic(directory / "request.json", {"backbone": key})
    local, official = arena.worker_command(directory), worker_command(tmp_path, directory, key)
    for command in (local, official):
        assert command[1:3] == ["-m", "opendpd.core.arena_runner"]
        assert command[command.index("--request") + 1] == str(directory / "request.json")
        assert command[command.index("--output") + 1] == str(directory / "result.json")
        assert "--device" not in command and "--prefetch" not in command
    assert "--cache" not in local
    assert official[official.index("--cache") + 1] == str(tmp_path / "cache")
