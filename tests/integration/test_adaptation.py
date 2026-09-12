"""S17: multi-condition adaptation under ``conditions-v1`` on a synthetic three-condition card.

The card is synthetic, so every report over it is below the evidence bar and says so: this file
tests the machinery (cells, budgets, warm starts, refusals, hash binding), never a scientific claim.
"""

import json

import pandas as pd
import pytest

from opendpd.commands import main
from opendpd.schemas import Condition, ConditionSet, DatasetOrigin, RunStatus, SignalSpec, TargetRule
from opendpd.schemas.conditions import MIN_CONDITIONS_FOR_EVIDENCE
from opendpd.services import adaptation as ad
from opendpd.services import datasets as ds
from opendpd.services.experiments import lineage, load_result
from opendpd.services.workspace import Workspace, WorkspaceError
from tests.fixtures.synthetic import Impairments, synthesize

SIGNAL = dict(sample_rate_hz=800e6, bandwidth_hz=200e6, n_sub_ch=10, nperseg=2560, amplitude_units="normalized")
# condition id -> (capture seed = its own acquisition, PA gain = the varied "drive")
CONDITIONS = {"drive-0": (11, 1.0), "drive-1": (12, 0.9), "drive-2": (13, 0.8)}
N = 12000


def _capture(path, seed, gain, n=N):
    x, y = synthesize(n, seed, impairments=Impairments(gain=gain))
    pd.DataFrame({"I_in": x[:, 0], "Q_in": x[:, 1], "I_out": y[:, 0], "Q_out": y[:, 1]}).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    src = tmp_path_factory.mktemp("src")
    ws = Workspace.create(tmp_path_factory.mktemp("ws"))
    for cid, (seed, gain) in CONDITIONS.items():
        ds.import_dataset(ws, _capture(src / f"{cid}.csv", seed, gain), dataset_id=cid, display_name=f"synthetic {cid}",
                          signal=SignalSpec(**SIGNAL), origin=DatasetOrigin.synthetic, guard_samples=64)
    return ws


def _card(prefix="", origin_note="synthetic"):
    conditions = [Condition(condition_id=cid, dataset_id=f"{prefix}{cid}", role="source" if i == 0 else "target",
                            capture_batch=f"seed-{seed}", values={"gain": gain})
                  for i, (cid, (seed, gain)) in enumerate(CONDITIONS.items())]
    return ad.seal_card(ConditionSet(set_id=f"{origin_note}-drive-v0", device=f"{origin_note} memory-polynomial PA (tests/fixtures)",
                                     dimension="drive", conditions=conditions))


@pytest.fixture(scope="module")
def plan(ws):
    # budget 20 is below the smoke recipes' frame length (50): a deliberate refusal in every few-shot column
    return ad.make_plan(ws, _card(), pa_recipe="pa-gru-smoke-v1", dpd_recipe="dpd-gru-smoke-v1", budgets=[2000, 20],
                        target=TargetRule(metric="NMSE", threshold=-10.0, better="lower"))


@pytest.fixture(scope="module")
def runs(ws, plan):
    return ad.run_plan(ws, plan)


def test_the_plan_is_hash_bound_and_lists_every_cell(plan):
    cells = ad.cells_of(plan)
    # PA: 3 full + 2 zero + 2x2 few; DPD: 3 full + 2 zero + 2x2 few
    assert len(cells) == 18
    assert [t for e, t, c, b in cells if e.entry_id == "pa"][:3] == ["full_retrain"] * 3
    assert cells[0][0].entry_id == "pa" and cells[-1][0].entry_id == "dpd"
    assert plan.plan_sha256 == plan.compute_sha256()
    assert plan.condition_set.independent_batches
    key = ad.cell_key(plan, "dpd", "few_shot", "drive-2", 2000, 0)
    assert key == f"adapt-{plan.plan_sha256[:12]}-dpd-few_shot-drive-2-b2000-s0"


def test_every_cell_is_an_ordinary_run_and_rerunning_creates_nothing(ws, plan, runs):
    assert len(runs) == 14, sorted(runs)                      # 18 cells - 4 refused budgets
    assert all(r.status == RunStatus.succeeded for r in runs.values()), {k: r.error for k, r in runs.items() if r.error}
    refusals = json.loads(ad.refusals_path(ws, plan).read_text())
    assert len(refusals) == 4 and all("b20-" in k for k in refusals)
    assert all("train_samples" in why or "frame" in why for why in refusals.values()), refusals
    again = ad.run_plan(ws, plan)
    assert {r.run_id for r in again.values()} == {r.run_id for r in runs.values()}


def test_zero_update_cells_move_the_source_model_without_new_samples(ws, plan, runs):
    source_dpd = runs[ad.cell_key(plan, "dpd", "full_retrain", "drive-0", None, 0)]
    target_pa = runs[ad.cell_key(plan, "pa", "full_retrain", "drive-1", None, 0)]
    cell = runs[ad.cell_key(plan, "dpd", "zero_update", "drive-1", None, 0)]
    parents = {(p.relation.value, p.run_id) for p in lineage(ws, cell.run_id).parents}
    assert ("dpd_model", source_dpd.run_id) in parents and ("pa_surrogate", target_pa.run_id) in parents
    result = load_result(ws, cell.run_id)
    assert result.evidence_type.value == "dpd_surrogate"
    assert any("transfer" in lim for lim in result.limitations)
    pa_cell = runs[ad.cell_key(plan, "pa", "zero_update", "drive-2", None, 0)]
    assert ("pa_model", runs[ad.cell_key(plan, "pa", "full_retrain", "drive-0", None, 0)].run_id) in \
        {(p.relation.value, p.run_id) for p in lineage(ws, pa_cell.run_id).parents}


def test_few_shot_cells_are_warm_started_from_the_source_within_the_budget(ws, plan, runs):
    source_pa = runs[ad.cell_key(plan, "pa", "full_retrain", "drive-0", None, 0)]
    cell = runs[ad.cell_key(plan, "pa", "few_shot", "drive-2", 2000, 0)]
    parents = {(p.relation.value, p.run_id) for p in lineage(ws, cell.run_id).parents}
    assert ("initialised_from", source_pa.run_id) in parents
    lims = load_result(ws, cell.run_id).limitations
    assert any("warm start" in lim for lim in lims) and any("first 2000 samples" in lim for lim in lims), lims


def test_the_report_keeps_every_cell_and_calls_itself_a_rehearsal(ws, plan, runs, tmp_path):
    report = ad.build_report(ws, plan)
    assert len(report.cells) == 18 and report.intact
    missing = [c for c in report.cells if c.status == "missing"]
    assert len(missing) == 4 and all(c.budget_samples == 20 and c.failure for c in missing)
    ok = [c for c in report.cells if c.status == "ok"]
    assert len(ok) == 14 and all(c.metrics.get("NMSE") is not None and c.checkpoint_sha256 for c in ok if c.task != "zero_update")
    by_task = {c.task: c for c in ok if c.entry_id == "dpd" and c.condition_id == "drive-1"}
    assert by_task["zero_update"].new_samples == 0
    assert by_task["few_shot"].new_samples == 2000
    assert by_task["full_retrain"].new_samples == next(a.train_samples for a in report.conditions if a.condition_id == "drive-1")
    assert all(c.reached_target is not None for c in ok)
    agg = next(a for a in report.aggregates if (a.entry_id, a.task, a.condition_id, a.budget_samples) == ("pa", "few_shot", "drive-1", 20))
    assert agg.n_ok == 0 and agg.n_failed == 1 and agg.metrics == {}
    assert report.evidence_bar.n_conditions == MIN_CONDITIONS_FOR_EVIDENCE and report.evidence_bar.independent_batches
    assert not report.evidence_bar.measured_origin and not report.evidence_bar.met
    assert any("rehearsal" in lim and "synthetic" in lim for lim in report.limitations), report.limitations
    assert any("4 of 18 cells" in lim for lim in report.limitations)
    assert "seeds: 1" in report.repeats and "one per condition" in report.repeats

    path = ad.store_report(ws, report)
    assert ad.load_report(path).report_sha256 == report.report_sha256
    assert [r.plan_sha256 for r in ad.list_reports(ws)] == [plan.plan_sha256]
    edited = json.loads(path.read_text())
    edited["cells"][0]["metrics"] = {"NMSE": -99.0}
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(edited))
    with pytest.raises(WorkspaceError, match="edited after it was sealed"):
        ad.load_report(tampered)

    md = ad.report_markdown(report)
    assert "Evidence bar: **not met**" in md and "FAILED 1/1" in md and "few_shot b2000" in md and "## Limitations" in md


def test_the_evidence_bar_is_about_the_card_not_the_numbers(ws, tmp_path_factory):
    src = tmp_path_factory.mktemp("measured")
    for cid, (seed, gain) in CONDITIONS.items():          # the same signals declared measured (a flag, nothing else)
        ds.import_dataset(ws, _capture(src / f"{cid}.csv", seed, gain), dataset_id=f"m-{cid}", display_name=f"measured {cid}",
                          signal=SignalSpec(**SIGNAL), origin=DatasetOrigin.measured, guard_samples=64)
    plan = ad.make_plan(ws, _card("m-", "measured"), pa_recipe="pa-gru-smoke-v1", tasks=["zero_update"])
    report = ad.build_report(ws, plan)                      # nothing ran: every cell is missing, the bar is still met
    assert report.evidence_bar.met and all(c.status == "missing" for c in report.cells)
    assert not any("rehearsal" in lim for lim in report.limitations)
    assert any("cells have no number" in lim for lim in report.limitations)


def test_a_card_whose_conditions_share_a_capture_is_refused(ws, tmp_path):
    ds.import_dataset(ws, _capture(tmp_path / "copy.csv", 11, 1.0), dataset_id="drive-0-copy", display_name="copy",
                      signal=SignalSpec(**SIGNAL), origin=DatasetOrigin.synthetic, guard_samples=64)
    card = ConditionSet(set_id="dup", device="x", dimension="drive", conditions=[
        Condition(condition_id="a", dataset_id="drive-0", role="source", capture_batch="1"),
        Condition(condition_id="b", dataset_id="drive-0-copy", role="target", capture_batch="2")])
    with pytest.raises(WorkspaceError, match="share the same raw capture"):
        ad.audit_card(ws, card)
    with pytest.raises(WorkspaceError, match="condition 'b': dataset 'nowhere'"):
        ad.audit_card(ws, ConditionSet(set_id="gone", device="x", dimension="drive", conditions=[
            Condition(condition_id="a", dataset_id="drive-0", role="source", capture_batch="1"),
            Condition(condition_id="b", dataset_id="nowhere", role="target", capture_batch="2")]))


def test_an_edited_plan_or_card_is_refused(plan, tmp_path):
    path = tmp_path / "plan.json"
    ad.write_plan(plan, path)
    assert ad.load_plan(path).plan_sha256 == plan.plan_sha256
    data = json.loads(path.read_text())
    data["seeds"] = [0, 1]
    path.write_text(json.dumps(data))
    with pytest.raises(WorkspaceError, match="edited after it was registered"):
        ad.load_plan(path)
    card_path = tmp_path / "card.json"
    card_path.write_text(plan.condition_set.model_dump_json())
    data = json.loads(card_path.read_text())
    data["conditions"][1]["role"] = "source"
    data["conditions"][0]["role"] = "target"
    card_path.write_text(json.dumps(data))
    with pytest.raises(WorkspaceError, match="edited after it was sealed"):
        ad.load_card(card_path)


def test_the_cli_reports_a_registered_plan(ws, plan, runs, tmp_path, capsys):
    plan_path = tmp_path / "plan.json"
    ad.write_plan(plan, plan_path)
    md = tmp_path / "report.md"
    assert main(["adaptation", "report", str(plan_path), "--workspace", str(ws.root), "--markdown", str(md)]) == 0
    out = capsys.readouterr().out
    assert "evidence bar NOT met" in out and "18 cells, 4 without a number" in out
    assert "not met" in md.read_text()
    assert main(["adaptation", "run", str(plan_path), "--workspace", str(ws.root)]) == 1   # 4 refused cells stay refused
    assert "4 refused" in capsys.readouterr().out
    assert main(["adaptation", "card", "apa-200mhz-batches-v1"]) == 0
    assert "2 conditions" in capsys.readouterr().out
    assert main(["adaptation", "card", "apa-200mhz-batches-v1", "--workspace", str(ws.root)]) == 2   # not registered here
    assert "condition 'batch-a'" in capsys.readouterr().err
