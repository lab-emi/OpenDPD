"""The independent matrix auditor must detect score, cost, sweep and checkpoint mistakes."""

import json
import math
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark import audit_arena_baselines as auditor
from benchmark.audit_arena_baselines import independent_score
from opendpd.core import arena
from tests.unit.test_arena_scoring import calibrated, cases  # noqa: F401  (calibrated is a fixture)

CREDIT = 10 * math.log10(2)      # The most a cost ratio below 0.5 can earn.


def observation(quality, **updates):
    """EVM gains quality + 1 dB and the worse (left) adjacent side quality − 1 dB; NMSE is only reported."""
    return dict(nmse_db=-20.-quality, baseline_nmse_db=-20., evm_db=-21.-quality, baseline_evm_db=-20.,
        aer_l_db=-29.-quality, baseline_aer_l_db=-30., aer_r_db=-40.-quality, baseline_aer_r_db=-40.,
        aclr_l_db=-29.-quality, baseline_aclr_l_db=-30., aclr_r_db=-40.-quality, baseline_aclr_r_db=-40.,
        power_error_db=0., ib_error_db=-35., baseline_ib_error_db=-21., **updates)


def row(costs, qualities=(4., 5., 6.), seeds=(0, 1, 2)):
    """``costs`` maps a budget to (parameters, MUL, ADD); a budget without a cost has no configuration."""
    budgets, observed = [], []
    for budget in (250, 500, 1000, 2000):
        if budget not in costs:
            budgets.append(dict(budget=budget, available=False, model_parameters=None))
            continue
        parameters, mul, add = costs[budget]
        budgets.append(dict(budget=budget, available=True, parameters=parameters, mul=mul, add=add, ops=mul + add))
        values = qualities[budget] if isinstance(qualities, dict) else qualities
        observed += [dict(budget=budget, condition_id="condition", seed=seed, judges=[observation(quality)],
                          checkpoint_sha256="a" * 64) for seed, quality in zip(seeds, values)]
    return dict(board_id="board", backbone="gru", seeds=list(seeds), budgets=budgets, cases=observed)


# One dense layer's worth of cost, ten times the operations, a tenth of everything; no 2,000 configuration.
COSTS = {250: (250, 250, 250), 500: (500, 5000, 5000), 1000: (100, 100, 100)}


def test_independent_fom_uses_mean_quality_and_fixed_costs():
    values=independent_score(row(COSTS))
    assert values['eligible']
    reference, heavy, tiny = (values['budgets'][b] for b in (250,500,1000))
    assert reference['quality_db']==5. and reference['quality_std_db']==1.
    assert reference['score']==pytest.approx(5+10*math.log10(4))
    assert heavy['score']==pytest.approx(5-5*math.log10(.5)-5*math.log10(5))
    assert tiny['score']==pytest.approx(15.)
    assert values['score']==15.
    assert all(values['rankings'][f'budget-{b}']==15. for b in auditor.BUDGETS)



def test_independent_costs_ignore_host_timing_and_keep_fixed_references():
    cheap,costly=row({1000:(1000,1000,1000)}),row({1000:(1000,1500,2500)})
    for case in costly['cases']: case.update(latency_ns_per_sample=1e-6)
    assert independent_score(cheap)['score']==5.
    assert independent_score(costly)['score']==pytest.approx(5-5*math.log10(2))



def test_independent_gate_rejects_cheap_identity_and_power_cheating():
    costs={b:(1,1,0) for b in auditor.BUDGETS}
    assert not independent_score(row(costs,(0.,)*3))['eligible']
    assert independent_score(row(costs,(-2.,)*3))['score'] is None
    assert independent_score(row(costs,(.5,)*3))['score']==pytest.approx(.5-5*math.log10(.001)-5*math.log10(.0005))
    example=row(costs,(30.,)*3)
    for case in example['cases']: case['judges'][0]['power_error_db']=-.51
    assert independent_score(example)['score'] is None



def test_independent_quality_is_the_equal_weight_mean_and_nmse_never_enters():
    example = row(COSTS, {250: (9., 9., 9.), 500: (9., 9., 9.), 1000: (9., 9., 9.)}, seeds=(0,))
    judge = example["cases"][0]["judges"][0]
    judge["nmse_db"] = judge["baseline_nmse_db"] + 5.        # a diagnostic: the score does not move
    assert independent_score(example)["budgets"][250]["quality_conservative_db"] == pytest.approx(9.)
    judge["evm_db"] = judge["baseline_evm_db"] + 2.          # EVM 2 dB worse, the worse side still 8 dB better
    values = independent_score(example)["budgets"][250]
    assert values["qualified"] and values["quality_conservative_db"] == pytest.approx(.5 * (-2. + 8.))
    judge["aclr_r_db"] = -31.                                 # the right side becomes the worse one: −30 → −31
    assert independent_score(example)["budgets"][250]["quality_conservative_db"] == pytest.approx(.5 * (-2. + 1.))


def test_deterministic_single_seed_has_no_invented_uncertainty():
    values=independent_score(row({500:(500,500,500)},(4.,),seeds=(0,)))
    point=values['budgets'][500]
    assert point['quality_std_db']==0. and point['quality_conservative_db']==4.
    assert values['score']==pytest.approx(4+10*math.log10(2))



PROFILES = [
    ("gru", "apa-200mhz-b", dict(qualities=(3., 5., 7.))),
    ("gru", "apa-200mhz-b", dict(qualities=(8., 8.5, 9.), by_budget={250: (-1., -1., -1.), 2000: (20., 20., 1.)})),
    ("mcldnn", "apa-200mhz-b", dict(qualities=(3., 3., 3.))),                     # negative arithmetic efficiency
    ("gmp", "apa-200mhz-b", dict(qualities=(6., 7., 7.5))),                  # one budget, one measured condition
    ("mp_ls", "apa-200mhz-b", dict(qualities=(4.,))),                        # deterministic: one seed
    ("gru_stream", "apa-200mhz-b", dict(qualities=(5., 5., 6.))),
    ("tcn", "apa-200mhz-b", dict(qualities=(2.9, 5., 7.1))),                      # a wide seed spread
]


@pytest.mark.parametrize("backbone,board_id,profile", PROFILES)
def test_independent_auditor_reproduces_the_production_summary_without_calling_it(
        calibrated, monkeypatch, backbone, board_id, profile):  # noqa: F811  (the imported fixture)
    from opendpd.schemas.arena import ArenaRow
    raw = cases(calibrated, board_id, backbone=backbone, **profile)
    raw[0]["judges"][-1]["aer_r_db"] += 1.5          # One judge and metric limits one seed.
    produced = arena.summarize_cases(backbone, board_id, raw, arena.sweep(backbone))
    published = ArenaRow(entry_id="job", board_id=board_id, backbone=backbone, display_name=backbone,
        origin="workspace", status="succeeded", protocol_sha256="a" * 64, cases=raw,
        evidence_type=arena.board(board_id).evidence_type, **produced).model_dump(mode="json")
    monkeypatch.setattr(arena, "summarize_cases", lambda *args: pytest.fail("the audit called the production scorer"))
    monkeypatch.setattr(arena, "_budget_result", lambda *args: pytest.fail("the audit called the production scorer"))
    audited = independent_score(published)
    assert audited["eligible"] == published["eligible"]
    assert audited["score"] == (None if published["score"] is None else pytest.approx(published["score"], abs=1e-9))
    assert {key: entry["score"] for key, entry in published["rankings"].items() if entry["score"] is not None} == {
        key: pytest.approx(value, abs=1e-9) for key, value in audited["rankings"].items() if value is not None}
    available = {point["budget"]: point for point in published["budgets"] if point["available"]}
    assert set(audited["budgets"]) == set(available)
    for budget, value in audited["budgets"].items():
        assert value["qualified"] == available[budget]["qualified"]
        for field in ("quality_db", "quality_std_db", "quality_conservative_db", "parameter_efficiency_db",
                      "arithmetic_efficiency_db", "score"):
            assert value[field] == pytest.approx(available[budget][field], abs=1e-9), (budget, field)
        for field in ("nmse_improvement_db", "evm_improvement_db", "aer_improvement_db"):
            assert value[field] == pytest.approx(available[budget]["metrics"][field], abs=1e-9)


def test_auditor_restates_the_protocol_constants_instead_of_importing_them():
    assert list(auditor.BUDGETS) == arena.BUDGETS == arena.protocol().budgets
    assert auditor.METRICS == ("evm_db", "aclr_l_db", "aclr_r_db")
    assert not hasattr(auditor, "quality_snapshot") and not any("timing" in name or "retim" in name for name in vars(auditor))
    assert independent_score.__defaults__ == (arena.TRAINING["output_power_tolerance_db"],)      # the only gate


# --- a real fitted, judged, published and audited entry ------------------------------------------------

def redirect(patch, assets):
    """A private asset folder, and one registered 48-parameter MP so that a real fit takes a moment:
    a fixed-size model has a configuration in its own budget class only."""
    patch.setattr(arena, "ASSETS", assets)
    patch.setattr(arena, "MP_PRESETS", {budget: dict(K=3, Q=8) for budget in arena.BUDGETS})


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    from opendpd.core import arena_runner
    root = tmp_path_factory.mktemp("official")
    shutil.copytree(arena.ASSETS, root / "assets", ignore=shutil.ignore_patterns("reference-results-*"))
    with pytest.MonkeyPatch.context() as patch:
        redirect(patch, root / "assets")
        folder = root / "matrix" / "jobs" / "apa-200mhz-b--mp_ls"
        folder.mkdir(parents=True)
        request = dict(board_id="apa-200mhz-b", backbone="mp_ls", protocol_sha256=arena.protocol().protocol_sha256,
                       model_parameters={}, model_provenance={"source": "bundled registry"})
        (folder / "request.json").write_text(json.dumps(request))
        result = arena_runner.evaluate_request(request, folder / "result.json", root / "matrix" / "cache", device="cpu")
        assert result.status == "succeeded", result.error
    return root


@pytest.fixture
def official(fitted, tmp_path, monkeypatch):
    """A private copy of that job and cache, honestly published; a test then alters what it audits."""
    from benchmark.run_arena_baselines import publish
    shutil.copytree(fitted, tmp_path / "official")
    redirect(monkeypatch, tmp_path / "official" / "assets")
    root = tmp_path / "official" / "matrix"
    publish(root, arena.protocol())

    def edit(path, change):
        payload = json.loads(path.read_text())
        change(payload)
        path.write_text(json.dumps(payload))

    return SimpleNamespace(root=root, job=root / "jobs" / "apa-200mhz-b--mp_ls" / "result.json", edit=edit,
        cache=next((root / "cache").glob("*/apa-200mhz-b/seed-0")),
        bundle=tmp_path / "official" / "assets" / arena.RESULTS_FILE,
        audit=lambda **options: auditor.audit(root, **{"allow_partial": True, "replay": False,
                                                       "operations": False, **options})[0])


def test_real_deterministic_entry_passes_the_development_audit_and_is_not_called_complete(official):
    result = json.loads(official.job.read_text())
    assert [point["available"] for point in result["budgets"]] == [True, False, False, False]
    assert result["budgets"][0]["parameters"] == 48 and result["expected_cases"] == result["completed_cases"] == 1
    report = official.audit()
    assert report["problems"] == [] and report["warnings"] == [] and report["integrity_ok"]
    assert (report["observed_rows"], report["succeeded_rows"], report["completed_cases"]) == (1, 1, 1)
    assert report["qualified_budget_points"] == sum(point["qualified"] for point in result["budgets"])
    assert report["mode"] == "development_partial" and not report["complete"] and len(report["missing_rows"]) == 22
    assert report["protocol_sha256"] == arena.protocol().protocol_sha256
    assert report["training_sha256"] == arena.training_fingerprint() == result["provenance"]["training_sha256"]
    final = official.audit(allow_partial=False)
    assert not final["integrity_ok"] and final["problems"] == ["Missing 22 of 23 required rows"]


def _raise_every_error_metric(result):
    for judge in result["cases"][0]["judges"]:
        for metric in ("nmse_db", "aer_l_db", "aer_r_db"):
            judge[metric] -= 3.


@pytest.mark.parametrize("change,finding", [
    (lambda result: result.update(score=result["score"] + .01), "independent FoM mismatch"),
    (lambda result: result["rankings"]["linearization"].update(score=9.), "independent ranking mismatch: linearization"),
    (lambda result: result["rankings"]["budget-500"].update(score=1.), "independent ranking mismatch: budget-500"),
    (lambda result: result.update(eligible=not result["eligible"]), "eligibility disagrees"),
    (lambda result: result["budgets"][0].update(qualified=not result["budgets"][0]["qualified"]), "qualification mismatch"),
    (lambda result: result["budgets"][0].update(quality_conservative_db=9.), "independent quality_conservative_db mismatch"),
    (lambda result: result["budgets"][0].update(arithmetic_efficiency_db=9.), "independent arithmetic_efficiency_db mismatch"),
    (lambda result: result["budgets"][0]["metrics"].update(aer_improvement_db=30.), "independent metric summary mismatch"),
    (_raise_every_error_metric, "mismatch"),
    (lambda result: result["budgets"][0].update(ops=result["budgets"][0]["ops"] - 1), "OPs is not MUL + ADD"),
    (lambda result: result["budgets"][0]["model_parameters"].update(K=4), "official sweep preset was changed"),
    (lambda result: result["budgets"].pop(), "budget sweep is incomplete"),
    (lambda result: result["cases"].append(dict(result["cases"][0])), "completed count does not match cases"),
    (lambda result: result["cases"][0].update(budget=500), "missing/duplicate budget, condition or seed"),
    (lambda result: result["cases"][0].update(reference_gain=result["cases"][0]["reference_gain"] * 1.0001), "reference gain"),
    (lambda result: result["cases"][0].update(parameters=50), "parameter count differs from checkpoint"),
    (lambda result: result["cases"][0].update(checkpoint_sha256="0" * 64), "checkpoint hash mismatch"),
    (lambda result: result["cases"][0]["cache_binding"].update(seed=1), "cache request binding mismatch"),
    (lambda result: result["cases"][0]["fit"].update(rank=1), "case training metadata differs"),
    (lambda result: result["cases"][0].update(selected_epoch=5), "case training metadata differs"),
    (lambda result: result["cases"][0]["judges"].pop(), "missing/duplicate final judge"),
    (lambda result: result["cases"][0]["judges"][0].update(checkpoint_sha256="0" * 64), "judge weight/equation hash mismatch"),
    (lambda result: result["cases"][0]["judges"][0].pop("baseline_power_error_db"), "missing/nonfinite raw metric"),
    (lambda result: result["provenance"].update(training_sha256="0" * 64), "training fingerprint mismatch"),
    (lambda result: result.update(execution_semantics="streaming_stateful"), "wrong execution semantics"),
    (lambda result: result.update(seeds=[0, 1, 2]), "wrong seed list"),
    (lambda result: result.update(protocol_sha256="0" * 64), "stale protocol artifact"),
])
def test_audit_detects_an_altered_result(official, change, finding):
    official.edit(official.job, change)
    report = official.audit()
    assert not report["integrity_ok"]
    assert any(finding in problem for problem in report["problems"]), report["problems"]


def _replace_weights(official):
    from opendpd.core import arena_engine
    arena_engine._save_weights(arena_engine.build_model("mp_ls", arena.model_parameters("mp_ls", 250)),
                               official.cache / "weights.npz")


@pytest.mark.parametrize("damage,finding", [
    (_replace_weights, "checkpoint hash mismatch"),
    (lambda official: official.edit(official.cache / "training.json",
                                    lambda record: record.update(weights_sha256="0" * 64)), "checkpoint hash mismatch"),
    (lambda official: official.edit(official.cache / "training.json",
                                    lambda record: record["cache_binding"]["model_parameters"].update(K=4)),
     "cache request binding mismatch"),
    (lambda official: official.edit(official.cache / "training.json",
                                    lambda record: record["cache_binding"].update(training_sha256="0" * 64)),
     "cache request binding mismatch"),
    (lambda official: shutil.rmtree(official.cache), "FileNotFoundError"),
])
def test_audit_detects_a_replaced_or_missing_checkpoint_and_an_edited_cache_record(official, damage, finding):
    damage(official)
    report = official.audit()
    assert not report["integrity_ok"] and any(finding in problem for problem in report["problems"]), report["problems"]


def test_audit_detects_a_fit_on_a_different_sample_budget(official):
    def fewer(record):
        record["fit"]["n_observations"] -= 1
    official.edit(official.cache / "training.json", fewer)
    official.edit(official.job, lambda result: fewer(result["cases"][0]))
    assert any("different sample budget" in problem for problem in official.audit()["problems"])


def test_audit_verifies_the_published_cost_against_the_instrumented_operation_count(official, monkeypatch):
    from benchmark import verify_arena_operations
    from opendpd.core import arena_ops
    parameters = arena.model_parameters("mp_ls", 250)
    cost = arena_ops.count("mp_ls", parameters)
    checked = dict(backbone="mp_ls", budget=250, parameters=48, mul=cost["mul"], add=cost["add"], ops=cost["ops"])
    verified = dict(samples=40, configurations=1, instrumented=0, problems=[], rows=[checked])
    monkeypatch.setattr(verify_arena_operations, "report", lambda: verified)
    report = official.audit(operations=True)
    assert report["problems"] == [] and report["operation_count_check"] == {k: v for k, v in verified.items() if k != "rows"}

    def exchanged(result):        # The same OPs and therefore the same score, but not the verified ledger.
        result["budgets"][0].update(mul=cost["mul"] + 1, add=cost["add"] - 1)
    official.edit(official.job, exchanged)
    assert official.audit()["problems"] == []          # Undetectable without the operation check.
    report = official.audit(operations=True)
    assert [problem for problem in report["problems"] if "verified operation count" in problem]
    disagreement = "gru@250: kernel products 1 != 2"
    monkeypatch.setattr(verify_arena_operations, "report", lambda: dict(verified, problems=[disagreement]))
    assert "Operation count: " + disagreement in official.audit(operations=True)["problems"]


def test_audit_compares_the_published_bundle_with_the_jobs(official):
    published = json.loads(official.bundle.read_text())
    assert [(entry["entry_id"], entry["origin"]) for entry in published["rows"]] == [("official-apa-200mhz-b-mp_ls", "official")]
    assert official.audit()["problems"] == official.audit()["warnings"] == []
    official.edit(official.job, lambda result: result["provenance"].update(note="edited after publication"))
    development = official.audit()
    assert development["integrity_ok"] and any("Published/job mismatch" in warning for warning in development["warnings"])
    assert any("Published/job mismatch" in problem for problem in official.audit(allow_partial=False)["problems"])
    official.edit(official.bundle, lambda bundle: bundle["rows"][0].update(score=99.))
    assert "Published bundle content seal mismatch" in official.audit()["problems"]
    official.edit(official.bundle, lambda bundle: bundle.update(protocol_sha256="0" * 64))
    assert "Published bundle protocol mismatch" in official.audit()["problems"]


COUNTERS = ("expected_cases_present_rows", "completed_cases", "succeeded_rows", "eligible_rows", "qualified_budget_points")
NO_BUNDLE = "No published bundle to compare with the jobs"


def test_audit_before_publication_reports_the_missing_bundle_and_still_audits_the_jobs(official):
    official.bundle.unlink()
    development = official.audit()
    assert development["integrity_ok"] and development["problems"] == []
    assert development["warnings"] == [NO_BUNDLE, "Published/job mismatch: ('apa-200mhz-b', 'mp_ls')"]
    result = json.loads(official.job.read_text())
    assert [development[counter] for counter in COUNTERS] == [
        1, 1, 1, int(result["eligible"]), sum(point["qualified"] for point in result["budgets"])]
    # The jobs are audited in full without a bundle: an altered one is still a problem, not a warning.
    official.edit(official.job, lambda job: job.update(score=job["score"] + .01))
    assert any("independent FoM mismatch" in problem for problem in official.audit()["problems"])
    # The final audit cannot pass before publication.
    final = official.audit(allow_partial=False)
    assert not final["integrity_ok"] and final["warnings"] == [] and NO_BUNDLE in final["problems"]


def test_audit_of_a_workspace_without_any_job_reports_every_counter_as_zero(official):
    shutil.rmtree(official.root / "jobs")
    official.bundle.unlink()
    report = official.audit()
    assert report["problems"] == [] and report["warnings"] == [NO_BUNDLE] and report["integrity_ok"]
    assert {counter: report[counter] for counter in COUNTERS} == dict.fromkeys(COUNTERS, 0)
    assert report["observed_rows"] == 0 and len(report["missing_rows"]) == report["expected_rows"] == 23
    assert not report["complete"] and report["execution_failures"] == []
    assert all(json.loads(json.dumps(report))[counter] == 0 for counter in COUNTERS)     # Also as the written document.


def test_development_audit_command_runs_while_the_matrix_is_still_unpublished(official, monkeypatch, capsys):
    official.bundle.unlink()
    out = official.root / "audit.json"
    arguments = ["audit", "--workspace", str(official.root), "--skip-streaming-replay", "--skip-operation-check",
                 "--out", str(out)]
    monkeypatch.setattr(sys, "argv", [*arguments, "--allow-partial"])
    assert auditor.main() == 0
    written = json.loads(out.read_text())
    assert written == json.loads(capsys.readouterr().out)
    assert written["integrity_ok"] and NO_BUNDLE in written["warnings"] and written["succeeded_rows"] == 1
    # Without --allow-partial the same workspace is a failed final audit, reported rather than raised.
    monkeypatch.setattr(sys, "argv", arguments)
    assert auditor.main() == 1
    final = json.loads(out.read_text())
    assert NO_BUNDLE in final["problems"] and not final["integrity_ok"]
    assert "Final audit must replay streaming checkpoints and verify operation counts" in final["problems"]


def test_failed_job_is_reported_as_an_execution_failure_and_cannot_carry_a_score(official):
    def failed(result):
        result.update(status="failed", eligible=False, score=None, rankings={}, cases=[], completed_cases=0,
                      error="FloatingPointError: fixture")
    official.edit(official.job, failed)
    report = official.audit()
    assert report["problems"] == [] and report["observed_rows"] == 1
    # A counter without anything to count is reported as zero, not left out of the report.
    assert [report[counter] for counter in COUNTERS] == [1, 0, 0, 0, 0]
    assert report["execution_failures"] == [
        {"board_id": "apa-200mhz-b", "backbone": "mp_ls", "error": "FloatingPointError: fixture"}]
    official.edit(official.job, lambda result: result.update(score=1.))
    assert any("failed row carries a rankable score" in problem for problem in official.audit()["problems"])


@pytest.mark.parametrize("device,chunk_error,fails", [("cuda", 1e-7, None),
                                                     ("cpu", 1e-7, "Original-device"),
                                                     ("cuda", .01, "DPD IQ depends")])
def test_streaming_audit_separates_chunk_invariance_from_judge_device(
        tmp_path, monkeypatch, device, chunk_error, fails):
    import torch
    from torch import nn
    from benchmark.audit_arena_baselines import replay_streaming
    from opendpd.core import arena_metrics, arena_runner, streaming
    from opendpd.services import streaming as adapters

    x = np.ones((128, 2), dtype=np.float32)
    np.savez(tmp_path / "data.npz", x_test=x, x_train=x)
    monkeypatch.setattr(arena, "ASSETS", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    built = []
    monkeypatch.setattr(arena_runner, "build_model", lambda *args: built.append(args) or nn.Identity())
    monkeypatch.setattr(arena_runner, "_load_weights", lambda model, path: model)
    monkeypatch.setattr(adapters, "streaming_model", lambda key, model: model)
    monkeypatch.setattr(streaming, "run_stream",
                        lambda model, data, chunk: data * (1-chunk_error if chunk == 137 else 1))
    monkeypatch.setattr(arena_runner, "load_frozen", lambda spec, where: where)
    devices = []

    def predict(model, data, where):
        devices.append(where)
        return data + (.01 if where == "cpu" else 0.)

    monkeypatch.setattr(arena_runner, "pa_output", predict)
    grid = dict(useful_samples=128, prefix_samples=0, occupied_hz=[[-1., 1.]])
    monkeypatch.setattr(arena, "TRAINING", {**arena.TRAINING, "evm_grids": {"condition": grid}})
    seen = []
    monkeypatch.setattr(arena_metrics, "compute",
                        lambda y, ref, signal, *where: seen.append(where) or {"nmse_db": float(y.sum())})
    sample_row = dict(backbone="gru_stream", provenance={"model_parameters": {}},
                      budgets=[dict(budget=point["budget"], model_parameters=point["model_parameters"])
                               for point in arena.sweep("gru_stream")])
    case = dict(budget=500, condition_id="condition", seed=0, training_device="cuda", reference_gain=1.,
        judges=[{"judge_id": key, "nmse_db": float(x.sum()), "evm_db": -300.} for key in ("pa",)])
    meta = dict(data_file="data.npz", peak_limit=2., teacher={}, evm_grid=grid, stimuli={"splits": {"test": {"metric_start": 0, "metric_stop": 128}}}, simulation=None, judges=[], split={"boundaries": {"test": [640, 768]}},
        signal=dict(sample_rate_hz=8., bandwidth_hz=2., n_sub_ch=1, nperseg=128))
    if fails:
        with pytest.raises(ValueError, match=fails):
            replay_streaming(sample_row, case, tmp_path / "unused.npz", meta, judge_device=device)
    else:
        result = replay_streaming(sample_row, case, tmp_path / "unused.npz", meta, judge_device=device)
        assert result["stored_metric_max_abs_delta_db"] == 0.
        assert result["same_device_chunk_max_abs_metric_delta_db"] < .001
        assert result["judge_device"] == "cuda" and set(devices) == {"cuda"}
        assert (result["backbone"], result["budget"], result["condition_id"], result["seed"]) == (
            "gru_stream", 500, "condition", 0)
        assert seen and all(where == (grid, 640, (0,128)) for where in seen)     # demodulated on the capture's own grid
        # The second demodulator guards the stored EVM: a published value it cannot reproduce stops the audit.
        case["judges"][0]["evm_db"] = -40.
        with pytest.raises(ValueError, match="Independent EVM"):
            replay_streaming(sample_row, case, tmp_path / "unused.npz", meta, judge_device=device)
    # The replayed network is the base model at the size registered for the case's budget.
    assert built and all(entry == ("gru", arena.model_parameters("gru", 500)) for entry in built)
