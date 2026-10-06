"""Analytic Arena quality, sweep and cost checks with complete frozen judging evidence."""

import copy
import math
import subprocess
import sys

import numpy as np
import pytest

from opendpd.core import arena, arena_ops


@pytest.fixture
def calibrated(monkeypatch):
    records = {}
    for index, condition in enumerate(condition for board in arena.BOARDS for condition in board["conditions"]):
        record = {"condition_id": condition, "data_sha256": str(index + 1) * 64,
                  "counts": {"train": 23040},
                  "teacher": {"sha256": "a" * 64}, "simulation": None,
                  "judges": [{"model": {"key": "gru", "parameters": {"hidden_size": 64}}, "sha256": "b" * 64},
                             {"model": {"key": "gmp_ls", "parameters": {}}, "sha256": "c" * 64}]}
        if condition.startswith("syn-"):
            record["simulation"] = {"virtual_pa": "test-pa", "parameters": {"gain": index + 1.0}}
            record["judges"] = []
        records[condition] = record
    monkeypatch.setattr(arena, "calibration", lambda: records)
    monkeypatch.setattr(arena, "file_hash", lambda _: "f" * 64)
    return records


def observations(judge_id, checkpoint, quality):
    """EVM gains quality + 2 dB and the worse (left) adjacent side quality − 2 dB: their mean is ``quality``."""
    return {"judge_id": judge_id, "checkpoint_sha256": checkpoint,
            "baseline_nmse_db": -20., "nmse_db": -20. - quality - 2,
            "baseline_evm_db": -20., "evm_db": -20. - quality - 2,
            "baseline_aclr_l_db": -30., "aclr_l_db": -30. - quality + 2,
            "baseline_aclr_r_db": -35., "aclr_r_db": -35. - quality - 1,
            "baseline_aer_l_db": -25., "aer_l_db": -25. - quality + 2,
            "baseline_aer_r_db": -28., "aer_r_db": -28. - quality + 1,
            "reference_aclr_l_db": -40., "reference_aclr_r_db": -42.,
            "baseline_ib_error_db": -21., "ib_error_db": -30., "power_error_db": 0.}


def cases(records, board_id="apa-200mhz-b", qualities=(3., 5., 7.), backbone="gru", by_budget=None):
    """Every available budget of ``backbone``; ``by_budget`` overrides the seed qualities of one budget."""
    board = next(b for b in arena.BOARDS if b["board_id"] == board_id)
    deterministic = backbone in arena.DETERMINISTIC
    output = []
    for point in arena.sweep(backbone):
        if point["model_parameters"] is None:
            continue
        values = (by_budget or {}).get(point["budget"], qualities)
        for seed, quality in zip(arena.SEEDS[:1] if deterministic else arena.SEEDS, values):
            for condition in board["conditions"]:
                record = records[condition]
                output.append({"budget": point["budget"], "condition_id": condition, "seed": seed,
                    "evaluation_split": "test",
                    "parameters": arena_ops.parameter_count(backbone, point["model_parameters"]),
                    "data_sha256": record["data_sha256"], "teacher_sha256": record["teacher"]["sha256"],
                    "attained_epochs": 0 if deterministic else arena.TRAINING["epochs"],
                    "optimizer_updates": 0 if deterministic else arena.training_budget(record["counts"]["train"])["optimizer_updates"],
                    "selected_epoch": None if deterministic else arena.TRAINING["epochs"],
                    **({} if deterministic else {"frame_draw_sha256": "d" * 64}),
                    "reference_gain": record.get("reference_gain", 1.2),
                    "judges": [observations(key, sha, quality) for key, sha in arena.judge_hashes(record).items()]})
    return output


def score(rows, board="apa-200mhz-b", backbone="gru"):
    return arena.summarize_cases(backbone, board, rows, arena.sweep(backbone))


def adjusted(backbone, budget, quality, parameters=.5, operations=.5):
    """The published law, written out: quality minus weighted 10 log10 cost ratios, floored at 0.5."""
    params = arena.model_parameters(backbone, budget)
    count, cost = arena_ops.parameter_count(backbone, params), arena_ops.count(backbone, params)
    return (quality - parameters * 10 * math.log10(count / 1000)
            - operations * 10 * math.log10(cost["ops"] / 2000))


def test_mean_quality_drives_score_and_seed_spread_remains_visible(calibrated):
    result = score(cases(calibrated))
    assert result['eligible'] and result['qualified_budgets'] == 4
    for point in result['budgets']:
        assert point['quality_db'] == 5. and point['quality_std_db'] == 2.
        assert point['quality_conservative_db'] == 3.
        assert point['score'] == pytest.approx(adjusted('gru', point['budget'], 5.))
        assert point['metrics'].evm_db == -27.
        assert point['metrics'].aclr_db == -33.
        assert point['metrics'].aer_db == -28.



def test_scoring_uses_output_aclr_quality():
    from opendpd.core.arena_metrics import quality_db
    judge = observations('pa', 'b'*64, 6.5)
    judge.update(aclr_l_db=-41., aclr_r_db=-33.)
    expected = .5*((-20.- -28.5)+(-30.- -33.))
    baseline = {k[len('baseline_'):]: v for k,v in judge.items() if k.startswith('baseline_')}
    assert arena.observation_quality(judge) == quality_db(judge, baseline) == expected



def test_configurations_use_fixed_costs_and_backbone_summary_is_explicit_best_observed(calibrated):
    result = score(cases(calibrated, qualities=(6.,)*3))
    expected = [adjusted('gru', budget, 6.) for budget in arena.BUDGETS]
    assert result['score'] == max(expected)
    assert result['best_budget'] == 250
    for budget in arena.BUDGETS:
        assert result['rankings'][f'budget-{budget}']['score'] == max(expected[:arena.BUDGETS.index(budget)+1])
    assert result['rankings']['parameter_efficiency']['score'] == max(adjusted('gru', b, 6.,1,0) for b in arena.BUDGETS)



def test_tenfold_cost_needs_ten_decibels_more_quality(calibrated):
    rows = [c for c in cases(calibrated, qualities=(9., 9., 9.)) if c["budget"] == 1000]
    params = arena.model_parameters("gru", 1000)
    ordinary = dict(ops=2000, mul=1000, add=1000, nonlinear={}, nonlinear_mul=0, nonlinear_add=0, items=[])
    expensive = dict(ordinary, ops=20000, mul=10000, add=10000)
    base = arena._budget_result(1000, params, 1000, ordinary, rows, arena.SEEDS)
    heavy = arena._budget_result(1000, params, 1000, expensive, rows, arena.SEEDS)
    assert base["score"] == base["parameter_efficiency_db"] == base["arithmetic_efficiency_db"] == 9.
    assert heavy["arithmetic_efficiency_db"] == pytest.approx(-1.) and heavy["parameter_efficiency_db"] == 9.
    assert heavy["score"] == pytest.approx(4.)     # half the operation cost
    both = arena._budget_result(1000, params, 10000, expensive, rows, arena.SEEDS)
    assert both["score"] == pytest.approx(-1.)     # ten times both costs the full 10 dB


def test_cost_credit_has_no_budget_dependent_floor(calibrated):
    rows = [c for c in cases(calibrated, qualities=(4.,)*3) if c['budget']==1000]
    cost = dict(ops=1,mul=1,add=0,nonlinear={},nonlinear_mul=0,nonlinear_add=0,items=[])
    point = arena._budget_result(1000,{},1,cost,rows,arena.SEEDS)
    assert point['score'] == pytest.approx(4-5*math.log10(1/1000)-5*math.log10(1/2000))



def test_power_failure_excludes_only_that_configuration_and_preserves_its_metrics(calibrated):
    rows = cases(calibrated, qualities=(8.,)*3, by_budget={250:(12.,)*3})
    rows[0]['judges'][0]['power_error_db'] = -.6
    result=score(rows)
    assert not result['budgets'][0]['qualified']
    assert result['budgets'][0]['quality_db']==12.
    assert result['rankings']['budget-250']['score'] is None
    assert result['score']==max(adjusted('gru', b,8.) for b in (500,1000,2000))



def test_small_positive_quality_can_rank_without_a_three_db_threshold(calibrated):
    result=score(cases(calibrated, qualities=(1.,)*3))
    assert result['eligible'] and result['qualified_budgets']==4
    assert result['rankings']['budget-250']['score']==pytest.approx(adjusted('gru',250,1.))



def test_missing_configuration_is_unavailable_and_never_a_zero_measurement(calibrated):
    result=score(cases(calibrated,backbone='mcldnn',qualities=(8.,)*3),backbone='mcldnn')
    assert result['available_budgets']==2
    assert result['rankings']['budget-250']['score'] is None
    assert result['rankings']['budget-500']['score'] is None
    assert result['score']==max(adjusted('mcldnn',b,8.) for b in (1000,2000))



def test_valid_negative_fom_is_preserved_instead_of_clipped_to_zero(calibrated):
    result=score(cases(calibrated,backbone='mcldnn',qualities=(.1,)*3),backbone='mcldnn')
    assert result['eligible'] and result['score']<0
    assert result['rankings']['budget-1000']['score']<0



def test_quality_and_efficiency_leaders_can_be_different_configurations(calibrated):
    rows=cases(calibrated,qualities=(4.,)*3,by_budget={1000:(9.,)*3,2000:(20.,)*3})
    for case in rows:
        if case['budget']==2000: case['judges'][0]['power_error_db']=.7
    result=score(rows)
    assert result['rankings']['linearization']['score']==9.
    assert result['rankings']['evm']['score']==11.
    assert result['rankings']['aclr']['score']==7.
    assert result['best_budget']==250 and result['quality_db']==4.



def test_seed_spread_is_reported_without_changing_mean_fom(calibrated):
    wide=score(cases(calibrated,qualities=(2.9,5.,7.1)))
    flat=score(cases(calibrated,qualities=(5.,)*3))
    assert wide['score']==flat['score']
    assert wide['budgets'][0]['quality_std_db']==pytest.approx(2.1)
    assert wide['budgets'][0]['quality_conservative_db']==pytest.approx(2.9)



def test_no_budget_inside_the_power_gate_means_no_rank(calibrated):
    rows = cases(calibrated, qualities=(9., 9., 9.))
    for case in rows:
        case["judges"][0]["power_error_db"] = -.51
    result = score(rows)
    assert not result["eligible"] and result["score"] is None and result["qualified_budgets"] == 0
    assert all(entry["score"] is None for entry in result["rankings"].values())
    assert result["eligibility_reasons"] == ["Output power falls outside the fixed-target ±0.5 dB envelope"]


def test_additional_pa_judges_cannot_change_the_frozen_plant(calibrated):
    rows=cases(calibrated)
    rows[0]['judges'].append(observations('gmp','b'*64,-20.))
    with pytest.raises(ValueError, match='exactly the frozen'):
        score(rows)



def test_only_apa_b_contributes_to_current_rankings(calibrated):
    results = {}
    for board, quality in zip(arena.BOARDS, (3., 5., 7., 9.)):
        identifier = board['board_id']
        rows = cases(calibrated, identifier, qualities=(quality,)*3)
        result = score(rows, identifier)
        assert result['expected_cases'] == 12
        assert all(p['quality_db'] == quality for p in result['budgets'])
        assert {case['condition_id'] for case in rows} == {identifier}
        results[identifier] = result['score']
    assert set(results) == {"apa-200mhz-b"}
    with pytest.raises(ValueError, match="Unknown Arena leaderboard"):
        score(rows, "dpa-200mhz")



def test_nmse_is_a_diagnostic_and_never_moves_a_score(calibrated):
    plain = score(cases(calibrated, qualities=(6., 6., 6.)))
    rows = cases(calibrated, qualities=(6., 6., 6.))
    for case in rows:
        for judge in case["judges"]:
            judge["nmse_db"] = judge["baseline_nmse_db"] + 3.   # a time-domain error the receiver never sees
    result = score(rows)
    assert result["score"] == plain["score"] and result["eligible"]
    assert result["metrics"].nmse_improvement_db == -3.


def test_output_aclr_changes_quality_while_aer_is_diagnostic(calibrated):
    rows=cases(calibrated,qualities=(10.,)*3)
    plain=score(copy.deepcopy(rows))
    rows[0]['judges'][0]['aclr_r_db']=-25.
    changed=score(rows)
    assert changed['budgets'][0]['quality_db'] < plain['budgets'][0]['quality_db']
    rows[0]['judges'][0]['aer_r_db']=100.
    assert score(rows)['score']==changed['score']



def test_a_loss_in_one_half_costs_quality_but_is_not_a_gate(calibrated):
    rows = [c for c in cases(calibrated, qualities=(10., 10., 10.)) if c["budget"] == 250]
    for case in rows:
        for judge in case["judges"]:
            judge["evm_db"] = judge["baseline_evm_db"] + 1.    # EVM 1 dB worse than no predistorter at all
    result = arena._budget_result(250, {}, 250, dict(ops=500, mul=250, add=250, nonlinear={}, nonlinear_mul=0,
                                                   nonlinear_add=0, items=[]), rows, arena.SEEDS)
    assert result["qualified"] and result["reasons"] == []
    assert result["quality_conservative_db"] == pytest.approx(.5 * (-1. + 8.))
    assert result["metrics"].evm_improvement_db == -1.         # and the loss stays visible


@pytest.mark.parametrize('quality',[-4.,0.])
def test_no_quality_gain_cannot_buy_a_rank_with_a_cost_bonus(calibrated, quality):
    result=score(cases(calibrated,qualities=(quality,)*3))
    assert not result['eligible'] and result['score'] is None
    assert result['rankings']['budget-2000']['score'] is None



def test_emitted_adjacent_energy_is_not_erased_by_subtracting_a_dirty_reference():
    from opendpd.core.arena_metrics import quality_db
    before=dict(evm_db=-60.,aclr_l_db=10*math.log10(.02**2),aclr_r_db=-100.)
    reproduces_dirty_reference=dict(evm_db=-60.,aclr_l_db=-20.,aclr_r_db=-100.)
    assert quality_db(reproduces_dirty_reference,before)==pytest.approx(-6.98970004)



@pytest.mark.parametrize("power_error", [-6., 0.501, -0.501])
def test_power_backoff_or_boost_cannot_buy_a_rank(calibrated, power_error):
    rows = cases(calibrated, qualities=(30., 30., 30.))
    for case in rows:
        case["judges"][0]["power_error_db"] = power_error
    result = score(rows)
    assert not result["eligible"] and result["score"] is None


@pytest.mark.parametrize("mutation", [
    lambda r: r.pop(),
    lambda r: r.append(copy.deepcopy(r[0])),
    lambda r: r[0].update(seed=True),
    lambda r: r[0].update(budget=True),
    lambda r: r[0].update(budget=125),
    lambda r: r[0].update(parameters=r[0]["parameters"] + 1),
    lambda r: r[0].pop("parameters"),
    lambda r: [r.remove(case) for case in list(r) if case["budget"] == 2000],
    lambda r: r[0].update(condition_id="unregistered"),
    lambda r: r[0].update(data_sha256="9" * 64),
    lambda r: r[0].update(teacher_sha256="9" * 64),
    lambda r: r[0]["judges"].pop(),
    lambda r: r[0]["judges"].append(copy.deepcopy(r[0]["judges"][0])),
    lambda r: r[0]["judges"][0].update(judge_id="preferred-surrogate"),
    lambda r: r[0]["judges"][0].update(checkpoint_sha256="9" * 64),
])
def test_missing_duplicate_or_changed_evidence_fails_closed(calibrated, mutation):
    rows = cases(calibrated)
    mutation(rows)
    with pytest.raises(ValueError):
        score(rows)


@pytest.mark.parametrize("mutation", [
    lambda points: points.pop(),
    lambda points: points.reverse(),
    lambda points: points[0].update(budget=300),
    lambda points: points[1]["model_parameters"].update(hidden_size=9),      # a cheaper model than registered
    lambda points: points[1].update(model_parameters=dict(points[0]["model_parameters"])),   # belongs to the 250 class
    lambda points: points[2].update(model_parameters="gru"),
])
def test_a_result_cannot_choose_its_own_sweep(calibrated, mutation):
    points = arena.sweep("gru")
    mutation(points)
    with pytest.raises(ValueError):
        arena.summarize_cases("gru", "apa-200mhz-b", cases(calibrated), points)


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf"), True, None])
def test_gain_must_be_a_finite_positive_number(calibrated, value):
    rows = cases(calibrated)
    rows[0]["reference_gain"] = value
    with pytest.raises(ValueError):
        score(rows)


def test_gain_cannot_change_between_seeds_or_budgets(calibrated):
    rows = cases(calibrated)
    rows[-1]["reference_gain"] = 1.3
    with pytest.raises(ValueError, match="fixed across seeds"):
        score(rows)


@pytest.mark.parametrize("field,value", [
    ("attained_epochs", 149), ("attained_epochs", None), ("attained_epochs", True),
    ("optimizer_updates", 600), ("optimizer_updates", 4799), ("optimizer_updates", 4801),
    ("selected_epoch", None), ("selected_epoch", 0), ("selected_epoch", 241),
    ("selected_epoch", 1.5), ("selected_epoch", True),
    ("frame_draw_sha256", None), ("frame_draw_sha256", "d" * 63), ("frame_draw_sha256", "z" * 64),
])
def test_incomplete_or_unverifiable_training_budget_cannot_rank(calibrated, field, value):
    rows = cases(calibrated)
    rows[0][field] = value
    with pytest.raises(ValueError):
        score(rows)


@pytest.mark.parametrize("field,value", [("attained_epochs", 150), ("optimizer_updates", 4800),
                                         ("selected_epoch", 5)])
def test_deterministic_fit_cannot_claim_training_curves(calibrated, field, value):
    rows = cases(calibrated, backbone="mp_ls")
    rows[0][field] = value
    with pytest.raises(ValueError, match="Deterministic"):
        score(rows, backbone="mp_ls")


@pytest.mark.parametrize("field", ["nmse_db", "aclr_l_db", "baseline_nmse_db", "ib_error_db", "power_error_db",
                                   "evm_db", "baseline_evm_db", "baseline_ib_error_db",
                                   "aer_l_db", "aer_r_db", "baseline_aer_l_db", "baseline_aer_r_db",
                                   "reference_aclr_l_db", "reference_aclr_r_db"])
def test_nonfinite_metrics_never_become_scores(calibrated, field):
    rows = cases(calibrated)
    rows[0]["judges"][0][field] = float("nan")
    with pytest.raises(ValueError):
        score(rows)


def test_deterministic_fit_uses_one_seed_without_invented_uncertainty(calibrated):
    result = score(cases(calibrated, qualities=(4.,), backbone="mp_ls"), backbone="mp_ls")
    assert all(point["quality_std_db"] == 0. and point["quality_conservative_db"] == 4. for point in result["budgets"])
    assert result["seeds"] == [0] and result["completed_cases"] == 4
    # Complex coefficients: two real parameters and four real multiplications each.
    assert all(1.9 < point["ops"] / (2 * point["parameters"]) < 2.1 for point in result["budgets"])
    with pytest.raises(ValueError):
        arena.summarize_cases("mp_ls", "apa-200mhz-b", cases(calibrated), arena.sweep("mp_ls"))


def test_frozen_pa_checkpoint_changes_invalidate_both_protocol_and_cases(calibrated):
    rows=cases(calibrated,'apa-200mhz-b')
    before=arena.protocol()
    calibrated['apa-200mhz-b']['teacher']['sha256']='e'*64
    after=arena.protocol()
    assert after.training_sha256!=before.training_sha256 and after.protocol_sha256!=before.protocol_sha256
    with pytest.raises(ValueError,match='teacher hash'):
        score(rows,'apa-200mhz-b')



EXPECTED_HIDDEN = {      # largest registered size within 250 / 500 / 1,000 / 2,000 real parameters
    "gru": (7, 10, 16, 23), "tres_gru": (5, 9, 15, 22), "tres_deltagru": (5, 9, 15, 22),
    "lstm": (5, 9, 13, 20), "vdlstm": (4, 7, 12, 18), "dgru": (5, 8, 12, 19), "tcn": (8, 17, 34, 68),
    "rvtdcnn": (5, 12, 24, 50), "pgjanet": (5, 7, 11, 16), "dvrjanet": (5, 7, 11, 16),
    "deltagru": (5, 9, 14, 21), "deltajanet": (7, 11, 18, 27), "qgru": (6, 9, 15, 22),
    "qgru_amp1": (6, 9, 15, 22), "bojanet": (1, 7, 14, 18), "apnrru": (None, 2, 9, 23),
    "mcldnn": (None, None, 2, 7), "gru_stream": (7, 10, 16, 23),
}


def test_sweep_presets_are_the_largest_size_inside_each_budget_class():
    for key, sizes in EXPECTED_HIDDEN.items():
        points = arena.sweep(key)
        assert tuple(p["model_parameters"] and p["model_parameters"]["hidden_size"] for p in points) == sizes, key
    for model in arena.bundled_backbones():
        for point in arena.sweep(model.key):
            if point["model_parameters"] is not None:
                count = arena_ops.parameter_count(model.key, point["model_parameters"])
                assert point["budget"] / 2 < count <= point["budget"] or point["budget"] == arena.BUDGETS[0], model.key
    # The gradient GMP is a fixed 495-parameter model: it exists in one class only.
    assert [p["model_parameters"] is not None for p in arena.sweep("gmp")] == [False, True, False, False]
    assert arena.sweep("gmp_stream") == arena.sweep("gmp")


def test_polynomial_families_fill_their_budget_and_fit_the_inference_halo():
    for budget in arena.BUDGETS:
        mp, gmp = (arena.model_parameters(key, budget) for key in ("mp_ls", "gmp_ls"))
        assert mp["Q"] - 1 <= arena.TRAINING["inference_center_start"] and mp["rcond"] == 1e-4
        with pytest.raises(ValueError, match="excluded"):
            arena.model_parameters("ilc_dpd", budget)
        assert max(gmp["La"], gmp["Lb"] + gmp["Mb"], gmp["Lc"]) <= arena.TRAINING["inference_center_start"]
        assert arena_ops.parameter_count("gmp_ls", gmp) == budget
    assert arena.model_parameters("mp_ls", 1000) == {"K": 10, "Q": 50, "rcond": 1e-4}
    with pytest.raises(ValueError):
        arena.model_parameters("gru", 1200)


def test_template_widths_scale_to_each_budget_and_fixed_widths_stay():
    from opendpd.core.backbone_template import parse_definition
    widths = []
    for point in arena.sweep("user_template"):
        nodes = {node["id"]: node for node in parse_definition(point["model_parameters"]["definition"])["nodes"]}
        widths.append(nodes["memory"]["features"])
        assert nodes["project"]["features"] == 2          # tied to the I/Q residual
    assert widths == [7, 10, 16, 23]
    definition = parse_definition(arena.model_parameters("user_template", 1000)["definition"])
    submitted = {"definition": arena_ops.__dict__ and __import__("json").dumps(definition)}
    # A design already inside a budget class is evaluated exactly as submitted there.
    assert parse_definition(arena.model_parameters("user_template", 1000, submitted)["definition"]) == definition
    assert arena.model_parameters("user_template", 250, submitted) == arena.model_parameters("user_template", 250)


def test_training_budget_and_arena_precision_labels_are_explicit():
    training = arena.TRAINING
    assert training["epochs"] == 240 and training["frames_per_epoch"] == "all training windows"
    assert training["quality_metric"] == "evm-aclr-v3"
    assert "minimum_quality_db" not in training and "regression_tolerance_db" not in training
    assert arena.SCORING["scoring_version"] == "configuration-ops-v3" and arena.SCORING["budgets"] == [250, 500, 1000, 2000]
    assert not any("latency" in key or "timing" in key for key in (*training, *arena.SCORING))
    assert arena.training_budget(25307) == dict(frames_per_epoch=25108, optimizer_updates=94320, frame_exposures=6025920)
    assert arena.training_budget(294912)["optimizer_updates"] == 1105200
    labels = {item.key: item.display_name for item in arena.bundled_backbones()}
    assert labels["qgru"] == "QGRU (FP32)" and labels["qgru_amp1"] == "QGRU amp1 (FP32)"
    assert all("thresholds 0" in labels[key] for key in ("deltagru", "deltajanet", "tres_deltagru"))
    assert labels["user_template"] == "Template GRU (bundled example)"


def test_every_condition_has_complete_declared_test_symbols():
    from opendpd.core.arena_metrics import symbol_windows
    for identifier,record in arena.active_calibration().items():
        grid=record['evm_grid']
        spec=record['stimuli']['splits']['test']
        start=record['split']['boundaries']['test'][0]+spec['metric_start']
        stop=record['split']['boundaries']['test'][0]+spec['metric_stop']
        if grid.get('carriers'):
            for carrier in grid['carriers']:
                assert len(carrier['fft_starts']) == 1
                assert all(start <= pos and pos+grid['useful_samples'] <= stop for pos in carrier['fft_starts'])
        else:
            windows=symbol_windows(grid,start,stop-start)
            assert len(windows)=={'apa-200mhz-b':4,'apa-200mhz-b':1}[identifier]
            assert all(size==grid['useful_samples'] for _,size in windows)
            assert grid['occupied_bins'] and all(0 <= b < grid['useful_samples'] for b in grid['occupied_bins'])



def test_public_arena_core_import_does_not_load_torch():
    completed = subprocess.run([sys.executable, "-c",
        "import sys; import opendpd.core.arena as a; a.protocol(); a.sweep('tcn'); assert 'torch' not in sys.modules"],
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_protocol_binds_metric_training_and_backbone_sources(calibrated, monkeypatch):
    manifest = arena.source_manifest()
    required = {"opendpd/core/arena_engine.py", "opendpd/core/arena_runner.py", "opendpd/core/arena_ops.py",
                "opendpd/core/arena_metrics.py",
                "opendpd/core/arena_cuda.py", "opendpd/core/metrics/spectral_v2.py",
                "opendpd/core/metrics/general_v1.py", "opendpd/core/template_network.py",
                "opendpd/core/backbone_template.py", "opendpd/core/polynomial.py",
                "opendpd/core/ilc.py", "opendpd/core/virtual_pa_kernel.py",
                "backbones/gru.py", "backbones/finite_iq.py", "models.py"}
    assert required <= manifest.keys()
    before = arena.protocol().protocol_sha256
    changed = dict(manifest)
    changed["opendpd/core/arena_metrics.py"] = "0" * 64
    monkeypatch.setattr(arena, "source_manifest", lambda: changed)
    assert arena.protocol().protocol_sha256 != before


def test_scoring_sources_change_the_protocol_but_never_a_checkpoint_binding(monkeypatch):
    before = arena.protocol()
    real = arena.file_hash
    monkeypatch.setattr(arena, "file_hash", lambda path: "1" * 64 if str(path).endswith("arena_ops.py") else real(path))
    rescored = arena.protocol()
    assert rescored.protocol_sha256 != before.protocol_sha256 and rescored.training_sha256 == before.training_sha256
    monkeypatch.setattr(arena, "file_hash", lambda path: "2" * 64 if str(path).endswith("arena_engine.py") else real(path))
    retrained = arena.protocol()
    assert retrained.training_sha256 != before.training_sha256
    assert set(arena.TRAINING_SOURCE_FILES).isdisjoint(arena.SCORING_SOURCE_FILES)
