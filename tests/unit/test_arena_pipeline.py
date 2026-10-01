"""Short CPU checks of actual Arena boundaries, selection, sweep orchestration and provenance paths."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from opendpd.core import arena, arena_engine as engine, arena_ops, arena_runner as runner
from opendpd.schemas.arena import ArenaRow
from opendpd.services.workspace import read_json, write_json_atomic
from tests.unit.test_arena_scoring import adjusted, observations


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_runner_is_orchestration_only_and_reexports_the_engine_primitives():
    moved = ("ArenaCascade", "Condition", "_load_weights", "_save_weights", "build_model",
             "calibrate_linear_baseline", "dpd_output", "fit_case", "fit_polynomial", "judge_case", "limit_array",
             "limit_tensor", "load_frozen", "offline_output", "pa_output", "parameter_count", "signal_metrics",
             "train_gradient")
    for name in moved:
        assert getattr(runner, name) is getattr(engine, name), name
        assert getattr(engine, name).__module__ == "opendpd.core.arena_engine", name
    for name in ("evaluate_request", "prefetch", "cache_folder", "default_device", "runtime_environment"):
        assert getattr(runner, name).__module__ == "opendpd.core.arena_runner" and not hasattr(engine, name)
    # Arena v2 times no host.
    for module in (runner, engine):
        assert not any("timing" in name or "latency" in name for name in vars(module))
    # The engine fixes weights and observations; orchestration and scoring never invalidate a checkpoint.
    assert "opendpd/core/arena_engine.py" in arena.TRAINING_SOURCE_FILES
    assert "opendpd/core/arena_runner.py" in arena.SCORING_SOURCE_FILES


@pytest.mark.parametrize("length", [1, 99, 100, 101, 203, 6401])
def test_overlap_save_identity_preserves_every_valid_sample(length):
    x = np.arange(length * 2, dtype=np.float32).reshape(length, 2)
    actual = engine.offline_output(nn.Identity(), x, batch_size=3)
    np.testing.assert_array_equal(actual, x)


@pytest.mark.parametrize("shift", [49, -30])
def test_retained_halo_supplies_past_and_future_context_across_blocks(shift):
    class Shift(nn.Module):
        def forward(self, x):
            y = torch.zeros_like(x)
            if shift > 0:
                y[:, shift:] = x[:, :-shift]
            else:
                y[:, :shift] = x[:, -shift:]
            return y

    x = np.random.default_rng(4).normal(size=(317, 2)).astype(np.float32)
    expected = np.zeros_like(x)
    if shift > 0:
        expected[shift:] = x[:-shift]
    else:
        expected[:shift] = x[-shift:]
    np.testing.assert_array_equal(engine.offline_output(Shift(), x, batch_size=2), expected)


def test_training_and_evaluation_limiters_agree_and_have_finite_zero_gradients():
    x = np.array([[0., 0.], [.1, -.2], [3., 4.], [-5., 12.]], dtype=np.float32)
    tensor = torch.tensor(x, requires_grad=True)
    limited = engine.limit_tensor(tensor, .5)
    np.testing.assert_allclose(limited.detach().numpy(), engine.limit_array(x, .5), rtol=1e-6)
    assert torch.all(torch.linalg.vector_norm(limited, dim=-1) <= .5 + 1e-7)
    limited.square().sum().backward()
    assert torch.isfinite(tensor.grad).all()


def test_baseline_complex_gain_uses_only_the_declared_training_probe():
    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.lengths = []

        def forward(self, x):
            self.lengths.append(x.shape[1])
            c, s = np.cos(.2), np.sin(.2)
            return 2 * torch.stack((c * x[..., 0] - s * x[..., 1],
                                    s * x[..., 0] + c * x[..., 1]), dim=-1)

    teacher = Teacher()
    x = np.random.default_rng(2).normal(size=(5000, 2)).astype(np.float32) * .1
    x[4096:] = 50.  # Outside the declared calibration probe.
    alpha = engine.calibrate_linear_baseline(teacher, x, 1.3, 100., "cpu")
    assert alpha == pytest.approx(.65 * np.exp(-.2j), abs=5e-6)
    assert set(teacher.lengths) == {4096}


def test_baseline_is_shared_across_judges_cached_and_uses_fixed_reference(monkeypatch):
    condition = engine.Condition.__new__(engine.Condition)
    condition.data = {"x_test": np.arange(12, dtype=np.float32).reshape(6, 2) / 100}
    condition.alpha, condition.peak, condition.gain = .8 + .1j, 1., 1.2
    condition.signal, condition.grid, condition.starts = object(), {"grid": 1}, {"test": 640}
    condition._baselines = None
    condition.manifest = {"stimuli": {"splits": {"test": {"metric_start": 0, "metric_stop": 6}}}}
    observed = []

    def outputs(iq):
        observed.append(iq.copy())
        yield "pa", "a" * 64, iq

    references = []

    def metrics(y, reference, signal, grid, start, window=None):
        assert (signal, grid, start) == (condition.signal, {"grid": 1}, 640)  # the test split on the capture's grid
        references.append(reference.copy())
        return {"nmse_db": float(y.sum())}

    condition.outputs = outputs
    monkeypatch.setattr(engine, "signal_metrics", metrics)
    first = condition.baseline_metrics()
    assert condition.baseline_metrics() is first
    assert set(first) == {"pa"} and len(observed) == 1
    expected = engine.limit_array(engine._complex_scale(condition.data["x_test"], condition.alpha), 1.)
    np.testing.assert_array_equal(observed[0], expected)
    for reference in references:
        np.testing.assert_array_equal(reference, 1.2 * condition.data["x_test"])


class Gain(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(value)))

    def forward(self, x):
        return self.weight * x


@pytest.mark.parametrize("quality_power,selected,feasible", [
    ([(8., .6), (3.5, 0.), (9., .7)], 2, True),      # only the second checkpoint keeps the output power
    ([(1., .6), (2., .7), (1.5, .8)], 2, False),     # none does: the best diagnostic checkpoint is kept
    ([(4., 0.), (5., 0.), (5., 0.)], 2, True),       # a tie keeps the earlier checkpoint
    ([(-1., 0.), (.5, 0.), (.2, 0.)], 2, True),      # no quality threshold: small gains still compete
])
def test_teacher_only_selection_feasibility_fallback_and_frame_budget(
        tmp_path, monkeypatch, quality_power, selected, feasible):
    settings = {**arena.TRAINING, "epochs": 3,
                "batch_size": 2, "validation_every_epochs": 1}
    monkeypatch.setattr(arena, "TRAINING", settings)
    model, teacher = Gain(.1), Gain(2.)
    monkeypatch.setattr(engine, "build_model", lambda *args: model)
    x = np.random.default_rng(8).normal(size=(260, 2)).astype(np.float32) * .1

    class NoTestAccess(dict):
        def __getitem__(self, key):
            assert key != "x_test", "test data entered checkpoint selection"
            return super().__getitem__(key)

    condition = SimpleNamespace(device="cpu", teacher=teacher, peak=10., gain=1.2,
        data=NoTestAccess(x_train=x, x_val=x[:201]), alpha=1. + 0.j,
        signal=object(), grid={"grid": 1}, starts={"val": 300}, identifier="unit-test",
        manifest={"stimuli": {"splits": {"val": {"metric_start": 0, "metric_stop": 201}}}})
    condition.metrics = lambda y, split: engine.Condition.metrics(condition, y, split)
    snapshots, calls = [], []

    def metrics(c, y):
        reference = c.gain * c.data["x_val"]
        calls.append(reference.copy())
        snapshots.append(model.weight.detach().clone())
        q, power = quality_power[len(calls)-1]
        return dict(nmse_db=-20.-q, objective_db=-q, power_error_db=power)

    monkeypatch.setattr(engine, "validation_objective", metrics)
    returned, info = engine.train_gradient(condition, "gru", {}, 2, tmp_path, lambda *args: None)
    assert info["selected_epoch"] == selected
    assert info["selected_validation_feasible"] is feasible
    assert info["attained_epochs"] == 3 and info["optimizer_updates"] == 93
    assert info["frames_per_epoch"] == 61 and info["frame_exposures"] == 183
    assert not any("timing" in key or "latency" in key for key in info)
    rng = np.random.default_rng(2)
    expected_draws = b"".join(rng.permutation(len(x) - 199).astype("<i8").tobytes() for _ in range(3))
    assert info["frame_draw_sha256"] == hashlib.sha256(expected_draws).hexdigest()
    assert torch.equal(returned.weight, snapshots[selected - 1])
    assert teacher.weight.item() == 2. and teacher.weight.grad is None
    assert len(calls) == 3  # Validation NMSE only; no test access or incomplete-symbol EVM.
    for reference in calls:
        np.testing.assert_array_equal(reference, 1.2 * condition.data["x_val"])


def test_stale_request_is_rejected_before_assets_or_training(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "Condition", lambda *args: pytest.fail("stale request reached assets"))
    with pytest.raises(ValueError, match="stale"):
        runner.evaluate_request({"protocol_sha256": "0" * 64}, tmp_path / "result.json")


@pytest.mark.parametrize("key,supplied", [
    ("gru", {"hidden_size": 3}), ("gru", {"definition": "{}"}), ("mp_ls", {"K": 1, "Q": 1}),
    ("user_template", {"hidden_size": 3}), ("user_template", {"definition": "{}", "hidden_size": 3})])
def test_request_cannot_supply_its_own_model_size(tmp_path, monkeypatch, key, supplied):
    monkeypatch.setattr(runner, "Condition", lambda *args: pytest.fail("an unregistered size reached assets"))
    request = {"protocol_sha256": arena.protocol().protocol_sha256, "board_id": "apa-200mhz-b",
               "backbone": key, "model_parameters": supplied}
    with pytest.raises(ValueError, match="server-verified template"):
        runner.evaluate_request(request, tmp_path / "result.json")
    assert not (tmp_path / "result.json").exists()


# --- the parameter sweep ------------------------------------------------------------------------------

def test_cache_identity_separates_every_budget_and_is_shared_with_the_streaming_variant(tmp_path, monkeypatch):
    points = arena.sweep("gru")
    folders = {runner.cache_folder(tmp_path, "gru", point["model_parameters"], condition, seed)
               for point in points for condition in ("dpa-200mhz", "apa-200mhz") for seed in arena.SEEDS}
    assert len(folders) == 4 * 2 * 3
    parameters = points[1]["model_parameters"]
    folder = runner.cache_folder(tmp_path, "gru", parameters, "dpa-200mhz", 1)
    identity = dict(key="gru", parameters=parameters, training=arena.training_fingerprint())
    assert folder == tmp_path / arena.canonical_hash(identity) / "dpa-200mhz" / "seed-1"
    assert arena.sweep("gru_stream") == points          # The streaming entry reloads exactly these checkpoints.
    assert engine.cache_binding("gru", parameters, "dpa-200mhz", 1) == dict(
        training_sha256=arena.training_fingerprint(), backbone="gru", model_parameters=parameters,
        condition_id="dpa-200mhz", seed=1)
    # Rescoring keeps every checkpoint; a change of the training engine does not.
    real = arena.file_hash
    monkeypatch.setattr(arena, "file_hash", lambda path: "1" * 64 if str(path).endswith("arena_ops.py") else real(path))
    assert runner.cache_folder(tmp_path, "gru", parameters, "dpa-200mhz", 1) == folder
    monkeypatch.setattr(arena, "file_hash", lambda path: "2" * 64 if str(path).endswith("arena_engine.py") else real(path))
    assert runner.cache_folder(tmp_path, "gru", parameters, "dpa-200mhz", 1) != folder


def test_cached_checkpoint_is_reloaded_only_for_its_own_configuration_condition_seed_and_sources(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    x = (rng.normal(size=(600, 2)) * .2).astype(np.float32)
    y = (1.5 * x - .3 * x * np.sum(x * x, axis=1, keepdims=True)).astype(np.float32)
    condition = SimpleNamespace(identifier="dpa-200mhz", device="cpu", teacher=Gain(1.5), gain=1.5, data={"x_train": x, "y_train": y})
    small, large = {"K": 3, "Q": 2, "rcond": 1e-4}, {"K": 3, "Q": 4, "rcond": 1e-4}
    folder = runner.cache_folder(tmp_path, "mp_ls", small, condition.identifier, 0)
    model, info = engine.fit_case(condition, "mp_ls", small, 0, folder, lambda *args: None)
    assert engine.parameter_count(model) == arena_ops.parameter_count("mp_ls", small) == 12
    assert info["cache_binding"] == engine.cache_binding("mp_ls", small, "dpa-200mhz", 0)
    assert info["weights_sha256"] == arena.file_hash(folder / "weights.npz")
    assert (info["attained_epochs"], info["optimizer_updates"], info["selected_epoch"]) == (0, 0, None)
    assert read_json(folder / "training.json") == info

    monkeypatch.setattr(engine, "fit_polynomial", lambda *args: pytest.fail("a verified checkpoint was fitted again"))
    reloaded, again = engine.fit_case(condition, "mp_ls", small, 0, folder, lambda *args: None)
    assert again == info and torch.equal(reloaded.coefficients, model.coefficients)
    elsewhere = SimpleNamespace(**{**vars(condition), "identifier": "apa-200mhz"})
    for other, parameters, seed in ((condition, large, 0), (condition, small, 1), (elsewhere, small, 0)):
        with pytest.raises(ValueError, match="integrity"):
            engine.fit_case(other, "mp_ls", parameters, seed, folder, lambda *args: None)
    with monkeypatch.context() as patch:
        patch.setattr(arena, "training_fingerprint", lambda: "0" * 64)
        with pytest.raises(ValueError, match="integrity"):
            engine.fit_case(condition, "mp_ls", small, 0, folder, lambda *args: None)
    engine._save_weights(engine.build_model("mp_ls", small), folder / "weights.npz")     # Replaced coefficients.
    with pytest.raises(ValueError, match="integrity"):
        engine.fit_case(condition, "mp_ls", small, 0, folder, lambda *args: None)


def stage(monkeypatch):
    """The real orchestration around instant fits and frozen judge observations of a declared quality."""
    records = arena.calibration()
    log = SimpleNamespace(conditions=[], fits=[], trained=[], events=[], quality=6., power=0., fail_after=None)

    class Condition:
        def __init__(self, identifier, device):
            self.identifier, self.device = identifier, device
            log.conditions.append((identifier, device))

    class EvaluationCondition:
        def __init__(self, identifier, device):
            self.identifier, self.device = identifier, device
            log.events.append("load-test")

    def fit_case(condition, key, parameters, seed, folder, emit):
        if log.fail_after is not None and len(log.fits) >= log.fail_after:
            raise FloatingPointError("Nonfinite training loss at epoch 7")
        log.fits.append((key, condition.identifier, parameters, seed, folder))
        assert isinstance(condition, Condition)
        log.events.append("fit")
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / "weights.npz").exists():       # As the engine does, a cached checkpoint is reloaded.
            log.trained.append(folder)
            (folder / "weights.npz").write_bytes(json.dumps([arena._base(key), parameters, seed]).encode())
        deterministic = arena._base(key) in arena.DETERMINISTIC
        emit("fitting" if deterministic else "training", 0 if deterministic else 5, "engine message")
        info = (dict(selected_epoch=None, attained_epochs=0, optimizer_updates=0) if deterministic else
                dict(selected_epoch=arena.TRAINING["epochs"], attained_epochs=arena.TRAINING["epochs"],
                     optimizer_updates=arena.training_budget(records[condition.identifier]["counts"]["train"])["optimizer_updates"],
                     frame_draw_sha256="d" * 64))
        model = nn.Identity()
        model.n_real_parameters = arena_ops.parameter_count(key, parameters)
        return model, dict(info, training_device="cpu")

    def judge_case(condition, model, key):
        assert isinstance(condition, EvaluationCondition)
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        log.events.append("judge")
        record = records[condition.identifier]
        return dict(judges=[dict(observations(name, sha, log.quality), power_error_db=log.power)
                            for name, sha in arena.judge_hashes(record).items()],
                    teacher_sha256=record["teacher"]["sha256"], data_sha256=record["data_sha256"], reference_gain=record["reference_gain"])

    monkeypatch.setattr(runner, "TrainingCondition", Condition)
    monkeypatch.setattr(runner, "Condition", EvaluationCondition)
    monkeypatch.setattr(runner, "fit_case", fit_case)
    monkeypatch.setattr(runner, "judge_case", judge_case)
    return log


@pytest.fixture
def staged(monkeypatch):
    return stage(monkeypatch)


def request_for(key, board_id, **updates):
    return dict(board_id=board_id, backbone=key, protocol_sha256=arena.protocol().protocol_sha256,
                model_parameters={}, model_provenance={}, **updates)


def test_evaluation_covers_every_available_budget_condition_and_seed_exactly_once(tmp_path, staged):
    output, cache = tmp_path / "job" / "result.json", tmp_path / "cache"
    output.parent.mkdir()
    row = runner.evaluate_request(request_for("mcldnn", "apa-200mhz-b"), output, cache, device="cpu")
    assert row.status == "succeeded" and row.error is None
    assert [(point.budget, point.available) for point in row.budgets] == [
        (250, False), (500, False), (1000, True), (2000, True)]
    assert row.available_budgets == row.qualified_budgets == 2 and row.expected_cases == row.completed_cases == 2 * 1 * 3
    configurations = {point["budget"]: point["model_parameters"] for point in arena.sweep("mcldnn")}
    expected = {(budget, condition, seed) for budget in (1000, 2000)
                for condition in ("apa-200mhz-b",) for seed in arena.SEEDS}
    assert {(case["budget"], case["condition_id"], case["seed"]) for case in row.cases} == expected and len(row.cases) == 6
    for case in row.cases:
        assert case["parameters"] == arena_ops.parameter_count("mcldnn", configurations[case["budget"]])
        folder = runner.cache_folder(cache, "mcldnn", configurations[case["budget"]], case["condition_id"], case["seed"])
        assert case["checkpoint_sha256"] == arena.file_hash(folder / "weights.npz")
    # Assets are verified once per condition; every sweep unit has a checkpoint folder of its own.
    assert staged.conditions == [("apa-200mhz-b", "cpu")]
    assert staged.events == ["fit"] * 6 + ["load-test"] + ["judge"] * 6
    assert all(case["evaluation_split"] == "test" for case in row.cases)
    assert len({fit[-1] for fit in staged.fits}) == 6
    assert {fit[-1] for fit in staged.fits} == {runner.cache_folder(cache, "mcldnn", configurations[budget], condition, seed)
                                                 for budget, condition, seed in expected}
    # Scores come from the production aggregation of those cases and the analytic cost.
    assert row.score == pytest.approx(max(adjusted("mcldnn", budget, 6.) for budget in (1000, 2000)))
    assert all(point.ops == arena_ops.count("mcldnn", point.model_parameters)["ops"]
               for point in row.budgets if point.available)
    assert row.rank is None and all(entry.rank is None for entry in row.rankings.values())
    assert row.execution_semantics == "offline_overlap_200_100" and row.evidence_type == "measured_data_simulation"
    assert read_json(output) == row.model_dump(mode="json")
    assert read_json(output.parent / "completed-cases.json") == row.cases
    progress = read_json(output.with_suffix(".progress.json"))
    assert (progress["phase"], progress["completed_cases"], progress["expected_cases"]) == ("complete", 6, 6)


def test_result_provenance_binds_training_sources_and_never_a_host_timing(tmp_path, staged):
    current = arena.protocol()
    row = runner.evaluate_request(request_for("gru", "apa-200mhz-b", ), tmp_path / "result.json", device="cpu")
    assert row.provenance["protocol_id"] == "dpd-arena-v6-apa-b"
    assert row.provenance["training_sha256"] == current.training_sha256 == arena.training_fingerprint()
    assert row.provenance["model_parameters"] == {}          # The request's definition; presets are in ``budgets``.
    assert [point.model_parameters for point in row.budgets] == [point["model_parameters"] for point in arena.sweep("gru")]
    assert set(row.provenance["environment"]) == {"processor", "architecture", "os", "torch", "numpy", "device", "precision"}
    assert row.provenance["environment"]["device"] == "cpu"
    assert all(case["evaluation_device"] == "cpu" for case in row.cases)
    stored = json.dumps(row.model_dump(mode="json"))
    assert not any(word in stored for word in ("latency", "timing", "anchor_ns", "macs_per_sample"))
    # Without --cache the checkpoints stay beside the result.
    assert all(fit[-1].is_relative_to(tmp_path / "evaluation") for fit in staged.fits)


def test_streaming_entry_is_judged_from_its_base_checkpoints_in_a_cohort_of_its_own(tmp_path, staged):
    cache = tmp_path / "cache"
    for name in ("base", "stream"):
        (tmp_path / name).mkdir()          # The command line creates the job folder; only a default cache is implicit.
    base = runner.evaluate_request(request_for("gru", "apa-200mhz-b"), tmp_path / "base" / "result.json", cache, device="cpu")
    fitted = [fit[-1] for fit in staged.fits]
    stream = runner.evaluate_request(request_for("gru_stream", "apa-200mhz-b"), tmp_path / "stream" / "result.json",
                                     cache, device="cpu")
    assert [fit[-1] for fit in staged.fits[len(fitted):]] == fitted == staged.trained      # Nothing is trained twice.
    assert (base.execution_semantics, stream.execution_semantics) == ("offline_overlap_200_100", "streaming_stateful")
    assert [point.ops for point in stream.budgets] == [point.ops for point in base.budgets]
    assert [case["checkpoint_sha256"] for case in stream.cases] == [case["checkpoint_sha256"] for case in base.cases]


def test_deterministic_fit_runs_one_seed_per_budget_and_template_sweeps_its_accepted_definition(tmp_path, staged):
    from opendpd.core.backbone_template import DEFAULT_DEFINITION
    fitted = runner.evaluate_request(request_for("mp_ls", "apa-200mhz-b"), tmp_path / "mp" / "result.json", device="cpu")
    assert fitted.status == "succeeded" and fitted.seeds == [0] and fitted.expected_cases == len(fitted.cases) == 4
    assert [case["budget"] for case in fitted.cases] == arena.BUDGETS and {case["seed"] for case in fitted.cases} == {0}
    definition = json.loads(DEFAULT_DEFINITION)
    definition["nodes"][0]["features"] = 12
    supplied = {"definition": json.dumps(definition)}
    request = {**request_for("user_template", "apa-200mhz-b"), "model_parameters": supplied}
    row = runner.evaluate_request(request, tmp_path / "template" / "result.json", device="cpu")
    assert row.status == "succeeded", row.error
    assert row.provenance["model_parameters"] == supplied
    assert [point.model_parameters for point in row.budgets] == [
        point["model_parameters"] for point in arena.sweep("user_template", supplied)]
    widths = [json.loads(point.model_parameters["definition"])["nodes"][0]["features"] for point in row.budgets]
    assert widths == [7, 10, 12, 23]


def test_interrupted_training_never_loads_test_or_emits_partial_test_evidence(tmp_path, staged):
    staged.fail_after = 5
    output = tmp_path / "job" / "result.json"
    output.parent.mkdir()
    row = runner.evaluate_request(request_for("gru", "apa-200mhz-b"), output, device="cpu")
    assert row.status == "failed" and row.error == "FloatingPointError: Nonfinite training loss at epoch 7"
    assert not row.eligible and row.score is None and row.rankings == {} and row.metrics is None
    assert (row.completed_cases, row.expected_cases, row.available_budgets) == (0, 4 * 1 * 3, 4)
    assert [(point.budget, point.available, point.qualified) for point in row.budgets] == [
        (budget, True, False) for budget in arena.BUDGETS]
    assert row.provenance["training_sha256"] == arena.protocol().training_sha256
    assert not (output.parent / "completed-cases.json").exists()
    assert staged.events == ["fit"] * 5
    progress = read_json(output.with_suffix(".progress.json"))
    assert progress["phase"] == "complete" and progress["message"] == row.error and progress["completed_cases"] == 0


def test_sweep_outside_the_power_gate_is_a_completed_evaluation_without_a_rank(tmp_path, staged):
    staged.quality, staged.power = 9., -.8
    row = runner.evaluate_request(request_for("gru", "apa-200mhz-b"), tmp_path / "result.json", device="cpu")
    assert row.status == "succeeded" and not row.eligible and row.score is None and row.qualified_budgets == 0
    assert all(entry.score is None for entry in row.rankings.values())
    assert all(point.available and not point.qualified and point.quality_conservative_db == 9. for point in row.budgets)
    assert row.eligibility_reasons == ["Output power falls outside the fixed-target ±0.5 dB envelope"]


def test_a_small_gain_is_ranked_for_what_it_is(tmp_path, staged):
    staged.quality = 1.
    row = runner.evaluate_request(request_for("gru", "apa-200mhz-b"), tmp_path / "result.json", device="cpu")
    assert row.status == "succeeded" and row.eligible and row.qualified_budgets == 4
    assert row.score == pytest.approx(max(adjusted("gru", budget, 1.) for budget in arena.BUDGETS))


def test_prefetch_trains_one_sweep_unit_and_skips_a_budget_without_a_configuration(tmp_path, staged):
    assert runner.prefetch("mcldnn", 250, "apa-200mhz-b", tmp_path) == 0
    assert staged.conditions == [] and staged.fits == []
    assert runner.prefetch("gru_stream", 500, "apa-200mhz-b", tmp_path, device="cpu") == 3
    parameters = arena.model_parameters("gru", 500)
    assert [(fit[0], fit[1], fit[2], fit[3]) for fit in staged.fits] == [
        ("gru_stream", "apa-200mhz-b", parameters, seed) for seed in arena.SEEDS]
    assert [fit[-1] for fit in staged.fits] == [runner.cache_folder(tmp_path, "gru", parameters, "apa-200mhz-b", seed)
                                                for seed in arena.SEEDS]
    with pytest.raises(ValueError, match="excluded"):
        runner.prefetch("ilc_dpd", 1000, "apa-200mhz-b", tmp_path, device="cpu")
    with pytest.raises(ValueError, match="budgets"):
        runner.prefetch("gru", 300, "apa-200mhz-b", tmp_path)


def test_official_test_phase_refuses_a_missing_checkpoint_without_training(tmp_path,staged):
    row=runner.evaluate_request(request_for('gru','apa-200mhz-b'),tmp_path/'result.json',
                                cache=tmp_path/'empty-cache',device='cpu',cached_only=True)
    assert row.status=='failed' and 'training is disabled' in row.error
    assert staged.fits==[] and row.cases==[]


def test_available_accelerator_is_the_default_and_an_explicit_device_wins(monkeypatch):
    for available in (False, True):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        accelerator = "cuda" if available else "cpu"
        assert [runner.default_device(key) for key in ("deltagru", "deltajanet")] == [accelerator] * 2
        assert [runner.default_device(key) for key in ("gru", "tres_deltagru", "mp_ls", "user_template")] == [accelerator] * 4
        assert runner.default_device("deltagru", "cuda") == "cuda" and runner.default_device("gru", "cpu") == "cpu"


def test_command_line_runs_either_one_prefetch_unit_or_one_request(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "prefetch", lambda *args, **kwargs: calls.append(("prefetch", args, kwargs)))
    monkeypatch.setattr(runner, "evaluate_request", lambda *args, **kwargs: calls.append(("evaluate", args, kwargs)))
    cache = tmp_path / "cache"
    monkeypatch.setattr("sys.argv", ["arena_runner", "--cache", str(cache), "--prefetch", "gru", "500", "apa-200mhz-b"])
    runner.main()
    request = tmp_path / "request.json"
    write_json_atomic(request, {"backbone": "gru"})
    output = tmp_path / "out" / "result.json"
    monkeypatch.setattr("sys.argv", ["arena_runner", "--request", str(request), "--output", str(output)])
    runner.main()
    assert calls == [("prefetch", ("gru", 500, "apa-200mhz-b", cache), {"device": None}),
                     ("evaluate", ({"backbone": "gru"}, output, None), {"device": None})]
    assert output.parent.is_dir()         # The command line creates the job folder.
    for arguments in (["--prefetch", "gru", "500", "apa-200mhz-b"], ["--request", str(request)], ["--timing"]):
        monkeypatch.setattr("sys.argv", ["arena_runner", *arguments])
        with pytest.raises(SystemExit):
            runner.main()
    assert len(calls) == 2


# --- publication of the official matrix ---------------------------------------------------------------

def test_publication_excludes_old_protocol_rows(tmp_path, monkeypatch):
    from benchmark.run_arena_baselines import publish

    protocol = arena.protocol()
    current = ArenaRow(entry_id="current", board_id="apa-200mhz-b", backbone="gru", display_name="GRU",
        origin="workspace", status="failed", protocol_sha256=protocol.protocol_sha256,
        evidence_type="measured_data_simulation", error="Deliberate fixture failure")
    old = current.model_copy(update={"entry_id": "old", "backbone": "lstm", "protocol_sha256": "0" * 64})
    for name, row in (("current", current), ("old", old)):
        folder = tmp_path / "jobs" / name
        folder.mkdir(parents=True)
        write_json_atomic(folder / "result.json", row)
    monkeypatch.setattr(arena, "ASSETS", tmp_path)
    rows = publish(tmp_path, protocol)
    assert len(rows) == 1 and rows[0]["backbone"] == "gru"
    assert rows[0]["origin"] == "official" and rows[0]["entry_id"] == "official-apa-200mhz-b-gru"
    assert arena.RESULTS_FILE == "reference-results-v6-apa-b.json" and not (tmp_path / "reference-results-v1.json").exists()
    payload = read_json(tmp_path / arena.RESULTS_FILE)
    assert payload["protocol_id"] == "dpd-arena-v6-apa-b" and payload["protocol_sha256"] == protocol.protocol_sha256
    assert payload["sha256"] == arena.canonical_hash({k: v for k, v in payload.items() if k != "sha256"})


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    """Worker results of two evaluated entries and one failure, produced once for this module."""
    root = tmp_path_factory.mktemp("evaluated")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: False)
        log = stage(patch)
        for key, board_id, quality in (("gru", "apa-200mhz-b", 6.), ("mcldnn", "apa-200mhz-b", 9.), ("lstm", "apa-200mhz-b", None)):
            log.quality, log.fail_after = quality, None if quality else len(log.fits)
            folder = root / "jobs" / f"{board_id}--{key}"
            folder.mkdir(parents=True)
            row = runner.evaluate_request(request_for(key, board_id), folder / "result.json", root / "cache", device="cpu")
            assert row.status == ("succeeded" if quality else "failed") and row.origin == "workspace"
    return root


@pytest.fixture
def matrix(tmp_path, monkeypatch, evaluated):
    """A private copy of those jobs, published into a private asset folder."""
    import shutil
    from benchmark.run_arena_baselines import publish
    protocol, calibration = arena.protocol(), arena.calibration()
    root = tmp_path / "matrix"
    shutil.copytree(evaluated / "jobs", root / "jobs")
    # Only the bundle is redirected: the protocol, its calibration and every hash stay the shipped ones.
    monkeypatch.setattr(arena, "ASSETS", tmp_path / "assets")
    monkeypatch.setattr(arena, "calibration", lambda: calibration)
    (tmp_path / "assets").mkdir()
    assert arena.protocol().protocol_sha256 == protocol.protocol_sha256 and arena.load_official_rows() == []
    return SimpleNamespace(root=root, protocol=protocol, bundle=tmp_path / "assets" / arena.RESULTS_FILE,
                           publish=lambda: publish(root, protocol))


def reseal(path, change):
    payload = read_json(path)
    payload.pop("sha256")
    change(payload)
    write_json_atomic(path, {**payload, "sha256": arena.canonical_hash(payload)})


def test_published_matrix_is_rescored_sealed_and_served_as_the_official_cohort(matrix, tmp_path):
    from opendpd.services.arena import ArenaController
    from opendpd.services.user_backbones import BackboneController
    from opendpd.services.workspace import NotFound, Workspace
    claimed = matrix.root / "jobs" / "apa-200mhz-b--gru" / "result.json"
    forged = read_json(claimed)
    honest = forged["score"]
    forged.update(score=999., rankings={key: {"score": 999., "rank": 1} for key in forged["rankings"]})
    write_json_atomic(claimed, forged)
    published = {row["backbone"]: row for row in matrix.publish()}
    assert set(published) == {"gru", "mcldnn", "lstm"}
    assert published["gru"]["score"] == pytest.approx(honest) and published["gru"]["rankings"]["overall"] == {
        "score": pytest.approx(honest), "rank": None}
    assert all(row["origin"] == "official" and row["entry_id"] == f"official-apa-200mhz-b-{key}"
               for key, row in published.items())
    loaded = {row.backbone: row for row in arena.load_official_rows()}
    assert {key: row.model_dump(mode="json") for key, row in loaded.items()} == published
    assert loaded["lstm"].status == "failed" and loaded["lstm"].score is None and loaded["lstm"].completed_cases == 0
    # A resealed bundle is still rescored from its raw cases.
    reseal(matrix.bundle, lambda payload: payload["rows"][0].update(score=999.))
    assert {row.backbone: row.score for row in arena.load_official_rows()}["gru"] == pytest.approx(honest)

    ws = Workspace.open_or_create(tmp_path / "workspace")
    board = ArenaController(ws, BackboneController(ws), protocol_provider=lambda: matrix.protocol).leaderboard("apa-200mhz-b")
    # MCLDNN linearizes best (9 dB against 6 dB), yet pays for its operations and its two missing budgets.
    assert [(row.backbone, row.origin, row.rank) for row in board.rows] == [
        ("gru", "official", 1), ("mcldnn", "official", 2), ("lstm", "official", None)]
    ranks = {row.backbone: {key: entry.rank for key, entry in row.rankings.items()} for row in board.rows}
    assert (ranks["mcldnn"]["linearization"], ranks["gru"]["linearization"]) == (1, 2)
    assert (ranks["mcldnn"]["budget-1000"], ranks["gru"]["budget-1000"]) == (2, 1)
    assert ranks["gru"]["arithmetic_efficiency"] == ranks["gru"]["budget-250"] == 1 and ranks["mcldnn"]["budget-250"] is None
    assert set(ranks["lstm"].values()) == {None}
    coverage = board.coverage
    assert (coverage.expected, coverage.evaluated, coverage.succeeded, coverage.failed) == (23, 3, 2, 1)
    assert len(coverage.missing) == 20 and "gru_stream" in coverage.missing
    with pytest.raises(NotFound, match="Unknown Arena leaderboard"):
        ArenaController(ws, BackboneController(ws)).leaderboard("dpa-200mhz")


def test_sealed_bundle_is_rescored_once_per_state_shared_by_workspaces_and_never_served_stale(matrix, tmp_path, monkeypatch):
    from opendpd.services.arena import ArenaController
    from opendpd.services.user_backbones import BackboneController
    from opendpd.services.workspace import NotFound, Workspace
    matrix.publish()
    loads, load = [], arena.load_official_rows
    monkeypatch.setattr(arena, "load_official_rows", lambda: loads.append(None) or load())

    def controller(name):
        ws = Workspace.open_or_create(tmp_path / name)
        return ArenaController(ws, BackboneController(ws), protocol_provider=lambda: matrix.protocol)

    first, second = controller("first"), controller("second")
    boards = [first.leaderboard("apa-200mhz-b"), first.leaderboard("apa-200mhz-b"), second.leaderboard("apa-200mhz-b")]
    assert len(loads) == 1 and boards[0] == boards[1] == boards[2]
    assert [(row.backbone, row.rank) for row in boards[0].rows] == [("gru", 1), ("mcldnn", 2), ("lstm", None)]
    # What every workspace shares is the verified evidence: a board's ranks never flow back into it.
    assert all(row.rank is None and all(entry.rank is None for entry in row.rankings.values())
               for row in first.official_rows())
    assert len(loads) == 1
    # A changed file is verified again, so an altered bundle is refused rather than answered from memory.
    reseal(matrix.bundle, lambda payload: payload["rows"][0].update(origin="workspace"))
    for _ in range(2):
        with pytest.raises(ValueError, match="identity"):
            second.leaderboard("apa-200mhz-b")
    assert len(loads) == 3                                     # A refused bundle is not remembered as acceptable.
    reseal(matrix.bundle, lambda payload: payload["rows"][0].update(origin="official"))
    assert first.leaderboard("apa-200mhz-b") == boards[0] and len(loads) == 4
    matrix.bundle.unlink()
    assert second.leaderboard("apa-200mhz-b").rows == [] and second.leaderboard("apa-200mhz-b").coverage.evaluated == 0


def test_a_board_request_derives_the_protocol_once_however_many_rows_it_checks(matrix, tmp_path):
    from opendpd.services.arena import ArenaController
    from opendpd.services.user_backbones import BackboneController
    from opendpd.services.workspace import NotFound, Workspace
    matrix.publish()
    derived = []
    ws = Workspace.open_or_create(tmp_path / "workspace")
    controller = ArenaController(ws, BackboneController(ws), protocol_provider=lambda: derived.append(None) or matrix.protocol)
    controller.leaderboard("apa-200mhz-b")
    derived.clear()
    board = controller.leaderboard("apa-200mhz-b")
    # Deriving the protocol hashes every frozen source file; doing it per checked row made a board request O(rows).
    assert len(board.rows) == 3 and len(derived) <= 2


def _cheaper_preset(payload):
    payload["rows"][0]["budgets"][1]["model_parameters"]["hidden_size"] -= 1


@pytest.mark.parametrize("change,message", [
    (lambda payload: payload.update(protocol_sha256="0" * 64), "different protocol"),
    (lambda payload: payload["rows"].append(copy.deepcopy(payload["rows"][0])), "identity"),
    (lambda payload: payload["rows"][0].update(origin="workspace"), "identity"),
    (lambda payload: payload["rows"][0].update(protocol_sha256="0" * 64), "identity"),
    (lambda payload: payload["rows"][1].update(board_id="synthetic-suite"), "identity"),
    (_cheaper_preset, "registered sweep"),
    (lambda payload: payload["rows"][0]["cases"].pop(), "exactly once"),
    (lambda payload: payload["rows"][0]["cases"][0]["judges"][0].update(checkpoint_sha256="9" * 64), "judge hash"),
    (lambda payload: payload["rows"][1].update(eligible=True), "cannot be ranked"),
    (lambda payload: payload["rows"][1].update(score=1.), "cannot be ranked"),
])
def test_bundle_with_foreign_duplicate_or_altered_evidence_is_refused(matrix, change, message):
    matrix.publish()         # The untouched bundle loads: see the test above.
    assert [row["backbone"] for row in read_json(matrix.bundle)["rows"]] == ["gru", "lstm", "mcldnn"]
    reseal(matrix.bundle, change)
    with pytest.raises(ValueError, match=message):
        arena.load_official_rows()


def test_bundle_edited_without_its_seal_is_refused(matrix):
    matrix.publish()
    payload = read_json(matrix.bundle)
    payload["rows"][0]["cases"][0]["judges"][0]["nmse_db"] -= 3.
    write_json_atomic(matrix.bundle, payload)
    with pytest.raises(ValueError, match="integrity"):
        arena.load_official_rows()
