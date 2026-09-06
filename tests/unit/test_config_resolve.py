"""One resolver for every entry point."""

import pytest

from opendpd.schemas import ExperimentConfig, ResolvedExperimentConfig, TaskType, TrainingConfig
from opendpd.schemas.examples import experiment_train_dpd_smoke, experiment_train_pa_smoke
from opendpd.services.config import ConfigError, canonical_json, config_sha256, resolve, validate


def test_resolve_fills_defaults_and_is_idempotent():
    cfg = experiment_train_pa_smoke()
    resolved = resolve(cfg)
    assert isinstance(resolved, ResolvedExperimentConfig)
    assert resolved.evaluation.checkpoint_selection_metric == "NMSE"
    assert resolved.model.parameters == {"hidden_size": 23, "num_layers": 1}
    again = resolve(resolved)
    assert again.resolution.config_sha256 == resolved.resolution.config_sha256
    assert config_sha256(again) == resolved.resolution.config_sha256


def test_hash_is_stable_across_field_order():
    a = experiment_train_pa_smoke().model_dump()
    b = dict(reversed(list(a.items())))
    assert canonical_json(ExperimentConfig.model_validate(a)) == canonical_json(ExperimentConfig.model_validate(b))


def test_hash_changes_with_any_result_affecting_field():
    base = resolve(experiment_train_pa_smoke())
    changed = experiment_train_pa_smoke().model_copy(update={"training": TrainingConfig(epochs=3, seed=1,
                                                                                         frame_length=50,
                                                                                         frame_stride=16,
                                                                                         batch_size_eval=256)})
    assert resolve(changed).resolution.config_sha256 != base.resolution.config_sha256


def test_unknown_model_reports_field_and_hint():
    cfg = experiment_train_pa_smoke().model_copy(update={"model": {"key": "transformer", "parameters": {}}})
    cfg = ExperimentConfig.model_validate(cfg.model_dump())
    with pytest.raises(ConfigError) as info:
        resolve(cfg)
    issue = info.value.issues[0]
    assert issue.field == "model.key" and "gru" in issue.hint


def test_dpd_without_bound_pa_reference_is_rejected():
    with pytest.raises(ConfigError) as info:
        resolve(experiment_train_dpd_smoke())
    assert info.value.issues[0].field == "pa_reference"


def test_selection_metric_is_protocol_not_user_choice():
    data = experiment_train_pa_smoke().model_dump()
    data["evaluation"]["checkpoint_selection_metric"] = "ACLR_AVG"
    with pytest.raises(ConfigError) as info:
        resolve(ExperimentConfig.model_validate(data))
    assert info.value.issues[0].field == "evaluation.checkpoint_selection_metric"


def test_unknown_profile_is_rejected():
    data = experiment_train_pa_smoke().model_dump()
    data["evaluation"]["profile_id"] = "nr-evm-v1"
    with pytest.raises(ConfigError) as info:
        resolve(ExperimentConfig.model_validate(data))
    assert info.value.issues[0].field == "evaluation.profile_id"


def test_warnings_for_smoke_length_and_untested_device():
    data = experiment_train_pa_smoke().model_dump()
    data["execution"]["device"] = "mps"
    report = validate(data)
    assert report.ok
    fields = {w.field for w in report.warnings}
    assert {"training.epochs", "execution.device"} <= fields
    assert any("mps" in w for w in report.resolved.resolution.warnings)


def test_validate_reports_schema_errors_with_paths():
    data = experiment_train_pa_smoke().model_dump()
    data["training"]["epochs"] = 0
    data["training"]["lr"] = 0.1
    report = validate(data)
    assert not report.ok
    fields = {e.field for e in report.errors}
    assert "training.epochs" in fields and "training.lr" in fields


def test_training_defaults_have_one_source():
    """TrainingConfig defaults must equal the legacy argparse defaults."""
    from arguments import build_parser

    ns = build_parser().parse_args([])
    t = TrainingConfig()
    assert t.epochs == ns.n_epochs and t.batch_size == ns.batch_size and t.batch_size_eval == ns.batch_size_eval
    assert t.learning_rate == ns.lr and t.lr_end == ns.lr_end and t.decay_factor == ns.decay_factor
    assert t.patience == ns.patience and t.optimizer == ns.opt_type and t.loss == ns.loss_type
    assert t.grad_clip == ns.grad_clip_val and t.frame_length == ns.frame_length
    assert t.frame_stride == ns.frame_stride and t.seed == ns.seed and t.reproducibility == ns.re_level
    assert t.lr_schedule == bool(ns.lr_schedule) and t.eval_val == bool(ns.eval_val)


def test_run_dpd_selection_metric_not_applicable():
    from opendpd.services.config import SELECTION_METRIC
    assert SELECTION_METRIC[TaskType.run_dpd] == "not_applicable"
