"""Public PA fits are bounded separately from neural training and ILC-DPD."""
import pytest
from fastapi import HTTPException

from opendpd.schemas import ExperimentConfig
from opendpd.services.recipes import instantiate
from opendpd.web.policy import check_config


@pytest.mark.parametrize("recipe", ["pa-mp-studio-v1", "pa-gmp-studio-v1"])
def test_compact_pa_fit_and_testing_are_available(recipe):
    config = instantiate(recipe, "capture")
    check_config(config)
    testing = ExperimentConfig(task="evaluate_pa", dataset={"id": "capture"}, model=config.model,
                               pa_reference={"run_id": "fitted-pa"}, execution={"device": "cpu"})
    check_config(testing)


@pytest.mark.parametrize("change", ["samples", "all_samples", "coefficients", "cuda", "dpd"])
def test_pa_fit_resource_or_task_escape_is_rejected(change):
    config = instantiate("pa-mp-studio-v1", "capture")
    if change == "samples":
        config.training.train_samples = 32769
    elif change == "all_samples":
        config.training.train_samples = None
    elif change == "coefficients":
        config.model.parameters.update(K=9, Q=150)
    elif change == "cuda":
        config.execution.device = "cuda"
    else:
        config = instantiate("dpd-mp-ila-v1", "capture", pa_run_id="fitted-pa")
        config.model.parameters = {"K": 7, "Q": 15, "rcond": 1e-4}
        config.training.train_samples = 32768
    with pytest.raises(HTTPException) as error:
        check_config(config)
    assert error.value.status_code == 422
    assert error.value.detail["error"]["code"] == "compute_limit"


def test_gmp_cross_terms_count_towards_memory_limit():
    config = instantiate("pa-gmp-studio-v1", "capture")
    config.model.parameters.update(Kb=4, Lb=30, Mb=5)
    with pytest.raises(HTTPException):
        check_config(config)


def test_frozen_full_benchmark_presets_are_unchanged_and_local():
    config = instantiate("pa-mp-ls-v1", "capture")
    assert config.model.parameters == {"K": 9, "Q": 150, "rcond": 0.0}
    assert config.training.train_samples is None
    with pytest.raises(HTTPException):
        check_config(config)
