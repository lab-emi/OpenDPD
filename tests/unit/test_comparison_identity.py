"""Ranking must reject distinct sources even when display labels coincide."""
from types import SimpleNamespace

import pytest

from opendpd.core.metrics import incompatibilities
from opendpd.schemas.examples import result_dpd_measured_mock, result_dpd_surrogate_mock


def test_identical_results_remain_comparable_without_new_optional_metadata():
    result = result_dpd_surrogate_mock()
    assert incompatibilities(result, result.model_copy(deep=True)) == []


@pytest.mark.parametrize("field", ["raw_sha256", "processed_sha256"])
def test_different_data_with_the_same_dataset_name_cannot_be_ranked(field):
    first = result_dpd_surrogate_mock()
    second = first.model_copy(deep=True)
    setattr(first.dataset, field, "a" * 64)
    setattr(second.dataset, field, "a" * 12 + "b" * 52)
    assert any("data hash:" in reason for reason in incompatibilities(first, second))


def test_surrogates_with_equal_display_prefixes_are_distinct():
    first = result_dpd_surrogate_mock()
    second = first.model_copy(deep=True)
    pa = next(model for model in first.models if model.role == "pa")
    other = next(model for model in second.models if model.role == "pa")
    other.weights_sha256 = pa.weights_sha256[:12] + "f" * 52
    assert any(reason.startswith("PA surrogate:") for reason in incompatibilities(first, second))


def test_reference_gains_are_not_rounded_to_display_precision():
    first = result_dpd_surrogate_mock()
    second = first.model_copy(deep=True)
    first.reference.gain_value = 1.0000001
    second.reference.gain_value = 1.0000002
    assert any(reason.startswith("reference gain:") for reason in incompatibilities(first, second))


def test_recorded_conditions_and_unknown_conditions_are_not_equal():
    first = result_dpd_surrogate_mock()
    # Optional context is populated by the Studio schema extension. Older
    # results without it remain readable by this comparison layer.
    second = first.model_copy(update={"rf_conditions": SimpleNamespace(average_output_power_dbm=20.0)})
    assert any("average output power dbm:" in reason for reason in incompatibilities(first, second))


def test_real_measurement_requires_a_declared_output_power_for_ranking():
    first = result_dpd_measured_mock().model_copy(update={"is_mock": False})
    first.measurement.captures[0].declared_output_power_dbm = None
    assert any("output power not declared" in reason for reason in incompatibilities(first, first))


def test_evaluation_ranges_cannot_be_joined_silently():
    first = result_dpd_surrogate_mock()
    second = first.model_copy(update={"valid_sample_range": (200, 1000)})
    assert any(reason.startswith("valid sample range:") for reason in incompatibilities(first, second))
