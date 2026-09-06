"""The registry is the one place scores come from: legacy stays frozen, profiles are well formed, comparisons are protocol-bound."""

import importlib.util
import json
from pathlib import Path

import pytest

from opendpd.core.metrics import DEFAULT_PROFILE_ID, PROFILES, comparison_key, evaluate, get_profile, incompatibilities
from opendpd.schemas import SignalSpec
from opendpd.schemas.examples import result_dpd_surrogate_mock, result_pa_modeling_mock

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"
_spec = importlib.util.spec_from_file_location("legacy_goldens", GOLDEN_DIR / "generate_legacy_metric_goldens.py")
goldens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(goldens)
GOLDEN = json.loads((GOLDEN_DIR / "legacy_metrics_v1.json").read_text())


@pytest.mark.parametrize("name", sorted(GOLDEN["profiles"]))
def test_legacy_profile_through_the_registry_reproduces_the_frozen_goldens(name):
    cfg = GOLDEN["profiles"][name]["config"]
    prediction, ground_truth, _ = goldens.stimulus(cfg)
    signal = SignalSpec(sample_rate_hz=cfg["fs"], bandwidth_hz=cfg["bw_main_ch"], n_sub_ch=cfg["n_sub_ch"], nperseg=cfg["nperseg"])
    values = {m.name: m.value for m in evaluate("legacy-opendpd-v1", prediction, ground_truth, signal)}
    for metric, expected in GOLDEN["profiles"][name]["expected"].items():
        if metric in values:
            assert values[metric] == pytest.approx(expected, rel=1e-7, abs=1e-7), metric


def test_profiles_are_well_formed_and_the_default_is_the_frozen_one():
    assert DEFAULT_PROFILE_ID == "legacy-opendpd-v1" and get_profile(DEFAULT_PROFILE_ID).frozen
    for profile in PROFILES.values():
        assert profile.version >= 1 and profile.description
        for m in profile.metrics:
            assert m.formula and m.aggregation and m.unit and m.display_name
        assert "aggregation" in profile.parameters and "normalization" in profile.parameters
    general = get_profile("general-spectral-v1")
    assert not general.frozen and general.metric("NMSE").display_name == "NMSE (pooled)"
    assert "not a demodulated EVM" in general.metric("IBE").notes.lower() or "not a demodulated evm" in general.metric("IBE").notes.lower()
    assert all(m.unit == "dBc" and m.better.value == "lower" for m in general.metrics if m.name.startswith("ACPR"))
    with pytest.raises(KeyError):
        get_profile("nope-v9")


def test_legacy_profile_states_explicit_statuses_without_metadata():
    prediction, ground_truth, _ = goldens.stimulus(GOLDEN["profiles"]["dpa_200mhz"]["config"])
    m = {v.name: v for v in evaluate("legacy-opendpd-v1", prediction, ground_truth, SignalSpec())}
    assert m["NMSE"].status.value == "ok"
    assert {m[k].status.value for k in ("EVM", "ACLR_L", "ACLR_R", "ACLR_AVG")} == {"not_applicable"}
    assert "metadata missing" in m["EVM"].reason


def test_results_under_different_protocols_are_not_comparable():
    pa, dpd = result_pa_modeling_mock(), result_dpd_surrogate_mock()
    assert incompatibilities(pa, pa) == []
    reasons = incompatibilities(pa, dpd)
    assert any(r.startswith("evidence type") for r in reasons) and any(r.startswith("reference kind") for r in reasons)
    assert any(r.startswith("PA surrogate: n/a vs ") for r in reasons)
    assert comparison_key(pa)["metric profile"] == "legacy-opendpd-v1 v1"
    # the same DPD scored through another surrogate is shown side by side, never ranked
    other = dpd.model_copy(update={"models": [m.model_copy(update={"weights_sha256": "f" * 64}) if m.role == "pa" else m
                                              for m in dpd.models]})
    assert [r for r in incompatibilities(dpd, other) if r.startswith("PA surrogate")] and comparison_key(dpd) != comparison_key(other)
