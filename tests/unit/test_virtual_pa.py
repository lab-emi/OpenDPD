"""Analytic limits, causality, time bases and effective controls of virtual PAs."""
import re

import numpy as np
import pytest

from opendpd.core.virtual_pa import catalog, resolve, simulate, analyze

MODELS = catalog()


@pytest.mark.parametrize("model", MODELS, ids=lambda model: model.model_id)
def test_every_control_is_bound_to_an_equation_and_has_a_real_effect(model):
    references = set(re.findall(r"\{\{(\w+)\}\}", " ".join(model.equations)))
    assert references == {p.key for p in model.parameters}
    n = np.arange(8192)
    x = (.03 + .6*(1+np.sin(2*np.pi*n/769))/2) * np.exp(1j*.031*n)
    x[(n//1024) % 2 == 0] *= .1
    _, defaults = resolve(model.model_id, {})
    for parameter in model.parameters:
        low, _ = simulate(x, 1e6, model.model_id, {**defaults, parameter.key: parameter.minimum})
        # -180° and +180° are the same physical phase; probe 90° instead.
        upper = 90 if parameter.key == 'phase_deg' else parameter.maximum
        high, _ = simulate(x, 1e6, model.model_id, {**defaults, parameter.key: upper})
        assert not np.allclose(low, high, rtol=1e-8, atol=1e-10), parameter.key


@pytest.mark.parametrize("model", MODELS, ids=lambda model: model.model_id)
def test_zero_input_and_causal_prefix_are_deterministic(model):
    zero, states = simulate(np.zeros(256), 80e6, model.model_id, {})
    np.testing.assert_array_equal(zero, 0)
    report = analyze(np.zeros(256), zero, 80e6, states, {})
    assert report.rms_gain_db is None and report.output_papr_db is None
    assert "NaN" not in report.model_dump_json()
    rng = np.random.default_rng(13)
    x = .2*(rng.normal(size=800)+1j*rng.normal(size=800))
    full, _ = simulate(x, 20e6, model.model_id, {})
    prefix, _ = simulate(x[:400], 20e6, model.model_id, {})
    np.testing.assert_array_equal(prefix, full[:400])


def test_linear_gain_and_phase_and_rapp_saturation_limits():
    x = np.array([0, 1e-9+1e-9j, .2+.1j, 10+2j])
    y, _ = simulate(x, 80e6, "linear-reference", {"gain": 3, "phase_deg": 90})
    np.testing.assert_allclose(y, 3j*x, atol=1e-14)
    y, _ = simulate(x, 80e6, "rapp-solid-state", {"gain": 2, "saturation": .7})
    np.testing.assert_allclose(y[1], 2*x[1], rtol=1e-12)
    assert np.max(np.abs(y)) <= .7 and np.abs(y[-1]) > .699


def test_memory_taps_and_cross_envelope_are_causal_and_distinct():
    x = np.zeros(20, complex)
    x[0] = .2
    y, _ = simulate(x, 1e6, "memory-polynomial", {"gain": 2, "cubic": 0, "quintic": 0,
        "memory": .3, "depth": 3, "decay": .5, "memory_phase": 0})
    np.testing.assert_allclose(y[:4], [.4, .4*.3/1.75, .4*.3*.5/1.75, .4*.3*.25/1.75])
    np.testing.assert_array_equal(y[4:], 0)
    with pytest.raises(ValueError, match="integer"):
        resolve("memory-polynomial", {"depth": 1.5})
    with pytest.raises(ValueError, match="Unknown PA parameters"):
        resolve("linear-reference", {"trap_strength": .5})


def test_dynamic_state_uses_physical_time_not_number_of_samples():
    params = {"capture_us": 10, "thermal_us": 20, "bias_us": 5}
    a, sa = simulate(np.full(200, .4+0j), 1e6, "gan-trap-thermal", params)
    b, sb = simulate(np.full(400, .4+0j), 2e6, "gan-trap-thermal", params)
    for state in sa:
        np.testing.assert_allclose(sa[state][-1], sb[state][-1], rtol=1e-10)
    np.testing.assert_allclose(a[-1], b[-1], rtol=1e-10)
    assert 25 < sa["effective_temperature_c"][-1] < 60
    assert 0 < sa["trap_occupancy"][-1] < 1


def test_switching_off_device_effects_recovers_static_rapp():
    x = .2*np.exp(1j*np.arange(500))
    dynamic, _ = simulate(x, 20e6, "gan-trap-thermal",
        {"trap_strength": 0, "trap_phase": 0, "thermal_gain": 0, "ir_drop": 0})
    static, _ = simulate(x, 20e6, "rapp-solid-state", {})
    np.testing.assert_array_equal(dynamic, static)
