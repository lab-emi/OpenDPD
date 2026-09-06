"""S16: the instrument contract and its fail-closed safety (mock adapter only; nothing here emits RF)."""

import json

import numpy as np
import pytest

from opendpd.instruments import (
    ALLOW_RF_ENV,
    AdapterInfo,
    Instrument,
    InstrumentError,
    Interlock,
    LinkLost,
    MockInstrument,
    SafetyLimits,
    SafetyViolation,
    list_adapters,
    run_capture_session,
)
from opendpd.instruments.mock import MOCK_DELAY_SAMPLES, MOCK_GAIN

def _signal(n=3000, peak=0.5, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n) + 1j * rng.normal(size=n)
    return z / np.max(np.abs(z)) * peak


def test_only_the_mock_ships_and_it_cannot_emit_rf():
    infos = list_adapters()
    assert [i.adapter_id for i in infos] == ["mock"]
    assert infos[0].kind == "mock" and infos[0].rf_output_capable is False


def test_rf_output_is_off_until_a_named_person_arms():
    inst = MockInstrument()
    inst.connect()
    lock = Interlock(inst)
    assert lock.state == "disarmed" and inst.output_on is False
    with pytest.raises(SafetyViolation, match="armed"):
        lock.play(_signal(), 1e6)
    with pytest.raises(SafetyViolation, match="operator's name"):
        lock.arm("   ")
    lock.arm("operator")
    assert lock.state == "armed"
    lock.play(_signal(), 1e6)
    assert inst.output_on is True
    lock.disarm()
    assert inst.output_on is False and lock.state == "disarmed"


class _RealLike(MockInstrument):
    info = AdapterInfo(adapter_id="lab-real", kind="real", rf_output_capable=True, description="pretend real chain")


def test_real_output_needs_the_environment_gate_that_ci_never_sets(monkeypatch):
    monkeypatch.delenv(ALLOW_RF_ENV, raising=False)
    lock = Interlock(_RealLike())
    with pytest.raises(SafetyViolation, match=ALLOW_RF_ENV):
        lock.arm("operator")
    assert lock.state == "disarmed"
    monkeypatch.setenv(ALLOW_RF_ENV, "1")
    Interlock(_RealLike()).arm("operator")


def test_limits_refuse_before_anything_is_sent():
    inst = MockInstrument()
    inst.connect()
    lock = Interlock(inst, SafetyLimits(max_peak_abs=0.4, max_output_power_dbm=20.0))
    lock.arm("operator")
    with pytest.raises(SafetyViolation, match="peak"):
        lock.play(_signal(peak=0.5), 1e6)
    with pytest.raises(SafetyViolation, match="ceiling"):
        lock.play(_signal(peak=0.3), 1e6, requested_power_dbm=25.0)
    with pytest.raises(SafetyViolation, match="empty or non-finite"):
        lock.play(np.array([np.nan + 0j]), 1e6)
    assert inst.playing is None and inst.output_on is False and lock.state == "armed"


def test_timeout_loss_of_link_failure_and_abort_all_land_in_tripped_with_rf_off():
    slow = MockInstrument(capture_delay_s=1.0)
    slow.connect()
    lock = Interlock(slow, SafetyLimits(timeout_s=0.2))
    lock.arm("operator")
    lock.play(_signal(), 1e6)
    with pytest.raises(InstrumentError, match="timed out"):
        lock.capture(100, 1e6)
    assert lock.state == "tripped" and slow.output_on is False and "exceeded" in lock.trip_reason
    with pytest.raises(SafetyViolation, match="tripped"):
        lock.arm("operator")

    lost = MockInstrument(lose_link_after=1, capture_delay_s=0.6)
    lost.connect()
    lock = Interlock(lost, SafetyLimits(link_timeout_s=0.2))
    lock.arm("operator")
    lock.play(_signal(), 1e6)
    with pytest.raises(LinkLost):
        lock.capture(100, 1e6)
    assert lock.state == "tripped" and lost.output_on is False

    failing = MockInstrument(fail_capture=True)
    failing.connect()
    lock = Interlock(failing)
    lock.arm("operator")
    lock.play(_signal(), 1e6)
    with pytest.raises(InstrumentError, match="overload"):
        lock.capture(100, 1e6)
    assert lock.state == "tripped" and failing.output_on is False

    inst = MockInstrument()
    inst.connect()
    lock = Interlock(inst)
    lock.arm("operator")
    lock.play(_signal(), 1e6)
    lock.abort("operator pressed stop")
    assert lock.state == "tripped" and inst.output_on is False
    assert [e["event"] for e in lock.log if e["event"] in ("armed", "play", "rf_off", "tripped")][:4] == ["armed", "play", "rf_off", "tripped"]


def test_leaving_the_session_block_turns_rf_off_even_on_an_exception():
    inst = MockInstrument()
    with pytest.raises(RuntimeError, match="boom"):
        with Interlock(inst) as lock:
            lock.arm("operator")
            lock.play(_signal(), 1e6)
            raise RuntimeError("boom")
    assert inst.output_on is False and inst.connected is False and lock.state == "tripped"


def test_a_capture_session_records_everything_and_ends_with_rf_off(tmp_path):
    inst = MockInstrument()
    played = _signal()
    out = run_capture_session(inst, played, 1e6, operator="operator", out=tmp_path / "with_dpd.npy")
    captured = np.load(out)
    assert captured.dtype == np.float32 and captured.shape == (played.size + 4096, 2)
    record = json.loads((tmp_path / "with_dpd.npy.session.json").read_text())
    assert record["mock"] is True and record["adapter"]["adapter_id"] == "mock" and record["operator"] == "operator"
    assert record["interlock"]["final_state"] == "disarmed" and record["interlock"]["trip_reason"] is None
    assert inst.output_on is False and inst.rf_off_calls >= 1
    # the stand-in PA is a known cubic with a known delay and gain: what the alignment must recover
    z = captured[:, 0] + 1j * captured[:, 1]
    corr = np.fft.ifft(np.fft.fft(z[:played.size]) * np.conj(np.fft.fft(played)))
    assert int(np.argmax(np.abs(corr))) == MOCK_DELAY_SAMPLES
    window = z[MOCK_DELAY_SAMPLES:MOCK_DELAY_SAMPLES + played.size]
    small = np.abs(played) < 0.05
    gain = np.vdot(played[small], window[small]) / np.vdot(played[small], played[small])
    assert abs(gain) == pytest.approx(abs(MOCK_GAIN), rel=0.02)


def test_the_contract_is_abstract():
    with pytest.raises(TypeError):
        Instrument()  # type: ignore[abstract]
