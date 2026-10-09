"""The parity harness's own logic (no MATLAB needed): registered budgets, verdict rule, signal construction, and a
committed record that regenerates the committed document block exactly."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "performance" / "matlab-parity.md"
RECORD = ROOT / "docs" / "performance" / "matlab-parity.json"
pytestmark = pytest.mark.skipif(not DOC.is_file(), reason="parity documents are not part of this checkout")


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("matlab_parity", ROOT / "scripts" / "matlab_parity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registration(text: str = None) -> str:
    """The registered part of the document: everything before the run log."""
    text = text or DOC.read_text(encoding="utf-8")
    return text.split("## Run log")[0]


def test_budgets_in_the_script_are_the_registered_ones(harness):
    text = registration()
    assert f"≤ {harness.BUDGET_ACLR_DB} dB" in text
    assert f"≤ {harness.BUDGET_EVM_PP} percentage points" in text
    assert f"≤ {harness.BUDGET_CFO_HZ} Hz" in text
    assert "≤ 1·10⁻⁶" in text and harness.BUDGET_WAVEFORM == 1e-6
    assert harness.NPERSEG_122 == (1024, 2048, 4096) and harness.NPERSEG_245 == (2048, 4096, 8192)
    assert harness.DRIVE == {"S2": 0.15, "S3": 0.25, "S4": 0.35, "S5": 0.50}
    assert harness.INJECTED_CFO_HZ == 350.0 and harness.NOISE_SEED == 7 and harness.NOISE_SNR_DB == 30.0
    assert "**+30 Hz**" in DOC.read_text(encoding="utf-8").split("## Amendment 1")[1]
    assert harness.CFO_AMENDED_HZ == 30.0


def test_a_refused_signal_never_counts_as_agreement(harness):
    scored = lambda passed: {"kind": "scored", "pass": passed, "group": "P3", "item": "x", "difference": 0, "budget": "b", "note": "n"}
    diagnostic = {"kind": "diagnostic", "pass": None, "group": "P3", "item": "d", "difference": 9, "budget": None, "note": ""}
    ok = harness.verdict([scored(True), diagnostic])
    assert ok["all_scored_within_budget"] and ok["passed"] == 1
    refused = harness.verdict([scored(True), scored(None)])
    assert not refused["all_scored_within_budget"] and refused["not_evaluable"] == 1 and refused["passed"] == 1
    failed = harness.verdict([scored(True), scored(False)])
    assert not failed["all_scored_within_budget"] and failed["failed"] == 1


def test_interpolation_is_exact_for_the_periodic_band_limited_waveform(harness):
    from opendpd.core.waveforms import ofdm
    from opendpd.schemas.waveform import WaveformSpec

    x = ofdm.generate(WaveformSpec(seed=1, n_subframes=1)).x
    up = harness.fft_interpolate(x, 4)
    assert up.size == 4 * x.size
    np.testing.assert_allclose(up[::4], x, atol=1e-12)


def test_capture_applies_shift_gain_and_offset_in_that_order(harness):
    y = np.ones(4 * 30720 * 10, dtype=np.complex128)
    out = harness.capture(y, 122.88e6, 4, cfo_hz=1000.0)
    assert out.size == y.size
    n = np.arange(out.size)
    np.testing.assert_allclose(out, harness.GAIN * np.exp(2j * math.pi * 1000.0 / 122.88e6 * n), atol=1e-12)
    ramp = np.arange(y.size, dtype=float)
    shifted = harness.capture(ramp.astype(complex), 122.88e6, 4, cfo_hz=0.0) / harness.GAIN
    assert shifted[0].real == pytest.approx(3 * 122880)       # the capture starts three subframes into the waveform


def test_committed_record_regenerates_the_committed_results_block(harness):
    if not RECORD.is_file():
        pytest.skip("no run recorded yet")
    report = json.loads(RECORD.read_text(encoding="utf-8"))
    text = DOC.read_text(encoding="utf-8")
    block = text.split(harness.BEGIN, 1)[1].split(harness.END, 1)[0]
    assert block.lstrip("\n") == harness.render(report), "regenerate with scripts/matlab_parity.py --write-report"
    verdict = report["verdict"]
    assert verdict["passed"] + verdict["failed"] + verdict["not_evaluable"] == verdict["scored"]
    # the headline numbers quoted in the prose are the record's
    assert f"Of {verdict['scored']} scored items {verdict['passed']} are within" in text
    assert f"{verdict['failed']} are outside it" in text and f"{verdict['not_evaluable']} cannot be evaluated" in text


def test_registration_sections_are_present_and_precede_the_log():
    text = DOC.read_text(encoding="utf-8")
    order = [text.index(h) for h in ("## Rules", "## Signals", "## P1 — waveform generation", "## P2 — ACLR against",
                                     "## P3 — EVM against", "## P4 — verdict rule", "## Run log", "## Amendment 1", "## Results")]
    assert order == sorted(order), "registration, then log, then amendments, then results"
