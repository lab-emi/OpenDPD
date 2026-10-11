"""The memory-polynomial parity harness's own logic (no MATLAB needed): the registered cases and budgets, the
independent reference polynomial, the layout mapping, and a committed record that regenerates the committed
document block exactly."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "performance" / "matlab-parity-dpd.md"
RECORD = ROOT / "docs" / "performance" / "matlab-parity-dpd.json"
pytestmark = pytest.mark.skipif(not DOC.is_file(), reason="parity documents are not part of this checkout")


@pytest.fixture(scope="module")
def harness():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location("matlab_parity_dpd", ROOT / "scripts" / "matlab_parity_dpd.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    return module


def test_cases_and_budgets_in_the_script_are_the_registered_ones(harness):
    text = DOC.read_text(encoding="utf-8").split("## Run log")[0]
    assert harness.BUDGET_COEFFICIENT == 1e-6 and harness.BUDGET_OUTPUT == 1e-6 and harness.BUDGET_DOUBLE == 1e-9
    assert text.count("≤ 1·10⁻⁶") >= 4 and "≤ 1·10⁻⁹" in text
    assert harness.DRIVE == {"S4": 0.35, "S5": 0.50} and harness.SEGMENT == 2048
    for case, record, degree, memory in harness.CASES:      # the registered case table
        assert re.search(rf"\| {case} \| {record} \| {degree}[^|]*\| {memory}[^|]*\|", text), case
    assert [c[0] for c in harness.CASES] == ["C1", "C2", "C3", "C4"]
    registered = "c_true[k,q] = 0.6^(k+q) · (−1)^k · exp(j(0.35(k+1) + 0.6(q+1)))"
    assert registered in text
    c = harness.true_coefficients(2, 3)
    assert c[1 * 3 + 2] == pytest.approx(0.6 ** 3 * (-1) * np.exp(1j * (0.35 * 2 + 0.6 * 3)))


def test_the_reference_polynomial_is_independent_of_mp_basis_and_agrees_with_it(harness):
    from opendpd.core.polynomial import mp_basis

    rng = np.random.default_rng(0)
    v = rng.standard_normal(400) + 1j * rng.standard_normal(400)
    for degree, memory in ((5, 3), (7, 5), (1, 1), (3, 4)):
        c = harness.true_coefficients(degree, memory)
        np.testing.assert_allclose(harness.direct_polynomial(c, v, degree, memory), mp_basis(v, degree, memory) @ c, atol=1e-12)


def test_matlab_layout_maps_to_the_flat_opendpd_order(harness):
    degree, memory = 3, 4
    flat = np.arange(degree * memory) + 0j                       # w[k * Q + q]
    matlab = flat.reshape(memory, degree, order="F")             # Coef(q + 1, k + 1) in MATLAB, column-major storage
    assert matlab[2, 1] == flat[1 * memory + 2]                  # lag 2, degree index 1
    np.testing.assert_array_equal(harness.flat(matlab), flat)
    np.testing.assert_array_equal(harness.flat(flat.reshape(memory, degree, order="F").copy()), flat)


def test_float32_interface_and_the_module_agree_with_the_core(harness):
    from opendpd.core.polynomial import mp_basis

    rng = np.random.default_rng(1)
    x = 0.3 * (rng.standard_normal(500) + 1j * rng.standard_normal(500))
    w = harness.true_coefficients(5, 3)
    x32 = harness.float32_rounded(x)
    assert np.max(np.abs(x32 - x)) < 1e-6 and np.all(x32.real.astype(np.float32).astype(np.float64) == x32.real)
    out = harness.module_output(5, 3, w, x32)
    np.testing.assert_allclose(out, mp_basis(x32, 5, 3) @ w, rtol=0, atol=2e-6)


def test_without_the_product_path_q6_is_not_evaluable_and_the_verdict_cannot_pass(harness):
    lte = harness.lte
    items = [lte.item("Q6", "product path", "scored", None, None, None, 1e-6, None, "product path not run"),
             lte.item("Q1", "x", "scored", None, 1e-14, 1e-14, 1e-6, True)]
    verdict = lte.verdict(items)
    assert verdict["not_evaluable"] == 1 and not verdict["all_scored_within_budget"]


def test_committed_record_regenerates_the_committed_results_block(harness):
    if not RECORD.is_file():
        pytest.skip("no run recorded yet")
    report = json.loads(RECORD.read_text(encoding="utf-8"))
    text = DOC.read_text(encoding="utf-8")
    block = text.split(harness.BEGIN, 1)[1].split(harness.END, 1)[0]
    assert block.lstrip("\n") == harness.render(report), "regenerate with scripts/matlab_parity_dpd.py --write-report"
    verdict = report["verdict"]
    assert verdict["passed"] + verdict["failed"] + verdict["not_evaluable"] == verdict["scored"] == 21
    assert f"**Verdict: the registered memory-polynomial parity {'passes' if verdict['all_scored_within_budget'] else 'does not pass'}**" in text
    assert f"{verdict['passed']} of {verdict['scored']} scored items are within budget" in text
    # the diagnostics quoted in the prose are the record's
    d4 = {i["item"][:20]: i["difference"] for i in report["items"] if i["group"] == "D4"}
    assert f"{100 * max(d4.values()):.3f} %" in text or f"{100 * min(d4.values()):.3f} %" in text


def test_registration_sections_precede_the_log_amendments_and_results():
    text = DOC.read_text(encoding="utf-8")
    order = [text.index(h) for h in ("## Rules", "## What this can and cannot show", "## Conventions", "## Data",
                                     "## MathWorks configuration", "## Items", "## Verdict rule", "## Run log",
                                     "## Amendment 1", "## Results")]
    assert order == sorted(order), "registration, then log, then amendments, then results"
