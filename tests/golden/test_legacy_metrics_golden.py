"""Legacy metric profile (legacy-opendpd-v1) must keep its frozen values.

The expected numbers live in ``legacy_metrics_v1.json`` (protected path). A
failure here means the historical NMSE / EVM / ACLR semantics changed; that is
only acceptable through a new metric-profile version, never by editing the
JSON to match new code.
"""

import json
from pathlib import Path

import pytest

from generate_legacy_metric_goldens import GOLDEN_PATH, compute

GOLDEN = json.loads(Path(GOLDEN_PATH).read_text())

# Deterministic CPU float64 algorithms; cross-platform FFT differences are far
# below this tolerance (documented in docs/protocols/acceptance-thresholds.md).
RTOL = 1e-7
ATOL = 1e-7


@pytest.mark.parametrize("profile_name", sorted(GOLDEN["profiles"]))
def test_legacy_metric_values_are_frozen(profile_name):
    entry = GOLDEN["profiles"][profile_name]
    actual = compute(entry["config"])
    for metric, expected in entry["expected"].items():
        assert actual[metric] == pytest.approx(expected, rel=RTOL, abs=ATOL), (
            f"{profile_name}/{metric}: expected {expected!r}, got {actual[metric]!r}"
        )


def test_golden_profiles_are_finite_and_negative_db():
    for name, entry in GOLDEN["profiles"].items():
        expected = entry["expected"]
        for metric in ("NMSE", "EVM", "ACLR_L", "ACLR_R", "ACLR_AVG"):
            assert expected[metric] < 0, f"{name}/{metric} should be a negative dB value"
