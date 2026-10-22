"""The test-model resource grids reproduce the ones MATLAB R2026a 5G Toolbox generated (digests only, no MathWorks data)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from opendpd.core.waveforms.nr_tm.grid import build_carrier_grid
from opendpd.core.waveforms.nr_tm.tables import MODEL_BY_ID

DATA = json.loads((Path(__file__).resolve().parents[1] / "data" / "nr_tm" / "matlab_grid_digests.json").read_text())
ENTRIES = DATA["entries"]


def digest(grid: np.ndarray) -> str:
    a = np.round(np.stack([grid.real, grid.imag]), 10) + 0.0
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()


def build(entry, **kwargs):
    return build_carrier_grid(MODEL_BY_ID[entry["model"]], entry["n_rb"], entry["scs_khz"], entry["duplex"],
                              entry["n_cell_id"], entry["subframes"], data_convention="matlab", **kwargs)


def test_data_file_covers_the_oracle_matrix():
    assert len(ENTRIES) >= 400
    assert {e["model"] for e in ENTRIES} == {m for m in MODEL_BY_ID if m.startswith("NR-FR1-")}
    assert {(e["scs_khz"], e["duplex"]) for e in ENTRIES} == {(s, d) for s in (15, 30, 60) for d in ("fdd", "tdd")}
    assert len({e["n_cell_id"] for e in ENTRIES}) >= 8 and {1, 2, 5, 7, 100, 503, 1006, 1007} <= {e["n_cell_id"] for e in ENTRIES}
    assert {1, 3, 10, 13, 20, 25, 40} <= {e["subframes"] for e in ENTRIES}
    # every model appears for both duplex modes at several subcarrier spacings and bandwidths
    for model in {e["model"] for e in ENTRIES}:
        rows = [e for e in ENTRIES if e["model"] == model]
        assert {e["duplex"] for e in rows} == {"fdd", "tdd"} and {e["scs_khz"] for e in rows} == {15, 30, 60}
        assert len({e["bandwidth_mhz"] for e in rows}) >= 4


def test_every_grid_matches_matlab_digest():
    bad = []
    for entry in ENTRIES:
        grid = build(entry).values
        same = (list(grid.shape) == entry["shape"] and int(np.count_nonzero(grid)) == entry["nnz"]
                and digest(grid) == entry["sha256"])
        if not same:
            bad.append(entry["id"])
    assert not bad, f"{len(bad)} of {len(ENTRIES)} grids differ from the MATLAB digests: {bad[:10]}"


@pytest.mark.parametrize("entry", [e for e in ENTRIES if e["n_cell_id"] != 1][:20], ids=lambda e: e["id"])
def test_cell_id_variants_match(entry):
    grid = build(entry).values
    assert digest(grid) == entry["sha256"]
    assert abs(float(np.sum(np.abs(grid) ** 2)) - entry["sum_abs2"]) < 1e-6 * entry["sum_abs2"]


def test_data_convention_is_ignored_for_pn_scope_in_matlab_mode():
    entry = next(e for e in ENTRIES if e["duplex"] == "tdd" and e["model"] == "NR-FR1-TM1.1" and e["subframes"] == 20)
    a = build(entry).values
    b = build_carrier_grid(MODEL_BY_ID[entry["model"]], entry["n_rb"], entry["scs_khz"], "tdd", entry["n_cell_id"],
                           entry["subframes"], data_convention="matlab", pn_scope="all_symbols").values
    assert np.array_equal(a, b)
