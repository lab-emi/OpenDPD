"""Test-model definitions against closed forms written independently from the text of TS 38.141-1 4.9.2.2 / 38.104 / 38.214."""
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from opendpd.core.waveforms.nr_tm import spec, tables
from opendpd.core.waveforms.nr_tm.tables import MODEL_BY_ID, NRB, TDD_PATTERNS

REFERENCE = json.loads((Path(__file__).resolve().parents[1] / "data" / "nr_tm" / "rbg_reference.json").read_text())["rows"]
CELLS = [(scs, bw, nrb) for scs, row in NRB.items() for bw, nrb in row.items()]
X_PERCENT = {"NR-FR1-TM1.2": 40, "NR-FR1-TM3.2": 60, "NR-FR1-TM3.3": 50}


def closed_form(nrb, x_percent):
    """RBG partition, N_RBG, selected RBG indices and PRBs straight from the text (Fractions, explicit partition)."""
    n = nrb - 3
    p = next(size for limit, size in ((36, 2), (72, 4), (144, 8), (275, 16)) if n <= limit)
    groups = [list(range(3, 3 + p - 3 % p))]
    while groups[-1][-1] + 1 < nrb:
        start = groups[-1][-1] + 1
        groups.append(list(range(start, min(start + p, nrb))))
    a = math.floor(Fraction(n + (3 % p) - p, p))
    count = min(math.floor(Fraction(x_percent, 100) * n / p), (a - a % 2) // 2 + 1)
    indices = {math.floor(Fraction(n + 3 % p, p)) - 1}
    if count > 1:
        indices |= {2 * j + 1 for j in range(count - 1)}
    prbs = sorted(prb for i in indices for prb in groups[i])
    return p, count, sorted(indices), prbs, groups


def test_resource_block_table_matches_the_generic_presets_and_the_guard_bands():
    from opendpd.core.waveforms import generator_presets as presets
    for scs, rbs in presets._NR_RBS.items():
        for bandwidth, nrb in zip(presets._NR_BANDWIDTHS, rbs):
            assert NRB[scs].get(bandwidth, 0) == nrb
    assert len(CELLS) == 40
    for scs, bw, nrb in CELLS:
        # table 5.3.2-1 is the largest N_RB that respects the minimum guard band of table 5.3.3-1
        assert nrb == math.floor((bw * 1000 - 2 * tables.GUARD_BAND_KHZ[scs][bw]) / (12 * scs))
    assert tables.n_rb(15, 3) == 15
    with pytest.raises(ValueError, match="not defined for 60 kHz"):
        tables.n_rb(60, 5)
    with pytest.raises(ValueError, match="not defined for NR FR1"):
        tables.n_rb(120, 100)


@pytest.mark.parametrize("model_id", list(X_PERCENT))
def test_rbg_selection_matches_closed_form_in_every_cell(model_id):
    model = MODEL_BY_ID[model_id]
    x = X_PERCENT[model_id]
    for scs, bw, nrb in CELLS:
        p, count, indices, prbs, groups = closed_form(nrb, x)
        assert spec.rbg_size(nrb - 3) == p
        assert spec.n_target_rbg(nrb, x) == count
        assert list(spec.target_rbg_indices(nrb, x)) == indices
        assert spec.target_prbs(nrb, x).tolist() == prbs
        assert len(prbs) == p * count and len(indices) == count          # all selected RBGs are full size, none coincide
        assert all(len(groups[i]) == p for i in indices)
        rest = [prb for prb in range(3, nrb) if prb not in set(prbs)]
        level_target = model.target_level_db
        level_rest = spec.model_levels(model, nrb)[0 if model.target_rnti == 1 else 1]
        # total power of PRB 3..N_RB-1 equals that of N_RB-3 PRBs at 0 dB
        total = len(prbs) * 10 ** (level_target / 10) + len(rest) * 10 ** (level_rest / 10)
        assert abs(total - (nrb - 3)) < 1e-9, (scs, bw)
        assert level_rest == pytest.approx(10 * math.log10((nrb - 3 - 10 ** (level_target / 10) * len(prbs)) / len(rest)), abs=1e-12)


def test_rbg_selection_matches_the_independent_research_table():
    assert len(REFERENCE) == 40
    for row in REFERENCE:
        assert tables.n_rb(row["scs_khz"], row["bandwidth_mhz"]) == row["n_rb"]
        assert spec.rbg_size(row["n_rb"] - 3) == row["p"]
        for name, x in (("TM1.2", 40), ("TM3.2", 60), ("TM3.3", 50)):
            model = MODEL_BY_ID["NR-FR1-" + name]
            ref = row[name]
            assert spec.n_target_rbg(row["n_rb"], x) == ref["n_rbg"]
            assert list(spec.target_rbg_indices(row["n_rb"], x)) == ref["rbg_indices"]
            target = spec.target_prbs(row["n_rb"], x)
            assert len(target) == ref["n_target_prb"] and row["n_rb"] - 3 - len(target) == ref["n_other_prb"]
            other = spec.model_levels(model, row["n_rb"])[0 if model.target_rnti == 1 else 1]
            assert other == pytest.approx(ref["other_db"], abs=6e-5)


def test_worked_example_106_prb():
    # N_RB = 106 (20 MHz, 15 kHz): P = 8, RBG 0 = PRB 3-7, RBG i = 8 i ... 8 i + 7, last RBG = PRB 104-105
    assert spec.rbg_size(103) == 8
    assert spec.prb_ranges(spec.target_prbs(106, 40)) == [[8, 15], [24, 31], [40, 47], [56, 63], [96, 103]]
    assert spec.prb_ranges(spec.target_prbs(106, 60)) == [[8, 15], [24, 31], [40, 47], [56, 63], [72, 79], [88, 103]]
    assert spec.prb_ranges(spec.target_prbs(106, 50)) == [[8, 15], [24, 31], [40, 47], [56, 63], [72, 79], [96, 103]]
    assert spec.model_levels(MODEL_BY_ID["NR-FR1-TM1.2"], 106)[1] == pytest.approx(-4.340490261365674, abs=1e-12)
    assert spec.model_levels(MODEL_BY_ID["NR-FR1-TM3.2"], 106)[0] == pytest.approx(2.0257824665223465, abs=1e-12)
    assert spec.model_levels(MODEL_BY_ID["NR-FR1-TM3.3"], 106)[0] == pytest.approx(2.1840632494990033, abs=1e-12)


def test_levels_of_the_target_groups_are_fixed_by_the_standard():
    levels = {m: spec.model_levels(MODEL_BY_ID[m], 106) for m in X_PERCENT}
    assert levels["NR-FR1-TM1.2"][0] == 3.0 and levels["NR-FR1-TM3.2"][1] == -3.0 and levels["NR-FR1-TM3.3"][1] == -6.0
    for model in ("NR-FR1-TM1.1", "NR-FR1-TM2", "NR-FR1-TM3.1b"):
        assert set(spec.model_levels(MODEL_BY_ID[model], 106).values()) == {0.0}
    assert spec.model_levels(MODEL_BY_ID["NR-FR1-TM1.2"], 106)[2] == 0.0                 # n_RNTI 2 stays at 0 dB


@pytest.mark.parametrize("mu", [0, 1, 2])
def test_tm2_slot_classes(mu):
    slots = 10 * 2 ** mu
    for scs, bw, nrb in CELLS:
        if tables.MU[scs] != mu:
            continue
        positions = [spec.tm2_prb(nrb, s) for s in range(slots)]
        for s, prb in enumerate(positions):
            assert prb == {0: 0, 1: nrb // 2, 2: nrb - 1}[s % 3]
    # table 4.9.2.2.3-1: slot 3n+r exists for n = 0 ... ceil((slots - r) / 3) - 1
    for r in range(3):
        assert len([s for s in range(slots) if s % 3 == r]) == math.ceil((slots - r) / 3)


def test_models_expose_the_per_user_structure_of_table_4_9_2_3_2_1():
    users = {m.id: sorted(u.rnti for u in m.users) for m in tables.MODELS}
    assert users["NR-FR1-TM1.1"] == [0, 2] and users["NR-FR1-TM1.2"] == [0, 1, 2] and users["NR-FR1-TM2"] == [2]
    assert users["NR-FR1-TM3.1b"] == [0, 2] and users["NR-FR1-TM3.2"] == [0, 1, 2] and users["NR-FR1-TM3.3"] == [0, 1, 2]
    qm = {m.id: [u.qm for u in sorted(m.users, key=lambda u: u.rnti)] for m in tables.MODELS}
    assert qm["NR-FR1-TM1.1"] == [2, 2] and qm["NR-FR1-TM1.2"] == [2, 2, 2] and qm["NR-FR1-TM2"] == [6]
    assert qm["NR-FR1-TM2a"] == [8] and qm["NR-FR1-TM2b"] == [10]
    assert qm["NR-FR1-TM3.1"] == [6, 6] and qm["NR-FR1-TM3.1a"] == [8, 8] and qm["NR-FR1-TM3.1b"] == [10, 10]
    assert qm["NR-FR1-TM3.2"] == [2, 4, 2] and qm["NR-FR1-TM3.3"] == [2, 2, 2]          # rnti 1 carries the measured 16QAM / QPSK
    assert {m.id: m.suffix for m in tables.MODELS}["NR-FR1-TM3.1a"] == "tm3p1a" and MODEL_BY_ID["NR-N-TM"].suffix == "ntm"
    assert [m.id for m in tables.MODELS if m.supports_layers] == ["NR-FR1-TM1.1"]
    assert [m.id for m in tables.MODELS if m.nb_iot] == ["NR-N-TM"]
    start = {u.rnti: u.start_symbol for m in tables.MODELS for u in m.users}
    assert start == {0: 0, 1: 0, 2: 2}


@pytest.mark.parametrize("model_id", [m.id for m in tables.MODELS])
def test_slot_allocations_are_disjoint_and_cover_the_defined_prbs(model_id):
    model = MODEL_BY_ID[model_id]
    for scs, bw, nrb in CELLS[::3]:
        for slot in range(10 * 2 ** tables.MU[scs]):
            allocs = spec.slot_allocations(model, nrb, slot, 14)
            seen = np.zeros(nrb, dtype=int)
            for a in allocs:
                seen[a.prbs] += 1
            assert seen.max() <= 1
            if model.structure == "single_prb":
                assert seen.sum() == 1 and [a.rnti for a in allocs] == [2]
            else:
                assert seen.sum() == nrb                                    # PRB 0-2 (n_RNTI 2) and PRB 3...N_RB-1 (rest) fully allocated
            assert all(a.sym_end == 14 for a in allocs)


@pytest.mark.parametrize("scs,pattern", [(15, [14, 14, 14, 10, 0]), (30, [14] * 7 + [6, 0, 0]), (60, [14] * 14 + [12, 0, 0, 0, 0, 0])])
def test_tdd_patterns_follow_table_4_9_2_2_1(scs, pattern):
    p = TDD_PATTERNS[scs]
    assert p.slots == len(pattern) == 5 * 2 ** tables.MU[scs]
    assert [p.dl_symbols_of_slot(i) for i in range(p.slots)] == pattern
    symbols = p.symbol_map()
    # all three columns describe the same physical time structure: 26/35 downlink, 8/35 uplink, 1/35 flexible
    total = len(symbols)
    assert Fraction(symbols.count("D"), total) == Fraction(26, 35)
    assert Fraction(symbols.count("U"), total) == Fraction(8, 35)
    assert Fraction(symbols.count("F"), total) == Fraction(1, 35)
    assert symbols.count("D") == {15: 52, 30: 104, 60: 208}[scs]


def test_tdd_slot_maps_and_labels():
    assert TDD_PATTERNS[15].slot_maps() == ["D" * 14] * 3 + ["DDDDDDDDDDFFUU", "U" * 14]
    assert TDD_PATTERNS[15].labels() == ["D", "D", "D", "S 10D/2U", "U"]
    assert TDD_PATTERNS[30].slot_maps()[7] == "DDDDDDFFFFUUUU" and TDD_PATTERNS[30].labels()[7:] == ["S 6D/4U", "U", "U"]
    maps60 = TDD_PATTERNS[60].slot_maps()
    assert maps60[14] == "DDDDDDDDDDDDFF" and maps60[15] == "FFFFFFUUUUUUUU" and maps60[16:] == ["U" * 14] * 4
    assert TDD_PATTERNS[60].labels()[14:16] == ["S 12D", "S 8U"]


def test_pdsch_resource_elements_per_allocation():
    model = MODEL_BY_ID["NR-FR1-TM1.1"]
    full = {a.rnti: spec.pdsch_data_res(a) for a in spec.slot_allocations(model, 52, 0, 14)}
    assert full[0] == 49 * 12 * 14 - 2 * 49 * 6                      # 14 symbols, two DM-RS symbols lose their even subcarriers
    assert full[2] == 3 * 12 * 12 - 2 * 3 * 6                        # symbols 2...13
    special = {a.rnti: spec.pdsch_data_res(a) for a in spec.slot_allocations(model, 52, 3, 10)}
    assert special[0] == 49 * 12 * 10 - 2 * 49 * 6 and special[2] == 3 * 12 * 8 - 2 * 3 * 6     # DM-RS {2, 9}
    six = {a.rnti: spec.pdsch_data_res(a) for a in spec.slot_allocations(model, 52, 7, 6)}
    assert six[0] == 49 * 12 * 6 - 49 * 6 and six[2] == 3 * 12 * 4 - 3 * 6                         # DM-RS {2}
    assert spec.slot_allocations(model, 52, 0, 0) == [] and [a.rnti for a in spec.slot_allocations(model, 52, 0, 2)] == [0]
