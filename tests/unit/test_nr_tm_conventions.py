"""PN23 conventions (TS 38.141-1 4.9.2.3), TDD slot handling and two-layer transmission, checked by reading the bits back out of the grid."""
from fractions import Fraction

import numpy as np
import pytest

from opendpd.core.waveforms.nr_tm import spec, tables
from opendpd.core.waveforms.nr_tm.grid import build_carrier_grid
from opendpd.core.waveforms.nr_tm.sequences import gold, pn23, scrambling_cinit_pdcch, scrambling_cinit_pdsch
from opendpd.core.waveforms.nr_tm.tables import MODEL_BY_ID, TDD_PATTERNS, dmrs_symbols

from test_nr_tm_sequences import literal_qam


def demap(symbols, qm):
    """Hard decision to the nearest constellation point of the literal 38.211 mapping (a chunked brute-force search)."""
    patterns = ((np.arange(1 << qm)[:, None] >> np.arange(qm - 1, -1, -1)[None, :]) & 1).astype(np.uint8)
    points = np.array([literal_qam(p) for p in patterns])
    out = np.empty((len(symbols), qm), dtype=np.uint8)
    for start in range(0, len(symbols), 20000):
        chunk = symbols[start:start + 20000]
        out[start:start + 20000] = patterns[np.argmin(np.abs(chunk[:, None] - points[None, :]), axis=1)]
    return out.reshape(-1)


def data_positions(prbs, start, end):
    """(subcarrier, symbol) of the PDSCH data REs of an allocation in mapping order (k first, then l), DM-RS comb removed; plain loops."""
    dm = set(dmrs_symbols(end))
    return [(k, l) for l in range(start, end) for prb in prbs for k in range(12 * prb, 12 * prb + 12) if not (l in dm and k % 2 == 0)]


def reader(model_id, nrb, scs, duplex, ncell, subframes, convention, scope="dl_symbols"):
    """Build a grid and return (grid, read, pdcch): read(slot, rnti) = descrambled bits of that user in that slot (None if absent)."""
    model = MODEL_BY_ID[model_id]
    spf = 10 * 2 ** tables.MU[scs]
    pattern = TDD_PATTERNS[scs] if duplex == "tdd" else None
    grid = build_carrier_grid(model, nrb, scs, duplex, ncell, subframes, data_convention=convention, pn_scope=scope).values

    def n_dl(slot):
        return 14 if pattern is None else pattern.dl_symbols_of_slot(slot % pattern.slots)

    def read(slot, rnti):
        for alloc in spec.slot_allocations(model, nrb, slot % spf, n_dl(slot)):
            if alloc.rnti == rnti:
                pos = data_positions(alloc.prbs.tolist(), alloc.sym_start, alloc.sym_end)
                values = np.array([grid[k, 14 * slot + l] for k, l in pos]) / 10 ** (alloc.power_db / 20)
                bits = demap(values, alloc.qm)
                return bits ^ gold(scrambling_cinit_pdsch(rnti, ncell), len(bits))
        return None

    def pdcch(slot):
        k_data = [k for k in range(36) if k % 12 not in (1, 5, 9)]
        values = np.array([grid[k, 14 * slot + l] for l in (0, 1) for k in k_data])
        return demap(values, 2) ^ gold(scrambling_cinit_pdcch(0, ncell), 108)

    return grid, read, pdcch


def test_standard_convention_fdd_streams_are_per_user_and_continuous():
    _, read, pdcch = reader("NR-FR1-TM1.1", 25, 15, "fdd", 7, 10, "standard")
    big = pn23(100_000)
    offset = {0: 0, 2: 0}
    for slot in range(10):
        for rnti in (0, 2):
            bits = read(slot, rnti)
            assert np.array_equal(bits, big[offset[rnti]:offset[rnti] + len(bits)]), (slot, rnti)
            offset[rnti] += len(bits)
        assert np.array_equal(pdcch(slot), big[108 * slot:108 * (slot + 1)])
    assert offset[0] == 10 * 2 * (22 * 12 * 14 - 2 * 22 * 6)         # QPSK, 22 PRBs, 14 symbols, DM-RS on the even half of 2 symbols
    assert offset[2] == 10 * 2 * (3 * 12 * 12 - 2 * 3 * 6)           # 3 PRBs, symbols 2...13


def test_conventions_agree_in_one_fdd_frame_of_single_object_models_and_differ_for_tm2():
    for model_id in ("NR-FR1-TM1.1", "NR-FR1-TM1.2", "NR-FR1-TM3.1", "NR-FR1-TM3.2", "NR-FR1-TM3.3"):
        a = build_carrier_grid(MODEL_BY_ID[model_id], 52, 15, "fdd", 3, 10, data_convention="standard").values
        b = build_carrier_grid(MODEL_BY_ID[model_id], 52, 15, "fdd", 3, 10, data_convention="matlab").values
        assert np.array_equal(a, b), model_id
    a = build_carrier_grid(MODEL_BY_ID["NR-FR1-TM2"], 52, 15, "fdd", 3, 10, data_convention="standard").values
    b = build_carrier_grid(MODEL_BY_ID["NR-FR1-TM2"], 52, 15, "fdd", 3, 10, data_convention="matlab").values
    assert np.array_equal(a[:, :14], b[:, :14])              # slot 0: both streams start from the all-ones state
    assert not np.array_equal(a[:, 14:28], b[:, 14:28])      # slot 1: the standard stream continues, the MATLAB class-1 stream starts


def test_tm2_standard_uses_one_stream_matlab_one_per_position_class():
    big = pn23(100_000)
    for convention in ("standard", "matlab"):
        _, read, _ = reader("NR-FR1-TM2", 25, 15, "fdd", 1, 10, convention)
        offsets = [0, 0, 0] if convention == "matlab" else [0]
        for slot in range(10):
            bits = read(slot, 2)
            idx = slot % 3 if convention == "matlab" else 0
            assert np.array_equal(bits, big[offsets[idx]:offsets[idx] + len(bits)]), (convention, slot)
            offsets[idx] += len(bits)
        assert len(bits) == 6 * (12 * 12 - 2 * 6)            # one 64QAM PRB, symbols 2...13, two DM-RS symbols


def test_tm2_occupies_one_prb_per_slot_next_to_the_pdcch():
    nrb = 25
    grid = build_carrier_grid(MODEL_BY_ID["NR-FR1-TM2"], nrb, 15, "fdd", 1, 3).values
    for slot in range(3):
        prb = (0, nrb // 2, nrb - 1)[slot]
        block = grid[:, 14 * slot + 2:14 * slot + 14]
        active = np.flatnonzero(np.any(block != 0, axis=1))
        assert active.tolist() == list(range(12 * prb, 12 * prb + 12))
        assert np.all(np.any(grid[:36, 14 * slot:14 * slot + 2] != 0, axis=1))          # PDCCH in RB 0-2 regardless of the data PRB
        assert not np.any(grid[36:, 14 * slot:14 * slot + 2])


def test_standard_convention_restarts_every_frame_matlab_does_not():
    model = MODEL_BY_ID["NR-FR1-TM1.1"]
    std = build_carrier_grid(model, 25, 15, "fdd", 2, 20, data_convention="standard").values
    assert np.array_equal(std[:, :140], std[:, 140:])
    mat = build_carrier_grid(model, 25, 15, "fdd", 2, 20, data_convention="matlab").values
    assert np.array_equal(mat[:, :140], std[:, :140]) and not np.array_equal(mat[:, :140], mat[:, 140:])
    assert np.array_equal(mat[:, 140:] != 0, mat[:, :140] != 0)                        # same occupancy, different bits
    # a TDD capture of two radio frames is two identical frames in the text's reading
    tdd = build_carrier_grid(MODEL_BY_ID["NR-FR1-TM3.2"], 52, 30, "tdd", 2, 20, data_convention="standard").values
    assert np.array_equal(tdd[:, :280], tdd[:, 280:])


def test_pn_scope_dl_symbols_versus_all_symbols_in_tdd():
    big = pn23(300_000)
    starts = {}
    for scope in ("dl_symbols", "all_symbols"):
        grid, read, pdcch = reader("NR-FR1-TM1.1", 25, 15, "tdd", 1, 20, "standard", scope)
        for rnti in (0, 2):
            sizes = [0 if read(s, rnti) is None else len(read(s, rnti)) for s in range(10)]
            full, special = sizes[0], sizes[3]
            assert sizes == [full, full, full, special, 0, full, full, full, special, 0] and 0 < special < full
            for frame in range(2):
                position = 0
                for s in range(10):
                    bits = read(10 * frame + s, rnti)
                    if scope == "all_symbols":
                        position = s * full                          # every slot, UL included, draws a whole slot of bits
                    if bits is not None:
                        assert np.array_equal(bits, big[position:position + len(bits)]), (scope, rnti, frame, s)
                        starts[(scope, rnti, s)] = position
                    if scope == "dl_symbols":
                        position += sizes[s]
        for frame in range(2):
            index = 0
            for s in range(10):
                if s % 5 == 4:                                       # uplink slot: nothing at all
                    assert not np.any(grid[:, 14 * (10 * frame + s):14 * (10 * frame + s + 1)])
                    continue
                assert np.array_equal(pdcch(10 * frame + s), big[108 * index:108 * (index + 1)]), (scope, frame, s)
                index += 1
    assert starts[("dl_symbols", 0, 5)] != starts[("all_symbols", 0, 5)]
    assert starts[("dl_symbols", 0, 0)] == starts[("all_symbols", 0, 0)] == 0


def test_matlab_tdd_uses_a_separate_stream_for_special_slots_and_never_restarts():
    _, read, pdcch = reader("NR-FR1-TM1.1", 25, 15, "tdd", 1, 20, "matlab")
    big = pn23(100_000)
    for rnti in (0, 2):
        full, special = len(read(0, rnti)), len(read(3, rnti))
        for n, slot in enumerate((0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17)):
            assert np.array_equal(read(slot, rnti), big[n * full:(n + 1) * full]), (rnti, slot)
        for n, slot in enumerate((3, 8, 13, 18)):
            assert np.array_equal(read(slot, rnti), big[n * special:(n + 1) * special]), (rnti, slot)
    for n, slot in enumerate((0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18)):
        assert np.array_equal(pdcch(slot), big[108 * n:108 * (n + 1)]), slot


def test_tdd_slot_and_symbol_activity():
    for scs, nrb in ((15, 25), (30, 24), (60, 11)):
        pattern = TDD_PATTERNS[scs]
        grid = build_carrier_grid(MODEL_BY_ID["NR-FR1-TM3.2"], nrb, scs, "tdd", 1, 20).values
        slots = grid.shape[1] // 14
        assert slots == 20 * 2 ** tables.MU[scs]
        dl_total = 0
        for slot in range(slots):
            n_dl = pattern.dl_symbols_of_slot(slot % pattern.slots)
            activity = [bool(np.any(grid[:, 14 * slot + l])) for l in range(14)]
            assert activity == [l < n_dl for l in range(14)], (scs, slot)
            dl_total += n_dl
        assert Fraction(dl_total, slots * 14) == Fraction(26, 35)


def test_pdcch_geometry_and_levels():
    grid = build_carrier_grid(MODEL_BY_ID["NR-FR1-TM1.2"], 52, 15, "fdd", 5, 1).values
    pdcch = grid[:36, :2]
    assert np.count_nonzero(pdcch) == 72                                 # 54 data + 18 DM-RS REs: RB 0-2, symbols 0-1 are full
    assert np.allclose(np.abs(pdcch), 1.0)                               # the PDCCH and its DM-RS sit at 0 dB
    dmrs = [k for k in range(36) if k % 12 in (1, 5, 9)]
    assert len(dmrs) == 9
    assert np.all(np.abs(grid[36:, 0]) > 0) and np.all(np.abs(grid[36:, 1]) > 0)       # n_RNTI 0 / 1 start at symbol 0
    assert np.allclose(np.abs(grid[:36, 2:14]), 1.0)                                   # n_RNTI 2: PRB 0-2, symbols 2...13, 0 dB


def test_boosted_rbgs_levels_and_power_conservation_in_the_grid():
    for model_id, nrb in (("NR-FR1-TM1.2", 52), ("NR-FR1-TM3.2", 106), ("NR-FR1-TM3.3", 25)):
        model = MODEL_BY_ID[model_id]
        grid = build_carrier_grid(model, nrb, 15, "fdd", 1, 1).values
        levels = spec.model_levels(model, nrb)
        target = set(spec.target_prbs(nrb, model.x_percent).tolist())
        other = 0 if model.target_rnti == 1 else 1
        power = 0.0
        for prb in range(3, nrb):
            expected = levels[model.target_rnti] if prb in target else levels[other]
            rnti = model.target_rnti if prb in target else other
            if model.user(rnti).qm == 2:                                 # every QPSK RE has exactly the level of its user
                assert np.allclose(np.abs(grid[12 * prb:12 * prb + 12, 3:11]), 10 ** (expected / 20), rtol=1e-12)
            dm_re = abs(grid[12 * prb + 4, 2])                           # DM-RS (even subcarrier, first DM-RS symbol) follows the PDSCH level
            assert dm_re == pytest.approx(10 ** (expected / 20), rel=1e-12)
            power += 10 ** (expected / 10)
        assert power == pytest.approx(nrb - 3, abs=1e-9)
        assert levels[model.target_rnti] == model.target_level_db


def test_two_layers_follow_the_layer_mapping_of_38_211():
    model = MODEL_BY_ID["NR-FR1-TM1.1"]
    nrb = 25
    g0 = build_carrier_grid(model, nrb, 15, "fdd", 4, 2, layers=2, output_port=0).values
    g1 = build_carrier_grid(model, nrb, 15, "fdd", 4, 2, layers=2, output_port=1).values
    single = build_carrier_grid(model, nrb, 15, "fdd", 4, 2).values
    assert np.array_equal(g0[:36, :2], single[:36, :2]) and not np.any(g1[:36, :2])      # the PDCCH is single-port: port 0 only
    alloc = next(a for a in spec.slot_allocations(model, nrb, 0, 14) if a.rnti == 0)
    pos = data_positions(alloc.prbs.tolist(), alloc.sym_start, alloc.sym_end)
    d0 = demap(np.array([g0[k, l] for k, l in pos]), 2).reshape(-1, 2)
    d1 = demap(np.array([g1[k, l] for k, l in pos]), 2).reshape(-1, 2)
    interleaved = np.empty((2 * len(d0), 2), dtype=np.uint8)
    interleaved[0::2], interleaved[1::2] = d0, d1                       # x(i)(j) = d(2j + i): layer 0 carries the even symbols of d
    scrambled = interleaved.reshape(-1)
    bits = scrambled ^ gold(scrambling_cinit_pdsch(0, 4), len(scrambled))
    assert np.array_equal(bits, pn23(len(bits)))                         # one codeword of twice the single-layer size, from one PN23 stream
    assert len(bits) == 2 * 2 * (22 * 12 * 14 - 2 * 22 * 6)
    for k in range(12 * 3, 12 * 4, 2):                                   # DM-RS of port 1001: same comb, w_f = (+1, -1) over m = k / 2
        assert g1[k, 2] == pytest.approx(g0[k, 2] * (1 if (k // 2) % 2 == 0 else -1), abs=1e-15)
    assert np.count_nonzero(g1) == np.count_nonzero(g0) - 72 * 2        # the second port has no PDCCH (72 REs per slot, 2 slots)


def test_overlay_hook_modifies_the_grid_and_excludes_resource_elements_from_the_reference():
    calls = []
    rows, cols = slice(120, 132), (3, 4)

    def overlay(slot_grid, slot_in_frame):
        calls.append(slot_in_frame)
        out = slot_grid.copy()
        out[rows, cols[0]:cols[1] + 1] = 5.0
        excluded = np.zeros(out.shape, dtype=bool)
        excluded[rows, cols[0]:cols[1] + 1] = True
        return out, excluded

    model = MODEL_BY_ID["NR-FR1-TM1.1"]
    plain = build_carrier_grid(model, 25, 15, "fdd", 1, 3)
    hooked = build_carrier_grid(model, 25, 15, "fdd", 1, 3, overlay=overlay)
    assert calls == [0, 1, 2]
    touched = np.zeros(plain.values.shape, dtype=bool)
    for slot in range(3):
        touched[rows, 14 * slot + cols[0]:14 * slot + cols[1] + 1] = True
    assert np.all(hooked.values[touched] == 5.0)
    assert np.array_equal(hooked.values[~touched], plain.values[~touched])
    assert np.all(plain.data_mask[touched]) and not np.any(hooked.data_mask[touched])
    assert np.array_equal(hooked.data_mask[~touched], plain.data_mask[~touched])
    # a hook that returns just a grid leaves the reference mask alone
    doubled = build_carrier_grid(model, 25, 15, "fdd", 1, 1, overlay=lambda g, s: g * 2)
    reference = build_carrier_grid(model, 25, 15, "fdd", 1, 1)
    assert np.allclose(doubled.values, 2 * reference.values) and np.array_equal(doubled.data_mask, reference.data_mask)
