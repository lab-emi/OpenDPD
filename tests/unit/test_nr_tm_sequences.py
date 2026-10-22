"""PN23, Gold, QAM and DM-RS sequences against literal implementations of the defining equations."""
import numpy as np
import pytest

from opendpd.core.waveforms.nr_tm import sequences as seq
from opendpd.core.waveforms.nr_tm import tables


def bits(a):
    return "".join(str(int(v)) for v in a)


def test_pn23_known_answers_and_recurrence():
    assert bits(seq.pn23(64)) == "1111111111111111111111100000000000000000011111000000000000011111"
    x = seq.pn23(200_000)
    assert x[:23].all()
    assert np.array_equal(x[23:], x[:-23] ^ x[5:-18])               # s(i) = s(i-23) xor s(i-18)
    literal = [1] * 23
    for i in range(23, 3000):
        literal.append(literal[i - 23] ^ literal[i - 18])
    assert np.array_equal(seq.pn23(3000), np.array(literal, dtype=np.uint8))
    assert len(seq.pn23(0)) == 0 and len(seq.pn23(5)) == 5


def test_pn23_is_maximal_length():
    period = 2 ** 23 - 1
    x = seq.pn23(period + 64)
    assert np.array_equal(x[:64], x[period:period + 64])
    assert int(x[:period].sum()) == 2 ** 22                        # a maximal-length sequence has 2^22 ones per period
    assert not np.array_equal(x[:64], x[period // 3:period // 3 + 64])


def gold_literal(c_init, n):
    """TS 38.211 5.2.1 bit by bit."""
    x1 = [1] + [0] * 30
    x2 = [(c_init >> i) & 1 for i in range(31)]
    for m in range(n + 1600):
        x1.append((x1[m + 3] + x1[m]) % 2)
        x2.append((x2[m + 3] + x2[m + 2] + x2[m + 1] + x2[m]) % 2)
    return np.array([(x1[m + 1600] + x2[m + 1600]) % 2 for m in range(n)], dtype=np.uint8)


@pytest.mark.parametrize("c_init", [0, 1, 7, 65537, 123456789, 0x1234567, 2 ** 31 - 1, 1179650])
def test_gold_matches_the_defining_recursion(c_init):
    assert np.array_equal(seq.gold(c_init, 2500), gold_literal(c_init, 2500))


def test_gold_known_answers():
    assert bits(seq.gold(1, 64)) == "0000001010000011000000110111010000101011100110101111110111100010"
    assert bits(seq.gold(0x1234567, 64)) == "0000011001010011100001101001010001011010111101011001000010100111"
    assert bits(seq.gold(2 ** 31 - 1, 64)) == "1111110100001011111100111000111000101110011000000101011110001110"
    assert bits(seq.gold(1179650, 16)) == "0000111000000100"          # PDSCH DM-RS, slot 0, l = 2, N_ID = 1
    assert bits(seq.gold(393218, 16)) == "1110010010101101"           # PDCCH DM-RS, slot 0, l = 0, N_ID = 1
    assert bits(seq.gold(65537, 32)) == "10001010000001000000010001110011"   # PDSCH scrambling n_RNTI 2, n_ID 1
    assert bits(seq.gold(1, 32)) == "00000010100000110000001101110100"      # n_RNTI 0, n_ID 1


def test_gold_is_a_prefix_code_and_long_sequences_are_fast():
    a, b = seq.gold(99, 1000), seq.gold(99, 5000)
    assert np.array_equal(a, b[:1000])
    assert len(seq.gold(5, 3_000_000)) == 3_000_000


def test_scrambling_and_dmrs_initialisation():
    assert seq.scrambling_cinit_pdsch(2, 1) == 65537 and seq.scrambling_cinit_pdsch(0, 1007) == 1007
    assert seq.scrambling_cinit_pdsch(1, 5) == (1 << 15) + 5
    assert seq.scrambling_cinit_pdcch(0, 503) == 503 and seq.scrambling_cinit_pdcch(1, 3) == (1 << 16) + 3
    assert seq.dmrs_cinit_pdsch(0, 2, 1) == 1179650 and seq.dmrs_cinit_pdsch(0, 11, 1) == 4718594
    assert seq.dmrs_cinit_pdcch(0, 0, 1) == 393218 and seq.dmrs_cinit_pdcch(0, 1, 1) == 786434
    assert seq.dmrs_cinit_pdsch(19, 13, 1007) == ((1 << 17) * (14 * 19 + 14) * 2015 + 2014) % (1 << 31)
    assert seq.dmrs_cinit_pdsch(3, 2, 9, 1) == seq.dmrs_cinit_pdsch(3, 2, 9) + 1
    r = seq.dmrs_pdsch(0, 2, 1, 4) * np.sqrt(2)
    assert np.allclose(r, [1 + 1j, 1 + 1j, -1 - 1j, -1 + 1j], atol=1e-15)
    a = seq.dmrs_pdsch(5, 2, 7, 600)
    assert np.allclose(np.abs(a), 1.0) and np.allclose(abs(a) ** 2, 1)
    assert len(seq.dmrs_pdcch(2, 1, 7, 9)) == 9


def literal_qam(b):
    """TS 38.211 5.1.3 ... 5.1.7 written out per modulation order (s(x) = 1 - 2x)."""
    s = [1 - 2 * int(v) for v in b]
    qm = len(b)
    if qm == 2:
        return (s[0] + 1j * s[1]) / np.sqrt(2)
    if qm == 4:
        return (s[0] * (2 - s[2]) + 1j * s[1] * (2 - s[3])) / np.sqrt(10)
    if qm == 6:
        return (s[0] * (4 - s[2] * (2 - s[4])) + 1j * s[1] * (4 - s[3] * (2 - s[5]))) / np.sqrt(42)
    if qm == 8:
        return (s[0] * (8 - s[2] * (4 - s[4] * (2 - s[6]))) + 1j * s[1] * (8 - s[3] * (4 - s[5] * (2 - s[7])))) / np.sqrt(170)
    return (s[0] * (16 - s[2] * (8 - s[4] * (4 - s[6] * (2 - s[8])))) + 1j * s[1] * (16 - s[3] * (8 - s[5] * (4 - s[7] * (2 - s[9]))))) / np.sqrt(682)


@pytest.mark.parametrize("qm", [2, 4, 6, 8, 10])
def test_qam_matches_the_standard_and_has_unit_power(qm):
    patterns = ((np.arange(1 << qm)[:, None] >> np.arange(qm - 1, -1, -1)[None, :]) & 1).astype(np.uint8)
    points = seq.qam_map(patterns.reshape(-1), qm)
    expected = np.array([literal_qam(p) for p in patterns])
    assert np.max(np.abs(points - expected)) < 1e-14
    assert abs(np.mean(np.abs(points) ** 2) - 1) < 1e-12
    assert len(np.unique(np.round(points, 9))) == 1 << qm


@pytest.mark.parametrize("qm", [4, 6, 8, 10])
def test_qam_is_gray_mapped(qm):
    patterns = ((np.arange(1 << qm)[:, None] >> np.arange(qm - 1, -1, -1)[None, :]) & 1).astype(np.uint8)
    points = seq.qam_map(patterns.reshape(-1), qm)
    step = np.min(np.abs(np.diff(np.sort(points.real))[np.diff(np.sort(points.real)) > 1e-9]))
    for i in range(0, 1 << qm, max(1, (1 << qm) // 64)):
        neighbours = np.flatnonzero(np.abs(points - points[i]) < step * 1.01)
        for j in neighbours:
            if j != i:
                assert int(np.sum(patterns[i] != patterns[j])) == 1


def test_qam_normalisation_and_even_odd_bit_split():
    # b0, b2, ... only change I; b1, b3, ... only change Q
    for qm in (2, 4, 6, 8, 10):
        base = np.zeros(qm, dtype=np.uint8)
        for bit in range(qm):
            flipped = base.copy()
            flipped[bit] = 1
            a, b = seq.qam_map(base, qm)[0], seq.qam_map(flipped, qm)[0]
            assert (a.imag == b.imag) if bit % 2 == 0 else (a.real == b.real)
    assert seq.qam_map(np.array([0, 0]), 2)[0] == complex(np.sqrt(0.5), np.sqrt(0.5))


def test_dmrs_positions_follow_table_7_4_1_1_2_3():
    expected = {3: (2,), 4: (2,), 5: (2,), 6: (2,), 7: (2,), 8: (2, 7), 9: (2, 7), 10: (2, 9), 11: (2, 9), 12: (2, 9), 13: (2, 11), 14: (2, 11)}
    for l_d, positions in expected.items():
        assert tables.dmrs_symbols(l_d) == positions
    assert tables.dmrs_symbols(14) == (2, 11) and tables.dmrs_symbols(10) == (2, 9)       # full slot, 15 kHz special slot
    assert tables.dmrs_symbols(6) == (2,) and tables.dmrs_symbols(12) == (2, 9)           # 30 kHz and 60 kHz special slots
    with pytest.raises(ValueError):
        tables.dmrs_symbols(2)


def test_pn_streams_are_independent_continuous_and_restartable():
    streams = seq.PnStreams()
    a1, b1 = streams.take("a", 100), streams.take("b", 50)
    a2 = streams.take("a", 100)
    reference = seq.pn23(200)
    assert np.array_equal(a1, reference[:100]) and np.array_equal(b1, reference[:50]) and np.array_equal(a2, reference[100:200])
    streams.restart()
    assert np.array_equal(streams.take("a", 30), reference[:30])
    assert len(streams.take("long", 300_000)) == 300_000
