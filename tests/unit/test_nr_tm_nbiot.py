"""NR-N-TM: the NB-IoT test model content (TS 36.141 6.1.3-6.1.6, TS 36.211 10.2) and its NR host overlay.

The numerical reference is ``tests/data/nr_tm/nbiot_oracle.json``: numbers produced by MATLAB R2026a LTE Toolbox NB-IoT functions
(no MathWorks code in the repository) for 13 (N_ID, host cell ID, operation mode) combinations, both RE budgets.
"""
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from opendpd.core.waveforms.nr_tm import nb_iot as nb

ORACLE = json.loads((Path(__file__).resolve().parents[1] / "data" / "nr_tm" / "nbiot_oracle.json").read_text())
CASE_NAMES = sorted(ORACLE["cases"])
TOL = 1e-12
TOL_NSSS = 2e-11    # MATLAB's NSSS values carry an argument-reduction error of up to 8e-12; the module itself is exact (below)

# TS 38.104 Table 5.3.2-1, 15 kHz column, and the table of nbiot_ntm.md 5.2: BW -> (N_RB, L_NB-IoT, R_NB-IoT, #eligible)
NR_15KHZ = {3: (15, 2, 12, 3), 5: (25, 2, 22, 5), 10: (52, 0, 50, 11), 15: (79, 4, 74, 15), 20: (106, 2, 102, 21),
            25: (133, 1, 131, 27), 30: (160, 4, 159, 32), 35: (188, 3, 183, 37), 40: (216, 2, 212, 43),
            45: (242, 0, 240, 49), 50: (270, 4, 269, 54)}


def bits(hexmask):
    m = int(hexmask, 16)
    return [i for i in range(nb.N_SYMB * nb.N_SC) if m >> i & 1]


def qpsk_codes(codes):
    return np.array([((1 - 2 * (int(c) & 1)) + 1j * (1 - 2 * (int(c) >> 1))) / math.sqrt(2) for c in codes])


def cplx(d):
    return np.array(d["re"]) + 1j * np.array(d["im"])


def n_re(mask):
    return [int(mask[sf].sum()) for sf in range(10)]


# ---------------------------------------------------------------------------------------------------------------------
# MATLAB oracle
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("budget", ["1", "2"])
@pytest.mark.parametrize("name", CASE_NAMES)
def test_frame_matches_matlab_oracle(name, budget):
    c = ORACLE["cases"][name]
    values, mask = nb.nb_iot_frame(c["n_id"], mode=c["mode"], host_cell_id=c["host_cell_id"], nrs_ports=int(budget))
    assert values.shape == mask.shape == (10, 14, 12)
    for sf, e in enumerate(c["budgets"][budget]["sf"]):
        fv, fm = values[sf].reshape(-1), mask[sf].reshape(-1)       # flat index = l * 12 + k
        nrs = e.get("nrs", [])
        nrs_idx = [i for i, _ in nrs]
        ch_idx = bits(e["ch"])
        assert sorted(nrs_idx + ch_idx) == list(np.flatnonzero(fm)), f"RE set of subframe {sf}"
        if nrs:
            assert len(nrs) == 8
            np.testing.assert_allclose(fv[nrs_idx], qpsk_codes([c for _, c in nrs]), atol=TOL, rtol=0)
        if "npss" in e:
            np.testing.assert_allclose(fv[ch_idx], cplx(e["npss"]), atol=TOL, rtol=0)
        elif "nsss" in e:
            np.testing.assert_allclose(fv[ch_idx], cplx(e["nsss"]), atol=TOL_NSSS, rtol=0)
        elif "sym" in e:
            np.testing.assert_allclose(fv[ch_idx], qpsk_codes(e["sym"]), atol=TOL, rtol=0)
        if budget == "2" and nrs:                                    # muted NRS port 2001: no NB-IoT signal there
            assert not fm[e["nrs1"]].any()
            assert sorted(e["nrs1"]) == sorted(l * 12 + k for k, l, _, _ in nb.nrs_positions(c["n_id"] % 6, 2001))


def test_oracle_covers_the_cases_of_the_report():
    assert {"sa103", "ib103h1", "ib110h2", "ib208h16", "ib500h7", "ib103h0"} <= set(CASE_NAMES)
    assert len(ORACLE["nsss_sweep"]) == 16
    # every N_ID q-range and the different-PCI (host v_shift != N_ID mod 6) situation is present
    assert {c["n_id"] // 126 for c in ORACLE["cases"].values()} == {0, 1, 3}
    assert any(c["mode"] == "inband" and c["n_id"] % 6 != c["host_cell_id"] % 6 for c in ORACLE["cases"].values())
    assert (Path(__file__).resolve().parents[1] / "data" / "nr_tm" / "nbiot_oracle.json").stat().st_size < 300_000


def test_nsss_sweep_matches_matlab_oracle():
    q_seen = set()
    for e in ORACLE["nsss_sweep"]:
        np.testing.assert_allclose(nb.nsss_sequence(e["n_id"], e["n_f"]), cplx(e), atol=TOL_NSSS, rtol=0)
        q_seen.add(e["n_id"] // 126)
    assert q_seen == {0, 1, 2, 3}


def test_nsss_is_exact_against_high_precision_evaluation():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    for n_id, n_f in ((0, 0), (103, 0), (125, 2), (126, 4), (300, 6), (503, 8)):
        d = nb.nsss_sequence(n_id, n_f)
        u, b = n_id % 126 + 3, [1 if ch == "+" else -1 for ch in nb._B_Q[n_id // 126]]
        theta = mp.mpf(33) / 132 * ((n_f // 2) % 4)
        for n in range(132):
            n1 = n % 131
            ref = b[n % 128] * mp.exp(-2j * mp.pi * theta * n) * mp.exp(-1j * mp.pi * u * n1 * (n1 + 1) / 131)
            assert abs(mp.mpc(d[n].real, d[n].imag) - ref) < 1e-14


# ---------------------------------------------------------------------------------------------------------------------
# frame structure, counts, closed forms
# ---------------------------------------------------------------------------------------------------------------------
def test_inband_budget_counts_per_subframe_and_frame():
    values, mask = nb.nb_iot_frame(103)                      # in-band, 2-port budget, host v_shift 1
    assert n_re(mask) == [108, 108, 108, 108, 108, 105, 108, 108, 108, 116]
    assert int(mask.sum()) == 1085
    # sums of the parts: NPBCH 100 + NRS 8; NPDCCH 100 + 8; 6 x (NPDSCH 100 + 8); NPSS 105; NSSS 116
    assert 8 * 108 + 105 + 116 == 1085


@pytest.mark.parametrize("host", range(0, 8))
def test_inband_count_depends_on_host_v_shift_only_through_the_npss(host):
    n_id = nb.nb_iot_n_id(host, 1)
    _, mask = nb.nb_iot_frame(n_id, host_cell_id=host)
    # the NPSS has no subcarrier 11, so a host CRS RE at k = 11 (v_shift 2 or 5) is not part of it
    assert n_re(mask)[5] == (109 if host % 6 in (2, 5) else 105)
    assert int(mask.sum()) == (1089 if host % 6 in (2, 5) else 1085)


def test_other_budgets():
    assert int(nb.nb_iot_frame(103, mode="standalone")[1].sum()) == 1481                      # 108 + 7 * 160 + 121 + 132
    assert int(nb.nb_iot_frame(103, nrs_ports=1)[1].sum()) == 1141                            # natural one-port in-band budget
    assert int(nb.nb_iot_frame(103, mode="standalone", nrs_ports=1)[1].sum()) == 1537
    _, m = nb.nb_iot_frame(103, mode="standalone")
    assert n_re(m) == [108, 160, 160, 160, 160, 121, 160, 160, 160, 132]


def test_subframe_map_and_symbol_ranges():
    values, mask = nb.nb_iot_frame(110, host_cell_id=2)
    assert not mask[:, :3, :].any(), "in-band: the first three symbols stay with the host"
    assert not values[~mask].any()
    assert (np.abs(values[mask]) > 0).all()
    np.testing.assert_allclose(np.abs(values[mask]), 1.0, atol=1e-12)                        # unit RE amplitude
    # NPSS spans subcarriers 0..10 only; NSSS and the data REs use all 12; subframe 5 and 9 carry no NRS
    assert not mask[5, :, 11].any() and mask[9, :, 11].any()
    n_id = 110
    nrs_kl = {(k, l) for k, l, _, _ in nb.nrs_positions(n_id % 6, 2000)}
    crs = nb.crs_positions(2, 4)
    for sf in (0, 1, 2, 3, 4, 6, 7, 8):
        assert all(mask[sf, l, k] for k, l in nrs_kl)
    # NPSS/NSSS subframes: exactly the synchronisation REs (no NRS is added on top of them)
    assert {(k, l) for l in range(14) for k in range(12) if mask[5, l, k]} == {
        (k, l) for l in range(3, 14) for k in range(11) if (k, l) not in crs}
    assert {(k, l) for l in range(14) for k in range(12) if mask[9, l, k]} == {
        (k, l) for l in range(3, 14) for k in range(12) if (k, l) not in crs}


def test_nrs_positions_closed_form():
    for v in range(6):
        for port, (v5, v6) in ((2000, (0, 3)), (2001, (3, 0))):
            pos = nb.nrs_positions(v, port)
            assert len(pos) == 8
            for k, l, slot, m in pos:
                assert l in (5, 6, 12, 13) and slot == (l >= 7)
                assert k == 6 * m + ((v5 if l % 7 == 5 else v6) + v) % 6
    # CRS of 4 ports in a PRB occupy 4 symbols x 4 subcarriers (l = 0, 4, 7, 11 for ports 0/1 and 1, 8 for ports 2/3), 6 subcarrier comb
    crs = nb.crs_positions(1, 4)
    assert len(crs) == 24 and {l for _, l in crs} == {0, 1, 4, 7, 8, 11}
    assert nb.crs_positions(1, 2) == {(k, l) for k, l in crs if l in (0, 4, 7, 11)}


def test_qpsk_mapping_is_the_table_of_36_211():
    s = 1 / math.sqrt(2)
    got = nb.qpsk([0, 0, 0, 1, 1, 0, 1, 1])
    np.testing.assert_allclose(got, [s + 1j * s, s - 1j * s, -s + 1j * s, -s - 1j * s], atol=1e-15)


def test_gold_sequence_matches_literal_recurrences():
    def literal(c_init, n):
        x1 = [1] + [0] * 30
        x2 = [(c_init >> i) & 1 for i in range(31)]
        for i in range(1600 + n):
            x1.append((x1[i + 3] + x1[i]) % 2)
            x2.append((x2[i + 3] + x2[i + 2] + x2[i + 1] + x2[i]) % 2)
        return [(x1[i + 1600] + x2[i + 1600]) % 2 for i in range(n)]
    for c_init in (0, 1, 103, 2 ** 31 - 1, 1000 * 2 ** 14 + 5 * 2 ** 9 + 103, 123456789):
        assert nb.gold_sequence(c_init, 300).tolist() == literal(c_init, 300)
    with pytest.raises(ValueError):
        nb.gold_sequence(2 ** 31, 10)


def test_nsss_table_rows_are_hadamard_rows():
    h = np.array([[1]])
    for _ in range(7):
        h = np.block([[h, h], [h, -h]])
    for q, row in zip(range(4), (0, 31, 63, 127)):
        assert len(nb._B_Q[q]) == 128
        np.testing.assert_array_equal(h[row], [1 if ch == "+" else -1 for ch in nb._B_Q[q]])


def test_npss_and_nsss_closed_forms():
    d = nb.npss_symbols()
    s = [1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1]
    for l in range(3, 14):
        for n in range(11):
            assert abs(d[l - 3, n] - s[l - 3] * np.exp(-1j * np.pi * 5 * n * (n + 1) / 11)) < TOL
    for n_id in (0, 103, 125, 126, 251, 252, 377, 378, 503):
        seq = nb.nsss_sequence(n_id, 0)
        u, q = n_id % 126 + 3, n_id // 126
        b = [1 if ch == "+" else -1 for ch in nb._B_Q[q]]
        n = np.arange(132)
        ref = np.array([b[i % 128] * np.exp(-1j * np.pi * u * (i % 131) * (i % 131 + 1) / 131) for i in n])
        np.testing.assert_allclose(seq, ref, atol=1e-9, rtol=0)
        assert (np.abs(np.abs(seq) - 1) < 1e-14).all()
    # theta_f of the odd radio frame pairs: frames 0, 2, 4, 6 -> 0, 1/4, 2/4, 3/4 turn per sequence index, period 8 frames
    d0, d2 = nb.nsss_sequence(103, 0), nb.nsss_sequence(103, 2)
    np.testing.assert_allclose(d2 / d0, np.exp(-2j * np.pi * np.arange(132) / 4), atol=1e-12)
    np.testing.assert_allclose(nb.nsss_sequence(103, 8), d0, atol=1e-12)


def test_npss_nsss_skip_host_crs_but_count_the_sequence_elements():
    values, mask = nb.nb_iot_frame(103)                       # host v_shift 1
    crs = nb.crs_positions(1, 4)
    d = nb.nsss_sequence(103, 0)
    for l in range(3, 14):
        for k in range(12):
            assert mask[9, l, k] == ((k, l) not in crs)
            if (k, l) not in crs:
                assert abs(values[9, l, k] - d[12 * (l - 3) + k]) < 1e-12      # index counted over the skipped elements too
    npss = nb.npss_symbols()
    for l in range(3, 14):
        for k in range(11):
            assert mask[5, l, k] == ((k, l) not in crs)
            if (k, l) not in crs:
                assert abs(values[5, l, k] - npss[l - 3, k]) < 1e-12


def test_data_channels_use_the_36_141_scrambling_initialisation():
    # NPDSCH n_RNTI 1000: c_init = 1000 2^14 + floor(n_s/2) 2^9 + N_ID; NPDCCH: floor(n_s/2) 2^9 + N_ID with n_s = 2
    n_id = 117
    values, mask = nb.nb_iot_frame(n_id, host_cell_id=3)
    nrs = {(k, l) for k, l, _, _ in nb.nrs_positions(n_id % 6, 2000)}
    for sf, c_init in ((1, 1 * 2 ** 9 + n_id), (2, 1000 * 2 ** 14 + 2 * 2 ** 9 + n_id), (8, 1000 * 2 ** 14 + 8 * 2 ** 9 + n_id)):
        res = [(k, l) for l in range(14) for k in range(12) if mask[sf, l, k] and (k, l) not in nrs]
        assert len(res) == 100
        np.testing.assert_allclose([values[sf, l, k] for k, l in res], nb.qpsk(nb.gold_sequence(c_init, 200)), atol=1e-15)


def test_determinism_and_copy_semantics():
    a = nb.nb_iot_frame(110, host_cell_id=2)
    b = nb.nb_iot_frame(110, host_cell_id=2)
    assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()
    a[0][:] = 0
    a[1][:] = False
    c = nb.nb_iot_frame(110, host_cell_id=2)
    assert c[0].tobytes() == b[0].tobytes() and c[1].tobytes() == b[1].tobytes()
    assert not np.array_equal(nb.nb_iot_frame(103)[0], nb.nb_iot_frame(110)[0])
    assert nb.nb_iot_frame(103)[0].dtype == np.complex128 and nb.nb_iot_frame(103)[1].dtype == bool


def test_frame_argument_validation():
    for kwargs in ({"n_id": -1}, {"n_id": 504}, {"n_id": 103, "mode": "guard"}, {"n_id": 103, "nrs_ports": 3},
                   {"n_id": 103, "host_crs_ports": 3}, {"n_id": 103, "control_symbols": 5}, {"n_id": 103, "host_cell_id": -1},
                   {"n_id": 1.5}, {"n_id": True}):
        with pytest.raises(ValueError):
            nb.nb_iot_frame(**kwargs)


# ---------------------------------------------------------------------------------------------------------------------
# eligible RBs
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bw", sorted(NR_15KHZ))
def test_eligible_rb_table_of_the_research_report(bw):
    nrb, low, high, count = NR_15KHZ[bw]
    rbs = nb.eligible_rbs(nrb)
    assert (rbs[0], rbs[-1], len(rbs)) == (low, high, count)
    assert nb.lower_rb(nrb) == low and nb.upper_rb(nrb) == high
    # closed form of the comb: even N_RB: N/2 + 4 + 5 i, odd N_RB: floor(N/2) + 5 i
    base = nrb // 2 + 4 if nrb % 2 == 0 else nrb // 2
    assert rbs == [m for m in range(nrb) if (m - base) % 5 == 0]


def first_principles_eligible(nrb, f_ref_mod_khz=0):
    """Absolute subcarrier frequencies: F_REF = centre of RE k=0 of PRB N/2 (even N) or of RE k=6 of PRB floor(N/2) (odd N)."""
    k_ref = 12 * (nrb // 2) + (0 if nrb % 2 == 0 else 6)
    out = []
    for rb in range(nrb):
        centre = sum(Fraction(f_ref_mod_khz) + 15 * (12 * rb + i - k_ref) for i in range(12)) / 12
        r = centre % 100
        r = r - 100 if r > 50 else r
        if r in (Fraction(-15, 2), Fraction(-5, 2), Fraction(5, 2), Fraction(15, 2)):
            out.append(rb)
    return out


@pytest.mark.parametrize("nrb", [n for n, *_ in NR_15KHZ.values()])
def test_eligible_rbs_follow_from_the_anchor_raster_rule(nrb):
    assert nb.eligible_rbs(nrb) == first_principles_eligible(nrb)
    for f0 in (5, 15, 30, 45, 60, 75, 90, 100, 205):
        try:
            expected = first_principles_eligible(nrb, f0)
        except Exception:
            continue
        if expected:
            assert nb.eligible_rbs(nrb, f0) == expected
    assert nb.eligible_rbs(nrb, 100) == nb.eligible_rbs(nrb, 0)
    assert nb.eligible_rbs(nrb, 0.0) == nb.eligible_rbs(nrb, Fraction(0))


def test_eligible_rbs_other_raster_offsets_and_errors():
    with pytest.raises(ValueError, match="eligible"):
        nb.eligible_rbs(52, 1)                              # no PRB centre within the anchor offsets
    assert nb.eligible_rbs(52, 15) != nb.eligible_rbs(52, 0)            # a 15 kHz raster shifts the comb
    assert nb.eligible_rbs(52, 5) == nb.eligible_rbs(52, 0)             # the anchor offsets are 5 kHz apart
    for bad in (14, 276, 0, -3, True, 52.0, "52", None):
        with pytest.raises(ValueError):
            nb.eligible_rbs(bad)


def test_lte_cross_check_of_the_raster_rule_against_table_16_8_1():
    """The same anchor-raster predicate reproduces the E-UTRA PRB lists of TS 36.213 Table 16.8-1 for LTE hosts."""
    def lte_prb_centre(n, big_n):
        half = big_n * 12 // 2
        sub = [15 * ((kp - half) if kp < half else (kp - half + 1)) for kp in range(n * 12, n * 12 + 12)]
        return Fraction(sum(sub), 12)

    t_odd = list(range(-35, 0, 5)) + list(range(5, 40, 5))
    t_even = list(range(-46, -5, 5)) + list(range(5, 50, 5))
    known = {15: [2, 12], 25: [2, 7, 17, 22], 50: [4, 9, 14, 19, 30, 35, 40, 45]}
    for big_n in (15, 25, 50, 75, 100):
        got = [n for n in range(big_n) if nb.raster_offset_khz(lte_prb_centre(n, big_n)) in nb.ANCHOR_OFFSETS_KHZ
               and not (big_n // 2 - 3 <= n <= big_n // 2 + 2)]
        ref = [x for x in (t_odd if big_n % 2 else t_even) if -(big_n // 2) <= x < big_n - big_n // 2]
        assert [n - big_n // 2 for n in got] == sorted(ref)
        if big_n in known:
            assert got == known[big_n]


# ---------------------------------------------------------------------------------------------------------------------
# N_ID rule and power
# ---------------------------------------------------------------------------------------------------------------------
def test_n_id_rule_of_36_141():
    assert nb.nb_iot_n_id(1, 1) == 103            # lowest in-band carrier (host cell ID 1)
    assert nb.nb_iot_n_id(2, 2) == 110            # in-band carrier of the 2nd lowest host
    assert nb.nb_iot_n_id(3, 3) == 117
    assert nb.nb_iot_n_id(0, 1) == 103            # lowest stand-alone carrier
    assert nb.nb_iot_n_id(0, 2) == 109            # 2nd lowest stand-alone carrier
    assert nb.nb_iot_n_ids([1, 2, 3]) == [103, 110, 117]
    assert nb.nb_iot_n_ids([1, 2, 3, 4, 5, 6, 7, 8]) == [96 + 7 * i for i in range(1, 9)]
    assert [n % 6 for n in nb.nb_iot_n_ids([1, 2, 3, 4, 5, 6])] == [1, 2, 3, 4, 5, 0]   # == host cell ID mod 6
    assert nb.nb_iot_n_id(401, 1) == 503
    for args in ((402, 1), (1007, 1), (1, 0), (-1, 1), (1, 1.5), (True, 1), (1, True)):
        with pytest.raises(ValueError):
            nb.nb_iot_n_id(*args)
    with pytest.raises(ValueError):
        nb.nb_iot_n_ids([1, 2, 407])


def total_res(nrb):
    return nrb * 12 * 140


@pytest.mark.parametrize("nrb, x, nr_db, rel_db", [(52, 6, -0.166, 6.166), (15, 6, -0.625, 6.625), (270, 6, -0.031, 6.031)])
def test_power_split_numbers_quoted_in_the_research_report(nrb, x, nr_db, rel_db):
    nr_gain, nb_gain = nb.power_split(nrb, 1085, x)
    assert 20 * math.log10(nr_gain) == pytest.approx(nr_db, abs=6e-4)
    assert 20 * math.log10(nb_gain / nr_gain) == pytest.approx(rel_db, abs=6e-4)
    assert 20 * math.log10(nb_gain) == pytest.approx(x, abs=1e-12)


def test_power_split_closed_form_and_energy_conservation():
    for bw, (nrb, *_rest) in NR_15KHZ.items():
        for x in (3, 6, 9.5, 12):
            for n in (1085, 1089, 2170):
                if x >= nb.max_boost_db(nrb, n):
                    with pytest.raises(ValueError):
                        nb.power_split(nrb, n, x)
                    continue
                nr_gain, nb_gain = nb.power_split(nrb, n, x)
                g = 10 ** (x / 10)
                n_tot = total_res(nrb)
                assert nr_gain ** 2 == pytest.approx((n_tot - n * g) / (n_tot - n), rel=1e-13)
                assert nb_gain ** 2 == pytest.approx(g, rel=1e-13)
                assert nr_gain ** 2 * (n_tot - n) + nb_gain ** 2 * n == pytest.approx(n_tot, rel=1e-12)
                assert 0 < nr_gain < 1 < nb_gain
    rel = [20 * math.log10(b / a) for a, b in (nb.power_split(nrb, 1085, 3) for nrb, *_ in NR_15KHZ.values())]
    assert 3.01 <= min(rel) and max(rel) <= 3.20            # X = 3 dB: NB-IoT / NR between 3.01 and 3.20 dB at every bandwidth


def test_per_symbol_power_reading_stays_within_the_documented_distance():
    _, mask = nb.nb_iot_frame(103)
    per_symbol = mask.sum(axis=2).reshape(-1)
    g = 10 ** 0.6
    for nrb, low, high in ((15, -0.42, 0.63), (52, -0.10, 0.17), (270, -0.02, 0.03)):
        frame_db = 20 * math.log10(nb.power_split(nrb, int(mask.sum()), 6)[0])
        dev = [10 * math.log10((12 * nrb - n * g) / (12 * nrb - n)) - frame_db for n in per_symbol]
        assert min(dev) == pytest.approx(low, abs=0.01) and max(dev) == pytest.approx(high, abs=0.01)


def test_power_split_limits_and_validation():
    assert nb.max_boost_db(15, 1085) == pytest.approx(10 * math.log10(25200 / 1085))      # < 13.7 dB at 3 MHz
    assert 13.6 < nb.max_boost_db(15, 1085) < 13.7 and 19.0 < nb.max_boost_db(52, 1085) < 19.1
    with pytest.raises(ValueError, match="no power"):
        nb.power_split(15, 2170, 11)                     # both carriers at 3 MHz: limit is 10.65 dB
    nb.power_split(15, 2170, 10.5)
    for args in ((52, 0, 6), (52, total_res(52), 6), (52, 1085.5, 6), (52, 1085, float("nan")), (52, 1085, float("inf")), (14, 1085, 6)):
        with pytest.raises(ValueError):
            nb.power_split(*args)
    # monotonic: a larger boost lowers the NR level and raises the NB-IoT one
    a, b = nb.power_split(52, 1085, 3), nb.power_split(52, 1085, 9)
    assert b[0] < a[0] and b[1] > a[1]


# ---------------------------------------------------------------------------------------------------------------------
# overlay plan
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("position", ["lower", "upper", "both"])
@pytest.mark.parametrize("bw", sorted(NR_15KHZ))
def test_overlay_plan_positions_ids_and_gains(bw, position):
    nrb, low, high, _ = NR_15KHZ[bw]
    plan = nb.nb_iot_overlay_plan(nrb, position, 6, [1])
    assert plan.rbs == {"lower": [low], "upper": [high], "both": [low, high]}[position]
    assert [c.side for c in plan.carriers] == {"lower": ["lower"], "upper": ["upper"], "both": ["lower", "upper"]}[position]
    assert all(c.n_id == 103 for c in plan.carriers)
    assert plan.n_nb_re == 1085 * len(plan.carriers) == sum(c.n_re for c in plan.carriers)
    assert (plan.nr_gain, plan.nb_gain) == nb.power_split(nrb, plan.n_nb_re, 6)
    values, mask = nb.nb_iot_frame(103, host_cell_id=1)
    for c in plan.carriers:
        assert c.values.shape == c.mask.shape == (10, 14, 12)
        np.testing.assert_array_equal(c.values, values)
        np.testing.assert_array_equal(c.mask, mask)
        assert not c.values.flags.writeable and not c.mask.flags.writeable


def test_overlay_plan_host_ordinals_and_cell_ids():
    hosts = [1, 2, 3]
    for i, (nrb, n_id) in enumerate(zip((106, 52, 25), (103, 110, 117))):
        plan = nb.nb_iot_overlay_plan(nrb, "both", 6, hosts, host_index=i)
        assert {c.n_id for c in plan.carriers} == {n_id} and plan.host_cell_id == hosts[i]
        values, mask = nb.nb_iot_frame(n_id, host_cell_id=hosts[i])
        np.testing.assert_array_equal(plan.carriers[0].mask, mask)
        np.testing.assert_array_equal(plan.carriers[0].values, values)
    assert nb.nb_iot_overlay_plan(52, "lower", 6, 1).carriers[0].n_id == 103           # a bare int is one host
    assert nb.nb_iot_overlay_plan(52, "lower", 6, (np.int64(5),)).carriers[0].n_id == 97 + 6 + 4
    for bad_index in (-1, 3):
        with pytest.raises(ValueError):
            nb.nb_iot_overlay_plan(52, "lower", 6, hosts, host_index=bad_index)
    for bad_hosts in ([], None, "1", [1, "x"], 1.5):
        with pytest.raises(ValueError):
            nb.nb_iot_overlay_plan(52, "lower", 6, bad_hosts)
    with pytest.raises(ValueError):
        nb.nb_iot_overlay_plan(52, "middle", 6, [1])
    with pytest.raises(ValueError):
        nb.nb_iot_overlay_plan(52, "lower", 6, [402])                                  # N_ID would exceed 503


def test_overlay_plan_rb_override_and_f_ref():
    plan = nb.nb_iot_overlay_plan(52, "both", 6, [1], rbs=[25 + 5, 5])                 # eligible comb: 0, 5, ..., 50
    assert plan.rbs == [5, 30] and [c.side for c in plan.carriers] == ["lower", "upper"]
    for bad in ([6], [], [0, 6]):
        with pytest.raises(ValueError, match="eligible"):
            nb.nb_iot_overlay_plan(52, "both", 6, [1], rbs=bad)
    shifted = nb.nb_iot_overlay_plan(52, "both", 6, [1], f_ref_mod_khz=5)
    assert shifted.rbs == [nb.lower_rb(52, 5), nb.upper_rb(52, 5)] and shifted.f_ref_mod_khz == 5


def test_overlay_plan_rejects_a_boost_that_exhausts_the_power():
    with pytest.raises(ValueError, match="no power"):
        nb.nb_iot_overlay_plan(15, "both", 12, [1])
    nb.nb_iot_overlay_plan(15, "lower", 12, [1])


def test_overlay_slot_replaces_only_the_nb_iot_res_and_scales_the_rest():
    rng = np.random.default_rng(7)
    plan = nb.nb_iot_overlay_plan(52, "both", 6, [1])
    nr = rng.standard_normal((52 * 12, 14)) + 1j * rng.standard_normal((52 * 12, 14))
    nr_copy = nr.copy()
    for slot in (0, 1, 5, 9, 12):
        out = plan.overlay_slot(nr, slot)
        sf = slot % 10
        np.testing.assert_array_equal(nr, nr_copy)                                      # input untouched
        expect = nr * plan.nr_gain
        for c in plan.carriers:
            blk = expect[12 * c.rb:12 * c.rb + 12]
            sel = c.mask[sf].T
            blk[sel] = plan.nb_gain * c.values[sf].T[sel]
        np.testing.assert_allclose(out, expect, atol=1e-14, rtol=0)
        # outside the punctured RBs (and on the NR-content REs inside them) the NR grid is only scaled
        keep = np.ones(out.shape, bool)
        for c in plan.carriers:
            keep[12 * c.rb:12 * c.rb + 12][c.mask[sf].T] = False
        np.testing.assert_allclose(out[keep], plan.nr_gain * nr[keep], atol=1e-14, rtol=0)
    # control symbols l = 0..2 of the punctured RB keep NR content (resolution 1)
    c = plan.carriers[0]
    out = plan.overlay_slot(nr, 3)
    np.testing.assert_allclose(out[12 * c.rb:12 * c.rb + 12, :3], plan.nr_gain * nr[12 * c.rb:12 * c.rb + 12, :3], atol=1e-14)


@pytest.mark.parametrize("bw, position, x", [(10, "lower", 6), (20, "both", 6), (3, "both", 3), (50, "upper", 12), (5, "both", 9)])
def test_overlay_conserves_the_frame_energy_for_a_flat_nr_grid(bw, position, x):
    nrb = NR_15KHZ[bw][0]
    plan = nb.nb_iot_overlay_plan(nrb, position, x, [1])
    out = plan.overlay(np.ones((nrb * 12, 14 * 10), dtype=complex))
    assert out.shape == (nrb * 12, 140)
    assert float(np.sum(np.abs(out) ** 2)) == pytest.approx(total_res(nrb), rel=1e-12)       # P_avg = 1 per RE
    # the boosted NB-IoT REs sit X dB above the average, the NR REs at the power_split level, ratio as quoted
    c0 = plan.carriers[0]
    nb_power = np.abs(out[12 * c0.rb:12 * c0.rb + 12, :14][c0.mask[0].T]) ** 2
    assert nb_power.size == 108
    np.testing.assert_allclose(nb_power, 10 ** (x / 10), rtol=1e-12)
    free = min(set(range(nrb)) - set(plan.rbs))
    np.testing.assert_allclose(np.abs(out[12 * free:12 * free + 12, :14]) ** 2, plan.nr_gain ** 2, rtol=1e-12)
    # whole-grid overlay equals the slot-by-slot overlay, slots beyond 10 repeat the frame
    rng = np.random.default_rng(3)
    nr = rng.standard_normal((nrb * 12, 14 * 12)) + 0j
    full = plan.overlay(nr)
    for s in (0, 7, 10, 11):
        np.testing.assert_array_equal(full[:, 14 * s:14 * (s + 1)], plan.overlay_slot(nr[:, 14 * s:14 * (s + 1)], s % 10))


def test_overlay_shape_validation():
    plan = nb.nb_iot_overlay_plan(52, "lower", 6, [1])
    for bad in (np.zeros((624, 13)), np.zeros((623, 14)), np.zeros((624,))):
        with pytest.raises(ValueError):
            plan.overlay_slot(bad, 0)
    for bad in (np.zeros((624, 15)), np.zeros((12, 14)), np.zeros((624,))):
        with pytest.raises(ValueError):
            plan.overlay(bad)
