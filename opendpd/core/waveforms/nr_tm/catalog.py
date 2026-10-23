"""The catalogue behind GET /signal-generator/nr-test-models: models, parameter ranges, fixed parameters, TDD patterns."""
from __future__ import annotations

from opendpd.schemas.nr_test_model import MAX_NB_IOT_BOOST_DB, NrTestModelConfig, NrTmCarrier

from .nb_iot import eligible_rbs, power_dynamic_range_floor_db
from .plan import MAX_SAMPLES, resolve
from .tables import (
    COMMON_FIXED_PARAMETERS, GAP_STEP_KHZ, MAX_CELL_ID, MODELS, NB_IOT_BANDWIDTHS_MHZ, NB_IOT_SCS_KHZ, NRB, QM_NAME, SCS_KHZ,
    STANDARD, STANDARD_SUBFRAMES, TDD_PATTERNS, ModelDef,
)

MAX_CARRIERS = 16


def _modulations(model: ModelDef) -> list[str]:
    seen: list[str] = []
    for user in model.users:
        name = QM_NAME[user.qm]
        if name not in seen:
            seen.append(name)
    return seen


def _model_parameters(model: ModelDef) -> list[tuple[str, str]]:
    users = "; ".join(f"n_RNTI {u.rnti}: {u.role} ({QM_NAME[u.qm]})" for u in model.users)
    params = [("Users (table 4.9.2.3.2-1)", users)]
    if model.structure == "rbg":
        other = 0 if model.target_rnti == 1 else 1
        params += [
            ("Target share x", f"{model.x_percent} % of the PRBs 3...N_RB-1 (resource-block groups of n_RNTI {model.target_rnti})"),
            ("RBG size P", "TS 38.214 table 5.1.2.2.1-1 configuration 1 with N_RB-3 PRBs: 2 (up to 36), 4 (up to 72), 8 (up to 144), 16 (up to 275)"),
            ("Number of target RBGs", "N_RBG = min(floor(x (N_RB-3) / P), (A - A mod 2) / 2 + 1), A = floor((N_RB-3 + (3 mod P) - P) / P)"),
            ("Target RBG locations", "RBG floor((N_RB-3 + (3 mod P)) / P) - 1 and, if N_RBG > 1, RBG 1, 3, ..., 2 (N_RBG-2) + 1 "
                                    "(0-based, RBG 0 being the partial first group that starts at PRB 3)"),
            ("Level of the target RBGs", f"{model.target_level_db:+g} dB (n_RNTI {model.target_rnti})"),
            ("Level of the other PRBs", f"n_RNTI {other}: 10 log10((N_RB-3 - 10^({model.target_level_db:g}/10) P N_RBG) / (N_RB-3 - P N_RBG)) dB "
                                        "so that the total power equals N_RB-3 PRBs at 0 dB"),
        ]
    elif model.structure == "single_prb":
        params += [
            ("PRB position", "slot 3n: PRB 0; slot 3n+1: PRB floor(N_RB/2); slot 3n+2: PRB N_RB-1 (slots numbered within the radio frame); "
                             "all other PRBs are unallocated (table 4.9.2.2.3-1)"),
            ("Level", "0 dB boost"),
        ]
    else:
        params.append(("Level", "0 dB boost on all PRBs"))
    if model.supports_layers:
        params.append(("Rank", "1 (one layer, port 1000); rank 2 (two layers, ports 1000 and 1001, no precoding) only for the TAE test; "
                               "the PDCCH is always single-port"))
    if model.nb_iot:
        params += [
            ("NB-IoT", "one NR resource block per NB-IoT carrier is punctured at the eligible in-band position closest to the lower and/or "
                       "upper minimum guard band (L_NB-IoT, R_NB-IoT) and carries the NB-IoT test model N-TM of TS 36.141 6.1.3-6.1.6, the only "
                       "NB-IoT test model there is; every other NR carrier is NR-FR1-TM1.1"),
            ("NB-IoT channels", "NPSS, NSSS, NRS, NPBCH, NPDCCH (format 1) and NPDSCH (single user, n_RNTI 1000, QPSK), normal cyclic prefix, "
                                "n_f = 0 and slot 0 at the start; all data bits are zero (scrambled as in TS 36.211 clause 10.2)"),
            ("NB-IoT subframe map", "one 10 ms frame repeated every frame: subframe 0 NPBCH, 1 NPDCCH, 2-4 and 6-8 NPDSCH, 5 NPSS, 9 NSSS; "
                                    "NRS in every subframe except 5 and 9"),
            ("NB-IoT antenna port", "one NB-IoT antenna port is exported (NRS port 2000); the 304 (stand-alone) / 200 (in-band) bit budgets of "
                                    "TS 36.141 6.1.4 are those of a two-port NRS, four host CRS ports and three control symbols, so the NRS "
                                    "positions of port 2001 stay reserved and only the 100 data resource elements per in-band subframe are used"),
            ("NB-IoT cell identity", "N_ID^Ncell = 97 + 6 n + max(0, m - 1) with m the cell identity of the host NR carrier and n the rank of the "
                                     "host among the NB-IoT hosts (103 for the lowest host with cell identity 1, 110 for the second); the lower "
                                     "and the upper NB-IoT carrier of one host share it; at most 503"),
            ("NB-IoT resource elements", "1085 of the 1680 N_RB resource elements of a frame per NB-IoT carrier: 108 (100 data + 8 NRS) in "
                                         "each of the eight subframes 0-4 and 6-8, NPSS 105 and NSSS 116; 1089 if the host cell identity "
                                         "mod 6 is 2 or 5 (the NPSS then skips no host CRS resource element)"),
            ("NB-IoT power", "P_avg = P_rated / (12 N_RB); NB-IoT RE: P_avg 10^(X/10); NR RE: (P_rated - N_NB-IoT P_NB-IoT) / N_NR, counted over "
                             "the resource elements of the whole 10 ms frame. X is the manufacturer's declared NB-IoT power dynamic range, at least "
                             "6 dB at the positions L_NB-IoT and R_NB-IoT of host carriers up to 20 MHz and at least 3 dB for 25 MHz and wider "
                             "(TS 38.104 Table 6.3.4.2-1, TS 38.141-1 D.43). Studio enforces that minimum and accepts X up to "
                             f"{MAX_NB_IOT_BOOST_DB} dB: the upper limit is a Studio limit, the standard sets none"),
            ("NR content around NB-IoT", "NR-FR1-TM1.1 on every resource element without NB-IoT signal, including the control symbols and host "
                                         "CRS positions of the punctured resource block"),
            ("Duplex / numerology", "Studio generates FDD with 15 kHz subcarrier spacing and channel bandwidths of 3 to 50 MHz. The standard also "
                                    "defines a TDD form (TS 38.141-1 Table 4.9.2.2-1a: D S U U D at 15 kHz, 5 ms); Studio does not implement "
                                    "NB-IoT in-band operation with TDD in this release. The NB-IoT bandwidth of 180 kHz is one NR resource "
                                    "block only at 15 kHz, which is why other numerologies are not offered"),
            ("EVM", "the NB-IoT resource elements are excluded from the PDSCH EVM reference"),
        ]
    return params


def _info(model: ModelDef):
    from opendpd.schemas.nr_test_model_api import NrTmFixedParameter, NrTmModelInfo, NrTmUserInfo

    fixed = [NrTmFixedParameter(name=n, value=v) for n, v in _model_parameters(model) + list(COMMON_FIXED_PARAMETERS)]
    return NrTmModelInfo(
        model=model.id, clause=model.clause, title=model.title, description=model.description, used_for=list(model.used_for),
        modulations=_modulations(model),
        users=[NrTmUserInfo(rnti=u.rnti, role=u.role, modulation=QM_NAME[u.qm], evm_measured=u.evm_measured) for u in model.users],
        fixed_parameters=fixed, supports_layers=model.supports_layers, nb_iot=model.nb_iot, fdd_only=model.fdd_only,
        scs_khz=[NB_IOT_SCS_KHZ] if model.nb_iot else list(SCS_KHZ))


def default_recipe() -> NrTestModelConfig:
    return NrTestModelConfig(model="NR-FR1-TM1.1", duplex="fdd", carriers=[NrTmCarrier(scs_khz=30, bandwidth_mhz=20)])


def catalog():
    """The NrTmCatalog (the API schema module is imported here so that importing the package stays light)."""
    from opendpd.schemas.nr_test_model_api import NrTmCatalog, NrTmLimits, NrTmNbIotInfo, NrTmTddPattern

    recipe = default_recipe()
    patterns = [NrTmTddPattern(scs_khz=p.scs_khz, period_ms=float(p.period_ms), slots_per_period=p.slots, dl_slots=p.dl_slots,
                               dl_symbols=p.dl_symbols, ul_slots=p.ul_slots, ul_symbols=p.ul_symbols, slot_map=p.slot_maps())
                for p in TDD_PATTERNS.values()]
    eligible = {str(bw): eligible_rbs(NRB[NB_IOT_SCS_KHZ][bw]) for bw in NB_IOT_BANDWIDTHS_MHZ}
    nb_iot = NrTmNbIotInfo(
        scs_khz=NB_IOT_SCS_KHZ, bandwidths_mhz=list(NB_IOT_BANDWIDTHS_MHZ),
        lower_rb={bw: rbs[0] for bw, rbs in eligible.items()}, upper_rb={bw: rbs[-1] for bw, rbs in eligible.items()}, eligible_rbs=eligible,
        min_boost_db={str(bw): float(max(power_dynamic_range_floor_db(bw, NRB[NB_IOT_SCS_KHZ][bw], rb) for rb in (eligible[str(bw)][0], eligible[str(bw)][-1])))
                      for bw in NB_IOT_BANDWIDTHS_MHZ},
        max_boost_db=float(MAX_NB_IOT_BOOST_DB))
    notes = [
        "Models of 3GPP TS 38.141-1 V18.14.0 clause 4.9.2.2 (FR1, BS type 1-C and 1-H); N_RB from TS 38.104 table 5.3.2-1. "
        "The FR2 models of TS 38.141-2 are not included.",
        "Parameters you can set: duplex mode, number of carriers, subcarrier spacing and channel bandwidth per carrier, the gap before a carrier "
        "(sub-block gap), rank for NR-FR1-TM1.1, the PN23 conventions, NB-IoT position and boost, oversampling and capture length. Studio "
        "extensions, not options of the standard: a cell identity other than N_ID^cell = n for the n-th lowest carrier (TS 38.141-1 4.9.2.3; "
        "flagged in the warnings), at most 16 carriers and an NB-IoT boost of at most 12 dB (the standard sets neither limit).",
        "Carriers are placed at the nominal channel spacing of TS 38.104 5.4.1.2 for 100 kHz-raster bands, the aggregate is centred at baseband DC and "
        "every carrier has the same resource-element amplitude (equal power spectral density).",
        "Only PDCCH, PDSCH and DM-RS are generated: no SSB, CSI-RS, PT-RS or channel coding; the NRTC6 ON-OFF patterns are not included.",
        f"At most {MAX_SAMPLES:,} samples are generated: the capture is shortened to whole subframes, or lower the oversampling.",
        "Not offered: NR-N-TM with TDD (defined in TS 38.141-1 Table 4.9.2.2-1a), the channel spacing rules of 15 kHz and 60 kHz raster bands "
        "(TS 38.104 5.4.1.2; Studio uses the 100 kHz-raster rule), a PN23 stream shared by all users, and the NRTC1-6 configurations as such.",
    ]
    return NrTmCatalog(
        standard=STANDARD, models=[_info(m) for m in MODELS], scs_khz=list(SCS_KHZ),
        resource_blocks={str(scs): {str(bw): n for bw, n in table.items()} for scs, table in NRB.items()},
        tdd_patterns=patterns, nb_iot=nb_iot,
        limits=NrTmLimits(max_carriers=MAX_CARRIERS, max_samples=MAX_SAMPLES, max_cell_id=MAX_CELL_ID,
                          standard_subframes_fdd=STANDARD_SUBFRAMES["fdd"], standard_subframes_tdd=STANDARD_SUBFRAMES["tdd"],
                          carrier_gap_step_mhz=GAP_STEP_KHZ / 1000),
        default_recipe=recipe, default_config=resolve(recipe).config, notes=notes)
