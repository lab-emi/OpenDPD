"""The catalogue served to the UI, the package API surface, and the import hygiene of the package."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from opendpd.core.waveforms import nr_tm
from opendpd.core.waveforms.nr_tm.tables import NRB
from opendpd.schemas.nr_test_model import NR_TM_MODELS, NrTestModelConfig, NrTmCarrier

CLAUSES = {
    "NR-FR1-TM1.1": "4.9.2.2.1", "NR-FR1-TM1.2": "4.9.2.2.2", "NR-FR1-TM2": "4.9.2.2.3", "NR-FR1-TM2a": "4.9.2.2.4", "NR-FR1-TM2b": "4.9.2.2.4A",
    "NR-FR1-TM3.1": "4.9.2.2.5", "NR-FR1-TM3.1a": "4.9.2.2.6", "NR-FR1-TM3.1b": "4.9.2.2.6A", "NR-FR1-TM3.2": "4.9.2.2.7",
    "NR-FR1-TM3.3": "4.9.2.2.8", "NR-N-TM": "4.9.2.2.9",
}
# model -> [(n_RNTI, modulation, EVM measured)] from the usage notes of TS 38.141-1 (tables 4.9.2.2-x, clause 6.5.3)
USERS = {
    "NR-FR1-TM1.1": [(0, "QPSK", False), (2, "QPSK", False)],
    "NR-FR1-TM1.2": [(0, "QPSK", False), (1, "QPSK", False), (2, "QPSK", False)],
    "NR-FR1-TM2": [(2, "64QAM", True)], "NR-FR1-TM2a": [(2, "256QAM", True)], "NR-FR1-TM2b": [(2, "1024QAM", True)],
    "NR-FR1-TM3.1": [(0, "64QAM", True), (2, "64QAM", True)], "NR-FR1-TM3.1a": [(0, "256QAM", True), (2, "256QAM", True)],
    "NR-FR1-TM3.1b": [(0, "1024QAM", True), (2, "1024QAM", True)],
    "NR-FR1-TM3.2": [(0, "QPSK", False), (1, "16QAM", True), (2, "QPSK", False)],
    "NR-FR1-TM3.3": [(0, "QPSK", False), (1, "QPSK", True), (2, "QPSK", False)],
    "NR-N-TM": [(0, "QPSK", False), (2, "QPSK", False)],
}
# TS 38.104 table 5.3.2-1 spot checks and the NB-IoT lowest / highest eligible NR resource block (research notes 5.2)
NB_IOT = {3: (2, 12, 15), 5: (2, 22, 25), 10: (0, 50, 52), 15: (4, 74, 79), 20: (2, 102, 106), 25: (1, 131, 133), 30: (4, 159, 160),
          35: (3, 183, 188), 40: (2, 212, 216), 45: (0, 240, 242), 50: (4, 269, 270)}


@pytest.fixture(scope="module")
def cat():
    return nr_tm.catalog()


def test_models_users_and_flags(cat):
    assert [m.model for m in cat.models] == list(NR_TM_MODELS) == list(CLAUSES)
    for info in cat.models:
        assert info.clause == CLAUSES[info.model]
        assert [(u.rnti, u.modulation, u.evm_measured) for u in info.users] == USERS[info.model]
        assert info.title and info.description and info.used_for and info.fixed_parameters
        assert info.supports_layers == (info.model == "NR-FR1-TM1.1")
        assert info.nb_iot == info.fdd_only == (info.model == "NR-N-TM")
        assert info.scs_khz == ([15] if info.nb_iot else [15, 30, 60])
        names = [p.name for p in info.fixed_parameters]
        assert len(names) == len(set(names)) and "Cyclic prefix" in names
        assert ("Target share x" in names) == (info.model in ("NR-FR1-TM1.2", "NR-FR1-TM3.2", "NR-FR1-TM3.3"))
        assert ("PRB position" in names) == info.model.startswith("NR-FR1-TM2")
        assert ("Rank" in names) == info.supports_layers and ("NB-IoT" in names) == info.nb_iot
    assert "3GPP TS 38.141-1" in cat.standard and "4.9.2" in cat.standard
    by_id = {m.model: m for m in cat.models}
    assert by_id["NR-FR1-TM3.2"].modulations == ["QPSK", "16QAM"] and by_id["NR-FR1-TM2b"].modulations == ["1024QAM"]


def test_resource_block_table_and_scs_list(cat):
    assert cat.scs_khz == [15, 30, 60]
    assert cat.resource_blocks == {str(scs): {str(bw): n for bw, n in table.items()} for scs, table in NRB.items()}
    assert cat.resource_blocks["15"]["50"] == 270 and cat.resource_blocks["15"]["3"] == 15 and cat.resource_blocks["30"]["100"] == 273
    assert cat.resource_blocks["60"]["100"] == 135 and cat.resource_blocks["60"]["10"] == 11 and "5" not in cat.resource_blocks["60"]
    assert "3" not in cat.resource_blocks["30"] and "60" not in cat.resource_blocks["15"]
    assert sum(len(t) for t in cat.resource_blocks.values()) == 40


def test_tdd_patterns_follow_the_ts_38_331_semantics(cat):
    patterns = {p.scs_khz: p for p in cat.tdd_patterns}
    assert sorted(patterns) == [15, 30, 60]
    expect = {15: (5, 3, 10, 1, 2, ["D"] * 3 + ["DDDDDDDDDDFFUU", "U"]),
              30: (10, 7, 6, 2, 4, ["D"] * 7 + ["DDDDDDFFFFUUUU", "U", "U"]),
              60: (20, 14, 12, 4, 8, ["D"] * 14 + ["DDDDDDDDDDDDFF", "FFFFFFUUUUUUUU"] + ["U"] * 4)}
    for scs, (slots, dl_slots, dl_symbols, ul_slots, ul_symbols, maps) in expect.items():
        p = patterns[scs]
        assert (p.period_ms, p.slots_per_period, p.dl_slots, p.dl_symbols, p.ul_slots, p.ul_symbols) == (5.0, slots, dl_slots, dl_symbols, ul_slots, ul_symbols)
        assert p.slot_map == [{"D": "D" * 14, "U": "U" * 14}.get(m, m) for m in maps]
        assert sum(m.count("D") for m in p.slot_map) / (14 * slots) == pytest.approx(26 / 35)


def test_nb_iot_table(cat):
    nb = cat.nb_iot
    assert nb.scs_khz == 15 and nb.bandwidths_mhz == sorted(NB_IOT)
    for bw, (low, high, nrb) in NB_IOT.items():
        assert nb.lower_rb[str(bw)] == low and nb.upper_rb[str(bw)] == high, bw
        rbs = nb.eligible_rbs[str(bw)]
        assert rbs[0] == low and rbs[-1] == high and NRB[15][bw] == nrb
        assert all(b - a == 5 for a, b in zip(rbs, rbs[1:]))                       # a comb with a spacing of five resource blocks
        assert all(0 <= rb < nrb for rb in rbs)


def test_limits_defaults_and_notes(cat):
    limits = cat.limits
    assert (limits.max_carriers, limits.max_samples, limits.max_cell_id) == (16, 1_000_000, 1007)
    assert (limits.standard_subframes_fdd, limits.standard_subframes_tdd, limits.carrier_gap_step_mhz) == (10, 20, 0.3)
    assert cat.default_recipe == NrTestModelConfig(model="NR-FR1-TM1.1", duplex="fdd", carriers=[NrTmCarrier(scs_khz=30, bandwidth_mhz=20)])
    assert cat.default_config.nr_test_model == cat.default_recipe and cat.default_config.preset_id.startswith("nr-tm1p1-fdd-30k-20m-c1-")
    assert cat.default_config == nr_tm.resolve(cat.default_recipe).config
    assert len(cat.notes) >= 4 and any("FR2" in note for note in cat.notes) and any("1,000,000" in note for note in cat.notes)
    assert 10_000 < len(cat.model_dump_json()) < 200_000
    assert nr_tm.catalog() == cat


def test_package_surface_and_source_files():
    assert set(nr_tm.__all__) == {"STRUCTURAL_FIELDS", "SOURCE_FILES", "NrTmDerived", "catalog", "clear_memo", "derive", "derived_fields", "recipe_label",
                                  "resolve", "synthesize_nr_tm", "validate_recipe"}
    names = {p.name for p in nr_tm.SOURCE_FILES}
    assert {"__init__.py", "tables.py", "spec.py", "sequences.py", "grid.py", "modulate.py", "plan.py", "synth.py", "catalog.py", "label.py"} <= names
    assert list(nr_tm.SOURCE_FILES) == sorted(nr_tm.SOURCE_FILES) and all(p.is_file() and p.suffix == ".py" for p in nr_tm.SOURCE_FILES)
    assert all(p.parent == Path(nr_tm.__file__).resolve().parent for p in nr_tm.SOURCE_FILES)
    assert callable(nr_tm.catalog) and callable(nr_tm.derive) and callable(nr_tm.resolve) and callable(nr_tm.validate_recipe) and callable(nr_tm.recipe_label)


def run_python(code):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)


def test_importing_the_schemas_and_the_package_stays_light():
    heavy = ("torch", "fastapi", "pandas", "matplotlib")
    code = ("import sys\nimport opendpd.schemas\n"
            f"assert not [m for m in {heavy!r} if m in sys.modules], [m for m in {heavy!r} if m in sys.modules]\n"
            "from opendpd.schemas.nr_test_model import NrTestModelConfig\n"
            "from opendpd.core.waveforms import nr_tm\n"
            "nr_tm.validate_recipe(NrTestModelConfig())\n"
            "d = nr_tm.derive(NrTestModelConfig())\n"
            f"assert not [m for m in {heavy + ('scipy',)!r} if m in sys.modules], [m for m in {heavy + ('scipy',)!r} if m in sys.modules]\n"
            "print('light')\n")
    result = run_python(code)
    assert result.returncode == 0 and "light" in result.stdout, result.stderr


def test_a_recipe_is_validated_when_it_is_built_and_the_validator_does_not_recurse():
    code = ("from opendpd.schemas.nr_test_model import NrTestModelConfig, NrTmCarrier\n"
            "from pydantic import ValidationError\n"
            "NrTestModelConfig(carriers=[NrTmCarrier(scs_khz=30, bandwidth_mhz=20)])\n"
            "try:\n    NrTestModelConfig(carriers=[NrTmCarrier(scs_khz=60, bandwidth_mhz=5)])\nexcept ValidationError as exc:\n    print('refused', 'MHz' in str(exc))\n")
    result = run_python(code)
    assert result.returncode == 0 and "refused True" in result.stdout, result.stderr


def test_the_notes_do_not_claim_more_than_studio_offers(cat):
    open_note = next(n for n in cat.notes if n.startswith("Parameters you can set:"))
    assert cat.notes[1] == open_note
    # the claim is a list of what can be set, not "every parameter the standard leaves open"
    assert not any("leaves open" in n or "every parameter" in n.lower() for n in cat.notes)
    # a free cell identity, the number of carriers and the NB-IoT boost cap are Studio's, and the note says so
    assert "Studio extensions, not options of the standard" in open_note
    for text in ("cell identity other than N_ID^cell = n", "TS 38.141-1 4.9.2.3", "at most 16 carriers", "boost of at most 12 dB"):
        assert text in open_note, text
    assert open_note.index("Studio extensions") > open_note.index("capture length")       # the plain parameters come first
    assert "cell identity," not in open_note.split("Studio extensions")[0]                 # it is not listed among the parameters the standard leaves open
    not_offered = next(n for n in cat.notes if n.startswith("Not offered:"))
    for text in ("NR-N-TM with TDD", "Table 4.9.2.2-1a", "15 kHz and 60 kHz raster", "PN23 stream shared by all users", "NRTC1-6"):
        assert text in not_offered, text
    assert any("NRTC6 ON-OFF" in n for n in cat.notes)                                      # the earlier note about NRTC6 is still there


def test_the_limits_say_whose_limits_they_are(cat):
    from opendpd.schemas.nr_test_model_api import NrTmLimits, NrTmNbIotInfo
    assert "Studio limit" in NrTmLimits.model_fields["max_carriers"].description and "standard sets none" in NrTmLimits.model_fields["max_carriers"].description
    assert "Studio limit" in NrTmLimits.model_fields["max_samples"].description
    assert "Studio limit" in NrTmNbIotInfo.model_fields["max_boost_db"].description
    assert cat.limits.max_carriers == 16 and cat.nb_iot.max_boost_db == 12.0
    # the catalog's per-model duplex flag: only NR-N-TM, and the text beside it names TS 38.141-1 Table 4.9.2.2-1a
    assert [m.model for m in cat.models if m.fdd_only] == ["NR-N-TM"]
    info = next(m for m in cat.models if m.model == "NR-N-TM")
    assert info.scs_khz == [15] and "Table 4.9.2.2-1a" in {p.name: p.value for p in info.fixed_parameters}["Duplex / numerology"]


def test_releases_of_ts_38_104_are_cited_consistently():
    """The N_RB, guard band and NB-IoT power tables were read in two copies of TS 38.104 (V18.10.0 and V18.15.0), identical for these tables.

    Every release number that follows a mention of 38.104 in the test-model code (up to the next mention of another specification) must be
    one of the two, so that a stray 'read in V18.9.0' anywhere in the package or in the schema is found."""
    import re
    root = Path(nr_tm.__file__).resolve().parents[3] / "schemas"
    package = Path(nr_tm.__file__).resolve().parent
    sources = sorted(package.glob("*.py")) + sorted(root.glob("nr_test_model*.py"))
    assert {"tables.py", "nb_iot.py", "plan.py", "catalog.py", "nr_test_model.py", "nr_test_model_api.py"} <= {s.name for s in sources}
    other_specification = re.compile(r"\b(?:TS|TR)\s?\d\d\.\d{3}")
    release = re.compile(r"\bV\d+\.\d+\.\d+\b")
    cited, mentions = {}, 0
    for source in sources:
        text = source.read_text()
        for mention in re.finditer(r"38\.104", text):
            mentions += 1
            window = text[mention.end():mention.end() + 400]
            stop = other_specification.search(window)
            for found in release.findall(window[:stop.start()] if stop else window):
                cited.setdefault(found, set()).add(source.name)
    OWNED = {"V18.10.0", "V18.15.0"}
    assert mentions > 10 and cited, "the scan must find the citations it is meant to check"
    assert set(cited) == OWNED, cited                      # no stray release such as V18.9.0, and both releases are really cited
    for found in OWNED:
        assert {"tables.py", "nb_iot.py"} <= cited[found], (found, cited[found])
    tables = (package / "tables.py").read_text()
    assert "V18.10.0 (2025-07)" in tables and "V18.15.0 (2026-09)" in tables and "identical in V18.10.0 and V18.15.0" in tables
