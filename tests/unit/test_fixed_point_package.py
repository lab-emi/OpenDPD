"""The committed ``fixed-point-v1`` deployment packages that the MATLAB toolbox tests run against.

``Matlab/toolbox/tests/data/gru-pa.fixed-point-v1.zip`` (default specification) and ``gru-pa-custom.fixed-point-v1.zip`` (a
specification that differs in every format) are real output of ``opendpd.services.deploy.export_deployment``, written by
``scripts/make_matlab_fixed_fixture.py``. The MATLAB reader takes the golden vectors as its oracle, so these tests keep the
oracle honest: the files are internally consistent, today's Python reference still computes exactly their golden outputs,
today's C99 generator still writes exactly their sources, and the rule texts the MATLAB reader insists on are the ones the
specification states. When one of them fails, the specification or a reference changed: regenerate the packages together
with that change and run the MATLAB tests.
"""

import hashlib
import importlib.util
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

from opendpd.core.fixed_point import FixedGRU, QuantisedGRU, table
from opendpd.export import c_backend
from opendpd.schemas.fixed_point import NONLINEARITY, ROUNDING, SATURATION, DeploymentManifest, FixedPointSpec

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Matlab" / "toolbox" / "tests" / "data"
PACKAGES = sorted(DATA.glob("*.fixed-point-v1.zip")) if DATA.is_dir() else []
CASES = ["normal", "extreme", "saturation", "all_zero", "state_reset", "long_sequence"]
GOLDEN_FILES = ["x.i16", "y.i16", "h_final.i16", "h_trace.i16", "meta.json"]
EXPECTED_ENTRIES = sorted(["manifest.json", "spec.json", "weights.json", "README.md", "c/gru_fixed.c", "c/gru_fixed.h", "c/harness.c"]
                          + [f"golden/{case}/{name}" for case in CASES for name in GOLDEN_FILES])

pytestmark = pytest.mark.skipif(len(PACKAGES) != 2, reason="the MATLAB toolbox's fixed-point test packages are not present")


def _read(path):
    archive = zipfile.ZipFile(path)
    return {info.filename: archive.read(info.filename) for info in archive.infolist()}


def _reference(contents, manifest):
    """The quantised model the package describes, rebuilt from its own weights.json."""
    w = json.loads(contents["weights.json"])
    spec = manifest.spec
    return QuantisedGRU(
        spec=spec, hidden=w["hidden"], inputs=w["inputs"], outputs=w["outputs"],
        w_ih=np.array(w["w_ih"], dtype=np.int64), w_hh=np.array(w["w_hh"], dtype=np.int64), w_out=np.array(w["w_out"], dtype=np.int64),
        f_ih=w["fractions"]["w_ih"], f_hh=w["fractions"]["w_hh"], f_out=w["fractions"]["w_out"],
        b_ih=np.array(w["b_ih"], dtype=np.int64), b_hh=np.array(w["b_hh"], dtype=np.int64), b_out=np.array(w["b_out"], dtype=np.int64),
        tensors=list(manifest.tensors),
        sigmoid_table=np.array(w["sigmoid_table"], dtype=np.int64), tanh_table=np.array(w["tanh_table"], dtype=np.int64))


def _manifest(path):
    return DeploymentManifest.model_validate_json(_read(path)["manifest.json"])


def _i16(contents, case, name, columns):
    return np.frombuffer(contents[f"golden/{case}/{name}"], dtype="<i2").astype(np.int64).reshape(-1, columns)


@pytest.mark.parametrize("path", PACKAGES, ids=lambda p: p.name)
def test_the_package_is_complete_consistent_and_verified(path):
    contents = _read(path)
    assert sorted(contents) == EXPECTED_ENTRIES
    manifest = DeploymentManifest.model_validate_json(contents["manifest.json"])
    assert FixedPointSpec.model_validate_json(contents["spec.json"]) == manifest.spec
    assert manifest.verification.status == "bit_exact" and manifest.verification.cases_checked == 6
    assert sorted(manifest.files) == sorted(name for name in contents if name != "manifest.json")
    for name, digest in manifest.files.items():
        assert hashlib.sha256(contents[name]).hexdigest() == digest, name
    assert [g.case_id for g in manifest.golden] == CASES
    assert manifest.golden[CASES.index("state_reset")].resets_at == [0, 256, 512, 768]
    for g in manifest.golden:
        assert hashlib.sha256(contents[f"golden/{g.case_id}/y.i16"]).hexdigest() == g.output_sha256
        assert hashlib.sha256(contents[f"golden/{g.case_id}/h_trace.i16"]).hexdigest() == g.trace_sha256
    assert all(t.saturated == 0 for t in manifest.tensors)


def test_the_two_packages_are_one_model_under_two_specifications():
    default = _manifest(next(p for p in PACKAGES if "custom" not in p.name))
    custom = _manifest(next(p for p in PACKAGES if "custom" in p.name))
    assert default.spec == FixedPointSpec()
    for field in ("x", "h", "y", "pre", "sigmoid", "tanh"):
        assert getattr(default.spec, field) != getattr(custom.spec, field), field
    assert default.spec.weight_bits != custom.spec.weight_bits and default.spec.accumulator_bits != custom.spec.accumulator_bits
    assert len({custom.spec.x.frac, custom.spec.h.frac, custom.spec.y.frac}) == 3, "a reader that scales by the wrong fraction must show"
    script = importlib.util.spec_from_file_location("make_matlab_fixed_fixture", ROOT / "scripts" / "make_matlab_fixed_fixture.py")
    module = importlib.util.module_from_spec(script)
    script.loader.exec_module(module)
    assert module.custom_spec() == custom.spec, "the committed package no longer matches scripts/make_matlab_fixed_fixture.py"


@pytest.mark.parametrize("path", PACKAGES, ids=lambda p: p.name)
def test_todays_reference_still_computes_the_golden_vectors_and_tables(path):
    contents = _read(path)
    manifest = DeploymentManifest.model_validate_json(contents["manifest.json"])
    q = _reference(contents, manifest)
    assert np.array_equal(q.sigmoid_table, table(manifest.spec.sigmoid)) and np.array_equal(q.tanh_table, table(manifest.spec.tanh))
    for g in manifest.golden:
        x = _i16(contents, g.case_id, "x.i16", 2)
        y, traces = FixedGRU(q).run(x, tuple(g.resets_at), trace=True)
        assert np.array_equal(y, _i16(contents, g.case_id, "y.i16", 2)), g.case_id
        assert np.array_equal(traces["h"], _i16(contents, g.case_id, "h_trace.i16", q.hidden)), g.case_id
        assert np.array_equal(traces["h"][-1], _i16(contents, g.case_id, "h_final.i16", q.hidden)[0]), g.case_id


@pytest.mark.parametrize("path", PACKAGES, ids=lambda p: p.name)
def test_todays_c99_generator_still_writes_the_packaged_sources(path, tmp_path):
    contents = _read(path)
    manifest = DeploymentManifest.model_validate_json(contents["manifest.json"])
    q = _reference(contents, manifest)
    for name, text in c_backend.generate(q).items():
        assert contents[f"c/{name}"].decode("utf-8") == text, name
    if c_backend.compiler() is None:
        pytest.skip("no C compiler: the packaged sources were compared but not compiled")
    cases = [(g.case_id, _i16(contents, g.case_id, "x.i16", 2), tuple(g.resets_at)) for g in manifest.golden]
    verification, _ = c_backend.verify(q, cases, tmp_path)
    assert verification.status == "bit_exact", verification.detail


def test_the_rules_the_matlab_reader_insists_on_are_the_ones_the_specification_states():
    text = (ROOT / "Matlab" / "toolbox" / "+opendpd" / "FixedModel.m").read_text(encoding="utf-8")
    for name, expected in (("Rounding", ROUNDING), ("Saturation", SATURATION), ("Nonlinearity", NONLINEARITY)):
        block = re.search(rf"^\s*{name} = (.*?)(?=^\s*\w+ = |^\s*end\b)", text, re.S | re.M).group(1)
        literal = "".join(re.findall(r"'((?:[^']|'')*)'", block)).replace("''", "'")
        assert literal == expected, f"FixedModel.{name} differs from opendpd/schemas/fixed_point.py"


def test_the_matlab_reader_accepts_what_the_specification_allows_and_no_more_than_the_c_backend_can_hold():
    """The reader's limits (16-bit words, 32-bit pre-activation, a 53-bit accumulator) are the C99 backend's types and the
    exactness of double precision; if the schema is widened, the backend and the reader must be widened with it."""
    source = (ROOT / "Matlab" / "toolbox" / "+opendpd" / "FixedModel.m").read_text(encoding="utf-8")
    assert "wholeNumber('weight_bits', spec.weight_bits, 4, 16)" in source
    assert "wholeNumber('accumulator_bits', spec.accumulator_bits, 32, 53)" in source
    assert "wordFormat('pre', spec.pre, 32)" in source
    contents = _read(PACKAGES[0])
    c_source = c_backend.generate(_reference(contents, DeploymentManifest.model_validate_json(contents["manifest.json"])))
    assert "static const int16_t W_IH" in c_source["gru_fixed.c"] and "static const int32_t B_IH" in c_source["gru_fixed.c"]
