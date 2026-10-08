"""The toolbox's version, supported models and documented calls agree with each other and with the code."""

import re
from pathlib import Path

import pytest

from opendpd.services.inference import APPLY_MODELS

ROOT = Path(__file__).resolve().parents[2]
TOOLBOX = ROOT / "Matlab" / "toolbox"
pytestmark = pytest.mark.skipif(not TOOLBOX.is_dir(), reason="toolbox sources are not part of this checkout")


def _text(*parts) -> str:
    return TOOLBOX.joinpath(*parts).read_text(encoding="utf-8")


def test_one_toolbox_version_everywhere():
    version = re.search(r"ToolboxVersion = '(\d+\.\d+\.\d+)'", _text("buildfile.m")).group(1)
    assert re.search(rf"^% Version {re.escape(version)}$", _text("Contents.m"), re.M)
    assert f"**{version}**" in _text("README.md") and f"OpenDPD-{version}.mltbx" in _text("README.md")
    # the package file name follows the version instead of repeating it
    assert "['OpenDPD-' char(opts.ToolboxVersion) '.mltbx']" in _text("buildfile.m")
    workflow = (ROOT / ".github" / "workflows" / "matlab-toolbox.yml").read_text(encoding="utf-8")
    assert not re.search(r"OpenDPD-\d", workflow), "the workflow must find the package, not name its version"


def test_docs_name_exactly_the_models_apply_supports():
    reference = next(line for line in _text("docs", "reference.md").splitlines() if "opendpd.apply(job, x" in line)
    workflow = _text("docs", "workflow.md")
    for key in APPLY_MODELS:
        assert f"`{key}`" in reference and f"`{key}`" in workflow, key
    for text in (reference, workflow, _text("README.md")):
        assert "ordinary, unquantized GRU" not in text and "ordinary unquantized GRU" not in text


def _import_statements(path: Path):
    """Every ``opendpd.importIQ(...)`` / ``opendpd.importMAT(...)`` statement, continuation lines joined."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines):
        if not re.search(r"opendpd\.import(?:IQ|MAT)\(", line):
            continue
        statement, index = line[line.index("opendpd.import"):], number
        while statement.rstrip().endswith("...") and index + 1 < len(lines):
            index += 1
            statement += " " + lines[index]
        yield path.relative_to(TOOLBOX), " ".join(statement.split())


def test_every_documented_import_states_its_segment_length():
    files = [TOOLBOX / "README.md", *sorted((TOOLBOX / "docs").glob("*.md")), *sorted((TOOLBOX / "examples").glob("*.m"))]
    statements = [item for path in files for item in _import_statements(path)]
    assert len(statements) >= 5, "the documented import examples were not found"
    missing = [f"{path}: {text}" for path, text in statements if "SegmentSamples" not in text]
    assert not missing, "\n".join(missing)


def test_segment_length_has_no_default_anywhere():
    for relative in ("+opendpd/importIQ.m", "+opendpd/importMAT.m"):
        text = _text(*relative.split("/"))
        assert re.search(r"options\.SegmentSamples \(1,1\) double \{[^}]*\}\n", text), relative
        assert "SegmentSamples (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.SegmentSamples, 2)} =" not in text
    sdk = (ROOT / "opendpd" / "sdk" / "client.py").read_text(encoding="utf-8")
    assert not re.search(r"nperseg\s*=\s*\d", sdk)
