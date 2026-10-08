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


def _public_functions():
    """Function files a user can call as ``opendpd.<name>`` or ``opendpd.metrics.<name>`` (classes are documented apart)."""
    classes = {"Job", "Project", "MATLABBridge"}
    top = [p.stem for p in (TOOLBOX / "+opendpd").glob("*.m") if p.stem not in classes]
    return sorted(top), sorted(p.stem for p in (TOOLBOX / "+opendpd" / "+metrics").glob("*.m"))


def test_every_public_function_is_documented_and_listed():
    top, metrics = _public_functions()
    assert {"apply", "waveform"} <= set(top) and {"evm", "aclr", "evaluate"} <= set(metrics)
    reference, contents = _text("docs", "reference.md"), _text("Contents.m")
    for name in top:
        assert f"opendpd.{name}(" in reference, f"{name} is not in the function reference"
        assert re.search(rf"\b{name}\b", contents), f"{name} is not listed in Contents.m"
    for name in metrics:
        assert f"opendpd.metrics.{name}(" in reference, f"metrics.{name} is not in the function reference"
        assert f"metrics.{name}" in contents


def test_shared_helpers_have_one_implementation():
    # a private folder is invisible to the +metrics sub-package, so the helpers live in opendpd.internal
    assert (TOOLBOX / "+opendpd" / "+internal" / "bridge.m").is_file()
    assert (TOOLBOX / "+opendpd" / "+internal" / "asPythonIQ.m").is_file()
    for name in ("bridge", "asPythonIQ"):
        delegate = _text("+opendpd", "private", f"{name}.m")
        assert f"opendpd.internal.{name}(" in delegate and len(delegate.splitlines()) <= 5, name


def test_the_packaged_guide_is_current_and_carries_the_toolbox_version():
    markdown = pytest.importorskip("markdown")        # noqa: F841  (the generator needs it)
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_matlab_docs", ROOT / "scripts" / "build_matlab_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    version = re.search(r"ToolboxVersion = '(\d+\.\d+\.\d+)'", _text("buildfile.m")).group(1)
    assert module.toolbox_version() == version
    module.build(check=True)                           # raises SystemExit if a packaged page is stale
    for page in (TOOLBOX / "resources" / "docs").glob("*.html"):
        text = page.read_text(encoding="utf-8")
        assert f'<span class="version">{version}</span>' in text and "PREVIEW" not in text, page.name
