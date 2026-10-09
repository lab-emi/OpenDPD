"""The toolbox's version, supported models and documented calls agree with each other and with the code."""

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from opendpd.core.measurement import to_iq
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
    classes = {"Job", "Project", "MATLABBridge", "Model", "FixedModel"}
    top = [p.stem for p in (TOOLBOX / "+opendpd").glob("*.m") if p.stem not in classes]
    return sorted(top), sorted(p.stem for p in (TOOLBOX / "+opendpd" / "+metrics").glob("*.m"))


def test_every_public_function_is_documented_and_listed():
    top, metrics = _public_functions()
    assert {"apply", "waveform", "export", "load", "verify", "fit"} <= set(top) and {"evm", "aclr", "evaluate"} <= set(metrics)
    reference, contents = _text("docs", "reference.md"), _text("Contents.m")
    for name in top:
        assert f"opendpd.{name}(" in reference, f"{name} is not in the function reference"
        assert re.search(rf"\b{name}\b", contents), f"{name} is not listed in Contents.m"
    for name in metrics:
        assert f"opendpd.metrics.{name}(" in reference, f"metrics.{name} is not in the function reference"
        assert f"metrics.{name}" in contents
    assert "`opendpd.Model`" in reference and re.search(r"\bModel\b", contents)
    assert "`opendpd.FixedModel`" in reference and re.search(r"\bFixedModel\b", contents)


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


def test_the_model_runtime_runs_no_python_and_loads_nothing():
    """The pure-MATLAB runtime and the package reader never call Python and never call anything that can run code.

    ``TestPackageSecurity`` checks the same in MATLAB; this copy fails in the Python CI too. ``whos -file`` and ``load`` are
    on the list because both call ``loadobj`` for classes on the MATLAB path (shown by ``TestPackageSecurity``)."""
    forbidden = re.compile(r"(?<![A-Za-z0-9_.])(load|whos|matfile|eval|evalc|evalin|feval|str2func|unzip|run|system|dos|unix|"
                           r"urlread|webread)\s*\(")
    files = [TOOLBOX / "+opendpd" / name for name in ("Model.m", "verify.m", "load.m")]
    files += [TOOLBOX / "+opendpd" / name for name in ("FixedModel.m",)]
    files += [TOOLBOX / "+opendpd" / "+internal" / name
              for name in ("readPackage.m", "readNpz.m", "readNpy.m", "copyZipEntry.m", "sha256.m", "readFixedPackage.m",
                           "packageKind.m")]
    files += sorted((TOOLBOX / "+opendpd" / "+runtime").glob("*.m"))
    assert len(files) >= 20
    for path in files:
        text = path.read_text(encoding="utf-8")
        if path.name == "load.m":
            text = text.replace("function model = load(", "")
        assert not forbidden.search(text), f"{path.name} calls something that can run code"
        assert not re.search(r"(?<![A-Za-z0-9_.])py\.", text), f"{path.name} refers to Python"
        assert not re.search(r"(?<![A-Za-z0-9_.])bridge\s*\(", text), f"{path.name} calls the bridge"


def test_the_process_transport_never_calls_python_inside_matlab():
    """``fit`` starts Python as a child process; nothing on that path may use MATLAB's Python integration.

    Reading ``pyenv().Executable`` is allowed (it does not load Python); the ``py.`` namespace and the bridge are not.
    ``TestFit`` also checks that the pyenv status does not change, but that check is vacuous in a session where an earlier
    test already loaded Python, so this scan is the one that always holds."""
    files = [TOOLBOX / "+opendpd" / "fit.m"]
    files += [TOOLBOX / "+opendpd" / "+internal" / name for name in
              ("runPython.m", "pythonExecutable.m", "writeNpy.m", "iqMatrix.m", "ProgressPrinter.m", "withoutMatlabEntries.m")]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"(?<![A-Za-z0-9_.])py\.", text), f"{path.name} refers to Python"
        assert not re.search(r"(?<![A-Za-z0-9_.])bridge\s*\(", text), f"{path.name} calls the bridge"
        assert not re.search(r"pyenv\s*\(\s*[^)\s]", text), f"{path.name} configures pyenv"


def test_the_lab_session_is_documented_and_never_opens_the_rf_gate_or_calls_python():
    """``opendpd.lab`` drives instruments, so its sources are held to the rules the Python interlock lives by.

    Nothing in the toolbox may set ``OPENDPD_ALLOW_RF_OUTPUT`` (``TestLab`` also scans for it); the check is repeated here so
    that the Python CI fails too. The session needs no Python at all."""
    folder = TOOLBOX / "+opendpd" / "+lab"
    files = sorted(folder.glob("*.m"))
    assert {p.stem for p in files} >= {"Session", "MockInstrument", "iqHash"}
    reference, contents, readme = _text("docs", "reference.md"), _text("Contents.m"), _text("README.md")
    for name in ("Session", "MockInstrument"):
        assert f"opendpd.lab.{name}(" in reference, f"lab.{name} is not in the function reference"
        assert f"lab.{name}" in contents, f"lab.{name} is not listed in Contents.m"
    assert "opendpd.lab.Session" in readme and "OPENDPD_ALLOW_RF_OUTPUT" in reference
    for path in [*files, *sorted((TOOLBOX / "tests").glob("*.m"))]:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"setenv\s*\(\s*['\"]OPENDPD_ALLOW_RF_OUTPUT", text), f"{path.name} opens the RF output gate"
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"setenv\s*\(", text), f"{path.name} sets an environment variable"
        assert not re.search(r"(?<![A-Za-z0-9_.])py\.", text), f"{path.name} refers to Python"
        assert not re.search(r"(?<![A-Za-z0-9_.])bridge\s*\(", text), f"{path.name} calls the bridge"


def test_the_lab_session_hashes_signals_the_way_python_session_records_do():
    """``TestLab`` pins the same literal for ``opendpd.lab.iqHash``: float32 interleaved I/Q, as ``run_capture_session`` hashes
    ``to_iq(z).tobytes()``. If either side changes how it hashes, one of the two tests fails."""
    z = np.arange(1, 9) / 16 + 1j * (np.arange(-4, 4) / 16)
    digest = hashlib.sha256(to_iq(z).tobytes()).hexdigest()
    assert digest == "45b6a1346b0daf8b0e93e0faf712d6245e4325c632d4007cbf186d66041d8f86"
    assert digest in _text("tests", "TestLab.m")



def test_the_code_generator_is_documented_runs_nothing_and_its_kernels_call_only_each_other():
    """``opendpd.generateCode`` writes code that users run, so its own sources are held to the rules of the runtime.

    The kernels it copies into a generated class are the files of ``+runtime``; the copy drops the ``opendpd.runtime.``
    qualifier, so a kernel that called anything else of the toolbox would produce a class that fails to run. ``TestCodegen``
    runs the generated classes with the toolbox off the path; this scan fails in the Python CI too."""
    reference, contents, readme = _text("docs", "reference.md"), _text("Contents.m"), _text("README.md")
    assert "opendpd.generateCode(" in reference and re.search(r"\bgenerateCode\b", contents) and "generateCode" in readme
    assert "examples/opendpdSimulink.m" in reference and (TOOLBOX / "examples" / "opendpdSimulink.m").is_file()
    forbidden = re.compile(r"(?<![A-Za-z0-9_.])(load|whos|matfile|eval|evalc|evalin|feval|str2func|unzip|run|system|dos|unix|"
                           r"urlread|webread)\s*\(")
    for relative in (("+opendpd", "generateCode.m"), ("+opendpd", "+internal", "generateSources.m"),
                     ("+opendpd", "+internal", "codegenMark.m")):
        text = _text(*relative)
        assert not forbidden.search(text), f"{relative[-1]} calls something that can run code"
        assert not re.search(r"(?<![A-Za-z0-9_.])py\.", text), f"{relative[-1]} refers to Python"
        assert not re.search(r"(?<![A-Za-z0-9_.])bridge\s*\(", text), f"{relative[-1]} calls the bridge"
    kernels = {path.stem for path in (TOOLBOX / "+opendpd" / "+runtime").glob("*.m")}
    assert {"gruLayer", "tresFeatures", "tresSkip", "mpForward", "gmpForward", "gmpPolynomialForward"} <= kernels
    assert "gruStack" not in kernels, "the cell-array GRU stack cannot be built by MATLAB Coder; generateCode needs gruLayer"
    for stem in kernels:
        text = _text("+opendpd", "+runtime", f"{stem}.m")
        for called in re.findall(r"(?<![A-Za-z0-9_.])opendpd\.([A-Za-z_.]+)\(", text):
            assert called.startswith("runtime.") and called.split(".")[1] in kernels, f"{stem}.m calls opendpd.{called}"
    generated = _text("+opendpd", "+internal", "generateSources.m")
    for stem in ("gruLayer", "tresFeatures", "tresSkip", "mpForward", "gmpForward", "gmpPolynomialForward"):
        assert f"'{stem}'" in generated, f"generateSources does not copy {stem}"


def test_the_fixed_point_reader_is_documented_and_its_kernel_is_plain_matlab():
    """``opendpd.load`` returns an ``opendpd.FixedModel`` for a fixed-point-v1 package; the documentation, the guide, the
    tutorial and the CI workflow all say so, and the integer kernel is a ``+runtime`` function that calls only other kernels."""
    reference, readme, contents = _text("docs", "reference.md"), _text("README.md"), _text("Contents.m")
    workflow = _text("docs", "workflow.md")
    for text in (reference, workflow, readme):
        assert "runInteger" in text or "FixedModel" in text, "a fixed-point page lacks the call"
    assert "model.runInteger(" in reference and "opendpd.FixedModel" in reference and "FixedModel" in contents
    assert "opendpd:CodegenFixedPoint" in reference
    tutorial = (ROOT / "docs" / "tutorials" / "deployment-export.md").read_text(encoding="utf-8")
    assert "## 4. Check it in MATLAB" in tutorial and "opendpd.verify(model)" in tutorial
    workflow_file = (ROOT / ".github" / "workflows" / "matlab-toolbox.yml").read_text(encoding="utf-8")
    assert "gru-pa.fixed-point-v1.zip" in workflow_file and "tests/unit/test_fixed_point_package.py" in workflow_file
    assert (TOOLBOX / "tests" / "data" / "gru-pa.fixed-point-v1.zip").is_file()
    assert (TOOLBOX / "tests" / "data" / "gru-pa-custom.fixed-point-v1.zip").is_file()
    kernels = {path.stem for path in (TOOLBOX / "+opendpd" / "+runtime").glob("*.m")}
    assert {"fixedGruRun", "fixedRescale"} <= kernels
    # the verdict and the arguments of the entry points name the new class
    for name in ("load", "verify", "apply"):
        assert "FixedModel" in _text("+opendpd", f"{name}.m"), name
