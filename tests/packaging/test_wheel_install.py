"""L2 packaging test: the built wheel works from outside the repository.

Builds the wheel, installs it (without the ``gui`` extra) into a fresh venv
that borrows the already-installed heavy dependencies through
``--system-site-packages``, then runs a headless experiment from a temporary
workspace with the repository *not* on sys.path. This proves that the wheel
ships the datasets and top-level modules, that ``import opendpd`` has no GUI
dependency, and that runs never write into the installation directory.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.packaging

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd, cwd, env, timeout=900):
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, f"{' '.join(map(str, cmd))}\n--- stdout ---\n{proc.stdout[-3000:]}\n--- stderr ---\n{proc.stderr[-3000:]}"
    return proc


@pytest.fixture(scope="module")
def installed_python(tmp_path_factory):
    work = tmp_path_factory.mktemp("pkg")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(MPLBACKEND="Agg", TQDM_DISABLE="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(work / "dist"), str(REPO_ROOT)], work, env)
    wheel = next((work / "dist").glob("opendpd-*.whl"))
    venv = work / "venv"
    _run([sys.executable, "-m", "venv", str(venv)], work, env)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    # Borrow the heavy dependencies (torch, numpy, pydantic ...) already present
    # in the developer environment instead of downloading them: a plain path
    # entry in a .pth file adds the directory without activating its editable
    # installs, so the wheel copy of opendpd is the one that gets imported.
    site_packages = Path(subprocess.run([str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                                        capture_output=True, text=True, env=env).stdout.strip())
    (site_packages / "zz_borrowed_deps.pth").write_text(sysconfig_purelib() + "\n")
    _run([str(python), "-m", "pip", "install", "--quiet", "--no-deps", str(wheel)], work, env)
    return python, work, env


def sysconfig_purelib() -> str:
    import sysconfig
    return sysconfig.get_paths()["purelib"]


def test_wheel_imports_without_gui_deps_and_ships_datasets(installed_python):
    python, work, env = installed_python
    check = (
        "import sys, opendpd, opendpd.schemas, opendpd.core.registry, opendpd.commands; "
        "assert opendpd.__file__.startswith(sys.prefix), opendpd.__file__; "
        "assert not any(m in sys.modules for m in ('fastapi', 'uvicorn', 'torch')); "
        "from opendpd.services.workspace import BUILTIN_DATASETS_DIR; "
        "assert (BUILTIN_DATASETS_DIR / 'DPA_200MHz' / 'spec.json').is_file(), BUILTIN_DATASETS_DIR; "
        "print('ok', opendpd.__version__)"
    )
    proc = _run([str(python), "-c", check], work, env)
    assert proc.stdout.startswith("ok")


def test_wheel_runs_headless_from_external_workspace(installed_python):
    python, work, env = installed_python
    ws = work / "workspace with spaces"
    opendpd_cli = [str(python), "-m", "opendpd.commands"]
    _run(opendpd_cli + ["datasets", "import-builtin", "DPA_200MHz", "--workspace", str(ws)], work, env)
    proc = _run(opendpd_cli + ["run", "--recipe", "pa-gru-smoke-v1", "--dataset", "dpa-200mhz",
                               "--workspace", str(ws)], work, env)
    assert "succeeded" in proc.stdout and "NMSE" in proc.stdout
    runs = list((ws / "runs").glob("run-*"))
    assert len(runs) == 1 and (runs[0] / "result.json").exists()
    # nothing was written next to the installed package or in the cwd
    site = Path(subprocess.run([str(python), "-c", "import opendpd, os; print(os.path.dirname(opendpd.__file__))"],
                               capture_output=True, text=True, env=env).stdout.strip())
    assert not (site.parent / "save").exists() and not (work / "save").exists()


STATIC_INDEX = REPO_ROOT / "opendpd" / "studio" / "static" / "index.html"


@pytest.mark.skipif(not STATIC_INDEX.exists(), reason="frontend not built (run `npm run build` in frontend/)")
def test_wheel_ships_gui_and_one_command_serves_it(installed_python):
    """`opendpd gui --no-browser` from the wheel: healthy, ready, serves the page, exits clean."""
    import json
    import socket
    import time
    import urllib.request

    python, work, env = installed_python
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ws = work / "gui 工作区 with spaces"
    env_gui = dict(env, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([str(python), "-m", "opendpd.commands", "gui", "--no-browser", "--port", str(port),
                             "--workspace", str(ws)], cwd=work, env=env_gui, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 60
        healthy = None
        while time.monotonic() < deadline and healthy is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as r:
                    healthy = json.loads(r.read())
            except Exception:
                time.sleep(0.2)
        assert healthy == {"status": "ok"}, "service never became healthy"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/readyz", timeout=5) as r:
            ready = json.loads(r.read())
        assert ready["ready"] is True and ready["frontend"]["present"] is True, ready
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            html = r.read().decode()
        assert "<div id=\"root\">" in html and "/assets/" in html
        # lock file exists while running, printed URL carries the bootstrap token
        assert (ws / ".studio.lock").exists()
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
            pytest.fail("gui did not stop within 30 s after SIGTERM")
    assert "OpenDPD Studio: http://127.0.0.1:" in out and "/bootstrap?token=" in out, out[-2000:]
    assert not (ws / ".studio.lock").exists(), "lock must be released on exit"
    site = Path(subprocess.run([str(python), "-c", "import opendpd.studio, os; print(os.path.dirname(opendpd.studio.__file__))"],
                               capture_output=True, text=True, env=env).stdout.strip())
    assert (Path(site) / "static" / "build-info.json").exists(), "wheel must carry the built frontend"
