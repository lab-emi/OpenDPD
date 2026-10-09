"""``opendpd.sdk.fit``: paired I/Q in, trained PA and DPD out as model packages, with cancel, timeout and a process transport."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from opendpd.sdk import SDKError, fit, open_project
from opendpd.services.workspace import Workspace
from tests.fixtures.synthetic import synthesize

pytestmark = pytest.mark.integration
FS = 80e6
SIGNAL = dict(sample_rate_hz=FS, bandwidth_hz=20e6, nperseg=128, origin="synthetic")
TINY = {"epochs": 1, "frame_length": 32, "frame_stride": 32, "batch_size": 16, "batch_size_eval": 16}
LONG = {**TINY, "epochs": 400}           # long enough that a cancel request always lands while it is running


@pytest.fixture(scope="module")
def capture():
    return synthesize(4096, fs=FS, bandwidth=20e6)


def manifest_of(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("manifest.json"))


def test_fit_trains_exports_and_reports_progress(tmp_path, capture):
    x, y = capture
    events = []
    result = fit(tmp_path / "ws", x, y, **SIGNAL, pa_model="gru", dpd_model="gmp", pa_parameters={"hidden_size": 5},
                 training=TINY, device="cpu", on_event=events.append, poll_interval=0.1)
    assert result["pa"]["status"] == result["dpd"]["status"] == "succeeded"
    for role, key in (("pa", "gru"), ("dpd", "gmp")):
        package = Path(result[role]["package"]["path"])
        assert package.parent == (tmp_path / "ws" / "exports").resolve() and package.name == f"{result[role]['run_id']}.opendpd.zip"
        manifest = manifest_of(package)
        assert manifest["model"]["key"] == key and manifest["run"]["role"] == role
        assert manifest["run"]["run_id"] == result[role]["run_id"]
        assert result[role]["package"]["sha256"]
    stages = [event["stage"] for event in events]
    assert stages[0] == "import" and stages.index("train_pa") < stages.index("train_dpd") < stages.index("export")
    assert {e["role"] for e in events if e["stage"] == "export"} == {"pa", "dpd"}
    assert any(e["stage"] == "train_pa" and e["status"] == "succeeded" for e in events)
    assert result["dataset"]["n_samples"] == len(x) and result["seconds"] > 0
    json.dumps(result, allow_nan=False)
    # the runs are ordinary workspace runs: another project can open them, and the DPD cascades with the fitted PA
    project = open_project(tmp_path / "ws")
    try:
        assert project.job(result["dpd"]["run_id"]).config()["pa_reference"]["run_id"] == result["pa"]["run_id"]
    finally:
        project.close(stop_service=True)


def test_fit_refuses_models_it_cannot_export_before_it_trains_anything(tmp_path, capture):
    x, y = capture
    for kwargs in ({"dpd_model": "lstm"}, {"pa_model": "deltagru"}):
        with pytest.raises(ValueError, match="cannot be exported; fit supports gru, tres_gru, gmp, mp_ls, gmp_ls"):
            fit(tmp_path / "ws", x, y, **SIGNAL, training=TINY, **kwargs)
    for key in ("mp_ls", "gmp_ls"):        # exportable, but the DPD cannot be trained through a least-squares PA
        with pytest.raises(ValueError, match="least-squares baseline, not a DPD surrogate"):
            fit(tmp_path / "ws", x, y, **SIGNAL, training=TINY, pa_model=key)
    assert not (tmp_path / "ws").exists()                      # nothing was created, let alone trained
    with pytest.raises(ValueError, match="timeout"):
        fit(tmp_path / "ws", x, y, **SIGNAL, timeout=0)


def test_fit_cancels_the_running_job_and_writes_nothing(tmp_path, capture):
    x, y = capture
    with pytest.raises(SDKError, match="cancelled"):
        fit(tmp_path / "ws", x, y, **SIGNAL, training=LONG, device="cpu", cancelled=lambda: True, poll_interval=0.1)
    ws = Workspace.open(tmp_path / "ws")
    states = {path.parent.name: json.loads(path.read_text())["status"] for path in (ws.root / "runs").glob("*/run.json")}
    assert states and set(states.values()) == {"cancelled"}
    assert not (ws.root / "exports").exists() or not list((ws.root / "exports").glob("*.zip"))


def test_fit_times_out_by_cancelling(tmp_path, capture):
    x, y = capture
    with pytest.raises(SDKError, match="timeout"):
        fit(tmp_path / "ws", x, y, **SIGNAL, training=LONG, device="cpu", timeout=0.5, poll_interval=0.1)
    ws = Workspace.open(tmp_path / "ws")
    assert {json.loads(p.read_text())["status"] for p in (ws.root / "runs").glob("*/run.json")} == {"cancelled"}


def test_the_process_transport_runs_the_same_fit(tmp_path, capture):
    """What the MATLAB toolbox does: files in a job folder, a plain Python process, no Python-in-MATLAB."""
    x, y = capture
    job = tmp_path / "job"
    job.mkdir()
    np.save(job / "x.npy", x.astype(np.float32), allow_pickle=False)
    np.save(job / "y.npy", y.astype(np.float32), allow_pickle=False)
    request = {"workspace": str(tmp_path / "ws"), **SIGNAL, "training": TINY, "device": "cpu", "pa_model": "gru",
               "pa_parameters": {"hidden_size": 5}, "dpd_model": "mp_ls", "dpd_parameters": {"K": 3, "Q": 4}, "poll_interval": 0.1}
    (job / "request.json").write_text(json.dumps(request))
    root = Path(__file__).resolve().parents[2]
    env = {**__import__("os").environ, "PYTHONPATH": str(root)}
    done = subprocess.run([sys.executable, "-m", "opendpd.sdk._fit", "--job", str(job)], env=env, cwd=root,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads((job / "result.json").read_text())
    assert manifest_of(result["dpd"]["package"]["path"])["model"]["key"] == "mp_ls"
    lines = [json.loads(line) for line in (job / "progress.jsonl").read_text().splitlines()]
    assert lines[0]["stage"] == "import" and any(line["stage"] == "export" for line in lines)
    assert not (job / "error.json").exists()


def test_the_process_transport_stops_on_the_cancel_file_and_reports_why(tmp_path, capture):
    x, y = capture
    job = tmp_path / "job"
    job.mkdir()
    np.save(job / "x.npy", x.astype(np.float32), allow_pickle=False)
    np.save(job / "y.npy", y.astype(np.float32), allow_pickle=False)
    (job / "request.json").write_text(json.dumps({"workspace": str(tmp_path / "ws"), **SIGNAL, "training": LONG,
                                                  "device": "cpu", "poll_interval": 0.1}))
    (job / "cancel").write_text("")                                  # the user pressed Ctrl+C before training got far
    root = Path(__file__).resolve().parents[2]
    done = subprocess.run([sys.executable, "-m", "opendpd.sdk._fit", "--job", str(job)],
                          env={**__import__("os").environ, "PYTHONPATH": str(root)}, cwd=root,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 130
    assert json.loads((job / "error.json").read_text())["code"] == "cancelled" and not (job / "result.json").exists()


def test_a_bad_request_is_an_error_file_not_a_hang(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "request.json").write_text("{not json")
    root = Path(__file__).resolve().parents[2]
    done = subprocess.run([sys.executable, "-m", "opendpd.sdk._fit", "--job", str(job)],
                          env={**__import__("os").environ, "PYTHONPATH": str(root)}, cwd=root,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 1 and json.loads((job / "error.json").read_text())["code"] == "fit_failed"


def test_fit_into_a_workspace_with_a_running_studio_returns_its_result_and_leaves_that_service_running(tmp_path, capture):
    """A person who has the workspace open in Studio calls fit again: the fit must finish normally, and the service they
    were using must still be there. (A service started outside the SDK cannot be stopped by it at all: asking raises.)"""
    from opendpd.studio.launcher import LOCK_FILE, Lock

    x, y = capture
    workspace = tmp_path / "ws"
    first = open_project(workspace)                       # the service somebody else is using
    try:
        lock = Lock.read(first.workspace / LOCK_FILE)
        (first.workspace / ".sdk-service.json").unlink()   # ... which the SDK did not launch (as a Studio started by hand)
        result = fit(workspace, x, y, **SIGNAL, pa_model="gru", dpd_model="gru", pa_parameters={"hidden_size": 4},
                     training=TINY, device="cpu", poll_interval=0.1)
        assert result["pa"]["status"] == result["dpd"]["status"] == "succeeded"
        assert lock.alive(), "fit stopped a service it did not start"
        assert first.models(), "the other connection still works"
    finally:
        first._process.terminate()
        first._process.wait(timeout=30)


def test_fit_leaves_a_service_that_another_connection_started_running(tmp_path, capture):
    """The service was launched by the SDK, but by another connection (a second MATLAB session, say): fit did not start
    it and must not stop it, because that other session may be using it."""
    from opendpd.studio.launcher import LOCK_FILE, Lock

    x, y = capture
    workspace = tmp_path / "ws"
    first = open_project(workspace)
    try:
        lock = Lock.read(first.workspace / LOCK_FILE)
        result = fit(workspace, x, y, **SIGNAL, pa_model="gru", dpd_model="gru", pa_parameters={"hidden_size": 4},
                     training=TINY, device="cpu", poll_interval=0.1)
        assert result["dpd"]["status"] == "succeeded"
        assert lock.alive(), "fit stopped a service that another connection started"
        assert first.models()
    finally:
        first.close(stop_service=True)


def test_fit_stops_the_service_it_started_and_only_that_one(tmp_path, capture):
    from opendpd.studio.launcher import LOCK_FILE, Lock

    x, y = capture
    workspace = tmp_path / "ws"
    fit(workspace, x, y, **SIGNAL, pa_model="gru", dpd_model="gru", pa_parameters={"hidden_size": 4},
        training=TINY, device="cpu", poll_interval=0.1)
    lock = Lock.read((workspace / LOCK_FILE).resolve())
    assert lock is None or not lock.alive(), "the service fit started is still running"


class _FakeProject:
    def __init__(self, active=0, fail=None):
        self.active, self.fail, self.stop_requests = active, fail, []

    def active_run_count(self):
        return self.active

    def close(self, *, stop_service=False):
        self.stop_requests.append(stop_service)
        if self.fail:
            raise self.fail


def test_release_stops_the_service_only_when_fit_started_it_and_nothing_else_is_running():
    from opendpd.sdk.workflow import _release

    for started, active, expected in ((True, 0, True), (True, 1, False), (False, 0, False), (False, 3, False)):
        project = _FakeProject(active=active)
        _release(project, started, None)
        assert project.stop_requests == [expected], (started, active)


def test_release_reports_a_failure_to_stop_and_never_raises_it():
    from opendpd.sdk.workflow import _release

    events = []
    _release(_FakeProject(fail=SDKError("shutdown_timeout", "Service is still stopping")), True, events.append)
    assert len(events) == 1 and events[0]["stage"] == "close" and "still stopping" in events[0]["warning"]
    _release(_FakeProject(fail=SDKError("external_service", "not ours")), False, None)       # no emitter: still silent
