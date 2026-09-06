"""S13 negative tests: the boundaries the threat model promises, exercised through the real ASGI app.

Every test here is a refusal: a malicious page, path, archive, checkpoint or body must be turned
away with a structured error and must leave no trace in the workspace.
"""

import json
import re
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from opendpd.schemas import RunStatus
from opendpd.server import routes as routes_module
from opendpd.server.app import _diagnostic_page, create_app
from opendpd.server.security import CONTENT_SECURITY_POLICY, CSRF_HEADER
from opendpd.services import packages
from opendpd.services.experiments import classify_failure
from opendpd.services.legacy_adapter import CheckpointRefused, load_checkpoint
from opendpd.services.recipes import instantiate
from tests.integration.test_studio_api import wait_terminal

pytestmark = pytest.mark.integration
TOKEN = "hardening-token"
FRONTEND_SRC = Path(__file__).resolve().parents[2] / "frontend" / "src"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(tmp_path_factory.mktemp("ws"), bootstrap_token=TOKEN,
                     supervisor_kwargs={"poll_interval": 0.1, "cancel_grace": 20}, shutdown_timeout=3)
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        r = c.post("/api/v1/session/bootstrap", json={"token": TOKEN})
        assert r.status_code == 200, r.text
        c.headers[CSRF_HEADER] = r.json()["csrf_token"]
        assert c.post("/api/v1/datasets/import-builtin", json={"name": "DPA_200MHz"}).status_code == 201
        yield c


@pytest.fixture(scope="module")
def ws(client):
    return client.app.state.ws


@pytest.fixture(scope="module")
def pa_run(client):
    cfg = instantiate("pa-gru-smoke-v1", "dpa-200mhz")
    cfg = cfg.model_copy(update={"training": cfg.training.model_copy(update={"epochs": 2})})
    r = client.post("/api/v1/runs", json={"config": json.loads(cfg.model_dump_json()), "name": "hardening PA"})
    assert r.status_code == 201, r.text
    record = wait_terminal(client, r.json()["run_id"])
    assert record["status"] == RunStatus.succeeded.value, record
    return record


@pytest.fixture(scope="module")
def share_package(client, pa_run, tmp_path_factory):
    r = client.post("/api/v1/exports", json={"run_id": pa_run["run_id"], "kind": "share"})
    assert r.status_code == 201, r.text
    body = client.get(r.json()["download_url"]).content
    path = tmp_path_factory.mktemp("pkg") / "share.zip"
    path.write_bytes(body)
    return path


def _rebuild(source: Path, target: Path, mutate) -> Path:
    """Copy a package, letting ``mutate(zin, zout)`` decide what changes."""
    with zipfile.ZipFile(source) as zin, zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        mutate(zin, zout)
    return target


# --- headers -------------------------------------------------------------------------------

def test_every_response_carries_the_security_headers(client):
    for path in ("/api/v1/runs", "/api/v1/nope", "/", "/readyz"):
        r = client.get(path)
        assert r.headers["content-security-policy"] == CONTENT_SECURITY_POLICY, path
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["referrer-policy"] == "same-origin"
    assert client.get("/api/v1/runs").headers["cache-control"] == "no-store"
    assert "cache-control" not in client.get("/").headers          # the static shell may be cached
    assert "script-src 'self'" in CONTENT_SECURITY_POLICY and "'unsafe-eval'" not in CONTENT_SECURITY_POLICY


def test_diagnostic_pages_escape_their_text():
    page = _diagnostic_page("<script>alert(1)</script>", "x & <b>y</b>")
    assert "<script>" not in page and "&lt;script&gt;" in page and "&amp; &lt;b&gt;" in page


def test_frontend_sources_have_no_html_sinks():
    sinks = ("dangerouslySetInnerHTML", ".innerHTML", "eval(", "new Function(", "document.write(")
    offenders = []
    for path in FRONTEND_SRC.rglob("*.ts*"):
        text = path.read_text(encoding="utf-8")
        offenders += [f"{path.relative_to(FRONTEND_SRC)}: {s}" for s in sinks if s in text]
    assert not offenders, offenders


def test_user_text_is_stored_and_returned_as_data(client, pa_run):
    payload = "<img src=x onerror=alert(1)>"
    cfg = instantiate("pa-gru-smoke-v1", "dpa-200mhz")
    r = client.post("/api/v1/experiments/validate", json={"config": json.loads(cfg.model_dump_json())})
    assert r.status_code == 200, r.text
    r = client.post("/api/v1/runs", json={"config": json.loads(cfg.model_dump_json()), "name": payload})
    assert r.status_code == 201, r.text
    assert client.get(f"/api/v1/runs/{r.json()['run_id']}").json()["name"] == payload   # JSON, escaped by React
    assert client.post(f"/api/v1/runs/{r.json()['run_id']}/cancel").status_code in (200, 409)


# --- bodies --------------------------------------------------------------------------------

def test_chunked_bodies_without_content_length_are_capped(client):
    def chunks():
        for _ in range(3):
            yield b"x" * (1024 * 1024)

    r = client.post("/api/v1/runs", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large", r.text


def test_oversized_uploads_are_413_and_leave_no_file(client, ws, share_package, monkeypatch):
    monkeypatch.setattr(routes_module, "UPLOAD_MAX_BODY", 1024)
    big_csv = b"I_in,Q_in,I_out,Q_out\n" + b"0.1,0.2,0.3,0.4\n" * 400
    r = client.post("/api/v1/datasets/upload", files={"file": ("big.csv", big_csv, "text/csv")})
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large", r.text
    assert not list((ws.imports_dir / "uploads").glob("*big.csv"))
    r = client.post("/api/v1/imports", files={"file": ("share.zip", share_package.read_bytes(), "application/zip")})
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_large", r.text
    assert not list((ws.imports_dir / "packages").glob("*share*.zip"))


# --- paths and symlinks ----------------------------------------------------------------------

def test_artifact_symlink_outside_the_run_is_refused(client, ws, pa_run, tmp_path):
    from opendpd.services.workspace import sha256_file

    run_id = pa_run["run_id"]
    secret = tmp_path / "secret.txt"
    secret.write_text("not for the browser")
    run_dir = ws.run_dir(run_id)
    (run_dir / "leak.txt").symlink_to(secret)
    manifest_path = run_dir / "artifacts.json"
    original = manifest_path.read_text()
    manifest = json.loads(original)
    leak = dict(manifest["artifacts"][0], artifact_id="leak", required=False,
                file={"path": "leak.txt", "sha256": sha256_file(secret), "size_bytes": secret.stat().st_size})
    manifest["artifacts"].append(leak)
    manifest_path.write_text(json.dumps(manifest))
    try:
        r = client.get(f"/api/v1/artifacts/{run_id}/leak")
        assert r.status_code == 404 and r.json()["error"]["code"] == "artifact_missing", r.text
        assert client.get(f"/api/v1/artifacts/{run_id}/..%2F..%2Fworkspace.json").status_code == 404
        assert client.get(f"/api/v1/artifacts/{run_id}/{manifest['artifacts'][0]['artifact_id']}").status_code == 200
    finally:
        manifest_path.write_text(original)
        (run_dir / "leak.txt").unlink()


def test_import_root_symlink_escapes_are_invisible_and_unreadable(client, ws, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.csv").write_text("I_in,Q_in,I_out,Q_out\n0,0,0,0\n")
    outside = tmp_path / "outside.csv"
    outside.write_text("I_in,Q_in,I_out,Q_out\n1,1,1,1\n")
    (root / "escape.csv").symlink_to(outside)
    (root / "escape-dir").symlink_to(tmp_path)
    ws.add_import_root("data", root)
    r = client.get("/api/v1/datasets/import-roots/data/files")
    assert r.status_code == 200 and [e["path"] for e in r.json()] == ["ok.csv"], r.text
    for path in ("escape.csv", "escape-dir/outside.csv", "../outside.csv"):
        r = client.post("/api/v1/datasets/inspect", json={"root_id": "data", "path": path})
        assert r.status_code in (404, 409) and "escapes" in r.text, (path, r.text)


# --- archives ----------------------------------------------------------------------------------

def test_symlink_members_are_refused(share_package, tmp_path, client):
    assert packages.inspect_package(share_package).run_id      # the unmodified package passes every check

    def mutate(zin, zout):
        for item in zin.infolist():
            zout.writestr(item, zin.read(item.filename))
        link = zipfile.ZipInfo("run/link")
        link.external_attr = (0o120777 << 16)      # a symlink entry pointing wherever the attacker likes
        zout.writestr(link, "/etc/passwd")

    bad = _rebuild(share_package, tmp_path / "symlink.zip", mutate)
    with pytest.raises(packages.PackageError) as info:
        packages.inspect_package(bad)
    assert info.value.code == "unsafe_member"
    r = client.post("/api/v1/imports", files={"file": ("symlink.zip", bad.read_bytes(), "application/zip")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "unsafe_member", r.text


def test_members_larger_than_recorded_are_refused_early(share_package, tmp_path):
    def mutate(zin, zout):
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith("config.resolved.json"):
                data = b"\0" * (64 * 1024 * 1024)          # 64 MB of zeros compress to a few KB: a small bomb
            zout.writestr(item, data)

    bomb = _rebuild(share_package, tmp_path / "bomb.zip", mutate)
    assert bomb.stat().st_size < 1024 * 1024
    with pytest.raises(packages.PackageError) as info:
        packages.inspect_package(bomb)
    assert info.value.code == "hash_mismatch" and "larger than its recorded size" in str(info.value)


def test_member_count_is_bounded(share_package, tmp_path):
    def mutate(zin, zout):
        for item in zin.infolist():
            zout.writestr(item, zin.read(item.filename))
        for i in range(packages.MAX_PACKAGE_MEMBERS + 1):
            zout.writestr(f"run/junk/{i}", b"")

    many = _rebuild(share_package, tmp_path / "many.zip", mutate)
    with pytest.raises(packages.PackageError) as info:
        packages.inspect_package(many)
    assert info.value.code == "too_many_members"


def test_traversal_members_are_refused_before_any_write(share_package, tmp_path, ws):
    def mutate(zin, zout):
        for item in zin.infolist():
            zout.writestr(item, zin.read(item.filename))
        zout.writestr("run/../../escaped.txt", b"x")

    bad = _rebuild(share_package, tmp_path / "traversal.zip", mutate)
    with pytest.raises(packages.PackageError) as info:
        packages.import_package(ws, bad)
    assert info.value.code == "unsafe_path"
    assert not (ws.root / "escaped.txt").exists() and not (ws.root.parent / "escaped.txt").exists()


# --- checkpoints ---------------------------------------------------------------------------------

class _Payload:
    """Pickles to a call of builtins.open(marker, "w"): loading it unrestricted would create the marker."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (str(self.marker), "w"))


def test_malicious_checkpoints_are_refused_not_executed(ws, pa_run, tmp_path):
    import torch

    marker = tmp_path / "executed.marker"
    evil = tmp_path / "evil.pt"
    torch.save({"weight": _Payload(marker)}, evil)
    with pytest.raises(CheckpointRefused) as info:
        load_checkpoint(evil)
    assert "was not executed" in str(info.value) and not marker.exists()

    # the same file in place of a real checkpoint: the evaluation refuses it with a workspace error
    from opendpd.services.evaluation import evaluate_run
    from opendpd.services.workspace import sha256_file
    run_dir = ws.run_dir(pa_run["run_id"])
    manifest = json.loads((run_dir / "artifacts.json").read_text())
    ckpt = next(a for a in manifest["artifacts"] if a["kind"] == "checkpoint")
    target = run_dir / ckpt["file"]["path"]
    original = target.read_bytes()
    try:
        target.write_bytes(evil.read_bytes())
        ckpt["file"]["sha256"] = sha256_file(target)
        (run_dir / "artifacts.json").write_text(json.dumps(manifest))
        with pytest.raises(CheckpointRefused):
            evaluate_run(ws, pa_run["run_id"], "legacy-opendpd-v1")
        assert not marker.exists()
    finally:
        target.write_bytes(original)
        ckpt["file"]["sha256"] = sha256_file(target)
        (run_dir / "artifacts.json").write_text(json.dumps(manifest))


def test_legacy_state_dict_checkpoints_still_load_under_the_restriction(tmp_path):
    import torch

    path = tmp_path / "legacy.pt"
    torch.save({"rnn.weight_ih_l0": torch.zeros(3, 2), "fc.bias": torch.ones(1)}, path)
    state = load_checkpoint(path)
    assert set(state) == {"rnn.weight_ih_l0", "fc.bias"}


# --- device refusals -------------------------------------------------------------------------------

def test_accelerator_refusals_are_classified_with_a_hint():
    try:
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 15.99 GiB total capacity)")
    except RuntimeError as err:
        error = classify_failure(err, stage="train")
    assert error.code == "device_busy_or_out_of_memory" and error.stage == "train"
    assert "another process may hold it" in (error.hint or "") and "automatically" in error.hint
    try:
        raise ValueError("shapes (3, 2) and (4,) do not match")
    except ValueError as err:
        error = classify_failure(err, stage="apply")
    assert error.code == "worker_exception" and error.hint is None and "ValueError" in error.message


def test_no_unrestricted_pickle_loading_in_the_tree():
    root = Path(__file__).resolve().parents[2]
    offenders = []
    for folder in ("opendpd", "steps", "modules", "utils"):
        for path in (root / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "torch.load(" in text and "weights_only=True" not in text:
                offenders.append(str(path.relative_to(root)))
            if "allow_pickle=True" in text or "pickle.load" in text:
                offenders.append(str(path.relative_to(root)))
    assert offenders == [], offenders


def test_the_service_and_the_page_never_call_out():
    """No telemetry, no CDN, no update checks: the only network client in the tree probes 127.0.0.1."""
    root = Path(__file__).resolve().parents[2]
    offenders = []
    for path in (root / "opendpd").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("urlopen(", "requests.get(", "requests.post(", "httpx.", "http.client.", "create_connection("):
            if needle in text and path.name != "launcher.py":
                offenders.append(f"{path.relative_to(root)}: {needle}")
    launcher = (root / "opendpd" / "studio" / "launcher.py").read_text(encoding="utf-8")
    assert "loopback only" in launcher and 'HOST = "127.0.0.1"' in launcher
    for path in FRONTEND_SRC.rglob("*.ts*"):
        text = path.read_text(encoding="utf-8")
        if path.name.endswith(".test.tsx") or path.name.endswith(".test.ts"):
            continue
        for needle in (r"\bfetch\(", "XMLHttpRequest", "sendBeacon", "new WebSocket", "new EventSource"):
            # \bfetch( is the browser call; react-query's refetch( is not a network client
            if re.search(needle, text) and path.name not in ("client.ts", "events.ts"):
                offenders.append(f"frontend/src/{path.relative_to(FRONTEND_SRC)}: {needle}")
    client = (FRONTEND_SRC / "api" / "client.ts").read_text(encoding="utf-8")
    assert "fetch(`${API}${path}`" in client and "const API = '/api/v1'" in client
    assert offenders == [], offenders
