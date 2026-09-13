"""Public boundary tests: real ASGI routing, independent runtimes and disk cleanup."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from opendpd.services.recipes import instantiate
from opendpd.web.app import create_web_app
from opendpd.web.policy import WebConfig
from opendpd.web.runtime import DAY, prepare_root

pytestmark = pytest.mark.integration
ORIGIN = "https://opendpd.com"
TUNNEL_HOST = "a" * 48 + ".internal"


@pytest.fixture
def public(tmp_path):
    now = [DAY * 20000 + 3600]
    config = WebConfig(root=tmp_path / "web", origin=ORIGIN, api_host="api.opendpd.com", tunnel_host=TUNNEL_HOST)
    app = create_web_app(config, now=lambda: now[0])
    with TestClient(app, base_url="http://" + TUNNEL_HOST, client=("127.0.0.1", 10000)) as client:
        client.headers.update({"Origin": ORIGIN, "X-Forwarded-Proto": "https", "CF-Connecting-IP": "203.0.113.10"})
        yield client, app.state.manager, now


def new_session(client):
    response = client.post("/api/v1/web/sessions", json={})
    assert response.status_code == 201, response.text
    return {"Authorization": "Bearer " + response.json()["access_token"]}


def test_no_auth_and_no_origin_bypass(public):
    client, manager, _ = public
    assert client.get("/api/v1/runs").status_code == 401
    auth = new_session(client)
    for bad in [{"Origin": "https://evil.example"}, {"Origin": "null"}, {"Host": "api.opendpd.com"},
                {"X-Forwarded-Proto": "http"}, {"CF-Connecting-IP": "garbage"}]:
        assert client.get("/api/v1/runs", headers={**auth, **bad}).status_code == 403
    assert len(manager.tenants) == 1
    result = client.get("/api/v1/runs", headers=auth)
    assert result.status_code == 200
    assert result.headers["access-control-allow-origin"] == ORIGIN
    assert result.headers["cache-control"] == "no-store"
    assert "set-cookie" not in result.headers
    assert "access-control-allow-credentials" not in result.headers
    assert client.get("/api/v1/runs?access_token=" + auth["Authorization"][7:]).status_code == 401


def test_preflight_and_body_limits(public):
    client, _, _ = public
    headers = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"}
    response = client.options("/api/v1/runs", headers=headers)
    assert response.status_code == 204
    assert response.content == b""
    assert client.options("/api/v1/datasets/upload", headers=headers).status_code == 204
    assert client.options("/api/v1/runs", headers={**headers, "Access-Control-Request-Headers": "x-forwarded-for"}).status_code == 403
    auth = new_session(client)
    assert client.post("/api/v1/runs", content="x" * 65537, headers={**auth, "Content-Type": "application/json"}).status_code == 413
    assert client.post("/api/v1/runs", content=iter([b"x" * 32000, b"x" * 34000]), headers={**auth, "Content-Type": "application/json"}).status_code == 413
    assert client.post("/api/v1/runs", content=b"x", headers=auth).status_code == 415


def test_same_ip_sessions_have_independent_files_and_auth(public):
    client, manager, _ = public
    a, b = new_session(client), new_session(client)
    response = client.post("/api/v1/datasets/import-builtin", json={"name": "MyCustomPA"}, headers=a)
    assert response.status_code == 201, response.text
    dataset_id = response.json()["dataset_id"]
    assert len(client.get("/api/v1/datasets", headers=a).json()) == 1
    assert client.get("/api/v1/datasets", headers=b).json() == []
    assert client.get(f"/api/v1/datasets/{dataset_id}", headers=b).status_code != 200
    roots = [t.root for t in manager.tenants.values()]
    assert roots[0] != roots[1]
    assert all("203.0.113" not in str(p) and (p.stat().st_mode & 0o777) == 0o700 for p in roots)
    client.put("/api/v1/settings", json={"language": "zh"}, headers=a)
    assert client.get("/api/v1/settings", headers=b).json()["language"] != "zh"
    # Even a known artifact ID cannot select another session's run or file.
    first = next(iter(manager.tenants.values()))
    path = first.app.state.ws.exports_dir / "private.zip"
    path.write_bytes(b"secret result")
    assert client.get("/api/v1/exports/private", headers=a).content == b"secret result"
    assert client.get("/api/v1/exports/private", headers=b).status_code == 404


def test_public_surface_blocks_upload_code_paths_and_huge_configs(public):
    client, _, _ = public
    auth = new_session(client)
    for path in ["/datasets/inspect", "/datasets/import", "/imports",
                 "/session/bootstrap", "/deploy/exports", "/datasets/a/manifest"]:
        assert client.post("/api/v1" + path, json={}, headers=auth).status_code == 403
    for path in ["/datasets/a/analysis?version=../../other", "/results/a?profile=../../other"]:
        assert client.get("/api/v1" + path, headers=auth).status_code == 422
    assert client.post("/api/v1/datasets/import-builtin", json={"name": "MyCustomPA", "dataset_id": "../../escape"}, headers=auth).status_code == 422
    config = json.loads(instantiate("pa-gru-smoke-v1", "mycustompa").model_dump_json())
    config["model"]["parameters"]["hidden_size"] = 1000000
    assert client.post("/api/v1/runs", json={"config": config}, headers=auth).status_code == 422
    assert client.post("/api/v1/experiments/validate", json={"config": config}, headers=auth).status_code == 422


@pytest.mark.parametrize('filename,data', [
    ('capture.py', b'1,2\n'), ('capture.csv.exe', b'1,2\n'), ('../capture.csv', b'1,2\n'),
    ('capture.csv', b'PK\x03\x04\x00\x00'), ('capture.csv', b'I_in,Q_in,I_out,Q_out\n1,2,3,=HYPERLINK("x")\n'),
    ('capture.csv', b'import os\nos.system("echo malicious")\n'), ('capture.csv', b'<script>alert(1)</script>\n'),
    ('capture.csv', b'input,output\n1,NaN\n'), ('capture.csv', b'input,output\n1,2,3\n'),
    ('capture.csv', b'input,output\n"1,2\n'), ('capture.csv', b'input,output\n\xff,2\n'),
    ('capture.csv', b'input,output\n1\x00,2\n'), ('capture.csv', b'input,output\n'+b'1'*4097+b',2\n'),
])
def test_rejected_csv_is_deleted_without_exposing_preview(public, filename, data):
    client, manager, _ = public
    auth = new_session(client)
    response = client.post('/api/v1/datasets/upload', params={'filename': filename}, content=data,
                           headers={**auth, 'Content-Type': 'text/csv'})
    assert response.status_code == 422, response.text
    assert response.json()['error']['code'] == 'csv_rejected'
    root = next(iter(manager.tenants.values())).app.state.ws.imports_dir
    assert not [p for p in root.rglob('*') if p.is_file()]
    assert 'preview' not in response.json()


@pytest.mark.parametrize('data', [b'I_in,Q_in,I_out,Q_out\n'+b'.1,.2,.3,.4\n'*4096,
                                b'input,output\n'+b'.1+.2j,.3-.4i\n'*4096])
def test_public_csv_validation_import_and_tenant_isolation(public, data):
    client, manager, _ = public
    auth, other = new_session(client), new_session(client)
    response = client.post('/api/v1/datasets/upload?filename=capture.csv', content=data,
                           headers={**auth, 'Content-Type': 'text/csv'})
    assert response.status_code == 201, response.text
    result = response.json()
    assert result['validation']['status'] == 'passed' and result['validation']['n_samples'] == 4096
    source = {k: result[k] for k in ('root_id', 'path')}
    body = {'source': source}
    assert client.post('/api/v1/datasets/csv/preview', json=body, headers=other).status_code == 422
    assert client.post('/api/v1/datasets/csv/preview', json={'source': {'root_id': 'imports', 'path': '../lease.json'}}, headers=auth).status_code == 422
    preview = client.post('/api/v1/datasets/csv/preview', json=body, headers=auth)
    assert preview.status_code == 200 and preview.json()['valid'], preview.text
    response = client.post('/api/v1/datasets/csv', json={**body, 'dataset_id': 'my-capture', 'display_name': 'My capture',
                           'origin': 'synthetic', 'expected_sha256': result['validation']['sha256']}, headers=auth)
    assert response.status_code == 201, response.text
    assert client.get('/api/v1/datasets', headers=other).json() == []
    assert client.get('/api/v1/system/capabilities', headers=auth).json()['custom_dataset_imports'] is True
    root = next(iter(manager.tenants.values())).app.state.ws.imports_dir
    assert not list((root / 'quarantine').iterdir())


def test_public_upload_size_and_unvalidated_paths_fail_closed(public):
    client, manager, _ = public
    auth = new_session(client)
    assert client.post('/api/v1/datasets/upload?filename=a.csv', content=b'1,2\n', headers={**auth, 'Content-Type': 'text/csv', 'Content-Length': str(26 * 1024 * 1024)}).status_code == 413
    assert client.post('/api/v1/datasets/upload?filename=a.csv', json={}, headers=auth).status_code == 415
    for source in ({'root_id': 'imports', 'path': None}, {'root_id': 'imports', 'path': 'uploads/'+'0'*32+'.csv'}):
        assert client.post('/api/v1/datasets/csv/preview', json={'source': source}, headers=auth).status_code == 422
    assert not [p for p in next(iter(manager.tenants.values())).app.state.ws.imports_dir.rglob('*') if p.is_file()]


@pytest.mark.parametrize('signal', [{'nperseg': 2**40}, {'n_sub_ch': 2**40}, {'sample_rate_hz': 'Infinity'},
                                    {'waveform': {}}, {'modulation': 'x' * 129}, None])
def test_public_csv_metadata_cannot_allocate_unbounded_compute(public, signal):
    client, _, _ = public
    auth = new_session(client)
    for path in ('/datasets/csv', '/datasets/csv/preview'):
        response = client.post('/api/v1' + path, json={'signal': signal}, headers=auth)
        assert response.status_code == 422
        assert response.json()['error']['code'] in {'compute_limit', 'feature_unavailable', 'invalid_request'}


def test_fixed_expiry_is_not_extended_and_purges_all_files(public):
    client, manager, now = public
    auth = new_session(client)
    tenant = next(iter(manager.tenants.values()))
    expiry = tenant.expires_at
    (tenant.root / "future-upload").write_bytes(b"future upload")
    (tenant.app.state.ws.cache_dir / "analysis").write_bytes(b"analysis")
    assert expiry - now[0] < DAY
    now[0] += 3600
    assert client.get("/api/v1/session", headers=auth).status_code == 200
    assert tenant.expires_at == expiry
    now[0] = expiry
    assert client.get("/api/v1/runs", headers=auth).status_code == 401
    client.portal.call(manager.sweep)
    assert not tenant.root.exists()
    assert not manager.tenants
    assert client.post("/api/v1/web/sessions", json={}).status_code == 503


def test_new_sessions_cannot_bypass_ip_quota_with_forged_forwarding(public):
    client, manager, _ = public
    for i in range(manager.config.sessions_per_ip):
        response = client.post("/api/v1/web/sessions", json={}, headers={"X-Forwarded-For": f"203.0.113.{i}"})
        assert response.status_code == 201
    assert client.post("/api/v1/web/sessions", json={}).status_code == 429
    assert manager.ip_key("2001:db8::1") == manager.ip_key("2001:db8::abcd")


def test_startup_cleans_orphans_and_refuses_shared_root(tmp_path):
    config = WebConfig(tmp_path / "web", ORIGIN, "api.opendpd.com", TUNNEL_HOST)
    first = create_web_app(config)
    with TestClient(first) as _:
        orphan = config.root / "sessions" / "orphan"
        orphan.mkdir()
        (orphan / "data").write_bytes(b"old data")
        with pytest.raises(RuntimeError, match="another public service"):
            prepare_root(config.root)
    with TestClient(create_web_app(config)):
        assert not orphan.exists()
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    (ordinary / "keep").write_text("user file")
    with pytest.raises(ValueError, match="not empty"):
        prepare_root(ordinary)
    assert (ordinary / "keep").read_text() == "user file"


def test_real_training_is_private_and_global_dispatch_is_serial(public):
    client, manager, _ = public
    a, b = new_session(client), new_session(client)
    for auth in [a, b]:
        assert client.post("/api/v1/datasets/import-builtin", json={"name": "MyCustomPA"}, headers=auth).status_code == 201
    cfg = json.loads(instantiate("pa-gru-smoke-v1", "mycustompa").model_dump_json())
    cfg["training"]["epochs"] = 1
    cfg["execution"]["num_threads"] = 1
    # All supervisors share the same semaphore, independent of selected device.
    assert manager.slots.acquire(blocking=False)
    response = client.post("/api/v1/runs", json={"config": cfg, "idempotency_key": "once"}, headers=a)
    assert response.status_code == 201, response.text
    rid = response.json()["run_id"]
    assert client.post("/api/v1/runs", json={"config": cfg, "idempotency_key": "once"}, headers=a).status_code == 200
    assert manager.total_runs == 1
    second = client.post("/api/v1/runs", json={"config": cfg}, headers=b)
    assert second.status_code == 201
    assert all(not t.app.state.supervisor.active_run_ids() for t in manager.tenants.values())
    assert client.get(f"/api/v1/runs/{rid}", headers=b).status_code == 404
    assert client.get(f"/api/v1/runs/{rid}/events/list", headers=b).status_code == 404
    assert client.post(f"/api/v1/runs/{rid}/cancel", json={}, headers=b).status_code == 404
    manager.slots.release()
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        assert sum(len(t.app.state.supervisor.active_run_ids()) for t in manager.tenants.values()) <= 1
        record = client.get(f"/api/v1/runs/{rid}", headers=a).json()
        if record.get("status") in {"succeeded", "failed", "cancelled"}:
            assert record["status"] == "succeeded", record
            break
        time.sleep(0.5)
    else:
        pytest.fail("training did not finish")
    assert client.get(f"/api/v1/results/{rid}", headers=a).status_code == 200
    assert client.get(f"/api/v1/results/{rid}", headers=b).status_code == 404
