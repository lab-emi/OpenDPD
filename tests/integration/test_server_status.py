"""Public telemetry, admission and user isolation through the real ASGI boundary."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import threading
import time

import pytest
from fastapi.testclient import TestClient

from opendpd.web.app import create_web_app
from opendpd.web.policy import WebConfig
from opendpd.web.runtime import DAY

pytestmark = pytest.mark.integration
TOKEN = 'private-status-token-' * 4
TUNNEL = 'a' * 48 + '.internal'


@pytest.fixture
def public(tmp_path):
    now = [DAY * 20000 + 3600]
    config = WebConfig(tmp_path / 'web', 'https://opendpd.com', 'api.opendpd.com', TUNNEL,
                       gpu_token=TOKEN, max_expensive_requests=1)
    app = create_web_app(config, now=lambda: now[0])
    with TestClient(app, base_url='http://127.0.0.1', client=('127.0.0.1', 10)) as client:
        headers = {'Host': TUNNEL, 'Origin': 'https://opendpd.com', 'X-Forwarded-Proto': 'https',
                   'CF-Connecting-IP': '203.0.113.10'}
        def session():
            result = client.post('/api/v1/web/sessions', json={}, headers=headers)
            assert result.status_code == 201, result.text
            return {**headers, 'Authorization': 'Bearer ' + result.json()['access_token']}
        yield client, app.state.manager, now, headers, session


def test_public_status_is_authenticated_aggregate_and_expires(public):
    client, manager, now, headers, session = public
    assert client.get('/api/v1/system/status', headers=headers).status_code == 401
    a, _ = session(), session()
    response = client.get('/api/v1/system/status', headers=a)
    assert response.status_code == 200, response.text
    data = response.json()
    assert (data['active_sessions'], data['workspaces']) == (2, 2)
    assert data['running_jobs'] == data['queued_jobs'] == 0
    assert data['compute']['stale'] and data['compute']['load']['cpu_percent'] is None
    assert response.headers['cache-control'] == 'no-store'
    assert 'set-cookie' not in response.headers
    assert '203.0.113.10' not in response.text and str(manager.config.root) not in response.text
    assert all(t.identifier not in response.text and t.token_hash not in response.text for t in manager.tenants.values())
    now[0] += 301
    data = client.get('/api/v1/system/status', headers=a).json()
    assert (data['active_sessions'], data['workspaces']) == (1, 2)
    now[0] = manager.cutoff
    assert client.get('/api/v1/system/status', headers=a).status_code == 401


def test_telemetry_requires_private_origin_and_validated_payload(public):
    client, manager, _, headers, session = public
    auth = session()
    payload = {'sampled_at': datetime.now(timezone.utc).isoformat(), 'cpu_percent': 63.5,
               'gpu': {'utilization_percent': 48, 'memory_used_bytes': 2**30, 'memory_total_bytes': 8 * 2**30}}
    private = {'X-OpenDPD-GPU': TOKEN}
    assert client.post('/_gpu/resources', json=payload).status_code == 403
    assert client.post('/_gpu/resources', json=payload, headers={**private, **headers}).status_code == 403
    for bad in ({**payload, 'processes': ['secret']}, {**payload, 'cpu_percent': 101},
                {**payload, 'sampled_at': '2026-01-01T00:00:00'}):
        assert client.post('/_gpu/resources', json=bad, headers=private).status_code == 409
    assert client.post('/_gpu/resources', json=payload, headers=private).status_code == 200
    result = client.get('/api/v1/system/status', headers=auth).json()['compute']
    assert not result['stale'] and result['load']['cpu_percent'] == 63.5
    manager.gpu.telemetry_seen = time.monotonic() - 25
    assert client.get('/api/v1/system/status', headers=auth).json()['compute']['stale']
    assert client.post('/_gpu/resources', content=b'x' * 4097, headers=private).status_code == 413


def test_concurrent_workspace_writes_are_refused_without_blocking_reads_or_cancel(public, monkeypatch):
    client, manager, _, _, session = public
    auth = session()
    workspace = next(iter(manager.tenants.values())).app.state.ws
    original = workspace.save_settings
    entered, release = threading.Event(), threading.Event()
    def slow(settings):
        entered.set()
        assert release.wait(5)
        return original(settings)
    monkeypatch.setattr(workspace, 'save_settings', slow)
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(client.put, '/api/v1/settings', json={'language': 'en'}, headers=auth)
        try:
            assert entered.wait(5)
            duplicate = client.put('/api/v1/settings', json={'language': 'zh'}, headers=auth)
            assert duplicate.status_code == 429 and duplicate.json()['error']['code'] == 'workspace_busy'
            assert client.get('/api/v1/system/status', headers=auth).status_code == 200
            assert client.post('/api/v1/runs/unknown/cancel', json={}, headers=auth).status_code == 404
        finally:
            release.set()
        assert pending.result().status_code == 200
    assert client.put('/api/v1/settings', json={'language': 'zh'}, headers=auth).status_code == 200
    assert manager.inflight == 0


def test_expensive_work_is_bounded_across_tenants(public, monkeypatch):
    from opendpd.server import routes
    client, manager, _, _, session = public
    a, b = session(), session()
    entered, release = threading.Event(), threading.Event()
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        raise routes._error(404, 'not_found', 'test analysis finished')
    monkeypatch.setattr(routes, 'analyze_dataset', slow)
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(client.get, '/api/v1/datasets/example/analysis', headers=a)
        try:
            assert entered.wait(5)
            rejected = client.get('/api/v1/datasets/example/analysis', headers=b)
            assert rejected.status_code == 429 and rejected.json()['error']['code'] == 'compute_busy'
            assert client.get('/api/v1/system/status', headers=b).status_code == 200
        finally:
            release.set()
        assert pending.result().status_code == 404
    assert manager.inflight == manager.expensive_requests == 0


def test_public_local_only_routes_and_outbound_activity_are_closed(public, monkeypatch):
    from opendpd.services import about
    client, _, _, _, session = public
    auth = session()
    monkeypatch.setattr(about, 'project_info', lambda: pytest.fail('public request made an outbound query'))
    assert client.get('/api/v1/datasets/import-roots', headers=auth).status_code == 403
    for path in ('/system/about', '/system/about?activity=true'):
        assert client.get('/api/v1' + path, headers=auth).status_code == 200


def test_storage_reservation_refuses_writes_before_disk_is_full(public, monkeypatch):
    from types import SimpleNamespace
    from opendpd.web import runtime
    client, manager, _, _, session = public
    auth = session()
    monkeypatch.setattr(runtime.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100 * 2**20))
    result = client.post('/api/v1/datasets/import-builtin', json={'name': 'MyCustomPA'}, headers=auth)
    assert result.status_code == 507
    assert client.get('/api/v1/datasets', headers=auth).json() == []
    assert client.get('/api/v1/system/status', headers=auth).status_code == 200
    assert manager.inflight == manager.storage_reserved == 0


def test_job_count_failure_is_unknown_load(public, monkeypatch):
    import sqlite3
    from opendpd.services import server_status
    client, _, _, _, session = public
    auth = session()
    def unavailable(_):
        raise sqlite3.OperationalError('database is unavailable')
    monkeypatch.setattr(server_status, 'job_counts', unavailable)
    response = client.get('/api/v1/system/status', headers=auth)
    assert response.status_code == 200
    assert response.json()['running_jobs'] is None
    assert response.json()['queued_jobs'] is None


def test_gpu_cleanup_failure_does_not_stop_tenant_expiry(public, monkeypatch):
    client, manager, now, headers, session = public
    session()
    tenant = next(iter(manager.tenants.values()))
    def failed_finalization():
        raise OSError('GPU finalization unavailable')
    monkeypatch.setattr(manager.gpu, 'sweep', failed_finalization)
    response = client.post('/api/v1/web/sessions', json={}, headers=headers)
    assert response.status_code == 503
    assert response.json()['error']['code'] == 'cleanup_unavailable'
    now[0] = tenant.expires_at
    client.portal.call(manager.sweep)
    assert not tenant.root.exists() and not manager.tenants
    assert client.get('/healthz').status_code == 503
