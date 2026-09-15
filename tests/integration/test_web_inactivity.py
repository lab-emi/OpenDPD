"""Inactivity is shared by creation network; background reads cannot renew it."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
import secrets
import threading

import pytest
from fastapi.testclient import TestClient

from opendpd.web.app import create_web_app
from opendpd.web.policy import WebConfig
from opendpd.web.runtime import DAY

pytestmark = pytest.mark.integration
TUNNEL = 'a' * 48 + '.internal'


@pytest.fixture
def studio(tmp_path):
    clock = [DAY * 20000 + 3600]
    app = create_web_app(WebConfig(tmp_path / 'web', 'https://opendpd.com', 'api.opendpd.com', TUNNEL), now=lambda: clock[0])
    with TestClient(app, base_url='http://' + TUNNEL, client=('127.0.0.1', 1234)) as client:
        client.headers.update({'Origin': 'https://opendpd.com', 'X-Forwarded-Proto': 'https', 'CF-Connecting-IP': '203.0.113.10'})
        yield client, app.state.manager, clock


def enter(client, ip='203.0.113.10'):
    response = client.post('/api/v1/web/sessions', json={}, headers={'CF-Connecting-IP': ip})
    assert response.status_code == 201, response.text
    return {'Authorization': 'Bearer ' + response.json()['access_token'], 'CF-Connecting-IP': ip}


def read(client, auth):
    return client.get('/api/v1/session', headers=auth)


def test_two_hours_of_polling_does_not_preserve_an_ip_group_or_revive_expired_data(studio):
    client, manager, clock = studio
    a, b = enter(client), enter(client)
    old = list(manager.tenants.values())
    for tenant in old:
        (tenant.root / 'private-result').write_text('private')
    clock[0] += 3600
    other = enter(client, '203.0.113.11')
    clock[0] += 3599
    assert client.get('/api/v1/system/status', headers=a).status_code == 200
    assert client.get('/api/v1/runs', headers=b).status_code == 200
    clock[0] += 1
    assert read(client, a).status_code == 401
    assert read(client, b).status_code == 401
    assert client.post('/api/v1/web/activity', json={}, headers=a).status_code == 401
    assert read(client, other).status_code == 200
    client.portal.call(manager.sweep)
    assert all(not tenant.root.exists() for tenant in old)
    assert len(manager.tenants) == len(manager.ip_activity) == 1
    assert read(client, enter(client)).status_code == 200


def test_interaction_renews_all_workspaces_on_same_ip_only_and_exposes_no_identity(studio):
    client, manager, clock = studio
    a, b, other = enter(client), enter(client), enter(client, '203.0.113.11')
    first = read(client, a).json()
    clock[0] += 3600
    renewed = client.post('/api/v1/web/activity', json={}, headers=b)
    assert renewed.status_code == 200
    data = renewed.json()
    assert data['inactivity_seconds'] == 7200
    assert datetime.fromisoformat(data['idle_expires_at']).timestamp() == clock[0] + 7200
    assert data['expires_at'] == first['expires_at']
    assert read(client, a).json()['idle_expires_at'] == data['idle_expires_at']
    assert 'access_token' not in data and '203.0.113' not in renewed.text
    assert str(manager.config.root) not in renewed.text
    clock[0] += 3601
    assert read(client, other).status_code == 401
    assert read(client, a).status_code == read(client, b).status_code == 200
    clock[0] += 3599
    assert read(client, a).status_code == read(client, b).status_code == 401


def test_ipv6_privacy_addresses_share_one_inactivity_group(studio):
    client, _, clock = studio
    a, b = enter(client, '2001:db8:1::1'), enter(client, '2001:db8:1::2')
    other = enter(client, '2001:db8:2::1')
    clock[0] += 3600
    client.post('/api/v1/web/activity', json={}, headers=a)
    clock[0] += 3601
    assert read(client, b).status_code == 200
    assert read(client, other).status_code == 401


def test_activity_requires_the_existing_origin_and_workspace_capability(studio):
    client, _, _ = studio
    assert client.post('/api/v1/web/activity', json={}).status_code == 401
    assert client.post('/api/v1/web/activity', json={}, headers={'Authorization': 'Queue ' + secrets.token_urlsafe(32)}).status_code == 401
    auth = enter(client)
    assert client.post('/api/v1/web/activity', json={}, headers={**auth, 'Origin': 'https://evil.example'}).status_code == 403
    assert client.post('/api/v1/web/activity', json={'expires_at': '2099-01-01'}, headers=auth).status_code == 422
    assert client.get('/api/v1/web/activity', headers=auth).status_code == 403
    preflight = client.options('/api/v1/web/activity', headers={'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'authorization, content-type'})
    assert preflight.status_code == 204


def test_expiry_blocks_new_work_while_an_existing_request_finishes(studio):
    client, manager, clock = studio
    auth = enter(client)
    tenant = next(iter(manager.tenants.values()))
    tenant.inflight = 1
    clock[0] += 7200
    client.portal.call(manager.sweep)
    assert tenant.closing and not tenant.app.state.supervisor._accepting
    assert tenant.root.exists()
    assert read(client, auth).status_code == 401
    tenant.inflight = 0
    client.portal.call(manager.sweep)
    assert not tenant.root.exists() and not manager.tenants


def test_waiting_visitor_enters_when_an_inactive_ip_is_cleaned(studio):
    client, manager, clock = studio
    manager.config = replace(manager.config, max_sessions=1)
    auth = enter(client)
    clock[0] += 7190
    waiting = {'Authorization': 'Queue ' + secrets.token_urlsafe(32), 'CF-Connecting-IP': '203.0.113.11'}
    assert client.post('/api/v1/web/sessions', json={}, headers=waiting).status_code == 202
    clock[0] += 10
    assert client.post('/api/v1/web/sessions', json={}, headers=waiting).status_code == 201
    assert read(client, auth).status_code == 401
    assert len(manager.tenants) == 1


def test_idle_status_performs_no_sqlite_counts(studio, monkeypatch):
    client, manager, _ = studio
    for _ in range(32):
        auth = enter(client)
    for tenant in manager.tenants.values():
        monkeypatch.setattr(tenant.app.state.store, 'count_runs', lambda *a, **kw: pytest.fail('idle workspace queried'))
    manager.status_updated = 0
    data = client.get('/api/v1/system/status', headers=auth).json()
    assert data['workspaces'] == 32
    assert data['running_jobs'] == data['queued_jobs'] == 0


@pytest.mark.parametrize('stage', ['runtime', 'files'])
def test_retirement_does_not_block_another_visitor(studio, monkeypatch, stage):
    from opendpd.web import runtime
    client, manager, clock = studio
    enter(client)
    retiring = next(iter(manager.tenants.values()))
    other = enter(client, '203.0.113.11')
    clock[0] += 7199
    client.post('/api/v1/web/activity', json={}, headers=other)
    clock[0] += 1
    started, release = threading.Event(), threading.Event()
    if stage == 'runtime':
        original = retiring.app.state.supervisor.stop
        def stop(*args, **kwargs):
            started.set()
            assert release.wait(5)
            return original(*args, **kwargs)
        monkeypatch.setattr(retiring.app.state.supervisor, 'stop', stop)
    else:
        original = runtime.shutil.rmtree
        def remove(path, *args, **kwargs):
            if path == retiring.root:
                started.set()
                assert release.wait(5)
            return original(path, *args, **kwargs)
        monkeypatch.setattr(runtime.shutil, 'rmtree', remove)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(client.portal.call, manager.sweep)
        try:
            assert started.wait(3)
            healthy = pool.submit(client.get, '/api/v1/runs', headers=other)
            assert healthy.result(timeout=2).status_code == 200
        finally:
            release.set()
        pending.result(timeout=3)
    assert not retiring.root.exists()
