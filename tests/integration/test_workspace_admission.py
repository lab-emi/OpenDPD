"""Real waiting-room requests: isolation, FIFO, bounded leases and fixed cleanup."""
from dataclasses import replace
import secrets
import threading

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from opendpd.web.app import create_web_app
from opendpd.web.policy import WebConfig
from opendpd.web.runtime import DAY

pytestmark = pytest.mark.integration
TUNNEL = 'a' * 48 + '.internal'


@pytest.fixture
def room(tmp_path):
    clock = [DAY * 20000 + 3600]
    config = WebConfig(tmp_path / 'room', 'https://opendpd.com', 'api.opendpd.com', TUNNEL, max_sessions=1)
    app = create_web_app(config, now=lambda: clock[0])
    with TestClient(app, base_url='http://' + TUNNEL, client=('127.0.0.1', 10000)) as client:
        client.headers.update({'Origin': 'https://opendpd.com', 'X-Forwarded-Proto': 'https', 'CF-Connecting-IP': '203.0.113.10'})
        yield client, app.state.manager, clock


def ticket():
    return {'Authorization': 'Queue ' + secrets.token_urlsafe(32)}


def enter(client, headers):
    return client.post('/api/v1/web/sessions', headers=headers, json={})


def bearer(response):
    assert response.status_code == 201, response.text
    return {'Authorization': 'Bearer ' + response.json()['access_token']}


def test_fifo_queue_is_lightweight_and_replays_admission_after_lost_response(room):
    client, manager, _ = room
    first = bearer(enter(client, ticket()))
    a, b, late = ticket(), ticket(), ticket()
    assert enter(client, a).json()['queue_position'] == 1
    second = enter(client, b)
    assert second.status_code == 202 and second.headers['retry-after'] == '5'
    assert second.json()['queue_position'] == 2
    assert len(manager.tenants) == 1
    assert len(list((manager.config.root / 'sessions').iterdir())) == 1
    status = client.get('/api/v1/system/status', headers=first).json()
    assert status['waiting_sessions'] == 2
    assert client.post('/api/v1/web/sessions/end', headers=first, json={}).status_code == 204
    # Polling faster or arriving later cannot bypass the first ticket.
    assert enter(client, b).json()['queue_position'] == 2
    assert enter(client, late).json()['queue_position'] == 3
    admitted = enter(client, a)
    auth = bearer(admitted)
    replay = enter(client, a)
    assert replay.json()['access_token'] == admitted.json()['access_token']
    assert len(manager.tenants) == 1
    assert client.get('/api/v1/session', headers=first).status_code == 401
    assert client.get('/api/v1/session', headers=auth).status_code == 200
    # Once acknowledged, no plaintext session capability remains in memory.
    assert all(entry.access_token is None for entry in manager.admissions.values())
    assert enter(client, a).status_code == 410


def test_ticket_is_not_a_workspace_or_gpu_credential_and_cancel_is_scoped(room):
    client, manager, _ = room
    auth = bearer(enter(client, ticket()))
    waiting = ticket()
    assert enter(client, waiting).status_code == 202
    assert client.get('/api/v1/runs', headers=waiting).status_code == 401
    assert client.post('/api/v1/web/sessions/end', headers=waiting, json={}).status_code == 401
    assert client.post('/_gpu/poll', headers=waiting, json={'name': 'test'}).status_code == 403
    assert client.post('/api/v1/web/queue/cancel', headers=ticket(), json={}).status_code == 204
    assert manager.waiting_count() == 1
    assert client.post('/api/v1/web/queue/cancel', headers=waiting, json={}).status_code == 204
    assert manager.waiting_count() == 0
    assert client.get('/api/v1/session', headers=auth).status_code == 200


def test_cancel_racing_with_admission_releases_only_an_unclaimed_workspace(room):
    client, manager, _ = room
    key = ticket()
    auth = bearer(enter(client, key))
    assert client.post('/api/v1/web/queue/cancel', headers=key, json={}).status_code == 204
    assert not manager.tenants
    assert client.get('/api/v1/session', headers=auth).status_code == 401
    key = ticket()
    auth = bearer(enter(client, key))
    assert client.get('/api/v1/session', headers=auth).status_code == 200
    assert client.post('/api/v1/web/queue/cancel', headers=key, json={}).status_code == 204
    assert client.get('/api/v1/session', headers=auth).status_code == 200


def test_abandoned_tickets_expire_and_do_not_stall_active_visitors(room):
    client, manager, clock = room
    auth = bearer(enter(client, ticket()))
    abandoned, active = ticket(), ticket()
    enter(client, abandoned)
    clock[0] += 100
    assert enter(client, active).json()['queue_position'] == 2
    clock[0] += 81
    assert enter(client, active).json()['queue_position'] == 1
    assert manager.waiting_count() == 1
    client.post('/api/v1/web/sessions/end', headers=auth, json={})
    assert enter(client, active).status_code == 201


def test_twelve_hour_cleanup_accepts_waiters_and_reopens_after_noon(room):
    client, manager, clock = room
    auth = bearer(enter(client, ticket()))
    tenant = next(iter(manager.tenants.values()))
    assert tenant.expires_at == DAY * 20000 + 12 * 3600 - 300
    (tenant.root / 'private-result').write_text('private')
    clock[0] = tenant.expires_at
    waiting = ticket()
    response = enter(client, waiting)
    assert response.status_code == 202 and response.json()['reason'] == 'cleanup'
    assert not tenant.root.exists()
    assert client.get('/api/v1/session', headers=auth).status_code == 401
    clock[0] += 150
    assert enter(client, waiting).status_code == 202
    clock[0] += 150
    admitted = enter(client, waiting)
    assert admitted.status_code == 201
    assert next(iter(manager.tenants.values())).expires_at == DAY * 20001 - 300


def test_global_and_network_queue_limits_apply_before_allocating_workspaces(room):
    client, manager, _ = room
    manager.config = replace(manager.config, max_waiting=2, waiting_per_ip=1)
    bearer(enter(client, ticket()))
    assert enter(client, ticket()).status_code == 202
    assert enter(client, ticket()).json()['error']['code'] == 'queue_quota'
    client.headers['CF-Connecting-IP'] = '203.0.113.11'
    assert enter(client, ticket()).status_code == 202
    client.headers['CF-Connecting-IP'] = '203.0.113.12'
    assert enter(client, ticket()).json()['error']['code'] == 'queue_full'
    assert len(manager.tenants) == 1


def test_low_shared_storage_waits_without_materializing_a_workspace(room, monkeypatch):
    client, manager, _ = room
    def full(_):
        raise HTTPException(507, {'error': {'code': 'storage_busy'}})
    monkeypatch.setattr(manager, 'reserve_storage', full)
    response = enter(client, ticket())
    assert response.status_code == 202 and response.json()['reason'] == 'storage'
    assert not manager.tenants and manager.storage_reserved == 0


def test_idle_capacity_exceeds_old_limit_without_per_visitor_threads(room):
    client, manager, _ = room
    manager.config = replace(manager.config, max_sessions=32)
    before = {thread.ident for thread in threading.enumerate() if thread.name.startswith('opendpd-')}
    for _ in range(32):
        auth = bearer(enter(client, ticket()))
        assert client.get('/api/v1/session', headers=auth).status_code == 200
    assert len(manager.tenants) == 32
    assert not manager.scheduled
    assert before == {thread.ident for thread in threading.enumerate() if thread.name.startswith('opendpd-')}
    assert all(t.app.state.sweeps._thread is None and t.app.state.supervisor._thread is None for t in manager.tenants.values())
