"""Browser handoff must stay on supported local routes and retain authentication."""
import json
import urllib.error
import urllib.parse
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from opendpd.sdk.client import Project
from opendpd.server.app import create_app
from opendpd.studio.navigation import destination, valid_destination


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("navigation")
    app = create_app(root / "workspace", bootstrap_token="navigation-test-token", static_dir=root / "absent")
    with TestClient(app, base_url="http://127.0.0.1:8765", follow_redirects=False) as c:
        yield c


def fresh_token(client):
    """Bootstrap tokens are single use; mint one the way the launcher does."""
    sessions = client.app.state.sessions
    return sessions.mint(sessions.launcher_secret)


@pytest.mark.parametrize("page,run_id,path", [
    ("home", None, "/"), ("datasets", None, "/datasets"),
    ("new-experiment", None, "/experiments/new"), ("results", None, "/results"),
    ("run", "run-abc.123", "/runs/run-abc.123"), ("result", "run_a", "/results/run_a")])
def test_authenticated_browser_handoff(client, page, run_id, path):
    assert destination(page, run_id) == path
    response = client.get("/bootstrap", params={"token": fresh_token(client), "next": path})
    assert response.status_code == 303
    assert response.headers["location"] == path
    assert "token" not in response.headers["location"]
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie
    assert client.get("/api/v1/system/capabilities").status_code == 200


@pytest.mark.parametrize("path", [
    "https://evil.example/", "//evil.example", "/\\evil.example", "/runs/../evil", "/runs/%2e%2e",
    "/runs/foo%2Fbar", "/runs/foo?next=https://evil.example", "/runs/a#fragment", "/runs/", "/unknown",
    "/datasets\r\nLocation: https://evil.example", "/results/" + "a" * 129])
def test_redirect_targets_rejected(client, path):
    assert not valid_destination(path)
    response = client.get("/bootstrap", params={"token": fresh_token(client), "next": path})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_destination"
    assert "location" not in response.headers and "set-cookie" not in response.headers


def test_destination_does_not_bypass_authentication(client):
    response = client.get("/bootstrap", params={"token": "wrong", "next": "/datasets"})
    assert response.status_code == 401
    assert "set-cookie" not in response.headers


def test_home_link_and_missing_frontend_remain_usable(client):
    response = client.get("/bootstrap", params={"token": "navigation-test-token"})
    assert response.status_code == 303 and response.headers["location"] == "/"
    ready = client.get("/readyz")
    assert ready.status_code == 503
    assert ready.json()["ready"] is False
    assert ready.json()["studio_navigation_version"] == 1


def test_sdk_returns_not_ready_snapshot_instead_of_http_error():
    class Opener:
        def open(self, request, timeout):
            assert urllib.parse.urlparse(request.full_url).path == "/readyz"
            raise urllib.error.HTTPError(request.full_url, 503, "Unavailable", {},
                BytesIO(json.dumps({"ready": False, "problems": ["frontend missing"]}).encode()))
    p = Project.__new__(Project)
    p._closed, p._csrf, p._base_url, p._opener = False, "csrf", "http://127.0.0.1:8765", Opener()
    p.workspace = Path("/tmp/navigation-workspace")
    info = p.studio_info()
    assert info["ready"] is False and info["problems"] == ["frontend missing"]
    assert info["workspace"] == str(p.workspace)


@pytest.mark.parametrize("page,run_id", [("run", None), ("result", "../x"),
    ("datasets", "unexpected"), ("https://evil.example", None)])
def test_sdk_invalid_destinations(page, run_id):
    with pytest.raises(ValueError):
        destination(page, run_id)
