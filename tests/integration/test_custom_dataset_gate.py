"""The initial Studio release cannot receive custom datasets through any web entry."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER, DatasetImportBoundary


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(tmp_path_factory.mktemp("closed-imports"), bootstrap_token="gate-test")
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        session = client.post("/api/v1/session/bootstrap", json={"token": "gate-test"}).json()
        client.headers[CSRF_HEADER] = session["csrf_token"]
        yield client


@pytest.mark.parametrize("path", [
    "/datasets/upload", "/datasets/upload/", "/datasets/import", "/datasets/csv",
    "/datasets/csv/preview", "/datasets/inspect", "/imports",
    "/datasets/import-roots", "/datasets/import-roots/imports/files",
])
def test_import_entry_points_are_closed_without_writing_files(client, path):
    before = list(client.app.state.ws.imports_dir.rglob("*"))
    response = client.request("GET" if "import-roots" in path else "POST", f"/api/v1{path}", content=b"untrusted body")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "custom_datasets_coming_soon"
    assert response.headers["cache-control"] == "no-store"
    assert list(client.app.state.ws.imports_dir.rglob("*")) == before


def test_builtin_dataset_remains_available_with_closed_imports(client):
    assert client.get("/api/v1/system/capabilities").json()["custom_dataset_imports"] is False
    response = client.post("/api/v1/datasets/import-builtin", json={"name": "DPA_200MHz"})
    assert response.status_code == 201
    analysis = client.get("/api/v1/datasets/dpa-200mhz/analysis").json()
    assert analysis["total_samples"] > 0
    assert analysis["inspection_ready"] is True


def test_closed_upload_does_not_read_or_parse_the_request_body():
    messages = []

    async def unexpected(*args):
        raise AssertionError("closed import must not read the body or call the route")

    async def send(message):
        messages.append(message)

    asyncio.run(DatasetImportBoundary(unexpected)({"type": "http", "path": "/api/v1/datasets/upload"}, unexpected, send))
    assert messages[0]["status"] == 403
