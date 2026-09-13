"""Authenticated generator → data export → PA dataset → exact test split."""
import pytest
from fastapi.testclient import TestClient

from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("allow_datasets", [True, False])
def test_generator_and_dataset_access_boundaries(tmp_path, allow_datasets):
    with TestClient(create_app(tmp_path / "ws", bootstrap_token="generator", allow_custom_datasets=allow_datasets), base_url="http://127.0.0.1:8877") as client:
        assert client.get('/api/v1/signal-generator/presets').status_code == 401
        auth = client.post('/api/v1/session/bootstrap', json={'token': 'generator'}).json()
        presets = client.get('/api/v1/signal-generator/presets').json()
        config = {**presets[0]['config'], 'n_samples': 16384}
        assert client.post('/api/v1/signal-generator/signals', json=config).status_code == 403
        client.headers[CSRF_HEADER] = auth['csrf_token']
        client.headers['origin'] = 'http://127.0.0.1:8877'
        assert client.post('/api/v1/signal-generator/validate', json={**config, 'fft_size': 999}).status_code == 422
        assert client.post('/api/v1/signal-generator/validate', json={**config, 'pilot_mode': 'explicit', 'pilot_indices': [999999]}).status_code == 422
        assert client.post('/api/v1/signal-generator/validate', json=config).json() == config
        generated = client.post('/api/v1/signal-generator/signals', json=config)
        assert generated.status_code == 201, generated.text
        data = generated.json()
        assert data['analysis']['sample_count'] == 16384
        assert client.get(data['download_url']).content[:2] == b'PK'
        saved = client.post(f"/api/v1/signal-generator/signals/{data['signal_id']}/dataset", json={'dataset_id': 'generated-pa'})
        if allow_datasets:
            assert saved.status_code == 201, saved.text
            counts = client.get('/api/v1/datasets/generated-pa/sample-counts').json()
            assert counts['counts']['test'] == saved.json()['test_samples']
            assert client.get('/api/v1/datasets/generated-pa/sample-counts?version=missing').status_code == 409
        else:
            assert saved.status_code == 403
