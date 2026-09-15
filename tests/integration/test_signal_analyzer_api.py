"""Signal import, authentication, source isolation and whole-capture validation."""
import pytest
from fastapi.testclient import TestClient

from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER
from opendpd.services.csv_upload import CsvUploadRejected
from opendpd.services.signal_analyzer import admit_signal_upload
from opendpd.services.workspace import Workspace
from opendpd.web.policy import allowed, expensive_request

pytestmark = pytest.mark.integration


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path/'ws', bootstrap_token='analyzer', monitor_resources=False,
                              start_sweeps=False), base_url='http://127.0.0.1:8877') as c:
        assert c.get('/api/v1/signal-analyzer/sources').status_code == 401
        auth = c.post('/api/v1/session/bootstrap', json={'token': 'analyzer'}).json()
        c.headers['origin'] = 'http://127.0.0.1:8877'
        assert c.post('/api/v1/signal-analyzer/analyze', json={}).status_code == 403
        c.headers[CSRF_HEADER] = auth['csrf_token']
        yield c


@pytest.mark.parametrize('header,rows,real', [
    ('voltage', ['0.25']*512, True), ('z', ['0.2+0.3i']*512, False),
    ('I,Q', ['0.2,0.3']*512, False), ('', ['-0.5']*512, True),
    ('I', ['0.25']*512, True), ('', ['i']*512, False),
])
def test_real_complex_and_iq_uploads_analyze_without_pa_output(client, header, rows, real):
    data = ('\n'.join(([header] if header else [])+rows)+'\n').encode()
    upload = client.post('/api/v1/signal-analyzer/upload', files={'file': ('capture.csv', data, 'text/csv')})
    assert upload.status_code == 201, upload.text
    info = upload.json()
    assert info['sample_count'] == 512 and info['origin'] == 'uploaded'
    assert info['sample_rate_hz'] is None
    request = {'source': info['source'], 'config': {'sample_rate_hz': 1e6, 'bandwidth_hz': 1e5, 'start_sample': 128, 'n_samples': 256}}
    response = client.post('/api/v1/signal-analyzer/analyze', json=request)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['sample_range'] == [128, 384] and result['real_signal'] == real
    assert result['source_sha256']
    assert client.get('/api/v1/signal-analyzer/sources').json()[0]['source'] == info['source']
    assert client.get('/api/v1/datasets').json() == []


def test_signal_generation_and_virtual_pa_are_direct_analyzer_sources(client):
    generated = client.post('/api/v1/signal-generator/signals', json={'n_samples': 8192}).json()
    model = client.get('/api/v1/pa-library/models').json()[0]['model_id']
    sim = client.post('/api/v1/pa-library/simulations', json={'input_signal_id': generated['signal_id'], 'model_id': model})
    assert sim.status_code == 201, sim.text
    entries = client.get('/api/v1/signal-analyzer/sources').json()
    assert len(entries) == 2
    inp = next(s['source'] for s in entries if s['source']['kind'] == 'generated')
    out = next(s['source'] for s in entries if s['source']['kind'] == 'virtual_pa')
    response = client.post('/api/v1/signal-analyzer/analyze', json={'source': out, 'reference': inp})
    assert response.status_code == 200, response.text
    assert any(m['key'] == 'reference_nmse' for m in response.json()['measurements'])
    bad = {**inp, 'source_id': 'sg-'+'0'*64}
    assert client.post('/api/v1/signal-analyzer/analyze', json={'source': bad}).status_code == 409
    assert client.post('/api/v1/signal-analyzer/analyze', json={'source': {**inp, 'source_id': '../outside'}}).status_code == 422
    assert client.post('/api/v1/signal-analyzer/analyze', json={'source': {**inp, 'role': 'output'}}).status_code == 422
    assert client.post('/api/v1/signal-analyzer/analyze', json={'source': {**out, 'role': 'input'}}).status_code == 422


def test_numeric_column_selection_preserves_complex_components(client):
    data = ('t,signal\n'+'\n'.join(f'{i},0.2+0.3j' for i in range(512))).encode()
    info = client.post('/api/v1/signal-analyzer/upload', files={'file': ('capture.csv', data)}).json()
    request = {'source': info['source'], 'config': {'sample_format': 'complex', 'i_column': 1}}
    result = client.post('/api/v1/signal-analyzer/analyze', json=request)
    assert result.status_code == 200
    assert result.json()['time_i'][0] == .2 and result.json()['time_q'][0] == .3
    request['config']['sample_format'] = 'real'
    assert client.post('/api/v1/signal-analyzer/analyze', json=request).status_code == 409


@pytest.mark.parametrize('bad', [b'z\n'+b'1\n'*256+b'nan\n', b'z\n'+b'1\n'*256+b'inf\n',
    b'z\n'+b'1\n'*256+b'=HYPERLINK("x")\n', b'z\n'+b'1\n'*256+b'1,2\n', b'z\n1\n', b'\x00'*512,
    b'z\n'+b'1\n'*256+b'1e300\n', b'z\n'+b'1\n'*256+b'exec(1)\n'])
def test_entire_csv_is_validated_and_rejected_files_are_removed(tmp_path, bad):
    ws = Workspace.create(tmp_path/'ws')
    path = tmp_path/'bad.csv'; path.write_bytes(bad)
    with pytest.raises(CsvUploadRejected):
        admit_signal_upload(ws, path)
    assert not path.exists()
    assert not list((ws.root/'signal_uploads').glob('sa-*'))


def test_new_public_surface_is_explicit_and_numeric_work_is_limited():
    for path in ('/signal-analyzer/analyze', '/signal-analyzer/upload'):
        assert allowed('POST', path) and expensive_request('POST', path)
    assert allowed('GET', '/signal-analyzer/sources')
    assert not allowed('GET', '/signal-analyzer/files')
    assert not allowed('POST', '/signal-analyzer/execute')
