import io
import json

from opendpd.services import about


def test_activity_reads_only_fixed_public_endpoints_without_user_data(monkeypatch):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        assert timeout == 6
        return io.BytesIO(json.dumps([]).encode())

    monkeypatch.setattr(about, '_cache', None)
    monkeypatch.setattr(about, '_checked', -float('inf'))
    monkeypatch.setattr(about.urllib.request, 'urlopen', urlopen)
    result = about.project_info()
    assert result['status'] == 'current'
    assert sorted(request.full_url for request in requests) == [
        'https://api.github.com/repos/lab-emi/OpenDPD/commits?per_page=5',
        'https://api.github.com/repos/lab-emi/OpenDPD/contributors?per_page=100',
    ]
    for request in requests:
        assert request.get_method() == 'GET'
        assert request.data is None
        assert {name.lower(): value for name, value in request.header_items()} == {
            'user-agent': 'OpenDPD-Studio', 'accept': 'application/vnd.github+json',
        }


def test_public_activity_cache_and_explicit_stale_fallback(monkeypatch):
    calls = []
    def get(endpoint):
        calls.append(endpoint)
        if endpoint.startswith('/contributors'):
            return [{"login": "contributor", "contributions": 2, "html_url": "https://github.com/contributor"}]
        return [{"sha": "abc", "html_url": "https://github.com/lab-emi/OpenDPD/commit/abc",
                 "commit": {"message": "Summary\n\nDetails", "author": {"name": "Contributor", "date": "2026-09-12"}}}]
    now = [1000]
    monkeypatch.setattr(about, '_cache', None)
    monkeypatch.setattr(about, '_checked', -float('inf'))
    monkeypatch.setattr(about, '_get', get)
    monkeypatch.setattr(about.time, 'monotonic', lambda: now[0])
    first = about.project_info()
    assert first['status'] == 'current'
    assert first['commits'][0]['message'] == 'Summary'
    assert about.project_info() == first and len(calls) == 2
    def offline(_): raise OSError('offline')
    monkeypatch.setattr(about, '_get', offline)
    now[0] += 301
    stale = about.project_info()
    assert stale['status'] == 'stale' and stale['updated_at'] == first['updated_at']
    assert stale['contributors'] == first['contributors']
