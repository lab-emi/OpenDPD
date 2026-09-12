from opendpd.services import about


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
