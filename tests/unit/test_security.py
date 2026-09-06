"""Pure checks for the local access boundary (no server needed)."""

from opendpd.server.security import SessionStore, host_is_loopback, origin_matches


def test_host_is_loopback():
    for ok in ("127.0.0.1:8765", "localhost", "LOCALHOST:80", "[::1]:8765", "[::1]"):
        assert host_is_loopback(ok), ok
    for bad in ("evil.example", "127.0.0.1.evil.example", "localhost.evil.example", "", None, "10.0.0.5:8765"):
        assert not host_is_loopback(bad), bad


def test_origin_must_match_host_when_present():
    assert origin_matches(None, "127.0.0.1:8765")          # non-browser client
    assert origin_matches("http://127.0.0.1:8765", "127.0.0.1:8765")
    assert origin_matches("http://127.0.0.1:8765/runs/1", "127.0.0.1:8765")   # Referer form
    assert not origin_matches("http://127.0.0.1:9999", "127.0.0.1:8765")
    assert not origin_matches("http://evil.example", "127.0.0.1:8765")
    assert not origin_matches("null", "127.0.0.1:8765")
    assert not origin_matches("file:///tmp/x.html", "127.0.0.1:8765")


def test_bootstrap_token_is_single_secret_and_sessions_are_random():
    store = SessionStore("secret")
    assert store.exchange("wrong") is None
    assert store.exchange("") is None
    a, b = store.exchange("secret"), store.exchange("secret")
    assert a.session_id != b.session_id and a.csrf_token != b.csrf_token
    assert len(a.session_id) >= 40
    assert store.get(a.session_id) is a and store.get("nope") is None and store.get(None) is None
