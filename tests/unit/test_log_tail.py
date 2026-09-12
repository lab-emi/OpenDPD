from types import SimpleNamespace

from opendpd.schemas import RunStatus
from opendpd.server import routes


def test_terminal_tail_keeps_byte_cursor_and_finishes_a_partial_line(tmp_path, monkeypatch):
    record = SimpleNamespace(status=RunStatus.running)
    monkeypatch.setattr(routes, "_get_run", lambda *_: record)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(ws=SimpleNamespace(run_dir=lambda _: tmp_path))))
    path = tmp_path / "logs" / "worker.log"
    path.parent.mkdir()
    complete = "".join(f"batch {i} · I/Q\n" for i in range(50_000)).encode()
    path.write_bytes(complete + b"final result")
    page = routes.runs_logs("run", request, offset=0, limit=3, tail=True)
    assert page.lines == [f"batch {i} · I/Q" for i in range(49_997, 50_000)]
    assert page.next_offset == len(complete)
    assert not page.eof
    record.status = RunStatus.succeeded
    last = routes.runs_logs("run", request, offset=page.next_offset, limit=3)
    assert last.lines == ["final result"]
    assert last.next_offset == path.stat().st_size and last.eof
