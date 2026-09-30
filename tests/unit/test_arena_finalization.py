"""No incomplete or unaudited matrix can replace the reference asset."""
from types import SimpleNamespace

import pytest

from benchmark import finalize_arena_reference as finalizer
from opendpd.services.workspace import read_json,write_json_atomic


@pytest.mark.parametrize('audit_ok',[False,True])
def test_publication_requires_freeze_test_and_successful_independent_audit(tmp_path,monkeypatch,audit_ok):
    events=[]
    root=tmp_path/'work';assets=tmp_path/'assets';assets.mkdir()
    repo=tmp_path/'repo';docs=repo/'docs/performance';docs.mkdir(parents=True)
    current=finalizer.arena.protocol()
    protocol=current.model_copy(update={'boards':current.boards[:1]})
    monkeypatch.setattr(finalizer.arena,'protocol',lambda:protocol)
    monkeypatch.setattr(finalizer.arena,'bundled_backbones',lambda:[SimpleNamespace(key='gru')])
    monkeypatch.setattr(finalizer.arena,'ASSETS',assets)
    monkeypatch.setattr(finalizer.arena,'__file__',str(repo/'opendpd/core/arena.py'))
    monkeypatch.setattr(finalizer,'units',lambda:[('gru',250,'apa-200mhz')])
    monkeypatch.setattr(finalizer,'freeze',lambda *_:events.append('freeze'))
    def evaluate(*_):
        assert events==['freeze'];events.append('test')
        return dict(status='succeeded')
    monkeypatch.setattr(finalizer.baseline,'evaluate',evaluate)
    def publish(*_,output):
        assert events==['freeze','test'];events.append('candidate')
        write_json_atomic(output,dict(sha256='s',rows=[]))
    monkeypatch.setattr(finalizer.baseline,'publish',publish)
    def audit(*_,bundle_path):
        assert events==['freeze','test','candidate'];events.append('audit')
        assert not (assets/finalizer.arena.RESULTS_FILE).exists()
        return dict(integrity_ok=audit_ok,complete=True,execution_failures=[]),[]
    monkeypatch.setattr(finalizer.auditing,'audit',audit)
    monkeypatch.setattr(finalizer.arena,'load_official_rows',lambda:[SimpleNamespace(completed_cases=3)])
    monkeypatch.setattr(finalizer,'report_markdown',lambda *_:'verified report')
    monkeypatch.setattr(finalizer,'verification_reports',lambda *_:None)
    if audit_ok:
        finalizer.complete(root)
        assert read_json(root/'finalization-progress.json')['phase']=='complete'
        assert (assets/finalizer.arena.RESULTS_FILE).exists()
        assert (docs/'arena-reference-results.md').read_text()=='verified report'
    else:
        with pytest.raises(RuntimeError,match='publication withheld'):finalizer.complete(root)
        assert not (assets/finalizer.arena.RESULTS_FILE).exists()
    assert events==['freeze','test','candidate','audit']
