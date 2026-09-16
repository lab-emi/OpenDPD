"""Regression cases for service, library and publishing boundaries."""
import asyncio
import io
from pathlib import Path
import sys
import threading
from unittest.mock import Mock

from fastapi import UploadFile
import pytest


def test_builtin_catalog_does_not_reread_content_and_returns_independent_values(monkeypatch):
    from opendpd.services import workspace
    workspace._builtin_catalog.cache_clear()
    counted = Mock(wraps=workspace.sha256_file)
    monkeypatch.setattr(workspace, 'sha256_file', counted)
    first = workspace.list_builtin_datasets()
    count = counted.call_count
    assert count > 0
    first[0].description = 'changed by a caller'
    second = workspace.list_builtin_datasets()
    assert counted.call_count == count
    assert second[0].description != first[0].description
    workspace._builtin_catalog.cache_clear()


def test_upload_consumer_runs_off_loop_and_closes_on_failure():
    from opendpd.server.uploads import consume_upload
    main_thread = threading.get_ident()
    file = UploadFile(io.BytesIO(b'csv'))
    def consumer(chunks):
        assert threading.get_ident() != main_thread
        assert b''.join(chunks) == b'csv'
        raise ValueError('rejected')
    with pytest.raises(ValueError, match='rejected'):
        asyncio.run(consume_upload(file, consumer))
    assert file.file.closed


def test_upload_cancellation_waits_for_receiver_before_close():
    from opendpd.server.uploads import consume_upload
    async def scenario():
        file = UploadFile(io.BytesIO(b'data'))
        ready, finish = threading.Event(), threading.Event()
        def consumer(chunks):
            ready.set()
            assert finish.wait(5)
            assert not file.file.closed
            return b''.join(chunks)
        task = asyncio.create_task(consume_upload(file, consumer))
        await asyncio.to_thread(ready.wait, 5)
        task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert file.file.closed
    asyncio.run(scenario())


def test_legacy_choices_and_constructors_have_bidirectional_parity():
    from arguments import build_parser
    from opendpd.core.backbone_builders import BACKBONE_BUILDERS
    from opendpd.core.registry import legacy_choices
    parser = build_parser()
    for role in ('pa', 'dpd'):
        choices = next(a.choices for a in parser._actions if a.dest == role.upper() + '_backbone')
        assert set(choices) == set(legacy_choices(role)) == set(BACKBONE_BUILDERS)


def test_public_api_preserves_argv_and_path_on_success_and_rejection(monkeypatch):
    from opendpd import api
    from types import SimpleNamespace
    before_argv, before_path = sys.argv[:], sys.path[:]
    captured = []
    monkeypatch.setattr(api, 'Project', lambda args: captured.append(args) or SimpleNamespace(path_save_file_best='model', path_log_file_best='log'))
    monkeypatch.setattr(api.train_pa_module, 'main', lambda project: None)
    api.train_pa(dataset_path='/tmp/custom', n_epochs=1)
    assert captured[0].dataset_path == '/tmp/custom'
    with pytest.raises(ValueError):
        api.train_pa(dataset_name='test', PA_backbone='missing')
    assert sys.argv == before_argv and sys.path == before_path


def test_hashed_stores_refuse_parent_symlinks(tmp_path):
    from opendpd.services.workspace import Workspace, WorkspaceError
    from opendpd.services import signal_generator, signal_analyzer, virtual_pa
    ws = Workspace.create(tmp_path / 'workspace')
    outside = tmp_path / 'outside'
    outside.mkdir()
    for kind, prefix, directory in [('signals', 'sg', signal_generator.directory), ('pa_simulations', 'vpa', virtual_pa.directory), ('signal_uploads', 'sa', signal_analyzer.upload_directory)]:
        (ws.root / kind).symlink_to(outside, target_is_directory=True)
        with pytest.raises(WorkspaceError, match='symbolic link'):
            directory(ws, prefix + '-' + 'a' * 64)


def test_action_references_are_immutable_and_publish_is_release_only():
    import re
    root = Path(__file__).resolve().parents[2]
    for file in (root / '.github/workflows').glob('*.yml'):
        for ref in re.findall(r'uses:\s+(\S+)', file.read_text()):
            assert re.fullmatch(r'[\w./-]+@[0-9a-f]{40}', ref), ref
    publish = (root / '.github/workflows/publish.yml').read_text()
    assert 'workflow_dispatch' not in publish
    assert 'RELEASE_TAG: ${{ github.event.release.tag_name }}' in publish


def test_late_gpu_result_cannot_recreate_retired_workspace(tmp_path):
    from types import SimpleNamespace
    from opendpd.web.gpu_broker import GpuBroker, Job
    from opendpd.services.workspace import Workspace
    from opendpd.web.gpu_archive import pack
    ws = Workspace.create(tmp_path / 'workspace')
    run = ws.run_dir('run-test')
    run.mkdir(parents=True)
    supervisor = SimpleNamespace(ws=ws)
    job = Job('a' * 32, supervisor, 'run-test', float('inf'))
    broker = GpuBroker()
    broker.jobs[job.id] = job
    broker.retire(supervisor, ws.root)
    assert job.done and not broker.jobs and not ws.root.exists()
    broker.result(job, pack(tmp_path, []), 0)
    assert not ws.root.exists()


def test_imported_plotting_keeps_the_callers_backend_and_styles():
    import importlib
    import matplotlib
    import matplotlib.pyplot as plt
    import utils.plotting
    import datasets.plot_utils
    with plt.rc_context({'font.size': 19}):
        before = dict(plt.rcParams)
        backend = matplotlib.get_backend()
        importlib.reload(utils.plotting)
        importlib.reload(datasets.plot_utils)
        assert dict(plt.rcParams) == before
        assert matplotlib.get_backend() == backend
