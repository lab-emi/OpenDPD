"""Data-only uploads, isolated contribution branches, catalog and real PA/DPD runs."""
import base64
import json

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from opendpd.core.backbone_template import TEMPLATE, scan_source
from opendpd.schemas import ExperimentConfig, RunStatus, SignalSpec
from opendpd.schemas.user_backbones import BackboneCapability, BackboneConsent
from opendpd.server.app import create_app
from opendpd.server.security import CSRF_HEADER
from opendpd.services import datasets
from opendpd.services.experiments import create_run, execute_run, load_resolved, load_result
from opendpd.services.recipes import instantiate
from opendpd.services.user_backbones import BackboneController, BackbonePublisher, package_manifest
from opendpd.services.workspace import Workspace, WorkspaceError
from tests.integration.test_dataset_publication import LocalGitHub, git, wait

pytestmark = pytest.mark.integration
SOURCE = TEMPLATE.replace('"author": ""', '"author": "Fixture contributor"').encode()


class LocalBackboneGitHub(BackbonePublisher, LocalGitHub):
    def __init__(self, root, **kwargs):
        LocalGitHub.__init__(self, root, **kwargs)

    def review_gate_available(self):
        return True

    def capability(self):
        return BackboneCapability(publication_available=True)

    def gh(self, *args, cwd=None, optional=False):
        if args[:2] == ('repo', 'fork') and self.fork_repo.exists():
            return ''  # gh repo fork reuses an existing verified fork.
        if args[:2] == ('pr','create'):
            from pathlib import Path
            self.calls.append(args)
            body = Path(args[args.index('--body-file')+1]).read_text()
            assert 'Human review is required' in body and 'Source SHA256' in body
            if self.fail_after_push:
                self.fail_after_push = False
                raise WorkspaceError('Simulated network failure after push')
            head=args[args.index('--head')+1]
            self.prs.append({'url':'https://github.com/lab-emi/OpenDPD/pull/456','state':'OPEN',
                             'head':head if ':' in head else f'lab-emi:{head}'})
            return self.prs[0]['url']
        return LocalGitHub.gh(self,*args,cwd=cwd,optional=optional)


def consent(record):
    return BackboneConsent(source_sha256=record.source_sha256, package_sha256=record.package_sha256,
                           publish_publicly=True, rights_confirmed=True)


@pytest.mark.parametrize('fork',[False,True])
def test_opt_in_creates_only_validated_branch_and_pr_recovers_without_merge(tmp_path,fork):
    remote = LocalBackboneGitHub(tmp_path, fork=fork, fail_after_push=True)
    controller = BackboneController(Workspace.create(tmp_path/'ws'),remote)
    record = controller.upload('my_model.py',SOURCE)
    assert not remote.calls
    assert controller.upload('renamed.py',SOURCE) == record
    assert not controller.catalog().entries
    controller.start(record.publication_id,consent(record))
    failed = wait(controller,record)
    assert failed.status == 'failed' and failed.commit_sha
    controller.start(record.publication_id,consent(record))
    done = wait(controller,record)
    assert done.status == 'submitted' and done.pull_request_url.endswith('/456')
    assert len(remote.prs) == 1
    assert controller.start(record.publication_id,consent(record)) == done
    assert git('--git-dir',str(remote.upstream),'rev-parse','main') == remote.base
    assert all(call[:2] not in (('pr','merge'),('api','merges')) for call in remote.calls)
    target = remote.fork_repo if fork else remote.upstream
    added = git('--git-dir',str(target),'diff','--name-only','main',record.branch).splitlines()
    assert set(added) == {f'{record.directory}/backbone.py',f'{record.directory}/manifest.json'}
    assert not controller.catalog().entries, 'An open PR must never enter the merged catalog.'


def test_tampering_and_mismatched_consent_cannot_publish(tmp_path):
    controller = BackboneController(Workspace.create(tmp_path/'ws'))
    record = controller.upload('safe.py',SOURCE)
    with pytest.raises(WorkspaceError,match='Consent'):
        controller.start(record.publication_id,consent(record).model_copy(update={'source_sha256':'f'*64}))
    (controller.directory(record.publication_id)/'package'/'backbone.py').write_bytes(SOURCE+b'\nimport os\n')
    with pytest.raises(WorkspaceError):
        controller.start(record.publication_id,consent(record))
    assert not controller.threads and controller.get(record.publication_id).status == 'prepared'


def test_utf8_bom_is_parsed_without_changing_stored_or_published_bytes(tmp_path):
    controller=BackboneController(Workspace.create(tmp_path/'ws'))
    source=b'\xef\xbb\xbf'+SOURCE
    record=controller.upload('windows.py',source)
    assert record.source_sha256==package_manifest(source)['source_sha256']
    assert (controller.directory(record.publication_id)/'package/backbone.py').read_bytes()==source
    assert record.model==controller.upload('unix.py',SOURCE).model


def test_bytes_changed_during_copy_never_reach_remote(tmp_path, monkeypatch):
    from opendpd.services import dataset_publication
    remote=LocalBackboneGitHub(tmp_path)
    controller=BackboneController(Workspace.create(tmp_path/'ws'),remote)
    record=controller.upload('safe.py',SOURCE)
    original=dataset_publication.shutil.copytree
    def changed(source,destination,*args,**kwargs):
        result=original(source,destination,*args,**kwargs)
        (destination/'backbone.py').write_bytes(b'import os\n')
        return result
    monkeypatch.setattr(dataset_publication.shutil,'copytree',changed)
    controller.start(record.publication_id,consent(record))
    failed=wait(controller,record)
    assert failed.status=='failed' and not failed.commit_sha
    assert not remote.prs
    assert record.branch not in git('--git-dir',str(remote.upstream),'branch','--list')


def test_unrelated_existing_branch_is_never_overwritten(tmp_path):
    remote=LocalBackboneGitHub(tmp_path)
    controller=BackboneController(Workspace.create(tmp_path/'ws'),remote)
    record=controller.upload('safe.py',SOURCE)
    source=tmp_path/'seed'
    git('checkout','-b',record.branch,cwd=source)
    import shutil
    shutil.copytree(controller.directory(record.publication_id)/'package',source/record.directory)
    (source/'README.md').write_text('An unrelated unapproved change.\n')
    git('add','.',cwd=source)
    git('-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit','-m','unexpected branch contents',cwd=source)
    git('push','origin',record.branch,cwd=source)
    before=git('rev-parse','HEAD',cwd=source)
    controller.start(record.publication_id,consent(record))
    assert wait(controller,record).status=='failed'
    assert not remote.prs
    assert git('--git-dir',str(remote.upstream),'rev-parse',record.branch)==before


def test_lost_response_after_fork_pr_creation_recovers_the_same_pr(tmp_path,monkeypatch):
    remote=LocalBackboneGitHub(tmp_path,fork=True)
    controller=BackboneController(Workspace.create(tmp_path/'ws'),remote)
    record=controller.upload('safe.py',SOURCE)
    original=remote.gh
    lost=True
    def command(*args,**kwargs):
        nonlocal lost
        result=original(*args,**kwargs)
        if args[:2]==('pr','create') and lost:
            lost=False
            raise WorkspaceError('PR was created, but its response was lost.')
        return result
    monkeypatch.setattr(remote,'gh',command)
    controller.start(record.publication_id,consent(record))
    assert wait(controller,record).status=='failed' and len(remote.prs)==1
    controller.start(record.publication_id,consent(record))
    assert wait(controller,record).status=='submitted' and len(remote.prs)==1
    assert any(args[:2]==('api','repos/lab-emi/OpenDPD/pulls') and f'head=fixture:{record.branch}' in args for args in remote.calls)


def test_private_store_rejects_root_entry_and_package_symlinks(tmp_path):
    ws=Workspace.create(tmp_path/'ws')
    outside=tmp_path/'outside'; outside.mkdir()
    (ws.root/'user-backbones').symlink_to(outside,target_is_directory=True)
    with pytest.raises(WorkspaceError,match='symbolic link'):
        BackboneController(ws)
    (ws.root/'user-backbones').unlink()
    controller=BackboneController(ws)
    digest=package_manifest(SOURCE)['source_sha256']
    location=controller.root/f'bbpr-{digest}'
    location.symlink_to(outside,target_is_directory=True)
    with pytest.raises(WorkspaceError,match='symbolic link'):
        controller.upload('safe.py',SOURCE)
    location.unlink(); location.mkdir()
    (location/'package').symlink_to(outside,target_is_directory=True)
    with pytest.raises(WorkspaceError,match='symbolic link'):
        controller.upload('safe.py',SOURCE)
    assert not list(outside.iterdir())


def test_api_auth_csrf_private_default_restart_and_workspace_isolation(tmp_path):
    app = create_app(tmp_path/'ws', bootstrap_token='backbones', allow_backbone_publications=False, monitor_resources=False, start_sweeps=False)
    with TestClient(app, base_url='http://127.0.0.1:8877') as client:
        assert client.get('/api/v1/backbones/template').status_code == 401
        auth=client.post('/api/v1/session/bootstrap',json={'token':'backbones'}).json()
        assert client.get('/api/v1/backbones/template').text == TEMPLATE
        body={'filename':'safe.py','source':SOURCE.decode()}
        assert client.post('/api/v1/backbones/uploads',json=body).status_code == 403
        client.headers[CSRF_HEADER]=auth['csrf_token']
        response=client.post('/api/v1/backbones/uploads',json=body)
        assert response.status_code == 201, response.text
        record=response.json()
        assert record['status']=='prepared' and record['consent_at'] is None
        assert not client.get('/api/v1/backbones/capability').json()['publication_available']
        assert client.post(f'/api/v1/backbones/uploads/{record["publication_id"]}/submit',json={
            'source_sha256':record['source_sha256'],'package_sha256':record['package_sha256'],
            'publish_publicly':True,'rights_confirmed':True}).status_code == 403
        for bad in ('import os\n', 'BACKBONE = __import__("os")'):
            assert client.post('/api/v1/backbones/uploads',json={**body,'source':bad}).status_code == 422
        assert client.post('/api/v1/backbones/uploads',content=json.dumps({**body,'source':'\ud800'}),
                           headers={'Content-Type':'application/json'}).status_code == 422
        assert client.post('/api/v1/backbones/uploads',json={**body,'source':'x'*100000}).status_code == 413
        assert client.get(f'/api/v1/backbones/uploads/{record["publication_id"]}/source').content == SOURCE
        assert len(client.get('/api/v1/backbones/uploads').json()) == 1
    again=BackboneController(Workspace.open_or_create(tmp_path/'ws'))
    assert again.list()[0].source_sha256 == record['source_sha256']
    other=BackboneController(Workspace.create(tmp_path/'other'))
    assert not other.list()
    with pytest.raises(WorkspaceError):
        other.get(record['publication_id'])


def test_only_main_catalog_data_is_loaded_atomically_and_invalid_updates_keep_previous(tmp_path):
    controller=BackboneController(Workspace.create(tmp_path/'ws'))
    manifest=package_manifest(SOURCE)
    name=f'demo_{manifest["source_sha256"][:16]}'
    commit='a'*40
    calls=[]
    source=SOURCE
    def get(path):
        calls.append(path)
        if path=='git/ref/heads/main': return {'object':{'sha':commit}}
        if path.startswith('contents/backbones/user_uploaded?'):
            assert path.endswith('?ref='+commit)
            return [{'name':name,'type':'dir','sha':'c'*40}]
        if path.startswith('git/trees/'):
            return {'truncated':False,'tree':[{'path':n,'type':'blob','mode':'100644','sha':s*40} for n,s in [('backbone.py','d'),('manifest.json','e')]]}
        data=source if path.endswith('d'*40) else json.dumps(manifest).encode()
        return {'sha':path.split('/')[-1],'size':len(data),'encoding':'base64','content':base64.b64encode(data).decode()}
    catalog=controller.refresh_catalog(get)
    assert len(catalog.entries)==1 and catalog.entries[0].origin=='community'
    assert catalog.entries[0].source_commit==commit
    assert all('pulls/' not in path for path in calls)
    source=SOURCE+b'\nprint("malicious")\n'
    controller.last_catalog_refresh=0
    refused=controller.refresh_catalog(get)
    assert refused.warning and refused.entries==catalog.entries
    assert controller.catalog().entries==catalog.entries


def test_repository_gate_requires_active_no_bypass_independent_reviews_and_bound_status(monkeypatch):
    from pathlib import Path
    rule=json.loads(Path('deployment/user-backbone-ruleset.json').read_text())
    publisher=BackbonePublisher()
    monkeypatch.setattr(publisher,'gh',lambda *args,**kw:json.dumps(
        [{'ruleset_id':1,'ruleset_source_type':'Repository'}] if args[1].endswith('/main') else rule))
    assert publisher.review_gate_available()
    del rule['bypass_actors']
    assert not publisher.review_gate_available(), 'Hidden bypass metadata is not evidence of no bypass actors.'
    rule['bypass_actors']=[{'actor_type':'RepositoryRole','actor_id':5,'bypass_mode':'always'}]
    assert not publisher.review_gate_available()
    rule['bypass_actors']=[]
    rule['rules'][2]['parameters']['dismiss_stale_reviews_on_push']=False
    assert not publisher.review_gate_available()


def test_public_upload_transfers_only_validated_graph_to_isolated_gpu_job(tmp_path, monkeypatch):
    import io
    import zipfile
    from opendpd.web.app import create_web_app
    from opendpd.web.policy import WebConfig
    from opendpd.web.runtime import DAY
    from opendpd.web import gpu_broker
    from types import SimpleNamespace
    import time
    from tests.integration.test_gpu_bridge import TOKEN, TUNNEL, eventually
    now=lambda:DAY*20000+3600
    monkeypatch.setattr(gpu_broker,'time',SimpleNamespace(time=now,monotonic=time.monotonic))
    config=WebConfig(tmp_path/'web','https://opendpd.com','api.opendpd.com',TUNNEL,gpu_token=TOKEN)
    app=create_web_app(config,now=now)
    with TestClient(app,base_url='http://127.0.0.1',client=('127.0.0.1',10)) as client:
        public={'Host':TUNNEL,'Origin':'https://opendpd.com','X-Forwarded-Proto':'https','CF-Connecting-IP':'203.0.113.10'}
        private={'X-OpenDPD-GPU':TOKEN}
        client.post('/_gpu/poll',json={'name':'Fixture CUDA'},headers=private)
        tokens=[]
        for _ in range(2):
            session=client.post('/api/v1/web/sessions',json={},headers=public)
            assert session.status_code==201,session.text
            tokens.append({**public,'Authorization':'Bearer '+session.json()['access_token']})
        response=client.post('/api/v1/backbones/uploads',json={'filename':'safe.py','source':SOURCE.decode()},headers=tokens[0])
        assert response.status_code==201,response.text
        upload=response.json()
        assert client.get(f'/api/v1/backbones/uploads/{upload["publication_id"]}/source',headers=tokens[1]).status_code==409
        assert not client.get('/api/v1/backbones/capability',headers=tokens[0]).json()['publication_available']
        assert client.post('/api/v1/backbones/uploads',json={'filename':'evil.py','source':'import os\n'},headers=tokens[0]).status_code==422
        assert any(m['key']=='user_template' for m in client.get('/api/v1/models',headers=tokens[0]).json())
        assert client.post('/api/v1/datasets/import-builtin',json={'name':'MyCustomPA'},headers=tokens[0]).status_code==201
        cfg=instantiate('pa-user_template-smoke-v1','mycustompa',device='cuda').model_dump(mode='json')
        cfg['model']=upload['model']
        response=client.post('/api/v1/runs',json={'config':cfg},headers=tokens[0])
        assert response.status_code==201,response.text
        rid=response.json()['run_id']
        job=eventually(lambda:client.post('/_gpu/poll',json={'name':'Fixture CUDA'},headers=private).json()['job'])
        archive=client.get(f'/_gpu/jobs/{job["id"]}/input',headers={**private,'X-OpenDPD-Lease':job['lease']})
        with zipfile.ZipFile(io.BytesIO(archive.content)) as package:
            assert not any(name.endswith('.py') or 'user-backbones/' in name for name in package.namelist())
            resolved=json.loads(package.read(f'runs/{rid}/config.resolved.json'))
            assert resolved['model']==upload['model']
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_uploaded_pa_and_distinct_dpd_train_and_test_with_hash_bound_graphs(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from tests.fixtures.synthetic import Impairments, synthesize
    x,y=synthesize(8192,5,impairments=Impairments())
    path=tmp_path/'capture.csv'
    pd.DataFrame({'I_in':x[:,0],'Q_in':x[:,1],'I_out':y[:,0],'Q_out':y[:,1]}).to_csv(path,index=False)
    ws=Workspace.create(tmp_path/'ws')
    datasets.import_dataset(ws,path,dataset_id='capture',display_name='Synthetic template test',origin='synthetic',
        signal=SignalSpec(sample_rate_hz=800e6,bandwidth_hz=200e6,n_sub_ch=10,nperseg=512,amplitude_units='normalized'),guard_samples=64)
    controller=BackboneController(ws)
    pa_template=controller.upload('pa.py',SOURCE)
    other=scan_source(SOURCE)
    other['name']='Small LSTM DPD'; other['nodes'][0].update(op='lstm',features=6)
    dpd_template=controller.upload('dpd.py',('BACKBONE = '+repr(other)+'\n').encode())
    def run(config):
        result=execute_run(ws,create_run(ws,config).run_id)
        assert result.status==RunStatus.succeeded,result.error
        report=load_result(ws,result.run_id)
        assert report and any(m.name=='NMSE' and m.value is not None and np.isfinite(m.value) for m in report.metrics)
        return result
    pa_config=instantiate('pa-user_template-smoke-v1','capture',device=device)
    pa_config.model=pa_template.model
    pa_config.training.epochs=2
    pa_config.execution.num_threads=2
    pa=run(pa_config)
    dpd_config=instantiate('dpd-user_template-smoke-v1','capture',pa_run_id=pa.run_id,device=device)
    dpd_config.model=dpd_template.model
    dpd_config.training.epochs=2
    dpd_config.execution.num_threads=2
    dpd=run(dpd_config)
    resolved=load_resolved(ws,dpd.run_id)
    assert resolved.model==dpd_template.model and resolved.pa_reference.model==pa_template.model
    for task,source_run,model in (('evaluate_pa',pa,pa_template.model),('run_dpd',dpd,dpd_template.model)):
        ref='pa_reference' if task=='evaluate_pa' else 'dpd_reference'
        config=ExperimentConfig.model_validate({'task':task,'dataset':{'id':'capture'},'model':model.model_dump(),
            'execution':{'device':device,'num_threads':2}, ref:{'run_id':source_run.run_id}})
        evaluated=run(config)
        assert load_resolved(ws,evaluated.run_id).model==model
