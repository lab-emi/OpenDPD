"""The distributed fitter cannot cross the global train/test boundary."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark import run_arena_distributed as distributed
from opendpd.core import arena
from opendpd.services.workspace import read_json,write_json_atomic


def test_plan_contains_all_unique_measured_training_units():
    units=distributed.units()
    assert len(units)==len(set(units))==78
    assert {u[2] for u in units}=={'apa-200mhz-b'}
    assert all(u[0]!='ilc_dpd' and not u[0].endswith('_stream') for u in units)
    assert sum(1 if u[0] in arena.DETERMINISTIC else len(arena.SEEDS) for u in units)==218


def test_local_only_ignores_remote_routing_and_starts_no_remote_slots(tmp_path,monkeypatch):
    seen=[]
    monkeypatch.setattr('sys.argv',['run_arena_distributed','--workspace',str(tmp_path),
        '--local-only','--remote','unused-host','--remote-gpu-workers','1'])
    monkeypatch.setattr(distributed,'coordinate',lambda args:seen.append(args))
    distributed.main()
    assert len(seen)==1 and seen[0].local_only
    assert seen[0].remote_cpu_workers==seen[0].remote_gpu_workers==0
    assert seen[0].remote is seen[0].remote_root is seen[0].remote_python is None


def test_all_checkpoints_are_verified_before_matrix_is_frozen(tmp_path,monkeypatch):
    unit=('mp_ls',250,'apa-200mhz-b')
    folder=tmp_path/'cache'/'case'/'seed-0';folder.mkdir(parents=True)
    (folder/'weights.npz').write_bytes(b'checkpoint')
    monkeypatch.setattr(distributed,'folders',lambda *_:[folder])
    calls=[]
    monkeypatch.setattr(distributed,'verify',lambda _,u:calls.append(u))
    distributed.freeze(tmp_path,[unit])
    record=read_json(tmp_path/'dpd-frozen.json')
    assert calls==[unit] and record['test_access'] is False
    assert record['checkpoints']==[dict(path=str(folder.relative_to(tmp_path)),
                                      weights_sha256=arena.file_hash(folder/'weights.npz'))]
    (tmp_path/'dpd-frozen.json').unlink()
    def reject(*_):raise ValueError('missing seed')
    monkeypatch.setattr(distributed,'verify',reject)
    with pytest.raises(ValueError,match='missing seed'):distributed.freeze(tmp_path,[unit])
    assert not (tmp_path/'dpd-frozen.json').exists()


def test_resumed_units_retain_their_optimizer_device(tmp_path,monkeypatch):
    import torch
    paths=[tmp_path/f'seed-{seed}' for seed in range(3)]
    for path in paths:path.mkdir()
    monkeypatch.setattr(distributed,'folders',lambda *_:paths)
    unit=('gru',250,'apa-200mhz-b')
    assert distributed.resume_device(tmp_path,unit) is None
    for path in paths:
        torch.save(dict(device='cpu'),path/'resume.pt')
    assert distributed.resume_device(tmp_path,unit)=='cpu'
    torch.save(dict(device='cuda'),paths[0]/'resume.pt')
    with pytest.raises(ValueError,match='Incompatible resume devices'):
        distributed.resume_device(tmp_path,unit)
    for path in paths:
        torch.save(dict(device='cuda'),path/'resume.pt')
    assert distributed.resume_device(tmp_path,unit)=='cuda'


@pytest.mark.parametrize('saved,available',[('cpu','cpu'),('cuda','cuda'),('cpu','cuda'),('cuda','cpu')])
def test_coordinator_only_assigns_resumes_to_a_compatible_slot(tmp_path,monkeypatch,saved,available):
    unit=('gru',250,'apa-200mhz-b');calls=[]
    monkeypatch.setattr(distributed,'units',lambda:[unit])
    monkeypatch.setattr(distributed,'resume_device',lambda *_:saved)
    monkeypatch.setattr(distributed,'verify',lambda *_:None)
    monkeypatch.setattr(distributed,'freeze',lambda *_:None)
    def execute(command,*_):
        calls.append(command[command.index('--device')+1]);return 0
    monkeypatch.setattr(distributed,'run_guarded',execute)
    args=SimpleNamespace(workspace=tmp_path,local_only=True,remote=None,remote_root=None,
        local_cpu_workers=int(available=='cpu'),local_gpu_workers=int(available=='cuda'),
        remote_cpu_workers=0,remote_gpu_workers=0)
    if saved==available:
        distributed.coordinate(args)
        assert calls==[saved]
    else:
        with pytest.raises(RuntimeError,match='1 unassigned'):
            distributed.coordinate(args)
        assert not calls


def test_a_busy_gpu_never_launches_a_second_worker(tmp_path,monkeypatch):
    monkeypatch.setattr(distributed,'gpu_worker_active',lambda _:True)
    monkeypatch.setattr(distributed,'run_guarded',lambda *_:pytest.fail('second GPU process launched'))
    assert distributed.worker(tmp_path,('gru',250,'apa-200mhz-b'),'cuda',arena.training_fingerprint())==75
    assert not (tmp_path/'prefetch'/'gpu.claim').exists()


def test_restarted_coordinator_waits_for_live_work_and_busy_leases(tmp_path,monkeypatch):
    unit=('gru',250,'apa-200mhz-b');calls=[];waits=[]
    running=iter([True,False,False]);codes=iter([75,0])
    monkeypatch.setattr(distributed,'units',lambda:[unit])
    monkeypatch.setattr(distributed,'resume_device',lambda *_:'cpu')
    monkeypatch.setattr(distributed,'foreign_worker',lambda *_:next(running))
    monkeypatch.setattr(distributed,'verify',lambda *_:None)
    monkeypatch.setattr(distributed,'freeze',lambda *_:None)
    monkeypatch.setattr(distributed.time,'sleep',lambda seconds:waits.append(seconds))
    def execute(command,*_):
        assert waits  # Existing training was allowed to finish first.
        calls.append(command);return next(codes)
    monkeypatch.setattr(distributed,'run_guarded',execute)
    args=SimpleNamespace(workspace=tmp_path,local_only=True,remote=None,remote_root=None,
        local_cpu_workers=1,local_gpu_workers=0,remote_cpu_workers=0,remote_gpu_workers=0)
    distributed.coordinate(args)
    state=read_json(tmp_path/'distributed-progress.json')
    assert len(calls)==2 and waits==[30,30]
    assert state['complete_units']==1 and state['failed']==[]
    assert state['all_checkpoints_frozen_before_test'] is True


@pytest.mark.parametrize('key',['pgjanet','dvrjanet','apnrru'])
def test_compiled_cells_serialize_seed_warmup_and_tail_updates(tmp_path,key):
    environment=distributed.execution_environment(tmp_path,(key,1000,'apa-200mhz-b'),'cuda')
    assert environment['OPENDPD_ARENA_PARALLEL_SEEDS']=='0'
    if key in ('dvrjanet','apnrru'):
        assert environment['TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS']=='ATEN'
    if key=='apnrru':
        assert environment['OPENDPD_DISABLE_ARENA_COMPILE']=='0'


def gpu_profile(tmp_path,monkeypatch):
    monkeypatch.setattr(distributed.os,'sched_getaffinity',lambda _:set(range(8)),raising=False)
    evidence=tmp_path/'validation.json'
    write_json_atomic(evidence,dict(passed=True,test_access=False,parity_steps=6,
        paced_stability_seconds=120,compile_disabled='0',cuda_launch_blocking='0'))
    profile=dict(profile='compiled-without-cuda-graphs',training_sha256=arena.training_fingerprint(),
        cpu_affinity=[4,5,6,7],validation_report='validation.json',validation_sha256=arena.file_hash(evidence))
    write_json_atomic(tmp_path/'local-gpu-runtime.json',profile)
    return profile


def test_validated_gpu_profile_disables_graphs_and_records_affinity(tmp_path,monkeypatch):
    profile=gpu_profile(tmp_path,monkeypatch)
    assert distributed.local_gpu_runtime(tmp_path)==profile
    environment=distributed.execution_environment(tmp_path,('apnrru',1000,'apa-200mhz-b'),'cuda')
    assert environment['OPENDPD_ARENA_PARALLEL_SEEDS']=='0'
    assert environment['OPENDPD_DISABLE_CUDA_FAST_PATH']=='1'
    assert environment['OPENDPD_DISABLE_CUDA_GRAPH_FROZEN_DGRU']=='1'
    assert environment['OPENDPD_DISABLE_ARENA_COMPILE']=='0'
    assert environment['CUDA_LAUNCH_BLOCKING']=='0'


def test_cpu_execution_does_not_read_or_apply_a_gpu_profile(tmp_path):
    (tmp_path/'local-gpu-runtime.json').write_text('invalid GPU policy')
    environment=distributed.execution_environment(tmp_path,('gru',1000,'apa-200mhz-b'),'cpu')
    assert environment['CUDA_VISIBLE_DEVICES']==''


@pytest.mark.parametrize('damage',['evidence','affinity','training','incomplete'])
def test_unverified_gpu_profiles_cannot_launch(tmp_path,monkeypatch,damage):
    profile=gpu_profile(tmp_path,monkeypatch)
    if damage=='evidence':(tmp_path/'validation.json').write_text('{}')
    elif damage=='affinity':profile['cpu_affinity']=[99]
    elif damage=='training':profile['training_sha256']='stale'
    else:
        write_json_atomic(tmp_path/'validation.json',dict(passed=False,test_access=False))
        profile['validation_sha256']=arena.file_hash(tmp_path/'validation.json')
    write_json_atomic(tmp_path/'local-gpu-runtime.json',profile)
    with pytest.raises(ValueError):distributed.local_gpu_runtime(tmp_path)


def replay_profile(tmp_path,monkeypatch):
    profile=gpu_profile(tmp_path,monkeypatch)
    report=dict(passed=True,test_access=False,training_sha256=profile['training_sha256'],
        unit=['dvrjanet',500,'apa-200mhz-b'],graph_enabled=True,frozen_pa_graph_disabled=True,
        parallel_seeds=False,cpu_affinity=profile['cpu_affinity'],cases=[
            dict(case=name,passed=True,parity_steps=16,speedup=3,
                 stability_updates=2048 if name=='resumed' else 0,
                 stability_seconds=180 if name=='resumed' else 0)
            for name in ['resumed','fresh']])
    write_json_atomic(tmp_path/'replay.json',report)
    profile['profile']='compiled-with-reviewed-replay'
    profile['replay_units']=[dict(unit=report['unit'],validation_report='replay.json',
        validation_sha256=arena.file_hash(tmp_path/'replay.json'))]
    write_json_atomic(tmp_path/'local-gpu-runtime.json',profile)
    return profile,report


def test_reviewed_replay_is_enabled_only_for_the_validated_unit(tmp_path,monkeypatch):
    replay_profile(tmp_path,monkeypatch)
    matched=distributed.execution_environment(tmp_path,('dvrjanet',500,'apa-200mhz-b'),'cuda')
    assert matched['OPENDPD_DISABLE_CUDA_FAST_PATH']=='0'
    assert matched['OPENDPD_ARENA_PARALLEL_SEEDS']=='0'
    assert matched['OPENDPD_DISABLE_CUDA_GRAPH_FROZEN_DGRU']=='1'
    for unit in [('dvrjanet',1000,'apa-200mhz-b'),('dvrjanet',500,'dpa-200mhz'),('apnrru',500,'apa-200mhz-b')]:
        assert distributed.execution_environment(tmp_path,unit,'cuda')['OPENDPD_DISABLE_CUDA_FAST_PATH']=='1'


@pytest.mark.parametrize('damage',['missing_fresh','short_stability','wrong_unit','wrong_affinity','changed_hash'])
def test_replay_requires_bound_fresh_and_resumed_evidence(tmp_path,monkeypatch,damage):
    profile,report=replay_profile(tmp_path,monkeypatch)
    if damage=='missing_fresh':report['cases']=report['cases'][:1]
    elif damage=='short_stability':report['cases'][0]['stability_updates']=1
    elif damage=='wrong_unit':report['unit']=['dvrjanet',1000,'apa-200mhz-b']
    elif damage=='wrong_affinity':report['cpu_affinity']=[0,1,2,3]
    else:report['passed']=False
    write_json_atomic(tmp_path/'replay.json',report)
    if damage!='changed_hash':profile['replay_units'][0]['validation_sha256']=arena.file_hash(tmp_path/'replay.json')
    write_json_atomic(tmp_path/'local-gpu-runtime.json',profile)
    with pytest.raises(ValueError):distributed.local_gpu_runtime(tmp_path)


@pytest.mark.parametrize('qualifies',[True,False])
def test_automatic_replay_review_is_isolated_memoized_and_falls_back(tmp_path,monkeypatch,qualifies):
    profile,report=replay_profile(tmp_path,monkeypatch)
    profile['replay_units']=[];profile['auto_review_replay']=True;profile['auto_review_scripts']={}
    for name in ['diagnose_graph_speed.py','run_bounded_graph_speed.py']:
        (tmp_path/name).write_text('# reviewed diagnostic fixture\n')
        profile['auto_review_scripts'][name]=arena.file_hash(tmp_path/name)
    write_json_atomic(tmp_path/'local-gpu-runtime.json',profile)
    write_json_atomic(tmp_path/'thermal-guard.json',dict(updated_at=distributed.time.time()))
    monkeypatch.setattr(distributed,'folders',lambda *_:[tmp_path/'missing-fit'])
    (tmp_path/'prefetch').mkdir();hold=distributed.claim(tmp_path/'prefetch/gpu.claim')
    calls=[];published=[]
    def record_write(path,value):
        if path==tmp_path/'local-gpu-runtime.json':published.append(value)
        write_json_atomic(path,value)
    monkeypatch.setattr(distributed,'write_json_atomic',record_write)
    def execute(command,*args):
        assert distributed.claim(tmp_path/'prefetch/gpu.claim') is None
        assert '--unit' in command and '--report-dir' in command
        out=Path(command[command.index('--report-dir')+1]);calls.append(command)
        if not qualifies:report['cases'][0]['stability_updates']=1
        write_json_atomic(out/'graph-speed-diagnostic.json',report)
        write_json_atomic(out/'graph-speed-manager.json',dict(phase='completed',exit_code=0))
        return 0
    monkeypatch.setattr(distributed,'run_guarded',execute)
    try:
        result=distributed.review_gpu_replay(tmp_path,('dvrjanet',500,'apa-200mhz-b'),profile)
        assert bool(result['replay_units'])==qualifies
        assert distributed.local_gpu_runtime(tmp_path)==result
        again=distributed.review_gpu_replay(tmp_path,('dvrjanet',500,'apa-200mhz-b'),result)
        assert again==result and len(calls)==1
        attempt=read_json(tmp_path/'replay-validation/dvrjanet--500--apa-200mhz-b/attempt.json')
        assert attempt['adopted']==qualifies and attempt['test_access'] is False
        # A concurrent reader must never see an unqualified candidate, even
        # temporarily before the reviewer falls back to the existing profile.
        assert published==([result] if qualifies else [])
        assert not (tmp_path/'missing-fit').exists()
    finally:distributed.release(tmp_path/'prefetch/gpu.claim',hold)


def test_automatic_replay_review_rejects_changed_diagnostic_code(tmp_path,monkeypatch):
    profile=gpu_profile(tmp_path,monkeypatch)
    profile.update(auto_review_replay=True,auto_review_scripts={})
    monkeypatch.setattr(distributed,'folders',lambda *_:[tmp_path/'missing-fit'])
    monkeypatch.setattr(distributed,'run_guarded',lambda *_:pytest.fail('Unverified code launched'))
    with pytest.raises(ValueError,match='scripts differ'):
        distributed.review_gpu_replay(tmp_path,('dvrjanet',500,'apa-200mhz-b'),profile)


def test_verification_rejects_an_incomplete_neural_training_budget(tmp_path):
    from opendpd.core.arena_engine import cache_binding
    unit=('tres_gru',250,'apa-200mhz-b')
    params=arena.model_parameters(unit[0],unit[1])
    for seed,folder in zip(arena.SEEDS,distributed.folders(tmp_path,unit)):
        folder.mkdir(parents=True);(folder/'weights.npz').write_bytes(b'checkpoint')
        write_json_atomic(folder/'training.json',dict(cache_binding=cache_binding(unit[0],params,unit[2],seed),
            weights_sha256=arena.file_hash(folder/'weights.npz'),attained_epochs=239,
            **arena.training_budget(arena.calibration()[unit[2]]['counts']['train'])))
    with pytest.raises(ValueError,match='Incomplete'):distributed.verify(tmp_path,unit)


@pytest.mark.parametrize('message,retries',[('TypeError: _foreach_addcdiv_ failed',True),
                                         ('FloatingPointError: Nonfinite training loss',False)])
def test_execution_recovery_keeps_gpu_lease_and_does_not_hide_numerical_failures(tmp_path,monkeypatch,message,retries):
    monkeypatch.setattr(distributed,'gpu_worker_active',lambda _:False)
    monkeypatch.setattr(distributed,'foreign_worker',lambda _:False)
    monkeypatch.setattr(distributed,'folders',lambda *_:[])
    monkeypatch.setattr(distributed,'verify',lambda *_:None)
    calls=[]
    def execute(command,log,timeout,environment):
        # Both attempts hold the same exclusive GPU lease; another helper
        # cannot start a second training process during recovery.
        assert distributed.claim(tmp_path/'prefetch/gpu.claim') is None
        calls.append(environment.copy());log.write_text(message if len(calls)==1 else 'complete')
        return 1 if len(calls)==1 else 0
    monkeypatch.setattr(distributed,'run_guarded',execute)
    result=distributed.worker(tmp_path,('gru',250,'apa-200mhz-b'),'cuda',arena.training_fingerprint())
    assert result==(0 if retries else 1)
    assert len(calls)==(2 if retries else 1)
    assert not (tmp_path/'prefetch/gpu.claim').exists()
    if retries:
        assert calls[1]['OPENDPD_ARENA_PARALLEL_SEEDS']=='0'
        assert calls[1]['OPENDPD_DISABLE_ARENA_COMPILE']=='1'
        assert calls[1]['OPENDPD_DISABLE_CUDA_FAST_PATH']=='1'
        assert (tmp_path/'prefetch/gru--250--apa-200mhz-b.cuda.attempt-1.log').exists()
    assert Path(calls[0]['TORCHINDUCTOR_CACHE_DIR']).is_relative_to(tmp_path)
