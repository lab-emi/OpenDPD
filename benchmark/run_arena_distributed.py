"""Prefetch the Arena matrix on a local host and one explicitly named SSH host.

Workers only fit train/validation data. Completed remote checkpoints are copied
back and verified; test evaluation remains a separate, local phase.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time

from benchmark.run_arena_baselines import (CPU_ONLY, CPU_SECONDS, GPU_SECONDS,
    claim, foreign_worker, release, run_guarded)
from opendpd.core import arena
from opendpd.core.registry import get_model
from opendpd.runtime.arena_limits import ARENA_MAX_RUNTIME_SECONDS
from opendpd.services.workspace import read_json, write_json_atomic

CPU_DENSE = frozenset({'deltagru','deltajanet','bojanet'})


def units():
    bases=sorted({get_model(m.key).weights_from or m.key for m in arena.bundled_backbones()})
    return [(key,budget,c) for key in bases for budget in arena.BUDGETS
            if arena.model_parameters(key,budget) is not None
            for c in sorted(arena.active_calibration())]


def folders(root,unit):
    from opendpd.core.arena_runner import cache_folder
    key,budget,condition=unit
    params=arena.model_parameters(key,budget)
    return [cache_folder(root/'cache',key,params,condition,seed)
            for seed in (arena.SEEDS[:1] if key in arena.DETERMINISTIC else arena.SEEDS)]


def verify(root,unit):
    from opendpd.core.arena_engine import cache_binding
    key,budget,condition=unit
    params=arena.model_parameters(key,budget)
    for seed,folder in zip(arena.SEEDS,folders(root,unit)):
        info=read_json(folder/'training.json')
        if (info['cache_binding'] != cache_binding(key,params,condition,seed)
                or info['weights_sha256'] != arena.file_hash(folder/'weights.npz')):
            raise ValueError(f'Checkpoint integrity failure: {unit}, seed {seed}')
        if key not in arena.DETERMINISTIC:
            expected=arena.training_budget(arena.calibration()[condition]['counts']['train'])
            if info['attained_epochs'] != arena.TRAINING['epochs'] or any(info[k] != v for k,v in expected.items()):
                raise ValueError(f'Incomplete full-window budget: {unit}, seed {seed}')


def resume_device(root,unit):
    """An incomplete optimizer state must continue on its recorded device."""
    paths=[folder/'resume.pt' for folder in folders(root,unit)
           if (folder/'resume.pt').exists()]
    if not paths:return None
    import torch
    devices={torch.load(path,map_location='cpu',weights_only=True)['device'] for path in paths}
    if len(devices)!=1 or not devices.issubset({'cpu','cuda'}):
        raise ValueError(f'Incompatible resume devices for {unit}: {sorted(devices)}')
    return devices.pop()


def freeze(root, planned):
    """Require every requested checkpoint before constructing any test context."""
    expected=arena.training_fingerprint()
    checkpoints=[]
    for unit in planned:
        verify(root,unit)
        for folder in folders(root,unit):
            checkpoints.append(dict(path=str(folder.relative_to(root)),
                                    weights_sha256=arena.file_hash(folder/'weights.npz')))
    if arena.training_fingerprint()!=expected:
        raise ValueError('Training sources changed while freezing checkpoints')
    frozen=root/'dpd-frozen.json'
    if frozen.exists():
        previous=read_json(frozen)
        if previous['training_sha256']!=expected or previous['checkpoints']!=checkpoints:
            raise ValueError('Previously frozen DPD matrix changed')
    else:
        write_json_atomic(frozen,dict(training_sha256=expected,
            frozen_at=time.time(),test_access=False,checkpoints=checkpoints))
    return checkpoints


def local_gpu_runtime(root):
    """Optional, evidence-bound execution profile for a thermally limited host."""
    path=root/'local-gpu-runtime.json'
    if not path.exists():return None
    return _validate_local_gpu_runtime(root,read_json(path))


def _validate_local_gpu_runtime(root,profile):
    """Validate before publishing so readers never observe a rejected profile."""
    if (profile.get('profile') not in ('compiled-without-cuda-graphs','compiled-with-reviewed-replay')
            or profile.get('training_sha256')!=arena.training_fingerprint()):
        raise ValueError('Unrecognized local GPU execution profile')
    cores=profile.get('cpu_affinity')
    if (not isinstance(cores,list) or not cores or any(type(c) is not int or c<0 for c in cores)
            or len(set(cores))!=len(cores) or not set(cores).issubset(os.sched_getaffinity(0))):
        raise ValueError('Local GPU CPU affinity is unavailable')
    evidence=root/profile['validation_report']
    if not evidence.resolve().is_relative_to(root.resolve()):
        raise ValueError('Local GPU validation evidence must be in the workspace')
    if arena.file_hash(evidence)!=profile['validation_sha256']:
        raise ValueError('Local GPU validation evidence changed')
    report=read_json(evidence)
    if (report.get('passed') is not True or report.get('test_access') is not False
            or report.get('parity_steps',0)<6 or report.get('paced_stability_seconds',0)<120
            or report.get('compile_disabled')!='0' or report.get('cuda_launch_blocking')!='0'):
        raise ValueError('Local GPU numerical/stability validation is incomplete')
    seen=set()
    for entry in profile.get('replay_units',[]):
        unit=tuple(entry['unit'])
        if unit not in units() or unit in seen:
            raise ValueError('Invalid or duplicate reviewed CUDA replay unit')
        seen.add(unit)
        evidence=root/entry['validation_report']
        if (not evidence.resolve().is_relative_to(root.resolve())
                or arena.file_hash(evidence)!=entry['validation_sha256']):
            raise ValueError('CUDA replay validation evidence changed')
        replay=read_json(evidence)
        cases={case['case']:case for case in replay.get('cases',[])}
        if (replay.get('passed') is not True or replay.get('test_access') is not False
                or replay.get('training_sha256')!=profile['training_sha256']
                or replay.get('unit')!=list(unit) or replay.get('graph_enabled') is not True
                or replay.get('frozen_pa_graph_disabled') is not True
                or replay.get('parallel_seeds') is not False
                or replay.get('cpu_affinity')!=cores or set(cases)!={'resumed','fresh'}
                or any(c.get('passed') is not True or c.get('parity_steps',0)<16
                       or c.get('speedup',0)<1.25 for c in cases.values())
                or cases['resumed'].get('stability_updates',0)<2048
                or cases['resumed'].get('stability_seconds',0)<180):
            raise ValueError('CUDA replay numerical/stability validation is incomplete')
    return profile


def execution_environment(root,unit,device):
    cache=root/'compiler-cache'
    for directory in ('inductor','triton','tmp'):(cache/directory).mkdir(parents=True,exist_ok=True)
    environment=dict(os.environ,OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='2',
        OPENDPD_ARENA_PARALLEL_SEEDS='1',NVIDIA_TF32_OVERRIDE='0',
        TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),TMPDIR=str(cache/'tmp'))
    if device=='cpu':environment['CUDA_VISIBLE_DEVICES']=''
    # A contiguous real tail batch can recompile after the strided warmup.
    # Dynamo's tracing state then interferes with another thread's Adam scalar
    # dispatch. Sequential seeds avoid that race without changing AdamW/math.
    if unit[0] in ('pgjanet','dvrjanet','apnrru'):environment['OPENDPD_ARENA_PARALLEL_SEEDS']='0'
    # APNRRU's complete 200-sample forward/backward now passes the native
    # numerical checks with the runner's bounded Inductor fusion profile.
    if unit[0]=='apnrru':environment['OPENDPD_DISABLE_ARENA_COMPILE']='0'
    if unit[0] in ('dvrjanet','apnrru'):environment['TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS']='ATEN'
    profile=local_gpu_runtime(root) if device=='cuda' else None
    if profile is not None:
        environment.update(OPENDPD_ARENA_PARALLEL_SEEDS='0',OPENDPD_DISABLE_CUDA_FAST_PATH='1',
            OPENDPD_DISABLE_CUDA_GRAPH_FROZEN_DGRU='1',OPENDPD_DISABLE_ARENA_COMPILE='0',
            CUDA_LAUNCH_BLOCKING='0')
        if any(entry['unit']==list(unit) for entry in profile.get('replay_units',[])):
            environment['OPENDPD_DISABLE_CUDA_FAST_PATH']='0'
    return environment


def review_gpu_replay(root,unit,profile):
    """Qualify an unreviewed unit once, under the already-held GPU lease.

    The diagnostic uses private train-only model copies. An unsuccessful
    optimization check leaves the proven non-replay fitting path available.
    """
    if (profile is None or not profile.get('auto_review_replay')
            or unit[0] in arena.DETERMINISTIC
            or any(entry['unit']==list(unit) for entry in profile.get('replay_units',[]))
            or all((f/'training.json').exists() and (f/'weights.npz').exists() for f in folders(root,unit))):
        return profile
    from opendpd.core.arena_cuda import ARENA_REPLAY_BACKBONES
    if unit[0] not in ARENA_REPLAY_BACKBONES:return profile
    directory=root/'replay-validation'/('--'.join(map(str,unit)))
    attempt=directory/'attempt.json'
    if attempt.exists():return profile
    sources=profile.get('auto_review_scripts',{})
    required={'diagnose_graph_speed.py','run_bounded_graph_speed.py'}
    if set(sources)!=required or any(arena.file_hash(root/name)!=value for name,value in sources.items()):
        raise ValueError('Automatic CUDA review scripts differ from the approved snapshot')
    guard=read_json(root/'thermal-guard.json')
    if time.time()-guard['updated_at']>15:
        raise RuntimeError('Thermal guard is stale before CUDA execution review')
    directory.mkdir(parents=True,exist_ok=True)
    for name in required:shutil.copy2(root/name,directory/name)
    record=dict(unit=list(unit),phase='running',started_at=time.time(),
                source_sha256=sources,test_access=False,official_experiment=False)
    write_json_atomic(attempt,record)
    command=[sys.executable,str(root/'run_bounded_graph_speed.py'),'--unit',*map(str,unit),
             '--report-dir',str(directory)]
    code=run_guarded(command,directory/'manager.log',1200,
        dict(os.environ,OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    manager=read_json(directory/'graph-speed-manager.json') if (directory/'graph-speed-manager.json').exists() else {}
    report=directory/'graph-speed-diagnostic.json'
    candidate={**profile,'profile':'compiled-with-reviewed-replay','replay_units':[
        *profile.get('replay_units',[]),dict(unit=list(unit),validation_report=str(report.relative_to(root)),
            validation_sha256=arena.file_hash(report) if report.exists() else '')]}
    adopted=False;reason=manager.get('error') or f'Qualification exited {code}'
    if code==0 and manager.get('phase')=='completed' and report.exists():
        # The observer and new workers read this file concurrently. Keep the
        # last valid profile visible until all candidate evidence passes.
        try:
            _validate_local_gpu_runtime(root,candidate)
        except (ValueError,KeyError,TypeError):
            reason='Qualification did not satisfy the replay evidence gate'
        else:
            write_json_atomic(root/'local-gpu-runtime.json',candidate)
            profile=candidate;adopted=True;reason='Verified fresh/resumed parity, speed and stability'
    record.update(phase='complete',finished_at=time.time(),adopted=adopted,reason=reason,exit_code=code)
    write_json_atomic(attempt,record)
    guard=read_json(root/'thermal-guard.json')
    if time.time()-guard['updated_at']>15:
        raise RuntimeError('Thermal guard unavailable after CUDA execution review')
    return profile


def worker(root,unit,device,expected):
    """A remote unit is protected against duplicate launches and lost SSH sessions."""
    if arena.training_fingerprint()!=expected:
        raise ValueError('Remote scientific sources/assets differ from the coordinator')
    profile=local_gpu_runtime(root) if device=='cuda' else None
    if profile is not None:
        os.sched_setaffinity(0,set(profile['cpu_affinity']))
    name='--'.join(map(str,unit));directory=root/'prefetch';directory.mkdir(parents=True,exist_ok=True)
    gpu_lease=directory/'gpu.claim'
    gpu_hold=claim(gpu_lease) if device=='cuda' else None
    if device=='cuda':
        # The host-wide lease also protects a restarted coordinator. An
        # orphaned child can outlive its SSH/helper, so inspect this workspace's
        # own workers before allowing another GPU training process.
        if gpu_hold is None or gpu_worker_active(root):
            release(gpu_lease,gpu_hold)
            return 75
    lease=directory/f'{name}.remote.claim'
    hold=claim(lease)
    if hold is None or foreign_worker([str(x) for x in unit]):
        release(lease,hold)
        release(gpu_lease,gpu_hold)
        return 75
    try:
        if device=='cuda':profile=review_gpu_replay(root,unit,profile)
        command=[sys.executable,'-m','opendpd.core.arena_runner','--cache',str(root/'cache'),
                 '--prefetch',*[str(x) for x in unit],'--device',device]
        environment=execution_environment(root,unit,device)
        provenance_keys=('NVIDIA_TF32_OVERRIDE','OPENDPD_ARENA_PARALLEL_SEEDS',
                         'TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS','OPENDPD_DISABLE_ARENA_COMPILE',
                         'OPENDPD_DISABLE_CUDA_FAST_PATH','OPENDPD_DISABLE_CUDA_GRAPH_FROZEN_DGRU',
                         'CUDA_LAUNCH_BLOCKING')
        attempt_path=directory/f'{name}.execution.json'
        attempts=read_json(attempt_path) if attempt_path.exists() else []
        prior={str(folder):read_json(folder/'training.json') for folder in folders(root,unit)
               if (folder/'training.json').exists()}
        log=directory/f'{name}.{device}.log'
        if log.exists():
            log.replace(directory/f'{name}.{device}.before-{time.time_ns()}.log')
        for attempt in range(2):
            started=time.time()
            code=run_guarded(command,log,ARENA_MAX_RUNTIME_SECONDS,environment)
            attempts.append(dict(started_at=started,finished_at=time.time(),exit_code=code,
                host=socket.gethostname(),device=device,
                environment={key:environment.get(key) for key in provenance_keys}))
            write_json_atomic(attempt_path,attempts)
            if code==0 or device!='cuda' or attempt==1:break
            error=log.read_text(errors='replace')
            if not any(marker in error for marker in ('InductorError','CUDA capture was unavailable',
                    '_foreach_','expected Tensor as element','No space left on device')):break
            # Retry only an execution failure. The previous child has exited;
            # keep the single-GPU lease and resume its last complete epoch.
            log.replace(directory/f'{name}.{device}.attempt-{len(attempts)}.log')
            environment['OPENDPD_ARENA_PARALLEL_SEEDS']='0'
            environment['OPENDPD_DISABLE_ARENA_COMPILE']='1'
            # Evidence for compiled replay does not qualify a native retry.
            environment['OPENDPD_DISABLE_CUDA_FAST_PATH']='1'
        if code==0:
            verify(root,unit)
            for folder in folders(root,unit):
                record=read_json(folder/'training.json')
                if str(folder) not in prior:
                    record['training_host']=socket.gethostname()
                    record['execution_environment']={key:environment.get(key) for key in provenance_keys}
                    if profile is not None:
                        record['execution_cpu_affinity']=sorted(os.sched_getaffinity(0))
                        record['execution_profile']=profile
                    if unit[0]=='apnrru':
                        record['execution_environment']['inductor_max_fusion_size']=8
                    record['execution_attempts']=attempts
                write_json_atomic(folder/'training.json',record)
        return code if code is not None else 124
    finally:
        release(lease,hold)
        release(gpu_lease,gpu_hold)


def gpu_worker_active(root):
    cache=str(root/'cache')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            if entry.stat().st_uid!=os.getuid():continue
            argv=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        if ('opendpd.core.arena_runner' in argv and '--prefetch' in argv
                and any(argv[i:i+2]==['--cache',cache] for i in range(len(argv)))
                and any(argv[i:i+2]==['--device','cuda'] for i in range(len(argv)))):
            return True
    return False


def ssh_command(host,remote_root,remote_python,unit,device,expected):
    root=Path(remote_root)
    command=['nice','-n','10','env',f'PYTHONPATH={root/"repo"}',
        'OMP_WAIT_POLICY=PASSIVE','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
        remote_python,'-m','benchmark.run_arena_distributed','--workspace',str(root/'work'),
        '--unit',*[str(x) for x in unit],'--device',device,'--expected-training-sha256',expected]
    script=f'cd {shlex.quote(str(root/"repo"))}\nexec {shlex.join(command)}'
    return ['ssh','-o','BatchMode=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6',
            host,script]


def coordinate(args):
    root=args.workspace.resolve();(root/'prefetch').mkdir(parents=True,exist_ok=True)
    expected=arena.training_fingerprint()
    planned=units();remaining=[]
    recovery_hosts=read_json(root/'recovery-hosts.json') if not args.local_only and (root/'recovery-hosts.json').exists() else {}
    for unit in planned:
        done=root/'prefetch'/('--'.join(map(str,unit))+'.done')
        if done.exists() and done.read_text()==expected:
            verify(root,unit)
        else: remaining.append(unit)
    resuming={unit:resume_device(root,unit) for unit in remaining}
    state=dict(training_sha256=expected,started_at=time.time(),expected_units=len(planned),
               complete_units=len(planned)-len(remaining),active={},failed=[])
    write_json_atomic(root/'protocol.json',arena.protocol())
    write_json_atomic(root/'distributed-plan.json',dict(training_sha256=expected,units=planned,
        remote=args.remote,remote_root=args.remote_root,local_cpu_workers=args.local_cpu_workers,
        local_gpu_workers=args.local_gpu_workers,remote_cpu_workers=args.remote_cpu_workers,
        remote_gpu_workers=args.remote_gpu_workers,local_only=args.local_only,test_access=False))
    lock=threading.Lock()

    def update():write_json_atomic(root/'distributed-progress.json',state)
    def take(slot,device):
        with lock:
            eligible=[u for u in remaining if u[0] in arena.DETERMINISTIC
                      or (u[0] in CPU_DENSE and u[2]!='dpa-160mhz')]
            if args.local_only and device=='cpu':
                # Let local CPUs cover short records while the GPU works on
                # long recurrent fits. Preserve both CPU and CUDA resumes.
                eligible=[u for u in remaining if resuming[u]=='cpu' or
                          (u[2]!='dpa-160mhz' and u[0] not in ('pgjanet','dvrjanet','apnrru'))]
            if device=='cuda':eligible=[u for u in remaining if u[0] not in CPU_ONLY]
            eligible=[u for u in eligible if resuming[u] in (None,device)]
            eligible=[u for u in eligible if recovery_hosts.get('--'.join(map(str,u)),slot.split('-')[0])==slot.split('-')[0]]
            if not eligible:return None
            # Long units first. The CPU pool handles explicit Python cells and LS;
            # GPU workers keep the other architectures ahead of CPU-capable work.
            unit=min(eligible,key=lambda u:(u not in [('tres_gru',1000,'apa-200mhz'),('tres_gru',1000,'apa-200mhz-b')],
                u[0] in CPU_DENSE or u[0] in arena.DETERMINISTIC if device=='cuda' else False,
                -arena.calibration()[u[2]]['counts']['train'],
                -(GPU_SECONDS if device=='cuda' else CPU_SECONDS).get(u[0],1),-u[1]))
            remaining.remove(unit);state['active'][slot]=dict(unit=unit,started_at=time.time())
            update();return unit

    def work(slot,remote,device):
        while (unit:=take(slot,device)) is not None:
            name='--'.join(map(str,unit));started=time.monotonic()
            status='ok';error=None
            try:
                if arena.training_fingerprint()!=expected:
                    raise ValueError('Scientific sources changed during training')
                command=(ssh_command(args.remote,args.remote_root,args.remote_python,unit,device,expected)
                    if remote else [sys.executable,'-m','benchmark.run_arena_distributed',
                        '--workspace',str(root),'--unit',*[str(x) for x in unit],'--device',device,
                        '--expected-training-sha256',expected])
                # A coordinator may restart while healthy local children are
                # still fitting. Wait for them and reuse their checkpoints;
                # an occupied lease is not a failed scientific experiment.
                lease_deadline=time.monotonic()+ARENA_MAX_RUNTIME_SECONDS
                while True:
                    if not remote and foreign_worker([str(x) for x in unit]):
                        code=75
                    else:
                        code=run_guarded(command,root/'prefetch'/f'{name}.{slot}.log',ARENA_MAX_RUNTIME_SECONDS+30,
                            dict(os.environ,OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
                    if code!=75:break
                    if time.monotonic()>=lease_deadline:
                        raise TimeoutError(f'Worker lease remained occupied: {unit}')
                    time.sleep(30)
                if code!=0:raise RuntimeError(f'Worker exit code {code}')
                if remote:
                    first=folders(root,unit)[0]
                    relative=first.parent.relative_to(root/'cache')
                    first.parent.mkdir(parents=True,exist_ok=True)
                    origin=f'{args.remote}:{Path(args.remote_root)/"work/cache"/relative}/'
                    subprocess.run(['rsync','-a','--protect-args','--include=*/','--include=training.json',
                        '--include=weights.npz','--include=history.json','--exclude=*',origin,str(first.parent)+'/'],check=True)
                verify(root,unit)
                (root/'prefetch'/f'{name}.done').write_text(expected)
            except Exception as exc:
                status='failed';error=f'{type(exc).__name__}: {exc}'
            with lock:
                state['active'].pop(slot)
                if status=='ok':state['complete_units']+=1
                else:state['failed'].append(dict(unit=unit,error=error))
                update()
                print(json.dumps(dict(unit=unit,slot=slot,status=status,error=error,
                    seconds=round(time.monotonic()-started,1),complete=state['complete_units'],
                    expected=len(planned))),flush=True)
    slots=[(f'{host}-{device}-{i}',host=='remote',device)
           for host,device,count in [('local','cpu',args.local_cpu_workers),('local','cuda',args.local_gpu_workers),
                                    ('remote','cpu',args.remote_cpu_workers),('remote','cuda',args.remote_gpu_workers)]
           for i in range(count)]
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        tasks=[pool.submit(work,*slot) for slot in slots]
        for task in tasks:task.result()
    if remaining or state['failed']:
        raise RuntimeError(f'Training remains incomplete: {len(remaining)} unassigned, {len(state["failed"])} failed; test phase not started')
    freeze(root,planned)
    state['finished_at']=time.time();state['all_checkpoints_frozen_before_test']=True;update()
    print('All training checkpoints verified and frozen. Ready for local test evaluation.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--remote');p.add_argument('--remote-root');p.add_argument('--remote-python')
    p.add_argument('--local-only',action='store_true',help='Run all remaining work locally; never launch SSH workers')
    p.add_argument('--local-cpu-workers',type=int,default=6);p.add_argument('--local-gpu-workers',type=int,default=1)
    p.add_argument('--remote-cpu-workers',type=int,default=4);p.add_argument('--remote-gpu-workers',type=int,default=0)
    p.add_argument('--unit',nargs=3);p.add_argument('--device',choices=['cpu','cuda'])
    p.add_argument('--expected-training-sha256')
    p.add_argument('--finalize',action='store_true',help='After all fits, evaluate, audit and update local Arena artifacts')
    args=p.parse_args()
    if args.unit:
        key,budget,condition=args.unit
        unit=(key,int(budget),condition)
        if unit not in units() or args.device is None or not args.expected_training_sha256:
            p.error('Worker requires a registered unit, device and frozen training SHA256')
        raise SystemExit(worker(args.workspace,unit,args.device,args.expected_training_sha256))
    if args.local_only:
        args.remote_cpu_workers=args.remote_gpu_workers=0
        args.remote=args.remote_root=args.remote_python=None
    if (args.remote_cpu_workers or args.remote_gpu_workers) and not all((args.remote,args.remote_root,args.remote_python)):
        p.error('Coordinator requires --remote, --remote-root and --remote-python')
    counts=(args.local_cpu_workers,args.local_gpu_workers,args.remote_cpu_workers,args.remote_gpu_workers)
    if min(counts)<0 or sum(counts)==0 or args.remote_gpu_workers>1:
        p.error('Worker counts must be nonnegative, nonzero in total, with at most one remote GPU process')
    coordinate(args)
    if args.finalize:
        from benchmark.finalize_arena_reference import complete
        complete(args.workspace)


if __name__=='__main__':main()
