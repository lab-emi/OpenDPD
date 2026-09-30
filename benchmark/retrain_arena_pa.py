"""Refit the published Arena PA on the existing disjoint measured partitions.

All seeds and datasets finish fitting and validation selection before any test
array is loaded. The measured assets, symbol grids, gains and limits stay fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from opendpd.core import arena, arena_engine as e
from opendpd.core.arena_cuda import make_arena_step
from opendpd.core.arena_metrics import adjacent_error_ratios, validation_metrics
from opendpd.schemas import SignalSpec
from opendpd.services.workspace import write_json_atomic


def train(meta, folder, seed, device):
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(e._asset(meta['data_file'], meta['data_sha256']), allow_pickle=False) as data:
        x, y, xv, yv = (data[key] for key in ('x_train', 'y_train', 'x_val', 'y_val'))
    budget = arena.training_budget(len(x))
    settings = dict(epochs=240, batch_size=64, frame_length=200, frame_stride=1,
        learning_rate=.005, weight_decay=.01, lr_patience=10, lr_factor=.5,
        lr_min=.0001, lr_threshold_db=.01, hidden_size=37, seed=seed,
        selection='minimum whole-validation-split NMSE', **budget)
    binding = dict(settings=settings, data_sha256=meta['data_sha256'],
        preparation_source_sha256=arena.file_hash(__file__),
        sources={p: h for p, h in arena.source_manifest().items() if p in arena.TRAINING_SOURCE_FILES})
    if (folder/'training.json').exists():
        info = json.loads((folder/'training.json').read_text())
        if info['binding'] != binding or arena.file_hash(folder/'weights.npz') != info['sha256']:
            raise ValueError('PA cache does not match training sources/settings')
        return info
    e._seed(seed)
    model = e.build_model('tres_gru', dict(hidden_size=37, num_layers=1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
        factor=.5, min_lr=.0001, threshold=.01, threshold_mode='abs')
    criterion = torch.nn.MSELoss()
    step = make_arena_step(model, criterion, tuple(model.parameters()), 200.) if device == 'cuda' else None
    xx, yy = e.training_frames(x, device), e.training_frames(y, device)
    history, best, selected, first, updates, seconds = [], float('inf'), None, 1, 0, 0.
    rng, draws = np.random.default_rng(seed), hashlib.sha256()
    resume = folder/'resume.pt'
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=True)
        if state['binding'] != binding or state['device'] != device:
            raise ValueError('PA resume request/device differs')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        history, best, selected = state['history'], state['best'], state['selected']
        first, updates, seconds = state['epoch']+1, state['updates'], state['seconds']
        for _ in range(first-1):
            draws.update(rng.permutation(len(xx)).astype('<i8').tobytes())
    started = time.monotonic()
    for epoch in range(first, 241):
        order = rng.permutation(len(xx)).astype('<i8'); draws.update(order.tobytes())
        order = torch.from_numpy(order).to(device)
        losses = []; model.train()
        for offset in range(0, len(xx), 64):
            index = order[offset:offset+64]
            features, target = xx[index], yy[index]
            loss = step(features, target) if step else None
            if loss is None:
                optimizer.zero_grad(set_to_none=step is None)
                loss = criterion(model(features), target); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 200.)
            optimizer.step(); updates += 1
            losses.append(loss.detach()*len(index))
        losses = torch.stack(losses)
        if not bool(torch.isfinite(losses).all()):
            raise FloatingPointError(f'Nonfinite PA loss at epoch {epoch}')
        model.eval()
        value = validation_metrics(e.pa_output(model, xv, device), yv)['nmse_db']
        scheduler.step(value)
        if value < best:
            best, selected = value, epoch
            e._save_weights(model, folder/'weights.npz')
        row = dict(epoch=epoch, loss=float(losses.sum().cpu())/len(xx),
            validation_nmse_db=value, learning_rate=optimizer.param_groups[0]['lr'], optimizer_updates=updates)
        history.append(row)
        write_json_atomic(folder/'history.json', history)
        temporary = resume.with_suffix('.tmp.pt')
        torch.save(dict(binding=binding, device=device, epoch=epoch, model=model.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), history=history,
            best=best, selected=selected, updates=updates, seconds=seconds+time.monotonic()-started), temporary)
        temporary.replace(resume)
        if epoch % 10 == 0:
            print(json.dumps(dict(condition=meta['condition_id'], seed=seed, **row)), flush=True)
    assert updates == budget['optimizer_updates']
    info = dict(binding=binding, device=device, parameters=4871, selected_epoch=selected,
        validation_nmse_db=best, sha256=arena.file_hash(folder/'weights.npz'),
        frame_draw_sha256=draws.hexdigest(), seconds=seconds+time.monotonic()-started,
        improvement_last_30_epochs_db=min(h['validation_nmse_db'] for h in history[:210])-best)
    write_json_atomic(folder/'training.json', info)
    resume.unlink(missing_ok=True)
    return info


def prepare(root, prior, device):
    selected, qualification = {}, []
    for identifier, meta in prior.items():
        candidates = []
        for seed in arena.SEEDS:
            folder = root/'pa'/identifier/f'seed-{seed}'
            info = train(meta, folder, seed, device)
            candidates.append(dict(info, seed=seed, file=str(folder/'weights.npz'), source='full-window PA identification'))
        # The previous PA was fit on these exact partitions; it may compete by
        # validation NMSE, but its published test score never enters selection.
        teacher = meta['teacher']
        model = e.load_frozen(teacher, device)
        with np.load(e._asset(meta['data_file'], meta['data_sha256']), allow_pickle=False) as data:
            xv, yv = data['x_val'], data['y_val']
        value = validation_metrics(e.pa_output(model, xv, device), yv)['nmse_db']
        candidates.append(dict(file=str(arena.ASSETS/teacher['file']), sha256=teacher['sha256'],
            validation_nmse_db=value, seed=teacher['seed'], parameters=teacher['parameters'],
            source='retained PA on identical measured partitions', prior_teacher=teacher))
        winner = min(candidates, key=lambda c:c['validation_nmse_db'])
        selected[identifier] = winner
        qualification.append(dict(condition=identifier, candidates=candidates, selected_sha256=winner['sha256'],
            test_used_for_selection=False))
        write_json_atomic(root/'pa-candidates.json', qualification)
        del model
        if device == 'cuda': torch.cuda.empty_cache()
    # The complete PA selection is now immutable. All subsequent access is evaluation.
    write_json_atomic(root/'pa-frozen.json', {key:{k:v for k,v in value.items() if k!='prior_teacher'}
                                            for key,value in selected.items()})
    calibrated = {}
    for identifier, winner in selected.items():
        meta = json.loads(json.dumps(prior[identifier]))
        filename = f'{identifier}-pa-v6.npz'
        assert arena.file_hash(Path(winner['file'])) == winner['sha256']
        destination = arena.ASSETS/filename
        if Path(winner['file']).resolve() != destination.resolve():
            shutil.copyfile(winner['file'], destination)
        model = e._load_weights(e.build_model('tres_gru',dict(hidden_size=37,num_layers=1)),
                                 arena.ASSETS/filename).to(device).eval()
        with np.load(e._asset(meta['data_file'],meta['data_sha256']),allow_pickle=False) as data:
            xt, yt, xv, yv = (data[k] for k in ('x_test','y_test','x_val','y_val'))
        test_nmse = validation_metrics(e.pa_output(model,xt,device),yt)['nmse_db']
        signal = SignalSpec.model_validate(meta['signal']).model_copy(update={'nperseg':4096})
        residual = adjacent_error_ratios(e.pa_output(model,xv,device)[200:-200],yv[200:-200],signal)
        meta['teacher'] = dict(file=filename, sha256=winner['sha256'],
            model=dict(key='tres_gru',parameters=dict(hidden_size=37,num_layers=1)),
            parameters=4871, seed=winner['seed'], pa_validation_nmse_db=winner['validation_nmse_db'],
            pa_test_nmse_db=test_nmse, validation_residual=residual,
            training=winner.get('prior_teacher',{}).get('training',winner.get('binding',{}).get('settings')),
            selection='minimum validation NMSE; all PA checkpoints and seeds frozen before test evaluation',
            source=winner['source'])
        calibrated[identifier] = meta
        print(json.dumps(dict(condition=identifier, teacher=meta['teacher'])),flush=True)
        del model
    write_json_atomic(root/'pa-selected.json',calibrated)
    write_json_atomic(arena.ASSETS/arena.CALIBRATION_FILE,calibrated)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--prior-calibration',type=Path,default=arena.ASSETS/arena.CALIBRATION_FILE)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args();a.workspace.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    prior = json.loads(a.prior_calibration.read_text())
    conditions = [condition for board in arena.BOARDS for condition in board['conditions']]
    prepare(a.workspace, {condition: prior[condition] for condition in conditions}, a.device)


if __name__ == '__main__':
    main()
