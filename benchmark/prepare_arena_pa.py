"""Select one frozen TRes-GRU per PA on validation data, then prepare DPD stimuli.

PA identification uses the original captures. DPD stimuli are independent,
filtered OFDM records, with complete symbols and a full symbol of context at
each end. Only the middle symbols enter metrics. No hardware is driven.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from opendpd.core import arena
from opendpd.core.arena_cuda import make_arena_step
from opendpd.core.arena_engine import _load_weights, _save_weights, build_model, pa_output
from opendpd.services.workspace import write_json_atomic

PA_TRAINING = dict(epochs=150, frames_per_epoch=512, batch_size=16, frame_length=200,
                   learning_rate=0.001, weight_decay=0.01, hidden_size=37, seeds=[0, 1, 2],
                   initialization="function-preserving expansion of the previous H27 PA; new units seeded independently",
                   selection="minimum whole-validation-split NMSE; test never selects weights or seed")


def nmse(prediction, reference):
    p, r = np.asarray(prediction, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    return float(10 * np.log10(max(np.sum((p-r)**2) / np.sum(r*r), 1e-30)))


def train_pa(identifier, data, folder, seed, device):
    folder.mkdir(parents=True, exist_ok=True)
    binding = dict(training=PA_TRAINING, seed=seed,
                   data_sha256=arena.file_hash(arena.ASSETS / data['data_file']))
    record = folder / 'training.json'
    if record.exists():
        info = json.loads(record.read_text())
        if info['binding'] != binding or info['sha256'] != arena.file_hash(folder / 'weights.npz'):
            raise ValueError('PA training cache does not match its inputs')
        return info
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model('tres_gru', dict(hidden_size=37, num_layers=1)).to(device)
    prior = _load_weights(build_model('tres_gru', data['teacher']['model']['parameters']), arena.ASSETS / data['teacher']['file'])
    old_state, expanded = prior.state_dict(), model.state_dict()
    for key, value in old_state.items():
        if expanded[key].shape == value.shape:
            expanded[key].copy_(value)
        elif 'weight_ih' in key:
            for gate in range(3):
                expanded[key][gate*37:gate*37+27].copy_(value[gate*27:(gate+1)*27])
        elif 'weight_hh' in key:
            for gate in range(3):
                expanded[key][gate*37:gate*37+27, :27].copy_(value[gate*27:(gate+1)*27])
                expanded[key][gate*37:gate*37+27, 27:].zero_()
        elif 'fc_out.weight' in key:
            expanded[key][:, :27].copy_(value)
            expanded[key][:, 27:].zero_()
        else:
            raise ValueError(f'Unreviewed PA expansion tensor: {key}')
    model.load_state_dict(expanded)
    del prior, expanded, old_state
    optimizer = torch.optim.AdamW(model.parameters(), lr=PA_TRAINING['learning_rate'], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.5, min_lr=.00005)
    criterion = torch.nn.MSELoss()
    step = make_arena_step(model, criterion, tuple(model.parameters()), 200.) if device == 'cuda' else None
    with np.load(arena.ASSETS / data['data_file'], allow_pickle=False) as arrays:
        x, y, xv, yv = (np.asarray(arrays[k], dtype=np.float32) for k in ('x_train', 'y_train', 'x_val', 'y_val'))
    rng = np.random.default_rng(seed)
    best, history, started = float('inf'), [], time.monotonic()
    for epoch in range(1, PA_TRAINING['epochs'] + 1):
        starts = rng.choice(len(x)-199, 512, replace=False)
        xx = torch.from_numpy(np.stack([x[i:i+200] for i in starts])).to(device)
        yy = torch.from_numpy(np.stack([y[i:i+200] for i in starts])).to(device)
        model.train()
        for i in range(0, 512, 16):
            loss = step(xx[i:i+16], yy[i:i+16]) if step else None
            if loss is None:
                optimizer.zero_grad(set_to_none=step is None)
                loss = criterion(model(xx[i:i+16]), yy[i:i+16])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 200.)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite PA training loss')
            optimizer.step()
        if epoch % 5 == 0:
            model.eval()
            value = nmse(pa_output(model, xv, device), yv)
            scheduler.step(value)
            history.append(dict(epoch=epoch, validation_nmse_db=value, learning_rate=optimizer.param_groups[0]['lr']))
            if value < best:
                best, selected = value, epoch
                _save_weights(model, folder / 'weights.npz')
            write_json_atomic(folder / 'history.json', history)
    info = dict(binding=binding, validation_nmse_db=best, selected_epoch=selected,
                sha256=arena.file_hash(folder / 'weights.npz'), parameters=4871,
                seconds=time.monotonic()-started, device=device)
    write_json_atomic(record, info)
    print(json.dumps(dict(condition=identifier, seed=seed, **info)), flush=True)
    del step, optimizer, model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return info


def waveform(identifier, meta, split, index):
    """Known-grid OFDM; the reference itself is filtered before it reaches DPD."""
    fs, bw, carriers = (meta['signal'][k] for k in ('sample_rate_hz', 'bandwidth_hz', 'n_sub_ch'))
    fft = 32768 if identifier.startswith('apa') else 2560 if identifier.startswith('dpa') else 1024
    cp = 2304 if identifier.startswith('apa') else fft // 16
    symbols = (4 if split == 'train' else 2) if identifier.startswith('apa') else (24 if split == 'train' else 12)
    qam = 256 if identifier.startswith(('apa', 'syn-et')) else 64
    frequency = np.fft.fftfreq(fft, 1/fs)
    width = bw / carriers
    bands = [[(i-(carriers-1)/2)*width-.45*width, (i-(carriers-1)/2)*width+.45*width] for i in range(carriers)]
    bins = np.flatnonzero(np.any([(frequency >= lo) & (frequency < hi) for lo, hi in bands], axis=0) & (frequency != 0))
    rng = np.random.default_rng(2026092000 + 10*index + {'train': 0, 'val': 1, 'test': 2}[split])
    blocks = []
    side = int(np.sqrt(qam))
    for _ in range(symbols+2):
        grid = np.zeros(fft, dtype=complex)
        grid[bins] = 2*rng.integers(0, side, len(bins))-side+1 + 1j*(2*rng.integers(0, side, len(bins))-side+1)
        block = np.fft.ifft(grid)
        blocks.append(np.r_[block[-cp:], block])
    x = np.concatenate(blocks)
    # Periodic-record raised-cosine low pass: transition .96..1 times half bandwidth.
    f = np.fft.fftfreq(len(x), 1/fs)
    transition = np.clip((np.abs(f)/(bw/2)-.96)/.04, 0, 1)
    x = np.fft.ifft(np.fft.fft(x) * (.5+.5*np.cos(np.pi*transition)))
    with np.load(arena.ASSETS / meta['data_file'], allow_pickle=False) as data:
        original = data['x_train'].astype(np.float64)
    rms = float(np.sqrt(np.mean(np.sum(original**2, axis=1))))
    x *= rms / np.sqrt(np.mean(abs(x)**2))
    # Stimulus scaling preserves spectral cleanliness; limiter acts downstream of DPD.
    peak = float(np.sqrt(np.sum(original**2, axis=1)).max())
    scale = min(1., .98*peak / abs(x).max())
    x *= scale
    spec = dict(fft_size=fft, prefix_samples=cp, symbols=symbols, guard_symbols=1,
                seed=2026092000+10*index+{'train': 0, 'val': 1, 'test': 2}[split],
                modulation_order=qam, rms=float(np.sqrt(np.mean(abs(x)**2))), scale_to_training_peak=scale,
                metric_start=fft+cp, metric_stop=(symbols+1)*(fft+cp))
    grid = dict(useful_samples=fft, prefix_samples=cp, occupied_hz=bands, occupied_bins=bins.tolist())
    return np.column_stack((x.real, x.imag)).astype(np.float32), grid, spec


def prepare(root, device, epochs=150):
    PA_TRAINING['epochs'] = epochs
    torch.set_num_threads(4)
    source = json.loads((arena.ASSETS / 'calibration-v1.json').read_text())
    output, qualification = {}, []
    for index, (identifier, original) in enumerate(sorted(source.items())):
        candidates = []
        for seed in PA_TRAINING['seeds']:
            folder = root / identifier / f'seed-{seed}'
            info = train_pa(identifier, original, folder, seed, device)
            candidates.append(dict(info, file=str(folder / 'weights.npz'), seed=seed, hidden_size=37))
        # A larger model is accepted only if its held-out validation error improves.
        prior = original['teacher']
        old = _load_weights(build_model('tres_gru', prior['model']['parameters']), arena.ASSETS / prior['file']).to(device).eval()
        with np.load(arena.ASSETS / original['data_file'], allow_pickle=False) as data:
            old_val = nmse(pa_output(old, data['x_val'], device), data['y_val'])
        candidates.append(dict(file=str(arena.ASSETS / prior['file']), seed=prior['training']['seed'],
                               hidden_size=27, parameters=2751, validation_nmse_db=old_val, source='previous frozen PA'))
        selected = min(candidates, key=lambda x: x['validation_nmse_db'])
        filename = f'{identifier}-pa-v2.npz'
        shutil.copyfile(selected['file'], arena.ASSETS / filename)
        model = _load_weights(build_model('tres_gru', dict(hidden_size=selected['hidden_size'], num_layers=1)), arena.ASSETS / filename).to(device).eval()
        with np.load(arena.ASSETS / original['data_file'], allow_pickle=False) as data:
            test_nmse = nmse(pa_output(model, data['x_test'], device), data['y_test'])
            peak = float(np.linalg.norm(data['x_train'], axis=1).max())
            gain = float(np.linalg.norm(data['y_train'], axis=1).max()/peak)
        meta = copy.deepcopy(original)
        meta['calibration_data_file'], meta['calibration_data_sha256'] = original['data_file'], original['data_sha256']
        meta['calibration_split'] = original['split']
        meta['teacher'] = dict(file=filename, sha256=arena.file_hash(arena.ASSETS / filename),
            model=dict(key='tres_gru', parameters=dict(hidden_size=selected['hidden_size'], num_layers=1)),
            parameters=selected['parameters'], pa_validation_nmse_db=selected['validation_nmse_db'], pa_test_nmse_db=test_nmse,
            selection=PA_TRAINING['selection'], seed=selected['seed'],
            training=prior['training'] if selected.get('source') == 'previous frozen PA' else PA_TRAINING,
            source='previous frozen PA' if selected.get('source') else 'expanded and fine-tuned TRes-GRU',
            candidate_training=PA_TRAINING)
        meta['judges'] = []
        meta['reference_gain'], meta['peak_limit'] = gain, peak
        arrays, specifications = {}, {}
        for split in ('train', 'val', 'test'):
            x, grid, specifications[split] = waveform(identifier, original, split, index)
            arrays[f'x_{split}'] = x
            arrays[f'y_{split}'] = pa_output(model, x, device)
        data_file = f'{identifier}-stimuli-v2.npz'
        np.savez_compressed(arena.ASSETS / data_file, **arrays)
        meta.update(data_file=data_file, data_sha256=arena.file_hash(arena.ASSETS / data_file), evm_grid=grid,
                    stimuli=dict(kind='independent filtered OFDM simulation inputs', splits=specifications,
                                 generator_source_sha256=arena.file_hash(__file__)),
                    split=dict(boundaries={s: [0, len(arrays[f'x_{s}'])] for s in specifications}, method='independent-generator-seeds'))
        meta['signal'].update(standard=None, waveform=None, modulation=f"{specifications['test']['modulation_order']}QAM OFDM (generated)")
        output[identifier] = meta
        qualification.append(dict(condition=identifier, candidates=candidates, selected=meta['teacher'], test_used_for_selection=False))
        write_json_atomic(root / 'qualification.json', qualification)
        print(json.dumps(dict(condition=identifier, selected=meta['teacher'], stimuli=specifications)), flush=True)
    write_json_atomic(arena.ASSETS / 'calibration-v2.json', output)
    write_json_atomic(root / 'qualification.json', qualification)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    prepare(args.workspace, args.device)
