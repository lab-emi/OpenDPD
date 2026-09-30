"""Freeze four measured captures, their disjoint splits and validation-selected PAs.

Every DPD input is an unchanged slice of a packaged measured capture. APA is
repartitioned before fitting so each independently timed carrier has a complete
held-out test symbol. Short validation records select by NMSE, never proxy EVM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from scipy.signal import find_peaks

from datasets.demodulator import Demodulator
from modules.data_collector import load_dataset
from opendpd.core import arena
from opendpd.core.arena_cuda import make_arena_step
from opendpd.core.arena_engine import _load_weights, _save_weights, build_model, pa_output
from opendpd.services.workspace import Workspace, write_json_atomic
from benchmark.prepare_arena_pa import nmse

DATASETS = {"dpa-160mhz": "DPA_160MHz", "dpa-200mhz": "DPA_200MHz",
            "apa-200mhz": "APA_200MHz", "apa-200mhz-b": "APA_200MHz_b"}
CONTEXT = 200


def apa_grid(x, name):
    """Input-only CP synchronization, separately for each independently timed carrier."""
    dm = Demodulator.from_dataset(name)
    signal = x[:, 0].astype(np.float64) + 1j*x[:, 1].astype(np.float64)
    size, cp, carriers = dm.ofdm_nfft, dm.cp_other, []
    # LTE 20 MHz has 1,200 occupied subcarriers (±600, excluding DC).
    # The capture runs at twice the generation rate: 36 MHz occupied per
    # 40 MHz carrier. The old constellation preview's central-600 subset is
    # not used for Arena's full-band data-aided EVM.
    bins = np.r_[np.arange(1, 601), np.arange(size-600, size)].tolist()
    for shift in dm._carrier_f_shifts():
        clean = dm._isolate_channel(signal, dm.fs, shift, dm._bp_half_bins(len(signal)))
        left, right = clean[:-size], clean[size:]
        def moving_sum(value):
            total = np.r_[0, np.cumsum(value)]
            return total[cp:] - total[:-cp]
        correlation = np.abs(moving_sum(left*np.conj(right))) / np.sqrt(
            moving_sum(abs(left)**2)*moving_sum(abs(right)**2) + 1e-30)
        peaks, _ = find_peaks(correlation, height=.5, distance=size//2)
        candidates = [int(p+cp+dm._fine_tune_offset(clean, int(p))) for p in peaks]
        candidates = [p for p in candidates if p >= CONTEXT and p+size <= len(x)-CONTEXT]
        if not candidates:
            raise ValueError(f"No complete held-out OFDM symbol for {name}, carrier {shift}")
        carriers.append(dict(frequency_shift_hz=shift, filter_half_bandwidth_hz=dm.bw_sub_ch/2,
            occupied_bins=bins, fft_starts=[max(candidates)],
            synchronization="last complete input-only CP-synchronized symbol; fixed kurtosis timing refinement"))
    return dict(kind="measured-independent-carrier-ofdm", useful_samples=size, prefix_samples=cp,
                carriers=carriers, n_active_per_carrier=1200)


def dpa_grid(spec):
    size, fs = spec['nperseg'], spec['input_signal_fs']
    active = int(round(spec['bw_sub_ch']/(fs/size)))
    bins = []
    for i in range(spec['n_sub_ch']):
        center = round((i-(spec['n_sub_ch']-1)/2)*spec['bw_sub_ch']/(fs/size))
        bins += [(center+k) % size for k in range(-active//2, active//2+1) if k]
    return dict(kind="measured-ifft-frames", useful_samples=size, prefix_samples=0, occupied_bins=sorted(set(bins)))


def capture(ws, identifier, name, prior):
    dataset = ws.register_builtin_dataset(name, dataset_id=identifier)
    values = load_dataset(dataset_path=ws.dataset_version_dir(identifier))
    keys = ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test")
    original = {key: np.asarray(value, dtype=np.float32) for key, value in zip(keys, values)}
    if not all(np.isfinite(value).all() for value in original.values()):
        raise ValueError(f"Nonfinite capture: {identifier}")
    # Bind the retained complete source capture, and verify it against the
    # packaged original CSV splits before choosing any new boundary.
    old = prior[identifier]
    source_file, source_sha = old['calibration_data_file'], old['calibration_data_sha256']
    if arena.file_hash(arena.ASSETS/source_file) != source_sha:
        raise ValueError("Original source capture hash mismatch")
    with np.load(arena.ASSETS/source_file, allow_pickle=False) as source:
        if any(not np.array_equal(source[key], value) for key, value in original.items()):
            raise ValueError("Retained source capture differs from the measured CSV input/output")
    x = np.concatenate([original[f'x_{split}'] for split in ('train','val','test')])
    y = np.concatenate([original[f'y_{split}'] for split in ('train','val','test')])
    spec = json.loads((Path(__file__).resolve().parents[1]/'datasets'/name/'spec.json').read_text())
    if identifier.startswith('apa-'):
        grid = apa_grid(x, name)
        test_start = min(c['fft_starts'][0] for c in grid['carriers']) - CONTEXT
        val_stop = test_start - CONTEXT
        train_stop = int(.8*(val_stop-CONTEXT))
        boundaries = dict(train=[0,train_stop], val=[train_stop+CONTEXT,val_stop], test=[test_start,len(x)])
        method = 'input-synchronized-heldout-symbol-v1'
    else:
        grid = dpa_grid(spec)
        boundaries = dataset.split.model_dump(mode='json')['boundaries']
        method = 'original-contiguous-60-20-20'
    arrays = {f'{kind}_{split}': values[first:last].copy() for split,(first,last) in boundaries.items()
              for kind,values in (('x',x),('y',y))}
    filename = f'{identifier}-measured-v4.npz'
    np.savez_compressed(arena.ASSETS/filename, **arrays)
    split = dict(method=method, boundaries=boundaries, guard_samples=CONTEXT if identifier.startswith('apa-') else 0)
    digest = arena.file_hash(arena.ASSETS/filename)
    return dict(condition_id=identifier, dataset=name, origin='measured', simulation=None, judges=[],
        data_file=filename, data_sha256=digest, calibration_data_file=filename, calibration_data_sha256=digest,
        source_capture_file=source_file, source_capture_sha256=source_sha, raw_sha256=dataset.raw_sha256,
        original_split=dataset.split.model_dump(mode='json'), split=split, calibration_split=split,
        counts={s:len(arrays[f'x_{s}']) for s in boundaries}, signal=dataset.signal.model_dump(mode='json'),
        evm_grid=grid, stimuli=dict(kind='unchanged original measured capture samples', context_samples=CONTEXT,
            preparation_source_sha256=arena.file_hash(__file__), splits={s:dict(
                source_start=first, source_stop=last, metric_start=CONTEXT, metric_stop=last-first-CONTEXT)
                for s,(first,last) in boundaries.items()}))


def train_pa(meta, folder, seed, device, epochs):
    folder.mkdir(parents=True, exist_ok=True)
    settings = dict(epochs=epochs, frames_per_epoch=512, batch_size=16, frame_length=200,
                    learning_rate=.005, weight_decay=.01, hidden_size=37, seed=seed,
                    initialization="seeded random initialization",
                    selection="minimum whole-validation-split NMSE; test never selects weights or seed")
    binding = dict(training=settings, data_sha256=meta["data_sha256"],
                   preparation_source_sha256=arena.file_hash(__file__),
                   training_sources={p: h for p, h in arena.source_manifest().items() if p in arena.TRAINING_SOURCE_FILES})
    if (folder / "training.json").exists():
        info = json.loads((folder / "training.json").read_text())
        if info["binding"] != binding or info["sha256"] != arena.file_hash(folder / "weights.npz"):
            raise ValueError("PA checkpoint binding differs")
        return info
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model("tres_gru", dict(hidden_size=37, num_layers=1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.5, min_lr=.00005)
    criterion = torch.nn.MSELoss()
    step = make_arena_step(model, criterion, tuple(model.parameters()), 200.) if device == "cuda" else None
    with np.load(arena.ASSETS / meta["data_file"], allow_pickle=False) as data:
        x, y, xv, yv = (data[k] for k in ("x_train", "y_train", "x_val", "y_val"))
    rng, best, history, started = np.random.default_rng(seed), float("inf"), [], time.monotonic()
    for epoch in range(1, epochs + 1):
        starts = rng.choice(len(x) - 199, 512, replace=False)
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
                raise ValueError("Nonfinite PA training loss")
            optimizer.step()
        if epoch % 5 == 0:
            model.eval()
            value = nmse(pa_output(model, xv, device), yv)
            scheduler.step(value)
            history.append(dict(epoch=epoch, validation_nmse_db=value, learning_rate=optimizer.param_groups[0]["lr"]))
            if value < best:
                best, selected = value, epoch
                _save_weights(model, folder / "weights.npz")
            write_json_atomic(folder / "history.json", history)
    info = dict(binding=binding, validation_nmse_db=best, selected_epoch=selected,
                sha256=arena.file_hash(folder / "weights.npz"), parameters=4871,
                seconds=time.monotonic() - started, device=device)
    write_json_atomic(folder / "training.json", info)
    print(json.dumps(dict(condition=meta["condition_id"], seed=seed, **info)), flush=True)
    del step, optimizer, model
    if device == "cuda":
        torch.cuda.empty_cache()
    return info



def prepare(root, device, epochs, prior):
    ws = Workspace.open_or_create(root/'identification')
    chosen, qualification = {}, []
    for identifier, name in DATASETS.items():
        meta = capture(ws, identifier, name, prior)
        candidates = []
        for seed in arena.SEEDS:
            folder = root/'pa'/identifier/f'seed-{seed}'
            info = train_pa(meta, folder, seed, device, epochs)
            candidates.append(dict(info, seed=seed, file=str(folder/'weights.npz'), source='fresh measured-capture PA identification'))
        # Only DPA retains the original partitions. An APA PA previously fit
        # on the old train split has seen part of the new test split and is
        # neither a candidate nor an initialization for this evaluation.
        if identifier.startswith('dpa-'):
            teacher = prior[identifier]['teacher']
            if arena.file_hash(arena.ASSETS/teacher['file']) != teacher['sha256']:
                raise ValueError('Prior PA integrity failure')
            model = _load_weights(build_model('tres_gru', teacher['model']['parameters']), arena.ASSETS/teacher['file']).to(device).eval()
            with np.load(arena.ASSETS/meta['data_file'], allow_pickle=False) as data:
                value = nmse(pa_output(model,data['x_val'],device),data['y_val'])
            candidates.append(dict(validation_nmse_db=value, parameters=teacher['parameters'],seed=teacher['seed'],
                file=str(arena.ASSETS/teacher['file']), sha256=teacher['sha256'],
                source='retained PA on identical original DPA partitions', prior_teacher=teacher))
            del model
        selected = min(candidates,key=lambda item:item['validation_nmse_db'])
        filename = f'{identifier}-pa-v4.npz'
        shutil.copyfile(selected['file'],arena.ASSETS/filename)
        params = selected.get('prior_teacher',{}).get('model',{}).get('parameters',dict(hidden_size=37,num_layers=1))
        model = _load_weights(build_model('tres_gru',params),arena.ASSETS/filename).to(device).eval()
        with np.load(arena.ASSETS/meta['data_file'],allow_pickle=False) as data:
            test_nmse = nmse(pa_output(model,data['x_test'],device),data['y_test'])
            peak = float(np.linalg.norm(data['x_train'],axis=1).max())
            gain = float(np.linalg.norm(data['y_train'],axis=1).max()/peak)
        meta.update(reference_gain=gain,peak_limit=peak)
        meta['teacher'] = dict(file=filename,sha256=arena.file_hash(arena.ASSETS/filename),
            model=dict(key='tres_gru',parameters=params),parameters=selected['parameters'],seed=selected['seed'],
            pa_validation_nmse_db=selected['validation_nmse_db'],pa_test_nmse_db=test_nmse,
            training=selected.get('prior_teacher',{}).get('training', selected.get('binding',{}).get('training')),
            selection='minimum validation NMSE; test input/output never selects weights, seed or model',source=selected['source'])
        chosen[identifier] = meta
        qualification.append(dict(condition=identifier,candidates=candidates,selected=meta['teacher'],test_used_for_selection=False))
        write_json_atomic(root/'pa-selected.json',chosen)
        write_json_atomic(root/'pa-qualification.json',qualification)
        del model
    write_json_atomic(arena.ASSETS/'calibration-v4.json',chosen)
    print('Prepared four entirely measured-input PA/DPD conditions',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--prior-calibration',type=Path,default=arena.ASSETS/'calibration-v3.json')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--epochs',type=int,default=300)
    args=parser.parse_args()
    args.workspace.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    prepare(args.workspace,args.device,args.epochs,json.loads(args.prior_calibration.read_text()))


if __name__ == '__main__':
    main()
