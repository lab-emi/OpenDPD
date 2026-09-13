"""ILC training data and separately labelled test-waveform ideal reference."""
from __future__ import annotations
import numpy as np
from opendpd.core.ilc import learn, options, torch_plant
from opendpd.core.polynomial import to_complex
from opendpd.services.workspace import read_json, write_json_atomic

ILC_STATEMENT = ('ILC + ILA: optimize only the training waveform through the bound PA surrogate; fit '
    'Phi(y_ILC / G) w ~= u_ILC and copy that postdistorter to the predistorter. '
    'The test waveform is never used to fit coefficients. The separate Ideal baseline uses test-waveform feedback; '
    'it is not transferable, globally optimal, or measured hardware evidence.')


def cancellation(run_dir):
    def check():
        if (run_dir / 'CANCEL').exists():
            from opendpd.services.legacy_adapter import RunCancelled
            raise RunCancelled('ILC cancelled between plant evaluations')
    return check


def training_waveform(ws, run_dir, resolved, x, gain, nperseg, on_epoch):
    from opendpd.services.evaluation import trained_model
    p = resolved.model.parameters
    x = to_complex(x)
    from opendpd.services.evaluation import _fitted_peak
    peak_limit = _fitted_peak(ws, resolved.pa_reference.run_id) * p['peak_factor']
    x = x[:int(p['fit_samples'])]
    loaded = trained_model(ws, resolved.pa_reference.run_id)
    plant = torch_plant(loaded.net, nperseg, resolved.execution.device)
    y0 = plant(x)
    g = np.vdot(x, y0) / np.vdot(x, x)
    if not np.isfinite(g) or abs(g) < 1e-8:
        raise ValueError('PA surrogate has near-zero linear gain; train a usable PA model before ILC.')
    inverse = 1 / g
    def progress(row):
        if on_epoch:
            on_epoch({'EPOCH': row['iteration'], 'N_EPOCH': int(p['iterations']) + 1, 'TRAIN_LOSS': row['nmse_db']})
    learned = learn(plant, x, gain, inverse, peak_limit=peak_limit, **options(p),
                    callback=progress, cancel=cancellation(run_dir))
    write_json_atomic(run_dir / 'ilc.json', {
        'version': 'ilc-ila-v1', 'method': ILC_STATEMENT, 'plant_run_id': resolved.pa_reference.run_id,
        'plant_sha256': resolved.pa_reference.checkpoint_sha256,
        'target_gain': gain, 'inverse_gain': [inverse.real, inverse.imag], 'peak_limit': peak_limit,
        'fit_samples': len(x), 'segment_length': nperseg, 'settings': p,
        'training_history': learned.history, 'training_stop_reason': learned.stop_reason,
        'training_plant_calls': learned.plant_calls, 'test_feedback_used_for_fit': False,
    })
    np.savez(run_dir / 'ilc-training.npz', x=x, u=learned.input, y=learned.output)
    return learned.output / gain, learned.input


def ideal_test(ws, run_dir, resolved, plant_model, x, gain, nperseg, *, save=False):
    source = resolved.dpd_reference.run_id if resolved.task.value == 'run_dpd' else run_dir.name
    record = read_json(ws.run_dir(source) / 'ilc.json')
    # Recompute when testing through another surrogate; do not reuse the old plant's inverse.
    plant = torch_plant(plant_model, nperseg, resolved.execution.device)
    inverse = complex(*record['inverse_gain'])
    if resolved.pa_reference.checkpoint_sha256 != record['plant_sha256']:
        from opendpd.services.datasets import load_version_arrays
        arrays = load_version_arrays(ws, resolved.dataset.id, resolved.dataset.preprocessing_version)
        # A different bound surrogate needs its own training-only gain approximation.
        lo, hi = arrays[2].boundaries['train']
        train = to_complex(arrays[0][lo:min(hi, lo + int(resolved.model.parameters['fit_samples']))])
        g = np.vdot(train, plant(train)) / np.vdot(train, train)
        if not np.isfinite(g) or abs(g) < 1e-8:
            raise ValueError('The selected PA surrogate has near-zero gain.')
        inverse = 1 / g
    learned = learn(plant, to_complex(x), gain, inverse, peak_limit=record['peak_limit'],
                    **options(resolved.model.parameters), cancel=cancellation(run_dir))
    evidence = {**record, 'test_history': learned.history, 'test_stop_reason': learned.stop_reason,
        'test_plant_calls': learned.plant_calls, 'test_samples': len(x),
        'test_inverse_gain': [inverse.real, inverse.imag],
        'test_plant_run_id': resolved.pa_reference.run_id, 'test_plant_sha256': resolved.pa_reference.checkpoint_sha256}
    if save:
        write_json_atomic(run_dir / 'ilc-benchmark.json', evidence)
        np.savetxt(run_dir / 'ilc-ideal-test.csv', np.column_stack((x, learned.input.real, learned.input.imag,
                   learned.output.real, learned.output.imag)), delimiter=',', header='I,Q,I_ideal,Q_ideal,I_pa_out,Q_pa_out', comments='')
    out = np.stack((learned.output.real, learned.output.imag), axis=-1).astype(np.float32)
    inp = np.stack((learned.input.real, learned.input.imag), axis=-1).astype(np.float32)
    return out, inp, evidence
