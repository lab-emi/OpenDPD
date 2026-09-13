"""Bounded gain-inverse ILC with a fixed target and monotone backtracking.

A waveform controller, not a forward PA identifier. The plant must reset its
state identically on each call. No measurement or generalization is implied.
Reference framework: Schoukens et al., doi:10.1109/TMTT.2017.2694822.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class ILCResult:
    input: np.ndarray
    output: np.ndarray
    history: list[dict]
    stop_reason: str
    plant_calls: int


def learn(plant: Callable, x, target_gain: float, inverse_gain: complex, *,
          iterations=30, learning_gain=.5, target_nmse_db=-45., peak_limit=1.,
          backtracking_steps=6, min_improvement_db=.001, callback=None, cancel=None):
    x = np.asarray(x, dtype=np.complex128)
    values = [target_gain, inverse_gain.real, inverse_gain.imag, learning_gain, target_nmse_db, peak_limit, min_improvement_db]
    if x.ndim != 1 or not x.size or not np.isfinite(x).all() or not np.isfinite(values).all():
        raise ValueError('ILC requires finite one-dimensional IQ and parameters.')
    if not (1 <= iterations <= 200 and int(iterations) == iterations and 0 <= backtracking_steps <= 10
            and int(backtracking_steps) == backtracking_steps and 0 < learning_gain <= 1.5
            and target_gain > 0 and abs(inverse_gain) > 0 and peak_limit > 0 and min_improvement_db >= 0):
        raise ValueError('ILC parameters are outside their supported bounds.')
    target = target_gain * x
    energy = float(np.vdot(target, target).real)
    if energy <= 0:
        raise ValueError('ILC needs a nonzero target waveform.')
    def clip(u):
        return u * np.minimum(1., peak_limit / np.maximum(np.abs(u), np.finfo(float).tiny))
    calls = 0
    def evaluate(u):
        nonlocal calls
        if cancel:
            cancel()
        out = np.asarray(plant(u), dtype=np.complex128)
        calls += 1
        if out.shape != x.shape or not np.isfinite(out).all():
            raise ValueError('ILC plant returned non-finite or mismatched IQ.')
        error = target - out
        nmse = float(10 * np.log10(max(float(np.vdot(error, error).real) / energy, 1e-30)))
        return out, error, nmse
    u = clip(x.copy())
    y, error, nmse = evaluate(u)
    history = []
    def record(i, step):
        row = {'iteration': i, 'nmse_db': nmse, 'learning_gain': step,
               'peak_abs': float(np.abs(u).max()), 'limited_fraction': float(np.mean(np.abs(u) >= peak_limit * (1 - 1e-10)))}
        history.append(row)
        if callback:
            callback(row)
    record(0, 0.)
    reason = 'iteration_limit'
    for i in range(1, iterations + 1):
        if nmse <= target_nmse_db:
            reason = 'target_reached'
            break
        previous = nmse
        accepted = False
        for backtrack in range(backtracking_steps + 1):
            step = learning_gain * 2. ** -backtrack
            candidate = clip(u + step * inverse_gain * error)
            yp, ep, np_db = evaluate(candidate)
            if np_db < nmse:
                u, y, error, nmse = candidate, yp, ep, np_db
                accepted = True
                break
        if not accepted:
            reason = 'no_improving_step'
            break
        record(i, step)
        if nmse <= target_nmse_db:
            reason = 'target_reached'
            break
        if previous - nmse < min_improvement_db:
            reason = 'improvement_below_tolerance'
            break
    return ILCResult(u, y, history, reason, calls)


def torch_plant(model, segment_length, device='cpu', batch_segments=32):
    """Replay independent, zero-initialized segments; pad only the final segment."""
    import torch
    model = model.to(device).eval()
    def plant(iq):
        n = len(iq)
        pad = (-n) % segment_length
        z = np.pad(iq, (0, pad))
        pairs = np.stack((z.real, z.imag), axis=-1).astype(np.float32).reshape(-1, segment_length, 2)
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(pairs), batch_segments):
                value = model(torch.from_numpy(pairs[start:start+batch_segments]).to(device)).cpu().numpy()
                outputs.append(value)
        flat = np.concatenate(outputs).reshape(-1, 2)[:n]
        return flat[:, 0].astype(np.float64) + 1j * flat[:, 1].astype(np.float64)
    return plant


def options(params):
    return {k: params[k] for k in ('iterations', 'learning_gain', 'target_nmse_db', 'backtracking_steps', 'min_improvement_db')}
