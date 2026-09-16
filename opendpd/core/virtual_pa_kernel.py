"""Portable complex-envelope PA equations used by simulation and dataset exports."""
import math

import numpy as np
from scipy.signal import lfilter


def _relax(target, tau_us, fs):
    b = -np.expm1(-1 / (fs * tau_us * 1e-6))
    return lfilter([b], [1, -(1-b)], target)


def _rapp(x, gain, saturation, smoothness):
    return gain * x / np.power(1 + np.power(gain * np.abs(x) / saturation, 2*smoothness), 1/(2*smoothness))


def simulate_resolved(x, fs, model_id, p):
    """Causal, cold-start complex-envelope output, exactly one output per input."""
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 1 or not len(x) or len(x) > 1_000_000 or not np.isfinite(x).all():
        raise ValueError("Use a finite one-dimensional PA input with at most 1,000,000 samples.")
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("The input sample rate must be positive.")
    r = np.abs(x)
    states = {}
    if model_id == "linear-reference":
        y = p["gain"] * x * np.exp(1j*np.deg2rad(p["phase_deg"]))
    elif model_id == "saleh-twta":
        y = p["gain"]*x / (1+p["compression"]*r*r) * np.exp(1j*p["phase"]*r*r/(1+p["phase_scale"]*r*r))
    elif model_id in ("memory-polynomial", "generalized-memory"):
        u = p["gain"]*x + p["cubic"]*x*r*r + p["quintic"]*x*r**4
        taps = np.arange(1, int(p["depth"])+1)
        weights = p["decay"]**(taps-1)
        weights /= weights.sum()
        phase = np.exp(1j*taps*np.deg2rad(p["memory_phase"]))
        y = lfilter(np.r_[1, p["memory"]*weights*phase], [1], u)
        if model_id == "generalized-memory":
            q = lfilter(np.r_[0, weights], [1], r*r)
            y += p["cross_memory"]*x*q
            states["delayed_power"] = q
    else:
        y = _rapp(x, p["gain"], p["saturation"], p["smoothness"])
        if model_id == "rapp-am-pm":
            y *= np.exp(1j*p["phase"]*r*r / (r*r+p["phase_scale"]**2))
        elif model_id == "gan-trap-thermal":
            power = r*r / (r*r+(p["saturation"]/p["gain"])**2)
            temperature = p["ambient_c"] + p["heating_c"]*_relax(power, p["thermal_us"], fs)
            release = p["release_us"]*np.exp(p["activation_ev"]/8.617333262e-5 *
                (1/(temperature+273.15) - 1/298.15))
            charge_b = -np.expm1(-1/(fs*p["capture_us"]*1e-6))
            release_b = -np.expm1(-1/(fs*release*1e-6))
            trap = np.empty(len(x))
            state = 0.
            for n, target in enumerate(power):
                b = charge_b if target >= state else release_b[n]
                state += b*(target-state)
                trap[n] = state
            supply = 1-p["ir_drop"]*_relax(power, p["bias_us"], fs)
            y *= (1-p["trap_strength"]*trap)*supply * np.exp(
                -p["thermal_gain"]*(temperature-25) + 1j*p["trap_phase"]*trap)
            states = {"trap_occupancy": trap, "effective_temperature_c": temperature, "supply_fraction": supply}
        elif model_id == "doherty-two-path":
            peaker = np.divide(x, r, out=np.zeros_like(x), where=r>0)*np.maximum(r-p["knee"], 0)
            y += p["peaker"]*_rapp(peaker, p["gain"], p["saturation"], p["smoothness"]) * np.exp(1j*np.deg2rad(p["phase_deg"]))
        elif model_id == "envelope-tracking":
            demand = np.minimum(p["gain"]*r/p["saturation"], 1)
            power = r*r/(r*r+(p["saturation"]/p["gain"])**2)
            supply = np.clip(p["supply_floor"] + p["supply_span"]*_relax(demand, p["tracking_us"], fs)
                - p["ir_drop"]*_relax(power, p["bias_us"], fs), p["supply_floor"], 1.5)
            y = _rapp(x, p["gain"], p["saturation"]*supply, p["smoothness"]) * np.exp(1j*p["phase"]*(supply-1))
            states = {"supply_fraction": supply, "envelope_demand": demand}
    if not np.isfinite(y).all() or np.max(np.abs(y)) > np.finfo(np.float32).max:
        raise ValueError("These parameters exceed the finite float32 output range.")
    return y, states

