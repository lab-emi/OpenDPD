"""Maintainer tool: the OpenDPD <-> MATLAB parity run registered in docs/performance/matlab-parity.md.

    python scripts/matlab_parity.py --work /tmp/parity --matlab ~/MATLAB/R2026a/bin/matlab
    python scripts/matlab_parity.py --work /tmp/parity --matlab ... --write-report   # json + the doc's Results block

Stages: build the registered signals, compute OpenDPD's values through the product path
(``ofdm-lte20-evm-v1``), run ``Matlab/toolbox/examples/parity/parityLTE.m`` with ``matlab -batch``, then apply the
registered budgets. The budgets below are copied from the registration and are never adjusted by this script.
MATLAB is needed only for this maintainer step; users of OpenDPD never need it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DOC = ROOT / "docs" / "performance" / "matlab-parity.md"
REPORT = ROOT / "docs" / "performance" / "matlab-parity.json"
PARITY_DIR = ROOT / "Matlab" / "toolbox" / "examples" / "parity"
BEGIN, END = "<!-- parity-results:begin -->", "<!-- parity-results:end -->"

# Registered budgets (docs/performance/matlab-parity.md; protocol docs/protocols/waveform-profiles.md section 5).
BUDGET_ACLR_DB = 0.1
BUDGET_EVM_PP = 0.05
BUDGET_WAVEFORM = 1e-6
BUDGET_CFO_HZ = 0.01
INJECTED_CFO_HZ = 350.0
CFO_AMENDED_HZ = 30.0          # amendment 1 (docs/performance/matlab-parity.md): inside the timing step's capture range
DRIVE = {"S2": 0.15, "S3": 0.25, "S4": 0.35, "S5": 0.50}
SHIFT_SUBFRAMES = 3
GAIN = 0.8 * np.exp(0.4j)
NOISE_SNR_DB, NOISE_SEED = 30.0, 7
NPERSEG_122 = (1024, 2048, 4096)
NPERSEG_245 = (2048, 4096, 8192)
SAMPLES_PER_SUBFRAME_30 = 30720


# --- signals ---------------------------------------------------------------------------------------------------

def fft_interpolate(x: np.ndarray, factor: int) -> np.ndarray:
    """Band-limited interpolation of a periodic signal by zero padding its spectrum (exact for this waveform)."""
    n = x.size
    spectrum = np.fft.fft(x)
    padded = np.zeros(factor * n, dtype=np.complex128)
    padded[:n // 2] = spectrum[:n // 2]
    padded[-(n // 2):] = spectrum[-(n // 2):]
    return np.fft.ifft(padded) * factor


def capture(y: np.ndarray, fs: float, factor: int, cfo_hz: float = INJECTED_CFO_HZ) -> np.ndarray:
    """Start the capture SHIFT_SUBFRAMES into the waveform, then apply the complex gain and the frequency offset."""
    shifted = np.roll(y, -SHIFT_SUBFRAMES * SAMPLES_PER_SUBFRAME_30 * factor)
    n = np.arange(shifted.size)
    return GAIN * shifted * np.exp(2j * math.pi * cfo_hz / fs * n)


def build_signals(work: Path) -> Dict[str, Any]:
    from opendpd.core.waveforms import ofdm
    from opendpd.schemas.waveform import WaveformSpec
    from tests.fixtures.synthetic import memory_polynomial_pa

    spec = WaveformSpec(seed=1, n_subframes=10)
    package = work / "package"
    ofdm.write_package(ofdm.generate(spec), package)
    x_iq = np.load(package / "x.npy")                                   # float32 (n, 2), as an instrument plays it
    symbols = np.load(package / "symbols.npy").astype(np.complex128)    # complex64 file, as shipped
    x30 = x_iq[:, 0].astype(np.float64) + 1j * x_iq[:, 1].astype(np.float64)
    up4, up8 = fft_interpolate(x30, 4), fft_interpolate(x30, 8)
    fs30, fs122, fs245 = 30.72e6, 122.88e6, 245.76e6

    doubles: Dict[str, np.ndarray] = {"S0": x30, "S1": up4}
    for name, drive in DRIVE.items():
        doubles[name] = memory_polynomial_pa(drive * up4)
    doubles["S6"] = capture(doubles["S4"], fs122, 4)
    rng = np.random.default_rng(NOISE_SEED)
    power = np.mean(np.abs(doubles["S4"]) ** 2) / 10 ** (NOISE_SNR_DB / 10)
    noise = (rng.standard_normal(doubles["S4"].size) + 1j * rng.standard_normal(doubles["S4"].size)) * math.sqrt(power / 2)
    doubles["S7"] = doubles["S4"] + noise
    doubles["S8"] = memory_polynomial_pa(0.35 * up8)
    doubles["S9"] = capture(up4, fs122, 4)
    doubles["S6b"] = capture(doubles["S4"], fs122, 4, CFO_AMENDED_HZ)       # amendment 1
    doubles["S9b"] = capture(up4, fs122, 4, CFO_AMENDED_HZ)

    at122 = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9", "S6b", "S9b")
    rates = {"S0": fs30, "S8": fs245, **{k: fs122 for k in at122}}
    inputs = {"S0": x30, "S8": up8, **{k: up4 for k in at122}}
    nperseg = {k: [] if rates[k] < 58e6 else list(NPERSEG_245 if k == "S8" else NPERSEG_122) for k in doubles}
    signals = {}
    for name, y in doubles.items():
        stored = np.stack([y.real, y.imag], axis=-1).astype(np.float32)        # the precision of an OpenDPD dataset
        signals[name] = {"fs": rates[name], "iq": stored, "nperseg": nperseg[name],
                         "input_iq": np.stack([inputs[name].real, inputs[name].imag], axis=-1).astype(np.float32),
                         "sha256": hashlib.sha256(stored.tobytes()).hexdigest()}
    return {"spec": spec, "package_x": x_iq, "symbols": symbols, "signals": signals,
            "package_sha256": json.loads((package / "waveform.json").read_text())["sha256"]}


def to_complex(iq: np.ndarray) -> np.ndarray:
    return iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)


# --- OpenDPD side ------------------------------------------------------------------------------------------------

def opendpd_values(built: Dict[str, Any]) -> Dict[str, Any]:
    from opendpd.core.metrics import registry
    from opendpd.core.waveforms import ofdm
    from opendpd.schemas.dataset import SignalSpec

    wf = ofdm.generate(built["spec"])
    out: Dict[str, Any] = {}
    for name, sig in built["signals"].items():
        fs = sig["fs"]
        binding = ofdm.bind_input(sig["input_iq"], fs, built["spec"], package_sha256=built["package_sha256"])
        row: Dict[str, Any] = {"aclr": {}}
        baseband = ofdm.to_baseband_rate(to_complex(sig["iq"]), fs)
        demod = ofdm.demodulate(baseband, wf)
        peak = ofdm.synchronize(baseband, wf)[1]
        row["baseband"] = baseband
        row["correlation"] = peak
        if demod is None:
            row["demodulation"] = None
        else:
            row["demodulation"] = {"timing_samples": demod.offset, "frequency_offset_hz": demod.cfo_hz,
                                   "evm_rms_percent": demod.evm_rms_pct, "n_symbols": demod.n_symbols}
        for nperseg in sig["nperseg"] or [2048]:
            signal = SignalSpec(sample_rate_hz=fs, bandwidth_hz=18e6, n_sub_ch=1, nperseg=nperseg, waveform=binding)
            values = {m.name: m for m in registry.evaluate("ofdm-lte20-evm-v1", sig["iq"], None, signal)}
            if "evm_status" not in row:
                row["evm_status"] = values["EVM_RMS"].status.value
                row["evm_rms_percent"] = values["EVM_RMS"].value
            elif row["evm_rms_percent"] != values["EVM_RMS"].value:
                raise RuntimeError(f"{name}: EVM depends on nperseg, which the profile does not allow")
            if sig["nperseg"]:
                row["aclr"][str(nperseg)] = {"left_dB": values["ACLR_L"].value, "right_dB": values["ACLR_R"].value}
        if (demod is None) != (row["evm_rms_percent"] is None) or (
                demod is not None and abs(row["evm_rms_percent"] - demod.evm_rms_pct) > 1e-9):
            raise RuntimeError(f"{name}: the profile's EVM differs from demodulate()")
        out[name] = row
    return out


def timing_peak_table() -> Dict[str, Any]:
    """Amendment A1-3 (OpenDPD only): normalised timing peak against an injected static carrier offset."""
    from opendpd.core.waveforms import ofdm
    from opendpd.schemas.waveform import WaveformSpec

    offsets = (0, 10, 30, 50, 100, 350, 1000)
    table: Dict[str, Any] = {"offsets_hz": list(offsets)}
    for subframes in (1, 10):
        wf = ofdm.generate(WaveformSpec(seed=1, n_subframes=subframes))
        n = np.arange(wf.x.size)
        table[f"{subframes}_subframes"] = [
            float(ofdm.synchronize(wf.x * np.exp(2j * math.pi * f / ofdm.FS * n), wf)[1]) for f in offsets]
    return table


# --- MATLAB side -------------------------------------------------------------------------------------------------

def write_matlab_inputs(work: Path, built: Dict[str, Any], opendpd: Dict[str, Any]) -> Path:
    from scipy.io import savemat

    arrays: Dict[str, Any] = {"symbols": built["symbols"], "x_package": to_complex(built["package_x"]).reshape(-1, 1)}
    description = []
    for name, sig in built["signals"].items():
        arrays[name] = (sig["iq"][:, 0] + 1j * sig["iq"][:, 1]).astype(np.complex64).reshape(-1, 1)
        if sig["fs"] != 30.72e6:
            arrays[name + "_baseband"] = opendpd[name]["baseband"].reshape(-1, 1)
        description.append({"id": name, "sample_rate_hz": sig["fs"], "nperseg": sig["nperseg"],
                            "toolbox_frequency_offset": name in ("S6", "S6b")})
    path = work / "signals.mat"
    savemat(path, arrays, do_compression=True)
    path.with_suffix(".json").write_text(json.dumps({"signals": description}, indent=2))
    return path


def run_matlab(matlab: str, signals: Path, output: Path, prefdir: Optional[Path], timeout: int) -> None:
    env = dict(os.environ)
    if prefdir is not None:
        prefdir.mkdir(parents=True, exist_ok=True)
        env["MATLAB_PREFDIR"] = str(prefdir)             # never touch a developer's own MATLAB preferences
    command = f"addpath('{PARITY_DIR}'); parityLTE('{signals}', '{output}');"
    done = subprocess.run([matlab, "-batch", command], env=env, capture_output=True, text=True, timeout=timeout)
    (output.parent / "matlab.log").write_text(done.stdout + "\n--- stderr ---\n" + done.stderr)
    if done.returncode != 0 or not output.is_file():
        raise RuntimeError(f"MATLAB failed ({done.returncode}); see {output.parent / 'matlab.log'}\n{done.stdout[-1500:]}")


# --- comparison ----------------------------------------------------------------------------------------------------

def item(group: str, name: str, kind: str, opendpd: Any, matlab: Any, difference: Any, budget: Any, passed: Optional[bool],
         note: str = "") -> Dict[str, Any]:
    return {"group": group, "item": name, "kind": kind, "opendpd": opendpd, "matlab": matlab, "difference": difference,
            "budget": budget, "pass": passed, "note": note}


def compare(opendpd: Dict[str, Any], matlab: Dict[str, Any], built: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    p1 = matlab["p1"]
    items.append(item("P1", "sample count", "scored", p1["n_samples_package"], p1["n_samples_matlab"],
                      p1["n_samples_matlab"] - p1["n_samples_package"], "exact",
                      p1["n_samples_matlab"] == p1["n_samples_package"]))
    items.append(item("P1", "first-sample alignment (peak lag)", "scored", 0, p1["peak_lag_samples"], p1["peak_lag_samples"],
                      "exact", p1["peak_lag_samples"] == 0))
    items.append(item("P1", "relative RMS difference after one real scale", "scored", None, p1["relative_rms_difference"],
                      p1["relative_rms_difference"], f"<= {BUDGET_WAVEFORM:g}", p1["relative_rms_difference"] <= BUDGET_WAVEFORM))
    items.append(item("P1", "scale factor", "diagnostic", None, p1["scale"], None, None, None))

    for sid, mrow in matlab["signals"].items():
        orow, sig = opendpd[sid], built["signals"][sid]
        scored_aclr = sid in ("S2", "S3", "S4", "S5", "S6", "S7", "S8")
        if "aclr" in mrow:
            for nperseg in sig["nperseg"]:
                key = f"n{nperseg}"
                for side, label in (("left_dB", "ACLR_L"), ("right_dB", "ACLR_R")):
                    reference = orow["aclr"][str(nperseg)][side]
                    replica = mrow["aclr"]["enclosing_rule_replica"][key][side]
                    items.append(item("P2", f"{sid} {label} nperseg={nperseg} vs enclosing-rule replica of comm.ACPR aligned",
                                      "diagnostic", mrow["aclr"]["aligned"][key][side], replica,
                                      replica - mrow["aclr"]["aligned"][key][side], None, None))
                    for mode, values in (("comm.ACPR default", mrow["aclr"]["default"]),
                                         ("comm.ACPR aligned", mrow["aclr"]["aligned"][key]),
                                         ("pwelch (OpenDPD settings)", mrow["aclr"]["pwelch"][key])):
                        diff = values[side] - reference
                        kind = "scored" if scored_aclr and not mode.startswith("pwelch") else "diagnostic"
                        passed = abs(diff) <= BUDGET_ACLR_DB if kind == "scored" else None
                        items.append(item("P2", f"{sid} {label} nperseg={nperseg} vs {mode}", kind, reference, values[side],
                                          diff, f"<= {BUDGET_ACLR_DB} dB" if kind == "scored" else None, passed))
        evm_d, mev = orow["evm_rms_percent"], mrow["evm"]
        refusal = None
        if orow["demodulation"] is None or mev["status"] != "ok":
            refusal = (f"not evaluable: OpenDPD {orow['evm_status']} (normalised peak {orow['correlation']:.3f}); "
                       f"MATLAB chain {mev['status']} (normalised peak {mev['correlation']:.3f}); "
                       + ("the two tools agree on the refusal" if (orow["demodulation"] is None) == (mev["status"] != "ok")
                          else "THE TWO TOOLS DISAGREE ON WHETHER THE SIGNAL SYNCHRONISES"))
        if refusal:
            for name_, budget in ((f"{sid} EVM_RMS (percent)", f"<= {BUDGET_EVM_PP} points"), (f"{sid} integer timing", "exact")):
                items.append(item("P3", name_, "scored", None, None, None, budget, None, refusal))
            if sid.startswith("S9"):
                for who in ("OpenDPD", "MATLAB"):
                    items.append(item("P3", f"{sid} frequency offset, {who}", "scored",
                                      INJECTED_CFO_HZ if sid == "S9" else CFO_AMENDED_HZ, None, None,
                                      f"<= {BUDGET_CFO_HZ} Hz of the injected value", None, refusal))
            continue
        evm_m = mev["evm_rms_percent"]
        items.append(item("P3", f"{sid} EVM_RMS (percent)", "scored", evm_d, evm_m, evm_m - evm_d,
                          f"<= {BUDGET_EVM_PP} points", abs(evm_m - evm_d) <= BUDGET_EVM_PP))
        items.append(item("P3", f"{sid} integer timing", "scored", orow["demodulation"]["timing_samples"],
                          mev["timing_samples"], mev["timing_samples"] - orow["demodulation"]["timing_samples"],
                          "exact", mev["timing_samples"] == orow["demodulation"]["timing_samples"]))
        if "evm_on_opendpd_baseband" in mrow:
            alt = mrow["evm_on_opendpd_baseband"]["evm_rms_percent"]
            items.append(item("P3", f"{sid} EVM_RMS, MATLAB chain on OpenDPD's baseband (isolates the resamplers)",
                              "diagnostic", evm_d, alt, alt - evm_d, None, None))
            items.append(item("P3", f"{sid} relative difference of the two rate converters", "diagnostic", None,
                              mrow["resampler_relative_difference"], None, None, None))
        if sid.startswith("S9"):
            injected = INJECTED_CFO_HZ if sid == "S9" else CFO_AMENDED_HZ
            for who, value in (("OpenDPD", orow["demodulation"]["frequency_offset_hz"]), ("MATLAB", mev["frequency_offset_hz"])):
                items.append(item("P3", f"{sid} frequency offset, {who}", "scored", injected, value, value - injected,
                                  f"<= {BUDGET_CFO_HZ} Hz of the injected value", abs(value - injected) <= BUDGET_CFO_HZ))
        if sid.startswith("S6"):
            d, m = orow["demodulation"]["frequency_offset_hz"], mev["frequency_offset_hz"]
            items.append(item("P3", f"{sid} frequency offset, OpenDPD vs MATLAB estimate", "diagnostic", d, m, m - d, None, None))
            if "frequency_offset_toolbox" in mrow:
                items.append(item("P3", f"{sid} frequency offset, lteFrequencyOffset (cyclic prefix, no guard)", "diagnostic", d,
                                  mrow["frequency_offset_toolbox"], mrow["frequency_offset_toolbox"] - d, None, None))
    return items


def verdict(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [i for i in items if i["kind"] == "scored"]
    unevaluable = [i for i in scored if i["pass"] is None]
    failed = [i for i in scored if i["pass"] is False]
    return {"scored": len(scored), "passed": len(scored) - len(failed) - len(unevaluable), "failed": len(failed),
            "not_evaluable": len(unevaluable), "all_scored_within_budget": not failed and not unevaluable,
            "failed_items": [f"{i['group']} {i['item']}: difference {i['difference']} (budget {i['budget']})" for i in failed],
            "not_evaluable_items": [f"{i['group']} {i['item']}: {i['note']}" for i in unevaluable]}


# --- report --------------------------------------------------------------------------------------------------------

def environment(matlab_env: Dict[str, Any]) -> Dict[str, Any]:
    import scipy

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "opendpd", "scripts", "Matlab"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    scripts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), PARITY_DIR / "parityLTE.m")}
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "opendpd_commit": commit, "working_tree_modified": dirty,
            "script_sha256": scripts, "matlab": matlab_env}


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{value:.{digits}g}"


def render(report: Dict[str, Any]) -> str:
    items, v, env = report["items"], report["verdict"], report["environment"]
    mat = env["matlab"]
    lines = [
        f"Run {report['run']} — {report['date']}. MATLAB {mat['matlab']}; Communications Toolbox "
        f"{mat['Communications_Toolbox']}, LTE Toolbox {mat['LTE_Toolbox']}, Signal Processing Toolbox "
        f"{mat['Signal_Processing_Toolbox']}. Python {env['python']}, NumPy {env['numpy']}, SciPy {env['scipy']}. "
        f"OpenDPD commit `{env['opendpd_commit'][:12]}`"
        + (" with local modifications" if env["working_tree_modified"] else "") + ".",
        "",
        f"**Scored items: {v['passed']} of {v['scored']} within budget, {v['failed']} outside budget, "
        f"{v['not_evaluable']} not evaluable"
        + (" — the registered cross-validation passes.**" if v["all_scored_within_budget"]
           else " — the registered cross-validation does not pass as registered.**"),
        ""]
    if v["failed_items"]:
        lines += ["Outside budget:", ""] + [f"* {line}" for line in v["failed_items"]] + [""]
    if v["not_evaluable_items"]:
        lines += ["Not evaluable (a tool refused the signal, so no difference exists to compare):", ""]
        lines += [f"* {line}" for line in v["not_evaluable_items"]] + [""]

    def table(group: str, kind: str, headers: List[str], rows):
        out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        out += ["| " + " | ".join(row) + " |" for row in rows]
        return out

    p1 = [i for i in items if i["group"] == "P1"]
    lines += ["### P1 — waveform generation", ""]
    lines += table("P1", "", ["item", "kind", "OpenDPD package", "MATLAB", "difference", "budget", "within"],
                   [[i["item"], i["kind"], fmt(i["opendpd"]), fmt(i["matlab"], 6), fmt(i["difference"], 6), i["budget"] or "—",
                     fmt(i["pass"])] for i in p1])
    lines += ["", "### P2 — ACLR against `comm.ACPR`", "",
              "Worst absolute difference over both sides and all `nperseg` values (full list in `matlab-parity.json`); "
              "budget 0.1 dB for scored rows.", ""]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for i in items:
        if i["group"] == "P2":
            sid, _, rest = i["item"].partition(" ")
            mode = rest.split(" vs ")[1]
            groups.setdefault(f"{sid}|{mode}|{i['kind']}", []).append(i)
    rows = []
    for key, members in groups.items():
        sid, mode, kind = key.split("|")
        worst = max(members, key=lambda m: abs(m["difference"]))
        rows.append([sid, mode, kind, str(len(members)), fmt(worst["difference"], 3),
                     "—" if kind != "scored" else fmt(all(m["pass"] for m in members))])
    lines += table("P2", "", ["signal", "MATLAB comparator", "kind", "values compared", "worst difference (dB)", "all within"], rows)
    lines += ["", "### P3 — EVM against the independent MATLAB chain", ""]
    p3 = [i for i in items if i["group"] == "P3"]
    lines += table("P3", "", ["item", "kind", "OpenDPD", "MATLAB", "difference", "budget", "within"],
                   [[i["item"], i["kind"], fmt(i["opendpd"], 6), fmt(i["matlab"], 6), fmt(i["difference"], 4), i["budget"] or "—",
                     fmt(i["pass"])] for i in p3])
    peak = report.get("timing_peak")
    if peak:
        lines += ["", "### A1-3 — timing peak against carrier offset (OpenDPD only)", ""]
        lines += table("A1-3", "", ["waveform"] + [f"{f} Hz" for f in peak["offsets_hz"]],
                       [[name.replace("_", " ")] + [f"{v:.3f}" for v in peak[name]]
                        for name in ("1_subframes", "10_subframes")])
        lines += ["", "The profile reports `missing_reference` below a peak of 0.3."]
    return "\n".join(lines) + "\n"


def write_report(report: Dict[str, Any], json_path: Path, update_doc: bool) -> None:
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not update_doc:
        return
    text = DOC.read_text(encoding="utf-8")
    block = f"{BEGIN}\n{render(report)}{END}"
    if BEGIN in text:
        head, rest = text.split(BEGIN, 1)
        text = head + block + rest.split(END, 1)[1]
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    DOC.write_text(text, encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True, help="scratch folder for signals and MATLAB output")
    parser.add_argument("--matlab", default="matlab", help="path to the MATLAB executable")
    parser.add_argument("--matlab-prefdir", type=Path, default=None, help="isolated MATLAB_PREFDIR (recommended)")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds allowed for the MATLAB run")
    parser.add_argument("--run", default="1", help="run label for the log")
    parser.add_argument("--date", default="", help="date label for the log")
    parser.add_argument("--write-report", action="store_true",
                        help="write the JSON record (default docs/performance/matlab-parity.json) and the Results block")
    parser.add_argument("--report-json", type=Path, default=REPORT, help="where the JSON record goes")
    parser.add_argument("--no-doc", action="store_true", help="write the JSON record only; leave the document alone")
    args = parser.parse_args(argv)
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)

    built = build_signals(work)
    opendpd = opendpd_values(built)
    inputs = write_matlab_inputs(work, built, opendpd)
    output = work / "matlab-results.json"
    run_matlab(args.matlab, inputs, output, args.matlab_prefdir, args.timeout)
    matlab = json.loads(output.read_text())
    items = compare(opendpd, matlab, built)
    report = {"run": args.run, "date": args.date, "environment": environment(matlab["environment"]),
              "package_sha256": built["package_sha256"],
              "signals": {k: {"sample_rate_hz": s["fs"], "n_samples": int(s["iq"].shape[0]), "sha256": s["sha256"],
                              "nperseg": s["nperseg"]} for k, s in built["signals"].items()},
              "budgets": {"aclr_db": BUDGET_ACLR_DB, "evm_percentage_points": BUDGET_EVM_PP, "waveform_relative_rms": BUDGET_WAVEFORM,
                          "frequency_offset_hz": BUDGET_CFO_HZ},
              "items": items, "verdict": verdict(items), "timing_peak": timing_peak_table()}
    print(render(report))
    if args.write_report:
        write_report(report, args.report_json, update_doc=not args.no_doc)
    return 0 if report["verdict"]["all_scored_within_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
