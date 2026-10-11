"""Run the preregistered, configuration-matched LTE v2 cross-validation.

Preserves the original v1 harness/record. Every score comes from actual product
profile output and the independent MATLAB implementation; no refusal is a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import matlab_parity as original

REGISTRATION_COMMIT = "8a8c77f594703da8cccf747cda18c905b109a491"
PROFILE_IDS = {"aligned": "ofdm-lte20-evm-v2", "default": "ofdm-lte20-evm-auto-v2"}
SIGNALS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S6b", "S9b")
ACLR_SIGNALS = ("S2", "S3", "S4", "S5", "S6", "S7", "S8")
BUDGETS = {"aclr_db": 0.1, "evm_percentage_points": 0.05, "waveform_relative_rms": 1e-6, "frequency_offset_hz": 0.01}


def expected_cases():
    cases = {("P1", item) for item in ("sample count", "first-sample alignment (peak lag)", "relative RMS difference after one real scale")}
    for sid in ACLR_SIGNALS:
        for size in original.NPERSEG_245 if sid == "S8" else original.NPERSEG_122:
            for name in ("ACLR_L", "ACLR_R"):
                for mode in ("default", "aligned"):
                    cases.add(("P2", f"{sid} {name} nperseg={size} vs comm.ACPR {mode}"))
    for sid in SIGNALS:
        cases.update(("P3", f"{sid} {name}") for name in ("EVM_RMS (percent)", "integer timing"))
    for sid in ("S9", "S9b"):
        cases.update(("P3", f"{sid} frequency offset, {who}") for who in ("OpenDPD", "MATLAB"))
    return cases


def verify_signals(built):
    registered = json.loads((ROOT / "docs/performance/matlab-parity.json").read_text())
    if set(built["signals"]) != set(SIGNALS) or built["package_sha256"] != registered["package_sha256"]:
        raise ValueError("Signal set or reference waveform differs from registration")
    for sid, signal in built["signals"].items():
        old = registered["signals"][sid]
        if (signal["sha256"] != old["sha256"] or signal["fs"] != old["sample_rate_hz"]
                or signal["iq"].shape[0] != old["n_samples"] or signal["nperseg"] != old["nperseg"]):
            raise ValueError(f"Registered signal changed: {sid}")


def product_values(built):
    from opendpd.core.metrics import registry
    from opendpd.core.waveforms import ofdm
    from opendpd.schemas.dataset import SignalSpec

    waveform = ofdm.generate(built["spec"])
    rows = {}
    for sid, sig in built["signals"].items():
        fs = sig["fs"]
        binding = ofdm.bind_input(sig["input_iq"], fs, built["spec"], package_sha256=built["package_sha256"])
        baseband = ofdm.to_baseband_rate(original.to_complex(sig["iq"]), fs)
        demod = ofdm.demodulate(baseband, waveform, frequency_tolerant=True)
        row = {"aclr": {}, "baseband": baseband, "demodulation": None if demod is None else {
            "timing_samples": demod.offset, "frequency_offset_hz": demod.cfo_hz,
            "evm_rms_percent": demod.evm_rms_pct, "n_symbols": demod.n_symbols},
            "correlation": ofdm.synchronize_frequency_tolerant(baseband, waveform)[1]}
        first_evm = None
        for size in sig["nperseg"] or [2048]:
            signal = SignalSpec(sample_rate_hz=fs, bandwidth_hz=18e6, n_sub_ch=1, nperseg=size, waveform=binding)
            row["aclr"][str(size)] = {}
            for mode, profile in PROFILE_IDS.items():
                values = {m.name: m for m in registry.evaluate(profile, sig["iq"], None, signal)}
                evm = (values["EVM_RMS"].value, values["EVM_RMS"].status.value)
                if first_evm is None:
                    first_evm = evm
                elif evm != first_evm:
                    raise ValueError("PSD configuration or nperseg changed EVM")
                if sig["nperseg"]:
                    row["aclr"][str(size)][mode] = {"left_dB": values["ACLR_L"].value,
                        "right_dB": values["ACLR_R"].value, "profile_id": profile}
        row["evm_rms_percent"], row["evm_status"] = first_evm
        if (demod is None) != (first_evm[0] is None) or (demod is not None and abs(first_evm[0] - demod.evm_rms_pct) > 1e-9):
            raise ValueError("Registry and direct demodulator disagree")
        rows[sid] = row
    return rows


def score(group, name, value, reference, budget, *, note="", profile_id=None):
    finite = all(isinstance(v, (float, int)) and math.isfinite(v) for v in (value, reference))
    difference = reference - value if finite else None
    return {"group": group, "item": name, "kind": "scored", "opendpd": value, "matlab": reference,
            "difference": difference, "budget": budget, "pass": bool(finite and abs(difference) <= budget),
            "note": note if finite else (note + "; missing/non-finite value"), "profile_id": profile_id}


def compare(product, matlab, built):
    if set(product) != set(SIGNALS) or set(matlab.get("signals", {})) != set(SIGNALS):
        raise ValueError("Incomplete or unexpected signal results")
    p = matlab["p1"]
    items = [score("P1", "sample count", p["n_samples_package"], p["n_samples_matlab"], 0),
             score("P1", "first-sample alignment (peak lag)", 0, p["peak_lag_samples"], 0),
             score("P1", "relative RMS difference after one real scale", 0, p["relative_rms_difference"], 1e-6)]
    for sid in SIGNALS:
        own, other = product[sid], matlab["signals"][sid]
        if sid in ACLR_SIGNALS:
            for size in built["signals"][sid]["nperseg"]:
                for mode, pid in PROFILE_IDS.items():
                    ours = own["aclr"][str(size)][mode]
                    if ours["profile_id"] != pid:
                        raise ValueError("Comparison uses the wrong production profile")
                    theirs = other["aclr"]["default"] if mode == "default" else other["aclr"]["aligned"][f"n{size}"]
                    for field, name in (("left_dB", "ACLR_L"), ("right_dB", "ACLR_R")):
                        items.append(score("P2", f"{sid} {name} nperseg={size} vs comm.ACPR {mode}",
                                           ours[field], theirs[field], 0.1, profile_id=pid))
        odm, mdm = own["demodulation"], other["evm"]
        valid = odm is not None and own["evm_status"] == "ok" and mdm["status"] == "ok"
        note = "" if valid else "one or both implementations refused the registered signal"
        items.append(score("P3", f"{sid} EVM_RMS (percent)", own["evm_rms_percent"] if valid else None,
                           mdm.get("evm_rms_percent") if valid else None, 0.05, note=note, profile_id=PROFILE_IDS["aligned"]))
        items.append(score("P3", f"{sid} integer timing", odm["timing_samples"] if valid else None,
                           mdm.get("timing_samples") if valid else None, 0, note=note))
        if sid in ("S9", "S9b"):
            injected = original.INJECTED_CFO_HZ if sid == "S9" else original.CFO_AMENDED_HZ
            for who, record in (("OpenDPD", odm), ("MATLAB", mdm)):
                value = record.get("frequency_offset_hz") if valid else None
                items.append(score("P3", f"{sid} frequency offset, {who}", injected, value, 0.01, note=note))
    return items


def verdict(items):
    actual = [(i["group"], i["item"]) for i in items if i["kind"] == "scored"]
    if len(actual) != 115 or len(set(actual)) != 115 or set(actual) != expected_cases():
        raise ValueError("The complete registered 115-case matrix is required")
    for item in items:
        expected = (0.1 if item["group"] == "P2" else 0.05 if "EVM_RMS" in item["item"] else
                    0.01 if "frequency offset" in item["item"] else 1e-6 if "relative RMS" in item["item"] else 0)
        if item["budget"] != expected:
            raise ValueError("A registered comparison budget changed")
        values = (item["opendpd"], item["matlab"])
        finite = all(isinstance(v, (float, int)) and math.isfinite(v) for v in values)
        difference = values[1] - values[0] if finite else None
        passed = bool(finite and abs(difference) <= expected)
        if item["difference"] != difference or item["pass"] is not passed:
            raise ValueError("Stored verdict differs from the measured values")
        if item["group"] == "P2":
            mode = "default" if item["item"].endswith("default") else "aligned"
            if item.get("profile_id") != PROFILE_IDS[mode]:
                raise ValueError("Scored item names the wrong production configuration")
    failed = [i["item"] for i in items if i["pass"] is not True]
    return {"scored": len(items), "passed": len(items) - len(failed), "failed": len(failed),
            "all_scored_within_budget": not failed, "failed_items": failed}


def render(report):
    v = report["verdict"]
    lines = [f"Run {report['run']} — {report['date']}: **{v['passed']}/{v['scored']} passed**; {v['failed']} failed.",
             "", "This is the separately registered v2 result; it does not change the old v1 verdict.",
             "For frequency-offset rows the first value is the injected offset; the second is the estimate by the tool named in the item.", "",
             "| Group | Scored item | OpenDPD | MATLAB | Difference | Budget | Pass |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    def fmt(x): return "unavailable" if x is None else f"{x:.12g}"
    for item in report["items"]:
        lines.append("| " + " | ".join([item['group'], item['item'], fmt(item['opendpd']), fmt(item['matlab']),
            fmt(item['difference']), fmt(item['budget']), 'yes' if item['pass'] else 'NO']) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument("--matlab-prefdir", type=Path)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--run", default="1")
    parser.add_argument("--date", default="")
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()
    work = args.work.resolve(); work.mkdir(parents=True, exist_ok=True)
    built = original.build_signals(work)
    verify_signals(built)
    product = product_values(built)
    inputs = original.write_matlab_inputs(work, built, product)
    output = work / "matlab-results.json"
    env = dict(os.environ)
    if args.matlab_prefdir:
        args.matlab_prefdir.mkdir(parents=True, exist_ok=True)
        env["MATLAB_PREFDIR"] = str(args.matlab_prefdir.resolve())
    quote = lambda path: str(path).replace("'", "''")
    command = f"addpath('{quote(original.PARITY_DIR)}'); parityLTEv2('{quote(inputs)}','{quote(output)}');"
    run = subprocess.run([args.matlab, "-batch", command], env=env, capture_output=True, text=True, timeout=args.timeout)
    (work / "matlab.log").write_text(run.stdout + "\n" + run.stderr)
    if run.returncode:
        raise RuntimeError(f"Independent MATLAB execution failed; inspect {work / 'matlab.log'}")
    matlab = json.loads(output.read_text())
    items = compare(product, matlab, built)
    environment = original.environment(matlab["environment"])
    paths = ["scripts/matlab_parity_v2.py", "opendpd/core/metrics/ofdm_evm_v2.py", "opendpd/core/metrics/registry.py",
             "opendpd/core/waveforms/ofdm.py", "Matlab/toolbox/examples/parity/parityLTEv2.m",
             "docs/protocols/waveform-profiles-v2.md"]
    environment["v2_source_sha256"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}
    report = {"run": args.run, "date": args.date, "registration_commit": REGISTRATION_COMMIT,
        "profiles": PROFILE_IDS, "environment": environment, "budgets": BUDGETS,
        "package_sha256": built["package_sha256"],
        "signals": {sid: {"sha256": sig["sha256"], "sample_rate_hz": sig["fs"], "n_samples": len(sig["iq"]),
                          "nperseg": sig["nperseg"]} for sid, sig in built["signals"].items()},
        "items": items, "verdict": verdict(items),
        "diagnostics": {
            "note": "Not scored: retain every product value and independent MATLAB diagnostic, including ideal-signal PSD floors and resampler comparisons.",
            "product": {sid: {key: value for key, value in row.items() if key != "baseband"}
                        for sid, row in product.items()},
            "matlab": matlab,
        }}
    target = args.report_json or work / "report.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (work / "report.md").write_text(render(report))
    print(json.dumps(report["verdict"]))
    return 0 if report["verdict"]["all_scored_within_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
