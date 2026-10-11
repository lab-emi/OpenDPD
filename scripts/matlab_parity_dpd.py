"""Maintainer tool: the memory-polynomial DPD parity registered in docs/performance/matlab-parity-dpd.md.

    python scripts/matlab_parity_dpd.py --work /tmp/parity-dpd --matlab ~/MATLAB/R2026a/bin/matlab
    python scripts/matlab_parity_dpd.py --work /tmp/parity-dpd --matlab ... --write-report   # json + the doc's Results block

Stages: build the registered records, compute OpenDPD's ``mp_ls`` values with the compute-core functions the run service
calls, train the product-path runs through the SDK, run ``Matlab/toolbox/examples/parity/parityMP.m`` with
``matlab -batch`` (comm.DPDCoefficientEstimator and comm.DPD), then apply the registered budgets. The budgets below are
copied from the registration and are never adjusted by this script. MATLAB is needed only for this maintainer step.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matlab_parity as lte  # noqa: E402  (shares the item/verdict/format helpers and the signal construction)

DOC = ROOT / "docs" / "performance" / "matlab-parity-dpd.md"
REPORT = ROOT / "docs" / "performance" / "matlab-parity-dpd.json"
PARITY_DIR = ROOT / "Matlab" / "toolbox" / "examples" / "parity"
BEGIN, END = "<!-- parity-dpd-results:begin -->", "<!-- parity-dpd-results:end -->"

# Registered budgets and cases (docs/performance/matlab-parity-dpd.md). Never adjusted by this script.
BUDGET_COEFFICIENT = 1e-6      # Q1, Q2, Q6: relative coefficient error
BUDGET_DOUBLE = 1e-9           # Q3: double against double
BUDGET_OUTPUT = 1e-6           # Q4, Q5: relative RMS output difference
DRIVE = {"S4": 0.35, "S5": 0.50}
CASES = (("C1", "S4", 5, 3), ("C2", "S4", 7, 5), ("C3", "S5", 5, 3), ("C4", "S5", 7, 5))
SEGMENT = 2048                 # nperseg of the boundary-row diagnostic and of the product-path dataset
PRODUCT_PATH = {"record": "S4", "degree": 5, "memory_depth": 3}
SAMPLE_RATE_HZ, BANDWIDTH_HZ = 122.88e6, 20e6       # dataset metadata of the product-path run (not used by the fit)
SURROGATE = {"hidden_size": 6}
TRAINING = {"epochs": 1, "frame_length": 32, "frame_stride": 32, "batch_size": 16, "batch_size_eval": 16}


# --- data --------------------------------------------------------------------------------------------------------

def build_records(work: Path) -> Dict[str, Any]:
    """PA input ``x = a * x_4`` and output ``y`` of the LTE registration's S4 and S5, in double precision."""
    from opendpd.core.waveforms import ofdm
    from opendpd.schemas.waveform import WaveformSpec
    from tests.fixtures.synthetic import memory_polynomial_pa

    package = work / "package"
    ofdm.write_package(ofdm.generate(WaveformSpec(seed=1, n_subframes=10)), package)
    x_iq = np.load(package / "x.npy")
    x30 = x_iq[:, 0].astype(np.float64) + 1j * x_iq[:, 1].astype(np.float64)
    up4 = lte.fft_interpolate(x30, 4)
    records = {}
    for name, drive in DRIVE.items():
        x = drive * up4
        records[name] = {"x": x, "y": memory_polynomial_pa(x)}
    return records


def true_coefficients(degree: int, memory: int) -> np.ndarray:
    """Registered constructed coefficients, flat order ``k * memory + q`` (lag fastest)."""
    c = np.empty(degree * memory, dtype=np.complex128)
    for k in range(degree):
        for q in range(memory):
            c[k * memory + q] = 0.6 ** (k + q) * (-1) ** k * np.exp(1j * (0.35 * (k + 1) + 0.6 * (q + 1)))
    return c


def direct_polynomial(coef: np.ndarray, v: np.ndarray, degree: int, memory: int) -> np.ndarray:
    """``sum c[k,q] v[n-q] |v[n-q]|^k`` by a direct double loop, zero before the start (independent of ``mp_basis``)."""
    out = np.zeros(v.size, dtype=np.complex128)
    for k in range(degree):
        for q in range(memory):
            delayed = np.concatenate([np.zeros(q, dtype=np.complex128), v[:v.size - q]])
            out += coef[k * memory + q] * delayed * np.abs(delayed) ** k
    return out


def relative(a: np.ndarray, b: np.ndarray) -> float:
    """``||a - b|| / ||b||`` (2-norm over the flattened complex vectors)."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def float32_rounded(x: np.ndarray) -> np.ndarray:
    """The values an OpenDPD dataset (or a float32 tensor) holds, as double."""
    return x.real.astype(np.float32).astype(np.float64) + 1j * x.imag.astype(np.float32).astype(np.float64)


# --- OpenDPD side ----------------------------------------------------------------------------------------------------

def fit_complete_memory(v: np.ndarray, target: np.ndarray, degree: int, memory: int):
    """OpenDPD's fit on the rows whose memory is complete: ``fit_least_squares`` on ``mp_basis``, ``rcond`` 0."""
    from opendpd.core.polynomial import fit_least_squares, mp_basis

    phi = mp_basis(v, degree, memory)[memory - 1:]
    w, diag = fit_least_squares(phi, target[memory - 1:], 0.0)
    return w, diag, phi


def residual_db(v: np.ndarray, target: np.ndarray, coef: np.ndarray, degree: int, memory: int) -> float:
    """``20 log10(||r|| / ||Phi w||)`` on the rows with complete memory."""
    from opendpd.core.polynomial import mp_basis

    fitted = mp_basis(v, degree, memory)[memory - 1:] @ np.asarray(coef, dtype=np.complex128).ravel()
    return float(20 * math.log10(np.linalg.norm(target[memory - 1:] - fitted) / np.linalg.norm(fitted)))


def module_output(degree: int, memory: int, w: np.ndarray, x32: np.ndarray) -> np.ndarray:
    """The ``PolynomialModel`` module as the run service applies it: float32 tensor in and out, one segment."""
    import torch
    from opendpd.core.polynomial import PolynomialModel

    model = PolynomialModel("mp_ls", {"K": degree, "Q": memory}, w)
    iq = np.stack([x32.real, x32.imag], axis=-1).astype(np.float32)[None]
    with torch.no_grad():
        out = model(torch.from_numpy(iq))
    out = out[0] if isinstance(out, tuple) else out
    out = out[0].numpy().astype(np.float64)
    return out[:, 0] + 1j * out[:, 1]


def opendpd_stage(records: Dict[str, Any]) -> Dict[str, Any]:
    """Everything OpenDPD contributes to the scored items and diagnostics of every case."""
    from opendpd.core.polynomial import mp_basis, segmented_basis, fit_least_squares

    out: Dict[str, Any] = {}
    for case, record, degree, memory in CASES:
        x, y = records[record]["x"], records[record]["y"]
        gain = float(np.max(np.abs(y)) / np.max(np.abs(x)))            # set_target_gain over the whole record
        v = y / gain
        c_true = true_coefficients(degree, memory)
        x_syn = direct_polynomial(c_true, v, degree, memory)

        w_exact, _, phi = fit_complete_memory(v, x_syn, degree, memory)
        kappa = float(np.linalg.cond(np.linalg.qr(phi, mode="r")))      # unscaled basis, rows with complete memory
        del phi
        w_od, diag, phi = fit_complete_memory(v, x, degree, memory)
        rho_db = float(20 * math.log10(np.linalg.norm(x[memory - 1:] - phi @ w_od) / np.linalg.norm(phi @ w_od)))
        del phi
        u_own = mp_basis(x, degree, memory) @ w_od                     # Q4: the predistorter applied to x in double
        x32 = float32_rounded(x)
        u_module = module_output(degree, memory, w_od, x32)            # Q5: the module through its real interface

        phi_seg = segmented_basis("mp_ls", {"K": degree, "Q": memory}, v, SEGMENT)    # D2: what fit_run builds
        w_seg, _ = fit_least_squares(phi_seg, x, 0.0)
        del phi_seg
        u_seg = segmented_basis("mp_ls", {"K": degree, "Q": memory}, x, SEGMENT) @ w_seg
        u_seg_continuous = mp_basis(x, degree, memory) @ w_seg
        out[case] = {"record": record, "degree": degree, "memory_depth": memory, "gain": gain, "c_true": c_true,
                     "x_syn": x_syn, "w_exact": w_exact, "w_od": w_od, "kappa": kappa, "rho_db": rho_db,
                     "rank": int(diag.rank), "u_own": u_own, "u_module": u_module, "w_seg": w_seg, "u_seg": u_seg,
                     "u_seg_continuous": u_seg_continuous}
    return out


def product_path(work: Path, records: Dict[str, Any]) -> Dict[str, Any]:
    """Train the registered ``mp_ls`` DPD run (and its GRU surrogate) through the SDK and read the DPD run's checkpoint."""
    from modules.data_collector import load_dataset
    from opendpd.core.polynomial import fit_least_squares, mp_basis, segmented_basis, to_complex
    from opendpd.schemas import ArtifactKind
    from opendpd.sdk import open_project
    from opendpd.services.experiments import load_artifacts
    from opendpd.services.legacy_adapter import load_checkpoint
    from opendpd.services.workspace import Workspace
    from utils.util import set_target_gain

    degree, memory = PRODUCT_PATH["degree"], PRODUCT_PATH["memory_depth"]
    x, y = records[PRODUCT_PATH["record"]]["x"], records[PRODUCT_PATH["record"]]["y"]
    iq = lambda z: np.stack([z.real, z.imag], axis=-1).astype(np.float32)
    parameters = {"K": degree, "Q": memory}
    project = open_project(work / "workspace")
    try:
        dataset = project.import_iq(iq(x), iq(y), dataset_id="parity-mp-s4", sample_rate_hz=SAMPLE_RATE_HZ,
                                    bandwidth_hz=BANDWIDTH_HZ, nperseg=SEGMENT, origin="synthetic")
        # The service refuses a least-squares baseline as a DPD surrogate (amendment 1 of the registration). The surrogate
        # only enters the evaluation: the postdistorter is identified on measured data, no gradient runs through it.
        pa = project.train_pa(dataset["dataset_id"], model="gru", parameters=SURROGATE, training=TRAINING,
                              device="cpu").wait(timeout=3600)
        dpd = project.train_dpd(dataset["dataset_id"], pa, model="mp_ls", parameters=parameters, training=TRAINING,
                                device="cpu").wait(timeout=3600)
        ws = Workspace.open(project.workspace)
        entry = load_artifacts(ws, dpd.run_id).by_kind(ArtifactKind.checkpoint)[0]
        state = load_checkpoint(ws.run_dir(dpd.run_id) / entry.file.path)
        checkpoint = state["coefficients"].numpy().astype(np.complex128).ravel()
        x_tr, y_tr = load_dataset(dataset_path=ws.dataset_version_dir(dataset["dataset_id"], "raw-v1"))[:2]
    finally:
        project.close(stop_service=True)

    gain = float(set_target_gain(x_tr, y_tr))
    xc, yc = to_complex(x_tr), to_complex(y_tr)
    w_seg, _ = fit_least_squares(segmented_basis("mp_ls", parameters, yc / gain, SEGMENT), xc, 0.0)
    w_full, _, _ = fit_complete_memory(yc / gain, xc, degree, memory)
    return {"degree": degree, "memory_depth": memory, "gain": gain, "run_id": dpd.run_id, "n_train": int(xc.size),
            "checkpoint": checkpoint, "w_seg": w_seg, "w_full": w_full, "x_tr": xc, "y_tr": yc}


# --- MATLAB side -------------------------------------------------------------------------------------------------------

def write_matlab_inputs(work: Path, records: Dict[str, Any], opendpd: Dict[str, Any], product: Optional[Dict[str, Any]]) -> Path:
    from scipy.io import savemat

    arrays: Dict[str, Any] = {}
    for name, rec in records.items():
        arrays[f"x_{name}"], arrays[f"y_{name}"] = rec["x"].reshape(-1, 1), rec["y"].reshape(-1, 1)
    cases = []
    for case, row in opendpd.items():
        arrays[f"xsyn_{case}"] = row["x_syn"].reshape(-1, 1)
        arrays[f"ctrue_{case}"] = row["c_true"].reshape(-1, 1)
        arrays[f"wod_{case}"] = row["w_od"].reshape(-1, 1)
        cases.append({"id": case, "record": row["record"], "degree": row["degree"], "memory_depth": row["memory_depth"],
                      "gain": row["gain"]})
    description: Dict[str, Any] = {"cases": cases}
    if product is not None:
        arrays["pp_x"], arrays["pp_y"] = product["x_tr"].reshape(-1, 1), product["y_tr"].reshape(-1, 1)
        description["product_path"] = {"degree": product["degree"], "memory_depth": product["memory_depth"],
                                       "gain": product["gain"]}
    path = work / "inputs.mat"
    savemat(path, arrays, do_compression=True)
    path.with_suffix(".json").write_text(json.dumps(description, indent=2))
    return path


def run_matlab(matlab: str, inputs: Path, output: Path, prefdir: Optional[Path], timeout: int) -> None:
    env = dict(os.environ)
    if prefdir is not None:
        prefdir.mkdir(parents=True, exist_ok=True)
        env["MATLAB_PREFDIR"] = str(prefdir)             # never touch a developer's own MATLAB preferences
    command = f"addpath('{PARITY_DIR}'); parityMP('{inputs}', '{output}');"
    done = subprocess.run([matlab, "-batch", command], env=env, capture_output=True, text=True, timeout=timeout)
    (output.parent / "matlab.log").write_text(done.stdout + "\n--- stderr ---\n" + done.stderr)
    if done.returncode != 0 or not output.is_file():
        raise RuntimeError(f"MATLAB failed ({done.returncode}); see {output.parent / 'matlab.log'}\n{done.stdout[-1500:]}")


# --- comparison --------------------------------------------------------------------------------------------------------

def flat(coef: np.ndarray) -> np.ndarray:
    """A MATLAB ``MemoryDepth x Degree`` matrix as OpenDPD's flat vector (lag fastest, then degree)."""
    return np.asarray(coef).reshape(-1, order="F")


def column(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).reshape(-1)


def check(value: float, budget: float) -> bool:
    return bool(math.isfinite(value) and value <= budget)


def compare(records: Dict[str, Any], opendpd: Dict[str, Any], product: Optional[Dict[str, Any]],
            arrays: Dict[str, Any]) -> List[Dict[str, Any]]:
    from scipy.io import loadmat

    m = loadmat(arrays["arrays_file"]) if "arrays_file" in arrays else {}
    items: List[Dict[str, Any]] = []
    add = lambda *a, **k: items.append(lte.item(*a, **k))
    for case, row in opendpd.items():
        K, Q, tag = row["degree"], row["memory_depth"], f"{case} ({row['record']}, K={row['degree']}, Q={row['memory_depth']})"
        x, y = records[row["record"]]["x"], records[row["record"]]["y"]
        v = y / row["gain"]

        e1 = relative(flat(m[f"{case}_coef_exact"]), row["c_true"])
        add("Q1", f"{tag}: MATLAB estimator on the exact problem vs c_true", "scored", None, e1, e1, BUDGET_COEFFICIENT,
            check(e1, BUDGET_COEFFICIENT))
        e2 = relative(row["w_exact"], row["c_true"])
        add("Q2", f"{tag}: OpenDPD fit on the exact problem vs c_true", "scored", e2, None, e2, BUDGET_COEFFICIENT,
            check(e2, BUDGET_COEFFICIENT))
        e3 = relative(column(m[f"{case}_u_true"]), row["x_syn"])
        add("Q3", f"{tag}: comm.DPD(c_true) on y/G vs the constructed input", "scored", None, e3, e3, BUDGET_DOUBLE,
            check(e3, BUDGET_DOUBLE))
        e4 = relative(column(m[f"{case}_u_own"]), row["u_own"])
        add("Q4", f"{tag}: predistorter outputs, each tool with its own coefficients", "scored", None, None, e4,
            BUDGET_OUTPUT, check(e4, BUDGET_OUTPUT))
        e5 = relative(column(m[f"{case}_u_od"]), row["u_module"])
        add("Q5", f"{tag}: the same coefficients applied by comm.DPD and by the PolynomialModel module (float32)", "scored",
            None, None, e5, BUDGET_OUTPUT, check(e5, BUDGET_OUTPUT))

        c_ml = flat(m[f"{case}_coef"])
        d1 = relative(c_ml, row["w_od"])
        add("D1", f"{tag}: coefficients, MATLAB estimator vs OpenDPD (complete-memory rows)", "diagnostic", None, None, d1, None,
            None, f"unscaled condition number {row['kappa']:.3g}; OpenDPD fit residual {row['rho_db']:.2f} dB; rank {row['rank']}")
        r_ml, r_od = residual_db(v, x, c_ml, K, Q), residual_db(v, x, row["w_od"], K, Q)
        r_seg = residual_db(v, x, row["w_seg"], K, Q)
        d2c = relative(c_ml, row["w_seg"])
        d2o = relative(column(m[f"{case}_u_own"]), row["u_seg"])
        d2oc = relative(column(m[f"{case}_u_own"]), row["u_seg_continuous"])
        add("D2", f"{tag}: coefficients, MATLAB estimator vs OpenDPD segmented fit (nperseg {SEGMENT})", "diagnostic", None, None,
            d2c, None, None, f"residual on complete-memory rows (dB): MATLAB {r_ml:.2f}, OpenDPD complete-memory {r_od:.2f}, "
                             f"OpenDPD segmented {r_seg:.2f}")
        add("D2", f"{tag}: predistorter outputs, MATLAB vs OpenDPD segmented fit applied per segment", "diagnostic", None, None,
            d2o, None, None, f"applied continuously instead: {d2oc:.3g}")
        d3 = relative(flat(m[f"{case}_coef_gain20"]), row["w_od"])
        add("D3", f"{tag}: coefficients with DesiredAmplitudeGaindB = 20 log10(G) vs OpenDPD", "diagnostic", None, None, d3,
            None, None)

    if product is None:
        for ident, name in (("Q6", "product-path checkpoint vs the fit recomputed from the run's train split"),
                            ("D4", "MATLAB estimator on the run's train split vs the checkpoint")):
            add(ident, name, "scored" if ident == "Q6" else "diagnostic", None, None, None,
                BUDGET_COEFFICIENT if ident == "Q6" else None, None, "product path not run")
    else:
        q6 = relative(product["checkpoint"], product["w_seg"])
        add("Q6", "C1 data through the SDK: DPD run checkpoint vs the segmented fit recomputed from the run's train split",
            "scored", None, None, q6, BUDGET_COEFFICIENT, check(q6, BUDGET_COEFFICIENT),
            f"run {product['run_id']}, {product['n_train']} train samples, gain {product['gain']:.6g}")
        d4 = relative(flat(m["pp_coef"]), product["checkpoint"])
        add("D4", "MATLAB estimator on the run's train split vs the checkpoint", "diagnostic", None, None, d4, None, None)
        add("D4", "segmented fit minus complete-memory fit on the same split (what the zero-filled rows change)", "diagnostic",
            None, None, relative(product["w_seg"], product["w_full"]), None, None)
    return items


# --- report ----------------------------------------------------------------------------------------------------------------

def environment(matlab_env: Dict[str, Any]) -> Dict[str, Any]:
    import scipy
    import torch

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "opendpd", "scripts", "Matlab"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    scripts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), PARITY_DIR / "parityMP.m")}
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "platform": platform.platform(), "opendpd_commit": commit,
            "working_tree_modified": dirty, "script_sha256": scripts, "matlab": matlab_env}


def render(report: Dict[str, Any]) -> str:
    items, v, env = report["items"], report["verdict"], report["environment"]
    mat = env["matlab"]
    lines = [
        f"Run {report['run']} — {report['date']}. MATLAB {mat['matlab']}; Communications Toolbox "
        f"{mat['Communications_Toolbox']}. Python {env['python']}, NumPy {env['numpy']}, SciPy {env['scipy']}, "
        f"PyTorch {env['torch']}. OpenDPD commit `{env['opendpd_commit'][:12]}`"
        + (" with local modifications" if env["working_tree_modified"] else "") + ".",
        "",
        f"**Scored items: {v['passed']} of {v['scored']} within budget, {v['failed']} outside budget, "
        f"{v['not_evaluable']} not evaluable"
        + (" — the registered memory-polynomial parity passes.**" if v["all_scored_within_budget"]
           else " — the registered memory-polynomial parity does not pass as registered.**"),
        "",
    ]
    if v["failed_items"]:
        lines += ["Outside budget:", ""] + [f"* {t}" for t in v["failed_items"]] + [""]
    if v["not_evaluable_items"]:
        lines += ["Not evaluable:", ""] + [f"* {t}" for t in v["not_evaluable_items"]] + [""]

    def table(rows: List[List[str]], header: List[str]) -> List[str]:
        out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
        return out + ["| " + " | ".join(r) + " |" for r in rows]

    scored = [i for i in items if i["kind"] == "scored"]
    lines += ["### Scored items", ""]
    lines += table([[i["group"], i["item"], lte.fmt(i["difference"], 4), lte.fmt(i["budget"], 3), lte.fmt(i["pass"])]
                    for i in scored], ["id", "item", "difference", "budget", "within"])
    lines += ["", "The difference of Q1–Q6 is relative: `‖a − b‖ / ‖b‖` over the flattened complex vectors.", ""]
    lines += ["### Diagnostics (no budget)", ""]
    lines += table([[i["group"], i["item"], lte.fmt(i["difference"], 4), i["note"] or "—"]
                    for i in items if i["kind"] == "diagnostic"], ["id", "item", "value", "note"])
    cases = report.get("cases")
    if cases:
        lines += ["", "### Cases", ""]
        lines += table([[c, r["record"], str(r["degree"]), str(r["memory_depth"]), f"{r['gain']:.6f}",
                         f"{r['kappa']:.3g}", f"{r['rho_db']:.2f}", str(r["rank"])] for c, r in cases.items()],
                       ["case", "record", "K", "Q", "G", "unscaled condition number", "OpenDPD fit residual (dB)", "rank"])
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
    parser.add_argument("--work", type=Path, required=True, help="scratch folder for data, workspace and MATLAB output")
    parser.add_argument("--matlab", default="matlab", help="path to the MATLAB executable")
    parser.add_argument("--matlab-prefdir", type=Path, default=None, help="isolated MATLAB_PREFDIR (recommended)")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds allowed for the MATLAB run")
    parser.add_argument("--run", default="1", help="run label for the log")
    parser.add_argument("--date", default="", help="date label for the log")
    parser.add_argument("--smoke", type=int, default=0, metavar="N",
                        help="plumbing check on the first N samples of each record; not the registered data, writes no report")
    parser.add_argument("--no-product-path", action="store_true",
                        help="development only: skip the SDK runs; Q6 is then reported as not evaluable")
    parser.add_argument("--write-report", action="store_true", help="write the JSON record and the doc's Results block")
    parser.add_argument("--report-json", type=Path, default=REPORT, help="where the JSON record goes")
    parser.add_argument("--no-doc", action="store_true", help="write the JSON record only; leave the document alone")
    args = parser.parse_args(argv)
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)

    if args.smoke and args.write_report:
        parser.error("--smoke checks the plumbing on data that is not the registered data; it cannot write a report")
    records = build_records(work)
    if args.smoke:
        records = {k: {"x": r["x"][:args.smoke], "y": r["y"][:args.smoke]} for k, r in records.items()}
    opendpd = opendpd_stage(records)
    product = None if args.no_product_path else product_path(work, records)
    inputs = write_matlab_inputs(work, records, opendpd, product)
    output = work / "matlab-results.json"
    run_matlab(args.matlab, inputs, output, args.matlab_prefdir, args.timeout)
    matlab = json.loads(output.read_text())
    items = compare(records, opendpd, product, matlab)
    report = {"run": args.run, "date": args.date, "environment": environment(matlab["environment"]),
              "records": {k: {"n_samples": int(r["x"].size),
                              "sha256_x": hashlib.sha256(r["x"].tobytes()).hexdigest(),
                              "sha256_y": hashlib.sha256(r["y"].tobytes()).hexdigest()} for k, r in records.items()},
              "cases": {c: {k: r[k] for k in ("record", "degree", "memory_depth", "gain", "kappa", "rho_db", "rank")}
                        for c, r in opendpd.items()},
              "budgets": {"coefficient_relative": BUDGET_COEFFICIENT, "double_relative": BUDGET_DOUBLE,
                          "output_relative": BUDGET_OUTPUT},
              "matlab_settings": matlab["settings"], "items": items, "verdict": lte.verdict(items)}
    print(render(report))
    if args.write_report:
        write_report(report, args.report_json, update_doc=not args.no_doc)
    return 0 if report["verdict"]["all_scored_within_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
