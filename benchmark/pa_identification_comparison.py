"""Reproduce forward PA identification comparisons, without test-set tuning.

ILC is a waveform controller, not a forward PA estimator. This experiment
compares the existing, approximately parameter-matched MP/GMP least-squares
PA estimators with Studio's default TRes-GRU. All runs use the Studio service.
Run prepare first; nn and polynomial can then run in separate processes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from opendpd.core.polynomial import segmented_basis, to_complex
from opendpd.core.virtual_pa import resolve as resolve_pa, simulate
from opendpd.core.waveforms.generator import synthesize
from opendpd.schemas import DatasetOrigin, SignalSpec
from opendpd.schemas.signal_generator import GeneratorConfig
from opendpd.services.datasets import import_dataset
from opendpd.services.experiments import create_run, execute_run, load_artifacts, load_resolved, load_result
from opendpd.services.recipes import instantiate
from opendpd.services.workspace import Workspace, read_json, sha256_file, write_json_atomic

CASES = ("syn-gmp-20mhz-q64", "syn-et-80mhz-q256", "dpa-200mhz", "apa-200mhz")
CUTOFFS = (0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2)


def nmse(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    return float(
        10 * np.log10(max(np.sum(np.abs(prediction - target) ** 2) / np.sum(np.abs(target) ** 2), 1e-30))
    )


def prepare(ws):
    import pandas as pd

    records = []
    for index, (identifier, bw, qam, model_id) in enumerate(
        [
            (CASES[0], 20e6, 64, "generalized-memory"),
            (CASES[1], 80e6, 256, "envelope-tracking"),
        ]
    ):
        config = GeneratorConfig(
            sample_rate_hz=4 * bw,
            bandwidth_hz=bw,
            n_samples=65536,
            channel_modulations=[qam],
            modulation_order=qam,
            seed=20260919 + index,
            rms=0.2,
        )
        _, parameters = resolve_pa(model_id, {})
        if not (ws.dataset_dir(identifier) / "manifest.json").exists():
            x, _ = synthesize(config)
            y, _ = simulate(x, config.sample_rate_hz, model_id, parameters)
            # Independent fixed-amplitude observation noise, not fitted to a
            # model's test result. The polynomial case is a positive control.
            rng = np.random.default_rng(20260929 + index)
            y = y + 0.001 / np.sqrt(2) * (rng.normal(size=len(y)) + 1j * rng.normal(size=len(y)))
            source = ws.root / f"{identifier}.csv"
            pd.DataFrame({"I_in": x.real, "Q_in": x.imag, "I_out": y.real, "Q_out": y.imag}).to_csv(
                source, index=False
            )
            data = import_dataset(
                ws,
                source,
                dataset_id=identifier,
                display_name=f"{identifier} (synthetic)",
                origin=DatasetOrigin.synthetic,
                guard_samples=256,
                signal=SignalSpec(
                    sample_rate_hz=4 * bw,
                    bandwidth_hz=bw,
                    sub_channel_bandwidth_hz=bw,
                    n_sub_ch=1,
                    nperseg=2048,
                    modulation=f"{qam}QAM OFDM (synthetic)",
                    amplitude_units="normalized",
                ),
                notes="Fixed benchmark generator; no physical capture. Independent additive complex noise RMS 0.001.",
            )
            data = data.model_copy(
                update={
                    "simulation": {
                        "virtual_pa": model_id,
                        "parameters": parameters,
                        "generator": config.model_dump(mode="json"),
                        "noise_seed": 20260929 + index,
                        "complex_noise_rms": 0.001,
                    }
                }
            )
            ws.save_dataset(data)
        records.append(ws.get_dataset(identifier).model_dump(mode="json"))
    for name in ("DPA_200MHz", "APA_200MHz"):
        records.append(ws.register_builtin_dataset(name).model_dump(mode="json"))
    write_json_atomic(ws.root / "comparison-data.json", records)
    print(
        json.dumps(
            [
                {"dataset": r["dataset_id"], "samples": r["n_samples"], "split": r["split"]["boundaries"]}
                for r in records
            ]
        ),
        flush=True,
    )


def run(ws, config, tag):
    destination = ws.root / "comparison-runs"
    destination.mkdir(exist_ok=True)
    pointer = destination / f"{tag}.json"
    if pointer.exists():
        item = read_json(pointer)
        if item["status"] == "succeeded":
            return item
    record = create_run(ws, config)
    start = time.monotonic()
    print(f"Start {tag}: {record.run_id}", flush=True)

    def emit(kind, payload):
        # Useful monitoring without retaining large live plot payloads.
        if kind.value in ("epoch", "metric") and payload.get("scope") != "live":
            write_json_atomic(
                destination / f"{tag}-progress.json",
                {"run_id": record.run_id, "kind": kind.value, "payload": payload},
            )

    with (
        (destination / f"{tag}.log").open("w") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        record = execute_run(ws, record.run_id, emit=emit)
    item = {
        "tag": tag,
        "run_id": record.run_id,
        "status": record.status.value,
        "seconds": time.monotonic() - start,
        "error": record.error.model_dump(mode="json") if record.error else None,
    }
    if record.status.value == "succeeded":
        result = load_result(ws, record.run_id)
        item["metrics"] = {m.name: m.value for m in result.metrics}
        item["config"] = load_resolved(ws, record.run_id).model_dump(mode="json")
        artifacts = load_artifacts(ws, record.run_id)
        item["checkpoints"] = [
            a.file.model_dump(mode="json") for a in artifacts.artifacts if a.kind.value == "checkpoint"
        ]
    write_json_atomic(pointer, item)
    print(
        json.dumps(
            {k: item[k] for k in ("tag", "run_id", "status", "seconds")} | {"metrics": item.get("metrics")}
        ),
        flush=True,
    )
    if record.status.value != "succeeded":
        raise RuntimeError(f"{tag}: {record.error}")
    return item


def neural(ws, cases, seeds, epochs):
    for seed in seeds:
        for case in cases:
            config = instantiate("pa-tres_gru-research-v1", case, device="cuda", seed=seed)
            config.name = f"PA comparison · TRes-GRU H27 · {case} · seed {seed}"
            config.training.epochs = epochs
            config.execution.num_threads = 4
            run(ws, config, f"{case}-tres-gru-s{seed}-e{epochs}")


def polynomial(ws, cases):
    from modules.data_collector import load_dataset
    import scipy.linalg

    destination = ws.root / "comparison-runs"
    destination.mkdir(exist_ok=True)
    for case in cases:
        dataset = ws.get_dataset(case)
        # Only train and validation enter selection; the loader also returns
        # the test split, which is ignored until the formal run evaluation.
        x, y, xv, yv, *_ = load_dataset(dataset_path=str(ws.dataset_version_dir(case, "raw-v1")))
        segment = int(dataset.signal.nperseg)
        for key, recipe in (("mp_ls", "pa-mp-ls-v1"), ("gmp_ls", "pa-gmp-ls-v1")):
            config = instantiate(recipe, case, device="cpu")
            params = config.model.parameters
            selection_file = destination / f"{case}-{key}-selection.json"
            if selection_file.exists():
                selection = read_json(selection_file)
            else:
                print(f"Validation-only SVD search: {case} {key}", flush=True)
                start = time.monotonic()
                phi = segmented_basis(key, params, to_complex(x), segment)
                norms = np.linalg.norm(phi, axis=0)
                if np.any(norms == 0):
                    raise ValueError("All-zero polynomial basis columns.")
                phi /= norms
                # One SVD is reused across cutoffs; no normal equations, no
                # test residual and no repeated expensive factorization.
                u, s, vh = scipy.linalg.svd(phi, full_matrices=False, check_finite=True, overwrite_a=True)
                del phi
                projected = u.conj().T @ to_complex(y)
                del u
                val_phi = segmented_basis(key, params, to_complex(xv), segment)
                candidates = []
                for cutoff in CUTOFFS:
                    relative = cutoff or np.finfo(float).eps * max(len(x), len(s))
                    keep = s > s[0] * relative
                    w = (vh[keep].conj().T @ (projected[keep] / s[keep])) / norms
                    prediction = val_phi @ w
                    prediction = prediction.real.astype(np.float32).astype(
                        float
                    ) + 1j * prediction.imag.astype(np.float32).astype(float)
                    candidates.append(
                        {
                            "rcond": cutoff,
                            "rank": int(keep.sum()),
                            "validation_nmse_db": nmse(prediction, to_complex(yv)),
                        }
                    )
                selected = min(candidates, key=lambda r: r["validation_nmse_db"])
                selection = {
                    "dataset": case,
                    "model": key,
                    "basis_parameters": params,
                    "selected": selected,
                    "candidates": candidates,
                    "selection_seconds": time.monotonic() - start,
                    "training_samples": len(x),
                    "validation_samples": len(xv),
                    "coefficients": len(s),
                    "condition_number": float(s[0] / s[-1]),
                    "test_used_for_selection": False,
                }
                write_json_atomic(selection_file, selection)
                del s, vh, val_phi
            # Both the existing benchmark preset and a separately labelled
            # validation-selected fit are evaluated by the ordinary run service.
            values = [("default", float(params["rcond"])), ("val-selected", selection["selected"]["rcond"])]
            for label, cutoff in values:
                cfg = config.model_copy(deep=True)
                cfg.model.parameters["rcond"] = cutoff
                cfg.name = f"PA comparison · {key} {label} · {case}"
                cfg.execution.num_threads = 4
                run(ws, cfg, f"{case}-{key}-{label}")


def report(ws, output):
    import torch
    from opendpd.core.metrics import evaluate
    from opendpd.services.evaluation import predict_test_split, trained_model

    items = []
    for file in sorted((ws.root / "comparison-runs").glob("*.json")):
        data = read_json(file)
        if data.get("status") != "succeeded":
            continue
        run_id = data["run_id"]
        config = load_resolved(ws, run_id)
        with (
            (ws.root / "comparison-runs/report-inference.log").open("a") as log,
            contextlib.redirect_stdout(log),
        ):
            pred = predict_test_split(ws, run_id, config, load_artifacts(ws, run_id))
        # Independent pooled complex-error computation, not an ILC tracking score.
        prediction = pred.prediction.reshape(-1, 2)[: pred.n_valid]
        reference = pred.ground_truth.reshape(-1, 2)[: pred.n_valid]
        actual = nmse(to_complex(prediction), to_complex(reference))
        if abs(actual - data["metrics"]["NMSE"]) > 1e-5:
            raise AssertionError(f'NMSE mismatch for {run_id}: {actual} vs {data["metrics"]["NMSE"]}')
        data["independent_test_nmse_db"] = actual
        dataset = ws.get_dataset(config.dataset.id)
        scores = {m.name: m.value for m in evaluate("opendpd-spectral-v2", reference, None, dataset.signal)}
        data["actual_pa_output_aclr_db"] = {k: v for k, v in scores.items() if k.startswith("ACLR_")}
        data["absolute_aclr_error_db"] = {
            k: abs(data["metrics"][k] - v) for k, v in data["actual_pa_output_aclr_db"].items()
        }
        # Diagnostic only, fixed equally for all models. MP uses 149 past
        # samples; TRes-GRU reads 16 future samples. Never select on this mask.
        indices = np.arange(pred.n_valid)
        segment = int(dataset.signal.nperseg)
        keep = (indices % segment >= 149) & (indices % segment < segment - 16) & (indices < pred.n_valid - 16)
        data["interior_test_nmse_db"] = nmse(to_complex(prediction[keep]), to_complex(reference[keep]))
        data["interior_test_samples"] = int(keep.sum())
        result = load_result(ws, run_id)
        data["model_metadata"] = [m.model_dump(mode="json") for m in result.models]
        expected_parameters = 2751 if config.model.key == "tres_gru" else 2700
        if result.models[0].n_parameters != expected_parameters:
            raise AssertionError(f"Unexpected parameter count for {run_id}")
        if data["tag"].endswith("-val-selected"):
            from modules.data_collector import load_dataset
            from opendpd.services.polynomial import _apply

            xv, yv = load_dataset(dataset_path=str(ws.dataset_version_dir(config.dataset.id, "raw-v1")))[2:4]
            with (
                (ws.root / "comparison-runs/report-inference.log").open("a") as log,
                contextlib.redirect_stdout(log),
            ):
                fitted = trained_model(ws, run_id)
            val_nmse = nmse(to_complex(_apply(fitted.net, xv, segment)), to_complex(yv))
            selection = read_json(
                ws.root / "comparison-runs" / f"{config.dataset.id}-{config.model.key}-selection.json"
            )
            if abs(val_nmse - selection["selected"]["validation_nmse_db"]) > 1e-4:
                raise AssertionError(f"Validation selection does not reproduce for {run_id}")
            data["verified_validation_nmse_db"] = val_nmse
        items.append(data)
    expected = {
        f"{case}-{key}-{label}"
        for case in CASES
        for key in ("mp_ls", "gmp_ls")
        for label in ("default", "val-selected")
    }
    expected |= {f"{case}-tres-gru-s{seed}-e150" for case in CASES for seed in (0, 1, 2)}
    missing = expected - {item["tag"] for item in items}
    if missing:
        raise ValueError("The registered comparison is incomplete: " + ", ".join(sorted(missing)))
    summaries = []
    for case in CASES:
        rows = [item for item in items if item["config"]["dataset"]["id"] == case]
        nn = [item for item in rows if item["config"]["model"]["key"] == "tres_gru"]
        mean = float(np.mean([r["metrics"]["NMSE"] for r in nn]))
        summary = {
            "dataset": case,
            "tres_gru_mean_nmse_db": mean,
            "tres_gru_sd_nmse_db": float(np.std([r["metrics"]["NMSE"] for r in nn], ddof=1)),
            "tres_gru_seeds": [r["config"]["training"]["seed"] for r in nn],
            "tres_gru_mean_seconds": float(np.mean([r["seconds"] for r in nn])),
        }
        for key in ("mp_ls", "gmp_ls"):
            selected = next(r for r in rows if r["tag"] == f"{case}-{key}-val-selected")
            selection = read_json(ws.root / "comparison-runs" / f"{case}-{key}-selection.json")
            summary[key] = {
                "test_nmse_db": selected["metrics"]["NMSE"],
                "rcond": selection["selected"]["rcond"],
                "neural_improvement_db": selected["metrics"]["NMSE"] - mean,
                "neural_error_power_reduction_percent": 100
                * (1 - 10 ** ((mean - selected["metrics"]["NMSE"]) / 10)),
                "fit_seconds": selected["seconds"],
                "selection_seconds": selection["selection_seconds"],
                "selection": selection,
            }
        summaries.append(summary)
    repo = Path(__file__).resolve().parents[1]
    source_files = [
        "benchmark/pa_identification_comparison.py",
        "opendpd/core/polynomial.py",
        "opendpd/core/virtual_pa_kernel.py",
        "opendpd/services/polynomial.py",
        "opendpd/services/recipes.py",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output,
        {
            "protocol": "forward-pa-identification-v1",
            "metric_profile": "opendpd-spectral-v2",
            "definition": "10 log10(sum |PA_prediction - PA_output|^2 / sum |PA_output|^2), valid test samples only",
            "ilc_is_not_a_forward_pa_estimator": True,
            "data": read_json(ws.root / "comparison-data.json"),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "threads": 4,
                "base_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                ).strip(),
                "report_source_sha256": {file: sha256_file(repo / file) for file in source_files},
            },
            "limitations": [
                "Held-out partitions of the same captures, not independent physical acquisitions.",
                "Two synthetic PA structures and two measured 200 MHz captures; no population-wide superiority claim.",
                "Approximately parameter-matched, not compute- or temporal-context-matched: TRes-GRU 16 future samples, GMP 5, MP 0.",
                "All models use Studio segmented zero-state prediction; continuous PA captures retain physical memory across boundaries.",
                "Interior NMSE removes 149 leading and 16 trailing samples per segment equally for all models; diagnostic only.",
                "SVD cutoff chosen on validation only. Existing default-cutoff results are retained separately.",
                "Reported primary metric is forward PA NMSE. Lower predicted-output ACLR alone is not greater modeling accuracy.",
            ],
            "summary": summaries,
            "runs": items,
        },
    )
    print(f"Wrote {output}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "nn", "polynomial", "report"])
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", choices=CASES, default=list(CASES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--output", type=Path, default=Path("benchmark/results/pa-identification-comparison.json"))
    args = p.parse_args()
    ws = Workspace.open_or_create(args.workspace)
    if args.stage == "prepare":
        prepare(ws)
    elif args.stage == "nn":
        neural(ws, args.datasets, args.seeds, args.epochs)
    elif args.stage == "polynomial":
        polynomial(ws, args.datasets)
    else:
        report(ws, args.output)


if __name__ == "__main__":
    main()
