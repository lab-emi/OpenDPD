"""Independent artifact and score audit of the frozen Arena reference matrix.

Recomputes every gate, cost-adjusted score and ranking value from the raw
cases without calling the production aggregation, verifies checkpoints, frame
draws and validation-only selection, replays streaming entries with a second
EVM demodulator, and checks the analytic operation count against an
instrumented forward pass. --allow-partial
is for development only and never reports a partial matrix as complete. No
fitting or training; final PA replay uses the original judging device rather
than silently substituting CPU kernels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from opendpd.core import arena
from opendpd.core.registry import get_model

METRICS = ("evm_db", "aclr_l_db", "aclr_r_db")      # what the quality is made of; NMSE and the rest are diagnostics
REQUIRED_METRICS = (*METRICS, "nmse_db", "aclr_l_db", "aclr_r_db", "ib_error_db", "power_error_db",
                    "reference_aclr_l_db", "reference_aclr_r_db", "baseline_power_error_db",
                    *("baseline_" + key for key in (*METRICS, "nmse_db", "ib_error_db", "aclr_l_db", "aclr_r_db")))
BUDGETS = (250, 500, 1000, 2000)      # restated here on purpose, not imported from the scorer
MEASURED_BOARDS = {"apa-200mhz-b": "APA_200MHz_b"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(a, b, tolerance=1e-8):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=tolerance)


def stdev(values):
    if len(values) == 1:
        return 0.
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((x - mean) ** 2 for x in values) / (len(values) - 1))


def independent_budget(point, cases, seeds, power_tolerance):
    """One sweep point. Deliberately does not call arena.summarize_cases or its helpers."""
    qualities, observations, gates = [], [], True
    for seed in seeds:
        gains = []
        for case in (c for c in cases if c["seed"] == seed):
            for judge in case["judges"]:
                in_band = judge["baseline_evm_db"] - judge["evm_db"]
                worse_before = max(judge["baseline_aclr_l_db"], judge["baseline_aclr_r_db"])
                worse_after = max(judge["aclr_l_db"], judge["aclr_r_db"])
                gains.append((in_band + (worse_before - worse_after)) / 2)      # in band and out of band weigh the same
                gates &= abs(judge["power_error_db"]) <= power_tolerance       # the only gate
                observations.append(judge)
        qualities.append(math.fsum(gains) / len(gains))
    mean_quality = math.fsum(qualities) / len(qualities)
    conservative = mean_quality - stdev(qualities)
    parameter_cost = 10 * math.log10(point["parameters"] / 1000)
    operation_cost = 10 * math.log10((point["mul"] + point["add"]) / 2000)
    mean = lambda key: math.fsum(j[key] for j in observations) / len(observations)
    worst = lambda prefix, name: math.fsum(max(j[prefix + name + "_l_db"], j[prefix + name + "_r_db"])
                                          for j in observations) / len(observations)
    return dict(qualified=bool(gates), quality_db=mean_quality, quality_std_db=stdev(qualities),
        quality_conservative_db=conservative, parameter_efficiency_db=mean_quality - parameter_cost,
        arithmetic_efficiency_db=mean_quality - operation_cost,
        score=mean_quality - parameter_cost / 2 - operation_cost / 2,
        nmse_improvement_db=mean("baseline_nmse_db") - mean("nmse_db"),
        evm_improvement_db=mean("baseline_evm_db") - mean("evm_db"),
        aclr_improvement_db=worst("baseline_", "aclr") - worst("", "aclr"),
        aer_improvement_db=worst("baseline_", "aer") - worst("", "aer"))


def independent_score(row, power_tolerance=.5):
    points = {}
    for point in row["budgets"]:
        if point["available"]:
            points[point["budget"]] = independent_budget(point, [c for c in row["cases"] if c["budget"] == point["budget"]],
                row["seeds"], power_tolerance)
    qualified = {budget: value for budget, value in points.items() if value["qualified"] and value["quality_db"] > 0}
    costs = {p["budget"]: p["parameters"] for p in row["budgets"] if p["available"]}
    rankings = {}
    if qualified:
        best = lambda field: max(v[field] for v in qualified.values())
        rankings = dict(overall=best("score"), parameter_efficiency=best("parameter_efficiency_db"),
            arithmetic_efficiency=best("arithmetic_efficiency_db"), linearization=best("quality_db"),
            evm=best("evm_improvement_db"), aclr=best("aclr_improvement_db"),
            **{f"budget-{b}": max((v["score"] for k, v in qualified.items() if costs[k] <= b), default=None) for b in BUDGETS})
    return dict(eligible=bool(qualified), score=rankings.get("overall"), rankings=rankings, budgets=points)


def independent_evm_db(y, reference, sample_rate, grid, start, window=None):
    """Independent receiver using FFT-shift indexing and a least-squares gain fit."""
    y, reference = y[:, 0] + 1j*y[:, 1], reference[:, 0] + 1j*reference[:, 1]
    first, last = (0,len(reference)) if window is None else window
    size = grid["useful_samples"]
    received, wanted = [], []
    if grid.get("carriers"):
        frequency = np.fft.fftshift(np.fft.fftfreq(len(reference),1/sample_rate))
        time = (np.arange(len(reference))+start)/sample_rate
        for carrier in grid["carriers"]:
            shift = np.cos(2*np.pi*carrier["frequency_shift_hz"]*time) + 1j*np.sin(2*np.pi*carrier["frequency_shift_hz"]*time)
            filtered = []
            for signal in (y,reference):
                spectrum = np.fft.fftshift(np.fft.fft(signal*shift))
                spectrum[abs(frequency)>carrier["filter_half_bandwidth_hz"]] = 0
                filtered.append(np.fft.ifft(np.fft.ifftshift(spectrum)))
            bins = np.asarray(carrier["occupied_bins"],dtype=int)
            spans = [p-start for p in carrier["fft_starts"] if start+first <= p and p+size <= start+last]
            if not spans:
                raise ValueError("Independent EVM: incomplete measured-carrier symbol")
            for offset in spans:
                received.append(np.fft.fft(filtered[0][offset:offset+size])[bins])
                wanted.append(np.fft.fft(filtered[1][offset:offset+size])[bins])
    else:
        period, k = size+grid["prefix_samples"], 0
        spans=[]
        while (k+1)*period <= start+last:
            offset=k*period+grid["prefix_samples"]-start
            if offset>=first:spans.append(offset)
            k+=1
        for offset in spans:
            if "occupied_bins" in grid:
                keep=np.asarray(grid["occupied_bins"],dtype=int)
            else:
                frequency=np.fft.fftfreq(size,1/sample_rate)
                keep=np.any([(frequency>=lo)&(frequency<hi) for lo,hi in grid["occupied_hz"]],axis=0)
            received.append(np.fft.fft(y[offset:offset+size])[keep])
            wanted.append(np.fft.fft(reference[offset:offset+size])[keep])
    received,wanted=np.concatenate(received),np.concatenate(wanted)
    gain=np.linalg.lstsq(wanted[:,None],received,rcond=None)[0][0]
    ratio=np.linalg.norm(received-gain*wanted)/np.linalg.norm(gain*wanted)
    return 20*math.log10(max(float(ratio),1e-15))


def replay_streaming(row, case, weights_path, meta, tolerance=1e-3, judge_device="recorded"):
    """Separate original-device reproduction from same-device chunk invariance.

    The original matrix uses CPU DPD streaming with four Torch threads and PA
    judging on the recorded evaluation device. A CPU PA is not a bit-equivalent
    substitute for a CUDA PA. An explicit device override is retained in the
    audit evidence for matrices produced with a different judging setup.
    """
    import torch
    from threadpoolctl import threadpool_limits
    from opendpd.core.arena_metrics import compute
    from opendpd.core.arena_runner import _load_weights, build_model, limit_array, load_frozen, pa_output
    from opendpd.core.streaming import CONSISTENCY_TOLERANCE, run_stream
    from opendpd.schemas import SignalSpec
    from opendpd.services.streaming import streaming_model

    torch.set_num_threads(4)
    with np.load(arena.ASSETS / meta["data_file"], allow_pickle=False) as data:
        x = np.asarray(data["x_test"], dtype=np.float32)
        peak = float(meta["peak_limit"])
    base = get_model(row["backbone"]).weights_from
    parameters = next(p["model_parameters"] for p in row["budgets"] if p["budget"] == case["budget"])
    model = _load_weights(build_model(base, parameters), weights_path).eval()
    with threadpool_limits(limits=4):
        raw200 = run_stream(streaming_model(row["backbone"], model), x, 200)
        raw137 = run_stream(streaming_model(row["backbone"], model), x, 137)
    iq_delta = float(np.max(np.abs(raw137.astype(np.float64)-raw200.astype(np.float64))))
    if iq_delta > CONSISTENCY_TOLERANCE:
        raise ValueError(f"DPD IQ depends on chunking: {iq_delta:.6g} > {CONSISTENCY_TOLERANCE:.6g}")
    u200, u137 = limit_array(raw200, peak), limit_array(raw137, peak)
    signal = SignalSpec.model_validate(meta["signal"])
    actual_device = case.get("evaluation_device", case["training_device"]) if judge_device == "recorded" else judge_device
    if actual_device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported recorded judging device: {actual_device}")
    if actual_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Stored PA judging used CUDA; expose CUDA for original-device reproduction. "
                         "CPU judging is a separate cross-device diagnostic, not an equivalent replay.")
    judge = load_frozen(meta["teacher"], actual_device)
    with threadpool_limits(limits=4):
        outputs = [("pa", pa_output(judge, u200, actual_device), pa_output(judge, u137, actual_device))]
    del judge
    reproduction_differences, chunk_differences, details = [], [], []
    stored = {judge["judge_id"]: judge for judge in case["judges"]}
    window = meta["stimuli"]["splits"]["test"]
    first, last = window["metric_start"], window["metric_stop"]
    grid, start = meta["evm_grid"], meta["split"]["boundaries"]["test"][0]
    for name, y200, y137 in outputs:
        observed = compute(y200, case["reference_gain"] * x, signal, grid, start, (first,last))
        rechunked = compute(y137, case["reference_gain"] * x, signal, grid, start, (first,last))
        second = independent_evm_db(np.asarray(y200, dtype=np.float64), case["reference_gain"] * x.astype(np.float64),
                                    signal.sample_rate_hz, grid, start, (first,last))
        if abs(second - stored[name]["evm_db"]) > tolerance:
            raise ValueError(f"Independent EVM demodulation differs for {name}: {abs(second - stored[name]['evm_db']):.6g} dB")
        reproduction, chunk = {}, {}
        for key, value in observed.items():
            delta = abs(value - stored[name][key])
            reproduction[key] = delta
            reproduction_differences.append(delta)
            if delta > tolerance:
                raise ValueError(f"Original-device 200-sample reproduction differs for {name}/{key}: {delta:.6g} dB")
            chunk_delta = abs(rechunked[key]-value)
            chunk[key] = chunk_delta
            chunk_differences.append(chunk_delta)
            if chunk_delta > tolerance:
                raise ValueError(f"Same-device 200→137 chunk comparison differs for {name}/{key}: {chunk_delta:.6g} dB")
        details.append(dict(judge_id=name, reproduction_deltas_db=reproduction, chunk_deltas_db=chunk))
    return dict(backbone=row["backbone"], budget=case["budget"], condition_id=case["condition_id"], seed=case["seed"],
        dpd_device="cpu", dpd_threads=4, original_chunk_samples=200, comparison_chunk_samples=137,
        judge_device=actual_device,
        device_basis="official matrix recorded evaluation device" if judge_device == "recorded" else "explicit audit override",
        dpd_chunk_max_abs_iq=iq_delta, dpd_chunk_iq_tolerance=CONSISTENCY_TOLERANCE,
        stored_metric_max_abs_delta_db=max(reproduction_differences),
        same_device_chunk_max_abs_metric_delta_db=max(chunk_differences), metric_tolerance_db=tolerance,
        judges=details)


def audit(workspace, *, allow_partial=False, replay=True, judge_device="recorded", operations=True, bundle_path=None):
    protocol = arena.protocol()
    calibration = arena.active_calibration()
    root = Path(workspace)
    models = arena.bundled_backbones()
    expected = {(board.board_id, model.key) for board in protocol.boards for model in models}
    boards = {board.board_id: board for board in protocol.boards}
    sweeps = {model.key: arena.sweep(model.key) for model in models}
    expected_cases = sum(sum(point["model_parameters"] is not None for point in sweeps[model.key])
                         * len(board.conditions) * (1 if model.deterministic else 3)
                         for board in protocol.boards for model in models)
    problems, warnings, rows = [], [], []
    failures, draws_by_pair, weights_by_identity = [], defaultdict(set), {}
    stream_requests, verified_assets = [], set()
    counts = Counter(dict.fromkeys(("expected_cases_present_rows", "completed_cases", "succeeded_rows", "eligible_rows",
                                   "qualified_budget_points"), 0))  # a zero is reported, not omitted
    if len(expected) != 23 or list(protocol.budgets) != list(BUDGETS):
        problems.append("The APA_200MHz_b audit requires exactly 23 identities and the 250/500/1000/2000 sweep")
    if {b.board_id: b.dataset for b in protocol.boards} != MEASURED_BOARDS or any(
            b.conditions != [b.board_id] or b.evidence_type != "measured_data_simulation" for b in protocol.boards):
        problems.append("Arena must contain only the APA_200MHz_b single-condition board")
    if set(calibration) != set(MEASURED_BOARDS) or any(
            meta.get("origin") != "measured" or meta.get("simulation") is not None
            or meta.get("dataset") != MEASURED_BOARDS.get(key)
            for key, meta in calibration.items()):
        problems.append("Active calibration must contain only the APA_200MHz_b measured PA capture")

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def asset(filename, expected_hash):
        path = arena.ASSETS / filename
        if str(path) not in verified_assets:
            require(path.is_file() and not path.is_symlink() and sha(path) == expected_hash,
                    f"Asset integrity failure: {filename}")
            verified_assets.add(str(path))

    for identifier, meta in calibration.items():
        try:
            asset(meta["source_capture_file"], meta["source_capture_sha256"])
            asset(meta["data_file"], meta["data_sha256"])
            require(meta["calibration_data_file"] == meta["data_file"]
                    and meta["calibration_split"] == meta["split"], "PA and DPD use different partitions")
            require(meta["stimuli"]["kind"] == "unchanged original measured capture samples", "Non-measured DPD input")
            with np.load(arena.ASSETS/meta["source_capture_file"], allow_pickle=False) as source, \
                    np.load(arena.ASSETS/meta["data_file"], allow_pickle=False) as data:
                previous_stop = 0
                for split in ("train", "val", "test"):
                    first, last = meta["split"]["boundaries"][split]
                    require(previous_stop <= first < last, "Overlapping or invalid measured partitions")
                    previous_stop = last
                    for kind in ("x", "y"):
                        original = np.concatenate([source[f"{kind}_{s}"] for s in ("train", "val", "test")])
                        require(np.array_equal(data[f"{kind}_{split}"], original[first:last]),
                                f"{split} {kind} is not an unchanged original measured slice")
                    window = meta["stimuli"]["splits"][split]
                    require(window["source_start"] == first and window["source_stop"] == last,
                            "Measured source bounds differ from evaluation metadata")
                    require(window["metric_start"] == 200 and window["metric_stop"] == last-first-200,
                            "Metric context differs from the fixed 200 samples")
            asset(meta["teacher"]["file"], meta["teacher"]["sha256"])
            with np.load(arena.ASSETS/meta["teacher"]["file"], allow_pickle=False) as pa:
                pa_parameters = sum(pa[key].size * (2 if np.iscomplexobj(pa[key]) else 1) for key in pa.files)
                require(all(np.isfinite(pa[key]).all() for key in pa.files), "Nonfinite frozen PA weights")
                require(pa_parameters == meta["teacher"]["parameters"] <= 5000,
                        "PA parameter count or limit differs from its checkpoint")
        except (KeyError, TypeError, ValueError, OSError) as error:
            problems.append(f"Measured provenance {identifier}: {error}")

    operation_report = None
    if operations:
        try:
            from benchmark.verify_arena_operations import report as operation_check
        except ImportError:        # executed as a script: benchmark/ itself is on sys.path
            from verify_arena_operations import report as operation_check
        operation_report = operation_check()
        problems.extend("Operation count: " + problem for problem in operation_report["problems"])
        verified_costs = {(r["backbone"], r["budget"]): r for r in operation_report["rows"]}

    for path in sorted((root / "jobs").glob("*/result.json")):
        row = read(path)
        identity = (row.get("board_id"), row.get("backbone"))
        label = "/".join(str(x) for x in identity)
        if row.get("protocol_sha256") != protocol.protocol_sha256:
            problems.append(f"{label}: stale protocol artifact remains in jobs")
            continue
        rows.append(row)
        try:
            require(identity in expected, "unexpected board/backbone identity")
            board = boards[identity[0]]
            key = identity[1]
            deterministic = key in {"mp_ls", "gmp_ls", "ilc_dpd"}
            seeds = [0] if deterministic else [0, 1, 2]
            require([point["budget"] for point in row["budgets"]] == list(BUDGETS), "budget sweep is incomplete")
            require([point["model_parameters"] for point in row["budgets"]]
                    == [point["model_parameters"] for point in sweeps[key]], "official sweep preset was changed")
            available = {point["budget"]: point for point in row["budgets"] if point["model_parameters"] is not None}
            triples = {(budget, condition, seed) for budget in available for condition in board.conditions for seed in seeds}
            counts["expected_cases_present_rows"] += len(triples)
            require(row["expected_cases"] == len(triples), "wrong required case count")
            require(row["seeds"] == seeds, "wrong seed list")
            require(row["completed_cases"] == len(row["cases"]), "completed count does not match cases")
            require(row["evidence_type"] == board.evidence_type, "wrong evidence label")
            if row["status"] == "failed":
                require(not row["eligible"] and row["score"] is None, "failed row carries a rankable score")
                failures.append({"board_id": identity[0], "backbone": key, "error": row.get("error")})
                continue
            require(row["status"] == "succeeded", "nonterminal row in reference jobs")
            case_triples = [(case["budget"], case["condition_id"], case["seed"]) for case in row["cases"]]
            require(len(case_triples) == len(triples) and set(case_triples) == triples,
                    "missing/duplicate budget, condition or seed")
            require(all(type(case["seed"]) is int for case in row["cases"]), "noninteger seed")
            require(row["provenance"]["training_sha256"] == protocol.training_sha256, "training fingerprint mismatch")
            require(row["provenance"].get("data_access") == dict(training_splits=["train", "val"],
                    evaluation_split="test", all_checkpoints_frozen_before_test=True),
                    "missing train/validation-to-frozen-test phase boundary")
            base = get_model(key).weights_from or key
            streaming = bool(get_model(key).weights_from)
            require(row["execution_semantics"] == ("streaming_stateful" if streaming else "offline_overlap_200_100"),
                    "wrong execution semantics")
            for budget, point in available.items():
                require(point["parameters"] <= budget and not any(point["parameters"] <= smaller for smaller in BUDGETS if smaller < budget),
                        "configuration is outside its budget class")
                require(point["ops"] == point["mul"] + point["add"] and point["mul"] > 0, "OPs is not MUL + ADD")
                if operations:
                    checked = verified_costs[(key, budget)]
                    require((checked["parameters"], checked["mul"], checked["add"]) == (point["parameters"], point["mul"], point["add"]),
                            "published cost differs from the verified operation count")
            for case in row["cases"]:
                require(case.get("evaluation_split") == "test", "ranked observations do not come from test")
                budget, condition, seed = case["budget"], case["condition_id"], case["seed"]
                params = available[budget]["model_parameters"]
                meta = calibration[condition]
                counts["completed_cases"] += 1
                asset(meta["data_file"], meta["data_sha256"])
                asset(meta["calibration_data_file"], meta["calibration_data_sha256"])
                asset(meta["teacher"]["file"], meta["teacher"]["sha256"])
                require(case["data_sha256"] == meta["data_sha256"], "case dataset hash mismatch")
                require(case["teacher_sha256"] == meta["teacher"]["sha256"], "case teacher hash mismatch")
                with np.load(arena.ASSETS / meta["data_file"], allow_pickle=False) as data:
                    train_x, train_y = np.asarray(data["x_train"], dtype=np.float32), np.asarray(data["y_train"], dtype=np.float32)
                    gain = float(meta["reference_gain"])
                    n_train = len(train_x)
                require(case["reference_gain"] == gain, "reference gain differs from the frozen calibration")
                model_hash = digest(dict(key=base, parameters=params, training=protocol.training_sha256))
                folder = root / "cache" / model_hash / condition / f"seed-{seed}"
                info = read(folder / "training.json")
                weights = folder / "weights.npz"
                weight_hash = sha(weights)
                binding = dict(training_sha256=protocol.training_sha256, backbone=base,
                               model_parameters=params, condition_id=condition, seed=seed)
                require(info["cache_binding"] == binding == case["cache_binding"], "cache request binding mismatch")
                require(info["weights_sha256"] == case["weights_sha256"] == case["checkpoint_sha256"] == weight_hash,
                        "cache or case checkpoint hash mismatch")
                require(all(case.get(k) == v for k, v in info.items()), "case training metadata differs from retained cache")
                require("reused_from" not in info, "Raw measured-input protocol requires fresh DPD training")
                with np.load(weights, allow_pickle=False) as arrays:
                    n_parameters = sum(arrays[name].size * (2 if np.iscomplexobj(arrays[name]) else 1) for name in arrays.files)
                    require(all(np.isfinite(arrays[name]).all() for name in arrays.files), "checkpoint contains nonfinite weights")
                require(n_parameters == case["parameters"] == available[budget]["parameters"],
                        "parameter count differs from checkpoint")
                weights_by_identity[(key, budget, condition, seed)] = weight_hash
                if deterministic:
                    require(case["attained_epochs"] == case["optimizer_updates"] == 0 and case["selected_epoch"] is None,
                            "deterministic fit claims neural training")
                    fit = case["fit"]
                    require(fit["rcond"] == params["rcond"] and fit["n_coefficients"] * 2 == n_parameters
                            and 0 < fit["rank"] <= fit["n_coefficients"], "polynomial fit diagnostics violate the preset")
                    require(fit["n_observations"] == (min(16384, n_train) if key == "ilc_dpd" else n_train),
                            "polynomial fit used a different sample budget")
                    if key == "ilc_dpd":
                        # The runner records history length, including the initial
                        # observation before the first accepted waveform update.
                        require(case["test_feedback"] is False and 1 <= case["ilc_iterations"] <= 31,
                                "ILC used test feedback or exceeded its iteration budget")
                else:
                    frames = n_train - 199
                    require(case["attained_epochs"] == 240 and case["optimizer_updates"] == 240 * math.ceil(frames / 64),
                            "incomplete training budget")
                    require(case["frames_per_epoch"] == frames and case["frame_exposures"] == 240 * frames,
                            "training window/sample exposure count differs")
                    require(type(case["selected_epoch"]) is int and 1 <= case["selected_epoch"] <= 240,
                            "checkpoint not selected at a declared validation check")
                    pair = (condition, seed)
                    if pair not in frame_cache:
                        rng = np.random.default_rng(seed)
                        draws = hashlib.sha256()
                        for _ in range(240):
                            draws.update(rng.permutation(frames).astype("<i8").tobytes())
                        frame_cache[pair] = draws.hexdigest()
                    require(case["frame_draw_sha256"] == frame_cache[pair], "frame draws differ from the prescribed generator/budget")
                    draws_by_pair[pair].add(case["frame_draw_sha256"])
                    history = read(folder / "history.json")
                    require([item["epoch"] for item in history] == list(range(1, 241)), "validation history is incomplete")
                    for item in history:
                        v = item["validation"]
                        objective = .5 * (v["ib_error_db"] + max(v["aclr_l_db"], v["aclr_r_db"]))
                        require(close(objective, item["validation_objective_db"]), "validation objective differs from the declared spectral rule")
                        require(item["optimizer_updates"] == item["epoch"] * math.ceil(frames / 64),
                                "epoch did not traverse all training windows")
                    chosen = max(history, key=lambda item: (item["validation_feasible"], -item["validation_objective_db"]))
                    require(chosen["epoch"] == case["selected_epoch"], "selected checkpoint violates feasible-first spectral validation rule")
                    require(close(chosen["validation_objective_db"], case["selected_validation_objective_db"]),
                            "selected validation spectral objective mismatch")
                    require(chosen["validation_feasible"] == case["selected_validation_feasible"], "selected feasibility mismatch")
                    require(close(chosen["validation_nmse_db"], case["selected_validation_nmse_db"]), "selected validation NMSE mismatch")
                expected_judges = arena.judge_hashes(meta)
                require(len(case["judges"]) == len(expected_judges)
                        and {j["judge_id"] for j in case["judges"]} == set(expected_judges), "missing/duplicate final judge")
                for judge in case["judges"]:
                    require(judge["checkpoint_sha256"] == expected_judges[judge["judge_id"]], "judge weight/equation hash mismatch")
                    require(all(finite(judge.get(metric)) for metric in REQUIRED_METRICS), "missing/nonfinite raw metric")
                for spec in meta["judges"]:
                    asset(spec["file"], spec["sha256"])
                if streaming:
                    stream_requests.append((row, case, weights, meta))
            calculated = independent_score(row)
            require(row["eligible"] == calculated["eligible"], "eligibility disagrees with independent gates")
            require((calculated["score"] is None and row["score"] is None) or close(row["score"], calculated["score"]),
                    "independent FoM mismatch")
            for name, entry in row["rankings"].items():
                value = calculated["rankings"].get(name)
                require((value is None and entry["score"] is None) or close(entry["score"], value),
                        f"independent ranking mismatch: {name}")
            for budget, value in calculated["budgets"].items():
                point = available[budget]
                require(point["qualified"] == value["qualified"], f"budget {budget} qualification mismatch")
                for field in ("quality_db", "quality_std_db", "quality_conservative_db", "parameter_efficiency_db",
                              "arithmetic_efficiency_db", "score"):
                    require(close(point[field], value[field]), f"independent {field} mismatch at budget {budget}")
                require(close(point["metrics"]["nmse_improvement_db"], value["nmse_improvement_db"])
                        and close(point["metrics"]["evm_improvement_db"], value["evm_improvement_db"])
                        and close(point["metrics"]["aer_improvement_db"], value["aer_improvement_db"]),
                        f"independent metric summary mismatch at budget {budget}")
            counts["succeeded_rows"] += 1
            counts["eligible_rows"] += int(row["eligible"])
            counts["qualified_budget_points"] += sum(point["qualified"] for point in row["budgets"])
        except (KeyError, TypeError, ValueError, OSError, AssertionError) as error:
            problems.append(f"{label}: {type(error).__name__}: {error}")
    identities = [(row["board_id"], row["backbone"]) for row in rows]
    if len(identities) != len(set(identities)):
        problems.append("Duplicate matrix identities")
    missing = sorted(expected - set(identities))
    if missing and not allow_partial:
        problems.append(f"Missing {len(missing)} of {len(expected)} required rows")
    for pair, hashes in draws_by_pair.items():
        if len(hashes) != 1:
            problems.append(f"Frame draws differ across architectures for {pair}")
    replayed, max_replay_delta, replay_details = 0, 0., []
    for row, case, weights, meta in stream_requests:
        base = get_model(row["backbone"]).weights_from
        lookup = (base, case["budget"], case["condition_id"], case["seed"])
        if weights_by_identity.get(lookup) != case["checkpoint_sha256"]:
            problems.append(f"Streaming weights differ from base: {row['backbone']}/{lookup[1:]}")
            continue
        if replay:
            try:
                detail = replay_streaming(row, case, weights, meta, judge_device=judge_device)
                replay_details.append(detail)
                max_replay_delta = max(max_replay_delta, detail["stored_metric_max_abs_delta_db"],
                                       detail["same_device_chunk_max_abs_metric_delta_db"])
                replayed += 1
            except (ValueError, RuntimeError, KeyError) as error:
                problems.append(f"Streaming replay {row['backbone']}/{lookup[1:]}: {error}")
    bundle_path = Path(bundle_path) if bundle_path is not None else arena.ASSETS / arena.RESULTS_FILE
    if not bundle_path.is_file():  # a development audit may precede publication; a final one may not
        (warnings if allow_partial else problems).append("No published bundle to compare with the jobs")
    bundle = read(bundle_path) if bundle_path.is_file() else {"rows": [], "protocol_sha256": protocol.protocol_sha256}
    if bundle_path.is_file() and digest({k: v for k, v in bundle.items() if k != "sha256"}) != bundle.get("sha256"):
        problems.append("Published bundle content seal mismatch")
    if bundle.get("protocol_sha256") != protocol.protocol_sha256:
        problems.append("Published bundle protocol mismatch")
    published = {(r["board_id"], r["backbone"]): r for r in bundle["rows"]}
    if len(published) != len(bundle["rows"]):
        problems.append("Duplicate identities in published bundle")
    for row in rows:
        identity = row["board_id"], row["backbone"]
        wanted = {**row, "origin": "official", "entry_id": f"official-{identity[0]}-{identity[1]}"}
        if published.get(identity) != wanted:
            (warnings if allow_partial else problems).append(f"Published/job mismatch: {identity}")
    if set(published) - set(identities):
        (warnings if allow_partial else problems).append("Published bundle contains an unaccounted row")
    report = dict(protocol_sha256=protocol.protocol_sha256, training_sha256=protocol.training_sha256,
        audit_source_sha256=sha(__file__), expected_rows=len(expected), observed_rows=len(rows),
        expected_cases=expected_cases, **counts, missing_rows=missing, execution_failures=failures,
        frame_draw_groups_verified=len(draws_by_pair), streaming_cases_replayed=replayed,
        streaming_replay_max_abs_metric_delta_db=max_replay_delta,
        operation_count_check=None if operation_report is None else {k: v for k, v in operation_report.items() if k != "rows"},
        streaming_replay_details=replay_details,
        problems=problems, warnings=warnings, mode="development_partial" if allow_partial else "complete_matrix",
        integrity_ok=not problems, complete=len(rows) == len(expected) and counts["completed_cases"] == expected_cases)
    return report, rows


# The exact frame generator is reproduced once per condition/seed, not per model.
frame_cache = {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-streaming-replay", action="store_true", help="Development audit only")
    parser.add_argument("--skip-operation-check", action="store_true", help="Development audit only")
    parser.add_argument("--judge-device", choices=("recorded", "cpu", "cuda"), default="recorded",
                        help="PA judge device for reproduction; official matrix defaults to recorded gradient device")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report, _ = audit(args.workspace, allow_partial=args.allow_partial, replay=not args.skip_streaming_replay,
                      judge_device=args.judge_device, operations=not args.skip_operation_check)
    if (args.skip_streaming_replay or args.skip_operation_check) and not args.allow_partial:
        report["problems"].append("Final audit must replay streaming checkpoints and verify operation counts")
        report["integrity_ok"] = False
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(not report["integrity_ok"] or bool(report["execution_failures"]) or
               (not args.allow_partial and not report["complete"]))


if __name__ == "__main__":
    raise SystemExit(main())
