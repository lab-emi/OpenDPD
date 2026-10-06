"""Publish the APA B subset of the completed v6 experiment without retraining.

The original matrix remains immutable. Copy its sealed weights and observations
into a separate workspace, rescore under the narrower public protocol, then run
the independent artifact, operation and streaming audit before publication.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil

from benchmark import audit_arena_baselines as auditing
from benchmark import run_arena_baselines as baseline
from benchmark.finalize_arena_reference import verification_reports
from benchmark.report_arena_results import report
from benchmark.run_arena_distributed import freeze, units
from opendpd.core import arena
from opendpd.core.arena_runner import cache_folder
from opendpd.services.workspace import read_json, write_json_atomic


def focus(source, workspace):
    source, root = Path(source).resolve(), Path(workspace).resolve()
    if source == root or source in root.parents:
        raise ValueError("Use a separate workspace; preserve the full experiment")
    protocol = arena.protocol()
    if [b.board_id for b in protocol.boards] != ["apa-200mhz-b"]:
        raise ValueError("This release requires the APA_200MHz_b board only")
    original = read_json(source / "candidate-reference.json")
    content = {k: v for k, v in original.items() if k != "sha256"}
    if arena.canonical_hash(content) != original["sha256"]:
        raise ValueError("Original reference seal is invalid")
    full_audit, frozen = read_json(source / "final-audit.json"), read_json(source / "dpd-frozen.json")
    original_calibration = read_json(source / "source-calibration-v6.json")
    code_root = Path(arena.__file__).resolve().parents[2]
    original_training = arena.canonical_hash(dict(training=arena.TRAINING, seeds=arena.SEEDS,
        calibration=original_calibration,
        judge_hashes={key: arena.judge_hashes(value) for key, value in original_calibration.items()},
        sources={name: arena.file_hash(code_root / name) for name in arena.TRAINING_SOURCE_FILES}))
    if (not full_audit["complete"] or not full_audit["integrity_ok"] or full_audit["problems"]
            or full_audit["execution_failures"] or len(frozen["checkpoints"]) != 872
            or frozen["training_sha256"] != original_training or frozen["test_access"] is not False
            or arena.calibration() != {"apa-200mhz-b": original_calibration["apa-200mhz-b"]}):
        raise ValueError("The completed experiment does not match the unchanged training protocol")
    checkpoints = [c for c in frozen["checkpoints"] if "/apa-200mhz-b/" in c["path"]]
    if len(checkpoints) != 218:
        raise ValueError("APA B requires all 218 independent fits")
    root.mkdir(parents=True, exist_ok=True)
    projected_checkpoints, projected_training = [], {}
    for checkpoint in checkpoints:
        relative = Path(checkpoint["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid archived checkpoint path")
        old = source / relative
        if arena.file_hash(old / "weights.npz") != checkpoint["weights_sha256"]:
            raise ValueError(f"Archived weight seal differs: {relative}")
        info = read_json(old / "training.json")
        binding = info["cache_binding"]
        if binding["training_sha256"] != original_training:
            raise ValueError("Original checkpoint training identity differs")
        new = cache_folder(root / "cache", binding["backbone"], binding["model_parameters"],
                           binding["condition_id"], binding["seed"])
        new.mkdir(parents=True, exist_ok=True)
        for filename in ("weights.npz", "history.json"):
            if (old / filename).exists():
                shutil.copy2(old / filename, new / filename)
        info.update(source_cache_binding=copy.deepcopy(binding),
                    source_training_record_sha256=arena.file_hash(old / "training.json"),
                    cache_binding={**binding, "training_sha256": protocol.training_sha256})
        if isinstance(info.get("execution_profile"), dict):
            execution = info["execution_profile"]
            info["source_execution_profile_sha256"] = arena.canonical_hash(execution)
            info["execution_profile"] = {key: execution[key] for key in ("profile", "cpu_affinity") if key in execution}
            info["execution_profile"]["replay_units"] = [item for item in execution.get("replay_units", [])
                                                         if item["unit"][2] == "apa-200mhz-b"]
        write_json_atomic(new / "training.json", info)
        projected_training[arena.canonical_hash(binding)] = info
        projected_checkpoints.append(dict(path=str(new.relative_to(root)), weights_sha256=checkpoint["weights_sha256"]))
    # Retain the actual pre-test freeze time, not the time of this projection.
    write_json_atomic(root / "dpd-frozen.json", {**frozen, "checkpoints": projected_checkpoints,
        "training_sha256": protocol.training_sha256, "source_training_sha256": original_training,
        "source_matrix_sha256": arena.file_hash(source / "dpd-frozen.json")})
    freeze(root, units())
    old_rows = {r["backbone"]: r for r in original["rows"] if r["board_id"] == "apa-200mhz-b"}
    if len(old_rows) != 23 or sum(r["completed_cases"] for r in old_rows.values()) != 233:
        raise ValueError("APA B reference observations are incomplete")
    for key, old in old_rows.items():
        row = copy.deepcopy(old)
        row["protocol_sha256"] = protocol.protocol_sha256
        row["provenance"].update(protocol_id=protocol.protocol_id,
            training_sha256=protocol.training_sha256, source_training_sha256=original_training,
            source_reference_bundle_sha256=original["sha256"],
            source_protocol_sha256=original["protocol_sha256"])
        for case in row["cases"]:
            case.update(projected_training[arena.canonical_hash(case["cache_binding"])])
        directory = root / "jobs" / f"apa-200mhz-b--{key}"
        directory.mkdir(parents=True, exist_ok=True)
        write_json_atomic(directory / "result.json", row)
    candidate = root / "candidate-reference.json"
    baseline.publish(root, protocol, output=candidate)
    for row in read_json(candidate)["rows"]:
        old = old_rows[row["backbone"]]
        for key in ("budgets", "metrics", "score", "rankings", "eligible"):
            if row[key] != old[key]:
                raise ValueError(f"Dataset projection changed {row['backbone']}/{key}")
        for current_case, old_case in zip(row["cases"], old["cases"], strict=True):
            if any(current_case[key] != value for key, value in old_case.items()
                   if key not in {"cache_binding", "execution_profile"}):
                raise ValueError("Dataset projection changed an original observation or training record")
    audit, _ = auditing.audit(root, bundle_path=candidate)
    write_json_atomic(root / "final-audit.json", audit)
    if not audit["complete"] or not audit["integrity_ok"] or audit["execution_failures"]:
        raise ValueError("APA B audit failed; publication withheld")
    write_json_atomic(arena.ASSETS / arena.RESULTS_FILE, read_json(candidate))
    rows = arena.load_official_rows()
    docs = Path(arena.__file__).resolve().parents[2] / "docs/performance"
    write_json_atomic(docs / "arena-reference-audit.json", audit)
    (docs / "arena-reference-results.md").write_text(report(rows, protocol, read_json(candidate)["sha256"]))
    verification_reports(root, docs, protocol, audit)
    write_json_atomic(root / "projection-verification.json", dict(passed=True,
        source_reference_bundle_sha256=original["sha256"], protocol_sha256=protocol.protocol_sha256,
        source_training_sha256=original_training, training_sha256=protocol.training_sha256,
        frozen_fits=218, rows=23, test_cases=233,
        observations_and_scores_unchanged=True, no_retraining=True))
    write_json_atomic(root / "protocol.json", protocol)
    print("APA_200MHz_b: 218 sealed fits, 23 results, 233 unchanged test observations; independent audit passed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    focus(args.source, args.workspace)
