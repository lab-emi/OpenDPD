"""Fixed container entrypoint: reviewed OpenDPD only, bounded allocator, no shell."""
import argparse
import json
import re
from pathlib import Path

import torch


def validate_arena_request(request):
    from opendpd.core import arena
    from opendpd.core.backbone_template import parse_definition, source_sha256, validate_definition

    required = {"board_id", "backbone", "protocol_sha256"}
    if (not isinstance(request, dict) or not required <= request.keys()
            or request.keys() - required - {"model_parameters", "model_provenance"}):
        raise ValueError("invalid Arena request")
    current = arena.protocol()
    if (request["protocol_sha256"] != current.protocol_sha256
            or request["board_id"] not in {board.board_id for board in current.boards}
            or request["backbone"] not in {item.key for item in arena.bundled_backbones()}):
        raise ValueError("Arena request does not match the installed protocol")
    parameters, provenance = request.get("model_parameters", {}), request.get("model_provenance", {})
    if not isinstance(parameters, dict) or not isinstance(provenance, dict):
        raise ValueError("invalid Arena model metadata")
    if parameters:
        if request["backbone"] != "user_template" or set(parameters) != {"definition"}:
            raise ValueError("Arena accepts only validated template definitions")
        if (set(provenance) != {"backbone_id", "source_sha256", "definition_sha256"}
                or not isinstance(provenance["backbone_id"], str)
                or re.fullmatch(r"ub-[a-f0-9]{64}", provenance["backbone_id"]) is None
                or any(not isinstance(provenance[key], str)
                       or re.fullmatch(r"[a-f0-9]{64}", provenance[key]) is None
                       for key in ("source_sha256", "definition_sha256"))):
            raise ValueError("invalid Arena template provenance")
        definition = parse_definition(parameters["definition"])
        if source_sha256(parameters["definition"].encode()) != provenance["definition_sha256"]:
            raise ValueError("Arena template definition hash mismatch")
        maximum = current.training.get("max_parameters")
        if type(maximum) is not int or maximum < 1:
            raise ValueError("invalid Arena parameter budget")
        if validate_definition(definition)["parameters"] > min(4096, maximum):
            raise ValueError("Arena template exceeds the parameter budget")
    elif provenance:
        raise ValueError("unexpected Arena template provenance")
    return request


def arena_capability():
    """Verify the pinned image contains the runner and every protocol asset."""
    from opendpd.core import arena
    from opendpd.core.arena_runner import evaluate_request
    if not callable(evaluate_request):
        raise ValueError("Arena runner unavailable")
    current, calibration = arena.protocol(), arena.calibration()
    for board in current.boards:
        for condition_id in board.conditions:
            condition = calibration[condition_id]
            assets = [(condition["data_file"], condition["data_sha256"])]
            assets.extend((item["file"], item["sha256"])
                          for item in [condition["teacher"], *condition.get("judges", [])])
            for filename, sha256 in assets:
                if (not isinstance(filename, str) or Path(filename).name != filename
                        or arena.file_hash(arena.ASSETS / filename) != sha256):
                    raise ValueError("Arena asset integrity failure")
    validate_arena_request({"board_id": current.boards[0].board_id, "backbone": "gru",
                            "protocol_sha256": current.protocol_sha256})
    return {"protocol_sha256": current.protocol_sha256}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena-capability", action="store_true")
    parser.add_argument("--workspace")
    parser.add_argument("--run-id")
    parser.add_argument("--kind", choices=("run", "arena"), default="run")
    args = parser.parse_args(argv)
    if args.arena_capability:
        print(json.dumps(arena_capability()))
        return 0
    if not args.workspace or not args.run_id:
        parser.error("--workspace and --run-id are required")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", args.run_id) is None:
        raise ValueError("invalid GPU run identifier")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the GPU container")
    # Cooperative allocator ceiling for reviewed models; the host scheduler
    # also serializes containers and reserves desktop GPU headroom.
    torch.cuda.set_per_process_memory_fraction(0.5, 0)
    if args.kind == "run":
        from opendpd.runtime.worker import main as run_main
        return run_main(["--workspace", args.workspace, "--run-id", args.run_id])
    if args.workspace != "/workspace":
        raise ValueError("Arena requires the fixed container workspace")
    from opendpd.web.gpu_archive import read_regular
    from opendpd.core.arena_runner import evaluate_request
    root = Path("/workspace") / "runs" / args.run_id
    data = read_regular(root, "request.json", limit=128 * 1024 + 1)
    if len(data) > 128 * 1024:
        raise ValueError("Arena request exceeds limit")
    request = validate_arena_request(json.loads(data))
    from opendpd.core.arena_runner import default_device
    device = default_device(request["backbone"])
    evaluate_request(request, root / "result.json", device=device)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
