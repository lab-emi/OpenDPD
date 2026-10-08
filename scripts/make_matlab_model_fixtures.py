"""Maintainer tool: regenerate the model packages the MATLAB toolbox tests run against (no Python is needed to use them).

    python scripts/make_matlab_model_fixtures.py [--output Matlab/toolbox/tests/data]

Trains tiny runs (a few hundred parameters, one epoch, CPU) on a synthetic memory-polynomial PA in a scratch workspace
and exports them with ``opendpd.services.model_export``. The weights are not meaningful DPDs: the packages exist to
check that the pure-MATLAB runtime computes the same outputs as the evaluator. ``tests/unit/test_model_export.py``
checks that the committed packages still match today's PyTorch code, so a change to a backbone shows up as a failing
test instead of a silently stale fixture. Training is not bit-reproducible across platforms: regenerate only when a
model's numerics change, and commit the packages together with the change.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FS, NPERSEG = 80e6, 128
TRAINING = {"epochs": 1, "frame_length": 32, "frame_stride": 32, "batch_size": 16, "batch_size_eval": 16}
SURROGATE = {"hidden_size": 6}
# fixture name -> (role, model key, parameters)
FIXTURES = {
    "gru-dpd": ("dpd", "gru", {"hidden_size": 6, "num_layers": 2}),
    "gru-pa": ("pa", "gru", {"hidden_size": 5, "num_layers": 1}),
    "tres_gru-dpd": ("dpd", "tres_gru", {"hidden_size": 6, "num_layers": 2}),
    "gmp-dpd": ("dpd", "gmp", {}),
    "mp_ls-dpd": ("dpd", "mp_ls", {"K": 3, "Q": 4}),
    "gmp_ls-dpd": ("dpd", "gmp_ls", {"Ka": 3, "La": 4, "Kb": 2, "Lb": 3, "Mb": 2, "Kc": 2, "Lc": 3, "Mc": 2, "rcond": 1e-6}),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "Matlab" / "toolbox" / "tests" / "data")
    args = parser.parse_args(argv)

    from opendpd.sdk import open_project
    from opendpd.services.model_export import export_model
    from opendpd.services.workspace import Workspace
    from tests.fixtures.synthetic import synthesize

    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="opendpd-fixtures-") as scratch:
        project = open_project(Path(scratch) / "workspace")
        try:
            x, y = synthesize(4096, fs=FS, bandwidth=20e6)
            dataset = project.import_iq(x, y, dataset_id="fixtures", sample_rate_hz=FS, bandwidth_hz=20e6,
                                        nperseg=NPERSEG, origin="synthetic")
            surrogate = project.train_pa(dataset["dataset_id"], parameters=SURROGATE, training=TRAINING,
                                         device="cpu").wait(timeout=600)
            ws = Workspace.open(project.workspace)
            for name, (role, key, parameters) in FIXTURES.items():
                if role == "pa":
                    job = project.train_pa(dataset["dataset_id"], model=key, parameters=parameters, training=TRAINING,
                                           device="cpu").wait(timeout=600)
                else:
                    job = project.train_dpd(dataset["dataset_id"], surrogate, model=key, parameters=parameters,
                                            training=TRAINING, device="cpu").wait(timeout=600)
                summary = export_model(ws, job.run_id, args.output / f"{name}.opendpd.zip")
                print(f"{name}: {Path(summary['path']).stat().st_size} bytes, sha256 {summary['sha256'][:12]}")
        finally:
            project.close(stop_service=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
