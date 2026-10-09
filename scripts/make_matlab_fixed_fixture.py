"""Maintainer tool: regenerate the fixed-point-v1 packages that the MATLAB toolbox tests run against.

    python scripts/make_matlab_fixed_fixture.py [--output-folder Matlab/toolbox/tests/data]

Trains one tiny one-layer GRU PA (one epoch, CPU) on a synthetic memory-polynomial PA in a scratch workspace and exports it
twice with ``opendpd.services.deploy.export_deployment``: ``gru-pa.fixed-point-v1.zip`` under the default specification and
``gru-pa-custom.fixed-point-v1.zip`` under a specification whose every format differs (narrower words, other fractions, a
narrower accumulator, coarser tables), so that a reader which hard-codes the default numbers fails on the second one. Both are
the real packages, with the real manifest, quantised weights, golden vectors and the C99 reference verified bit for bit (that
needs a C compiler; without one the verdict is ``not_run`` and the golden vectors are still written). The only change to the
production path is the length of the ``long_sequence`` golden vector, 2048 samples instead of 65536, so that the committed
files stay small; the six cases and every rule are unchanged. The weights are not a meaningful PA: the packages exist to check
that the MATLAB fixed-point reference computes the same integers as the Python and C99 references. Training is not
bit-reproducible across platforms: regenerate only when the specification changes, and commit the files together with the
change.
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
LONG_SEQUENCE = 2048


def custom_spec():
    """A specification that differs from the default in every format the reader must take from the package, with the input,
    state and output fractions all different from each other (a reader that scales by the wrong one is caught)."""
    from opendpd.schemas.fixed_point import FixedPointSpec, TableSpec, WordFormat

    return FixedPointSpec(
        x=WordFormat(bits=14, frac=12), h=WordFormat(bits=14, frac=13), y=WordFormat(bits=14, frac=11), weight_bits=12,
        pre=WordFormat(bits=28, frac=18), accumulator_bits=40,
        sigmoid=TableSpec(function="sigmoid", range=6.0, index_frac=6, value=WordFormat(bits=14, frac=13)),
        tanh=TableSpec(function="tanh", range=3.0, index_frac=7, value=WordFormat(bits=14, frac=13)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-folder", type=Path, default=ROOT / "Matlab" / "toolbox" / "tests" / "data")
    args = parser.parse_args(argv)

    from opendpd.sdk import open_project
    from opendpd.services import deploy
    from opendpd.services.workspace import Workspace
    from tests.fixtures.synthetic import synthesize

    deploy.LONG_SEQUENCE = LONG_SEQUENCE
    args.output_folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="opendpd-fixed-fixture-") as scratch:
        project = open_project(Path(scratch) / "workspace")
        try:
            x, y = synthesize(4096, fs=FS, bandwidth=20e6)
            dataset = project.import_iq(x, y, dataset_id="fixtures", sample_rate_hz=FS, bandwidth_hz=20e6,
                                        nperseg=NPERSEG, origin="synthetic")
            job = project.train_pa(dataset["dataset_id"], model="gru", parameters={"hidden_size": 6, "num_layers": 1},
                                   training=TRAINING, device="cpu").wait(timeout=600)
            ws = Workspace.open(project.workspace)
            for name, spec in (("gru-pa", None), ("gru-pa-custom", custom_spec())):
                out = args.output_folder / f"{name}.fixed-point-v1.zip"
                manifest = deploy.export_deployment(ws, job.run_id, out, spec=spec)
                print(f"{out.name}: {out.stat().st_size} bytes; verification {manifest.verification.status}; "
                      + ", ".join(f"{g.case_id} {g.n_samples}" for g in manifest.golden))
        finally:
            project.close(stop_service=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
