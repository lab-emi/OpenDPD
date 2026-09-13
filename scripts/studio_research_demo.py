"""Build a small, reproducible Studio review demonstration from registered data.

python scripts/studio_research_demo.py --workspace /tmp/studio-review-demo

CPU smoke training only. The measured no-DPD source is the bundled DPA dataset;
DPD results are surrogate simulations, not physical with-DPD captures.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opendpd.schemas import RunStatus
from opendpd.schemas.review import FigureSpec
from opendpd.services import experiments, figures, review
from opendpd.services.recipes import instantiate
from opendpd.services.workspace import Workspace, write_json_atomic


def build(workspace):
    ws = Workspace.create(workspace)
    dataset = ws.register_builtin_dataset("DPA_200MHz")
    ids = {}
    for key, recipe in (("pa", "pa-gru-smoke-v1"), ("gru", "dpd-gru-smoke-v1"), ("mp", "dpd-mp-ila-v1")):
        config = instantiate(recipe, dataset.dataset_id, name=f"Review demo: {key} (CPU smoke)",
                             **({"pa_run_id": ids["pa"]} if key != "pa" else {}))
        record = experiments.execute_run(ws, experiments.create_run(ws, config).run_id)
        if record.status != RunStatus.succeeded:
            raise RuntimeError(f"{key}: {record.error}")
        ids[key] = record.run_id
    traces = []
    for key, color, dash in (("mp", "#2563EB", "dash"), ("gru", "#D97706", "dot")):
        data, _, _ = review.plot_artifact(ws, ids[key], "spectrum")
        primary = next(t for t in data["traces"] if t["role"] == "primary")
        traces.append(dict(run_id=ids[key], trace_name=primary["name"], color=color, dash=dash))
    spec = FigureSpec(title="MP / GRU DPD · surrogate smoke demo", reference_run_id=ids["mp"],
                      profiles={ids[key]: "legacy-opendpd-v1" for key in ("mp", "gru")},
                      panels=[dict(traces=traces, show_bands=True, x_range=[-150, 150], y_range=[-160, -50])])
    saved = figures.save_figure(ws, spec)
    out = ws.root / "review-demo"
    out.mkdir(exist_ok=True)
    (out / "figure.zip").write_bytes(figures.export_figure(ws, saved.figure_id))
    index = {"runs": ids, "figure_id": saved.figure_id,
             "compare_path": f"/results/compare?runs={ids['mp']}&runs={ids['gru']}&reference={ids['mp']}",
             "evidence": {"source": "bundled DPA_200MHz measured no-DPD IQ", "dpd": "PA surrogate simulation; smoke training, not a benchmark"},
             "missing": ["physical with-DPD capture", "independent repeat captures", "three independent working conditions",
                         "declared DC power and calibrated RF power", "orthogonal I/Q scan", "target hardware cost report"]}
    write_json_atomic(out / "index.json", index)
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    build(parser.parse_args().workspace)
