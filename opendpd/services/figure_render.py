"""Standalone figure replay. Exported verbatim with saved figure bundles.

Run ``python replay.py <bundle-directory>`` (requires matplotlib).
Only renders stored arrays and metrics; never trains, aligns or evaluates.
"""

import hashlib
import json
import textwrap
from pathlib import Path


def plot_kind(kind):
    return 'amam' if kind in {'amam', 'ampm'} else kind


def coordinates(panel, plot, trace):
    """Stored numerical coordinates and units, shared by CSV and rendering."""
    kind = panel['kind']
    if kind == 'spectrum':
        hz = plot['axis'] == 'hz'
        return plot['frequency'], trace['psd_db'], 'Hz' if hz else 'cycles/sample', 'dB re amplitude²/Hz' if hz else 'dB re amplitude²/(cycles/sample)', 'Frequency', 'PSD'
    if kind in {'amam', 'ampm'}:
        return plot['amp_in'], trace['amp_out'] if kind == 'amam' else trace['phase_deg'], 'stored units', 'stored units' if kind == 'amam' else 'degrees', 'Input amplitude', 'Output amplitude' if kind == 'amam' else 'Phase difference'
    return plot['x'], trace['y'], plot['x_unit'], trace.get('y_unit', plot.get('y_unit')), plot['x_label'], trace.get('y_label', plot.get('y_label'))


def render_figure(figure, plots, output):
    import matplotlib
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": .6,
                                    "legend.frameon": False, "path.simplify": False})
        return _draw(figure, plots, output)


def _draw(figure, plots, output):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    spec = figure["spec"]
    panels = spec["panels"]
    columns = 1 if spec["width"] == "single_column" else min(2, len(panels))
    rows = (len(panels) + columns - 1) // columns
    fig = Figure(figsize=(3.5 if spec["width"] == "single_column" else 7.2, rows * 3.2), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(rows, columns, squeeze=False).ravel()
    dash = {"solid": "-", "dash": "--", "dot": ":", "dashdot": "-.", "longdash": (0, (7, 3))}
    contexts = {b["run_id"]: b["review"] for b in figure["bindings"]}
    aliases = {run: f'R{i + 1}' for i, run in enumerate(contexts)}
    for panel, ax in zip(panels, axes):
        spectrum = panel["kind"] == "spectrum"
        for trace in panel["traces"]:
            if not trace["visible"]:
                continue
            data = plots[f'{trace["run_id"]}/{plot_kind(panel["kind"])}']
            tr = next(t for t in data["traces"] if t["name"] == trace["trace_name"])
            x, y, xu, yu, xl, yl = coordinates(panel, data, tr)
            if spectrum and xu == 'Hz':
                x, xu = [v / 1e6 for v in x], 'MHz'
            ax.set_xlabel(f'{xl} ({xu})')
            ax.set_ylabel(f'{yl} ({yu})')
            result = contexts[trace["run_id"]]["result"]
            evidence = "MOCK" if result["is_mock"] else tr.get("source", result["evidence_type"] if tr["role"] == "primary" else tr["role"])
            if any(f['key'] == 'dataset_origin' and f['value'] == 'synthetic' for f in contexts[trace['run_id']]['facts']):
                evidence = 'SYNTHETIC / ' + evidence
            label = '\n'.join(textwrap.wrap(f'{aliases[trace["run_id"]]} · {tr["name"]} [{evidence}]', width=48))
            connected = spectrum or panel['kind'] == 'error_distribution'
            cloud = panel['kind'] in {'amam', 'ampm'}
            ax.plot(x, y, color=trace["color"], linestyle=dash[trace["dash"]] if connected else "None",
                    marker=None if connected else "." if cloud else {'solid': 'o', 'dash': '^', 'dot': 's', 'dashdot': 'D', 'longdash': '+'}[trace['dash']], markersize=1.5 if cloud else 4,
                    linewidth=.85, label=label, rasterized=cloud)
        if spectrum and panel["show_bands"]:
            for band in contexts[spec["reference_run_id"]]["bands"]:
                if band["role"] == "subchannel":
                    continue
                ax.axvspan(band["edges_hz"][0] / 1e6, band["edges_hz"][1] / 1e6,
                           color="#2563EB" if band["role"] == "main" else "#D97706", alpha=.07,
                           hatch=None if band["available"] else "//")
        if panel.get("x_range"):
            ax.set_xlim(panel["x_range"])
        if panel.get("y_range"):
            ax.set_ylim(panel["y_range"])
        if panel.get("cursor_x") is not None:
            ax.axvline(panel["cursor_x"], color="#64748B", linestyle=":", linewidth=.8)
        titles = {'spectrum': 'PSD', 'amam': 'AM/AM', 'ampm': 'AM/PM', 'power_scan': 'Declared power scan', 'error_distribution': 'Residual CDF (not EVM)'}
        ax.set_title(titles[panel['kind']], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.2)
        ax.legend(fontsize=5, loc="best", frameon=True, framealpha=.9, facecolor='white', edgecolor='none')
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(spec["title"], fontsize=10)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output / f"figure.{ext}", dpi=200)


def replay(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("bundle member outside root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"hash mismatch: {name}")
    figure = json.loads((root / "figure.json").read_text())
    plots = json.loads((root / "plot-data.json").read_text())
    render_figure(figure, plots, root / "replayed")


if __name__ == "__main__":
    import sys
    replay(sys.argv[1] if len(sys.argv) > 1 else ".")
