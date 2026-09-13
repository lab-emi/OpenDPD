"""Save hash-bound review views and export plots/data with a standalone replay."""

import csv
import hashlib
import io
import json
import tempfile
import uuid
import zipfile
from pathlib import Path

from opendpd.core.metrics import incompatibilities
from opendpd.core.spectrum_layout import signal_node, has_dpd
from opendpd.schemas.review import FigureBinding, FigureSpec, SavedFigure, FigureSources, FigureSource, FigurePreview
from opendpd.services import figure_render
from opendpd.services.review import object_sha, plot_artifact, review_result
from opendpd.services.workspace import WorkspaceError, read_json, write_json_atomic


def _power_plot(ws, run, profile):
    context = review_result(ws, run, profile)
    if context.profile.validation.value == 'pending_cross_validation':
        raise WorkspaceError('The selected RF profile is pending independent validation.')
    fact = next((f for f in context.facts if f.key == 'average_output_power_dbm'), None)
    power = float(fact.value) if fact and fact.value is not None else None
    if power is None:
        raise WorkspaceError(f'{run}: declared average output power in dBm is not provided; normalized IQ cannot supply it')
    return dict(kind='power_scan', version='declared-power-scan-v1', x=[power], x_unit='dBm', x_label='Declared average RF output power',
        note='Each point uses a declared RF output power and an unchanged saved metric. Points are not joined or ranked; inspect all other working conditions.',
        traces=[dict(name=m.name, role='primary', source=context.result.evidence_type.value, y=[m.value], y_unit=m.unit, y_label=m.name)
                for m in context.result.metrics if m.status.value == 'ok' and m.value is not None])


def sources(ws, profiles):
    available, missing = [], []
    for run, profile in profiles.items():
        for kind in ('spectrum', 'amam', 'ampm', 'error_distribution', 'power_scan'):
            try:
                data = _power_plot(ws, run, profile) if kind == 'power_scan' else plot_artifact(ws, run, figure_render.plot_kind(kind))[0]
                available.extend(FigureSource(run_id=run, kind=kind, trace_name=t['name'], role=t['role'], source=t.get('source', t['role']), signal_node=signal_node(t, has_dpd(data['traces'])) if kind == 'spectrum' else None) for t in data['traces'])
            except WorkspaceError as exc:
                missing.append(str(exc))
    return FigureSources(sources=available, missing=list(dict.fromkeys(missing)))


def _plots(ws, spec):
    plots, files = {}, {run: {} for run in spec.profiles}
    for panel in spec.panels:
        axes = set()
        kind = figure_render.plot_kind(panel.kind)
        if kind == "spectrum" and panel.show_bands:
            run = spec.reference_run_id
            key = f"{run}/spectrum"
            data, path, digest = plot_artifact(ws, run, "spectrum")
            plots[key] = data
            files[run][path] = digest
            axes.add(data["axis"])
        for trace in panel.traces:
            key = f"{trace.run_id}/{kind}"
            if key not in plots:
                if kind == 'power_scan':
                    data = _power_plot(ws, trace.run_id, spec.profiles[trace.run_id])
                else:
                    data, path, digest = plot_artifact(ws, trace.run_id, kind)
                    files[trace.run_id][path] = digest
                plots[key] = data
            data = plots[key]
            tr = next((t for t in data["traces"] if t["name"] == trace.trace_name), None)
            if tr is None:
                raise WorkspaceError(f"trace not found: {trace.run_id}/{trace.trace_name}")
            x, y, xu, yu, xl, yl = figure_render.coordinates(panel.model_dump(), data, tr)
            if len(x) != len(y) or len(x) == 0:
                raise WorkspaceError("plot has inconsistent or empty coordinates")
            axes.add(data['axis'] if kind == 'spectrum' else (xu, yu, xl, yl))
        if len(axes) != 1:
            raise WorkspaceError("panel mixes incompatible coordinate units")
    return plots, files


def preview_figure(ws, spec: FigureSpec):
    plots, files = _plots(ws, spec)
    spec = FigureSpec.model_validate({**spec.model_dump(), "panels": figure_render.split_spectrum_panels([p.model_dump() for p in spec.panels], plots)})
    bindings = []
    for run, profile in spec.profiles.items():
        context = review_result(ws, run, profile)
        if context.profile.validation.value == 'pending_cross_validation':
            raise WorkspaceError('The selected RF profile is pending independent validation.')
        bindings.append(FigureBinding(run_id=run, result_id=context.result.result_id,
                                      result_sha256=object_sha(context.result), files=files[run], review=context))
    return FigurePreview(figure=SavedFigure(figure_id='fig-preview', spec=spec, bindings=bindings), plots=plots)


def save_figure(ws, spec: FigureSpec):
    figure = preview_figure(ws, spec).figure
    figure.figure_id = f"fig-{uuid.uuid4().hex}"
    root = ws.root / "figures"
    root.mkdir(exist_ok=True)
    write_json_atomic(root / f"{figure.figure_id}.json", figure)
    return figure


def list_figures(ws, runs=()):
    selected = set(runs)
    figures = [SavedFigure.model_validate(read_json(p)) for p in (ws.root / "figures").glob("fig-*.json")]
    return sorted((f for f in figures if not selected or set(f.spec.profiles) == selected),
                  key=lambda f: f.created_at, reverse=True)


def load_figure(ws, figure_id):
    from pydantic import TypeAdapter
    from opendpd.schemas.common import Slug
    TypeAdapter(Slug).validate_python(figure_id)
    path = ws.root / "figures" / f"{figure_id}.json"
    if not path.is_file():
        raise WorkspaceError("saved figure not found")
    return SavedFigure.model_validate(read_json(path))


def export_figure(ws, figure_id):
    figure = load_figure(ws, figure_id)
    plots = validate_sources(ws, figure)
    return _export(figure, plots)


def validate_sources(ws, figure):
    plots, files = _plots(ws, figure.spec)
    for binding in figure.bindings:
        current = review_result(ws, binding.run_id, figure.spec.profiles[binding.run_id])
        if (object_sha(current.result) != binding.result_sha256 or files[binding.run_id] != binding.files
                or object_sha(current) != object_sha(binding.review)):
            raise WorkspaceError("saved figure sources changed; save a new review to export the current evidence")
    return plots


def _export(figure, plots):
    figure = figure.model_copy(deep=True)
    figure.spec = FigureSpec.model_validate({**figure.spec.model_dump(), "panels": figure_render.split_spectrum_panels([p.model_dump() for p in figure.spec.panels], plots)})
    members = {}

    def add_json(name, value):
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        members[name] = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()

    add_json("figure.json", figure)
    add_json("plot-data.json", plots)
    add_json("results.json", [b.review.result.model_dump(mode="json") for b in figure.bindings])
    add_json("profiles.json", {b.run_id: b.review.profile.model_dump(mode="json") for b in figure.bindings})
    import matplotlib
    add_json("rendering.json", {"matplotlib_version": matplotlib.__version__, "font": "DejaVu Sans",
                                "renderer": "Matplotlib Figure / Agg, rcParamsDefault with explicit publication settings"})
    from opendpd.core import spectrum_layout
    members["spectrum_layout.py"] = Path(spectrum_layout.__file__).read_bytes()
    members["requirements.txt"] = f"matplotlib=={matplotlib.__version__}\n".encode()
    data = io.StringIO()
    writer = csv.writer(data)
    writer.writerow(["panel", "run_id", "trace", "role", "visible", "x", "x_unit", "y", "y_unit"])
    for i, panel in enumerate(figure.spec.panels):
        for trace in panel.traces:
            kind = figure_render.plot_kind(panel.kind)
            plot = plots[f"{trace.run_id}/{kind}"]
            tr = next(t for t in plot["traces"] if t["name"] == trace.trace_name)
            x, y, xunit, yunit, _, _ = figure_render.coordinates(panel.model_dump(), plot, tr)
            for xv, yv in zip(x, y):
                writer.writerow([i, trace.run_id, trace.trace_name, tr["role"], trace.visible, xv, xunit, yv, yunit])
    members["chart-data.csv"] = data.getvalue().encode()
    data = io.StringIO()
    writer = csv.writer(data)
    writer.writerow(["run_id", "result_id", "profile", "profile_version", "evidence", "mock", "metric", "value", "unit", "status", "reason"])
    for b in figure.bindings:
        r = b.review.result
        for m in r.metrics:
            writer.writerow([b.run_id, r.result_id, r.metric_profile_id, r.metric_profile_version, r.evidence_type.value, r.is_mock,
                             m.name, m.value, m.unit, m.status.value, m.reason])
    members["metrics.csv"] = data.getvalue().encode()
    caption = [figure.spec.title, "", f"Reference: {figure.spec.reference_run_id}; mode: {figure.spec.mode}.",
               "Metrics are copied verbatim from stored results. Chart CSV contains stored display points, not raw IQ.",
               "Zoom, trace visibility and cursors change the view only. Missing conditions are not provided.", ""]
    caption += [str(plot['note']) for plot in plots.values() if plot.get('note')]
    for i, b in enumerate(figure.bindings):
        r = b.review.result
        caption += [f"R{i + 1} = {b.run_id}: {'MOCK / ' if r.is_mock else ''}{r.evidence_type.value}; {r.metric_profile_id} v{r.metric_profile_version}.",
                    b.review.band_note, r.reference.description]
        caption += [f"- {f.label}: {f.value if f.value is not None else 'not provided'} {f.unit or ''} ({f.source})" for f in b.review.facts]
        caption += [f"- {key}: {value or 'not provided'}" for key, value in b.review.provenance.items()]
        caption += [f"- Limitation: {text}" for text in r.limitations]
    for i, a in enumerate(figure.bindings):
        for b in figure.bindings[i + 1:]:
            caption += [f"Incompatible {a.run_id} / {b.run_id}: {s}" for s in incompatibilities(a.review.result, b.review.result)]
    members["caption.md"] = ("\n".join(caption) + "\n").encode()
    members["replay.py"] = Path(figure_render.__file__).read_bytes()
    members["README.md"] = ("Extract this bundle, install matplotlib, then run `python replay.py .`.\n"
                             "The command verifies all packaged hashes and writes PNG, SVG and PDF to replayed/.\n"
                             "It replays saved plot data; it does not re-evaluate raw IQ. Use the Studio full run package\n"
                             "and its recorded evaluation command to independently reproduce formal metrics.\n"
                             "figure.json includes full result/model/data/config hashes, software, profiles, conditions and view state.\n").encode()
    with tempfile.TemporaryDirectory(prefix="opendpd-figure-") as tmp:
        figure_render.render_figure(figure.model_dump(mode="json"), plots, tmp)
        for ext in ("png", "svg", "pdf"):
            members[f"figure.{ext}"] = (Path(tmp) / f"figure.{ext}").read_bytes()
    add_json("manifest.json", {"version": "figure-bundle-v1", "files": {name: hashlib.sha256(content).hexdigest() for name, content in members.items()}})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()
