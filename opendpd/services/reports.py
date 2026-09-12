"""Reports bound to their sources: every number is copied from the stored result, every chart is drawn
from the stored plots-v1 data. Nothing is recomputed here (plan S11)."""

from __future__ import annotations

import base64
import html
import io
import json
import re
from typing import Dict, List, Optional

from opendpd import __version__
from opendpd.studio.localization import localize, language_tag
from opendpd.schemas import ArtifactKind, EvaluationResult, MetricValue
from opendpd.services import experiments
from opendpd.services.evaluation import available_profiles
from opendpd.services.workspace import Workspace, read_json


def _metric_text(m: MetricValue) -> str:
    return f"{m.value:.4f} {m.unit}" if m.value is not None else f"{m.status.value}: {m.reason}"


def _plot_png(spectrum: Optional[Dict], language: str = "en") -> Optional[str]:
    """Spectrum PNG (base64) drawn from the stored plots-v1 data; None when matplotlib is unavailable."""
    if not spectrum:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - reports degrade to text without matplotlib
        return None
    from matplotlib import font_manager
    installed = {font.name for font in font_manager.fontManager.ttflist}
    preferred = ["Arial Unicode MS", "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK KR", "DejaVu Sans"]
    family = next((font for font in preferred if font in installed), "DejaVu Sans")
    freq = spectrum["frequency"]
    scale = 1e6 if spectrum.get("axis") == "hz" else 1.0
    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
    for trace in spectrum["traces"]:
        ax.plot([f / scale for f in freq], trace["psd_db"], linewidth=0.9, label=localize(trace["name"], language))
    bands = spectrum.get("bands")
    if bands:
        ax.axvspan(bands["main"][0] / scale, bands["main"][1] / scale, color="#2563EB", alpha=0.06)
        for lo, hi in bands["adjacent"]:
            ax.axvspan(lo / scale, hi / scale, color="#D97706", alpha=0.06)
    ax.set_xlabel(localize("Frequency (MHz)" if scale != 1.0 else "Frequency (cycles per sample)", language))
    ax.set_ylabel("PSD (dB)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    for text in fig.findobj(matplotlib.text.Text):
        text.set_fontfamily(family)
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _Report:
    """Everything a report shows, gathered once from the run directory."""

    def __init__(self, ws: Workspace, run_id: str, language: str = "en"):
        self.language = language
        self.text = lambda value: localize(str(value), language) if language != "en" else str(value)
        self.run_id = run_id
        self.record = experiments.load_run(ws, run_id)
        self.resolved = experiments.load_resolved(ws, run_id)
        self.result: Optional[EvaluationResult] = experiments.load_result(ws, run_id)
        self.profiles = available_profiles(ws, run_id)
        self.provenance = read_json(ws.run_dir(run_id) / "provenance.json")
        self.lineage = experiments.lineage(ws, run_id)
        manifest = experiments.load_artifacts(ws, run_id)
        self.artifacts = manifest.artifacts if manifest else []
        self.spectrum = None
        for a in manifest.by_kind(ArtifactKind.plot) if manifest else []:
            if a.artifact_id == "plot-spectrum":
                self.spectrum = read_json(ws.run_dir(run_id) / a.file.path)
        self.dataset = ws.get_dataset(self.resolved.dataset.id)

    @property
    def title(self) -> str:
        return self.record.name or self.run_id

    def format(self, template: str, **values: object) -> str:
        """Translate the template before inserting identifiers or other source values."""
        return re.sub(r"\{(\w+)\}", lambda match: str(values[match[1]]), self.text(template))

    def metric_rows(self) -> List[List[str]]:
        if self.result is None:
            return []
        heads = ["metric", "value"] + [b.kind for b in self.result.baselines]
        rows = [heads]
        for m in self.result.metrics:
            value = lambda metric: _metric_text(metric) if metric.value is not None or self.language == "en" else f"{self.text(metric.status.value)}: {self.text(metric.reason)}"
            row = [m.name, value(m)]
            for b in self.result.baselines:
                bm = next((x for x in b.metrics if x.name == m.name), None)
                row.append(value(bm) if bm else "n/a")
            rows.append(row)
        return [[self.text(cell) for cell in row] for row in rows]

    def facts(self) -> List[tuple]:
        r = self.result
        facts = [("Run", self.run_id), ("Task", self.record.task.value), ("Status", self.record.status.value),
                 ("Configuration hash", self.resolved.resolution.config_sha256),
                 ("Dataset", f"{self.dataset.dataset_id} · version {self.resolved.dataset.preprocessing_version} · "
                             f"split {self.resolved.dataset.split_version} · raw sha256 {self.dataset.raw_sha256}"),
                 ("Seed / reproducibility", f"{self.resolved.training.seed} / {self.resolved.training.reproducibility}"),
                 ("Device", self.resolved.execution.device)]
        if r is not None:
            facts += [("Evidence", r.evidence_type.value),
                      ("Metric profile", f"{r.metric_profile_id} v{r.metric_profile_version} (stored: {', '.join(self.profiles)})"),
                      ("Reference", f"{self.text(r.reference.kind)}: {self.text(r.reference.description)}"
                                    + (f" (gain {r.reference.gain_value})" if r.reference.gain_value is not None else "")),
                      ("Selected epoch", str(r.selected_epoch) if r.selected_epoch is not None else "n/a")]
            for m in r.models:
                facts.append((self.format("Model ({role})", role=self.text(m.role)),
                              self.format("{model} · run {run} · weights {weights}",
                                          model=f"{m.model.key} {json.dumps(m.model.parameters)}", run=m.run_id, weights=m.weights_sha256)
                              + (" · " + self.format("training path {path}", path=m.training_path) if m.training_path else "")))
            if r.measurement is not None:
                c = r.measurement.conditions
                facts.append(("Measurement", f"{r.measurement.attestation}. PA {c.pa}; chain {c.capture_chain}; "
                                             f"{c.sample_rate_hz:g} Hz; drive {c.drive}; gain "
                                             f"{c.gain_db if c.gain_db is not None else 'n/a'} dB; calibration "
                                             f"{c.calibration}; measured {c.measured_at.isoformat()}; operator "
                                             f"{c.operator or 'n/a'}; played export of run {r.measurement.apply_run_id} "
                                             f"(sha256 {r.measurement.played_sha256})"))
                for cap in r.measurement.captures:
                    power = f"{cap.declared_output_power_dbm:g} dBm" if cap.declared_output_power_dbm is not None else "not declared"
                    facts.append((f"Capture ({cap.role.replace('_', ' ')})",
                                  f"{cap.artifact_id} raw sha256 {cap.raw_sha256}; {cap.n_samples_raw} samples at "
                                  f"{cap.sample_rate_hz:g} Hz; delay {cap.delay_samples} samples"
                                  f"{' (wrapped)' if cap.wrapped else ''}; correlation {cap.correlation:.4f}; least-squares "
                                  f"gain {cap.gain_db:+.2f} dB at {cap.gain_phase_deg:+.1f} deg; rms {cap.rms:.4g}; "
                                  f"declared output power {power}"))
                if r.measurement.level_difference_db is not None:
                    facts.append(("Output level with DPD relative to without",
                                  f"{r.measurement.level_difference_db:+.2f} dB (capture units); reported, never normalised"))
        sw = self.provenance.get("software", {})
        facts.append(("Software", f"opendpd {sw.get('opendpd_version')} · python {sw.get('python_version')} · "
                                  f"torch {sw.get('torch_version')} · {sw.get('platform')} · git {sw.get('git_commit')}"
                                  f"{' (dirty)' if sw.get('git_dirty') else ''}"))
        return [(self.text(key), self.text(value)) for key, value in facts]

    def scaling(self) -> str:
        sc = self.result.scaling
        return self.format("Scaling: units {units}; input scaling {scaling}; reference gain {gain}; no physical calibration, no absolute power derived.",
                           units=self.text(sc.amplitude_units), scaling=self.text(sc.input_scaling), gain=sc.reference_gain)

    def commands(self) -> List[tuple]:
        return [("Re-evaluate the stored checkpoint", f"opendpd evaluate {self.run_id} --workspace <workspace> --profile "
                                                      f"{self.resolved.evaluation.profile_id}"),
                ("Re-run the configuration (a new experiment)", f"opendpd run --config config.user.json --workspace <workspace>"),
                ("Legacy equivalent", self.provenance.get("legacy_equivalent_command")
                 or "n/a (least-squares baselines run only through opendpd)")]


def report_markdown(ws: Workspace, run_id: str, *, language: str = "en") -> str:
    rep = _Report(ws, run_id, language)
    out = [f"# OpenDPD Studio report: {rep.title}", "",
           f"Generated by opendpd {__version__}. Every number below is copied from the stored result; charts in the "
           "HTML report are drawn from the stored plots-v1 data. Nothing is recomputed in a report.", ""]
    out += ["## Facts", ""] + [f"- **{k}**: {v}" for k, v in rep.facts()] + [""]
    rows = rep.metric_rows()
    if rows:
        out += ["## Metrics", "", "| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
        out += ["| " + " | ".join(r) + " |" for r in rows[1:]] + [""]
        if rep.result.baselines:
            out += ["Baseline columns are scored against the same reference as the result (no separate normalisation).", ""]
    if rep.result is not None and rep.result.signal_chain:
        out += ["## Signal chain", ""] + [f"- `{s.symbol}` {rep.text(s.role)} — {rep.text(s.source)}{(' (' + rep.text('simulated') + ')') if s.simulated else ''}"
                                          for s in rep.result.signal_chain] + [""]
        if rep.result.surrogate_coverage:
            out += [rep.format("Surrogate coverage: {note}", note=rep.text(rep.result.surrogate_coverage.note)), ""]
        if rep.result.scaling:
            out += [rep.scaling(), ""]
    if rep.result is not None and rep.result.limitations:
        out += ["## Limitations", ""] + [f"- {rep.text(lim)}" for lim in rep.result.limitations] + [""]
    if rep.lineage.parents or rep.lineage.children:
        out += ["## Lineage", ""]
        out += ["- " + rep.format("uses {relation}: {run} ({sha})", relation=rep.text(l.relation.value), run=l.run_id, sha=(l.checkpoint_sha256 or '')[:12]) for l in rep.lineage.parents]
        out += ["- " + rep.format("used by ({relation}): {run}", relation=rep.text(l.relation.value), run=l.run_id) for l in rep.lineage.children] + [""]
    out += ["## Reproduce", ""] + [f"- {rep.text(k)}: `{v}`" for k, v in rep.commands()] + [""]
    out += ["## Resolved configuration", "", "```json", rep.resolved.model_dump_json(indent=2), "```", ""]
    out += ["## Artifacts", ""] + [f"- `{a.artifact_id}` ({rep.text(a.kind.value)}) {a.file.path} sha256 {a.file.sha256}" for a in rep.artifacts]
    # Translate prose only. Configuration JSON and reproduction commands remain exact.
    translated, fenced = [], False
    for line in out:
        if line.startswith("```"):
            fenced = not fenced
            translated.append(line)
        elif fenced:
            translated.append(line)
        elif line.startswith("#"):
            prefix, _, title = line.partition(" ")
            translated.append(prefix + " " + rep.text(title))
        else:
            translated.append(rep.text(line))
    return "\n".join(translated) + "\n"


def report_html(ws: Workspace, run_id: str, *, language: str = "en") -> str:
    rep = _Report(ws, run_id, language)
    e = html.escape
    tr = lambda value: e(rep.text(value))
    parts = [f"<!doctype html><html lang='{language_tag(language)}'><head><meta charset='utf-8'><title>{e(rep.title)}</title>",
             "<style>body{font:14px/1.5 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#111827}"
             "table{border-collapse:collapse}td,th{border:1px solid #D1D5DB;padding:4px 8px;text-align:left}"
             "code,pre{background:#F3F4F6;border-radius:4px;padding:2px 4px}pre{padding:8px;overflow:auto}"
             ".note{color:#4B5563;font-size:12px}</style></head><body>",
             f"<h1>{tr(f'OpenDPD Studio report: {rep.title}')}</h1>",
             "<p class='note'>" + tr(f"Generated by opendpd {__version__}. Every number is copied from the stored result; the chart "
             "is drawn from the stored plots-v1 data. Nothing is recomputed in this report.") + "</p>",
             f"<h2>{tr('Facts')}</h2><table>" + "".join(f"<tr><th>{tr(k)}</th><td>{e(str(v))}</td></tr>" for k, v in rep.facts()) + "</table>"]
    rows = rep.metric_rows()
    if rows:
        parts.append(f"<h2>{tr('Metrics')}</h2><table><tr>" + "".join(f"<th>{e(h)}</th>" for h in rows[0]) + "</tr>"
                     + "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</table>")
        if rep.result.baselines:
            parts.append("<p class='note'>" + tr("Baseline columns are scored against the same reference as the result.") + "</p>")
    png = _plot_png(rep.spectrum, language)
    if png:
        parts.append(f"<h2>{tr('Spectrum')}</h2><img alt='{tr('Power spectral density')}' src='data:image/png;base64,{png}' style='max-width:100%'>")
    if rep.result is not None and rep.result.signal_chain:
        parts.append(f"<h2>{tr('Signal chain')}</h2><ul>" + "".join(
            f"<li><code>{e(s.symbol)}</code> {tr(s.role)} — {tr(s.source)}{(' <b>(' + tr('simulated') + ')</b>') if s.simulated else ''}</li>"
            for s in rep.result.signal_chain) + "</ul>")
        if rep.result.surrogate_coverage:
            parts.append(f"<p>{tr(rep.result.surrogate_coverage.note)}</p>")
        if rep.result.scaling:
            parts.append(f"<p>{e(rep.scaling())}</p>")
    if rep.result is not None and rep.result.limitations:
        parts.append(f"<h2>{tr('Limitations')}</h2><ul>" + "".join(f"<li>{tr(lim)}</li>" for lim in rep.result.limitations) + "</ul>")
    if rep.lineage.parents or rep.lineage.children:
        parts.append(f"<h2>{tr('Lineage')}</h2><ul>" + "".join(
            "<li>" + e(rep.format("uses {relation}: {run} ({sha})", relation=rep.text(l.relation.value), run=l.run_id, sha=(l.checkpoint_sha256 or '')[:12])) + "</li>" for l in rep.lineage.parents)
            + "".join("<li>" + e(rep.format("used by ({relation}): {run}", relation=rep.text(l.relation.value), run=l.run_id)) + "</li>" for l in rep.lineage.children) + "</ul>")
    parts.append(f"<h2>{tr('Reproduce')}</h2><ul>" + "".join(f"<li>{tr(k)}: <code>{e(v)}</code></li>" for k, v in rep.commands()) + "</ul>")
    parts.append(f"<h2>{tr('Resolved configuration')}</h2><details><summary>config.resolved.json</summary><pre>"
                 + e(rep.resolved.model_dump_json(indent=2)) + "</pre></details>")
    parts.append(f"<h2>{tr('Artifacts')}</h2><ul>" + "".join(
        f"<li><code>{e(a.artifact_id)}</code> ({tr(a.kind.value)}) {e(a.file.path)} sha256 {e(a.file.sha256 or '')}</li>"
        for a in rep.artifacts) + "</ul></body></html>")
    return "".join(parts)
