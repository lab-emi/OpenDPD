"""Reports bound to their sources: every number is copied from the stored result, every chart is drawn
from the stored plots-v1 data. Nothing is recomputed here (plan S11)."""

from __future__ import annotations

import base64
import html
import io
import json
from typing import Dict, List, Optional

from opendpd import __version__
from opendpd.schemas import ArtifactKind, EvaluationResult, MetricValue
from opendpd.services import experiments
from opendpd.services.evaluation import available_profiles
from opendpd.services.workspace import Workspace, read_json


def _metric_text(m: MetricValue) -> str:
    return f"{m.value:.4f} {m.unit}" if m.value is not None else f"{m.status.value}: {m.reason}"


def _plot_png(spectrum: Optional[Dict]) -> Optional[str]:
    """Spectrum PNG (base64) drawn from the stored plots-v1 data; None when matplotlib is unavailable."""
    if not spectrum:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - reports degrade to text without matplotlib
        return None
    freq = spectrum["frequency"]
    scale = 1e6 if spectrum.get("axis") == "hz" else 1.0
    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
    for trace in spectrum["traces"]:
        ax.plot([f / scale for f in freq], trace["psd_db"], linewidth=0.9, label=trace["name"])
    bands = spectrum.get("bands")
    if bands:
        ax.axvspan(bands["main"][0] / scale, bands["main"][1] / scale, color="#2563EB", alpha=0.06)
        for lo, hi in bands["adjacent"]:
            ax.axvspan(lo / scale, hi / scale, color="#D97706", alpha=0.06)
    ax.set_xlabel("Frequency (MHz)" if scale != 1.0 else "Frequency (cycles/sample)")
    ax.set_ylabel("PSD (dB)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _Report:
    """Everything a report shows, gathered once from the run directory."""

    def __init__(self, ws: Workspace, run_id: str):
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

    def metric_rows(self) -> List[List[str]]:
        if self.result is None:
            return []
        heads = ["metric", "value"] + [b.kind for b in self.result.baselines]
        rows = [heads]
        for m in self.result.metrics:
            row = [m.name, _metric_text(m)]
            for b in self.result.baselines:
                bm = next((x for x in b.metrics if x.name == m.name), None)
                row.append(_metric_text(bm) if bm else "n/a")
            rows.append(row)
        return rows

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
                      ("Reference", f"{r.reference.kind}: {r.reference.description}"
                                    + (f" (gain {r.reference.gain_value})" if r.reference.gain_value is not None else "")),
                      ("Selected epoch", str(r.selected_epoch) if r.selected_epoch is not None else "n/a")]
            for m in r.models:
                facts.append((f"Model ({m.role})", f"{m.model.key} {json.dumps(m.model.parameters)} · run {m.run_id} · "
                                                   f"weights {m.weights_sha256}"
                                                   + (f" · training path {m.training_path}" if m.training_path else "")))
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
        return facts

    def commands(self) -> List[tuple]:
        return [("Re-evaluate the stored checkpoint", f"opendpd evaluate {self.run_id} --workspace <workspace> --profile "
                                                      f"{self.resolved.evaluation.profile_id}"),
                ("Re-run the configuration (a new experiment)", f"opendpd run --config config.user.json --workspace <workspace>"),
                ("Legacy equivalent", self.provenance.get("legacy_equivalent_command")
                 or "n/a (least-squares baselines run only through opendpd)")]


def report_markdown(ws: Workspace, run_id: str) -> str:
    rep = _Report(ws, run_id)
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
        out += ["## Signal chain", ""] + [f"- `{s.symbol}` {s.role} — {s.source}{' (simulated)' if s.simulated else ''}"
                                          for s in rep.result.signal_chain] + [""]
        if rep.result.surrogate_coverage:
            out += [f"Surrogate coverage: {rep.result.surrogate_coverage.note}", ""]
        if rep.result.scaling:
            sc = rep.result.scaling
            out += [f"Scaling: units {sc.amplitude_units}; input scaling {sc.input_scaling}; reference gain {sc.reference_gain}; "
                    "no physical calibration, no absolute power derived.", ""]
    if rep.result is not None and rep.result.limitations:
        out += ["## Limitations", ""] + [f"- {lim}" for lim in rep.result.limitations] + [""]
    if rep.lineage.parents or rep.lineage.children:
        out += ["## Lineage", ""]
        out += [f"- uses {l.relation.value}: {l.run_id} ({(l.checkpoint_sha256 or '')[:12]})" for l in rep.lineage.parents]
        out += [f"- used by ({l.relation.value}): {l.run_id}" for l in rep.lineage.children] + [""]
    out += ["## Reproduce", ""] + [f"- {k}: `{v}`" for k, v in rep.commands()] + [""]
    out += ["## Resolved configuration", "", "```json", rep.resolved.model_dump_json(indent=2), "```", ""]
    out += ["## Artifacts", ""] + [f"- `{a.artifact_id}` ({a.kind.value}) {a.file.path} sha256 {a.file.sha256}" for a in rep.artifacts]
    return "\n".join(out) + "\n"


def report_html(ws: Workspace, run_id: str) -> str:
    rep = _Report(ws, run_id)
    e = html.escape
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{e(rep.title)}</title>",
             "<style>body{font:14px/1.5 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#111827}"
             "table{border-collapse:collapse}td,th{border:1px solid #D1D5DB;padding:4px 8px;text-align:left}"
             "code,pre{background:#F3F4F6;border-radius:4px;padding:2px 4px}pre{padding:8px;overflow:auto}"
             ".note{color:#4B5563;font-size:12px}</style></head><body>",
             f"<h1>OpenDPD Studio report: {e(rep.title)}</h1>",
             f"<p class='note'>Generated by opendpd {e(__version__)}. Every number is copied from the stored result; the chart "
             "is drawn from the stored plots-v1 data. Nothing is recomputed in this report.</p>",
             "<h2>Facts</h2><table>" + "".join(f"<tr><th>{e(k)}</th><td>{e(str(v))}</td></tr>" for k, v in rep.facts()) + "</table>"]
    rows = rep.metric_rows()
    if rows:
        parts.append("<h2>Metrics</h2><table><tr>" + "".join(f"<th>{e(h)}</th>" for h in rows[0]) + "</tr>"
                     + "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</table>")
        if rep.result.baselines:
            parts.append("<p class='note'>Baseline columns are scored against the same reference as the result.</p>")
    png = _plot_png(rep.spectrum)
    if png:
        parts.append(f"<h2>Spectrum</h2><img alt='power spectral density from plots-v1 data' src='data:image/png;base64,{png}' style='max-width:100%'>")
    if rep.result is not None and rep.result.signal_chain:
        parts.append("<h2>Signal chain</h2><ul>" + "".join(
            f"<li><code>{e(s.symbol)}</code> {e(s.role)} — {e(s.source)}{' <b>(simulated)</b>' if s.simulated else ''}</li>"
            for s in rep.result.signal_chain) + "</ul>")
        if rep.result.surrogate_coverage:
            parts.append(f"<p>{e(rep.result.surrogate_coverage.note)}</p>")
        if rep.result.scaling:
            sc = rep.result.scaling
            parts.append(f"<p>Scaling: units {e(sc.amplitude_units)}; input scaling {e(sc.input_scaling)}; reference gain "
                         f"{sc.reference_gain}; no physical calibration, no absolute power derived.</p>")
    if rep.result is not None and rep.result.limitations:
        parts.append("<h2>Limitations</h2><ul>" + "".join(f"<li>{e(lim)}</li>" for lim in rep.result.limitations) + "</ul>")
    if rep.lineage.parents or rep.lineage.children:
        parts.append("<h2>Lineage</h2><ul>" + "".join(
            f"<li>uses {e(l.relation.value)}: {e(l.run_id)} ({e((l.checkpoint_sha256 or '')[:12])})</li>" for l in rep.lineage.parents)
            + "".join(f"<li>used by ({e(l.relation.value)}): {e(l.run_id)}</li>" for l in rep.lineage.children) + "</ul>")
    parts.append("<h2>Reproduce</h2><ul>" + "".join(f"<li>{e(k)}: <code>{e(v)}</code></li>" for k, v in rep.commands()) + "</ul>")
    parts.append("<h2>Resolved configuration</h2><details><summary>config.resolved.json</summary><pre>"
                 + e(rep.resolved.model_dump_json(indent=2)) + "</pre></details>")
    parts.append("<h2>Artifacts</h2><ul>" + "".join(
        f"<li><code>{e(a.artifact_id)}</code> ({e(a.kind.value)}) {e(a.file.path)} sha256 {e(a.file.sha256 or '')}</li>"
        for a in rep.artifacts) + "</ul></body></html>")
    return "".join(parts)
