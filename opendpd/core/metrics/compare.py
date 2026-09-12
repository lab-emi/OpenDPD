"""When may two results be ranked against each other?

Only results produced under the same protocol: same metric profile and
version, same evidence type, same data (id, preprocessing version, split
protocol and split), the same kind of reference and operating point
(reference gain), the same PA surrogate for simulated DPD results and the
same execution semantics. Anything else may be shown side by side but never
auto-ranked (plan S08, S11).
"""

from __future__ import annotations

from typing import Dict, List

from opendpd.schemas.results import EvaluationResult, EvidenceType


def _reference_gain(r: EvaluationResult) -> str:
    """The reference gain is a protocol choice for simulated results; a measured capture's least-squares gain is an
    alignment of that capture (capture units), so measured results compare on their operating point instead."""
    if r.evidence_type == EvidenceType.dpd_measured:
        return "per-capture least-squares alignment"
    return "none" if r.reference.gain_value is None else f"{r.reference.gain_value:.6g}"


def _operating_point(r: EvaluationResult) -> str:
    """PA, drive, declared output power, capture chain and rate, as the operator declared them (S16). Two measured
    results rank only when every one of them agrees; a lower drive or output power is a different operating point."""
    m = r.measurement
    if r.evidence_type != EvidenceType.dpd_measured or m is None:
        return "n/a"
    c = m.conditions
    power = m.captures[0].declared_output_power_dbm
    return " | ".join([c.pa, c.drive, f"{power:g} dBm" if power is not None else "output power not declared",
                       c.capture_chain, f"{c.sample_rate_hz:g} Hz"])


def _surrogate(r: EvaluationResult) -> str:
    """The PA surrogate a simulated DPD result was scored through; PA-modeling results have none."""
    if r.evidence_type != EvidenceType.dpd_surrogate:
        return "n/a"
    pa = next((m for m in r.models if m.role == "pa"), None)
    return (pa.weights_sha256 or pa.run_id or "unknown")[:12] if pa else "unknown"


_FIELDS = (
    ("metric profile", lambda r: f"{r.metric_profile_id} v{r.metric_profile_version}"),
    ("evidence type", lambda r: r.evidence_type.value),
    ("dataset", lambda r: r.dataset.dataset_id),
    ("preprocessing version", lambda r: r.dataset.preprocessing_version),
    ("split protocol", lambda r: r.dataset.split_version),
    ("evaluated split", lambda r: r.dataset.split),
    ("reference kind", lambda r: r.reference.kind),
    ("reference gain", _reference_gain),
    ("PA surrogate", _surrogate),
    ("operating point", _operating_point),
    ("execution semantics", lambda r: ",".join(sorted({m.execution_semantics for m in r.models})) or "none"),
    ("mock", lambda r: "mock" if r.is_mock else "real"),
)


def comparison_key(result: EvaluationResult) -> Dict[str, str]:
    """The protocol a result was produced under; equal keys may be ranked."""
    return {name: str(get(result)) for name, get in _FIELDS}


def incompatibilities(a: EvaluationResult, b: EvaluationResult) -> List[str]:
    """Human-readable reasons why ``a`` and ``b`` must not be ranked against each other (empty = comparable)."""
    ka, kb = comparison_key(a), comparison_key(b)
    return [f"{name}: {ka[name]} vs {kb[name]}" for name in ka if ka[name] != kb[name]]
