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
    ("reference gain", lambda r: "none" if r.reference.gain_value is None else f"{r.reference.gain_value:.6g}"),
    ("PA surrogate", _surrogate),
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
