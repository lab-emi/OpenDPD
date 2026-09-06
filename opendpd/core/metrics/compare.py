"""When may two results be ranked against each other?

Only results produced under the same protocol: same metric profile and
version, same evidence type, same data (id, preprocessing version, split
protocol and split) and the same kind of reference. Anything else may be
shown side by side but never auto-ranked (plan S08).
"""

from __future__ import annotations

from typing import Dict, List

from opendpd.schemas.results import EvaluationResult

_FIELDS = (
    ("metric profile", lambda r: f"{r.metric_profile_id} v{r.metric_profile_version}"),
    ("evidence type", lambda r: r.evidence_type.value),
    ("dataset", lambda r: r.dataset.dataset_id),
    ("preprocessing version", lambda r: r.dataset.preprocessing_version),
    ("split protocol", lambda r: r.dataset.split_version),
    ("evaluated split", lambda r: r.dataset.split),
    ("reference kind", lambda r: r.reference.kind),
    ("mock", lambda r: "mock" if r.is_mock else "real"),
)


def comparison_key(result: EvaluationResult) -> Dict[str, str]:
    """The protocol a result was produced under; equal keys may be ranked."""
    return {name: str(get(result)) for name, get in _FIELDS}


def incompatibilities(a: EvaluationResult, b: EvaluationResult) -> List[str]:
    """Human-readable reasons why ``a`` and ``b`` must not be ranked against each other (empty = comparable)."""
    ka, kb = comparison_key(a), comparison_key(b)
    return [f"{name}: {ka[name]} vs {kb[name]}" for name in ka if ka[name] != kb[name]]
