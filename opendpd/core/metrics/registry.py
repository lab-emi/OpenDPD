"""Profile registry and the single ``evaluate`` entry point used by the worker, the CLI and the API."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from opendpd.schemas.common import MetricValue
from opendpd.schemas.dataset import SignalSpec
from opendpd.schemas.metrics import MetricProfile

from . import general_v1, legacy_v1, ofdm_evm_v1

PROFILES: Dict[str, MetricProfile] = {
    legacy_v1.PROFILE.profile_id: legacy_v1.PROFILE,
    general_v1.PROFILE.profile_id: general_v1.PROFILE,
    ofdm_evm_v1.PROFILE.profile_id: ofdm_evm_v1.PROFILE,
}
DEFAULT_PROFILE_ID = legacy_v1.PROFILE.profile_id


def list_profiles() -> List[MetricProfile]:
    return list(PROFILES.values())


def get_profile(profile_id: str) -> MetricProfile:
    try:
        return PROFILES[profile_id]
    except KeyError:
        raise KeyError(f"metric profile '{profile_id}' is not registered; known: {', '.join(sorted(PROFILES))}") from None


def evaluate(profile_id: str, prediction: np.ndarray, reference: Optional[np.ndarray], signal: SignalSpec, *,
             valid_samples: Optional[int] = None) -> List[MetricValue]:
    """Compute every metric of ``profile_id``.

    ``prediction`` / ``reference`` are I/Q arrays shaped ``(n_segments, nperseg, 2)``
    (as produced by the trainer's test evaluation) or ``(n, 2)``. ``valid_samples``
    is the number of real samples; anything beyond it is zero padding of the last
    segment. The legacy profile ignores it (its frozen semantics include the
    padding); the general profile excludes it.
    """
    get_profile(profile_id)
    if profile_id == legacy_v1.PROFILE.profile_id:
        return legacy_v1.compute(prediction, reference, signal)
    if profile_id == ofdm_evm_v1.PROFILE_ID:
        return ofdm_evm_v1.compute(prediction, reference, signal, valid_samples=valid_samples)
    return general_v1.compute(prediction, reference, signal, valid_samples=valid_samples)
