"""Application profile catalogue, extending the frozen Arena metric registry.

Arena imports opendpd.core.metrics directly; that catalogue and its dispatch
remain untouched. Studio, CLI and SDK use this facade for additional validated
profiles. Existing identifiers always delegate to the original evaluator.
"""
from __future__ import annotations

from opendpd.core.metrics import registry as frozen
from opendpd.core.metrics import comparison_key, incompatibilities
from opendpd.core.metrics import ofdm_evm_v2

DEFAULT_PROFILE_ID = frozen.DEFAULT_PROFILE_ID
PROFILES = {
    **frozen.PROFILES,
    ofdm_evm_v2.PROFILE_ID: ofdm_evm_v2.PROFILE,
    ofdm_evm_v2.AUTO_PROFILE_ID: ofdm_evm_v2.AUTO_PROFILE,
}


def list_profiles():
    return list(PROFILES.values())


def get_profile(profile_id):
    try:
        return PROFILES[profile_id]
    except KeyError:
        raise KeyError(f"metric profile '{profile_id}' is not registered; known: {', '.join(sorted(PROFILES))}") from None


def evaluate(profile_id, prediction, reference, signal, *, valid_samples=None):
    if profile_id in (ofdm_evm_v2.PROFILE_ID, ofdm_evm_v2.AUTO_PROFILE_ID):
        return ofdm_evm_v2.compute(prediction, reference, signal, valid_samples=valid_samples,
                                  auto=profile_id == ofdm_evm_v2.AUTO_PROFILE_ID)
    return frozen.evaluate(profile_id, prediction, reference, signal, valid_samples=valid_samples)


__all__ = ["DEFAULT_PROFILE_ID", "PROFILES", "evaluate", "get_profile", "list_profiles",
           "comparison_key", "incompatibilities"]
