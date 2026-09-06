"""Metric registry: every score in a result comes from a versioned profile here.

Protected path (``.github/workflows/protected-paths.yml``): changing what a
profile computes is a scientific-semantics change and needs a new version.
"""

from .registry import DEFAULT_PROFILE_ID, PROFILES, evaluate, get_profile, list_profiles
from .compare import comparison_key, incompatibilities

__all__ = ["DEFAULT_PROFILE_ID", "PROFILES", "evaluate", "get_profile", "list_profiles",
           "comparison_key", "incompatibilities"]
