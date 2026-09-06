"""Metric profiles: the versioned definition behind every score."""

from __future__ import annotations

from enum import Enum

from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator

from .common import BetterDirection, Slug, StrictModel


class MetricDefinition(StrictModel):
    name: str = Field(min_length=1)          # key used in MetricValue.name, e.g. "NMSE"
    display_name: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    better: BetterDirection
    formula: str = Field(min_length=1)       # human-readable definition
    aggregation: str = Field(min_length=1)   # e.g. "mean of per-segment dB"
    requires_reference: bool = True
    notes: Optional[str] = None


class ProfileValidation(str, Enum):
    """How a profile's numbers have been checked. The GUI offers only profiles past ``pending_cross_validation``;
    the CLI and the API compute every profile and say which status it has."""

    golden = "golden"                                          # pinned to frozen references of the historical code
    analytic = "analytic"                                      # closed-form references on deterministic signals
    pending_cross_validation = "pending_cross_validation"      # implemented and tested; the independent backend comparison has not run
    cross_validated = "cross_validated"                        # agreed with an independent implementation on the same signals


class MetricProfile(StrictModel):
    profile_id: Slug
    version: int = Field(ge=1)
    description: str = Field(min_length=1)
    metrics: List[MetricDefinition] = Field(min_length=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)   # FFT, window, segments, normalisation
    frozen: bool = False        # historical profiles never change numerically
    deprecated: bool = False
    validation: ProfileValidation = ProfileValidation.pending_cross_validation   # a new profile is hidden until it says otherwise

    @model_validator(mode="after")
    def _unique_metric_names(self) -> "MetricProfile":
        names = [m.name for m in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("metric names within a profile must be unique")
        return self

    def metric(self, name: str) -> MetricDefinition:
        for m in self.metrics:
            if m.name == name:
                return m
        raise KeyError(name)
