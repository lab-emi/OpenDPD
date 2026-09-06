"""Metric profiles: the versioned definition behind every score."""

from __future__ import annotations

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


class MetricProfile(StrictModel):
    profile_id: Slug
    version: int = Field(ge=1)
    description: str = Field(min_length=1)
    metrics: List[MetricDefinition] = Field(min_length=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)   # FFT, window, segments, normalisation
    frozen: bool = False        # historical profiles never change numerically
    deprecated: bool = False

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
