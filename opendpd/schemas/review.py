"""Read-only RF review and replayable figure contracts; no metric algorithms."""

from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import ConfigDict, Field, model_validator

from .common import Sha256, Slug, StrictModel, utcnow
from .dataset import SignalSpec
from .metrics import MetricProfile
from .results import EvaluationResult


class ReviewFact(StrictModel):
    key: str
    label: str
    value: Optional[str] = None
    unit: Optional[str] = None
    source: str


class ReviewBand(StrictModel):
    label: str
    role: Literal["main", "adjacent", "subchannel"]
    edges_hz: Tuple[float, float]
    available: bool
    reason: Optional[str] = None


class ReviewContext(StrictModel):
    version: Literal["rf-review-v1"] = "rf-review-v1"
    result: EvaluationResult
    signal: Optional[SignalSpec] = None
    signal_source: str
    profile: MetricProfile
    facts: List[ReviewFact]
    bands: List[ReviewBand] = Field(default_factory=list)
    band_note: str
    provenance: Dict[str, Optional[str]] = Field(default_factory=dict)


class FigureTrace(StrictModel):
    run_id: Slug
    trace_name: str = Field(min_length=1, max_length=500)
    color: str = Field(default="#2563EB", pattern=r"^#[0-9a-fA-F]{6}$")
    dash: Literal["solid", "dash", "dot", "dashdot", "longdash"] = "solid"
    visible: bool = True


class FigurePanel(StrictModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)
    kind: Literal["spectrum", "amam", "ampm", "power_scan", "error_distribution"] = "spectrum"
    traces: List[FigureTrace] = Field(min_length=1, max_length=64)
    # Spectrum ranges are MHz (Hz axis) or cycles/sample; AM axes use stored amplitude units.
    x_range: Optional[Tuple[float, float]] = None
    y_range: Optional[Tuple[float, float]] = None
    show_bands: bool = True
    cursor_x: Optional[float] = None

    @model_validator(mode="after")
    def _ranges(self):
        for bounds in (self.x_range, self.y_range):
            if bounds is not None and bounds[0] >= bounds[1]:
                raise ValueError("axis ranges must be strictly increasing")
        ids = [(t.run_id, t.trace_name) for t in self.traces]
        if len(ids) != len(set(ids)):
            raise ValueError("a trace may appear only once per panel")
        if not any(t.visible for t in self.traces):
            raise ValueError("a panel needs at least one visible trace")
        if self.kind == 'power_scan' and len({t.trace_name for t in self.traces}) != 1:
            raise ValueError('a power scan panel must use one RF metric')
        return self


class FigureSpec(StrictModel):
    version: Literal["figure-v1", "figure-v2"] = "figure-v1"
    title: str = Field(min_length=1, max_length=200)
    reference_run_id: Slug
    mode: Literal["same_condition", "cross_condition"] = "same_condition"
    # Explicit profiles, including in comparisons: changing a view never changes evaluation.
    profiles: Dict[Slug, Slug]
    panels: List[FigurePanel] = Field(min_length=1, max_length=4)
    width: Literal["single_column", "double_column"] = "double_column"

    @model_validator(mode="after")
    def _references(self):
        if not 1 <= len(self.profiles) <= 8 or self.reference_run_id not in self.profiles:
            raise ValueError("reference must belong to the 1–8 selected results")
        if any(t.run_id not in self.profiles for p in self.panels for t in p.traces):
            raise ValueError("every trace must reference a selected result")
        if self.version == 'figure-v1' and any(p.kind in {'power_scan', 'error_distribution'} for p in self.panels):
            raise ValueError('power scans and residual distributions require figure-v2')
        return self


class FigureBinding(StrictModel):
    run_id: Slug
    result_id: Slug
    result_sha256: Sha256
    files: Dict[str, Sha256]
    review: ReviewContext


class SavedFigure(StrictModel):
    figure_id: Slug
    created_at: datetime = Field(default_factory=utcnow)
    spec: FigureSpec
    bindings: List[FigureBinding]


class FigureSourcesRequest(StrictModel):
    profiles: Dict[Slug, Slug] = Field(min_length=1, max_length=8)


class FigureSource(StrictModel):
    run_id: Slug
    kind: str
    trace_name: str
    role: str
    source: str


class FigureSources(StrictModel):
    sources: List[FigureSource]
    missing: List[str]


class FigurePreview(StrictModel):
    figure: SavedFigure
    plots: Dict[str, dict]
