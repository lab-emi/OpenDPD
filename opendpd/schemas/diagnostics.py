"""Dataset Doctor reports."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator

from .common import SCHEMA_VERSION, Severity, Sha256, Slug, StrictModel, utcnow


class DiagnosticItem(StrictModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    severity: Severity
    title: str = Field(min_length=1)
    message: str = Field(min_length=1)
    evidence: Dict[str, Any] = Field(default_factory=dict)   # the numbers behind the finding
    suggestion: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    blocking: bool = False   # blocks evaluation that would be unreliable

    @model_validator(mode="after")
    def _blocking_needs_error(self) -> "DiagnosticItem":
        if self.blocking and self.severity != Severity.error:
            raise ValueError("only error-level diagnostics can block")
        return self


class DiagnosticReport(StrictModel):
    schema_version: int = SCHEMA_VERSION
    report_id: Slug
    dataset_id: Slug
    dataset_raw_sha256: Optional[Sha256] = None
    doctor_version: str = "dataset-doctor-v1"
    generated_at: datetime = Field(default_factory=utcnow)
    items: List[DiagnosticItem] = Field(default_factory=list)
    evaluation_blocked: bool = False

    def counts(self) -> Dict[str, int]:
        return {s.value: sum(1 for i in self.items if i.severity == s) for s in Severity}

    @model_validator(mode="after")
    def _blocked_matches_items(self) -> "DiagnosticReport":
        if self.evaluation_blocked != any(i.blocking for i in self.items):
            raise ValueError("evaluation_blocked must equal any(item.blocking)")
        return self
