"""Primitive contract types shared by every schema module."""

from __future__ import annotations

import math
import posixpath
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1

# Identifiers used in paths, URLs and databases: no spaces, no path separators.
Slug = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    """Base for all contracts: unknown fields are errors, not silently dropped."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=False)


class EvidenceType(str, Enum):
    """What kind of evidence a result represents. Never inferred, always declared."""

    pa_modeling = "pa_modeling"        # PA behavioural model vs measured PA output
    dpd_surrogate = "dpd_surrogate"    # DPD evaluated through a learned PA surrogate
    dpd_measured = "dpd_measured"      # DPD evaluated with a real PA measurement


class Severity(str, Enum):
    error = "error"
    warning = "warning"
    info = "info"


class BetterDirection(str, Enum):
    lower = "lower"
    higher = "higher"


class MetricStatus(str, Enum):
    ok = "ok"
    not_applicable = "not_applicable"      # metric undefined for this data/profile
    missing_reference = "missing_reference"  # no valid reference signal
    invalid = "invalid"                    # computed but non-finite or out of domain
    failed = "failed"                      # computation raised


class MetricValue(StrictModel):
    """One formal metric. ``value`` is only present when ``status`` is ``ok``
    and is always finite; every other status carries a reason. This is how
    "null / not applicable / failed / infinite" get explicit meanings and how
    zero-filling of missing metrics is made impossible."""

    name: str = Field(min_length=1)
    value: Optional[float] = None
    unit: str = Field(min_length=1)
    better: BetterDirection
    status: MetricStatus = MetricStatus.ok
    reason: Optional[str] = None

    @model_validator(mode="after")
    def _consistent(self) -> "MetricValue":
        if self.status == MetricStatus.ok:
            if self.value is None or not math.isfinite(self.value):
                raise ValueError(f"metric {self.name}: status ok requires a finite value")
        else:
            if self.value is not None:
                raise ValueError(f"metric {self.name}: status {self.status.value} must not carry a value")
            if not self.reason:
                raise ValueError(f"metric {self.name}: status {self.status.value} requires a reason")
        return self


class SoftwareProvenance(StrictModel):
    opendpd_version: str
    python_version: str
    platform: str
    torch_version: Optional[str] = None
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None


class FileRef(StrictModel):
    """A file inside a known container (workspace, run or dataset directory).
    Paths are relative POSIX paths; absolute paths and ``..`` are rejected so
    a manifest can never point outside its container."""

    path: str = Field(min_length=1)
    sha256: Optional[Sha256] = None
    size_bytes: Optional[int] = Field(default=None, ge=0)

    @field_validator("path")
    @classmethod
    def _relative_and_contained(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("use POSIX separators in FileRef.path")
        if posixpath.isabs(value) or value.startswith("~"):
            raise ValueError("FileRef.path must be relative")
        parts = value.split("/")
        if any(part in ("", ".", "..") for part in parts):
            raise ValueError("FileRef.path must not contain empty, '.' or '..' segments")
        if ":" in parts[0] and len(parts[0]) == 2:  # C: style drive prefix
            raise ValueError("FileRef.path must not contain a drive prefix")
        return value
