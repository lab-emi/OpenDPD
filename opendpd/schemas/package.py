"""Reproducible experiment packages: what a zip contains and what it does not."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import Field

from .common import SCHEMA_VERSION, Sha256, Slug, SoftwareProvenance, StrictModel, utcnow
from .dataset import DatasetSourceKind
from .experiment import TaskType

PACKAGE_VERSION = 1
PackageKind = Literal["full", "share"]


class PackageFile(StrictModel):
    """A member of the archive; the hash is verified before anything is imported."""

    path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(ge=0)


class PackageDataset(StrictModel):
    dataset_id: Slug
    raw_sha256: Optional[Sha256] = None
    preprocessing_version: str
    split_version: str
    source_kind: DatasetSourceKind
    builtin_name: Optional[str] = None
    included: bool                              # raw data and the used version travel with the package
    how_to_obtain: str = Field(min_length=1)    # when not included: what to ask for and how to check it


class PackageReference(StrictModel):
    """Another run this one depends on (PA surrogate, DPD model); it always travels under refs/ with its checkpoint."""

    run_id: Slug
    role: Literal["pa_surrogate", "dpd_model"]
    checkpoint_sha256: Sha256


class PackageManifest(StrictModel):
    schema_version: int = SCHEMA_VERSION
    package_version: int = PACKAGE_VERSION
    kind: PackageKind
    created_at: datetime = Field(default_factory=utcnow)
    opendpd_version: str
    software: SoftwareProvenance
    run_id: Slug
    task: TaskType
    config_sha256: Sha256
    seed: int
    result_id: Optional[Slug] = None
    metric_profile_id: Optional[Slug] = None
    dataset: PackageDataset
    references: List[PackageReference] = Field(default_factory=list)
    files: List[PackageFile] = Field(default_factory=list)
    reproduction: Dict[str, str] = Field(default_factory=dict)   # named commands, relative to the imported workspace
    redaction: List[str] = Field(default_factory=list)           # what a share package left out or rewrote
    missing: List[str] = Field(default_factory=list)             # what is needed to re-evaluate and is not inside
    retraining_note: str = Field(min_length=1)


class ImportReport(StrictModel):
    package_version: int
    kind: PackageKind
    run_id: Slug
    imported_runs: List[Slug] = Field(default_factory=list)      # the run and every included reference
    dataset_status: Literal["imported", "existing", "registered_builtin", "missing"]
    dataset_id: Slug
    missing: List[str] = Field(default_factory=list)
    evaluate_command: str = Field(min_length=1)
    note: str = Field(min_length=1)
