"""Data-only community packages and explicitly synthetic research fixtures."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from .common import FileRef, Sha256, Slug, StrictModel, utcnow
from .conditions import ConditionSet
from .dataset import DatasetManifest, DatasetOrigin, SignalSpec, SplitSpec


class SyntheticSuiteRequest(StrictModel):
    prefix: str = Field(default="synthetic-research", pattern=r"^[a-z0-9][a-z0-9-]{0,35}$")
    seed: int = Field(default=20260913, ge=0, le=2**32 - 1)
    samples_per_capture: int = Field(default=16384, ge=8192, le=131072, multiple_of=512)
    repeats: int = Field(default=2, ge=1, le=4)


class SyntheticSuite(StrictModel):
    generator: Literal["synthetic-memory-pa-v1"] = "synthetic-memory-pa-v1"
    request: SyntheticSuiteRequest
    datasets: list[DatasetManifest]
    condition_set: ConditionSet
    limitations: list[str]


class CatalogDataset(StrictModel):
    format: Literal["opendpd-dataset-v1"] = "opendpd-dataset-v1"
    dataset_id: Slug
    display_name: str = Field(min_length=1, max_length=200)
    origin: DatasetOrigin
    description: str = Field(min_length=1, max_length=2000)
    license: Literal["CC0-1.0", "CC-BY-4.0"]
    attribution: str = Field(min_length=1, max_length=200)
    signal: SignalSpec
    split: SplitSpec
    n_samples: int = Field(ge=1, le=1_000_000)
    data: FileRef
    source_raw_sha256: Sha256
    simulation: dict[str, object] | None = None

    @model_validator(mode="after")
    def _data_only(self):
        if self.data.path != "data.csv":
            raise ValueError("the only data file is data.csv")
        if self.origin == DatasetOrigin.synthetic and "synthetic" not in self.display_name.lower():
            raise ValueError("synthetic datasets must say synthetic in their display name")
        if self.simulation is not None and self.origin != DatasetOrigin.synthetic:
            raise ValueError("a simulated dataset cannot be labelled measured or unknown")
        return self


class DatasetPublicationDraft(StrictModel):
    dataset_id: Slug
    description: str = Field(min_length=1, max_length=2000)
    license: Literal["CC0-1.0", "CC-BY-4.0"]
    attribution: str = Field(min_length=1, max_length=200)


class DatasetPublication(StrictModel):
    publication_id: str = Field(pattern=r"^dspr-[a-f0-9]{64}$")
    dataset_id: Slug
    package_sha256: Sha256
    source_manifest_sha256: Sha256
    repository: str = "lab-emi/OpenDPD"
    branch: str
    directory: str
    files: list[FileRef]
    catalog: CatalogDataset
    status: Literal["prepared", "queued", "branch", "push", "pull_request", "submitted", "failed", "interrupted"] = "prepared"
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    consent_at: datetime | None = None
    commit_sha: str | None = None
    pull_request_url: str | None = None
    pull_request_state: str | None = None
    error: str | None = None
    contact_email: Literal["emi.lab@outlook.com"] = "emi.lab@outlook.com"


class PublicationConsent(StrictModel):
    package_sha256: Sha256
    publish_publicly: Literal[True]
    rights_confirmed: Literal[True]


class PublicationCapability(StrictModel):
    available: bool
    repository: str = "lab-emi/OpenDPD"
    reason: str | None = None
    contact_email: str = "emi.lab@outlook.com"
