"""Workspace-private templates and optional, hash-bound public contributions."""
from datetime import datetime
from typing import Literal

from pydantic import Field

from .common import FileRef, Sha256, StrictModel, utcnow
from .experiment import ModelSpec


class BackboneUploadRequest(StrictModel):
    filename: str = Field(min_length=1, max_length=100)
    source: str = Field(min_length=1, max_length=32768)


class BackboneEntry(StrictModel):
    backbone_id: str = Field(pattern=r"^ub-[a-f0-9]{64}$")
    name: str
    description: str
    author: str
    license: Literal["Apache-2.0"] = "Apache-2.0"
    source_sha256: Sha256
    definition_sha256: Sha256
    parameter_count: int
    node_count: int
    model: ModelSpec
    origin: Literal["private", "community"] = "private"
    source_commit: str | None = None


class BackboneUpload(BackboneEntry):
    publication_id: str = Field(pattern=r"^bbpr-[a-f0-9]{64}$")
    package_sha256: Sha256
    repository: Literal["lab-emi/OpenDPD"] = "lab-emi/OpenDPD"
    branch: str
    directory: str
    files: list[FileRef]
    status: Literal["prepared", "queued", "branch", "push", "pull_request", "submitted", "failed", "interrupted"] = "prepared"
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    consent_at: datetime | None = None
    commit_sha: str | None = None
    pull_request_url: str | None = None
    pull_request_state: str | None = None
    error: str | None = None


class BackboneConsent(StrictModel):
    source_sha256: Sha256
    package_sha256: Sha256
    publish_publicly: Literal[True]
    rights_confirmed: Literal[True]


class BackboneCapability(StrictModel):
    uploads_available: bool = True
    publication_available: bool = False
    reason: str | None = None
    template_version: Literal[1] = 1
    max_source_bytes: Literal[32768] = 32768
    repository: Literal["lab-emi/OpenDPD"] = "lab-emi/OpenDPD"


class BackboneCatalog(StrictModel):
    entries: list[BackboneEntry]
    commit: str | None = None
    warning: str | None = None
