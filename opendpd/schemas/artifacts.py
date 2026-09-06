"""Artifact manifests: registered files a run produced, with hashes."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import Field, model_validator

from .common import SCHEMA_VERSION, FileRef, Slug, StrictModel, utcnow


class ArtifactKind(str, Enum):
    checkpoint = "checkpoint"
    log_history = "log_history"
    log_best = "log_best"
    worker_log = "worker_log"
    dpd_output = "dpd_output"
    plot = "plot"
    config = "config"
    provenance = "provenance"
    events = "events"
    result = "result"
    export = "export"
    other = "other"


class Artifact(StrictModel):
    artifact_id: Slug
    kind: ArtifactKind
    file: FileRef
    created_at: datetime = Field(default_factory=utcnow)
    description: Optional[str] = None
    required: bool = False      # a run cannot be "succeeded" without its required artifacts


class ArtifactManifest(StrictModel):
    schema_version: int = SCHEMA_VERSION
    run_id: Slug
    artifacts: List[Artifact] = Field(default_factory=list)
    complete: bool = False

    def by_kind(self, kind: ArtifactKind) -> List[Artifact]:
        return [a for a in self.artifacts if a.kind == kind]

    @model_validator(mode="after")
    def _complete_means_verified(self) -> "ArtifactManifest":
        ids = [a.artifact_id for a in self.artifacts]
        if len(ids) != len(set(ids)):
            raise ValueError("artifact ids must be unique within a run")
        required_verified = all(a.file.sha256 is not None for a in self.artifacts if a.required)
        if self.complete and not required_verified:
            raise ValueError("complete manifests need a hash for every required artifact")
        return self
