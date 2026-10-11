"""MATLINK v1: bounded metadata and named MATLAB actions, never executable code."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from .common import Slug, StrictModel

MATLAB_KEYWORDS = frozenset("break case catch classdef continue else elseif end for function global if otherwise parfor persistent return spmd switch try while".split())
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,62}$")]


def matlab_identifier(value: str) -> str:
    if value in MATLAB_KEYWORDS:
        raise ValueError("Choose a MATLAB variable name, not a language keyword.")
    return value


class MatlabVariable(StrictModel):
    name: Identifier
    class_name: str = Field(min_length=1, max_length=128)
    size: list[int] = Field(min_length=2, max_length=32)
    complex: bool = False
    eligible: bool = False
    n_samples: int = Field(default=0, ge=0, le=2**53)

    _identifier = field_validator("name")(matlab_identifier)

    @field_validator("size")
    @classmethod
    def _dimensions(cls, value):
        if any(n < 0 or n > 2**53 for n in value):
            raise ValueError("Variable dimensions must be nonnegative and finite.")
        return value

    @model_validator(mode="after")
    def _eligible_shape(self):
        if self.eligible:
            if self.class_name not in ("single", "double") or len(self.size) != 2:
                raise ValueError("I/Q import requires a single or double vector.")
            rows, cols = self.size
            expected = rows * cols if rows == 1 or cols == 1 else 0
            if not expected or self.n_samples != expected:
                raise ValueError("Eligible I/Q size and sample count must agree.")
        return self


class MatlinkHeartbeat(StrictModel):
    variables: list[MatlabVariable] = Field(default_factory=list, max_length=256)

    @field_validator("variables")
    @classmethod
    def _unique(cls, values):
        if len({v.name for v in values}) != len(values):
            raise ValueError("Variable names must be unique.")
        return values


class MatlinkConnect(MatlinkHeartbeat):
    label: str = Field(default="MATLAB", min_length=1, max_length=120)
    release: str = Field(default="", max_length=40)
    capabilities: list[Literal["dataset_export", "result_bundle"]] = Field(default_factory=list, max_length=2)


class MatlinkConnection(StrictModel):
    client_id: str
    bridge_token: str
    lease_seconds: int
    protocol_version: Literal[1] = 1


class CreateDemoPayload(StrictModel):
    pass


class ImportIQPayload(StrictModel):
    input: Identifier
    output: Identifier
    name: Slug | Literal[""] = ""
    sample_rate_mhz: float = Field(gt=0, allow_inf_nan=False)
    bandwidth_mhz: float = Field(gt=0, allow_inf_nan=False)
    # PSD segment length and the evaluation reset interval; deliberately without a default.
    segment_samples: int = Field(ge=2, le=1_048_576)
    origin: Literal["unknown", "measured", "synthetic"] = "unknown"

    _identifier = field_validator("input", "output")(matlab_identifier)

    @model_validator(mode="after")
    def _pair(self):
        if self.input == self.output:
            raise ValueError("Choose separate input and output variables.")
        if self.bandwidth_mhz > self.sample_rate_mhz:
            raise ValueError("Bandwidth must not exceed the sample rate.")
        return self


class ImportResultPayload(StrictModel):
    run_id: Slug
    variable: Identifier | Literal[""] = ""
    bundle: bool = False

    _identifier = field_validator("variable")(matlab_identifier)


class ImportDatasetPayload(StrictModel):
    dataset_id: Slug


class OpenVariablePayload(StrictModel):
    variable: Identifier

    _identifier = field_validator("variable")(matlab_identifier)


ACTION_PAYLOADS = {
    "create_demo": CreateDemoPayload,
    "import_iq": ImportIQPayload,
    "import_result": ImportResultPayload,
    "import_dataset": ImportDatasetPayload,
    "open_variable": OpenVariablePayload,
}
MatlinkAction = Literal["create_demo", "import_iq", "import_result", "import_dataset", "open_variable"]
MatlinkStatus = Literal["waiting", "queued", "succeeded", "failed"]


class MatlinkRequest(StrictModel):
    client_id: str = Field(min_length=1, max_length=64)
    action: MatlinkAction
    payload: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def _payload(self):
        parsed = ACTION_PAYLOADS[self.action].model_validate(self.payload)
        object.__setattr__(self, "payload", parsed.model_dump())
        return self


class MatlinkCompletion(StrictModel):
    status: Literal["succeeded", "failed"]
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _bounded(self):
        try:
            serialized = json.dumps(self.result, allow_nan=False)
        except (TypeError, ValueError):
            raise ValueError("Transfer results must contain finite JSON metadata.") from None
        if len(serialized.encode("utf-8")) > 32768:
            raise ValueError("Transfer result metadata must not exceed 32 KiB.")
        if self.status == "failed" and not self.error:
            raise ValueError("A failed transfer must explain the error.")
        if self.status == "succeeded" and self.error is not None:
            raise ValueError("Successful transfers cannot carry an error.")
        return self


class MatlinkTransfer(StrictModel):
    request_id: str
    client_id: str
    action: MatlinkAction
    status: MatlinkStatus
    payload: dict[str, Any]
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class MatlinkSession(StrictModel):
    client_id: str
    label: str
    release: str
    connected: bool
    last_seen: datetime
    variables: list[MatlabVariable]
    pending_count: int
    capabilities: list[Literal["dataset_export", "result_bundle"]] = Field(default_factory=list)


class MatlinkState(StrictModel):
    available: Literal[True] = True
    protocol_version: Literal[1] = 1
    workspace: str
    sessions: list[MatlinkSession]
    transfers: list[MatlinkTransfer]


class MatlinkPoll(StrictModel):
    requests: list[MatlinkTransfer]
    lease_seconds: int


class MatlinkDisconnected(StrictModel):
    disconnected: Literal[True] = True
