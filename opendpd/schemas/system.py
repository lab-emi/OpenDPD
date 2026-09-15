"""Aggregate telemetry only: never process names, paths, network addresses or users."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import Field

from opendpd.schemas.common import StrictModel


class GpuLoad(StrictModel):
    utilization_percent: Optional[float] = Field(default=None, ge=0, le=100)
    memory_used_bytes: Optional[int] = Field(default=None, ge=0, le=2**60)
    memory_total_bytes: Optional[int] = Field(default=None, gt=0, le=2**60)


class MachineLoad(StrictModel):
    sampled_at: Optional[datetime] = None
    cpu_percent: Optional[float] = Field(default=None, ge=0, le=100)
    memory_percent: Optional[float] = Field(default=None, ge=0, le=100)
    memory_used_bytes: Optional[int] = Field(default=None, ge=0, le=2**60)
    memory_total_bytes: Optional[int] = Field(default=None, gt=0, le=2**60)
    gpu: Optional[GpuLoad] = None


class ResourceStatus(StrictModel):
    load: MachineLoad = Field(default_factory=MachineLoad)
    stale: bool = True
    age_seconds: Optional[float] = Field(default=None, ge=0)


class ServerStatus(StrictModel):
    mode: Literal['web', 'local']
    sampled_at: datetime
    refresh_seconds: int = 5
    active_sessions: Optional[int] = Field(default=None, ge=0)
    active_window_seconds: int = 300
    workspaces: int = Field(ge=0)
    workspace_capacity: Optional[int] = Field(default=None, ge=1)
    waiting_sessions: Optional[int] = Field(default=None, ge=0)
    running_jobs: Optional[int] = Field(ge=0)
    queued_jobs: Optional[int] = Field(ge=0)
    parallel_capacity: Optional[int] = Field(default=None, ge=1)
    api: ResourceStatus
    compute: Optional[ResourceStatus] = None
