from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="Service health status.")
    app_name: str
    version: str
    data_file_exists: bool


class PipelineState(StrEnum):
    idle = "idle"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class AnalysisRunRequest(BaseModel):
    force: bool = Field(
        default=False,
        description="Reserved for clients that want to explicitly request a full rebuild.",
    )


class AnalysisRunResponse(BaseModel):
    status: PipelineState
    message: str
    started_at: datetime | None = None


class AnalysisStatusResponse(BaseModel):
    status: PipelineState
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class ArtifactSummary(BaseModel):
    path: str
    size_bytes: int
    modified_at: datetime


class ArtifactListResponse(BaseModel):
    artifacts: list[ArtifactSummary]
    count: int


class ErrorResponse(BaseModel):
    detail: str
    context: dict[str, Any] = Field(default_factory=dict)
