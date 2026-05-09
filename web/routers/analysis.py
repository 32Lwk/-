from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, status
from fastapi.responses import FileResponse

from web.config import AppSettings
from web.dependencies import get_analysis_runner, get_settings
from web.schemas import (
    AnalysisRunRequest,
    AnalysisRunResponse,
    AnalysisStatusResponse,
    ArtifactListResponse,
)
from web.services import AnalysisRunner, list_artifacts, resolve_artifact_path

router = APIRouter(prefix="/api", tags=["analysis"])


@router.post(
    "/analysis/run",
    response_model=AnalysisRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def run_analysis(
    payload: AnalysisRunRequest,
    background_tasks: BackgroundTasks,
    runner: AnalysisRunner = Depends(get_analysis_runner),
) -> AnalysisRunResponse:
    snapshot = runner.start(background_tasks)
    message = "analysis pipeline accepted"
    if payload.force:
        message = "analysis pipeline accepted with full rebuild requested"
    return AnalysisRunResponse(
        status=snapshot.status,
        message=message,
        started_at=snapshot.started_at,
    )


@router.get("/analysis/status", response_model=AnalysisStatusResponse)
def analysis_status(
    runner: AnalysisRunner = Depends(get_analysis_runner),
) -> AnalysisStatusResponse:
    snapshot = runner.snapshot()
    return AnalysisStatusResponse(
        status=snapshot.status,
        started_at=snapshot.started_at,
        finished_at=snapshot.finished_at,
        error=snapshot.error,
    )


@router.get("/artifacts", response_model=ArtifactListResponse)
def artifacts(settings: AppSettings = Depends(get_settings)) -> ArtifactListResponse:
    items = list_artifacts(settings)
    return ArtifactListResponse(artifacts=items, count=len(items))


@router.get("/artifacts/{artifact_path:path}")
def artifact(
    artifact_path: str,
    settings: AppSettings = Depends(get_settings),
) -> FileResponse:
    path = resolve_artifact_path(settings, artifact_path)
    return FileResponse(path)
