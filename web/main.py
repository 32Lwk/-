from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from web.config import get_default_settings
from web.exceptions import ArtifactNotFoundError, PipelineAlreadyRunningError
from web.routers.analysis import router as analysis_router
from web.routers.health import router as health_router
from web.schemas import ErrorResponse
from web.services import AnalysisRunner, initialize_runtime


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_default_settings()
    initialize_runtime(settings)
    app.state.settings = settings
    app.state.analysis_runner = AnalysisRunner()
    yield


def create_app() -> FastAPI:
    settings = get_default_settings()
    fastapi_app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=lifespan,
    )
    fastapi_app.include_router(health_router)
    fastapi_app.include_router(analysis_router)
    fastapi_app.add_exception_handler(PipelineAlreadyRunningError, pipeline_already_running_handler)
    fastapi_app.add_exception_handler(ArtifactNotFoundError, artifact_not_found_handler)
    return fastapi_app


async def pipeline_already_running_handler(
    request: Request,
    exc: PipelineAlreadyRunningError,
) -> JSONResponse:
    response = ErrorResponse(detail=str(exc), context={"path": str(request.url.path)})
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content=response.model_dump(mode="json"),
    )


async def artifact_not_found_handler(
    request: Request,
    exc: ArtifactNotFoundError,
) -> JSONResponse:
    response = ErrorResponse(detail=str(exc), context={"path": str(request.url.path)})
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=response.model_dump(mode="json"),
    )


app = create_app()
