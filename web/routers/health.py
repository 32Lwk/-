from __future__ import annotations

from fastapi import APIRouter, Depends

from web.config import AppSettings
from web.dependencies import get_settings
from web.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: AppSettings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.version,
        data_file_exists=settings.data_path.is_file(),
    )
