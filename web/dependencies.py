from __future__ import annotations

from fastapi import Request

from web.config import AppSettings
from web.services import AnalysisRunner


def get_settings(request: Request) -> AppSettings:
    return request.app.state.settings


def get_analysis_runner(request: Request) -> AnalysisRunner:
    return request.app.state.analysis_runner
