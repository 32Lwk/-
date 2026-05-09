from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from analytics.config import ART_DIR, DATA_PATH, FIG_DIR, FIG_HTML_DIR, LATEX_DIR, ROOT


@dataclass(frozen=True)
class AppSettings:
    """Runtime settings for the FastAPI application."""

    app_name: str = "Kikagaku Analysis API"
    version: str = "1.0.0"
    root_dir: Path = ROOT
    data_path: Path = DATA_PATH
    artifact_dir: Path = ART_DIR
    figure_dir: Path = FIG_DIR
    figure_html_dir: Path = FIG_HTML_DIR
    latex_dir: Path = LATEX_DIR

    @property
    def allowed_artifact_roots(self) -> tuple[Path, ...]:
        return (
            self.artifact_dir,
            self.figure_dir,
            self.figure_html_dir,
            self.latex_dir,
        )


def get_default_settings() -> AppSettings:
    return AppSettings()
