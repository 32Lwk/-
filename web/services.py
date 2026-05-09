from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from fastapi import BackgroundTasks

from analytics.config import ensure_all_dirs
from analytics.pipeline import run_pipeline
from web.config import AppSettings
from web.exceptions import ArtifactNotFoundError, PipelineAlreadyRunningError
from web.schemas import ArtifactSummary, PipelineState


@dataclass
class PipelineSnapshot:
    status: PipelineState = PipelineState.idle
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class AnalysisRunner:
    """Coordinates background execution of the existing analysis pipeline."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._snapshot = PipelineSnapshot()

    def snapshot(self) -> PipelineSnapshot:
        with self._lock:
            return PipelineSnapshot(
                status=self._snapshot.status,
                started_at=self._snapshot.started_at,
                finished_at=self._snapshot.finished_at,
                error=self._snapshot.error,
            )

    def start(self, background_tasks: BackgroundTasks) -> PipelineSnapshot:
        with self._lock:
            if self._snapshot.status == PipelineState.running:
                raise PipelineAlreadyRunningError("analysis pipeline is already running")

            started_at = datetime.now(timezone.utc)
            self._snapshot = PipelineSnapshot(
                status=PipelineState.running,
                started_at=started_at,
                finished_at=None,
                error=None,
            )
            background_tasks.add_task(self._run_pipeline)
            return PipelineSnapshot(
                status=self._snapshot.status,
                started_at=self._snapshot.started_at,
                finished_at=self._snapshot.finished_at,
                error=self._snapshot.error,
            )

    def _run_pipeline(self) -> None:
        try:
            run_pipeline()
        except Exception as exc:
            with self._lock:
                self._snapshot.status = PipelineState.failed
                self._snapshot.finished_at = datetime.now(timezone.utc)
                self._snapshot.error = str(exc)
            raise

        with self._lock:
            self._snapshot.status = PipelineState.succeeded
            self._snapshot.finished_at = datetime.now(timezone.utc)
            self._snapshot.error = None


def initialize_runtime(settings: AppSettings) -> None:
    """Prepare directories needed by both the API and analysis pipeline."""

    ensure_all_dirs()
    settings.latex_dir.mkdir(parents=True, exist_ok=True)


def list_artifacts(settings: AppSettings) -> list[ArtifactSummary]:
    artifacts: list[ArtifactSummary] = []
    for root in settings.allowed_artifact_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            artifacts.append(_to_artifact_summary(settings.root_dir, path))

    for path in (settings.root_dir / "final_report.md", settings.root_dir / "report.md"):
        if path.is_file():
            artifacts.append(_to_artifact_summary(settings.root_dir, path))

    return sorted(artifacts, key=lambda item: item.path)


def resolve_artifact_path(settings: AppSettings, requested_path: str) -> Path:
    candidate = (settings.root_dir / requested_path).resolve()
    allowed_roots = [root.resolve() for root in settings.allowed_artifact_roots]
    allowed_roots.append((settings.root_dir / "final_report.md").resolve())
    allowed_roots.append((settings.root_dir / "report.md").resolve())

    if not candidate.is_file() or not _is_allowed_path(candidate, allowed_roots):
        raise ArtifactNotFoundError(f"artifact not found: {requested_path}")
    return candidate


def _is_allowed_path(candidate: Path, allowed_roots: list[Path]) -> bool:
    for root in allowed_roots:
        if candidate == root:
            return True
        if root.is_dir():
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            return True
    return False


def _to_artifact_summary(root_dir: Path, path: Path) -> ArtifactSummary:
    stat = path.stat()
    return ArtifactSummary(
        path=path.relative_to(root_dir).as_posix(),
        size_bytes=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc),
    )
