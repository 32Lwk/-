from __future__ import annotations


class PipelineAlreadyRunningError(RuntimeError):
    """Raised when a pipeline run is requested while another run is active."""


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when a requested generated artifact cannot be served safely."""
