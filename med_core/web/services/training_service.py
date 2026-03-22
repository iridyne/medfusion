"""Deprecated legacy training service shim.

The real web training path is now owned by ``med_core.web.api.training`` and the
``/api/training/*`` endpoints. This shim remains only to fail loudly and point
callers at the supported path instead of silently running a second fake
execution model.
"""

from __future__ import annotations


class RemovedTrainingServiceError(RuntimeError):
    """Raised when legacy callers try to use the removed training service."""


class TrainingService:
    """Compatibility shim for the removed legacy training service."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise RemovedTrainingServiceError(
            "Legacy med_core.web.services.TrainingService has been removed. "
            "Use the real web training API (/api/training/start, /status, /history) "
            "or the CLI entrypoints (`medfusion train`, `medfusion build-results`).",
        )
