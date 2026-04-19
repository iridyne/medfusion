"""Local Python worker registry for real training subprocesses.

This module is intentionally small: the API/BFF layer should not own raw
``subprocess.Popen`` instances directly. Instead, training process lifecycle is
tracked here so the rest of the application can evolve toward external queue /
GPU worker deployments without changing route contracts first.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path


class LocalTrainingWorkerRegistry:
    """Track locally launched Python training workers."""

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[str]] = {}

    def start(
        self,
        *,
        job_id: str,
        command: list[str],
        cwd: Path,
        log_path: Path,
        env: dict[str, str],
    ) -> subprocess.Popen[str]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,
            )
        except Exception:
            log_handle.close()
            raise

        self._processes[job_id] = process
        return process

    def get(self, job_id: str) -> subprocess.Popen[str] | None:
        return self._processes.get(job_id)

    def signal(self, job_id: str, process_signal: signal.Signals) -> bool:
        process = self._processes.get(job_id)
        if process is None or process.poll() is not None:
            return False

        try:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, process_signal)
            else:
                process.send_signal(int(process_signal))
        except (ProcessLookupError, OSError, ValueError):
            return False

        return True

    def remove(self, job_id: str) -> subprocess.Popen[str] | None:
        return self._processes.pop(job_id, None)


local_training_worker_registry = LocalTrainingWorkerRegistry()
