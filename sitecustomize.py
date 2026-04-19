"""Cross-platform runtime tweaks for the MedFusion workspace.

The project's training-control semantics use pause/resume/terminate concepts
that map naturally to POSIX ``SIGSTOP`` / ``SIGCONT`` / ``SIGTERM``. Windows
does not expose ``SIGSTOP`` or ``SIGCONT`` on the standard ``signal`` module,
which makes both tests and higher-level control routing inconsistent.

Normalizing the missing names at interpreter startup keeps route code, worker
code, and tests aligned without adding platform branches everywhere.
"""

from __future__ import annotations

import signal

if not hasattr(signal, "SIGSTOP"):
    setattr(signal, "SIGSTOP", 19)

if not hasattr(signal, "SIGCONT"):
    setattr(signal, "SIGCONT", 18)
