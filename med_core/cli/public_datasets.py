"""CLI wrapper for public dataset quick-validation utilities."""

from __future__ import annotations

from collections.abc import Sequence

from med_core.public_datasets import main as public_datasets_main


def public_datasets(
    argv: Sequence[str] | None = None,
    prog: str = "medfusion public-datasets",
) -> None:
    """Run the public dataset quick-validation CLI."""
    public_datasets_main(argv=argv, prog=prog)
