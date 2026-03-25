#!/usr/bin/env python3
"""Backward-compatible wrapper for public dataset preparation."""

from __future__ import annotations

import sys

from med_core.public_datasets import main


if __name__ == "__main__":
    main(argv=["prepare", *sys.argv[1:]], prog="prepare_public_dataset.py")
