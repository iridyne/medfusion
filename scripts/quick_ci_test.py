#!/usr/bin/env python3
"""
Compatibility wrapper for the repository quick validation entrypoint.
"""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    print(
        "scripts/quick_ci_test.py 已降级为兼容入口，转调 scripts/full_regression.sh --quick"
    )
    return subprocess.call(
        ["bash", "scripts/full_regression.sh", "--quick", *sys.argv[1:]],
        cwd=repo_root,
    )


if __name__ == "__main__":
    sys.exit(main())
