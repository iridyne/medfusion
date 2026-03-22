"""Deprecated legacy script.

This file targets the pre-``med_core.web`` backend layout and is no longer a
valid test entrypoint for the current MVP. Use:

- ``tests/test_web_api_minimal.py`` for the real web training chain
- ``tests/test_workflow_api.py`` for workflow feature gating
"""


if __name__ == "__main__":
    raise SystemExit(
        "Deprecated script: use `uv run pytest tests/test_web_api_minimal.py "
        "tests/test_workflow_api.py` instead.",
    )
