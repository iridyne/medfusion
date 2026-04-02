#!/usr/bin/env bash
# 统一验证入口：快速检查 / CI 对齐 / 完整回归

set -euo pipefail

MODE="full"
QUICK_PY_LINT_TARGETS=(
  tests/test_output_layout.py
  tests/test_build_results.py
  tests/test_validation_workflow.py
)
QUICK_TEST_TARGETS=(
  tests/test_output_layout.py
  tests/test_build_results.py
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/full_regression.sh [--mode quick|ci|full]
  bash scripts/full_regression.sh --quick
  bash scripts/full_regression.sh --ci
  bash scripts/full_regression.sh --full
  bash scripts/full_regression.sh --help

Modes:
  quick   Fast local validation for everyday development.
          Runs:
            - uv sync --extra dev
            - bash -n scripts/full_regression.sh
            - uv run ruff check <focused validation python files>
            - uv run ruff format --check <focused validation python files>
            - uv run pytest tests/test_output_layout.py tests/test_build_results.py -v
          Notes:
            - quick is intentionally scoped to the current validation workflow files.
            - It does not attempt to clean up all historical repo-wide formatting debt.

  ci      Closest local approximation of GitHub CI.
          Runs:
            - uv sync --extra dev --extra web
            - uv run pytest tests/ -v --cov=med_core --cov-report=xml --cov-report=term \
                --ignore=tests/test_config_validation.py \
                --ignore=tests/test_export.py
            - uv run pytest tests/test_end_to_end.py -v --tb=short
            - bash test/smoke.sh
          Notes:
            - The CI-aligned suite intentionally excludes:
              tests/test_config_validation.py
              tests/test_export.py

  full    Broader local regression pass.
          Runs:
            - bash scripts/local_ci_test.sh

Why this exists:
  This is the repository's script-based validation entrypoint.
  It replaces “remember a bunch of commands in your head” with one real command.
EOF
}

run() {
  echo
  echo "==> $*"
  "$@"
}

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --quick)
      MODE="quick"
      shift
      ;;
    --ci)
      MODE="ci"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo >&2
      usage >&2
      exit 2
      ;;
  esac
done

ensure_uv

case "$MODE" in
  quick)
    run uv sync --extra dev
    run bash -n scripts/full_regression.sh
    run uv run ruff check "${QUICK_PY_LINT_TARGETS[@]}"
    run uv run ruff format --check "${QUICK_PY_LINT_TARGETS[@]}"
    run uv run pytest "${QUICK_TEST_TARGETS[@]}" -v
    ;;
  ci)
    run uv sync --extra dev --extra web
    run uv run pytest tests/ -v --cov=med_core --cov-report=xml --cov-report=term \
      --ignore=tests/test_config_validation.py \
      --ignore=tests/test_export.py
    run uv run pytest tests/test_end_to_end.py -v --tb=short
    run bash test/smoke.sh
    ;;
  full)
    run bash scripts/local_ci_test.sh
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac
