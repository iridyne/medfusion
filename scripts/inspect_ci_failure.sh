#!/usr/bin/env bash
# Inspect recent GitHub Actions CI failures via gh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORKFLOW="ci.yml"
RUN_ID=""
LIMIT="${CI_RUN_LIMIT:-10}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/inspect_ci_failure.sh [--workflow ci.yml] [--run-id <id>] [--limit 10]

What it does:
  - lists recent runs for the target workflow
  - if --run-id is omitted, picks the latest failed/cancelled/in-progress run
  - prints failed logs via `gh run view --log-failed`

Requirements:
  - GitHub CLI (`gh`) installed and authenticated
EOF
}

ensure_gh() {
  if ! command -v gh >/dev/null 2>&1; then
    echo "gh not found. Install GitHub CLI first." >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workflow)
      WORKFLOW="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --limit)
      LIMIT="${2:-}"
      shift 2
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

ensure_gh

echo
echo "Recent runs for workflow: $WORKFLOW"
gh run list --workflow "$WORKFLOW" --limit "$LIMIT"

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(
    gh run list \
      --workflow "$WORKFLOW" \
      --limit "$LIMIT" \
      --json databaseId,conclusion,status \
      --jq 'map(select(.conclusion == "failure" or .conclusion == "cancelled" or .status == "in_progress")) | .[0].databaseId'
  )"
fi

if [[ -z "$RUN_ID" || "$RUN_ID" == "null" ]]; then
  echo
  echo "No failed/cancelled/in-progress runs found in the latest $LIMIT entries."
  exit 0
fi

echo
echo "Inspecting failed logs for run: $RUN_ID"
gh run view "$RUN_ID" --log-failed
