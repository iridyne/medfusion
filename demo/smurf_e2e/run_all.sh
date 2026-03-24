#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/demo/smurf_e2e/config.yaml}"

cd "$ROOT_DIR"

echo "[smurf-e2e] 1/4 prepare mock data"
uv run python demo/smurf_e2e/smurf_e2e.py --config "$CONFIG_PATH" prepare-mock

echo "[smurf-e2e] 2/4 train"
uv run python demo/smurf_e2e/smurf_e2e.py --config "$CONFIG_PATH" train

echo "[smurf-e2e] 3/4 evaluate"
uv run python demo/smurf_e2e/smurf_e2e.py --config "$CONFIG_PATH" evaluate

echo "[smurf-e2e] 4/4 report"
uv run python demo/smurf_e2e/smurf_e2e.py --config "$CONFIG_PATH" report

echo "[smurf-e2e] done. check demo/smurf_e2e/outputs"
