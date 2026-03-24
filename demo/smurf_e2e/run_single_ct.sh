#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-fast}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

case "$MODE" in
  fast)
    CONFIG_PATH="${ROOT_DIR}/demo/smurf_e2e/config.elbow_single_ct_fast.yaml"
    ;;
  stable)
    CONFIG_PATH="${ROOT_DIR}/demo/smurf_e2e/config.elbow_single_ct_stable.yaml"
    ;;
  base)
    CONFIG_PATH="${ROOT_DIR}/demo/smurf_e2e/config.elbow_single_ct.yaml"
    ;;
  *)
    echo "[smurf-e2e] unknown mode: $MODE (supported: fast|stable|base)"
    exit 1
    ;;
esac

if [[ -x "${ROOT_DIR}/.venv-medml/bin/python" ]]; then
  PY=("${ROOT_DIR}/.venv-medml/bin/python")
else
  PY=(uv run python)
fi

cd "$ROOT_DIR"

echo "[smurf-e2e] mode=$MODE"
echo "[smurf-e2e] config=$CONFIG_PATH"

echo "[smurf-e2e] python=${PY[*]}"

if [[ "$SKIP_PREPARE" != "1" ]]; then
  echo "[smurf-e2e] 1/4 build elbow dryrun"
  "${PY[@]}" demo/smurf_e2e/build_elbow_dryrun.py
else
  echo "[smurf-e2e] skip prepare (SKIP_PREPARE=1)"
fi

echo "[smurf-e2e] 2/4 train"
"${PY[@]}" demo/smurf_e2e/smurf_e2e.py --config "$CONFIG_PATH" train

echo "[smurf-e2e] 3/4 evaluate"
"${PY[@]}" demo/smurf_e2e/smurf_e2e.py --config "$CONFIG_PATH" evaluate

echo "[smurf-e2e] 4/4 report"
"${PY[@]}" demo/smurf_e2e/smurf_e2e.py --config "$CONFIG_PATH" report

echo "[smurf-e2e] done"
