#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PROFILE_ID="medmnist-breastmnist"
CONFIG_PATH="configs/public_datasets/breastmnist_quickstart.yaml"
DATASET_DIR="data/public/medmnist/breastmnist-demo"
OUTPUT_DIR="outputs/public_datasets/breastmnist_quickstart"
CHECKPOINT_PATH="$OUTPUT_DIR/checkpoints/best.pth"

ARTIFACTS=(
  "$OUTPUT_DIR/checkpoints/best.pth"
  "$OUTPUT_DIR/logs/history.json"
  "$OUTPUT_DIR/metrics/metrics.json"
  "$OUTPUT_DIR/metrics/validation.json"
  "$OUTPUT_DIR/reports/summary.json"
  "$OUTPUT_DIR/reports/report.md"
)

UV_BIN=""

find_first_command() {
  local candidate
  for candidate in "$@"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

prefer_windows_uv() {
  [[ -n "${WSL_DISTRO_NAME:-}" && "$REPO_ROOT" == /mnt/c/* ]]
}

resolve_uv_bin() {
  local candidate

  if prefer_windows_uv; then
    if candidate="$(find_first_command uv.exe)"; then
      printf '%s\n' "$candidate"
      return 0
    fi

    candidate="/mnt/c/Users/Administrator/.local/bin/uv.exe"
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  if candidate="$(find_first_command uv uv.exe)"; then
    printf '%s\n' "$candidate"
    return 0
  fi

  for candidate in "$HOME/.local/bin/uv" "/home/yixian/.local/bin/uv"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

step() {
  echo
  echo "==> $1"
}

run() {
  echo "+ $*"
  "$@"
}

ensure_uv() {
  if [[ -n "$UV_BIN" ]]; then
    return 0
  fi

  if ! UV_BIN="$(resolve_uv_bin)"; then
    echo "uv not found in PATH" >&2
    exit 1
  fi
}

uv() {
  ensure_uv
  "$UV_BIN" "$@"
}

prepare_dataset_if_needed() {
  if [[ "${MEDFUSION_SMOKE_FORCE_PREPARE:-0}" == "1" ]] || [[ ! -f "$DATASET_DIR/metadata.csv" ]]; then
    step "Prepare public dataset"
    run uv run medfusion public-datasets prepare medmnist-breastmnist --overwrite
    return
  fi

  step "Reuse prepared dataset"
  echo "Using existing prepared dataset at $DATASET_DIR"
}

verify_artifacts() {
  step "Verify canonical artifacts"
  for artifact in "${ARTIFACTS[@]}"; do
    if [[ ! -f "$artifact" ]]; then
      echo "Missing artifact: $artifact" >&2
      exit 1
    fi
    echo "✓ $artifact"
  done
}

main() {
  ensure_uv

  step "Smoke profile"
  echo "Profile: $PROFILE_ID"
  echo "Config: $CONFIG_PATH"
  echo "Output: $OUTPUT_DIR"

  prepare_dataset_if_needed

  step "Validate config"
  run uv run medfusion validate-config --config "$CONFIG_PATH"

  step "Train quickstart profile"
  run uv run medfusion train --config "$CONFIG_PATH"

  step "Build results"
  run uv run medfusion build-results --config "$CONFIG_PATH" --checkpoint "$CHECKPOINT_PATH"

  verify_artifacts

  step "Smoke summary"
  echo "Smoke run completed successfully."
  echo "Output directory: $OUTPUT_DIR"
}

main "$@"
