#!/usr/bin/env bash
# 兼容包装：保留历史入口，但真实实现统一在 full_regression.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "scripts/local_ci_test.sh 已降级为兼容入口，转调 scripts/full_regression.sh --full"
exec bash scripts/full_regression.sh --full "$@"
