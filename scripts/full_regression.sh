#!/usr/bin/env bash
# 统一验证入口：本地轻量预检 + GitHub Actions CI handoff

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="full"
UV_BIN=""
PYTHON_BIN=""
QUICK_PY_LINT_TARGETS=(
  tests/test_output_layout.py
  tests/test_build_results.py
  tests/test_validation_workflow.py
)

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

resolve_python_bin() {
  local candidate
  if candidate="$(find_first_command python python3)"; then
    printf '%s\n' "$candidate"
    return 0
  fi

  for candidate in \
    "/usr/bin/python3" \
    "/mnt/c/Users/Administrator/AppData/Local/Microsoft/WindowsApps/python.exe"
  do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/full_regression.sh [--mode quick|ci|full]
  bash scripts/full_regression.sh --quick
  bash scripts/full_regression.sh --ci
  bash scripts/full_regression.sh --full
  bash scripts/full_regression.sh --help

Modes:
  quick   Fast local preflight for everyday development.
          Runs:
            - uv sync --extra dev
            - bash -n scripts/full_regression.sh
            - uv run ruff check <focused validation python files>
            - uv run ruff format --check <focused validation python files>
          Notes:
            - quick stays local and intentionally does not run pytest.
            - pytest ownership lives in GitHub Actions CI: .github/workflows/ci.yml

  ci      Local handoff check before GitHub Actions CI.
          Runs:
            - uv sync --extra dev --extra web
            - bash test/smoke.sh
          Notes:
            - pytest is intentionally delegated to GitHub Actions CI.
            - inspect recent failures with: bash scripts/inspect_ci_failure.sh

  full    Broader local non-pytest regression pass.
          Runs:
            - embedded local checks (env, lint, format, mypy, smoke, build, safety)
          Notes:
            - pytest is intentionally excluded from local full mode.
            - use CI logs as the source of truth for pytest failures.

Why this exists:
  Local scripts stay focused on preflight and smoke.
  Full pytest responsibility is owned by .github/workflows/ci.yml.
EOF
}

run() {
  echo
  echo "==> $*"
  "$@"
}

FULL_RED='\033[0;31m'
FULL_GREEN='\033[0;32m'
FULL_YELLOW='\033[1;33m'
FULL_NC='\033[0m'
FULL_PASSED=0
FULL_FAILED=0
FULL_WARNINGS=0

full_print_step() {
  echo
  echo "=========================================="
  echo "$1"
  echo "=========================================="
}

full_print_success() {
  echo -e "${FULL_GREEN}✓ $1${FULL_NC}"
  FULL_PASSED=$((FULL_PASSED + 1))
}

full_print_error() {
  echo -e "${FULL_RED}✗ $1${FULL_NC}"
  FULL_FAILED=$((FULL_FAILED + 1))
}

full_print_warning() {
  echo -e "${FULL_YELLOW}⚠ $1${FULL_NC}"
  FULL_WARNINGS=$((FULL_WARNINGS + 1))
}

ensure_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    return 0
  fi

  if ! PYTHON_BIN="$(resolve_python_bin)"; then
    echo "python not found" >&2
    exit 1
  fi
}

ensure_uv() {
  if [[ -n "$UV_BIN" ]]; then
    return 0
  fi

  if ! UV_BIN="$(resolve_uv_bin)"; then
    echo "uv not found" >&2
    exit 1
  fi
}

python() {
  ensure_python
  "$PYTHON_BIN" "$@"
}

uv() {
  ensure_uv
  "$UV_BIN" "$@"
}

print_ci_pytest_notice() {
  echo
  echo "pytest 已迁移到 GitHub Actions CI"
  echo "pytest now runs in GitHub Actions CI: .github/workflows/ci.yml"
  echo "Inspect recent failed logs with: bash scripts/inspect_ci_failure.sh"
}

run_full_validation() {
  full_print_step "步骤 0: 检查环境依赖"

  ensure_python
  full_print_success "Python 已安装: $(python --version)"

  ensure_uv
  full_print_success "uv 已安装: $(uv --version)"

  local python_version
  python_version="$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
  if [[ "$python_version" < "3.11" ]]; then
    full_print_error "Python 版本过低 (需要 >= 3.11，当前: $python_version)"
    exit 1
  fi
  full_print_success "Python 版本符合要求: $python_version"

  full_print_step "步骤 1: 安装开发依赖"

  if [[ ! -d ".venv" ]]; then
    echo "创建虚拟环境..."
    uv venv
  fi

  echo "同步开发依赖 (dev extras)..."
  if uv sync --extra dev > /tmp/uv_install.log 2>&1; then
    full_print_success "依赖同步成功"
  else
    full_print_error "依赖同步失败，查看日志: /tmp/uv_install.log"
    cat /tmp/uv_install.log
    exit 1
  fi

  full_print_step "步骤 2: 代码质量检查 (Ruff Linting)"
  if uv run ruff check med_core/ tests/ --output-format=text > /tmp/ruff_check.log 2>&1; then
    full_print_success "Ruff 检查通过"
  else
    full_print_warning "Ruff 检查发现问题 (continue-on-error)"
    echo "查看详情: /tmp/ruff_check.log"
    head -20 /tmp/ruff_check.log
  fi

  full_print_step "步骤 3: 代码格式检查 (Ruff Format)"
  if uv run ruff format --check med_core/ tests/ > /tmp/ruff_format.log 2>&1; then
    full_print_success "代码格式检查通过"
  else
    full_print_warning "代码格式检查发现问题 (continue-on-error)"
    echo "查看详情: /tmp/ruff_format.log"
    head -20 /tmp/ruff_format.log
  fi

  full_print_step "步骤 4: 类型检查 (mypy)"
  if uv run mypy med_core/ --ignore-missing-imports > /tmp/mypy.log 2>&1; then
    full_print_success "类型检查通过"
  else
    full_print_warning "类型检查发现问题 (continue-on-error)"
    echo "查看详情: /tmp/mypy.log"
    head -30 /tmp/mypy.log
  fi

  full_print_step "步骤 5: 仓库 smoke 主链"
  if bash test/smoke.sh > /tmp/medfusion_smoke.log 2>&1; then
    full_print_success "Smoke 主链通过"
  else
    full_print_error "Smoke 主链失败"
    echo "查看详情: /tmp/medfusion_smoke.log"
    tail -50 /tmp/medfusion_smoke.log
  fi

  full_print_step "步骤 6: pytest 交给 GitHub Actions CI"
  full_print_success "本地 full 模式不运行 pytest"
  echo "查看 CI 失败日志: bash scripts/inspect_ci_failure.sh"

  full_print_step "步骤 7: 检查集成测试脚本"
  local script
  for script in "scripts/generate_mock_data.py" "test/smoke.sh"; do
    if [[ ! -f "$script" ]]; then
      full_print_warning "脚本不存在: $script"
      continue
    fi
    if [[ "$script" == *.sh ]]; then
      if bash -n "$script" 2>&1; then
        full_print_success "脚本语法正确: $script"
      else
        full_print_error "脚本语法错误: $script"
      fi
    elif python -m py_compile "$script" 2>&1; then
      full_print_success "脚本语法正确: $script"
    else
      full_print_error "脚本语法错误: $script"
    fi
  done

  full_print_step "步骤 8: 检查 Docker 配置"
  if [[ -f "Dockerfile" ]]; then
    full_print_success "Dockerfile 存在"
    if command -v docker >/dev/null 2>&1; then
      if docker build --dry-run . > /dev/null 2>&1; then
        full_print_success "Dockerfile 语法正确"
      else
        full_print_warning "Dockerfile 可能有问题（需要实际构建验证）"
      fi
    else
      full_print_warning "Docker 未安装，跳过 Dockerfile 验证"
    fi
  else
    full_print_warning "Dockerfile 不存在，跳过 Docker 检查"
  fi

  full_print_step "步骤 9: 检查示例代码语法"
  if [[ -d "examples" ]]; then
    local example example_errors=0
    for example in examples/*.py; do
      if [[ -f "$example" ]]; then
        if python -m py_compile "$example" 2>&1; then
          echo "  ✓ $example"
        else
          echo "  ✗ $example"
          example_errors=$((example_errors + 1))
        fi
      fi
    done
    if [[ $example_errors -eq 0 ]]; then
      full_print_success "所有示例代码语法正确"
    else
      full_print_error "发现 $example_errors 个示例代码语法错误"
    fi
  else
    full_print_warning "examples/ 目录不存在"
  fi

  full_print_step "步骤 10: 测试包构建"
  if uv run --with build python -m build --outdir /tmp/dist > /tmp/build.log 2>&1; then
    full_print_success "包构建成功"
    ls -lh /tmp/dist/
  else
    full_print_error "包构建失败"
    cat /tmp/build.log
  fi

  full_print_step "步骤 11: 安全扫描"
  echo "执行安全扫描..."
  if uv run --with bandit bandit -r med_core/ -f json -o /tmp/bandit-report.json > /tmp/bandit.log 2>&1; then
    full_print_success "Bandit 安全扫描通过"
  else
    full_print_warning "Bandit 发现潜在安全问题"
    if [[ -f /tmp/bandit-report.json ]]; then
      echo "查看报告: /tmp/bandit-report.json"
    fi
  fi
  if uv run --with safety safety check --json > /tmp/safety-report.json 2>&1; then
    full_print_success "Safety 依赖检查通过"
  else
    full_print_warning "Safety 发现已知漏洞或命令兼容问题"
    echo "查看报告: /tmp/safety-report.json"
  fi

  full_print_step "测试总结"
  echo
  echo "通过: $FULL_PASSED"
  echo "失败: $FULL_FAILED"
  echo "警告: $FULL_WARNINGS"
  echo

  print_ci_pytest_notice

  if [[ $FULL_FAILED -eq 0 ]]; then
    echo -e "${FULL_GREEN}=========================================="
    echo "所有关键测试通过！✓"
    echo -e "==========================================${FULL_NC}"
    return 0
  fi

  echo -e "${FULL_RED}=========================================="
  echo "发现 $FULL_FAILED 个失败项！✗"
  echo -e "==========================================${FULL_NC}"
  return 1
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
    print_ci_pytest_notice
    ;;
  ci)
    run uv sync --extra dev --extra web
    run bash test/smoke.sh
    print_ci_pytest_notice
    ;;
  full)
    run_full_validation
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac
