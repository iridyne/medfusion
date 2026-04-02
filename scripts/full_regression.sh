#!/usr/bin/env bash
# 统一验证入口：快速检查 / CI 对齐 / 完整回归

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

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
            - embedded full local validation suite

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

run_full_validation() {
  full_print_step "步骤 0: 检查环境依赖"

  if ! command -v python >/dev/null 2>&1; then
    full_print_error "Python 未安装"
    exit 1
  fi
  full_print_success "Python 已安装: $(python --version)"

  if ! command -v uv >/dev/null 2>&1; then
    full_print_error "uv 未安装"
    exit 1
  fi
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

  full_print_step "步骤 5: 运行单元测试"
  if [[ ! -d "tests" ]]; then
    full_print_error "tests/ 目录不存在"
  elif uv run pytest tests/ -v --cov=med_core --cov-report=term --cov-report=xml > /tmp/pytest.log 2>&1; then
    full_print_success "单元测试通过"
    echo
    echo "覆盖率报告:"
    grep -A 5 "TOTAL" /tmp/pytest.log || echo "未找到覆盖率信息"
  else
    full_print_error "单元测试失败"
    echo "查看详情: /tmp/pytest.log"
    tail -50 /tmp/pytest.log
  fi

  full_print_step "步骤 6: 检查集成测试脚本"
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

  full_print_step "步骤 7: 检查 Docker 配置"
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

  full_print_step "步骤 8: 检查示例代码语法"
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

  full_print_step "步骤 9: 测试包构建"
  if uv run --with build python -m build --outdir /tmp/dist > /tmp/build.log 2>&1; then
    full_print_success "包构建成功"
    ls -lh /tmp/dist/
  else
    full_print_error "包构建失败"
    cat /tmp/build.log
  fi

  full_print_step "步骤 10: 安全扫描"
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
    run_full_validation
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac
