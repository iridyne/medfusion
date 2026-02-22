#!/bin/bash
# 本地 CI 测试脚本 - 模拟 GitHub Actions 流程

set -e  # 遇到错误立即退出

echo "=========================================="
echo "MedFusion 本地 CI 测试"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 测试结果统计
PASSED=0
FAILED=0
WARNINGS=0

# 辅助函数
print_step() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
    ((PASSED++))
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
    ((FAILED++))
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
    ((WARNINGS++))
}

# 检查依赖
print_step "步骤 0: 检查环境依赖"

if ! command -v python &> /dev/null; then
    print_error "Python 未安装"
    exit 1
fi
print_success "Python 已安装: $(python --version)"

if ! command -v uv &> /dev/null; then
    print_error "uv 未安装"
    exit 1
fi
print_success "uv 已安装: $(uv --version)"

# 检查 Python 版本
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PYTHON_VERSION" < "3.11" ]]; then
    print_error "Python 版本过低 (需要 >= 3.11，当前: $PYTHON_VERSION)"
    exit 1
fi
print_success "Python 版本符合要求: $PYTHON_VERSION"

# 步骤 1: 安装依赖
print_step "步骤 1: 安装开发依赖"

if [ ! -d ".venv" ]; then
    echo "创建虚拟环境..."
    uv venv
fi

echo "安装依赖 (dev extras)..."
if uv pip install -e ".[dev]" > /tmp/uv_install.log 2>&1; then
    print_success "依赖安装成功"
else
    print_error "依赖安装失败，查看日志: /tmp/uv_install.log"
    cat /tmp/uv_install.log
    exit 1
fi

# 激活虚拟环境
source .venv/bin/activate

# 步骤 2: 代码质量检查
print_step "步骤 2: 代码质量检查 (Ruff Linting)"

if ruff check med_core/ tests/ --output-format=text > /tmp/ruff_check.log 2>&1; then
    print_success "Ruff 检查通过"
else
    print_warning "Ruff 检查发现问题 (continue-on-error)"
    echo "查看详情: /tmp/ruff_check.log"
    head -20 /tmp/ruff_check.log
fi

# 步骤 3: 代码格式检查
print_step "步骤 3: 代码格式检查 (Ruff Format)"

if ruff format --check med_core/ tests/ > /tmp/ruff_format.log 2>&1; then
    print_success "代码格式检查通过"
else
    print_warning "代码格式检查发现问题 (continue-on-error)"
    echo "查看详情: /tmp/ruff_format.log"
    head -20 /tmp/ruff_format.log
fi

# 步骤 4: 类型检查
print_step "步骤 4: 类型检查 (mypy)"

if mypy med_core/ --ignore-missing-imports > /tmp/mypy.log 2>&1; then
    print_success "类型检查通过"
else
    print_warning "类型检查发现问题 (continue-on-error)"
    echo "查看详情: /tmp/mypy.log"
    head -30 /tmp/mypy.log
fi

# 步骤 5: 单元测试
print_step "步骤 5: 运行单元测试"

if [ ! -d "tests" ]; then
    print_error "tests/ 目录不存在"
else
    if pytest tests/ -v --cov=med_core --cov-report=term --cov-report=xml > /tmp/pytest.log 2>&1; then
        print_success "单元测试通过"
        echo ""
        echo "覆盖率报告:"
        grep -A 5 "TOTAL" /tmp/pytest.log || echo "未找到覆盖率信息"
    else
        print_error "单元测试失败"
        echo "查看详情: /tmp/pytest.log"
        tail -50 /tmp/pytest.log
    fi
fi

# 步骤 6: 检查关键脚本
print_step "步骤 6: 检查集成测试脚本"

SCRIPTS=(
    "scripts/generate_mock_data.py"
    "scripts/smoke_test.py"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if python -m py_compile "$script" 2>&1; then
            print_success "脚本语法正确: $script"
        else
            print_error "脚本语法错误: $script"
        fi
    else
        print_warning "脚本不存在: $script"
    fi
done

# 步骤 7: 检查 Docker 配置
print_step "步骤 7: 检查 Docker 配置"

if [ -f "Dockerfile" ]; then
    print_success "Dockerfile 存在"

    # 检查 Dockerfile 语法（如果安装了 docker）
    if command -v docker &> /dev/null; then
        if docker build --dry-run . > /dev/null 2>&1; then
            print_success "Dockerfile 语法正确"
        else
            print_warning "Dockerfile 可能有问题（需要实际构建验证）"
        fi
    else
        print_warning "Docker 未安装，跳过 Dockerfile 验证"
    fi
else
    print_error "Dockerfile 不存在"
fi

# 步骤 8: 检查示例代码
print_step "步骤 8: 检查示例代码语法"

if [ -d "examples" ]; then
    EXAMPLE_ERRORS=0
    for example in examples/*.py; do
        if [ -f "$example" ]; then
            if python -m py_compile "$example" 2>&1; then
                echo "  ✓ $example"
            else
                echo "  ✗ $example"
                ((EXAMPLE_ERRORS++))
            fi
        fi
    done

    if [ $EXAMPLE_ERRORS -eq 0 ]; then
        print_success "所有示例代码语法正确"
    else
        print_error "发现 $EXAMPLE_ERRORS 个示例代码语法错误"
    fi
else
    print_warning "examples/ 目录不存在"
fi

# 步骤 9: 检查包构建
print_step "步骤 9: 测试包构建"

if uv pip install build > /dev/null 2>&1; then
    if python -m build --outdir /tmp/dist > /tmp/build.log 2>&1; then
        print_success "包构建成功"
        ls -lh /tmp/dist/
    else
        print_error "包构建失败"
        cat /tmp/build.log
    fi
else
    print_warning "无法安装 build 工具"
fi

# 步骤 10: 安全检查
print_step "步骤 10: 安全扫描"

echo "安装安全工具..."
if uv pip install bandit safety > /dev/null 2>&1; then

    # Bandit 扫描
    if bandit -r med_core/ -f json -o /tmp/bandit-report.json > /dev/null 2>&1; then
        print_success "Bandit 安全扫描通过"
    else
        print_warning "Bandit 发现潜在安全问题"
        if [ -f /tmp/bandit-report.json ]; then
            echo "查看报告: /tmp/bandit-report.json"
        fi
    fi

    # Safety 检查
    if safety check --json > /tmp/safety-report.json 2>&1; then
        print_success "Safety 依赖检查通过"
    else
        print_warning "Safety 发现已知漏洞"
        echo "查看报告: /tmp/safety-report.json"
    fi
else
    print_warning "无法安装安全工具"
fi

# 总结
print_step "测试总结"

echo ""
echo "通过: $PASSED"
echo "失败: $FAILED"
echo "警告: $WARNINGS"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}=========================================="
    echo "所有关键测试通过！✓"
    echo -e "==========================================${NC}"
    exit 0
else
    echo -e "${RED}=========================================="
    echo "发现 $FAILED 个失败项！✗"
    echo -e "==========================================${NC}"
    exit 1
fi
