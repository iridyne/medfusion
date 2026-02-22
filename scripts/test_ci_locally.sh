#!/bin/bash
# 简化的本地 CI 测试脚本

set -e

echo "=========================================="
echo "本地 CI 快速测试"
echo "=========================================="

# 1. 测试依赖安装
echo ""
echo "1. 测试依赖安装..."
if uv pip install -e ".[dev]" > /tmp/install.log 2>&1; then
    echo "✓ 依赖安装成功"
else
    echo "✗ 依赖安装失败"
    tail -20 /tmp/install.log
    exit 1
fi

# 2. 测试代码检查
echo ""
echo "2. 测试代码检查..."
ruff check med_core/ --output-format=text || echo "⚠ Ruff 发现问题"

# 3. 测试格式检查
echo ""
echo "3. 测试格式检查..."
ruff format --check med_core/ || echo "⚠ 格式需要调整"

# 4. 测试类型检查
echo ""
echo "4. 测试类型检查..."
mypy med_core/ --ignore-missing-imports || echo "⚠ 类型检查发现问题"

# 5. 检查关键文件
echo ""
echo "5. 检查关键文件..."
for file in scripts/generate_mock_data.py scripts/smoke_test.py; do
    if [ -f "$file" ]; then
        python -m py_compile "$file" && echo "✓ $file"
    else
        echo "✗ $file 不存在"
    fi
done

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
