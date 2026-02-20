#!/bin/bash
# 一键构建和测试脚本

set -e

echo "🦀 MedCore Rust 加速模块 - 构建脚本"
echo "======================================"

# 检查 Rust 是否安装
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust 未安装"
    echo "请运行: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

echo "✅ Rust 已安装: $(rustc --version)"

# 检查 maturin 是否安装
if ! command -v maturin &> /dev/null; then
    echo "📦 安装 maturin..."
    pip install maturin
fi

echo "✅ Maturin 已安装: $(maturin --version)"

# 构建模块
echo ""
echo "🔨 构建 Rust 模块（发布模式）..."
maturin develop --release

# 验证安装
echo ""
echo "🧪 验证安装..."
python -c "from med_core_rs import normalize_intensity_minmax; print('✅ 模块导入成功!')"

# 运行测试
echo ""
echo "🧪 运行 Rust 单元测试..."
cargo test

# 运行基准测试（可选）
echo ""
read -p "是否运行性能基准测试？(y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📊 运行 Python vs Rust 性能对比..."
    python benchmark_comparison.py
fi

echo ""
echo "======================================"
echo "✅ 构建完成！"
echo ""
echo "下一步："
echo "  1. 运行示例: python example_integration.py"
echo "  2. 运行基准测试: python benchmark_comparison.py"
echo "  3. 运行 Rust 基准测试: cargo bench"
echo "======================================"
