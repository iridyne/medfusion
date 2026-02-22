#!/usr/bin/env python3
"""
简化的 CI 本地测试脚本
"""
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    """运行命令"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 60)
    print("  MedFusion CI 本地快速测试")
    print("=" * 60)

    # 1. 检查项目文件
    print("\n1. 检查项目结构...")
    files = ["pyproject.toml", "med_core/__init__.py", "Dockerfile"]
    for f in files:
        if Path(f).exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} 不存在")
            return 1

    # 2. 测试依赖安装
    print("\n2. 测试依赖安装...")
    success, output = run_cmd(["uv", "pip", "install", "-e", ".[dev]"])
    if success:
        print("  ✓ 依赖安装成功")
    else:
        print("  ✗ 依赖安装失败")
        print(output[-500:])
        return 1

    # 3. 代码检查
    print("\n3. 代码质量检查...")

    print("  - Ruff linting...")
    success, _ = run_cmd(["ruff", "check", "med_core/", "--output-format=text"])
    print("    ✓ 通过" if success else "    ⚠ 发现问题")

    print("  - Ruff format...")
    success, _ = run_cmd(["ruff", "format", "--check", "med_core/"])
    print("    ✓ 通过" if success else "    ⚠ 需要格式化")

    print("  - mypy...")
    success, _ = run_cmd(["mypy", "med_core/", "--ignore-missing-imports"])
    print("    ✓ 通过" if success else "    ⚠ 发现类型问题")

    # 4. 检查脚本
    print("\n4. 检查关键脚本...")
    scripts = ["scripts/generate_mock_data.py", "scripts/smoke_test.py"]
    for script in scripts:
        if Path(script).exists():
            success, _ = run_cmd([sys.executable, "-m", "py_compile", script])
            print(f"  {'✓' if success else '✗'} {script}")
        else:
            print(f"  ⚠ {script} 不存在")

    # 5. 检查配置文件
    print("\n5. 检查 CI 配置...")
    ci_files = [
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
        ".github/actions/setup-python-env/action.yml",
    ]
    for f in ci_files:
        if Path(f).exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} 不存在")

    print("\n" + "=" * 60)
    print("  测试完成！")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
