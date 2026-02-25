#!/usr/bin/env python3
"""
CI 配置验证脚本 - 检查所有修复是否正确应用
"""

import sys
from pathlib import Path


def check_pyproject():
    """检查 pyproject.toml 配置"""
    print("检查 pyproject.toml...")
    content = Path("pyproject.toml").read_text()

    issues = []

    # 检查 dev 依赖位置
    if "[project.optional-dependencies]" in content and "dev = [" in content:
        print("  ✓ dev 依赖在正确位置 (optional-dependencies)")
    else:
        issues.append("dev 依赖配置错误")

    # 检查是否还有 dependency-groups
    if "[dependency-groups]" in content:
        issues.append("仍然存在 [dependency-groups] 节")
    else:
        print("  ✓ 已移除 [dependency-groups]")

    # 检查 Python 版本限制
    if 'requires-python = ">=3.11,<3.14"' in content:
        print("  ✓ Python 版本限制正确")
    else:
        issues.append("Python 版本限制不正确")

    return len(issues) == 0, issues


def check_ci_workflow():
    """检查 CI workflow 配置"""
    print("\n检查 .github/workflows/ci.yml...")
    content = Path(".github/workflows/ci.yml").read_text()

    issues = []

    # 检查测试矩阵
    if 'python-version: ["3.11", "3.12", "3.13"]' in content:
        print("  ✓ 测试矩阵包含 Python 3.11-3.13")
    else:
        issues.append("测试矩阵配置不正确")

    # 检查安全工具版本
    if "bandit==1.7.5 safety==3.0.1" in content:
        print("  ✓ 安全工具版本已固定")
    else:
        issues.append("安全工具版本未固定")

    # 检查 Docker 平台
    if "platforms: linux/amd64" in content:
        print("  ✓ Docker 构建限制为单平台")
    else:
        issues.append("Docker 平台配置不正确")

    return len(issues) == 0, issues


def check_setup_action():
    """检查 setup-python-env action"""
    print("\n检查 .github/actions/setup-python-env/action.yml...")
    content = Path(".github/actions/setup-python-env/action.yml").read_text()

    issues = []

    # 检查缓存配置
    if "actions/cache@v4" in content:
        print("  ✓ 已添加依赖缓存")
    else:
        issues.append("缺少缓存配置")

    # 检查缓存路径
    if "~/.cache/uv" in content:
        print("  ✓ uv 缓存路径正确")
    else:
        issues.append("uv 缓存路径不正确")

    return len(issues) == 0, issues


def check_project_structure():
    """检查项目结构"""
    print("\n检查项目结构...")

    required = {
        "med_core/__init__.py": "核心模块",
        "tests/": "测试目录",
        "scripts/generate_mock_data.py": "数据生成脚本",
        "scripts/smoke_test.py": "冒烟测试脚本",
        "Dockerfile": "Docker 配置",
        ".github/workflows/ci.yml": "CI 配置",
        ".github/workflows/release.yml": "发布配置",
    }

    issues = []
    for path, desc in required.items():
        p = Path(path)
        if p.exists():
            print(f"  ✓ {desc}: {path}")
        else:
            issues.append(f"缺少{desc}: {path}")

    # 统计测试文件
    test_files = list(Path("tests").glob("test_*.py"))
    print(f"  ✓ 找到 {len(test_files)} 个测试文件")

    return len(issues) == 0, issues


def main():
    print("=" * 70)
    print("  MedFusion CI 配置验证")
    print("=" * 70)
    print()

    results = {}
    all_issues = []

    # 运行所有检查
    checks = [
        ("pyproject.toml", check_pyproject),
        ("CI Workflow", check_ci_workflow),
        ("Setup Action", check_setup_action),
        ("项目结构", check_project_structure),
    ]

    for name, check_func in checks:
        success, issues = check_func()
        results[name] = success
        if issues:
            all_issues.extend([(name, issue) for issue in issues])

    # 总结
    print("\n" + "=" * 70)
    print("  验证总结")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status}: {name}")

    if all_issues:
        print("\n发现的问题:")
        for name, issue in all_issues:
            print(f"  - [{name}] {issue}")

    print(f"\n结果: {passed}/{total} 项检查通过")

    if passed == total:
        print("\n✓ 所有配置修复已正确应用！")
        print("\n下一步:")
        print(
            "  1. 提交更改: git add -A && git commit -m 'fix: CI configuration issues'"
        )
        print("  2. 推送到远程: git push")
        print("  3. 观察 GitHub Actions 运行结果")
        return 0
    else:
        print("\n✗ 仍有配置问题需要修复")
        return 1


if __name__ == "__main__":
    sys.exit(main())
