#!/usr/bin/env python3
"""
本地 CI 诊断脚本 - 检查 GitHub Actions 配置的所有依赖和步骤
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'


def print_header(text: str):
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print('=' * 80)


def print_step(text: str):
    print(f"\n{Colors.BLUE}▶ {text}{Colors.NC}")


def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")


def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.NC}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")


def run_command(cmd: List[str], check: bool = True) -> Tuple[bool, str]:
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
            timeout=60
        )
        return True, result.stdout + result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout + e.stderr
    except subprocess.TimeoutExpired:
        return False, "命令超时"
    except Exception as e:
        return False, str(e)


def check_environment():
    """检查基础环境"""
    print_header("步骤 1: 环境检查")

    # Python 版本
    print_step("检查 Python 版本")
    success, output = run_command([sys.executable, "--version"])
    if success:
        print_success(f"Python: {output.strip()}")
    else:
        print_error("Python 检查失败")
        return False

    # 检查 Python 版本是否 >= 3.11
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print_error(f"Python 版本过低: {version.major}.{version.minor} (需要 >= 3.11)")
        return False

    # uv 版本
    print_step("检查 uv")
    success, output = run_command(["uv", "--version"])
    if success:
        print_success(f"uv: {output.strip()}")
    else:
        print_warning("uv 未安装或不在 PATH 中")

    return True


def check_project_structure():
    """检查项目结构"""
    print_header("步骤 2: 项目结构检查")

    required_files = [
        "pyproject.toml",
        "README.md",
        "med_core/__init__.py",
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
        ".github/actions/setup-python-env/action.yml",
        "Dockerfile",
    ]

    required_dirs = [
        "med_core",
        "tests",
        "scripts",
        "configs",
        "examples",
    ]

    all_ok = True

    print_step("检查必需文件")
    for file in required_files:
        path = Path(file)
        if path.exists():
            print_success(f"{file}")
        else:
            print_error(f"{file} 不存在")
            all_ok = False

    print_step("检查必需目录")
    for dir_name in required_dirs:
        path = Path(dir_name)
        if path.is_dir():
            print_success(f"{dir_name}/")
        else:
            print_error(f"{dir_name}/ 不存在")
            all_ok = False

    return all_ok


def check_dependencies():
    """检查依赖安装"""
    print_header("步骤 3: 依赖检查")

    critical_packages = [
        "torch",
        "numpy",
        "pandas",
        "pytest",
        "ruff",
        "mypy",
    ]

    print_step("检查关键包")
    all_ok = True

    for package in critical_packages:
        try:
            __import__(package)
            print_success(f"{package} 已安装")
        except ImportError:
            print_error(f"{package} 未安装")
            all_ok = False

    return all_ok


def check_code_quality():
    """检查代码质量"""
    print_header("步骤 4: 代码质量检查")

    # Ruff 检查
    print_step("Ruff Linting")
    success, output = run_command(
        ["ruff", "check", "med_core/", "tests/", "--output-format=text"],
        check=False
    )
    if success:
        print_success("Ruff 检查通过")
    else:
        print_warning("Ruff 发现问题")
        print(output[:500])

    # Ruff 格式检查
    print_step("Ruff Format")
    success, output = run_command(
        ["ruff", "format", "--check", "med_core/", "tests/"],
        check=False
    )
    if success:
        print_success("代码格式正确")
    else:
        print_warning("代码格式需要调整")
        print(output[:500])

    # mypy 类型检查
    print_step("mypy 类型检查")
    success, output = run_command(
        ["mypy", "med_core/", "--ignore-missing-imports"],
        check=False
    )
    if success:
        print_success("类型检查通过")
    else:
        print_warning("类型检查发现问题")
        print(output[:500])

    return True


def check_tests():
    """检查测试"""
    print_header("步骤 5: 测试检查")

    # 检查测试文件
    print_step("检查测试文件")
    test_files = list(Path("tests").glob("test_*.py"))
    if test_files:
        print_success(f"找到 {len(test_files)} 个测试文件")
        for f in test_files[:5]:
            print(f"  - {f}")
    else:
        print_error("未找到测试文件")
        return False

    # 运行测试
    print_step("运行测试")
    success, output = run_command(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        check=False
    )
    if success:
        print_success("测试通过")
    else:
        print_error("测试失败")
        print(output[-1000:])
        return False

    return True


def check_scripts():
    """检查脚本"""
    print_header("步骤 6: 脚本检查")

    critical_scripts = [
        "scripts/generate_mock_data.py",
        "scripts/smoke_test.py",
    ]

    print_step("检查关键脚本")
    all_ok = True

    for script in critical_scripts:
        path = Path(script)
        if not path.exists():
            print_error(f"{script} 不存在")
            all_ok = False
            continue

        # 检查语法
        success, output = run_command(
            [sys.executable, "-m", "py_compile", script],
            check=False
        )
        if success:
            print_success(f"{script}")
        else:
            print_error(f"{script} 语法错误")
            print(output)
            all_ok = False

    return all_ok


def check_docker():
    """检查 Docker 配置"""
    print_header("步骤 7: Docker 检查")

    dockerfile = Path("Dockerfile")
    if not dockerfile.exists():
        print_error("Dockerfile 不存在")
        return False

    print_success("Dockerfile 存在")

    # 检查 Docker 是否安装
    success, output = run_command(["docker", "--version"], check=False)
    if success:
        print_success(f"Docker: {output.strip()}")

        # 尝试验证 Dockerfile 语法
        print_step("验证 Dockerfile 语法")
        success, output = run_command(
            ["docker", "build", "--no-cache", "-f", "Dockerfile", "-t", "medfusion:test", "."],
            check=False
        )
        if success:
            print_success("Docker 构建成功")
        else:
            print_warning("Docker 构建失败（可能需要依赖）")
            print(output[-500:])
    else:
        print_warning("Docker 未安装，跳过构建测试")

    return True


def check_examples():
    """检查示例代码"""
    print_header("步骤 8: 示例代码检查")

    examples_dir = Path("examples")
    if not examples_dir.exists():
        print_warning("examples/ 目录不存在")
        return True

    example_files = list(examples_dir.glob("*.py"))
    if not example_files:
        print_warning("未找到示例文件")
        return True

    print_step(f"检查 {len(example_files)} 个示例文件")
    all_ok = True

    for example in example_files:
        success, output = run_command(
            [sys.executable, "-m", "py_compile", str(example)],
            check=False
        )
        if success:
            print_success(f"{example.name}")
        else:
            print_error(f"{example.name} 语法错误")
            all_ok = False

    return all_ok


def main():
    print_header("MedFusion CI 本地诊断")
    print(f"工作目录: {Path.cwd()}")

    results = {
        "环境检查": check_environment(),
        "项目结构": check_project_structure(),
        "依赖检查": check_dependencies(),
        "代码质量": check_code_quality(),
        "测试": check_tests(),
        "脚本检查": check_scripts(),
        "Docker": check_docker(),
        "示例代码": check_examples(),
    }

    # 总结
    print_header("诊断总结")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        if result:
            print_success(f"{name}")
        else:
            print_error(f"{name}")

    print(f"\n通过: {passed}/{total}")

    if passed == total:
        print(f"\n{Colors.GREEN}所有检查通过！✓{Colors.NC}")
        return 0
    else:
        print(f"\n{Colors.RED}发现 {total - passed} 个问题{Colors.NC}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
