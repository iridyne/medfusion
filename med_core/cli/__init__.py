"""MedFusion CLI package.

This module exposes:
- ``main``: unified CLI entry for ``medfusion``
- ``train``/``evaluate``/``preprocess``: legacy callable entry points used by
  ``med-train``/``med-evaluate``/``med-preprocess`` scripts.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from med_core.version import __version__

from .build_results import build_results
from .doctor import doctor, validate_config
from .evaluate import evaluate
from .import_run import import_run
from .preprocess import preprocess
from .public_datasets import public_datasets
from .run import run
from .train import train

__all__ = [
    "build_results",
    "doctor",
    "evaluate",
    "import_run",
    "main",
    "preprocess",
    "public_datasets",
    "run",
    "train",
    "validate_config",
]


def _print_help() -> None:
    print("MedFusion - 医学多模态深度学习框架")
    print("")
    print("Usage:")
    print("  medfusion <command> [args...]")
    print("")
    print("Commands:")
    print("  start       启动 MedFusion 工作台（推荐入口）")
    print("  run         一条命令执行 validate-config -> train -> build-results")
    print("  train       训练模型")
    print("  evaluate    评估模型")
    print("  preprocess  数据预处理")
    print("  validate-config  训练前配置与数据体检")
    print("  doctor      validate-config 的短别名")
    print("  build-results   训练后生成 validation / 图表 / 报告 artifact")
    print("  import-run      生成 artifact 并导入模型库，供 dashboard 直接展示")
    print("  public-datasets 公开数据集快速验证入口（list/show/prepare）")
    print("  web         Web UI 管理命令")
    print("  data        Web UI 数据管理命令")
    print("")
    print("Global options:")
    print("  -h, --help     显示帮助")
    print("  -V, --version  显示版本")
    print("")
    print("Recommended entrypoints:")
    print("  medfusion start")
    print("    Guided first-run entry for onboarding, quick validation, and result handoff")
    print("  medfusion run --config configs/starter/quickstart.yaml")
    print("  medfusion validate-config --config configs/starter/quickstart.yaml")
    print("  medfusion train --config configs/starter/quickstart.yaml")
    print(
        "  medfusion build-results --config configs/starter/quickstart.yaml --checkpoint <path>"
    )
    print(
        "  medfusion import-run --config configs/starter/quickstart.yaml --checkpoint <path>"
    )
    print(
        "  medfusion evaluate --config configs/starter/quickstart.yaml --checkpoint <path>"
    )
    print("  medfusion public-datasets list")
    print("  medfusion public-datasets prepare uci-heart-disease --overwrite")
    print("  medfusion web")
    print("")
    print("YAML mainline:")
    print("  Use one training YAML to validate, train, and build structured results")
    print("")
    print("Config directories:")
    print("  configs/starter/          CLI 训练主链入门配置")
    print("  configs/public_datasets/  公开数据集快速验证配置")
    print("  configs/testing/          smoke / workflow 测试配置")
    print("  configs/builder/          模型结构实验配置，不等价于 train schema")
    print("")
    print("Legacy aliases:")
    print("  med-train / med-evaluate / med-preprocess")


def _dispatch_web_command(command: str, args: list[str]) -> None:
    try:
        from med_core.web.cli import data, web
    except ImportError:
        print("❌ Web UI 依赖未安装")
        print("💡 请运行: pip install medfusion[web]")
        raise SystemExit(1) from None

    target = web if command == "web" else data
    target.main(args=args, prog_name=f"medfusion {command}", standalone_mode=True)


def _dispatch_start_command(args: list[str]) -> None:
    try:
        from med_core.web.cli import start as web_start
    except ImportError:
        print("❌ Web UI 依赖未安装")
        print("💡 请运行: pip install medfusion[web]")
        raise SystemExit(1) from None

    web_start.main(args=args, prog_name="medfusion start", standalone_mode=True)


def _run_legacy_command(
    command: Callable[..., None],
    args: Sequence[str],
    canonical_prog: str,
) -> None:
    """Run argparse-based subcommands under the unified medfusion CLI."""
    command(argv=list(args), prog=canonical_prog)


def main() -> None:
    """Unified CLI entry point for ``medfusion``."""
    argv = sys.argv[1:]

    if not argv or argv[0] in {"-h", "--help"}:
        _print_help()
        return

    if argv[0] in {"-V", "--version"}:
        print(f"MedFusion {__version__}")
        return

    command = argv[0]
    args = argv[1:]

    if command == "start":
        _dispatch_start_command(args)
        return

    if command == "train":
        _run_legacy_command(train, args, "medfusion train")
        return

    if command == "run":
        _run_legacy_command(run, args, "medfusion run")
        return

    if command == "evaluate":
        _run_legacy_command(evaluate, args, "medfusion evaluate")
        return

    if command == "preprocess":
        _run_legacy_command(preprocess, args, "medfusion preprocess")
        return

    if command == "validate-config":
        _run_legacy_command(validate_config, args, "medfusion validate-config")
        return

    if command == "doctor":
        _run_legacy_command(doctor, args, "medfusion doctor")
        return

    if command == "build-results":
        _run_legacy_command(build_results, args, "medfusion build-results")
        return

    if command == "import-run":
        _run_legacy_command(import_run, args, "medfusion import-run")
        return

    if command == "public-datasets":
        _run_legacy_command(public_datasets, args, "medfusion public-datasets")
        return

    if command in {"web", "data"}:
        _dispatch_web_command(command, args)
        return

    print(f"❌ 未知命令: {command}")
    print("")
    _print_help()
    raise SystemExit(2)
