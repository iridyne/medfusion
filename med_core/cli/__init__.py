"""MedFusion CLI package.

This module exposes:
- ``main``: unified CLI entry for ``medfusion``
- ``train``/``evaluate``/``preprocess``: legacy callable entry points used by
  ``med-train``/``med-evaluate``/``med-preprocess`` scripts.

The command handlers are imported lazily so lightweight commands do not
implicitly require heavyweight training dependencies.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from med_core.version import __version__

__all__ = ["evaluate", "main", "preprocess", "train"]


def _load_train() -> Callable[[], None]:
    from .train import train

    return train


def _load_evaluate() -> Callable[[], None]:
    from .evaluate import evaluate

    return evaluate


def _load_preprocess() -> Callable[[], None]:
    from .preprocess import preprocess

    return preprocess


def __getattr__(name: str) -> Callable[[], None]:
    """Lazy attribute loader for command functions."""
    if name == "train":
        return _load_train()
    if name == "evaluate":
        return _load_evaluate()
    if name == "preprocess":
        return _load_preprocess()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _print_help() -> None:
    print("MedFusion - 医学多模态深度学习框架")
    print("")
    print("Usage:")
    print("  medfusion <command> [args...]")
    print("")
    print("Commands:")
    print("  train       训练模型")
    print("  evaluate    评估模型")
    print("  preprocess  数据预处理")
    print("  web         Web UI 管理命令")
    print("  data        Web UI 数据管理命令")
    print("")
    print("Global options:")
    print("  -h, --help     显示帮助")
    print("  -V, --version  显示版本")


def _dispatch_web_command(command: str, args: list[str]) -> None:
    try:
        from med_core.web.cli import data, web
    except ImportError:
        print("❌ Web UI 依赖未安装")
        print("💡 请运行: pip install medfusion[web]")
        raise SystemExit(1) from None

    target = web if command == "web" else data
    target.main(args=args, prog_name=f"medfusion {command}", standalone_mode=True)


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

    if command == "train":
        sys.argv = ["med-train", *args]
        _load_train()()
        return

    if command == "evaluate":
        sys.argv = ["med-evaluate", *args]
        _load_evaluate()()
        return

    if command == "preprocess":
        sys.argv = ["med-preprocess", *args]
        _load_preprocess()()
        return

    if command in {"web", "data"}:
        _dispatch_web_command(command, args)
        return

    print(f"❌ 未知命令: {command}")
    print("")
    _print_help()
    raise SystemExit(2)
