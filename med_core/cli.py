"""
CLI entry points for Med-Core framework.

This module has been refactored into a package structure.
The functions are now imported from med_core.cli submodules.

Provides command-line interfaces for:
- med-train: Train multimodal models
- med-evaluate: Evaluate trained models
- med-preprocess: Preprocess medical images
- medfusion web: Web UI commands
"""

import click

# Import from new modular structure
from med_core.cli.evaluate import evaluate
from med_core.cli.preprocess import preprocess
from med_core.cli.train import train

__all__ = ["train", "evaluate", "preprocess", "main"]


@click.group()
@click.version_option(version="0.3.0", prog_name="MedFusion")
def main():
    """MedFusion - 医学多模态深度学习框架"""
    pass


# 添加子命令
main.add_command(train, name="train")
main.add_command(evaluate, name="evaluate")
main.add_command(preprocess, name="preprocess")

# 添加 Web UI 命令
try:
    from med_core.web.cli import web, data
    main.add_command(web, name="web")
    main.add_command(data, name="data")
except ImportError:
    # Web 依赖未安装
    @main.command()
    def web():
        """启动 Web UI（需要安装 web 依赖）"""
        click.echo("❌ Web UI 依赖未安装")
        click.echo("💡 请运行: pip install medfusion[web]")
        raise click.Abort()


if __name__ == "__main__":
    main()
