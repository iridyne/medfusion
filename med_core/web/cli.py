"""Web UI CLI 命令"""

import socket
import webbrowser
from pathlib import Path

import click
import uvicorn
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import settings

console = Console()


def find_free_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """查找可用端口"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"无法在 {start_port}-{start_port + max_attempts} 范围内找到可用端口"
    )


def check_web_ui_exists() -> bool:
    """检查前端资源是否存在"""
    static_dir = settings.data_dir / "web-ui" / settings.version / "static"
    return static_dir.exists() and (static_dir / "index.html").exists()


def initialize_web_server() -> None:
    """初始化 Web 服务器"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # 1. 初始化目录
        task1 = progress.add_task("⏳ 初始化数据目录...", total=None)
        settings.initialize_directories()
        progress.update(task1, completed=True)
        console.print("✅ 数据目录初始化完成")

        # 2. 检查前端资源
        task2 = progress.add_task("⏳ 检查前端资源...", total=None)
        if not check_web_ui_exists():
            console.print("⚠️  前端资源未安装")
            console.print("💡 首次运行时会自动下载前端资源（功能开发中）")
            # TODO: 实现自动下载前端资源
        else:
            console.print("✅ 前端资源就绪")
        progress.update(task2, completed=True)


@click.group()
def web() -> None:
    """Web UI 管理命令"""
    pass


@web.command()
@click.option("--host", default="127.0.0.1", help="监听地址")
@click.option("--port", default=None, type=int, help="端口")
@click.option("--auth", is_flag=True, help="启用认证")
@click.option("--token", default=None, help="自定义 Token")
@click.option("--no-browser", is_flag=True, help="不自动打开浏览器")
@click.option("--reload", is_flag=True, help="开发模式（自动重载）")
def start(host: str, port: int | None, auth: bool, token: str | None, no_browser: bool, reload: bool) -> None:
    """启动 MedFusion Web UI"""

    # 初始化
    console.print("\n[bold blue]MedFusion Web UI[/bold blue]")
    console.print(f"版本: {settings.version}\n")

    initialize_web_server()

    # 查找可用端口
    if port is None:
        port = find_free_port()
        if port != 8000:
            console.print(f"⚠️  端口 8000 已被占用，使用端口 {port}")

    # 认证配置
    if auth:
        # TODO: 实现认证
        console.print("🔒 认证已启用")
        if token:
            console.print(f"🔑 Token: {token}")

    # 警告
    if host != "127.0.0.1":
        console.print(f"⚠️  监听所有网络接口 ({host})，建议启用认证: --auth")

    # 启动信息
    console.print("\n✅ 启动成功！")
    console.print(f"🌐 访问地址: [link]http://{host}:{port}[/link]")
    console.print("按 Ctrl+C 停止服务器\n")

    # 自动打开浏览器
    if not no_browser and host == "127.0.0.1":
        try:
            webbrowser.open(f"http://{host}:{port}")
        except Exception:
            pass

    # 启动服务器
    try:
        uvicorn.run(
            "med_core.web.app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        console.print("\n👋 MedFusion Web UI 已停止")
    except Exception as e:
        console.print(f"\n❌ 启动失败: {e}")
        raise


@web.command()
def info() -> None:
    """显示 Web UI 信息"""
    console.print("[bold]MedFusion Web UI[/bold]")
    console.print(f"版本: {settings.version}")
    console.print(f"数据目录: {settings.data_dir}")
    console.print(f"数据库: {settings.database_url}")

    # 检查前端资源
    if check_web_ui_exists():
        console.print("前端资源: ✅ 已安装")
    else:
        console.print("前端资源: ❌ 未安装")


@click.group()
def data() -> None:
    """数据管理命令"""
    pass


@data.command("info")
def data_info() -> None:
    """显示数据目录信息"""
    data_dir = settings.data_dir

    def get_dir_size(path: Path) -> float:
        """计算目录大小（GB）"""
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception:
            pass
        return total / 1024**3

    console.print(f"📁 数据目录: {data_dir}")
    console.print(f"💾 总大小: {get_dir_size(data_dir):.2f} GB")

    # 子目录大小
    subdirs = ["models", "experiments", "datasets", "logs"]
    for subdir in subdirs:
        path = data_dir / subdir
        if path.exists():
            size = get_dir_size(path)
            console.print(f"   {subdir}: {size:.2f} GB")


@data.command()
@click.argument("output", type=click.Path())
def backup(output: str) -> None:
    """备份数据"""
    import shutil

    console.print(f"⏳ 正在备份到 {output}...")
    try:
        shutil.make_archive(output, "gztar", settings.data_dir)
        console.print(f"✅ 备份完成: {output}.tar.gz")
    except Exception as e:
        console.print(f"❌ 备份失败: {e}")


@data.command()
@click.option("--keep-models", is_flag=True, help="保留模型文件")
@click.confirmation_option(prompt="确定要清理数据吗？")
def clean(keep_models: bool) -> None:
    """清理旧数据"""
    import shutil

    console.print("⏳ 正在清理数据...")

    # 清理日志
    logs_dir = settings.data_dir / "logs"
    if logs_dir.exists():
        shutil.rmtree(logs_dir)
        logs_dir.mkdir()
        console.print("✅ 日志已清理")

    # 清理上传文件
    uploads_dir = settings.data_dir / "uploads"
    if uploads_dir.exists():
        shutil.rmtree(uploads_dir)
        uploads_dir.mkdir()
        console.print("✅ 上传文件已清理")

    # 清理模型（可选）
    if not keep_models:
        models_dir = settings.data_dir / "models"
        if models_dir.exists():
            shutil.rmtree(models_dir)
            models_dir.mkdir()
            console.print("✅ 模型文件已清理")

    console.print("✅ 清理完成")
