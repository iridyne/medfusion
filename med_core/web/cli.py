"""Web UI CLI 命令"""

import contextlib
import socket
import tempfile
import webbrowser
from pathlib import Path

import click
import uvicorn
from rich.console import Console

from .config import settings
from .static_assets import StaticAssetLocation, resolve_static_asset_location

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
        f"无法在 {start_port}-{start_port + max_attempts} 范围内找到可用端口",
    )


def get_web_ui_location() -> StaticAssetLocation | None:
    """检查前端资源位置。"""
    return resolve_static_asset_location(
        package_root=Path(__file__).parent,
        data_dir=settings.data_dir,
        version=settings.version,
    )


def check_web_ui_exists() -> bool:
    """检查前端资源是否存在"""
    return get_web_ui_location() is not None


def _first_run_marker_path() -> Path:
    return settings.data_dir / "settings" / ".web-first-run-complete"


def _is_first_run() -> bool:
    return not _first_run_marker_path().exists()


def _mark_first_run_complete() -> None:
    marker = _first_run_marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")


def initialize_web_server() -> bool:
    """初始化 Web 服务器"""
    console.print("初始化数据目录...")
    settings.initialize_directories()
    console.print("数据目录初始化完成")
    first_run = _is_first_run()

    console.print("检查前端资源...")
    location = get_web_ui_location()
    if location is None:
        console.print("前端资源未安装")
        console.print(
            "若你是源码运行，请先执行: `cd web/frontend && node --run build`，"
            "然后把 `dist/*` 同步到 `med_core/web/static/`。",
        )
    else:
        source_label = "内置资源" if location.source == "bundled" else "下载资源"
        console.print(f"前端资源就绪（{source_label}）")
    if first_run:
        console.print("首次启动引导: 推荐从 /start 进入，再按 /config -> /training -> /models 主线体验。")
        _mark_first_run_complete()
    return first_run


def _start_entrypoint_hint(command_path: str) -> str | None:
    normalized = " ".join(command_path.strip().split()).lower()
    if normalized.endswith("web start"):
        return "提示: 推荐直接使用 `medfusion start`，`medfusion web start` 仅作为兼容入口。"
    return None


@click.group()
def web() -> None:
    """Web UI 管理命令"""


@web.command()
@click.pass_context
@click.option("--host", default="127.0.0.1", help="监听地址")
@click.option("--port", default=None, type=int, help="端口")
@click.option("--auth", is_flag=True, help="启用认证")
@click.option("--token", default=None, help="自定义 Token")
@click.option("--no-browser", is_flag=True, help="不自动打开浏览器")
@click.option("--reload", is_flag=True, help="开发模式（自动重载）")
def start(
    ctx: click.Context,
    host: str,
    port: int | None,
    auth: bool,
    token: str | None,
    no_browser: bool,
    reload: bool,
) -> None:
    """启动 MedFusion Web UI"""
    # 初始化
    console.print("\n[bold blue]MedFusion Web UI[/bold blue]")
    console.print(f"版本: {settings.version}\n")
    hint = _start_entrypoint_hint(ctx.command_path)
    if hint:
        console.print(hint)

    first_run = initialize_web_server()

    # 查找可用端口
    if port is None:
        port = find_free_port()
        if port != 8000:
            console.print(f"端口 8000 已被占用，使用端口 {port}")

    # 认证配置
    if auth:
        # TODO: 实现认证
        console.print("认证已启用")
        if token:
            console.print(f"Token: {token}")

    # 警告
    if host != "127.0.0.1":
        console.print(f"监听所有网络接口 ({host})，建议启用认证: --auth")

    # 启动信息
    console.print("\n启动成功")
    console.print(f"访问地址: [link]http://{host}:{port}[/link]")
    console.print("按 Ctrl+C 停止服务器\n")

    # 自动打开浏览器
    if not no_browser and host == "127.0.0.1":
        with contextlib.suppress(Exception):
            landing = "/start" if first_run else ""
            webbrowser.open(f"http://{host}:{port}{landing}")

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
        console.print("\nMedFusion Web UI 已停止")
    except Exception as e:
        console.print(f"\n启动失败: {e}")
        raise


@web.command()
def info() -> None:
    """显示 Web UI 信息"""
    console.print("[bold]MedFusion Web UI[/bold]")
    console.print(f"版本: {settings.version}")
    console.print(f"数据目录: {settings.data_dir}")
    console.print(f"数据库: {settings.database_url}")

    # 检查前端资源
    location = get_web_ui_location()
    if location is not None:
        source_label = "内置资源" if location.source == "bundled" else "下载资源"
        console.print(f"前端资源: 已安装（{source_label}）")
        console.print(f"静态目录: {location.directory}")
    else:
        console.print("前端资源: 未安装")


@click.group()
def data() -> None:
    """数据管理命令"""


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

    console.print(f"数据目录: {data_dir}")
    console.print(f"总大小: {get_dir_size(data_dir):.2f} GB")

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

    console.print(f"正在备份到 {output}...")
    try:
        shutil.make_archive(output, "gztar", settings.data_dir)
        console.print(f"备份完成: {output}.tar.gz")
    except Exception as e:
        console.print(f"备份失败: {e}")


@data.command()
@click.argument("archive", type=click.Path(exists=True, dir_okay=False))
@click.option("--overwrite", is_flag=True, help="覆盖现有数据目录")
@click.option("--dry-run", is_flag=True, help="仅检查备份结构，不写入文件")
@click.confirmation_option(prompt="确定要从备份恢复数据吗？")
def restore(archive: str, overwrite: bool, dry_run: bool) -> None:
    """从备份恢复数据目录"""
    import shutil

    archive_path = Path(archive).resolve()
    data_dir = settings.data_dir.resolve()
    data_dir.parent.mkdir(parents=True, exist_ok=True)

    if data_dir.exists() and any(data_dir.iterdir()) and not overwrite:
        raise click.ClickException("目标数据目录非空，请加 --overwrite 执行恢复。")

    console.print(f"正在读取备份: {archive_path}")
    with tempfile.TemporaryDirectory(prefix="medfusion-restore-") as temp_dir:
        temp_root = Path(temp_dir)
        try:
            shutil.unpack_archive(str(archive_path), str(temp_root))
        except Exception as exc:
            raise click.ClickException(f"备份解压失败: {exc}") from exc

        extracted_dirs = [item for item in temp_root.iterdir() if item.is_dir()]
        extracted_files = [item for item in temp_root.iterdir() if item.is_file()]
        if len(extracted_dirs) == 1 and not extracted_files:
            extracted_root = extracted_dirs[0]
        else:
            # 兼容历史 backup 格式：根目录直接是数据内容
            extracted_root = temp_root

        if dry_run:
            file_count = sum(1 for entry in extracted_root.rglob("*") if entry.is_file())
            console.print(f"dry-run: 备份根目录 {extracted_root.name}，文件数 {file_count}")
            return

        if data_dir.exists():
            shutil.rmtree(data_dir)
        shutil.copytree(extracted_root, data_dir)
        console.print(f"恢复完成: {data_dir}")


@data.command()
@click.option("--keep-models", is_flag=True, help="保留模型文件")
@click.confirmation_option(prompt="确定要清理数据吗？")
def clean(keep_models: bool) -> None:
    """清理旧数据"""
    import shutil

    console.print("正在清理数据...")

    # 清理日志
    logs_dir = settings.data_dir / "logs"
    if logs_dir.exists():
        shutil.rmtree(logs_dir)
        logs_dir.mkdir()
        console.print("日志已清理")

    # 清理上传文件
    uploads_dir = settings.data_dir / "uploads"
    if uploads_dir.exists():
        shutil.rmtree(uploads_dir)
        uploads_dir.mkdir()
        console.print("上传文件已清理")

    # 清理模型（可选）
    if not keep_models:
        models_dir = settings.data_dir / "models"
        if models_dir.exists():
            shutil.rmtree(models_dir)
            models_dir.mkdir()
            console.print("模型文件已清理")

    console.print("清理完成")
