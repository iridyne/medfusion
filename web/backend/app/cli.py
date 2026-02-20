"""MedFusion Web UI 命令行工具

提供简洁的命令行接口来管理 Web UI 服务。

使用方式:
    web start          # 启动前后端服务
    web start-backend  # 只启动后端
    web start-frontend # 只启动前端
    web stop           # 停止所有服务
    web status         # 查看服务状态
    web logs           # 查看日志
"""

import subprocess
import sys
import time
from pathlib import Path

import click
import psutil

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
LOGS_DIR = PROJECT_ROOT / "logs"
PID_DIR = PROJECT_ROOT / "logs"


def ensure_dirs():
    """确保必要的目录存在"""
    LOGS_DIR.mkdir(exist_ok=True)
    PID_DIR.mkdir(exist_ok=True)


def get_pid(service: str) -> int | None:
    """获取服务的 PID"""
    pid_file = PID_DIR / f"{service}.pid"
    if pid_file.exists():
        try:
            return int(pid_file.read_text().strip())
        except (ValueError, FileNotFoundError):
            return None
    return None


def save_pid(service: str, pid: int):
    """保存服务的 PID"""
    pid_file = PID_DIR / f"{service}.pid"
    pid_file.write_text(str(pid))


def is_process_running(pid: int) -> bool:
    """检查进程是否运行"""
    try:
        process = psutil.Process(pid)
        return process.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def kill_process(pid: int, timeout: int = 5):
    """终止进程"""
    try:
        process = psutil.Process(pid)
        process.terminate()

        # 等待进程终止
        try:
            process.wait(timeout=timeout)
        except psutil.TimeoutExpired:
            # 强制杀死
            process.kill()
            process.wait(timeout=2)

        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def check_port(port: int) -> bool:
    """检查端口是否被占用"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@click.group()
@click.version_option(version="0.1.0", prog_name="web")
def cli():
    """MedFusion Web UI 命令行工具

    一个用于医学深度学习的 Web 界面管理工具。
    """
    ensure_dirs()


@cli.command()
@click.option("--host", default="0.0.0.0", help="后端服务主机地址")
@click.option("--port", default=8000, help="后端服务端口")
@click.option("--reload", is_flag=True, help="开发模式（热重载）")
@click.option("--daemon", is_flag=True, help="后台运行")
def start_backend(host: str, port: int, reload: bool, daemon: bool):
    """启动后端 API 服务"""
    click.echo(click.style("🚀 启动后端服务...", fg="blue", bold=True))

    # 检查是否已经运行
    pid = get_pid("backend")
    if pid and is_process_running(pid):
        click.echo(click.style(f"❌ 后端服务已在运行 (PID: {pid})", fg="red"))
        return

    # 检查端口
    if check_port(port):
        click.echo(click.style(f"❌ 端口 {port} 已被占用", fg="red"))
        return

    # 构建启动命令
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    if reload:
        cmd.append("--reload")

    # 启动服务
    try:
        if daemon:
            # 后台运行
            log_file = LOGS_DIR / "backend.log"
            with open(log_file, "w") as f:
                process = subprocess.Popen(
                    cmd,
                    cwd=BACKEND_DIR,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

            save_pid("backend", process.pid)
            click.echo(
                click.style(f"✅ 后端服务已启动 (PID: {process.pid})", fg="green")
            )
            click.echo(f"   API 地址: http://{host}:{port}")
            click.echo(f"   API 文档: http://{host}:{port}/docs")
            click.echo(f"   日志文件: {log_file}")
        else:
            # 前台运行
            click.echo(click.style("✅ 后端服务启动中...", fg="green"))
            click.echo(f"   API 地址: http://{host}:{port}")
            click.echo(f"   API 文档: http://{host}:{port}/docs")
            click.echo(click.style("\n按 Ctrl+C 停止服务\n", fg="yellow"))

            subprocess.run(cmd, cwd=BACKEND_DIR)

    except KeyboardInterrupt:
        click.echo(click.style("\n⏹️  后端服务已停止", fg="yellow"))
    except Exception as e:
        click.echo(click.style(f"❌ 启动失败: {e}", fg="red"))
        sys.exit(1)


@cli.command()
@click.option("--port", default=5173, help="前端服务端口")
@click.option("--daemon", is_flag=True, help="后台运行")
def start_frontend(port: int, daemon: bool):
    """启动前端开发服务器"""
    click.echo(click.style("🚀 启动前端服务...", fg="blue", bold=True))

    # 检查是否已经运行
    pid = get_pid("frontend")
    if pid and is_process_running(pid):
        click.echo(click.style(f"❌ 前端服务已在运行 (PID: {pid})", fg="red"))
        return

    # 检查 node_modules
    if not (FRONTEND_DIR / "node_modules").exists():
        click.echo(
            click.style("⚠️  未检测到 node_modules，正在安装依赖...", fg="yellow")
        )
        subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)

    # 构建启动命令
    cmd = ["npm", "run", "dev", "--", "--port", str(port)]

    # 启动服务
    try:
        if daemon:
            # 后台运行
            log_file = LOGS_DIR / "frontend.log"
            with open(log_file, "w") as f:
                process = subprocess.Popen(
                    cmd,
                    cwd=FRONTEND_DIR,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

            save_pid("frontend", process.pid)
            click.echo(
                click.style(f"✅ 前端服务已启动 (PID: {process.pid})", fg="green")
            )
            click.echo(f"   访问地址: http://localhost:{port}")
            click.echo(f"   日志文件: {log_file}")
        else:
            # 前台运行
            click.echo(click.style("✅ 前端服务启动中...", fg="green"))
            click.echo(f"   访问地址: http://localhost:{port}")
            click.echo(click.style("\n按 Ctrl+C 停止服务\n", fg="yellow"))

            subprocess.run(cmd, cwd=FRONTEND_DIR)

    except KeyboardInterrupt:
        click.echo(click.style("\n⏹️  前端服务已停止", fg="yellow"))
    except Exception as e:
        click.echo(click.style(f"❌ 启动失败: {e}", fg="red"))
        sys.exit(1)


@cli.command()
@click.option("--backend-host", default="0.0.0.0", help="后端服务主机地址")
@click.option("--backend-port", default=8000, help="后端服务端口")
@click.option("--frontend-port", default=5173, help="前端服务端口")
@click.option("--reload", is_flag=True, help="开发模式（热重载）")
@click.option("--daemon", is_flag=True, help="后台运行")
def start(
    backend_host: str, backend_port: int, frontend_port: int, reload: bool, daemon: bool
):
    """启动完整的 Web UI 服务（前端 + 后端）"""
    click.echo(click.style("🚀 启动 MedFusion Web UI", fg="blue", bold=True))
    click.echo()

    # 启动后端
    ctx = click.get_current_context()
    ctx.invoke(
        start_backend,
        host=backend_host,
        port=backend_port,
        reload=reload,
        daemon=True,  # 后端总是后台运行
    )

    # 等待后端启动
    click.echo(click.style("⏳ 等待后端服务启动...", fg="yellow"))
    time.sleep(3)

    # 检查后端健康状态
    if check_port(backend_port):
        click.echo(click.style("✅ 后端服务健康检查通过", fg="green"))
    else:
        click.echo(click.style("⚠️  后端服务可能未完全启动", fg="yellow"))

    click.echo()

    # 启动前端
    ctx.invoke(start_frontend, port=frontend_port, daemon=daemon)

    if daemon:
        click.echo()
        click.echo(click.style("=" * 60, fg="cyan"))
        click.echo(click.style("✨ MedFusion Web UI 已启动", fg="green", bold=True))
        click.echo(click.style("=" * 60, fg="cyan"))
        click.echo()
        click.echo(f"  🌐 前端界面: http://localhost:{frontend_port}")
        click.echo(f"  🔌 后端 API: http://{backend_host}:{backend_port}")
        click.echo(f"  📚 API 文档: http://{backend_host}:{backend_port}/docs")
        click.echo()
        click.echo(click.style("管理命令:", fg="cyan"))
        click.echo("  web status  # 查看服务状态")
        click.echo("  web logs    # 查看日志")
        click.echo("  web stop    # 停止服务")
        click.echo()


@cli.command()
@click.option(
    "--service",
    type=click.Choice(["backend", "frontend", "all"]),
    default="all",
    help="要停止的服务",
)
def stop(service: str):
    """停止 Web UI 服务"""
    click.echo(click.style("⏹️  停止服务...", fg="yellow", bold=True))

    services = ["backend", "frontend"] if service == "all" else [service]

    for svc in services:
        pid = get_pid(svc)
        if pid and is_process_running(pid):
            click.echo(f"停止 {svc} 服务 (PID: {pid})...")
            if kill_process(pid):
                click.echo(click.style(f"✅ {svc} 服务已停止", fg="green"))
                # 删除 PID 文件
                pid_file = PID_DIR / f"{svc}.pid"
                pid_file.unlink(missing_ok=True)
            else:
                click.echo(click.style(f"❌ 停止 {svc} 服务失败", fg="red"))
        else:
            click.echo(click.style(f"⚠️  {svc} 服务未运行", fg="yellow"))


@cli.command()
def status():
    """查看服务状态"""
    click.echo(click.style("📊 服务状态", fg="blue", bold=True))
    click.echo()

    services = ["backend", "frontend"]

    for service in services:
        pid = get_pid(service)
        if pid and is_process_running(pid):
            try:
                process = psutil.Process(pid)
                cpu = process.cpu_percent(interval=0.1)
                mem = process.memory_info().rss / 1024 / 1024  # MB

                click.echo(f"  {service.capitalize()}: ", nl=False)
                click.echo(click.style("✅ 运行中", fg="green"), nl=False)
                click.echo(f" (PID: {pid}, CPU: {cpu:.1f}%, 内存: {mem:.1f}MB)")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                click.echo(f"  {service.capitalize()}: ", nl=False)
                click.echo(click.style("❌ 已停止", fg="red"))
        else:
            click.echo(f"  {service.capitalize()}: ", nl=False)
            click.echo(click.style("❌ 已停止", fg="red"))

    click.echo()

    # 检查端口
    if check_port(8000):
        click.echo("  后端端口 8000: ", nl=False)
        click.echo(click.style("✅ 可访问", fg="green"))

    if check_port(5173):
        click.echo("  前端端口 5173: ", nl=False)
        click.echo(click.style("✅ 可访问", fg="green"))


@cli.command()
@click.option(
    "--service",
    type=click.Choice(["backend", "frontend", "all"]),
    default="all",
    help="要查看的日志",
)
@click.option("--follow", "-f", is_flag=True, help="实时跟踪日志")
@click.option("--lines", "-n", default=50, help="显示的行数")
def logs(service: str, follow: bool, lines: int):
    """查看服务日志"""
    services = ["backend", "frontend"] if service == "all" else [service]

    log_files = [LOGS_DIR / f"{svc}.log" for svc in services]
    existing_logs = [f for f in log_files if f.exists()]

    if not existing_logs:
        click.echo(click.style("⚠️  没有找到日志文件", fg="yellow"))
        return

    if follow:
        # 实时跟踪日志
        cmd = ["tail", "-f"] + [str(f) for f in existing_logs]
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            click.echo()
    else:
        # 显示最近的日志
        for log_file in existing_logs:
            click.echo(click.style(f"\n=== {log_file.name} ===", fg="cyan", bold=True))
            cmd = ["tail", "-n", str(lines), str(log_file)]
            subprocess.run(cmd)


@cli.command()
def init():
    """初始化 Web UI 环境（安装依赖、初始化数据库）"""
    click.echo(click.style("🔧 初始化 MedFusion Web UI", fg="blue", bold=True))
    click.echo()

    # 安装后端依赖
    click.echo(click.style("📦 安装后端依赖...", fg="cyan"))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=BACKEND_DIR,
        check=True,
    )
    click.echo(click.style("✅ 后端依赖安装完成", fg="green"))
    click.echo()

    # 初始化数据库
    click.echo(click.style("🗄️  初始化数据库...", fg="cyan"))
    init_db_script = BACKEND_DIR / "scripts" / "init_db.py"
    if init_db_script.exists():
        subprocess.run(
            [sys.executable, str(init_db_script)], cwd=BACKEND_DIR, check=True
        )
        click.echo(click.style("✅ 数据库初始化完成", fg="green"))
    else:
        click.echo(click.style("⚠️  数据库初始化脚本不存在，跳过", fg="yellow"))
    click.echo()

    # 安装前端依赖
    click.echo(click.style("📦 安装前端依赖...", fg="cyan"))
    subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)
    click.echo(click.style("✅ 前端依赖安装完成", fg="green"))
    click.echo()

    click.echo(click.style("=" * 60, fg="cyan"))
    click.echo(click.style("✨ 初始化完成！", fg="green", bold=True))
    click.echo(click.style("=" * 60, fg="cyan"))
    click.echo()
    click.echo("现在可以运行以下命令启动服务：")
    click.echo(click.style("  web start", fg="cyan", bold=True))
    click.echo()


if __name__ == "__main__":
    cli()
