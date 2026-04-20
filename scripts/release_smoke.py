#!/usr/bin/env python3
"""Run the MedFusion formal-release smoke paths.

This script is intentionally minimal and focuses on two deployment shapes:
1) local browser mode
2) Docker private-deployment mode
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/public_datasets/breastmnist_quickstart.yaml"
DEFAULT_PROFILE = "medmnist-breastmnist"
DATASET_METADATA = Path("data/public/medmnist/breastmnist-demo/metadata.csv")
OUTPUT_DIR = Path("outputs/public_datasets/breastmnist_quickstart")
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoints" / "best.pth"
ARTIFACTS = [
    OUTPUT_DIR / "checkpoints" / "best.pth",
    OUTPUT_DIR / "logs" / "history.json",
    OUTPUT_DIR / "metrics" / "metrics.json",
    OUTPUT_DIR / "metrics" / "validation.json",
    OUTPUT_DIR / "reports" / "summary.json",
    OUTPUT_DIR / "reports" / "report.md",
]


def _print_step(title: str) -> None:
    print()
    print(f"==> {title}")


def _run_command(command: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def _wait_http(url: str, timeout_seconds: int) -> None:
    start = time.time()
    last_error: str | None = None
    while time.time() - start < timeout_seconds:
        try:
            with urlopen(url, timeout=5) as response:  # noqa: S310
                if response.status < 500:
                    return
        except HTTPError as exc:
            # 4xx still means the service is reachable.
            if 400 <= exc.code < 500:
                return
            last_error = f"HTTP {exc.code}"
        except URLError as exc:
            last_error = str(exc.reason)
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}. Last error: {last_error}")


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _check_local_web(host: str, port: int, timeout_seconds: int) -> None:
    _print_step("Local Web start check")
    log_dir = REPO_ROOT / "outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "release_smoke_local_web.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "med_core.cli",
                "start",
                "--host",
                host,
                "--port",
                str(port),
                "--no-browser",
            ],
            cwd=str(REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_http(f"http://{host}:{port}/health", timeout_seconds)
            _wait_http(f"http://{host}:{port}/start", timeout_seconds)
            print(f"Local Web endpoints reachable on http://{host}:{port}")
        except Exception as exc:  # pragma: no cover - runtime-facing
            raise RuntimeError(f"Local Web start check failed. See: {log_path}") from exc
        finally:
            _stop_process(process)


def _prepare_dataset_if_needed(profile: str, force_prepare: bool) -> None:
    metadata_path = REPO_ROOT / DATASET_METADATA
    if force_prepare or not metadata_path.exists():
        _print_step("Prepare public dataset")
        _run_command(
            [
                sys.executable,
                "-m",
                "med_core.cli",
                "public-datasets",
                "prepare",
                profile,
                "--overwrite",
            ]
        )
        return

    _print_step("Reuse prepared dataset")
    print(f"Using existing metadata: {metadata_path}")


def _run_local_mainline(config_path: str) -> None:
    _print_step("Validate config")
    _run_command(
        [
            sys.executable,
            "-m",
            "med_core.cli",
            "validate-config",
            "--config",
            config_path,
        ]
    )

    _print_step("Train")
    _run_command(
        [sys.executable, "-m", "med_core.cli", "train", "--config", config_path]
    )

    _print_step("Build results")
    _run_command(
        [
            sys.executable,
            "-m",
            "med_core.cli",
            "build-results",
            "--config",
            config_path,
            "--checkpoint",
            str(CHECKPOINT_PATH),
        ]
    )


def _verify_artifacts() -> None:
    _print_step("Verify canonical artifacts")
    missing = [artifact for artifact in ARTIFACTS if not (REPO_ROOT / artifact).exists()]
    if missing:
        for artifact in missing:
            print(f"Missing artifact: {artifact}")
        raise RuntimeError("Smoke mainline failed: canonical artifacts are missing.")
    for artifact in ARTIFACTS:
        print(f"OK: {artifact}")


def run_local(*, host: str, port: int, timeout_seconds: int, profile: str, config_path: str, force_prepare: bool) -> None:
    _check_local_web(host, port, timeout_seconds)
    _prepare_dataset_if_needed(profile, force_prepare)
    _run_local_mainline(config_path)
    _verify_artifacts()


def run_docker(*, port: int, image: str, timeout_seconds: int) -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required for --mode docker but was not found in PATH.")

    _print_step("Build Docker image")
    _run_command(
        [
            "docker",
            "build",
            "-f",
            "docker/Dockerfile",
            "-t",
            image,
            ".",
        ]
    )

    container_name = f"medfusion-release-smoke-{int(time.time())}"
    _print_step("Run Docker container")
    _run_command(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            container_name,
            "-p",
            f"{port}:8000",
            image,
        ]
    )

    try:
        _print_step("Check Docker Web endpoints")
        _wait_http(f"http://127.0.0.1:{port}/health", timeout_seconds)
        _wait_http(f"http://127.0.0.1:{port}/start", timeout_seconds)
        print(f"Docker Web endpoints reachable on http://127.0.0.1:{port}")
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MedFusion formal-release smoke paths (local and/or Docker)."
    )
    parser.add_argument(
        "--mode",
        choices=["local", "docker", "all"],
        default="local",
        help="Smoke target to run.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for local web smoke.")
    parser.add_argument("--port", type=int, default=18080, help="Port for local web smoke.")
    parser.add_argument(
        "--docker-port",
        type=int,
        default=18081,
        help="Host port for Docker web smoke.",
    )
    parser.add_argument(
        "--docker-image",
        default="medfusion/medfusion:release-smoke",
        help="Docker image tag used by docker smoke.",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="Public dataset profile for local mainline smoke.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Config path for local mainline smoke.",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Always re-run public dataset preparation.",
    )
    parser.add_argument(
        "--web-timeout",
        type=int,
        default=90,
        help="Timeout seconds when waiting for Web endpoints.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.mode in {"local", "all"}:
        run_local(
            host=args.host,
            port=args.port,
            timeout_seconds=args.web_timeout,
            profile=args.profile,
            config_path=args.config,
            force_prepare=args.force_prepare,
        )

    if args.mode in {"docker", "all"}:
        run_docker(
            port=args.docker_port,
            image=args.docker_image,
            timeout_seconds=args.web_timeout,
        )

    _print_step("Smoke summary")
    print(f"Mode: {args.mode}")
    print("Formal-release smoke completed.")


if __name__ == "__main__":
    main()
