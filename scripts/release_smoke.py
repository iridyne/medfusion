#!/usr/bin/env python3
"""Run the MedFusion formal-release smoke paths.

This script is intentionally minimal and focuses on two deployment shapes:
1) local browser mode
2) Docker private-deployment mode
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

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


def _request_json(
    method: str,
    url: str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    request = Request(
        url,
        data=(
            json.dumps(payload).encode("utf-8")
            if payload is not None
            else None
        ),
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310
        body = response.read().decode("utf-8")
        return json.loads(body) if body else {}


def _check_local_custom_model_capabilities(host: str, port: int) -> None:
    _print_step("Check local custom model + preference APIs")
    base_url = f"http://{host}:{port}/api"

    preferences = _request_json("GET", f"{base_url}/system/preferences")
    if preferences.get("storage") != "filesystem":
        raise RuntimeError("UI preferences are not backed by filesystem storage.")

    updated_preferences = _request_json(
        "PUT",
        f"{base_url}/system/preferences",
        {"history_display_mode": "technical"},
    )
    if updated_preferences.get("preferences", {}).get("history_display_mode") != "technical":
        raise RuntimeError("Failed to update machine-wide UI preferences.")

    custom_model_payload = {
        "schema_version": "0.1",
        "id": "release-smoke-custom-model",
        "source": "custom",
        "label": "Release Smoke Custom Model",
        "description": "filesystem smoke",
        "status": "local_custom",
        "based_on_model_id": "quickstart_multimodal",
        "unit_map": {
            "vision_encoder": "resnet18_encoder_bundle",
            "tabular_encoder": "mlp_tabular_encoder_bundle",
            "fusion_bundle": "concatenate_fusion_bundle",
            "task_head": "classification_head_bundle",
            "training_strategy": "standard_training_bundle",
        },
        "editable_slots": [
            "vision_encoder",
            "tabular_encoder",
            "fusion_bundle",
            "task_head",
            "training_strategy",
        ],
        "component_ids": [
            "resnet18_encoder_bundle",
            "mlp_tabular_encoder_bundle",
            "concatenate_fusion_bundle",
            "classification_head_bundle",
            "standard_training_bundle",
        ],
        "data_requirements": ["CSV + image_dir", "至少 1 个表格特征"],
        "compute_profile": {
            "tier": "light",
            "gpu_vram_hint": "8GB+",
            "notes": "release smoke",
        },
        "wizard_prefill": {
            "modelTemplateId": "quickstart_multimodal",
            "customModelLabel": "Release Smoke Custom Model",
        },
        "created_at": "2026-04-21T00:00:00",
        "updated_at": "2026-04-21T00:00:00",
    }
    _request_json("POST", f"{base_url}/models/custom", custom_model_payload)
    custom_list = _request_json("GET", f"{base_url}/models/custom")
    if not any(item.get("id") == "release-smoke-custom-model" for item in custom_list.get("items", [])):
        raise RuntimeError("Failed to persist custom model to local filesystem store.")

    _request_json("DELETE", f"{base_url}/models/custom/release-smoke-custom-model")
    custom_list_after_delete = _request_json("GET", f"{base_url}/models/custom")
    if not any(item.get("id") == "release-smoke-custom-model" for item in custom_list_after_delete.get("trash_items", [])):
        raise RuntimeError("Deleted custom model was not moved to recycle bin.")


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
            _wait_http(f"http://{host}:{port}/evaluation", timeout_seconds)
            _check_local_custom_model_capabilities(host, port)
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
        _wait_http(f"http://127.0.0.1:{port}/evaluation", timeout_seconds)
        print(f"Docker Web endpoints reachable on http://127.0.0.1:{port}")
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _normalize_compose_environment(
    raw_environment: object,
) -> dict[str, str]:
    if isinstance(raw_environment, Mapping):
        return {str(k): str(v) for k, v in raw_environment.items()}

    if isinstance(raw_environment, list):
        normalized: dict[str, str] = {}
        for item in raw_environment:
            if not isinstance(item, str):
                continue
            if "=" in item:
                key, value = item.split("=", 1)
                normalized[key] = value
            else:
                normalized[item] = ""
        return normalized

    return {}


def run_docker_dry_run() -> None:
    """Validate Docker artifacts without requiring Docker daemon/runtime."""
    _print_step("Docker dry-run: validate Dockerfile and compose contracts")

    dockerfile_path = REPO_ROOT / "docker" / "Dockerfile"
    compose_path = REPO_ROOT / "docker" / "docker-compose.yml"

    if not dockerfile_path.exists():
        raise RuntimeError(f"Missing Dockerfile: {dockerfile_path}")
    if not compose_path.exists():
        raise RuntimeError(f"Missing compose file: {compose_path}")

    dockerfile_text = dockerfile_path.read_text(encoding="utf-8")
    required_dockerfile_fragments = [
        "FROM python:3.11-slim",
        "uv pip install \".[web]\"",
        "HEALTHCHECK",
        "\"python\", \"-m\", \"med_core.cli\", \"start\"",
    ]
    for fragment in required_dockerfile_fragments:
        if fragment not in dockerfile_text:
            raise RuntimeError(f"Dockerfile missing required fragment: {fragment}")

    compose_data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    if not isinstance(compose_data, dict):
        raise RuntimeError("Compose file parse failed: top-level is not a mapping.")

    services = compose_data.get("services")
    if not isinstance(services, dict):
        raise RuntimeError("Compose file parse failed: services section is missing.")

    required_services = ("medfusion-web", "postgres", "redis")
    for service_name in required_services:
        if service_name not in services:
            raise RuntimeError(f"Compose missing required service: {service_name}")

    medfusion_web = services["medfusion-web"]
    if not isinstance(medfusion_web, dict):
        raise RuntimeError("Compose service medfusion-web is invalid.")

    environment = _normalize_compose_environment(medfusion_web.get("environment"))
    required_env_keys = (
        "MEDFUSION_DATABASE_URL",
        "MEDFUSION_REDIS_URL",
        "MEDFUSION_TRAINING_QUEUE_BACKEND",
    )
    for key in required_env_keys:
        if key not in environment:
            raise RuntimeError(f"Compose medfusion-web missing env: {key}")

    depends_on = medfusion_web.get("depends_on")
    if not isinstance(depends_on, dict):
        raise RuntimeError("Compose medfusion-web depends_on should be a mapping.")
    for dependency in ("postgres", "redis"):
        if dependency not in depends_on:
            raise RuntimeError(f"Compose medfusion-web missing dependency: {dependency}")

    for service_name in required_services:
        service = services[service_name]
        if not isinstance(service, dict):
            raise RuntimeError(f"Compose service {service_name} is invalid.")
        if "healthcheck" not in service:
            raise RuntimeError(f"Compose service {service_name} missing healthcheck.")

    print("Docker dry-run checks passed:")
    print(f"- Dockerfile: {dockerfile_path}")
    print(f"- Compose: {compose_path}")
    print("- Required services/env/dependencies/healthchecks: OK")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MedFusion formal-release smoke paths (local and/or Docker)."
    )
    parser.add_argument(
        "--mode",
        choices=["local", "docker", "docker-dry-run", "all"],
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
    if args.mode == "docker-dry-run":
        run_docker_dry_run()

    _print_step("Smoke summary")
    print(f"Mode: {args.mode}")
    print("Formal-release smoke completed.")


if __name__ == "__main__":
    main()
