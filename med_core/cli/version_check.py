"""CLI version compatibility checks."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from med_core.version import __version__
from med_core.web.config import settings
from med_core.web.static_assets import resolve_static_asset_location


def _fetch_server_version(server_url: str, timeout: float) -> dict[str, Any]:
    base_url = server_url.rstrip("/")
    endpoint = f"{base_url}/api/system/version"
    request = urllib.request.Request(
        endpoint,
        headers={
            "Accept": "application/json",
            "X-Client-Version": __version__,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return {
                "ok": True,
                "url": endpoint,
                "status_code": response.getcode(),
                "backend_version": payload.get("backend"),
                "api_version": payload.get("api"),
                "frontend_required": payload.get("frontend_required"),
                "server_header_version": response.headers.get("X-Server-Version"),
            }
    except urllib.error.URLError as exc:
        return {
            "ok": False,
            "url": endpoint,
            "error": str(exc.reason),
        }
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {
            "ok": False,
            "url": endpoint,
            "error": str(exc),
        }


def _local_frontend_asset_summary() -> dict[str, Any]:
    location = resolve_static_asset_location(
        package_root=Path(__file__).resolve().parents[1] / "web",
        data_dir=settings.data_dir,
        version=settings.version,
    )
    if location is None:
        return {
            "installed": False,
            "source": None,
            "directory": None,
        }
    return {
        "installed": True,
        "source": location.source,
        "directory": str(location.directory),
    }


def _build_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="检查 MedFusion CLI、本地 Web 资源和运行中服务的版本一致性。",
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000",
        help="运行中服务地址（默认: http://127.0.0.1:8000）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="请求运行中服务的超时时间（秒）",
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="只做本地版本检查，不请求运行中服务",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="把服务不可达也视为失败（默认仅提示 warning）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON",
    )
    return parser


def version_check(argv: list[str] | None = None, prog: str = "medfusion version-check") -> None:
    parser = _build_parser(prog)
    args = parser.parse_args(argv)

    local = {
        "cli_version": __version__,
        "web_settings_version": settings.version,
        "cli_matches_web_settings": __version__ == settings.version,
        "frontend_assets": _local_frontend_asset_summary(),
    }

    server: dict[str, Any] | None = None
    issues: list[str] = []
    warnings: list[str] = []

    if not local["cli_matches_web_settings"]:
        issues.append("CLI version does not match web settings version.")

    if args.skip_server:
        server = {
            "checked": False,
            "reason": "skip-server",
        }
    else:
        server = _fetch_server_version(args.server_url, args.timeout)
        if not server.get("ok"):
            warning = f"Server version check failed: {server.get('error', 'unknown error')}"
            if args.strict:
                issues.append(warning)
            else:
                warnings.append(warning)
        else:
            backend_version = str(server.get("backend_version") or "")
            if backend_version != __version__:
                issues.append(
                    f"Running backend version mismatch: expected {__version__}, got {backend_version or '<empty>'}.",
                )

    ok = len(issues) == 0
    summary = {
        "ok": ok,
        "local": local,
        "server": server,
        "issues": issues,
        "warnings": warnings,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("MedFusion version-check")
        print(f"- CLI version: {local['cli_version']}")
        print(f"- Web settings version: {local['web_settings_version']}")
        print(
            f"- Local version alignment: {'ok' if local['cli_matches_web_settings'] else 'mismatch'}",
        )
        frontend_assets = local["frontend_assets"]
        if frontend_assets["installed"]:
            print(
                f"- Frontend assets: installed ({frontend_assets['source']}) @ {frontend_assets['directory']}",
            )
        else:
            print("- Frontend assets: not installed")
        if server and server.get("checked") is False:
            print("- Running server check: skipped")
        elif server and server.get("ok"):
            print(f"- Running server version: {server.get('backend_version')}")
            print(f"- API version: {server.get('api_version')}")
        else:
            print(
                f"- Running server check: failed ({(server or {}).get('error', 'unknown error')})",
            )
        if warnings:
            print("- Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        if issues:
            print("- Issues:")
            for issue in issues:
                print(f"  - {issue}")
        print(f"- Result: {'ok' if ok else 'failed'}")

    if not ok:
        raise SystemExit(1)
