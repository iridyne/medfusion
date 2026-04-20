"""CLI entry point for uninstalling MedFusion runtime assets."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _is_dangerous_target(path: Path) -> bool:
    resolved = path.resolve()
    anchor = Path(resolved.anchor)
    if resolved == anchor:
        return True
    if resolved == Path.home():
        return True
    return False


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def uninstall(
    argv: Sequence[str] | None = None,
    prog: str = "medfusion uninstall",
) -> None:
    """Uninstall local MedFusion runtime assets."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Uninstall local MedFusion runtime assets. "
            "Default mode keeps data, --purge-data removes local outputs and user data."
        ),
    )
    parser.add_argument(
        "--purge-data",
        action="store_true",
        help="Also delete local outputs/logs/checkpoints and user data directory",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting files",
    )
    parser.add_argument(
        "--venv-path",
        default=".venv",
        help="Runtime environment directory to remove",
    )
    parser.add_argument(
        "--user-data-dir",
        default=str(Path.home() / ".medfusion"),
        help="User data directory used in --purge-data mode",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output",
    )
    args = parser.parse_args(argv)

    cwd = Path.cwd()
    venv_path = (cwd / args.venv_path).resolve()
    purge_paths = [
        (cwd / "outputs").resolve(),
        (cwd / "logs").resolve(),
        (cwd / "checkpoints").resolve(),
        Path(args.user_data_dir).expanduser().resolve(),
    ]

    targets = [venv_path]
    mode = "keep-data"
    if args.purge_data:
        mode = "purge-data"
        targets.extend(purge_paths)

    deduped_targets: list[Path] = []
    seen: set[str] = set()
    for target in targets:
        key = str(target)
        if key in seen:
            continue
        seen.add(key)
        deduped_targets.append(target)

    dangerous = [str(path) for path in deduped_targets if _is_dangerous_target(path)]
    if dangerous:
        message = "Refuse to remove dangerous target(s): " + ", ".join(dangerous)
        if args.json:
            print(json.dumps({"ok": False, "error": message}, ensure_ascii=False, indent=2))
        else:
            print(f"❌ {message}")
        raise SystemExit(2)

    if not args.yes and not args.dry_run:
        print("MedFusion Uninstall")
        print("")
        print(f"Mode: {mode}")
        print("Targets:")
        for target in deduped_targets:
            print(f"  - {target}")
        confirm = input("Proceed with uninstall? [y/N]: ").strip().lower()
        if confirm not in {"y", "yes"}:
            print("Cancelled.")
            return

    removed: list[str] = []
    would_remove: list[str] = []
    missing: list[str] = []
    failed: list[dict[str, str]] = []

    for target in deduped_targets:
        if not target.exists():
            missing.append(str(target))
            continue

        if args.dry_run:
            would_remove.append(str(target))
            continue

        try:
            _remove_path(target)
            removed.append(str(target))
        except Exception as exc:  # pragma: no cover - exercised in integration/manual runs
            failed.append({"path": str(target), "error": str(exc)})

    payload: dict[str, Any] = {
        "ok": len(failed) == 0,
        "mode": mode,
        "dry_run": args.dry_run,
        "targets": [str(path) for path in deduped_targets],
        "removed": removed,
        "would_remove": would_remove,
        "missing": missing,
        "failed": failed,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("MedFusion Uninstall")
        print("")
        print(f"Mode: {mode}")
        if args.dry_run:
            print(f"Would remove: {len(would_remove)}")
        else:
            print(f"Removed: {len(removed)}")
        print(f"Missing: {len(missing)}")
        print(f"Failed: {len(failed)}")

    if failed:
        raise SystemExit(1)
