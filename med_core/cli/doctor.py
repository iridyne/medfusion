"""CLI entry point for config validation and doctor checks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from med_core.configs.doctor import analyze_config


def _print_issue_block(title: str, items: list[dict]) -> None:
    if not items:
        return
    print(f"\n{title}:")
    for index, item in enumerate(items, start=1):
        prefix = item.get("code") or "-"
        print(f"  {index}. [{prefix}] {item['path']}: {item['message']}")
        suggestion = item.get("suggestion")
        if suggestion:
            print(f"     -> {suggestion}")


def validate_config(
    argv: Sequence[str] | None = None,
    prog: str = "medfusion validate-config",
) -> None:
    """Validate config structure and dataset readiness."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Validate training config, dataset paths, CSV columns, and sample readiness.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument(
        "--image-sample-limit",
        type=int,
        default=32,
        help="How many image paths to probe from CSV for existence checks",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON report",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when warnings exist",
    )
    args = parser.parse_args(argv)

    report = analyze_config(args.config, image_sample_limit=max(args.image_sample_limit, 1))
    payload = report.to_dict()

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("MedFusion Config Doctor")
        print("")
        print(f"Config: {payload['config_path']}")
        print(
            "Summary: "
            f"errors={len(payload['errors'])}, "
            f"warnings={len(payload['warnings'])}, "
            f"dataset_rows={payload['summary'].get('dataset_rows', 0)}, "
            f"splits={payload['summary'].get('estimated_split_counts', {})}"
        )
        _print_issue_block("Errors", payload["errors"])
        _print_issue_block("Warnings", payload["warnings"])
        if payload["info"]:
            print("\nInfo:")
            for item in payload["info"]:
                print(f"  - {json.dumps(item, ensure_ascii=False)}")

    if payload["errors"]:
        raise SystemExit(2)
    if args.strict and payload["warnings"]:
        raise SystemExit(1)


def doctor(
    argv: Sequence[str] | None = None,
    prog: str = "medfusion doctor",
) -> None:
    """Alias of validate-config for a shorter entrypoint."""
    validate_config(argv=argv, prog=prog)
