"""User-facing CLI entry point for the YAML mainline end-to-end run flow."""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from med_core.configs import load_config
from med_core.output_layout import RunOutputLayout, format_oss_display_path

from .build_results import build_results
from .doctor import validate_config
from .train import train


def _capture_json_command(
    command: Callable[..., None],
    argv: Sequence[str],
    prog: str,
) -> dict[str, Any]:
    buffer = io.StringIO()
    original_streams = _redirect_stream_handlers(sys.stderr)
    try:
        with redirect_stdout(buffer):
            command(argv=list(argv), prog=prog)
    except BaseException:
        captured_output = buffer.getvalue()
        if captured_output:
            sys.stderr.write(captured_output)
        raise
    finally:
        _restore_stream_handlers(original_streams)
    return json.loads(buffer.getvalue())


def _redirect_stream_handlers(
    target_stream: Any,
) -> list[tuple[logging.StreamHandler[Any], Any]]:
    original_streams: list[tuple[logging.StreamHandler[Any], Any]] = []
    seen_handler_ids: set[int] = set()

    for logger in (logging.getLogger(), logging.getLogger("med_core.cli.train")):
        for handler in logger.handlers:
            if not isinstance(handler, logging.StreamHandler):
                continue
            if id(handler) in seen_handler_ids:
                continue
            seen_handler_ids.add(id(handler))
            original_streams.append((handler, handler.stream))
            handler.setStream(target_stream)

    return original_streams


def _restore_stream_handlers(
    original_streams: list[tuple[logging.StreamHandler[Any], Any]],
) -> None:
    for handler, stream in original_streams:
        handler.setStream(stream)


def _run_train_command(
    argv: Sequence[str],
    *,
    prog: str,
    json_mode: bool,
) -> None:
    if not json_mode:
        train(argv, prog=prog)
        return

    buffer = io.StringIO()
    original_streams = _redirect_stream_handlers(sys.stderr)
    try:
        with redirect_stdout(buffer):
            train(argv, prog=prog)
    except BaseException:
        captured_output = buffer.getvalue()
        if captured_output:
            sys.stderr.write(captured_output)
        raise
    finally:
        _restore_stream_handlers(original_streams)


def _build_payload(
    *,
    config_path: str,
    run_layout: RunOutputLayout,
    build_results_skipped: bool,
    build_results_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    checkpoint_path = run_layout.checkpoints_dir / "best.pth"
    artifact_paths = {
        "checkpoint": str(checkpoint_path),
        "history": str(run_layout.history_path),
    }
    if build_results_payload is not None:
        result_paths = build_results_payload.get("artifact_paths", {})
        artifact_paths.update(
            {
                "metrics": result_paths.get("metrics_path", str(run_layout.metrics_path)),
                "validation": result_paths.get(
                    "validation_path", str(run_layout.validation_path)
                ),
                "summary": result_paths.get("summary_path", str(run_layout.summary_path)),
                "report": result_paths.get("report_path", str(run_layout.report_path)),
            }
        )

    return {
        "config_path": config_path,
        "output_dir": str(run_layout.root_dir),
        "checkpoint": str(checkpoint_path),
        "build_results_skipped": build_results_skipped,
        "artifact_paths": artifact_paths,
    }


def _print_human_summary(payload: dict[str, Any]) -> None:
    print("MedFusion Run")
    print("")
    print(f"Config: {payload['config_path']}")
    print(f"Output: {format_oss_display_path(payload['output_dir'])}")
    print(f"Checkpoint: {format_oss_display_path(payload['checkpoint'])}")
    print(f"History: {format_oss_display_path(payload['artifact_paths']['history'])}")

    if payload["build_results_skipped"]:
        print("Build results: skipped")
        return

    print(f"Metrics: {format_oss_display_path(payload['artifact_paths']['metrics'])}")
    print(
        f"Validation: {format_oss_display_path(payload['artifact_paths']['validation'])}"
    )
    print(f"Summary: {format_oss_display_path(payload['artifact_paths']['summary'])}")
    print(f"Report: {format_oss_display_path(payload['artifact_paths']['report'])}")


def run(
    argv: Sequence[str] | None = None,
    prog: str = "medfusion run",
) -> None:
    """Run the YAML mainline end-to-end: validate, train, and build results."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Run the MedFusion YAML mainline end-to-end: "
            "validate-config -> train -> build-results."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument(
        "--skip-build-results",
        action="store_true",
        help="Stop after training and do not generate validation/report artifacts",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final run summary as JSON",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    output_dir = args.output_dir or config.logging.output_dir
    run_layout = RunOutputLayout(output_dir).ensure_exists()

    _capture_json_command(
        validate_config,
        ["--config", args.config, "--json"],
        "medfusion validate-config",
    )
    train_args = ["--config", args.config]
    if args.output_dir:
        train_args.extend(["--output-dir", args.output_dir])
    _run_train_command(train_args, prog="medfusion train", json_mode=args.json)

    checkpoint_path = run_layout.checkpoints_dir / "best.pth"
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Training completed but canonical checkpoint is missing: {checkpoint_path}"
        )

    build_results_payload: dict[str, Any] | None = None
    if not args.skip_build_results:
        build_args = [
            "--config",
            args.config,
            "--checkpoint",
            str(checkpoint_path),
            "--json",
        ]
        if args.output_dir:
            build_args.extend(["--output-dir", args.output_dir])
        build_results_payload = _capture_json_command(
            build_results,
            build_args,
            "medfusion build-results",
        )

    payload = _build_payload(
        config_path=args.config,
        run_layout=run_layout,
        build_results_skipped=args.skip_build_results,
        build_results_payload=build_results_payload,
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    _print_human_summary(payload)
