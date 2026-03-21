"""CLI entry point for importing real training runs into the Web model library."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence


def import_run(
    argv: Sequence[str] | None = None,
    prog: str = "medfusion import-run",
) -> None:
    """Build artifacts from a real checkpoint and register the run in ModelInfo."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Generate real artifacts from a trained checkpoint and import the run into the Web model library.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", help="Override artifact output directory")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split used for post-training validation",
    )
    parser.add_argument(
        "--attention-samples",
        type=int,
        default=4,
        help="How many attention / Grad-CAM samples to export",
    )
    parser.add_argument("--name", help="Override model name in the library")
    parser.add_argument("--description", help="Override model description in the library")
    parser.add_argument(
        "--tag",
        dest="tags",
        action="append",
        default=[],
        help="Attach a tag to the imported model; can be used multiple times",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the imported model summary as JSON",
    )
    args = parser.parse_args(argv)

    from med_core.web.database import SessionLocal, init_db
    from med_core.web.model_registry import import_model_run as import_model_run_into_db

    init_db()
    db = SessionLocal()
    try:
        model = import_model_run_into_db(
            db=db,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            split=args.split,
            attention_samples=max(args.attention_samples, 0),
            name=args.name,
            description=args.description,
            tags=args.tags,
            import_source="cli",
        )
    finally:
        db.close()

    payload = {
        "model_id": model.id,
        "name": model.name,
        "checkpoint_path": model.checkpoint_path,
        "config_path": model.config_path,
        "dataset_name": model.dataset_name,
        "accuracy": model.accuracy,
        "loss": model.loss,
        "artifact_paths": (model.config or {}).get("artifact_paths", {}),
        "tags": model.tags,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("MedFusion Run Importer")
    print("")
    print(f"Model ID: {model.id}")
    print(f"Name: {model.name}")
    print(f"Dataset: {model.dataset_name or '-'}")
    print(f"Accuracy: {model.accuracy}")
    print(f"Loss: {model.loss}")
    print(f"Artifacts: {len((model.config or {}).get('artifact_paths', {}))}")
    print(f"Checkpoint: {model.checkpoint_path}")
