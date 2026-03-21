"""CLI entry point for building post-training result artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from med_core.postprocessing import build_results_artifacts


def build_results(
    argv: Sequence[str] | None = None,
    prog: str = "medfusion build-results",
) -> None:
    """Build validation/visualization artifacts from a real checkpoint."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run validation on a trained checkpoint and generate result artifacts for the Web result panel.",
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the generated artifact payload as JSON",
    )
    args = parser.parse_args(argv)

    result = build_results_artifacts(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        attention_samples=max(args.attention_samples, 0),
    )
    payload = {
        "output_dir": result.output_dir,
        "artifact_paths": result.artifact_paths,
        "metrics": result.metrics,
        "validation": result.validation,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("MedFusion Result Builder")
    print("")
    print(f"Output: {result.output_dir}")
    print(f"Artifacts: {len(result.artifact_paths)}")
    print(
        "Metrics: "
        f"accuracy={result.metrics.get('accuracy')}, "
        f"auc={result.metrics.get('auc')}, "
        f"macro_f1={result.metrics.get('macro_f1')}"
    )
    print(
        "Validation: "
        f"samples={result.validation.get('overview', {}).get('sample_count')}, "
        f"split={result.validation.get('overview', {}).get('split')}"
    )
    for key, value in sorted(result.artifact_paths.items()):
        print(f"- {key}: {value}")
