#!/usr/bin/env python3
"""
Prepare public datasets for MedFusion quick validation.

This script adapts a small set of public datasets to the current MedFusion
multimodal CLI pipeline. The current main training path expects both image
and tabular inputs, so:

- `medmnist-pathmnist` keeps image data and relies on the dataset loader's
  dummy tabular fallback.
- `uci-heart-disease` keeps the tabular features and writes one neutral
  placeholder image so the data can flow through the same multimodal path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

DEFAULT_PATHMNIST_DIR = Path("data/public/medmnist/pathmnist-demo")
DEFAULT_HEART_DIR = Path("data/public/uci/heart-disease-demo")
UCI_HEART_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


def _reset_output_dir(output_dir: Path, overwrite: bool) -> None:
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _summarize_labels(labels: list[int]) -> dict[str, int]:
    return {str(label): count for label, count in sorted(Counter(labels).items())}


def _prepare_pathmnist(
    output_dir: Path,
    train_limit: int,
    val_limit: int,
    test_limit: int,
    overwrite: bool,
) -> None:
    try:
        from medmnist import PathMNIST
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: medmnist. "
            "Use `uv run python scripts/prepare_public_dataset.py medmnist-pathmnist` "
            "after installing `medmnist` with `uv pip install medmnist`."
        ) from exc

    _reset_output_dir(output_dir, overwrite)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    download_root = output_dir / "downloads"
    download_root.mkdir(parents=True, exist_ok=True)

    split_limits = {
        "train": train_limit,
        "val": val_limit,
        "test": test_limit,
    }

    records: list[dict[str, Any]] = []
    all_labels: list[int] = []

    for split, limit in split_limits.items():
        dataset = PathMNIST(split=split, root=str(download_root), download=True)
        actual_limit = min(limit, len(dataset))

        for index in range(actual_limit):
            image, label = dataset[index]
            pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)
            pil_image = pil_image.convert("RGB")

            label_value = int(np.asarray(label).reshape(-1)[0])
            filename = f"{split}_{index:05d}.png"
            relative_image_path = Path("images") / filename
            pil_image.save(output_dir / relative_image_path)

            records.append(
                {
                    "patient_id": f"PATHMNIST_{split.upper()}_{index:05d}",
                    "source_split": split,
                    "image_path": relative_image_path.as_posix(),
                    "label": label_value,
                }
            )
            all_labels.append(label_value)

    metadata = pd.DataFrame.from_records(records)
    metadata_path = output_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    _write_json(
        output_dir / "summary.json",
        {
            "dataset": "PathMNIST",
            "output_dir": str(output_dir),
            "num_samples": len(records),
            "split_limits": split_limits,
            "class_distribution": _summarize_labels(all_labels),
            "metadata_path": str(metadata_path),
        },
    )

    print(f"Prepared PathMNIST quick-validation dataset at: {output_dir}")
    print(f"Metadata: {metadata_path}")
    print("Recommended training command:")
    print(
        "  uv run medfusion train "
        "--config configs/public_datasets/pathmnist_quickstart.yaml"
    )


def _build_placeholder_image(path: Path, image_size: int = 96) -> None:
    background = Image.new("RGB", (image_size, image_size), color=(234, 236, 240))
    draw = ImageDraw.Draw(background)

    margin = image_size // 8
    draw.rounded_rectangle(
        [(margin, margin), (image_size - margin, image_size - margin)],
        radius=image_size // 6,
        outline=(176, 183, 194),
        width=2,
        fill=(244, 246, 248),
    )
    draw.line(
        [(margin + 8, image_size // 2), (image_size - margin - 8, image_size // 2)],
        fill=(196, 201, 209),
        width=2,
    )
    draw.line(
        [(image_size // 2, margin + 8), (image_size // 2, image_size - margin - 8)],
        fill=(196, 201, 209),
        width=2,
    )
    background.save(path)


def _prepare_uci_heart_disease(output_dir: Path, overwrite: bool) -> None:
    _reset_output_dir(output_dir, overwrite)
    raw_dir = output_dir / "raw"
    image_dir = output_dir / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "processed.cleveland.data"
    urllib.request.urlretrieve(UCI_HEART_URL, raw_path)

    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num",
    ]
    df = pd.read_csv(raw_path, header=None, names=columns, na_values="?")
    df = df.dropna().reset_index(drop=True)
    df["diagnosis_binary"] = (df["num"] > 0).astype(int)

    placeholder_path = image_dir / "placeholder.png"
    _build_placeholder_image(placeholder_path)

    metadata = df.copy()
    metadata.insert(0, "patient_id", [f"HEART_{index:04d}" for index in range(len(df))])
    metadata.insert(1, "image_path", "images/placeholder.png")
    metadata = metadata.drop(columns=["num"])

    metadata_path = output_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    _write_json(
        output_dir / "summary.json",
        {
            "dataset": "UCI Heart Disease (Cleveland)",
            "output_dir": str(output_dir),
            "num_samples": int(len(metadata)),
            "class_distribution": _summarize_labels(
                metadata["diagnosis_binary"].astype(int).tolist()
            ),
            "metadata_path": str(metadata_path),
            "raw_path": str(raw_path),
        },
    )

    print(f"Prepared UCI Heart Disease quick-validation dataset at: {output_dir}")
    print(f"Metadata: {metadata_path}")
    print("Recommended training command:")
    print(
        "  uv run medfusion train --config "
        "configs/public_datasets/uci_heart_disease_quickstart.yaml"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare public datasets for MedFusion quick validation.",
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    pathmnist_parser = subparsers.add_parser(
        "medmnist-pathmnist",
        help="Download and export a small PathMNIST quick-validation dataset.",
    )
    pathmnist_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PATHMNIST_DIR,
        help=f"Output directory (default: {DEFAULT_PATHMNIST_DIR})",
    )
    pathmnist_parser.add_argument(
        "--train-limit",
        type=int,
        default=270,
        help="Number of training samples to export from the official train split.",
    )
    pathmnist_parser.add_argument(
        "--val-limit",
        type=int,
        default=45,
        help="Number of validation samples to export from the official val split.",
    )
    pathmnist_parser.add_argument(
        "--test-limit",
        type=int,
        default=45,
        help="Number of test samples to export from the official test split.",
    )
    pathmnist_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before exporting.",
    )

    heart_parser = subparsers.add_parser(
        "uci-heart-disease",
        help="Download and export UCI Heart Disease for quick validation.",
    )
    heart_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_HEART_DIR,
        help=f"Output directory (default: {DEFAULT_HEART_DIR})",
    )
    heart_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before exporting.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset == "medmnist-pathmnist":
        _prepare_pathmnist(
            output_dir=args.output_dir,
            train_limit=args.train_limit,
            val_limit=args.val_limit,
            test_limit=args.test_limit,
            overwrite=args.overwrite,
        )
        return

    if args.dataset == "uci-heart-disease":
        _prepare_uci_heart_disease(output_dir=args.output_dir, overwrite=args.overwrite)
        return

    parser.error(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()
