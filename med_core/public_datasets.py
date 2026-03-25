"""Public dataset quick-validation helpers and CLI entrypoints."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

DEFAULT_PATHMNIST_DIR = Path("data/public/medmnist/pathmnist-demo")
DEFAULT_BREASTMNIST_DIR = Path("data/public/medmnist/breastmnist-demo")
DEFAULT_HEART_DIR = Path("data/public/uci/heart-disease-demo")
UCI_HEART_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

PUBLIC_DATASET_SPECS: dict[str, dict[str, Any]] = {
    "medmnist-pathmnist": {
        "id": "medmnist-pathmnist",
        "label": "PathMNIST",
        "description": "最快验证图像训练、结果页和 artifact 输出。",
        "modality": "image",
        "task": "classification",
        "default_output_dir": DEFAULT_PATHMNIST_DIR,
        "config_path": "configs/public_datasets/pathmnist_quickstart.yaml",
        "notes": [
            "依赖 medmnist 包",
            "走 dummy tabular fallback 进入统一多模态训练主链",
        ],
    },
    "medmnist-breastmnist": {
        "id": "medmnist-breastmnist",
        "label": "BreastMNIST",
        "description": "最小二分类图像 quick validation，适合快速录屏和演示。",
        "modality": "image",
        "task": "binary-classification",
        "default_output_dir": DEFAULT_BREASTMNIST_DIR,
        "config_path": "configs/public_datasets/breastmnist_quickstart.yaml",
        "notes": [
            "依赖 medmnist 包",
            "走 dummy tabular fallback 进入统一多模态训练主链",
        ],
    },
    "uci-heart-disease": {
        "id": "uci-heart-disease",
        "label": "UCI Heart Disease",
        "description": "最快验证 tabular 指标链路和二分类结果展示。",
        "modality": "tabular",
        "task": "binary-classification",
        "default_output_dir": DEFAULT_HEART_DIR,
        "config_path": "configs/public_datasets/uci_heart_disease_quickstart.yaml",
        "notes": [
            "保留真实表格特征",
            "自动生成 placeholder 图像以进入统一多模态训练主链",
        ],
    },
}


def list_public_datasets() -> list[dict[str, Any]]:
    """Return a JSON-serializable public dataset registry."""
    datasets: list[dict[str, Any]] = []
    for dataset_id in sorted(PUBLIC_DATASET_SPECS):
        spec = PUBLIC_DATASET_SPECS[dataset_id]
        datasets.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "description": spec["description"],
                "modality": spec["modality"],
                "task": spec["task"],
                "default_output_dir": str(spec["default_output_dir"]),
                "config_path": spec["config_path"],
                "notes": list(spec.get("notes", [])),
            }
        )
    return datasets


def get_public_dataset(dataset_id: str) -> dict[str, Any]:
    """Return one public dataset spec or raise a ValueError."""
    try:
        return next(item for item in list_public_datasets() if item["id"] == dataset_id)
    except StopIteration as exc:
        available = ", ".join(sorted(PUBLIC_DATASET_SPECS))
        raise ValueError(
            f"Unsupported public dataset: {dataset_id}. Available: {available}"
        ) from exc


def _reset_output_dir(output_dir: Path, overwrite: bool) -> None:
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _summarize_labels(labels: list[int]) -> dict[str, int]:
    return {str(label): count for label, count in sorted(Counter(labels).items())}


def _prepare_medmnist_dataset(
    dataset_class_name: str,
    dataset_label: str,
    output_dir: Path,
    train_limit: int,
    val_limit: int,
    test_limit: int,
    overwrite: bool,
) -> None:
    try:
        import medmnist
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: medmnist. "
            f"Run `uv pip install medmnist` before preparing {dataset_label}."
        ) from exc

    dataset_class = getattr(medmnist, dataset_class_name)
    patient_prefix = dataset_label.upper().replace("-", "_")

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
        dataset = dataset_class(split=split, root=str(download_root), download=True)
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
                    "patient_id": f"{patient_prefix}_{split.upper()}_{index:05d}",
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
            "dataset": dataset_label,
            "output_dir": str(output_dir),
            "num_samples": len(records),
            "split_limits": split_limits,
            "class_distribution": _summarize_labels(all_labels),
            "metadata_path": str(metadata_path),
        },
    )


def _prepare_pathmnist(
    output_dir: Path,
    train_limit: int,
    val_limit: int,
    test_limit: int,
    overwrite: bool,
) -> None:
    _prepare_medmnist_dataset(
        dataset_class_name="PathMNIST",
        dataset_label="PathMNIST",
        output_dir=output_dir,
        train_limit=train_limit,
        val_limit=val_limit,
        test_limit=test_limit,
        overwrite=overwrite,
    )


def _prepare_breastmnist(
    output_dir: Path,
    train_limit: int,
    val_limit: int,
    test_limit: int,
    overwrite: bool,
) -> None:
    _prepare_medmnist_dataset(
        dataset_class_name="BreastMNIST",
        dataset_label="BreastMNIST",
        output_dir=output_dir,
        train_limit=train_limit,
        val_limit=val_limit,
        test_limit=test_limit,
        overwrite=overwrite,
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


def prepare_public_dataset(
    dataset_id: str,
    *,
    output_dir: Path | None = None,
    overwrite: bool = False,
    train_limit: int = 270,
    val_limit: int = 45,
    test_limit: int = 45,
) -> dict[str, Any]:
    """Prepare one public dataset and return a summary payload."""
    dataset = get_public_dataset(dataset_id)
    resolved_output_dir = Path(output_dir) if output_dir else Path(dataset["default_output_dir"])

    if dataset_id == "medmnist-pathmnist":
        _prepare_pathmnist(
            output_dir=resolved_output_dir,
            train_limit=train_limit,
            val_limit=val_limit,
            test_limit=test_limit,
            overwrite=overwrite,
        )
    elif dataset_id == "medmnist-breastmnist":
        _prepare_breastmnist(
            output_dir=resolved_output_dir,
            train_limit=train_limit,
            val_limit=val_limit,
            test_limit=test_limit,
            overwrite=overwrite,
        )
    elif dataset_id == "uci-heart-disease":
        _prepare_uci_heart_disease(
            output_dir=resolved_output_dir,
            overwrite=overwrite,
        )
    else:
        raise ValueError(f"Unsupported public dataset: {dataset_id}")

    summary_path = resolved_output_dir / "summary.json"
    summary_payload = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    return {
        "dataset": dataset,
        "output_dir": str(resolved_output_dir),
        "summary_path": str(summary_path),
        "summary": summary_payload,
        "recommended_commands": _recommended_commands(dataset, resolved_output_dir),
    }


def _recommended_commands(dataset: dict[str, Any], output_dir: Path | None = None) -> dict[str, str]:
    prepare_command = f"medfusion public-datasets prepare {dataset['id']} --overwrite"
    if output_dir is not None and output_dir != Path(dataset["default_output_dir"]):
        prepare_command += f" --output-dir {output_dir}"

    return {
        "prepare": prepare_command,
        "train": f"medfusion train --config {dataset['config_path']}",
        "build_results": (
            f"medfusion build-results --config {dataset['config_path']} "
            "--checkpoint <path>"
        ),
    }


def _print_json(payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _print_list(datasets: list[dict[str, Any]]) -> None:
    print("MedFusion Public Dataset Quick Validation")
    print("")
    for item in datasets:
        print(f"- {item['id']} ({item['label']})")
        print(f"  {item['description']}")
        print(f"  config: {item['config_path']}")
        print(f"  output: {item['default_output_dir']}")
        print("")


def _print_show(dataset: dict[str, Any]) -> None:
    commands = _recommended_commands(dataset)
    print(f"{dataset['label']} ({dataset['id']})")
    print("")
    print(dataset["description"])
    print(f"modality: {dataset['modality']}")
    print(f"task: {dataset['task']}")
    print(f"config: {dataset['config_path']}")
    print(f"default output: {dataset['default_output_dir']}")
    print("")
    print("notes:")
    for note in dataset.get("notes", []):
        print(f"- {note}")
    print("")
    print("recommended commands:")
    for name, command in commands.items():
        print(f"- {name}: {command}")


def build_parser(prog: str = "medfusion public-datasets") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Manage public datasets for MedFusion quick validation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List supported public datasets.")
    list_parser.add_argument("--json", action="store_true", help="Print as JSON")

    show_parser = subparsers.add_parser("show", help="Show one public dataset profile.")
    show_parser.add_argument("dataset", choices=sorted(PUBLIC_DATASET_SPECS))
    show_parser.add_argument("--json", action="store_true", help="Print as JSON")

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Download/adapt one public dataset for quick validation.",
    )
    prepare_parser.add_argument("dataset", choices=sorted(PUBLIC_DATASET_SPECS))
    prepare_parser.add_argument("--output-dir", type=Path, help="Override output directory")
    prepare_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before exporting.",
    )
    prepare_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the resolved plan and recommended commands.",
    )
    prepare_parser.add_argument(
        "--train-limit",
        type=int,
        default=270,
        help="PathMNIST only: train split export size.",
    )
    prepare_parser.add_argument(
        "--val-limit",
        type=int,
        default=45,
        help="PathMNIST only: val split export size.",
    )
    prepare_parser.add_argument(
        "--test-limit",
        type=int,
        default=45,
        help="PathMNIST only: test split export size.",
    )
    prepare_parser.add_argument("--json", action="store_true", help="Print as JSON")

    return parser


def main(
    argv: Sequence[str] | None = None,
    prog: str = "medfusion public-datasets",
) -> None:
    """CLI entry point for public dataset quick-validation assets."""
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    if args.command == "list":
        datasets = list_public_datasets()
        if args.json:
            _print_json(datasets)
            return
        _print_list(datasets)
        return

    if args.command == "show":
        dataset = get_public_dataset(args.dataset)
        if args.json:
            payload = dict(dataset)
            payload["recommended_commands"] = _recommended_commands(dataset)
            _print_json(payload)
            return
        _print_show(dataset)
        return

    if args.command == "prepare":
        dataset = get_public_dataset(args.dataset)
        resolved_output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else Path(dataset["default_output_dir"])
        )
        payload = {
            "dataset": dataset,
            "output_dir": str(resolved_output_dir),
            "overwrite": bool(args.overwrite),
            "dry_run": bool(args.dry_run),
            "recommended_commands": _recommended_commands(dataset, resolved_output_dir),
        }
        if args.dataset in {"medmnist-pathmnist", "medmnist-breastmnist"}:
            payload["split_limits"] = {
                "train": int(args.train_limit),
                "val": int(args.val_limit),
                "test": int(args.test_limit),
            }
        if args.dry_run:
            if args.json:
                _print_json(payload)
                return
            print("Dry run only. Nothing downloaded yet.")
            print("")
            _print_show(dataset)
            print("")
            print(f"resolved output: {resolved_output_dir}")
            if "split_limits" in payload:
                print(f"split limits: {payload['split_limits']}")
            return

        result = prepare_public_dataset(
            args.dataset,
            output_dir=resolved_output_dir,
            overwrite=args.overwrite,
            train_limit=args.train_limit,
            val_limit=args.val_limit,
            test_limit=args.test_limit,
        )
        if args.json:
            _print_json(result)
            return

        print(
            f"Prepared {result['dataset']['label']} quick-validation dataset at: "
            f"{result['output_dir']}"
        )
        print(f"Summary: {result['summary_path']}")
        print("Recommended commands:")
        for name, command in result["recommended_commands"].items():
            print(f"- {name}: {command}")
        return

    parser.error(f"Unsupported command: {args.command}")
