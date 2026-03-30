"""Configuration doctor utilities for training readiness checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from med_core.configs.base_config import ExperimentConfig
from med_core.configs.config_loader import load_config
from med_core.configs.validation import validate_config


@dataclass
class DoctorIssue:
    """Single doctor finding."""

    severity: str
    path: str
    message: str
    suggestion: str | None = None
    code: str | None = None


@dataclass
class ConfigDoctorReport:
    """Aggregated doctor report for a config file."""

    config_path: str
    ok: bool
    errors: list[DoctorIssue] = field(default_factory=list)
    warnings: list[DoctorIssue] = field(default_factory=list)
    info: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a JSON-serializable dictionary."""
        return {
            "config_path": self.config_path,
            "ok": self.ok,
            "errors": [asdict(item) for item in self.errors],
            "warnings": [asdict(item) for item in self.warnings],
            "info": self.info,
            "summary": self.summary,
        }


class ConfigDoctor:
    """Perform practical training-readiness checks on a config file."""

    def __init__(self, image_sample_limit: int = 32):
        self.image_sample_limit = image_sample_limit

    @staticmethod
    def _read_manifest_dataframe(
        csv_path: Path,
        patient_id_column: str | None,
    ) -> pd.DataFrame:
        read_csv_kwargs = {}
        if patient_id_column:
            read_csv_kwargs["dtype"] = {patient_id_column: "string"}
        return pd.read_csv(csv_path, **read_csv_kwargs)

    def analyze(self, config_path: str | Path) -> ConfigDoctorReport:
        config_path = Path(config_path)
        errors: list[DoctorIssue] = []
        warnings: list[DoctorIssue] = []
        info: list[dict[str, Any]] = []
        summary: dict[str, Any] = {
            "estimated_split_counts": {},
            "dataset_rows": 0,
            "valid_image_rows": 0,
            "missing_image_rows": 0,
        }

        if not config_path.exists():
            errors.append(
                DoctorIssue(
                    severity="error",
                    path="config",
                    message=f"配置文件不存在: {config_path}",
                    suggestion="确认 --config 路径是否正确",
                    code="D001",
                )
            )
            return ConfigDoctorReport(
                config_path=str(config_path),
                ok=False,
                errors=errors,
                warnings=warnings,
                info=info,
                summary=summary,
            )

        try:
            config = load_config(config_path)
        except Exception as exc:  # pragma: no cover - exercised via CLI smoke
            errors.append(
                DoctorIssue(
                    severity="error",
                    path="config",
                    message=f"配置解析失败: {exc}",
                    suggestion="检查 YAML 语法和字段名是否匹配当前 schema",
                    code="D002",
                )
            )
            return ConfigDoctorReport(
                config_path=str(config_path),
                ok=False,
                errors=errors,
                warnings=warnings,
                info=info,
                summary=summary,
            )

        self._collect_structural_issues(config, errors)
        self._collect_runtime_warnings(config, warnings)
        self._collect_dataset_checks(config, errors, warnings, info, summary)

        return ConfigDoctorReport(
            config_path=str(config_path),
            ok=not errors,
            errors=errors,
            warnings=warnings,
            info=info,
            summary=summary,
        )

    def _collect_structural_issues(
        self,
        config: ExperimentConfig,
        errors: list[DoctorIssue],
    ) -> None:
        for item in validate_config(config):
            errors.append(
                DoctorIssue(
                    severity="error",
                    path=item.path,
                    message=item.message,
                    suggestion=item.suggestion,
                    code=item.error_code,
                )
            )

    def _collect_runtime_warnings(
        self,
        config: ExperimentConfig,
        warnings: list[DoctorIssue],
    ) -> None:
        if config.device == "cpu" and config.data.pin_memory:
            warnings.append(
                DoctorIssue(
                    severity="warning",
                    path="data.pin_memory",
                    message="当前环境是 CPU，但 pin_memory=true，训练时会出现无效 pin_memory 警告。",
                    suggestion="如果主要在 CPU 上运行，可改成 data.pin_memory=false",
                    code="W001",
                )
            )

        if config.training.monitor == "val_auc" and config.model.num_classes != 2:
            warnings.append(
                DoctorIssue(
                    severity="warning",
                    path="training.monitor",
                    message="当前 monitor=val_auc，但配置是多分类任务，AUC 监控未必是最稳的默认选择。",
                    suggestion="多分类主链可考虑监控 accuracy、macro_f1 或 balanced_accuracy",
                    code="W002",
                )
            )

        if config.data.batch_size <= 4:
            warnings.append(
                DoctorIssue(
                    severity="warning",
                    path="data.batch_size",
                    message=f"batch_size={config.data.batch_size}，更适合 smoke 或小数据调试，不一定适合作为正式基线。",
                    suggestion="如果显存允许，可适当增大 batch_size 提升吞吐稳定性",
                    code="W003",
                )
            )

    def _collect_dataset_checks(
        self,
        config: ExperimentConfig,
        errors: list[DoctorIssue],
        warnings: list[DoctorIssue],
        info: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> None:
        if config.data.dataset_type == "three_phase_ct_tabular":
            self._collect_three_phase_dataset_checks(
                config=config,
                errors=errors,
                warnings=warnings,
                info=info,
                summary=summary,
            )
            return

        csv_path = Path(config.data.csv_path)
        image_dir = Path(config.data.image_dir)

        if not csv_path.exists():
            errors.append(
                DoctorIssue(
                    severity="error",
                    path="data.csv_path",
                    message=f"CSV 不存在: {csv_path}",
                    suggestion="先准备 metadata CSV，或改成仓库中真实存在的路径",
                    code="D101",
                )
            )
            return

        if not image_dir.exists():
            errors.append(
                DoctorIssue(
                    severity="error",
                    path="data.image_dir",
                    message=f"图像目录不存在: {image_dir}",
                    suggestion="确认 image_dir 指向真实图像目录",
                    code="D102",
                )
            )
            return

        try:
            dataframe = self._read_manifest_dataframe(
                csv_path,
                config.data.patient_id_column,
            )
        except Exception as exc:
            errors.append(
                DoctorIssue(
                    severity="error",
                    path="data.csv_path",
                    message=f"CSV 读取失败: {exc}",
                    suggestion="检查编码、分隔符或文件内容是否损坏",
                    code="D103",
                )
            )
            return

        row_count = int(len(dataframe))
        summary["dataset_rows"] = row_count
        info.append({"section": "dataset", "rows": row_count})

        required_columns = [config.data.image_path_column, config.data.target_column]
        optional_columns = (
            list(config.data.numerical_features)
            + list(config.data.categorical_features)
            + ([config.data.survival_time_column] if config.data.survival_time_column else [])
            + ([config.data.survival_event_column] if config.data.survival_event_column else [])
            + ([config.data.patient_id_column] if config.data.patient_id_column else [])
        )
        expected_columns = [column for column in required_columns + optional_columns if column]

        missing_columns = [column for column in expected_columns if column not in dataframe.columns]
        if missing_columns:
            errors.append(
                DoctorIssue(
                    severity="error",
                    path="data.columns",
                    message=f"CSV 缺少配置里引用的列: {', '.join(missing_columns)}",
                    suggestion="同步修正 CSV 列名和 config 中的 image/target/feature 配置",
                    code="D104",
                )
            )
            return

        if row_count == 0:
            errors.append(
                DoctorIssue(
                    severity="error",
                    path="data.csv_path",
                    message="CSV 没有任何样本行。",
                    suggestion="至少准备一条有效样本再启动训练",
                    code="D105",
                )
            )
            return

        target_distribution = (
            dataframe[config.data.target_column].value_counts(dropna=False).to_dict()
        )
        info.append({"section": "label_distribution", "values": target_distribution})

        unique_labels = int(dataframe[config.data.target_column].nunique(dropna=True))
        if unique_labels < 2:
            errors.append(
                DoctorIssue(
                    severity="error",
                    path="data.target_column",
                    message=f"目标列只有 {unique_labels} 个有效类别，无法进行分类训练。",
                    suggestion="确认 target 列中至少存在两个类别",
                    code="D106",
                )
            )

        if config.model.num_classes != unique_labels:
            warnings.append(
                DoctorIssue(
                    severity="warning",
                    path="model.num_classes",
                    message=f"model.num_classes={config.model.num_classes}，但 CSV 中检测到 {unique_labels} 个类别。",
                    suggestion="如果是单标签分类，建议把 model.num_classes 与真实类别数对齐",
                    code="W101",
                )
            )

        image_values = dataframe[config.data.image_path_column].astype(str)
        missing_images = 0
        checked_images = 0
        for image_name in image_values.head(self.image_sample_limit):
            image_path = Path(image_name)
            if not image_path.is_absolute():
                image_path = image_dir / image_name
            checked_images += 1
            if not image_path.exists():
                missing_images += 1

        summary["missing_image_rows"] = missing_images
        summary["valid_image_rows"] = max(checked_images - missing_images, 0)
        info.append(
            {
                "section": "image_probe",
                "checked": checked_images,
                "missing": missing_images,
                "sample_limit": self.image_sample_limit,
            }
        )

        if missing_images == checked_images and checked_images > 0:
            errors.append(
                DoctorIssue(
                    severity="error",
                    path="data.image_path_column",
                    message="抽样检查到的图像文件全部缺失。",
                    suggestion="检查 image_dir、image_path_column 或文件名相对路径是否正确",
                    code="D107",
                )
            )
        elif missing_images > 0:
            warnings.append(
                DoctorIssue(
                    severity="warning",
                    path="data.image_path_column",
                    message=f"抽样检查 {checked_images} 条图像路径，其中 {missing_images} 条缺失。",
                    suggestion="训练前建议先清洗 metadata，避免运行时样本被大量丢弃",
                    code="W102",
                )
            )

        train_count = int(round(row_count * config.data.train_ratio))
        val_count = int(round(row_count * config.data.val_ratio))
        test_count = max(row_count - train_count - val_count, 0)
        summary["estimated_split_counts"] = {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        }

        if val_count < max(2, config.model.num_classes):
            warnings.append(
                DoctorIssue(
                    severity="warning",
                    path="data.val_ratio",
                    message=f"按当前比例估算，验证集只有 {val_count} 条样本，指标波动会比较大。",
                    suggestion="演示可接受；如果要做稳定验证，建议提高数据量或增大 val_ratio",
                    code="W103",
                )
            )

        if config.data.batch_size > row_count:
            warnings.append(
                DoctorIssue(
                    severity="warning",
                    path="data.batch_size",
                    message=f"batch_size={config.data.batch_size} 已超过数据总量 {row_count}。",
                    suggestion="通常把 batch_size 调整到不超过训练集规模更合理",
                    code="W104",
                )
            )

    def _collect_three_phase_dataset_checks(
        self,
        config: ExperimentConfig,
        errors: list[DoctorIssue],
        warnings: list[DoctorIssue],
        info: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> None:
        csv_path = Path(config.data.csv_path)
        if not csv_path.exists():
            errors.append(
                DoctorIssue(
                    severity="error",
                    path="data.csv_path",
                    message=f"CSV 不存在: {csv_path}",
                    suggestion="先准备三相 CT manifest CSV，或改成真实存在的路径",
                    code="D201",
                )
            )
            return

        try:
            dataframe = self._read_manifest_dataframe(
                csv_path,
                config.data.patient_id_column,
            )
        except Exception as exc:
            errors.append(
                DoctorIssue(
                    severity="error",
                    path="data.csv_path",
                    message=f"CSV 读取失败: {exc}",
                    suggestion="检查编码、分隔符或文件内容是否损坏",
                    code="D202",
                )
            )
            return

        row_count = int(len(dataframe))
        summary["dataset_rows"] = row_count
        info.append({"section": "dataset", "rows": row_count})

        phase_columns = list(config.data.phase_dir_columns.values())
        required_columns = (
            [config.data.target_column]
            + ([config.data.patient_id_column] if config.data.patient_id_column else [])
            + phase_columns
            + list(config.data.clinical_feature_columns)
        )
        missing_columns = [
            column for column in required_columns if column and column not in dataframe.columns
        ]
        if missing_columns:
            errors.append(
                DoctorIssue(
                    severity="error",
                    path="data.columns",
                    message=f"CSV 缺少三相 CT 配置里引用的列: {', '.join(missing_columns)}",
                    suggestion="同步修正 manifest 列名和 config 中的 phase/clinical/target 配置",
                    code="D203",
                )
            )
            return

        if row_count == 0:
            errors.append(
                DoctorIssue(
                    severity="error",
                    path="data.csv_path",
                    message="CSV 没有任何样本行。",
                    suggestion="至少准备两条带标签的病例再启动训练",
                    code="D204",
                )
            )
            return

        unique_labels = int(dataframe[config.data.target_column].nunique(dropna=True))
        if unique_labels < 2:
            errors.append(
                DoctorIssue(
                    severity="error",
                    path="data.target_column",
                    message=f"目标列只有 {unique_labels} 个有效类别，无法进行分类训练。",
                    suggestion="确认 target 列中至少存在两个类别",
                    code="D205",
                )
            )

        checked_cases = 0
        missing_phase_paths = 0
        sample_rows = dataframe.head(self.image_sample_limit).to_dict(orient="records")
        for row in sample_rows:
            checked_cases += 1
            for phase_name, column_name in config.data.phase_dir_columns.items():
                phase_dir = Path(str(row[column_name]))
                if not phase_dir.exists():
                    missing_phase_paths += 1
                    warnings.append(
                        DoctorIssue(
                            severity="warning",
                            path=f"data.phase_dir_columns.{phase_name}",
                            message=(
                                f"病例 {row.get(config.data.patient_id_column or 'case_id', checked_cases)} "
                                f"的 {phase_name} 目录缺失: {phase_dir}"
                            ),
                            suggestion="确认 manifest 中的三相目录路径是否正确",
                            code="W201",
                        )
                    )

        summary["valid_image_rows"] = max(
            checked_cases - min(missing_phase_paths, checked_cases),
            0,
        )
        summary["missing_image_rows"] = missing_phase_paths
        info.append(
            {
                "section": "three_phase_probe",
                "checked_cases": checked_cases,
                "missing_phase_paths": missing_phase_paths,
            }
        )

        if checked_cases > 0 and missing_phase_paths >= checked_cases * max(
            len(config.data.phase_dir_columns),
            1,
        ):
            errors.append(
                DoctorIssue(
                    severity="error",
                    path="data.phase_dir_columns",
                    message="抽样检查到的三相目录全部缺失。",
                    suggestion="检查 manifest 中的三相目录列和真实 DICOM 路径是否一致",
                    code="D206",
                )
            )


def analyze_config(config_path: str | Path, image_sample_limit: int = 32) -> ConfigDoctorReport:
    """Convenience wrapper for config analysis."""
    return ConfigDoctor(image_sample_limit=image_sample_limit).analyze(config_path)
