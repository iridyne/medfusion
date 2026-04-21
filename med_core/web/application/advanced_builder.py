"""Advanced-builder application services.

This module turns a constrained GraphSpec into a formal-release RunSpec draft.
The graph itself is not an execution source of truth; it is compiled into a
deterministic configuration payload that can be inspected, edited, and later
executed by the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .model_catalog import export_advanced_builder_contract, export_model_catalog
from med_core.configs.base_config import (
    DataConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from med_core.configs.validation import ValidationError, validate_config
from med_core.output_layout import RunOutputLayout, format_oss_display_path

AdvancedBuilderFamily = Literal[
    "data_input",
    "vision_backbone",
    "tabular_encoder",
    "fusion",
    "head",
    "training_strategy",
]
AdvancedBuilderStatus = Literal["compile_ready", "conditional", "draft_only"]


@dataclass(frozen=True)
class AdvancedBuilderComponent:
    id: str
    family: AdvancedBuilderFamily
    label: str
    status: AdvancedBuilderStatus
    description: str
    schema_path: str | None = None


@dataclass(frozen=True)
class AdvancedBuilderConnectionRule:
    from_family: AdvancedBuilderFamily
    to_family: AdvancedBuilderFamily
    status: Literal["required", "conditional", "blocked"]
    description: str


@dataclass(frozen=True)
class AdvancedBuilderBlueprint:
    id: str
    label: str
    status: Literal["compile_ready", "draft_only"]
    description: str
    components: tuple[str, ...]
    compiles_to: str | None = None
    blockers: tuple[str, ...] = ()


def _display_path(path: str | Path) -> str:
    candidate = Path(format_oss_display_path(path))
    return candidate.as_posix()


def _slugify(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "-")
        .replace("_", "-")
        .replace("/", "-")
    )


def _issue(
    *,
    level: Literal["error", "warning"],
    code: str,
    message: str,
    path: str | None = None,
    context: dict[str, Any] | None = None,
    suggestion: str | None = None,
) -> dict[str, Any]:
    issue: dict[str, Any] = {
        "level": level,
        "code": code,
        "message": message,
    }
    if path:
        issue["path"] = path
    if context:
        issue["context"] = context
    if suggestion:
        issue["suggestion"] = suggestion
    return issue


def _create_run_spec_preset(preset: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "projectName": "medfusion-oss",
        "experimentName": "quickstart-run",
        "description": "",
        "tags": ["oss", "multimodal"],
        "seed": 42,
        "device": "auto",
        "data": {
            "csvPath": "data/mock/metadata.csv",
            "imageDir": "data/mock",
            "imagePathColumn": "image_path",
            "targetColumn": "diagnosis",
            "patientIdColumn": "",
            "numericalFeatures": ["age"],
            "categoricalFeatures": ["gender"],
            "trainRatio": 0.7,
            "valRatio": 0.15,
            "testRatio": 0.15,
            "imageSize": 224,
            "batchSize": 4,
            "numWorkers": 0,
            "pinMemory": False,
            "augmentationStrength": "light",
        },
        "model": {
            "numClasses": 2,
            "useAuxiliaryHeads": True,
            "vision": {
                "backbone": "resnet18",
                "pretrained": True,
                "freezeBackbone": False,
                "featureDim": 128,
                "dropout": 0.3,
                "attentionType": "cbam",
            },
            "tabular": {
                "hiddenDims": [32],
                "outputDim": 16,
                "dropout": 0.2,
            },
            "fusion": {
                "fusionType": "concatenate",
                "hiddenDim": 144,
                "dropout": 0.3,
                "numHeads": 4,
            },
        },
        "training": {
            "numEpochs": 3,
            "mixedPrecision": False,
            "gradientClip": 1,
            "useProgressiveTraining": False,
            "stage1Epochs": 1,
            "stage2Epochs": 1,
            "stage3Epochs": 1,
            "useAttentionSupervision": False,
            "attentionLossWeight": 0.1,
            "optimizer": {
                "optimizer": "adam",
                "learningRate": 0.001,
                "weightDecay": 0,
                "momentum": 0.9,
            },
            "scheduler": {
                "scheduler": "step",
                "warmupEpochs": 0,
                "minLr": 1e-6,
                "stepSize": 1,
                "gamma": 0.1,
                "patience": 5,
                "factor": 0.5,
            },
        },
        "logging": {
            "outputDir": "outputs/medfusion-mvp/quickstart-run",
            "useTensorboard": False,
            "useWandb": False,
        },
    }

    if preset == "showcase":
        base["projectName"] = "medfusion-results"
        base["experimentName"] = "attention-audit"
        base["description"] = "用于结果审查和可视化产物补齐的多模态实验。"
        base["tags"] = ["results", "attention", "audit"]
        base["model"]["vision"]["backbone"] = "resnet50"
        base["model"]["vision"]["featureDim"] = 256
        base["model"]["fusion"]["fusionType"] = "attention"
        base["model"]["fusion"]["hiddenDim"] = 192
        base["training"]["numEpochs"] = 8
        base["training"]["mixedPrecision"] = True
        base["training"]["scheduler"]["scheduler"] = "cosine"
        base["training"]["optimizer"]["optimizer"] = "adamw"
        base["training"]["optimizer"]["learningRate"] = 3e-4
        base["training"]["optimizer"]["weightDecay"] = 1e-2
        base["data"]["batchSize"] = 8
        base["data"]["numWorkers"] = 2
        base["data"]["pinMemory"] = True
        base["data"]["augmentationStrength"] = "medium"
        base["logging"]["useTensorboard"] = True

    if preset == "clinical":
        base["projectName"] = "medfusion-clinical"
        base["experimentName"] = "baseline-clinical"
        base["description"] = "偏稳健的临床基线配置，适合在真实数据集上继续迭代。"
        base["tags"] = ["baseline", "clinical", "validation"]
        base["model"]["vision"]["backbone"] = "efficientnet_b0"
        base["model"]["vision"]["featureDim"] = 192
        base["model"]["fusion"]["fusionType"] = "gated"
        base["model"]["fusion"]["hiddenDim"] = 160
        base["training"]["numEpochs"] = 18
        base["training"]["mixedPrecision"] = True
        base["training"]["useProgressiveTraining"] = True
        base["training"]["stage1Epochs"] = 6
        base["training"]["stage2Epochs"] = 8
        base["training"]["stage3Epochs"] = 4
        base["training"]["scheduler"]["scheduler"] = "plateau"
        base["training"]["scheduler"]["patience"] = 4
        base["training"]["scheduler"]["factor"] = 0.5
        base["training"]["optimizer"]["optimizer"] = "adamw"
        base["training"]["optimizer"]["learningRate"] = 2e-4
        base["training"]["optimizer"]["weightDecay"] = 1e-2
        base["data"]["batchSize"] = 12
        base["data"]["numWorkers"] = 4
        base["data"]["pinMemory"] = True
        base["data"]["augmentationStrength"] = "medium"
        base["logging"]["useTensorboard"] = True

    base["logging"]["outputDir"] = (
        f"outputs/{_slugify(base['projectName'])}/{_slugify(base['experimentName'])}"
    )
    return base


def _to_experiment_config_payload(run_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_name": run_spec["projectName"],
        "experiment_name": run_spec["experimentName"],
        "description": run_spec["description"],
        "tags": run_spec["tags"],
        "seed": run_spec["seed"],
        "device": run_spec["device"],
        "data": {
            "dataset_type": "image_tabular",
            "csv_path": run_spec["data"]["csvPath"],
            "image_dir": run_spec["data"]["imageDir"],
            "image_path_column": run_spec["data"]["imagePathColumn"],
            "target_column": run_spec["data"]["targetColumn"],
            "patient_id_column": run_spec["data"]["patientIdColumn"] or None,
            "numerical_features": run_spec["data"]["numericalFeatures"],
            "categorical_features": run_spec["data"]["categoricalFeatures"],
            "train_ratio": run_spec["data"]["trainRatio"],
            "val_ratio": run_spec["data"]["valRatio"],
            "test_ratio": run_spec["data"]["testRatio"],
            "image_size": run_spec["data"]["imageSize"],
            "batch_size": run_spec["data"]["batchSize"],
            "num_workers": run_spec["data"]["numWorkers"],
            "pin_memory": run_spec["data"]["pinMemory"],
            "augmentation_strength": run_spec["data"]["augmentationStrength"],
        },
        "model": {
            "model_type": "multimodal_fusion",
            "num_classes": run_spec["model"]["numClasses"],
            "use_auxiliary_heads": run_spec["model"]["useAuxiliaryHeads"],
            "vision": {
                "backbone": run_spec["model"]["vision"]["backbone"],
                "pretrained": run_spec["model"]["vision"]["pretrained"],
                "freeze_backbone": run_spec["model"]["vision"]["freezeBackbone"],
                "feature_dim": run_spec["model"]["vision"]["featureDim"],
                "dropout": run_spec["model"]["vision"]["dropout"],
                "attention_type": run_spec["model"]["vision"]["attentionType"],
                "enable_attention_supervision": run_spec["training"][
                    "useAttentionSupervision"
                ],
            },
            "tabular": {
                "hidden_dims": run_spec["model"]["tabular"]["hiddenDims"],
                "output_dim": run_spec["model"]["tabular"]["outputDim"],
                "dropout": run_spec["model"]["tabular"]["dropout"],
            },
            "fusion": {
                "fusion_type": run_spec["model"]["fusion"]["fusionType"],
                "hidden_dim": run_spec["model"]["fusion"]["hiddenDim"],
                "dropout": run_spec["model"]["fusion"]["dropout"],
                "num_heads": run_spec["model"]["fusion"]["numHeads"],
            },
        },
        "training": {
            "num_epochs": run_spec["training"]["numEpochs"],
            "gradient_clip": run_spec["training"]["gradientClip"],
            "mixed_precision": run_spec["training"]["mixedPrecision"],
            "use_progressive_training": run_spec["training"][
                "useProgressiveTraining"
            ],
            "stage1_epochs": run_spec["training"]["stage1Epochs"],
            "stage2_epochs": run_spec["training"]["stage2Epochs"],
            "stage3_epochs": run_spec["training"]["stage3Epochs"],
            "use_attention_supervision": run_spec["training"][
                "useAttentionSupervision"
            ],
            "attention_loss_weight": run_spec["training"]["attentionLossWeight"],
            "optimizer": {
                "optimizer": run_spec["training"]["optimizer"]["optimizer"],
                "learning_rate": run_spec["training"]["optimizer"]["learningRate"],
                "weight_decay": run_spec["training"]["optimizer"]["weightDecay"],
                "momentum": run_spec["training"]["optimizer"]["momentum"],
            },
            "scheduler": {
                "scheduler": run_spec["training"]["scheduler"]["scheduler"],
                "warmup_epochs": run_spec["training"]["scheduler"]["warmupEpochs"],
                "min_lr": run_spec["training"]["scheduler"]["minLr"],
                "step_size": run_spec["training"]["scheduler"]["stepSize"],
                "gamma": run_spec["training"]["scheduler"]["gamma"],
                "patience": run_spec["training"]["scheduler"]["patience"],
                "factor": run_spec["training"]["scheduler"]["factor"],
            },
        },
        "logging": {
            "output_dir": run_spec["logging"]["outputDir"],
            "use_tensorboard": run_spec["logging"]["useTensorboard"],
            "use_wandb": run_spec["logging"]["useWandb"],
        },
    }


def _instantiate_experiment_config(run_spec: dict[str, Any]) -> ExperimentConfig:
    payload = _to_experiment_config_payload(run_spec)
    training_payload = payload["training"]

    return ExperimentConfig(
        project_name=payload["project_name"],
        experiment_name=payload["experiment_name"],
        description=payload["description"],
        tags=payload["tags"],
        seed=payload["seed"],
        device=payload["device"],
        data=DataConfig(**payload["data"]),
        model=ModelConfig(**payload["model"]),
        training=TrainingConfig(
            num_epochs=training_payload["num_epochs"],
            gradient_clip=training_payload["gradient_clip"],
            mixed_precision=training_payload["mixed_precision"],
            use_progressive_training=training_payload["use_progressive_training"],
            stage1_epochs=training_payload["stage1_epochs"],
            stage2_epochs=training_payload["stage2_epochs"],
            stage3_epochs=training_payload["stage3_epochs"],
            use_attention_supervision=training_payload["use_attention_supervision"],
            attention_loss_weight=training_payload["attention_loss_weight"],
            optimizer=OptimizerConfig(**training_payload["optimizer"]),
            scheduler=SchedulerConfig(**training_payload["scheduler"]),
        ),
        logging=LoggingConfig(**payload["logging"]),
    )


def _serialize_validation_errors(
    errors: list[ValidationError],
) -> list[dict[str, str | None]]:
    return [
        {
            "path": error.path,
            "message": error.message,
            "error_code": error.error_code,
            "suggestion": error.suggestion,
        }
        for error in errors
    ]


def _collect_contract_warnings(config: ExperimentConfig) -> list[dict[str, str | None]]:
    warnings: list[dict[str, str | None]] = []

    if config.data.batch_size <= 4:
        warnings.append(
            {
                "path": "data.batch_size",
                "message": (
                    f"batch_size={config.data.batch_size}，更适合 smoke 或小数据调试，"
                    "不一定适合作为正式基线。"
                ),
                "error_code": "W003",
                "suggestion": "如果显存允许，可适当增大 batch_size 提升吞吐稳定性",
            }
        )

    if not config.data.resolved_csv_path.exists():
        warnings.append(
            {
                "path": "data.csv_path",
                "message": f"CSV 路径当前不存在：{_display_path(config.data.resolved_csv_path)}",
                "error_code": "W101",
                "suggestion": "在真正执行训练前，确认 CSV 已存在并可被当前运行环境读取",
            }
        )

    if not config.data.resolved_image_dir.exists():
        warnings.append(
            {
                "path": "data.image_dir",
                "message": f"图像目录当前不存在：{_display_path(config.data.resolved_image_dir)}",
                "error_code": "W102",
                "suggestion": "在真正执行训练前，确认图像目录已存在并且路径正确",
            }
        )

    return warnings


def _build_mainline_contract(config: ExperimentConfig) -> dict[str, Any]:
    layout = RunOutputLayout(config.logging.output_dir)
    checkpoint_path = layout.checkpoints_dir / "best.pth"
    config_path_str = "./advanced-builder-graph.yaml"
    display_output_dir = _display_path(layout.root_dir)
    display_checkpoint_path = _display_path(checkpoint_path)

    return {
        "schema_family": "experiment",
        "config_path": config_path_str,
        "dataset_type": config.data.dataset_type,
        "output_dir": display_output_dir,
        "model": {
            "model_type": config.model.model_type,
            "vision_backbone": config.model.vision.backbone,
            "fusion_type": config.model.fusion.fusion_type,
            "num_classes": config.model.num_classes,
        },
        "artifacts": {
            "checkpoint": display_checkpoint_path,
            "metrics": _display_path(layout.metrics_path),
            "validation": _display_path(layout.validation_path),
            "summary": _display_path(layout.summary_path),
            "report": _display_path(layout.report_path),
        },
        "recommended_commands": {
            "validate": f"medfusion validate-config --config {config_path_str}",
            "train": f"medfusion train --config {config_path_str}",
            "build_results": (
                "medfusion build-results "
                f"--config {config_path_str} --checkpoint {display_checkpoint_path}"
            ),
            "import_run": (
                "medfusion import-run "
                f"--config {config_path_str} --checkpoint {display_checkpoint_path}"
            ),
        },
    }


def _validate_compiled_run_spec(run_spec: dict[str, Any]) -> dict[str, Any]:
    config = _instantiate_experiment_config(run_spec)
    errors = validate_config(config)
    warnings = _collect_contract_warnings(config)
    return {
        "ok": len(errors) == 0,
        "errors": _serialize_validation_errors(errors),
        "warnings": warnings,
        "mainline_contract": _build_mainline_contract(config),
        "experiment_config": config.to_dict(),
    }


def build_training_payload_from_runspec(run_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_name": run_spec["experimentName"],
        "training_model_config": {
            "backbone": run_spec["model"]["vision"]["backbone"],
            "num_classes": run_spec["model"]["numClasses"],
            "pretrained": run_spec["model"]["vision"]["pretrained"],
            "freeze_backbone": run_spec["model"]["vision"]["freezeBackbone"],
            "feature_dim": run_spec["model"]["vision"]["featureDim"],
            "tabular_output_dim": run_spec["model"]["tabular"]["outputDim"],
            "fusion_type": run_spec["model"]["fusion"]["fusionType"],
        },
        "dataset_config": {
            "dataset": run_spec["experimentName"],
            "csv_path": run_spec["data"]["csvPath"],
            "image_dir": run_spec["data"]["imageDir"],
            "image_path_column": run_spec["data"]["imagePathColumn"],
            "target_column": run_spec["data"]["targetColumn"],
            "patient_id_column": run_spec["data"]["patientIdColumn"] or None,
            "numerical_features": run_spec["data"]["numericalFeatures"],
            "categorical_features": run_spec["data"]["categoricalFeatures"],
            "num_classes": run_spec["model"]["numClasses"],
        },
        "training_config": {
            "epochs": run_spec["training"]["numEpochs"],
            "batch_size": run_spec["data"]["batchSize"],
            "num_workers": run_spec["data"]["numWorkers"],
            "image_size": run_spec["data"]["imageSize"],
            "mixed_precision": run_spec["training"]["mixedPrecision"],
            "use_progressive_training": run_spec["training"]["useProgressiveTraining"],
            "optimizer": run_spec["training"]["optimizer"]["optimizer"],
            "learning_rate": run_spec["training"]["optimizer"]["learningRate"],
            "weight_decay": run_spec["training"]["optimizer"]["weightDecay"],
            "scheduler": run_spec["training"]["scheduler"]["scheduler"],
            "step_size": run_spec["training"]["scheduler"]["stepSize"],
            "patience": run_spec["training"]["scheduler"]["patience"],
            "monitor": "accuracy",
            "mode": "max",
            "use_tensorboard": run_spec["logging"]["useTensorboard"],
        },
        "source_context": {
            "source_type": "advanced_builder",
            "entrypoint": "advanced-builder-canvas",
        },
    }


def _infer_preset(chosen_components: dict[AdvancedBuilderFamily, str]) -> str:
    contract = _advanced_builder_contract()
    chosen_component_ids = set(chosen_components.values())
    ranked_rules = sorted(
        contract.get("preset_rules", []),
        key=lambda item: item.get("priority", 0),
        reverse=True,
    )
    for rule in ranked_rules:
        if chosen_component_ids.intersection(set(rule.get("match_any_components", []))):
            return str(rule["preset"])
    return str(contract.get("default_preset", "quickstart"))


def _component_contract_map() -> dict[str, dict[str, Any]]:
    catalog = export_model_catalog()
    mapping: dict[str, dict[str, Any]] = {}
    for unit in catalog["units"]:
        advanced_component_id = unit.get("advanced_builder_component_id")
        if advanced_component_id:
            mapping[str(advanced_component_id)] = dict(
                unit.get("advanced_builder_contract") or {}
            )
    return mapping


def _catalog_units_by_advanced_component() -> dict[str, dict[str, Any]]:
    catalog = export_model_catalog()
    return {
        str(unit["advanced_builder_component_id"]): unit
        for unit in catalog["units"]
        if unit.get("advanced_builder_component_id")
    }


def _apply_component_to_spec(
    spec: dict[str, Any],
    component_id: str,
    issues: list[dict[str, Any]],
) -> None:
    prefill = _component_prefill_map().get(component_id)
    if prefill:
        if "csvPath" in prefill:
            spec["data"]["csvPath"] = prefill["csvPath"]
        if "imageDir" in prefill:
            spec["data"]["imageDir"] = prefill["imageDir"]
        if "imagePathColumn" in prefill:
            spec["data"]["imagePathColumn"] = prefill["imagePathColumn"]
        if "targetColumn" in prefill:
            spec["data"]["targetColumn"] = prefill["targetColumn"]
        if "patientIdColumn" in prefill:
            spec["data"]["patientIdColumn"] = prefill["patientIdColumn"]
        if "numericalFeatures" in prefill:
            spec["data"]["numericalFeatures"] = list(prefill["numericalFeatures"])
        if "categoricalFeatures" in prefill:
            spec["data"]["categoricalFeatures"] = list(prefill["categoricalFeatures"])
        if "backbone" in prefill:
            spec["model"]["vision"]["backbone"] = prefill["backbone"]
        if "featureDim" in prefill:
            spec["model"]["vision"]["featureDim"] = prefill["featureDim"]
        if "attentionType" in prefill:
            spec["model"]["vision"]["attentionType"] = prefill["attentionType"]
        if "pretrained" in prefill:
            spec["model"]["vision"]["pretrained"] = prefill["pretrained"]
        if "freezeBackbone" in prefill:
            spec["model"]["vision"]["freezeBackbone"] = prefill["freezeBackbone"]
        if "tabularHiddenDims" in prefill:
            spec["model"]["tabular"]["hiddenDims"] = list(prefill["tabularHiddenDims"])
        if "tabularOutputDim" in prefill:
            spec["model"]["tabular"]["outputDim"] = prefill["tabularOutputDim"]
        if "tabularDropout" in prefill:
            spec["model"]["tabular"]["dropout"] = prefill["tabularDropout"]
        if "fusionType" in prefill:
            spec["model"]["fusion"]["fusionType"] = prefill["fusionType"]
        if "fusionHiddenDim" in prefill:
            spec["model"]["fusion"]["hiddenDim"] = prefill["fusionHiddenDim"]
        if "fusionDropout" in prefill:
            spec["model"]["fusion"]["dropout"] = prefill["fusionDropout"]
        if "fusionNumHeads" in prefill:
            spec["model"]["fusion"]["numHeads"] = prefill["fusionNumHeads"]
        if "numClasses" in prefill:
            spec["model"]["numClasses"] = prefill["numClasses"]
        if "useAuxiliaryHeads" in prefill:
            spec["model"]["useAuxiliaryHeads"] = prefill["useAuxiliaryHeads"]
        if "useAttentionSupervision" in prefill:
            spec["training"]["useAttentionSupervision"] = prefill["useAttentionSupervision"]
        if "useProgressiveTraining" in prefill:
            spec["training"]["useProgressiveTraining"] = prefill["useProgressiveTraining"]
        if "numEpochs" in prefill:
            spec["training"]["numEpochs"] = prefill["numEpochs"]
        if "stage1Epochs" in prefill:
            spec["training"]["stage1Epochs"] = prefill["stage1Epochs"]
        if "stage2Epochs" in prefill:
            spec["training"]["stage2Epochs"] = prefill["stage2Epochs"]
        if "stage3Epochs" in prefill:
            spec["training"]["stage3Epochs"] = prefill["stage3Epochs"]
    for warning in _component_contract_map().get(component_id, {}).get(
        "warning_metadata",
        [],
    ):
        issues.append(
            _issue(
                level="warning",
                code=warning["code"],
                path=warning.get("path"),
                message=warning["message"],
                suggestion=warning.get("suggestion"),
            )
        )
    if component_id == "concatenate_fusion":
        spec["model"]["fusion"]["hiddenDim"] = (
            spec["model"]["vision"]["featureDim"] + spec["model"]["tabular"]["outputDim"]
        )
    if prefill:
        return

    issues.append(
        _issue(
            level="error",
            code="ABG-E010",
            path="nodes[].data.componentId",
            message=f"当前编译器还不能把组件 {component_id} 降级映射到正式版 RunSpec。",
            context={"component_id": component_id},
            suggestion="替换为当前编译器已支持的组件后重试。",
        )
    )


def _component_map() -> dict[str, AdvancedBuilderComponent]:
    return {component.id: component for component in _projected_components()}


def _advanced_builder_contract() -> dict[str, Any]:
    return export_advanced_builder_contract()


def _family_labels() -> dict[AdvancedBuilderFamily, str]:
    labels = _advanced_builder_contract()["family_labels"]
    return {key: value for key, value in labels.items()}


def _required_families() -> tuple[AdvancedBuilderFamily, ...]:
    return tuple(_advanced_builder_contract()["required_families"])


def _connection_rules() -> tuple[AdvancedBuilderConnectionRule, ...]:
    return tuple(
        AdvancedBuilderConnectionRule(
            from_family=rule["from_family"],
            to_family=rule["to_family"],
            status=rule["status"],
            description=rule["description"],
        )
        for rule in _advanced_builder_contract()["connection_rules"]
    )


def _rule_map() -> dict[tuple[AdvancedBuilderFamily, AdvancedBuilderFamily], AdvancedBuilderConnectionRule]:
    return {
        (rule.from_family, rule.to_family): rule
        for rule in _connection_rules()
    }


def _projected_components() -> tuple[AdvancedBuilderComponent, ...]:
    """Project advanced-builder components from the official model catalog."""
    catalog = export_model_catalog()
    family_map: dict[str, AdvancedBuilderFamily] = {
        family: metadata["advanced_family"]
        for family, metadata in catalog["advanced_builder"]["family_projection"].items()
    }

    projected: list[AdvancedBuilderComponent] = []
    for unit in catalog["units"]:
        advanced_component_id = unit.get("advanced_builder_component_id")
        family = family_map.get(unit.get("family"))
        if not advanced_component_id or family is None:
            continue
        projected.append(
            AdvancedBuilderComponent(
                id=str(advanced_component_id),
                family=family,
                label=str(unit["label"]),
                status=unit["status"],
                description=str(unit["description"]),
                schema_path=(
                    unit.get("config_requirements", [None])[0]
                    if unit.get("config_requirements")
                    else None
                ),
            )
        )
    return tuple(projected)


def _projected_blueprints() -> tuple[AdvancedBuilderBlueprint, ...]:
    """Project advanced-builder blueprints from the official model catalog."""
    catalog = export_model_catalog()
    unit_advanced_ids = {
        unit["id"]: unit.get("advanced_builder_component_id")
        for unit in catalog["units"]
    }
    projected: list[AdvancedBuilderBlueprint] = []
    for model in catalog["models"]:
        advanced_blueprint_id = model.get("advanced_builder_blueprint_id")
        if not advanced_blueprint_id:
            continue
        projected_components = tuple(
            advanced_id
            for component_id in model.get("component_ids", [])
            for advanced_id in [unit_advanced_ids.get(component_id)]
            if advanced_id
        )
        projected.append(
            AdvancedBuilderBlueprint(
                id=str(advanced_blueprint_id),
                label=str(model["label"]),
                status=model["status"],
                description=str(model["description"]),
                components=projected_components,
                compiles_to=(
                    "ExperimentConfig / official model catalog projection"
                    if model["status"] == "compile_ready"
                    else None
                ),
                blockers=tuple(
                    []
                    if model["status"] != "draft_only"
                    else ["当前模板仍处于草稿层，不能进入默认正式版主线。"]
                ),
            )
        )
    return tuple(projected)


def _component_prefill_map() -> dict[str, dict[str, Any]]:
    catalog = export_model_catalog()
    mapping: dict[str, dict[str, Any]] = {}
    for unit in catalog["units"]:
        advanced_component_id = unit.get("advanced_builder_component_id")
        wizard_prefill = unit.get("wizard_prefill") or {}
        if advanced_component_id and wizard_prefill:
            mapping[str(advanced_component_id)] = dict(wizard_prefill)
    return mapping


# Compatibility aliases for modules that still import the old names directly.
ADVANCED_BUILDER_FAMILY_LABELS: dict[AdvancedBuilderFamily, str] = _family_labels()
ADVANCED_BUILDER_COMPONENTS: tuple[AdvancedBuilderComponent, ...] = _projected_components()
ADVANCED_BUILDER_BLUEPRINTS: tuple[AdvancedBuilderBlueprint, ...] = _projected_blueprints()
ADVANCED_BUILDER_CONNECTION_RULES: tuple[AdvancedBuilderConnectionRule, ...] = _connection_rules()
REQUIRED_FAMILIES: tuple[AdvancedBuilderFamily, ...] = _required_families()


def export_catalog() -> dict[str, Any]:
    components = _projected_components()
    blueprints = _projected_blueprints()
    return {
        "families": _family_labels(),
        "status_labels": _advanced_builder_contract()["status_labels"],
        "components": [
            {
                "id": component.id,
                "family": component.family,
                "label": component.label,
                "status": component.status,
                "description": component.description,
                "schema_path": component.schema_path,
                "notes": (
                    _catalog_units_by_advanced_component()
                    .get(component.id, {})
                    .get("advanced_builder_contract", {})
                    .get("compile_notes", [])
                ),
                "advanced_builder_contract": (
                    _catalog_units_by_advanced_component()
                    .get(component.id, {})
                    .get("advanced_builder_contract", {})
                ),
            }
            for component in components
        ],
        "connection_rules": [
            {
                "from_family": rule.from_family,
                "to_family": rule.to_family,
                "status": rule.status,
                "description": rule.description,
            }
            for rule in _connection_rules()
        ],
        "blueprints": [
            {
                "id": blueprint.id,
                "label": blueprint.label,
                "status": blueprint.status,
                "description": blueprint.description,
                "components": list(blueprint.components),
                "compiles_to": blueprint.compiles_to,
                "blockers": list(blueprint.blockers),
                "recommended_preset": (
                    next(
                        (
                            model.get("advanced_builder_contract", {}).get(
                                "recommended_preset"
                            )
                            for model in export_model_catalog()["models"]
                            if model.get("advanced_builder_blueprint_id") == blueprint.id
                        ),
                        None,
                    )
                ),
            }
            for blueprint in blueprints
        ],
    }


def compile_graph_to_runspec(
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    components = _component_map()
    rules = _rule_map()

    if not nodes:
        issues.append(
            _issue(
                level="error",
                code="ABG-E001",
                path="nodes",
                message="当前画布为空，无法生成配置草案。",
                suggestion="先选择一条 compile-ready blueprint，或手动补齐 6 个核心组件家族。",
            )
        )
        return {
            "preset": "quickstart",
            "run_spec": None,
            "experiment_config": None,
            "contract_validation": None,
            "mainline_contract": None,
            "issues": issues,
            "chosen_components": {},
        }

    chosen_components: dict[AdvancedBuilderFamily, str] = {}
    duplicate_families: list[AdvancedBuilderFamily] = []
    node_by_id: dict[str, dict[str, Any]] = {}
    family_node_ids: dict[AdvancedBuilderFamily, list[str]] = {}

    for node in nodes:
        node_id = str(node.get("id") or "")
        data = node.get("data") or {}
        component_id = data.get("componentId")
        if component_id not in components:
            issues.append(
                _issue(
                    level="error",
                    code="ABG-E002",
                    path="nodes[].data.componentId",
                    message=f"节点 {node_id or '<unknown>'} 缺少合法 componentId。",
                    context={"node_id": node_id or "<unknown>"},
                    suggestion="为该节点重新选择一个合法组件，或删除这个无效节点。",
                )
            )
            continue
        component = components[component_id]
        node_by_id[node_id] = node
        family_node_ids.setdefault(component.family, []).append(node_id or "<unknown>")
        if component.family in chosen_components:
            duplicate_families.append(component.family)
        else:
            chosen_components[component.family] = component_id

        if component.status == "draft_only":
            issues.append(
                _issue(
                    level="error",
                    code="ABG-E003",
                    path="nodes[].data.componentId",
                    message=f"当前图包含仅草稿组件：{component.label}。这些组件还不能编译进正式版主链。",
                    context={
                        "component_id": component.id,
                        "node_id": node_id or "<unknown>",
                    },
                    suggestion="把该节点替换为 compile-ready 或 conditional 组件后再编译。",
                )
            )

    if duplicate_families:
        family_labels = " / ".join(
            ADVANCED_BUILDER_FAMILY_LABELS[family] for family in duplicate_families
        )
        duplicate_node_ids = list(
            dict.fromkeys(
                node_id
                for family in duplicate_families
                for node_id in family_node_ids.get(family, [])
                if node_id and not node_id.startswith("<")
            )
        )
        issues.append(
            _issue(
                level="error",
                code="ABG-E004",
                path="nodes[].data.componentId",
                message=f"当前图存在重复组件家族：{family_labels}。正式版编译层当前要求每个核心家族最多一个组件。",
                context={
                    "families": duplicate_families,
                    "node_ids": duplicate_node_ids,
                },
                suggestion="每个核心 family 仅保留一个节点，删除重复节点后重试。",
            )
        )

    family_labels = _family_labels()

    missing_families = [family for family in _required_families() if family not in chosen_components]
    if missing_families:
        issues.append(
            _issue(
                level="error",
                code="ABG-E005",
                path="nodes",
                message=f"缺少必需组件家族：{' / '.join(family_labels[f] for f in missing_families)}。",
                context={"families": missing_families},
                suggestion="补齐缺失家族对应的组件节点，再重新编译。",
            )
        )

    family_by_node_id: dict[str, AdvancedBuilderFamily] = {}
    for node_id, node in node_by_id.items():
        component_id = (node.get("data") or {}).get("componentId")
        if component_id in components:
            family_by_node_id[node_id] = components[component_id].family

    present_connections: set[tuple[AdvancedBuilderFamily, AdvancedBuilderFamily]] = set()
    for edge in edges:
        source_node_id = str(edge.get("source") or "")
        target_node_id = str(edge.get("target") or "")
        source_family = family_by_node_id.get(source_node_id)
        target_family = family_by_node_id.get(target_node_id)
        if not source_family or not target_family:
            missing_endpoints: list[str] = []
            if source_node_id not in family_by_node_id:
                missing_endpoints.append(f"source={source_node_id or '<empty>'}")
            if target_node_id not in family_by_node_id:
                missing_endpoints.append(f"target={target_node_id or '<empty>'}")
            if missing_endpoints:
                issues.append(
                    _issue(
                        level="error",
                        code="ABG-E006",
                        path="edges",
                        message=(
                            "发现悬空连接，当前连接引用了不存在的节点："
                            f"{', '.join(missing_endpoints)}。"
                        ),
                        context={
                            "source_node_id": source_node_id or "<empty>",
                            "target_node_id": target_node_id or "<empty>",
                        },
                        suggestion="修复或删除悬空连接，确保 source/target 都指向现存节点。",
                    )
                )
            continue
        rule = rules.get((source_family, target_family))
        if rule is None:
            issues.append(
                _issue(
                    level="error",
                    code="ABG-E007",
                    path="edges",
                    message=f"当前没有定义 {family_labels[source_family]} -> {family_labels[target_family]} 的正式版连接规则。",
                    context={
                        "from_family": source_family,
                        "to_family": target_family,
                        "source_node_id": source_node_id or "<empty>",
                        "target_node_id": target_node_id or "<empty>",
                    },
                    suggestion="按连接规则重新连线，使用已定义的 family 连接组合。",
                )
            )
            continue
        if rule.status == "blocked":
            issues.append(
                _issue(
                    level="error",
                    code="ABG-E008",
                    path="edges",
                    message=rule.description,
                    context={
                        "from_family": source_family,
                        "to_family": target_family,
                        "source_node_id": source_node_id or "<empty>",
                        "target_node_id": target_node_id or "<empty>",
                    },
                    suggestion="删除这条被禁止的连接，并按 required/conditional 规则重连。",
                )
            )
            continue
        present_connections.add((source_family, target_family))

    for rule in _connection_rules():
        if rule.status != "required":
            continue
        if rule.from_family in chosen_components and rule.to_family in chosen_components:
            if (rule.from_family, rule.to_family) not in present_connections:
                issues.append(
                    _issue(
                        level="error",
                        code="ABG-E009",
                        path="edges",
                        message=f"缺少必需连接：{family_labels[rule.from_family]} -> {family_labels[rule.to_family]}。",
                        context={
                            "from_family": rule.from_family,
                            "to_family": rule.to_family,
                        },
                        suggestion="补上这条 required 连接后再编译。",
                    )
                )

    preset = _infer_preset(chosen_components)
    if any(issue["level"] == "error" for issue in issues):
        return {
            "preset": preset,
            "run_spec": None,
            "experiment_config": None,
            "contract_validation": None,
            "mainline_contract": None,
            "issues": issues,
            "chosen_components": chosen_components,
        }

    spec = _create_run_spec_preset(preset)
    spec["projectName"] = "medfusion-formal"
    spec["experimentName"] = f"advanced-{preset}-graph"
    spec["description"] = "Compiled from the formal-release advanced builder graph API."
    spec["tags"] = sorted(set(spec["tags"] + ["advanced-builder", "compiled", "api"]))
    spec["logging"]["outputDir"] = (
        f"outputs/{_slugify(spec['projectName'])}/{_slugify(spec['experimentName'])}"
    )

    for family in REQUIRED_FAMILIES:
        component_id = chosen_components.get(family)
        if component_id:
            _apply_component_to_spec(spec, component_id, issues)

    contract_validation = _validate_compiled_run_spec(spec)

    return {
        "preset": preset,
        "run_spec": spec,
        "experiment_config": contract_validation["experiment_config"],
        "contract_validation": {
            "ok": contract_validation["ok"],
            "errors": contract_validation["errors"],
            "warnings": contract_validation["warnings"],
        },
        "mainline_contract": contract_validation["mainline_contract"],
        "issues": issues,
        "chosen_components": chosen_components,
    }
