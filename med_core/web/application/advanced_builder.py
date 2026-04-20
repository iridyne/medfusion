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


ADVANCED_BUILDER_FAMILY_LABELS: dict[AdvancedBuilderFamily, str] = {
    "data_input": "数据输入",
    "vision_backbone": "视觉 backbone",
    "tabular_encoder": "表格编码器",
    "fusion": "融合层",
    "head": "任务头",
    "training_strategy": "训练策略",
}

ADVANCED_BUILDER_COMPONENTS: tuple[AdvancedBuilderComponent, ...] = (
    AdvancedBuilderComponent(
        id="image_tabular_dataset",
        family="data_input",
        label="图像 + 表格输入",
        status="compile_ready",
        description="当前正式版默认主链，映射到 image_tabular 训练 schema。",
        schema_path="data.dataset_type=image_tabular",
    ),
    AdvancedBuilderComponent(
        id="three_phase_ct_dataset",
        family="data_input",
        label="三相 CT + 临床输入",
        status="draft_only",
        description="runtime 已有专项模型，但还没有进入正式版高级建模器的通用编译面。",
        schema_path="data.dataset_type=three_phase_ct_tabular",
    ),
    AdvancedBuilderComponent(
        id="resnet18_backbone",
        family="vision_backbone",
        label="ResNet18",
        status="compile_ready",
        description="默认轻量视觉 backbone，适合作为正式版起步骨架。",
        schema_path="model.vision.backbone=resnet18",
    ),
    AdvancedBuilderComponent(
        id="efficientnet_b0_backbone",
        family="vision_backbone",
        label="EfficientNet-B0",
        status="compile_ready",
        description="更稳健的常规研究基线，适合中等规模正式版模板。",
        schema_path="model.vision.backbone=efficientnet_b0",
    ),
    AdvancedBuilderComponent(
        id="attention_backbone_bundle",
        family="vision_backbone",
        label="Attention-supervised backbone",
        status="conditional",
        description="当前仅在 CBAM 注意力路径下可用，需要前台显式提示条件。",
        schema_path="model.vision.attention_type=cbam",
    ),
    AdvancedBuilderComponent(
        id="mlp_tabular_encoder",
        family="tabular_encoder",
        label="MLP 表格编码器",
        status="compile_ready",
        description="当前正式版默认的表格分支编码器。",
        schema_path="model.tabular",
    ),
    AdvancedBuilderComponent(
        id="concatenate_fusion",
        family="fusion",
        label="Concatenate Fusion",
        status="compile_ready",
        description="最稳的正式版起步融合层，默认 quickstart 使用此路径。",
        schema_path="model.fusion.fusion_type=concatenate",
    ),
    AdvancedBuilderComponent(
        id="gated_fusion",
        family="fusion",
        label="Gated Fusion",
        status="compile_ready",
        description="适合作为更稳健的研究基线融合策略。",
        schema_path="model.fusion.fusion_type=gated",
    ),
    AdvancedBuilderComponent(
        id="attention_fusion",
        family="fusion",
        label="Attention Fusion",
        status="conditional",
        description="适合结果强化和注意力审查，但正式版应在骨架推荐后再开放。",
        schema_path="model.fusion.fusion_type=attention",
    ),
    AdvancedBuilderComponent(
        id="cross_attention_fusion",
        family="fusion",
        label="Cross Attention Fusion",
        status="draft_only",
        description="runtime 有探索性空间，但当前还不进入正式版高级模式的可编译默认集。",
        schema_path="model.fusion.fusion_type=cross_attention",
    ),
    AdvancedBuilderComponent(
        id="classification_head",
        family="head",
        label="分类头",
        status="compile_ready",
        description="当前正式版主链默认任务头。",
        schema_path="model.num_classes",
    ),
    AdvancedBuilderComponent(
        id="survival_head",
        family="head",
        label="生存任务头",
        status="draft_only",
        description="仓库有相关能力，但还没有进入正式版高级建模器的主叙事。",
    ),
    AdvancedBuilderComponent(
        id="standard_training",
        family="training_strategy",
        label="标准训练",
        status="compile_ready",
        description="当前最稳定的正式版训练策略。",
        schema_path="training",
    ),
    AdvancedBuilderComponent(
        id="progressive_training",
        family="training_strategy",
        label="分阶段训练",
        status="conditional",
        description="可以编译，但需要显式满足 stage epoch 总和约束。",
        schema_path="training.use_progressive_training=true",
    ),
)

ADVANCED_BUILDER_CONNECTION_RULES: tuple[AdvancedBuilderConnectionRule, ...] = (
    AdvancedBuilderConnectionRule(
        from_family="data_input",
        to_family="vision_backbone",
        status="required",
        description="只要选择图像模态，必须先接一个视觉 backbone 才能继续编译。",
    ),
    AdvancedBuilderConnectionRule(
        from_family="data_input",
        to_family="tabular_encoder",
        status="required",
        description="图像 + 表格主链要求至少一条表格编码分支。",
    ),
    AdvancedBuilderConnectionRule(
        from_family="vision_backbone",
        to_family="fusion",
        status="required",
        description="视觉特征必须经过正式版支持的融合层才能进入任务头。",
    ),
    AdvancedBuilderConnectionRule(
        from_family="tabular_encoder",
        to_family="fusion",
        status="required",
        description="当前正式版多模态主链要求把表格特征也接入融合层。",
    ),
    AdvancedBuilderConnectionRule(
        from_family="fusion",
        to_family="head",
        status="required",
        description="融合层输出必须进入任务头，才能定义最终训练目标。",
    ),
    AdvancedBuilderConnectionRule(
        from_family="head",
        to_family="training_strategy",
        status="required",
        description="任务头之后必须接训练策略，图才进入可执行主链。",
    ),
    AdvancedBuilderConnectionRule(
        from_family="data_input",
        to_family="head",
        status="blocked",
        description="不允许跳过 backbone / fusion 直接把原始输入接到任务头。",
    ),
    AdvancedBuilderConnectionRule(
        from_family="vision_backbone",
        to_family="training_strategy",
        status="blocked",
        description="不允许跳过融合层和任务头直接进入训练策略。",
    ),
)

ADVANCED_BUILDER_BLUEPRINTS: tuple[AdvancedBuilderBlueprint, ...] = (
    AdvancedBuilderBlueprint(
        id="quickstart_multimodal",
        label="正式版 quickstart 多模态骨架",
        status="compile_ready",
        description="图像 + 表格输入，ResNet18 + MLP + Concatenate + 分类头 + 标准训练。",
        components=(
            "image_tabular_dataset",
            "resnet18_backbone",
            "mlp_tabular_encoder",
            "concatenate_fusion",
            "classification_head",
            "standard_training",
        ),
        compiles_to="ExperimentConfig / configs/starter/quickstart.yaml",
    ),
    AdvancedBuilderBlueprint(
        id="clinical_gated_baseline",
        label="稳健研究基线骨架",
        status="compile_ready",
        description="图像 + 表格输入，EfficientNet-B0 + MLP + Gated Fusion + 分类头 + 分阶段训练。",
        components=(
            "image_tabular_dataset",
            "efficientnet_b0_backbone",
            "mlp_tabular_encoder",
            "gated_fusion",
            "classification_head",
            "progressive_training",
        ),
        compiles_to="ExperimentConfig / formal release baseline preset",
    ),
    AdvancedBuilderBlueprint(
        id="attention_audit_path",
        label="结果审查 / 注意力增强骨架",
        status="compile_ready",
        description="用于结果交付和注意力可视化审查的正式版骨架。",
        components=(
            "image_tabular_dataset",
            "attention_backbone_bundle",
            "mlp_tabular_encoder",
            "attention_fusion",
            "classification_head",
            "standard_training",
        ),
        compiles_to="ExperimentConfig / result-audit preset",
    ),
    AdvancedBuilderBlueprint(
        id="three_phase_ct_builder",
        label="三相 CT 通用高级建模骨架",
        status="draft_only",
        description="当前仍是专项 runtime 能力，还没有进入正式版高级建模器的通用编译层。",
        components=(
            "three_phase_ct_dataset",
            "gated_fusion",
            "classification_head",
            "standard_training",
        ),
        blockers=("需要正式版图编译层先支持三相 CT 专项 schema。",),
    ),
    AdvancedBuilderBlueprint(
        id="survival_builder",
        label="生存分析建模骨架",
        status="draft_only",
        description="当前不作为正式版默认高级模式开放。",
        components=(
            "image_tabular_dataset",
            "efficientnet_b0_backbone",
            "mlp_tabular_encoder",
            "gated_fusion",
            "survival_head",
            "standard_training",
        ),
        blockers=("正式版主叙事当前先围绕分类主链，不把 survival 作为默认承诺。",),
    ),
)

REQUIRED_FAMILIES: tuple[AdvancedBuilderFamily, ...] = (
    "data_input",
    "vision_backbone",
    "tabular_encoder",
    "fusion",
    "head",
    "training_strategy",
)


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
    vision_component = chosen_components.get("vision_backbone", "")
    fusion_component = chosen_components.get("fusion", "")
    training_component = chosen_components.get("training_strategy", "")

    if vision_component == "attention_backbone_bundle" or fusion_component == "attention_fusion":
        return "showcase"
    if (
        vision_component == "efficientnet_b0_backbone"
        or fusion_component == "gated_fusion"
        or training_component == "progressive_training"
    ):
        return "clinical"
    return "quickstart"


def _apply_component_to_spec(
    spec: dict[str, Any],
    component_id: str,
    issues: list[dict[str, Any]],
) -> None:
    if component_id == "image_tabular_dataset":
        return
    if component_id == "resnet18_backbone":
        spec["model"]["vision"]["backbone"] = "resnet18"
        spec["model"]["vision"]["featureDim"] = 128
        spec["model"]["vision"]["attentionType"] = "cbam"
        return
    if component_id == "efficientnet_b0_backbone":
        spec["model"]["vision"]["backbone"] = "efficientnet_b0"
        spec["model"]["vision"]["featureDim"] = 192
        spec["model"]["vision"]["attentionType"] = "cbam"
        return
    if component_id == "attention_backbone_bundle":
        spec["model"]["vision"]["backbone"] = "resnet50"
        spec["model"]["vision"]["featureDim"] = 256
        spec["model"]["vision"]["attentionType"] = "cbam"
        spec["training"]["useAttentionSupervision"] = True
        issues.append(
            _issue(
                level="warning",
                code="ABG-W001",
                path="model.vision",
                message="当前图使用了 attention-supervised backbone，编译结果会默认走 CBAM + attention supervision 条件路径。",
                suggestion="确认这是预期路径；如需更稳妥的默认链，改用 ResNet18 或 EfficientNet-B0 backbone。",
            )
        )
        return
    if component_id == "mlp_tabular_encoder":
        spec["model"]["tabular"]["hiddenDims"] = [32]
        spec["model"]["tabular"]["outputDim"] = 16
        spec["model"]["tabular"]["dropout"] = 0.2
        return
    if component_id == "concatenate_fusion":
        spec["model"]["fusion"]["fusionType"] = "concatenate"
        spec["model"]["fusion"]["hiddenDim"] = (
            spec["model"]["vision"]["featureDim"] + spec["model"]["tabular"]["outputDim"]
        )
        spec["model"]["fusion"]["dropout"] = 0.3
        return
    if component_id == "gated_fusion":
        spec["model"]["fusion"]["fusionType"] = "gated"
        spec["model"]["fusion"]["hiddenDim"] = 160
        spec["model"]["fusion"]["dropout"] = 0.3
        return
    if component_id == "attention_fusion":
        spec["model"]["fusion"]["fusionType"] = "attention"
        spec["model"]["fusion"]["hiddenDim"] = 192
        spec["model"]["fusion"]["numHeads"] = 4
        spec["model"]["fusion"]["dropout"] = 0.25
        issues.append(
            _issue(
                level="warning",
                code="ABG-W002",
                path="model.fusion",
                message="当前图使用了 attention fusion，编译结果会保留注意力路径，但仍受正式版主链的现有 fusion schema 约束。",
                suggestion="如果只需要最稳主链，可改用 concatenate fusion 或 gated fusion。",
            )
        )
        return
    if component_id == "classification_head":
        spec["model"]["numClasses"] = 2
        spec["model"]["useAuxiliaryHeads"] = True
        return
    if component_id == "standard_training":
        spec["training"]["useProgressiveTraining"] = False
        return
    if component_id == "progressive_training":
        spec["training"]["useProgressiveTraining"] = True
        spec["training"]["numEpochs"] = 18
        spec["training"]["stage1Epochs"] = 6
        spec["training"]["stage2Epochs"] = 8
        spec["training"]["stage3Epochs"] = 4
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
    return {component.id: component for component in ADVANCED_BUILDER_COMPONENTS}


def _rule_map() -> dict[tuple[AdvancedBuilderFamily, AdvancedBuilderFamily], AdvancedBuilderConnectionRule]:
    return {
        (rule.from_family, rule.to_family): rule
        for rule in ADVANCED_BUILDER_CONNECTION_RULES
    }


def export_catalog() -> dict[str, Any]:
    return {
        "families": ADVANCED_BUILDER_FAMILY_LABELS,
        "components": [
            {
                "id": component.id,
                "family": component.family,
                "label": component.label,
                "status": component.status,
                "description": component.description,
                "schema_path": component.schema_path,
            }
            for component in ADVANCED_BUILDER_COMPONENTS
        ],
        "connection_rules": [
            {
                "from_family": rule.from_family,
                "to_family": rule.to_family,
                "status": rule.status,
                "description": rule.description,
            }
            for rule in ADVANCED_BUILDER_CONNECTION_RULES
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
            }
            for blueprint in ADVANCED_BUILDER_BLUEPRINTS
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

    missing_families = [family for family in REQUIRED_FAMILIES if family not in chosen_components]
    if missing_families:
        issues.append(
            _issue(
                level="error",
                code="ABG-E005",
                path="nodes",
                message=f"缺少必需组件家族：{' / '.join(ADVANCED_BUILDER_FAMILY_LABELS[f] for f in missing_families)}。",
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
                    message=f"当前没有定义 {ADVANCED_BUILDER_FAMILY_LABELS[source_family]} -> {ADVANCED_BUILDER_FAMILY_LABELS[target_family]} 的正式版连接规则。",
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

    for rule in ADVANCED_BUILDER_CONNECTION_RULES:
        if rule.status != "required":
            continue
        if rule.from_family in chosen_components and rule.to_family in chosen_components:
            if (rule.from_family, rule.to_family) not in present_connections:
                issues.append(
                    _issue(
                        level="error",
                        code="ABG-E009",
                        path="edges",
                        message=f"缺少必需连接：{ADVANCED_BUILDER_FAMILY_LABELS[rule.from_family]} -> {ADVANCED_BUILDER_FAMILY_LABELS[rule.to_family]}。",
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
