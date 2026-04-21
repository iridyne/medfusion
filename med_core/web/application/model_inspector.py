"""Application helpers for standalone model skeleton inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from med_core.backbones import create_tabular_backbone, create_vision_backbone
from med_core.configs import (
    DataConfig,
    ExperimentConfig,
    FusionConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TabularConfig,
    TrainingConfig,
    VisionConfig,
    validate_config,
)
from med_core.fusion import MultiModalFusionModel, create_fusion_module


@dataclass
class ModelInspectPayload:
    num_classes: int
    use_auxiliary_heads: bool
    vision: dict[str, Any]
    tabular: dict[str, Any]
    fusion: dict[str, Any]
    numerical_features: list[str]
    categorical_features: list[str]
    image_size: int = 224
    use_attention_supervision: bool = False
    num_epochs: int = 3


def _build_config(payload: ModelInspectPayload) -> ExperimentConfig:
    return ExperimentConfig(
        project_name="web-model-inspect",
        experiment_name="model-inspect",
        model=ModelConfig(
            num_classes=payload.num_classes,
            use_auxiliary_heads=payload.use_auxiliary_heads,
            vision=VisionConfig(
                backbone=payload.vision["backbone"],
                pretrained=bool(payload.vision.get("pretrained", True)),
                freeze_backbone=bool(payload.vision.get("freeze_backbone", False)),
                feature_dim=int(payload.vision.get("feature_dim", 128)),
                dropout=float(payload.vision.get("dropout", 0.3)),
                attention_type=payload.vision.get("attention_type", "none"),
                enable_attention_supervision=bool(payload.use_attention_supervision),
            ),
            tabular=TabularConfig(
                hidden_dims=list(payload.tabular.get("hidden_dims", [32])),
                output_dim=int(payload.tabular.get("output_dim", 16)),
                dropout=float(payload.tabular.get("dropout", 0.2)),
            ),
            fusion=FusionConfig(
                fusion_type=payload.fusion.get("fusion_type", "concatenate"),
                hidden_dim=int(payload.fusion.get("hidden_dim", 144)),
                dropout=float(payload.fusion.get("dropout", 0.3)),
                num_heads=int(payload.fusion.get("num_heads", 4)),
            ),
        ),
        data=DataConfig(
            csv_path="data/mock/metadata.csv",
            image_dir="data/mock",
            image_path_column="image_path",
            target_column="diagnosis",
            patient_id_column="",
            numerical_features=payload.numerical_features,
            categorical_features=payload.categorical_features,
            batch_size=4,
            image_size=max(int(payload.image_size), 32),
            num_workers=0,
            pin_memory=False,
        ),
        training=TrainingConfig(
            num_epochs=max(int(payload.num_epochs), 1),
            use_progressive_training=False,
            use_attention_supervision=bool(payload.use_attention_supervision),
            optimizer=OptimizerConfig(
                optimizer="adam",
                learning_rate=1e-3,
            ),
            scheduler=SchedulerConfig(scheduler="none"),
        ),
        logging=LoggingConfig(output_dir="outputs/model-inspect"),
    )


def _runtime_instantiate(config: ExperimentConfig) -> dict[str, Any]:
    # Avoid implicit network/downloads in inspect. Skeleton instantiation is enough.
    requested_pretrained = bool(config.model.vision.pretrained)
    vision_backbone = create_vision_backbone(
        backbone_name=config.model.vision.backbone,
        pretrained=False,
        freeze=config.model.vision.freeze_backbone,
        feature_dim=config.model.vision.feature_dim,
        dropout=config.model.vision.dropout,
        attention_type=config.model.vision.attention_type,
        enable_attention_supervision=config.model.vision.enable_attention_supervision,
    )
    tabular_input_dim = max(
        len(config.data.numerical_features) + len(config.data.categorical_features),
        1,
    )
    tabular_backbone = create_tabular_backbone(
        input_dim=tabular_input_dim,
        output_dim=config.model.tabular.output_dim,
        hidden_dims=config.model.tabular.hidden_dims,
        dropout=config.model.tabular.dropout,
    )
    fusion_kwargs = {"dropout": config.model.fusion.dropout}
    if config.model.fusion.fusion_type in {"attention", "cross_attention"}:
        fusion_kwargs["num_heads"] = config.model.fusion.num_heads
    fusion_module = create_fusion_module(
        fusion_type=config.model.fusion.fusion_type,
        vision_dim=config.model.vision.feature_dim,
        tabular_dim=config.model.tabular.output_dim,
        output_dim=config.model.fusion.hidden_dim,
        **fusion_kwargs,
    )
    model = MultiModalFusionModel(
        vision_backbone=vision_backbone,
        tabular_backbone=tabular_backbone,
        fusion_module=fusion_module,
        num_classes=config.model.num_classes,
        use_auxiliary_heads=config.model.use_auxiliary_heads,
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "vision_output_dim": int(vision_backbone.output_dim),
        "tabular_input_dim": tabular_input_dim,
        "tabular_output_dim": int(tabular_backbone.output_dim),
        "fusion_output_dim": int(fusion_module.output_dim),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "frozen_params": int(total_params - trainable_params),
        "pretrained_requested": requested_pretrained,
        "pretrained_materialized": False,
        "auxiliary_heads": bool(config.model.use_auxiliary_heads),
    }


def inspect_model_skeleton(payload: ModelInspectPayload) -> dict[str, Any]:
    config = _build_config(payload)
    validation_errors = validate_config(config)

    issues = [
        {
            "path": item.path,
            "message": item.message,
            "error_code": item.error_code,
            "suggestion": item.suggestion,
        }
        for item in validation_errors
    ]
    errors = [item for item in issues]
    warnings: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    tabular_feature_count = len(payload.numerical_features) + len(payload.categorical_features)
    if tabular_feature_count == 0:
        issues.append(
            {
                "path": "data.features",
                "message": "当前正式版主线要求至少一个表格特征进入模型分支。",
                "error_code": "M001",
                "suggestion": "先在数据步骤中选择 numerical_features 或 categorical_features",
            }
        )
        errors = list(issues)

    checks.append(
        {
            "key": "config_validation",
            "label": "结构校验",
            "status": "pass" if not validation_errors else "fail",
            "detail": "ExperimentConfig 结构校验"
            if not validation_errors
            else f"{len(validation_errors)} 个错误",
        }
    )
    checks.append(
        {
            "key": "tabular_features",
            "label": "表格特征输入",
            "status": "pass" if tabular_feature_count > 0 else "fail",
            "detail": f"共 {tabular_feature_count} 个特征",
        }
    )

    runtime_summary: dict[str, Any] | None = None
    runtime_error: str | None = None
    if not errors:
        try:
            runtime_summary = _runtime_instantiate(config)
            checks.append(
                {
                    "key": "runtime_instantiation",
                    "label": "Runtime 实例化",
                    "status": "pass",
                    "detail": "vision / tabular / fusion / model skeleton 已成功实例化",
                }
            )
        except Exception as exc:
            runtime_error = str(exc)
            errors.append(
                {
                    "path": "model.runtime",
                    "message": runtime_error,
                    "error_code": "M002",
                    "suggestion": "回退到主链支持的 backbone / fusion 组合，或检查维度配置",
                }
            )
            checks.append(
                {
                    "key": "runtime_instantiation",
                    "label": "Runtime 实例化",
                    "status": "fail",
                    "detail": runtime_error,
                }
            )

    if payload.vision.get("pretrained", True):
        warnings.append(
            {
                "path": "model.vision.pretrained",
                "message": "当前检查只实例化 skeleton，不下载预训练权重。",
                "error_code": "M101",
                "suggestion": "正式训练仍会按配置请求 pretrained 权重",
            }
        )

    if payload.vision.get("freeze_backbone", False) and not payload.vision.get(
        "pretrained", True
    ):
        warnings.append(
            {
                "path": "model.vision.freeze_backbone",
                "message": "当前配置会冻结一个未加载预训练的视觉 backbone，通常不利于训练收敛。",
                "error_code": "M102",
                "suggestion": "更常见的组合是 pretrained=true 或 freeze_backbone=false",
            }
        )

    if payload.fusion.get("fusion_type") == "concatenate":
        recommended_hidden_dim = int(payload.vision.get("feature_dim", 128)) + int(
            payload.tabular.get("output_dim", 16)
        )
        if int(payload.fusion.get("hidden_dim", recommended_hidden_dim)) != recommended_hidden_dim:
            warnings.append(
                {
                    "path": "model.fusion.hidden_dim",
                    "message": f"concatenate 常见起点是 vision_dim + tabular_dim = {recommended_hidden_dim}。",
                    "error_code": "M103",
                    "suggestion": f"可先把 fusion.hidden_dim 设为 {recommended_hidden_dim}",
                }
            )

    if runtime_summary and runtime_summary["total_params"] > 50_000_000:
        warnings.append(
            {
                "path": "model.parameters",
                "message": "当前模型参数量较大，建议确认是否符合你的显存预算。",
                "error_code": "M104",
                "suggestion": "可优先减小 feature_dim、tabular.output_dim 或更换更轻量 backbone",
            }
        )

    status = "ready" if not errors else "blocked"
    if status == "ready" and warnings:
        status = "warning"

    return {
        "status": status,
        "can_enter_training": status != "blocked",
        "issues": {
            "errors": errors,
            "warnings": warnings,
        },
        "checks": checks,
        "summary": {
            "num_classes": config.model.num_classes,
            "backbone": config.model.vision.backbone,
            "attention_type": config.model.vision.attention_type,
            "fusion_type": config.model.fusion.fusion_type,
            "tabular_feature_count": tabular_feature_count,
            "recommended_fusion_hidden_dim": config.model.vision.feature_dim
            + config.model.tabular.output_dim,
        },
        "runtime": runtime_summary,
        "config_contract": asdict(config.model),
        "next_step": (
            "模型骨架已可进入训练主线"
            if status != "blocked"
            else "先修复模型结构或特征输入问题，再进入训练"
        ),
    }
