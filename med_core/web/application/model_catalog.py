"""Structured catalog for packaged model-building modules."""

from __future__ import annotations

from typing import Any


MODEL_CATALOG_ADVANCED_BUILDER_FAMILY_PROJECTION: dict[str, dict[str, str]] = {
    "data_bundle": {
        "advanced_family": "data_input",
        "label": "数据输入",
    },
    "vision_encoder": {
        "advanced_family": "vision_backbone",
        "label": "视觉 backbone",
    },
    "tabular_encoder": {
        "advanced_family": "tabular_encoder",
        "label": "表格编码器",
    },
    "fusion_bundle": {
        "advanced_family": "fusion",
        "label": "融合层",
    },
    "task_head": {
        "advanced_family": "head",
        "label": "任务头",
    },
    "training_strategy": {
        "advanced_family": "training_strategy",
        "label": "训练策略",
    },
}

MODEL_CATALOG_ADVANCED_BUILDER_STATUS_LABELS: dict[str, str] = {
    "compile_ready": "可编译",
    "conditional": "有条件开放",
    "draft_only": "仅草稿",
}

MODEL_CATALOG_ADVANCED_BUILDER_DEFAULT_PRESET = "quickstart"

MODEL_CATALOG_ADVANCED_BUILDER_PRESET_RULES: list[dict[str, Any]] = [
    {
        "preset": "showcase",
        "priority": 20,
        "match_any_components": [
            "attention_backbone_bundle",
            "attention_fusion",
        ],
        "description": "包含 attention 路径时，默认更接近结果审查 / 可解释性路线。",
    },
    {
        "preset": "clinical",
        "priority": 10,
        "match_any_components": [
            "efficientnet_b0_backbone",
            "gated_fusion",
            "progressive_training",
        ],
        "description": "包含更稳健 backbone / fusion / 分阶段训练时，默认更接近临床基线路线。",
    },
]

MODEL_CATALOG_ADVANCED_BUILDER_REQUIRED_FAMILIES: list[str] = [
    "data_input",
    "vision_backbone",
    "tabular_encoder",
    "fusion",
    "head",
    "training_strategy",
]

MODEL_CATALOG_ADVANCED_BUILDER_CONNECTION_RULES: list[dict[str, str]] = [
    {
        "from_family": "data_input",
        "to_family": "vision_backbone",
        "status": "required",
        "description": "只要选择图像模态，必须先接一个视觉 backbone 才能继续编译。",
    },
    {
        "from_family": "data_input",
        "to_family": "tabular_encoder",
        "status": "required",
        "description": "图像 + 表格主链要求至少一条表格编码分支。",
    },
    {
        "from_family": "vision_backbone",
        "to_family": "fusion",
        "status": "required",
        "description": "视觉特征必须经过正式版支持的融合层才能进入任务头。",
    },
    {
        "from_family": "tabular_encoder",
        "to_family": "fusion",
        "status": "required",
        "description": "当前正式版多模态主链要求把表格特征也接入融合层。",
    },
    {
        "from_family": "fusion",
        "to_family": "head",
        "status": "required",
        "description": "融合层输出必须进入任务头，才能定义最终训练目标。",
    },
    {
        "from_family": "head",
        "to_family": "training_strategy",
        "status": "required",
        "description": "任务头之后必须接训练策略，图才进入可执行主链。",
    },
    {
        "from_family": "data_input",
        "to_family": "head",
        "status": "blocked",
        "description": "不允许跳过 backbone / fusion 直接把原始输入接到任务头。",
    },
    {
        "from_family": "vision_backbone",
        "to_family": "training_strategy",
        "status": "blocked",
        "description": "不允许跳过融合层和任务头直接进入训练策略。",
    },
]

MODEL_CATALOG_ADVANCED_COMPONENT_CONTRACTS: dict[str, dict[str, Any]] = {
    "image_tabular_dataset": {
        "preset_hints": ["quickstart", "clinical", "showcase"],
        "compile_boundary": "default_mainline",
        "compile_notes": [
            "这是当前正式版默认数据输入单元，会直接映射到 image_tabular 主链。",
        ],
        "patch_target_hints": [
            {
                "path": "data.*",
                "mode": "set",
                "description": "会设置 csv/image_dir/列名与特征列等数据输入字段。",
            }
        ],
        "warning_metadata": [],
    },
    "resnet18_backbone": {
        "preset_hints": ["quickstart", "showcase"],
        "compile_boundary": "default_mainline",
        "compile_notes": [
            "更适合作为 quickstart 或 attention 审查路径的轻量 backbone。",
        ],
        "patch_target_hints": [
            {
                "path": "model.vision.backbone",
                "mode": "set",
                "description": "直接决定视觉 backbone。",
            },
            {
                "path": "model.vision.featureDim",
                "mode": "set",
                "description": "会同步设置默认视觉特征维度。",
            },
        ],
        "warning_metadata": [],
    },
    "efficientnet_b0_backbone": {
        "preset_hints": ["clinical"],
        "compile_boundary": "default_mainline",
        "compile_notes": [
            "更适合作为稳健研究基线，而不是最低门槛 smoke 路径。",
        ],
        "patch_target_hints": [
            {
                "path": "model.vision.backbone",
                "mode": "set",
                "description": "直接决定视觉 backbone。",
            },
            {
                "path": "model.vision.featureDim",
                "mode": "set",
                "description": "会同步设置更高的默认视觉特征维度。",
            },
        ],
        "warning_metadata": [],
    },
    "attention_backbone_bundle": {
        "preset_hints": ["showcase"],
        "compile_boundary": "conditional_attention_path",
        "compile_notes": [
            "会默认走 CBAM + attention supervision 条件路径。",
        ],
        "patch_target_hints": [
            {
                "path": "model.vision.attentionType",
                "mode": "set",
                "description": "会把注意力路径固定到 CBAM。",
            },
            {
                "path": "training.useAttentionSupervision",
                "mode": "toggle",
                "description": "会打开 attention supervision 路径。",
            },
        ],
        "warning_metadata": [
            {
                "code": "ABG-W001",
                "path": "model.vision",
                "message": "当前图使用了 attention-supervised backbone，编译结果会默认走 CBAM + attention supervision 条件路径。",
                "suggestion": "确认这是预期路径；如需更稳妥的默认链，改用 ResNet18 或 EfficientNet-B0 backbone。",
            }
        ],
    },
    "mlp_tabular_encoder": {
        "preset_hints": ["quickstart", "clinical", "showcase"],
        "compile_boundary": "default_mainline",
        "compile_notes": [
            "当前正式版默认表格编码分支。",
        ],
        "patch_target_hints": [
            {
                "path": "model.tabular.*",
                "mode": "set",
                "description": "会设置 hidden_dims / output_dim / dropout 等表格编码字段。",
            }
        ],
        "warning_metadata": [],
    },
    "concatenate_fusion": {
        "preset_hints": ["quickstart"],
        "compile_boundary": "default_mainline",
        "compile_notes": [
            "编译时会把 fusion hidden dim 对齐到 vision feature dim + tabular output dim。",
        ],
        "patch_target_hints": [
            {
                "path": "model.fusion.fusionType",
                "mode": "set",
                "description": "会把 fusion 类型设为 concatenate。",
            },
            {
                "path": "model.fusion.hiddenDim",
                "mode": "derived",
                "description": "会根据 vision feature dim + tabular output dim 派生 hidden dim。",
            },
        ],
        "warning_metadata": [],
    },
    "gated_fusion": {
        "preset_hints": ["clinical"],
        "compile_boundary": "default_mainline",
        "compile_notes": [
            "更接近稳健研究基线，而不是最轻量起步路径。",
        ],
        "patch_target_hints": [
            {
                "path": "model.fusion.*",
                "mode": "set",
                "description": "会设置 gated fusion 的类型、hidden dim 与 dropout。",
            }
        ],
        "warning_metadata": [],
    },
    "attention_fusion": {
        "preset_hints": ["showcase"],
        "compile_boundary": "conditional_attention_path",
        "compile_notes": [
            "会保留 attention 路径，但仍受正式版当前 fusion schema 约束。",
        ],
        "patch_target_hints": [
            {
                "path": "model.fusion.*",
                "mode": "set",
                "description": "会设置 attention fusion 的类型、hidden dim、num_heads 与 dropout。",
            }
        ],
        "warning_metadata": [
            {
                "code": "ABG-W002",
                "path": "model.fusion",
                "message": "当前图使用了 attention fusion，编译结果会保留注意力路径，但仍受正式版主链的现有 fusion schema 约束。",
                "suggestion": "如果只需要最稳主链，可改用 concatenate fusion 或 gated fusion。",
            }
        ],
    },
    "classification_head": {
        "preset_hints": ["quickstart", "clinical", "showcase"],
        "compile_boundary": "default_mainline",
        "compile_notes": [
            "当前正式版默认任务头，只承诺分类主链。",
        ],
        "patch_target_hints": [
            {
                "path": "model.numClasses",
                "mode": "set",
                "description": "会决定分类任务的输出类别数。",
            },
            {
                "path": "model.useAuxiliaryHeads",
                "mode": "toggle",
                "description": "会决定是否启用辅助头。",
            },
        ],
        "warning_metadata": [],
    },
    "standard_training": {
        "preset_hints": ["quickstart", "showcase"],
        "compile_boundary": "default_mainline",
        "compile_notes": [
            "当前最稳的正式版训练路径。",
        ],
        "patch_target_hints": [
            {
                "path": "training.useProgressiveTraining",
                "mode": "toggle",
                "description": "会关闭 progressive training。",
            }
        ],
        "warning_metadata": [],
    },
    "progressive_training": {
        "preset_hints": ["clinical"],
        "compile_boundary": "conditional_stage_sum",
        "compile_notes": [
            "要求 stage1 + stage2 + stage3 == num_epochs。",
        ],
        "patch_target_hints": [
            {
                "path": "training.useProgressiveTraining",
                "mode": "toggle",
                "description": "会打开 progressive training。",
            },
            {
                "path": "training.stage*",
                "mode": "set",
                "description": "会设置 stage1/stage2/stage3 的默认轮次。",
            },
        ],
        "warning_metadata": [],
    },
}

MODEL_CATALOG_ADVANCED_TEMPLATE_CONTRACTS: dict[str, dict[str, Any]] = {
    "quickstart_multimodal": {
        "recommended_preset": "quickstart",
        "compile_boundary": "default_mainline",
        "compile_notes": [
            "当前最适合作为正式版默认起步骨架。",
        ],
        "patch_target_hints": [
            {
                "path": "data.* / model.* / training.*",
                "mode": "seed",
                "description": "会以 quickstart 路线初始化整套默认配置。",
            }
        ],
    },
    "clinical_gated_baseline": {
        "recommended_preset": "clinical",
        "compile_boundary": "default_mainline",
        "compile_notes": [
            "更适合真实数据集上的稳健研究基线。",
        ],
        "patch_target_hints": [
            {
                "path": "model.vision.* / model.fusion.* / training.stage*",
                "mode": "seed",
                "description": "会以 clinical 路线初始化 backbone、fusion 和训练节奏。",
            }
        ],
    },
    "attention_audit_path": {
        "recommended_preset": "showcase",
        "compile_boundary": "conditional_attention_path",
        "compile_notes": [
            "更偏结果审查和可解释性路径，不建议替代默认起步模板。",
        ],
        "patch_target_hints": [
            {
                "path": "model.vision.attentionType / training.useAttentionSupervision / model.fusion.*",
                "mode": "seed",
                "description": "会以 showcase 路线初始化注意力相关配置。",
            }
        ],
    },
}


MODEL_CATALOG_COMPONENTS: list[dict[str, Any]] = [
    {
        "id": "image_tabular_input_bundle",
        "source": "official",
        "label": "图像 + 表格输入包",
        "family": "data_bundle",
        "status": "compile_ready",
        "description": "正式版当前默认数据包，假定你已经准备好 CSV、图像目录和至少一个表格特征。",
        "data_requirements": [
            "CSV manifest",
            "image_path_column",
            "target_column",
            "at least 1 tabular feature",
        ],
        "config_requirements": [
            "data.csv_path",
            "data.image_dir",
            "data.image_path_column",
            "data.target_column",
            "data.numerical_features or data.categorical_features",
        ],
        "compute_profile": {
            "tier": "light",
            "gpu_vram_hint": "8GB+",
            "notes": "主要取决于后续 backbone 和 batch_size。",
        },
        "upstream": [],
        "outputs": ["image_data", "tabular_data", "dataset_split"],
        "advanced_builder_component_id": "image_tabular_dataset",
        "wizard_prefill": {
            "csvPath": "data/mock/metadata.csv",
            "imageDir": "data/mock",
            "imagePathColumn": "image_path",
            "targetColumn": "diagnosis",
            "patientIdColumn": "",
            "numericalFeatures": ["age"],
            "categoricalFeatures": ["gender"],
        },
    },
    {
        "id": "three_phase_ct_input_bundle",
        "source": "official",
        "label": "三相 CT + 临床输入包",
        "family": "data_bundle",
        "status": "draft_only",
        "description": "专项 runtime 已存在，但目前不进入正式版默认模型数据库主叙事。",
        "data_requirements": [
            "three-phase manifest CSV",
            "phase_dir_columns",
            "clinical_feature_columns",
            "target_shape",
        ],
        "config_requirements": [
            "data.dataset_type=three_phase_ct_tabular",
            "data.phase_dir_columns",
            "data.clinical_feature_columns",
            "data.target_shape",
        ],
        "compute_profile": {
            "tier": "heavy",
            "gpu_vram_hint": "24GB+",
            "notes": "3D 输入对显存和 IO 压力明显更高。",
        },
        "upstream": [],
        "outputs": ["phase_volume", "clinical_features"],
        "advanced_builder_component_id": "three_phase_ct_dataset",
    },
    {
        "id": "resnet18_encoder_bundle",
        "source": "official",
        "label": "ResNet18 特征提取包",
        "family": "vision_encoder",
        "status": "compile_ready",
        "description": "轻量视觉 backbone，适合 quickstart 和低门槛实验。",
        "data_requirements": ["2D image tensors"],
        "config_requirements": [
            "model.vision.backbone=resnet18",
            "model.vision.feature_dim",
            "model.vision.attention_type",
        ],
        "compute_profile": {
            "tier": "light",
            "gpu_vram_hint": "8GB+",
            "notes": "优先用于快速验证、CI smoke 和低成本迭代。",
        },
        "upstream": ["data_bundle"],
        "outputs": ["vision_features"],
        "advanced_builder_component_id": "resnet18_backbone",
        "wizard_prefill": {
            "backbone": "resnet18",
            "featureDim": 128,
            "attentionType": "cbam",
            "pretrained": True,
            "freezeBackbone": False,
        },
    },
    {
        "id": "efficientnet_b0_encoder_bundle",
        "source": "official",
        "label": "EfficientNet-B0 特征提取包",
        "family": "vision_encoder",
        "status": "compile_ready",
        "description": "更稳健的中等复杂度视觉编码器，适合正式版稳健基线。",
        "data_requirements": ["2D image tensors"],
        "config_requirements": [
            "model.vision.backbone=efficientnet_b0",
            "model.vision.feature_dim",
        ],
        "compute_profile": {
            "tier": "medium",
            "gpu_vram_hint": "12GB+",
            "notes": "较 ResNet18 略重，但通常更适合真实数据集基线。",
        },
        "upstream": ["data_bundle"],
        "outputs": ["vision_features"],
        "advanced_builder_component_id": "efficientnet_b0_backbone",
        "wizard_prefill": {
            "backbone": "efficientnet_b0",
            "featureDim": 192,
            "attentionType": "cbam",
            "pretrained": True,
            "freezeBackbone": False,
        },
    },
    {
        "id": "attention_cbam_encoder_bundle",
        "source": "official",
        "label": "CBAM 注意力视觉编码包",
        "family": "vision_encoder",
        "status": "conditional",
        "description": "适合注意力可视化和审查，但只在 CBAM 路径下开放。",
        "data_requirements": ["2D image tensors"],
        "config_requirements": [
            "model.vision.attention_type=cbam",
            "training.use_attention_supervision (optional)",
        ],
        "compute_profile": {
            "tier": "medium",
            "gpu_vram_hint": "12GB+",
            "notes": "比轻量 backbone 多一层注意力路径与解释成本。",
        },
        "upstream": ["data_bundle"],
        "outputs": ["vision_features", "attention_maps"],
        "advanced_builder_component_id": "attention_backbone_bundle",
        "wizard_prefill": {
            "backbone": "resnet18",
            "featureDim": 128,
            "attentionType": "cbam",
            "pretrained": True,
            "freezeBackbone": False,
            "useAttentionSupervision": True,
        },
    },
    {
        "id": "mlp_tabular_encoder_bundle",
        "source": "official",
        "label": "MLP 表格编码包",
        "family": "tabular_encoder",
        "status": "compile_ready",
        "description": "当前正式版默认表格特征编码器。",
        "data_requirements": ["at least 1 tabular feature"],
        "config_requirements": [
            "model.tabular.hidden_dims",
            "model.tabular.output_dim",
        ],
        "compute_profile": {
            "tier": "light",
            "gpu_vram_hint": "negligible",
            "notes": "参数量小，主要影响融合前特征表达。",
        },
        "upstream": ["data_bundle"],
        "outputs": ["tabular_features"],
        "advanced_builder_component_id": "mlp_tabular_encoder",
        "wizard_prefill": {
            "tabularHiddenDims": [32],
            "tabularOutputDim": 16,
            "tabularDropout": 0.2,
        },
    },
    {
        "id": "concatenate_fusion_bundle",
        "source": "official",
        "label": "Concatenate 融合包",
        "family": "fusion_bundle",
        "status": "compile_ready",
        "description": "最稳的正式版起步融合方式，适合作为默认黑盒拼装单元。",
        "data_requirements": ["vision_features", "tabular_features"],
        "config_requirements": [
            "model.fusion.fusion_type=concatenate",
            "model.fusion.hidden_dim ~= vision_dim + tabular_dim",
        ],
        "compute_profile": {
            "tier": "light",
            "gpu_vram_hint": "8GB+",
            "notes": "适合作为 quickstart 和主线默认融合层。",
        },
        "upstream": ["vision_encoder", "tabular_encoder"],
        "outputs": ["fused_features"],
        "advanced_builder_component_id": "concatenate_fusion",
        "wizard_prefill": {
            "fusionType": "concatenate",
            "fusionHiddenDim": 144,
            "fusionDropout": 0.3,
            "fusionNumHeads": 4,
        },
    },
    {
        "id": "gated_fusion_bundle",
        "source": "official",
        "label": "Gated 融合包",
        "family": "fusion_bundle",
        "status": "compile_ready",
        "description": "更稳健的研究基线融合方式，适合正式版基线模型库。",
        "data_requirements": ["vision_features", "tabular_features"],
        "config_requirements": [
            "model.fusion.fusion_type=gated",
            "model.fusion.hidden_dim",
        ],
        "compute_profile": {
            "tier": "medium",
            "gpu_vram_hint": "12GB+",
            "notes": "相较 concatenate 略复杂，但通常更适合真实研究迭代。",
        },
        "upstream": ["vision_encoder", "tabular_encoder"],
        "outputs": ["fused_features"],
        "advanced_builder_component_id": "gated_fusion",
        "wizard_prefill": {
            "fusionType": "gated",
            "fusionHiddenDim": 160,
            "fusionDropout": 0.3,
            "fusionNumHeads": 4,
        },
    },
    {
        "id": "attention_fusion_bundle",
        "source": "official",
        "label": "Attention 融合包",
        "family": "fusion_bundle",
        "status": "conditional",
        "description": "用于结果强化和注意力审查，但不作为默认起步包。",
        "data_requirements": ["vision_features", "tabular_features"],
        "config_requirements": [
            "model.fusion.fusion_type=attention",
            "model.fusion.num_heads",
        ],
        "compute_profile": {
            "tier": "medium",
            "gpu_vram_hint": "12GB+",
            "notes": "注意力融合更适合解释性和审查，不适合作为最低门槛路径。",
        },
        "upstream": ["vision_encoder", "tabular_encoder"],
        "outputs": ["fused_features", "attention_weights"],
        "advanced_builder_component_id": "attention_fusion",
        "wizard_prefill": {
            "fusionType": "attention",
            "fusionHiddenDim": 144,
            "fusionDropout": 0.3,
            "fusionNumHeads": 4,
        },
    },
    {
        "id": "cross_attention_fusion_bundle",
        "source": "official",
        "label": "Cross Attention 融合包",
        "family": "fusion_bundle",
        "status": "draft_only",
        "description": "保留为官方草稿单元，但当前不进入默认正式版主线。",
        "data_requirements": ["vision_features", "tabular_features"],
        "config_requirements": [
            "model.fusion.fusion_type=cross_attention",
            "model.fusion.num_heads",
        ],
        "compute_profile": {
            "tier": "medium",
            "gpu_vram_hint": "12GB+",
            "notes": "目前只作为草稿能力保留，不建议默认使用。",
        },
        "upstream": ["vision_encoder", "tabular_encoder"],
        "outputs": ["fused_features"],
        "advanced_builder_component_id": "cross_attention_fusion",
        "wizard_prefill": {
            "fusionType": "cross_attention",
            "fusionHiddenDim": 160,
            "fusionDropout": 0.3,
            "fusionNumHeads": 4,
        },
    },
    {
        "id": "classification_head_bundle",
        "source": "official",
        "label": "分类头包",
        "family": "task_head",
        "status": "compile_ready",
        "description": "当前正式版默认任务头，只承诺分类主链。",
        "data_requirements": ["fused_features"],
        "config_requirements": ["model.num_classes >= 2"],
        "compute_profile": {
            "tier": "light",
            "gpu_vram_hint": "negligible",
            "notes": "头部成本低，主要受 fusion 输出维度影响。",
        },
        "upstream": ["fusion_bundle"],
        "outputs": ["logits"],
        "advanced_builder_component_id": "classification_head",
        "wizard_prefill": {
            "numClasses": 2,
            "useAuxiliaryHeads": True,
        },
    },
    {
        "id": "survival_head_bundle",
        "source": "official",
        "label": "生存任务头包",
        "family": "task_head",
        "status": "draft_only",
        "description": "保留为官方草稿任务头，但当前不作为默认主线承诺。",
        "data_requirements": ["fused_features", "survival labels"],
        "config_requirements": ["生存分析输出定义", "风险头配置"],
        "compute_profile": {
            "tier": "light",
            "gpu_vram_hint": "negligible",
            "notes": "当前只保留草稿位，不进入默认正式版说明。",
        },
        "upstream": ["fusion_bundle"],
        "outputs": ["risk_score"],
        "advanced_builder_component_id": "survival_head",
        "wizard_prefill": {},
    },
    {
        "id": "standard_training_bundle",
        "source": "official",
        "label": "标准训练策略包",
        "family": "training_strategy",
        "status": "compile_ready",
        "description": "当前最稳定的正式版训练策略。",
        "data_requirements": ["dataset_split", "task head output"],
        "config_requirements": ["training.use_progressive_training=false"],
        "compute_profile": {
            "tier": "light",
            "gpu_vram_hint": "depends on model",
            "notes": "默认策略，适合正式版主线。",
        },
        "upstream": ["task_head"],
        "outputs": ["checkpoint", "history", "validation"],
        "advanced_builder_component_id": "standard_training",
        "wizard_prefill": {
            "useProgressiveTraining": False,
            "useAttentionSupervision": False,
        },
    },
    {
        "id": "progressive_training_bundle",
        "source": "official",
        "label": "分阶段训练策略包",
        "family": "training_strategy",
        "status": "conditional",
        "description": "允许分阶段训练，但需要满足 stage epoch 总和约束。",
        "data_requirements": ["dataset_split", "task head output"],
        "config_requirements": [
            "training.use_progressive_training=true",
            "stage1+stage2+stage3 == num_epochs",
        ],
        "compute_profile": {
            "tier": "medium",
            "gpu_vram_hint": "depends on model",
            "notes": "适合更稳健的基线训练，但配置更复杂。",
        },
        "upstream": ["task_head"],
        "outputs": ["checkpoint", "history", "validation"],
        "advanced_builder_component_id": "progressive_training",
        "wizard_prefill": {
            "useProgressiveTraining": True,
            "numEpochs": 18,
            "stage1Epochs": 6,
            "stage2Epochs": 8,
            "stage3Epochs": 4,
        },
    },
]


MODEL_CATALOG_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "quickstart_multimodal",
        "source": "official",
        "label": "Quickstart 多模态模板",
        "status": "compile_ready",
        "description": "图像 + 表格输入，ResNet18 + MLP + Concatenate + 分类头。",
        "component_ids": [
            "image_tabular_input_bundle",
            "resnet18_encoder_bundle",
            "mlp_tabular_encoder_bundle",
            "concatenate_fusion_bundle",
            "classification_head_bundle",
            "standard_training_bundle",
        ],
        "unit_map": {
            "vision_encoder": "resnet18_encoder_bundle",
            "tabular_encoder": "mlp_tabular_encoder_bundle",
            "fusion_bundle": "concatenate_fusion_bundle",
            "task_head": "classification_head_bundle",
            "training_strategy": "standard_training_bundle",
        },
        "editable_slots": [
            "vision_encoder",
            "tabular_encoder",
            "fusion_bundle",
            "task_head",
            "training_strategy",
        ],
        "data_requirements": [
            "CSV + image_dir",
            "至少 1 个表格特征",
            "二分类或多分类标签列",
        ],
        "compute_profile": {
            "tier": "light",
            "gpu_vram_hint": "8GB+",
            "notes": "适合第一条正式版主线和本机 smoke。",
        },
        "advanced_builder_blueprint_id": "quickstart_multimodal",
        "wizard_prefill": {
            "numClasses": 2,
            "useAuxiliaryHeads": True,
            "backbone": "resnet18",
            "attentionType": "cbam",
            "featureDim": 128,
            "pretrained": True,
            "freezeBackbone": False,
            "tabularHiddenDims": [32],
            "tabularOutputDim": 16,
            "tabularDropout": 0.2,
            "fusionType": "concatenate",
            "fusionHiddenDim": 144,
            "fusionDropout": 0.3,
            "fusionNumHeads": 4,
            "useAttentionSupervision": False,
        },
    },
    {
        "id": "clinical_gated_baseline",
        "source": "official",
        "label": "稳健研究基线模板",
        "status": "compile_ready",
        "description": "图像 + 表格输入，EfficientNet-B0 + MLP + Gated 融合 + 分类头。",
        "component_ids": [
            "image_tabular_input_bundle",
            "efficientnet_b0_encoder_bundle",
            "mlp_tabular_encoder_bundle",
            "gated_fusion_bundle",
            "classification_head_bundle",
            "progressive_training_bundle",
        ],
        "unit_map": {
            "vision_encoder": "efficientnet_b0_encoder_bundle",
            "tabular_encoder": "mlp_tabular_encoder_bundle",
            "fusion_bundle": "gated_fusion_bundle",
            "task_head": "classification_head_bundle",
            "training_strategy": "progressive_training_bundle",
        },
        "editable_slots": [
            "vision_encoder",
            "tabular_encoder",
            "fusion_bundle",
            "task_head",
            "training_strategy",
        ],
        "data_requirements": [
            "CSV + image_dir",
            "至少 1 个表格特征",
            "更适合样本规模稍大的真实数据集",
        ],
        "compute_profile": {
            "tier": "medium",
            "gpu_vram_hint": "12GB+",
            "notes": "推荐作为对外正式版基线，而不是最轻 smoke 路径。",
        },
        "advanced_builder_blueprint_id": "clinical_gated_baseline",
        "wizard_prefill": {
            "numClasses": 2,
            "useAuxiliaryHeads": True,
            "backbone": "efficientnet_b0",
            "attentionType": "cbam",
            "featureDim": 192,
            "pretrained": True,
            "freezeBackbone": False,
            "tabularHiddenDims": [64, 32],
            "tabularOutputDim": 32,
            "tabularDropout": 0.2,
            "fusionType": "gated",
            "fusionHiddenDim": 160,
            "fusionDropout": 0.3,
            "fusionNumHeads": 4,
            "useAttentionSupervision": False,
        },
    },
    {
        "id": "attention_audit_path",
        "source": "official",
        "label": "注意力审查模板",
        "status": "conditional",
        "description": "用于结果解释和注意力可视化审查，不作为默认起步模板。",
        "component_ids": [
            "image_tabular_input_bundle",
            "attention_cbam_encoder_bundle",
            "mlp_tabular_encoder_bundle",
            "attention_fusion_bundle",
            "classification_head_bundle",
            "standard_training_bundle",
        ],
        "unit_map": {
            "vision_encoder": "attention_cbam_encoder_bundle",
            "tabular_encoder": "mlp_tabular_encoder_bundle",
            "fusion_bundle": "attention_fusion_bundle",
            "task_head": "classification_head_bundle",
            "training_strategy": "standard_training_bundle",
        },
        "editable_slots": [
            "vision_encoder",
            "tabular_encoder",
            "fusion_bundle",
            "task_head",
            "training_strategy",
        ],
        "data_requirements": [
            "CSV + image_dir",
            "至少 1 个表格特征",
            "更适合结果分析而不是最低门槛首跑",
        ],
        "compute_profile": {
            "tier": "medium",
            "gpu_vram_hint": "12GB+",
            "notes": "适合 attention heatmap / 结果解释，不建议替代默认模板。",
        },
        "advanced_builder_blueprint_id": "attention_audit_path",
        "wizard_prefill": {
            "numClasses": 2,
            "useAuxiliaryHeads": True,
            "backbone": "resnet18",
            "attentionType": "cbam",
            "featureDim": 128,
            "pretrained": True,
            "freezeBackbone": False,
            "tabularHiddenDims": [32],
            "tabularOutputDim": 16,
            "tabularDropout": 0.2,
            "fusionType": "attention",
            "fusionHiddenDim": 144,
            "fusionDropout": 0.3,
            "fusionNumHeads": 4,
            "useAttentionSupervision": True,
        },
    },
]


def export_advanced_builder_contract() -> dict[str, Any]:
    family_projection = {
        family: dict(metadata)
        for family, metadata in MODEL_CATALOG_ADVANCED_BUILDER_FAMILY_PROJECTION.items()
    }
    family_labels = {
        metadata["advanced_family"]: metadata["label"]
        for metadata in family_projection.values()
    }

    return {
        "family_projection": family_projection,
        "family_labels": family_labels,
        "status_labels": dict(MODEL_CATALOG_ADVANCED_BUILDER_STATUS_LABELS),
        "required_families": list(MODEL_CATALOG_ADVANCED_BUILDER_REQUIRED_FAMILIES),
        "default_preset": MODEL_CATALOG_ADVANCED_BUILDER_DEFAULT_PRESET,
        "preset_rules": [
            dict(rule) for rule in MODEL_CATALOG_ADVANCED_BUILDER_PRESET_RULES
        ],
        "connection_rules": [
            dict(rule) for rule in MODEL_CATALOG_ADVANCED_BUILDER_CONNECTION_RULES
        ],
    }


def _unit_with_advanced_builder_contract(unit: dict[str, Any]) -> dict[str, Any]:
    advanced_component_id = unit.get("advanced_builder_component_id")
    if not advanced_component_id:
        return dict(unit)
    return {
        **unit,
        "advanced_builder_contract": dict(
            MODEL_CATALOG_ADVANCED_COMPONENT_CONTRACTS.get(
                str(advanced_component_id),
                {
                    "preset_hints": [],
                    "compile_boundary": "unspecified",
                    "compile_notes": [],
                    "patch_target_hints": [],
                    "warning_metadata": [],
                },
            )
        ),
    }


def _model_with_advanced_builder_contract(model: dict[str, Any]) -> dict[str, Any]:
    advanced_blueprint_id = model.get("advanced_builder_blueprint_id")
    if not advanced_blueprint_id:
        return dict(model)
    return {
        **model,
        "advanced_builder_contract": dict(
            MODEL_CATALOG_ADVANCED_TEMPLATE_CONTRACTS.get(
                str(advanced_blueprint_id),
                {
                    "recommended_preset": MODEL_CATALOG_ADVANCED_BUILDER_DEFAULT_PRESET,
                    "compile_boundary": "unspecified",
                    "compile_notes": [],
                    "patch_target_hints": [],
                },
            )
        ),
    }


def export_model_catalog() -> dict[str, Any]:
    units = [_unit_with_advanced_builder_contract(unit) for unit in MODEL_CATALOG_COMPONENTS]
    models = [_model_with_advanced_builder_contract(model) for model in MODEL_CATALOG_TEMPLATES]
    return {
        "sources": {
            "official": {
                "enabled": True,
                "label": "官方模型来源",
                "description": "当前默认只开放官方打包组件和模板，确保能回落到现有 runtime 主线。",
                "entry_path": "/config/model",
            },
            "custom": {
                "enabled": True,
                "label": "用户自定义来源",
                "description": "当前允许基于官方最小组织单元组合本地自定义模型模板，但不开放服务端共享和审核流。",
                "entry_path": "/config/model/custom",
            },
        },
        "principles": [
            "不暴露 pooling、FC 等过底层神经网络部件",
            "只暴露已经打包好的黑盒组件，供用户连接和组合",
            "每个组件都必须说明数据要求、配置要求和算力要求",
            "模板优先服务正式版主链，而不是任意 builder 幻觉",
        ],
        "advanced_builder": export_advanced_builder_contract(),
        "units": units,
        "models": models,
        "components": units,
        "templates": models,
    }
