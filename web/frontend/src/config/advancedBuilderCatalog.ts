export type AdvancedBuilderFamily =
  | "data_input"
  | "vision_backbone"
  | "tabular_encoder"
  | "fusion"
  | "head"
  | "training_strategy";

export type AdvancedBuilderStatus =
  | "compile_ready"
  | "conditional"
  | "draft_only";

export interface AdvancedBuilderComponent {
  id: string;
  family: AdvancedBuilderFamily;
  label: string;
  status: AdvancedBuilderStatus;
  description: string;
  schemaPath?: string;
  inputs?: string[];
  outputs?: string[];
  notes?: string[];
}

export interface AdvancedBuilderConnectionRule {
  fromFamily: AdvancedBuilderFamily;
  toFamily: AdvancedBuilderFamily;
  status: "required" | "conditional" | "blocked";
  description: string;
}

export interface AdvancedBuilderBlueprint {
  id: string;
  label: string;
  status: "compile_ready" | "draft_only";
  description: string;
  components: string[];
  compilesTo?: string;
  blockers?: string[];
}

export const ADVANCED_BUILDER_FAMILY_LABELS: Record<
  AdvancedBuilderFamily,
  string
> = {
  data_input: "数据输入",
  vision_backbone: "视觉 backbone",
  tabular_encoder: "表格编码器",
  fusion: "融合层",
  head: "任务头",
  training_strategy: "训练策略",
};

export const ADVANCED_BUILDER_STATUS_LABELS: Record<
  AdvancedBuilderStatus,
  string
> = {
  compile_ready: "可编译",
  conditional: "有条件开放",
  draft_only: "仅草稿",
};

export const ADVANCED_BUILDER_COMPONENTS: AdvancedBuilderComponent[] = [
  {
    id: "image_tabular_dataset",
    family: "data_input",
    label: "图像 + 表格输入",
    status: "compile_ready",
    description: "当前正式版默认主链，映射到 image_tabular 训练 schema。",
    schemaPath: "data.dataset_type=image_tabular",
    outputs: ["image_data", "tabular_data", "dataset_split"],
  },
  {
    id: "three_phase_ct_dataset",
    family: "data_input",
    label: "三相 CT + 临床输入",
    status: "draft_only",
    description: "runtime 已有专项模型，但还没有进入正式版高级建模器的通用编译面。",
    schemaPath: "data.dataset_type=three_phase_ct_tabular",
    outputs: ["phase_volume", "clinical_features"],
  },
  {
    id: "resnet18_backbone",
    family: "vision_backbone",
    label: "ResNet18",
    status: "compile_ready",
    description: "默认轻量视觉 backbone，适合作为正式版起步骨架。",
    schemaPath: "model.vision.backbone=resnet18",
    inputs: ["image_data"],
    outputs: ["vision_features"],
  },
  {
    id: "efficientnet_b0_backbone",
    family: "vision_backbone",
    label: "EfficientNet-B0",
    status: "compile_ready",
    description: "更稳健的常规研究基线，适合中等规模正式版模板。",
    schemaPath: "model.vision.backbone=efficientnet_b0",
    inputs: ["image_data"],
    outputs: ["vision_features"],
  },
  {
    id: "attention_backbone_bundle",
    family: "vision_backbone",
    label: "Attention-supervised backbone",
    status: "conditional",
    description: "当前仅在 CBAM 注意力路径下可用，需要前台显式提示条件。",
    schemaPath: "model.vision.attention_type=cbam",
    inputs: ["image_data"],
    outputs: ["vision_features", "attention_maps"],
    notes: ["启用注意力监督时，当前正式版只允许 CBAM 路径。"],
  },
  {
    id: "mlp_tabular_encoder",
    family: "tabular_encoder",
    label: "MLP 表格编码器",
    status: "compile_ready",
    description: "当前正式版默认的表格分支编码器。",
    schemaPath: "model.tabular",
    inputs: ["tabular_data"],
    outputs: ["tabular_features"],
  },
  {
    id: "concatenate_fusion",
    family: "fusion",
    label: "Concatenate Fusion",
    status: "compile_ready",
    description: "最稳的正式版起步融合层，默认 quickstart 使用此路径。",
    schemaPath: "model.fusion.fusion_type=concatenate",
    inputs: ["vision_features", "tabular_features"],
    outputs: ["fused_features"],
  },
  {
    id: "gated_fusion",
    family: "fusion",
    label: "Gated Fusion",
    status: "compile_ready",
    description: "适合作为更稳健的研究基线融合策略。",
    schemaPath: "model.fusion.fusion_type=gated",
    inputs: ["vision_features", "tabular_features"],
    outputs: ["fused_features"],
  },
  {
    id: "attention_fusion",
    family: "fusion",
    label: "Attention Fusion",
    status: "conditional",
    description: "适合结果强化和注意力审查，但正式版应在骨架推荐后再开放。",
    schemaPath: "model.fusion.fusion_type=attention",
    inputs: ["vision_features", "tabular_features"],
    outputs: ["fused_features", "attention_weights"],
  },
  {
    id: "cross_attention_fusion",
    family: "fusion",
    label: "Cross Attention Fusion",
    status: "draft_only",
    description: "runtime 有探索性空间，但当前还不进入正式版高级模式的可编译默认集。",
    schemaPath: "model.fusion.fusion_type=cross_attention",
    inputs: ["vision_features", "tabular_features"],
    outputs: ["fused_features"],
  },
  {
    id: "classification_head",
    family: "head",
    label: "分类头",
    status: "compile_ready",
    description: "当前正式版主链默认任务头。",
    schemaPath: "model.num_classes",
    inputs: ["fused_features"],
    outputs: ["logits"],
  },
  {
    id: "survival_head",
    family: "head",
    label: "生存任务头",
    status: "draft_only",
    description: "仓库有相关能力，但还没有进入正式版高级建模器的主叙事。",
    inputs: ["fused_features"],
    outputs: ["risk_score"],
  },
  {
    id: "standard_training",
    family: "training_strategy",
    label: "标准训练",
    status: "compile_ready",
    description: "当前最稳定的正式版训练策略。",
    schemaPath: "training",
    inputs: ["dataset_split", "logits"],
    outputs: ["checkpoint", "history", "validation"],
  },
  {
    id: "progressive_training",
    family: "training_strategy",
    label: "分阶段训练",
    status: "conditional",
    description: "可以编译，但需要显式满足 stage epoch 总和约束。",
    schemaPath: "training.use_progressive_training=true",
    inputs: ["dataset_split", "logits"],
    outputs: ["checkpoint", "history", "validation"],
  },
];

export const ADVANCED_BUILDER_CONNECTION_RULES: AdvancedBuilderConnectionRule[] = [
  {
    fromFamily: "data_input",
    toFamily: "vision_backbone",
    status: "required",
    description: "只要选择图像模态，必须先接一个视觉 backbone 才能继续编译。",
  },
  {
    fromFamily: "data_input",
    toFamily: "tabular_encoder",
    status: "required",
    description: "图像 + 表格主链要求至少一条表格编码分支。",
  },
  {
    fromFamily: "vision_backbone",
    toFamily: "fusion",
    status: "required",
    description: "视觉特征必须经过正式版支持的融合层才能进入任务头。",
  },
  {
    fromFamily: "tabular_encoder",
    toFamily: "fusion",
    status: "required",
    description: "当前正式版多模态主链要求把表格特征也接入融合层。",
  },
  {
    fromFamily: "fusion",
    toFamily: "head",
    status: "required",
    description: "融合层输出必须进入任务头，才能定义最终训练目标。",
  },
  {
    fromFamily: "head",
    toFamily: "training_strategy",
    status: "required",
    description: "任务头之后必须接训练策略，图才进入可执行主链。",
  },
  {
    fromFamily: "data_input",
    toFamily: "head",
    status: "blocked",
    description: "不允许跳过 backbone / fusion 直接把原始输入接到任务头。",
  },
  {
    fromFamily: "vision_backbone",
    toFamily: "training_strategy",
    status: "blocked",
    description: "不允许跳过融合层和任务头直接进入训练策略。",
  },
];

export const ADVANCED_BUILDER_BLUEPRINTS: AdvancedBuilderBlueprint[] = [
  {
    id: "quickstart_multimodal",
    label: "正式版 quickstart 多模态骨架",
    status: "compile_ready",
    description: "图像 + 表格输入，ResNet18 + MLP + Concatenate + 分类头 + 标准训练。",
    components: [
      "image_tabular_dataset",
      "resnet18_backbone",
      "mlp_tabular_encoder",
      "concatenate_fusion",
      "classification_head",
      "standard_training",
    ],
    compilesTo: "ExperimentConfig / configs/starter/quickstart.yaml",
  },
  {
    id: "clinical_gated_baseline",
    label: "稳健研究基线骨架",
    status: "compile_ready",
    description: "图像 + 表格输入，EfficientNet-B0 + MLP + Gated Fusion + 分类头 + 分阶段训练。",
    components: [
      "image_tabular_dataset",
      "efficientnet_b0_backbone",
      "mlp_tabular_encoder",
      "gated_fusion",
      "classification_head",
      "progressive_training",
    ],
    compilesTo: "ExperimentConfig / formal release baseline preset",
  },
  {
    id: "attention_audit_path",
    label: "结果审查 / 注意力增强骨架",
    status: "compile_ready",
    description: "用于结果交付和注意力可视化审查的正式版骨架。",
    components: [
      "image_tabular_dataset",
      "attention_backbone_bundle",
      "mlp_tabular_encoder",
      "attention_fusion",
      "classification_head",
      "standard_training",
    ],
    compilesTo: "ExperimentConfig / result-audit preset",
  },
  {
    id: "three_phase_ct_builder",
    label: "三相 CT 通用高级建模骨架",
    status: "draft_only",
    description: "当前仍是专项 runtime 能力，还没有进入正式版高级建模器的通用编译层。",
    components: [
      "three_phase_ct_dataset",
      "gated_fusion",
      "classification_head",
      "standard_training",
    ],
    blockers: ["需要正式版图编译层先支持三相 CT 专项 schema。"],
  },
  {
    id: "survival_builder",
    label: "生存分析建模骨架",
    status: "draft_only",
    description: "当前不作为正式版默认高级模式开放。",
    components: [
      "image_tabular_dataset",
      "efficientnet_b0_backbone",
      "mlp_tabular_encoder",
      "gated_fusion",
      "survival_head",
      "standard_training",
    ],
    blockers: ["正式版主叙事当前先围绕分类主链，不把 survival 作为默认承诺。"],
  },
];
