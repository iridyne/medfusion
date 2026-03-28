export type RunPresetId = "quickstart" | "showcase" | "clinical";
export type DeviceType = "auto" | "cpu" | "cuda" | "mps";
export type VisionBackbone =
  | "resnet18"
  | "resnet34"
  | "resnet50"
  | "resnet101"
  | "mobilenetv2"
  | "mobilenetv3_small"
  | "mobilenetv3_large"
  | "efficientnet_b0"
  | "efficientnet_b1"
  | "efficientnet_b2"
  | "vit_b_16"
  | "vit_b_32"
  | "swin_t"
  | "swin_s";
export type AttentionType = "cbam" | "se" | "eca" | "none";
export type FusionType = "concatenate" | "gated" | "attention" | "cross_attention" | "bilinear";
export type OptimizerType = "adam" | "adamw" | "sgd";
export type SchedulerType = "cosine" | "step" | "plateau" | "onecycle" | "none";
export type AugmentationStrength = "light" | "medium" | "heavy";

export interface RunSpec {
  projectName: string;
  experimentName: string;
  description: string;
  tags: string[];
  seed: number;
  device: DeviceType;
  data: {
    csvPath: string;
    imageDir: string;
    imagePathColumn: string;
    targetColumn: string;
    patientIdColumn: string;
    numericalFeatures: string[];
    categoricalFeatures: string[];
    trainRatio: number;
    valRatio: number;
    testRatio: number;
    imageSize: number;
    batchSize: number;
    numWorkers: number;
    pinMemory: boolean;
    augmentationStrength: AugmentationStrength;
  };
  model: {
    numClasses: number;
    useAuxiliaryHeads: boolean;
    vision: {
      backbone: VisionBackbone;
      pretrained: boolean;
      freezeBackbone: boolean;
      featureDim: number;
      dropout: number;
      attentionType: AttentionType;
    };
    tabular: {
      hiddenDims: number[];
      outputDim: number;
      dropout: number;
    };
    fusion: {
      fusionType: FusionType;
      hiddenDim: number;
      dropout: number;
      numHeads: number;
    };
  };
  training: {
    numEpochs: number;
    mixedPrecision: boolean;
    gradientClip: number | null;
    useProgressiveTraining: boolean;
    stage1Epochs: number;
    stage2Epochs: number;
    stage3Epochs: number;
    useAttentionSupervision: boolean;
    attentionLossWeight: number;
    optimizer: {
      optimizer: OptimizerType;
      learningRate: number;
      weightDecay: number;
      momentum: number;
    };
    scheduler: {
      scheduler: SchedulerType;
      warmupEpochs: number;
      minLr: number;
      stepSize: number;
      gamma: number;
      patience: number;
      factor: number;
    };
  };
  logging: {
    outputDir: string;
    useTensorboard: boolean;
    useWandb: boolean;
  };
}

export interface RunSpecValidationIssue {
  level: "error" | "warning";
  message: string;
}

export const RUN_PRESET_OPTIONS: Array<{ id: RunPresetId; label: string; description: string }> = [
  {
    id: "quickstart",
    label: "快速验证",
    description: "最短路径跑通多模态训练，适合本地 smoke test。",
  },
  {
    id: "showcase",
    label: "结果增强",
    description: "强化结果分析与可视化产物，适合做结果回归与审查。",
  },
  {
    id: "clinical",
    label: "稳健基线",
    description: "偏稳健训练配置，适合后续扩展成真实问题求解基线。",
  },
];

export const VISION_BACKBONE_OPTIONS: VisionBackbone[] = [
  "resnet18",
  "resnet34",
  "resnet50",
  "resnet101",
  "mobilenetv2",
  "mobilenetv3_small",
  "mobilenetv3_large",
  "efficientnet_b0",
  "efficientnet_b1",
  "efficientnet_b2",
  "vit_b_16",
  "vit_b_32",
  "swin_t",
  "swin_s",
];

export const FUSION_TYPE_OPTIONS: FusionType[] = [
  "concatenate",
  "gated",
  "attention",
  "cross_attention",
  "bilinear",
];

export const ATTENTION_TYPE_OPTIONS: AttentionType[] = ["cbam", "se", "eca", "none"];
export const OPTIMIZER_OPTIONS: OptimizerType[] = ["adam", "adamw", "sgd"];
export const SCHEDULER_OPTIONS: SchedulerType[] = ["cosine", "step", "plateau", "onecycle", "none"];
export const DEVICE_OPTIONS: DeviceType[] = ["auto", "cuda", "cpu", "mps"];
export const AUGMENTATION_OPTIONS: AugmentationStrength[] = ["light", "medium", "heavy"];

const PRESET_BASE: RunSpec = {
  projectName: "medfusion-oss",
  experimentName: "quickstart-run",
  description: "",
  tags: ["oss", "multimodal"],
  seed: 42,
  device: "auto",
  data: {
    csvPath: "data/mock/metadata.csv",
    imageDir: "data/mock",
    imagePathColumn: "image_path",
    targetColumn: "diagnosis",
    patientIdColumn: "",
    numericalFeatures: ["age"],
    categoricalFeatures: ["gender"],
    trainRatio: 0.7,
    valRatio: 0.15,
    testRatio: 0.15,
    imageSize: 224,
    batchSize: 4,
    numWorkers: 0,
    pinMemory: false,
    augmentationStrength: "light",
  },
  model: {
    numClasses: 2,
    useAuxiliaryHeads: true,
    vision: {
      backbone: "resnet18",
      pretrained: true,
      freezeBackbone: false,
      featureDim: 128,
      dropout: 0.3,
      attentionType: "cbam",
    },
    tabular: {
      hiddenDims: [32],
      outputDim: 16,
      dropout: 0.2,
    },
    fusion: {
      fusionType: "concatenate",
      hiddenDim: 144,
      dropout: 0.3,
      numHeads: 4,
    },
  },
  training: {
    numEpochs: 3,
    mixedPrecision: false,
    gradientClip: 1,
    useProgressiveTraining: false,
    stage1Epochs: 1,
    stage2Epochs: 1,
    stage3Epochs: 1,
    useAttentionSupervision: false,
    attentionLossWeight: 0.1,
    optimizer: {
      optimizer: "adam",
      learningRate: 0.001,
      weightDecay: 0,
      momentum: 0.9,
    },
    scheduler: {
      scheduler: "step",
      warmupEpochs: 0,
      minLr: 1e-6,
      stepSize: 1,
      gamma: 0.1,
      patience: 5,
      factor: 0.5,
    },
  },
  logging: {
    outputDir: "outputs/medfusion-mvp/quickstart-run",
    useTensorboard: false,
    useWandb: false,
  },
};

function cloneRunSpec(spec: RunSpec): RunSpec {
  return JSON.parse(JSON.stringify(spec)) as RunSpec;
}

export function inferOutputDir(projectName: string, experimentName: string): string {
  const project = slugify(projectName || "medfusion");
  const experiment = slugify(experimentName || "run");
  return `outputs/${project}/${experiment}`;
}

export function createRunSpecPreset(preset: RunPresetId): RunSpec {
  const base = cloneRunSpec(PRESET_BASE);

  if (preset === "showcase") {
    base.projectName = "medfusion-results";
    base.experimentName = "attention-audit";
    base.description = "用于结果审查和可视化产物补齐的多模态实验。";
    base.tags = ["results", "attention", "audit"];
    base.model.vision.backbone = "resnet50";
    base.model.vision.featureDim = 256;
    base.model.fusion.fusionType = "attention";
    base.model.fusion.hiddenDim = 192;
    base.training.numEpochs = 8;
    base.training.mixedPrecision = true;
    base.training.scheduler.scheduler = "cosine";
    base.training.optimizer.optimizer = "adamw";
    base.training.optimizer.learningRate = 3e-4;
    base.training.optimizer.weightDecay = 1e-2;
    base.data.batchSize = 8;
    base.data.numWorkers = 2;
    base.data.pinMemory = true;
    base.data.augmentationStrength = "medium";
    base.logging.useTensorboard = true;
  }

  if (preset === "clinical") {
    base.projectName = "medfusion-clinical";
    base.experimentName = "baseline-clinical";
    base.description = "偏稳健的临床基线配置，适合在真实数据集上继续迭代。";
    base.tags = ["baseline", "clinical", "validation"];
    base.model.vision.backbone = "efficientnet_b0";
    base.model.vision.featureDim = 192;
    base.model.fusion.fusionType = "gated";
    base.model.fusion.hiddenDim = 160;
    base.training.numEpochs = 18;
    base.training.mixedPrecision = true;
    base.training.useProgressiveTraining = true;
    base.training.stage1Epochs = 6;
    base.training.stage2Epochs = 8;
    base.training.stage3Epochs = 4;
    base.training.scheduler.scheduler = "plateau";
    base.training.scheduler.patience = 4;
    base.training.scheduler.factor = 0.5;
    base.training.optimizer.optimizer = "adamw";
    base.training.optimizer.learningRate = 2e-4;
    base.training.optimizer.weightDecay = 1e-2;
    base.data.batchSize = 12;
    base.data.numWorkers = 4;
    base.data.pinMemory = true;
    base.data.augmentationStrength = "medium";
    base.logging.useTensorboard = true;
  }

  base.logging.outputDir = inferOutputDir(base.projectName, base.experimentName);
  return base;
}

export function validateRunSpec(spec: RunSpec): RunSpecValidationIssue[] {
  const issues: RunSpecValidationIssue[] = [];
  const ratioSum = spec.data.trainRatio + spec.data.valRatio + spec.data.testRatio;
  const tabularFeatureCount = spec.data.numericalFeatures.length + spec.data.categoricalFeatures.length;

  if (!spec.projectName.trim()) {
    issues.push({ level: "error", message: "project_name 不能为空。" });
  }
  if (!spec.experimentName.trim()) {
    issues.push({ level: "error", message: "experiment_name 不能为空。" });
  }
  if (!spec.data.csvPath.trim()) {
    issues.push({ level: "error", message: "需要填写 CSV 标注文件路径。" });
  }
  if (!spec.data.imageDir.trim()) {
    issues.push({ level: "error", message: "需要填写图像目录路径。" });
  }
  if (!spec.data.imagePathColumn.trim()) {
    issues.push({ level: "error", message: "需要指定 image_path_column。" });
  }
  if (!spec.data.targetColumn.trim()) {
    issues.push({ level: "error", message: "需要指定 target_column。" });
  }
  if (tabularFeatureCount === 0) {
    issues.push({ level: "error", message: "当前训练链路要求至少提供一个表格特征。" });
  }
  if (Math.abs(ratioSum - 1) > 0.001) {
    issues.push({ level: "error", message: "train/val/test 划分比例之和必须等于 1。" });
  }
  if (spec.training.numEpochs < 1) {
    issues.push({ level: "error", message: "训练轮数必须大于等于 1。" });
  }
  if (spec.training.useProgressiveTraining) {
    const stageTotal =
      spec.training.stage1Epochs + spec.training.stage2Epochs + spec.training.stage3Epochs;
    if (stageTotal !== spec.training.numEpochs) {
      issues.push({
        level: "error",
        message: "启用 progressive training 时，三个 stage 的 epoch 总和必须等于总 epoch。",
      });
    }
  }
  if (spec.model.fusion.hiddenDim < 2) {
    issues.push({ level: "error", message: "fusion.hidden_dim 不能小于 2。" });
  }
  if ((spec.model.fusion.fusionType === "attention" || spec.model.fusion.fusionType === "cross_attention") && spec.model.fusion.numHeads < 1) {
    issues.push({ level: "error", message: "attention 类 fusion 需要至少 1 个 attention head。" });
  }
  if (spec.training.useAttentionSupervision && spec.model.vision.attentionType !== "cbam") {
    issues.push({
      level: "error",
      message: "注意力监督目前要求 vision.attention_type = cbam。",
    });
  }
  if (!spec.logging.outputDir.trim()) {
    issues.push({ level: "error", message: "output_dir 不能为空。" });
  }
  if (spec.training.optimizer.learningRate > 0.01) {
    issues.push({ level: "warning", message: "learning_rate 偏大，容易让训练不稳定。" });
  }
  if (spec.data.batchSize > 32) {
    issues.push({ level: "warning", message: "batch_size 较大，单卡显存压力可能偏高。" });
  }
  if (spec.model.vision.freezeBackbone && spec.training.numEpochs <= 3) {
    issues.push({
      level: "warning",
      message: "冻结 backbone 且总 epoch 很小，可能学不到有效融合表示。",
    });
  }

  return issues;
}

export function buildYamlFromRunSpec(spec: RunSpec): string {
  const payload = buildConfigObject(spec);
  const header = [
    "# MedFusion real training config",
    "# Generated by the web run wizard",
    "# Recommended execution:",
    "#   uv run medfusion validate-config --config ./generated-run.yaml",
    "#   uv run medfusion train --config ./generated-run.yaml",
    "",
  ].join("\n");
  return `${header}${serializeYaml(payload)}\n`;
}

export function buildTrainCommand(configPath: string): string {
  return `uv run medfusion validate-config --config ${configPath} && uv run medfusion train --config ${configPath}`;
}

export function buildResultsCommand(configPath: string, checkpointPath = "outputs/.../checkpoints/best.pth"): string {
  return `uv run medfusion build-results --config ${configPath} --checkpoint ${checkpointPath}`;
}

function buildConfigObject(spec: RunSpec): Record<string, unknown> {
  const config: Record<string, unknown> = {
    project_name: spec.projectName,
    experiment_name: spec.experimentName,
    description: spec.description,
    tags: spec.tags,
    seed: spec.seed,
    device: spec.device,
    data: {
      csv_path: spec.data.csvPath,
      image_dir: spec.data.imageDir,
      image_path_column: spec.data.imagePathColumn,
      target_column: spec.data.targetColumn,
      numerical_features: spec.data.numericalFeatures,
      categorical_features: spec.data.categoricalFeatures,
      train_ratio: spec.data.trainRatio,
      val_ratio: spec.data.valRatio,
      test_ratio: spec.data.testRatio,
      image_size: spec.data.imageSize,
      batch_size: spec.data.batchSize,
      num_workers: spec.data.numWorkers,
      pin_memory: spec.data.pinMemory,
      augmentation_strength: spec.data.augmentationStrength,
    },
    model: {
      num_classes: spec.model.numClasses,
      use_auxiliary_heads: spec.model.useAuxiliaryHeads,
      vision: {
        backbone: spec.model.vision.backbone,
        pretrained: spec.model.vision.pretrained,
        freeze_backbone: spec.model.vision.freezeBackbone,
        feature_dim: spec.model.vision.featureDim,
        dropout: spec.model.vision.dropout,
        attention_type: spec.model.vision.attentionType,
      },
      tabular: {
        hidden_dims: spec.model.tabular.hiddenDims,
        output_dim: spec.model.tabular.outputDim,
        dropout: spec.model.tabular.dropout,
      },
      fusion: {
        fusion_type: spec.model.fusion.fusionType,
        hidden_dim: spec.model.fusion.hiddenDim,
        dropout: spec.model.fusion.dropout,
        num_heads: spec.model.fusion.numHeads,
      },
    },
    training: {
      num_epochs: spec.training.numEpochs,
      mixed_precision: spec.training.mixedPrecision,
      gradient_clip: spec.training.gradientClip,
      use_progressive_training: spec.training.useProgressiveTraining,
      monitor: "accuracy",
      mode: "max",
      optimizer: {
        optimizer: spec.training.optimizer.optimizer,
        learning_rate: spec.training.optimizer.learningRate,
        weight_decay: spec.training.optimizer.weightDecay,
        momentum: spec.training.optimizer.momentum,
        use_differential_lr: false,
      },
      scheduler: {
        scheduler: spec.training.scheduler.scheduler,
        warmup_epochs: spec.training.scheduler.warmupEpochs,
        min_lr: spec.training.scheduler.minLr,
        step_size: spec.training.scheduler.stepSize,
        gamma: spec.training.scheduler.gamma,
        patience: spec.training.scheduler.patience,
        factor: spec.training.scheduler.factor,
      },
    },
    logging: {
      output_dir: spec.logging.outputDir,
      use_tensorboard: spec.logging.useTensorboard,
      use_wandb: spec.logging.useWandb,
    },
  };

  if (spec.data.patientIdColumn.trim()) {
    (config.data as Record<string, unknown>).patient_id_column = spec.data.patientIdColumn.trim();
  }

  if (spec.training.useProgressiveTraining) {
    Object.assign(config.training as Record<string, unknown>, {
      stage1_epochs: spec.training.stage1Epochs,
      stage2_epochs: spec.training.stage2Epochs,
      stage3_epochs: spec.training.stage3Epochs,
    });
  }

  if (spec.training.useAttentionSupervision) {
    Object.assign((config.model as Record<string, unknown>).vision as Record<string, unknown>, {
      enable_attention_supervision: true,
    });
    Object.assign(config.training as Record<string, unknown>, {
      use_attention_supervision: true,
      attention_loss_weight: spec.training.attentionLossWeight,
      attention_supervision_method: "cam",
    });
  }

  return config;
}

function slugify(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "run";
}

function serializeYaml(value: unknown, indent = 0): string {
  const prefix = "  ".repeat(indent);

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return "[]";
    }

    if (value.every(isScalar)) {
      return `[${value.map(serializeScalar).join(", ")}]`;
    }

    return value
      .map((item) => {
        if (isScalar(item)) {
          return `${prefix}- ${serializeScalar(item)}`;
        }
        return `${prefix}-\n${serializeYaml(item, indent + 1)}`;
      })
      .join("\n");
  }

  if (isPlainObject(value)) {
    return Object.entries(value)
      .filter(([, item]) => item !== undefined)
      .map(([key, item]) => {
        if (isScalar(item)) {
          return `${prefix}${key}: ${serializeScalar(item)}`;
        }
        if (Array.isArray(item) && item.every(isScalar)) {
          return `${prefix}${key}: ${serializeYaml(item, 0)}`;
        }
        return `${prefix}${key}:\n${serializeYaml(item, indent + 1)}`;
      })
      .join("\n");
  }

  return `${prefix}${serializeScalar(value)}`;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isScalar(value: unknown): boolean {
  return value === null || ["string", "number", "boolean"].includes(typeof value);
}

function serializeScalar(value: unknown): string {
  if (typeof value === "string") {
    return JSON.stringify(value);
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (value === null) {
    return "null";
  }
  return String(value);
}
