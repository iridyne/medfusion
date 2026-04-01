import type { TrainingLaunchPrefill } from "@/utils/trainingPrefill";

export interface QuickstartProfile {
  id: string;
  title: string;
  description: string;
  configPath: string;
  prepareCommand: string;
  validateCommand: string;
  trainCommand: string;
  buildResultsCommand: string;
}

export interface QuickstartStage {
  key: "prepare" | "validate-config" | "train" | "build-results";
  title: string;
  description: string;
  command: string;
}

export const RECOMMENDED_QUICKSTART_PROFILE: QuickstartProfile = {
  id: "medmnist-breastmnist",
  title: "BreastMNIST public dataset quickstart",
  description:
    "先用公开数据跑通一条最小闭环，避免一开始就被私有数据准备和自定义配置放大变量。",
  configPath: "configs/public_datasets/breastmnist_quickstart.yaml",
  prepareCommand:
    "uv run medfusion public-datasets prepare medmnist-breastmnist --overwrite",
  validateCommand:
    "uv run medfusion validate-config --config configs/public_datasets/breastmnist_quickstart.yaml",
  trainCommand:
    "uv run medfusion train --config configs/public_datasets/breastmnist_quickstart.yaml",
  buildResultsCommand:
    "uv run medfusion build-results --config configs/public_datasets/breastmnist_quickstart.yaml --checkpoint outputs/public_datasets/breastmnist_quickstart/checkpoints/best.pth",
};

export const QUICKSTART_STAGES: QuickstartStage[] = [
  {
    key: "prepare",
    title: "准备公开数据",
    description: "先把推荐数据 profile 拉到本地，让第一次运行不依赖你自己的目录结构。",
    command: RECOMMENDED_QUICKSTART_PROFILE.prepareCommand,
  },
  {
    key: "validate-config",
    title: "校验配置",
    description: "在真正训练前先验证 YAML，尽早发现路径、字段和 schema 问题。",
    command: RECOMMENDED_QUICKSTART_PROFILE.validateCommand,
  },
  {
    key: "train",
    title: "启动训练",
    description: "训练阶段继续走同一条 YAML + CLI 主链，不切换到另一套执行语义。",
    command: RECOMMENDED_QUICKSTART_PROFILE.trainCommand,
  },
  {
    key: "build-results",
    title: "构建结果",
    description: "把 checkpoint 转成 summary、validation 和 report，方便回到 Web 理解结果。",
    command: RECOMMENDED_QUICKSTART_PROFILE.buildResultsCommand,
  },
];

export const QUICKSTART_RESULT_PATHS = [
  "outputs/public_datasets/breastmnist_quickstart/checkpoints/best.pth",
  "outputs/public_datasets/breastmnist_quickstart/metrics/metrics.json",
  "outputs/public_datasets/breastmnist_quickstart/metrics/validation.json",
  "outputs/public_datasets/breastmnist_quickstart/reports/summary.json",
  "outputs/public_datasets/breastmnist_quickstart/reports/report.md",
] as const;

export const QUICKSTART_TRAINING_PREFILL: TrainingLaunchPrefill = {
  experimentName: "breastmnist-quickstart",
  backbone: "resnet18",
  numClasses: 2,
  epochs: 10,
  batchSize: 32,
  learningRate: 0.001,
};
