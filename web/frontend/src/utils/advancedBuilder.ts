import type { Edge, Node } from "reactflow";

import {
  ADVANCED_BUILDER_BLUEPRINTS,
  ADVANCED_BUILDER_COMPONENTS,
  ADVANCED_BUILDER_CONNECTION_RULES,
  ADVANCED_BUILDER_FAMILY_LABELS,
  type AdvancedBuilderBlueprint,
  type AdvancedBuilderComponent,
  type AdvancedBuilderFamily,
} from "@/config/advancedBuilderCatalog";
import {
  createRunSpecPreset,
  inferOutputDir,
  type RunPresetId,
  type RunSpec,
} from "@/utils/runSpec";

export interface AdvancedBuilderNodeData {
  componentId: string;
  label: string;
  family: AdvancedBuilderFamily;
  status: AdvancedBuilderComponent["status"];
  description: string;
  schemaPath?: string;
}

export interface AdvancedBuilderEvaluation {
  compileReady: boolean;
  missingFamilies: AdvancedBuilderFamily[];
  missingRequiredConnections: Array<{
    fromFamily: AdvancedBuilderFamily;
    toFamily: AdvancedBuilderFamily;
    description: string;
  }>;
  conditionalComponents: string[];
  draftOnlyComponents: string[];
}

export interface AdvancedBuilderCompileIssue {
  level: "error" | "warning";
  message: string;
}

export interface AdvancedBuilderCompileResult {
  preset: RunPresetId;
  spec: RunSpec | null;
  issues: AdvancedBuilderCompileIssue[];
  chosenComponents: Partial<Record<AdvancedBuilderFamily, string>>;
}

const REQUIRED_FAMILIES: AdvancedBuilderFamily[] = [
  "data_input",
  "vision_backbone",
  "tabular_encoder",
  "fusion",
  "head",
  "training_strategy",
];

const FAMILY_POSITIONS: Record<AdvancedBuilderFamily, { x: number; y: number }> = {
  data_input: { x: 80, y: 160 },
  vision_backbone: { x: 320, y: 60 },
  tabular_encoder: { x: 320, y: 260 },
  fusion: { x: 560, y: 160 },
  head: { x: 800, y: 160 },
  training_strategy: { x: 1040, y: 160 },
};

function getComponentOrThrow(componentId: string): AdvancedBuilderComponent {
  const component = ADVANCED_BUILDER_COMPONENTS.find(
    (item) => item.id === componentId,
  );
  if (!component) {
    throw new Error(`Unknown advanced builder component: ${componentId}`);
  }
  return component;
}

function getBlueprintOrThrow(blueprintId: string): AdvancedBuilderBlueprint {
  const blueprint = ADVANCED_BUILDER_BLUEPRINTS.find(
    (item) => item.id === blueprintId,
  );
  if (!blueprint) {
    throw new Error(`Unknown advanced builder blueprint: ${blueprintId}`);
  }
  return blueprint;
}

function slugify(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function createNodeId(componentId: string, index: number): string {
  return `${componentId}__${index}`;
}

function createEdgeId(source: string, target: string): string {
  return `${source}__${target}`;
}

export function createBuilderNode(
  componentId: string,
  index = 0,
): Node<AdvancedBuilderNodeData> {
  const component = getComponentOrThrow(componentId);
  const position = FAMILY_POSITIONS[component.family];

  return {
    id: createNodeId(componentId, index),
    type: "advancedBuilderComponent",
    position: {
      x: position.x + index * 24,
      y: position.y + index * 16,
    },
    data: {
      componentId: component.id,
      label: component.label,
      family: component.family,
      status: component.status,
      description: component.description,
      schemaPath: component.schemaPath,
    },
  };
}

export function buildBlueprintGraph(blueprintId: string): {
  blueprint: AdvancedBuilderBlueprint;
  nodes: Array<Node<AdvancedBuilderNodeData>>;
  edges: Edge[];
} {
  const blueprint = getBlueprintOrThrow(blueprintId);
  const nodes = blueprint.components.map((componentId, index) =>
    createBuilderNode(componentId, index),
  );

  const familyToNode = new Map<AdvancedBuilderFamily, Node<AdvancedBuilderNodeData>>();
  for (const node of nodes) {
    if (!familyToNode.has(node.data.family)) {
      familyToNode.set(node.data.family, node);
    }
  }

  const edges: Edge[] = [];
  const connectFamilies = (
    fromFamily: AdvancedBuilderFamily,
    toFamily: AdvancedBuilderFamily,
  ) => {
    const source = familyToNode.get(fromFamily);
    const target = familyToNode.get(toFamily);
    if (!source || !target) {
      return;
    }
    edges.push({
      id: createEdgeId(source.id, target.id),
      source: source.id,
      target: target.id,
      animated: fromFamily === "head" && toFamily === "training_strategy",
      label:
        ADVANCED_BUILDER_CONNECTION_RULES.find(
          (rule) => rule.fromFamily === fromFamily && rule.toFamily === toFamily,
        )?.status ?? "link",
    });
  };

  connectFamilies("data_input", "vision_backbone");
  connectFamilies("data_input", "tabular_encoder");
  connectFamilies("vision_backbone", "fusion");
  connectFamilies("tabular_encoder", "fusion");
  connectFamilies("fusion", "head");
  connectFamilies("head", "training_strategy");

  return { blueprint, nodes, edges };
}

export function canConnectFamilies(
  fromFamily: AdvancedBuilderFamily,
  toFamily: AdvancedBuilderFamily,
): {
  allowed: boolean;
  status: "required" | "conditional" | "blocked" | "unsupported";
  description: string;
} {
  const rule = ADVANCED_BUILDER_CONNECTION_RULES.find(
    (item) => item.fromFamily === fromFamily && item.toFamily === toFamily,
  );

  if (!rule) {
    return {
      allowed: false,
      status: "unsupported",
      description: `当前没有定义 ${ADVANCED_BUILDER_FAMILY_LABELS[fromFamily]} -> ${ADVANCED_BUILDER_FAMILY_LABELS[toFamily]} 的正式版连接规则。`,
    };
  }

  return {
    allowed: rule.status !== "blocked",
    status: rule.status,
    description: rule.description,
  };
}

export function evaluateAdvancedBuilderGraph(
  nodes: Array<Node<AdvancedBuilderNodeData>>,
  edges: Edge[],
): AdvancedBuilderEvaluation {
  const familySet = new Set(nodes.map((node) => node.data.family));
  const missingFamilies = REQUIRED_FAMILIES.filter((family) => !familySet.has(family));

  const missingRequiredConnections = ADVANCED_BUILDER_CONNECTION_RULES.filter(
    (rule) => rule.status === "required",
  )
    .filter((rule) => familySet.has(rule.fromFamily) && familySet.has(rule.toFamily))
    .filter((rule) => {
      return !edges.some((edge) => {
        const sourceNode = nodes.find((node) => node.id === edge.source);
        const targetNode = nodes.find((node) => node.id === edge.target);
        return (
          sourceNode?.data.family === rule.fromFamily &&
          targetNode?.data.family === rule.toFamily
        );
      });
    })
    .map((rule) => ({
      fromFamily: rule.fromFamily,
      toFamily: rule.toFamily,
      description: rule.description,
    }));

  const conditionalComponents = nodes
    .filter((node) => node.data.status === "conditional")
    .map((node) => node.data.label);
  const draftOnlyComponents = nodes
    .filter((node) => node.data.status === "draft_only")
    .map((node) => node.data.label);

  return {
    compileReady:
      missingFamilies.length === 0 &&
      missingRequiredConnections.length === 0 &&
      draftOnlyComponents.length === 0,
    missingFamilies,
    missingRequiredConnections,
    conditionalComponents,
    draftOnlyComponents,
  };
}

function inferPresetFromChosenComponents(
  chosenComponents: Partial<Record<AdvancedBuilderFamily, string>>,
): RunPresetId {
  const visionComponent = chosenComponents.vision_backbone || "";
  const fusionComponent = chosenComponents.fusion || "";
  const trainingComponent = chosenComponents.training_strategy || "";

  if (
    visionComponent === "attention_backbone_bundle" ||
    fusionComponent === "attention_fusion"
  ) {
    return "showcase";
  }

  if (
    visionComponent === "efficientnet_b0_backbone" ||
    fusionComponent === "gated_fusion" ||
    trainingComponent === "progressive_training"
  ) {
    return "clinical";
  }

  return "quickstart";
}

function createCompileBaseSpec(preset: RunPresetId): RunSpec {
  const spec = createRunSpecPreset(preset);
  spec.projectName = "medfusion-formal";
  spec.experimentName = `advanced-${preset}-graph`;
  spec.description =
    "Compiled from the formal-release advanced builder graph prototype.";
  spec.tags = Array.from(new Set([...spec.tags, "advanced-builder", "compiled"]));
  spec.logging.outputDir = inferOutputDir(spec.projectName, spec.experimentName);
  return spec;
}

function applyComponentToSpec(
  spec: RunSpec,
  componentId: string,
  issues: AdvancedBuilderCompileIssue[],
): void {
  switch (componentId) {
    case "image_tabular_dataset":
      spec.data.csvPath = "data/mock/metadata.csv";
      spec.data.imageDir = "data/mock";
      spec.data.imagePathColumn = "image_path";
      spec.data.targetColumn = "diagnosis";
      spec.data.numericalFeatures = ["age"];
      spec.data.categoricalFeatures = ["gender"];
      return;
    case "resnet18_backbone":
      spec.model.vision.backbone = "resnet18";
      spec.model.vision.featureDim = 128;
      spec.model.vision.attentionType = "cbam";
      return;
    case "efficientnet_b0_backbone":
      spec.model.vision.backbone = "efficientnet_b0";
      spec.model.vision.featureDim = 192;
      spec.model.vision.attentionType = "cbam";
      return;
    case "attention_backbone_bundle":
      spec.model.vision.backbone = "resnet50";
      spec.model.vision.featureDim = 256;
      spec.model.vision.attentionType = "cbam";
      spec.training.useAttentionSupervision = true;
      issues.push({
        level: "warning",
        message:
          "当前图使用了 attention-supervised backbone，编译结果会默认走 CBAM + attention supervision 条件路径。",
      });
      return;
    case "mlp_tabular_encoder":
      spec.model.tabular.hiddenDims = [32];
      spec.model.tabular.outputDim = 16;
      spec.model.tabular.dropout = 0.2;
      return;
    case "concatenate_fusion":
      spec.model.fusion.fusionType = "concatenate";
      spec.model.fusion.hiddenDim =
        spec.model.vision.featureDim + spec.model.tabular.outputDim;
      spec.model.fusion.dropout = 0.3;
      return;
    case "gated_fusion":
      spec.model.fusion.fusionType = "gated";
      spec.model.fusion.hiddenDim = 160;
      spec.model.fusion.dropout = 0.3;
      return;
    case "attention_fusion":
      spec.model.fusion.fusionType = "attention";
      spec.model.fusion.hiddenDim = 192;
      spec.model.fusion.numHeads = 4;
      spec.model.fusion.dropout = 0.25;
      issues.push({
        level: "warning",
        message:
          "当前图使用了 attention fusion，编译结果会保留注意力路径，但仍受正式版主链的现有 fusion schema 约束。",
      });
      return;
    case "classification_head":
      spec.model.numClasses = 2;
      spec.model.useAuxiliaryHeads = true;
      return;
    case "standard_training":
      spec.training.useProgressiveTraining = false;
      spec.training.useAttentionSupervision =
        spec.training.useAttentionSupervision && spec.model.vision.attentionType === "cbam";
      return;
    case "progressive_training":
      spec.training.useProgressiveTraining = true;
      spec.training.numEpochs = 18;
      spec.training.stage1Epochs = 6;
      spec.training.stage2Epochs = 8;
      spec.training.stage3Epochs = 4;
      return;
    default:
      issues.push({
        level: "error",
        message: `当前编译器还不能把组件 ${componentId} 降级映射到正式版 RunSpec。`,
      });
  }
}

function collectChosenComponents(
  nodes: Array<Node<AdvancedBuilderNodeData>>,
): {
  chosenComponents: Partial<Record<AdvancedBuilderFamily, string>>;
  duplicateFamilies: AdvancedBuilderFamily[];
} {
  const grouped = new Map<AdvancedBuilderFamily, Array<Node<AdvancedBuilderNodeData>>>();
  for (const node of nodes) {
    const entries = grouped.get(node.data.family) || [];
    entries.push(node);
    grouped.set(node.data.family, entries);
  }

  const chosenComponents: Partial<Record<AdvancedBuilderFamily, string>> = {};
  const duplicateFamilies: AdvancedBuilderFamily[] = [];
  for (const [family, familyNodes] of grouped.entries()) {
    if (familyNodes.length > 1) {
      duplicateFamilies.push(family);
      continue;
    }
    chosenComponents[family] = familyNodes[0]?.data.componentId;
  }

  return { chosenComponents, duplicateFamilies };
}

export function compileAdvancedBuilderGraphToRunSpec(
  nodes: Array<Node<AdvancedBuilderNodeData>>,
  edges: Edge[],
): AdvancedBuilderCompileResult {
  const issues: AdvancedBuilderCompileIssue[] = [];
  const evaluation = evaluateAdvancedBuilderGraph(nodes, edges);
  const { chosenComponents, duplicateFamilies } = collectChosenComponents(nodes);

  if (!nodes.length) {
    issues.push({
      level: "error",
      message: "当前画布为空，无法生成配置草案。",
    });
  }

  if (duplicateFamilies.length) {
    issues.push({
      level: "error",
      message: `当前图存在重复组件家族：${duplicateFamilies
        .map((family) => ADVANCED_BUILDER_FAMILY_LABELS[family])
        .join(" / ")}。正式版编译层当前要求每个核心家族最多一个组件。`,
    });
  }

  if (evaluation.missingFamilies.length) {
    issues.push({
      level: "error",
      message: `缺少必需组件家族：${evaluation.missingFamilies
        .map((family) => ADVANCED_BUILDER_FAMILY_LABELS[family])
        .join(" / ")}。`,
    });
  }

  if (evaluation.missingRequiredConnections.length) {
    issues.push({
      level: "error",
      message: `缺少必需连接：${evaluation.missingRequiredConnections
        .map(
          (rule) =>
            `${ADVANCED_BUILDER_FAMILY_LABELS[rule.fromFamily]} -> ${ADVANCED_BUILDER_FAMILY_LABELS[rule.toFamily]}`,
        )
        .join(" / ")}。`,
    });
  }

  if (evaluation.draftOnlyComponents.length) {
    issues.push({
      level: "error",
      message: `当前图包含仅草稿组件：${evaluation.draftOnlyComponents.join(
        " / ",
      )}。这些组件还不能编译进正式版主链。`,
    });
  }

  const preset = inferPresetFromChosenComponents(chosenComponents);
  if (issues.some((issue) => issue.level === "error")) {
    return {
      preset,
      spec: null,
      issues,
      chosenComponents,
    };
  }

  const spec = createCompileBaseSpec(preset);
  spec.projectName = "medfusion-formal";
  spec.experimentName = `advanced-${slugify(preset)}-graph`;
  spec.logging.outputDir = inferOutputDir(spec.projectName, spec.experimentName);

  for (const family of REQUIRED_FAMILIES) {
    const componentId = chosenComponents[family];
    if (!componentId) {
      continue;
    }
    applyComponentToSpec(spec, componentId, issues);
  }

  return {
    preset,
    spec,
    issues,
    chosenComponents,
  };
}
