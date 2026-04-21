import { describe, expect, it } from "vitest";

import {
  buildBlueprintGraph,
  canConnectFamilies,
  createBuilderNode,
  evaluateAdvancedBuilderGraph,
} from "@/utils/advancedBuilder";
import type {
  AdvancedBuilderBlueprint,
  AdvancedBuilderComponent,
  AdvancedBuilderConnectionRule,
} from "@/config/advancedBuilderCatalog";

const FAMILY_LABELS = {
  data_input: "数据输入",
  vision_backbone: "视觉 backbone",
  tabular_encoder: "表格编码器",
  fusion: "融合层",
  head: "任务头",
  training_strategy: "训练策略",
} as const;

const STATUS_LABELS = {
  compile_ready: "可编译",
  conditional: "有条件开放",
  draft_only: "仅草稿",
} as const;

const COMPONENTS: AdvancedBuilderComponent[] = [
  {
    id: "image_tabular_dataset",
    family: "data_input",
    label: "图像 + 表格输入",
    status: "compile_ready",
    description: "default",
  },
  {
    id: "three_phase_ct_dataset",
    family: "data_input",
    label: "三相 CT + 临床输入",
    status: "draft_only",
    description: "draft",
  },
  {
    id: "resnet18_backbone",
    family: "vision_backbone",
    label: "ResNet18",
    status: "compile_ready",
    description: "resnet18",
  },
  {
    id: "efficientnet_b0_backbone",
    family: "vision_backbone",
    label: "EfficientNet-B0",
    status: "compile_ready",
    description: "efficientnet",
  },
  {
    id: "mlp_tabular_encoder",
    family: "tabular_encoder",
    label: "MLP 表格编码器",
    status: "compile_ready",
    description: "tabular",
  },
  {
    id: "concatenate_fusion",
    family: "fusion",
    label: "Concatenate Fusion",
    status: "compile_ready",
    description: "concat",
  },
  {
    id: "classification_head",
    family: "head",
    label: "分类头",
    status: "compile_ready",
    description: "head",
  },
  {
    id: "standard_training",
    family: "training_strategy",
    label: "标准训练",
    status: "compile_ready",
    description: "train",
  },
];

const CONNECTION_RULES: AdvancedBuilderConnectionRule[] = [
  {
    fromFamily: "data_input",
    toFamily: "vision_backbone",
    status: "required",
    description: "required",
  },
  {
    fromFamily: "data_input",
    toFamily: "tabular_encoder",
    status: "required",
    description: "required",
  },
  {
    fromFamily: "vision_backbone",
    toFamily: "fusion",
    status: "required",
    description: "required",
  },
  {
    fromFamily: "tabular_encoder",
    toFamily: "fusion",
    status: "required",
    description: "required",
  },
  {
    fromFamily: "fusion",
    toFamily: "head",
    status: "required",
    description: "required",
  },
  {
    fromFamily: "head",
    toFamily: "training_strategy",
    status: "required",
    description: "required",
  },
  {
    fromFamily: "data_input",
    toFamily: "head",
    status: "blocked",
    description: "blocked",
  },
];

const BLUEPRINTS: AdvancedBuilderBlueprint[] = [
  {
    id: "quickstart_multimodal",
    label: "Quickstart",
    status: "compile_ready",
    description: "ready",
    components: [
      "image_tabular_dataset",
      "resnet18_backbone",
      "mlp_tabular_encoder",
      "concatenate_fusion",
      "classification_head",
      "standard_training",
    ],
  },
];

describe("advancedBuilder utils", () => {
  it("builds a compile-ready graph for the quickstart blueprint", () => {
    const graph = buildBlueprintGraph(
      "quickstart_multimodal",
      BLUEPRINTS,
      COMPONENTS,
      CONNECTION_RULES,
      FAMILY_LABELS,
      STATUS_LABELS,
    );
    const evaluation = evaluateAdvancedBuilderGraph(
      graph.nodes,
      graph.edges,
      CONNECTION_RULES,
    );

    expect(graph.nodes.length).toBeGreaterThan(0);
    expect(graph.edges.length).toBeGreaterThan(0);
    expect(evaluation.compileReady).toBe(true);
    expect(evaluation.missingFamilies).toEqual([]);
  });

  it("blocks invalid connections across unsupported families", () => {
    const result = canConnectFamilies("data_input", "head", CONNECTION_RULES, FAMILY_LABELS);

    expect(result.allowed).toBe(false);
    expect(result.status).toBe("blocked");
  });

  it("marks graphs with draft-only components as not compile-ready", () => {
    const draftNode = createBuilderNode(
      "three_phase_ct_dataset",
      0,
      COMPONENTS,
      FAMILY_LABELS,
      STATUS_LABELS,
    );
    const evaluation = evaluateAdvancedBuilderGraph([draftNode], [], CONNECTION_RULES);

    expect(evaluation.compileReady).toBe(false);
    expect(evaluation.draftOnlyComponents).toContain("三相 CT + 临床输入");
  });

  it("marks graphs with duplicate required families as not compile-ready", () => {
    const graph = buildBlueprintGraph(
      "quickstart_multimodal",
      BLUEPRINTS,
      COMPONENTS,
      CONNECTION_RULES,
      FAMILY_LABELS,
      STATUS_LABELS,
    );
    const duplicateBackbone = createBuilderNode(
      "efficientnet_b0_backbone",
      99,
      COMPONENTS,
      FAMILY_LABELS,
      STATUS_LABELS,
    );
    const evaluation = evaluateAdvancedBuilderGraph(
      [...graph.nodes, duplicateBackbone],
      graph.edges,
      CONNECTION_RULES,
    );

    expect(evaluation.compileReady).toBe(false);
    expect(evaluation.duplicateFamilies).toContain("vision_backbone");
  });
});
