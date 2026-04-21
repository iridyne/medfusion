import type { Edge, Node } from "reactflow";

import {
  ADVANCED_BUILDER_BLUEPRINTS,
  ADVANCED_BUILDER_COMPONENTS,
  ADVANCED_BUILDER_CONNECTION_RULES,
  ADVANCED_BUILDER_FAMILY_LABELS,
  type AdvancedBuilderBlueprint,
  type AdvancedBuilderComponent,
  type AdvancedBuilderFamily,
  type AdvancedBuilderConnectionRule,
} from "@/config/advancedBuilderCatalog";

export interface AdvancedBuilderNodeData {
  componentId: string;
  label: string;
  family: AdvancedBuilderFamily;
  familyLabel?: string;
  status: AdvancedBuilderComponent["status"];
  statusLabel?: string;
  description: string;
  schemaPath?: string;
  notes?: string[];
}

export interface AdvancedBuilderEvaluation {
  compileReady: boolean;
  missingFamilies: AdvancedBuilderFamily[];
  duplicateFamilies: AdvancedBuilderFamily[];
  missingRequiredConnections: Array<{
    fromFamily: AdvancedBuilderFamily;
    toFamily: AdvancedBuilderFamily;
    description: string;
  }>;
  conditionalComponents: string[];
  draftOnlyComponents: string[];
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

function getComponentOrThrowFromCatalog(
  componentId: string,
  components: AdvancedBuilderComponent[],
): AdvancedBuilderComponent {
  const component = components.find((item) => item.id === componentId);
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

function getBlueprintOrThrowFromCatalog(
  blueprintId: string,
  blueprints: AdvancedBuilderBlueprint[],
): AdvancedBuilderBlueprint {
  const blueprint = blueprints.find((item) => item.id === blueprintId);
  if (!blueprint) {
    throw new Error(`Unknown advanced builder blueprint: ${blueprintId}`);
  }
  return blueprint;
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
  components: AdvancedBuilderComponent[] = ADVANCED_BUILDER_COMPONENTS,
  familyLabels: Record<AdvancedBuilderFamily, string> = ADVANCED_BUILDER_FAMILY_LABELS,
  statusLabels: Record<string, string> = {
    compile_ready: "可编译",
    conditional: "有条件开放",
    draft_only: "仅草稿",
  },
): Node<AdvancedBuilderNodeData> {
  const component = getComponentOrThrowFromCatalog(componentId, components);
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
      familyLabel: familyLabels[component.family],
      status: component.status,
      statusLabel: statusLabels[component.status],
      description: component.description,
      schemaPath: component.schemaPath,
      notes: component.notes || [],
    },
  };
}

export function buildBlueprintGraph(
  blueprintId: string,
  blueprints: AdvancedBuilderBlueprint[] = ADVANCED_BUILDER_BLUEPRINTS,
  components: AdvancedBuilderComponent[] = ADVANCED_BUILDER_COMPONENTS,
  connectionRules: AdvancedBuilderConnectionRule[] = ADVANCED_BUILDER_CONNECTION_RULES,
  familyLabels: Record<AdvancedBuilderFamily, string> = ADVANCED_BUILDER_FAMILY_LABELS,
  statusLabels: Record<string, string> = {
    compile_ready: "可编译",
    conditional: "有条件开放",
    draft_only: "仅草稿",
  },
): {
  blueprint: AdvancedBuilderBlueprint;
  nodes: Array<Node<AdvancedBuilderNodeData>>;
  edges: Edge[];
} {
  const blueprint = getBlueprintOrThrowFromCatalog(blueprintId, blueprints);
  const nodes = blueprint.components.map((componentId, index) =>
    createBuilderNode(componentId, index, components, familyLabels, statusLabels),
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
        connectionRules.find(
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
  connectionRules: AdvancedBuilderConnectionRule[] = ADVANCED_BUILDER_CONNECTION_RULES,
  familyLabels: Record<AdvancedBuilderFamily, string> = ADVANCED_BUILDER_FAMILY_LABELS,
): {
  allowed: boolean;
  status: "required" | "conditional" | "blocked" | "unsupported";
  description: string;
} {
  const rule = connectionRules.find(
    (item) => item.fromFamily === fromFamily && item.toFamily === toFamily,
  );

  if (!rule) {
    return {
      allowed: false,
      status: "unsupported",
      description: `当前没有定义 ${familyLabels[fromFamily]} -> ${familyLabels[toFamily]} 的正式版连接规则。`,
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
  connectionRules: AdvancedBuilderConnectionRule[] = ADVANCED_BUILDER_CONNECTION_RULES,
  requiredFamilies: AdvancedBuilderFamily[] = REQUIRED_FAMILIES,
): AdvancedBuilderEvaluation {
  const familyCounts = new Map<AdvancedBuilderFamily, number>();
  for (const node of nodes) {
    familyCounts.set(node.data.family, (familyCounts.get(node.data.family) || 0) + 1);
  }
  const familySet = new Set(nodes.map((node) => node.data.family));
  const missingFamilies = requiredFamilies.filter((family) => !familySet.has(family));
  const duplicateFamilies = Array.from(familyCounts.entries())
    .filter(([, count]) => count > 1)
    .map(([family]) => family);

  const missingRequiredConnections = connectionRules.filter(
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
      duplicateFamilies.length === 0 &&
      missingRequiredConnections.length === 0 &&
      draftOnlyComponents.length === 0,
    missingFamilies,
    duplicateFamilies,
    missingRequiredConnections,
    conditionalComponents,
    draftOnlyComponents,
  };
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
