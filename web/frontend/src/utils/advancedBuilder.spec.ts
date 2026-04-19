import { describe, expect, it } from "vitest";

import {
  buildBlueprintGraph,
  canConnectFamilies,
  compileAdvancedBuilderGraphToRunSpec,
  createBuilderNode,
  evaluateAdvancedBuilderGraph,
} from "@/utils/advancedBuilder";

describe("advancedBuilder utils", () => {
  it("builds a compile-ready graph for the quickstart blueprint", () => {
    const graph = buildBlueprintGraph("quickstart_multimodal");
    const evaluation = evaluateAdvancedBuilderGraph(graph.nodes, graph.edges);

    expect(graph.nodes.length).toBeGreaterThan(0);
    expect(graph.edges.length).toBeGreaterThan(0);
    expect(evaluation.compileReady).toBe(true);
    expect(evaluation.missingFamilies).toEqual([]);
  });

  it("blocks invalid connections across unsupported families", () => {
    const result = canConnectFamilies("data_input", "head");

    expect(result.allowed).toBe(false);
    expect(result.status).toBe("blocked");
  });

  it("marks graphs with draft-only components as not compile-ready", () => {
    const draftNode = createBuilderNode("three_phase_ct_dataset");
    const evaluation = evaluateAdvancedBuilderGraph([draftNode], []);

    expect(evaluation.compileReady).toBe(false);
    expect(evaluation.draftOnlyComponents).toContain("三相 CT + 临床输入");
  });

  it("compiles the quickstart blueprint into a real RunSpec draft", () => {
    const graph = buildBlueprintGraph("quickstart_multimodal");
    const result = compileAdvancedBuilderGraphToRunSpec(
      graph.nodes,
      graph.edges,
    );

    expect(result.spec).not.toBeNull();
    expect(result.preset).toBe("quickstart");
    expect(result.spec?.model.vision.backbone).toBe("resnet18");
    expect(result.spec?.model.fusion.fusionType).toBe("concatenate");
    expect(result.spec?.training.useProgressiveTraining).toBe(false);
  });

  it("rejects graphs with duplicate required families", () => {
    const graph = buildBlueprintGraph("quickstart_multimodal");
    const duplicateBackbone = createBuilderNode("efficientnet_b0_backbone", 99);
    const result = compileAdvancedBuilderGraphToRunSpec(
      [...graph.nodes, duplicateBackbone],
      graph.edges,
    );

    expect(result.spec).toBeNull();
    expect(result.issues.some((issue) => issue.level === "error")).toBe(true);
  });
});
