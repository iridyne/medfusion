import { describe, expect, it } from "vitest";

import {
  buildTrainingPrefillQuery,
  consumeTrainingLaunchParams,
  parseTrainingPrefillParams,
} from "@/utils/trainingPrefill";

describe("training prefill helpers", () => {
  const backboneOptions = ["resnet18", "resnet50", "vit_b16"] as const;

  it("builds query string from run wizard payload", () => {
    const query = buildTrainingPrefillQuery({
      experimentName: "baseline-run",
      backbone: "resnet50",
      numClasses: 2,
      epochs: 12,
      batchSize: 8,
      learningRate: 0.001,
    });
    const params = new URLSearchParams(query);

    expect(params.get("action")).toBe("start");
    expect(params.get("experimentName")).toBe("baseline-run");
    expect(params.get("backbone")).toBe("resnet50");
    expect(params.get("numClasses")).toBe("2");
    expect(params.get("epochs")).toBe("12");
    expect(params.get("batchSize")).toBe("8");
    expect(params.get("learningRate")).toBe("0.001");
  });

  it("parses valid prefill fields", () => {
    const params = new URLSearchParams(
      "action=start&experimentName=run-a&backbone=vit_b16&numClasses=3&epochs=5&batchSize=4&learningRate=0.0005",
    );
    const parsed = parseTrainingPrefillParams(params, backboneOptions);

    expect(parsed).toEqual({
      experimentName: "run-a",
      backbone: "vit_b16",
      numClasses: 3,
      epochs: 5,
      batchSize: 4,
      learningRate: 0.0005,
    });
  });

  it("drops invalid numeric values and unsupported backbone", () => {
    const params = new URLSearchParams(
      "action=start&experimentName=run-b&backbone=invalid&numClasses=0&epochs=-1&batchSize=abc&learningRate=0",
    );
    const parsed = parseTrainingPrefillParams(params, backboneOptions);

    expect(parsed).toEqual({
      experimentName: "run-b",
    });
  });

  it("consumes guided start source and strips launch params from query", () => {
    const params = new URLSearchParams(
      "action=start&source=guided-start&experimentName=run-c&backbone=resnet18&keep=1",
    );

    const consumed = consumeTrainingLaunchParams(params, backboneOptions);

    expect(consumed.source).toBe("guided-start");
    expect(consumed.prefill).toEqual({
      experimentName: "run-c",
      backbone: "resnet18",
    });
    expect(consumed.nextSearchParams.toString()).toBe("keep=1");
  });
});
