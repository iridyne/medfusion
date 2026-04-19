import { describe, expect, it } from "vitest";

import {
  buildTrainingMonitorLink,
  buildTrainingResultLink,
  clearTrainingResultHandoffParams,
  getTrainingResultState,
  parseTrainingResultHandoff,
  type TrainingJob,
} from "@/api/training";

const baseJob: TrainingJob = {
  id: "job-42",
  name: "lung-risk-run",
  status: "completed",
  progress: 100,
  epoch: 12,
  totalEpochs: 12,
  loss: 0.128,
  accuracy: 0.91,
  startTime: "2026-04-19T12:00:00Z",
};

describe("training result handoff helpers", () => {
  it("classifies result readiness from training jobs", () => {
    expect(
      getTrainingResultState({
        status: "completed",
        resultModelId: 17,
      }),
    ).toBe("ready");

    expect(
      getTrainingResultState({
        status: "completed",
        resultModelId: undefined,
      }),
    ).toBe("pending");

    expect(
      getTrainingResultState({
        status: "running",
        resultModelId: undefined,
      }),
    ).toBe("unavailable");
  });

  it("builds a model-library deep link with training context", () => {
    const link = buildTrainingResultLink({
      ...baseJob,
      resultModelId: 17,
      resultModelName: "lung-risk-run-best",
    });

    expect(link).toBe(
      "/models?model=17&source=training-monitor&job=job-42&jobName=lung-risk-run&result=lung-risk-run-best",
    );
  });

  it("round-trips training result handoff params and preserves unrelated params when clearing", () => {
    const params = new URLSearchParams(
      "model=17&source=training-monitor&job=job-42&jobName=lung-risk-run&result=lung-risk-run-best&view=detail",
    );

    expect(parseTrainingResultHandoff(params)).toEqual({
      modelId: 17,
      source: "training-monitor",
      jobId: "job-42",
      jobName: "lung-risk-run",
      resultModelName: "lung-risk-run-best",
    });

    expect(clearTrainingResultHandoffParams(params).toString()).toBe("view=detail");
  });

  it("builds a return link back to the training monitor", () => {
    expect(buildTrainingMonitorLink("job-42")).toBe(
      "/training?job=job-42&source=model-library",
    );
  });
});
