import { describe, expect, it } from "vitest";

describe("quickstart run contract", () => {
  it("defines one recommended quickstart profile and four core stages", async () => {
    const config = await import("@/config/quickstartRun");

    expect(config.RECOMMENDED_QUICKSTART_PROFILE.id).toBe(
      "medmnist-breastmnist",
    );
    expect(config.QUICKSTART_STAGES.map((stage) => stage.key)).toEqual([
      "prepare",
      "validate-config",
      "train",
      "build-results",
    ]);
    expect(config.QUICKSTART_RESULT_PATHS).toEqual([
      "outputs/public_datasets/breastmnist_quickstart/checkpoints/best.pth",
      "outputs/public_datasets/breastmnist_quickstart/metrics/metrics.json",
      "outputs/public_datasets/breastmnist_quickstart/metrics/validation.json",
      "outputs/public_datasets/breastmnist_quickstart/reports/summary.json",
      "outputs/public_datasets/breastmnist_quickstart/reports/report.md",
    ]);
  });
});
