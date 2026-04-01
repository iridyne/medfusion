import { describe, expect, it } from "vitest";

describe("workbench experience contract", () => {
  it("defines workbench as a post-run medium overview", async () => {
    const experience = await import("@/config/workbenchExperience").catch(
      () => null,
    );

    expect(experience?.WORKBENCH_MODE).toBe("post-run-overview");
    expect(experience?.WORKBENCH_PRIMARY_CARDS).toEqual([
      "latest-run",
      "key-metric",
      "next-steps",
    ]);
  });
});
