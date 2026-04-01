import { describe, expect, it } from "vitest";

describe("start experience contract", () => {
  it("defines a guided onboarding payload", async () => {
    const startExperience = await import("@/config/startExperience").catch(
      () => null,
    );

    expect(startExperience?.DEFAULT_START_ROUTE).toBe("/start");
    expect(startExperience?.START_CHECKS).toHaveLength(4);
    expect(startExperience?.START_NEXT_STEPS).toEqual([
      "quickstart",
      "bring-your-own-yaml",
    ]);
  });
});
