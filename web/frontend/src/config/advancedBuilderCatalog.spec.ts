import { describe, expect, it } from "vitest";

import {
  ADVANCED_BUILDER_BLUEPRINTS,
  ADVANCED_BUILDER_COMPONENTS,
  ADVANCED_BUILDER_CONNECTION_RULES,
} from "@/config/advancedBuilderCatalog";

describe("advanced builder catalog", () => {
  it("defines both compile-ready and draft-only components", () => {
    const statuses = new Set(
      ADVANCED_BUILDER_COMPONENTS.map((component) => component.status),
    );

    expect(statuses.has("compile_ready")).toBe(true);
    expect(statuses.has("draft_only")).toBe(true);
  });

  it("defines explicit blocked connection rules", () => {
    expect(
      ADVANCED_BUILDER_CONNECTION_RULES.some((rule) => rule.status === "blocked"),
    ).toBe(true);
  });

  it("keeps at least one compile-ready blueprint for the formal release path", () => {
    expect(
      ADVANCED_BUILDER_BLUEPRINTS.some(
        (blueprint) => blueprint.status === "compile_ready",
      ),
    ).toBe(true);
  });
});
