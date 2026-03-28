import { describe, expect, it } from "vitest";

import {
  buildWorkbenchFallbackSearch,
  consumeWorkbenchFallback,
  isWorkbenchFallbackSource,
} from "@/utils/workbenchFallback";

describe("workbench fallback helpers", () => {
  it("builds fallback query for known source", () => {
    expect(buildWorkbenchFallbackSearch("workflow")).toBe("from=workflow");
    expect(buildWorkbenchFallbackSearch("preprocessing")).toBe(
      "from=preprocessing",
    );
  });

  it("consumes and strips valid fallback source", () => {
    const params = new URLSearchParams("from=experiments&keep=1");
    const { source, nextSearchParams } = consumeWorkbenchFallback(params);

    expect(source).toBe("experiments");
    expect(nextSearchParams.toString()).toBe("keep=1");
    expect(params.toString()).toBe("from=experiments&keep=1");
  });

  it("ignores unknown source and still strips from query", () => {
    const params = new URLSearchParams("from=unknown&keep=1");
    const { source, nextSearchParams } = consumeWorkbenchFallback(params);

    expect(source).toBeNull();
    expect(nextSearchParams.toString()).toBe("keep=1");
  });

  it("recognizes allowed fallback sources", () => {
    expect(isWorkbenchFallbackSource("workflow")).toBe(true);
    expect(isWorkbenchFallbackSource("settings")).toBe(true);
    expect(isWorkbenchFallbackSource("random")).toBe(false);
    expect(isWorkbenchFallbackSource(null)).toBe(false);
  });
});
