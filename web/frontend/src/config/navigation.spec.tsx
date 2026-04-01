import { describe, expect, it } from "vitest";

import {
  NAVIGATION_ITEMS,
  PRIMARY_ENTRY_COMMAND,
  getCurrentNavigation,
} from "@/config/navigation";

const translate = (key: string) => key;

describe("navigation config", () => {
  it("keeps medfusion start as the primary entry command", () => {
    expect(PRIMARY_ENTRY_COMMAND).toBe("uv run medfusion start");
  });

  it("includes a dedicated start page as the first navigation item", () => {
    expect(NAVIGATION_ITEMS[0]?.path).toBe("/start");
  });

  it("treats /start as its own navigation context", () => {
    const current = getCurrentNavigation("/start", translate);

    expect(current.path).toBe("/start");
    expect(current.label).toBe("nav.start");
  });
});
