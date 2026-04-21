import { describe, expect, it } from "vitest"

import {
  DEFAULT_UI_PREFERENCES,
  buildMigratedPreferences,
} from "@/utils/legacyPreferenceMigration"

describe("legacy preference migration", () => {
  it("migrates legacy browser settings when machine preferences are still default", () => {
    const result = buildMigratedPreferences(DEFAULT_UI_PREFERENCES, {
      language: "en",
      themeMode: "dark",
    })

    expect(result).toEqual({
      history_display_mode: "friendly",
      language: "en",
      theme_mode: "dark",
    })
  })

  it("does not override non-default machine preferences", () => {
    const result = buildMigratedPreferences(
      {
        history_display_mode: "technical",
        language: "zh",
        theme_mode: "light",
      },
      {
        language: "en",
        themeMode: "dark",
      },
    )

    expect(result).toBeNull()
  })
})
