import type { UIPreferences } from "@/api/system"
import type { ThemeMode } from "@/theme/config"

export const DEFAULT_UI_PREFERENCES: UIPreferences = {
  history_display_mode: "friendly",
  language: "zh",
  theme_mode: "auto",
}

export interface LegacyPreferenceSnapshot {
  language: string | null
  themeMode: string | null
}

export function extractLegacyPreferences(): LegacyPreferenceSnapshot {
  if (typeof window === "undefined" || typeof window.localStorage === "undefined") {
    return { language: null, themeMode: null }
  }

  return {
    language: window.localStorage.getItem("language"),
    themeMode: window.localStorage.getItem("themeMode"),
  }
}

export function buildMigratedPreferences(
  current: UIPreferences,
  legacy: LegacyPreferenceSnapshot,
): UIPreferences | null {
  const currentIsDefault =
    current.language === DEFAULT_UI_PREFERENCES.language &&
    current.theme_mode === DEFAULT_UI_PREFERENCES.theme_mode &&
    current.history_display_mode === DEFAULT_UI_PREFERENCES.history_display_mode

  if (!currentIsDefault) {
    return null
  }

  const next: UIPreferences = { ...current }
  let changed = false

  if (legacy.language === "zh" || legacy.language === "en") {
    next.language = legacy.language
    changed = true
  }

  if (
    legacy.themeMode === "light" ||
    legacy.themeMode === "dark" ||
    legacy.themeMode === "auto"
  ) {
    next.theme_mode = legacy.themeMode as ThemeMode
    changed = true
  }

  return changed ? next : null
}

export function clearLegacyPreferences(): void {
  if (typeof window === "undefined" || typeof window.localStorage === "undefined") {
    return
  }

  window.localStorage.removeItem("language")
  window.localStorage.removeItem("themeMode")
}
