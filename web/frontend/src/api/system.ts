import api from "./index"

export interface UIPreferences {
  history_display_mode: "friendly" | "technical"
  language: "zh" | "en"
  theme_mode: "light" | "dark" | "auto"
}

export interface SystemResources {
  cpu: {
    usage_percent: number
    count: number
  }
  memory: {
    used: number
    total: number
    percent: number
  }
  gpu: Array<{
    id: number
    name: string
    memory_allocated: number
    memory_total: number
    memory_reserved: number
  }>
}

export const getSystemResources = async (): Promise<SystemResources> => {
  const response = await api.get("/system/resources")
  return response.data
}

export const getUIPreferences = async (): Promise<{
  preferences: UIPreferences
  storage: string
  path: string
  history_display_scope: string
}> => {
  const response = await api.get("/system/preferences")
  return response.data
}

export const updateUIPreferences = async (
  preferences: UIPreferences,
): Promise<{
  preferences: UIPreferences
  storage: string
  path: string
  history_display_scope: string
}> => {
  const response = await api.put("/system/preferences", preferences)
  return response.data
}

export const resetUIPreferences = async (): Promise<{
  preferences: UIPreferences
  storage: string
  path: string
  history_display_scope: string
}> => {
  const response = await api.delete("/system/preferences")
  return response.data
}
