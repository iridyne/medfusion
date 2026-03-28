import { theme, type ThemeConfig } from "antd";

export type ThemeMode = "light" | "dark" | "auto";

interface ThemePalette {
  primary: string;
  success: string;
  warning: string;
  error: string;
  info: string;
  base: string;
  layout: string;
  surface: string;
  surfaceElevated: string;
  text: string;
  muted: string;
  soft: string;
  border: string;
  borderSoft: string;
  fill: string;
  fillAlt: string;
  ring: string;
  shadow: string;
  shadowSoft: string;
}

function createPalette(mode: "light" | "dark"): ThemePalette {
  if (mode === "dark") {
    return {
      primary: "#6eb7aa",
      success: "#7dc98b",
      warning: "#d9b56d",
      error: "#d98278",
      info: "#78a8d8",
      base: "#071119",
      layout: "#08131d",
      surface: "#0e1b25",
      surfaceElevated: "#132633",
      text: "#f3eee2",
      muted: "#b8c4c6",
      soft: "#8e9ca0",
      border: "rgba(160, 192, 191, 0.18)",
      borderSoft: "rgba(160, 192, 191, 0.1)",
      fill: "rgba(110, 183, 170, 0.12)",
      fillAlt: "rgba(120, 168, 216, 0.1)",
      ring: "rgba(110, 183, 170, 0.32)",
      shadow: "0 28px 70px rgba(2, 9, 14, 0.42)",
      shadowSoft: "0 18px 38px rgba(2, 9, 14, 0.28)",
    };
  }

  return {
    primary: "#2d887d",
    success: "#3b8d5a",
    warning: "#a97a2a",
    error: "#b4564d",
    info: "#3d6fa0",
    base: "#f7f3eb",
    layout: "#f3efe5",
    surface: "#fffdf8",
    surfaceElevated: "#f7f2e8",
    text: "#1e2a31",
    muted: "#5b676d",
    soft: "#7e8a91",
    border: "rgba(64, 82, 92, 0.12)",
    borderSoft: "rgba(64, 82, 92, 0.08)",
    fill: "rgba(45, 136, 125, 0.09)",
    fillAlt: "rgba(61, 111, 160, 0.08)",
    ring: "rgba(45, 136, 125, 0.22)",
    shadow: "0 24px 60px rgba(42, 38, 30, 0.12)",
    shadowSoft: "0 16px 36px rgba(42, 38, 30, 0.08)",
  };
}

function createThemeConfig(mode: "light" | "dark"): ThemeConfig {
  const palette = createPalette(mode);
  const isDark = mode === "dark";

  return {
    algorithm: isDark ? theme.darkAlgorithm : theme.defaultAlgorithm,
    token: {
      colorPrimary: palette.primary,
      colorSuccess: palette.success,
      colorWarning: palette.warning,
      colorError: palette.error,
      colorInfo: palette.info,
      colorBgBase: palette.base,
      colorBgContainer: palette.surface,
      colorBgElevated: palette.surfaceElevated,
      colorBgLayout: palette.layout,
      colorTextBase: palette.text,
      colorText: palette.text,
      colorTextSecondary: palette.muted,
      colorTextTertiary: palette.soft,
      colorBorder: palette.border,
      colorBorderSecondary: palette.borderSoft,
      colorFillSecondary: palette.fill,
      colorFillTertiary: palette.fillAlt,
      borderRadius: 16,
      borderRadiusLG: 24,
      borderRadiusSM: 12,
      fontSize: 14,
      controlHeight: 40,
      controlHeightLG: 46,
      fontFamily:
        '"IBM Plex Sans", "Avenir Next", "Segoe UI", "PingFang SC", "Hiragino Sans GB", sans-serif',
      fontFamilyCode:
        '"JetBrains Mono", "SFMono-Regular", "Cascadia Code", monospace',
      controlOutline: palette.ring,
      boxShadow: palette.shadow,
      boxShadowSecondary: palette.shadowSoft,
      wireframe: false,
    },
    components: {
      Layout: {
        colorBgHeader: "transparent",
        colorBgBody: palette.layout,
        colorBgTrigger: palette.surfaceElevated,
      },
      Card: {
        colorBgContainer: palette.surface,
        boxShadow: palette.shadowSoft,
      },
      Table: {
        colorBgContainer: "transparent",
      },
      Drawer: {
        colorBgElevated: palette.surfaceElevated,
      },
      Modal: {
        colorBgElevated: palette.surfaceElevated,
      },
    },
  };
}

export const lightTheme = createThemeConfig("light");
export const darkTheme = createThemeConfig("dark");

export const getSystemTheme = (): "light" | "dark" => {
  if (typeof window === "undefined") {
    return "dark";
  }

  const darkModeQuery = window.matchMedia("(prefers-color-scheme: dark)");
  return darkModeQuery.matches ? "dark" : "light";
};

export const getCurrentTheme = (mode: ThemeMode): ThemeConfig => {
  if (mode === "auto") {
    return getSystemTheme() === "dark" ? darkTheme : lightTheme;
  }

  return mode === "dark" ? darkTheme : lightTheme;
};

export const saveThemeMode = (mode: ThemeMode): void => {
  localStorage.setItem("themeMode", mode);
};

export const loadThemeMode = (): ThemeMode => {
  const saved = localStorage.getItem("themeMode");
  if (saved === "light" || saved === "dark" || saved === "auto") {
    return saved;
  }
  return "dark";
};

export const watchSystemTheme = (
  callback: (themeMode: "light" | "dark") => void,
): (() => void) => {
  if (typeof window === "undefined") {
    return () => {};
  }

  const darkModeQuery = window.matchMedia("(prefers-color-scheme: dark)");

  const handler = (event: MediaQueryListEvent) => {
    callback(event.matches ? "dark" : "light");
  };

  darkModeQuery.addEventListener("change", handler);

  return () => {
    darkModeQuery.removeEventListener("change", handler);
  };
};
