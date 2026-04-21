import { Suspense, lazy, useEffect, useState } from "react";
import { Routes, Route, Navigate, useLocation } from "react-router-dom";
import { ConfigProvider, Spin } from "antd";
import zhCN from "antd/locale/zh_CN";
import enUS from "antd/locale/en_US";
import { useTranslation } from "react-i18next";

import AppShell from "./components/layout/AppShell";
import {
  ThemeMode,
  getCurrentTheme,
  getSystemTheme,
  watchSystemTheme,
} from "./theme/config";
import { getUIPreferences, updateUIPreferences } from "./api/system";
import { getCurrentNavigation } from "./config/navigation";
import { buildWorkbenchFallbackSearch } from "./utils/workbenchFallback";
import {
  buildMigratedPreferences,
  clearLegacyPreferences,
  extractLegacyPreferences,
} from "./utils/legacyPreferenceMigration";
import "./i18n/config";
import "./App.css";

const Workbench = lazy(() => import("./pages/Workbench"));
const GettingStarted = lazy(() => import("./pages/GettingStarted"));
const QuickstartRun = lazy(() => import("./pages/QuickstartRun"));
const DatasetManager = lazy(() => import("./pages/DatasetManager"));
const EvaluationWorkbench = lazy(() => import("./pages/EvaluationWorkbench"));
const TrainingMonitor = lazy(() => import("./pages/TrainingMonitor"));
const ModelLibrary = lazy(() => import("./pages/ModelLibrary"));
const SystemMonitor = lazy(() => import("./pages/SystemMonitor"));
const RunWizard = lazy(() => import("./pages/RunWizard"));
const ModelCatalog = lazy(() => import("./pages/ModelCatalog"));
const ModelCatalogCustom = lazy(() => import("./pages/ModelCatalogCustom"));
const AdvancedBuilder = lazy(() => import("./pages/AdvancedBuilder"));
const AdvancedBuilderCanvas = lazy(() => import("./pages/AdvancedBuilderCanvas"));
const ComfyUIBridge = lazy(() => import("./pages/ComfyUIBridge"));
const WorkflowEditor = lazy(() => import("./pages/WorkflowEditor"));
const Settings = lazy(() => import("./pages/Settings"));

function getDocumentTitle(pathname: string, translate: (key: string) => string) {
  if (pathname === "/quickstart-run") {
    return `Quickstart Run · MedFusion OSS`;
  }

  const currentPage = getCurrentNavigation(pathname, translate);
  return `${currentPage.label} · MedFusion OSS`;
}

function RouteFallback() {
  const { t } = useTranslation();

  return (
    <div className="route-fallback surface-frame">
      <Spin size="large" />
      <div className="route-fallback__copy">
        <strong>{t("app.routeFallback.title")}</strong>
        <p>{t("app.routeFallback.description")}</p>
      </div>
    </div>
  );
}

function App() {
  const location = useLocation();
  const { i18n, t } = useTranslation();
  const [themeMode, setThemeMode] = useState<ThemeMode>("auto");
  const [resolvedTheme, setResolvedTheme] = useState<"light" | "dark">(
    themeMode === "auto" ? getSystemTheme() : themeMode,
  );
  const [currentTheme, setCurrentTheme] = useState(getCurrentTheme(themeMode));

  useEffect(() => {
    const nextResolved = themeMode === "auto" ? getSystemTheme() : themeMode;
    setResolvedTheme(nextResolved);
    setCurrentTheme(getCurrentTheme(themeMode));
  }, [themeMode]);

  useEffect(() => {
    if (themeMode !== "auto") {
      return;
    }

    const unwatch = watchSystemTheme((nextTheme) => {
      setResolvedTheme(nextTheme);
      setCurrentTheme(getCurrentTheme("auto"));
    });

    return unwatch;
  }, [themeMode]);

  useEffect(() => {
    document.documentElement.dataset.appTheme = resolvedTheme;
    document.body.dataset.appTheme = resolvedTheme;
  }, [resolvedTheme]);

  useEffect(() => {
    const load = async () => {
      try {
        const payload = await getUIPreferences();
        const migratedPreferences = buildMigratedPreferences(
          payload.preferences,
          extractLegacyPreferences(),
        );
        const effectivePreferences = migratedPreferences
          ? (await updateUIPreferences(migratedPreferences)).preferences
          : payload.preferences;

        if (migratedPreferences) {
          clearLegacyPreferences();
        }

        setThemeMode(effectivePreferences.theme_mode);
        if (i18n.language !== effectivePreferences.language) {
          await i18n.changeLanguage(effectivePreferences.language);
        }
      } catch (error) {
        console.error("Failed to load machine UI preferences:", error);
      }
    };

    void load();
  }, [i18n]);

  useEffect(() => {
    document.title = getDocumentTitle(location.pathname, t);
  }, [location.pathname, i18n.language, t]);

  const getAntdLocale = () => {
    return i18n.language === "en" ? enUS : zhCN;
  };

  const handleThemeChange = async (mode: ThemeMode) => {
    setThemeMode(mode);
    try {
      const payload = await getUIPreferences();
      await updateUIPreferences({
        ...payload.preferences,
        theme_mode: mode,
      });
    } catch (error) {
      console.error("Failed to persist machine theme preference:", error);
    }
  };

  return (
    <ConfigProvider theme={currentTheme} locale={getAntdLocale()}>
      <AppShell
        currentPath={location.pathname}
        themeMode={themeMode}
        resolvedTheme={resolvedTheme}
        onThemeChange={handleThemeChange}
      >
        <Suspense fallback={<RouteFallback />}>
          <Routes>
            <Route path="/" element={<Navigate to="/start" replace />} />
            <Route path="/start" element={<GettingStarted />} />
            <Route path="/quickstart-run" element={<QuickstartRun />} />
            <Route path="/workbench" element={<Workbench />} />
            <Route path="/datasets" element={<DatasetManager />} />
            <Route path="/evaluation" element={<EvaluationWorkbench />} />
            <Route path="/training" element={<TrainingMonitor />} />
            <Route path="/config" element={<RunWizard />} />
            <Route path="/config/model" element={<ModelCatalog />} />
            <Route path="/config/model/custom" element={<ModelCatalogCustom />} />
            <Route path="/config/advanced" element={<AdvancedBuilder />} />
            <Route path="/config/advanced/canvas" element={<AdvancedBuilderCanvas />} />
            <Route path="/config/comfyui" element={<ComfyUIBridge />} />
            <Route path="/workflow" element={<WorkflowEditor />} />
            <Route path="/models" element={<ModelLibrary />} />
            <Route path="/system" element={<SystemMonitor />} />
            <Route path="/settings" element={<Settings />} />
            <Route
              path="/experiments"
              element={
                <Navigate
                  to={`/workbench?${buildWorkbenchFallbackSearch("experiments")}`}
                  replace
                />
              }
            />
            <Route
              path="/preprocessing"
              element={
                <Navigate
                  to={`/workbench?${buildWorkbenchFallbackSearch("preprocessing")}`}
                  replace
                />
              }
            />
            <Route path="*" element={<Navigate to="/start" replace />} />
          </Routes>
        </Suspense>
      </AppShell>
    </ConfigProvider>
  );
}

export default App;
