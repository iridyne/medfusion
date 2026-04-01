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
  loadThemeMode,
  saveThemeMode,
  watchSystemTheme,
} from "./theme/config";
import { getCurrentNavigation } from "./config/navigation";
import { buildWorkbenchFallbackSearch } from "./utils/workbenchFallback";
import "./i18n/config";
import "./App.css";

const Workbench = lazy(() => import("./pages/Workbench"));
const GettingStarted = lazy(() => import("./pages/GettingStarted"));
const QuickstartRun = lazy(() => import("./pages/QuickstartRun"));
const DatasetManager = lazy(() => import("./pages/DatasetManager"));
const TrainingMonitor = lazy(() => import("./pages/TrainingMonitor"));
const ModelLibrary = lazy(() => import("./pages/ModelLibrary"));
const SystemMonitor = lazy(() => import("./pages/SystemMonitor"));
const RunWizard = lazy(() => import("./pages/RunWizard"));

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
  const [themeMode, setThemeMode] = useState<ThemeMode>(loadThemeMode());
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
    document.title = getDocumentTitle(location.pathname, t);
  }, [location.pathname, i18n.language, t]);

  const getAntdLocale = () => {
    return i18n.language === "en" ? enUS : zhCN;
  };

  const handleThemeChange = (mode: ThemeMode) => {
    saveThemeMode(mode);
    setThemeMode(mode);
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
            <Route path="/training" element={<TrainingMonitor />} />
            <Route path="/config" element={<RunWizard />} />
            <Route path="/models" element={<ModelLibrary />} />
            <Route path="/system" element={<SystemMonitor />} />
            <Route
              path="/workflow"
              element={
                <Navigate
                  to={`/workbench?${buildWorkbenchFallbackSearch("workflow")}`}
                  replace
                />
              }
            />
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
            <Route
              path="/settings"
              element={
                <Navigate
                  to={`/workbench?${buildWorkbenchFallbackSearch("settings")}`}
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
