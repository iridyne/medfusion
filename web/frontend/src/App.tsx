import { useState, useEffect } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { Layout, ConfigProvider } from "antd";
import zhCN from "antd/locale/zh_CN";
import enUS from "antd/locale/en_US";
import { useTranslation } from "react-i18next";
import DatasetManager from "./pages/DatasetManager";
import TrainingMonitor from "./pages/TrainingMonitor";
import ModelLibrary from "./pages/ModelLibrary";
import SystemMonitor from "./pages/SystemMonitor";
import Workbench from "./pages/Workbench";
import RunWizard from "./pages/RunWizard";
import Sidebar from "./components/Sidebar";
import {
  ThemeMode,
  getCurrentTheme,
  loadThemeMode,
  watchSystemTheme,
} from "./theme/config";
import "./i18n/config";
import "./App.css";

const { Content } = Layout;

function App() {
  const { i18n } = useTranslation();
  const [themeMode, setThemeMode] = useState<ThemeMode>(loadThemeMode());
  const [currentTheme, setCurrentTheme] = useState(getCurrentTheme(themeMode));

  // 监听系统主题变化（仅在 auto 模式下）
  useEffect(() => {
    if (themeMode === "auto") {
      const unwatch = watchSystemTheme(() => {
        setCurrentTheme(getCurrentTheme("auto"));
      });
      return unwatch;
    }
  }, [themeMode]);

  // 主题变化时更新
  useEffect(() => {
    setCurrentTheme(getCurrentTheme(themeMode));
  }, [themeMode]);

  // 获取 Ant Design 语言包
  const getAntdLocale = () => {
    return i18n.language === "en" ? enUS : zhCN;
  };

  const handleThemeChange = (mode: ThemeMode) => {
    setThemeMode(mode);
  };

  return (
    <ConfigProvider theme={currentTheme} locale={getAntdLocale()}>
      <Layout style={{ height: "100vh" }}>
        <Sidebar />
        <Layout>
          <Content style={{ overflow: "auto" }}>
            <Routes>
              <Route path="/" element={<Navigate to="/workbench" replace />} />
              <Route path="/workbench" element={<Workbench />} />
              <Route path="/datasets" element={<DatasetManager />} />
              <Route path="/training" element={<TrainingMonitor />} />
              <Route path="/config" element={<RunWizard />} />
              <Route path="/models" element={<ModelLibrary />} />
              <Route path="/system" element={<SystemMonitor />} />
              <Route path="/workflow" element={<Navigate to="/workbench" replace />} />
              <Route path="/experiments" element={<Navigate to="/workbench" replace />} />
              <Route path="/preprocessing" element={<Navigate to="/workbench" replace />} />
              <Route path="/settings" element={<Navigate to="/workbench" replace />} />
              <Route path="*" element={<Navigate to="/workbench" replace />} />
            </Routes>
          </Content>
        </Layout>
      </Layout>
    </ConfigProvider>
  );
}

export default App;
