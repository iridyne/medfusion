import { useState, useEffect } from "react";
import { Routes, Route } from "react-router-dom";
import { Layout, ConfigProvider } from "antd";
import zhCN from "antd/locale/zh_CN";
import enUS from "antd/locale/en_US";
import { useTranslation } from "react-i18next";
import WorkflowEditor from "./pages/WorkflowEditor";
import ConfigGenerator from "./pages/ConfigGenerator";
import DatasetManager from "./pages/DatasetManager";
import TrainingMonitor from "./pages/TrainingMonitor";
import ModelLibrary from "./pages/ModelLibrary";
import Preprocessing from "./pages/Preprocessing";
import SystemMonitor from "./pages/SystemMonitor";
import Settings from "./pages/Settings";
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
              <Route path="/" element={<WorkflowEditor />} />
              <Route path="/workflow" element={<WorkflowEditor />} />
              <Route path="/config" element={<ConfigGenerator />} />
              <Route path="/datasets" element={<DatasetManager />} />
              <Route path="/training" element={<TrainingMonitor />} />
              <Route path="/models" element={<ModelLibrary />} />
              <Route path="/preprocessing" element={<Preprocessing />} />
              <Route path="/system" element={<SystemMonitor />} />
              <Route
                path="/settings"
                element={<Settings onThemeChange={handleThemeChange} />}
              />
            </Routes>
          </Content>
        </Layout>
      </Layout>
    </ConfigProvider>
  );
}

export default App;
