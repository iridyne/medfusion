import { useEffect, useState, type ReactNode } from "react";
import { Button, Drawer, Grid, Segmented } from "antd";
import {
  DesktopOutlined,
  MenuOutlined,
  MoonOutlined,
  SunOutlined,
} from "@ant-design/icons";
import { useTranslation } from "react-i18next";

import Sidebar from "@/components/Sidebar";
import {
  type NavigationMode,
  getCurrentNavigation,
} from "@/config/navigation";
import type { ThemeMode } from "@/theme/config";

interface AppShellProps {
  children: ReactNode;
  currentPath: string;
  navigationMode: NavigationMode;
  themeMode: ThemeMode;
  resolvedTheme: "light" | "dark";
  onThemeChange: (mode: ThemeMode) => void;
}

export default function AppShell({
  children,
  currentPath,
  navigationMode,
  themeMode,
  resolvedTheme,
  onThemeChange,
}: AppShellProps) {
  const { t } = useTranslation();
  const screens = Grid.useBreakpoint();
  const isDesktop = Boolean(screens.lg);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const currentPage = getCurrentNavigation(currentPath, t);

  useEffect(() => {
    if (isDesktop) {
      setDrawerOpen(false);
    }
  }, [isDesktop, currentPath]);

  const topbarControls = (
    <div className="app-topbar__actions">
      {!isDesktop ? (
        <Button
          className="app-topbar__menu-button"
          icon={<MenuOutlined />}
          onClick={() => setDrawerOpen(true)}
        >
          导航
        </Button>
      ) : null}

      <div className="app-topbar__theme-control">
        <Segmented
          size="middle"
          value={themeMode}
          onChange={(value) => onThemeChange(value as ThemeMode)}
          options={[
            {
              label: (
                <span className="theme-mode-option">
                  <DesktopOutlined />
                  {t("shell.theme.auto")}
                </span>
              ),
              value: "auto",
            },
            {
              label: (
                <span className="theme-mode-option">
                  <MoonOutlined />
                  {t("shell.theme.dark")}
                </span>
              ),
              value: "dark",
            },
            {
              label: (
                <span className="theme-mode-option">
                  <SunOutlined />
                  {t("shell.theme.light")}
                </span>
              ),
              value: "light",
            },
          ]}
        />
      </div>
    </div>
  );

  return (
    <div className="app-shell">
      <header className="app-topbar surface-frame">
        <div className="app-topbar__left">
          <div className="app-topbar__brand">
            <div className="app-topbar__brand-mark" aria-hidden="true">
              <span />
              <span />
            </div>
            <div className="app-topbar__brand-copy">
              <strong>MedFusion</strong>
              <span>Clinical Research Workspace</span>
            </div>
          </div>

          <div className="app-topbar__page">
            <div className="app-topbar__page-copy">
              <strong>{currentPage.label}</strong>
            </div>
          </div>
        </div>

        <div className="app-topbar__right">
          <span className="app-topbar__runtime-state">
            {resolvedTheme === "dark"
              ? t("shell.themeState.dark")
              : t("shell.themeState.light")}
          </span>
          {topbarControls}
        </div>
      </header>

      <div className="app-shell__body">
        {isDesktop ? (
          <aside className="app-shell__sidebar">
            <Sidebar navigationMode={navigationMode} />
          </aside>
        ) : (
          <Drawer
            open={drawerOpen}
            placement="left"
            onClose={() => setDrawerOpen(false)}
            closable={false}
            width={320}
            rootClassName="surface-drawer surface-drawer--nav"
          >
            <Sidebar
              navigationMode={navigationMode}
              onNavigate={() => setDrawerOpen(false)}
            />
          </Drawer>
        )}

        <main className="app-shell__main">
          <section className="app-shell__content">
            <div className="app-shell__viewport">{children}</div>
          </section>
        </main>
      </div>
    </div>
  );
}
