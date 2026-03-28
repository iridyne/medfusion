import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Button, Drawer, Grid, Segmented, Space } from "antd";
import {
  DesktopOutlined,
  MenuOutlined,
  MoonOutlined,
  RightOutlined,
  SunOutlined,
} from "@ant-design/icons";
import { useTranslation } from "react-i18next";

import Sidebar from "@/components/Sidebar";
import {
  NAVIGATION_ITEMS,
  PRIMARY_ENTRY_COMMAND,
  getCurrentNavigation,
} from "@/config/navigation";
import type { ThemeMode } from "@/theme/config";

interface AppShellProps {
  children: ReactNode;
  currentPath: string;
  themeMode: ThemeMode;
  resolvedTheme: "light" | "dark";
  onThemeChange: (mode: ThemeMode) => void;
}

export default function AppShell({
  children,
  currentPath,
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

  const contextualNotes = useMemo(
    () => [
      t("shell.notes.viewCount", { count: NAVIGATION_ITEMS.length }),
      resolvedTheme === "dark"
        ? t("shell.notes.themeDark")
        : t("shell.notes.themeLight"),
      t("shell.notes.cliBridge"),
    ],
    [resolvedTheme, t],
  );

  const headerControls = (
    <div className="app-shell__header-actions">
      {!isDesktop ? (
        <Button
          className="app-shell__menu-button"
          icon={<MenuOutlined />}
          onClick={() => setDrawerOpen(true)}
        >
          导航
        </Button>
      ) : null}

      <div className="app-shell__command-pill">
        <span>{t("shell.entryCommandLabel")}</span>
        <code>{PRIMARY_ENTRY_COMMAND}</code>
      </div>

      <div className="app-shell__theme-control">
        <span>{t("shell.theme.label")}</span>
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
      {isDesktop ? (
        <aside className="app-shell__sidebar">
          <Sidebar />
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
          <Sidebar onNavigate={() => setDrawerOpen(false)} />
        </Drawer>
      )}

      <main className="app-shell__main">
        <header className="app-shell__header">
          <div className="app-shell__masthead surface-frame">
            <div className="app-shell__masthead-copy">
              <span className="app-shell__eyebrow">{t("shell.masthead.kicker")}</span>
              <div className="app-shell__page-title-row">
                <strong>{currentPage.label}</strong>
                <span className="app-shell__theme-state">
                  {resolvedTheme === "dark"
                    ? t("shell.themeState.dark")
                    : t("shell.themeState.light")}
                </span>
              </div>
              <p>{currentPage.description}</p>
            </div>

            <div className="app-shell__masthead-rail">
              <div className="app-shell__masthead-path">
                <span>{t("shell.masthead.pathPrefix")}</span>
                <RightOutlined />
                <strong>{currentPage.eyebrow}</strong>
              </div>
              <div className="app-shell__masthead-notes">
                {contextualNotes.map((note) => (
                  <span key={note} className="page-chip is-neutral">
                    {note}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="app-shell__header-panel surface-frame">
            <div className="app-shell__page-copy">
              <span className="app-shell__eyebrow">{currentPage.eyebrow}</span>
              <div className="app-shell__page-title-row">
                <strong>{t("shell.header.title")}</strong>
              </div>
              <p>{t("shell.header.description")}</p>
            </div>
            {headerControls}
          </div>
        </header>

        <section className="app-shell__content">
          <div className="app-shell__viewport">{children}</div>
        </section>
      </main>
    </div>
  );
}
