import type { CSSProperties } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Space } from "antd";
import { useTranslation } from "react-i18next";

import {
  CLI_EXECUTION_COMMAND,
  PRIMARY_ENTRY_COMMAND,
  getResolvedNavigationItems,
} from "@/config/navigation";

interface SidebarProps {
  onNavigate?: () => void;
}

export default function Sidebar({ onNavigate }: SidebarProps) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const navigationItems = getResolvedNavigationItems(t);

  const handleNavigate = (path: string) => {
    navigate(path);
    onNavigate?.();
  };

  return (
    <div className="app-sidebar">
      <div className="app-sidebar__brand-band">
        <div className="app-sidebar__brand">
          <div className="app-sidebar__brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <div className="app-sidebar__brand-name">MedFusion OSS</div>
            <div className="app-sidebar__brand-caption">{t("sidebar.brandCaption")}</div>
          </div>
        </div>

        <div className="app-sidebar__manifest">
          <span className="app-sidebar__manifest-kicker">{t("sidebar.manifest.kicker")}</span>
          <strong>{t("sidebar.manifest.title")}</strong>
          <p>{t("sidebar.manifest.description")}</p>
        </div>
      </div>

      <div className="app-sidebar__cluster">
        <div className="app-sidebar__cluster-row">
          <span className="app-sidebar__cluster-label">{t("sidebar.cluster.label")}</span>
          <span className="app-sidebar__cluster-count">
            {t("sidebar.cluster.count", { count: navigationItems.length })}
          </span>
        </div>
        <div className="app-sidebar__nav-list">
          {navigationItems.map((item, index) => {
            const active =
              location.pathname === item.path ||
              location.pathname.startsWith(`${item.path}/`);

            return (
              <button
                key={item.path}
                type="button"
                className={`nav-card ${active ? "is-active" : ""}`}
                style={{ "--nav-accent": item.accent } as CSSProperties}
                onClick={() => handleNavigate(item.path)}
              >
                <span className="nav-card__index">{String(index + 1).padStart(2, "0")}</span>
                <span className="nav-card__icon">{item.icon}</span>
                <span className="nav-card__copy">
                  <strong>{item.shortLabel}</strong>
                  <small>{item.description}</small>
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="app-sidebar__support surface-frame">
        <span className="app-sidebar__cluster-label">{t("sidebar.support.label")}</span>
        <h3>{t("sidebar.support.title")}</h3>
        <p>{t("sidebar.support.description")}</p>

        <Space direction="vertical" size={10} style={{ width: "100%" }}>
          <div className="sidebar-command">
            <span>{t("sidebar.commands.entry")}</span>
            <code>{PRIMARY_ENTRY_COMMAND}</code>
          </div>
          <div className="sidebar-command">
            <span>{t("sidebar.commands.execution")}</span>
            <code>{CLI_EXECUTION_COMMAND}</code>
          </div>
        </Space>

        <div className="app-sidebar__footer-note">
          <span>{t("sidebar.footer.researchGrade")}</span>
          <span>{t("sidebar.footer.localFirst")}</span>
          <span>{t("sidebar.footer.artifactAware")}</span>
        </div>
      </div>
    </div>
  );
}
