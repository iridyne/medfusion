import type { CSSProperties } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";

import {
  type NavigationMode,
  getResolvedNavigationItems,
} from "@/config/navigation";

interface SidebarProps {
  onNavigate?: () => void;
  navigationMode?: NavigationMode;
}

export default function Sidebar({
  onNavigate,
  navigationMode = "full",
}: SidebarProps) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const navigationItems = getResolvedNavigationItems(t, navigationMode);

  const handleNavigate = (path: string) => {
    navigate(path);
    onNavigate?.();
  };

  return (
    <div className="app-sidebar">
      <div className="app-sidebar__header">
        <div className="app-sidebar__header-main">
          <strong>研究导航</strong>
        </div>
      </div>

      <div className="app-sidebar__nav-list">
        {navigationItems.map((item) => {
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
              <span className="nav-card__icon">{item.icon}</span>
              <span className="nav-card__copy">
                <strong>
                  {item.shortLabel}
                  {active ? <span className="nav-card__active-dot" aria-hidden="true" /> : null}
                </strong>
                <small>{item.description}</small>
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
