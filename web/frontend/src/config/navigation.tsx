import type { ReactNode } from "react";
import {
  ControlOutlined,
  DatabaseOutlined,
  DotChartOutlined,
  ExperimentOutlined,
  HomeOutlined,
  MonitorOutlined,
  PlayCircleOutlined,
  RadarChartOutlined,
  SettingOutlined,
  ShareAltOutlined,
} from "@ant-design/icons";

type TranslateFn = (key: string) => string;

export interface NavigationItem {
  path: string;
  label: string;
  shortLabel: string;
  eyebrow: string;
  description: string;
  accent: string;
  icon: ReactNode;
}

interface NavigationMetaItem {
  path: string;
  labelKey: string;
  shortLabelKey: string;
  eyebrowKey: string;
  descriptionKey: string;
  accent: string;
  icon: ReactNode;
}

export const PRIMARY_ENTRY_COMMAND = "uv run medfusion start";
export const CLI_EXECUTION_COMMAND = "uv run medfusion train --config <yaml>";

export const NAVIGATION_ITEMS: NavigationMetaItem[] = [
  {
    path: "/start",
    labelKey: "nav.start",
    shortLabelKey: "navShort.start",
    eyebrowKey: "navEyebrow.start",
    descriptionKey: "navDescription.start",
    accent: "var(--accent-amber)",
    icon: <PlayCircleOutlined />,
  },
  {
    path: "/workbench",
    labelKey: "nav.workbench",
    shortLabelKey: "navShort.workbench",
    eyebrowKey: "navEyebrow.workbench",
    descriptionKey: "navDescription.workbench",
    accent: "var(--accent-teal)",
    icon: <HomeOutlined />,
  },
  {
    path: "/datasets",
    labelKey: "nav.datasets",
    shortLabelKey: "navShort.datasets",
    eyebrowKey: "navEyebrow.datasets",
    descriptionKey: "navDescription.datasets",
    accent: "var(--accent-blue)",
    icon: <DatabaseOutlined />,
  },
  {
    path: "/training",
    labelKey: "nav.training",
    shortLabelKey: "navShort.training",
    eyebrowKey: "navEyebrow.training",
    descriptionKey: "navDescription.training",
    accent: "var(--accent-rose)",
    icon: <ExperimentOutlined />,
  },
  {
    path: "/evaluation",
    labelKey: "nav.evaluation",
    shortLabelKey: "navShort.evaluation",
    eyebrowKey: "navEyebrow.evaluation",
    descriptionKey: "navDescription.evaluation",
    accent: "var(--accent-teal)",
    icon: <DotChartOutlined />,
  },
  {
    path: "/config",
    labelKey: "nav.config",
    shortLabelKey: "navShort.config",
    eyebrowKey: "navEyebrow.config",
    descriptionKey: "navDescription.config",
    accent: "var(--accent-amber)",
    icon: <ControlOutlined />,
  },
  {
    path: "/workflow",
    labelKey: "nav.workflow",
    shortLabelKey: "navShort.workflow",
    eyebrowKey: "navEyebrow.workflow",
    descriptionKey: "navDescription.workflow",
    accent: "var(--accent-rose)",
    icon: <ShareAltOutlined />,
  },
  {
    path: "/models",
    labelKey: "nav.models",
    shortLabelKey: "navShort.models",
    eyebrowKey: "navEyebrow.models",
    descriptionKey: "navDescription.models",
    accent: "var(--accent-teal)",
    icon: <RadarChartOutlined />,
  },
  {
    path: "/system",
    labelKey: "nav.system",
    shortLabelKey: "navShort.system",
    eyebrowKey: "navEyebrow.system",
    descriptionKey: "navDescription.system",
    accent: "var(--accent-blue)",
    icon: <MonitorOutlined />,
  },
  {
    path: "/settings",
    labelKey: "nav.settings",
    shortLabelKey: "navShort.settings",
    eyebrowKey: "navEyebrow.settings",
    descriptionKey: "navDescription.settings",
    accent: "var(--accent-blue)",
    icon: <SettingOutlined />,
  },
];

export function resolveNavigationItem(
  item: NavigationMetaItem,
  translate: TranslateFn,
): NavigationItem {
  return {
    path: item.path,
    label: translate(item.labelKey),
    shortLabel: translate(item.shortLabelKey),
    eyebrow: translate(item.eyebrowKey),
    description: translate(item.descriptionKey),
    accent: item.accent,
    icon: item.icon,
  };
}

export function getResolvedNavigationItems(
  translate: TranslateFn,
): NavigationItem[] {
  return NAVIGATION_ITEMS.map((item) => resolveNavigationItem(item, translate));
}

export function getCurrentNavigation(
  pathname: string,
  translate: TranslateFn,
): NavigationItem {
  const resolvedItems = getResolvedNavigationItems(translate);
  return (
    resolvedItems.find(
      (item) => pathname === item.path || pathname.startsWith(`${item.path}/`),
    ) || resolvedItems[0]
  );
}
