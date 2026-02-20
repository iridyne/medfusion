import { useState } from "react";
import { Layout, Menu } from "antd";
import { useNavigate, useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";
import {
  AppstoreOutlined,
  ExperimentOutlined,
  DatabaseOutlined,
  DashboardOutlined,
  SettingOutlined,
} from "@ant-design/icons";

const { Sider } = Layout;

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { t } = useTranslation();

  const menuItems = [
    {
      key: "/workflow",
      icon: <AppstoreOutlined />,
      label: t("nav.workflow"),
    },
    {
      key: "/training",
      icon: <ExperimentOutlined />,
      label: t("nav.training"),
    },
    {
      key: "/models",
      icon: <DatabaseOutlined />,
      label: t("nav.models"),
    },
    {
      key: "/system",
      icon: <DashboardOutlined />,
      label: t("nav.system"),
    },
    {
      key: "/settings",
      icon: <SettingOutlined />,
      label: t("nav.settings"),
    },
  ];

  return (
    <Sider
      collapsible
      collapsed={collapsed}
      onCollapse={setCollapsed}
      theme="dark"
      width={220}
    >
      <div
        style={{
          height: 64,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#fff",
          fontSize: collapsed ? 16 : 20,
          fontWeight: "bold",
        }}
      >
        {collapsed ? "MF" : "MedFusion"}
      </div>
      <Menu
        theme="dark"
        mode="inline"
        selectedKeys={[location.pathname]}
        items={menuItems}
        onClick={({ key }) => navigate(key)}
      />
    </Sider>
  );
}
