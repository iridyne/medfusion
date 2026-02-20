import { useState } from "react";
import { Layout, Menu } from "antd";
import { useNavigate, useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";
import {
  AppstoreOutlined,
  ExperimentOutlined,
  DatabaseOutlined,
  FileImageOutlined,
  DashboardOutlined,
  SettingOutlined,
  ToolOutlined,
  InboxOutlined,
  BarChartOutlined,
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
      key: "/config",
      icon: <ToolOutlined />,
      label: "配置生成器",
    },
    {
      key: "/datasets",
      icon: <InboxOutlined />,
      label: "数据管理",
    },
    {
      key: "/training",
      icon: <ExperimentOutlined />,
      label: t("nav.training"),
    },
    {
      key: "/experiments",
      icon: <BarChartOutlined />,
      label: "实验对比",
    },
    {
      key: "/models",
      icon: <DatabaseOutlined />,
      label: t("nav.models"),
    },
    {
      key: "/preprocessing",
      icon: <FileImageOutlined />,
      label: t("nav.preprocessing"),
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
