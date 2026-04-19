import { Card, Space, Tag, Typography } from "antd";

import {
  ADVANCED_BUILDER_COMPONENTS,
  ADVANCED_BUILDER_FAMILY_LABELS,
  ADVANCED_BUILDER_STATUS_LABELS,
} from "@/config/advancedBuilderCatalog";

const { Text } = Typography;

interface AdvancedComponentPaletteProps {
  onAddComponent: (componentId: string) => void;
}

const STATUS_COLORS = {
  compile_ready: "success",
  conditional: "warning",
  draft_only: "default",
} as const;

export default function AdvancedComponentPalette({
  onAddComponent,
}: AdvancedComponentPaletteProps) {
  const groupedComponents = Object.entries(ADVANCED_BUILDER_FAMILY_LABELS).map(
    ([family, label]) => ({
      family,
      label,
      components: ADVANCED_BUILDER_COMPONENTS.filter(
        (component) => component.family === family,
      ),
    }),
  );

  return (
    <Card
      title="组件注册表"
      size="small"
      style={{
        position: "absolute",
        top: 16,
        left: 16,
        zIndex: 10,
        width: 320,
        maxHeight: "calc(100vh - 120px)",
        overflowY: "auto",
      }}
    >
      <Space direction="vertical" size={12} style={{ width: "100%" }}>
        {groupedComponents.map((group) => (
          <div key={group.family}>
            <Text strong>{group.label}</Text>
            <Space direction="vertical" size={8} style={{ width: "100%", marginTop: 8 }}>
              {group.components.map((component) => (
                <Card
                  key={component.id}
                  size="small"
                  hoverable
                  onClick={() => onAddComponent(component.id)}
                  style={{ cursor: "pointer" }}
                >
                  <Space direction="vertical" size={4} style={{ width: "100%" }}>
                    <Space wrap>
                      <Text strong>{component.label}</Text>
                      <Tag color={STATUS_COLORS[component.status]}>
                        {ADVANCED_BUILDER_STATUS_LABELS[component.status]}
                      </Tag>
                    </Space>
                    <Text type="secondary">{component.description}</Text>
                  </Space>
                </Card>
              ))}
            </Space>
          </div>
        ))}
      </Space>
    </Card>
  );
}
