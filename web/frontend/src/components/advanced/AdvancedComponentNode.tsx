import { memo } from "react";
import { Card, Space, Tag, Typography } from "antd";
import { Handle, Position, type NodeProps } from "reactflow";

import type { AdvancedBuilderNodeData } from "@/utils/advancedBuilder";

const { Text } = Typography;

const STATUS_COLORS: Record<AdvancedBuilderNodeData["status"], string> = {
  compile_ready: "success",
  conditional: "warning",
  draft_only: "default",
};

function AdvancedComponentNode({
  data,
  selected,
}: NodeProps<AdvancedBuilderNodeData>) {
  return (
    <div style={{ position: "relative" }}>
      <Handle
        type="target"
        position={Position.Left}
        style={{ background: "#94a3b8" }}
      />
      <Card
        size="small"
        style={{
          minWidth: 220,
          border: selected ? "2px solid #1677ff" : "1px solid #d9d9d9",
          boxShadow: selected ? "0 0 0 2px rgba(22,119,255,0.12)" : undefined,
        }}
      >
        <Space direction="vertical" size={6} style={{ width: "100%" }}>
          <Space wrap>
            <Text strong>{data.label}</Text>
            <Tag>{data.familyLabel || data.family}</Tag>
            <Tag color={STATUS_COLORS[data.status]}>
              {data.statusLabel || data.status}
            </Tag>
          </Space>
          <Text type="secondary">{data.description}</Text>
          {data.schemaPath ? (
            <Text type="secondary">schema: {data.schemaPath}</Text>
          ) : null}
          {data.notes?.length ? (
            <div className="surface-note surface-note--dense">
              {data.notes.join(" ")}
            </div>
          ) : null}
        </Space>
      </Card>
      <Handle
        type="source"
        position={Position.Right}
        style={{ background: "#1677ff" }}
      />
    </div>
  );
}

export default memo(AdvancedComponentNode);
