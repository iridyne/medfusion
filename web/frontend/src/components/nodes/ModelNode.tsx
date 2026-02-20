import { memo } from "react";
import { Handle, Position, NodeProps } from "reactflow";
import { Card } from "antd";
import { ExperimentOutlined } from "@ant-design/icons";

interface ModelNodeData {
  label: string;
  backbone?: string;
  numClasses?: number;
}

function ModelNode({ data, selected }: NodeProps<ModelNodeData>) {
  return (
    <>
      <Handle type="target" position={Position.Left} />
      <Card
        size="small"
        style={{
          minWidth: 180,
          border: selected ? "2px solid #1890ff" : "1px solid #d9d9d9",
          boxShadow: selected ? "0 0 0 2px rgba(24, 144, 255, 0.2)" : undefined,
        }}
        title={
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <ExperimentOutlined />
            <span>模型</span>
          </div>
        }
      >
        <div style={{ fontSize: 12 }}>
          <div>Backbone: {data.backbone || "resnet18"}</div>
          <div>类别数: {data.numClasses || 2}</div>
        </div>
      </Card>
      <Handle type="source" position={Position.Right} />
    </>
  );
}

export default memo(ModelNode);
