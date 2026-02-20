import { memo } from "react";
import { Handle, Position, NodeProps } from "reactflow";
import { Card } from "antd";
import { ExperimentOutlined } from "@ant-design/icons";

interface ModelNodeData {
  label: string;
  backbone?: string;
  numClasses?: number;
  fusion?: string;
  aggregator?: string;
  hiddenDim?: number;
}

function ModelNode({ data, selected }: NodeProps<ModelNodeData>) {
  return (
    <div style={{ position: "relative" }}>
      {/* 输入端口 */}
      <Handle
        type="target"
        position={Position.Left}
        id="image_data"
        style={{ top: "40%", background: "#1890ff" }}
      />
      <div
        style={{
          position: "absolute",
          left: -75,
          top: "40%",
          transform: "translateY(-50%)",
          fontSize: 11,
          color: "#999",
        }}
      >
        image_data
      </div>

      <Handle
        type="target"
        position={Position.Left}
        id="tabular_data"
        style={{ top: "60%", background: "#52c41a" }}
      />
      <div
        style={{
          position: "absolute",
          left: -85,
          top: "60%",
          transform: "translateY(-50%)",
          fontSize: 11,
          color: "#999",
        }}
      >
        tabular_data
      </div>

      <Card
        size="small"
        style={{
          minWidth: 220,
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
        <div style={{ fontSize: 12, lineHeight: "20px" }}>
          <div style={{ marginBottom: 4 }}>
            <strong>Backbone:</strong> {data.backbone || "resnet18"}
          </div>
          <div style={{ marginBottom: 4 }}>
            <strong>Fusion:</strong> {data.fusion || "concatenate"}
          </div>
          <div style={{ marginBottom: 4 }}>
            <strong>Aggregator:</strong> {data.aggregator || "mean"}
          </div>
          <div style={{ color: "#999", fontSize: 11 }}>
            类别: {data.numClasses || 2} | 维度: {data.hiddenDim || 512}
          </div>
        </div>
      </Card>

      {/* 输出端口 */}
      <Handle
        type="source"
        position={Position.Right}
        id="model"
        style={{ top: "50%", background: "#722ed1" }}
      />
      <div
        style={{
          position: "absolute",
          right: -45,
          top: "50%",
          transform: "translateY(-50%)",
          fontSize: 11,
          color: "#999",
        }}
      >
        model
      </div>
    </div>
  );
}

export default memo(ModelNode);
