import { memo } from "react";
import { Handle, Position, NodeProps } from "reactflow";
import { Card, Tag } from "antd";
import { ExperimentOutlined } from "@ant-design/icons";

interface TrainingNodeData {
  label: string;
  epochs?: number;
  learningRate?: number;
  optimizer?: string;
  useAmp?: boolean;
  gradientCheckpointing?: boolean;
}

function TrainingNode({ data, selected }: NodeProps<TrainingNodeData>) {
  const getOptimizerColor = (optimizer?: string) => {
    const colors: Record<string, string> = {
      adam: "#1890ff",
      sgd: "#52c41a",
      adamw: "#722ed1",
    };
    return colors[optimizer || "adam"] || "#d9d9d9";
  };

  return (
    <div style={{ position: "relative" }}>
      {/* 输入端口 */}
      <Handle
        type="target"
        position={Position.Left}
        id="model"
        style={{ top: "30%", background: "#722ed1" }}
      />
      <div
        style={{
          position: "absolute",
          left: -50,
          top: "30%",
          transform: "translateY(-50%)",
          fontSize: 11,
          color: "#999",
        }}
      >
        model
      </div>

      <Handle
        type="target"
        position={Position.Left}
        id="train_data"
        style={{ top: "50%", background: "#52c41a" }}
      />
      <div
        style={{
          position: "absolute",
          left: -70,
          top: "50%",
          transform: "translateY(-50%)",
          fontSize: 11,
          color: "#999",
        }}
      >
        train_data
      </div>

      <Handle
        type="target"
        position={Position.Left}
        id="val_data"
        style={{ top: "70%", background: "#faad14" }}
      />
      <div
        style={{
          position: "absolute",
          left: -60,
          top: "70%",
          transform: "translateY(-50%)",
          fontSize: 11,
          color: "#999",
        }}
      >
        val_data
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
            <span>训练</span>
          </div>
        }
      >
        <div style={{ fontSize: 12, lineHeight: "20px" }}>
          <div style={{ marginBottom: 4 }}>
            <strong>轮数:</strong> {data.epochs || 50}
          </div>
          <div style={{ marginBottom: 4 }}>
            <strong>学习率:</strong> {data.learningRate || 0.001}
          </div>
          <div style={{ marginBottom: 4 }}>
            <strong>优化器:</strong>{" "}
            <Tag
              color={getOptimizerColor(data.optimizer)}
              style={{ fontSize: 11 }}
            >
              {(data.optimizer || "adam").toUpperCase()}
            </Tag>
          </div>
          <div style={{ color: "#999", fontSize: 11 }}>
            {data.useAmp ? "⚡ 混合精度" : "🔋 全精度"} |{" "}
            {data.gradientCheckpointing ? "💾 梯度检查点" : "🚀 标准"}
          </div>
        </div>
      </Card>

      {/* 输出端口 */}
      <Handle
        type="source"
        position={Position.Right}
        id="trained_model"
        style={{ top: "40%", background: "#722ed1" }}
      />
      <div
        style={{
          position: "absolute",
          right: -90,
          top: "40%",
          transform: "translateY(-50%)",
          fontSize: 11,
          color: "#999",
        }}
      >
        trained_model
      </div>

      <Handle
        type="source"
        position={Position.Right}
        id="history"
        style={{ top: "60%", background: "#1890ff" }}
      />
      <div
        style={{
          position: "absolute",
          right: -50,
          top: "60%",
          transform: "translateY(-50%)",
          fontSize: 11,
          color: "#999",
        }}
      >
        history
      </div>
    </div>
  );
}

export default memo(TrainingNode);
