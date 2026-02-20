import { memo } from "react";
import { Handle, Position, NodeProps } from "reactflow";
import { Card, Tag, Space } from "antd";
import { LineChartOutlined } from "@ant-design/icons";

interface EvaluationNodeData {
  label: string;
  metrics?: string[];
  saveResults?: boolean;
}

function EvaluationNode({ data, selected }: NodeProps<EvaluationNodeData>) {
  const defaultMetrics = ["accuracy", "loss"];
  const metrics = data.metrics || defaultMetrics;

  const getMetricColor = (metric: string) => {
    const colors: Record<string, string> = {
      accuracy: "#52c41a",
      loss: "#f5222d",
      precision: "#1890ff",
      recall: "#722ed1",
      f1: "#faad14",
      auc: "#13c2c2",
    };
    return colors[metric] || "#d9d9d9";
  };

  return (
    <div style={{ position: "relative" }}>
      {/* 输入端口 */}
      <Handle
        type="target"
        position={Position.Left}
        id="model"
        style={{ top: "40%", background: "#722ed1" }}
      />
      <div
        style={{
          position: "absolute",
          left: -50,
          top: "40%",
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
        id="test_data"
        style={{ top: "60%", background: "#1890ff" }}
      />
      <div
        style={{
          position: "absolute",
          left: -65,
          top: "60%",
          transform: "translateY(-50%)",
          fontSize: 11,
          color: "#999",
        }}
      >
        test_data
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
            <LineChartOutlined />
            <span>评估</span>
          </div>
        }
      >
        <div style={{ fontSize: 12, lineHeight: "20px" }}>
          <div style={{ marginBottom: 8 }}>
            <strong>评估指标:</strong>
          </div>
          <Space size={4} wrap style={{ marginBottom: 8 }}>
            {metrics.map((metric) => (
              <Tag
                key={metric}
                color={getMetricColor(metric)}
                style={{ fontSize: 10, margin: 0 }}
              >
                {metric}
              </Tag>
            ))}
          </Space>
          <div style={{ color: "#999", fontSize: 11 }}>
            {data.saveResults ? "💾 保存结果" : "👁️ 仅查看"}
          </div>
        </div>
      </Card>

      {/* 输出端口 */}
      <Handle
        type="source"
        position={Position.Right}
        id="metrics"
        style={{ top: "40%", background: "#52c41a" }}
      />
      <div
        style={{
          position: "absolute",
          right: -55,
          top: "40%",
          transform: "translateY(-50%)",
          fontSize: 11,
          color: "#999",
        }}
      >
        metrics
      </div>

      <Handle
        type="source"
        position={Position.Right}
        id="report"
        style={{ top: "60%", background: "#1890ff" }}
      />
      <div
        style={{
          position: "absolute",
          right: -45,
          top: "60%",
          transform: "translateY(-50%)",
          fontSize: 11,
          color: "#999",
        }}
      >
        report
      </div>
    </div>
  );
}

export default memo(EvaluationNode);
