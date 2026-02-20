import { memo } from "react";
import { Handle, Position, NodeProps } from "reactflow";
import { Card } from "antd";
import { DatabaseOutlined } from "@ant-design/icons";

interface DataLoaderNodeData {
  label: string;
  datasetId?: string;
  datasetName?: string;
  split?: "train" | "val" | "test" | "all";
  batchSize?: number;
  shuffle?: boolean;
  numWorkers?: number;
  seed?: number;
}

function DataLoaderNode({ data, selected }: NodeProps<DataLoaderNodeData>) {
  const getSplitColor = (split?: string) => {
    const colors: Record<string, string> = {
      train: "#52c41a",
      val: "#faad14",
      test: "#1890ff",
      all: "#722ed1",
    };
    return colors[split || "all"] || "#d9d9d9";
  };

  return (
    <Card
      size="small"
      style={{
        minWidth: 200,
        border: selected ? "2px solid #1890ff" : "1px solid #d9d9d9",
        boxShadow: selected ? "0 0 0 2px rgba(24, 144, 255, 0.2)" : undefined,
      }}
      title={
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <DatabaseOutlined />
          <span>数据加载器</span>
        </div>
      }
    >
      <div style={{ fontSize: 12, lineHeight: "20px" }}>
        <div style={{ marginBottom: 4 }}>
          <strong>数据集:</strong> {data.datasetName || "未选择"}
        </div>
        <div style={{ marginBottom: 4 }}>
          <strong>划分:</strong>{" "}
          <span
            style={{
              padding: "2px 6px",
              borderRadius: 4,
              backgroundColor: getSplitColor(data.split),
              color: "#fff",
              fontSize: 11,
            }}
          >
            {data.split || "all"}
          </span>
        </div>
        <div style={{ marginBottom: 4 }}>
          <strong>批次:</strong> {data.batchSize || 32}
        </div>
        <div style={{ color: "#999", fontSize: 11 }}>
          {data.shuffle ? "🔀 打乱" : "📋 顺序"} | Workers:{" "}
          {data.numWorkers || 4}
        </div>
      </div>

      {/* 输出端口 */}
      <Handle
        type="source"
        position={Position.Right}
        id="dataset"
        style={{ top: "50%", background: "#52c41a" }}
      />
      <div
        style={{
          position: "absolute",
          right: -60,
          top: "50%",
          transform: "translateY(-50%)",
          fontSize: 11,
          color: "#999",
        }}
      >
        dataset
      </div>
    </Card>
  );
}

export default memo(DataLoaderNode);
