import {
  Alert,
  Button,
  Card,
  Col,
  Empty,
  Row,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
  message,
} from "antd";
import {
  AppstoreOutlined,
  DownloadOutlined,
  DotChartOutlined,
  FileTextOutlined,
} from "@ant-design/icons";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  downloadModelArtifact,
  type Model,
} from "@/api/models";

const { Paragraph, Text } = Typography;

interface ModelResultPanelProps {
  model: Model;
}

function formatDuration(seconds?: number) {
  if (!seconds) {
    return "-";
  }
  if (seconds < 60) {
    return `${seconds.toFixed(1)} 秒`;
  }
  const minutes = Math.floor(seconds / 60);
  const remain = Math.round(seconds % 60);
  return `${minutes} 分 ${remain} 秒`;
}

function formatPath(path?: string) {
  return path || "-";
}

function formatAccuracy(value?: number) {
  return value !== undefined ? `${(value * 100).toFixed(2)}%` : "-";
}

function formatLoss(value?: number) {
  return value !== undefined ? value.toFixed(4) : "-";
}

function HeatmapCard({
  title,
  modality,
  grid,
}: {
  title: string;
  modality: string;
  grid: number[][];
}) {
  return (
    <Card size="small" title={title} extra={<Tag>{modality}</Tag>}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${grid[0]?.length || 8}, 1fr)`,
          gap: 4,
          aspectRatio: "1 / 1",
        }}
      >
        {grid.flatMap((row, rowIndex) =>
          row.map((value, colIndex) => (
            <div
              key={`${rowIndex}-${colIndex}`}
              style={{
                borderRadius: 6,
                background: `rgba(22, 119, 255, ${Math.max(0.12, value)})`,
                boxShadow:
                  value > 0.75 ? "0 0 12px rgba(22, 119, 255, 0.35)" : "none",
                minHeight: 20,
              }}
              title={`attention=${value}`}
            />
          )),
        )}
      </div>
    </Card>
  );
}

export default function ModelResultPanel({ model }: ModelResultPanelProps) {
  const rocCurve = model.visualizations?.roc_curve;
  const confusionMatrix = model.visualizations?.confusion_matrix;
  const attentionMaps = model.visualizations?.attention_maps || [];
  const summary = model.config?.result_summary as Record<string, any> | undefined;

  const handleDownloadArtifact = async (
    artifactKey: string,
    fallbackName: string,
  ) => {
    try {
      await downloadModelArtifact(model.id, artifactKey, fallbackName);
      message.success(`已开始下载 ${fallbackName}`);
    } catch (error) {
      console.error("Failed to download artifact:", error);
      message.error("下载结果文件失败");
    }
  };

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Alert
        type="info"
        showIcon
        message="多模态结果面板"
        description="这里集中展示当前模型的关键指标、ROC/AUC、混淆矩阵以及多模态注意力热力图，适合直接用于演示和录屏。"
      />

      <Row gutter={16}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Best Accuracy"
              value={(summary?.best_accuracy ?? model.accuracy ?? 0) * 100}
              precision={2}
              suffix="%"
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="AUC"
              value={rocCurve?.auc ?? summary?.best_accuracy ?? model.accuracy ?? 0}
              precision={4}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Best Loss"
              value={summary?.best_loss ?? model.loss ?? 0}
              precision={4}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="训练时长"
              value={formatDuration(model.training_time)}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16}>
        <Col span={14}>
          <Card
            title={
              <Space>
                <DotChartOutlined />
                <span>ROC 曲线</span>
              </Space>
            }
          >
            {rocCurve ? (
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={rocCurve.points}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="fpr" type="number" domain={[0, 1]} />
                  <YAxis type="number" domain={[0, 1]} />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="tpr"
                    stroke="#1677ff"
                    strokeWidth={3}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <Empty description="暂无 ROC 数据" />
            )}
            <Paragraph style={{ marginTop: 12, marginBottom: 0 }}>
              <Text strong>AUC：</Text>
              {rocCurve ? rocCurve.auc.toFixed(4) : "-"}
            </Paragraph>
          </Card>
        </Col>
        <Col span={10}>
          <Card
            title="结果摘要"
            extra={
              model.tags?.length ? (
                <Space wrap>
                  {model.tags.map((tag) => (
                    <Tag key={tag}>{tag}</Tag>
                  ))}
                </Space>
              ) : null
            }
          >
            <Space direction="vertical" size={8} style={{ width: "100%" }}>
              <div>
                <Text type="secondary">数据集</Text>
                <div>{model.dataset_name || "-"}</div>
              </div>
              <div>
                <Text type="secondary">骨干网络</Text>
                <div>{model.backbone}</div>
              </div>
              <div>
                <Text type="secondary">训练轮数</Text>
                <div>{model.trained_epochs || "-"}</div>
              </div>
              <div>
                <Text type="secondary">结果权重</Text>
                <div style={{ wordBreak: "break-all" }}>
                  {formatPath(model.checkpoint_path)}
                </div>
              </div>
              <div>
                <Text type="secondary">配置文件</Text>
                <div style={{ wordBreak: "break-all" }}>
                  {formatPath(model.config_path)}
                </div>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      <Card title="混淆矩阵">
        {confusionMatrix ? (
          <Table
            pagination={false}
            size="small"
            scroll={{ x: true }}
            dataSource={confusionMatrix.matrix.map((row, rowIndex) => {
              const item: Record<string, any> = {
                key: confusionMatrix.labels[rowIndex],
                actual: confusionMatrix.labels[rowIndex],
              };
              confusionMatrix.labels.forEach((label, colIndex) => {
                item[label] = row[colIndex];
              });
              return item;
            })}
            columns={[
              {
                title: "Actual \\ Predicted",
                dataIndex: "actual",
                key: "actual",
                fixed: "left",
              },
              ...((confusionMatrix?.labels || []).map((label) => ({
                title: label,
                dataIndex: label,
                key: label,
                align: "center" as const,
              })) ?? []),
            ]}
          />
        ) : (
          <Empty description="暂无混淆矩阵数据" />
        )}
      </Card>

      <Card
        title={
          <Space>
            <AppstoreOutlined />
            <span>多模态注意力结果图</span>
          </Space>
        }
      >
        <Row gutter={[16, 16]}>
          {attentionMaps.length > 0 ? (
            attentionMaps.map((map) => (
              <Col span={12} key={`${map.modality}-${map.title}`}>
                <HeatmapCard
                  title={map.title}
                  modality={map.modality}
                  grid={map.grid}
                />
              </Col>
            ))
          ) : (
            <Col span={24}>
              <Empty description="暂无注意力热力图" />
            </Col>
          )}
        </Row>
      </Card>

      <Card
        title={
          <Space>
            <FileTextOutlined />
            <span>结果文件</span>
          </Space>
        }
      >
        <Space direction="vertical" style={{ width: "100%" }} size={12}>
          {(model.result_files || []).map((artifact) => (
            <Card
              key={artifact.key}
              size="small"
              bodyStyle={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontWeight: 600 }}>
                  {artifact.label}
                  <Tag
                    color={artifact.exists ? "success" : "default"}
                    style={{ marginLeft: 8 }}
                  >
                    {artifact.exists ? "已生成" : "缺失"}
                  </Tag>
                </div>
                <div
                  style={{
                    color: "#666",
                    marginTop: 4,
                    wordBreak: "break-all",
                  }}
                >
                  {artifact.path}
                </div>
              </div>
              <Button
                type="default"
                icon={<DownloadOutlined />}
                disabled={!artifact.exists}
                onClick={() =>
                  void handleDownloadArtifact(
                    artifact.key,
                    `${model.name}-${artifact.key}`,
                  )
                }
              >
                下载
              </Button>
            </Card>
          ))}
        </Space>
      </Card>

      <Card title="关键指标">
        <Row gutter={16}>
          <Col span={8}>
            <Statistic title="Accuracy" value={formatAccuracy(model.accuracy)} />
          </Col>
          <Col span={8}>
            <Statistic title="Loss" value={formatLoss(model.loss)} />
          </Col>
          <Col span={8}>
            <Statistic title="数据集类别数" value={model.num_classes || 0} />
          </Col>
        </Row>
      </Card>
    </Space>
  );
}
