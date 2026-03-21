import {
  Alert,
  Button,
  Card,
  Col,
  Empty,
  Image,
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
  DotChartOutlined,
  DownloadOutlined,
  FileTextOutlined,
  LineChartOutlined,
} from "@ant-design/icons";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { downloadModelArtifact, type Model } from "@/api/models";

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

function formatChartTooltipValue(value?: number | string) {
  if (typeof value === "number") {
    return value.toFixed(4);
  }
  return value ?? "-";
}

function PreviewImageCard({
  title,
  subtitle,
  imageUrl,
  onDownload,
}: {
  title: string;
  subtitle?: string;
  imageUrl?: string | null;
  onDownload?: () => void;
}) {
  return (
    <Card
      size="small"
      title={title}
      extra={
        onDownload ? (
          <Button size="small" icon={<DownloadOutlined />} onClick={onDownload}>
            下载
          </Button>
        ) : null
      }
    >
      {imageUrl ? (
        <Space direction="vertical" size={8} style={{ width: "100%" }}>
          <Image
            src={imageUrl}
            alt={title}
            style={{ width: "100%", borderRadius: 8, objectFit: "cover" }}
          />
          {subtitle ? <Text type="secondary">{subtitle}</Text> : null}
        </Space>
      ) : (
        <Empty description="暂无图像产物" />
      )}
    </Card>
  );
}

export default function ModelResultPanel({ model }: ModelResultPanelProps) {
  const rocCurve = model.visualizations?.roc_curve;
  const confusionMatrix = model.visualizations?.confusion_matrix;
  const attentionMaps = model.visualizations?.attention_maps || [];
  const summary = model.config?.result_summary as Record<string, any> | undefined;
  const trainingHistory = model.training_history?.entries || [];
  const auxiliaryVisuals = [
    {
      key: model.visualizations?.training_curves?.artifact_key,
      title: "训练曲线图",
      imageUrl: model.visualizations?.training_curves?.image_url,
    },
    {
      key: model.visualizations?.calibration_curve?.artifact_key,
      title: "校准曲线",
      imageUrl: model.visualizations?.calibration_curve?.image_url,
    },
    {
      key: model.visualizations?.probability_distribution?.artifact_key,
      title: "概率分布图",
      imageUrl: model.visualizations?.probability_distribution?.image_url,
    },
    {
      key: model.visualizations?.attention_statistics?.artifact_key,
      title: "注意力统计图",
      imageUrl: model.visualizations?.attention_statistics?.image_url,
    },
  ].filter((item) => item.imageUrl);

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

  const historyChartData = trainingHistory.map((entry) => ({
    epoch: entry.epoch,
    trainLoss: entry.train_loss,
    valLoss: entry.val_loss,
    trainAcc: entry.train_accuracy,
    valAcc: entry.val_accuracy,
    learningRate: entry.learning_rate,
  }));

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Alert
        type="info"
        showIcon
        message="多模态结果面板"
        description="当前面板直接消费训练后沉淀下来的 artifact，包括 history、ROC、混淆矩阵、注意力图和报告文件，不再依赖运行时 mock 数据。"
      />

      <Row gutter={16}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Best Accuracy"
              value={(summary?.best_accuracy ?? model.metrics?.best_accuracy ?? model.accuracy ?? 0) * 100}
              precision={2}
              suffix="%"
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="AUC"
              value={rocCurve?.auc ?? model.metrics?.auc ?? 0}
              precision={4}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Best Loss"
              value={summary?.best_loss ?? model.metrics?.best_loss ?? model.loss ?? 0}
              precision={4}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="训练时长" value={formatDuration(model.training_time)} />
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
            extra={
              rocCurve?.plot_artifact_key ? (
                <Button
                  size="small"
                  icon={<DownloadOutlined />}
                  onClick={() =>
                    void handleDownloadArtifact(
                      rocCurve.plot_artifact_key!,
                      `${model.name}-roc-curve.png`,
                    )
                  }
                >
                  下载图
                </Button>
              ) : null
            }
          >
            {rocCurve?.points?.length ? (
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={rocCurve.points}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="fpr" type="number" domain={[0, 1]} />
                  <YAxis type="number" domain={[0, 1]} />
                  <Tooltip formatter={formatChartTooltipValue} />
                  <ReferenceLine
                    segment={[
                      { x: 0, y: 0 },
                      { x: 1, y: 1 },
                    ]}
                    stroke="#bfbfbf"
                    strokeDasharray="5 5"
                  />
                  <Line
                    type="monotone"
                    dataKey="tpr"
                    stroke="#1677ff"
                    strokeWidth={3}
                    dot={false}
                    name="TPR"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <Empty description="暂无 ROC 数据" />
            )}
            <Paragraph style={{ marginTop: 12, marginBottom: 0 }}>
              <Text strong>AUC：</Text>
              {rocCurve?.auc?.toFixed(4) || "-"}
              {rocCurve?.positive_class_label ? (
                <Text type="secondary" style={{ marginLeft: 12 }}>
                  正类: {rocCurve.positive_class_label}
                </Text>
              ) : null}
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

      <Row gutter={16}>
        <Col span={12}>
          <Card
            title={
              <Space>
                <LineChartOutlined />
                <span>训练损失曲线</span>
              </Space>
            }
          >
            {historyChartData.length ? (
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={historyChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="trainLoss"
                    stroke="#1677ff"
                    strokeWidth={2}
                    dot={false}
                    name="Train Loss"
                  />
                  <Line
                    type="monotone"
                    dataKey="valLoss"
                    stroke="#ff4d4f"
                    strokeWidth={2}
                    dot={false}
                    name="Val Loss"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <Empty description="暂无训练历史" />
            )}
          </Card>
        </Col>
        <Col span={12}>
          <Card
            title={
              <Space>
                <LineChartOutlined />
                <span>训练准确率曲线</span>
              </Space>
            }
          >
            {historyChartData.length ? (
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={historyChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="trainAcc"
                    stroke="#52c41a"
                    strokeWidth={2}
                    dot={false}
                    name="Train Acc"
                  />
                  <Line
                    type="monotone"
                    dataKey="valAcc"
                    stroke="#722ed1"
                    strokeWidth={2}
                    dot={false}
                    name="Val Acc"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <Empty description="暂无训练历史" />
            )}
          </Card>
        </Col>
      </Row>

      <Card
        title="混淆矩阵"
        extra={
          confusionMatrix?.plot_artifact_key ? (
            <Space>
              <Button
                size="small"
                icon={<DownloadOutlined />}
                onClick={() =>
                  void handleDownloadArtifact(
                    confusionMatrix.plot_artifact_key!,
                    `${model.name}-confusion-matrix.png`,
                  )
                }
              >
                下载图
              </Button>
              {confusionMatrix.normalized_plot_artifact_key ? (
                <Button
                  size="small"
                  icon={<DownloadOutlined />}
                  onClick={() =>
                    void handleDownloadArtifact(
                      confusionMatrix.normalized_plot_artifact_key!,
                      `${model.name}-confusion-matrix-normalized.png`,
                    )
                  }
                >
                  下载归一化图
                </Button>
              ) : null}
            </Space>
          ) : null
        }
      >
        {confusionMatrix ? (
          <Space direction="vertical" size={16} style={{ width: "100%" }}>
            {confusionMatrix.plot_url ? (
              <Image
                src={confusionMatrix.plot_url}
                alt="Confusion Matrix"
                style={{ width: "100%", borderRadius: 8 }}
              />
            ) : null}
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
                ...confusionMatrix.labels.map((label) => ({
                  title: label,
                  dataIndex: label,
                  key: label,
                  align: "center" as const,
                })),
              ]}
            />
          </Space>
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
                <PreviewImageCard
                  title={map.title}
                  subtitle={`modality=${map.modality} · mean=${map.mean_attention ?? "-"} · peak=${map.peak_attention ?? "-"}`}
                  imageUrl={map.image_url}
                  onDownload={
                    map.artifact_key
                      ? () =>
                          void handleDownloadArtifact(
                            map.artifact_key!,
                            `${model.name}-${map.modality}.png`,
                          )
                      : undefined
                  }
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

      {auxiliaryVisuals.length ? (
        <Card title="辅助图表">
          <Row gutter={[16, 16]}>
            {auxiliaryVisuals.map((item) => (
              <Col span={12} key={item.title}>
                <PreviewImageCard
                  title={item.title}
                  imageUrl={item.imageUrl}
                  onDownload={
                    item.key
                      ? () =>
                          void handleDownloadArtifact(
                            item.key!,
                            `${model.name}-${item.title}.png`,
                          )
                      : undefined
                  }
                />
              </Col>
            ))}
          </Row>
        </Card>
      ) : null}

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
                gap: 16,
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
                  {artifact.is_image ? <Tag color="blue">图像</Tag> : null}
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
          <Col span={6}>
            <Statistic title="Accuracy" value={formatAccuracy(model.accuracy)} />
          </Col>
          <Col span={6}>
            <Statistic title="Loss" value={formatLoss(model.loss)} />
          </Col>
          <Col span={6}>
            <Statistic
              title="Precision"
              value={model.metrics?.precision !== undefined ? model.metrics.precision.toFixed(4) : "-"}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Recall"
              value={model.metrics?.recall !== undefined ? model.metrics.recall.toFixed(4) : "-"}
            />
          </Col>
        </Row>
      </Card>
    </Space>
  );
}
