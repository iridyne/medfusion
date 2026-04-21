import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
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
import type { ValueType } from "recharts/types/component/DefaultTooltipContent";

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

function formatPercent(value?: number | null, digits = 2) {
  return value !== undefined && value !== null ? `${(value * 100).toFixed(digits)}%` : "-";
}

function formatMetric(value?: number | null, digits = 4) {
  return value !== undefined && value !== null ? value.toFixed(digits) : "-";
}

function formatCount(value?: number | null) {
  return value !== undefined && value !== null ? `${value}` : "-";
}

function formatChartTooltipValue(value?: ValueType) {
  const scalarValue = Array.isArray(value) ? value[0] : value;
  if (typeof scalarValue === "number") {
    return scalarValue.toFixed(4);
  }
  return scalarValue === undefined ? "-" : String(scalarValue);
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
  const phaseImportance = model.visualizations?.phase_importance;
  const caseExplanations = model.visualizations?.case_explanations;
  const threePhaseHeatmaps = model.visualizations?.three_phase_heatmaps;
  const validation = model.validation;
  const validationOverview = validation?.overview;
  const validationDataset = validation?.dataset;
  const predictionSummary = validation?.prediction_summary;
  const thresholdAnalysis = validation?.threshold_analysis;
  const calibration = validation?.calibration;
  const survival = validation?.survival;
  const globalFeatureImportance = validation?.global_feature_importance;
  const perClassRows = validation?.per_class || [];
  const importanceTopFeatures = globalFeatureImportance?.top_features || [];
  const summary = (model.config?.result_summary || {}) as Record<string, any>;
  const sourceContext = (model.config?.source_context || {}) as Record<string, any>;
  const sourceContract = model.source_contract;
  const importSource = model.config?.import_source as string | undefined;
  const trainingHistory = model.training_history?.entries || [];
  const resultFiles = model.result_files || [];
  const existingResultFiles = resultFiles.filter((artifact) => artifact.exists);
  const phaseImportanceEntries = Object.entries(phaseImportance?.mean_importance || {});
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
      key: model.visualizations?.survival_curve?.artifact_key,
      title: "Kaplan-Meier 曲线",
      imageUrl: model.visualizations?.survival_curve?.image_url,
    },
    {
      key: model.visualizations?.risk_score_distribution?.artifact_key,
      title: "风险分数分布图",
      imageUrl: model.visualizations?.risk_score_distribution?.image_url,
    },
    {
      key: model.visualizations?.feature_importance_bar?.artifact_key,
      title: "变量重要性柱状图",
      imageUrl: model.visualizations?.feature_importance_bar?.image_url,
    },
    {
      key: model.visualizations?.feature_importance_beeswarm?.artifact_key,
      title: "变量重要性蜂群图",
      imageUrl: model.visualizations?.feature_importance_beeswarm?.image_url,
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
  const visualizationArtifactCount =
    attentionMaps.length +
    auxiliaryVisuals.length +
    (threePhaseHeatmaps?.heatmap_count || 0) +
    (rocCurve?.plot_url ? 1 : 0) +
    (confusionMatrix?.plot_url ? 1 : 0) +
    (confusionMatrix?.normalized_plot_url ? 1 : 0);

  const primaryStats = [
    {
      title: "Best Accuracy",
      value: formatPercent(
        summary?.best_accuracy ?? model.metrics?.best_accuracy ?? model.accuracy,
      ),
    },
    {
      title: "ROC AUC",
      value: formatMetric(
        rocCurve?.auc ?? validationOverview?.auc ?? model.metrics?.auc,
      ),
    },
    {
      title: "C-index",
      value: formatMetric(survival?.c_index ?? model.metrics?.c_index),
    },
    {
      title: "宏平均 F1",
      value: formatMetric(
        validationOverview?.macro_f1 ??
          model.metrics?.macro_f1 ??
          model.metrics?.f1_score,
      ),
    },
    {
      title: "Balanced Acc",
      value: formatMetric(
        validationOverview?.balanced_accuracy ??
          model.metrics?.balanced_accuracy,
      ),
    },
    {
      title: "Specificity",
      value: formatMetric(
        thresholdAnalysis?.specificity ?? model.metrics?.specificity,
      ),
    },
    {
      title: "训练时长",
      value: formatDuration(model.training_time),
    },
  ];

  const validationMetricCards = [
    {
      title: "验证样本数",
      value: formatCount(validationOverview?.sample_count),
    },
    {
      title: "平均置信度",
      value: formatPercent(
        predictionSummary?.mean_confidence ?? validationOverview?.mean_confidence,
      ),
    },
    {
      title: "错误率",
      value: formatPercent(
        predictionSummary?.error_rate ?? validationOverview?.error_rate,
      ),
    },
    {
      title: "Brier Score",
      value: formatMetric(calibration?.brier_score ?? model.metrics?.brier_score),
    },
    {
      title: "ECE",
      value: formatMetric(calibration?.ece ?? model.metrics?.ece),
    },
    {
      title: "最佳轮次",
      value: formatCount(validationOverview?.best_epoch ?? model.metrics?.best_epoch),
    },
  ];

  return (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Alert
        type="info"
        showIcon
        message="正式版结果详情"
        description="结果详情现在按四层来读：先看结论层，再看指标层，再下钻可视化层，最后回到文件层做下载、复盘和交付。"
      />

      <Card
        title="1. 结论层"
        extra={<Tag color="processing">Conclusion first</Tag>}
      >
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <Alert
            type={existingResultFiles.length > 0 ? "success" : "warning"}
            showIcon
            message="先看这条 run 值不值得继续下钻"
            description="这一层只回答四个问题：这次 run 在做什么、主要结果如何、有没有可交付的资产、接下来值不值得去看更细的指标和图。"
          />

          <Row gutter={[16, 16]}>
            <Col xs={24} xl={15}>
              <Card
                size="small"
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
                <Descriptions column={1} size="small" labelStyle={{ width: 120 }}>
                  <Descriptions.Item label="实验">
                    {summary?.experiment_name || model.name || "-"}
                  </Descriptions.Item>
                  <Descriptions.Item label="来源">
                    {sourceContext.source_type || importSource || "-"}
                  </Descriptions.Item>
                  <Descriptions.Item label="入口">
                    {sourceContext.entrypoint || "-"}
                  </Descriptions.Item>
                  <Descriptions.Item label="Blueprint">
                    {sourceContext.blueprint_id || "-"}
                  </Descriptions.Item>
                  <Descriptions.Item label="来源模板">
                    {sourceContract?.template_label || "-"}
                  </Descriptions.Item>
                  <Descriptions.Item label="来源说明">
                    {sourceContract?.message || "-"}
                  </Descriptions.Item>
                  <Descriptions.Item label="推荐 preset">
                    {sourceContract?.recommended_preset || "-"}
                  </Descriptions.Item>
                  <Descriptions.Item label="编译边界">
                    {sourceContract?.compile_boundary || "-"}
                  </Descriptions.Item>
                  <Descriptions.Item label="数据集">
                    {model.dataset_name || validationDataset?.name || "-"}
                  </Descriptions.Item>
                  <Descriptions.Item label="骨干网络">
                    {model.backbone || "-"}
                  </Descriptions.Item>
                  <Descriptions.Item label="训练轮数">
                    {formatCount(model.trained_epochs)}
                  </Descriptions.Item>
                  <Descriptions.Item label="验证样本数">
                    {formatCount(validationOverview?.sample_count)}
                  </Descriptions.Item>
                  <Descriptions.Item label="结果 split">
                    {summary?.split || validationOverview?.positive_class_label || "-"}
                  </Descriptions.Item>
                  <Descriptions.Item label="正类标签">
                    {validationOverview?.positive_class_label || "-"}
                  </Descriptions.Item>
                </Descriptions>

                {validationDataset?.class_distribution?.length ? (
                  <div style={{ marginTop: 12 }}>
                    <Text type="secondary">类别分布</Text>
                    <div style={{ marginTop: 8 }}>
                      <Space wrap>
                        {validationDataset.class_distribution.map((item) => (
                          <Tag key={item.label} color="blue">
                            {item.label}: {item.count} / {formatPercent(item.rate)}
                          </Tag>
                        ))}
                      </Space>
                    </div>
                  </div>
                ) : null}
                {sourceContract?.compile_notes?.length ? (
                  <div style={{ marginTop: 12 }}>
                    <Text type="secondary">来源 contract 说明</Text>
                    <div style={{ marginTop: 8 }}>
                      <Space direction="vertical" size={4} style={{ width: "100%" }}>
                        {sourceContract.compile_notes.map((item) => (
                          <div key={item} className="surface-note surface-note--dense">
                            {item}
                          </div>
                        ))}
                      </Space>
                    </div>
                  </div>
                ) : null}
                {sourceContract?.patch_target_hints?.length ? (
                  <div style={{ marginTop: 12 }}>
                    <Text type="secondary">来源 patch target hints</Text>
                    <div style={{ marginTop: 8 }}>
                      <Space direction="vertical" size={4} style={{ width: "100%" }}>
                        {sourceContract.patch_target_hints.map((item) => (
                          <div
                            key={`${item.path}-${item.mode}`}
                            className="surface-note surface-note--dense"
                          >
                            {item.mode}: {item.path} · {item.description}
                          </div>
                        ))}
                      </Space>
                    </div>
                  </div>
                ) : null}
              </Card>
            </Col>

            <Col xs={24} xl={9}>
              <Card size="small" title="结果交付摘要">
                <Descriptions column={1} size="small" labelStyle={{ width: 132 }}>
                  <Descriptions.Item label="主准确率">
                    {formatPercent(
                      summary?.best_accuracy ?? model.metrics?.best_accuracy ?? model.accuracy,
                    )}
                  </Descriptions.Item>
                  <Descriptions.Item label="ROC AUC">
                    {formatMetric(rocCurve?.auc ?? validationOverview?.auc ?? model.metrics?.auc)}
                  </Descriptions.Item>
                  <Descriptions.Item label="宏平均 F1">
                    {formatMetric(
                      validationOverview?.macro_f1 ??
                        model.metrics?.macro_f1 ??
                        model.metrics?.f1_score,
                    )}
                  </Descriptions.Item>
                  <Descriptions.Item label="可下载文件">
                    {existingResultFiles.length} / {resultFiles.length}
                  </Descriptions.Item>
                  <Descriptions.Item label="可视化资产">
                    {visualizationArtifactCount}
                  </Descriptions.Item>
                  <Descriptions.Item label="训练时长">
                    {formatDuration(model.training_time)}
                  </Descriptions.Item>
                </Descriptions>
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]}>
            {primaryStats.map((item) => (
              <Col xs={24} sm={12} lg={8} xl={6} key={item.title}>
                <Card size="small">
                  <Statistic title={item.title} value={item.value} />
                </Card>
              </Col>
            ))}
          </Row>
        </Space>
      </Card>

      <Card
        title="2. 指标层"
        extra={<Tag color="gold">Metric deep dive</Tag>}
      >
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <Card size="small" title="Validation 快照">
            <Row gutter={[16, 16]}>
              {validationMetricCards.map((item) => (
                <Col xs={24} sm={12} lg={8} xl={4} key={item.title}>
                  <Card size="small">
                    <Statistic title={item.title} value={item.value} />
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>

          <Row gutter={[16, 16]}>
            <Col xs={24} xl={12}>
              <Card
                size="small"
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

            <Col xs={24} xl={12}>
              <Card
                size="small"
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

          <Card size="small" title="Per-class Metrics">
            {perClassRows.length ? (
              <Table
                rowKey="label"
                pagination={false}
                size="small"
                scroll={{ x: true }}
                dataSource={perClassRows}
                columns={[
                  {
                    title: "类别",
                    dataIndex: "label",
                    key: "label",
                    fixed: "left",
                  },
                  {
                    title: "Support",
                    dataIndex: "support",
                    key: "support",
                    align: "right",
                  },
                  {
                    title: "Prevalence",
                    dataIndex: "prevalence",
                    key: "prevalence",
                    align: "right",
                    render: (value: number) => formatPercent(value),
                  },
                  {
                    title: "Precision",
                    dataIndex: "precision",
                    key: "precision",
                    align: "right",
                    render: (value: number) => formatMetric(value),
                  },
                  {
                    title: "Recall",
                    dataIndex: "recall",
                    key: "recall",
                    align: "right",
                    render: (value: number) => formatMetric(value),
                  },
                  {
                    title: "F1",
                    dataIndex: "f1_score",
                    key: "f1_score",
                    align: "right",
                    render: (value: number) => formatMetric(value),
                  },
                  {
                    title: "Predicted",
                    dataIndex: "predicted_count",
                    key: "predicted_count",
                    align: "right",
                  },
                  {
                    title: "Predicted Rate",
                    dataIndex: "predicted_rate",
                    key: "predicted_rate",
                    align: "right",
                    render: (value: number) => formatPercent(value),
                  },
                ]}
              />
            ) : (
              <Empty description="暂无 per-class 指标" />
            )}
          </Card>

          <Row gutter={[16, 16]}>
            <Col xs={24} xl={12}>
              <Card size="small" title="阈值分析">
                {thresholdAnalysis ? (
                  <Descriptions column={1} size="small" labelStyle={{ width: 140 }}>
                    <Descriptions.Item label="Optimal Threshold">
                      {formatMetric(thresholdAnalysis.threshold)}
                    </Descriptions.Item>
                    <Descriptions.Item label="Youden J">
                      {formatMetric(thresholdAnalysis.youden_j)}
                    </Descriptions.Item>
                    <Descriptions.Item label="Sensitivity">
                      {formatMetric(thresholdAnalysis.sensitivity)}
                    </Descriptions.Item>
                    <Descriptions.Item label="Specificity">
                      {formatMetric(thresholdAnalysis.specificity)}
                    </Descriptions.Item>
                    <Descriptions.Item label="PPV">
                      {formatMetric(thresholdAnalysis.ppv)}
                    </Descriptions.Item>
                    <Descriptions.Item label="NPV">
                      {formatMetric(thresholdAnalysis.npv)}
                    </Descriptions.Item>
                  </Descriptions>
                ) : (
                  <Empty description="当前任务暂无阈值分析" />
                )}
              </Card>
            </Col>

            <Col xs={24} xl={12}>
              <Card size="small" title="校准与误差">
                <Descriptions column={1} size="small" labelStyle={{ width: 140 }}>
                  <Descriptions.Item label="Brier Score">
                    {formatMetric(calibration?.brier_score ?? model.metrics?.brier_score)}
                  </Descriptions.Item>
                  <Descriptions.Item label="ECE">
                    {formatMetric(calibration?.ece ?? model.metrics?.ece)}
                  </Descriptions.Item>
                  <Descriptions.Item label="Mean Confidence">
                    {formatPercent(
                      predictionSummary?.mean_confidence ?? validationOverview?.mean_confidence,
                    )}
                  </Descriptions.Item>
                  <Descriptions.Item label="Error Rate">
                    {formatPercent(
                      predictionSummary?.error_rate ?? validationOverview?.error_rate,
                    )}
                  </Descriptions.Item>
                  <Descriptions.Item label="Correct Confidence">
                    {formatPercent(predictionSummary?.mean_confidence_correct)}
                  </Descriptions.Item>
                  <Descriptions.Item label="Error Confidence">
                    {formatPercent(predictionSummary?.mean_confidence_error)}
                  </Descriptions.Item>
                </Descriptions>

                <div style={{ marginTop: 12 }}>
                  <Text type="secondary">常见误分类</Text>
                  <div style={{ marginTop: 8 }}>
                    {predictionSummary?.top_misclassifications?.length ? (
                      <Space wrap>
                        {predictionSummary.top_misclassifications.map((item) => (
                          <Tag key={`${item.actual}-${item.predicted}`}>
                            {item.actual} {"->"} {item.predicted}: {item.count}
                          </Tag>
                        ))}
                      </Space>
                    ) : (
                      <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无误分类摘要" />
                    )}
                  </div>
                </div>
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]}>
            <Col xs={24} xl={12}>
              <Card size="small" title="Survival 分析">
                {survival ? (
                  <Descriptions column={1} size="small" labelStyle={{ width: 160 }}>
                    <Descriptions.Item label="C-index">
                      {formatMetric(survival.c_index)}
                    </Descriptions.Item>
                    <Descriptions.Item label="样本数">
                      {formatCount(survival.sample_count)}
                    </Descriptions.Item>
                    <Descriptions.Item label="事件数">
                      {formatCount(survival.event_count)}
                    </Descriptions.Item>
                    <Descriptions.Item label="事件率">
                      {formatPercent(survival.event_rate)}
                    </Descriptions.Item>
                    <Descriptions.Item label="删失率">
                      {formatPercent(survival.censoring_rate)}
                    </Descriptions.Item>
                    <Descriptions.Item label="风险来源">
                      {survival.risk_score_source || "-"}
                    </Descriptions.Item>
                    <Descriptions.Item label="中位生存时间">
                      {formatMetric(survival.median_survival_time)}
                    </Descriptions.Item>
                    <Descriptions.Item label="分组阈值">
                      {formatMetric(survival.risk_group_threshold)}
                    </Descriptions.Item>
                  </Descriptions>
                ) : (
                  <Empty description="当前结果未配置 survival 分析" />
                )}
              </Card>
            </Col>

            <Col xs={24} xl={12}>
              <Card size="small" title="全局变量重要性">
                {importanceTopFeatures.length ? (
                  <Table
                    rowKey="feature"
                    pagination={false}
                    size="small"
                    dataSource={importanceTopFeatures}
                    columns={[
                      {
                        title: "变量",
                        dataIndex: "feature",
                        key: "feature",
                      },
                      {
                        title: "Mean |Contribution|",
                        dataIndex: "mean_abs_contribution",
                        key: "mean_abs_contribution",
                        align: "right" as const,
                        render: (value: number) => formatMetric(value, 6),
                      },
                      {
                        title: "Mean Contribution",
                        dataIndex: "mean_contribution",
                        key: "mean_contribution",
                        align: "right" as const,
                        render: (value: number) => formatMetric(value, 6),
                      },
                    ]}
                  />
                ) : (
                  <Empty description="当前结果暂无变量重要性摘要" />
                )}
                {globalFeatureImportance?.method ? (
                  <Paragraph style={{ marginTop: 12, marginBottom: 0 }}>
                    <Text type="secondary">
                      {globalFeatureImportance.method}
                      {" · "}
                      {globalFeatureImportance.score_name || "-"}
                      {" · samples="}
                      {globalFeatureImportance.sample_count ?? "-"}
                    </Text>
                  </Paragraph>
                ) : null}
              </Card>
            </Col>
          </Row>
        </Space>
      </Card>

      <Card
        title="3. 可视化层"
        extra={<Tag color="blue">Visual evidence</Tag>}
      >
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <Row gutter={[16, 16]}>
            <Col xs={24} xl={15}>
              <Card
                size="small"
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
                  <ResponsiveContainer width="100%" height={300}>
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
                  {formatMetric(rocCurve?.auc ?? validationOverview?.auc)}
                  {rocCurve?.positive_class_label ? (
                    <Text type="secondary" style={{ marginLeft: 12 }}>
                      正类: {rocCurve.positive_class_label}
                    </Text>
                  ) : null}
                </Paragraph>
              </Card>
            </Col>

            <Col xs={24} xl={9}>
              <Card
                size="small"
                title={
                  <Space>
                    <AppstoreOutlined />
                    <span>可视化摘要</span>
                  </Space>
                }
              >
                <Descriptions column={1} size="small" labelStyle={{ width: 132 }}>
                  <Descriptions.Item label="ROC 曲线">
                    {rocCurve?.plot_url ? "已生成" : "缺失"}
                  </Descriptions.Item>
                  <Descriptions.Item label="混淆矩阵">
                    {confusionMatrix?.plot_url ? "已生成" : "缺失"}
                  </Descriptions.Item>
                  <Descriptions.Item label="注意力热图">
                    {attentionMaps.length}
                  </Descriptions.Item>
                  <Descriptions.Item label="三期热图病例">
                    {threePhaseHeatmaps?.case_count || 0}
                  </Descriptions.Item>
                  <Descriptions.Item label="三期贡献指标">
                    {phaseImportanceEntries.length}
                  </Descriptions.Item>
                  <Descriptions.Item label="辅助图表">
                    {auxiliaryVisuals.length}
                  </Descriptions.Item>
                  <Descriptions.Item label="总可视化资产">
                    {visualizationArtifactCount}
                  </Descriptions.Item>
                </Descriptions>
              </Card>
            </Col>
          </Row>

          <Card
            size="small"
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
                <Row gutter={[16, 16]}>
                  {confusionMatrix.plot_url ? (
                    <Col xs={24} xl={12}>
                      <PreviewImageCard title="原始混淆矩阵" imageUrl={confusionMatrix.plot_url} />
                    </Col>
                  ) : null}
                  {confusionMatrix.normalized_plot_url ? (
                    <Col xs={24} xl={12}>
                      <PreviewImageCard
                        title="归一化混淆矩阵"
                        imageUrl={confusionMatrix.normalized_plot_url}
                      />
                    </Col>
                  ) : null}
                </Row>
                <Table
                  pagination={false}
                  size="small"
                  scroll={{ x: true }}
                  dataSource={confusionMatrix.matrix.map((row, rowIndex) => {
                    const item: Record<string, string | number> = {
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
            size="small"
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
                  <Col xs={24} xl={12} key={`${map.modality}-${map.title}`}>
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

          <Card
            size="small"
            title="三期解释热图（模型空间 + 原始切片）"
            extra={
              threePhaseHeatmaps?.artifact_key ? (
                <Button
                  size="small"
                  icon={<DownloadOutlined />}
                  onClick={() =>
                    void handleDownloadArtifact(
                      threePhaseHeatmaps.artifact_key!,
                      `${model.name}-heatmap-manifest.json`,
                    )
                  }
                >
                  下载清单
                </Button>
              ) : null
            }
          >
            <Space direction="vertical" size={12} style={{ width: "100%" }}>
              <Descriptions column={1} size="small" labelStyle={{ width: 160 }}>
                <Descriptions.Item label="热图方法">
                  {threePhaseHeatmaps?.method || "-"}
                </Descriptions.Item>
                <Descriptions.Item label="病例数">
                  {threePhaseHeatmaps?.case_count || 0}
                </Descriptions.Item>
                <Descriptions.Item label="热图条目">
                  {threePhaseHeatmaps?.heatmap_count || 0}
                </Descriptions.Item>
                <Descriptions.Item label="病例解释数">
                  {caseExplanations?.cases?.length || 0}
                </Descriptions.Item>
                <Descriptions.Item label="三期贡献">
                  {phaseImportanceEntries.length ? (
                    <Space wrap>
                      {phaseImportanceEntries.map(([phase, value]) => (
                        <Tag key={phase} color="geekblue">
                          {phase}: {formatPercent(value)}
                        </Tag>
                      ))}
                    </Space>
                  ) : (
                    "暂无"
                  )}
                </Descriptions.Item>
              </Descriptions>

              {threePhaseHeatmaps?.cases?.length ? (
                <Space wrap>
                  {threePhaseHeatmaps.cases.slice(0, 6).map((caseItem) => (
                    <Tag key={caseItem.case_id} color="blue">
                      病例 {caseItem.case_id} · 热图 {(caseItem.heatmaps || []).length} 张
                    </Tag>
                  ))}
                </Space>
              ) : (
                <Empty description="暂无三期解释热图" />
              )}
            </Space>
          </Card>

          {auxiliaryVisuals.length ? (
            <Card size="small" title="辅助图表">
              <Row gutter={[16, 16]}>
                {auxiliaryVisuals.map((item) => (
                  <Col xs={24} xl={12} key={item.title}>
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
        </Space>
      </Card>

      <Card
        title="4. 文件层"
        extra={<Tag color="purple">Artifact handoff</Tag>}
      >
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <Card size="small" title="文件与路径摘要">
            <Row gutter={[16, 16]}>
              <Col xs={24} sm={12} lg={6}>
                <Statistic title="已生成文件" value={existingResultFiles.length} />
              </Col>
              <Col xs={24} sm={12} lg={6}>
                <Statistic title="总索引文件" value={resultFiles.length} />
              </Col>
              <Col xs={24} sm={12} lg={6}>
                <Statistic title="图像型 artifact" value={resultFiles.filter((artifact) => artifact.is_image && artifact.exists).length} />
              </Col>
              <Col xs={24} sm={12} lg={6}>
                <Statistic title="缺失文件" value={resultFiles.length - existingResultFiles.length} />
              </Col>
            </Row>

            <Descriptions
              column={1}
              size="small"
              labelStyle={{ width: 132 }}
              style={{ marginTop: 16 }}
            >
              <Descriptions.Item label="权重路径">
                <div style={{ wordBreak: "break-all" }}>
                  {formatPath(model.checkpoint_path)}
                </div>
              </Descriptions.Item>
              <Descriptions.Item label="配置路径">
                <div style={{ wordBreak: "break-all" }}>
                  {formatPath(model.config_path)}
                </div>
              </Descriptions.Item>
            </Descriptions>
          </Card>

          <Card
            size="small"
            title={
              <Space>
                <FileTextOutlined />
                <span>结果文件</span>
              </Space>
            }
          >
            <Space direction="vertical" style={{ width: "100%" }} size={12}>
              {resultFiles.length ? (
                resultFiles.map((artifact) => (
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
                ))
              ) : (
                <Empty description="当前结果尚未索引文件" />
              )}
            </Space>
          </Card>
        </Space>
      </Card>
    </Space>
  );
}
