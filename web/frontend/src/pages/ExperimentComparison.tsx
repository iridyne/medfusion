import React, { useState, useEffect } from "react";
import {
  Card,
  Table,
  Button,
  Space,
  Select,
  Checkbox,
  Row,
  Col,
  Statistic,
  Tag,
  message,
  Empty,
} from "antd";
import {
  BarChartOutlined,
  DownloadOutlined,
  FileWordOutlined,
  FilePdfOutlined,
  ReloadOutlined,
  StarOutlined,
  StarFilled,
} from "@ant-design/icons";
import type { ColumnsType } from "antd/es/table";
import MetricsChart from "../components/experiment/MetricsChart";
import ConfusionMatrix from "../components/experiment/ConfusionMatrix";
import ROCCurve from "../components/experiment/ROCCurve";
import api from "../api";

interface Experiment {
  id: string;
  name: string;
  status: "completed" | "running" | "failed" | "pending";
  config: {
    backbone: string;
    fusion: string;
    aggregator?: string;
    learning_rate: number;
    batch_size: number;
    epochs: number;
  };
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    auc?: number;
    loss: number;
  };
  training_time: number; // seconds
  created_at: string;
  is_favorite?: boolean;
}

interface ComparisonMetrics {
  metric: string;
  experiments: { [key: string]: number };
  best_experiment: string;
}

interface ExperimentListResponse {
  experiments: Experiment[];
  total: number;
  page: number;
  page_size: number;
}

interface ComparisonResponse {
  experiments: Experiment[];
  metrics: ComparisonMetrics[];
  summary: Record<string, unknown>;
}

interface ReportResponse {
  report_id: string;
  download_url: string;
  format: "word" | "pdf";
  created_at: string;
}

const ExperimentComparison: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExperiments, setSelectedExperiments] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [comparisonData, setComparisonData] = useState<ComparisonMetrics[]>([]);
  const [chartType, setChartType] = useState<"bar" | "line">("bar");
  const [showCharts, setShowCharts] = useState(false);

  useEffect(() => {
    void fetchExperiments();
  }, []);

  const fetchExperiments = async () => {
    setLoading(true);
    try {
      const { data } = await api.get<ExperimentListResponse>("/experiments/", {
        params: {
          page: 1,
          page_size: 100,
          sort_by: "created_at",
          order: "desc",
        },
      });

      const items = data.experiments ?? [];
      setExperiments(items);
      setSelectedExperiments((prev) =>
        prev.filter((id) => items.some((exp) => exp.id === id)),
      );
    } catch (error) {
      message.error("Failed to fetch experiments");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = async () => {
    if (selectedExperiments.length < 2) {
      message.warning("Please select at least 2 experiments to compare");
      return;
    }

    setComparing(true);
    try {
      const { data } = await api.post<ComparisonResponse>(
        "/experiments/compare",
        selectedExperiments,
      );
      setComparisonData(data.metrics ?? []);
      setShowCharts(true);
    } catch (error) {
      message.error("Failed to compare experiments");
      console.error(error);
    } finally {
      setComparing(false);
    }
  };

  const handleGenerateReport = async (format: "word" | "pdf") => {
    if (selectedExperiments.length === 0) {
      message.warning("Please select at least one experiment");
      return;
    }

    const messageKey = `report-${format}`;
    message.loading({
      content: `Generating ${format.toUpperCase()} report...`,
      key: messageKey,
      duration: 0,
    });
    try {
      const { data } = await api.post<ReportResponse>("/experiments/report", {
        experiment_ids: selectedExperiments,
        format,
        include_charts: true,
        include_config: true,
        include_metrics: true,
      });

      message.success({
        content: `${format.toUpperCase()} report generated successfully`,
        key: messageKey,
      });
      if (data.download_url) {
        window.open(data.download_url, "_blank", "noopener,noreferrer");
      }
    } catch (error) {
      message.error({ content: "Failed to generate report", key: messageKey });
      console.error(error);
    }
  };

  const toggleFavorite = async (experimentId: string) => {
    try {
      const { data } = await api.patch<{ is_favorite?: boolean }>(
        `/experiments/${experimentId}/favorite`,
      );
      setExperiments((prev) =>
        prev.map((exp) => {
          if (exp.id !== experimentId) {
            return exp;
          }
          if (typeof data.is_favorite === "boolean") {
            return { ...exp, is_favorite: data.is_favorite };
          }
          return { ...exp, is_favorite: !exp.is_favorite };
        }),
      );
    } catch (error) {
      message.error("Failed to update favorite status");
      console.error(error);
    }
  };

  const columns: ColumnsType<Experiment> = [
    {
      title: "Select",
      key: "select",
      width: 60,
      render: (_, record) => (
        <Checkbox
          checked={selectedExperiments.includes(record.id)}
          onChange={(e) => {
            if (e.target.checked) {
              setSelectedExperiments([...selectedExperiments, record.id]);
            } else {
              setSelectedExperiments(
                selectedExperiments.filter((id) => id !== record.id)
              );
            }
          }}
        />
      ),
    },
    {
      title: "Favorite",
      key: "favorite",
      width: 80,
      render: (_, record) => (
        <Button
          type="text"
          icon={
            record.is_favorite ? (
              <StarFilled style={{ color: "#faad14" }} />
            ) : (
              <StarOutlined />
            )
          }
          onClick={() => void toggleFavorite(record.id)}
        />
      ),
    },
    {
      title: "Experiment Name",
      dataIndex: "name",
      key: "name",
      width: 200,
      sorter: (a, b) => a.name.localeCompare(b.name),
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      width: 100,
      render: (status: string) => {
        const colorMap = {
          completed: "success",
          running: "processing",
          failed: "error",
          pending: "default",
        };
        return <Tag color={colorMap[status as keyof typeof colorMap]}>{status}</Tag>;
      },
    },
    {
      title: "Backbone",
      dataIndex: ["config", "backbone"],
      key: "backbone",
      width: 150,
    },
    {
      title: "Fusion",
      dataIndex: ["config", "fusion"],
      key: "fusion",
      width: 120,
    },
    {
      title: "Accuracy",
      dataIndex: ["metrics", "accuracy"],
      key: "accuracy",
      width: 100,
      sorter: (a, b) => a.metrics.accuracy - b.metrics.accuracy,
      render: (value: number) => (
        <span style={{ fontWeight: 500 }}>
          {typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "-"}
        </span>
      ),
    },
    {
      title: "F1 Score",
      dataIndex: ["metrics", "f1_score"],
      key: "f1_score",
      width: 100,
      sorter: (a, b) => a.metrics.f1_score - b.metrics.f1_score,
      render: (value: number) => (
        <span style={{ fontWeight: 500 }}>{(value * 100).toFixed(2)}%</span>
      ),
    },
    {
      title: "AUC",
      dataIndex: ["metrics", "auc"],
      key: "auc",
      width: 100,
      sorter: (a, b) => (a.metrics.auc || 0) - (b.metrics.auc || 0),
      render: (value: number) => (
        <span style={{ fontWeight: 500 }}>{(value * 100).toFixed(2)}%</span>
      ),
    },
    {
      title: "Training Time",
      dataIndex: "training_time",
      key: "training_time",
      width: 120,
      sorter: (a, b) => a.training_time - b.training_time,
      render: (seconds: number) => {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
      },
    },
    {
      title: "Created At",
      dataIndex: "created_at",
      key: "created_at",
      width: 180,
      sorter: (a, b) =>
        new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      render: (date: string) => new Date(date).toLocaleString(),
    },
  ];

  const comparisonColumns: ColumnsType<ComparisonMetrics> = [
    {
      title: "Metric",
      dataIndex: "metric",
      key: "metric",
      width: 120,
      fixed: "left",
    },
    ...selectedExperiments.map((expId) => {
      const exp = experiments.find((e) => e.id === expId);
      return {
        title: exp?.name || expId,
        key: expId,
        width: 150,
      render: (record: ComparisonMetrics) => {
          const value = record.experiments[exp?.name || ""];
          if (typeof value !== "number") {
            return <span>-</span>;
          }
          const isBest = record.best_experiment === exp?.name;

          return (
            <span
              style={{
                fontWeight: isBest ? 600 : 400,
                color: isBest ? "#52c41a" : undefined,
              }}
            >
              {record.metric === "Loss"
                ? value.toFixed(4)
                : `${(value * 100).toFixed(2)}%`}
              {isBest && (
                <Tag color="success" style={{ marginLeft: 8 }}>
                  Best
                </Tag>
              )}
            </span>
          );
        },
      };
    }),
  ];

  const getBestExperiment = () => {
    if (experiments.length === 0) return null;
    const metric = (value: number | undefined) =>
      typeof value === "number" ? value : 0;
    return experiments.reduce((prev, curr) =>
      metric(curr.metrics.accuracy) > metric(prev.metrics.accuracy) ? curr : prev
    );
  };

  const bestExperiment = getBestExperiment();

  return (
    <div style={{ padding: "24px" }}>
      <Card
        title={
          <Space>
            <BarChartOutlined />
            <span>Experiment Comparison</span>
          </Space>
        }
        extra={
          <Space>
            <Button icon={<ReloadOutlined />} onClick={() => void fetchExperiments()}>
              Refresh
            </Button>
            <Button
              type="primary"
              icon={<BarChartOutlined />}
              onClick={() => void handleCompare()}
              loading={comparing}
              disabled={selectedExperiments.length < 2}
            >
              Compare Selected ({selectedExperiments.length})
            </Button>
            <Button
              icon={<FileWordOutlined />}
              onClick={() => void handleGenerateReport("word")}
              disabled={selectedExperiments.length === 0}
            >
              Export Word
            </Button>
            <Button
              icon={<FilePdfOutlined />}
              onClick={() => void handleGenerateReport("pdf")}
              disabled={selectedExperiments.length === 0}
            >
              Export PDF
            </Button>
          </Space>
        }
      >
        {/* Summary Statistics */}
        {bestExperiment && (
          <Row gutter={16} style={{ marginBottom: 24 }}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="Total Experiments"
                  value={experiments.length}
                  prefix={<BarChartOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="Best Accuracy"
                  value={bestExperiment.metrics.accuracy * 100}
                  precision={2}
                  suffix="%"
                  valueStyle={{ color: "#3f8600" }}
                />
                <div style={{ fontSize: 12, color: "#8c8c8c", marginTop: 8 }}>
                  {bestExperiment.name}
                </div>
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="Best F1 Score"
                  value={bestExperiment.metrics.f1_score * 100}
                  precision={2}
                  suffix="%"
                  valueStyle={{ color: "#3f8600" }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="Best AUC"
                  value={(bestExperiment.metrics.auc || 0) * 100}
                  precision={2}
                  suffix="%"
                  valueStyle={{ color: "#3f8600" }}
                />
              </Card>
            </Col>
          </Row>
        )}

        {/* Experiments Table */}
        <Table
          columns={columns}
          dataSource={experiments}
          rowKey="id"
          loading={loading}
          scroll={{ x: 1500 }}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showTotal: (total) => `Total ${total} experiments`,
          }}
        />

        {/* Comparison Results */}
        {showCharts && comparisonData.length > 0 && (
          <div style={{ marginTop: 32 }}>
            <Card
              title="Comparison Results"
              extra={
                <Space>
                  <Select
                    value={chartType}
                    onChange={setChartType}
                    style={{ width: 120 }}
                  >
                    <Select.Option value="bar">Bar Chart</Select.Option>
                    <Select.Option value="line">Line Chart</Select.Option>
                  </Select>
                  <Button
                    icon={<DownloadOutlined />}
                    onClick={() => message.info("Chart download feature coming soon")}
                  >
                    Download Chart
                  </Button>
                </Space>
              }
            >
              <Table
                columns={comparisonColumns}
                dataSource={comparisonData}
                rowKey="metric"
                pagination={false}
                scroll={{ x: 800 }}
                style={{ marginBottom: 24 }}
              />

              <MetricsChart
                data={comparisonData}
                chartType={chartType}
                height={400}
              />
            </Card>

            {/* Advanced Visualizations */}
            <Row gutter={16} style={{ marginTop: 24 }}>
              <Col span={12}>
                <Card title="Confusion Matrix" size="small">
                  <ConfusionMatrix experimentId={selectedExperiments[0]} />
                </Card>
              </Col>
              <Col span={12}>
                <Card title="ROC Curve" size="small">
                  <ROCCurve experimentIds={selectedExperiments} />
                </Card>
              </Col>
            </Row>
          </div>
        )}

        {experiments.length === 0 && !loading && (
          <Empty
            description="No experiments found. Start training to see results here."
            style={{ marginTop: 48 }}
          />
        )}
      </Card>
    </div>
  );
};

export default ExperimentComparison;
