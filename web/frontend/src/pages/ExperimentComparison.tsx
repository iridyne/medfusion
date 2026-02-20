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
  Spin,
  Empty,
  Tooltip,
  Modal,
} from "antd";
import {
  BarChartOutlined,
  LineChartOutlined,
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

interface Experiment {
  id: string;
  name: string;
  status: "completed" | "running" | "failed";
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

const ExperimentComparison: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExperiments, setSelectedExperiments] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [comparisonData, setComparisonData] = useState<ComparisonMetrics[]>([]);
  const [chartType, setChartType] = useState<"bar" | "line">("bar");
  const [showCharts, setShowCharts] = useState(false);

  useEffect(() => {
    fetchExperiments();
  }, []);

  const fetchExperiments = async () => {
    setLoading(true);
    try {
      // Mock data - replace with actual API call
      const mockExperiments: Experiment[] = [
        {
          id: "exp-001",
          name: "ResNet50 + Concatenate",
          status: "completed",
          config: {
            backbone: "resnet50",
            fusion: "concatenate",
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 50,
          },
          metrics: {
            accuracy: 0.892,
            precision: 0.885,
            recall: 0.878,
            f1_score: 0.881,
            auc: 0.945,
            loss: 0.234,
          },
          training_time: 3600,
          created_at: "2026-02-15T10:30:00Z",
          is_favorite: true,
        },
        {
          id: "exp-002",
          name: "ViT-Base + Attention",
          status: "completed",
          config: {
            backbone: "vit_base",
            fusion: "attention",
            learning_rate: 0.0001,
            batch_size: 16,
            epochs: 50,
          },
          metrics: {
            accuracy: 0.915,
            precision: 0.908,
            recall: 0.902,
            f1_score: 0.905,
            auc: 0.962,
            loss: 0.198,
          },
          training_time: 5400,
          created_at: "2026-02-16T14:20:00Z",
          is_favorite: false,
        },
        {
          id: "exp-003",
          name: "Swin-Base + Gated",
          status: "completed",
          config: {
            backbone: "swin_base",
            fusion: "gated",
            aggregator: "attention",
            learning_rate: 0.0005,
            batch_size: 24,
            epochs: 50,
          },
          metrics: {
            accuracy: 0.928,
            precision: 0.922,
            recall: 0.918,
            f1_score: 0.920,
            auc: 0.971,
            loss: 0.176,
          },
          training_time: 7200,
          created_at: "2026-02-17T09:15:00Z",
          is_favorite: true,
        },
        {
          id: "exp-004",
          name: "EfficientNet-B3 + Bilinear",
          status: "completed",
          config: {
            backbone: "efficientnet_b3",
            fusion: "bilinear",
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 50,
          },
          metrics: {
            accuracy: 0.901,
            precision: 0.895,
            recall: 0.889,
            f1_score: 0.892,
            auc: 0.953,
            loss: 0.215,
          },
          training_time: 4200,
          created_at: "2026-02-18T11:45:00Z",
          is_favorite: false,
        },
      ];

      setExperiments(mockExperiments);
    } catch (error) {
      message.error("Failed to fetch experiments");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = () => {
    if (selectedExperiments.length < 2) {
      message.warning("Please select at least 2 experiments to compare");
      return;
    }

    setComparing(true);
    try {
      const selected = experiments.filter((exp) =>
        selectedExperiments.includes(exp.id)
      );

      const metrics: ComparisonMetrics[] = [
        {
          metric: "Accuracy",
          experiments: {},
          best_experiment: "",
        },
        {
          metric: "Precision",
          experiments: {},
          best_experiment: "",
        },
        {
          metric: "Recall",
          experiments: {},
          best_experiment: "",
        },
        {
          metric: "F1 Score",
          experiments: {},
          best_experiment: "",
        },
        {
          metric: "AUC",
          experiments: {},
          best_experiment: "",
        },
        {
          metric: "Loss",
          experiments: {},
          best_experiment: "",
        },
      ];

      selected.forEach((exp) => {
        metrics[0].experiments[exp.name] = exp.metrics.accuracy;
        metrics[1].experiments[exp.name] = exp.metrics.precision;
        metrics[2].experiments[exp.name] = exp.metrics.recall;
        metrics[3].experiments[exp.name] = exp.metrics.f1_score;
        metrics[4].experiments[exp.name] = exp.metrics.auc || 0;
        metrics[5].experiments[exp.name] = exp.metrics.loss;
      });

      // Find best experiment for each metric
      metrics.forEach((m) => {
        const isLowerBetter = m.metric === "Loss";
        const values = Object.entries(m.experiments);
        const best = values.reduce((prev, curr) =>
          isLowerBetter
            ? curr[1] < prev[1]
              ? curr
              : prev
            : curr[1] > prev[1]
            ? curr
            : prev
        );
        m.best_experiment = best[0];
      });

      setComparisonData(metrics);
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

    message.loading(`Generating ${format.toUpperCase()} report...`, 0);
    try {
      // Mock API call - replace with actual implementation
      await new Promise((resolve) => setTimeout(resolve, 2000));
      message.destroy();
      message.success(`${format.toUpperCase()} report generated successfully`);
    } catch (error) {
      message.destroy();
      message.error("Failed to generate report");
      console.error(error);
    }
  };

  const toggleFavorite = (experimentId: string) => {
    setExperiments((prev) =>
      prev.map((exp) =>
        exp.id === experimentId
          ? { ...exp, is_favorite: !exp.is_favorite }
          : exp
      )
    );
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
          onClick={() => toggleFavorite(record.id)}
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
        <span style={{ fontWeight: 500 }}>{(value * 100).toFixed(2)}%</span>
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
          const isBest = record.best_experiment === exp?.name;
          const isLowerBetter = record.metric === "Loss";

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
    return experiments.reduce((prev, curr) =>
      curr.metrics.accuracy > prev.metrics.accuracy ? curr : prev
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
            <Button icon={<ReloadOutlined />} onClick={fetchExperiments}>
              Refresh
            </Button>
            <Button
              type="primary"
              icon={<BarChartOutlined />}
              onClick={handleCompare}
              loading={comparing}
              disabled={selectedExperiments.length < 2}
            >
              Compare Selected ({selectedExperiments.length})
            </Button>
            <Button
              icon={<FileWordOutlined />}
              onClick={() => handleGenerateReport("word")}
              disabled={selectedExperiments.length === 0}
            >
              Export Word
            </Button>
            <Button
              icon={<FilePdfOutlined />}
              onClick={() => handleGenerateReport("pdf")}
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
