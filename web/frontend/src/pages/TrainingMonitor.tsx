import { useEffect, useState, useRef } from "react";
import {
  Card,
  Progress,
  Statistic,
  Row,
  Col,
  Table,
  Tag,
  Button,
  Space,
  message,
  Tabs,
  Badge,
} from "antd";
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  WifiOutlined,
  DisconnectOutlined,
} from "@ant-design/icons";
import { EChartsOption } from "echarts";
import { useTranslation } from "react-i18next";
import WebSocketClient from "../utils/websocket";
import trainingApi from "../api/training";
import LazyChart from "../components/LazyChart";

interface TrainingJob {
  id: string;
  name: string;
  status: "running" | "paused" | "completed" | "failed";
  progress: number;
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  startTime: string;
}

interface MetricHistory {
  epochs: number[];
  trainLoss: number[];
  valLoss: number[];
  trainAcc: number[];
  valAcc: number[];
  learningRate: number[];
}

export default function TrainingMonitor() {
  const { t } = useTranslation();
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [selectedJob, setSelectedJob] = useState<string>("");
  const [metricHistory, setMetricHistory] = useState<MetricHistory>({
    epochs: [],
    trainLoss: [],
    valLoss: [],
    trainAcc: [],
    valAcc: [],
    learningRate: [],
  });
  const [wsConnected, setWsConnected] = useState(false);
  const wsClient = useRef<WebSocketClient | null>(null);

  const currentJob = jobs.find((j) => j.id === selectedJob);

  // 初始化 WebSocket 连接
  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.hostname}:8000/ws/training/${selectedJob || "all"}`;

    wsClient.current = new WebSocketClient({
      url: wsUrl,
      onOpen: () => {
        console.log("WebSocket connected");
        setWsConnected(true);
        message.success(t("training.wsConnected"));
      },
      onClose: () => {
        console.log("WebSocket disconnected");
        setWsConnected(false);
      },
      onError: (error) => {
        console.error("WebSocket error:", error);
        message.error(t("training.wsError"));
      },
      onMessage: (data) => {
        handleWebSocketMessage(data);
      },
    });

    wsClient.current.connect();

    return () => {
      wsClient.current?.close();
    };
  }, [selectedJob]);

  // 处理 WebSocket 消息
  const handleWebSocketMessage = (data: any) => {
    console.log("Received WebSocket message:", data);

    switch (data.type) {
      case "status_update":
        // 更新任务状态
        setJobs((prevJobs) =>
          prevJobs.map((job) =>
            job.id === data.job_id
              ? {
                  ...job,
                  status: data.status,
                  progress: data.progress || job.progress,
                  epoch: data.epoch || job.epoch,
                  loss: data.loss || job.loss,
                  accuracy: data.accuracy || job.accuracy,
                }
              : job,
          ),
        );
        break;

      case "batch_progress":
        // 批次进度更新（可选：显示更细粒度的进度）
        console.log(`Batch ${data.batch}/${data.total_batches} completed`);
        break;

      case "epoch_complete":
        // Epoch 完成，更新指标历史
        setMetricHistory((prev) => ({
          epochs: [...prev.epochs, data.epoch],
          trainLoss: [...prev.trainLoss, data.train_loss],
          valLoss: [...prev.valLoss, data.val_loss],
          trainAcc: [...prev.trainAcc, data.train_acc],
          valAcc: [...prev.valAcc, data.val_acc],
          learningRate: [...prev.learningRate, data.learning_rate],
        }));
        break;

      case "training_complete":
        // 训练完成
        message.success(t("training.trainingComplete", { jobId: data.job_id }));
        setJobs((prevJobs) =>
          prevJobs.map((job) =>
            job.id === data.job_id
              ? { ...job, status: "completed", progress: 100 }
              : job,
          ),
        );
        break;

      case "error":
        // 错误消息
        message.error(t("training.trainingError", { message: data.message }));
        break;

      case "heartbeat":
        // 心跳消息，忽略
        break;

      default:
        console.warn("Unknown message type:", data.type);
    }
  };

  // 加载训练任务列表
  const loadJobs = async () => {
    try {
      const response = await trainingApi.listJobs();
      setJobs(response.data);
      if (response.data.length > 0 && !selectedJob) {
        setSelectedJob(response.data[0].id);
      }
    } catch (error) {
      console.error("Failed to load training jobs:", error);
      message.error(t("training.loadJobsError"));
    }
  };

  // 初始加载
  useEffect(() => {
    loadJobs();
  }, []);

  // 任务控制
  const handleJobControl = async (
    jobId: string,
    action: "pause" | "resume" | "stop",
  ) => {
    try {
      if (action === "pause") {
        await trainingApi.pauseJob(jobId);
        message.success(t("training.pauseSuccess"));
      } else if (action === "resume") {
        await trainingApi.resumeJob(jobId);
        message.success(t("training.resumeSuccess"));
      } else if (action === "stop") {
        await trainingApi.stopJob(jobId);
        message.success(t("training.stopSuccess"));
      }

      // 通过 WebSocket 发送控制命令
      wsClient.current?.send({
        type: "control",
        job_id: jobId,
        action: action,
      });
    } catch (error) {
      console.error("Failed to control job:", error);
      message.error(t("training.controlError", { action }));
    }
  };

  const handleRefresh = () => {
    loadJobs();
    message.success(t("training.refreshSuccess"));
  };

  const lossOption: EChartsOption = {
    title: { text: t("training.lossChart"), left: "center" },
    tooltip: { trigger: "axis" },
    legend: {
      data: [t("training.trainLoss"), t("training.valLoss")],
      bottom: 10,
    },
    grid: { left: "3%", right: "4%", bottom: "15%", containLabel: true },
    xAxis: {
      type: "category",
      data: metricHistory.epochs,
      name: "Epoch",
    },
    yAxis: { type: "value", name: "Loss" },
    series: [
      {
        name: t("training.trainLoss"),
        data: metricHistory.trainLoss,
        type: "line",
        smooth: true,
        itemStyle: { color: "#1890ff" },
      },
      {
        name: t("training.valLoss"),
        data: metricHistory.valLoss,
        type: "line",
        smooth: true,
        itemStyle: { color: "#52c41a" },
      },
    ],
  };

  const accuracyOption: EChartsOption = {
    title: { text: t("training.accuracyChart"), left: "center" },
    tooltip: { trigger: "axis" },
    legend: {
      data: [t("training.trainAcc"), t("training.valAcc")],
      bottom: 10,
    },
    grid: { left: "3%", right: "4%", bottom: "15%", containLabel: true },
    xAxis: {
      type: "category",
      data: metricHistory.epochs,
      name: "Epoch",
    },
    yAxis: { type: "value", name: "Accuracy", min: 0, max: 1 },
    series: [
      {
        name: t("training.trainAcc"),
        data: metricHistory.trainAcc,
        type: "line",
        smooth: true,
        itemStyle: { color: "#1890ff" },
      },
      {
        name: t("training.valAcc"),
        data: metricHistory.valAcc,
        type: "line",
        smooth: true,
        itemStyle: { color: "#52c41a" },
      },
    ],
  };

  const learningRateOption: EChartsOption = {
    title: { text: t("training.learningRateChart"), left: "center" },
    tooltip: { trigger: "axis" },
    grid: { left: "3%", right: "4%", bottom: "10%", containLabel: true },
    xAxis: {
      type: "category",
      data: metricHistory.epochs,
      name: "Epoch",
    },
    yAxis: { type: "value", name: "Learning Rate" },
    series: [
      {
        data: metricHistory.learningRate,
        type: "line",
        smooth: true,
        itemStyle: { color: "#faad14" },
      },
    ],
  };

  const columns = [
    {
      title: t("training.jobName"),
      dataIndex: "name",
      key: "name",
    },
    {
      title: t("training.status"),
      dataIndex: "status",
      key: "status",
      render: (status: string) => {
        const colorMap: Record<string, string> = {
          running: "blue",
          paused: "orange",
          completed: "green",
          failed: "red",
        };
        return (
          <Tag color={colorMap[status]}>{t(`training.status_${status}`)}</Tag>
        );
      },
    },
    {
      title: t("training.progress"),
      dataIndex: "progress",
      key: "progress",
      render: (progress: number) => (
        <Progress percent={progress} size="small" />
      ),
    },
    {
      title: t("training.epoch"),
      key: "epoch",
      render: (_: any, record: TrainingJob) =>
        `${record.epoch}/${record.totalEpochs}`,
    },
    {
      title: t("training.loss"),
      dataIndex: "loss",
      key: "loss",
      render: (loss: number) => loss.toFixed(4),
    },
    {
      title: t("training.accuracy"),
      dataIndex: "accuracy",
      key: "accuracy",
      render: (acc: number) => `${(acc * 100).toFixed(2)}%`,
    },
    {
      title: t("training.startTime"),
      dataIndex: "startTime",
      key: "startTime",
    },
    {
      title: t("training.actions"),
      key: "action",
      render: (_: any, record: TrainingJob) => (
        <Space>
          {record.status === "running" && (
            <>
              <Button
                size="small"
                icon={<PauseCircleOutlined />}
                onClick={() => handleJobControl(record.id, "pause")}
              >
                {t("training.pause")}
              </Button>
              <Button
                size="small"
                danger
                icon={<StopOutlined />}
                onClick={() => handleJobControl(record.id, "stop")}
              >
                {t("training.stop")}
              </Button>
            </>
          )}
          {record.status === "paused" && (
            <Button
              size="small"
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={() => handleJobControl(record.id, "resume")}
            >
              {t("training.resume")}
            </Button>
          )}
          <Button size="small" onClick={() => setSelectedJob(record.id)}>
            {t("common.detail")}
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: 24 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <h1>{t("nav.training")}</h1>
        <Space>
          <Badge
            status={wsConnected ? "success" : "error"}
            text={
              wsConnected ? t("training.connected") : t("training.disconnected")
            }
          />
          {wsConnected ? (
            <WifiOutlined style={{ color: "#52c41a" }} />
          ) : (
            <DisconnectOutlined style={{ color: "#ff4d4f" }} />
          )}
          <Button icon={<ReloadOutlined />} onClick={handleRefresh}>
            {t("common.refresh")}
          </Button>
        </Space>
      </div>

      <Tabs
        defaultActiveKey="1"
        items={[
          {
            key: "1",
            label: t("training.jobList"),
            children: (
              <Card>
                <Table
                  columns={columns}
                  dataSource={jobs}
                  rowKey="id"
                  pagination={false}
                />
              </Card>
            ),
          },
          {
            key: "2",
            label: t("training.realTimeMonitor"),
            children: currentJob ? (
              <>
                <Card style={{ marginBottom: 16 }}>
                  <h3>{currentJob.name}</h3>
                  <Progress
                    percent={currentJob.progress}
                    status={
                      currentJob.status === "running" ? "active" : "normal"
                    }
                  />
                </Card>

                <Row gutter={16} style={{ marginBottom: 16 }}>
                  <Col span={6}>
                    <Card>
                      <Statistic
                        title={t("training.currentEpoch")}
                        value={currentJob.epoch}
                        suffix={`/ ${currentJob.totalEpochs}`}
                      />
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card>
                      <Statistic
                        title={t("training.trainLoss")}
                        value={currentJob.loss}
                        precision={4}
                      />
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card>
                      <Statistic
                        title={t("training.trainAcc")}
                        value={currentJob.accuracy * 100}
                        precision={2}
                        suffix="%"
                      />
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card>
                      <Statistic
                        title={t("training.learningRate")}
                        value={
                          metricHistory.learningRate[
                            metricHistory.learningRate.length - 1
                          ]
                        }
                        precision={6}
                      />
                    </Card>
                  </Col>
                </Row>

                <Row gutter={16} style={{ marginBottom: 16 }}>
                  <Col span={12}>
                    <Card>
                      <LazyChart option={lossOption} style={{ height: 350 }} />
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card>
                      <LazyChart
                        option={accuracyOption}
                        style={{ height: 350 }}
                      />
                    </Card>
                  </Col>
                </Row>

                <Row gutter={16}>
                  <Col span={24}>
                    <Card>
                      <LazyChart
                        option={learningRateOption}
                        style={{ height: 250 }}
                      />
                    </Card>
                  </Col>
                </Row>
              </>
            ) : (
              <Card>
                <p>{t("training.selectJobPrompt")}</p>
              </Card>
            ),
          },
        ]}
      />
    </div>
  );
}
