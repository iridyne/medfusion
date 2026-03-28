import { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  Alert,
  Badge,
  Button,
  Card,
  Col,
  Empty,
  Form,
  Input,
  InputNumber,
  Modal,
  Progress,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  message,
  Tabs,
} from "antd";
import {
  DisconnectOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  PlusOutlined,
  ReloadOutlined,
  StopOutlined,
  WifiOutlined,
} from "@ant-design/icons";
import type { ColumnsType } from "antd/es/table";
import { EChartsOption } from "echarts";

import { getDatasets } from "../api/datasets";
import { getModels, type Model as ResultModel } from "../api/models";
import trainingApi, {
  type TrainingHistoryEntry,
  type TrainingJob,
  type TrainingJobCreate,
} from "../api/training";
import LazyChart from "../components/LazyChart";
import WebSocketClient from "../utils/websocket";
import PageScaffold from "@/components/layout/PageScaffold";

interface DatasetOption {
  id: string;
  name: string;
  numClasses: number;
  dataPath?: string;
}

interface MetricHistory {
  epochs: number[];
  trainLoss: number[];
  valLoss: number[];
  trainAcc: number[];
  valAcc: number[];
  learningRate: number[];
}

interface CreateTrainingValues {
  experimentName: string;
  datasetId: string;
  backbone: string;
  numClasses: number;
  epochs: number;
  batchSize: number;
  learningRate: number;
}

const EMPTY_HISTORY: MetricHistory = {
  epochs: [],
  trainLoss: [],
  valLoss: [],
  trainAcc: [],
  valAcc: [],
  learningRate: [],
};

const BACKBONE_OPTIONS = [
  "resnet18",
  "resnet34",
  "resnet50",
  "efficientnet_b0",
  "vit_b16",
  "swin_tiny",
];

function toMetricHistory(entries: TrainingHistoryEntry[]): MetricHistory {
  if (!entries.length) {
    return EMPTY_HISTORY;
  }

  return {
    epochs: entries.map((entry) => entry.epoch),
    trainLoss: entries.map((entry) => Number(entry.train_loss ?? 0)),
    valLoss: entries.map((entry) => Number(entry.val_loss ?? entry.train_loss ?? 0)),
    trainAcc: entries.map((entry) => Number(entry.train_accuracy ?? 0)),
    valAcc: entries.map(
      (entry) => Number(entry.val_accuracy ?? entry.train_accuracy ?? 0),
    ),
    learningRate: entries.map((entry) => Number(entry.learning_rate ?? 0)),
  };
}

export default function TrainingMonitor() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [form] = Form.useForm<CreateTrainingValues>();
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [datasets, setDatasets] = useState<DatasetOption[]>([]);
  const [latestOutputs, setLatestOutputs] = useState<ResultModel[]>([]);
  const [selectedJob, setSelectedJob] = useState<string>("");
  const [metricHistory, setMetricHistory] = useState<MetricHistory>(EMPTY_HISTORY);
  const [wsConnected, setWsConnected] = useState(false);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const wsClient = useRef<WebSocketClient | null>(null);

  const currentJob = jobs.find((job) => job.id === selectedJob);

  const loadJobs = async () => {
    try {
      const jobList = await trainingApi.listJobs();
      setJobs(jobList);
      setSelectedJob((current) => current || jobList[0]?.id || "");
    } catch (error) {
      console.error("Failed to load training jobs:", error);
      message.error("加载训练任务失败");
    }
  };

  const loadDatasets = async () => {
    try {
      const data = await getDatasets({ limit: 200 });
      const mapped = (data || []).map((item: any) => ({
        id: String(item.id),
        name: item.name,
        numClasses: item.num_classes ?? 2,
        dataPath: item.data_path,
      }));
      setDatasets(mapped);
    } catch (error) {
      console.error("Failed to load datasets:", error);
      message.error("加载数据集失败");
    }
  };

  const loadOutputs = async () => {
    try {
      const data = await getModels({ limit: 6 });
      setLatestOutputs(data || []);
    } catch (error) {
      console.error("Failed to load outputs:", error);
    }
  };

  const loadJobHistory = async (jobId: string) => {
    if (!jobId) {
      setMetricHistory(EMPTY_HISTORY);
      return;
    }

    try {
      const entries = await trainingApi.getJobHistory(jobId);
      setMetricHistory(toMetricHistory(entries));
    } catch (error) {
      console.error("Failed to load training history:", error);
      setMetricHistory(EMPTY_HISTORY);
    }
  };

  useEffect(() => {
    void loadJobs();
    void loadDatasets();
    void loadOutputs();

    const interval = window.setInterval(() => {
      void loadJobs();
      void loadOutputs();
    }, 3000);

    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    if (searchParams.get("action") !== "start") {
      return;
    }
    setCreateModalOpen(true);
    const next = new URLSearchParams(searchParams);
    next.delete("action");
    setSearchParams(next, { replace: true });
  }, [searchParams, setSearchParams]);

  useEffect(() => {
    void loadJobHistory(selectedJob);
  }, [selectedJob]);

  useEffect(() => {
    if (!selectedJob) {
      setWsConnected(false);
      return;
    }

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/api/training/ws/${selectedJob}`;

    wsClient.current?.close();
    wsClient.current = new WebSocketClient({
      url: wsUrl,
      onOpen: () => setWsConnected(true),
      onClose: () => setWsConnected(false),
      onError: (error) => {
        console.error("WebSocket error:", error);
        setWsConnected(false);
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

  const appendMetricSnapshot = (snapshot: {
    epoch?: number;
    trainLoss?: number;
    valLoss?: number;
    trainAccuracy?: number;
    valAccuracy?: number;
    learningRate?: number;
  }) => {
    const { epoch, trainLoss, valLoss, trainAccuracy, valAccuracy, learningRate } =
      snapshot;

    if (
      !epoch ||
      trainLoss === undefined ||
      valLoss === undefined ||
      trainAccuracy === undefined ||
      valAccuracy === undefined ||
      epoch <= 0
    ) {
      return;
    }

    setMetricHistory((prev) => {
      if (prev.epochs[prev.epochs.length - 1] === epoch) {
        return prev;
      }

      return {
        epochs: [...prev.epochs, epoch],
        trainLoss: [...prev.trainLoss, trainLoss],
        valLoss: [...prev.valLoss, valLoss],
        trainAcc: [...prev.trainAcc, trainAccuracy],
        valAcc: [...prev.valAcc, valAccuracy],
        learningRate: [
          ...prev.learningRate,
          learningRate ?? prev.learningRate[prev.learningRate.length - 1] ?? 0.001,
        ],
      };
    });
  };

  const handleWebSocketMessage = (data: any) => {
    if (data.type === "status_update") {
      setJobs((prevJobs) => {
        const existing = prevJobs.find((job) => job.id === data.job_id);
        if (!existing) {
          return [
            {
              id: data.job_id,
              name: data.experiment_name || data.job_id,
              status: data.status,
              progress: Math.round(data.progress ?? 0),
              epoch: data.epoch ?? 0,
              totalEpochs: data.total_epochs ?? 0,
              loss: data.loss ?? 0,
              accuracy: data.accuracy ?? 0,
              startTime: "",
            },
            ...prevJobs,
          ];
        }

        return prevJobs.map((job) =>
          job.id === data.job_id
            ? {
                ...job,
                status: data.status,
                progress: Math.round(data.progress ?? job.progress),
                epoch: data.epoch ?? job.epoch,
                loss: data.loss ?? job.loss,
                accuracy: data.accuracy ?? job.accuracy,
              }
            : job,
        );
      });

      if (data.job_id === selectedJob) {
        appendMetricSnapshot({
          epoch: data.epoch,
          trainLoss: data.train_loss,
          valLoss: data.val_loss ?? data.loss,
          trainAccuracy: data.train_accuracy,
          valAccuracy: data.val_accuracy ?? data.accuracy,
          learningRate: data.learning_rate,
        });
      }
      return;
    }

    if (data.type === "training_complete") {
      message.success("训练任务已完成，模型已同步到模型库");
      void loadJobs();
      void loadOutputs();
      if (data.job_id === selectedJob) {
        void loadJobHistory(data.job_id);
      }
      return;
    }

    if (data.type === "error") {
      message.error(data.message || "训练任务出现错误");
    }
  };

  const handleJobControl = async (
    jobId: string,
    action: "pause" | "resume" | "stop",
  ) => {
    try {
      if (action === "pause") {
        await trainingApi.pauseJob(jobId);
        message.success("训练任务已暂停");
      } else if (action === "resume") {
        await trainingApi.resumeJob(jobId);
        message.success("训练任务已恢复");
      } else {
        await trainingApi.stopJob(jobId);
        message.success("训练任务已停止");
      }

      void loadJobs();
    } catch (error) {
      console.error("Failed to control job:", error);
      message.error("更新训练任务状态失败");
    }
  };

  const handleDatasetChange = (datasetId: string) => {
    const dataset = datasets.find((item) => item.id === datasetId);
    if (dataset) {
      form.setFieldsValue({
        numClasses: dataset.numClasses || 2,
      });
    }
  };

  const handleCreateJob = async () => {
    try {
      const values = await form.validateFields();
      const dataset = datasets.find((item) => item.id === values.datasetId);
      if (!dataset) {
        message.error("请选择有效的数据集");
        return;
      }

      setSubmitting(true);
      const payload: TrainingJobCreate = {
        experiment_name: values.experimentName,
        training_model_config: {
          backbone: values.backbone,
          num_classes: values.numClasses,
        },
        dataset_config: {
          dataset_id: dataset.id,
          dataset: dataset.name,
          data_path: dataset.dataPath,
          num_classes: dataset.numClasses,
        },
        training_config: {
          epochs: values.epochs,
          batch_size: values.batchSize,
          learning_rate: values.learningRate,
        },
      };

      const response = await trainingApi.createJob(payload);
      message.success("训练任务已启动");
      setCreateModalOpen(false);
      form.resetFields();
      setSelectedJob(response.job_id);
      setMetricHistory(EMPTY_HISTORY);
      await loadJobs();
    } catch (error: any) {
      if (error?.errorFields) {
        return;
      }

      console.error("Failed to create job:", error);
      message.error(error?.response?.data?.detail || "启动训练任务失败");
    } finally {
      setSubmitting(false);
    }
  };

  const handleRefresh = () => {
    void loadJobs();
    void loadDatasets();
    void loadOutputs();
    message.success("训练看板已刷新");
  };

  const formatDateTime = (value: string) => {
    if (!value) {
      return "-";
    }
    return new Date(value).toLocaleString("zh-CN");
  };

  const lossOption: EChartsOption = {
    title: { text: "损失曲线", left: "center" },
    tooltip: { trigger: "axis" },
    legend: { data: ["训练损失", "验证损失"], bottom: 10 },
    grid: { left: "3%", right: "4%", bottom: "15%", containLabel: true },
    xAxis: { type: "category", data: metricHistory.epochs, name: "Epoch" },
    yAxis: { type: "value", name: "Loss" },
    series: [
      {
        name: "训练损失",
        data: metricHistory.trainLoss,
        type: "line",
        smooth: true,
        itemStyle: { color: "#1677ff" },
      },
      {
        name: "验证损失",
        data: metricHistory.valLoss,
        type: "line",
        smooth: true,
        itemStyle: { color: "#52c41a" },
      },
    ],
  };

  const accuracyOption: EChartsOption = {
    title: { text: "准确率曲线", left: "center" },
    tooltip: { trigger: "axis" },
    legend: { data: ["训练准确率", "验证准确率"], bottom: 10 },
    grid: { left: "3%", right: "4%", bottom: "15%", containLabel: true },
    xAxis: { type: "category", data: metricHistory.epochs, name: "Epoch" },
    yAxis: { type: "value", name: "Accuracy", min: 0, max: 1 },
    series: [
      {
        name: "训练准确率",
        data: metricHistory.trainAcc,
        type: "line",
        smooth: true,
        itemStyle: { color: "#1677ff" },
      },
      {
        name: "验证准确率",
        data: metricHistory.valAcc,
        type: "line",
        smooth: true,
        itemStyle: { color: "#faad14" },
      },
    ],
  };

  const learningRateOption: EChartsOption = {
    title: { text: "学习率变化", left: "center" },
    tooltip: { trigger: "axis" },
    grid: { left: "3%", right: "4%", bottom: "10%", containLabel: true },
    xAxis: { type: "category", data: metricHistory.epochs, name: "Epoch" },
    yAxis: { type: "value", name: "Learning Rate" },
    series: [
      {
        name: "学习率",
        data: metricHistory.learningRate,
        type: "line",
        smooth: true,
        itemStyle: { color: "#722ed1" },
      },
    ],
  };

  const columns: ColumnsType<TrainingJob> = [
    {
      title: "任务名称",
      dataIndex: "name",
      key: "name",
    },
    {
      title: "状态",
      dataIndex: "status",
      key: "status",
      render: (status: string) => {
        const colorMap: Record<string, string> = {
          queued: "default",
          running: "processing",
          paused: "warning",
          completed: "success",
          failed: "error",
          stopped: "default",
        };
        const labelMap: Record<string, string> = {
          queued: "排队中",
          running: "训练中",
          paused: "已暂停",
          completed: "已完成",
          failed: "失败",
          stopped: "已停止",
        };
        return <Tag color={colorMap[status]}>{labelMap[status] || status}</Tag>;
      },
    },
    {
      title: "进度",
      dataIndex: "progress",
      key: "progress",
      render: (progress: number) => <Progress percent={progress} size="small" />,
    },
    {
      title: "Epoch",
      key: "epoch",
      render: (_, record) => `${record.epoch}/${record.totalEpochs}`,
    },
    {
      title: "Loss",
      dataIndex: "loss",
      key: "loss",
      render: (loss: number) => loss.toFixed(4),
    },
    {
      title: "Accuracy",
      dataIndex: "accuracy",
      key: "accuracy",
      render: (accuracy: number) => `${(accuracy * 100).toFixed(2)}%`,
    },
    {
      title: "创建时间",
      dataIndex: "startTime",
      key: "startTime",
      render: (value: string) => formatDateTime(value),
    },
    {
      title: "操作",
      key: "action",
      render: (_, record) => (
        <Space>
          {record.status === "running" && (
            <>
              <Button
                size="small"
                icon={<PauseCircleOutlined />}
                onClick={() => handleJobControl(record.id, "pause")}
              >
                暂停
              </Button>
              <Button
                size="small"
                danger
                icon={<StopOutlined />}
                onClick={() => handleJobControl(record.id, "stop")}
              >
                停止
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
              恢复
            </Button>
          )}
          <Button size="small" onClick={() => setSelectedJob(record.id)}>
            查看
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <PageScaffold
      eyebrow="Execution Monitor"
      title="把实验运行状态做成一张可持续阅读的控制面"
      description="这里承接训练任务启动、实时进度观察和最近结果回看。它既是运行中的观察面，也是评估阶段验证主链可执行的证据页。"
      chips={[
        { label: "Real-time telemetry", tone: "rose" },
        { label: "Job orchestration", tone: "teal" },
        { label: "Artifact handoff", tone: "amber" },
      ]}
      actions={
        <>
          <Button icon={<ReloadOutlined />} onClick={handleRefresh}>
            刷新
          </Button>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => setCreateModalOpen(true)}
          >
            启动训练
          </Button>
        </>
      }
      aside={
        <div className="hero-aside-panel">
          <span className="hero-aside-panel__label">Live lane</span>
          <div className="hero-aside-panel__value">
            {currentJob ? currentJob.name : "等待选择训练任务"}
          </div>
          <div className="hero-aside-panel__copy">
            {currentJob
              ? `当前任务进度 ${currentJob.progress}% ，状态为 ${currentJob.status}。`
              : "先在任务列表中选择一个训练任务，或直接启动新的训练任务。"}
          </div>
          <Badge
            status={wsConnected ? "success" : "default"}
            text={wsConnected ? "实时连接已建立" : "当前未建立实时连接"}
          />
          {currentJob ? (
            <Progress
              percent={currentJob.progress}
              status={currentJob.status === "running" ? "active" : "normal"}
            />
          ) : (
            <div className="surface-note">训练曲线会在选择任务后出现在监控面板中。</div>
          )}
        </div>
      }
      metrics={[
        {
          label: "Training jobs",
          value: jobs.length.toLocaleString(),
          hint: "All tracked runs",
          tone: "blue",
        },
        {
          label: "Running now",
          value: jobs.filter((job) => job.status === "running").length.toLocaleString(),
          hint: wsConnected ? "实时连接在线" : "等待新的 websocket 连接",
          tone: "rose",
        },
        {
          label: "Completed",
          value: jobs.filter((job) => job.status === "completed").length.toLocaleString(),
          hint: "Finished jobs available for result review",
          tone: "teal",
        },
        {
          label: "Recent outputs",
          value: latestOutputs.length.toLocaleString(),
          hint: "Latest artifacts previewed below",
          tone: "amber",
        },
      ]}
    >
      <Card className="surface-card">
        <Tabs
          defaultActiveKey="jobs"
          items={[
            {
              key: "jobs",
              label: "任务列表",
              children: (
                <Table
                  columns={columns}
                  dataSource={jobs}
                  rowKey="id"
                    pagination={false}
                    locale={{
                      emptyText: (
                        <Empty description="暂无训练任务，先点击右上角“启动训练”创建一个任务。" />
                      ),
                    }}
                  />
              ),
            },
            {
              key: "monitor",
              label: "实时监控",
              children: currentJob ? (
                <div className="stack-grid">
                  <div className="surface-note">
                    <strong>{currentJob.name}</strong>
                    <p>当前任务会在训练过程中持续刷新 loss、accuracy 和 learning rate 曲线。</p>
                    <Progress
                      percent={currentJob.progress}
                      status={currentJob.status === "running" ? "active" : "normal"}
                    />
                  </div>

                  <Row gutter={[16, 16]}>
                    <Col xs={24} md={12} xl={6}>
                      <Card>
                        <Statistic
                          title="当前 Epoch"
                          value={currentJob.epoch}
                          suffix={`/ ${currentJob.totalEpochs}`}
                        />
                      </Card>
                    </Col>
                    <Col xs={24} md={12} xl={6}>
                      <Card>
                        <Statistic
                          title="训练损失"
                          value={currentJob.loss}
                          precision={4}
                        />
                      </Card>
                    </Col>
                    <Col xs={24} md={12} xl={6}>
                      <Card>
                        <Statistic
                          title="训练准确率"
                          value={currentJob.accuracy * 100}
                          precision={2}
                          suffix="%"
                        />
                      </Card>
                    </Col>
                    <Col xs={24} md={12} xl={6}>
                      <Card>
                        <Statistic
                          title="学习率"
                          value={
                            metricHistory.learningRate[
                              metricHistory.learningRate.length - 1
                            ] || 0.001
                          }
                          precision={6}
                        />
                      </Card>
                    </Col>
                  </Row>

                  <Row gutter={[16, 16]}>
                    <Col xs={24} xl={12}>
                      <Card>
                        <LazyChart option={lossOption} style={{ height: 320 }} />
                      </Card>
                    </Col>
                    <Col xs={24} xl={12}>
                      <Card>
                        <LazyChart option={accuracyOption} style={{ height: 320 }} />
                      </Card>
                    </Col>
                  </Row>

                  <Card>
                    <LazyChart option={learningRateOption} style={{ height: 240 }} />
                  </Card>
                </div>
              ) : (
                <Empty description="选择一个训练任务后，这里会显示实时指标和曲线。" />
              ),
            },
            {
              key: "outputs",
              label: "结果输出",
              children:
                latestOutputs.length > 0 ? (
                  <Row gutter={[16, 16]}>
                    {latestOutputs.map((output) => (
                      <Col xs={24} xl={12} key={output.id}>
                        <Card
                          title={output.name}
                          extra={
                            <Space wrap>
                              <Tag color="blue">{output.backbone}</Tag>
                              <Tag color="purple">
                                {output.format?.toUpperCase() || "PYTORCH"}
                              </Tag>
                            </Space>
                          }
                        >
                          <Space
                            direction="vertical"
                            size={12}
                            style={{ width: "100%" }}
                          >
                            <div style={{ color: "var(--text-secondary)" }}>
                              数据集：{output.dataset_name || "-"}
                            </div>
                            <Row gutter={12}>
                              <Col span={8}>
                                <Statistic
                                  title="Accuracy"
                                  value={(output.accuracy ?? 0) * 100}
                                  precision={2}
                                  suffix="%"
                                />
                              </Col>
                              <Col span={8}>
                                <Statistic
                                  title="AUC"
                                  value={output.visualizations?.roc_curve?.auc ?? 0}
                                  precision={4}
                                />
                              </Col>
                              <Col span={8}>
                                <Statistic
                                  title="Attention"
                                  value={output.visualizations?.attention_maps?.length || 0}
                                />
                              </Col>
                            </Row>
                            <Alert
                              type="success"
                              showIcon
                              message="结果文件已生成"
                              description={`共 ${output.result_files?.length || 0} 个结果产物，包括模型权重、训练配置、结果摘要和日志。`}
                            />
                            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                              ROC AUC: {output.visualizations?.roc_curve?.auc?.toFixed(4) || "-"}{" "}
                              | Loss: {output.loss?.toFixed(4) || "-"} | 创建时间:{" "}
                              {output.created_at
                                ? new Date(output.created_at).toLocaleString("zh-CN")
                                : "-"}
                            </div>
                            <Button onClick={() => navigate("/models")}>
                              查看完整结果页
                            </Button>
                          </Space>
                        </Card>
                      </Col>
                    ))}
                  </Row>
                ) : (
                  <Empty description="训练完成后，这里会集中展示 ROC/AUC、attention 热力图和结果文件概览。" />
                ),
            },
          ]}
        />
      </Card>

      <Modal
        title="启动训练任务"
        open={createModalOpen}
        onCancel={() => setCreateModalOpen(false)}
        onOk={() => void handleCreateJob()}
        okText="启动训练"
        cancelText="取消"
        confirmLoading={submitting}
        width={720}
        rootClassName="surface-modal"
      >
        {datasets.length === 0 && (
          <Alert
            type="warning"
            showIcon
            style={{ marginBottom: 16 }}
            message="当前还没有可用数据集"
            description="请先到数据集页面登记一个本地目录，再回来启动训练。"
          />
        )}

        <Form
          form={form}
          layout="vertical"
          initialValues={{
            experimentName: `run-${new Date().toISOString().slice(11, 19).replace(/:/g, "")}`,
            backbone: "resnet18",
            numClasses: 2,
            epochs: 12,
            batchSize: 8,
            learningRate: 0.001,
          }}
        >
          <Form.Item
            label="实验名称"
            name="experimentName"
            rules={[{ required: true, message: "请输入实验名称" }]}
          >
            <Input placeholder="例如：chest-xray-run" />
          </Form.Item>

          <Form.Item
            label="数据集"
            name="datasetId"
            rules={[{ required: true, message: "请选择一个数据集" }]}
          >
            <Select
              placeholder="选择要用于训练的数据集"
              onChange={handleDatasetChange}
              options={datasets.map((dataset) => ({
                value: dataset.id,
                label: `${dataset.name} (${dataset.numClasses} 类)`,
              }))}
            />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="Backbone"
                name="backbone"
                rules={[{ required: true, message: "请选择骨干网络" }]}
              >
                <Select
                  options={BACKBONE_OPTIONS.map((backbone) => ({
                    value: backbone,
                    label: backbone,
                  }))}
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="类别数"
                name="numClasses"
                rules={[{ required: true, message: "请输入类别数" }]}
              >
                <InputNumber min={2} style={{ width: "100%" }} />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                label="Epoch"
                name="epochs"
                rules={[{ required: true, message: "请输入训练轮数" }]}
              >
                <InputNumber min={1} max={200} style={{ width: "100%" }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                label="Batch Size"
                name="batchSize"
                rules={[{ required: true, message: "请输入 batch size" }]}
              >
                <InputNumber min={1} max={256} style={{ width: "100%" }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                label="Learning Rate"
                name="learningRate"
                rules={[{ required: true, message: "请输入学习率" }]}
              >
                <InputNumber
                  min={0.000001}
                  max={1}
                  step={0.0001}
                  style={{ width: "100%" }}
                />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>
    </PageScaffold>
  );
}
