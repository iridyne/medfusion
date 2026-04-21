import { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import {
  Alert,
  Card,
  Tag,
  Space,
  Button,
  Input,
  InputNumber,
  Select,
  Row,
  Col,
  Statistic,
  Modal,
  Descriptions,
  message,
  Divider,
  Typography,
  Form,
} from "antd";
import {
  ControlOutlined,
  SearchOutlined,
  DownloadOutlined,
  DeleteOutlined,
  EyeOutlined,
  DotChartOutlined,
  ImportOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import {
  deleteModel,
  downloadModel,
  getModels,
  importModelRun,
  type Model as ApiModel,
  type ModelImportRequest,
} from "@/api/models";
import {
  buildTrainingMonitorLink,
  clearTrainingResultHandoffParams,
  parseTrainingResultHandoff,
  type TrainingResultHandoff,
} from "@/api/training";
import ModelResultPanel from "@/components/model/ModelResultPanel";
import VirtualList from "@/components/VirtualList";
import PageScaffold from "@/components/layout/PageScaffold";
import { QUICKSTART_TRAINING_PREFILL } from "@/config/quickstartRun";
import { buildTrainingPrefillQuery } from "@/utils/trainingPrefill";

const { Text } = Typography;

interface Model extends ApiModel {
  displayFormat: "pytorch" | "onnx" | "torchscript";
  numClasses: number;
  params: number;
  createdAt: string;
  descriptionText: string;
  size: number;
}

interface ImportFormValues {
  config_path: string;
  checkpoint_path: string;
  output_dir?: string;
  split: "train" | "val" | "test";
  attention_samples: number;
  survival_time_column?: string;
  survival_event_column?: string;
  importance_sample_limit: number;
  name?: string;
  description?: string;
  tags?: string;
}

interface ModelImportPrefillState {
  importPrefill?: Partial<ImportFormValues>;
  importSource?: string;
}

function mapModelPayload(item: any): Model {
  return {
    ...item,
    name: item.name || `model-${item.id}`,
    backbone: item.backbone || item.architecture || "unknown",
    numClasses: item.num_classes ?? 0,
    params: item.num_parameters ?? item.params ?? 0,
    accuracy: item.accuracy ?? undefined,
    createdAt: item.created_at || "",
    descriptionText: item.description || "",
    size:
      item.file_size ??
      (item.model_size_mb ? Math.round(item.model_size_mb * 1024 * 1024) : 0),
    displayFormat: (item.format || "pytorch") as "pytorch" | "onnx" | "torchscript",
  };
}

export default function ModelLibrary() {
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const importPrefillConsumedRef = useRef(false);
  const [importForm] = Form.useForm<ImportFormValues>();
  const [models, setModels] = useState<Model[]>([]);
  const [filteredModels, setFilteredModels] = useState<Model[]>(models);
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [filterBackbone, setFilterBackbone] = useState<string>("all");
  const [filterFormat, setFilterFormat] = useState<string>("all");
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [resultHandoff, setResultHandoff] = useState<TrainingResultHandoff | null>(null);
  const [attemptedHandoffModelId, setAttemptedHandoffModelId] = useState<number | null>(
    null,
  );

  useEffect(() => {
    filterModels();
  }, [searchText, filterBackbone, filterFormat, models]);

  const loadModels = async (focusModelId?: number) => {
    setLoading(true);
    try {
      const data = await getModels();
      const mapped: Model[] = (data || []).map(mapModelPayload);
      setModels(mapped);

      const targetModelId = focusModelId ?? selectedModel?.id;
      if (targetModelId) {
        const refreshedSelected = mapped.find((item) => item.id === targetModelId) || null;
        setSelectedModel(refreshedSelected);
        if (focusModelId && refreshedSelected) {
          setAttemptedHandoffModelId(null);
          setDetailModalOpen(true);
        }
      }
    } catch (error) {
      console.error("Failed to load models:", error);
      message.error("加载模型列表失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadModels();
  }, []);

  useEffect(() => {
    if (searchParams.get("action") === "import") {
      setImportModalOpen(true);
      const next = new URLSearchParams(searchParams);
      next.delete("action");
      setSearchParams(next, { replace: true });
      return;
    }

    const focusModelId = searchParams.get("modelId") || searchParams.get("model");
    if (focusModelId) {
      const parsedModelId = Number(focusModelId);
      if (!Number.isNaN(parsedModelId)) {
        const matchedFocus = models.find((item) => item.id === parsedModelId) || null;
        if (matchedFocus) {
          setSelectedModel(matchedFocus);
          setDetailModalOpen(true);
          const next = new URLSearchParams(searchParams);
          next.delete("modelId");
          next.delete("model");
          setSearchParams(next, { replace: true });
          return;
        }
        if (!loading && attemptedHandoffModelId !== parsedModelId) {
          setAttemptedHandoffModelId(parsedModelId);
          void loadModels(parsedModelId);
        }
      }
    }

    const handoff = parseTrainingResultHandoff(searchParams);
    if (!handoff?.modelId) {
      return;
    }

    setResultHandoff(handoff);
    const matched = models.find((item) => item.id === handoff.modelId) || null;
    if (matched) {
      setSelectedModel(matched);
      setDetailModalOpen(true);
      setAttemptedHandoffModelId(null);
      setSearchParams(clearTrainingResultHandoffParams(searchParams), { replace: true });
      return;
    }

    if (!loading && attemptedHandoffModelId !== handoff.modelId) {
      setAttemptedHandoffModelId(handoff.modelId);
      void loadModels(handoff.modelId);
    }
  }, [
    searchParams,
    setSearchParams,
    models,
    loading,
    attemptedHandoffModelId,
  ]);

  useEffect(() => {
    if (importPrefillConsumedRef.current) {
      return;
    }

    const state = location.state as ModelImportPrefillState | null;
    const prefill = state?.importPrefill;
    if (!prefill) {
      return;
    }

    importForm.setFieldsValue({
      config_path: prefill.config_path,
      checkpoint_path: prefill.checkpoint_path,
      output_dir: prefill.output_dir,
      split: prefill.split,
      attention_samples: prefill.attention_samples,
      survival_time_column: prefill.survival_time_column,
      survival_event_column: prefill.survival_event_column,
      importance_sample_limit: prefill.importance_sample_limit,
      name: prefill.name,
      description: prefill.description,
      tags: prefill.tags,
    });
    setImportModalOpen(true);
    importPrefillConsumedRef.current = true;
    message.success(
      state?.importSource
        ? `已从 ${state.importSource} 预填导入参数`
        : "已预填导入参数",
    );
  }, [location.state, importForm]);

  const filterModels = () => {
    let filtered = models;

    if (searchText) {
      filtered = filtered.filter(
        (m) =>
          m.name.toLowerCase().includes(searchText.toLowerCase()) ||
          m.descriptionText.toLowerCase().includes(searchText.toLowerCase()),
      );
    }

    if (filterBackbone !== "all") {
      filtered = filtered.filter((m) => m.backbone === filterBackbone);
    }

    if (filterFormat !== "all") {
      filtered = filtered.filter((m) => m.displayFormat === filterFormat);
    }

    setFilteredModels(filtered);
  };

  const formatParams = (params: number) => {
    if (params >= 1_000_000) {
      return `${(params / 1_000_000).toFixed(1)}M`;
    }
    return `${(params / 1_000).toFixed(1)}K`;
  };

  const formatSize = (bytes: number) => {
    if (bytes >= 1_000_000_000) {
      return `${(bytes / 1_000_000_000).toFixed(2)} GB`;
    }
    if (bytes >= 1_000_000) {
      return `${(bytes / 1_000_000).toFixed(2)} MB`;
    }
    return `${(bytes / 1_000).toFixed(2)} KB`;
  };

  const handleViewDetail = (model: Model) => {
    setSelectedModel(model);
    setDetailModalOpen(true);
  };

  const handleDownload = (model: Model) => {
    void (async () => {
      try {
        await downloadModel(Number(model.id), `${model.name}.${model.displayFormat === "onnx" ? "onnx" : "pth"}`);
        message.success(`已开始下载 ${model.name}`);
      } catch (error) {
        console.error("Failed to download model:", error);
        message.error("下载失败");
      }
    })();
  };

  const handleDelete = (model: Model) => {
    Modal.confirm({
      title: "确认删除",
      content: `确定要删除模型 "${model.name}" 吗？`,
      onOk: async () => {
        try {
          await deleteModel(Number(model.id));
          setModels((prev) => prev.filter((m) => m.id !== model.id));
          if (selectedModel?.id === model.id) {
            setSelectedModel(null);
            setDetailModalOpen(false);
          }
          message.success("模型已删除");
        } catch (error) {
          console.error("Failed to delete model:", error);
          message.error("删除失败");
        }
      },
    });
  };

  const handleImport = async (values: ImportFormValues) => {
    const parsedTags = (values.tags || "")
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);

    const payload: ModelImportRequest = {
      config_path: values.config_path,
      checkpoint_path: values.checkpoint_path,
      output_dir: values.output_dir?.trim() || undefined,
      split: values.split,
      attention_samples: values.attention_samples,
      survival_time_column: values.survival_time_column?.trim() || undefined,
      survival_event_column: values.survival_event_column?.trim() || undefined,
      importance_sample_limit: values.importance_sample_limit,
      name: values.name?.trim() || undefined,
      description: values.description?.trim() || undefined,
      tags: parsedTags.length ? parsedTags : undefined,
    };

    setImporting(true);
    try {
      const imported = await importModelRun(payload);
      const mappedImported = mapModelPayload(imported);
      setImportModalOpen(false);
      importForm.resetFields();
      setSelectedModel(mappedImported);
      setDetailModalOpen(true);
      await loadModels(mappedImported.id);
      message.success(`已导入训练结果：${mappedImported.name}`);
    } catch (error: any) {
      console.error("Failed to import model run:", error);
      const detail = error?.response?.data?.detail;
      message.error(typeof detail === "string" ? detail : "导入训练结果失败");
    } finally {
      setImporting(false);
    }
  };

  const backboneOptions = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "vit_b16",
    "vit_b32",
    "swin_tiny",
  ];

  const totalParams = models.reduce((sum, m) => sum + m.params, 0);
  const totalSize = models.reduce((sum, m) => sum + m.size, 0);
  const modelsWithAccuracy = models.filter((m) => m.accuracy !== undefined);
  const avgAccuracy =
    modelsWithAccuracy.length > 0
      ? modelsWithAccuracy.reduce((sum, m) => sum + (m.accuracy || 0), 0) /
        modelsWithAccuracy.length
      : 0;
  const latestModel = models[0];
  const handoffModel =
    (resultHandoff?.modelId &&
      (models.find((item) => item.id === resultHandoff.modelId) ||
        (selectedModel?.id === resultHandoff.modelId ? selectedModel : null))) ||
    null;
  const handoffResolved = Boolean(resultHandoff?.modelId && handoffModel);
  const handoffPending =
    Boolean(resultHandoff?.modelId) &&
    !handoffResolved &&
    !loading &&
    attemptedHandoffModelId === resultHandoff?.modelId;
  const handoffMatchesSelectedModel =
    Boolean(resultHandoff?.modelId) && selectedModel?.id === resultHandoff?.modelId;

  const handleLoopBackToConfig = () => {
    const sourceModel = selectedModel || latestModel || null;
    if (!sourceModel) {
      navigate("/config");
      return;
    }

    const sourceModelAny = sourceModel as Model & {
      project_name?: string;
      output_dir?: string;
    };
    navigate("/config", {
      state: {
        source: "model-library",
        wizardPrefill: {
          projectName: sourceModelAny.project_name || "medfusion-rerun",
          experimentName: `${sourceModel.name}-rerun`,
          description: `基于结果后台模型 ${sourceModel.name} 发起下一轮主线配置`,
          outputDir: sourceModelAny.output_dir,
          backbone: sourceModel.backbone,
          numClasses: sourceModel.numClasses,
        },
      },
    });
  };

  const handleLoopBackToTraining = () => {
    const sourceModel = selectedModel || latestModel || null;
    if (!sourceModel) {
      navigate("/training");
      return;
    }

    const prefillQuery = buildTrainingPrefillQuery({
      experimentName: `${sourceModel.name}-rerun`,
      backbone: sourceModel.backbone || QUICKSTART_TRAINING_PREFILL.backbone,
      numClasses:
        sourceModel.numClasses > 0
          ? sourceModel.numClasses
          : QUICKSTART_TRAINING_PREFILL.numClasses,
      epochs: QUICKSTART_TRAINING_PREFILL.epochs,
      batchSize: QUICKSTART_TRAINING_PREFILL.batchSize,
      learningRate: QUICKSTART_TRAINING_PREFILL.learningRate,
    });

    navigate(`/training?source=model-library&${prefillQuery}`);
  };

  const handleOpenEvaluation = (model?: Model | null) => {
    const sourceModel = model || selectedModel || latestModel || null;
    if (!sourceModel) {
      navigate("/evaluation");
      return;
    }
    navigate("/evaluation", {
      state: {
        configPath: sourceModel.config_path,
        checkpointPath: sourceModel.checkpoint_path || sourceModel.model_path,
        name: `${sourceModel.name}-eval`,
        description: `基于结果后台模型 ${sourceModel.name} 发起独立评估`,
        source: "model-library",
      },
    });
  };

  return (
    <PageScaffold
      eyebrow="Result backend"
      title="正式版结果后台：归档、回流并展示真实训练结果"
      description="Model Library 不只是下载 checkpoint 的地方，它是正式版结果后台。这里集中承接真实 run 回流后的 summary、validation、report 和可视化 artifact，用于复盘、演示和交付。"
      chips={[
        { label: "Result backend", tone: "teal" },
        { label: "Real-run import", tone: "amber" },
        { label: "Artifact delivery", tone: "blue" },
      ]}
      actions={
        <>
          <Button
            icon={<ReloadOutlined />}
            onClick={() => void loadModels()}
            loading={loading}
          >
            刷新
          </Button>
          <Button
            type="primary"
            icon={<ImportOutlined />}
            onClick={() => setImportModalOpen(true)}
          >
            导入训练结果
          </Button>
          <Button icon={<DotChartOutlined />} onClick={() => handleOpenEvaluation()}>
            打开独立评估
          </Button>
        </>
      }
      aside={
        <div className="hero-aside-panel">
          <span className="hero-aside-panel__label">Latest archived run</span>
          <div className="hero-aside-panel__value">
            {latestModel ? latestModel.name : "尚未导入模型产物"}
          </div>
          <div className="hero-aside-panel__copy">
            {latestModel
              ? latestModel.descriptionText ||
                "最近一次训练产物已经被索引，可用于展示多模态研究结果。"
              : "完成一次训练或导入真实 run 后，这里会出现最新归档的结果摘要，并承接训练完成后的直接深链打开。"}
          </div>
          {latestModel ? (
            <>
              <div className="surface-note">
                <strong>Accuracy snapshot</strong>
                <p>
                  {((latestModel.accuracy ?? 0) * 100).toFixed(2)}% | AUC{" "}
                  {(latestModel.visualizations?.roc_curve?.auc ?? latestModel.accuracy ?? 0).toFixed(4)}
                </p>
              </div>
              <Space wrap>
                <Tag color="blue">{latestModel.backbone}</Tag>
                <Tag color="green">
                  {latestModel.dataset_name || "未命名数据集"}
                </Tag>
                <Tag color="purple">{latestModel.displayFormat.toUpperCase()}</Tag>
              </Space>
            </>
          ) : (
            <div className="surface-note">先完成训练或导入结果以填充结果库。</div>
          )}
        </div>
      }
      metrics={[
        {
          label: "Models archived",
          value: models.length.toLocaleString(),
          hint: "Total indexed results",
          tone: "blue",
        },
        {
          label: "Parameter volume",
          value: formatParams(totalParams),
          hint: "Combined across archived models",
          tone: "teal",
        },
        {
          label: "Storage",
          value: formatSize(totalSize),
          hint: "Artifact footprint",
          tone: "amber",
        },
        {
          label: "Average accuracy",
          value: `${(avgAccuracy * 100).toFixed(2)}%`,
          hint: "Across models with recorded accuracy",
          tone: "rose",
        },
      ]}
    >
      {handoffResolved && resultHandoff ? (
        <Alert
          type="success"
          showIcon
          style={{ marginBottom: 16 }}
          message="训练结果已回流到结果后台"
          description={
            <Space direction="vertical" size={12} style={{ width: "100%" }}>
              <div>
                {resultHandoff.jobName ? `训练任务 ${resultHandoff.jobName}` : "训练任务"}
                已经把结果交付到结果后台。
                {resultHandoff.resultModelName || handoffModel?.name
                  ? ` 当前可直接打开结果详情：${resultHandoff.resultModelName || handoffModel?.name}。`
                  : " 当前可直接打开结果详情并继续复盘。"}
              </div>
              <Space wrap>
                <Button
                  type="primary"
                  icon={<EyeOutlined />}
                  onClick={() => {
                    setSelectedModel(handoffModel);
                    setDetailModalOpen(true);
                  }}
                >
                  打开结果详情
                </Button>
                {resultHandoff.jobId ? (
                  <Button onClick={() => navigate(buildTrainingMonitorLink(resultHandoff.jobId!))}>
                    返回训练看板
                  </Button>
                ) : null}
              </Space>
            </Space>
          }
          closable
          onClose={() => setResultHandoff(null)}
        />
      ) : null}

      {handoffPending && resultHandoff ? (
        <Alert
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
          message="结果深链已收到，但结果后台还没定位到对应模型"
          description={
            <Space direction="vertical" size={12} style={{ width: "100%" }}>
              <div>
                {resultHandoff.jobName ? `训练任务 ${resultHandoff.jobName}` : "对应训练任务"}
                已完成，但结果后台暂时还没有查到模型记录。你可以先刷新一次，或回到训练看板确认归档状态。
              </div>
              <Space wrap>
                <Button
                  type="primary"
                  icon={<ReloadOutlined />}
                  onClick={() => void loadModels(resultHandoff.modelId)}
                >
                  重新查询结果
                </Button>
                {resultHandoff.jobId ? (
                  <Button onClick={() => navigate(buildTrainingMonitorLink(resultHandoff.jobId!))}>
                    返回训练看板
                  </Button>
                ) : null}
              </Space>
            </Space>
          }
          closable
          onClose={() => setResultHandoff(null)}
        />
      ) : null}

      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="当前主线步骤：结果（3/3）"
        description={
          <Space wrap>
            <span>这里负责结果复盘与交付；如需重跑可直接回到配置或训练页。</span>
            <Button size="small" icon={<ControlOutlined />} onClick={handleLoopBackToConfig}>
              基于当前结果重开配置
            </Button>
            <Button size="small" icon={<PlayCircleOutlined />} onClick={handleLoopBackToTraining}>
              基于当前结果直接重跑训练
            </Button>
            <Button size="small" icon={<ControlOutlined />} onClick={() => navigate("/config")}>
              回到配置主线
            </Button>
            <Button size="small" icon={<PlayCircleOutlined />} onClick={() => navigate("/training")}>
              回到训练监控
            </Button>
          </Space>
        }
      />

      <Card className="surface-card" loading={loading}>
        <div className="section-heading">
          <div>
            <div className="section-heading__eyebrow">Archive explorer</div>
            <h2 className="section-heading__title">过滤、浏览并打开结果详情</h2>
            <p className="section-heading__description">
              用名称、骨干网络和导出格式快速缩小范围，然后直接进入多模态结果详情面板。
            </p>
          </div>
        </div>

        <Space style={{ marginBottom: 16, width: "100%" }} direction="vertical">
          <Row gutter={[16, 16]}>
            <Col xs={24} xl={12}>
              <Input
                placeholder="搜索模型名称或描述"
                prefix={<SearchOutlined />}
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                allowClear
              />
            </Col>
            <Col xs={24} md={12} xl={6}>
              <Select
                style={{ width: "100%" }}
                placeholder="按骨干网络筛选"
                value={filterBackbone}
                onChange={setFilterBackbone}
              >
                <Select.Option value="all">全部骨干网络</Select.Option>
                {backboneOptions.map((opt) => (
                  <Select.Option key={opt} value={opt}>
                    {opt}
                  </Select.Option>
                ))}
              </Select>
            </Col>
            <Col xs={24} md={12} xl={6}>
              <Select
                style={{ width: "100%" }}
                placeholder="按格式筛选"
                value={filterFormat}
                onChange={setFilterFormat}
              >
                <Select.Option value="all">全部格式</Select.Option>
                <Select.Option value="pytorch">PyTorch</Select.Option>
                <Select.Option value="onnx">ONNX</Select.Option>
                <Select.Option value="torchscript">TorchScript</Select.Option>
              </Select>
            </Col>
          </Row>
        </Space>

        <div className="library-list">
          <VirtualList
            data={filteredModels}
            itemHeight={132}
            renderItem={(model) => (
              <div
                className="library-row"
                style={
                  resultHandoff?.modelId === model.id
                    ? {
                        boxShadow: "0 0 0 1px rgba(22, 119, 255, 0.24) inset",
                        background: "rgba(22, 119, 255, 0.03)",
                      }
                    : undefined
                }
              >
                <div className="library-row__main">
                  <div className="library-row__title">
                    <strong>{model.name}</strong>
                    {resultHandoff?.modelId === model.id ? (
                      <Tag color="gold">结果刚送达</Tag>
                    ) : null}
                    <Tag color="blue">{model.backbone}</Tag>
                    <Tag color="green">{formatParams(model.params)}</Tag>
                    {model.accuracy ? (
                      <Tag color="orange">
                        {(model.accuracy * 100).toFixed(2)}%
                      </Tag>
                    ) : null}
                    <Tag>{model.displayFormat.toUpperCase()}</Tag>
                  </div>
                  <p className="library-row__description">
                    {model.descriptionText || "训练主链自动沉淀的模型记录"}
                  </p>
                  <p className="library-row__meta">
                    数据集: {model.dataset_name || "-"} | 类别数: {model.numClasses} | 文件大小:{" "}
                    {formatSize(model.size)} | 创建时间:{" "}
                    {model.createdAt
                      ? new Date(model.createdAt).toLocaleString("zh-CN")
                      : "-"}
                  </p>
                  <p className="library-row__meta">
                    AUC: {model.visualizations?.roc_curve?.auc?.toFixed(4) || "-"} | Loss:{" "}
                    {model.loss?.toFixed(4) || "-"} | Attention Maps:{" "}
                    {model.visualizations?.attention_maps?.length || 0}
                  </p>
                </div>
                <div className="library-row__actions">
                  <Button
                    size="small"
                    icon={<DotChartOutlined />}
                    onClick={() => handleOpenEvaluation(model)}
                  >
                    评估
                  </Button>
                  <Button
                    size="small"
                    icon={<EyeOutlined />}
                    onClick={() => handleViewDetail(model)}
                  >
                    详情
                  </Button>
                  <Button
                    size="small"
                    icon={<DownloadOutlined />}
                    onClick={() => handleDownload(model)}
                  >
                    下载
                  </Button>
                  <Button
                    size="small"
                    danger
                    icon={<DeleteOutlined />}
                    onClick={() => handleDelete(model)}
                  >
                    删除
                  </Button>
                </div>
              </div>
            )}
          />
        </div>
      </Card>

      <div className="split-grid">
        <Card className="surface-card">
          <div className="section-heading">
            <div>
              <div className="section-heading__eyebrow">Import chain</div>
              <h2 className="section-heading__title">真实 run 如何回流到结果后台</h2>
              <p className="section-heading__description">
                先验证配置、再执行训练、最后导入 artifact；结果后台负责把这些产物组织成一套可浏览、可汇报、可交付的研究证据。
              </p>
            </div>
          </div>
          <pre className="command-block">
            uv run medfusion validate-config --config &lt;config&gt;
            {"\n"}uv run medfusion train --config &lt;config&gt;
            {"\n"}uv run medfusion import-run --config &lt;config&gt; --checkpoint &lt;path&gt;
          </pre>
        </Card>

        <Card className="surface-card">
          <div className="section-heading">
            <div>
              <div className="section-heading__eyebrow">Archive intent</div>
              <h2 className="section-heading__title">为什么正式版必须有结果后台</h2>
              <p className="section-heading__description">
                正式版不能只做到“能训练”，还要做到“能交付结果”。这里帮助评估者判断系统输出是否真实可用，也帮助研究者进行复盘和分享。
              </p>
            </div>
          </div>
          <div className="stack-grid">
            <div className="surface-note">
              <strong>Artifact 完整性</strong>
              <p>不只保留权重，还保留 ROC、混淆矩阵、注意力图、日志和配置。</p>
            </div>
            <div className="surface-note">
              <strong>检索友好</strong>
              <p>名称、骨干网络、格式和描述可以组合过滤，方便找到适合展示的结果。</p>
            </div>
          </div>
        </Card>
      </div>

      <Modal
        title="导入真实训练结果"
        open={importModalOpen}
        onCancel={() => {
          if (!importing) {
            setImportModalOpen(false);
          }
        }}
        onOk={() => importForm.submit()}
        confirmLoading={importing}
        okText="开始导入"
        cancelText="取消"
        destroyOnClose
        width={760}
        rootClassName="surface-modal"
      >
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <Alert
            type="info"
            showIcon
            message="把真实 CLI 训练产物接进结果页"
            description="这里会直接调用 /api/models/import-run：读取 config 和 checkpoint，生成 validation / ROC / 混淆矩阵 / attention artifact，并在可配置时附加 survival 和 SHAP-style 全局变量重要性。"
          />

          <Form<ImportFormValues>
            form={importForm}
            layout="vertical"
            initialValues={{
              split: "test",
              attention_samples: 4,
              importance_sample_limit: 128,
              config_path: "configs/starter/quickstart.yaml",
            }}
            onFinish={(values) => void handleImport(values)}
          >
            <Row gutter={16}>
              <Col span={24}>
                <Form.Item
                  label="配置文件路径"
                  name="config_path"
                  rules={[{ required: true, message: "请输入训练配置 YAML 路径" }]}
                >
                  <Input placeholder="例如：configs/starter/quickstart.yaml" />
                </Form.Item>
              </Col>
              <Col span={24}>
                <Form.Item
                  label="模型权重路径"
                  name="checkpoint_path"
                  rules={[{ required: true, message: "请输入 checkpoint 路径" }]}
                >
                  <Input placeholder="例如：outputs/quickstart/checkpoints/best.pth" />
                </Form.Item>
              </Col>
              <Col span={24}>
                <Form.Item label="结果输出目录（可选）" name="output_dir">
                  <Input placeholder="例如：outputs/quickstart" />
                </Form.Item>
              </Col>
              <Col xs={24} md={8}>
                <Form.Item label="Validation Split" name="split">
                  <Select
                    options={[
                      { label: "test", value: "test" },
                      { label: "val", value: "val" },
                      { label: "train", value: "train" },
                    ]}
                  />
                </Form.Item>
              </Col>
              <Col xs={24} md={8}>
                <Form.Item label="Attention 样本数" name="attention_samples">
                  <InputNumber min={0} max={16} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col xs={24} md={8}>
                <Form.Item label="Importance 样本数" name="importance_sample_limit">
                  <InputNumber min={0} max={512} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col xs={24} md={8}>
                <Form.Item label="模型名称（可选）" name="name">
                  <Input placeholder="例如：pathology-mvp-v1" />
                </Form.Item>
              </Col>
              <Col xs={24} md={12}>
                <Form.Item label="Survival 时间列（可选）" name="survival_time_column">
                  <Input placeholder="例如：survival_time" />
                </Form.Item>
              </Col>
              <Col xs={24} md={12}>
                <Form.Item label="Survival 事件列（可选）" name="survival_event_column">
                  <Input placeholder="例如：event" />
                </Form.Item>
              </Col>
              <Col span={24}>
                <Form.Item label="描述（可选）" name="description">
                  <Input.TextArea
                    rows={3}
                    placeholder="例如：真实 CLI 训练导入，用于结果复盘与回归对照"
                  />
                </Form.Item>
              </Col>
              <Col span={24}>
                <Form.Item label="标签（可选）" name="tags">
                  <Input placeholder="多个标签用英文逗号分隔，例如：mvp, xiaohongshu, real-run" />
                </Form.Item>
              </Col>
            </Row>
          </Form>

          <Card size="small" title="推荐链路">
            <Space direction="vertical" size={4}>
              <Text code>uv run medfusion validate-config --config &lt;config&gt;</Text>
              <Text code>uv run medfusion train --config &lt;config&gt;</Text>
              <Text code>uv run medfusion import-run --config &lt;config&gt; --checkpoint &lt;path&gt;</Text>
            </Space>
          </Card>
        </Space>
      </Modal>

      <Modal
        title="模型详情"
        open={detailModalOpen}
        onCancel={() => setDetailModalOpen(false)}
        width={1100}
        rootClassName="surface-modal"
        footer={[
          resultHandoff?.jobId && handoffMatchesSelectedModel ? (
            <Button
              key="back-to-job"
              onClick={() => navigate(buildTrainingMonitorLink(resultHandoff.jobId!))}
            >
              返回训练看板
            </Button>
          ) : null,
          <Button key="close" onClick={() => setDetailModalOpen(false)}>
            关闭
          </Button>,
          <Button
            key="evaluate"
            icon={<DotChartOutlined />}
            onClick={() => {
              if (selectedModel) {
                handleOpenEvaluation(selectedModel);
                setDetailModalOpen(false);
              }
            }}
          >
            独立评估
          </Button>,
          <Button
            key="download"
            type="primary"
            icon={<DownloadOutlined />}
            onClick={() => {
              if (selectedModel) handleDownload(selectedModel);
              setDetailModalOpen(false);
            }}
          >
            下载
          </Button>,
        ]}
      >
        {selectedModel && (
          <Space direction="vertical" size={16} style={{ width: "100%" }}>
            {handoffMatchesSelectedModel && resultHandoff ? (
              <Alert
                type="success"
                showIcon
                message="这是从训练完成态直接打开的结果详情"
                description={
                  resultHandoff.jobName
                    ? `来源任务：${resultHandoff.jobName}。现在可以直接复盘结果、下载产物，或返回训练看板继续查看运行记录。`
                    : "这个详情页来自训练完成后的直接 handoff。现在可以直接复盘结果、下载产物，或返回训练看板继续查看运行记录。"
                }
              />
            ) : null}

            <Alert
              type="success"
              showIcon
              message="结果详情已强化"
              description="当前详情页会同时展示多模态指标、ROC/AUC、混淆矩阵、survival 分析、SHAP-style 变量重要性、注意力热力图和结果文件。"
            />

            <Descriptions bordered column={2}>
              <Descriptions.Item label="模型名称" span={2}>
                {selectedModel.name}
              </Descriptions.Item>
              <Descriptions.Item label="骨干网络">
                {selectedModel.backbone}
              </Descriptions.Item>
              <Descriptions.Item label="格式">
                {selectedModel.displayFormat.toUpperCase()}
              </Descriptions.Item>
              <Descriptions.Item label="数据集">
                {selectedModel.dataset_name || "-"}
              </Descriptions.Item>
              <Descriptions.Item label="类别数">
                {selectedModel.numClasses}
              </Descriptions.Item>
              <Descriptions.Item label="参数量">
                {formatParams(selectedModel.params)}
              </Descriptions.Item>
              <Descriptions.Item label="准确率">
                {selectedModel.accuracy
                  ? `${(selectedModel.accuracy * 100).toFixed(2)}%`
                  : "N/A"}
              </Descriptions.Item>
              <Descriptions.Item label="Loss">
                {selectedModel.loss?.toFixed(4) || "-"}
              </Descriptions.Item>
              <Descriptions.Item label="文件大小">
                {formatSize(selectedModel.size)}
              </Descriptions.Item>
              <Descriptions.Item label="创建时间" span={2}>
                {selectedModel.createdAt
                  ? new Date(selectedModel.createdAt).toLocaleString("zh-CN")
                  : "-"}
              </Descriptions.Item>
              <Descriptions.Item label="描述" span={2}>
                {selectedModel.descriptionText || "训练主链自动沉淀的模型记录"}
              </Descriptions.Item>
            </Descriptions>

            <Divider style={{ margin: 0 }} />
            <ModelResultPanel model={selectedModel} />
          </Space>
        )}
      </Modal>
    </PageScaffold>
  );
}
