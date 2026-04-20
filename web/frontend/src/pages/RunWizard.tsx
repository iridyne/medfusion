import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  Alert,
  Button,
  Card,
  Col,
  Form,
  Input,
  InputNumber,
  Row,
  Select,
  Space,
  Steps,
  Switch,
  Tag,
  Typography,
  message,
} from "antd";
import {
  CheckCircleOutlined,
  CopyOutlined,
  DownloadOutlined,
  ExperimentOutlined,
  LinkOutlined,
  ReloadOutlined,
  WarningOutlined,
} from "@ant-design/icons";

import {
  ATTENTION_TYPE_OPTIONS,
  AUGMENTATION_OPTIONS,
  buildResultsCommand,
  buildTrainCommand,
  buildYamlFromRunSpec,
  createRunSpecPreset,
  DEVICE_OPTIONS,
  FUSION_TYPE_OPTIONS,
  inferOutputDir,
  OPTIMIZER_OPTIONS,
  RUN_PRESET_OPTIONS,
  SCHEDULER_OPTIONS,
  type RunPresetId,
  type RunSpec,
  validateRunSpec,
  VISION_BACKBONE_OPTIONS,
} from "@/utils/runSpec";
import { buildTrainingPrefillQuery } from "@/utils/trainingPrefill";
import PageScaffold from "@/components/layout/PageScaffold";

const { Paragraph, Text, Title } = Typography;

function toConfigFileName(experimentName: string): string {
  const normalized = experimentName
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

  return `${normalized || "generated-run"}.yaml`;
}

type BuilderPathId =
  | "first-success"
  | "clinical-baseline"
  | "result-handoff";

interface BuilderPathOption {
  id: BuilderPathId;
  label: string;
  problem: string;
  runtimeShape: string;
  editorMode: string;
  preset: RunPresetId;
}

const BUILDER_PATH_OPTIONS: BuilderPathOption[] = [
  {
    id: "first-success",
    label: "第一次先跑通一条真实主链",
    problem: "我想先完成公开数据、训练和结果回流的一次成功闭环。",
    runtimeShape: "image + tabular / classification / quickstart",
    editorMode: "默认模式：向导 + 参数编辑",
    preset: "quickstart",
  },
  {
    id: "clinical-baseline",
    label: "我要搭一个更稳的研究基线",
    problem: "我已经知道任务方向，希望从正式版前台拿到一套更稳的分类基线骨架。",
    runtimeShape: "image + tabular / classification / clinical baseline",
    editorMode: "默认模式：骨架推荐 -> 参数细化",
    preset: "clinical",
  },
  {
    id: "result-handoff",
    label: "我要做结果审查与对外交付",
    problem: "我更关心结果 artifact、可视化和导入结果后台后的展示质量。",
    runtimeShape: "image + tabular / classification / result audit",
    editorMode: "默认模式：结果导向 preset + artifact 强化",
    preset: "showcase",
  },
];

export default function RunWizard() {
  const navigate = useNavigate();
  const location = useLocation();
  const [currentStep, setCurrentStep] = useState(0);
  const [builderPath, setBuilderPath] = useState<BuilderPathId>("first-success");
  const [preset, setPreset] = useState<RunPresetId>("quickstart");
  const [spec, setSpec] = useState<RunSpec>(() => createRunSpecPreset("quickstart"));
  const [compiledImportSource, setCompiledImportSource] = useState<string | null>(
    null,
  );

  const issues = useMemo(() => validateRunSpec(spec), [spec]);
  const yamlPreview = useMemo(() => buildYamlFromRunSpec(spec), [spec]);
  const configFileName = useMemo(() => toConfigFileName(spec.experimentName), [spec.experimentName]);
  const configPath = `./${configFileName}`;
  const trainCommand = useMemo(() => buildTrainCommand(configPath), [configPath]);
  const resultsCommand = useMemo(() => buildResultsCommand(configPath), [configPath]);
  const errorCount = issues.filter((item) => item.level === "error").length;
  const warningCount = issues.filter((item) => item.level === "warning").length;
  const recommendedFusionDim = spec.model.vision.featureDim + spec.model.tabular.outputDim;
  const readyToRun = errorCount === 0;

  const updateSpec = (patch: Partial<RunSpec>) => {
    setSpec((prev) => ({ ...prev, ...patch }));
  };

  const updateData = (patch: Partial<RunSpec["data"]>) => {
    setSpec((prev) => ({ ...prev, data: { ...prev.data, ...patch } }));
  };

  const updateModel = (patch: Partial<RunSpec["model"]>) => {
    setSpec((prev) => ({ ...prev, model: { ...prev.model, ...patch } }));
  };

  const updateVision = (patch: Partial<RunSpec["model"]["vision"]>) => {
    setSpec((prev) => ({
      ...prev,
      model: {
        ...prev.model,
        vision: { ...prev.model.vision, ...patch },
      },
    }));
  };

  const updateTabular = (patch: Partial<RunSpec["model"]["tabular"]>) => {
    setSpec((prev) => ({
      ...prev,
      model: {
        ...prev.model,
        tabular: { ...prev.model.tabular, ...patch },
      },
    }));
  };

  const updateFusion = (patch: Partial<RunSpec["model"]["fusion"]>) => {
    setSpec((prev) => ({
      ...prev,
      model: {
        ...prev.model,
        fusion: { ...prev.model.fusion, ...patch },
      },
    }));
  };

  const updateTraining = (patch: Partial<RunSpec["training"]>) => {
    setSpec((prev) => ({ ...prev, training: { ...prev.training, ...patch } }));
  };

  const updateOptimizer = (patch: Partial<RunSpec["training"]["optimizer"]>) => {
    setSpec((prev) => ({
      ...prev,
      training: {
        ...prev.training,
        optimizer: { ...prev.training.optimizer, ...patch },
      },
    }));
  };

  const updateScheduler = (patch: Partial<RunSpec["training"]["scheduler"]>) => {
    setSpec((prev) => ({
      ...prev,
      training: {
        ...prev.training,
        scheduler: { ...prev.training.scheduler, ...patch },
      },
    }));
  };

  const updateLogging = (patch: Partial<RunSpec["logging"]>) => {
    setSpec((prev) => ({ ...prev, logging: { ...prev.logging, ...patch } }));
  };

  const applyPreset = (nextPreset: RunPresetId) => {
    setPreset(nextPreset);
    setSpec(createRunSpecPreset(nextPreset));
    setCurrentStep(0);
  };

  const applyBuilderPath = (pathId: BuilderPathId) => {
    const option = BUILDER_PATH_OPTIONS.find((item) => item.id === pathId);
    if (!option) {
      return;
    }
    setBuilderPath(pathId);
    applyPreset(option.preset);
  };

  const syncOutputDir = () => {
    updateLogging({ outputDir: inferOutputDir(spec.projectName, spec.experimentName) });
    message.success("已按项目名和实验名更新输出目录");
  };

  const copyText = async (text: string, successMessage: string) => {
    try {
      await navigator.clipboard.writeText(text);
      message.success(successMessage);
    } catch (error) {
      console.error("Clipboard write failed:", error);
      message.error("复制失败，请检查浏览器权限");
    }
  };

  const downloadYaml = () => {
    const blob = new Blob([yamlPreview], { type: "text/yaml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = configFileName;
    link.click();
    URL.revokeObjectURL(url);
    message.success("配置文件已下载");
  };

  const stepItems = [
    { title: "问题定义", description: "先选问题路径和推荐骨架" },
    { title: "数据与输入", description: "CSV、图像列与表格特征" },
    { title: "模型骨架", description: "backbone、fusion、attention" },
    { title: "训练策略", description: "epoch、optimizer、scheduler" },
    { title: "导出与执行", description: "YAML 与 CLI 命令" },
  ];
  const selectedPresetLabel =
    RUN_PRESET_OPTIONS.find((item) => item.id === preset)?.label ?? preset;
  const selectedBuilderPath =
    BUILDER_PATH_OPTIONS.find((item) => item.id === builderPath) ??
    BUILDER_PATH_OPTIONS[0];
  const trainingPrefillQuery = useMemo(() => {
    return buildTrainingPrefillQuery({
      experimentName: spec.experimentName,
      backbone: spec.model.vision.backbone,
      numClasses: spec.model.numClasses,
      epochs: spec.training.numEpochs,
      batchSize: spec.data.batchSize,
      learningRate: spec.training.optimizer.learningRate,
    });
  }, [
    spec.experimentName,
    spec.model.vision.backbone,
    spec.model.numClasses,
    spec.training.numEpochs,
    spec.data.batchSize,
    spec.training.optimizer.learningRate,
  ]);

  useEffect(() => {
    const state = location.state as
      | {
          compiledRunSpec?: RunSpec;
          compiledPreset?: RunPresetId;
          source?: string;
        }
      | undefined;

    if (!state?.compiledRunSpec) {
      return;
    }

    setSpec(state.compiledRunSpec);
    setPreset(state.compiledPreset || "quickstart");
    setCurrentStep(0);
    setCompiledImportSource(state.source || "advanced-builder");
  }, [location.state]);

  const renderBasicsStep = () => (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Alert
        type="info"
        showIcon
        message="当前正式版默认开放的是问题向导 + 参数编辑层"
        description="这一步先从问题定义进入，再把你映射到当前 runtime 真正支持的模型骨架。节点式编辑仍然是高级模式，不直接替代这条默认路径。"
      />

      <Card size="small" title="先说你现在要解决什么问题">
        <Row gutter={[12, 12]}>
          {BUILDER_PATH_OPTIONS.map((item) => (
            <Col xs={24} md={8} key={item.id}>
              <Card
                size="small"
                hoverable
                onClick={() => applyBuilderPath(item.id)}
                style={{
                  borderColor: builderPath === item.id ? "#1677ff" : undefined,
                  boxShadow:
                    builderPath === item.id
                      ? "0 0 0 2px rgba(22,119,255,0.12)"
                      : undefined,
                  cursor: "pointer",
                }}
              >
                <Space direction="vertical" size={6} style={{ width: "100%" }}>
                  <Space>
                    <Text strong>{item.label}</Text>
                    {builderPath === item.id ? (
                      <Tag color="processing">当前</Tag>
                    ) : null}
                  </Space>
                  <Text type="secondary">{item.problem}</Text>
                  <Text>当前会映射到：{item.runtimeShape}</Text>
                  <Text type="secondary">{item.editorMode}</Text>
                </Space>
              </Card>
            </Col>
          ))}
        </Row>
      </Card>

      <Card size="small" title="当前阶段的前台边界">
        <div className="editorial-grid">
          <div className="surface-note surface-note--dense">
            <strong>默认模式</strong>
            <p>先通过问题向导得到推荐骨架，再在参数级编辑层里调整字段和导出 YAML。</p>
          </div>
          <div className="surface-note surface-note--dense">
            <strong>高级模式</strong>
            <p>节点式编辑当前仍然保留为高级结构编辑面，不对外承诺可任意组合出 runtime 尚未支持的新能力。</p>
          </div>
        </div>
      </Card>

      <Card size="small" title="实验元信息">
        <Form layout="vertical">
          <Row gutter={16}>
            <Col xs={24} md={12}>
              <Form.Item label="project_name">
                <Input
                  value={spec.projectName}
                  onChange={(event) => updateSpec({ projectName: event.target.value })}
                  placeholder="例如：medfusion-research"
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={12}>
              <Form.Item label="experiment_name">
                <Input
                  value={spec.experimentName}
                  onChange={(event) => updateSpec({ experimentName: event.target.value })}
                  placeholder="例如：baseline-run"
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={12}>
              <Form.Item label="device">
                <Select
                  value={spec.device}
                  options={DEVICE_OPTIONS.map((value) => ({ label: value, value }))}
                  onChange={(value) => updateSpec({ device: value })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={12}>
              <Form.Item label="seed">
                <InputNumber
                  style={{ width: "100%" }}
                  value={spec.seed}
                  onChange={(value) => updateSpec({ seed: Number(value ?? 42) })}
                />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item label="description">
            <Input.TextArea
              rows={3}
              value={spec.description}
              onChange={(event) => updateSpec({ description: event.target.value })}
              placeholder="可选，用于结果页和 README 说明"
            />
          </Form.Item>
          <Form.Item label="tags">
            <Select
              mode="tags"
              tokenSeparators={[","]}
              value={spec.tags}
              onChange={(value) => updateSpec({ tags: value })}
              placeholder="输入标签后回车，例如 quickstart, validation"
            />
          </Form.Item>
        </Form>
      </Card>
    </Space>
  );

  const renderDataStep = () => (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Alert
        type="info"
        showIcon
        message="当前训练主链是图像 + 表格双分支"
        description="除了图像路径和标签列，还需要至少一个 numerical_features 或 categorical_features。"
      />

      <Card size="small" title="数据路径与字段">
        <Form layout="vertical">
          <Row gutter={16}>
            <Col xs={24} md={12}>
              <Form.Item label="csv_path">
                <Input
                  value={spec.data.csvPath}
                  onChange={(event) => updateData({ csvPath: event.target.value })}
                  placeholder="例如：data/mock/metadata.csv"
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={12}>
              <Form.Item label="image_dir">
                <Input
                  value={spec.data.imageDir}
                  onChange={(event) => updateData({ imageDir: event.target.value })}
                  placeholder="例如：data/mock"
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item label="image_path_column">
                <Input
                  value={spec.data.imagePathColumn}
                  onChange={(event) => updateData({ imagePathColumn: event.target.value })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item label="target_column">
                <Input
                  value={spec.data.targetColumn}
                  onChange={(event) => updateData({ targetColumn: event.target.value })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item label="patient_id_column（可选）">
                <Input
                  value={spec.data.patientIdColumn}
                  onChange={(event) => updateData({ patientIdColumn: event.target.value })}
                  placeholder="没有可留空"
                />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col xs={24} md={12}>
              <Form.Item label="numerical_features">
                <Select
                  mode="tags"
                  tokenSeparators={[","]}
                  value={spec.data.numericalFeatures}
                  onChange={(value) => updateData({ numericalFeatures: value })}
                  placeholder="输入后回车，例如 age, bmi"
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={12}>
              <Form.Item label="categorical_features">
                <Select
                  mode="tags"
                  tokenSeparators={[","]}
                  value={spec.data.categoricalFeatures}
                  onChange={(value) => updateData({ categoricalFeatures: value })}
                  placeholder="输入后回车，例如 gender, stage"
                />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Card>

      <Card size="small" title="数据划分与 dataloader">
        <Form layout="vertical">
          <Row gutter={16}>
            <Col xs={24} md={8}>
              <Form.Item label="train_ratio">
                <InputNumber
                  style={{ width: "100%" }}
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  value={spec.data.trainRatio}
                  onChange={(value) => updateData({ trainRatio: Number(value ?? 0.7) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item label="val_ratio">
                <InputNumber
                  style={{ width: "100%" }}
                  min={0.05}
                  max={0.9}
                  step={0.05}
                  value={spec.data.valRatio}
                  onChange={(value) => updateData({ valRatio: Number(value ?? 0.15) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item label="test_ratio">
                <InputNumber
                  style={{ width: "100%" }}
                  min={0.05}
                  max={0.9}
                  step={0.05}
                  value={spec.data.testRatio}
                  onChange={(value) => updateData({ testRatio: Number(value ?? 0.15) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="image_size">
                <InputNumber
                  style={{ width: "100%" }}
                  min={64}
                  step={32}
                  value={spec.data.imageSize}
                  onChange={(value) => updateData({ imageSize: Number(value ?? 224) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="batch_size">
                <InputNumber
                  style={{ width: "100%" }}
                  min={1}
                  value={spec.data.batchSize}
                  onChange={(value) => updateData({ batchSize: Number(value ?? 4) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="num_workers">
                <InputNumber
                  style={{ width: "100%" }}
                  min={0}
                  value={spec.data.numWorkers}
                  onChange={(value) => updateData({ numWorkers: Number(value ?? 0) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="augmentation_strength">
                <Select
                  value={spec.data.augmentationStrength}
                  options={AUGMENTATION_OPTIONS.map((value) => ({ label: value, value }))}
                  onChange={(value) => updateData({ augmentationStrength: value })}
                />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item label="pin_memory" valuePropName="checked">
            <Switch
              checked={spec.data.pinMemory}
              onChange={(checked) => updateData({ pinMemory: checked })}
            />
          </Form.Item>
        </Form>
      </Card>
    </Space>
  );

  const renderModelStep = () => (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Card size="small" title="任务与视觉 backbone">
        <Form layout="vertical">
          <Row gutter={16}>
            <Col xs={24} md={8}>
              <Form.Item label="num_classes">
                <InputNumber
                  style={{ width: "100%" }}
                  min={2}
                  value={spec.model.numClasses}
                  onChange={(value) => updateModel({ numClasses: Number(value ?? 2) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item label="vision.backbone">
                <Select
                  showSearch
                  optionFilterProp="label"
                  value={spec.model.vision.backbone}
                  options={VISION_BACKBONE_OPTIONS.map((value) => ({ label: value, value }))}
                  onChange={(value) => updateVision({ backbone: value })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item label="vision.attention_type">
                <Select
                  value={spec.model.vision.attentionType}
                  options={ATTENTION_TYPE_OPTIONS.map((value) => ({ label: value, value }))}
                  onChange={(value) => {
                    updateVision({ attentionType: value });
                    if (value !== "cbam" && spec.training.useAttentionSupervision) {
                      updateTraining({ useAttentionSupervision: false });
                    }
                  }}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="vision.feature_dim">
                <InputNumber
                  style={{ width: "100%" }}
                  min={16}
                  step={16}
                  value={spec.model.vision.featureDim}
                  onChange={(value) => updateVision({ featureDim: Number(value ?? 128) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="vision.dropout">
                <InputNumber
                  style={{ width: "100%" }}
                  min={0}
                  max={0.9}
                  step={0.05}
                  value={spec.model.vision.dropout}
                  onChange={(value) => updateVision({ dropout: Number(value ?? 0.3) })}
                />
              </Form.Item>
            </Col>
            <Col xs={12} md={6}>
              <Form.Item label="pretrained" valuePropName="checked">
                <Switch
                  checked={spec.model.vision.pretrained}
                  onChange={(checked) => updateVision({ pretrained: checked })}
                />
              </Form.Item>
            </Col>
            <Col xs={12} md={6}>
              <Form.Item label="freeze_backbone" valuePropName="checked">
                <Switch
                  checked={spec.model.vision.freezeBackbone}
                  onChange={(checked) => updateVision({ freezeBackbone: checked })}
                />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Card>

      <Card size="small" title="表格分支与融合">
        <Form layout="vertical">
          <Row gutter={16}>
            <Col xs={24} md={12}>
              <Form.Item label="tabular.hidden_dims">
                <Select
                  mode="tags"
                  tokenSeparators={[","]}
                  value={spec.model.tabular.hiddenDims.map(String)}
                  onChange={(value) => {
                    const parsed = value
                      .map((item) => Number(item))
                      .filter((item) => Number.isFinite(item) && item > 0);
                    updateTabular({ hiddenDims: parsed.length > 0 ? parsed : [32] });
                  }}
                  placeholder="例如：64, 32"
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="tabular.output_dim">
                <InputNumber
                  style={{ width: "100%" }}
                  min={4}
                  step={4}
                  value={spec.model.tabular.outputDim}
                  onChange={(value) => updateTabular({ outputDim: Number(value ?? 16) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="tabular.dropout">
                <InputNumber
                  style={{ width: "100%" }}
                  min={0}
                  max={0.9}
                  step={0.05}
                  value={spec.model.tabular.dropout}
                  onChange={(value) => updateTabular({ dropout: Number(value ?? 0.2) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item label="fusion.fusion_type">
                <Select
                  value={spec.model.fusion.fusionType}
                  options={FUSION_TYPE_OPTIONS.map((value) => ({ label: value, value }))}
                  onChange={(value) => updateFusion({ fusionType: value })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item
                label="fusion.hidden_dim"
                extra={`concatenate 常见起点可以直接设成 ${recommendedFusionDim}`}
              >
                <InputNumber
                  style={{ width: "100%" }}
                  min={8}
                  step={8}
                  value={spec.model.fusion.hiddenDim}
                  onChange={(value) => updateFusion({ hiddenDim: Number(value ?? recommendedFusionDim) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item label="fusion.dropout">
                <InputNumber
                  style={{ width: "100%" }}
                  min={0}
                  max={0.9}
                  step={0.05}
                  value={spec.model.fusion.dropout}
                  onChange={(value) => updateFusion({ dropout: Number(value ?? 0.3) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item label="fusion.num_heads">
                <InputNumber
                  style={{ width: "100%" }}
                  min={1}
                  value={spec.model.fusion.numHeads}
                  onChange={(value) => updateFusion({ numHeads: Number(value ?? 4) })}
                />
              </Form.Item>
            </Col>
            <Col xs={12} md={8}>
              <Form.Item label="use_auxiliary_heads" valuePropName="checked">
                <Switch
                  checked={spec.model.useAuxiliaryHeads}
                  onChange={(checked) => updateModel({ useAuxiliaryHeads: checked })}
                />
              </Form.Item>
            </Col>
            <Col xs={12} md={8}>
              <Form.Item
                label="attention supervision"
                extra={spec.model.vision.attentionType !== "cbam" ? "当前仅在 CBAM 下可用" : undefined}
                valuePropName="checked"
              >
                <Switch
                  checked={spec.training.useAttentionSupervision}
                  disabled={spec.model.vision.attentionType !== "cbam"}
                  onChange={(checked) => updateTraining({ useAttentionSupervision: checked })}
                />
              </Form.Item>
            </Col>
            {spec.training.useAttentionSupervision ? (
              <Col xs={24} md={8}>
                <Form.Item label="attention_loss_weight">
                  <InputNumber
                    style={{ width: "100%" }}
                    min={0.01}
                    max={1}
                    step={0.01}
                    value={spec.training.attentionLossWeight}
                    onChange={(value) => updateTraining({ attentionLossWeight: Number(value ?? 0.1) })}
                  />
                </Form.Item>
              </Col>
            ) : null}
          </Row>
        </Form>
      </Card>
    </Space>
  );

  const renderTrainingStep = () => (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Card size="small" title="训练节奏与策略">
        <Form layout="vertical">
          <Row gutter={16}>
            <Col xs={24} md={6}>
              <Form.Item label="num_epochs">
                <InputNumber
                  style={{ width: "100%" }}
                  min={1}
                  value={spec.training.numEpochs}
                  onChange={(value) => updateTraining({ numEpochs: Number(value ?? 3) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="gradient_clip">
                <InputNumber
                  style={{ width: "100%" }}
                  min={0}
                  step={0.1}
                  value={spec.training.gradientClip ?? 0}
                  onChange={(value) => updateTraining({ gradientClip: value === null ? null : Number(value) })}
                />
              </Form.Item>
            </Col>
            <Col xs={12} md={6}>
              <Form.Item label="mixed_precision" valuePropName="checked">
                <Switch
                  checked={spec.training.mixedPrecision}
                  onChange={(checked) => updateTraining({ mixedPrecision: checked })}
                />
              </Form.Item>
            </Col>
            <Col xs={12} md={6}>
              <Form.Item label="use_progressive_training" valuePropName="checked">
                <Switch
                  checked={spec.training.useProgressiveTraining}
                  onChange={(checked) => updateTraining({ useProgressiveTraining: checked })}
                />
              </Form.Item>
            </Col>
          </Row>

          {spec.training.useProgressiveTraining ? (
            <Row gutter={16}>
              <Col xs={24} md={8}>
                <Form.Item label="stage1_epochs">
                  <InputNumber
                    style={{ width: "100%" }}
                    min={1}
                    value={spec.training.stage1Epochs}
                    onChange={(value) => updateTraining({ stage1Epochs: Number(value ?? 1) })}
                  />
                </Form.Item>
              </Col>
              <Col xs={24} md={8}>
                <Form.Item label="stage2_epochs">
                  <InputNumber
                    style={{ width: "100%" }}
                    min={1}
                    value={spec.training.stage2Epochs}
                    onChange={(value) => updateTraining({ stage2Epochs: Number(value ?? 1) })}
                  />
                </Form.Item>
              </Col>
              <Col xs={24} md={8}>
                <Form.Item label="stage3_epochs">
                  <InputNumber
                    style={{ width: "100%" }}
                    min={1}
                    value={spec.training.stage3Epochs}
                    onChange={(value) => updateTraining({ stage3Epochs: Number(value ?? 1) })}
                  />
                </Form.Item>
              </Col>
            </Row>
          ) : null}
        </Form>
      </Card>

      <Card size="small" title="优化器与 scheduler">
        <Form layout="vertical">
          <Row gutter={16}>
            <Col xs={24} md={6}>
              <Form.Item label="optimizer.optimizer">
                <Select
                  value={spec.training.optimizer.optimizer}
                  options={OPTIMIZER_OPTIONS.map((value) => ({ label: value, value }))}
                  onChange={(value) => updateOptimizer({ optimizer: value })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="optimizer.learning_rate">
                <InputNumber
                  style={{ width: "100%" }}
                  min={1e-6}
                  step={1e-4}
                  value={spec.training.optimizer.learningRate}
                  onChange={(value) => updateOptimizer({ learningRate: Number(value ?? 1e-3) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="optimizer.weight_decay">
                <InputNumber
                  style={{ width: "100%" }}
                  min={0}
                  step={1e-4}
                  value={spec.training.optimizer.weightDecay}
                  onChange={(value) => updateOptimizer({ weightDecay: Number(value ?? 0) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="optimizer.momentum">
                <InputNumber
                  style={{ width: "100%" }}
                  min={0}
                  max={0.999}
                  step={0.05}
                  disabled={spec.training.optimizer.optimizer !== "sgd"}
                  value={spec.training.optimizer.momentum}
                  onChange={(value) => updateOptimizer({ momentum: Number(value ?? 0.9) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="scheduler.scheduler">
                <Select
                  value={spec.training.scheduler.scheduler}
                  options={SCHEDULER_OPTIONS.map((value) => ({ label: value, value }))}
                  onChange={(value) => updateScheduler({ scheduler: value })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="scheduler.min_lr">
                <InputNumber
                  style={{ width: "100%" }}
                  min={0}
                  step={1e-6}
                  value={spec.training.scheduler.minLr}
                  onChange={(value) => updateScheduler({ minLr: Number(value ?? 1e-6) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="scheduler.step_size">
                <InputNumber
                  style={{ width: "100%" }}
                  min={1}
                  disabled={spec.training.scheduler.scheduler !== "step"}
                  value={spec.training.scheduler.stepSize}
                  onChange={(value) => updateScheduler({ stepSize: Number(value ?? 1) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="scheduler.gamma">
                <InputNumber
                  style={{ width: "100%" }}
                  min={0.01}
                  max={0.99}
                  step={0.05}
                  disabled={spec.training.scheduler.scheduler !== "step"}
                  value={spec.training.scheduler.gamma}
                  onChange={(value) => updateScheduler({ gamma: Number(value ?? 0.1) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="scheduler.patience">
                <InputNumber
                  style={{ width: "100%" }}
                  min={1}
                  disabled={spec.training.scheduler.scheduler !== "plateau"}
                  value={spec.training.scheduler.patience}
                  onChange={(value) => updateScheduler({ patience: Number(value ?? 5) })}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={6}>
              <Form.Item label="scheduler.factor">
                <InputNumber
                  style={{ width: "100%" }}
                  min={0.05}
                  max={0.9}
                  step={0.05}
                  disabled={spec.training.scheduler.scheduler !== "plateau"}
                  value={spec.training.scheduler.factor}
                  onChange={(value) => updateScheduler({ factor: Number(value ?? 0.5) })}
                />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Card>

      <Card size="small" title="输出与追踪">
        <Form layout="vertical">
          <Form.Item label="logging.output_dir">
            <Input
              value={spec.logging.outputDir}
              onChange={(event) => updateLogging({ outputDir: event.target.value })}
              addonAfter={
                <Button type="link" size="small" onClick={syncOutputDir} icon={<ReloadOutlined />}>
                  同步
                </Button>
              }
            />
          </Form.Item>
          <Row gutter={16}>
            <Col xs={12} md={8}>
              <Form.Item label="use_tensorboard" valuePropName="checked">
                <Switch
                  checked={spec.logging.useTensorboard}
                  onChange={(checked) => updateLogging({ useTensorboard: checked })}
                />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Card>
    </Space>
  );

  const renderPreviewStep = () => (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Alert
        type={readyToRun ? "success" : "warning"}
        showIcon
        message={readyToRun ? "配置已达到可运行状态" : "配置仍有待修正项"}
        description={
          readyToRun
            ? "这份 YAML 已经对齐真实 CLI schema，可以直接下载后执行 medfusion validate-config / train。"
            : "请先处理右侧就绪检查里的 error，再导出配置。warning 不阻塞运行，但建议看一遍。"
        }
      />

      <Card
        size="small"
        title="导出动作"
        extra={<Tag color={readyToRun ? "success" : "warning"}>{readyToRun ? "Ready" : "Needs Fix"}</Tag>}
      >
        <Space wrap>
          <Button icon={<CopyOutlined />} onClick={() => void copyText(yamlPreview, "YAML 已复制")}>复制 YAML</Button>
          <Button icon={<DownloadOutlined />} onClick={downloadYaml}>下载 YAML</Button>
          <Button icon={<CopyOutlined />} onClick={() => void copyText(trainCommand, "训练命令已复制")}>复制训练命令</Button>
          <Button
            icon={<ExperimentOutlined />}
            onClick={() => navigate(`/training?${trainingPrefillQuery}`)}
          >
            打开训练监控
          </Button>
        </Space>
      </Card>

      <Card size="small" title="YAML 预览">
        <pre
          style={{
            margin: 0,
            padding: 16,
            borderRadius: 12,
            background: "#0f172a",
            color: "#e2e8f0",
            overflowX: "auto",
            fontSize: 13,
            lineHeight: 1.6,
          }}
        >
          {yamlPreview}
        </pre>
      </Card>
    </Space>
  );

  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        return renderBasicsStep();
      case 1:
        return renderDataStep();
      case 2:
        return renderModelStep();
      case 3:
        return renderTrainingStep();
      case 4:
        return renderPreviewStep();
      default:
        return null;
    }
  };

  return (
    <PageScaffold
      eyebrow="Problem-first builder"
      title="先定义问题，再生成可运行的模型骨架"
      description="Run Wizard 现在承担正式版默认的模型搭建入口第一阶段：先把用户问题收敛成推荐骨架，再落到当前 runtime 已支持的参数编辑层、YAML 导出和真实训练链。"
      chips={[
        { label: "Problem-first", tone: "amber" },
        { label: "Runtime-backed", tone: "teal" },
        { label: "Formal release slice", tone: "blue" },
      ]}
      actions={
        <>
          <Button icon={<DownloadOutlined />} onClick={downloadYaml}>
            下载当前 YAML
          </Button>
          <Button onClick={() => navigate("/config/advanced")}>
            打开高级模式
          </Button>
          <Button icon={<LinkOutlined />} onClick={() => navigate("/config/comfyui")}>
            打开 ComfyUI 入口
          </Button>
          <Button
            icon={<CopyOutlined />}
            onClick={() => void copyText(trainCommand, "训练命令已复制")}
          >
            复制训练命令
          </Button>
          <Button icon={<ReloadOutlined />} onClick={syncOutputDir}>
            同步输出目录
          </Button>
        </>
      }
      aside={
        <div className="hero-aside-panel">
          <span className="hero-aside-panel__label">Current recommendation</span>
          <div className="hero-aside-panel__value">{selectedBuilderPath.label}</div>
          <div className="hero-aside-panel__copy">
            当前问题路径会映射到 <strong>{selectedPresetLabel}</strong> preset，
            并继续通过当前 runtime 支持的配置空间导出真实 YAML。
          </div>
          <div className="surface-note">
            当前支持：<strong>{selectedBuilderPath.runtimeShape}</strong>
          </div>
        </div>
      }
      metrics={[
        {
          label: "Problem path",
          value: selectedPresetLabel,
          hint: selectedBuilderPath.label,
          tone: "amber",
        },
        {
          label: "Wizard step",
          value: `${currentStep + 1}/${stepItems.length}`,
          hint: stepItems[currentStep]?.title,
          tone: "blue",
        },
        {
          label: "Edit mode",
          value: "guided by default",
          hint: "Node editing remains an advanced mode",
          tone: "teal",
        },
        {
          label: "Blocking errors",
          value: errorCount.toLocaleString(),
          hint: readyToRun
            ? `Warnings: ${warningCount}`
            : "Must be zero before running",
          tone: errorCount > 0 ? "rose" : "teal",
        },
      ]}
    >
      <Alert
        type="info"
        showIcon
        message="这一步先解决的是“你到底在搭什么模型”"
        description="CLI 仍然保留作为执行和自动化层，但普通用户不应该再从手写 YAML 和 schema 字段开始。Run Wizard 需要先解释问题路径、模板边界和推荐骨架，再把配置导回真实主链。"
      />
      {compiledImportSource ? (
        <Alert
          type="success"
          showIcon
          style={{ marginTop: 16 }}
          message="已从高级模式导入一份可编辑配置草案"
          description={`当前这份 RunSpec 来自 ${compiledImportSource} 的图编译结果。你现在可以继续在默认模式里微调字段，再导出 YAML 或进入训练。`}
        />
      ) : null}

      <div className="split-grid">
        <Card className="surface-card">
          <Space direction="vertical" size={20} style={{ width: "100%" }}>
            <Steps current={currentStep} items={stepItems} responsive />
            {renderStepContent()}
            <Space>
              <Button
                disabled={currentStep === 0}
                onClick={() => setCurrentStep((prev) => prev - 1)}
              >
                上一步
              </Button>
              <Button
                type="primary"
                disabled={currentStep === stepItems.length - 1}
                onClick={() => setCurrentStep((prev) => prev + 1)}
              >
                下一步
              </Button>
              {currentStep === stepItems.length - 1 ? (
                <Button icon={<DownloadOutlined />} onClick={downloadYaml}>
                  直接下载
                </Button>
              ) : null}
            </Space>
          </Space>
        </Card>

        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <Card
            className="surface-card"
            title="运行摘要"
            extra={
              readyToRun ? (
                <CheckCircleOutlined style={{ color: "var(--accent-teal)" }} />
              ) : (
                <WarningOutlined style={{ color: "var(--accent-amber)" }} />
              )
            }
          >
            <Space direction="vertical" size={8} style={{ width: "100%" }}>
              <Text strong>{spec.projectName}</Text>
              <Text type="secondary">实验名：{spec.experimentName}</Text>
              <Text type="secondary">backbone：{spec.model.vision.backbone}</Text>
              <Text type="secondary">fusion：{spec.model.fusion.fusionType}</Text>
              <Text type="secondary">epoch：{spec.training.numEpochs}</Text>
              <Text type="secondary">batch size：{spec.data.batchSize}</Text>
              <Text type="secondary">device：{spec.device}</Text>
            </Space>
          </Card>

          <Card className="surface-card" title="就绪检查">
            <Space direction="vertical" size={10} style={{ width: "100%" }}>
              {issues.length === 0 ? (
                <Alert type="success" showIcon message="没有发现阻塞项" />
              ) : (
                issues.map((issue, index) => (
                  <Alert
                    key={`${issue.level}-${index}`}
                    type={issue.level === "error" ? "error" : "warning"}
                    showIcon
                    message={issue.message}
                  />
                ))
              )}
            </Space>
          </Card>

          <Card className="surface-card" title="CLI 命令">
            <Space direction="vertical" size={12} style={{ width: "100%" }}>
              <div>
                <Text strong>训练</Text>
                <pre className="command-block">{trainCommand}</pre>
              </div>
              <div>
                <Text strong>结果构建</Text>
                <pre className="command-block">{resultsCommand}</pre>
              </div>
              <Text type="secondary">
                当前建议优先下载 YAML 并通过 CLI 执行，以保证训练和结果链路可复现。Web 训练入口适合本地快速验证。
              </Text>
              <Button onClick={() => navigate("/config/advanced")}>
                查看高级模式的组件注册表
              </Button>
              <Button onClick={() => navigate("/config/comfyui")}>
                打开 ComfyUI 上线入口
              </Button>
            </Space>
          </Card>
        </Space>
      </div>
    </PageScaffold>
  );
}
