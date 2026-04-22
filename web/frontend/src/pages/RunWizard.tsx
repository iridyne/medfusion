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
  ControlOutlined,
  CopyOutlined,
  DownloadOutlined,
  ExperimentOutlined,
  LinkOutlined,
  ReloadOutlined,
  WarningOutlined,
} from "@ant-design/icons";
import {
  getModelCatalog,
  inspectModelConfig,
  type ModelCatalogTemplate,
  type ModelInspectResponse,
} from "@/api/models";

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
  SCHEDULER_OPTIONS,
  type RunPresetId,
  type RunSpec,
  validateRunSpec,
  VISION_BACKBONE_OPTIONS,
} from "@/utils/runSpec";
import {
  buildTrainingPrefillQuery,
  type TrainingLaunchSource,
} from "@/utils/trainingPrefill";
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

interface WizardPrefillState {
  projectName?: string;
  experimentName?: string;
  description?: string;
  outputDir?: string;
  backbone?: string;
  numClasses?: number;
}

interface DatasetPrefillState {
  csvPath?: string;
  imageDir?: string;
  imagePathColumn?: string;
  targetColumn?: string;
  patientIdColumn?: string;
  numericalFeatures?: string[];
  categoricalFeatures?: string[];
  numClasses?: number;
}

interface ModelPrefillState {
  modelTemplateId?: string;
  customModelLabel?: string;
  numClasses?: number;
  useAuxiliaryHeads?: boolean;
  backbone?: string;
  attentionType?: string;
  featureDim?: number;
  pretrained?: boolean;
  freezeBackbone?: boolean;
  tabularHiddenDims?: number[];
  tabularOutputDim?: number;
  tabularDropout?: number;
  fusionType?: string;
  fusionHiddenDim?: number;
  fusionDropout?: number;
  fusionNumHeads?: number;
  useAttentionSupervision?: boolean;
}

const DEFAULT_MODEL_TEMPLATE_ID = "quickstart_multimodal";

function resolveTemplatePreset(
  templateId: string,
  templates: ModelCatalogTemplate[],
): RunPresetId {
  const fromCatalog = templates.find((item) => item.id === templateId)
    ?.advanced_builder_contract?.recommended_preset;
  if (
    fromCatalog === "quickstart" ||
    fromCatalog === "clinical" ||
    fromCatalog === "showcase"
  ) {
    return fromCatalog;
  }

  if (templateId === "clinical_gated_baseline") {
    return "clinical";
  }
  if (templateId === "attention_audit_path") {
    return "showcase";
  }
  return "quickstart";
}

function applyModelPrefillToSpec(prev: RunSpec, modelPrefill: ModelPrefillState): RunSpec {
  const nextBackbone = modelPrefill.backbone;
  const supportedBackbone =
    typeof nextBackbone === "string" &&
    VISION_BACKBONE_OPTIONS.includes(nextBackbone as (typeof VISION_BACKBONE_OPTIONS)[number])
      ? (nextBackbone as (typeof VISION_BACKBONE_OPTIONS)[number])
      : null;
  const nextAttention = modelPrefill.attentionType;
  const supportedAttention =
    typeof nextAttention === "string" &&
    ATTENTION_TYPE_OPTIONS.includes(nextAttention as (typeof ATTENTION_TYPE_OPTIONS)[number])
      ? (nextAttention as (typeof ATTENTION_TYPE_OPTIONS)[number])
      : null;
  const nextFusion = modelPrefill.fusionType;
  const supportedFusion =
    typeof nextFusion === "string" &&
    FUSION_TYPE_OPTIONS.includes(nextFusion as (typeof FUSION_TYPE_OPTIONS)[number])
      ? (nextFusion as (typeof FUSION_TYPE_OPTIONS)[number])
      : null;

  return {
    ...prev,
    model: {
      ...prev.model,
      numClasses:
        typeof modelPrefill?.numClasses === "number" && modelPrefill.numClasses > 0
          ? modelPrefill.numClasses
          : prev.model.numClasses,
      useAuxiliaryHeads:
        typeof modelPrefill?.useAuxiliaryHeads === "boolean"
          ? modelPrefill.useAuxiliaryHeads
          : prev.model.useAuxiliaryHeads,
      vision: {
        ...prev.model.vision,
        backbone: supportedBackbone || prev.model.vision.backbone,
        attentionType: supportedAttention || prev.model.vision.attentionType,
        featureDim:
          typeof modelPrefill?.featureDim === "number" && modelPrefill.featureDim > 0
            ? modelPrefill.featureDim
            : prev.model.vision.featureDim,
        pretrained:
          typeof modelPrefill?.pretrained === "boolean"
            ? modelPrefill.pretrained
            : prev.model.vision.pretrained,
        freezeBackbone:
          typeof modelPrefill?.freezeBackbone === "boolean"
            ? modelPrefill.freezeBackbone
            : prev.model.vision.freezeBackbone,
      },
      tabular: {
        ...prev.model.tabular,
        hiddenDims:
          modelPrefill?.tabularHiddenDims?.length
            ? modelPrefill.tabularHiddenDims
            : prev.model.tabular.hiddenDims,
        outputDim:
          typeof modelPrefill?.tabularOutputDim === "number" &&
          modelPrefill.tabularOutputDim > 0
            ? modelPrefill.tabularOutputDim
            : prev.model.tabular.outputDim,
        dropout:
          typeof modelPrefill?.tabularDropout === "number"
            ? modelPrefill.tabularDropout
            : prev.model.tabular.dropout,
      },
      fusion: {
        ...prev.model.fusion,
        fusionType: supportedFusion || prev.model.fusion.fusionType,
        hiddenDim:
          typeof modelPrefill?.fusionHiddenDim === "number" &&
          modelPrefill.fusionHiddenDim > 0
            ? modelPrefill.fusionHiddenDim
            : prev.model.fusion.hiddenDim,
        dropout:
          typeof modelPrefill?.fusionDropout === "number"
            ? modelPrefill.fusionDropout
            : prev.model.fusion.dropout,
        numHeads:
          typeof modelPrefill?.fusionNumHeads === "number" &&
          modelPrefill.fusionNumHeads > 0
            ? modelPrefill.fusionNumHeads
            : prev.model.fusion.numHeads,
      },
    },
    training: {
      ...prev.training,
      useAttentionSupervision:
        typeof modelPrefill?.useAttentionSupervision === "boolean"
          ? modelPrefill.useAttentionSupervision
          : prev.training.useAttentionSupervision,
    },
  };
}

export default function RunWizard() {
  const navigate = useNavigate();
  const location = useLocation();
  const [currentStep, setCurrentStep] = useState(0);
  const [preset, setPreset] = useState<RunPresetId>("quickstart");
  const [selectedModelTemplateId, setSelectedModelTemplateId] = useState<string>(
    DEFAULT_MODEL_TEMPLATE_ID,
  );
  const [officialModelTemplates, setOfficialModelTemplates] = useState<ModelCatalogTemplate[]>([]);
  const [selectedTemplateDisplayLabel, setSelectedTemplateDisplayLabel] = useState<string | null>(
    null,
  );
  const [spec, setSpec] = useState<RunSpec>(() =>
    createRunSpecPreset("quickstart"),
  );
  const [compiledImportSource, setCompiledImportSource] = useState<string | null>(
    null,
  );
  const [compiledImportMode, setCompiledImportMode] = useState<"compiled" | "prefill">(
    "compiled",
  );
  const [modelInspection, setModelInspection] = useState<ModelInspectResponse | null>(
    null,
  );
  const [modelInspectionLoading, setModelInspectionLoading] = useState(false);
  const [modelInspectionError, setModelInspectionError] = useState<string | null>(
    null,
  );

  useEffect(() => {
    const loadTemplates = async () => {
      try {
        const payload = await getModelCatalog();
        const templates = payload.templates.filter((item) => item.source === "official");
        setOfficialModelTemplates(templates);
      } catch (error) {
        console.error("Failed to load official model templates:", error);
      }
    };

    void loadTemplates();
  }, []);

  useEffect(() => {
    if (!officialModelTemplates.length) {
      return;
    }
    const selectedTemplate = officialModelTemplates.find(
      (item) => item.id === selectedModelTemplateId,
    );
    if (!selectedTemplate?.wizard_prefill) {
      return;
    }
    const nextPreset = resolveTemplatePreset(
      selectedTemplate.id,
      officialModelTemplates,
    );
    setPreset(nextPreset);
    setSelectedTemplateDisplayLabel(selectedTemplate.label);
    setSpec((prev) => applyModelPrefillToSpec(createRunSpecPreset(nextPreset), {
      modelTemplateId: selectedTemplate.id,
      ...(selectedTemplate.wizard_prefill as ModelPrefillState),
    }));
  }, [officialModelTemplates, selectedModelTemplateId]);

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

  const applyModelTemplate = (
    templateId: string,
    modelPrefill?: ModelPrefillState,
    displayLabel?: string | null,
  ) => {
    const nextPreset = resolveTemplatePreset(templateId, officialModelTemplates);
    setPreset(nextPreset);
    setSelectedModelTemplateId(templateId);
    setSelectedTemplateDisplayLabel(displayLabel || null);
    const baseSpec = createRunSpecPreset(nextPreset);
    setSpec(modelPrefill ? applyModelPrefillToSpec(baseSpec, modelPrefill) : baseSpec);
    setCurrentStep(0);
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
  const selectedModelTemplate =
    officialModelTemplates.find((item) => item.id === selectedModelTemplateId) || null;
  const selectedPresetLabel =
    selectedTemplateDisplayLabel ||
    selectedModelTemplate?.label ||
    {
      quickstart: "快速验证",
      clinical: "稳健研究基线",
      showcase: "注意力审查",
    }[preset];
  const selectedTemplateDescription =
    selectedModelTemplate?.description ||
    "当前会通过官方模型模板，把问题路径映射到正式版支持的模型骨架。";
  const selectedTemplateCompute =
    selectedModelTemplate?.compute_profile
      ? `${selectedModelTemplate.compute_profile.gpu_vram_hint} · ${selectedModelTemplate.compute_profile.notes}`
      : "image + tabular / classification / quickstart";
  const selectedTemplateContract = selectedModelTemplate?.advanced_builder_contract || null;
  const selectedTemplateRecommendedPreset = selectedTemplateContract?.recommended_preset || null;
  const selectedTemplateCompileBoundary = selectedTemplateContract?.compile_boundary || null;
  const selectedTemplateCompileNotes = selectedTemplateContract?.compile_notes || [];
  const selectedTemplatePatchHints = selectedTemplateContract?.patch_target_hints || [];
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
  const trainingLaunchSource: Exclude<TrainingLaunchSource, null> = useMemo(() => {
    if (compiledImportMode === "prefill") {
      if (compiledImportSource === "comfyui-bridge") {
        return "comfyui-bridge";
      }
      if (compiledImportSource === "model-library") {
        return "model-library";
      }
      if (compiledImportSource === "training-monitor") {
        return "training-monitor";
      }
    }
    return "run-wizard";
  }, [compiledImportMode, compiledImportSource]);

  useEffect(() => {
    const state = location.state as
      | {
          compiledRunSpec?: RunSpec;
          compiledPreset?: RunPresetId;
          wizardPrefill?: WizardPrefillState;
          datasetPrefill?: DatasetPrefillState;
          modelPrefill?: ModelPrefillState;
          source?: string;
        }
      | undefined;

    if (!state) {
      return;
    }

    if (state.compiledRunSpec) {
      setSpec(state.compiledRunSpec);
      setPreset(state.compiledPreset || "quickstart");
      setCurrentStep(0);
      setCompiledImportMode("compiled");
      setCompiledImportSource(state.source || "advanced-builder");
      return;
    }

    if (state.wizardPrefill) {
      const nextBackbone = state.wizardPrefill?.backbone;
      const nextNumClasses = state.wizardPrefill?.numClasses;
      const supportedBackbone =
        typeof nextBackbone === "string" &&
        VISION_BACKBONE_OPTIONS.includes(nextBackbone as (typeof VISION_BACKBONE_OPTIONS)[number])
          ? (nextBackbone as (typeof VISION_BACKBONE_OPTIONS)[number])
          : null;
      setSpec((prev) => ({
        ...prev,
        projectName: state.wizardPrefill?.projectName || prev.projectName,
        experimentName: state.wizardPrefill?.experimentName || prev.experimentName,
        description: state.wizardPrefill?.description || prev.description,
        model: {
          ...prev.model,
          numClasses:
            typeof nextNumClasses === "number" && nextNumClasses > 0
              ? nextNumClasses
              : prev.model.numClasses,
          vision: {
            ...prev.model.vision,
            backbone: supportedBackbone || prev.model.vision.backbone,
          },
        },
        logging: {
          ...prev.logging,
          outputDir: state.wizardPrefill?.outputDir || prev.logging.outputDir,
        },
      }));
      setCurrentStep(0);
      setCompiledImportMode("prefill");
      setCompiledImportSource(state.source || "comfyui-bridge");
      return;
    }

    if (state.datasetPrefill) {
      setSpec((prev) => ({
        ...prev,
        data: {
          ...prev.data,
          csvPath: state.datasetPrefill?.csvPath || prev.data.csvPath,
          imageDir: state.datasetPrefill?.imageDir || prev.data.imageDir,
          imagePathColumn:
            state.datasetPrefill?.imagePathColumn || prev.data.imagePathColumn,
          targetColumn:
            state.datasetPrefill?.targetColumn || prev.data.targetColumn,
          patientIdColumn:
            state.datasetPrefill?.patientIdColumn || prev.data.patientIdColumn,
          numericalFeatures:
            state.datasetPrefill?.numericalFeatures || prev.data.numericalFeatures,
          categoricalFeatures:
            state.datasetPrefill?.categoricalFeatures ||
            prev.data.categoricalFeatures,
        },
        model: {
          ...prev.model,
          numClasses:
            typeof state.datasetPrefill?.numClasses === "number" &&
            state.datasetPrefill.numClasses > 0
              ? state.datasetPrefill.numClasses
              : prev.model.numClasses,
        },
      }));
      setCurrentStep(1);
      setCompiledImportMode("prefill");
      setCompiledImportSource(state.source || "dataset-manager");
      return;
    }

    if (state.modelPrefill) {
      const templateId = state.modelPrefill.modelTemplateId || DEFAULT_MODEL_TEMPLATE_ID;
      applyModelTemplate(
        templateId,
        state.modelPrefill,
        state.modelPrefill.customModelLabel || null,
      );
      setCurrentStep(2);
      setCompiledImportMode("prefill");
      setCompiledImportSource(state.source || "model-catalog");
    }
  }, [location.state]);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      setModelInspectionLoading(true);
      setModelInspectionError(null);
      try {
        const inspection = await inspectModelConfig({
          num_classes: spec.model.numClasses,
          use_auxiliary_heads: spec.model.useAuxiliaryHeads,
          vision: {
            backbone: spec.model.vision.backbone,
            pretrained: spec.model.vision.pretrained,
            freeze_backbone: spec.model.vision.freezeBackbone,
            feature_dim: spec.model.vision.featureDim,
            dropout: spec.model.vision.dropout,
            attention_type: spec.model.vision.attentionType,
          },
          tabular: {
            hidden_dims: spec.model.tabular.hiddenDims,
            output_dim: spec.model.tabular.outputDim,
            dropout: spec.model.tabular.dropout,
          },
          fusion: {
            fusion_type: spec.model.fusion.fusionType,
            hidden_dim: spec.model.fusion.hiddenDim,
            dropout: spec.model.fusion.dropout,
            num_heads: spec.model.fusion.numHeads,
          },
          numerical_features: spec.data.numericalFeatures,
          categorical_features: spec.data.categoricalFeatures,
          image_size: spec.data.imageSize,
          use_attention_supervision: spec.training.useAttentionSupervision,
          num_epochs: spec.training.numEpochs,
        });
        if (!cancelled) {
          setModelInspection(inspection);
        }
      } catch (error: any) {
        console.error("Failed to inspect model config:", error);
        if (!cancelled) {
          setModelInspectionError(
            error?.response?.data?.detail || error?.message || "模型检查失败",
          );
        }
      } finally {
        if (!cancelled) {
          setModelInspectionLoading(false);
        }
      }
    }, 300);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [
    spec.model.numClasses,
    spec.model.useAuxiliaryHeads,
    spec.model.vision.backbone,
    spec.model.vision.pretrained,
    spec.model.vision.freezeBackbone,
    spec.model.vision.featureDim,
    spec.model.vision.dropout,
    spec.model.vision.attentionType,
    spec.model.tabular.hiddenDims,
    spec.model.tabular.outputDim,
    spec.model.tabular.dropout,
    spec.model.fusion.fusionType,
    spec.model.fusion.hiddenDim,
    spec.model.fusion.dropout,
    spec.model.fusion.numHeads,
    spec.data.numericalFeatures,
    spec.data.categoricalFeatures,
    spec.data.imageSize,
    spec.training.useAttentionSupervision,
    spec.training.numEpochs,
  ]);

  const renderBasicsStep = () => (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      <Alert
        type="info"
        showIcon
        message="当前正式版默认开放的是问题向导 + 参数编辑层"
        description="这一步先从问题定义进入，再把你映射到当前 runtime 真正支持的模型骨架。节点式编辑仍然是高级模式，不直接替代这条默认路径。"
      />
      <Alert
        type="success"
        showIcon
        message="当前默认配置已切到 ComfyUI 适配模式"
        description="你仍在同一条 MedFusion 主线里。若需要检查 ComfyUI 连通性或选择适配档案，可打开 ComfyUI 入口页。"
        action={
          <Button size="small" onClick={() => navigate("/config/comfyui")}>
            打开 ComfyUI 入口
          </Button>
        }
      />

      <Card size="small" title="先说你现在要解决什么问题">
        <Space direction="vertical" size={6} style={{ width: "100%" }}>
          <Space>
            <Text strong>{selectedPresetLabel}</Text>
            <Tag color="processing">当前主线默认路径</Tag>
          </Space>
          <Text type="secondary">
            官方模型模板现在来自模型数据库，而不是前端硬编码 preset。
          </Text>
          <Text>{selectedTemplateDescription}</Text>
          <Text type="secondary">算力建议：{selectedTemplateCompute}</Text>
          {selectedTemplateRecommendedPreset ? (
            <Space wrap size={[8, 8]}>
              <Tag color="blue">recommended preset: {selectedTemplateRecommendedPreset}</Tag>
              {selectedTemplateCompileBoundary ? (
                <Tag color="geekblue">compile boundary: {selectedTemplateCompileBoundary}</Tag>
              ) : null}
            </Space>
          ) : null}
          {selectedTemplateCompileNotes.length ? (
            <Space direction="vertical" size={2} style={{ width: "100%" }}>
              {selectedTemplateCompileNotes.map((note) => (
                <Text key={note} type="secondary">
                  {note}
                </Text>
              ))}
            </Space>
          ) : null}
          {selectedTemplatePatchHints.length ? (
            <Space direction="vertical" size={2} style={{ width: "100%" }}>
              <Text type="secondary">模板 patch target hints：</Text>
              {selectedTemplatePatchHints.map((hint) => (
                <Text key={`${hint.path}-${hint.mode}`} type="secondary">
                  {hint.path} ({hint.mode}) - {hint.description}
                </Text>
              ))}
            </Space>
          ) : null}
          {officialModelTemplates.length ? (
            <Select
              value={selectedModelTemplateId}
              style={{ width: 360, maxWidth: "100%" }}
              options={officialModelTemplates.map((item) => ({
                label: `${item.label} · ${item.status}`,
                value: item.id,
              }))}
              onChange={(value) => {
                const template = officialModelTemplates.find((item) => item.id === value);
                applyModelTemplate(
                  value,
                  template?.wizard_prefill as ModelPrefillState | undefined,
                  template?.label || null,
                );
              }}
            />
          ) : null}
          <Button icon={<ControlOutlined />} onClick={() => navigate("/config/model")}>
            去官方模型数据库选模板
          </Button>
        </Space>
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

      <Card size="small" title="模型骨架检查">
        <Space direction="vertical" size={12} style={{ width: "100%" }}>
          <Space wrap>
            <Button icon={<ControlOutlined />} onClick={() => navigate("/config/model")}>
              打开模型数据库
            </Button>
            <Button icon={<ExperimentOutlined />} onClick={() => navigate("/config/advanced")}>
              看高级模式组件注册表
            </Button>
          </Space>

          {modelInspectionError ? (
            <Alert type="error" showIcon message={modelInspectionError} />
          ) : null}
          {modelInspection ? (
            <Alert
              type={
                modelInspection.can_enter_training
                  ? modelInspection.status === "warning"
                    ? "warning"
                    : "success"
                  : "error"
              }
              showIcon
              message={modelInspection.next_step}
              description={
                <Space wrap>
                  <Tag
                    color={
                      modelInspection.status === "ready"
                        ? "success"
                        : modelInspection.status === "warning"
                          ? "warning"
                          : "error"
                    }
                  >
                    {modelInspection.status}
                  </Tag>
                  <span>backbone: {modelInspection.summary.backbone}</span>
                  <span>fusion: {modelInspection.summary.fusion_type}</span>
                  <span>tabular features: {modelInspection.summary.tabular_feature_count}</span>
                </Space>
              }
            />
          ) : (
            <Alert
              type="info"
              showIcon
              message={modelInspectionLoading ? "正在检查模型骨架..." : "等待模型检查结果"}
            />
          )}

          {modelInspection?.runtime ? (
            <Row gutter={16}>
              <Col xs={24} md={8}>
                <Card size="small">
                  <Text type="secondary">总参数量</Text>
                  <Title level={4} style={{ margin: 0 }}>
                    {modelInspection.runtime.total_params.toLocaleString()}
                  </Title>
                </Card>
              </Col>
              <Col xs={24} md={8}>
                <Card size="small">
                  <Text type="secondary">可训练参数</Text>
                  <Title level={4} style={{ margin: 0 }}>
                    {modelInspection.runtime.trainable_params.toLocaleString()}
                  </Title>
                </Card>
              </Col>
              <Col xs={24} md={8}>
                <Card size="small">
                  <Text type="secondary">fusion 输出维度</Text>
                  <Title level={4} style={{ margin: 0 }}>
                    {modelInspection.runtime.fusion_output_dim}
                  </Title>
                </Card>
              </Col>
            </Row>
          ) : null}

          {modelInspection ? (
            <div className="editorial-grid">
              {modelInspection.checks.map((check) => (
                <div key={check.key} className="surface-note surface-note--dense">
                  <Space wrap>
                    <Tag
                      color={
                        check.status === "pass"
                          ? "success"
                          : check.status === "warning"
                            ? "warning"
                            : "error"
                      }
                    >
                      {check.status}
                    </Tag>
                    <strong>{check.label}</strong>
                  </Space>
                  <p>{check.detail}</p>
                </div>
              ))}
            </div>
          ) : null}

          {modelInspection?.issues.errors.length ? (
            modelInspection.issues.errors.map((issue) => (
              <Alert
                key={`${issue.error_code}-${issue.path}`}
                type="error"
                showIcon
                message={`${issue.path}: ${issue.message}`}
                description={issue.suggestion || undefined}
              />
            ))
          ) : null}
          {modelInspection?.issues.warnings.length ? (
            modelInspection.issues.warnings.map((issue) => (
              <Alert
                key={`${issue.error_code}-${issue.path}`}
                type="warning"
                showIcon
                message={`${issue.path}: ${issue.message}`}
                description={issue.suggestion || undefined}
              />
            ))
          ) : null}
        </Space>
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
            onClick={() => navigate(`/training?source=${trainingLaunchSource}&${trainingPrefillQuery}`)}
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
          <div className="hero-aside-panel__value">{selectedPresetLabel}</div>
          <div className="hero-aside-panel__copy">
            当前问题路径会映射到官方模型数据库里的 <strong>{selectedPresetLabel}</strong>
            ，并继续通过当前 runtime 支持的配置空间导出真实 YAML。
          </div>
          <div className="surface-note">
            算力建议：<strong>{selectedTemplateCompute}</strong>
          </div>
        </div>
      }
      metrics={[
        {
          label: "Official template",
          value: selectedPresetLabel,
          hint: selectedModelTemplate?.id || preset,
          tone: "amber",
        },
        {
          label: "Wizard step",
          value: `${currentStep + 1}/${stepItems.length}`,
          hint: stepItems[currentStep]?.title,
          tone: "blue",
        },
        {
          label: "Template source",
          value: "official catalog",
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
        style={{ marginBottom: 16 }}
        message="当前主线步骤：配置（1/3）"
        description={
          <Space wrap>
            <span>完成配置后继续进入训练监控，再到结果后台复盘。</span>
            <Button
              size="small"
              icon={<ExperimentOutlined />}
              onClick={() => navigate(`/training?source=${trainingLaunchSource}&${trainingPrefillQuery}`)}
            >
              下一步：训练监控
            </Button>
            <Button size="small" icon={<LinkOutlined />} onClick={() => navigate("/models")}>
              跳到结果后台
            </Button>
          </Space>
        }
      />
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
          message={
            compiledImportMode === "compiled"
              ? "已从高级模式导入一份可编辑配置草案"
              : "已从上游页面预填一份可编辑配置草案"
          }
          description={
            compiledImportMode === "compiled"
              ? `当前这份 RunSpec 来自 ${compiledImportSource} 的图编译结果。你现在可以继续在默认模式里微调字段，再导出 YAML 或进入训练。`
              : `当前字段来自 ${compiledImportSource} 的预填。你可以继续在默认模式里微调字段，再导出 YAML 或进入训练。`
          }
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
