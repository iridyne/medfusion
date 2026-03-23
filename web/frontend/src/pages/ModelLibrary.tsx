import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
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
  SearchOutlined,
  DownloadOutlined,
  DeleteOutlined,
  EyeOutlined,
  ExperimentOutlined,
  TrophyOutlined,
  ImportOutlined,
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
import ModelResultPanel from "@/components/model/ModelResultPanel";
import VirtualList from "@/components/VirtualList";

const { Paragraph, Text } = Typography;

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
  name?: string;
  description?: string;
  tags?: string;
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
  const [searchParams, setSearchParams] = useSearchParams();
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
    if (searchParams.get("action") !== "import") {
      return;
    }
    setImportModalOpen(true);
    const next = new URLSearchParams(searchParams);
    next.delete("action");
    setSearchParams(next, { replace: true });
  }, [searchParams, setSearchParams]);

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

  return (
    <div style={{ padding: 24 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <h1 style={{ marginBottom: 0 }}>模型库</h1>
        <Space>
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
        </Space>
      </div>

      {latestModel && (
        <Card
          style={{ marginTop: 16 }}
          bodyStyle={{ paddingBottom: 8 }}
        >
          <Row gutter={24}>
            <Col span={16}>
              <Space align="start">
                <TrophyOutlined style={{ fontSize: 28, color: "#faad14", marginTop: 4 }} />
                <div>
                  <div style={{ fontSize: 20, fontWeight: 700 }}>
                    最新结果：{latestModel.name}
                  </div>
                  <Paragraph style={{ marginBottom: 12, marginTop: 8 }}>
                    {latestModel.descriptionText || "这是最近一次训练自动沉淀下来的模型结果，适合直接展示多模态项目的训练产出。"}
                  </Paragraph>
                  <Space wrap>
                    <Tag color="blue">{latestModel.backbone}</Tag>
                    <Tag color="green">{latestModel.dataset_name || "未命名数据集"}</Tag>
                    <Tag color="purple">{latestModel.displayFormat.toUpperCase()}</Tag>
                    {latestModel.tags?.map((tag) => (
                      <Tag key={tag}>{tag}</Tag>
                    ))}
                  </Space>
                </div>
              </Space>
            </Col>
            <Col span={8}>
              <Row gutter={[12, 12]}>
                <Col span={12}>
                  <Statistic
                    title="Accuracy"
                    value={(latestModel.accuracy ?? 0) * 100}
                    precision={2}
                    suffix="%"
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="AUC"
                    value={latestModel.visualizations?.roc_curve?.auc ?? latestModel.accuracy ?? 0}
                    precision={4}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Epochs"
                    value={latestModel.trained_epochs ?? 0}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="训练时长"
                    value={
                      latestModel.training_time
                        ? `${Math.round(latestModel.training_time)}s`
                        : "-"
                    }
                  />
                </Col>
              </Row>
            </Col>
          </Row>
        </Card>
      )}

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="模型总数"
              value={models.length}
              prefix={<ExperimentOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="总参数量"
              value={formatParams(totalParams)}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="总文件大小"
              value={formatSize(totalSize)}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均准确率"
              value={avgAccuracy * 100}
              precision={2}
              suffix="%"
            />
          </Card>
        </Col>
      </Row>

      <Card style={{ marginTop: 16 }} loading={loading}>
        <Space style={{ marginBottom: 16, width: "100%" }} direction="vertical">
          <Row gutter={16}>
            <Col span={12}>
              <Input
                placeholder="搜索模型名称或描述"
                prefix={<SearchOutlined />}
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                allowClear
              />
            </Col>
            <Col span={6}>
              <Select
                style={{ width: "100%" }}
                placeholder="按骨干网络筛选"
                value={filterBackbone}
                onChange={setFilterBackbone}
              >
                <Select.Option value="all">
                  全部骨干网络
                </Select.Option>
                {backboneOptions.map((opt) => (
                  <Select.Option key={opt} value={opt}>
                    {opt}
                  </Select.Option>
                ))}
              </Select>
            </Col>
            <Col span={6}>
              <Select
                style={{ width: "100%" }}
                placeholder="按格式筛选"
                value={filterFormat}
                onChange={setFilterFormat}
              >
                <Select.Option value="all">
                  全部格式
                </Select.Option>
                <Select.Option value="pytorch">PyTorch</Select.Option>
                <Select.Option value="onnx">ONNX</Select.Option>
                <Select.Option value="torchscript">TorchScript</Select.Option>
              </Select>
            </Col>
          </Row>
        </Space>

        <div style={{ height: 600 }}>
          <VirtualList
            data={filteredModels}
            itemHeight={120}
            renderItem={(model) => (
              <div
                style={{
                  padding: "16px",
                  borderBottom: "1px solid #f0f0f0",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                }}
              >
                <div style={{ flex: 1 }}>
                  <Space style={{ marginBottom: 8 }}>
                    <span style={{ fontWeight: "bold", fontSize: 16 }}>
                      {model.name}
                    </span>
                    <Tag color="blue">{model.backbone}</Tag>
                    <Tag color="green">{formatParams(model.params)}</Tag>
                    {model.accuracy && (
                      <Tag color="orange">
                        {(model.accuracy * 100).toFixed(2)}%
                      </Tag>
                    )}
                    <Tag>{model.displayFormat.toUpperCase()}</Tag>
                  </Space>
                  <div style={{ color: "#666", marginBottom: 8 }}>
                    {model.descriptionText || "演示型 MVP 自动沉淀的模型记录"}
                  </div>
                  <div style={{ fontSize: 12, color: "#999" }}>
                    数据集: {model.dataset_name || "-"} | 类别数: {model.numClasses} | 文件大小: {formatSize(model.size)} | 创建时间:{" "}
                    {model.createdAt ? new Date(model.createdAt).toLocaleString("zh-CN") : "-"}
                  </div>
                  <div style={{ fontSize: 12, color: "#999", marginTop: 6 }}>
                    AUC: {model.visualizations?.roc_curve?.auc?.toFixed(4) || "-"} | Loss:{" "}
                    {model.loss?.toFixed(4) || "-"} | Attention Maps:{" "}
                    {model.visualizations?.attention_maps?.length || 0}
                  </div>
                </div>
                <Space>
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
                </Space>
              </div>
            )}
          />
        </div>
      </Card>

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
      >
        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <Alert
            type="info"
            showIcon
            message="把真实 CLI 训练产物接进结果页"
            description="这里会直接调用 /api/models/import-run：读取 config 和 checkpoint，生成 validation / ROC / 混淆矩阵 / attention artifact，并把结果写入模型库。"
          />

          <Form<ImportFormValues>
            form={importForm}
            layout="vertical"
            initialValues={{
              split: "test",
              attention_samples: 4,
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
                <Form.Item label="模型名称（可选）" name="name">
                  <Input placeholder="例如：pathology-mvp-v1" />
                </Form.Item>
              </Col>
              <Col span={24}>
                <Form.Item label="描述（可选）" name="description">
                  <Input.TextArea
                    rows={3}
                    placeholder="例如：真实 CLI 训练导入，用于 dashboard 演示和 README 截图"
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
        footer={[
          <Button key="close" onClick={() => setDetailModalOpen(false)}>
            关闭
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
            <Alert
              type="success"
              showIcon
              message="结果详情已强化"
              description="当前详情页会同时展示多模态指标、ROC/AUC、混淆矩阵、注意力热力图和结果文件，适合直接用于演示。"
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
                {selectedModel.descriptionText || "演示型 MVP 自动沉淀的模型记录"}
              </Descriptions.Item>
            </Descriptions>

            <Divider style={{ margin: 0 }} />
            <ModelResultPanel model={selectedModel} />
          </Space>
        )}
      </Modal>
    </div>
  );
}
