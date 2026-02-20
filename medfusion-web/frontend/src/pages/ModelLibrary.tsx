import { useEffect, useState } from "react";
import {
  Card,
  Tag,
  Space,
  Button,
  Input,
  Select,
  Row,
  Col,
  Statistic,
  Modal,
  Descriptions,
  message,
  Upload,
  Tabs,
} from "antd";
import {
  SearchOutlined,
  DownloadOutlined,
  DeleteOutlined,
  EyeOutlined,
  UploadOutlined,
  ExperimentOutlined,
} from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import { getModels } from "@/api/models";
import VirtualList from "@/components/VirtualList";

interface Model {
  id: string;
  name: string;
  backbone: string;
  numClasses: number;
  params: number;
  accuracy?: number;
  createdAt: string;
  description: string;
  size: number;
  format: "pytorch" | "onnx" | "torchscript";
}

export default function ModelLibrary() {
  const { t } = useTranslation();
  const [models, setModels] = useState<Model[]>([
    {
      id: "1",
      name: "肺癌分类模型 v1.0",
      backbone: "resnet50",
      numClasses: 3,
      params: 25600000,
      accuracy: 0.9234,
      createdAt: "2024-02-20 10:30:00",
      description: "基于 ResNet50 的肺癌三分类模型，在 LIDC-IDRI 数据集上训练",
      size: 102400000,
      format: "pytorch",
    },
    {
      id: "2",
      name: "多视图融合模型 v2.1",
      backbone: "efficientnet_b2",
      numClasses: 2,
      params: 9200000,
      accuracy: 0.9567,
      createdAt: "2024-02-19 14:20:00",
      description: "使用注意力融合的多视图模型，支持 4 个视图输入",
      size: 36800000,
      format: "pytorch",
    },
    {
      id: "3",
      name: "ViT 分类器",
      backbone: "vit_b16",
      numClasses: 5,
      params: 86400000,
      accuracy: 0.8891,
      createdAt: "2024-02-18 09:15:00",
      description: "Vision Transformer 基础模型，适用于高分辨率医学图像",
      size: 345600000,
      format: "pytorch",
    },
  ]);
  const [filteredModels, setFilteredModels] = useState<Model[]>(models);
  const [loading, setLoading] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [filterBackbone, setFilterBackbone] = useState<string>("all");
  const [filterFormat, setFilterFormat] = useState<string>("all");
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);

  useEffect(() => {
    filterModels();
  }, [searchText, filterBackbone, filterFormat, models]);

  const filterModels = () => {
    let filtered = models;

    if (searchText) {
      filtered = filtered.filter(
        (m) =>
          m.name.toLowerCase().includes(searchText.toLowerCase()) ||
          m.description.toLowerCase().includes(searchText.toLowerCase()),
      );
    }

    if (filterBackbone !== "all") {
      filtered = filtered.filter((m) => m.backbone === filterBackbone);
    }

    if (filterFormat !== "all") {
      filtered = filtered.filter((m) => m.format === filterFormat);
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
    message.success(t("models.downloadStarted", { name: model.name }));
    // TODO: 实现下载逻辑
  };

  const handleDelete = (model: Model) => {
    Modal.confirm({
      title: t("common.confirmDelete"),
      content: t("models.deleteConfirm", { name: model.name }),
      onOk: () => {
        setModels(models.filter((m) => m.id !== model.id));
        message.success(t("models.deleteSuccess"));
      },
    });
  };

  const handleUpload = () => {
    message.info(t("models.uploadInProgress"));
    // TODO: 实现上传逻辑
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
  const avgAccuracy =
    models
      .filter((m) => m.accuracy)
      .reduce((sum, m) => sum + (m.accuracy || 0), 0) /
    models.filter((m) => m.accuracy).length;

  return (
    <div style={{ padding: 24 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <h1>{t("nav.models")}</h1>
        <Button type="primary" icon={<UploadOutlined />} onClick={handleUpload}>
          {t("models.uploadModel")}
        </Button>
      </div>

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title={t("models.totalModels")}
              value={models.length}
              prefix={<ExperimentOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title={t("models.totalParams")}
              value={formatParams(totalParams)}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title={t("models.totalSize")}
              value={formatSize(totalSize)}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title={t("models.avgAccuracy")}
              value={avgAccuracy * 100}
              precision={2}
              suffix="%"
            />
          </Card>
        </Col>
      </Row>

      <Card style={{ marginTop: 16 }}>
        <Space style={{ marginBottom: 16, width: "100%" }} direction="vertical">
          <Row gutter={16}>
            <Col span={12}>
              <Input
                placeholder={t("models.searchPlaceholder")}
                prefix={<SearchOutlined />}
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                allowClear
              />
            </Col>
            <Col span={6}>
              <Select
                style={{ width: "100%" }}
                placeholder={t("models.filterBackbone")}
                value={filterBackbone}
                onChange={setFilterBackbone}
              >
                <Select.Option value="all">
                  {t("models.allBackbones")}
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
                placeholder={t("models.filterFormat")}
                value={filterFormat}
                onChange={setFilterFormat}
              >
                <Select.Option value="all">
                  {t("models.allFormats")}
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
                    <Tag>{model.format.toUpperCase()}</Tag>
                  </Space>
                  <div style={{ color: "#666", marginBottom: 8 }}>
                    {model.description}
                  </div>
                  <div style={{ fontSize: 12, color: "#999" }}>
                    {t("models.numClasses")}: {model.numClasses} |{" "}
                    {t("models.size")}: {formatSize(model.size)} |{" "}
                    {t("models.createdAt")}: {model.createdAt}
                  </div>
                </div>
                <Space>
                  <Button
                    size="small"
                    icon={<EyeOutlined />}
                    onClick={() => handleViewDetail(model)}
                  >
                    {t("common.detail")}
                  </Button>
                  <Button
                    size="small"
                    icon={<DownloadOutlined />}
                    onClick={() => handleDownload(model)}
                  >
                    {t("common.download")}
                  </Button>
                  <Button
                    size="small"
                    danger
                    icon={<DeleteOutlined />}
                    onClick={() => handleDelete(model)}
                  >
                    {t("common.delete")}
                  </Button>
                </Space>
              </div>
            )}
          />
        </div>
      </Card>

      <Modal
        title={t("models.modelDetail")}
        open={detailModalOpen}
        onCancel={() => setDetailModalOpen(false)}
        width={700}
        footer={[
          <Button key="close" onClick={() => setDetailModalOpen(false)}>
            {t("common.close")}
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
            {t("common.download")}
          </Button>,
        ]}
      >
        {selectedModel && (
          <Descriptions bordered column={2}>
            <Descriptions.Item label={t("models.modelName")} span={2}>
              {selectedModel.name}
            </Descriptions.Item>
            <Descriptions.Item label={t("models.backbone")}>
              {selectedModel.backbone}
            </Descriptions.Item>
            <Descriptions.Item label={t("models.format")}>
              {selectedModel.format.toUpperCase()}
            </Descriptions.Item>
            <Descriptions.Item label={t("models.numClasses")}>
              {selectedModel.numClasses}
            </Descriptions.Item>
            <Descriptions.Item label={t("models.params")}>
              {formatParams(selectedModel.params)}
            </Descriptions.Item>
            <Descriptions.Item label={t("models.accuracy")}>
              {selectedModel.accuracy
                ? `${(selectedModel.accuracy * 100).toFixed(2)}%`
                : "N/A"}
            </Descriptions.Item>
            <Descriptions.Item label={t("models.size")}>
              {formatSize(selectedModel.size)}
            </Descriptions.Item>
            <Descriptions.Item label={t("models.createdAt")} span={2}>
              {selectedModel.createdAt}
            </Descriptions.Item>
            <Descriptions.Item label={t("models.description")} span={2}>
              {selectedModel.description}
            </Descriptions.Item>
          </Descriptions>
        )}
      </Modal>
    </div>
  );
}
