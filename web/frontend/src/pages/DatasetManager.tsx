import React, { useState, useEffect } from "react";
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Modal,
  Drawer,
  Input,
  Descriptions,
  message,
  Statistic,
  Row,
  Col,
  Progress,
  Tooltip,
} from "antd";
import {
  PlusOutlined,
  SearchOutlined,
  EyeOutlined,
  DeleteOutlined,
  DatabaseOutlined,
  FileImageOutlined,
  TableOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  SyncOutlined,
} from "@ant-design/icons";
import type { ColumnsType } from "antd/es/table";
import DatasetUploader from "../components/dataset/DatasetUploader";
import { deleteDataset, getDatasets } from "../api/datasets";
import PageScaffold from "@/components/layout/PageScaffold";

interface Dataset {
  id: string;
  name: string;
  type: "image" | "tabular" | "multimodal";
  status: "uploading" | "processing" | "ready" | "error";
  size: number;
  samples: number;
  classes: number;
  created_at: string;
  updated_at: string;
  data_path?: string;
  description?: string;
  tags?: string[];
  progress?: number;
  error_message?: string;
  analysis?: Record<string, any>;
}

const DatasetManager: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [registerModalVisible, setRegisterModalVisible] = useState(false);
  const [searchText, setSearchText] = useState("");

  // 获取数据集列表
  const fetchDatasets = async () => {
    setLoading(true);
    try {
      const data = await getDatasets({ limit: 200 });
      const mapped: Dataset[] = (data || []).map((item: any) => ({
        id: String(item.id),
        name: item.name,
        type: (item.dataset_type || "image") as Dataset["type"],
        status: (item.status || "ready") as Dataset["status"],
        size: item.size_bytes || 0,
        samples: item.num_samples || 0,
        classes: item.num_classes || 0,
        created_at: item.created_at || new Date().toISOString(),
        updated_at: item.updated_at || item.created_at || new Date().toISOString(),
        data_path: item.data_path,
        description: item.description,
        tags: item.tags || [],
        analysis: item.analysis || undefined,
        progress:
          item.status === "processing" || item.status === "uploading"
            ? item.progress ?? 50
            : undefined,
        error_message: item.error_message,
      }));
      setDatasets(mapped);
    } catch (error) {
      message.error("获取数据集列表失败");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  // 删除数据集
  const handleDelete = async (id: string) => {
    try {
      await deleteDataset(Number(id));
      message.success("数据集删除成功");
      fetchDatasets();
    } catch (error) {
      message.error("删除失败");
      console.error(error);
    }
  };

  // 查看详情
  const handleViewDetails = (dataset: Dataset) => {
    setSelectedDataset(dataset);
    setDrawerVisible(true);
  };

  // 格式化文件大小
  const formatSize = (bytes: number): string => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  };

  // 格式化日期
  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return date.toLocaleString("zh-CN");
  };

  // 获取状态标签
  const getStatusTag = (status: Dataset["status"], progress?: number) => {
    const statusConfig = {
      uploading: { color: "blue", icon: <SyncOutlined spin />, text: "登记中" },
      processing: { color: "orange", icon: <ClockCircleOutlined />, text: "处理中" },
      ready: { color: "green", icon: <CheckCircleOutlined />, text: "就绪" },
      error: { color: "red", icon: <ExclamationCircleOutlined />, text: "错误" },
    };

    const config = statusConfig[status];
    return (
      <Space>
        <Tag color={config.color} icon={config.icon}>
          {config.text}
        </Tag>
        {(status === "uploading" || status === "processing") && progress !== undefined && (
          <Progress percent={progress} size="small" style={{ width: 100 }} />
        )}
      </Space>
    );
  };

  // 获取类型图标
  const getTypeIcon = (type: Dataset["type"]) => {
    const icons = {
      image: <FileImageOutlined style={{ color: "#1890ff" }} />,
      tabular: <TableOutlined style={{ color: "#52c41a" }} />,
      multimodal: <DatabaseOutlined style={{ color: "#722ed1" }} />,
    };
    return icons[type];
  };

  // 表格列定义
  const columns: ColumnsType<Dataset> = [
    {
      title: "数据集名称",
      dataIndex: "name",
      key: "name",
      width: 250,
      render: (text, record) => (
        <Space>
          {getTypeIcon(record.type)}
          <a onClick={() => handleViewDetails(record)}>{text}</a>
        </Space>
      ),
      filteredValue: searchText ? [searchText] : null,
      onFilter: (value, record) =>
        record.name.toLowerCase().includes((value as string).toLowerCase()) ||
        (record.description?.toLowerCase().includes((value as string).toLowerCase()) ?? false),
    },
    {
      title: "类型",
      dataIndex: "type",
      key: "type",
      width: 100,
      filters: [
        { text: "图像", value: "image" },
        { text: "表格", value: "tabular" },
        { text: "多模态", value: "multimodal" },
      ],
      onFilter: (value, record) => record.type === value,
      render: (type: Dataset["type"]) => {
        const typeMap = {
          image: "图像",
          tabular: "表格",
          multimodal: "多模态",
        };
        return typeMap[type];
      },
    },
    {
      title: "状态",
      dataIndex: "status",
      key: "status",
      width: 200,
      render: (status, record) => getStatusTag(status, record.progress),
    },
    {
      title: "样本数",
      dataIndex: "samples",
      key: "samples",
      width: 100,
      sorter: (a, b) => a.samples - b.samples,
      render: (samples) => samples.toLocaleString(),
    },
    {
      title: "类别数",
      dataIndex: "classes",
      key: "classes",
      width: 100,
      sorter: (a, b) => a.classes - b.classes,
    },
    {
      title: "大小",
      dataIndex: "size",
      key: "size",
      width: 120,
      sorter: (a, b) => a.size - b.size,
      render: (size) => formatSize(size),
    },
    {
      title: "创建时间",
      dataIndex: "created_at",
      key: "created_at",
      width: 180,
      sorter: (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      render: (date) => formatDate(date),
    },
    {
      title: "操作",
      key: "action",
      width: 160,
      render: (_, record) => (
        <Space>
          <Tooltip title="查看详情">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => handleViewDetails(record)}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button
              type="text"
              danger
              icon={<DeleteOutlined />}
              onClick={() => {
                Modal.confirm({
                  title: "确认删除",
                  content: `确定要删除数据集 "${record.name}" 吗？此操作不可恢复。`,
                  okText: "删除",
                  okType: "danger",
                  cancelText: "取消",
                  onOk: () => handleDelete(record.id),
                });
              }}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  // 统计信息
  const statistics = {
    total: datasets.length,
    ready: datasets.filter((d) => d.status === "ready").length,
    processing: datasets.filter((d) => d.status === "processing").length,
    totalSize: datasets.reduce((sum, d) => sum + d.size, 0),
  };

  return (
    <PageScaffold
      eyebrow="Data Intake Registry"
      title="把研究数据入口整理成一套可审查的资产台账"
      description="登记本地目录、观察处理状态，并把实验前置的数据上下文固定下来。这里既是训练前的入口，也是 OSS 评估者理解可用资产的第一屏。"
      chips={[
        { label: "Local-first registry", tone: "blue" },
        { label: "Processing visibility", tone: "teal" },
        { label: "Research dataset ledger", tone: "amber" },
      ]}
      actions={
        <Button
          type="primary"
          size="large"
          icon={<PlusOutlined />}
          onClick={() => setRegisterModalVisible(true)}
        >
          登记数据集
        </Button>
      }
      aside={
        <div className="hero-aside-panel">
          <span className="hero-aside-panel__label">Registry pulse</span>
          <div className="hero-aside-panel__value">
            {statistics.ready} 个数据集已准备好进入训练
          </div>
          <div className="hero-aside-panel__copy">
            {statistics.processing > 0
              ? `${statistics.processing} 个数据集仍在处理链路中，建议先确认目录结构与分析状态。`
              : "当前没有数据集积压在处理中，适合直接进入训练或结果验证。"}
          </div>
          <div className="surface-note">
            总体积 {formatSize(statistics.totalSize)}，覆盖 {statistics.total} 份登记记录。
          </div>
        </div>
      }
      metrics={[
        {
          label: "Registered datasets",
          value: statistics.total.toLocaleString(),
          hint: "All indexed directories",
          tone: "blue",
        },
        {
          label: "Ready for training",
          value: statistics.ready.toLocaleString(),
          hint: "已完成分析或无需预处理",
          tone: "teal",
        },
        {
          label: "In processing",
          value: statistics.processing.toLocaleString(),
          hint: "等待处理完成后再进入训练",
          tone: "rose",
        },
        {
          label: "Storage footprint",
          value: formatSize(statistics.totalSize),
          hint: "Current indexed volume",
          tone: "amber",
        },
      ]}
    >
      <Card
        className="surface-card"
        title={
          <Space>
            <DatabaseOutlined />
            <span>数据登记面板</span>
          </Space>
        }
        extra={
          <Input.Search
            placeholder="搜索数据集"
            allowClear
            style={{ width: 260 }}
            onChange={(e) => setSearchText(e.target.value)}
            prefix={<SearchOutlined />}
          />
        }
      >
        <Table
          columns={columns}
          dataSource={datasets}
          rowKey="id"
          loading={loading}
          pagination={{
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 个数据集`,
          }}
          scroll={{ x: 1400 }}
        />
      </Card>

      <div className="split-grid">
        <Card className="surface-card">
          <div className="section-heading">
            <div>
              <div className="section-heading__eyebrow">Registry protocol</div>
              <h2 className="section-heading__title">推荐的数据登记节奏</h2>
              <p className="section-heading__description">
                先把目录和字段登记清楚，再进入训练或结果导入，能明显减少路径和 schema 层面的歧义。
              </p>
            </div>
          </div>

          <div className="workbench-flow">
            <div className="flow-step">
              <strong>1. 登记本地目录</strong>
              <p>把路径、类型、标签和描述补齐，便于团队成员快速判断数据用途。</p>
            </div>
            <div className="flow-step">
              <strong>2. 观察处理状态</strong>
              <p>等待分析链路完成，再根据样本量、类别数和体量决定训练策略。</p>
            </div>
            <div className="flow-step">
              <strong>3. 回到训练或向导</strong>
              <p>数据资产稳定后，再进入训练监控页或 RunSpec 向导配置真实实验。</p>
            </div>
          </div>
        </Card>

        <Card className="surface-card">
          <div className="section-heading">
            <div>
              <div className="section-heading__eyebrow">Information scent</div>
              <h2 className="section-heading__title">当前页面提供什么线索</h2>
              <p className="section-heading__description">
                这张表既服务日常操作，也让 OSS 评估者一眼看出系统是否真的管理数据资产。
              </p>
            </div>
          </div>

          <div className="stack-grid">
            <div className="surface-note">
              <strong>状态语义</strong>
              <p>区分 `登记中`、`处理中`、`就绪` 和 `错误`，帮助你判断是否可以进入下一步。</p>
            </div>
            <div className="surface-note">
              <strong>结构线索</strong>
              <p>样本数、类别数、体积和时间戳被放到同一视图，便于快速评估实验成熟度。</p>
            </div>
            <div className="surface-note">
              <strong>详情抽屉</strong>
              <p>点击名称即可展开完整路径、标签和分析 JSON，不需要离开主表格。</p>
            </div>
          </div>
        </Card>
      </div>

      <Modal
        title="登记本地数据集"
        open={registerModalVisible}
        onCancel={() => setRegisterModalVisible(false)}
        footer={null}
        width={720}
        destroyOnClose
        rootClassName="surface-modal"
      >
        <DatasetUploader
          onSuccess={() => {
            setRegisterModalVisible(false);
            fetchDatasets();
          }}
          onCancel={() => setRegisterModalVisible(false)}
        />
      </Modal>

      <Drawer
        title="数据集详情"
        placement="right"
        width={600}
        open={drawerVisible}
        onClose={() => setDrawerVisible(false)}
        destroyOnClose
        rootClassName="surface-drawer"
      >
        {selectedDataset && (
          <Descriptions column={1} bordered size="small">
            <Descriptions.Item label="数据集名称">
              {selectedDataset.name}
            </Descriptions.Item>
            <Descriptions.Item label="数据类型">
              {selectedDataset.type === "image"
                ? "图像"
                : selectedDataset.type === "tabular"
                  ? "表格"
                  : "多模态"}
            </Descriptions.Item>
            <Descriptions.Item label="状态">
              {getStatusTag(selectedDataset.status, selectedDataset.progress)}
            </Descriptions.Item>
            <Descriptions.Item label="本地目录路径">
              <span style={{ wordBreak: "break-all" }}>
                {selectedDataset.data_path || "-"}
              </span>
            </Descriptions.Item>
            <Descriptions.Item label="样本数">
              {selectedDataset.samples.toLocaleString()}
            </Descriptions.Item>
            <Descriptions.Item label="类别数">
              {selectedDataset.classes || "-"}
            </Descriptions.Item>
            <Descriptions.Item label="数据集大小">
              {formatSize(selectedDataset.size)}
            </Descriptions.Item>
            <Descriptions.Item label="创建时间">
              {formatDate(selectedDataset.created_at)}
            </Descriptions.Item>
            <Descriptions.Item label="更新时间">
              {formatDate(selectedDataset.updated_at)}
            </Descriptions.Item>
            <Descriptions.Item label="描述">
              {selectedDataset.description || "暂无描述"}
            </Descriptions.Item>
            <Descriptions.Item label="标签">
              {selectedDataset.tags && selectedDataset.tags.length > 0 ? (
                <Space wrap>
                  {selectedDataset.tags.map((tag) => (
                    <Tag key={tag}>{tag}</Tag>
                  ))}
                </Space>
              ) : (
                "暂无标签"
              )}
            </Descriptions.Item>
            <Descriptions.Item label="分析结果">
              {selectedDataset.analysis ? (
                <pre
                  style={{
                    margin: 0,
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  {JSON.stringify(selectedDataset.analysis, null, 2)}
                </pre>
              ) : (
                "暂无分析结果"
              )}
            </Descriptions.Item>
          </Descriptions>
        )}
      </Drawer>
    </PageScaffold>
  );
};

export default DatasetManager;
