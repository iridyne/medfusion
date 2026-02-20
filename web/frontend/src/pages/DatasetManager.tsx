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
  message,
  Dropdown,
  Popconfirm,
  Statistic,
  Row,
  Col,
  Progress,
  Empty,
  Tooltip,
} from "antd";
import {
  PlusOutlined,
  SearchOutlined,
  EyeOutlined,
  DeleteOutlined,
  DownloadOutlined,
  MoreOutlined,
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
import DatasetPreview from "../components/dataset/DatasetPreview";
import DatasetStatistics from "../components/dataset/DatasetStatistics";

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
  description?: string;
  tags?: string[];
  progress?: number;
  error_message?: string;
}

const DatasetManager: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [uploadModalVisible, setUploadModalVisible] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [previewVisible, setPreviewVisible] = useState(false);

  // 获取数据集列表
  const fetchDatasets = async () => {
    setLoading(true);
    try {
      // TODO: 调用后端 API
      // const response = await fetch("/api/datasets");
      // const data = await response.json();
      // setDatasets(data);

      // 模拟数据
      const mockData: Dataset[] = [
        {
          id: "1",
          name: "Chest X-Ray Dataset",
          type: "image",
          status: "ready",
          size: 1024 * 1024 * 500, // 500MB
          samples: 5000,
          classes: 2,
          created_at: "2024-01-15T10:30:00Z",
          updated_at: "2024-01-15T11:00:00Z",
          description: "胸部 X 光图像数据集，包含正常和肺炎两类",
          tags: ["医学影像", "分类"],
        },
        {
          id: "2",
          name: "Clinical Records",
          type: "tabular",
          status: "ready",
          size: 1024 * 1024 * 10, // 10MB
          samples: 10000,
          classes: 3,
          created_at: "2024-01-16T09:00:00Z",
          updated_at: "2024-01-16T09:30:00Z",
          description: "临床记录表格数据",
          tags: ["表格数据", "多分类"],
        },
        {
          id: "3",
          name: "Multimodal Cancer Dataset",
          type: "multimodal",
          status: "processing",
          size: 1024 * 1024 * 1024 * 2, // 2GB
          samples: 3000,
          classes: 4,
          created_at: "2024-01-17T14:00:00Z",
          updated_at: "2024-01-17T14:30:00Z",
          description: "多模态癌症数据集（影像 + 临床数据）",
          tags: ["多模态", "癌症"],
          progress: 65,
        },
      ];
      setDatasets(mockData);
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
      // TODO: 调用后端 API
      // await fetch(`/api/datasets/${id}`, { method: "DELETE" });
      message.success("数据集删除成功");
      fetchDatasets();
    } catch (error) {
      message.error("删除失败");
      console.error(error);
    }
  };

  // 下载数据集
  const handleDownload = async (dataset: Dataset) => {
    try {
      // TODO: 调用后端 API
      message.info(`开始下载 ${dataset.name}`);
    } catch (error) {
      message.error("下载失败");
      console.error(error);
    }
  };

  // 查看详情
  const handleViewDetails = (dataset: Dataset) => {
    setSelectedDataset(dataset);
    setDrawerVisible(true);
  };

  // 预览数据
  const handlePreview = (dataset: Dataset) => {
    setSelectedDataset(dataset);
    setPreviewVisible(true);
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
      uploading: { color: "blue", icon: <SyncOutlined spin />, text: "上传中" },
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
      width: 200,
      fixed: "right",
      render: (_, record) => (
        <Space>
          <Tooltip title="预览">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => handlePreview(record)}
              disabled={record.status !== "ready"}
            />
          </Tooltip>
          <Tooltip title="下载">
            <Button
              type="text"
              icon={<DownloadOutlined />}
              onClick={() => handleDownload(record)}
              disabled={record.status !== "ready"}
            />
          </Tooltip>
          <Dropdown
            menu={{
              items: [
                {
                  key: "details",
                  label: "查看详情",
                  onClick: () => handleViewDetails(record),
                },
                {
                  key: "delete",
                  label: "删除",
                  danger: true,
                  onClick: () => {
                    Modal.confirm({
                      title: "确认删除",
                      content: `确定要删除数据集 "${record.name}" 吗？此操作不可恢复。`,
                      okText: "删除",
                      okType: "danger",
                      cancelText: "取消",
                      onOk: () => handleDelete(record.id),
                    });
                  },
                },
              ],
            }}
          >
            <Button type="text" icon={<MoreOutlined />} />
          </Dropdown>
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
    <div style={{ padding: "24px" }}>
      {/* 统计卡片 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="数据集总数"
              value={statistics.total}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="就绪"
              value={statistics.ready}
              valueStyle={{ color: "#3f8600" }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="处理中"
              value={statistics.processing}
              valueStyle={{ color: "#cf1322" }}
              prefix={<SyncOutlined spin />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="总大小"
              value={formatSize(statistics.totalSize)}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* 主内容卡片 */}
      <Card
        title={
          <Space>
            <DatabaseOutlined />
            <span>数据集管理</span>
          </Space>
        }
        extra={
          <Space>
            <Input.Search
              placeholder="搜索数据集"
              allowClear
              style={{ width: 250 }}
              onChange={(e) => setSearchText(e.target.value)}
              prefix={<SearchOutlined />}
            />
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setUploadModalVisible(true)}
            >
              上传数据集
            </Button>
          </Space>
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

      {/* 上传模态框 */}
      <Modal
        title="上传数据集"
        open={uploadModalVisible}
        onCancel={() => setUploadModalVisible(false)}
        footer={null}
        width={800}
        destroyOnClose
      >
        <DatasetUploader
          onSuccess={() => {
            setUploadModalVisible(false);
            fetchDatasets();
          }}
          onCancel={() => setUploadModalVisible(false)}
        />
      </Modal>

      {/* 详情抽屉 */}
      <Drawer
        title="数据集详情"
        placement="right"
        width={600}
        open={drawerVisible}
        onClose={() => setDrawerVisible(false)}
        destroyOnClose
      >
        {selectedDataset && (
          <DatasetStatistics dataset={selectedDataset} />
        )}
      </Drawer>

      {/* 预览模态框 */}
      <Modal
        title={`预览: ${selectedDataset?.name}`}
        open={previewVisible}
        onCancel={() => setPreviewVisible(false)}
        footer={null}
        width={1200}
        destroyOnClose
      >
        {selectedDataset && (
          <DatasetPreview dataset={selectedDataset} />
        )}
      </Modal>
    </div>
  );
};

export default DatasetManager;
