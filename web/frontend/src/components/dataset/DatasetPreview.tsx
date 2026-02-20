import React, { useState, useEffect } from "react";
import {
  Tabs,
  Image,
  Table,
  Card,
  Row,
  Col,
  Select,
  Pagination,
  Empty,
  Spin,
  Tag,
  Space,
  Statistic,
} from "antd";
import {
  FileImageOutlined,
  TableOutlined,
  BarChartOutlined,
  EyeOutlined,
} from "@ant-design/icons";
import type { ColumnsType } from "antd/es/table";

interface Dataset {
  id: string;
  name: string;
  type: "image" | "tabular" | "multimodal";
  status: string;
  size: number;
  samples: number;
  classes: number;
  created_at: string;
  updated_at: string;
  description?: string;
  tags?: string[];
}

interface DatasetPreviewProps {
  dataset: Dataset;
}

interface ImageSample {
  id: string;
  url: string;
  label: string;
  metadata?: Record<string, any>;
}

interface TabularSample {
  id: string;
  [key: string]: any;
}

const DatasetPreview: React.FC<DatasetPreviewProps> = ({ dataset }) => {
  const [loading, setLoading] = useState(false);
  const [imageSamples, setImageSamples] = useState<ImageSample[]>([]);
  const [tabularSamples, setTabularSamples] = useState<TabularSample[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [selectedClass, setSelectedClass] = useState<string | null>(null);
  const [classDistribution, setClassDistribution] = useState<Record<string, number>>({});

  // 获取预览数据
  const fetchPreviewData = async () => {
    setLoading(true);
    try {
      // TODO: 调用后端 API
      // const response = await fetch(
      //   `/api/datasets/${dataset.id}/preview?page=${currentPage}&size=${pageSize}&class=${selectedClass || ""}`
      // );
      // const data = await response.json();

      // 模拟图像数据
      if (dataset.type === "image" || dataset.type === "multimodal") {
        const mockImages: ImageSample[] = Array.from({ length: 20 }, (_, i) => ({
          id: `img-${i}`,
          url: `https://via.placeholder.com/300x300?text=Sample+${i + 1}`,
          label: i % 2 === 0 ? "Normal" : "Abnormal",
          metadata: {
            width: 512,
            height: 512,
            format: "DICOM",
          },
        }));
        setImageSamples(mockImages);
      }

      // 模拟表格数据
      if (dataset.type === "tabular" || dataset.type === "multimodal") {
        const mockTabular: TabularSample[] = Array.from({ length: 20 }, (_, i) => ({
          id: `row-${i}`,
          patient_id: `P${1000 + i}`,
          age: 45 + Math.floor(Math.random() * 30),
          gender: i % 2 === 0 ? "Male" : "Female",
          diagnosis: i % 3 === 0 ? "Normal" : i % 3 === 1 ? "Pneumonia" : "COVID-19",
          temperature: (36.5 + Math.random() * 2).toFixed(1),
          heart_rate: 60 + Math.floor(Math.random() * 40),
        }));
        setTabularSamples(mockTabular);
      }

      // 模拟类别分布
      setClassDistribution({
        Normal: 2500,
        Abnormal: 2500,
      });
    } catch (error) {
      console.error("Failed to fetch preview data:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPreviewData();
  }, [dataset.id, currentPage, pageSize, selectedClass]);

  // 渲染图像预览
  const renderImagePreview = () => {
    if (imageSamples.length === 0) {
      return <Empty description="暂无图像数据" />;
    }

    return (
      <div>
        {/* 筛选器 */}
        <Space style={{ marginBottom: 16 }}>
          <span>筛选类别:</span>
          <Select
            style={{ width: 200 }}
            placeholder="全部类别"
            allowClear
            value={selectedClass}
            onChange={setSelectedClass}
          >
            {Object.keys(classDistribution).map((cls) => (
              <Select.Option key={cls} value={cls}>
                {cls} ({classDistribution[cls]})
              </Select.Option>
            ))}
          </Select>
        </Space>

        {/* 图像网格 */}
        <Image.PreviewGroup>
          <Row gutter={[16, 16]}>
            {imageSamples.map((sample) => (
              <Col key={sample.id} xs={12} sm={8} md={6} lg={4}>
                <Card
                  hoverable
                  cover={
                    <Image
                      src={sample.url}
                      alt={sample.label}
                      style={{ height: 150, objectFit: "cover" }}
                      preview={{
                        mask: (
                          <Space direction="vertical" align="center">
                            <EyeOutlined style={{ fontSize: 24 }} />
                            <span>查看</span>
                          </Space>
                        ),
                      }}
                    />
                  }
                  bodyStyle={{ padding: 8 }}
                >
                  <div style={{ textAlign: "center" }}>
                    <Tag color={sample.label === "Normal" ? "green" : "red"}>
                      {sample.label}
                    </Tag>
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
        </Image.PreviewGroup>

        {/* 分页 */}
        <div style={{ marginTop: 24, textAlign: "center" }}>
          <Pagination
            current={currentPage}
            pageSize={pageSize}
            total={dataset.samples}
            onChange={(page, size) => {
              setCurrentPage(page);
              setPageSize(size || 20);
            }}
            showSizeChanger
            showQuickJumper
            showTotal={(total) => `共 ${total} 个样本`}
          />
        </div>
      </div>
    );
  };

  // 渲染表格预览
  const renderTabularPreview = () => {
    if (tabularSamples.length === 0) {
      return <Empty description="暂无表格数据" />;
    }

    // 动态生成列
    const columns: ColumnsType<TabularSample> = Object.keys(tabularSamples[0] || {})
      .filter((key) => key !== "id")
      .map((key) => ({
        title: key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase()),
        dataIndex: key,
        key,
        width: 150,
        render: (value: any) => {
          // 特殊处理诊断列
          if (key === "diagnosis") {
            const colorMap: Record<string, string> = {
              Normal: "green",
              Pneumonia: "orange",
              "COVID-19": "red",
            };
            return <Tag color={colorMap[value] || "default"}>{value}</Tag>;
          }
          return value;
        },
      }));

    return (
      <div>
        <Table
          columns={columns}
          dataSource={tabularSamples}
          rowKey="id"
          pagination={{
            current: currentPage,
            pageSize: pageSize,
            total: dataset.samples,
            onChange: (page, size) => {
              setCurrentPage(page);
              setPageSize(size || 20);
            },
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 条记录`,
          }}
          scroll={{ x: 1000 }}
        />
      </div>
    );
  };

  // 渲染统计信息
  const renderStatistics = () => {
    return (
      <div>
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={8}>
            <Card>
              <Statistic
                title="总样本数"
                value={dataset.samples}
                prefix={<FileImageOutlined />}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card>
              <Statistic
                title="类别数"
                value={dataset.classes}
                prefix={<BarChartOutlined />}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card>
              <Statistic
                title="数据集大小"
                value={formatSize(dataset.size)}
                prefix={<TableOutlined />}
              />
            </Card>
          </Col>
        </Row>

        {/* 类别分布 */}
        <Card title="类别分布" style={{ marginBottom: 24 }}>
          <Space direction="vertical" style={{ width: "100%" }}>
            {Object.entries(classDistribution).map(([cls, count]) => (
              <div key={cls}>
                <div style={{ marginBottom: 8 }}>
                  <Space>
                    <Tag>{cls}</Tag>
                    <span>{count} 样本</span>
                    <span style={{ color: "#999" }}>
                      ({((count / dataset.samples) * 100).toFixed(1)}%)
                    </span>
                  </Space>
                </div>
                <div
                  style={{
                    height: 8,
                    background: "#f0f0f0",
                    borderRadius: 4,
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      width: `${(count / dataset.samples) * 100}%`,
                      height: "100%",
                      background: "#1890ff",
                    }}
                  />
                </div>
              </div>
            ))}
          </Space>
        </Card>

        {/* 数据集信息 */}
        <Card title="数据集信息">
          <Space direction="vertical" style={{ width: "100%" }}>
            <div>
              <strong>名称:</strong> {dataset.name}
            </div>
            <div>
              <strong>类型:</strong>{" "}
              <Tag>
                {dataset.type === "image"
                  ? "图像"
                  : dataset.type === "tabular"
                  ? "表格"
                  : "多模态"}
              </Tag>
            </div>
            {dataset.description && (
              <div>
                <strong>描述:</strong> {dataset.description}
              </div>
            )}
            {dataset.tags && dataset.tags.length > 0 && (
              <div>
                <strong>标签:</strong>{" "}
                <Space>
                  {dataset.tags.map((tag) => (
                    <Tag key={tag}>{tag}</Tag>
                  ))}
                </Space>
              </div>
            )}
            <div>
              <strong>创建时间:</strong> {new Date(dataset.created_at).toLocaleString("zh-CN")}
            </div>
          </Space>
        </Card>
      </div>
    );
  };

  // 格式化文件大小
  const formatSize = (bytes: number): string => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  };

  // 根据数据集类型渲染不同的标签页
  const renderTabs = () => {
    const items = [];

    // 统计信息标签页（始终显示）
    items.push({
      key: "statistics",
      label: (
        <span>
          <BarChartOutlined />
          统计信息
        </span>
      ),
      children: renderStatistics(),
    });

    // 图像预览标签页
    if (dataset.type === "image" || dataset.type === "multimodal") {
      items.push({
        key: "images",
        label: (
          <span>
            <FileImageOutlined />
            图像预览
          </span>
        ),
        children: renderImagePreview(),
      });
    }

    // 表格预览标签页
    if (dataset.type === "tabular" || dataset.type === "multimodal") {
      items.push({
        key: "tabular",
        label: (
          <span>
            <TableOutlined />
            表格数据
          </span>
        ),
        children: renderTabularPreview(),
      });
    }

    return items;
  };

  return (
    <Spin spinning={loading}>
      <Tabs defaultActiveKey="statistics" items={renderTabs()} />
    </Spin>
  );
};

export default DatasetPreview;
