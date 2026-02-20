import React, { useState, useEffect } from "react";
import {
  Descriptions,
  Card,
  Space,
  Tag,
  Progress,
  Divider,
  List,
  Statistic,
  Row,
  Col,
  Alert,
  Timeline,
  Empty,
  Spin,
} from "antd";
import {
  CheckCircleOutlined,
  WarningOutlined,
  ClockCircleOutlined,
  FileImageOutlined,
  TableOutlined,
  DatabaseOutlined,
} from "@ant-design/icons";

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

interface DatasetStatisticsProps {
  dataset: Dataset;
}

interface QualityMetrics {
  completeness: number;
  balance: number;
  duplicates: number;
  missing_values: number;
}

interface ClassDistribution {
  class_name: string;
  count: number;
  percentage: number;
}

const DatasetStatistics: React.FC<DatasetStatisticsProps> = ({ dataset }) => {
  const [loading, setLoading] = useState(false);
  const [qualityMetrics, setQualityMetrics] = useState<QualityMetrics | null>(null);
  const [classDistribution, setClassDistribution] = useState<ClassDistribution[]>([]);

  // 获取统计数据
  const fetchStatistics = async () => {
    setLoading(true);
    try {
      // TODO: 调用后端 API
      // const response = await fetch(`/api/datasets/${dataset.id}/statistics`);
      // const data = await response.json();

      // 模拟数据
      setQualityMetrics({
        completeness: 98.5,
        balance: 85.0,
        duplicates: 12,
        missing_values: 75,
      });

      setClassDistribution([
        { class_name: "Normal", count: 2500, percentage: 50.0 },
        { class_name: "Abnormal", count: 2500, percentage: 50.0 },
      ]);
    } catch (error) {
      console.error("Failed to fetch statistics:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (dataset.status === "ready") {
      fetchStatistics();
    }
  }, [dataset.id, dataset.status]);

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
  const getStatusTag = () => {
    const statusConfig = {
      uploading: { color: "blue", icon: <ClockCircleOutlined />, text: "上传中" },
      processing: { color: "orange", icon: <ClockCircleOutlined />, text: "处理中" },
      ready: { color: "green", icon: <CheckCircleOutlined />, text: "就绪" },
      error: { color: "red", icon: <WarningOutlined />, text: "错误" },
    };

    const config = statusConfig[dataset.status];
    return (
      <Tag color={config.color} icon={config.icon}>
        {config.text}
      </Tag>
    );
  };

  // 获取类型图标和文本
  const getTypeInfo = () => {
    const typeConfig = {
      image: { icon: <FileImageOutlined />, text: "图像数据集", color: "#1890ff" },
      tabular: { icon: <TableOutlined />, text: "表格数据集", color: "#52c41a" },
      multimodal: { icon: <DatabaseOutlined />, text: "多模态数据集", color: "#722ed1" },
    };
    return typeConfig[dataset.type];
  };

  // 获取质量评分颜色
  const getQualityColor = (score: number): string => {
    if (score >= 90) return "#52c41a";
    if (score >= 70) return "#faad14";
    return "#f5222d";
  };

  // 渲染基本信息
  const renderBasicInfo = () => {
    const typeInfo = getTypeInfo();
    return (
      <Card title="基本信息" style={{ marginBottom: 16 }}>
        <Descriptions column={1} bordered size="small">
          <Descriptions.Item label="数据集名称">{dataset.name}</Descriptions.Item>
          <Descriptions.Item label="类型">
            <Space>
              {typeInfo.icon}
              <span style={{ color: typeInfo.color }}>{typeInfo.text}</span>
            </Space>
          </Descriptions.Item>
          <Descriptions.Item label="状态">{getStatusTag()}</Descriptions.Item>
          <Descriptions.Item label="样本数量">
            {dataset.samples.toLocaleString()}
          </Descriptions.Item>
          <Descriptions.Item label="类别数量">{dataset.classes}</Descriptions.Item>
          <Descriptions.Item label="数据集大小">{formatSize(dataset.size)}</Descriptions.Item>
          <Descriptions.Item label="创建时间">{formatDate(dataset.created_at)}</Descriptions.Item>
          <Descriptions.Item label="更新时间">{formatDate(dataset.updated_at)}</Descriptions.Item>
          {dataset.description && (
            <Descriptions.Item label="描述">{dataset.description}</Descriptions.Item>
          )}
          {dataset.tags && dataset.tags.length > 0 && (
            <Descriptions.Item label="标签">
              <Space>
                {dataset.tags.map((tag) => (
                  <Tag key={tag}>{tag}</Tag>
                ))}
              </Space>
            </Descriptions.Item>
          )}
        </Descriptions>
      </Card>
    );
  };

  // 渲染数据质量指标
  const renderQualityMetrics = () => {
    if (!qualityMetrics) return null;

    return (
      <Card title="数据质量" style={{ marginBottom: 16 }}>
        <Row gutter={16}>
          <Col span={12}>
            <Card bordered={false}>
              <Statistic
                title="完整性"
                value={qualityMetrics.completeness}
                precision={1}
                suffix="%"
                valueStyle={{ color: getQualityColor(qualityMetrics.completeness) }}
              />
              <Progress
                percent={qualityMetrics.completeness}
                strokeColor={getQualityColor(qualityMetrics.completeness)}
                showInfo={false}
              />
            </Card>
          </Col>
          <Col span={12}>
            <Card bordered={false}>
              <Statistic
                title="类别平衡性"
                value={qualityMetrics.balance}
                precision={1}
                suffix="%"
                valueStyle={{ color: getQualityColor(qualityMetrics.balance) }}
              />
              <Progress
                percent={qualityMetrics.balance}
                strokeColor={getQualityColor(qualityMetrics.balance)}
                showInfo={false}
              />
            </Card>
          </Col>
        </Row>

        <Divider />

        <Space direction="vertical" style={{ width: "100%" }}>
          <div>
            <Space>
              <WarningOutlined style={{ color: "#faad14" }} />
              <span>缺失值: {qualityMetrics.missing_values} 个</span>
            </Space>
          </div>
          <div>
            <Space>
              <WarningOutlined style={{ color: "#faad14" }} />
              <span>重复样本: {qualityMetrics.duplicates} 个</span>
            </Space>
          </div>
        </Space>

        {(qualityMetrics.missing_values > 0 || qualityMetrics.duplicates > 0) && (
          <Alert
            message="数据质量建议"
            description="检测到缺失值和重复样本，建议在训练前进行数据清洗。"
            type="warning"
            showIcon
            style={{ marginTop: 16 }}
          />
        )}
      </Card>
    );
  };

  // 渲染类别分布
  const renderClassDistribution = () => {
    if (classDistribution.length === 0) return null;

    return (
      <Card title="类别分布" style={{ marginBottom: 16 }}>
        <List
          dataSource={classDistribution}
          renderItem={(item) => (
            <List.Item>
              <div style={{ width: "100%" }}>
                <div style={{ marginBottom: 8 }}>
                  <Space style={{ width: "100%", justifyContent: "space-between" }}>
                    <Tag>{item.class_name}</Tag>
                    <Space>
                      <span>{item.count.toLocaleString()} 样本</span>
                      <span style={{ color: "#999" }}>({item.percentage.toFixed(1)}%)</span>
                    </Space>
                  </Space>
                </div>
                <Progress
                  percent={item.percentage}
                  strokeColor="#1890ff"
                  showInfo={false}
                />
              </div>
            </List.Item>
          )}
        />
      </Card>
    );
  };

  // 渲染处理历史
  const renderProcessingHistory = () => {
    const history = [
      {
        time: dataset.created_at,
        status: "success",
        title: "数据集创建",
        description: "数据集已成功创建",
      },
      {
        time: dataset.updated_at,
        status: "success",
        title: "数据处理完成",
        description: "数据预处理和验证已完成",
      },
    ];

    if (dataset.status === "error" && dataset.error_message) {
      history.push({
        time: dataset.updated_at,
        status: "error",
        title: "处理失败",
        description: dataset.error_message,
      });
    }

    return (
      <Card title="处理历史" style={{ marginBottom: 16 }}>
        <Timeline
          items={history.map((item) => ({
            color: item.status === "error" ? "red" : "green",
            children: (
              <div>
                <div style={{ fontWeight: 500 }}>{item.title}</div>
                <div style={{ color: "#999", fontSize: 12 }}>
                  {formatDate(item.time)}
                </div>
                <div style={{ marginTop: 4 }}>{item.description}</div>
              </div>
            ),
          }))}
        />
      </Card>
    );
  };

  // 渲染错误信息
  const renderErrorInfo = () => {
    if (dataset.status !== "error" || !dataset.error_message) return null;

    return (
      <Alert
        message="处理错误"
        description={dataset.error_message}
        type="error"
        showIcon
        style={{ marginBottom: 16 }}
      />
    );
  };

  // 渲染处理进度
  const renderProgress = () => {
    if (dataset.status !== "uploading" && dataset.status !== "processing") return null;

    return (
      <Card title="处理进度" style={{ marginBottom: 16 }}>
        <Progress
          percent={dataset.progress || 0}
          status="active"
          strokeColor={{
            "0%": "#108ee9",
            "100%": "#87d068",
          }}
        />
        <div style={{ marginTop: 8, color: "#999" }}>
          {dataset.status === "uploading" ? "正在上传文件..." : "正在处理数据..."}
        </div>
      </Card>
    );
  };

  return (
    <Spin spinning={loading}>
      <Space direction="vertical" style={{ width: "100%" }}>
        {renderErrorInfo()}
        {renderProgress()}
        {renderBasicInfo()}
        {dataset.status === "ready" && (
          <>
            {renderQualityMetrics()}
            {renderClassDistribution()}
          </>
        )}
        {renderProcessingHistory()}
      </Space>
    </Spin>
  );
};

export default DatasetStatistics;
