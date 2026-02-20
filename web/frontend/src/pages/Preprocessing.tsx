/**
 * 预处理管理页面
 *
 * 提供图像预处理功能，包括：
 * - 创建预处理任务
 * - 监控任务进度
 * - 查看任务历史
 * - 配置预处理参数
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Button,
  Table,
  Tag,
  Progress,
  Modal,
  Form,
  Input,
  InputNumber,
  Select,
  Switch,
  Space,
  message,
  Statistic,
  Row,
  Col,
  Descriptions,
  Tooltip,
} from 'antd';
import {
  PlayCircleOutlined,
  StopOutlined,
  DeleteOutlined,
  EyeOutlined,
  ReloadOutlined,
  PlusOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import {
  preprocessingAPI,
  PreprocessingTask,
  PreprocessingTaskCreate,
  PreprocessingStatistics,
  formatStatus,
  getStatusColor,
  formatNormalizeMethod,
  formatDuration,
  calculateSuccessRate,
} from '../api/preprocessing';

const { Option } = Select;

const Preprocessing: React.FC = () => {
  // 状态管理
  const [tasks, setTasks] = useState<PreprocessingTask[]>([]);
  const [statistics, setStatistics] = useState<PreprocessingStatistics | null>(null);
  const [loading, setLoading] = useState(false);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [selectedTask, setSelectedTask] = useState<PreprocessingTask | null>(null);
  const [form] = Form.useForm();

  // 加载任务列表
  const loadTasks = async () => {
    setLoading(true);
    try {
      const response = await preprocessingAPI.list({ limit: 100 });
      setTasks(response.data.tasks);
    } catch (error) {
      message.error('加载任务列表失败');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  // 加载统计信息
  const loadStatistics = async () => {
    try {
      const response = await preprocessingAPI.statistics();
      setStatistics(response.data);
    } catch (error) {
      console.error('加载统计信息失败:', error);
    }
  };

  // 初始加载
  useEffect(() => {
    loadTasks();
    loadStatistics();

    // 定时刷新
    const interval = setInterval(() => {
      loadTasks();
      loadStatistics();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // 创建任务
  const handleCreateTask = async (values: any) => {
    try {
      const taskData: PreprocessingTaskCreate = {
        name: values.name,
        description: values.description,
        input_dir: values.input_dir,
        output_dir: values.output_dir,
        config: {
          size: values.size,
          normalize: values.normalize,
          remove_artifacts: values.remove_artifacts,
          enhance_contrast: values.enhance_contrast,
        },
      };

      await preprocessingAPI.start(taskData);
      message.success('预处理任务已启动');
      setCreateModalVisible(false);
      form.resetFields();
      loadTasks();
    } catch (error: any) {
      message.error(error.response?.data?.detail || '启动任务失败');
      console.error(error);
    }
  };

  // 取消任务
  const handleCancelTask = async (taskId: string) => {
    try {
      await preprocessingAPI.cancel(taskId);
      message.success('任务已取消');
      loadTasks();
    } catch (error: any) {
      message.error(error.response?.data?.detail || '取消任务失败');
    }
  };

  // 删除任务
  const handleDeleteTask = async (taskId: number) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个任务吗？此操作不可恢复。',
      onOk: async () => {
        try {
          await preprocessingAPI.delete(taskId);
          message.success('任务已删除');
          loadTasks();
        } catch (error: any) {
          message.error(error.response?.data?.detail || '删除任务失败');
        }
      },
    });
  };

  // 查看任务详情
  const handleViewDetails = (task: PreprocessingTask) => {
    setSelectedTask(task);
    setDetailModalVisible(true);
  };

  // 表格列定义
  const columns: ColumnsType<PreprocessingTask> = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      width: 200,
      ellipsis: true,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>{formatStatus(status)}</Tag>
      ),
    },
    {
      title: '进度',
      key: 'progress',
      width: 200,
      render: (_, record) => (
        <div>
          <Progress
            percent={Math.round(record.progress * 100)}
            size="small"
            status={
              record.status === 'failed'
                ? 'exception'
                : record.status === 'completed'
                ? 'success'
                : 'active'
            }
          />
          <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
            {record.processed_images} / {record.total_images} 张
            {record.failed_images > 0 && (
              <span style={{ color: '#ff4d4f', marginLeft: '8px' }}>
                失败: {record.failed_images}
              </span>
            )}
          </div>
        </div>
      ),
    },
    {
      title: '配置',
      key: 'config',
      width: 150,
      render: (_, record) => (
        <div style={{ fontSize: '12px' }}>
          <div>大小: {record.config.size}px</div>
          <div>归一化: {formatNormalizeMethod(record.config.normalize)}</div>
        </div>
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 180,
      render: (time: string) => new Date(time).toLocaleString('zh-CN'),
    },
    {
      title: '耗时',
      dataIndex: 'duration',
      key: 'duration',
      width: 100,
      render: (duration?: number) => formatDuration(duration),
    },
    {
      title: '操作',
      key: 'actions',
      width: 150,
      fixed: 'right',
      render: (_, record) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button
              type="link"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handleViewDetails(record)}
            />
          </Tooltip>
          {(record.status === 'running' || record.status === 'pending') && (
            <Tooltip title="取消任务">
              <Button
                type="link"
                size="small"
                danger
                icon={<StopOutlined />}
                onClick={() => handleCancelTask(record.task_id)}
              />
            </Tooltip>
          )}
          {(record.status === 'completed' ||
            record.status === 'failed' ||
            record.status === 'cancelled') && (
            <Tooltip title="删除任务">
              <Button
                type="link"
                size="small"
                danger
                icon={<DeleteOutlined />}
                onClick={() => handleDeleteTask(record.id)}
              />
            </Tooltip>
          )}
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <h1>图像预处理</h1>

      {/* 统计信息 */}
      {statistics && (
        <Row gutter={16} style={{ marginBottom: '24px' }}>
          <Col span={6}>
            <Card>
              <Statistic title="总任务数" value={statistics.total_tasks} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="运行中"
                value={statistics.status_counts.running}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="已完成"
                value={statistics.status_counts.completed}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="已处理图像"
                value={statistics.total_processed_images}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* 任务列表 */}
      <Card
        title="预处理任务"
        extra={
          <Space>
            <Button icon={<ReloadOutlined />} onClick={loadTasks}>
              刷新
            </Button>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setCreateModalVisible(true)}
            >
              新建任务
            </Button>
          </Space>
        }
      >
        <Table
          columns={columns}
          dataSource={tasks}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showTotal: (total) => `共 ${total} 个任务`,
          }}
          scroll={{ x: 1200 }}
        />
      </Card>

      {/* 创建任务模态框 */}
      <Modal
        title="创建预处理任务"
        open={createModalVisible}
        onCancel={() => {
          setCreateModalVisible(false);
          form.resetFields();
        }}
        onOk={() => form.submit()}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateTask}
          initialValues={{
            size: 224,
            normalize: 'percentile',
            remove_artifacts: false,
            enhance_contrast: false,
          }}
        >
          <Form.Item
            label="任务名称"
            name="name"
            rules={[{ required: true, message: '请输入任务名称' }]}
          >
            <Input placeholder="例如：肺部CT预处理" />
          </Form.Item>

          <Form.Item label="任务描述" name="description">
            <Input.TextArea rows={2} placeholder="可选的任务描述" />
          </Form.Item>

          <Form.Item
            label="输入目录"
            name="input_dir"
            rules={[{ required: true, message: '请输入输入目录路径' }]}
          >
            <Input placeholder="/path/to/input/images" />
          </Form.Item>

          <Form.Item
            label="输出目录"
            name="output_dir"
            rules={[{ required: true, message: '请输入输出目录路径' }]}
          >
            <Input placeholder="/path/to/output/images" />
          </Form.Item>

          <Form.Item
            label="目标图像大小"
            name="size"
            rules={[{ required: true, message: '请输入图像大小' }]}
          >
            <InputNumber
              min={32}
              max={1024}
              style={{ width: '100%' }}
              addonAfter="px"
            />
          </Form.Item>

          <Form.Item
            label="归一化方法"
            name="normalize"
            rules={[{ required: true, message: '请选择归一化方法' }]}
          >
            <Select>
              <Option value="minmax">Min-Max 归一化</Option>
              <Option value="zscore">Z-Score 标准化</Option>
              <Option value="percentile">百分位归一化</Option>
              <Option value="none">不归一化</Option>
            </Select>
          </Form.Item>

          <Form.Item
            label="去除伪影"
            name="remove_artifacts"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>

          <Form.Item
            label="增强对比度 (CLAHE)"
            name="enhance_contrast"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>
        </Form>
      </Modal>

      {/* 任务详情模态框 */}
      <Modal
        title="任务详情"
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setDetailModalVisible(false)}>
            关闭
          </Button>,
        ]}
        width={800}
      >
        {selectedTask && (
          <div>
            <Descriptions bordered column={2}>
              <Descriptions.Item label="任务 ID" span={2}>
                {selectedTask.task_id}
              </Descriptions.Item>
              <Descriptions.Item label="任务名称" span={2}>
                {selectedTask.name}
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                <Tag color={getStatusColor(selectedTask.status)}>
                  {formatStatus(selectedTask.status)}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="进度">
                {Math.round(selectedTask.progress * 100)}%
              </Descriptions.Item>
              <Descriptions.Item label="输入目录" span={2}>
                {selectedTask.input_dir}
              </Descriptions.Item>
              <Descriptions.Item label="输出目录" span={2}>
                {selectedTask.output_dir}
              </Descriptions.Item>
              <Descriptions.Item label="总图像数">
                {selectedTask.total_images}
              </Descriptions.Item>
              <Descriptions.Item label="已处理">
                {selectedTask.processed_images}
              </Descriptions.Item>
              <Descriptions.Item label="失败数">
                <span style={{ color: selectedTask.failed_images > 0 ? '#ff4d4f' : undefined }}>
                  {selectedTask.failed_images}
                </span>
              </Descriptions.Item>
              <Descriptions.Item label="成功率">
                {calculateSuccessRate(selectedTask).toFixed(1)}%
              </Descriptions.Item>
              <Descriptions.Item label="图像大小">
                {selectedTask.config.size}px
              </Descriptions.Item>
              <Descriptions.Item label="归一化方法">
                {formatNormalizeMethod(selectedTask.config.normalize)}
              </Descriptions.Item>
              <Descriptions.Item label="去除伪影">
                {selectedTask.config.remove_artifacts ? '是' : '否'}
              </Descriptions.Item>
              <Descriptions.Item label="增强对比度">
                {selectedTask.config.enhance_contrast ? '是' : '否'}
              </Descriptions.Item>
              <Descriptions.Item label="创建时间" span={2}>
                {new Date(selectedTask.created_at).toLocaleString('zh-CN')}
              </Descriptions.Item>
              {selectedTask.started_at && (
                <Descriptions.Item label="开始时间" span={2}>
                  {new Date(selectedTask.started_at).toLocaleString('zh-CN')}
                </Descriptions.Item>
              )}
              {selectedTask.completed_at && (
                <Descriptions.Item label="完成时间" span={2}>
                  {new Date(selectedTask.completed_at).toLocaleString('zh-CN')}
                </Descriptions.Item>
              )}
              {selectedTask.duration && (
                <Descriptions.Item label="耗时" span={2}>
                  {formatDuration(selectedTask.duration)}
                </Descriptions.Item>
              )}
              {selectedTask.error && (
                <Descriptions.Item label="错误信息" span={2}>
                  <span style={{ color: '#ff4d4f' }}>{selectedTask.error}</span>
                </Descriptions.Item>
              )}
            </Descriptions>

            {selectedTask.description && (
              <div style={{ marginTop: '16px' }}>
                <h4>任务描述</h4>
                <p>{selectedTask.description}</p>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default Preprocessing;
