import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Drawer,
  Form,
  Input,
  Modal,
  Popconfirm,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
  message,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import {
  AppstoreOutlined,
  DeleteOutlined,
  EyeOutlined,
  ExportOutlined,
  FolderOpenOutlined,
  PlusOutlined,
  RocketOutlined,
} from "@ant-design/icons";

import { getDatasets, type Dataset } from "@/api/datasets";
import {
  createProject,
  deleteProject,
  exportProjectBundle,
  getProject,
  getProjects,
  getProjectTemplates,
  type Project,
  type ProjectCreate,
  type ProjectTaskType,
  type ProjectTemplate,
} from "@/api/projects";

const { Paragraph, Text, Title } = Typography;

const TASK_LABELS: Record<ProjectTaskType, string> = {
  binary_classification: "二分类预测",
  cox_survival: "Cox 生存分析",
  multimodal_research: "多模态研究",
};

interface CreateProjectValues {
  name: string;
  description?: string;
  template_id: string;
  dataset_id?: number;
}

export default function ProjectWorkspace() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [form] = Form.useForm<CreateProjectValues>();
  const [projects, setProjects] = useState<Project[]>([]);
  const [templates, setTemplates] = useState<ProjectTemplate[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const selectedTemplateId = Form.useWatch("template_id", form);

  const selectedTemplate = useMemo(
    () => templates.find((item) => item.id === selectedTemplateId) || null,
    [templates, selectedTemplateId],
  );

  const loadAll = async (focusProjectId?: number) => {
    setLoading(true);
    try {
      const [projectData, templateData, datasetData] = await Promise.all([
        getProjects({ limit: 100 }),
        getProjectTemplates(),
        getDatasets({ limit: 200 }),
      ]);

      setProjects(projectData);
      setTemplates(templateData);
      setDatasets(datasetData || []);

      if (focusProjectId) {
        const detail = await getProject(focusProjectId);
        setSelectedProject(detail);
        setDrawerOpen(true);
      }
    } catch (error) {
      console.error("Failed to load project workspace:", error);
      message.error("加载项目工作区失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadAll();
  }, []);

  useEffect(() => {
    if (searchParams.get("action") !== "create") {
      return;
    }

    const templateId = searchParams.get("template");
    setCreateModalOpen(true);
    form.setFieldsValue({
      template_id: templateId || templates[0]?.id,
    });

    const next = new URLSearchParams(searchParams);
    next.delete("action");
    next.delete("template");
    setSearchParams(next, { replace: true });
  }, [form, searchParams, setSearchParams, templates]);

  const handleCreateProject = async () => {
    try {
      const values = await form.validateFields();
      const template = templates.find((item) => item.id === values.template_id);
      if (!template) {
        message.error("请选择有效模板");
        return;
      }

      setSubmitting(true);
      const payload: ProjectCreate = {
        name: values.name,
        description: values.description?.trim() || undefined,
        template_id: template.id,
        task_type: template.task_type,
        dataset_id: values.dataset_id,
        status: "draft",
        project_meta: {
          recommended_backbone: template.recommended_backbone,
          recommended_fusion: template.recommended_fusion,
        },
      };

      const created = await createProject(payload);
      message.success("项目已创建");
      setCreateModalOpen(false);
      form.resetFields();
      await loadAll(created.id);
    } catch (error: any) {
      if (error?.errorFields) {
        return;
      }
      console.error("Failed to create project:", error);
      message.error("创建项目失败");
    } finally {
      setSubmitting(false);
    }
  };

  const handleViewProject = async (projectId: number) => {
    try {
      const detail = await getProject(projectId);
      setSelectedProject(detail);
      setDrawerOpen(true);
    } catch (error) {
      console.error("Failed to load project detail:", error);
      message.error("加载项目详情失败");
    }
  };

  const handleDeleteProject = async (projectId: number) => {
    try {
      await deleteProject(projectId);
      if (selectedProject?.id === projectId) {
        setSelectedProject(null);
        setDrawerOpen(false);
      }
      await loadAll();
      message.success("项目已删除");
    } catch (error) {
      console.error("Failed to delete project:", error);
      message.error("删除项目失败");
    }
  };

  const handleExport = async (projectId: number) => {
    try {
      const exported = await exportProjectBundle(projectId);
      window.open(exported.download_url, "_blank", "noopener,noreferrer");
      message.success("已开始导出项目交付包");
    } catch (error: any) {
      console.error("Failed to export project bundle:", error);
      message.error(error?.response?.data?.detail || "导出项目交付包失败");
    }
  };

  const columns: ColumnsType<Project> = [
    {
      title: "项目名称",
      dataIndex: "name",
      key: "name",
      render: (_, record) => (
        <Space>
          <FolderOpenOutlined style={{ color: "#1677ff" }} />
          <a onClick={() => void handleViewProject(record.id)}>{record.name}</a>
        </Space>
      ),
    },
    {
      title: "任务类型",
      dataIndex: "task_type",
      key: "task_type",
      render: (taskType: ProjectTaskType) => (
        <Tag color="blue">{TASK_LABELS[taskType] || taskType}</Tag>
      ),
    },
    {
      title: "状态",
      dataIndex: "status",
      key: "status",
      render: (status: string) => <Tag color={status === "completed" ? "success" : "processing"}>{status}</Tag>,
    },
    {
      title: "数据集",
      dataIndex: "dataset_name",
      key: "dataset_name",
      render: (value: string | null | undefined) => value || "-",
    },
    {
      title: "训练记录",
      dataIndex: "job_count",
      key: "job_count",
    },
    {
      title: "结果记录",
      dataIndex: "model_count",
      key: "model_count",
    },
    {
      title: "更新时间",
      dataIndex: "updated_at",
      key: "updated_at",
      render: (value: string | null | undefined) =>
        value ? new Date(value).toLocaleString("zh-CN") : "-",
    },
    {
      title: "操作",
      key: "action",
      render: (_, record) => (
        <Space>
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => void handleViewProject(record.id)}
          >
            查看
          </Button>
          <Button
            size="small"
            icon={<ExportOutlined />}
            onClick={() => void handleExport(record.id)}
          >
            导出
          </Button>
          <Popconfirm
            title="确定删除这个项目吗？"
            onConfirm={() => void handleDeleteProject(record.id)}
          >
            <Button size="small" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  const totalProjects = projects.length;
  const completedProjects = projects.filter((item) => item.status === "completed").length;
  const runningProjects = projects.filter((item) => item.status === "running").length;

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" size={20} style={{ width: "100%" }}>
        <Card bordered={false}>
          <Row gutter={[16, 16]} align="middle">
            <Col xs={24} lg={16}>
              <Space direction="vertical" size={8}>
                <Title level={2} style={{ margin: 0 }}>
                  项目工作区
                </Title>
                <Paragraph style={{ margin: 0 }}>
                  Local Pro v1 以项目为中心，把数据、配置、训练、结果和导出收口到同一条本地工作流里。
                </Paragraph>
                <Space wrap>
                  <Button
                    type="primary"
                    icon={<PlusOutlined />}
                    onClick={() => setCreateModalOpen(true)}
                  >
                    新建项目
                  </Button>
                  <Button
                    icon={<AppstoreOutlined />}
                    onClick={() => navigate("/templates")}
                  >
                    查看模板库
                  </Button>
                </Space>
              </Space>
            </Col>
            <Col xs={24} lg={8}>
              <Alert
                type="info"
                showIcon
                message="当前是项目骨架版"
                description="项目已可创建、查看、导出，并能挂接训练/结果。下一步会继续把医生模式向导接入项目上下文。"
              />
            </Col>
          </Row>
        </Card>

        <Row gutter={[16, 16]}>
          <Col xs={24} md={8}>
            <Card loading={loading}>
              <Statistic title="项目总数" value={totalProjects} prefix={<FolderOpenOutlined />} />
            </Card>
          </Col>
          <Col xs={24} md={8}>
            <Card loading={loading}>
              <Statistic title="运行中项目" value={runningProjects} prefix={<RocketOutlined />} />
            </Card>
          </Col>
          <Col xs={24} md={8}>
            <Card loading={loading}>
              <Statistic title="已完成项目" value={completedProjects} />
            </Card>
          </Col>
        </Row>

        <Card
          title="模板入口"
          extra={
            <Button type="link" onClick={() => navigate("/templates")}>
              查看全部模板
            </Button>
          }
        >
          <Row gutter={[16, 16]}>
            {templates.map((template) => (
              <Col xs={24} lg={8} key={template.id}>
                <Card
                  size="small"
                  hoverable
                  onClick={() => {
                    form.setFieldsValue({ template_id: template.id });
                    setCreateModalOpen(true);
                  }}
                >
                  <Space direction="vertical" size={8} style={{ width: "100%" }}>
                    <Text strong>{template.name}</Text>
                    <Text type="secondary">{template.description}</Text>
                    <Space wrap>
                      <Tag color="blue">{TASK_LABELS[template.task_type]}</Tag>
                      <Tag>{template.recommended_backbone}</Tag>
                    </Space>
                  </Space>
                </Card>
              </Col>
            ))}
          </Row>
        </Card>

        <Card title="项目列表" loading={loading}>
          <Table
            rowKey="id"
            dataSource={projects}
            columns={columns}
            pagination={{ pageSize: 8 }}
          />
        </Card>
      </Space>

      <Modal
        title="新建项目"
        open={createModalOpen}
        onCancel={() => setCreateModalOpen(false)}
        onOk={() => void handleCreateProject()}
        confirmLoading={submitting}
        destroyOnHidden
      >
        <Form form={form} layout="vertical">
          <Form.Item
            label="项目名称"
            name="name"
            rules={[{ required: true, message: "请输入项目名称" }]}
          >
            <Input placeholder="例如：lung-survival-prototype" />
          </Form.Item>
          <Form.Item
            label="模板"
            name="template_id"
            rules={[{ required: true, message: "请选择模板" }]}
          >
            <Select
              options={templates.map((template) => ({
                label: template.name,
                value: template.id,
              }))}
              placeholder="选择一个专业版模板"
            />
          </Form.Item>
          <Form.Item label="关联数据集" name="dataset_id">
            <Select
              allowClear
              options={datasets.map((dataset) => ({
                label: dataset.name,
                value: dataset.id,
              }))}
              placeholder="可选，后续也可以再绑定"
            />
          </Form.Item>
          <Form.Item label="项目说明" name="description">
            <Input.TextArea rows={3} placeholder="描述这个项目的科研目标或数据背景" />
          </Form.Item>
          {selectedTemplate ? (
            <Alert
              type="info"
              showIcon
              message={selectedTemplate.name}
              description={
                <Space direction="vertical" size={4}>
                  <Text>{selectedTemplate.description}</Text>
                  <Text type="secondary">
                    必填字段：{selectedTemplate.required_fields.join(" / ")}
                  </Text>
                </Space>
              }
            />
          ) : null}
        </Form>
      </Modal>

      <Drawer
        title={selectedProject?.name || "项目详情"}
        open={drawerOpen}
        width={720}
        onClose={() => setDrawerOpen(false)}
      >
        {selectedProject ? (
          <Space direction="vertical" size={16} style={{ width: "100%" }}>
            <Descriptions bordered column={2} size="small">
              <Descriptions.Item label="任务类型">
                {TASK_LABELS[selectedProject.task_type]}
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                <Tag color={selectedProject.status === "completed" ? "success" : "processing"}>
                  {selectedProject.status}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="模板">{selectedProject.template_id}</Descriptions.Item>
              <Descriptions.Item label="数据集">
                {selectedProject.dataset_name || "-"}
              </Descriptions.Item>
              <Descriptions.Item label="配置路径" span={2}>
                {selectedProject.config_path || "-"}
              </Descriptions.Item>
              <Descriptions.Item label="输出目录" span={2}>
                {selectedProject.output_dir || "-"}
              </Descriptions.Item>
            </Descriptions>

            <Card size="small" title="最近运行">
              {selectedProject.latest_job ? (
                <Descriptions column={2} size="small">
                  <Descriptions.Item label="实验名">
                    {selectedProject.latest_job.experiment_name || "-"}
                  </Descriptions.Item>
                  <Descriptions.Item label="状态">
                    {selectedProject.latest_job.status}
                  </Descriptions.Item>
                  <Descriptions.Item label="进度">
                    {Math.round(selectedProject.latest_job.progress)}%
                  </Descriptions.Item>
                  <Descriptions.Item label="Epoch">
                    {selectedProject.latest_job.current_epoch}/{selectedProject.latest_job.total_epochs}
                  </Descriptions.Item>
                </Descriptions>
              ) : (
                <Text type="secondary">项目还没有训练记录。</Text>
              )}
            </Card>

            <Card size="small" title="结果与导出">
              <Space wrap>
                <Button onClick={() => navigate(`/config?projectId=${selectedProject.id}`)}>
                  打开项目向导
                </Button>
                <Button
                  onClick={() =>
                    navigate(
                      `/training?projectId=${selectedProject.id}&projectName=${encodeURIComponent(
                        selectedProject.name,
                      )}&taskType=${selectedProject.task_type}&action=start`,
                    )
                  }
                >
                  查看训练
                </Button>
                <Button onClick={() => navigate(`/models?projectId=${selectedProject.id}`)}>
                  查看结果
                </Button>
                <Button
                  type="primary"
                  icon={<ExportOutlined />}
                  onClick={() => void handleExport(selectedProject.id)}
                >
                  导出项目交付包
                </Button>
              </Space>
            </Card>
          </Space>
        ) : null}
      </Drawer>
    </div>
  );
}
