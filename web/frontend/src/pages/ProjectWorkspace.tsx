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
  Progress,
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
  ControlOutlined,
} from "@ant-design/icons";

import { getDatasets, type Dataset } from "@/api/datasets";
import {
  createProject,
  deleteProject,
  exportProjectBundle,
  getProject,
  getProjects,
  getProjectTemplates,
  updateProject,
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

type ProjectStageStatus = "pending" | "active" | "completed" | "blocked";

interface ProjectStage {
  key: string;
  title: string;
  description: string;
  status: ProjectStageStatus;
}

interface ExportArtifactSummary {
  label: string;
  ready: boolean;
}

function getStageStatusLabel(status: ProjectStageStatus): string {
  switch (status) {
    case "completed":
      return "已完成";
    case "active":
      return "进行中";
    case "blocked":
      return "受阻";
    default:
      return "待开始";
  }
}

interface CreateProjectValues {
  name: string;
  description?: string;
  template_id: string;
  dataset_id?: number;
}

function getStageColor(status: ProjectStageStatus): string {
  switch (status) {
    case "completed":
      return "#52c41a";
    case "active":
      return "#1677ff";
    case "blocked":
      return "#faad14";
    default:
      return "#d9d9d9";
  }
}

function getStatusTagColor(status: string): string {
  switch (status) {
    case "completed":
      return "success";
    case "running":
    case "active":
      return "processing";
    case "paused":
    case "blocked":
      return "warning";
    case "pending":
    case "draft":
      return "default";
    case "failed":
      return "error";
    default:
      return "default";
  }
}

function getProjectStages(project: Project): ProjectStage[] {
  const hasDataset = Boolean(project.dataset_id || project.dataset_name);
  const hasConfig = Boolean(project.config_path);
  const latestJob = project.latest_job;
  const hasTraining = Boolean(latestJob);
  const hasResults = Boolean(project.latest_model_id || project.latest_model);

  return [
    {
      key: "dataset",
      title: "数据准备",
      description: hasDataset ? "已绑定数据集" : "需要先选择项目数据集",
      status: hasDataset ? "completed" : "active",
    },
    {
      key: "config",
      title: "配置生成",
      description: hasConfig ? "已生成配置" : "建议先进入项目向导生成配置",
      status: hasConfig ? "completed" : hasDataset ? "active" : "pending",
    },
    {
      key: "training",
      title: "训练执行",
      description: hasTraining
        ? `最近状态：${latestJob?.status || "unknown"}`
        : "尚未启动训练任务",
      status: hasTraining
        ? latestJob?.status === "completed"
          ? "completed"
          : latestJob?.status === "failed"
            ? "blocked"
            : "active"
        : hasConfig
          ? "active"
          : "pending",
    },
    {
      key: "results",
      title: "结果导出",
      description: hasResults ? "已有结果可查看和导出" : "训练完成后可生成结果包",
      status: hasResults ? "completed" : hasTraining ? "active" : "pending",
    },
  ];
}

function getProjectNextAction(project: Project): {
  label: string;
  description: string;
  route: string;
} {
  if (!project.dataset_id && !project.dataset_name) {
    return {
      label: "绑定数据集",
      description: "先去数据管理准备或登记项目数据集。",
      route: "/datasets",
    };
  }

  if (!project.config_path) {
    return {
      label: "生成项目配置",
      description: "进入项目向导，按模板生成真实训练配置。",
      route: `/config?projectId=${project.id}&projectName=${encodeURIComponent(
        project.name,
      )}&taskType=${project.task_type}&template=${project.template_id}`,
    };
  }

  if (!project.latest_job) {
    return {
      label: "启动训练",
      description: "配置已就绪，建议进入训练监控发起项目训练。",
      route: `/training?projectId=${project.id}&projectName=${encodeURIComponent(
        project.name,
      )}&taskType=${project.task_type}&action=start`,
    };
  }

  if (!project.latest_model) {
    return {
      label: "查看训练进度",
      description: "项目已启动训练，先关注训练状态和历史曲线。",
      route: `/training?projectId=${project.id}&projectName=${encodeURIComponent(
        project.name,
      )}&taskType=${project.task_type}`,
    };
  }

  return {
    label: "查看项目结果",
    description: "结果已经生成，可以查看 artifact 并导出项目交付包。",
    route: `/models?projectId=${project.id}`,
  };
}

function getProjectStageProgress(project: Project): number {
  const stages = getProjectStages(project);
  const completed = stages.filter((stage) => stage.status === "completed").length;
  return Math.round((completed / stages.length) * 100);
}

function getProjectExportArtifacts(project: Project): ExportArtifactSummary[] {
  return [
    {
      label: "训练配置 YAML",
      ready: Boolean(project.config_path),
    },
    {
      label: "训练输出目录",
      ready: Boolean(project.output_dir),
    },
    {
      label: "训练任务记录",
      ready: Boolean(project.latest_job),
    },
    {
      label: "模型结果",
      ready: Boolean(project.latest_model),
    },
    {
      label: "项目交付包",
      ready: Boolean(project.output_dir || project.config_path),
    },
  ];
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
  const [updatingProject, setUpdatingProject] = useState(false);

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
      await loadAll();
      navigate(
        `/config?projectId=${created.id}&projectName=${encodeURIComponent(
          created.name,
        )}&taskType=${created.task_type}&template=${created.template_id}`,
      );
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

  const handleBindDataset = async (projectId: number, datasetId?: number) => {
    try {
      setUpdatingProject(true);
      await updateProject(projectId, {
        dataset_id: datasetId,
      });
      await loadAll(projectId);
      message.success(datasetId ? "项目已绑定数据集" : "项目数据集已清空");
    } catch (error) {
      console.error("Failed to update project dataset:", error);
      message.error("更新项目数据集失败");
    } finally {
      setUpdatingProject(false);
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
      render: (status: string) => <Tag color={getStatusTagColor(status)}>{status}</Tag>,
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
      title: "下一步",
      key: "next",
      render: (_, record) => {
        const nextAction = getProjectNextAction(record);
        return (
          <Button size="small" type="link" onClick={() => navigate(nextAction.route)}>
            {nextAction.label}
          </Button>
        );
      },
    },
    {
      title: "操作",
      key: "action",
      render: (_, record) => (
        <Space>
          <Button
            size="small"
            icon={<ControlOutlined />}
            onClick={() =>
              navigate(
                `/config?projectId=${record.id}&projectName=${encodeURIComponent(
                  record.name,
                )}&taskType=${record.task_type}&template=${record.template_id}`,
              )
            }
          >
            向导
          </Button>
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
  const focusProject = selectedProject ?? projects[0] ?? null;
  const focusNextAction = focusProject ? getProjectNextAction(focusProject) : null;

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

        {focusProject && focusNextAction ? (
          <Card
            title={`当前焦点项目：${focusProject.name}`}
            extra={<Tag color="blue">{TASK_LABELS[focusProject.task_type]}</Tag>}
          >
            <Row gutter={[16, 16]} align="middle">
              <Col xs={24} lg={14}>
                <Space direction="vertical" size={8} style={{ width: "100%" }}>
                  <Text>{focusNextAction.description}</Text>
                  <Progress
                    percent={getProjectStageProgress(focusProject)}
                    status={focusProject.status === "failed" ? "exception" : "active"}
                  />
                  <Space wrap>
                    <Tag color={getStatusTagColor(focusProject.status)}>
                      {focusProject.status}
                    </Tag>
                    <Tag>{focusProject.template_id}</Tag>
                    {focusProject.dataset_name ? (
                      <Tag color="green">{focusProject.dataset_name}</Tag>
                    ) : null}
                  </Space>
                </Space>
              </Col>
              <Col xs={24} lg={10}>
                <Space wrap>
                  <Button
                    type="primary"
                    icon={<RocketOutlined />}
                    onClick={() => navigate(focusNextAction.route)}
                  >
                    {focusNextAction.label}
                  </Button>
                  <Button onClick={() => void handleViewProject(focusProject.id)}>
                    查看详情
                  </Button>
                  {focusProject.dataset_id || focusProject.dataset_name ? null : (
                    <Button onClick={() => void handleViewProject(focusProject.id)}>
                      绑定数据集
                    </Button>
                  )}
                </Space>
              </Col>
            </Row>
          </Card>
        ) : null}

        {focusProject ? (
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="最近结果摘要">
                {focusProject.latest_model ? (
                  <Space direction="vertical" size={10} style={{ width: "100%" }}>
                    <Descriptions size="small" column={1}>
                      <Descriptions.Item label="模型名称">
                        {focusProject.latest_model.name}
                      </Descriptions.Item>
                      <Descriptions.Item label="架构">
                        {focusProject.latest_model.architecture || "-"}
                      </Descriptions.Item>
                      <Descriptions.Item label="Accuracy">
                        {focusProject.latest_model.accuracy !== null &&
                        focusProject.latest_model.accuracy !== undefined
                          ? `${(focusProject.latest_model.accuracy * 100).toFixed(2)}%`
                          : "-"}
                      </Descriptions.Item>
                      <Descriptions.Item label="Loss">
                        {focusProject.latest_model.loss !== null &&
                        focusProject.latest_model.loss !== undefined
                          ? focusProject.latest_model.loss.toFixed(4)
                          : "-"}
                      </Descriptions.Item>
                    </Descriptions>
                    <Space wrap>
                      <Button onClick={() => navigate(`/models?projectId=${focusProject.id}`)}>
                        打开结果页
                      </Button>
                      <Button
                        type="primary"
                        icon={<ExportOutlined />}
                        onClick={() => void handleExport(focusProject.id)}
                      >
                        导出项目包
                      </Button>
                    </Space>
                  </Space>
                ) : (
                  <Text type="secondary">项目还没有沉淀下来的模型结果。</Text>
                )}
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="导出内容概览">
                <Row gutter={[12, 12]}>
                  {getProjectExportArtifacts(focusProject).map((item) => (
                    <Col xs={24} md={12} key={item.label}>
                      <Card
                        size="small"
                        bodyStyle={{ padding: 12 }}
                        style={{
                          borderColor: item.ready ? "#b7eb8f" : "#f0f0f0",
                          background: item.ready ? "#f6ffed" : undefined,
                        }}
                      >
                        <Space direction="vertical" size={6}>
                          <Text strong>{item.label}</Text>
                          <Tag color={item.ready ? "success" : "default"}>
                            {item.ready ? "已就绪" : "未生成"}
                          </Tag>
                        </Space>
                      </Card>
                    </Col>
                  ))}
                </Row>
              </Card>
            </Col>
          </Row>
        ) : null}

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

            <Card size="small" title="项目阶段">
              <Row gutter={[12, 12]}>
                {getProjectStages(selectedProject).map((stage) => (
                  <Col xs={24} md={12} key={stage.key}>
                    <Card
                      size="small"
                      bodyStyle={{ padding: 12 }}
                      style={{
                        borderColor: getStageColor(stage.status),
                        boxShadow:
                          stage.status === "active"
                            ? "0 0 0 2px rgba(22,119,255,0.08)"
                            : undefined,
                      }}
                    >
                      <Space direction="vertical" size={6} style={{ width: "100%" }}>
                        <Space>
                          <Text strong>{stage.title}</Text>
                          <Tag color={getStatusTagColor(stage.status === "blocked" ? "failed" : stage.status)}>
                            {getStageStatusLabel(stage.status)}
                          </Tag>
                        </Space>
                        <Text type="secondary">{stage.description}</Text>
                      </Space>
                    </Card>
                  </Col>
                ))}
              </Row>
            </Card>

            <Card size="small" title="下一步建议">
              <Space direction="vertical" size={8} style={{ width: "100%" }}>
                <Text>{getProjectNextAction(selectedProject).description}</Text>
                <Button
                  type="primary"
                  onClick={() => navigate(getProjectNextAction(selectedProject).route)}
                >
                  {getProjectNextAction(selectedProject).label}
                </Button>
              </Space>
            </Card>

            <Card size="small" title="项目数据绑定">
              <Space direction="vertical" size={12} style={{ width: "100%" }}>
                <Text type="secondary">
                  如果项目还没有绑定数据集，可以直接在这里补上，不需要回到数据管理页。
                </Text>
                <Select
                  allowClear
                  showSearch
                  optionFilterProp="label"
                  value={selectedProject.dataset_id ?? undefined}
                  loading={updatingProject}
                  placeholder="选择一个已登记数据集"
                  options={datasets.map((dataset) => ({
                    label: dataset.name,
                    value: dataset.id,
                  }))}
                  onChange={(value) =>
                    void handleBindDataset(
                      selectedProject.id,
                      value ? Number(value) : undefined,
                    )
                  }
                />
              </Space>
            </Card>

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
                <Button
                  onClick={() =>
                    navigate(
                      `/config?projectId=${selectedProject.id}&projectName=${encodeURIComponent(
                        selectedProject.name,
                      )}&taskType=${selectedProject.task_type}&template=${selectedProject.template_id}`,
                    )
                  }
                >
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
