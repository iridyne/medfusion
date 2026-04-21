import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Alert,
  Button,
  Card,
  Col,
  Empty,
  Row,
  Select,
  Space,
  Tag,
  Typography,
  message,
} from "antd";
import {
  ApartmentOutlined,
  ArrowLeftOutlined,
  ControlOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  PlusOutlined,
} from "@ant-design/icons";

import { getModelCatalog, type ModelCatalogResponse } from "@/api/models";
import PageScaffold from "@/components/layout/PageScaffold";

const { Paragraph, Text } = Typography;

const STATUS_COLORS: Record<string, string> = {
  compile_ready: "success",
  conditional: "warning",
  draft_only: "default",
};

export default function ModelCatalog() {
  const navigate = useNavigate();
  const [catalog, setCatalog] = useState<ModelCatalogResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [familyFilter, setFamilyFilter] = useState<string>("all");

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const payload = await getModelCatalog();
        setCatalog(payload);
      } catch (error) {
        console.error("Failed to load model catalog:", error);
        message.error("加载模型数据库失败");
      } finally {
        setLoading(false);
      }
    };

    void load();
  }, []);

  const families = useMemo(() => {
    if (!catalog) {
      return [];
    }
    return Array.from(new Set(catalog.components.map((item) => item.family)));
  }, [catalog]);

  const filteredComponents = useMemo(() => {
    if (!catalog) {
      return [];
    }
    if (familyFilter === "all") {
      return catalog.components;
    }
    return catalog.components.filter((item) => item.family === familyFilter);
  }, [catalog, familyFilter]);

  const openWizardWithTemplate = (templateId: string, template: Record<string, any>) => {
    navigate("/config", {
      state: {
        modelPrefill: {
          modelTemplateId: templateId,
          ...template,
        },
        source: "model-catalog-official",
      },
    });
  };

  return (
    <PageScaffold
      eyebrow="Model database"
      title="模型数据库：只暴露可连接的打包黑盒，不暴露神经网络底层零件"
      description="这里记录当前正式版主线允许用户搭建的模型组件和模板。每个条目都要讲清楚：它处理什么数据、要求哪些配置、算力大概在哪个档位，以及它在整个模型图里应该接在哪里。"
      chips={[
        { label: "Packaged black boxes", tone: "amber" },
        { label: "Model composition", tone: "teal" },
        { label: "Runtime-backed catalog", tone: "blue" },
      ]}
      actions={
        <>
          <Button icon={<ArrowLeftOutlined />} onClick={() => navigate("/config")}>
            回到 Run Wizard
          </Button>
          <Button icon={<ApartmentOutlined />} onClick={() => navigate("/config/advanced")}>
            看高级模式注册表
          </Button>
          <Button type="primary" icon={<ControlOutlined />} onClick={() => navigate("/config")}>
            去模型配置主线
          </Button>
        </>
      }
      aside={
        <div className="hero-aside-panel">
          <span className="hero-aside-panel__label">Catalog posture</span>
          <div className="hero-aside-panel__value">不从 pooling / FC 这种底层开始搭</div>
          <div className="hero-aside-panel__copy">
            用户在这里看到的应该是已经封装好的模型单元，例如“CT 特征提取包”“表格编码包”“融合包”“分类头包”，而不是神经网络最底层零件。
          </div>
          <div className="surface-note">
            当前条目：
            <strong> {catalog?.components.length ?? 0} 个组件 / {catalog?.templates.length ?? 0} 个模板</strong>
          </div>
        </div>
      }
      metrics={[
        {
          label: "Packaged modules",
          value: catalog?.components.length ?? 0,
          hint: "黑盒组件总数",
          tone: "amber",
        },
        {
          label: "Ready templates",
          value: catalog?.templates.filter((item) => item.status === "compile_ready").length ?? 0,
          hint: "可直接进入主线的模型模板",
          tone: "teal",
        },
        {
          label: "Conditional items",
          value: catalog?.components.filter((item) => item.status === "conditional").length ?? 0,
          hint: "有额外条件的组件",
          tone: "rose",
        },
        {
          label: "Draft-only items",
          value: catalog?.components.filter((item) => item.status === "draft_only").length ?? 0,
          hint: "暂不进入默认主线的专项能力",
          tone: "blue",
        },
      ]}
    >
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="模型数据库的边界"
        description="这页记录的是“可以拿来连接的组件”和“已经打包好的模型模板”。它不是论文式模块大全，也不是低层神经网络编辑器。"
      />

      {catalog?.sources ? (
        <Card className="surface-card" style={{ marginBottom: 16 }}>
          <div className="section-heading">
            <div>
              <div className="section-heading__eyebrow">Model sources</div>
              <h2 className="section-heading__title">先启用官方来源，同时预留用户自定义入口</h2>
              <p className="section-heading__description">
                当前阶段只允许官方打包组件和模板进入模型数据库；用户自定义入口已经保留，但暂不开放写入。
              </p>
            </div>
          </div>

          <Row gutter={[16, 16]}>
            <Col xs={24} xl={12}>
              <Card size="small">
                <Space direction="vertical" size={10} style={{ width: "100%" }}>
                  <Space wrap>
                    <Text strong>{catalog.sources.official.label}</Text>
                    <Tag color={catalog.sources.official.enabled ? "success" : "default"}>
                      {catalog.sources.official.enabled ? "enabled" : "disabled"}
                    </Tag>
                  </Space>
                  <Paragraph style={{ marginBottom: 0 }}>
                    {catalog.sources.official.description}
                  </Paragraph>
                  <Button
                    type="primary"
                    icon={<ControlOutlined />}
                    onClick={() => navigate(catalog.sources.official.entry_path)}
                  >
                    当前正在使用官方来源
                  </Button>
                </Space>
              </Card>
            </Col>
            <Col xs={24} xl={12}>
              <Card size="small">
                <Space direction="vertical" size={10} style={{ width: "100%" }}>
                  <Space wrap>
                    <Text strong>{catalog.sources.custom.label}</Text>
                    <Tag color={catalog.sources.custom.enabled ? "success" : "warning"}>
                      {catalog.sources.custom.enabled ? "enabled" : "reserved"}
                    </Tag>
                  </Space>
                  <Paragraph style={{ marginBottom: 0 }}>
                    {catalog.sources.custom.description}
                  </Paragraph>
                  <Button
                    icon={<PlusOutlined />}
                    onClick={() => navigate(catalog.sources.custom.entry_path)}
                  >
                    查看预留入口
                  </Button>
                </Space>
              </Card>
            </Col>
          </Row>
        </Card>
      ) : null}

      {catalog?.principles?.length ? (
        <Card className="surface-card" style={{ marginBottom: 16 }}>
          <div className="section-heading">
            <div>
              <div className="section-heading__eyebrow">Catalog principles</div>
              <h2 className="section-heading__title">为什么这里不暴露更底层部件</h2>
              <p className="section-heading__description">
                先固定可连接的上层单元，才能让模型搭建过程既可解释、也可落回现有 runtime。
              </p>
            </div>
          </div>

          <div className="editorial-grid">
            {catalog.principles.map((item) => (
              <div key={item} className="surface-note surface-note--dense">
                <strong>{item}</strong>
              </div>
            ))}
          </div>
        </Card>
      ) : null}

      <Card className="surface-card" style={{ marginBottom: 16 }}>
        <div className="section-heading">
          <div>
            <div className="section-heading__eyebrow">Model templates</div>
            <h2 className="section-heading__title">当前可拿来直接起步的模型模板</h2>
            <p className="section-heading__description">
              模板先给你一组能直接进入主线的打包模型，再决定要不要进一步拆回组件层。
            </p>
          </div>
          <Tag color="processing">Template-first</Tag>
        </div>

        {catalog?.templates?.length ? (
          <Row gutter={[16, 16]}>
            {catalog.templates.map((template) => (
              <Col xs={24} xl={8} key={template.id}>
                <Card size="small">
                  <Space direction="vertical" size={10} style={{ width: "100%" }}>
                    <Space wrap>
                      <Text strong>{template.label}</Text>
                      <Tag color={STATUS_COLORS[template.status] || "default"}>
                        {template.status}
                      </Tag>
                    </Space>
                    <Paragraph style={{ marginBottom: 0 }}>{template.description}</Paragraph>
                    <div className="surface-note surface-note--dense">
                      <strong>数据要求</strong>
                      <p>{template.data_requirements.join(" / ")}</p>
                    </div>
                    <div className="surface-note surface-note--dense">
                      <strong>算力需求</strong>
                      <p>
                        {template.compute_profile.gpu_vram_hint} · {template.compute_profile.notes}
                      </p>
                    </div>
                    <div className="surface-note surface-note--dense">
                      <strong>组件链</strong>
                      <p>{template.component_ids.join(" -> ")}</p>
                    </div>
                    <Space wrap>
                      {template.wizard_prefill ? (
                        <Button
                          type="primary"
                          icon={<ControlOutlined />}
                          onClick={() => openWizardWithTemplate(template.id, template.wizard_prefill!)}
                        >
                          带入 Run Wizard
                        </Button>
                      ) : null}
                      {template.advanced_builder_blueprint_id ? (
                        <Button
                          icon={<ApartmentOutlined />}
                          onClick={() =>
                            navigate(
                              `/config/advanced/canvas?blueprint=${template.advanced_builder_blueprint_id}`,
                            )
                          }
                        >
                          去高级模式画布
                        </Button>
                      ) : null}
                    </Space>
                  </Space>
                </Card>
              </Col>
            ))}
          </Row>
        ) : (
          <Empty description="暂无模板" />
        )}
      </Card>

      <Card className="surface-card">
        <div className="section-heading">
          <div>
            <div className="section-heading__eyebrow">Packaged modules</div>
            <h2 className="section-heading__title">当前允许用户拿来连接的黑盒组件</h2>
            <p className="section-heading__description">
              模型不是一个大黑盒，但也不从 pooling / FC 这种底层开始拼。这里列出的是已经封装好的上层单元。
            </p>
          </div>
          <Select
            value={familyFilter}
            style={{ width: 220 }}
            onChange={setFamilyFilter}
            options={[
              { label: "全部组件", value: "all" },
              ...families.map((item) => ({ label: item, value: item })),
            ]}
          />
        </div>

        {filteredComponents.length ? (
          <Row gutter={[16, 16]}>
            {filteredComponents.map((component) => (
              <Col xs={24} lg={12} key={component.id}>
                <Card size="small">
                  <Space direction="vertical" size={10} style={{ width: "100%" }}>
                    <Space wrap>
                      {component.family === "data_bundle" ? (
                        <DatabaseOutlined />
                      ) : component.family === "vision_encoder" ? (
                        <ExperimentOutlined />
                      ) : (
                        <ControlOutlined />
                      )}
                      <Text strong>{component.label}</Text>
                      <Tag color={STATUS_COLORS[component.status] || "default"}>
                        {component.status}
                      </Tag>
                    </Space>
                    <Paragraph style={{ marginBottom: 0 }}>{component.description}</Paragraph>
                    <div className="surface-note surface-note--dense">
                      <strong>数据要求</strong>
                      <p>{component.data_requirements.join(" / ")}</p>
                    </div>
                    <div className="surface-note surface-note--dense">
                      <strong>配置要求</strong>
                      <p>{component.config_requirements.join(" / ")}</p>
                    </div>
                    <div className="surface-note surface-note--dense">
                      <strong>算力需求</strong>
                      <p>
                        {component.compute_profile.tier} · {component.compute_profile.gpu_vram_hint}
                        {" · "}
                        {component.compute_profile.notes}
                      </p>
                    </div>
                    <div className="surface-note surface-note--dense">
                      <strong>上下游</strong>
                      <p>
                        in: {component.upstream.join(" / ") || "-"} | out:{" "}
                        {component.outputs.join(" / ") || "-"}
                      </p>
                    </div>
                    <Space wrap>
                      {component.wizard_prefill ? (
                        <Button
                          icon={<ControlOutlined />}
                          onClick={() => openWizardWithTemplate(component.id, component.wizard_prefill!)}
                        >
                          带入 Run Wizard
                        </Button>
                      ) : null}
                      {component.advanced_builder_component_id ? (
                        <Button icon={<ApartmentOutlined />} onClick={() => navigate("/config/advanced")}>
                          看高级模式映射
                        </Button>
                      ) : null}
                    </Space>
                  </Space>
                </Card>
              </Col>
            ))}
          </Row>
        ) : (
          <Empty description="当前筛选下没有组件" />
        )}
      </Card>
    </PageScaffold>
  );
}
