import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Alert, Button, Card, Col, Empty, Row, Space, Table, Tag, Typography, message } from "antd";
import {
  ApartmentOutlined,
  ArrowLeftOutlined,
  ControlOutlined,
  ExperimentOutlined,
} from "@ant-design/icons";

import PageScaffold from "@/components/layout/PageScaffold";
import { getAdvancedBuilderCatalog, type AdvancedBuilderCatalogResponse } from "@/api/advancedBuilder";
import {
  type AdvancedBuilderFamily,
  type AdvancedBuilderStatus,
} from "@/config/advancedBuilderCatalog";

const { Paragraph, Text } = Typography;

const STATUS_COLORS: Record<AdvancedBuilderStatus, string> = {
  compile_ready: "success",
  conditional: "warning",
  draft_only: "default",
};

export default function AdvancedBuilder() {
  const navigate = useNavigate();
  const [catalog, setCatalog] = useState<AdvancedBuilderCatalogResponse | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const payload = await getAdvancedBuilderCatalog();
        setCatalog(payload);
      } catch (error) {
        console.error("Failed to load advanced builder catalog:", error);
        message.error("加载高级模式目录失败");
      }
    };

    void load();
  }, []);

  const familyEntries = useMemo(() => {
    return Object.entries(catalog?.familyLabels || {}) as Array<
      [AdvancedBuilderFamily, string]
    >;
  }, [catalog]);

  const compileReadyBlueprints = (catalog?.blueprints || []).filter(
    (item) => item.status === "compile_ready",
  );
  const draftBlueprints = (catalog?.blueprints || []).filter(
    (item) => item.status === "draft_only",
  );

  return (
    <PageScaffold
      eyebrow="Advanced mode"
      title="高级建模模式：先看组件注册表，再谈节点编辑"
      description="这页先把正式版高级模式需要的前置条件讲清楚：当前支持哪些组件家族、哪些连接合法、哪些骨架能编译、哪些组合还只是草稿。它不是旧 workflow editor 的翻版，而是正式版高级模式的约束面。"
      chips={[
        { label: "Advanced mode", tone: "amber" },
        { label: "Registry-backed", tone: "teal" },
        { label: "Compile-ready boundaries", tone: "blue" },
      ]}
      actions={
        <>
          <Button
            icon={<ArrowLeftOutlined />}
            onClick={() => navigate("/config")}
          >
            返回问题向导
          </Button>
          <Button
            icon={<ApartmentOutlined />}
            onClick={() => navigate("/config/advanced/canvas")}
          >
            打开节点图原型
          </Button>
          <Button
            icon={<ControlOutlined />}
            onClick={() => navigate("/training")}
          >
            去训练执行
          </Button>
        </>
      }
      aside={
        <div className="hero-aside-panel">
          <span className="hero-aside-panel__label">Current posture</span>
          <div className="hero-aside-panel__value">高级模式不是默认入口</div>
          <div className="hero-aside-panel__copy">
            当前阶段先把组件、连接和编译边界收紧，再逐步进入真正的节点式编辑。默认用户仍然优先走问题向导和参数编辑层。
          </div>
          <div className="surface-note">
            当前定位：
            <strong> registry + constraints + blueprint preview</strong>
          </div>
        </div>
      }
      metrics={[
        {
          label: "Component families",
          value: familyEntries.length,
          hint: "Input / backbone / fusion / head / training",
          tone: "amber",
        },
        {
          label: "Compile-ready components",
          value: (catalog?.components || []).filter(
            (item) => item.status === "compile_ready",
          ).length,
          hint: "Safe to compile into the current formal release path",
          tone: "teal",
        },
        {
          label: "Compile-ready blueprints",
          value: compileReadyBlueprints.length,
          hint: "Current formal release advanced-mode starting points",
          tone: "blue",
        },
        {
          label: "Blocked rules",
          value: (catalog?.connectionRules || []).filter(
            (rule) => rule.status === "blocked",
          ).length,
          hint: "Human-readable rules that prevent fake builder freedom",
          tone: "rose",
        },
      ]}
    >
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="为什么先做注册表和约束，而不是先给空白画布"
        description="因为正式版高级模式不能只是“能画图”。在进入节点编辑之前，必须先说明哪些组件真的存在、哪些连接真的合法、哪些骨架真的能编译回 runtime 主链。否则前台只会把用户带到一个仓库实际上跑不起来的假交互里。"
      />
      <Alert
        type="warning"
        showIcon
        style={{ marginBottom: 16 }}
        message="当前高级模式只编辑官方来源"
        description="本地自定义模型当前通过模型数据库页和槽位替换来完成，不直接进入高级图编辑器。这是为了把学习成本和结构复杂度控制在当前产品阶段可接受的范围内。"
      />

      <Card className="surface-card surface-card--accent" style={{ marginBottom: 16 }}>
        <div className="section-heading">
          <div>
              <div className="section-heading__eyebrow">Mode boundary</div>
              <h2 className="section-heading__title">默认模式与高级模式的真实分工</h2>
              <p className="section-heading__description">
                高级模式不接管默认首页，只承担结构编辑和组件约束可视化；正式版默认链路仍然是“问题定义 / 骨架推荐 / 参数编辑 / 训练 / 结果”。
              </p>
            </div>
          <Tag color="processing">Formal release policy</Tag>
        </div>

        <div className="editorial-grid">
          <div className="surface-note surface-note--dense">
            <strong>默认模式</strong>
            <p>面向非技术用户，先做问题收敛和骨架推荐，不要求理解节点连接细节。</p>
          </div>
          <div className="surface-note surface-note--dense">
            <strong>高级模式</strong>
            <p>面向结构编辑和扩展需求，先显示组件注册表、连接规则和编译边界，再进入更细的节点交互。</p>
          </div>
        </div>
      </Card>

      <Card className="surface-card" style={{ marginBottom: 16 }}>
        <div className="section-heading">
          <div>
            <div className="section-heading__eyebrow">Component registry</div>
            <h2 className="section-heading__title">正式版高级模式当前能看到哪些组件家族</h2>
            <p className="section-heading__description">
              组件现在直接从官方模型真源投影过来，不再维护第二套前端本地清单。
            </p>
          </div>
          <Tag color="processing">{catalog?.components.length ?? 0} components</Tag>
        </div>

        {catalog?.components.length ? (
          <Space direction="vertical" size={16} style={{ width: "100%" }}>
          {familyEntries.map(([familyKey, familyLabel]) => {
            const components = (catalog?.components || []).filter(
              (item) => item.family === familyKey,
            );
            return (
              <Card key={familyKey} size="small" title={familyLabel}>
                <Row gutter={[16, 16]}>
                  {components.map((component) => (
                    <Col xs={24} xl={12} key={component.id}>
                      <Card size="small">
                        <Space direction="vertical" size={8} style={{ width: "100%" }}>
                          <Space wrap>
                            <Text strong>{component.label}</Text>
                            <Tag color={STATUS_COLORS[component.status]}>
                              {catalog?.statusLabels?.[component.status] || component.status}
                            </Tag>
                          </Space>
                          <Paragraph style={{ marginBottom: 0 }}>
                            {component.description}
                          </Paragraph>
                          {component.schemaPath ? (
                            <Text type="secondary">
                              schema: {component.schemaPath}
                            </Text>
                          ) : null}
                          {component.inputs?.length ? (
                            <Text type="secondary">
                              inputs: {component.inputs.join(" / ")}
                            </Text>
                          ) : null}
                          {component.outputs?.length ? (
                            <Text type="secondary">
                              outputs: {component.outputs.join(" / ")}
                            </Text>
                          ) : null}
                          {component.notes?.length ? (
                            <div className="surface-note surface-note--dense">
                              {component.notes.join(" ")}
                            </div>
                          ) : null}
                        </Space>
                      </Card>
                    </Col>
                  ))}
                </Row>
              </Card>
            );
          })}
          </Space>
        ) : (
          <Empty description="暂无高级模式组件目录" />
        )}
      </Card>

      <Card className="surface-card" style={{ marginBottom: 16 }}>
        <div className="section-heading">
          <div>
            <div className="section-heading__eyebrow">连接约束</div>
            <h2 className="section-heading__title">哪些连接是必须的，哪些组合当前被阻断</h2>
            <p className="section-heading__description">
              这层规则的目标不是限制用户，而是阻止前台把用户带入当前 runtime 根本编译不出来的假路径。
            </p>
          </div>
          <Tag color="warning">Human-readable constraints</Tag>
        </div>

        <Table
          size="small"
          pagination={false}
          rowKey={(record) => `${record.fromFamily}-${record.toFamily}`}
          dataSource={catalog?.connectionRules || []}
          columns={[
            {
              title: "From",
              dataIndex: "fromFamily",
              key: "fromFamily",
              render: (value: AdvancedBuilderFamily) =>
                catalog?.familyLabels?.[value] || value,
            },
            {
              title: "To",
              dataIndex: "toFamily",
              key: "toFamily",
              render: (value: AdvancedBuilderFamily) =>
                catalog?.familyLabels?.[value] || value,
            },
            {
              title: "状态",
              dataIndex: "status",
              key: "status",
              render: (value: string) => {
                const color =
                  value === "required"
                    ? "success"
                    : value === "conditional"
                      ? "warning"
                      : "default";
                return <Tag color={color}>{value}</Tag>;
              },
            },
            {
              title: "说明",
              dataIndex: "description",
              key: "description",
            },
          ]}
        />
      </Card>

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={12}>
          <Card className="surface-card">
            <div className="section-heading">
              <div>
                <div className="section-heading__eyebrow">Compile-ready blueprints</div>
                <h2 className="section-heading__title">当前能编译回主链的高级骨架</h2>
                <p className="section-heading__description">
                  这些蓝图代表高级模式当前最安全的起点，它们仍然服从正式版主链的 runtime 边界。
                </p>
              </div>
              <Tag color="success">{compileReadyBlueprints.length} ready</Tag>
            </div>

            <Space direction="vertical" size={12} style={{ width: "100%" }}>
              {compileReadyBlueprints.map((blueprint) => (
                <Card key={blueprint.id} size="small">
                  <Space direction="vertical" size={8} style={{ width: "100%" }}>
                    <Space wrap>
                      <Text strong>{blueprint.label}</Text>
                      <Tag color="success">可编译</Tag>
                    </Space>
                    <Paragraph style={{ marginBottom: 0 }}>
                      {blueprint.description}
                    </Paragraph>
                    <Text type="secondary">
                      components: {blueprint.components.join(" -> ")}
                    </Text>
                    <Text type="secondary">
                      compiles to: {blueprint.compilesTo || "-"}
                    </Text>
                  </Space>
                </Card>
              ))}
            </Space>
          </Card>
        </Col>

        <Col xs={24} xl={12}>
          <Card className="surface-card">
            <div className="section-heading">
              <div>
                <div className="section-heading__eyebrow">Draft-only area</div>
                <h2 className="section-heading__title">当前只允许停留在草稿层的组合</h2>
                <p className="section-heading__description">
                  这些组合不应该被包装成“已经支持”，但应该在高级模式里被明确标注出来，方便后续 runtime 和编译层补齐。
                </p>
              </div>
              <Tag>{draftBlueprints.length} draft</Tag>
            </div>

            <Space direction="vertical" size={12} style={{ width: "100%" }}>
              {draftBlueprints.map((blueprint) => (
                <Card key={blueprint.id} size="small">
                  <Space direction="vertical" size={8} style={{ width: "100%" }}>
                    <Space wrap>
                      <Text strong>{blueprint.label}</Text>
                      <Tag>仅草稿</Tag>
                    </Space>
                    <Paragraph style={{ marginBottom: 0 }}>
                      {blueprint.description}
                    </Paragraph>
                    <Text type="secondary">
                      components: {blueprint.components.join(" -> ")}
                    </Text>
                    {blueprint.blockers?.length ? (
                      <div className="surface-note surface-note--dense">
                        {blueprint.blockers.join(" ")}
                      </div>
                    ) : null}
                  </Space>
                </Card>
              ))}
            </Space>
          </Card>
        </Col>
      </Row>

      <Card className="surface-card surface-card--editorial">
        <div className="section-heading">
          <div>
            <div className="section-heading__eyebrow">Legacy workflow boundary</div>
            <h2 className="section-heading__title">为什么这页不是旧 workflow editor</h2>
            <p className="section-heading__description">
              因为旧 workflow editor 的端口语义更偏执行图，不等于正式版模型组件图。正式版高级模式要先把“模型组件如何连接”这件事讲清楚，而不是直接让用户进入一张看起来能拖拽、但语义上并不稳定的画布。
            </p>
          </div>
          <Tag color="default">Experimental executor remains separate</Tag>
        </div>

        <div className="editorial-quote">
          <span className="editorial-quote__mark">/</span>
          <p>
            先收紧组件注册表、连接规则和蓝图边界，再开放真正的节点编辑。这是为了保证高级模式最终输出的是可解释、可校验、可编译的模型图，而不是一张仓库当前根本跑不起来的好看画布。
          </p>
        </div>

        <Space>
          <Button
            icon={<ArrowLeftOutlined />}
            onClick={() => navigate("/config")}
          >
            回到默认模式
          </Button>
          <Button
            icon={<ApartmentOutlined />}
            onClick={() => navigate("/config/advanced/canvas")}
          >
            进入节点图原型
          </Button>
          <Button
            icon={<ExperimentOutlined />}
            onClick={() => navigate("/models")}
          >
            看结果后台结构
          </Button>
          <Button
            icon={<ApartmentOutlined />}
            onClick={() => navigate("/training")}
          >
            回到真实执行主链
          </Button>
        </Space>
      </Card>
    </PageScaffold>
  );
}
