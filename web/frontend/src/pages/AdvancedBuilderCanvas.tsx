import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
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
  ArrowLeftOutlined,
  ClearOutlined,
  DeleteOutlined,
  NodeIndexOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import ReactFlow, {
  Background,
  BackgroundVariant,
  Controls,
  addEdge,
  type Connection,
  type Edge,
  type Node,
  useEdgesState,
  useNodesState,
} from "reactflow";
import "reactflow/dist/style.css";

import {
  getAdvancedBuilderCatalog,
  type AdvancedBuilderCompileIssue,
  type AdvancedBuilderCatalogResponse,
  compileAdvancedBuilder,
  startTrainingFromAdvancedBuilder,
} from "@/api/advancedBuilder";
import AdvancedComponentNode from "@/components/advanced/AdvancedComponentNode";
import AdvancedComponentPalette from "@/components/advanced/AdvancedComponentPalette";
import PageScaffold from "@/components/layout/PageScaffold";
import {
  type AdvancedBuilderFamily,
} from "@/config/advancedBuilderCatalog";
import {
  buildBlueprintGraph,
  canConnectFamilies,
  createBuilderNode,
  evaluateAdvancedBuilderGraph,
  type AdvancedBuilderNodeData,
} from "@/utils/advancedBuilder";
import {
  buildResultsCommand,
  buildTrainCommand,
  buildYamlFromRunSpec,
  type RunPresetId,
  type RunSpec,
} from "@/utils/runSpec";

const { Paragraph, Text } = Typography;

const nodeTypes = {
  advancedBuilderComponent: AdvancedComponentNode,
};

function formatConnectionLabel(
  familyLabels: Record<AdvancedBuilderFamily, string>,
  sourceNode?: Node<AdvancedBuilderNodeData>,
  targetNode?: Node<AdvancedBuilderNodeData>,
) {
  if (!sourceNode || !targetNode) {
    return "-";
  }
  return `${familyLabels[sourceNode.data.family]} -> ${familyLabels[targetNode.data.family]}`;
}

function toConfigFileName(experimentName: string): string {
  return `${experimentName
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "advanced-builder-graph"}.yaml`;
}

function pickContextString(
  issue: AdvancedBuilderCompileIssue,
  key: string,
): string | null {
  const value = issue.context?.[key];
  return typeof value === "string" ? value : null;
}

function pickContextStringArray(
  issue: AdvancedBuilderCompileIssue,
  key: string,
): string[] {
  const value = issue.context?.[key];
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

function renderIssueDescription(issue: AdvancedBuilderCompileIssue) {
  const issueMeta = [issue.code, issue.path].filter(Boolean).join(" · ") || null;
  if (!issueMeta && !issue.suggestion) {
    return undefined;
  }
  return (
    <Space direction="vertical" size={0}>
      {issueMeta ? <Text type="secondary">{issueMeta}</Text> : null}
      {issue.suggestion ? <Text type="secondary">{issue.suggestion}</Text> : null}
    </Space>
  );
}

export default function AdvancedBuilderCanvas() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [catalog, setCatalog] = useState<AdvancedBuilderCatalogResponse | null>(null);
  const familyLabels =
    catalog?.familyLabels || ({} as Record<AdvancedBuilderFamily, string>);
  const statusLabels =
    catalog?.statusLabels ||
    ({} as Record<"compile_ready" | "conditional" | "draft_only", string>);
  const requestedBlueprintId = searchParams.get("blueprint");
  const [selectedBlueprintId, setSelectedBlueprintId] = useState("quickstart_multimodal");
  const [nodes, setNodes, onNodesChange] = useNodesState<AdvancedBuilderNodeData>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  useEffect(() => {
    const load = async () => {
      try {
        const payload = await getAdvancedBuilderCatalog();
        setCatalog(payload);
        const nextBlueprintId =
          requestedBlueprintId &&
          payload.blueprints.some(
            (blueprint: { id: string }) => blueprint.id === requestedBlueprintId,
          )
            ? requestedBlueprintId
            : "quickstart_multimodal";
        const graph = buildBlueprintGraph(
          nextBlueprintId,
          payload.blueprints,
          payload.components,
          payload.connectionRules,
          payload.familyLabels,
          payload.statusLabels,
        );
        setSelectedBlueprintId(nextBlueprintId);
        setNodes(graph.nodes);
        setEdges(graph.edges);
      } catch (error) {
        console.error("Failed to load advanced builder catalog:", error);
        message.error("加载高级模式目录失败");
      }
    };

    void load();
  }, [requestedBlueprintId, setEdges, setNodes]);

  const evaluation = useMemo(
    () =>
      evaluateAdvancedBuilderGraph(
        nodes,
        edges,
        catalog?.connectionRules,
      ),
    [nodes, edges, catalog?.connectionRules],
  );
  const [compileResult, setCompileResult] = useState<{
    preset: RunPresetId;
    spec: RunSpec | null;
    issues: AdvancedBuilderCompileIssue[];
    contractValidation: {
      ok: boolean;
      errors: Array<{
        path: string;
        message: string;
        error_code: string;
        suggestion?: string | null;
      }>;
      warnings: Array<{
        path: string;
        message: string;
        error_code: string;
        suggestion?: string | null;
      }>;
    } | null;
    mainlineContract: {
      schema_family: string;
      output_dir: string;
      model: {
        model_type: string;
        vision_backbone: string;
        fusion_type: string;
        num_classes: number;
      };
    } | null;
    source: "server";
  }>({
    preset: "quickstart",
    spec: null,
    issues: [],
    contractValidation: null,
    mainlineContract: null,
    source: "server",
  });
  const [compilePending, setCompilePending] = useState(false);
  const [compileRequestError, setCompileRequestError] = useState<string | null>(
    null,
  );
  const compiledYaml = useMemo(
    () => (compileResult.spec ? buildYamlFromRunSpec(compileResult.spec) : ""),
    [compileResult.spec],
  );
  const compiledConfigPath = compileResult.spec
    ? `./${toConfigFileName(compileResult.spec.experimentName)}`
    : "./advanced-builder-graph.yaml";
  const compiledTrainCommand = buildTrainCommand(compiledConfigPath);
  const compiledResultsCommand = buildResultsCommand(compiledConfigPath);

  const selectedNodes = nodes.filter((node) => node.selected);
  const selectedEdges = edges.filter((edge) => edge.selected);

  const locateIssueOnCanvas = (issue: AdvancedBuilderCompileIssue) => {
    const nodeIdSet = new Set<string>();
    const contextNodeIds = pickContextStringArray(issue, "node_ids");
    const contextFamilies = pickContextStringArray(issue, "families");
    const sourceNodeId = pickContextString(issue, "source_node_id");
    const targetNodeId = pickContextString(issue, "target_node_id");
    const nodeId = pickContextString(issue, "node_id");
    const fromFamily = pickContextString(issue, "from_family");
    const toFamily = pickContextString(issue, "to_family");

    for (const item of contextNodeIds) {
      if (!item.startsWith("<")) {
        nodeIdSet.add(item);
      }
    }
    if (nodeId && !nodeId.startsWith("<")) {
      nodeIdSet.add(nodeId);
    }
    if (sourceNodeId && !sourceNodeId.startsWith("<")) {
      nodeIdSet.add(sourceNodeId);
    }
    if (targetNodeId && !targetNodeId.startsWith("<")) {
      nodeIdSet.add(targetNodeId);
    }

    const resolveNodeIdsByFamily = (family: string | null): string[] => {
      if (!family) {
        return [];
      }
      return nodes
        .filter((item) => item.data.family === (family as AdvancedBuilderFamily))
        .map((item) => item.id);
    };

    for (const family of contextFamilies) {
      for (const resolvedId of resolveNodeIdsByFamily(family)) {
        nodeIdSet.add(resolvedId);
      }
    }

    const fromFamilyNodeIds = resolveNodeIdsByFamily(fromFamily);
    const toFamilyNodeIds = resolveNodeIdsByFamily(toFamily);
    const fromFamilyNodeId = fromFamilyNodeIds[0] || null;
    const toFamilyNodeId = toFamilyNodeIds[0] || null;
    if (fromFamilyNodeId) {
      nodeIdSet.add(fromFamilyNodeId);
    }
    if (toFamilyNodeId) {
      nodeIdSet.add(toFamilyNodeId);
    }

    const resolvedSource = sourceNodeId || fromFamilyNodeId;
    const resolvedTarget = targetNodeId || toFamilyNodeId;
    const matchedEdgeId =
      resolvedSource && resolvedTarget
        ? edges.find(
            (edge) =>
              edge.source === resolvedSource && edge.target === resolvedTarget,
          )?.id || null
        : null;

    if (!nodeIdSet.size && !matchedEdgeId) {
      message.info("该问题当前没有可定位的节点或连线");
      return;
    }

    setNodes((current) =>
      current.map((node) => ({
        ...node,
        selected: nodeIdSet.has(node.id),
      })),
    );
    setEdges((current) =>
      current.map((edge) => ({
        ...edge,
        selected: matchedEdgeId ? edge.id === matchedEdgeId : false,
      })),
    );
  };

  const loadBlueprint = (blueprintId: string) => {
    if (!catalog) {
      return;
    }
    const graph = buildBlueprintGraph(
      blueprintId,
      catalog.blueprints,
      catalog.components,
      catalog.connectionRules,
      catalog.familyLabels,
      catalog.statusLabels,
    );
    setSelectedBlueprintId(blueprintId);
    setNodes(graph.nodes);
    setEdges(graph.edges);
  };

  const handleAddComponent = (componentId: string) => {
    setNodes((current) => [
      ...current,
      createBuilderNode(
        componentId,
        current.length,
        catalog?.components,
        catalog?.familyLabels,
        catalog?.statusLabels,
      ),
    ]);
    message.success("组件已加入画布");
  };

  const handleConnect = (connection: Connection) => {
    const sourceNode = nodes.find((node) => node.id === connection.source);
    const targetNode = nodes.find((node) => node.id === connection.target);

    if (!sourceNode || !targetNode) {
      message.error("连接失败：找不到节点");
      return;
    }

    const rule = canConnectFamilies(
      sourceNode.data.family,
      targetNode.data.family,
      catalog?.connectionRules,
      catalog?.familyLabels,
    );
    if (!rule.allowed) {
      message.error(rule.description);
      return;
    }

    if (rule.status === "conditional") {
      message.warning(`有条件连接：${rule.description}`);
    } else {
      message.success("连接已添加");
    }

    setEdges((current) =>
      addEdge(
        {
          ...connection,
          label: rule.status,
          animated: rule.status === "conditional",
        },
        current,
      ),
    );
  };

  const handleDeleteSelected = () => {
    setNodes((current) => current.filter((node) => !node.selected));
    setEdges((current) => current.filter((edge) => !edge.selected));
    message.success("已移除选中的节点或连接");
  };

  const handleClear = () => {
    setNodes([]);
    setEdges([]);
    message.success("高级模式画布已清空");
  };

  const handleCopyText = async (text: string, successMessage: string) => {
    try {
      await navigator.clipboard.writeText(text);
      message.success(successMessage);
    } catch (error) {
      console.error("Clipboard write failed:", error);
      message.error("复制失败，请检查浏览器权限");
    }
  };

  const handleDownloadYaml = () => {
    if (!compileResult.spec) {
      message.warning("当前图还没有可编译的配置草案");
      return;
    }

    const blob = new Blob([compiledYaml], { type: "text/yaml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = toConfigFileName(compileResult.spec.experimentName);
    link.click();
    URL.revokeObjectURL(url);
    message.success("已下载编译后的 YAML 草案");
  };

  const buildCompileRequestPayload = () => ({
    nodes: nodes.map((node) => ({
      id: node.id,
      type: node.type,
      data: node.data,
      position: node.position,
    })),
    edges: edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      sourceHandle: edge.sourceHandle,
      targetHandle: edge.targetHandle,
      label: edge.label,
    })),
    blueprint_id: selectedBlueprintId,
  });

  const handleStartTraining = async () => {
    try {
      const response = await startTrainingFromAdvancedBuilder(
        buildCompileRequestPayload(),
      );
      message.success("已从高级模式直接创建真实训练任务");
      navigate(`/training?job=${response.job_id}`);
    } catch (error: any) {
      console.error("Failed to start training from advanced builder:", error);
      const detail = error?.response?.data?.detail;
      if (typeof detail?.message === "string") {
        message.error(detail.message);
      } else {
        message.error("从高级模式创建训练任务失败");
      }
    }
  };

  useEffect(() => {
    let cancelled = false;

    const runCompile = async () => {
      setCompilePending(true);
      setCompileRequestError(null);
      try {
        const payload = await compileAdvancedBuilder(buildCompileRequestPayload());

        if (cancelled) {
          return;
        }

        setCompileResult({
          preset: payload.preset,
          spec: (payload.run_spec as RunSpec | null) ?? null,
          issues: payload.issues,
          contractValidation: payload.contract_validation,
          mainlineContract: payload.mainline_contract,
          source: "server",
        });
      } catch (error) {
        console.error("Advanced builder compile failed:", error);
        if (cancelled) {
          return;
        }
        setCompileRequestError("后端编译服务暂时不可用");
        setCompileResult({
          preset: "quickstart",
          spec: null,
          issues: [
            {
              level: "error",
              code: "ABG-E999",
              message:
                "后端编译服务暂时不可用，当前无法把节点图降级生成正式版 RunSpec 草案。",
              suggestion: "检查后端服务状态后重试。",
            },
          ],
          contractValidation: null,
          mainlineContract: null,
          source: "server",
        });
      } finally {
        if (!cancelled) {
          setCompilePending(false);
        }
      }
    };

    void runCompile();
    return () => {
      cancelled = true;
    };
  }, [nodes, edges]);

  return (
    <PageScaffold
      eyebrow="Advanced canvas prototype"
      title="高级模式节点图原型"
      description="这不是旧 workflow executor 的包装层，而是正式版高级模式的结构编辑原型。当前允许你基于组件注册表添加节点、按连接约束拉线，并实时看到这张图是否还处在可编译边界内。"
      chips={[
        { label: "Node prototype", tone: "amber" },
        { label: "Constraint-aware", tone: "teal" },
        { label: "Compile boundary", tone: "blue" },
      ]}
      actions={
        <>
          <Button
            icon={<ArrowLeftOutlined />}
            onClick={() => navigate("/config/advanced")}
          >
            返回注册表页
          </Button>
          <Button
            icon={<ReloadOutlined />}
            onClick={() => loadBlueprint(selectedBlueprintId)}
          >
            重载蓝图
          </Button>
          <Button
            icon={<DeleteOutlined />}
            onClick={handleDeleteSelected}
            disabled={!selectedNodes.length && !selectedEdges.length}
          >
            删除选中
          </Button>
          <Button icon={<ClearOutlined />} onClick={handleClear}>
            清空画布
          </Button>
        </>
      }
      aside={
        <div className="hero-aside-panel">
          <span className="hero-aside-panel__label">Compile status</span>
          <div className="hero-aside-panel__value">
            {evaluation.compileReady ? "当前图可编译" : "当前图未达可编译边界"}
          </div>
          <div className="hero-aside-panel__copy">
            当前画布不会直接执行训练，它只回答一件事：这张图是否还落在正式版主链可解释、可校验、可编译的边界里。
          </div>
          <div className="surface-note">
            当前蓝图：
              <strong>
                {" "}
              {catalog?.blueprints.find((item) => item.id === selectedBlueprintId)?.label || "-"}
            </strong>
          </div>
        </div>
      }
      metrics={[
        {
          label: "Nodes",
          value: nodes.length,
          hint: "Components on canvas",
          tone: "amber",
        },
        {
          label: "Edges",
          value: edges.length,
          hint: "Connections on canvas",
          tone: "blue",
        },
        {
          label: "Missing families",
          value: evaluation.missingFamilies.length,
          hint: evaluation.missingFamilies.length
            ? evaluation.missingFamilies
                .map((family) => familyLabels[family])
                .join(" / ")
            : "All required families present",
          tone: evaluation.missingFamilies.length ? "rose" : "teal",
        },
        {
          label: "Draft-only nodes",
          value: evaluation.draftOnlyComponents.length,
          hint: evaluation.draftOnlyComponents.length
            ? evaluation.draftOnlyComponents.join(" / ")
            : "No draft-only blockers",
          tone: evaluation.draftOnlyComponents.length ? "rose" : "teal",
        },
      ]}
    >
      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="当前原型的边界"
        description="当前允许你编辑节点图，但还不把这张图直接交给后端执行。真正的目标是先把高级模式的前台语义收紧成一张可解释、可校验、可编译的模型图，再决定如何把它降级编译回正式版主链配置。"
      />
      <Alert
        type="warning"
        showIcon
        style={{ marginBottom: 16 }}
        message="官方来源专用"
        description="这张画布当前只服务官方模型来源。用户自定义模型仍然通过模型数据库里的本地自定义模板进行维护，不直接进入图编辑器。"
      />

      <div className="split-grid">
        <Card className="surface-card" style={{ minHeight: 760 }}>
          <Space direction="vertical" size={16} style={{ width: "100%" }}>
            <div className="section-heading">
              <div>
                <div className="section-heading__eyebrow">Blueprint seed</div>
                <h2 className="section-heading__title">先选一个正式版蓝图再编辑</h2>
                <p className="section-heading__description">
                  这样可以避免高级模式一上来就变成没有约束的空白画布。
                </p>
              </div>
              <Tag color="processing">Seed before freeform</Tag>
            </div>

            <Select
              value={selectedBlueprintId}
              onChange={loadBlueprint}
              options={(catalog?.blueprints || []).map((blueprint) => ({
                label: `${blueprint.label} · ${statusLabels[blueprint.status] || blueprint.status}`,
                value: blueprint.id,
              }))}
            />

            <div
              style={{
                position: "relative",
                width: "100%",
                height: 620,
                borderRadius: 16,
                border: "1px solid var(--surface-border, #d9d9d9)",
                overflow: "hidden",
                background:
                  "linear-gradient(180deg, rgba(249,250,251,0.9), rgba(241,245,249,0.9))",
              }}
            >
              <AdvancedComponentPalette
                onAddComponent={handleAddComponent}
                components={catalog?.components || []}
                familyLabels={familyLabels}
                statusLabels={statusLabels}
              />
              <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={handleConnect}
                nodeTypes={nodeTypes}
                fitView
              >
                <Controls />
                <Background variant={BackgroundVariant.Dots} gap={18} size={1} />
              </ReactFlow>
            </div>
          </Space>
        </Card>

        <Space direction="vertical" size={16} style={{ width: "100%" }}>
          <Card className="surface-card" title="编译边界检查">
            <Space direction="vertical" size={12} style={{ width: "100%" }}>
              <Tag color={evaluation.compileReady ? "success" : "warning"}>
                {evaluation.compileReady ? "可编译" : "未达可编译条件"}
              </Tag>
              {evaluation.missingFamilies.length ? (
                <div className="surface-note surface-note--dense">
                  缺少组件家族：
                  {" "}
                  {evaluation.missingFamilies
                    .map((family) => familyLabels[family])
                    .join(" / ")}
                </div>
              ) : null}
              {evaluation.missingRequiredConnections.length ? (
                <div className="surface-note surface-note--dense">
                  缺少必须连接：
                  {" "}
                  {evaluation.missingRequiredConnections
                    .map(
                      (rule) =>
                        `${familyLabels[rule.fromFamily]} -> ${familyLabels[rule.toFamily]}`,
                    )
                    .join(" / ")}
                </div>
              ) : null}
              {evaluation.conditionalComponents.length ? (
                <div className="surface-note surface-note--dense">
                  有条件组件：
                  {" "}
                  {evaluation.conditionalComponents.join(" / ")}
                </div>
              ) : null}
              {evaluation.draftOnlyComponents.length ? (
                <div className="surface-note surface-note--dense">
                  草稿组件：
                  {" "}
                  {evaluation.draftOnlyComponents.join(" / ")}
                </div>
              ) : null}
              {!evaluation.missingFamilies.length &&
              !evaluation.missingRequiredConnections.length &&
              !evaluation.draftOnlyComponents.length ? (
                <Alert
                  type="success"
                  showIcon
                  message="当前图满足正式版高级模式的最小可编译边界"
                />
              ) : null}
            </Space>
          </Card>

          <Card className="surface-card" title="图编译结果">
            <Space direction="vertical" size={12} style={{ width: "100%" }}>
              <Space wrap>
                <Tag color={compileResult.spec ? "success" : "warning"}>
                  {compileResult.spec ? "已生成 RunSpec 草案" : "当前还不能生成 RunSpec"}
                </Tag>
                <Tag color={compilePending ? "processing" : "default"}>
                  {compilePending ? "编译中" : "后端编译"}
                </Tag>
              </Space>

              {compileRequestError ? (
                <Alert type="error" showIcon message={compileRequestError} />
              ) : null}

              {compileResult.issues.length ? (
                compileResult.issues.map((issue, index) => (
                  <Alert
                    key={`${issue.level}-${issue.code || "na"}-${index}`}
                    type={issue.level === "error" ? "error" : "warning"}
                    showIcon
                    message={issue.message}
                    description={renderIssueDescription(issue)}
                    action={
                      issue.context ? (
                        <Button
                          size="small"
                          onClick={() => locateIssueOnCanvas(issue)}
                        >
                          定位到画布
                        </Button>
                      ) : undefined
                    }
                  />
                ))
              ) : (
                <Alert
                  type="success"
                  showIcon
                  message="当前图已经可以降级编译到正式版 RunSpec"
                />
              )}

              {compileResult.spec ? (
                <>
                  <div className="surface-note surface-note--dense">
                    <strong>当前预设</strong>
                    <p>{compileResult.preset}</p>
                  </div>
                  {compileResult.mainlineContract ? (
                    <div className="surface-note surface-note--dense">
                      <strong>正式 contract</strong>
                      <p>
                        {compileResult.mainlineContract.model.model_type} /{" "}
                        {compileResult.mainlineContract.model.vision_backbone} /{" "}
                        {compileResult.mainlineContract.model.fusion_type}
                      </p>
                    </div>
                  ) : null}
                  <div className="surface-note surface-note--dense">
                    <strong>编译来源</strong>
                    <p>{compileResult.source}</p>
                  </div>
                  <div className="surface-note surface-note--dense">
                    <strong>输出文件</strong>
                    <p>{compiledConfigPath}</p>
                  </div>
                  <Space wrap>
                    <Button
                      onClick={() =>
                        void handleCopyText(compiledYaml, "YAML 已复制")
                      }
                    >
                      复制 YAML
                    </Button>
                    <Button onClick={handleDownloadYaml}>下载 YAML</Button>
                    <Button
                      type="primary"
                      disabled={
                        compilePending ||
                        !compileResult.spec ||
                        !compileResult.contractValidation?.ok
                      }
                      onClick={() => void handleStartTraining()}
                    >
                      直接创建训练任务
                    </Button>
                    <Button
                      onClick={() =>
                        navigate("/config", {
                          state: {
                            compiledRunSpec: compileResult.spec,
                            compiledPreset: compileResult.preset,
                            source: "advanced builder canvas",
                          },
                        })
                      }
                    >
                      回到默认模式继续编辑
                    </Button>
                  </Space>
                  <div>
                    <Text strong>训练命令</Text>
                    <pre className="command-block">{compiledTrainCommand}</pre>
                  </div>
                  <div>
                    <Text strong>结果构建</Text>
                    <pre className="command-block">{compiledResultsCommand}</pre>
                  </div>
                </>
              ) : null}
            </Space>
          </Card>

          {compileResult.contractValidation ? (
            <Card className="surface-card" title="正式配置 contract 校验">
              <Space direction="vertical" size={12} style={{ width: "100%" }}>
                <Tag color={compileResult.contractValidation.ok ? "success" : "warning"}>
                  {compileResult.contractValidation.ok
                    ? "ExperimentConfig 校验通过"
                    : "ExperimentConfig 校验未通过"}
                </Tag>

                {compileResult.contractValidation.errors.length ? (
                  compileResult.contractValidation.errors.map((issue, index) => (
                    <Alert
                      key={`contract-error-${index}`}
                      type="error"
                      showIcon
                      message={`${issue.path}: ${issue.message}`}
                      description={issue.suggestion || undefined}
                    />
                  ))
                ) : (
                  <Alert
                    type="success"
                    showIcon
                    message="当前图已经通过正式 ExperimentConfig 结构校验"
                  />
                )}

                {compileResult.contractValidation.warnings.length
                  ? compileResult.contractValidation.warnings.map((issue, index) => (
                      <Alert
                        key={`contract-warning-${index}`}
                        type="warning"
                        showIcon
                        message={`${issue.path}: ${issue.message}`}
                        description={issue.suggestion || undefined}
                      />
                    ))
                  : null}

                {compileResult.mainlineContract ? (
                  <div className="surface-note surface-note--dense">
                    <strong>输出目录</strong>
                    <p>{compileResult.mainlineContract.output_dir}</p>
                  </div>
                ) : null}
              </Space>
            </Card>
          ) : null}

          <Card className="surface-card" title="当前选中对象">
            {selectedNodes.length ? (
              <Space direction="vertical" size={12} style={{ width: "100%" }}>
                {selectedNodes.map((node) => (
                  <Card key={node.id} size="small">
                    <Space direction="vertical" size={6} style={{ width: "100%" }}>
                      <Text strong>{node.data.label}</Text>
                      <Text type="secondary">
                        family: {familyLabels[node.data.family] || node.data.family}
                      </Text>
                      <Text type="secondary">
                        status: {statusLabels[node.data.status] || node.data.status}
                      </Text>
                      <Paragraph style={{ marginBottom: 0 }}>
                        {node.data.description}
                      </Paragraph>
                      {node.data.schemaPath ? (
                        <Text type="secondary">
                          schema: {node.data.schemaPath}
                        </Text>
                      ) : null}
                    </Space>
                  </Card>
                ))}
              </Space>
            ) : selectedEdges.length ? (
              <Space direction="vertical" size={12} style={{ width: "100%" }}>
                {selectedEdges.map((edge) => {
                  const sourceNode = nodes.find((node) => node.id === edge.source);
                  const targetNode = nodes.find((node) => node.id === edge.target);
                  const rule =
                    sourceNode && targetNode
                      ? canConnectFamilies(
                          sourceNode.data.family,
                          targetNode.data.family,
                          catalog?.connectionRules,
                          familyLabels,
                        )
                      : null;
                  return (
                    <Card key={edge.id} size="small">
                      <Space direction="vertical" size={6} style={{ width: "100%" }}>
                        <Text strong>
                          {formatConnectionLabel(familyLabels, sourceNode, targetNode)}
                        </Text>
                        {rule ? (
                          <>
                            <Tag
                              color={
                                rule.status === "required"
                                  ? "success"
                                  : rule.status === "conditional"
                                    ? "warning"
                                    : "default"
                              }
                            >
                              {rule.status}
                            </Tag>
                            <Paragraph style={{ marginBottom: 0 }}>
                              {rule.description}
                            </Paragraph>
                          </>
                        ) : null}
                      </Space>
                    </Card>
                  );
                })}
              </Space>
            ) : (
              <Empty description="选择一个节点或连接，查看它在正式版高级模式里的含义" />
            )}
          </Card>

          {compileResult.spec ? (
            <Card className="surface-card" title="YAML 草案预览">
              <pre
                style={{
                  margin: 0,
                  padding: 16,
                  borderRadius: 12,
                  background: "#0f172a",
                  color: "#e2e8f0",
                  overflowX: "auto",
                  fontSize: 13,
                  lineHeight: 1.6,
                }}
              >
                {compiledYaml}
              </pre>
            </Card>
          ) : null}

          <Card className="surface-card" title="当前原型不做什么">
            <div className="editorial-stack">
              <div className="surface-note surface-note--dense">
                <strong>不直接执行</strong>
                <p>这张图当前不会直接提交给实验 workflow executor。</p>
              </div>
              <div className="surface-note surface-note--dense">
                <strong>不假装全能</strong>
                <p>只有落在组件注册表和连接规则里的结构才被视为正式版高级模式的一部分。</p>
              </div>
              <div className="surface-note surface-note--dense">
                <strong>先编译边界，再谈自动生成</strong>
                <p>只有这张图的约束稳定下来，后续 AI 初始化节点图才有意义。</p>
              </div>
            </div>
          </Card>
        </Space>
      </div>
    </PageScaffold>
  );
}
