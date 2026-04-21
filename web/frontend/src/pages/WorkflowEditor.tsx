import { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  BackgroundVariant,
  NodeMouseHandler,
} from "reactflow";
import "reactflow/dist/style.css";
import {
  Alert,
  Button,
  Card,
  Input,
  Modal,
  Space,
  Tag,
  Typography,
  message,
} from "antd";
import {
  ArrowRightOutlined,
  ClearOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  SaveOutlined,
} from "@ant-design/icons";

import {
  executeWorkflow,
  getWorkflowStatus,
  validateWorkflow,
} from "@/api/workflow";
import NodeConfigPanel from "@/components/NodeConfigPanel";
import { nodeTypes } from "@/components/nodes";
import NodePalette from "@/components/NodePalette";

const initialNodes: Node[] = [];
const initialEdges: Edge[] = [];

let nodeId = 0;
const getNodeId = () => `node_${nodeId++}`;

function getNodeLabel(type: string): string {
  const labels: Record<string, string> = {
    dataLoader: "数据加载器",
    model: "模型",
    training: "训练",
    evaluation: "评估",
  };
  return labels[type] || type;
}

export default function WorkflowEditor() {
  const navigate = useNavigate();
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [configPanelOpen, setConfigPanelOpen] = useState(false);
  const [workflowName, setWorkflowName] = useState("未命名工作流");
  const [nameModalOpen, setNameModalOpen] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null);
  const [resultModelId, setResultModelId] = useState<number | null>(null);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  );

  const handleAddNode = (type: string) => {
    const newNode: Node = {
      id: getNodeId(),
      type,
      position: {
        x: Math.random() * 400 + 100,
        y: Math.random() * 300 + 140,
      },
      data: {
        label: getNodeLabel(type),
      },
    };
    setNodes((existing) => [...existing, newNode]);
    message.success(`已添加${getNodeLabel(type)}节点`);
  };

  const handleNodeDoubleClick: NodeMouseHandler = useCallback((_, node) => {
    setSelectedNode(node);
    setConfigPanelOpen(true);
  }, []);

  const handleSaveNodeConfig = (targetNodeId: string, data: any) => {
    setNodes((existing) =>
      existing.map((node) =>
        node.id === targetNodeId
          ? {
              ...node,
              data: { ...node.data, ...data },
            }
          : node,
      ),
    );
    message.success("节点配置已保存");
  };

  const handleDeleteSelected = () => {
    setNodes((existing) => existing.filter((node) => !node.selected));
    setEdges((existing) => existing.filter((edge) => !edge.selected));
    message.success("已删除选中的节点和连接");
  };

  const handleClear = () => {
    Modal.confirm({
      title: "确认清空",
      content: "确定要清空整个工作流吗？此操作不可恢复。",
      onOk: () => {
        setNodes([]);
        setEdges([]);
        setTrainingJobId(null);
        setResultModelId(null);
        message.success("工作流已清空");
      },
    });
  };

  const buildWorkflowData = () => ({
    nodes: nodes.map((node) => ({
      id: node.id,
      type: node.type || "default",
      position: node.position,
      data: node.data,
    })),
    edges: edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      sourceHandle: edge.sourceHandle,
      targetHandle: edge.targetHandle,
    })),
  });

  const handleExecute = async () => {
    if (nodes.length === 0) {
      message.warning("工作流为空，请先添加节点");
      return;
    }

    try {
      setExecuting(true);
      setTrainingJobId(null);
      setResultModelId(null);

      const workflowData = buildWorkflowData();

      message.loading("验证工作流...", 0);
      const validation = await validateWorkflow(workflowData);
      if (!validation.valid) {
        message.destroy();
        Modal.error({
          title: "工作流验证失败",
          content: (
            <div>
              <p>请修复以下错误：</p>
              <ul>
                {validation.errors.map((error: string, index: number) => (
                  <li key={index}>{error}</li>
                ))}
              </ul>
            </div>
          ),
        });
        setExecuting(false);
        return;
      }

      message.destroy();
      message.loading("启动真实训练任务...", 0);
      const result = await executeWorkflow({
        workflow: workflowData,
        name: workflowName,
      });

      message.destroy();
      message.success("工作流已启动");
      setTrainingJobId(result.training_job_id ?? null);

      const workflowId = result.workflow_id;
      const pollInterval = setInterval(async () => {
        try {
          const status = await getWorkflowStatus(workflowId);

          setNodes((existing) =>
            existing.map((node) => {
              const nodeStatus = status.status.nodes[node.id];
              if (!nodeStatus) {
                return node;
              }
              return {
                ...node,
                data: {
                  ...node.data,
                  status: nodeStatus.status,
                  error: nodeStatus.error,
                },
              };
            }),
          );

          if (status.status.completed + status.status.failed === status.status.total) {
            clearInterval(pollInterval);
            setExecuting(false);
            setResultModelId(status.results?.result_model_id ?? null);

            if (status.status.failed > 0) {
              message.error("工作流执行失败，部分节点出错");
            } else {
              message.success("工作流执行成功");
            }
          }
        } catch (error) {
          console.error("Failed to poll workflow status:", error);
          clearInterval(pollInterval);
          setExecuting(false);
          message.error("获取执行状态失败");
        }
      }, 2000);
    } catch (error: any) {
      message.destroy();
      setExecuting(false);
      const detail = error?.response?.data?.detail;
      if (detail?.errors?.length) {
        message.error(detail.errors.join("；"));
      } else if (typeof detail?.message === "string") {
        message.error(detail.message);
      } else {
        message.error(error.message || "执行工作流失败");
      }
      console.error(error);
    }
  };

  const handleSave = () => {
    if (nodes.length === 0) {
      message.warning("工作流为空，无需保存");
      return;
    }
    setNameModalOpen(true);
  };

  const handleSaveConfirm = () => {
    message.success(`工作流 "${workflowName}" 已保存`);
    setNameModalOpen(false);
  };

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <Card
        size="small"
        style={{
          position: "absolute",
          top: 16,
          left: 232,
          right: 16,
          zIndex: 10,
        }}
      >
        <Space direction="vertical" size={8} style={{ width: "100%" }}>
          <Alert
            type="info"
            showIcon
            message="节点工作流 preview"
            description="当前只支持一条受限主线：dataLoader -> model -> training -> optional evaluation。图负责组织语义，真实执行直接回到稳定 training runtime。"
          />
          <Space wrap>
            <Tag color="processing">单数据节点</Tag>
            <Tag color="processing">单模型节点</Tag>
            <Tag color="processing">单训练节点</Tag>
            <Tag color="default">评估节点可选</Tag>
          </Space>
          <Typography.Text type="secondary">
            推荐先双击数据节点，套用“仓库 mock 数据（推荐）”模板，再按默认模型与训练参数直接启动。
          </Typography.Text>
          {trainingJobId ? (
            <Space wrap>
              <Tag color="gold">Linked job: {trainingJobId}</Tag>
              <Button size="small" icon={<ArrowRightOutlined />} onClick={() => navigate("/training")}>
                打开训练监控
              </Button>
              {resultModelId ? (
                <Button size="small" icon={<ArrowRightOutlined />} onClick={() => navigate("/models")}>
                  打开结果库
                </Button>
              ) : null}
            </Space>
          ) : null}
        </Space>
      </Card>

      <NodePalette onAddNode={handleAddNode} />

      <div
        style={{
          position: "absolute",
          top: 176,
          right: 16,
          zIndex: 10,
        }}
      >
        <Space>
          <Button
            icon={<DeleteOutlined />}
            onClick={handleDeleteSelected}
            disabled={!nodes.some((node) => node.selected) && !edges.some((edge) => edge.selected)}
          >
            删除选中
          </Button>
          <Button icon={<ClearOutlined />} onClick={handleClear} danger>
            清空
          </Button>
          <Button icon={<SaveOutlined />} onClick={handleSave}>
            保存
          </Button>
          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={handleExecute}
            loading={executing}
            disabled={executing}
          >
            {executing ? "执行中..." : "执行"}
          </Button>
        </Space>
      </div>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeDoubleClick={handleNodeDoubleClick}
        nodeTypes={nodeTypes}
        fitView
      >
        <Controls />
        <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
      </ReactFlow>

      <NodeConfigPanel
        node={selectedNode}
        open={configPanelOpen}
        onClose={() => setConfigPanelOpen(false)}
        onSave={handleSaveNodeConfig}
      />

      <Modal
        title="保存工作流"
        open={nameModalOpen}
        onOk={handleSaveConfirm}
        onCancel={() => setNameModalOpen(false)}
      >
        <Input
          placeholder="请输入工作流名称"
          value={workflowName}
          onChange={(event) => setWorkflowName(event.target.value)}
        />
      </Modal>
    </div>
  );
}
