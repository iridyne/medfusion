import { useCallback, useState } from "react";
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
import { Button, Space, message, Modal, Input } from "antd";
import {
  PlayCircleOutlined,
  SaveOutlined,
  DeleteOutlined,
  ClearOutlined,
} from "@ant-design/icons";
import {
  validateWorkflow,
  executeWorkflow,
  getWorkflowStatus,
} from "@/api/workflow";
import { nodeTypes } from "@/components/nodes";
import NodePalette from "@/components/NodePalette";
import NodeConfigPanel from "@/components/NodeConfigPanel";

const initialNodes: Node[] = [];
const initialEdges: Edge[] = [];

let nodeId = 0;
const getNodeId = () => `node_${nodeId++}`;

export default function WorkflowEditor() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [configPanelOpen, setConfigPanelOpen] = useState(false);
  const [workflowName, setWorkflowName] = useState("未命名工作流");
  const [nameModalOpen, setNameModalOpen] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [executionProgress, setExecutionProgress] = useState<
    Record<string, any>
  >({});

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
        y: Math.random() * 300 + 100,
      },
      data: {
        label: getNodeLabel(type),
      },
    };
    setNodes((nds) => [...nds, newNode]);
    message.success(`已添加${getNodeLabel(type)}节点`);
  };

  const getNodeLabel = (type: string): string => {
    const labels: Record<string, string> = {
      dataLoader: "数据加载器",
      model: "模型",
      training: "训练",
      evaluation: "评估",
    };
    return labels[type] || type;
  };

  const handleNodeDoubleClick: NodeMouseHandler = useCallback((_, node) => {
    setSelectedNode(node);
    setConfigPanelOpen(true);
  }, []);

  const handleSaveNodeConfig = (nodeId: string, data: any) => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === nodeId) {
          return {
            ...node,
            data: { ...node.data, ...data },
          };
        }
        return node;
      }),
    );
    message.success("节点配置已保存");
  };

  const handleDeleteSelected = () => {
    setNodes((nds) => nds.filter((node) => !node.selected));
    setEdges((eds) => eds.filter((edge) => !edge.selected));
    message.success("已删除选中的节点和连接");
  };

  const handleClear = () => {
    Modal.confirm({
      title: "确认清空",
      content: "确定要清空整个工作流吗？此操作不可恢复。",
      onOk: () => {
        setNodes([]);
        setEdges([]);
        message.success("工作流已清空");
      },
    });
  };

  const handleExecute = async () => {
    if (nodes.length === 0) {
      message.warning("工作流为空，请先添加节点");
      return;
    }

    try {
      setExecuting(true);

      const workflowData = {
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
      };

      // 步骤 1: 验证工作流
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

      // 步骤 2: 执行工作流
      message.destroy();
      message.loading("启动工作流执行...", 0);

      const result = await executeWorkflow({
        workflow: workflowData,
        name: workflowName,
      });

      message.destroy();
      message.success("工作流已启动");

      // 步骤 3: 监控执行进度
      const workflowId = result.workflow_id;
      const pollInterval = setInterval(async () => {
        try {
          const status = await getWorkflowStatus(workflowId);
          setExecutionProgress(status.status);

          // 更新节点状态
          setNodes((nds) =>
            nds.map((node) => {
              const nodeStatus = status.status.nodes[node.id];
              if (nodeStatus) {
                return {
                  ...node,
                  data: {
                    ...node.data,
                    status: nodeStatus.status,
                    error: nodeStatus.error,
                  },
                };
              }
              return node;
            }),
          );

          // 检查是否完成
          if (
            status.status.completed + status.status.failed ===
            status.status.total
          ) {
            clearInterval(pollInterval);
            setExecuting(false);

            if (status.status.failed > 0) {
              message.error("工作流执行失败，部分节点出错");
            } else {
              message.success("工作流执行成功！");
            }
          }
        } catch (error) {
          console.error("Failed to poll workflow status:", error);
          clearInterval(pollInterval);
          setExecuting(false);
          message.error("获取执行状态失败");
        }
      }, 2000); // 每 2 秒轮询一次
    } catch (error: any) {
      message.destroy();
      setExecuting(false);
      message.error(error.message || "执行工作流失败");
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
    // TODO: 调用 API 保存工作流
    message.success(`工作流 "${workflowName}" 已保存`);
    setNameModalOpen(false);
  };

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <NodePalette onAddNode={handleAddNode} />

      <div
        style={{
          position: "absolute",
          top: 16,
          right: 16,
          zIndex: 10,
        }}
      >
        <Space>
          <Button
            icon={<DeleteOutlined />}
            onClick={handleDeleteSelected}
            disabled={
              !nodes.some((n) => n.selected) && !edges.some((e) => e.selected)
            }
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
          onChange={(e) => setWorkflowName(e.target.value)}
        />
      </Modal>
    </div>
  );
}
