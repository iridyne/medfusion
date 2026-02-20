"""
工作流执行引擎

实现节点化工作流的解析、验证和执行。
"""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .checkpoint_manager import CheckpointManager
from .node_executors import ExecutorFactory, NodeExecutionError
from .resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """节点执行状态"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PortType(Enum):
    """端口类型"""

    DATASET = "dataset"
    IMAGE_DATA = "image_data"
    TABULAR_DATA = "tabular_data"
    MODEL = "model"
    TRAINED_MODEL = "trained_model"
    METRICS = "metrics"
    HISTORY = "history"
    REPORT = "report"


@dataclass
class Port:
    """节点端口"""

    id: str
    type: PortType
    node_id: str
    is_input: bool
    value: Any = None


@dataclass
class Node:
    """工作流节点"""

    id: str
    type: str
    data: Dict[str, Any]
    position: Dict[str, float]
    inputs: Dict[str, Port] = field(default_factory=dict)
    outputs: Dict[str, Port] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.PENDING
    error: Optional[str] = None
    result: Any = None


@dataclass
class Edge:
    """节点连接边"""

    id: str
    source: str  # 源节点 ID
    target: str  # 目标节点 ID
    source_handle: str  # 源端口 ID
    target_handle: str  # 目标端口 ID


class WorkflowValidationError(Exception):
    """工作流验证错误"""

    pass


class WorkflowExecutionError(Exception):
    """工作流执行错误"""

    pass


class WorkflowEngine:
    """工作流执行引擎"""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        enable_checkpoints: bool = True,
        enable_monitoring: bool = True,
    ):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.adjacency_list: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency_list: Dict[str, List[str]] = defaultdict(list)
        self.data_dir = data_dir or Path.cwd() / "data"
        self.executor_factory = ExecutorFactory

        # 检查点管理
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_manager = None
        if enable_checkpoints:
            checkpoint_dir = self.data_dir / "checkpoints"
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)
            logger.info("Checkpoint manager enabled")

        # 资源监控
        self.enable_monitoring = enable_monitoring
        self.resource_monitor = None
        if enable_monitoring:
            self.resource_monitor = ResourceMonitor(interval=2.0, history_size=300)
            logger.info("Resource monitor enabled")
</text>

<old_text line=401>
    async def _execute_data_loader(
        self, node: Node, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行数据加载节点"""
        logger.info(f"Loading dataset: {node.data.get('datasetId')}")

        # TODO: 实际的数据加载逻辑
        # 这里返回模拟数据
        return {
            "dataset": {
                "id": node.data.get("datasetId"),
                "name": node.data.get("datasetName"),
                "split": node.data.get("split", "train"),
                "batch_size": node.data.get("batchSize", 32),
            }
        }

    def load_workflow(self, workflow_data: Dict[str, Any]) -> None:
        """
        加载工作流数据

        Args:
            workflow_data: 包含 nodes 和 edges 的工作流数据
        """
        logger.info("Loading workflow...")

        # 解析节点
        self.nodes = {}
        for node_data in workflow_data.get("nodes", []):
            node = Node(
                id=node_data["id"],
                type=node_data["type"],
                data=node_data.get("data", {}),
                position=node_data.get("position", {"x": 0, "y": 0}),
            )

            # 根据节点类型初始化端口
            self._initialize_ports(node)
            self.nodes[node.id] = node

        # 解析边
        self.edges = []
        self.adjacency_list = defaultdict(list)
        self.reverse_adjacency_list = defaultdict(list)

        for edge_data in workflow_data.get("edges", []):
            edge = Edge(
                id=edge_data["id"],
                source=edge_data["source"],
                target=edge_data["target"],
                source_handle=edge_data.get("sourceHandle", "default"),
                target_handle=edge_data.get("targetHandle", "default"),
            )
            self.edges.append(edge)

            # 构建邻接表
            self.adjacency_list[edge.source].append(edge.target)
            self.reverse_adjacency_list[edge.target].append(edge.source)

        logger.info(f"Loaded {len(self.nodes)} nodes and {len(self.edges)} edges")

    def _initialize_ports(self, node: Node) -> None:
        """初始化节点端口"""
        if node.type == "dataLoader":
            node.outputs["dataset"] = Port(
                id="dataset",
                type=PortType.DATASET,
                node_id=node.id,
                is_input=False,
            )

        elif node.type == "model":
            node.inputs["image_data"] = Port(
                id="image_data",
                type=PortType.IMAGE_DATA,
                node_id=node.id,
                is_input=True,
            )
            node.inputs["tabular_data"] = Port(
                id="tabular_data",
                type=PortType.TABULAR_DATA,
                node_id=node.id,
                is_input=True,
            )
            node.outputs["model"] = Port(
                id="model", type=PortType.MODEL, node_id=node.id, is_input=False
            )

        elif node.type == "training":
            node.inputs["model"] = Port(
                id="model", type=PortType.MODEL, node_id=node.id, is_input=True
            )
            node.inputs["dataset"] = Port(
                id="dataset", type=PortType.DATASET, node_id=node.id, is_input=True
            )
            node.outputs["trained_model"] = Port(
                id="trained_model",
                type=PortType.TRAINED_MODEL,
                node_id=node.id,
                is_input=False,
            )
            node.outputs["history"] = Port(
                id="history", type=PortType.HISTORY, node_id=node.id, is_input=False
            )

        elif node.type == "evaluation":
            node.inputs["model"] = Port(
                id="model", type=PortType.TRAINED_MODEL, node_id=node.id, is_input=True
            )
            node.inputs["test_data"] = Port(
                id="test_data", type=PortType.DATASET, node_id=node.id, is_input=True
            )
            node.outputs["metrics"] = Port(
                id="metrics", type=PortType.METRICS, node_id=node.id, is_input=False
            )
            node.outputs["report"] = Port(
                id="report", type=PortType.REPORT, node_id=node.id, is_input=False
            )

    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证工作流合法性

        Returns:
            (is_valid, errors): 验证结果和错误列表
        """
        errors = []

        # 检查是否有节点
        if not self.nodes:
            errors.append("工作流为空，至少需要一个节点")
            return False, errors

        # 检查循环依赖
        try:
            self._topological_sort()
        except WorkflowValidationError as e:
            errors.append(f"检测到循环依赖: {e!s}")

        # 检查边的有效性
        for edge in self.edges:
            if edge.source not in self.nodes:
                errors.append(f"边 {edge.id} 的源节点 {edge.source} 不存在")
            if edge.target not in self.nodes:
                errors.append(f"边 {edge.id} 的目标节点 {edge.target} 不存在")

            # 检查端口类型匹配
            if edge.source in self.nodes and edge.target in self.nodes:
                source_node = self.nodes[edge.source]
                target_node = self.nodes[edge.target]

                if edge.source_handle not in source_node.outputs:
                    errors.append(
                        f"节点 {edge.source} 没有输出端口 {edge.source_handle}"
                    )
                if edge.target_handle not in target_node.inputs:
                    errors.append(
                        f"节点 {edge.target} 没有输入端口 {edge.target_handle}"
                    )

        # 检查必需的输入
        for node in self.nodes.values():
            if node.type == "training":
                if not self.reverse_adjacency_list.get(node.id):
                    errors.append(f"训练节点 {node.id} 缺少输入连接")
            elif node.type == "evaluation":
                if not self.reverse_adjacency_list.get(node.id):
                    errors.append(f"评估节点 {node.id} 缺少输入连接")

        return len(errors) == 0, errors

    def _topological_sort(self) -> List[str]:
        """
        拓扑排序，返回节点执行顺序

        Returns:
            节点 ID 列表（按执行顺序）

        Raises:
            WorkflowValidationError: 如果存在循环依赖
        """
        # 计算入度
        in_degree = {node_id: 0 for node_id in self.nodes}
        for node_id in self.nodes:
            in_degree[node_id] = len(self.reverse_adjacency_list.get(node_id, []))

        # 找到所有入度为 0 的节点
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            node_id = queue.popleft()
            result.append(node_id)

            # 减少相邻节点的入度
            for neighbor in self.adjacency_list.get(node_id, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 检查是否所有节点都被访问
        if len(result) != len(self.nodes):
            unvisited = set(self.nodes.keys()) - set(result)
            raise WorkflowValidationError(f"检测到循环依赖，未访问的节点: {unvisited}")

        return result

    async def execute(
        self,
        progress_callback: Optional[callable] = None,
        workflow_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行工作流

        Args:
            progress_callback: 进度回调函数，接收 (node_id, status, progress) 参数
            workflow_id: 工作流 ID，用于检查点保存

        Returns:
            执行结果字典

        Raises:
            WorkflowExecutionError: 执行失败
        """
        logger.info("Starting workflow execution...")

        # 启动资源监控
        if self.resource_monitor:
            await self.resource_monitor.start()

        try:
            # 验证工作流
            is_valid, errors = self.validate()
            if not is_valid:
                raise WorkflowExecutionError(f"工作流验证失败: {errors}")

            # 获取执行顺序
            execution_order = self._topological_sort()
            logger.info(f"Execution order: {execution_order}")

            # 执行节点
            results = {}
            for i, node_id in enumerate(execution_order):
                node = self.nodes[node_id]
                logger.info(f"Executing node {node_id} ({node.type})...")

                try:
                    # 更新状态
                    node.status = NodeStatus.RUNNING
                    if progress_callback:
                        await progress_callback(
                            node_id, NodeStatus.RUNNING, (i + 1) / len(execution_order)
                        )

                    # 收集输入数据
                    inputs = self._collect_inputs(node)

                    # 执行节点
                    result = await self._execute_node(node, inputs)
                    node.result = result
                    node.status = NodeStatus.COMPLETED

                    # 更新输出端口的值
                    for port_id, port in node.outputs.items():
                        if isinstance(result, dict) and port_id in result:
                            port.value = result[port_id]
                        else:
                            port.value = result

                    results[node_id] = result

                    if progress_callback:
                        await progress_callback(
                            node_id, NodeStatus.COMPLETED, (i + 1) / len(execution_order)
                        )

                    logger.info(f"Node {node_id} completed successfully")

                    # 保存检查点（每个节点完成后）
                    if self.checkpoint_manager and workflow_id:
                        await self._save_checkpoint(workflow_id, results)

                except Exception as e:
                    node.status = NodeStatus.FAILED
                    node.error = str(e)
                    logger.error(f"Node {node_id} failed: {e}")

                    if progress_callback:
                        await progress_callback(node_id, NodeStatus.FAILED, None)

                    # 保存失败状态的检查点
                    if self.checkpoint_manager and workflow_id:
                        await self._save_checkpoint(workflow_id, results, failed=True)

                    raise WorkflowExecutionError(f"节点 {node_id} 执行失败: {e}") from e

            logger.info("Workflow execution completed successfully")
            return results

        finally:
            # 停止资源监控
            if self.resource_monitor:
                await self.resource_monitor.stop()

    def _collect_inputs(self, node: Node) -> Dict[str, Any]:
        """收集节点的输入数据"""
        inputs = {}

        for edge in self.edges:
            if edge.target == node.id:
                source_node = self.nodes[edge.source]
                source_port = source_node.outputs.get(edge.source_handle)

                if source_port and source_port.value is not None:
                    inputs[edge.target_handle] = source_port.value

        return inputs

    async def _execute_node(self, node: Node, inputs: Dict[str, Any]) -> Any:
        """
        执行单个节点

        Args:
            node: 要执行的节点
            inputs: 输入数据

        Returns:
            节点执行结果
        """
        if node.type == "dataLoader":
            return await self._execute_data_loader(node, inputs)
        elif node.type == "model":
            return await self._execute_model(node, inputs)
        elif node.type == "training":
            return await self._execute_training(node, inputs)
        elif node.type == "evaluation":
            return await self._execute_evaluation(node, inputs)
        else:
            raise WorkflowExecutionError(f"未知的节点类型: {node.type}")

    async def _execute_data_loader(
        self, node: Node, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行数据加载节点"""
        try:
            executor = self.executor_factory.create("dataLoader", self.data_dir)
            result = await executor.execute(node.data, inputs)
            return result
        except NodeExecutionError as e:
            logger.error(f"DataLoader node execution failed: {e}")
            raise WorkflowExecutionError(f"数据加载节点执行失败: {e}") from e

    async def _execute_model(
        self, node: Node, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行模型构建节点"""
        try:
            executor = self.executor_factory.create("model", self.data_dir)
            result = await executor.execute(node.data, inputs)
            return result
        except NodeExecutionError as e:
            logger.error(f"Model node execution failed: {e}")
            raise WorkflowExecutionError(f"模型构建节点执行失败: {e}") from e

    async def _execute_training(
        self, node: Node, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行训练节点"""
        try:
            executor = self.executor_factory.create("training", self.data_dir)
            result = await executor.execute(node.data, inputs)
            return result
        except NodeExecutionError as e:
            logger.error(f"Training node execution failed: {e}")
            raise WorkflowExecutionError(f"训练节点执行失败: {e}") from e

    async def _execute_evaluation(
        self, node: Node, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行评估节点"""
        try:
            executor = self.executor_factory.create("evaluation", self.data_dir)
            result = await executor.execute(node.data, inputs)
            return result
        except NodeExecutionError as e:
            logger.error(f"Evaluation node execution failed: {e}")
            raise WorkflowExecutionError(f"评估节点执行失败: {e}") from e

    async def _save_checkpoint(
        self, workflow_id: str, results: Dict[str, Any], failed: bool = False
    ) -> None:
        """保存检查点"""
        try:
            execution_state = {
                node_id: {"status": node.status.value, "error": node.error, "result": node.result}
                for node_id, node in self.nodes.items()
            }

            metadata = {
                "failed": failed,
                "completed_nodes": sum(
                    1 for node in self.nodes.values() if node.status == NodeStatus.COMPLETED
                ),
                "total_nodes": len(self.nodes),
            }

            # 在线程池中执行保存操作
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.checkpoint_manager.save_checkpoint,
                workflow_id,
                {node_id: {"id": node_id, "type": node.type, "data": node.data, "position": node.position} for node_id, node in self.nodes.items()},
                [{"id": edge.id, "source": edge.source, "target": edge.target, "sourceHandle": edge.source_handle, "targetHandle": edge.target_handle} for edge in self.edges],
                execution_state,
                metadata,
            )

            logger.info(f"Checkpoint saved for workflow {workflow_id}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def get_status(self) -> Dict[str, Any]:
        """获取工作流执行状态"""
        status = {
            "nodes": {
                node_id: {
                    "status": node.status.value,
                    "error": node.error,
                }
                for node_id, node in self.nodes.items()
            },
            "total": len(self.nodes),
            "completed": sum(
                1 for node in self.nodes.values() if node.status == NodeStatus.COMPLETED
            ),
            "failed": sum(
                1 for node in self.nodes.values() if node.status == NodeStatus.FAILED
            ),
        }

        # 添加资源监控信息
        if self.resource_monitor:
            status["resources"] = self.resource_monitor.get_current_status()

        return status

    def get_resource_statistics(self, duration: Optional[float] = None) -> Dict[str, Any]:
        """获取资源统计信息"""
        if not self.resource_monitor:
            return {}
        return self.resource_monitor.get_statistics(duration)
