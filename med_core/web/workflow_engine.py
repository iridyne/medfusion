"""
工作流执行引擎

实现节点化工作流的解析、验证和执行。
"""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

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

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.adjacency_list: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency_list: Dict[str, List[str]] = defaultdict(list)

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
        self, progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        执行工作流

        Args:
            progress_callback: 进度回调函数，接收 (node_id, status, progress) 参数

        Returns:
            执行结果字典

        Raises:
            WorkflowExecutionError: 执行失败
        """
        logger.info("Starting workflow execution...")

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

            except Exception as e:
                node.status = NodeStatus.FAILED
                node.error = str(e)
                logger.error(f"Node {node_id} failed: {e}")

                if progress_callback:
                    await progress_callback(node_id, NodeStatus.FAILED, None)

                raise WorkflowExecutionError(f"节点 {node_id} 执行失败: {e}") from e

        logger.info("Workflow execution completed successfully")
        return results

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

    async def _execute_model(
        self, node: Node, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行模型构建节点"""
        logger.info(f"Building model: {node.data.get('backbone')}")

        # TODO: 实际的模型构建逻辑
        return {
            "model": {
                "backbone": node.data.get("backbone", "resnet18"),
                "fusion": node.data.get("fusion", "concatenate"),
                "aggregator": node.data.get("aggregator", "mean"),
                "num_classes": node.data.get("numClasses", 2),
            }
        }

    async def _execute_training(
        self, node: Node, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行训练节点"""
        logger.info("Starting training...")

        model = inputs.get("model")
        dataset = inputs.get("dataset")

        if not model or not dataset:
            raise WorkflowExecutionError("训练节点缺少必需的输入（model 或 dataset）")

        # TODO: 实际的训练逻辑
        # 这里返回模拟结果
        return {
            "trained_model": {
                **model,
                "trained": True,
                "epochs": node.data.get("epochs", 10),
            },
            "history": {
                "loss": [0.5, 0.4, 0.3, 0.2, 0.1],
                "accuracy": [0.7, 0.75, 0.8, 0.85, 0.9],
            },
        }

    async def _execute_evaluation(
        self, node: Node, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行评估节点"""
        logger.info("Starting evaluation...")

        model = inputs.get("model")
        test_data = inputs.get("test_data")

        if not model or not test_data:
            raise WorkflowExecutionError("评估节点缺少必需的输入（model 或 test_data）")

        # TODO: 实际的评估逻辑
        return {
            "metrics": {
                "accuracy": 0.92,
                "precision": 0.91,
                "recall": 0.93,
                "f1": 0.92,
            },
            "report": {
                "confusion_matrix": [[45, 5], [3, 47]],
                "classification_report": "...",
            },
        }

    def get_status(self) -> Dict[str, Any]:
        """获取工作流执行状态"""
        return {
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
