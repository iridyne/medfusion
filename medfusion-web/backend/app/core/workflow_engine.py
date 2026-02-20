"""工作流执行引擎

支持依赖解析、并行执行、错误处理
"""
from typing import Dict, Any, List, Set, Optional
from collections import defaultdict, deque
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from app.core.node_registry import node_registry


class NodeStatus(str, Enum):
    """节点执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeExecution:
    """节点执行记录"""
    node_id: str
    node_type: str
    status: NodeStatus = NodeStatus.PENDING
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration(self) -> Optional[float]:
        """执行时长（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class WorkflowEngine:
    """工作流执行引擎
    
    功能:
    - 依赖关系解析
    - 拓扑排序
    - 并行执行
    - 错误处理
    - 执行状态跟踪
    """
    
    def __init__(self, workflow: Dict[str, Any]):
        """
        Args:
            workflow: 工作流定义，包含 nodes 和 edges
        """
        self.workflow = workflow
        self.nodes = {node["id"]: node for node in workflow["nodes"]}
        self.edges = workflow["edges"]
        
        # 构建依赖图
        self.dependencies = self._build_dependency_graph()
        self.reverse_dependencies = self._build_reverse_dependency_graph()
        
        # 执行状态
        self.executions: Dict[str, NodeExecution] = {}
        self.node_outputs: Dict[str, Dict[str, Any]] = {}
        
    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """构建依赖图：node_id -> 依赖的 node_ids"""
        dependencies = defaultdict(set)
        
        for edge in self.edges:
            target = edge["target"]
            source = edge["source"]
            dependencies[target].add(source)
        
        # 确保所有节点都在图中
        for node_id in self.nodes:
            if node_id not in dependencies:
                dependencies[node_id] = set()
        
        return dict(dependencies)
    
    def _build_reverse_dependency_graph(self) -> Dict[str, Set[str]]:
        """构建反向依赖图：node_id -> 依赖它的 node_ids"""
        reverse_deps = defaultdict(set)
        
        for edge in self.edges:
            source = edge["source"]
            target = edge["target"]
            reverse_deps[source].add(target)
        
        return dict(reverse_deps)
    
    def _topological_sort(self) -> List[List[str]]:
        """拓扑排序，返回可并行执行的节点分层
        
        Returns:
            List[List[str]]: 每层包含可并行执行的节点 ID
        """
        # 计算入度
        in_degree = {node_id: len(deps) for node_id, deps in self.dependencies.items()}
        
        # 找到所有入度为 0 的节点（起始节点）
        layers = []
        current_layer = [node_id for node_id, degree in in_degree.items() if degree == 0]
        
        while current_layer:
            layers.append(current_layer)
            next_layer = []
            
            # 处理当前层的节点
            for node_id in current_layer:
                # 减少依赖此节点的节点的入度
                for dependent in self.reverse_dependencies.get(node_id, []):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_layer.append(dependent)
            
            current_layer = next_layer
        
        # 检查是否有环
        if sum(in_degree.values()) > 0:
            raise ValueError("工作流包含循环依赖")
        
        return layers
    
    def _get_node_inputs(self, node_id: str) -> Dict[str, Any]:
        """获取节点的输入数据
        
        从前置节点的输出中收集数据
        """
        node = self.nodes[node_id]
        node_config = node.get("data", {}).get("config", {})
        
        # 从配置中获取静态输入
        inputs = dict(node_config)
        
        # 从前置节点获取动态输入
        for edge in self.edges:
            if edge["target"] == node_id:
                source_id = edge["source"]
                source_handle = edge.get("sourceHandle")
                target_handle = edge.get("targetHandle")
                
                # 获取源节点的输出
                if source_id in self.node_outputs:
                    source_output = self.node_outputs[source_id]
                    
                    if source_handle and source_handle in source_output:
                        # 使用指定的输出
                        value = source_output[source_handle]
                    else:
                        # 使用整个输出
                        value = source_output
                    
                    # 设置到目标输入
                    if target_handle:
                        inputs[target_handle] = value
                    else:
                        # 合并所有输出
                        if isinstance(value, dict):
                            inputs.update(value)
        
        return inputs
    
    async def _execute_node(self, node_id: str) -> NodeExecution:
        """执行单个节点"""
        node = self.nodes[node_id]
        node_type = node["type"]
        
        # 创建执行记录
        execution = NodeExecution(
            node_id=node_id,
            node_type=node_type,
            start_time=datetime.now()
        )
        
        try:
            # 获取节点类
            node_class = node_registry.get(node_type)
            if not node_class:
                raise ValueError(f"未知的节点类型: {node_type}")
            
            # 创建节点实例
            node_instance = node_class()
            
            # 获取输入
            inputs = self._get_node_inputs(node_id)
            execution.inputs = inputs
            
            # 执行节点
            execution.status = NodeStatus.RUNNING
            outputs = await node_instance.execute(inputs)
            
            # 保存输出
            execution.outputs = outputs
            execution.status = NodeStatus.COMPLETED
            self.node_outputs[node_id] = outputs
            
        except Exception as e:
            execution.status = NodeStatus.FAILED
            execution.error = str(e)
            raise
        
        finally:
            execution.end_time = datetime.now()
        
        return execution
    
    async def execute(
        self,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """执行工作流
        
        Args:
            progress_callback: 进度回调函数，接收 (node_id, status, execution) 参数
            
        Returns:
            执行结果，包含所有节点的输出和执行统计
        """
        # 拓扑排序
        try:
            layers = self._topological_sort()
        except ValueError as e:
            return {
                "status": "error",
                "error": str(e),
                "executions": {},
            }
        
        # 按层执行
        total_nodes = len(self.nodes)
        completed_nodes = 0
        
        for layer_idx, layer in enumerate(layers):
            # 并行执行当前层的所有节点
            tasks = []
            
            for node_id in layer:
                task = self._execute_node(node_id)
                tasks.append((node_id, task))
            
            # 并行等待当前层所有节点完成
            results = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            # 处理执行结果
            for (node_id, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    # 节点执行失败
                    execution = self.executions.get(node_id)
                    if not execution:
                        execution = NodeExecution(
                            node_id=node_id,
                            node_type=self.nodes[node_id]["type"],
                            status=NodeStatus.FAILED,
                            error=str(result)
                        )
                        self.executions[node_id] = execution
                    else:
                        execution.status = NodeStatus.FAILED
                        execution.error = str(result)
                    
                    # 标记依赖此节点的所有节点为跳过
                    self._mark_dependent_nodes_skipped(node_id)
                    
                    # 调用进度回调
                    if progress_callback:
                        await progress_callback(
                            node_id=node_id,
                            status=NodeStatus.FAILED,
                            execution=execution,
                            progress=completed_nodes / total_nodes * 100
                        )
                    
                    # 返回错误结果
                    return {
                        "status": "error",
                        "error": f"节点 {node_id} 执行失败: {str(result)}",
                        "failed_node": node_id,
                        "executions": {
                            nid: {
                                "status": exec.status,
                                "inputs": exec.inputs,
                                "outputs": exec.outputs,
                                "error": exec.error,
                                "duration": exec.duration,
                            }
                            for nid, exec in self.executions.items()
                        },
                    }
                else:
                    # 节点执行成功
                    execution = result
                    self.executions[node_id] = execution
                    completed_nodes += 1
                    
                    # 调用进度回调
                    if progress_callback:
                        await progress_callback(
                            node_id=node_id,
                            status=execution.status,
                            execution=execution,
                            progress=completed_nodes / total_nodes * 100
                        )
        
        # 所有节点执行成功
        return {
            "status": "success",
            "executions": {
                node_id: {
                    "status": exec.status,
                    "inputs": exec.inputs,
                    "outputs": exec.outputs,
                    "error": exec.error,
                    "duration": exec.duration,
                }
                for node_id, exec in self.executions.items()
            },
            "outputs": self.node_outputs,
            "statistics": {
                "total_nodes": total_nodes,
                "completed_nodes": completed_nodes,
                "total_duration": sum(
                    exec.duration for exec in self.executions.values()
                    if exec.duration is not None
                ),
            },
        }
    
    def _mark_dependent_nodes_skipped(self, failed_node_id: str):
        """标记依赖失败节点的所有节点为跳过状态"""
        to_skip = deque([failed_node_id])
        skipped = set()
        
        while to_skip:
            node_id = to_skip.popleft()
            
            for dependent in self.reverse_dependencies.get(node_id, []):
                if dependent not in skipped:
                    skipped.add(dependent)
                    to_skip.append(dependent)
                    
                    # 创建跳过的执行记录
                    self.executions[dependent] = NodeExecution(
                        node_id=dependent,
                        node_type=self.nodes[dependent]["type"],
                        status=NodeStatus.SKIPPED,
                        error=f"依赖的节点 {node_id} 执行失败"
                    )
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        status_counts = defaultdict(int)
        for execution in self.executions.values():
            status_counts[execution.status] += 1
        
        return {
            "total_nodes": len(self.nodes),
            "executed_nodes": len(self.executions),
            "status_counts": dict(status_counts),
            "executions": {
                node_id: {
                    "status": exec.status,
                    "duration": exec.duration,
                    "error": exec.error,
                }
                for node_id, exec in self.executions.items()
            },
        }
