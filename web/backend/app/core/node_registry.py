"""节点注册表"""
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class NodeConfig(BaseModel):
    """节点配置"""
    pass


class NodePlugin(ABC):
    """节点插件基类"""

    name: str = "Base Node"
    category: str = "base"
    description: str = ""

    @property
    @abstractmethod
    def inputs(self) -> list[str]:
        """输入端口"""
        pass

    @property
    @abstractmethod
    def outputs(self) -> list[str]:
        """输出端口"""
        pass

    @abstractmethod
    async def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """执行节点逻辑"""
        pass

    def validate(self, config: dict[str, Any]) -> bool:
        """验证配置"""
        return True

    def get_schema(self) -> dict[str, Any]:
        """获取节点配置 schema"""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }


class NodeRegistry:
    """节点注册表"""

    def __init__(self):
        self._nodes: dict[str, type[NodePlugin]] = {}

    def register(self, node_type: str, node_class: type[NodePlugin]):
        """注册节点"""
        self._nodes[node_type] = node_class

    def get(self, node_type: str) -> type[NodePlugin]:
        """获取节点类"""
        if node_type not in self._nodes:
            raise ValueError(f"Unknown node type: {node_type}")
        return self._nodes[node_type]

    def list_nodes(self) -> list[dict[str, Any]]:
        """列出所有节点"""
        return [
            node_class().get_schema()
            for node_class in self._nodes.values()
        ]

    def get_by_category(self, category: str) -> list[dict[str, Any]]:
        """按类别获取节点"""
        return [
            node_class().get_schema()
            for node_class in self._nodes.values()
            if node_class().category == category
        ]


# 全局注册表
node_registry = NodeRegistry()


def register_node(node_type: str):
    """节点注册装饰器"""
    def decorator(node_class: type[NodePlugin]):
        node_registry.register(node_type, node_class)
        return node_class
    return decorator
