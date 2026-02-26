"""
检查点管理器

支持工作流执行的保存、恢复和中断恢复。
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """检查点管理器"""

    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 5):
        """
        初始化检查点管理器

        Args:
            checkpoint_dir: 检查点保存目录
            max_checkpoints: 最大保留检查点数量
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")

    def save_checkpoint(
        self,
        workflow_id: str,
        nodes: dict[str, Any],
        edges: list[dict[str, Any]],
        execution_state: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        保存检查点

        Args:
            workflow_id: 工作流 ID
            nodes: 节点状态字典
            edges: 边列表
            execution_state: 执行状态（包含中间结果）
            metadata: 额外的元数据

        Returns:
            检查点文件路径
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"{workflow_id}_{timestamp}"
            checkpoint_path = self.checkpoint_dir / checkpoint_name

            # 创建检查点目录
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            # 保存工作流定义
            workflow_data = {"nodes": nodes, "edges": edges}
            with open(checkpoint_path / "workflow.json", "w") as f:
                json.dump(workflow_data, f, indent=2)

            # 保存执行状态
            state_data = {
                "workflow_id": workflow_id,
                "timestamp": timestamp,
                "execution_state": self._serialize_execution_state(execution_state),
                "metadata": metadata or {},
            }
            with open(checkpoint_path / "state.json", "w") as f:
                json.dump(state_data, f, indent=2)

            # 保存模型权重（如果有）
            self._save_models(checkpoint_path, execution_state)

            # 清理旧检查点
            self._cleanup_old_checkpoints(workflow_id)

            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径

        Returns:
            包含工作流定义和执行状态的字典
        """
        try:
            checkpoint_path = Path(checkpoint_path)

            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            # 加载工作流定义
            with open(checkpoint_path / "workflow.json") as f:
                workflow_data = json.load(f)

            # 加载执行状态
            with open(checkpoint_path / "state.json") as f:
                state_data = json.load(f)

            # 加载模型权重
            models = self._load_models(checkpoint_path)
            state_data["models"] = models

            logger.info(f"Checkpoint loaded: {checkpoint_path}")

            return {
                "workflow": workflow_data,
                "state": state_data,
            }

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def list_checkpoints(self, workflow_id: str | None = None) -> list[dict[str, Any]]:
        """
        列出检查点

        Args:
            workflow_id: 可选的工作流 ID 过滤

        Returns:
            检查点信息列表
        """
        checkpoints = []

        for checkpoint_path in sorted(self.checkpoint_dir.iterdir(), reverse=True):
            if not checkpoint_path.is_dir():
                continue

            # 检查是否匹配工作流 ID
            if workflow_id and not checkpoint_path.name.startswith(workflow_id):
                continue

            try:
                # 读取状态文件
                state_file = checkpoint_path / "state.json"
                if state_file.exists():
                    with open(state_file) as f:
                        state_data = json.load(f)

                    checkpoints.append(
                        {
                            "path": str(checkpoint_path),
                            "name": checkpoint_path.name,
                            "workflow_id": state_data.get("workflow_id"),
                            "timestamp": state_data.get("timestamp"),
                            "metadata": state_data.get("metadata", {}),
                            "size": self._get_dir_size(checkpoint_path),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {checkpoint_path}: {e}")
                continue

        return checkpoints

    def get_latest_checkpoint(self, workflow_id: str) -> Path | None:
        """
        获取最新的检查点

        Args:
            workflow_id: 工作流 ID

        Returns:
            最新检查点路径，如果不存在则返回 None
        """
        checkpoints = self.list_checkpoints(workflow_id)
        if checkpoints:
            return Path(checkpoints[0]["path"])
        return None

    def delete_checkpoint(self, checkpoint_path: Path) -> None:
        """
        删除检查点

        Args:
            checkpoint_path: 检查点路径
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
                logger.info(f"Checkpoint deleted: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            raise

    def _cleanup_old_checkpoints(self, workflow_id: str) -> None:
        """清理旧检查点"""
        checkpoints = self.list_checkpoints(workflow_id)

        if len(checkpoints) > self.max_checkpoints:
            # 删除最旧的检查点
            for checkpoint in checkpoints[self.max_checkpoints :]:
                try:
                    self.delete_checkpoint(Path(checkpoint["path"]))
                    logger.info(f"Cleaned up old checkpoint: {checkpoint['name']}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup checkpoint: {e}")

    def _serialize_execution_state(
        self, execution_state: dict[str, Any]
    ) -> dict[str, Any]:
        """
        序列化执行状态

        将不可序列化的对象（如 PyTorch 模型、DataLoader）转换为可序列化的格式
        """
        serialized = {}

        for node_id, node_state in execution_state.items():
            serialized[node_id] = {
                "status": node_state.get("status"),
                "error": node_state.get("error"),
                "result_type": str(type(node_state.get("result"))),
                # 不保存实际的 PyTorch 对象，只保存元数据
                "has_model": "model" in str(node_state.get("result", {})),
                "has_dataset": "dataset" in str(node_state.get("result", {})),
            }

        return serialized

    def _save_models(
        self, checkpoint_path: Path, execution_state: dict[str, Any]
    ) -> None:
        """保存模型权重"""
        models_dir = checkpoint_path / "models"
        models_dir.mkdir(exist_ok=True)

        for node_id, node_state in execution_state.items():
            result = node_state.get("result")
            if result and isinstance(result, dict):
                # 保存模型
                if "model" in result:
                    model = result["model"]
                    if hasattr(model, "state_dict"):
                        model_path = models_dir / f"{node_id}_model.pt"
                        torch.save(model.state_dict(), model_path)
                        logger.debug(f"Model saved: {model_path}")

                # 保存训练后的模型
                if "trained_model" in result:
                    model = result["trained_model"]
                    if hasattr(model, "state_dict"):
                        model_path = models_dir / f"{node_id}_trained_model.pt"
                        torch.save(model.state_dict(), model_path)
                        logger.debug(f"Trained model saved: {model_path}")

                # 保存训练历史
                if "history" in result:
                    history_path = models_dir / f"{node_id}_history.json"
                    with open(history_path, "w") as f:
                        json.dump(result["history"], f, indent=2)
                    logger.debug(f"Training history saved: {history_path}")

    def _load_models(self, checkpoint_path: Path) -> dict[str, Any]:
        """加载模型权重"""
        models: dict[str, Any] = {}
        models_dir = checkpoint_path / "models"

        if not models_dir.exists():
            return models

        for model_file in models_dir.glob("*.pt"):
            node_id = model_file.stem.rsplit("_", 1)[0]
            model_type = model_file.stem.rsplit("_", 1)[1]

            # 只加载权重字典，不加载模型结构
            # 模型结构需要在恢复时重新构建
            state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
            models[f"{node_id}_{model_type}"] = state_dict
            logger.debug(f"Model weights loaded: {model_file}")

        # 加载训练历史
        for history_file in models_dir.glob("*_history.json"):
            node_id = history_file.stem.rsplit("_", 1)[0]
            with open(history_file) as f:
                history = json.load(f)
            models[f"{node_id}_history"] = history
            logger.debug(f"Training history loaded: {history_file}")

        return models

    def _get_dir_size(self, path: Path) -> int:
        """获取目录大小（字节）"""
        total_size = 0
        for item in path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
        return total_size

    def resume_workflow(
        self, workflow_id: str, checkpoint_path: Path | None = None
    ) -> dict[str, Any]:
        """
        恢复工作流执行

        Args:
            workflow_id: 工作流 ID
            checkpoint_path: 可选的检查点路径，如果不提供则使用最新的

        Returns:
            恢复的工作流数据和状态
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint(workflow_id)

        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint found for workflow {workflow_id}")

        logger.info(f"Resuming workflow {workflow_id} from {checkpoint_path}")
        return self.load_checkpoint(checkpoint_path)
