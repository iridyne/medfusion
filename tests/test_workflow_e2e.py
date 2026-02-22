"""
工作流端到端测试

测试完整的工作流执行流程，包括：
- 数据加载
- 模型构建
- 训练
- 评估
- 检查点保存和恢复
- 资源监控
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip this test file if med_core.web.workflow_engine cannot be imported
pytest.importorskip("med_core.web.workflow_engine")

from med_core.web.workflow_engine import (
    NodeStatus,
    WorkflowEngine,
    WorkflowExecutionError,
)


@pytest.fixture
def temp_data_dir():
    """创建临时数据目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # 创建必要的子目录
        (data_dir / "datasets").mkdir()
        (data_dir / "checkpoints").mkdir()
        yield data_dir


@pytest.fixture
def simple_workflow():
    """简单的工作流定义"""
    return {
        "nodes": [
            {
                "id": "data_loader_1",
                "type": "dataLoader",
                "data": {
                    "datasetId": "test_dataset",
                    "split": "train",
                    "batchSize": 32,
                },
                "position": {"x": 100, "y": 100},
            },
            {
                "id": "model_1",
                "type": "model",
                "data": {
                    "backbone": "resnet18",
                    "fusion": "concatenate",
                    "numClasses": 2,
                },
                "position": {"x": 300, "y": 100},
            },
            {
                "id": "training_1",
                "type": "training",
                "data": {
                    "epochs": 5,
                    "learningRate": 0.001,
                    "optimizer": "adam",
                },
                "position": {"x": 500, "y": 100},
            },
            {
                "id": "evaluation_1",
                "type": "evaluation",
                "data": {
                    "metrics": ["accuracy", "loss"],
                },
                "position": {"x": 700, "y": 100},
            },
        ],
        "edges": [
            {
                "id": "e1",
                "source": "data_loader_1",
                "target": "model_1",
                "sourceHandle": "dataset",
                "targetHandle": "image_data",
            },
            {
                "id": "e2",
                "source": "model_1",
                "target": "training_1",
                "sourceHandle": "model",
                "targetHandle": "model",
            },
            {
                "id": "e3",
                "source": "data_loader_1",
                "target": "training_1",
                "sourceHandle": "dataset",
                "targetHandle": "dataset",
            },
            {
                "id": "e4",
                "source": "training_1",
                "target": "evaluation_1",
                "sourceHandle": "trained_model",
                "targetHandle": "model",
            },
            {
                "id": "e5",
                "source": "data_loader_1",
                "target": "evaluation_1",
                "sourceHandle": "dataset",
                "targetHandle": "test_data",
            },
        ],
    }


@pytest.fixture
def cyclic_workflow():
    """包含循环依赖的工作流"""
    return {
        "nodes": [
            {
                "id": "node_1",
                "type": "dataLoader",
                "data": {},
                "position": {"x": 100, "y": 100},
            },
            {
                "id": "node_2",
                "type": "model",
                "data": {},
                "position": {"x": 300, "y": 100},
            },
            {
                "id": "node_3",
                "type": "training",
                "data": {},
                "position": {"x": 500, "y": 100},
            },
        ],
        "edges": [
            {"id": "e1", "source": "node_1", "target": "node_2"},
            {"id": "e2", "source": "node_2", "target": "node_3"},
            {"id": "e3", "source": "node_3", "target": "node_1"},  # 循环
        ],
    }


class TestWorkflowValidation:
    """工作流验证测试"""

    def test_valid_workflow(self, temp_data_dir, simple_workflow):
        """测试有效的工作流"""
        engine = WorkflowEngine(data_dir=temp_data_dir, enable_monitoring=False)
        engine.load_workflow(simple_workflow)

        is_valid, errors = engine.validate()
        assert is_valid
        assert len(errors) == 0

    def test_cyclic_workflow(self, temp_data_dir, cyclic_workflow):
        """测试循环依赖检测"""
        engine = WorkflowEngine(data_dir=temp_data_dir, enable_monitoring=False)
        engine.load_workflow(cyclic_workflow)

        is_valid, errors = engine.validate()
        assert not is_valid
        assert any("循环依赖" in error for error in errors)

    def test_empty_workflow(self, temp_data_dir):
        """测试空工作流"""
        engine = WorkflowEngine(data_dir=temp_data_dir, enable_monitoring=False)
        engine.load_workflow({"nodes": [], "edges": []})

        is_valid, errors = engine.validate()
        assert not is_valid
        assert any("工作流为空" in error for error in errors)

    def test_invalid_edge(self, temp_data_dir):
        """测试无效的边"""
        workflow = {
            "nodes": [
                {
                    "id": "node_1",
                    "type": "dataLoader",
                    "data": {},
                    "position": {"x": 100, "y": 100},
                }
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "node_1",
                    "target": "nonexistent_node",
                }
            ],
        }

        engine = WorkflowEngine(data_dir=temp_data_dir, enable_monitoring=False)
        engine.load_workflow(workflow)

        is_valid, errors = engine.validate()
        assert not is_valid
        assert any("不存在" in error for error in errors)


class TestWorkflowExecution:
    """工作流执行测试"""

    @pytest.mark.asyncio
    @patch("med_core.web.node_executors.DataLoaderExecutor.execute")
    @patch("med_core.web.node_executors.ModelExecutor.execute")
    @patch("med_core.web.node_executors.TrainingExecutor.execute")
    @patch("med_core.web.node_executors.EvaluationExecutor.execute")
    async def test_simple_workflow_execution(
        self,
        mock_eval,
        mock_train,
        mock_model,
        mock_data,
        temp_data_dir,
        simple_workflow,
    ):
        """测试简单工作流执行"""
        # Mock 执行器返回值
        mock_data.return_value = {
            "dataset": MagicMock(),
            "dataset_info": {"size": 100},
        }
        mock_model.return_value = {
            "model": MagicMock(),
            "model_config": {},
        }
        mock_train.return_value = {
            "trained_model": MagicMock(),
            "history": {"loss": [0.5, 0.4, 0.3]},
        }
        mock_eval.return_value = {
            "metrics": {"accuracy": 0.9},
            "report": {},
        }

        engine = WorkflowEngine(
            data_dir=temp_data_dir,
            enable_checkpoints=False,
            enable_monitoring=False,
        )
        engine.load_workflow(simple_workflow)

        # 执行工作流
        results = await engine.execute()

        # 验证结果
        assert len(results) == 4
        assert "data_loader_1" in results
        assert "model_1" in results
        assert "training_1" in results
        assert "evaluation_1" in results

        # 验证所有节点都成功完成
        for node in engine.nodes.values():
            assert node.status == NodeStatus.COMPLETED

    @pytest.mark.asyncio
    @patch("med_core.web.node_executors.DataLoaderExecutor.execute")
    async def test_execution_with_failure(
        self, mock_data, temp_data_dir, simple_workflow
    ):
        """测试执行失败的情况"""
        # Mock 执行器抛出异常
        mock_data.side_effect = Exception("数据加载失败")

        engine = WorkflowEngine(
            data_dir=temp_data_dir,
            enable_checkpoints=False,
            enable_monitoring=False,
        )
        engine.load_workflow(simple_workflow)

        # 执行应该失败
        with pytest.raises(WorkflowExecutionError):
            await engine.execute()

        # 验证失败节点的状态
        assert engine.nodes["data_loader_1"].status == NodeStatus.FAILED
        assert engine.nodes["data_loader_1"].error is not None

    @pytest.mark.asyncio
    @patch("med_core.web.node_executors.DataLoaderExecutor.execute")
    @patch("med_core.web.node_executors.ModelExecutor.execute")
    async def test_execution_order(
        self, mock_model, mock_data, temp_data_dir, simple_workflow
    ):
        """测试执行顺序"""
        execution_order = []

        async def track_data(*args, **kwargs):
            execution_order.append("data_loader_1")
            return {"dataset": MagicMock(), "dataset_info": {}}

        async def track_model(*args, **kwargs):
            execution_order.append("model_1")
            return {"model": MagicMock(), "model_config": {}}

        mock_data.side_effect = track_data
        mock_model.side_effect = track_model

        engine = WorkflowEngine(
            data_dir=temp_data_dir,
            enable_checkpoints=False,
            enable_monitoring=False,
        )
        engine.load_workflow(simple_workflow)

        # 只执行前两个节点
        workflow_subset = {
            "nodes": simple_workflow["nodes"][:2],
            "edges": simple_workflow["edges"][:1],
        }
        engine.load_workflow(workflow_subset)

        await engine.execute()

        # 验证执行顺序：data_loader 应该在 model 之前
        assert execution_order.index("data_loader_1") < execution_order.index("model_1")


class TestCheckpointManagement:
    """检查点管理测试"""

    @pytest.mark.asyncio
    @patch("med_core.web.node_executors.DataLoaderExecutor.execute")
    async def test_checkpoint_save(self, mock_data, temp_data_dir, simple_workflow):
        """测试检查点保存"""
        mock_data.return_value = {
            "dataset": MagicMock(),
            "dataset_info": {},
        }

        engine = WorkflowEngine(
            data_dir=temp_data_dir,
            enable_checkpoints=True,
            enable_monitoring=False,
        )

        # 只加载第一个节点
        workflow_subset = {
            "nodes": [simple_workflow["nodes"][0]],
            "edges": [],
        }
        engine.load_workflow(workflow_subset)

        workflow_id = "test_workflow_123"
        await engine.execute(workflow_id=workflow_id)

        # 验证检查点已保存
        checkpoints = engine.checkpoint_manager.list_checkpoints(workflow_id)
        assert len(checkpoints) > 0
        assert checkpoints[0]["workflow_id"] == workflow_id

    @pytest.mark.asyncio
    async def test_checkpoint_resume(self, temp_data_dir, simple_workflow):
        """测试从检查点恢复"""
        engine = WorkflowEngine(
            data_dir=temp_data_dir,
            enable_checkpoints=True,
            enable_monitoring=False,
        )

        workflow_id = "test_workflow_456"

        # 手动创建一个检查点
        checkpoint_path = engine.checkpoint_manager.save_checkpoint(
            workflow_id=workflow_id,
            nodes={node["id"]: node for node in simple_workflow["nodes"]},
            edges=simple_workflow["edges"],
            execution_state={},
            metadata={"test": True},
        )

        # 恢复检查点
        restored = engine.checkpoint_manager.load_checkpoint(checkpoint_path)

        assert restored["workflow"]["nodes"] is not None
        assert restored["workflow"]["edges"] is not None
        assert restored["state"]["workflow_id"] == workflow_id


class TestResourceMonitoring:
    """资源监控测试"""

    @pytest.mark.asyncio
    async def test_resource_monitor_start_stop(self, temp_data_dir):
        """测试资源监控启动和停止"""
        engine = WorkflowEngine(
            data_dir=temp_data_dir,
            enable_checkpoints=False,
            enable_monitoring=True,
        )

        assert engine.resource_monitor is not None

        # 启动监控
        await engine.resource_monitor.start()
        assert engine.resource_monitor.monitoring

        # 等待一些数据收集
        await asyncio.sleep(2)

        # 停止监控
        await engine.resource_monitor.stop()
        assert not engine.resource_monitor.monitoring

        # 验证有历史记录
        assert len(engine.resource_monitor.history) > 0

    @pytest.mark.asyncio
    async def test_resource_statistics(self, temp_data_dir):
        """测试资源统计"""
        engine = WorkflowEngine(
            data_dir=temp_data_dir,
            enable_checkpoints=False,
            enable_monitoring=True,
        )

        await engine.resource_monitor.start()
        await asyncio.sleep(2)
        await engine.resource_monitor.stop()

        # 获取统计信息
        stats = engine.get_resource_statistics()

        assert "cpu" in stats
        assert "memory" in stats
        assert "avg" in stats["cpu"]
        assert "max" in stats["cpu"]
        assert "min" in stats["cpu"]


class TestProgressCallback:
    """进度回调测试"""

    @pytest.mark.asyncio
    @patch("med_core.web.node_executors.DataLoaderExecutor.execute")
    async def test_progress_callback(self, mock_data, temp_data_dir, simple_workflow):
        """测试进度回调"""
        mock_data.return_value = {
            "dataset": MagicMock(),
            "dataset_info": {},
        }

        progress_updates = []

        async def progress_callback(node_id, status, progress):
            progress_updates.append(
                {
                    "node_id": node_id,
                    "status": status,
                    "progress": progress,
                }
            )

        engine = WorkflowEngine(
            data_dir=temp_data_dir,
            enable_checkpoints=False,
            enable_monitoring=False,
        )

        # 只加载第一个节点
        workflow_subset = {
            "nodes": [simple_workflow["nodes"][0]],
            "edges": [],
        }
        engine.load_workflow(workflow_subset)

        await engine.execute(progress_callback=progress_callback)

        # 验证进度回调被调用
        assert len(progress_updates) >= 2  # 至少 RUNNING 和 COMPLETED
        assert any(
            update["status"] == NodeStatus.RUNNING for update in progress_updates
        )
        assert any(
            update["status"] == NodeStatus.COMPLETED for update in progress_updates
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
