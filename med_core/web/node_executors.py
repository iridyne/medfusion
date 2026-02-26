"""
节点执行器模块

实现工作流节点的真实执行逻辑，集成 MedFusion 核心功能。
"""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class NodeExecutionError(Exception):
    """节点执行错误"""

    pass


class NodeExecutor:
    """节点执行器基类"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"NodeExecutor initialized with device: {self.device}")

    async def execute(
        self,
        node_data: dict[str, Any],
        inputs: dict[str, Any],
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """
        执行节点

        Args:
            node_data: 节点配置数据
            inputs: 输入数据（来自上游节点）
            progress_callback: 进度回调函数

        Returns:
            执行结果字典
        """
        # TODO: 子类需要实现具体的执行逻辑
        raise NotImplementedError


class DataLoaderExecutor(NodeExecutor):
    """数据加载器执行器"""

    async def execute(
        self,
        node_data: dict[str, Any],
        inputs: dict[str, Any],
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """
        加载数据集

        Args:
            node_data: 包含 datasetId, split, batchSize 等配置
            inputs: 空（数据加载器是起始节点）
            progress_callback: 进度回调

        Returns:
            包含 dataset 和 dataset_info 的字典
        """
        try:
            dataset_id = node_data.get("datasetId")
            if not dataset_id:
                raise NodeExecutionError("缺少 datasetId 参数")

            split = node_data.get("split", "train")
            batch_size = node_data.get("batchSize", 32)
            shuffle = node_data.get("shuffle", True if split == "train" else False)
            num_workers = node_data.get("numWorkers", 4)
            seed = node_data.get("seed")

            logger.info(
                f"Loading dataset {dataset_id}, split={split}, batch_size={batch_size}"
            )

            # 数据集路径
            dataset_path = self.data_dir / "datasets" / dataset_id

            if not dataset_path.exists():
                raise NodeExecutionError(f"数据集不存在: {dataset_path}")

            # 动态导入以避免循环依赖
            from med_core.data.dataset import create_dataset

            # 创建数据集
            dataset = await asyncio.get_event_loop().run_in_executor(
                None, create_dataset, str(dataset_path), split
            )

            # 设置随机种子
            if seed is not None:
                torch.manual_seed(seed)

            # 创建 DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True if self.device.type == "cuda" else False,
                drop_last=False,
            )

            logger.info(
                f"Dataset loaded: {len(dataset)} samples, {len(dataloader)} batches"
            )

            return {
                "dataset": dataloader,
                "dataset_info": {
                    "id": dataset_id,
                    "split": split,
                    "size": len(dataset),
                    "batch_size": batch_size,
                    "num_batches": len(dataloader),
                    "num_workers": num_workers,
                },
            }

        except Exception as e:
            logger.error(f"DataLoader execution failed: {e}")
            raise NodeExecutionError(f"数据加载失败: {e}") from e


class ModelExecutor(NodeExecutor):
    """模型构建执行器"""

    async def execute(
        self,
        node_data: dict[str, Any],
        inputs: dict[str, Any],
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """
        构建模型

        Args:
            node_data: 包含 backbone, fusion, aggregator 等配置
            inputs: 可选的数据集信息（用于推断输入维度）
            progress_callback: 进度回调

        Returns:
            包含 model 和 model_config 的字典
        """
        try:
            backbone = node_data.get("backbone", "resnet18")
            fusion = node_data.get("fusion", "concatenate")
            aggregator = node_data.get("aggregator", "mean")
            num_classes = node_data.get("numClasses", 2)
            hidden_dim = node_data.get("hiddenDim", 512)
            pretrained = node_data.get("pretrained", True)

            logger.info(f"Building model: backbone={backbone}, fusion={fusion}")

            # 动态导入
            from med_core.models import ModelFactory

            # 创建模型配置
            config = {
                "backbone": backbone,
                "fusion_strategy": fusion,
                "aggregator": aggregator,
                "num_classes": num_classes,
                "hidden_dim": hidden_dim,
                "pretrained": pretrained,
            }

            # 创建模型
            model = await asyncio.get_event_loop().run_in_executor(
                None, ModelFactory.create, config
            )

            # 移动到设备
            model = model.to(self.device)

            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            logger.info(
                f"Model created: {total_params:,} params ({trainable_params:,} trainable)"
            )

            return {
                "model": model,
                "model_config": config,
                "model_info": {
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "device": str(self.device),
                },
            }

        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            raise NodeExecutionError(f"模型构建失败: {e}") from e


class TrainingExecutor(NodeExecutor):
    """训练执行器"""

    async def execute(
        self,
        node_data: dict[str, Any],
        inputs: dict[str, Any],
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """
        训练模型

        Args:
            node_data: 包含 epochs, learningRate, optimizer 等配置
            inputs: 包含 model 和 dataset
            progress_callback: 进度回调

        Returns:
            包含 trained_model 和 history 的字典
        """
        try:
            model = inputs.get("model")
            dataloader = inputs.get("dataset")

            if model is None:
                raise NodeExecutionError("缺少输入: model")
            if dataloader is None:
                raise NodeExecutionError("缺少输入: dataset")

            # 训练配置
            epochs = node_data.get("epochs", 10)
            learning_rate = node_data.get("learningRate", 1e-4)
            optimizer_name = node_data.get("optimizer", "adam")
            scheduler_name = node_data.get("scheduler")
            use_amp = node_data.get("useAmp", False)
            gradient_accumulation_steps = node_data.get("gradientAccumulationSteps", 1)
            early_stopping_patience = node_data.get("earlyStoppingPatience")
            checkpoint_dir = node_data.get("checkpointDir")

            logger.info(
                f"Starting training: epochs={epochs}, lr={learning_rate}, optimizer={optimizer_name}"
            )

            # 动态导入
            from med_core.training.trainer import Trainer

            # 创建训练器
            trainer = Trainer(
                model=model,
                train_loader=dataloader,
                epochs=epochs,
                learning_rate=learning_rate,
                optimizer=optimizer_name,
                scheduler=scheduler_name,
                device=self.device,
                use_amp=use_amp,
                gradient_accumulation_steps=gradient_accumulation_steps,
                early_stopping_patience=early_stopping_patience,
                checkpoint_dir=checkpoint_dir,
            )

            # 异步训练
            history = await self._train_async(trainer, progress_callback)

            logger.info(
                f"Training completed: final loss={history['loss'][-1]:.4f}, "
                f"final acc={history.get('accuracy', [0])[-1]:.4f}"
            )

            return {
                "trained_model": model,
                "history": history,
                "final_metrics": {
                    "train_loss": history["loss"][-1],
                    "train_acc": history.get("accuracy", [0])[-1],
                    "epochs_trained": len(history["loss"]),
                },
            }

        except Exception as e:
            logger.error(f"Training execution failed: {e}")
            raise NodeExecutionError(f"训练失败: {e}") from e

    async def _train_async(self, trainer: Any, progress_callback: Callable | None = None) -> Any:
        """异步训练"""
        loop = asyncio.get_event_loop()

        # 如果有进度回调，设置训练器的回调
        if progress_callback:
            trainer.set_progress_callback(progress_callback)

        # 在线程池中运行训练
        history = await loop.run_in_executor(None, trainer.train)
        return history


class EvaluationExecutor(NodeExecutor):
    """评估执行器"""

    async def execute(
        self,
        node_data: dict[str, Any],
        inputs: dict[str, Any],
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """
        评估模型

        Args:
            node_data: 包含 metrics, saveResults 等配置
            inputs: 包含 model 和 test_data
            progress_callback: 进度回调

        Returns:
            包含 metrics 和 report 的字典
        """
        try:
            model = inputs.get("model")
            test_data = inputs.get("test_data")

            if model is None:
                raise NodeExecutionError("缺少输入: model")
            if test_data is None:
                raise NodeExecutionError("缺少输入: test_data")

            # 评估配置
            metrics = node_data.get("metrics", ["accuracy", "loss"])
            save_results = node_data.get("saveResults", False)
            output_dir = node_data.get("outputDir")

            logger.info(f"Starting evaluation: metrics={metrics}")

            # 动态导入
            from med_core.evaluation.evaluator import Evaluator

            # 创建评估器
            evaluator = Evaluator(
                model=model, test_loader=test_data, device=self.device
            )

            # 异步评估
            results = await self._evaluate_async(evaluator, metrics)

            # 保存结果
            if save_results and output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                evaluator.save_results(results, output_path)
                logger.info(f"Results saved to {output_path}")

            logger.info(f"Evaluation completed: {results['metrics']}")

            return {"metrics": results["metrics"], "report": results.get("report", {})}

        except Exception as e:
            logger.error(f"Evaluation execution failed: {e}")
            raise NodeExecutionError(f"评估失败: {e}") from e

    async def _evaluate_async(self, evaluator: Any, metrics: Any) -> Any:
        """异步评估"""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, evaluator.evaluate, metrics)
        return results


# 执行器工厂
class ExecutorFactory:
    """执行器工厂"""

    _executors = {
        "dataLoader": DataLoaderExecutor,
        "model": ModelExecutor,
        "training": TrainingExecutor,
        "evaluation": EvaluationExecutor,
    }

    @classmethod
    def create(cls, node_type: str, data_dir: Path) -> NodeExecutor:
        """
        创建执行器

        Args:
            node_type: 节点类型
            data_dir: 数据目录

        Returns:
            节点执行器实例
        """
        executor_class = cls._executors.get(node_type)
        if executor_class is None:
            raise ValueError(f"未知的节点类型: {node_type}")

        return executor_class(data_dir)

    @classmethod
    def register(cls, node_type: str, executor_class: type) -> None:
        """注册新的执行器"""
        cls._executors[node_type] = executor_class
