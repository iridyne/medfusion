"""训练服务

集成 med_core 训练器，提供真实的训练功能
"""
import asyncio
import sys
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

# 添加 med_core 到路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TrainingStatus(StrEnum):
    """训练状态"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TrainingService:
    """训练服务

    集成 med_core 训练器，提供：
    - 真实的模型训练
    - 进度回调
    - 训练控制（暂停/恢复/停止）
    - 指标收集
    """

    def __init__(self, job_id: str, config: dict[str, Any]):
        """
        Args:
            job_id: 训练任务 ID
            config: 训练配置
        """
        self.job_id = job_id
        self.config = config
        self.status = TrainingStatus.PENDING
        self.progress = 0.0
        self.current_epoch = 0
        self.total_epochs = config.get("training_config", {}).get("epochs", 10)
        self.metrics = {}
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.error: str | None = None

        # 控制标志
        self._should_stop = False
        self._should_pause = False
        self._is_paused = False

    async def run(self, progress_callback: Callable | None = None):
        """运行训练

        Args:
            progress_callback: 进度回调函数
        """
        self.status = TrainingStatus.INITIALIZING
        self.start_time = datetime.now()

        try:
            # 导入 med_core
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset

            from med_core.backbones import create_backbone
            from med_core.models import SimpleClassifier

            # 解析配置
            model_config = self.config.get("model_config", {})
            data_config = self.config.get("data_config", {})
            training_config = self.config.get("training_config", {})

            # 创建模型
            if progress_callback:
                await progress_callback({
                    "type": "status_update",
                    "status": "创建模型...",
                    "progress": 5,
                })

            backbone_name = model_config.get("backbone", "resnet18")
            num_classes = model_config.get("num_classes", 10)

            backbone = create_backbone(
                backbone_name,
                pretrained=model_config.get("pretrained", True),
                feature_dim=model_config.get("feature_dim", 128)
            )

            model = SimpleClassifier(
                backbone=backbone,
                num_classes=num_classes
            )

            # 设置设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            # 启用梯度检查点（如果配置）
            if training_config.get("gradient_checkpointing", False):
                backbone.enable_gradient_checkpointing()

            # 创建模拟数据（实际应用中应该加载真实数据）
            if progress_callback:
                await progress_callback({
                    "type": "status_update",
                    "status": "加载数据...",
                    "progress": 10,
                })

            batch_size = training_config.get("batch_size", 32)
            num_samples = data_config.get("num_samples", 1000)

            # 模拟数据
            X_train = torch.randn(num_samples, 3, 224, 224)
            y_train = torch.randint(0, num_classes, (num_samples,))
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            X_val = torch.randn(num_samples // 5, 3, 224, 224)
            y_val = torch.randint(0, num_classes, (num_samples // 5,))
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # 创建优化器和损失函数
            optimizer_name = training_config.get("optimizer", "adam")
            learning_rate = training_config.get("learning_rate", 0.001)

            if optimizer_name.lower() == "adam":
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_name.lower() == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            else:
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            criterion = nn.CrossEntropyLoss()

            # 学习率调度器
            scheduler = None
            if training_config.get("use_scheduler", False):
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.total_epochs
                )

            # 混合精度训练
            use_amp = training_config.get("use_amp", False)
            scaler = torch.cuda.amp.GradScaler() if use_amp else None

            # 训练循环
            self.status = TrainingStatus.RUNNING

            for epoch in range(self.total_epochs):
                # 检查停止标志
                if self._should_stop:
                    self.status = TrainingStatus.STOPPED
                    break

                # 检查暂停标志
                while self._should_pause:
                    self._is_paused = True
                    self.status = TrainingStatus.PAUSED
                    await asyncio.sleep(0.5)

                self._is_paused = False
                self.status = TrainingStatus.RUNNING
                self.current_epoch = epoch + 1

                # 训练阶段
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()

                    # 前向传播
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()

                    # 统计
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += targets.size(0)
                    train_correct += predicted.eq(targets).sum().item()

                    # 批次进度回调
                    if progress_callback and batch_idx % 5 == 0:
                        batch_progress = (epoch + batch_idx / len(train_loader)) / self.total_epochs * 100
                        await progress_callback({
                            "type": "batch_progress",
                            "epoch": epoch + 1,
                            "batch": batch_idx,
                            "total_batches": len(train_loader),
                            "loss": loss.item(),
                            "progress": batch_progress,
                        })

                train_loss /= len(train_loader)
                train_acc = 100.0 * train_correct / train_total

                # 验证阶段
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(device)
                        targets = targets.to(device)

                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()

                val_loss /= len(val_loader)
                val_acc = 100.0 * val_correct / val_total

                # 更新历史
                self.history["train_loss"].append(train_loss)
                self.history["train_acc"].append(train_acc)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                # 更新指标
                self.metrics = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }

                # 更新进度
                self.progress = (epoch + 1) / self.total_epochs * 100

                # 学习率调度
                if scheduler:
                    scheduler.step()

                # Epoch 进度回调
                if progress_callback:
                    await progress_callback({
                        "type": "epoch_completed",
                        "epoch": epoch + 1,
                        "total_epochs": self.total_epochs,
                        "metrics": self.metrics,
                        "progress": self.progress,
                    })

            # 训练完成
            if self.status != TrainingStatus.STOPPED:
                self.status = TrainingStatus.COMPLETED

            self.end_time = datetime.now()

            # 保存模型（可选）
            if training_config.get("save_model", False):
                output_dir = Path(training_config.get("output_dir", "./outputs"))
                output_dir.mkdir(parents=True, exist_ok=True)
                model_path = output_dir / f"{self.job_id}_model.pth"
                torch.save(model.state_dict(), model_path)

            if progress_callback:
                await progress_callback({
                    "type": "training_completed",
                    "status": self.status,
                    "final_metrics": self.metrics,
                    "history": self.history,
                })

        except Exception as e:
            self.status = TrainingStatus.FAILED
            self.error = str(e)
            self.end_time = datetime.now()

            if progress_callback:
                await progress_callback({
                    "type": "training_failed",
                    "error": str(e),
                })

            raise

    def stop(self):
        """停止训练"""
        self._should_stop = True

    def pause(self):
        """暂停训练"""
        self._should_pause = True

    def resume(self):
        """恢复训练"""
        self._should_pause = False

    def get_status(self) -> dict[str, Any]:
        """获取训练状态"""
        duration = None
        if self.start_time:
            end = self.end_time or datetime.now()
            duration = (end - self.start_time).total_seconds()

        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "metrics": self.metrics,
            "history": self.history,
            "duration": duration,
            "error": self.error,
            "is_paused": self._is_paused,
        }
