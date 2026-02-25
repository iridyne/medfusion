"""混合精度训练"""

from typing import Any

import torch


class AMPTrainer:
    """AMP 训练器"""

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()

    def train_step(
        self, data: torch.Tensor, target: torch.Tensor, criterion: Any
    ) -> float:
        with torch.cuda.amp.autocast():
            output = self.model(data)
            loss = criterion(output, target)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return loss.item()


def create_amp_trainer(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> AMPTrainer:
    return AMPTrainer(model, optimizer)
