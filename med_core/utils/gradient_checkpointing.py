"""
梯度检查点工具模块。

梯度检查点 (Gradient Checkpointing) 是一种以计算换内存的技术，
通过在反向传播时重新计算前向传播的中间激活值，而不是存储它们，
从而大幅降低内存占用。

适用场景：
- 训练大型模型时内存不足
- 希望使用更大的 batch size
- 模型层数很深

权衡：
- 优点：显著降低内存占用（通常可节省 30-50%）
- 缺点：增加约 20-30% 的训练时间
"""

import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


def apply_gradient_checkpointing(
    module: nn.Module,
    checkpoint_segments: int | None = None,
    use_reentrant: bool = True,
) -> None:
    """
    为模块启用梯度检查点。

    Args:
        module: 要应用梯度检查点的模块
        checkpoint_segments: 将模块分成多少段进行检查点（None 表示自动）
        use_reentrant: 是否使用可重入的检查点（PyTorch 2.0+ 推荐 False）

    Example:
        >>> model = ResNet50()
        >>> apply_gradient_checkpointing(model)
        >>> # 训练时内存占用会显著降低
    """
    if not hasattr(module, "_gradient_checkpointing_enabled"):
        module._gradient_checkpointing_enabled = True
        module._checkpoint_segments = checkpoint_segments
        module._use_reentrant = use_reentrant
        logger.info(f"Enabled gradient checkpointing for {module.__class__.__name__}")
    else:
        logger.warning(f"Gradient checkpointing already enabled for {module.__class__.__name__}")


def checkpoint_sequential(
    functions: list[nn.Module],
    segments: int,
    input: torch.Tensor,
    use_reentrant: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """
    对顺序模块列表应用分段梯度检查点。

    将模块列表分成多个段，每个段使用梯度检查点。
    这比对每个模块单独使用检查点更高效。

    Args:
        functions: 模块列表
        segments: 分段数量
        input: 输入张量
        use_reentrant: 是否使用可重入检查点
        **kwargs: 传递给模块的额外参数

    Returns:
        输出张量

    Example:
        >>> layers = [layer1, layer2, layer3, layer4]
        >>> output = checkpoint_sequential(layers, segments=2, input=x)
    """
    if segments <= 0:
        raise ValueError(f"segments must be positive, got {segments}")

    if segments > len(functions):
        segments = len(functions)

    def run_function(start: int, end: int, functions: list[nn.Module]) -> Callable:
        def forward(input: torch.Tensor) -> torch.Tensor:
            for j in range(start, end):
                input = functions[j](input)
            return input

        return forward

    # 分段处理
    segment_size = len(functions) // segments
    output = input

    for i in range(segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < segments - 1 else len(functions)

        if end_idx > start_idx:
            output = checkpoint(
                run_function(start_idx, end_idx, functions),
                output,
                use_reentrant=use_reentrant,
            )

    return output


def create_checkpoint_wrapper(
    module: nn.Module,
    use_reentrant: bool = True,
) -> Callable:
    """
    创建一个包装函数，使模块的 forward 使用梯度检查点。

    Args:
        module: 要包装的模块
        use_reentrant: 是否使用可重入检查点

    Returns:
        包装后的 forward 函数

    Example:
        >>> layer = nn.Linear(512, 512)
        >>> checkpointed_forward = create_checkpoint_wrapper(layer)
        >>> output = checkpointed_forward(input)
    """

    def forward_with_checkpoint(*args: Any, **kwargs: Any) -> torch.Tensor:
        def custom_forward(*inputs: Any) -> torch.Tensor:
            return module(*inputs, **kwargs)

        return checkpoint(custom_forward, *args, use_reentrant=use_reentrant)

    return forward_with_checkpoint


class CheckpointedSequential(nn.Sequential):
    """
    支持梯度检查点的 Sequential 容器。

    自动对内部的模块应用梯度检查点，可以指定分段数量。

    Args:
        *args: 模块列表
        segments: 分段数量（None 表示每个模块一个检查点）
        use_reentrant: 是否使用可重入检查点

    Example:
        >>> model = CheckpointedSequential(
        ...     nn.Linear(512, 512),
        ...     nn.ReLU(),
        ...     nn.Linear(512, 512),
        ...     segments=2,
        ... )
    """

    def __init__(
        self,
        *args: nn.Module,
        segments: int | None = None,
        use_reentrant: bool = True,
    ):
        super().__init__(*args)
        self.segments = segments if segments is not None else len(self)
        self.use_reentrant = use_reentrant

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """使用梯度检查点的前向传播。"""
        if not self.training:
            # 推理时不使用检查点
            return super().forward(input)

        modules = list(self.children())
        return checkpoint_sequential(
            modules,
            segments=self.segments,
            input=input,
            use_reentrant=self.use_reentrant,
        )


def estimate_memory_savings(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: str = "cuda",
) -> dict[str, float]:
    """
    估算启用梯度检查点后的内存节省。

    Args:
        model: 模型
        input_shape: 输入形状（不包括 batch 维度）
        device: 设备

    Returns:
        包含内存统计的字典：
            - without_checkpoint: 不使用检查点的内存（MB）
            - with_checkpoint: 使用检查点的内存（MB）
            - savings: 节省的内存（MB）
            - savings_percent: 节省百分比

    Example:
        >>> model = ResNet50()
        >>> stats = estimate_memory_savings(model, (3, 224, 224))
        >>> print(f"Memory savings: {stats['savings_percent']:.1f}%")
    """
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU for estimation")
        device = "cpu"

    model = model.to(device)
    model.train()

    # 测试不使用检查点
    torch.cuda.empty_cache() if device == "cuda" else None
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None

    dummy_input = torch.randn(1, *input_shape, device=device)
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()

    if device == "cuda":
        memory_without = torch.cuda.max_memory_allocated() / 1024**2  # MB
    else:
        memory_without = 0.0

    # 清理
    model.zero_grad()
    del output, loss, dummy_input

    # 测试使用检查点
    if hasattr(model, "enable_gradient_checkpointing"):
        torch.cuda.empty_cache() if device == "cuda" else None
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None

        model.enable_gradient_checkpointing()
        dummy_input = torch.randn(1, *input_shape, device=device)
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()

        if device == "cuda":
            memory_with = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            memory_with = 0.0

        savings = memory_without - memory_with
        savings_percent = (savings / memory_without * 100) if memory_without > 0 else 0.0

        return {
            "without_checkpoint": memory_without,
            "with_checkpoint": memory_with,
            "savings": savings,
            "savings_percent": savings_percent,
        }
    else:
        logger.warning(f"Model {model.__class__.__name__} does not support gradient checkpointing")
        return {
            "without_checkpoint": memory_without,
            "with_checkpoint": memory_without,
            "savings": 0.0,
            "savings_percent": 0.0,
        }


__all__ = [
    "apply_gradient_checkpointing",
    "checkpoint_sequential",
    "create_checkpoint_wrapper",
    "CheckpointedSequential",
    "estimate_memory_savings",
]
