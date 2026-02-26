"""
分布式训练工具

支持 DDP (DistributedDataParallel) 和 FSDP (Fully Sharded Data Parallel)。
"""

import logging
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def setup_distributed(
    backend: str = "nccl",
    init_method: str | None = None,
) -> tuple[int, int, int]:
    """
    设置分布式训练环境

    Args:
        backend: 分布式后端 ("nccl", "gloo", "mpi")
        init_method: 初始化方法

    Returns:
        (rank, local_rank, world_size)

    Example:
        >>> rank, local_rank, world_size = setup_distributed()
        >>> print(f"Rank {rank}/{world_size}")
    """
    # 从环境变量获取信息
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        # 初始化进程组
        if init_method is None:
            init_method = "env://"

        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )

        # 设置当前设备
        torch.cuda.set_device(local_rank)

        logger.info(
            f"Initialized distributed training: rank={rank}, "
            f"local_rank={local_rank}, world_size={world_size}"
        )
    else:
        logger.info("Running in single-process mode")

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """
    清理分布式训练环境

    Example:
        >>> cleanup_distributed()
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Destroyed process group")


def is_main_process() -> bool:
    """
    检查是否为主进程

    Returns:
        是否为主进程（rank 0）

    Example:
        >>> if is_main_process():
        ...     print("I am the main process")
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """
    获取当前进程的 rank

    Returns:
        当前 rank
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """
    获取总进程数

    Returns:
        world_size
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    """
    同步所有进程

    Example:
        >>> barrier()  # 等待所有进程
    """
    if dist.is_initialized():
        dist.barrier()


def reduce_dict(input_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    在所有进程间归约字典

    Args:
        input_dict: 输入字典

    Returns:
        归约后的字典

    Example:
        >>> metrics = {"loss": torch.tensor(0.5)}
        >>> avg_metrics = reduce_dict(metrics)
    """
    if not dist.is_initialized():
        return input_dict

    world_size = get_world_size()
    if world_size == 1:
        return input_dict

    with torch.no_grad():
        names = sorted(input_dict.keys())
        values = [input_dict[k] for k in names]

        # 归约
        for value in values:
            dist.all_reduce(value)

        # 平均
        reduced_dict = {k: v / world_size for k, v in zip(names, values)}

    return reduced_dict


class DDPWrapper:
    """
    DDP (DistributedDataParallel) 包装器

    简化 DDP 的使用。

    Args:
        model: PyTorch 模型
        device_ids: 设备 ID 列表
        output_device: 输出设备
        find_unused_parameters: 是否查找未使用的参数

    Example:
        >>> model = MyModel()
        >>> ddp_model = DDPWrapper(model)
        >>> output = ddp_model(input)
    """

    def __init__(
        self,
        model: nn.Module,
        device_ids: list[int] | None = None,
        output_device: int | None = None,
        find_unused_parameters: bool = False,
    ) -> None:
        self.model = model

        if dist.is_initialized():
            # 获取本地 rank
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

            if device_ids is None:
                device_ids = [local_rank]
            if output_device is None:
                output_device = local_rank

            # 移动模型到设备
            self.model = self.model.to(f"cuda:{local_rank}")

            # 包装为 DDP
            self.model = DDP(
                self.model,
                device_ids=device_ids,
                output_device=output_device,
                find_unused_parameters=find_unused_parameters,
            )

            logger.info(f"Wrapped model with DDP on device {local_rank}")
        else:
            logger.info("DDP not initialized, using single-process model")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """前向传播"""
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """代理属性访问"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def state_dict(self) -> dict[str, Any]:
        """获取状态字典"""
        if isinstance(self.model, DDP):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """加载状态字典"""
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)


class FSDPWrapper:
    """
    FSDP (Fully Sharded Data Parallel) 包装器

    支持更大规模的模型训练。

    Args:
        model: PyTorch 模型
        sharding_strategy: 分片策略
        auto_wrap_policy: 自动包装策略
        min_num_params: 最小参数数量（用于自动包装）

    Example:
        >>> model = MyModel()
        >>> fsdp_model = FSDPWrapper(model)
        >>> output = fsdp_model(input)
    """

    def __init__(
        self,
        model: nn.Module,
        sharding_strategy: str = "FULL_SHARD",
        auto_wrap_policy: Callable | None = None,
        min_num_params: int = 1000000,
    ) -> None:
        self.model = model

        if dist.is_initialized():
            # 获取本地 rank
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

            # 移动模型到设备
            self.model = self.model.to(f"cuda:{local_rank}")

            # 分片策略
            strategy_map = {
                "FULL_SHARD": ShardingStrategy.FULL_SHARD,
                "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
                "NO_SHARD": ShardingStrategy.NO_SHARD,
                "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
            }

            sharding_strategy_enum = strategy_map.get(
                sharding_strategy,
                ShardingStrategy.FULL_SHARD,
            )

            # 自动包装策略
            if auto_wrap_policy is None:
                auto_wrap_policy = size_based_auto_wrap_policy

            # 包装为 FSDP
            self.model = FSDP(
                self.model,
                sharding_strategy=sharding_strategy_enum,
                auto_wrap_policy=auto_wrap_policy,
                device_id=local_rank,
            )

            logger.info(
                f"Wrapped model with FSDP (strategy={sharding_strategy}) "
                f"on device {local_rank}"
            )
        else:
            logger.info("FSDP not initialized, using single-process model")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """前向传播"""
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """代理属性访问"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def state_dict(self) -> dict[str, Any]:
        """获取状态字典"""
        if isinstance(self.model, FSDP):
            # FSDP 需要特殊处理
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                return self.model.state_dict()
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """加载状态字典"""
        if isinstance(self.model, FSDP):
            # FSDP 需要特殊处理
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)


def create_distributed_model(
    model: nn.Module,
    strategy: str = "ddp",
    **kwargs: Any,
) -> nn.Module:
    """
    创建分布式模型的工厂函数

    Args:
        model: PyTorch 模型
        strategy: 分布式策略 ("ddp" 或 "fsdp")
        **kwargs: 额外参数

    Returns:
        分布式模型

    Example:
        >>> model = MyModel()
        >>> dist_model = create_distributed_model(model, strategy="ddp")
    """
    if strategy == "ddp":
        return DDPWrapper(model, **kwargs)
    elif strategy == "fsdp":
        return FSDPWrapper(model, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'ddp' or 'fsdp'.")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    filepath: str,
    **kwargs: Any,
) -> None:
    """
    保存检查点（仅在主进程）

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前 epoch
        filepath: 保存路径
        **kwargs: 额外信息

    Example:
        >>> save_checkpoint(model, optimizer, epoch, "checkpoint.pt")
    """
    if not is_main_process():
        return

    # 获取模型状态
    if isinstance(model, (DDPWrapper, FSDPWrapper)):
        model_state = model.state_dict()
    elif isinstance(model, (DDP, FSDP)):
        if isinstance(model, DDP):
            model_state = model.module.state_dict()
        else:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                model_state = model.state_dict()
    else:
        model_state = model.state_dict()

    # 保存
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        **kwargs,
    }

    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    filepath: str,
) -> dict[str, Any]:
    """
    加载检查点

    Args:
        model: 模型
        optimizer: 优化器（可选）
        filepath: 检查点路径

    Returns:
        检查点字典

    Example:
        >>> checkpoint = load_checkpoint(model, optimizer, "checkpoint.pt")
        >>> epoch = checkpoint["epoch"]
    """
    # 加载检查点
    map_location = {"cuda:0": f"cuda:{get_rank()}"}
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)

    # 加载模型状态
    if isinstance(model, (DDPWrapper, FSDPWrapper)):
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(model, (DDP, FSDP)):
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # 加载优化器状态
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(f"Loaded checkpoint from {filepath}")

    return checkpoint
