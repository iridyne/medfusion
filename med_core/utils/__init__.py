"""
Utility functions and helper modules for Med-Core framework.

This module contains common utilities used across the framework.
"""

from med_core.utils.checkpoint import (
    cleanup_checkpoints,
    find_best_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from med_core.utils.device import (
    get_device,
    get_device_info,
    move_to_device,
)
from med_core.utils.gradient_checkpointing import (
    CheckpointedSequential,
    apply_gradient_checkpointing,
    checkpoint_sequential,
    create_checkpoint_wrapper,
    estimate_memory_savings,
)
from med_core.utils.logging import (
    get_logger,
    setup_logging,
)
from med_core.utils.seed import set_seed

__all__ = [
    # Seed utilities
    "set_seed",
    # Device utilities
    "get_device",
    "get_device_info",
    "move_to_device",
    # Logging utilities
    "setup_logging",
    "get_logger",
    # Checkpoint utilities
    "save_checkpoint",
    "load_checkpoint",
    "find_best_checkpoint",
    "cleanup_checkpoints",
    # Gradient checkpointing utilities
    "apply_gradient_checkpointing",
    "checkpoint_sequential",
    "create_checkpoint_wrapper",
    "CheckpointedSequential",
    "estimate_memory_savings",
]
