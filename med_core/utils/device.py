"""Device management utilities for PyTorch."""

from typing import Any

import torch


def get_device(device: str = "auto") -> torch.device:
    """
    Get PyTorch device based on availability.

    Args:
        device: Device specification ("auto", "cuda", "cpu", "mps")

    Returns:
        torch.device object
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def get_device_info() -> dict[str, Any]:
    """
    Get information about available devices.

    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
        "mps_available": hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available(),
    }

    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory

    return info


def move_to_device(obj: Any, device: torch.device) -> Any:
    """
    Move tensor or model to device.

    Args:
        obj: Tensor, model, or dict/list of tensors
        device: Target device

    Returns:
        Object moved to device
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    return obj
