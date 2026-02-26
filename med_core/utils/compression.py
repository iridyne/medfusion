"""
模型压缩工具

支持量化和剪枝。
"""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.quantization as quant
from torch import nn

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """
    模型量化器

    支持动态量化、静态量化和 QAT（量化感知训练）。

    Args:
        model: PyTorch 模型
        backend: 量化后端 ("fbgemm" 或 "qnnpack")

    Example:
        >>> quantizer = ModelQuantizer(model)
        >>> quantized_model = quantizer.dynamic_quantize()
    """

    def __init__(
        self,
        model: nn.Module,
        backend: str = "fbgemm",
    ):
        self.model = model
        self.backend = backend
        torch.backends.quantized.engine = backend

    def dynamic_quantize(
        self,
        dtype: torch.dtype = torch.qint8,
        modules: set | None = None,
    ) -> nn.Module:
        """
        动态量化

        Args:
            dtype: 量化数据类型
            modules: 要量化的模块类型

        Returns:
            量化后的模型
        """
        if modules is None:
            modules = {nn.Linear, nn.LSTM, nn.GRU}

        quantized_model = quant.quantize_dynamic(
            self.model,
            modules,
            dtype=dtype,
        )

        logger.info("✓ Dynamic quantization completed")
        logger.info(f"  Backend: {self.backend}")
        logger.info(f"  Dtype: {dtype}")

        return quantized_model

    def static_quantize(
        self,
        calibration_data: torch.utils.data.DataLoader,
    ) -> nn.Module:
        """
        静态量化

        Args:
            calibration_data: 校准数据

        Returns:
            量化后的模型
        """
        # 准备模型
        self.model.eval()
        self.model.qconfig = quant.get_default_qconfig(self.backend)

        # 融合模块
        model_fused = self._fuse_modules()

        # 准备量化
        model_prepared = quant.prepare(model_fused)

        # 校准
        logger.info("Calibrating model...")
        with torch.no_grad():
            for data, _ in calibration_data:
                model_prepared(data)

        # 转换
        quantized_model = quant.convert(model_prepared)

        logger.info("✓ Static quantization completed")

        return quantized_model

    def _fuse_modules(self) -> nn.Module:
        """融合模块"""
        # 简化版本，实际使用需要根据模型结构定制
        return self.model

    def compare_size(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
    ) -> None:
        """比较模型大小"""

        def get_size(model: nn.Module) -> float:
            torch.save(model.state_dict(), "temp.pth")
            size = Path("temp.pth").stat().st_size / (1024 * 1024)
            Path("temp.pth").unlink()
            return size

        original_size = get_size(original_model)
        quantized_size = get_size(quantized_model)

        logger.info("\nModel Size Comparison:")
        logger.info(f"  Original: {original_size:.2f} MB")
        logger.info(f"  Quantized: {quantized_size:.2f} MB")
        logger.info(f"  Reduction: {(1 - quantized_size / original_size) * 100:.1f}%")


class ModelPruner:
    """
    模型剪枝器

    支持非结构化剪枝和结构化剪枝。

    Args:
        model: PyTorch 模型

    Example:
        >>> pruner = ModelPruner(model)
        >>> pruned_model = pruner.prune_unstructured(amount=0.3)
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def prune_unstructured(
        self,
        amount: float = 0.3,
        method: str = "l1",
    ) -> nn.Module:
        """
        非结构化剪枝

        Args:
            amount: 剪枝比例
            method: 剪枝方法 ("l1" 或 "random")

        Returns:
            剪枝后的模型
        """
        from torch.nn.utils import prune

        parameters_to_prune = []
        for _name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, "weight"))

        if method == "l1":
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )
        elif method == "random":
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=amount,
            )

        # 移除重参数化
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        logger.info("✓ Unstructured pruning completed")
        logger.info(f"  Amount: {amount * 100:.1f}%")
        logger.info(f"  Method: {method}")

        return self.model

    def prune_structured(
        self,
        amount: float = 0.3,
        dim: int = 0,
    ) -> nn.Module:
        """
        结构化剪枝

        Args:
            amount: 剪枝比例
            dim: 剪枝维度

        Returns:
            剪枝后的模型
        """
        from torch.nn.utils import prune

        for _name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=amount,
                    n=2,
                    dim=dim,
                )
                prune.remove(module, "weight")

        logger.info("✓ Structured pruning completed")
        logger.info(f"  Amount: {amount * 100:.1f}%")
        logger.info(f"  Dim: {dim}")

        return self.model

    def get_sparsity(self) -> float:
        """获取稀疏度"""
        total_params = 0
        zero_params = 0

        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

        sparsity = zero_params / total_params

        logger.info("\nModel Sparsity:")
        logger.info(f"  Total params: {total_params:,}")
        logger.info(f"  Zero params: {zero_params:,}")
        logger.info(f"  Sparsity: {sparsity * 100:.2f}%")

        return sparsity


def quantize_model(
    model: nn.Module,
    method: str = "dynamic",
    calibration_data: torch.utils.data.DataLoader | None = None,
    **kwargs: Any,
) -> nn.Module:
    """
    量化模型的便捷函数

    Args:
        model: PyTorch 模型
        method: 量化方法 ("dynamic" 或 "static")
        calibration_data: 校准数据（静态量化需要）
        **kwargs: 额外参数

    Returns:
        量化后的模型

    Example:
        >>> quantized_model = quantize_model(model, method="dynamic")
    """
    quantizer = ModelQuantizer(model, **kwargs)

    if method == "dynamic":
        return quantizer.dynamic_quantize()
    if method == "static":
        if calibration_data is None:
            raise ValueError("Static quantization requires calibration_data")
        return quantizer.static_quantize(calibration_data)
    raise ValueError(f"Unknown method: {method}")


def prune_model(
    model: nn.Module,
    amount: float = 0.3,
    method: str = "unstructured",
    **kwargs: Any,
) -> nn.Module:
    """
    剪枝模型的便捷函数

    Args:
        model: PyTorch 模型
        amount: 剪枝比例
        method: 剪枝方法 ("unstructured" 或 "structured")
        **kwargs: 额外参数

    Returns:
        剪枝后的模型

    Example:
        >>> pruned_model = prune_model(model, amount=0.3)
    """
    pruner = ModelPruner(model)

    if method == "unstructured":
        return pruner.prune_unstructured(amount, **kwargs)
    if method == "structured":
        return pruner.prune_structured(amount, **kwargs)
    raise ValueError(f"Unknown method: {method}")


def compress_model(
    model: nn.Module,
    quantize: bool = True,
    prune: bool = True,
    prune_amount: float = 0.3,
    calibration_data: torch.utils.data.DataLoader | None = None,
) -> nn.Module:
    """
    压缩模型（量化 + 剪枝）

    Args:
        model: PyTorch 模型
        quantize: 是否量化
        prune: 是否剪枝
        prune_amount: 剪枝比例
        calibration_data: 校准数据

    Returns:
        压缩后的模型

    Example:
        >>> compressed_model = compress_model(model, quantize=True, prune=True)
    """
    compressed_model = model

    # 剪枝
    if prune:
        logger.info("Step 1: Pruning...")
        compressed_model = prune_model(compressed_model, amount=prune_amount)

    # 量化
    if quantize:
        logger.info("\nStep 2: Quantizing...")
        method = "static" if calibration_data else "dynamic"
        compressed_model = quantize_model(
            compressed_model,
            method=method,
            calibration_data=calibration_data,
        )

    logger.info("\n✓ Model compression completed!")

    return compressed_model
