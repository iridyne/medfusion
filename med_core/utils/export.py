"""
模型导出工具

支持将 PyTorch 模型导出为 ONNX 和 TorchScript 格式。
"""

import logging
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


class ModelExporter:
    """
    模型导出器

    支持导出为 ONNX 和 TorchScript 格式。

    Args:
        model: PyTorch 模型
        input_shape: 输入形状（不包括 batch 维度）
        device: 设备

    Example:
        >>> model = MyModel()
        >>> exporter = ModelExporter(model, input_shape=(3, 224, 224))
        >>> exporter.export_onnx("model.onnx")
        >>> exporter.export_torchscript("model.pt")
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...],
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.input_shape = input_shape
        self.device = device

        # 设置为评估模式
        self.model.eval()

    def export_onnx(
        self,
        output_path: str | Path,
        opset_version: int = 11,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        导出为 ONNX 格式

        Args:
            output_path: 输出路径
            opset_version: ONNX opset 版本
            input_names: 输入名称列表
            output_names: 输出名称列表
            dynamic_axes: 动态轴配置
            verbose: 是否打印详细信息
            **kwargs: 传递给 torch.onnx.export 的额外参数

        Example:
            >>> exporter.export_onnx(
            ...     "model.onnx",
            ...     input_names=["input"],
            ...     output_names=["output"],
            ...     dynamic_axes={"input": {0: "batch_size"}},
            ... )
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 创建示例输入
        dummy_input = torch.randn(1, *self.input_shape).to(self.device)

        # 默认输入输出名称
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        # 默认动态轴（batch 维度）
        if dynamic_axes is None:
            dynamic_axes = {
                input_names[0]: {0: "batch_size"},
                output_names[0]: {0: "batch_size"},
            }

        logger.info(f"Exporting model to ONNX: {output_path}")
        logger.info(f"  Input shape: {dummy_input.shape}")
        logger.info(f"  Opset version: {opset_version}")

        # 导出
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            torch.onnx.export(
                self.model,
                dummy_input,
                str(output_path),
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=verbose,
                **kwargs,
            )

        logger.info(f"✓ Model exported to {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    def export_torchscript(
        self,
        output_path: str | Path,
        method: str = "trace",
        optimize: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        导出为 TorchScript 格式

        Args:
            output_path: 输出路径
            method: 导出方法 ("trace" 或 "script")
            optimize: 是否优化
            **kwargs: 传递给 torch.jit.trace/script 的额外参数

        Example:
            >>> exporter.export_torchscript("model.pt", method="trace")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting model to TorchScript: {output_path}")
        logger.info(f"  Method: {method}")
        logger.info(f"  Optimize: {optimize}")

        if method == "trace":
            # 使用 trace 方法
            dummy_input = torch.randn(1, *self.input_shape).to(self.device)
            logger.info(f"  Input shape: {dummy_input.shape}")

            traced_model = torch.jit.trace(self.model, dummy_input, **kwargs)

            if optimize:
                traced_model = torch.jit.optimize_for_inference(traced_model)

            traced_model.save(str(output_path))

        elif method == "script":
            # 使用 script 方法
            scripted_model = torch.jit.script(self.model, **kwargs)

            if optimize:
                scripted_model = torch.jit.optimize_for_inference(scripted_model)

            scripted_model.save(str(output_path))

        else:
            raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'.")

        logger.info(f"✓ Model exported to {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    def verify_onnx(
        self,
        onnx_path: str | Path,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ) -> bool:
        """
        验证 ONNX 模型

        Args:
            onnx_path: ONNX 模型路径
            rtol: 相对容差
            atol: 绝对容差

        Returns:
            是否验证通过

        Example:
            >>> exporter.export_onnx("model.onnx")
            >>> exporter.verify_onnx("model.onnx")
        """
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            logger.warning(
                "⚠ ONNX or ONNXRuntime not installed. Skipping verification.",
            )
            return False

        logger.info(f"Verifying ONNX model: {onnx_path}")

        # 加载 ONNX 模型
        onnx_model = onnx.load(str(onnx_path))

        # 检查模型
        try:
            onnx.checker.check_model(onnx_model)
            logger.info("  ✓ ONNX model is valid")
        except Exception as e:
            logger.error(f"  ✗ ONNX model check failed: {e}")
            return False

        # 创建测试输入
        dummy_input = torch.randn(1, *self.input_shape).to(self.device)

        # PyTorch 输出
        with torch.inference_mode():
            pytorch_output = self.model(dummy_input)

        if isinstance(pytorch_output, tuple):
            pytorch_output = pytorch_output[0]

        pytorch_output = pytorch_output.cpu().numpy()

        # ONNX Runtime 输出
        ort_session = ort.InferenceSession(str(onnx_path))
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]

        # 比较输出
        import numpy as np

        if np.allclose(pytorch_output, ort_output, rtol=rtol, atol=atol):
            max_diff = np.abs(pytorch_output - ort_output).max()
            logger.info(f"  ✓ Outputs match (max diff: {max_diff:.6f})")
            return True
        max_diff = np.abs(pytorch_output - ort_output).max()
        logger.error(f"  ✗ Outputs differ (max diff: {max_diff:.6f})")
        return False

    def verify_torchscript(
        self,
        torchscript_path: str | Path,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ) -> bool:
        """
        验证 TorchScript 模型

        Args:
            torchscript_path: TorchScript 模型路径
            rtol: 相对容差
            atol: 绝对容差

        Returns:
            是否验证通过

        Example:
            >>> exporter.export_torchscript("model.pt")
            >>> exporter.verify_torchscript("model.pt")
        """
        logger.info(f"Verifying TorchScript model: {torchscript_path}")

        # 加载 TorchScript 模型
        loaded_model = torch.jit.load(str(torchscript_path))
        loaded_model.eval()
        loaded_model.to(self.device)

        # 创建测试输入
        dummy_input = torch.randn(1, *self.input_shape).to(self.device)

        # 原始模型输出
        with torch.inference_mode():
            original_output = self.model(dummy_input)

        if isinstance(original_output, tuple):
            original_output = original_output[0]

        # TorchScript 模型输出
        with torch.inference_mode():
            loaded_output = loaded_model(dummy_input)

        if isinstance(loaded_output, tuple):
            loaded_output = loaded_output[0]

        # 比较输出
        if torch.allclose(original_output, loaded_output, rtol=rtol, atol=atol):
            max_diff = (original_output - loaded_output).abs().max().item()
            logger.info(f"  ✓ Outputs match (max diff: {max_diff:.6f})")
            return True
        max_diff = (original_output - loaded_output).abs().max().item()
        logger.error(f"  ✗ Outputs differ (max diff: {max_diff:.6f})")
        return False


def export_model(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, ...],
    format: str = "onnx",
    verify: bool = True,
    **kwargs: Any,
) -> None:
    """
    导出模型的便捷函数

    Args:
        model: PyTorch 模型
        output_path: 输出路径
        input_shape: 输入形状（不包括 batch 维度）
        format: 导出格式 ("onnx" 或 "torchscript")
        verify: 是否验证导出的模型
        **kwargs: 传递给导出函数的额外参数

    Example:
        >>> model = MyModel()
        >>> export_model(model, "model.onnx", (3, 224, 224))
    """
    exporter = ModelExporter(model, input_shape)

    if format == "onnx":
        exporter.export_onnx(output_path, **kwargs)
        if verify:
            exporter.verify_onnx(output_path)
    elif format == "torchscript":
        exporter.export_torchscript(output_path, **kwargs)
        if verify:
            exporter.verify_torchscript(output_path)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'onnx' or 'torchscript'.")


class MultiModalExporter(ModelExporter):
    """
    多模态模型导出器

    支持多个输入的模型导出。

    Args:
        model: PyTorch 模型
        input_shapes: 输入形状字典
        device: 设备

    Example:
        >>> model = MultiModalModel()
        >>> exporter = MultiModalExporter(
        ...     model,
        ...     input_shapes={
        ...         "image": (3, 224, 224),
        ...         "tabular": (10,),
        ...     }
        ... )
        >>> exporter.export_onnx("model.onnx")
    """

    def __init__(
        self,
        model: nn.Module,
        input_shapes: dict[str, tuple[int, ...]],
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.input_shapes = input_shapes
        self.device = device
        self.model.eval()

    def _create_dummy_inputs(self) -> tuple[Any, ...]:
        """创建示例输入"""
        dummy_inputs = []
        for shape in self.input_shapes.values():
            dummy_input = torch.randn(1, *shape).to(self.device)
            dummy_inputs.append(dummy_input)
        return tuple(dummy_inputs)

    def export_onnx(
        self,
        output_path: str | Path,
        opset_version: int = 11,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """导出为 ONNX 格式"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 创建示例输入
        dummy_inputs = self._create_dummy_inputs()

        # 默认输入输出名称
        if input_names is None:
            input_names = list(self.input_shapes.keys())
        if output_names is None:
            output_names = ["output"]

        # 默认动态轴
        if dynamic_axes is None:
            dynamic_axes = {name: {0: "batch_size"} for name in input_names}
            dynamic_axes[output_names[0]] = {0: "batch_size"}

        logger.info(f"Exporting multimodal model to ONNX: {output_path}")
        for name, dummy_input in zip(input_names, dummy_inputs):
            logger.info(f"  {name}: {dummy_input.shape}")

        # 导出
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            torch.onnx.export(
                self.model,
                dummy_inputs,
                str(output_path),
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=verbose,
                **kwargs,
            )

        logger.info(f"✓ Model exported to {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    def export_torchscript(
        self,
        output_path: str | Path,
        method: str = "trace",
        optimize: bool = True,
        **kwargs: Any,
    ) -> None:
        """导出为 TorchScript 格式"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting multimodal model to TorchScript: {output_path}")
        logger.info(f"  Method: {method}")

        if method == "trace":
            dummy_inputs = self._create_dummy_inputs()
            for name, dummy_input in zip(self.input_shapes.keys(), dummy_inputs):
                logger.info(f"  {name}: {dummy_input.shape}")

            traced_model = torch.jit.trace(self.model, dummy_inputs, **kwargs)

            if optimize:
                traced_model = torch.jit.optimize_for_inference(traced_model)

            traced_model.save(str(output_path))

        elif method == "script":
            scripted_model = torch.jit.script(self.model, **kwargs)

            if optimize:
                scripted_model = torch.jit.optimize_for_inference(scripted_model)

            scripted_model.save(str(output_path))

        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"✓ Model exported to {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
