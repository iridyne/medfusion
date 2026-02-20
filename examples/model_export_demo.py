"""
模型导出示例

演示如何将 PyTorch 模型导出为 ONNX 和 TorchScript 格式。
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn


def demo_simple_export():
    """演示简单模型导出"""
    print("=" * 60)
    print("简单模型导出演示")
    print("=" * 60)
    
    from med_core.utils.export import ModelExporter
    
    # 创建一个简单的模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.flatten(1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    
    # 创建导出器
    exporter = ModelExporter(
        model=model,
        input_shape=(3, 224, 224),
        device="cpu",
    )
    
    # 导出为 ONNX
    print("\n1. 导出为 ONNX:")
    exporter.export_onnx(
        "outputs/simple_model.onnx",
        opset_version=11,
        input_names=["image"],
        output_names=["logits"],
    )
    
    # 验证 ONNX 模型
    print("\n2. 验证 ONNX 模型:")
    exporter.verify_onnx("outputs/simple_model.onnx")
    
    # 导出为 TorchScript (trace)
    print("\n3. 导出为 TorchScript (trace):")
    exporter.export_torchscript(
        "outputs/simple_model_trace.pt",
        method="trace",
        optimize=True,
    )
    
    # 验证 TorchScript 模型
    print("\n4. 验证 TorchScript 模型:")
    exporter.verify_torchscript("outputs/simple_model_trace.pt")
    
    # 导出为 TorchScript (script)
    print("\n5. 导出为 TorchScript (script):")
    exporter.export_torchscript(
        "outputs/simple_model_script.pt",
        method="script",
        optimize=True,
    )


def demo_multimodal_export():
    """演示多模态模型导出"""
    print("\n" + "=" * 60)
    print("多模态模型导出演示")
    print("=" * 60)
    
    from med_core.utils.export import MultiModalExporter
    
    # 创建一个多模态模型
    class MultiModalModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 图像分支
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            # 表格分支
            self.tabular_encoder = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
            )
            # 融合
            self.fusion = nn.Linear(128, 10)
        
        def forward(self, image, tabular):
            image_feat = self.image_encoder(image)
            tabular_feat = self.tabular_encoder(tabular)
            fused = torch.cat([image_feat, tabular_feat], dim=1)
            output = self.fusion(fused)
            return output
    
    model = MultiModalModel()
    
    # 创建多模态导出器
    exporter = MultiModalExporter(
        model=model,
        input_shapes={
            "image": (3, 224, 224),
            "tabular": (10,),
        },
        device="cpu",
    )
    
    # 导出为 ONNX
    print("\n1. 导出为 ONNX:")
    exporter.export_onnx(
        "outputs/multimodal_model.onnx",
        input_names=["image", "tabular"],
        output_names=["logits"],
    )
    
    # 导出为 TorchScript
    print("\n2. 导出为 TorchScript:")
    exporter.export_torchscript(
        "outputs/multimodal_model.pt",
        method="trace",
    )


def demo_convenience_function():
    """演示便捷函数"""
    print("\n" + "=" * 60)
    print("便捷函数演示")
    print("=" * 60)
    
    from med_core.utils.export import export_model
    
    # 创建模型
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    )
    
    # 使用便捷函数导出
    print("\n1. 导出为 ONNX:")
    export_model(
        model=model,
        output_path="outputs/model_convenience.onnx",
        input_shape=(3, 224, 224),
        format="onnx",
        verify=True,
    )
    
    print("\n2. 导出为 TorchScript:")
    export_model(
        model=model,
        output_path="outputs/model_convenience.pt",
        input_shape=(3, 224, 224),
        format="torchscript",
        verify=True,
    )


def demo_dynamic_axes():
    """演示动态轴"""
    print("\n" + "=" * 60)
    print("动态轴演示")
    print("=" * 60)
    
    from med_core.utils.export import ModelExporter
    
    # 创建模型
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    )
    
    exporter = ModelExporter(model, input_shape=(3, 224, 224))
    
    # 导出时指定动态轴
    print("\n导出支持动态 batch size 和图像尺寸的模型:")
    exporter.export_onnx(
        "outputs/model_dynamic.onnx",
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {
                0: "batch_size",
                2: "height",
                3: "width",
            },
            "logits": {0: "batch_size"},
        },
    )
    
    print("\n✓ 模型支持:")
    print("  • 动态 batch size")
    print("  • 动态图像高度")
    print("  • 动态图像宽度")


def demo_inference():
    """演示推理"""
    print("\n" + "=" * 60)
    print("推理演示")
    print("=" * 60)
    
    # 创建并导出模型
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    )
    
    from med_core.utils.export import ModelExporter
    
    exporter = ModelExporter(model, input_shape=(3, 224, 224))
    exporter.export_onnx("outputs/model_inference.onnx")
    exporter.export_torchscript("outputs/model_inference.pt")
    
    # 1. PyTorch 推理
    print("\n1. PyTorch 推理:")
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        pytorch_output = model(x)
    print(f"   输出形状: {pytorch_output.shape}")
    print(f"   输出范围: [{pytorch_output.min():.3f}, {pytorch_output.max():.3f}]")
    
    # 2. TorchScript 推理
    print("\n2. TorchScript 推理:")
    loaded_model = torch.jit.load("outputs/model_inference.pt")
    loaded_model.eval()
    with torch.no_grad():
        torchscript_output = loaded_model(x)
    print(f"   输出形状: {torchscript_output.shape}")
    print(f"   与 PyTorch 的差异: {(pytorch_output - torchscript_output).abs().max():.6f}")
    
    # 3. ONNX 推理
    print("\n3. ONNX 推理:")
    try:
        import onnxruntime as ort
        
        ort_session = ort.InferenceSession("outputs/model_inference.onnx")
        ort_inputs = {ort_session.get_inputs()[0].name: x.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        print(f"   输出形状: {ort_output.shape}")
        print(f"   与 PyTorch 的差异: {abs(pytorch_output.numpy() - ort_output).max():.6f}")
    except ImportError:
        print("   ⚠ ONNXRuntime 未安装，跳过 ONNX 推理")


def demo_best_practices():
    """演示最佳实践"""
    print("\n" + "=" * 60)
    print("最佳实践")
    print("=" * 60)
    
    print("\n1. 选择合适的导出格式:")
    print("   • ONNX: 跨平台部署，支持多种推理引擎")
    print("   • TorchScript: PyTorch 生态，性能优化")
    
    print("\n2. 导出前的准备:")
    print("   • 设置模型为评估模式 (model.eval())")
    print("   • 移除训练相关的操作（dropout、batch norm）")
    print("   • 测试模型在不同输入下的行为")
    
    print("\n3. 验证导出的模型:")
    print("   • 比较输出是否一致")
    print("   • 测试不同的输入尺寸（如果使用动态轴）")
    print("   • 测试边界情况")
    
    print("\n4. 优化建议:")
    print("   • 使用 optimize_for_inference (TorchScript)")
    print("   • 选择合适的 opset 版本 (ONNX)")
    print("   • 考虑量化和剪枝")
    
    print("\n5. 常见问题:")
    print("   • 动态控制流: 使用 script 而不是 trace")
    print("   • 自定义算子: 需要注册 ONNX 算子")
    print("   • 版本兼容性: 注意 PyTorch 和 ONNX 版本")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("MedFusion 模型导出演示")
    print("=" * 60)
    
    # 创建输出目录
    Path("outputs").mkdir(exist_ok=True)
    
    try:
        # 演示 1: 简单模型导出
        demo_simple_export()
        
        # 演示 2: 多模态模型导出
        demo_multimodal_export()
        
        # 演示 3: 便捷函数
        demo_convenience_function()
        
        # 演示 4: 动态轴
        demo_dynamic_axes()
        
        # 演示 5: 推理
        demo_inference()
        
        # 演示 6: 最佳实践
        demo_best_practices()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
        print("\n💡 关键要点:")
        print("  1. ONNX 适合跨平台部署")
        print("  2. TorchScript 适合 PyTorch 生态")
        print("  3. 始终验证导出的模型")
        print("  4. 使用动态轴支持不同输入尺寸")
        print("  5. 优化模型以提高推理性能")
        
        print("\n📖 相关资源:")
        print("  • med_core/utils/export.py")
        print("  • examples/model_export_demo.py")
        print("  • docs/guides/model_export.md")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
