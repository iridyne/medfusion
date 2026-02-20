"""
测试模型导出功能
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from med_core.utils.export import (
    ModelExporter,
    MultiModalExporter,
    export_model,
)


class SimpleModel(nn.Module):
    """简单测试模型"""
    
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


class MultiModalModel(nn.Module):
    """多模态测试模型"""
    
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.tabular_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
        )
        self.fusion = nn.Linear(128, 10)
    
    def forward(self, image, tabular):
        image_feat = self.image_encoder(image)
        tabular_feat = self.tabular_encoder(tabular)
        fused = torch.cat([image_feat, tabular_feat], dim=1)
        output = self.fusion(fused)
        return output


class TestModelExporter:
    """测试 ModelExporter"""
    
    def test_init(self):
        """测试初始化"""
        model = SimpleModel()
        exporter = ModelExporter(model, input_shape=(3, 224, 224))
        
        assert exporter.model is not None
        assert exporter.input_shape == (3, 224, 224)
        assert exporter.device == "cpu"
    
    def test_export_onnx(self):
        """测试导出 ONNX"""
        model = SimpleModel()
        exporter = ModelExporter(model, input_shape=(3, 224, 224))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"
            exporter.export_onnx(output_path)
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_export_torchscript_trace(self):
        """测试导出 TorchScript (trace)"""
        model = SimpleModel()
        exporter = ModelExporter(model, input_shape=(3, 224, 224))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"
            exporter.export_torchscript(output_path, method="trace")
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_export_torchscript_script(self):
        """测试导出 TorchScript (script)"""
        model = SimpleModel()
        exporter = ModelExporter(model, input_shape=(3, 224, 224))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"
            exporter.export_torchscript(output_path, method="script")
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_verify_torchscript(self):
        """测试验证 TorchScript"""
        model = SimpleModel()
        exporter = ModelExporter(model, input_shape=(3, 224, 224))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"
            exporter.export_torchscript(output_path, method="trace")
            
            result = exporter.verify_torchscript(output_path)
            assert result is True
    
    def test_dynamic_axes(self):
        """测试动态轴"""
        model = SimpleModel()
        exporter = ModelExporter(model, input_shape=(3, 224, 224))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"
            exporter.export_onnx(
                output_path,
                dynamic_axes={
                    "input": {0: "batch_size", 2: "height", 3: "width"},
                    "output": {0: "batch_size"},
                },
            )
            
            assert output_path.exists()


class TestMultiModalExporter:
    """测试 MultiModalExporter"""
    
    def test_init(self):
        """测试初始化"""
        model = MultiModalModel()
        exporter = MultiModalExporter(
            model,
            input_shapes={
                "image": (3, 224, 224),
                "tabular": (10,),
            },
        )
        
        assert exporter.model is not None
        assert len(exporter.input_shapes) == 2
    
    def test_export_onnx(self):
        """测试导出 ONNX"""
        model = MultiModalModel()
        exporter = MultiModalExporter(
            model,
            input_shapes={
                "image": (3, 224, 224),
                "tabular": (10,),
            },
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"
            exporter.export_onnx(
                output_path,
                input_names=["image", "tabular"],
                output_names=["logits"],
            )
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_export_torchscript(self):
        """测试导出 TorchScript"""
        model = MultiModalModel()
        exporter = MultiModalExporter(
            model,
            input_shapes={
                "image": (3, 224, 224),
                "tabular": (10,),
            },
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"
            exporter.export_torchscript(output_path, method="trace")
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0


class TestExportModel:
    """测试 export_model 便捷函数"""
    
    def test_export_onnx(self):
        """测试导出 ONNX"""
        model = SimpleModel()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"
            export_model(
                model,
                output_path,
                input_shape=(3, 224, 224),
                format="onnx",
                verify=False,
            )
            
            assert output_path.exists()
    
    def test_export_torchscript(self):
        """测试导出 TorchScript"""
        model = SimpleModel()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"
            export_model(
                model,
                output_path,
                input_shape=(3, 224, 224),
                format="torchscript",
                verify=False,
            )
            
            assert output_path.exists()
    
    def test_invalid_format(self):
        """测试无效格式"""
        model = SimpleModel()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.invalid"
            
            with pytest.raises(ValueError):
                export_model(
                    model,
                    output_path,
                    input_shape=(3, 224, 224),
                    format="invalid",
                )


class TestInference:
    """测试推理"""
    
    def test_torchscript_inference(self):
        """测试 TorchScript 推理"""
        model = SimpleModel()
        model.eval()
        
        exporter = ModelExporter(model, input_shape=(3, 224, 224))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"
            exporter.export_torchscript(output_path, method="trace")
            
            # 加载模型
            loaded_model = torch.jit.load(str(output_path))
            loaded_model.eval()
            
            # 推理
            x = torch.randn(1, 3, 224, 224)
            
            with torch.no_grad():
                original_output = model(x)
                loaded_output = loaded_model(x)
            
            # 比较输出
            assert torch.allclose(original_output, loaded_output, rtol=1e-3, atol=1e-5)
    
    def test_different_batch_sizes(self):
        """测试不同的 batch size"""
        model = SimpleModel()
        model.eval()
        
        exporter = ModelExporter(model, input_shape=(3, 224, 224))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"
            exporter.export_torchscript(output_path, method="trace")
            
            loaded_model = torch.jit.load(str(output_path))
            loaded_model.eval()
            
            # 测试不同的 batch size
            for batch_size in [1, 2, 4]:
                x = torch.randn(batch_size, 3, 224, 224)
                
                with torch.no_grad():
                    output = loaded_model(x)
                
                assert output.shape == (batch_size, 10)


class TestEdgeCases:
    """测试边界情况"""
    
    def test_small_model(self):
        """测试小模型"""
        model = nn.Linear(10, 5)
        exporter = ModelExporter(model, input_shape=(10,))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"
            exporter.export_torchscript(output_path)
            
            assert output_path.exists()
    
    def test_large_input(self):
        """测试大输入"""
        model = SimpleModel()
        exporter = ModelExporter(model, input_shape=(3, 512, 512))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"
            exporter.export_torchscript(output_path)
            
            assert output_path.exists()
    
    def test_model_with_dropout(self):
        """测试带 dropout 的模型"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Dropout(0.5),
            nn.Linear(20, 5),
        )
        model.eval()  # 重要：设置为评估模式
        
        exporter = ModelExporter(model, input_shape=(10,))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"
            exporter.export_torchscript(output_path)
            
            assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
