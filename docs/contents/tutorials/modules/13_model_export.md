# 模型导出

**预计时间：15分钟**

本教程介绍如何将训练好的 MedFusion 模型导出为 ONNX 和 TorchScript 格式以便部署。

## 导出格式选择

### ONNX（推荐跨平台部署）

**优势：**
- 跨平台、跨框架
- 支持多种推理引擎（ONNX Runtime、TensorRT、OpenVINO）
- 硬件加速支持好
- 模型优化和量化

**适用场景：**
- 生产环境部署
- 移动端和边缘设备
- 需要最大兼容性

### TorchScript（推荐 PyTorch 生态）

**优势：**
- 完全兼容 PyTorch
- 保留模型结构
- 支持 C++ 部署
- 易于调试

**适用场景：**
- PyTorch 服务环境
- 需要保留动态特性
- C++ 集成

## 快速开始

### 导出为 ONNX

```python
from med_core.utils.export import export_model
import torch

# 加载训练好的模型
model = torch.load("outputs/checkpoints/best.pth")
model.eval()

# 导出为 ONNX
export_model(
    model=model,
    output_path="model.onnx",
    input_shape=(3, 224, 224),
    format="onnx"
)
```

### 导出为 TorchScript

```python
# 导出为 TorchScript
export_model(
    model=model,
    output_path="model.pt",
    input_shape=(3, 224, 224),
    format="torchscript"
)
```

## ONNX 导出详解

### 基本导出

```python
from med_core.utils.export import ModelExporter

# 创建导出器
exporter = ModelExporter(
    model=model,
    input_shape=(3, 224, 224),
    device="cpu"
)

# 导出
exporter.export_onnx(
    output_path="model.onnx",
    opset_version=11,
    input_names=["image"],
    output_names=["logits"]
)

# 验证
exporter.verify_onnx("model.onnx")
```

### 动态批次大小

```python
exporter.export_onnx(
    output_path="model_dynamic.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "logits": {0: "batch_size"}
    }
)
```

### ONNX 推理

```python
import onnxruntime as ort
import numpy as np

# 加载模型
session = ort.InferenceSession("model.onnx")

# 准备输入
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
input_name = session.get_inputs()[0].name

# 推理
outputs = session.run(None, {input_name: input_data})
print(f"Output shape: {outputs[0].shape}")
```

## TorchScript 导出详解

### Trace 方法（推荐）

```python
exporter.export_torchscript(
    output_path="model_trace.pt",
    method="trace",
    optimize=True
)
```

### Script 方法

适用于包含动态控制流的模型。

```python
exporter.export_torchscript(
    output_path="model_script.pt",
    method="script",
    optimize=True
)
```

### TorchScript 推理

```python
import torch

# 加载模型
model = torch.jit.load("model.pt")
model.eval()

# 推理
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(x)
print(f"Output shape: {output.shape}")
```

## 多模态模型导出

### 导出多输入模型

```python
from med_core.utils.export import MultiModalExporter

# 创建导出器
exporter = MultiModalExporter(
    model=multimodal_model,
    input_shapes={
        "image": (3, 224, 224),
        "tabular": (10,)
    },
    device="cpu"
)

# 导出为 ONNX
exporter.export_onnx(
    output_path="multimodal_model.onnx",
    input_names=["image", "tabular"],
    output_names=["logits"]
)

# 导出为 TorchScript
exporter.export_torchscript(
    output_path="multimodal_model.pt",
    method="trace"
)
```

### 多模态推理

```python
# ONNX 推理
session = ort.InferenceSession("multimodal_model.onnx")
outputs = session.run(
    None,
    {
        "image": image_data,
        "tabular": tabular_data
    }
)

# TorchScript 推理
model = torch.jit.load("multimodal_model.pt")
output = model(image_tensor, tabular_tensor)
```

## 完整示例

### 单模态分类模型

```python
from med_core.models import MultiModalModelBuilder
from med_core.utils.export import ModelExporter

# 1. 加载训练好的模型
builder = MultiModalModelBuilder(num_classes=2)
builder.add_modality("ct", backbone="resnet50", pretrained=True)
builder.set_fusion("none")
builder.set_head("classification")
model = builder.build()

# 加载权重
checkpoint = torch.load("outputs/checkpoints/best.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. 导出
exporter = ModelExporter(model, input_shape=(3, 224, 224))

# ONNX
exporter.export_onnx(
    "model.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "logits": {0: "batch_size"}
    }
)

# 验证
assert exporter.verify_onnx("model.onnx")

# TorchScript
exporter.export_torchscript("model.pt", method="trace")
assert exporter.verify_torchscript("model.pt")
```

### 多模态融合模型

```python
# 1. 构建多模态模型
builder = MultiModalModelBuilder(num_classes=2)
builder.add_modality("ct", backbone="resnet50", pretrained=True)
builder.add_modality("tabular", input_dim=10)
builder.set_fusion("gated")
builder.set_head("classification")
model = builder.build()

# 加载权重
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# 2. 导出
exporter = MultiModalExporter(
    model,
    input_shapes={
        "image": (3, 224, 224),
        "tabular": (10,)
    }
)

# ONNX
exporter.export_onnx(
    "multimodal_model.onnx",
    input_names=["image", "tabular"],
    output_names=["logits"]
)

# TorchScript
exporter.export_torchscript("multimodal_model.pt")
```

## 验证导出模型

### 输出一致性检查

```python
import numpy as np

# 准备测试输入
test_input = torch.randn(1, 3, 224, 224)

# PyTorch 输出
model.eval()
with torch.no_grad():
    pytorch_output = model(test_input)

# ONNX 输出
session = ort.InferenceSession("model.onnx")
onnx_output = session.run(
    None,
    {"image": test_input.numpy()}
)[0]

# 比较
np.testing.assert_allclose(
    pytorch_output.numpy(),
    onnx_output,
    rtol=1e-3,
    atol=1e-5
)
print("✓ 输出一致性验证通过")
```

### 性能测试

```python
import time

# PyTorch 性能
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(test_input)
pytorch_time = time.time() - start

# ONNX 性能
start = time.time()
for _ in range(100):
    _ = session.run(None, {"image": test_input.numpy()})
onnx_time = time.time() - start

print(f"PyTorch: {pytorch_time:.3f}s")
print(f"ONNX: {onnx_time:.3f}s")
print(f"加速比: {pytorch_time / onnx_time:.2f}x")
```

## 最佳实践

### 1. 导出前准备

```python
# 设置为评估模式
model.eval()

# 移除 Dropout
for module in model.modules():
    if isinstance(module, torch.nn.Dropout):
        module.p = 0.0

# 测试模型
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(x)
print(f"Output shape: {output.shape}")
```

### 2. 选择合适的格式

| 场景 | 推荐格式 | 原因 |
|------|---------|------|
| 跨平台部署 | ONNX | 支持多种推理引擎 |
| PyTorch 生态 | TorchScript | 完全兼容 |
| 移动端部署 | ONNX | 轻量级 |
| 边缘设备 | ONNX | 硬件加速 |
| C++ 部署 | TorchScript | 易于集成 |

### 3. 优化导出模型

```python
# ONNX 优化
exporter.export_onnx(
    "model.onnx",
    opset_version=11,
    do_constant_folding=True  # 常量折叠优化
)

# TorchScript 优化
exporter.export_torchscript(
    "model.pt",
    method="trace",
    optimize=True  # 启用优化
)
```

## 常见问题

### Q1: 导出失败怎么办？

A: 检查以下几点：
1. 模型是否设置为评估模式
2. 是否有不支持的操作
3. 是否有动态控制流（使用 script 而不是 trace）
4. PyTorch 和 ONNX 版本是否兼容

### Q2: 如何处理自定义算子？

A: 使用 TorchScript 或注册 ONNX 算子：

```python
# 方法 1: 使用 TorchScript
exporter.export_torchscript("model.pt", method="script")

# 方法 2: 注册 ONNX 算子
from torch.onnx import register_custom_op_symbolic

@register_custom_op_symbolic("custom::my_op", opset_version=11)
def my_op_symbolic(g, input):
    return g.op("custom::MyOp", input)
```

### Q3: 导出的模型太大怎么办？

A: 四种方法：
1. 使用模型压缩（量化、剪枝）
2. 移除不必要的层
3. 使用更小的骨干网络
4. 压缩模型文件（gzip）

### Q4: 如何优化推理性能？

A:
1. 使用 ONNX Runtime 的优化选项
2. 启用硬件加速（GPU、TensorRT）
3. 使用量化和剪枝
4. 批处理推理

## 下一步

- [Docker 部署](14_docker_deployment.md) - 容器化部署模型
- [生产环境清单](15_production_checklist.md) - 部署前检查
- [性能优化指南](../../guides/performance_optimization.md) - 进一步优化

## 参考资源

详细的模型导出指南请参考：
- [完整模型导出文档](/home/yixian/Projects/med-ml/medfusion/docs/guides/model_export.md)
- [ONNX 官方文档](https://onnx.ai/)
- [TorchScript 文档](https://pytorch.org/docs/stable/jit.html)
