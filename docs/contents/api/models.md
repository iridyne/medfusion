# Models API

MedFusion 提供了灵活的模型构建系统，支持多种架构组合。

## 核心类

### GenericMultiModalModel

通用多模态模型，支持任意数量的模态组合。

**特性：**
- 支持 2+ 个模态
- 任意骨干网络组合
- 可选的 MIL 聚合
- 灵活的融合策略

**示例：**
```python
from med_core.models import GenericMultiModalModel

backbones = {
    'ct': swin3d_backbone,
    'pathology': resnet_backbone
}
fusion = attention_fusion
head = classification_head

model = GenericMultiModalModel(backbones, fusion, head)
outputs = model({'ct': ct_data, 'pathology': path_data})
```

### MultiModalModelBuilder

流式 API 构建器，用于快速构建多模态模型。

**示例：**
```python
from med_core.models import MultiModalModelBuilder

builder = MultiModalModelBuilder(num_classes=2)
builder.add_modality("ct", backbone="swin3d_tiny", input_channels=1)
builder.add_modality("pathology", backbone="resnet50", pretrained=True)
builder.set_fusion("attention", hidden_dim=256)
builder.set_head("classification")
model = builder.build()
```

## 工厂函数

### build_model_from_config

从配置字典或 YAML 文件构建模型。

**参数：**
- `config`: 配置字典或 YAML 文件路径

**返回：**
- `GenericMultiModalModel`: 配置好的模型实例

**示例：**
```python
from med_core.models import build_model_from_config

# 从字典构建
config = {
    'modalities': {...},
    'fusion': {...},
    'head': {...}
}
model = build_model_from_config(config)

# 从 YAML 文件构建
model = build_model_from_config('configs/builder/generic_multimodal.yaml')
```

注意：

- `build_model_from_config()` 读取的是 builder 风格配置
- 它和 `medfusion train --config ...` 使用的 starter/public/testing 配置不是同一套 schema

## 配置格式

模型配置包含以下部分：

```yaml
modalities:
  ct:
    backbone: swin3d_tiny
    modality_type: vision3d
    input_channels: 1
    feature_dim: 512

  pathology:
    backbone: resnet50
    modality_type: vision
    pretrained: true
    feature_dim: 512

fusion:
  strategy: attention
  hidden_dim: 256
  num_heads: 8

head:
  task_type: classification
  num_classes: 4
  hidden_dims: [256, 128]
  dropout: 0.5

mil:  # 可选
  ct:
    strategy: attention
    hidden_dim: 128
```

## 参考

完整的 API 文档请参考源代码：
- `med_core/models/builder.py` - 模型构建器
- `med_core/models/three_phase_ct_fusion.py` - 原生 three-phase CT 融合模型
