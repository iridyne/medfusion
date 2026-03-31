# 模型构建器 API

**预计时间：25分钟**

本教程详细讲解 MedFusion 的模型构建器 API，展示如何通过代码灵活构建各种多模态模型。

先说明这页和 CLI 主链的关系：

- 这页主要服务于**结构实验**
- Builder 配置更适合研究原型和结构拼装
- 它**不是当前 `medfusion train` 主链 YAML**

如果你是为了“新建一个能直接进入训练主链的 YAML”，先看
[如何新建模型与 YAML](../../getting-started/model-creation-paths.md)。

## 三种模型构建方式

MedFusion 提供三种构建模型的方式：

### 1. Builder API（推荐）

```python
from med_core.models import MultiModalModelBuilder

model = (
    MultiModalModelBuilder()
    .add_modality("xray", backbone="resnet18")
    .add_modality("clinical", backbone="mlp", input_dim=10)
    .set_fusion("attention")
    .set_head("classification", num_classes=2)
    .build()
)
```

**优点**：链式调用，代码简洁，易于理解

### 2. 配置文件

```python
from med_core.models import build_model_from_config
import yaml

with open("configs/builder/generic_multimodal.yaml") as f:
    config = yaml.safe_load(f)

model = build_model_from_config(config)
```

**优点**：配置与代码分离，便于实验管理
**注意**：这里使用的是 `configs/builder/` 下的 builder 配置，不是 `medfusion train` 主链配置。

### 3. 直接构建

```python
from med_core.models import GenericMultiModalModel
from med_core.backbones import create_vision_backbone
from med_core.fusion import create_fusion_module
from med_core.heads import ClassificationHead

backbones = {
    'xray': create_vision_backbone('resnet18'),
    'clinical': create_tabular_backbone('mlp', input_dim=10)
}
fusion = create_fusion_module('attention', input_dims=[512, 64], output_dim=256)
head = ClassificationHead(input_dim=256, num_classes=2)
model = GenericMultiModalModel(backbones, fusion, head)
```

**优点**：完全控制，适合高级定制

## Builder API 详解

### 基本结构

```python
builder = MultiModalModelBuilder()
builder.add_modality(...)      # 添加模态
builder.add_mil_aggregation(...)  # 添加 MIL 聚合（可选）
builder.set_fusion(...)        # 设置融合策略
builder.set_head(...)          # 设置任务头
model = builder.build()        # 构建模型
```

### 链式调用

```python
model = (
    MultiModalModelBuilder()
    .add_modality("xray", backbone="resnet18")
    .add_modality("clinical", backbone="mlp", input_dim=10)
    .set_fusion("attention")
    .set_head("classification", num_classes=2)
    .build()
)
```

## 示例 1：基础多模态模型

```python
import torch
from med_core.models import MultiModalModelBuilder

# 构建模型
model = (
    MultiModalModelBuilder()
    .add_modality("xray", backbone="resnet18", modality_type="vision")
    .add_modality("clinical", backbone="mlp", modality_type="tabular", input_dim=10)
    .set_fusion("attention")
    .set_head("classification", num_classes=2)
    .build()
)

# 准备输入
xray = torch.randn(4, 3, 224, 224)
clinical = torch.randn(4, 10)
inputs = {"xray": xray, "clinical": clinical}

# 前向传播
logits = model(inputs)
print(f"Output shape: {logits.shape}")  # [4, 2]
```

**关键参数：**
- `modality_type`: 模态类型（`"vision"`, `"vision3d"`, `"tabular"`, `"custom"`）
- `backbone`: 骨干网络名称或自定义模块
- `input_dim`: 表格数据的输入维度

## 示例 2：影像-病理双模态模型

```python
# 使用 Builder API 构建一个影像-病理双模态分类模型
model = (
    MultiModalModelBuilder()
    .add_modality(
        "radiology",
        backbone="swin3d_small",
        modality_type="vision3d",
        in_channels=1,
        feature_dim=512,
    )
    .add_modality(
        "pathology",
        backbone="swin2d_small",
        modality_type="vision",
        in_channels=3,
        feature_dim=512,
    )
    .set_fusion("fused_attention", num_heads=8, use_kronecker=True, output_dim=256)
    .set_head("classification", num_classes=4, dropout=0.3)
    .build()
)

# 测试
ct = torch.randn(2, 1, 64, 128, 128)
pathology = torch.randn(2, 3, 224, 224)
inputs = {"radiology": ct, "pathology": pathology}
logits = model(inputs)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**这个示例的特点：**
- 3D Swin Transformer 处理 CT
- 2D Swin Transformer 处理病理图像
- Fused Attention 融合策略
- Kronecker 积增强特征交互

## 示例 3：多实例学习（MIL）

```python
# 构建带 MIL 的模型
model = (
    MultiModalModelBuilder()
    .add_modality(
        "radiology",
        backbone="swin3d_small",
        modality_type="vision3d",
        in_channels=1,
        feature_dim=512,
    )
    .add_modality(
        "pathology",
        backbone="swin2d_small",
        modality_type="vision",
        in_channels=3,
        feature_dim=512,
    )
    .add_mil_aggregation("pathology", strategy="attention", attention_dim=128)
    .set_fusion("fused_attention", num_heads=8, output_dim=256)
    .set_head("classification", num_classes=4)
    .build()
)

# 测试（病理图像有 10 个 patches）
ct = torch.randn(2, 1, 64, 128, 128)
pathology_patches = torch.randn(2, 10, 3, 224, 224)  # [batch, num_patches, C, H, W]
inputs = {"radiology": ct, "pathology": pathology_patches}

logits, features = model(inputs, return_features=True)

# 查看 MIL 注意力权重
if "mil_attention_weights" in features:
    attention = features["mil_attention_weights"]["pathology"]
    print(f"Attention weights shape: {attention.shape}")  # [2, 10]
    print(f"Sample 0 attention: {attention[0].squeeze().tolist()}")
```

**MIL 聚合策略：**
- `"mean"`: 平均池化
- `"max"`: 最大池化
- `"attention"`: 注意力加权（推荐）
- `"gated"`: 门控注意力
- `"deepsets"`: DeepSets 聚合
- `"transformer"`: Transformer 聚合

## 示例 4：不同融合策略对比

```python
strategies = ["concat", "gated", "attention", "kronecker", "fused_attention"]

xray = torch.randn(4, 3, 224, 224)
clinical = torch.randn(4, 10)
inputs = {"xray": xray, "clinical": clinical}

for strategy in strategies:
    model = (
        MultiModalModelBuilder()
        .add_modality("xray", backbone="resnet18", modality_type="vision")
        .add_modality("clinical", backbone="mlp", modality_type="tabular", input_dim=10)
        .set_fusion(strategy)
        .set_head("classification", num_classes=2)
        .build()
    )

    logits = model(inputs)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{strategy:20s} - Params: {num_params:,} - Output: {logits.shape}")
```

**输出示例：**
```
concat               - Params: 11,234,562 - Output: torch.Size([4, 2])
gated                - Params: 11,298,178 - Output: torch.Size([4, 2])
attention            - Params: 11,312,450 - Output: torch.Size([4, 2])
kronecker            - Params: 11,445,826 - Output: torch.Size([4, 2])
fused_attention      - Params: 11,523,906 - Output: torch.Size([4, 2])
```

**融合策略选择：**
- `concat`: 最简单，参数最少
- `gated`: 平衡性能和复杂度（推荐）
- `attention`: 自适应权重
- `kronecker`: 强特征交互
- `fused_attention`: 最强性能，参数最多

## 示例 5：三模态模型

```python
model = (
    MultiModalModelBuilder()
    .add_modality("xray", backbone="resnet18", modality_type="vision", feature_dim=256)
    .add_modality("ct", backbone="swin3d_tiny", modality_type="vision3d", in_channels=1, feature_dim=256)
    .add_modality("clinical", backbone="mlp", modality_type="tabular", input_dim=15, feature_dim=256)
    .set_fusion("concat", output_dim=256)  # 对于 >2 模态，实际执行 concatenate + projection
    .set_head("classification", num_classes=3)
    .build()
)

# 测试
xray = torch.randn(2, 3, 224, 224)
ct = torch.randn(2, 1, 32, 64, 64)
clinical = torch.randn(2, 15)
inputs = {"xray": xray, "ct": ct, "clinical": clinical}

logits, features = model(inputs, return_features=True)

print(f"Fused features shape: {features['fused_features'].shape}")

# 获取模态贡献度
contributions = model.get_modality_contribution()
print("\nModality contributions:")
for modality, contrib in contributions.items():
    print(f"  {modality}: {contrib:.3f}")
```

**输出示例：**
```
Fused features shape: torch.Size([2, 256])

Modality contributions:
  xray: 0.342
  ct: 0.381
  clinical: 0.277
```

## 示例 6：生存分析模型

```python
# Cox 生存分析
model = (
    MultiModalModelBuilder()
    .add_modality("radiology", backbone="swin3d_small", modality_type="vision3d", in_channels=1, feature_dim=512)
    .add_modality("pathology", backbone="swin2d_small", modality_type="vision", in_channels=3, feature_dim=512)
    .set_fusion("fused_attention", num_heads=8, output_dim=256)
    .set_head("survival_cox", hidden_dims=[128, 64])
    .build()
)

# 测试
ct = torch.randn(2, 1, 64, 128, 128)
pathology = torch.randn(2, 3, 224, 224)
inputs = {"radiology": ct, "pathology": pathology}

risk_scores = model(inputs)
print(f"Risk scores: {risk_scores.squeeze().tolist()}")
```

**生存分析任务头：**
- `"survival_cox"`: Cox 比例风险模型
- `"survival_deep"`: DeepSurv
- `"survival_discrete"`: 离散时间生存分析

## 示例 7：从配置字典构建

```python
config = {
    "modalities": {
        "xray": {
            "backbone": "resnet18",
            "modality_type": "vision",
            "feature_dim": 256,
        },
        "clinical": {
            "backbone": "mlp",
            "modality_type": "tabular",
            "input_dim": 10,
            "feature_dim": 64,
        },
    },
    "fusion": {
        "strategy": "attention",
        "output_dim": 128,
    },
    "head": {
        "task_type": "classification",
        "num_classes": 2,
        "dropout": 0.3,
    },
}

# 从配置构建
builder = MultiModalModelBuilder.from_config(config)
model = builder.build()

# 测试
xray = torch.randn(4, 3, 224, 224)
clinical = torch.randn(4, 10)
inputs = {"xray": xray, "clinical": clinical}
logits = model(inputs)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**配置字典结构：**
- `modalities`: 模态配置字典
- `fusion`: 融合策略配置
- `head`: 任务头配置

## 示例 8：特征提取

```python
model = (
    MultiModalModelBuilder()
    .add_modality("xray", backbone="resnet18", modality_type="vision", in_channels=1)
    .add_modality("clinical", backbone="mlp", modality_type="tabular", input_dim=10)
    .set_fusion("attention")
    .set_head("classification", num_classes=2)
    .build()
)

# 提取中间特征
xray = torch.randn(4, 3, 224, 224)
clinical = torch.randn(4, 10)
inputs = {"xray": xray, "clinical": clinical}

logits, features = model(inputs, return_features=True)

print("Extracted features:")
print(f"  X-ray features: {features['modality_features']['xray'].shape}")
print(f"  Clinical features: {features['modality_features']['clinical'].shape}")
print(f"  Fused features: {features['fused_features'].shape}")

if "fusion_aux" in features and features["fusion_aux"] is not None:
    print(f"  Fusion auxiliary outputs: {list(features['fusion_aux'].keys())}")
```

**可提取的特征：**
- `modality_features`: 各模态的特征
- `fused_features`: 融合后的特征
- `fusion_aux`: 融合模块的辅助输出（如注意力权重）
- `mil_attention_weights`: MIL 注意力权重

## 示例 9：不同 MIL 策略对比

```python
strategies = ["mean", "max", "attention", "gated", "deepsets", "transformer"]

ct = torch.randn(2, 1, 64, 128, 128)
pathology_patches = torch.randn(2, 10, 3, 224, 224)
inputs = {"radiology": ct, "pathology": pathology_patches}

for strategy in strategies:
    model = (
        MultiModalModelBuilder()
        .add_modality("radiology", backbone="swin3d_tiny", modality_type="vision3d", in_channels=1, feature_dim=256)
        .add_modality("pathology", backbone="resnet18", modality_type="vision", feature_dim=256)
        .add_mil_aggregation("pathology", strategy=strategy)
        .set_fusion("concat")
        .set_head("classification", num_classes=4)
        .build()
    )

    logits = model(inputs)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{strategy:15s} - Params: {num_params:,} - Output: {logits.shape}")
```

**MIL 策略选择：**
- `mean/max`: 简单快速，无额外参数
- `attention`: 自适应权重（推荐）
- `gated`: 门控注意力，更强表达能力
- `deepsets`: 排列不变性
- `transformer`: 最强性能，参数最多

## 示例 10：自定义骨干网络

```python
import torch.nn as nn

# 定义自定义骨干网络
class CustomBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 128)
        self.output_dim = 128  # 必须定义 output_dim

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

# 使用自定义骨干网络
custom_backbone = CustomBackbone()

model = (
    MultiModalModelBuilder()
    .add_modality("xray", backbone=custom_backbone, modality_type="custom")
    .add_modality("clinical", backbone="mlp", modality_type="tabular", input_dim=10)
    .set_fusion("concat")
    .set_head("classification", num_classes=2)
    .build()
)

# 测试
xray = torch.randn(4, 3, 224, 224)
clinical = torch.randn(4, 10)
inputs = {"xray": xray, "clinical": clinical}
logits = model(inputs)

print(f"Custom backbone output dim: {custom_backbone.output_dim}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**自定义骨干网络要求：**
1. 继承 `nn.Module`
2. 定义 `output_dim` 属性
3. 实现 `forward` 方法

## API 参考

### MultiModalModelBuilder

#### add_modality()

```python
def add_modality(
    self,
    name: str,
    backbone: str | nn.Module,
    modality_type: str = "vision",
    **kwargs
) -> "MultiModalModelBuilder":
    """添加模态"""
```

**参数：**
- `name`: 模态名称（用于输入字典的键）
- `backbone`: 骨干网络名称或自定义模块
- `modality_type`: 模态类型
  - `"vision"`: 2D 视觉
  - `"vision3d"`: 3D 视觉
  - `"tabular"`: 表格数据
  - `"custom"`: 自定义
- `**kwargs`: 传递给骨干网络的额外参数
  - `in_channels`: 输入通道数
  - `feature_dim`: 输出特征维度
  - `input_dim`: 表格数据输入维度
  - `pretrained`: 是否使用预训练权重

#### add_mil_aggregation()

```python
def add_mil_aggregation(
    self,
    modality_name: str,
    strategy: str = "attention",
    **kwargs
) -> "MultiModalModelBuilder":
    """为指定模态添加 MIL 聚合"""
```

**参数：**
- `modality_name`: 模态名称
- `strategy`: 聚合策略
  - `"mean"`, `"max"`, `"attention"`, `"gated"`, `"deepsets"`, `"transformer"`
- `**kwargs`: 策略特定参数
  - `attention_dim`: 注意力维度（用于 attention/gated）
  - `num_heads`: 注意力头数（用于 transformer）

#### set_fusion()

```python
def set_fusion(
    self,
    fusion_type: str,
    **kwargs
) -> "MultiModalModelBuilder":
    """设置融合策略"""
```

**参数：**
- `fusion_type`: 融合类型
  - `"concat"`, `"gated"`, `"attention"`, `"cross_attention"`, `"bilinear"`, `"kronecker"`, `"fused_attention"`
- `**kwargs`: 融合特定参数
  - `hidden_dim`: 隐藏层维度
  - `output_dim`: 输出维度
  - `num_heads`: 注意力头数
  - `dropout`: Dropout 比例
  - `use_kronecker`: 是否使用 Kronecker 积

#### set_head()

```python
def set_head(
    self,
    task_type: str,
    **kwargs
) -> "MultiModalModelBuilder":
    """设置任务头"""
```

**参数：**
- `task_type`: 任务类型
  - `"classification"`: 分类
  - `"survival_cox"`: Cox 生存分析
  - `"survival_deep"`: DeepSurv
  - `"survival_discrete"`: 离散时间生存分析
- `**kwargs`: 任务特定参数
  - `num_classes`: 类别数（分类任务）
  - `hidden_dims`: 隐藏层维度列表
  - `dropout`: Dropout 比例

#### build()

```python
def build(self) -> nn.Module:
    """构建并返回模型"""
```

#### from_config()

```python
@classmethod
def from_config(cls, config: dict) -> "MultiModalModelBuilder":
    """从配置字典创建 Builder"""
```

## 最佳实践

### 1. 特征维度对齐

```python
# 推荐：所有模态使用相同的 feature_dim
model = (
    MultiModalModelBuilder()
    .add_modality("xray", backbone="resnet18", feature_dim=256)
    .add_modality("ct", backbone="swin3d_tiny", feature_dim=256)
    .add_modality("clinical", backbone="mlp", input_dim=10, feature_dim=256)
    .set_fusion("concat")
    .build()
)
```

### 2. 渐进式构建

```python
# 先构建简单模型验证
simple_model = (
    MultiModalModelBuilder()
    .add_modality("xray", backbone="resnet18")
    .add_modality("clinical", backbone="mlp", input_dim=10)
    .set_fusion("concat")
    .set_head("classification", num_classes=2)
    .build()
)

# 验证通过后，升级到复杂模型
complex_model = (
    MultiModalModelBuilder()
    .add_modality("xray", backbone="swin_s", feature_dim=512)
    .add_modality("clinical", backbone="mlp", input_dim=10, feature_dim=128)
    .set_fusion("fused_attention", num_heads=8)
    .set_head("classification", num_classes=2, dropout=0.3)
    .build()
)
```

### 3. 模型验证

```python
# 构建模型后立即验证
model = builder.build()

# 检查参数量
num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params:,}")

# 测试前向传播
dummy_inputs = {
    "xray": torch.randn(2, 3, 224, 224),
    "clinical": torch.randn(2, 10)
}
try:
    output = model(dummy_inputs)
    print(f"Output shape: {output.shape}")
except Exception as e:
    print(f"Forward pass failed: {e}")
```

### 4. 保存和加载

```python
# 保存模型
torch.save(model.state_dict(), "model.pth")

# 保存完整模型（包括架构）
torch.save(model, "model_full.pth")

# 加载模型
model = builder.build()
model.load_state_dict(torch.load("model.pth"))

# 或加载完整模型
model = torch.load("model_full.pth")
```

## 常见问题

**Q: Builder API 和配置文件哪个更好？**
A: 配置文件适合实验管理，Builder API 适合快速原型和自定义。可以结合使用。

**Q: 如何添加自定义融合策略？**
A: 实现自定义融合模块，然后作为 `nn.Module` 传递给 Builder。

**Q: 可以动态修改模型吗？**
A: 可以，但需要重新构建。建议使用配置文件管理不同版本。

**Q: 如何处理不同尺寸的输入？**
A: 使用自适应池化或在数据加载时统一尺寸。

## 下一步

- [训练第一个模型](../../getting-started/first-model.md) - 开始训练
- [配置文件详解](configs.md) - 深入理解配置
- [数据准备指南](data-prep.md) - 准备训练数据
