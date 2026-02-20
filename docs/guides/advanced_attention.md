# 高级注意力机制指南

本指南介绍 MedFusion v0.2.0 中新增的高级注意力机制，包括 SE、ECA、Transformer 等。

## 概述

MedFusion 现在支持多种先进的注意力机制：

1. **SE (Squeeze-and-Excitation)**: 通道注意力
2. **ECA (Efficient Channel Attention)**: 高效通道注意力
3. **Spatial Attention**: 空间注意力
4. **CBAM**: 通道 + 空间注意力
5. **Transformer Attention**: 多头自注意力

## 快速开始

### 基本使用

```python
from med_core.attention_supervision import create_attention_module

# 创建 SE 注意力
se_attention = create_attention_module(
    attention_type="se",
    channels=256,
    reduction=16,
)

# 在模型中使用
import torch
x = torch.randn(2, 256, 14, 14)
out = se_attention(x)  # (2, 256, 14, 14)
```

### 工厂函数

使用工厂函数可以轻松切换不同的注意力类型：

```python
from med_core.attention_supervision import create_attention_module

# 支持的类型
attention_types = ["se", "eca", "spatial", "cbam", "transformer"]

for attn_type in attention_types:
    attention = create_attention_module(attn_type, channels=256)
    x = torch.randn(2, 256, 14, 14)
    out = attention(x)
```

## 注意力模块详解

### 1. SE (Squeeze-and-Excitation) 注意力

**原理**: 通过全局平均池化和两层全连接网络学习通道注意力权重。

**优点**:
- 参数量小
- 计算高效
- 性能提升明显

**使用场景**:
- 需要增强重要通道
- 计算资源有限
- 通道数较多

**示例**:

```python
from med_core.attention_supervision import SEAttention

# 创建 SE 模块
se = SEAttention(
    channels=256,      # 输入通道数
    reduction=16,      # 降维比例
    activation="relu", # 激活函数
)

# 前向传播
x = torch.randn(2, 256, 14, 14)
out = se(x)  # (2, 256, 14, 14)

# 获取注意力权重（用于可视化）
weights = se.get_attention_weights(x)  # (2, 256)
```

**参数说明**:
- `channels`: 输入通道数
- `reduction`: 降维比例，越大参数越少（默认 16）
- `activation`: 激活函数，可选 "relu", "gelu", "silu"

**参考文献**:
- Hu et al. "Squeeze-and-Excitation Networks" CVPR 2018

---

### 2. ECA (Efficient Channel Attention) 注意力

**原理**: 使用 1D 卷积实现高效的通道注意力，避免降维。

**优点**:
- 参数量更少
- 性能优于 SE
- 自适应卷积核大小

**使用场景**:
- 需要极致的效率
- 通道数变化较大
- 追求最佳性能

**示例**:

```python
from med_core.attention_supervision import ECAAttention

# 创建 ECA 模块（自动计算卷积核大小）
eca = ECAAttention(channels=256)

# 或手动指定卷积核大小
eca = ECAAttention(
    channels=256,
    kernel_size=5,  # 手动指定
)

# 前向传播
x = torch.randn(2, 256, 14, 14)
out = eca(x)  # (2, 256, 14, 14)

# 获取注意力权重
weights = eca.get_attention_weights(x)  # (2, 256)
```

**参数说明**:
- `channels`: 输入通道数
- `kernel_size`: 1D 卷积核大小（默认自动计算）
- `gamma`, `b`: 自动计算卷积核大小的参数

**参考文献**:
- Wang et al. "ECA-Net: Efficient Channel Attention for Deep CNNs" CVPR 2020

---

### 3. Spatial Attention (空间注意力)

**原理**: 学习空间维度的注意力权重，关注重要的空间位置。

**优点**:
- 适合目标定位
- 提高空间感知能力
- 可视化效果好

**使用场景**:
- 目标检测
- 病灶定位
- 显著性检测

**示例**:

```python
from med_core.attention_supervision import SpatialAttention

# 创建空间注意力模块
spatial = SpatialAttention(kernel_size=7)

# 前向传播
x = torch.randn(2, 256, 14, 14)
out = spatial(x)  # (2, 256, 14, 14)

# 获取注意力权重
weights = spatial.get_attention_weights(x)  # (2, 1, 14, 14)
```

**参数说明**:
- `kernel_size`: 卷积核大小（默认 7）

---

### 4. CBAM (Convolutional Block Attention Module)

**原理**: 结合通道注意力和空间注意力，先通道后空间。

**优点**:
- 性能强大
- 结合两者优势
- 广泛应用

**使用场景**:
- 需要同时关注通道和空间
- 追求最佳性能
- 通用场景

**示例**:

```python
from med_core.attention_supervision import CBAM

# 创建 CBAM 模块
cbam = CBAM(
    channels=256,
    reduction=16,      # 通道注意力的降维比例
    spatial_kernel=7,  # 空间注意力的卷积核大小
)

# 前向传播
x = torch.randn(2, 256, 14, 14)
out = cbam(x)  # (2, 256, 14, 14)
```

**参数说明**:
- `channels`: 输入通道数
- `reduction`: 通道注意力的降维比例
- `spatial_kernel`: 空间注意力的卷积核大小

**参考文献**:
- Woo et al. "CBAM: Convolutional Block Attention Module" ECCV 2018

---

### 5. Transformer Attention (Transformer 注意力)

**原理**: 多头自注意力机制，全局感受野。

**优点**:
- 全局建模能力
- 捕获长距离依赖
- 性能优异

**使用场景**:
- 需要全局信息
- 长距离依赖建模
- 大规模数据

**示例**:

```python
from med_core.attention_supervision import TransformerAttention2D

# 创建 Transformer 注意力模块
transformer = TransformerAttention2D(
    channels=256,
    num_heads=8,
    qkv_bias=False,
    attn_drop=0.0,
    proj_drop=0.0,
)

# 前向传播
x = torch.randn(2, 256, 14, 14)
out = transformer(x)  # (2, 256, 14, 14)

# 获取注意力权重
weights = transformer.get_attention_weights(x)  # (2, 8, 196, 196)
```

**参数说明**:
- `channels`: 输入通道数
- `num_heads`: 注意力头数
- `qkv_bias`: 是否使用 QKV 偏置
- `attn_drop`: 注意力 dropout
- `proj_drop`: 投影 dropout

---

## 注意力监督

为了提高注意力的可解释性和性能，MedFusion 提供了注意力监督机制。

### 1. 通道注意力监督

```python
from med_core.attention_supervision import (
    SEAttention,
    ChannelAttentionSupervision,
)

# 创建注意力模块和监督
se = SEAttention(channels=256, reduction=16)
supervision = ChannelAttentionSupervision(
    loss_weight=0.1,           # 损失权重
    diversity_weight=0.1,      # 多样性损失权重
    sparsity_weight=0.1,       # 稀疏性损失权重
)

# 前向传播
x = torch.randn(2, 256, 14, 14)
out = se(x)

# 获取注意力权重
weights = se.get_attention_weights(x)

# 计算监督损失
loss = supervision(weights, x)
print(f"Total loss: {loss.total_loss.item():.4f}")
print(f"Components: {loss.components}")
```

### 2. 空间注意力监督

```python
from med_core.attention_supervision import (
    SpatialAttention,
    SpatialAttentionSupervision,
)

# 创建注意力模块和监督
spatial = SpatialAttention(kernel_size=7)
supervision = SpatialAttentionSupervision(
    loss_weight=0.1,
    consistency_weight=0.1,    # 一致性损失权重
    smoothness_weight=0.1,     # 平滑性损失权重
)

# 前向传播
x = torch.randn(2, 256, 14, 14)
out = spatial(x)

# 获取注意力权重
weights = spatial.get_attention_weights(x)

# 计算监督损失（可选提供目标掩码）
targets = torch.randint(0, 2, (2, 1, 14, 14)).float()
loss = supervision(weights, x, targets)
```

### 3. Transformer 注意力监督

```python
from med_core.attention_supervision import (
    TransformerAttention2D,
    TransformerAttentionSupervision,
)

# 创建注意力模块和监督
transformer = TransformerAttention2D(channels=256, num_heads=8)
supervision = TransformerAttentionSupervision(
    loss_weight=0.1,
    head_diversity_weight=0.1,  # 头多样性损失权重
    locality_weight=0.1,         # 局部性损失权重
)

# 前向传播
x = torch.randn(2, 256, 14, 14)
out = transformer(x)

# 获取注意力权重
weights = transformer.get_attention_weights(x)

# 计算监督损失
loss = supervision(weights, x)
```

### 4. 混合注意力监督

```python
from med_core.attention_supervision import HybridAttentionSupervision

# 创建混合监督
supervision = HybridAttentionSupervision(
    loss_weight=0.1,
    channel_weight=1.0,
    spatial_weight=1.0,
    transformer_weight=1.0,
)

# 收集多种注意力权重
attentions = {
    "channel": channel_weights,    # (B, C)
    "spatial": spatial_weights,    # (B, 1, H, W)
    "transformer": transformer_weights,  # (B, num_heads, N, N)
}

# 计算监督损失
loss = supervision(attentions, features)
```

---

## 完整示例

### 示例 1: 在 ResNet 中添加 SE 注意力

```python
import torch.nn as nn
from med_core.attention_supervision import SEAttention

class SEResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEAttention(channels, reduction=16)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 应用 SE 注意力
        out = self.se(out)
        
        out += identity
        out = self.relu(out)
        
        return out
```

### 示例 2: 带注意力监督的分类模型

```python
import torch.nn as nn
from med_core.attention_supervision import (
    SEAttention,
    ChannelAttentionSupervision,
)

class AttentionClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 骨干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.ReLU(),
        )
        
        # SE 注意力
        self.attention = SEAttention(channels=256, reduction=16)
        
        # 注意力监督
        self.attention_supervision = ChannelAttentionSupervision(
            loss_weight=0.1,
            diversity_weight=0.1,
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x, return_attention=False):
        # 特征提取
        features = self.backbone(x)
        
        # 应用注意力
        attended_features = self.attention(features)
        
        # 分类
        logits = self.classifier(attended_features)
        
        if return_attention:
            attn_weights = self.attention.get_attention_weights(features)
            return logits, attn_weights
        
        return logits
    
    def compute_loss(self, x, y):
        # 前向传播
        logits, attn_weights = self.forward(x, return_attention=True)
        
        # 分类损失
        cls_loss = nn.CrossEntropyLoss()(logits, y)
        
        # 注意力监督损失
        features = self.backbone(x)
        attn_loss = self.attention_supervision(attn_weights, features)
        
        # 总损失
        total_loss = cls_loss + attn_loss.total_loss
        
        return total_loss, {
            "cls_loss": cls_loss.item(),
            "attn_loss": attn_loss.total_loss.item(),
            **{k: v.item() for k, v in attn_loss.components.items()},
        }

# 使用
model = AttentionClassifier(num_classes=10)
x = torch.randn(2, 3, 224, 224)
y = torch.randint(0, 10, (2,))

loss, loss_dict = model.compute_loss(x, y)
print(f"Total loss: {loss.item():.4f}")
print(f"Loss components: {loss_dict}")
```

---

## 性能对比

| 注意力类型 | 参数量 | 计算量 | 性能 | 适用场景 |
|-----------|--------|--------|------|---------|
| SE | 中 | 低 | 好 | 通用 |
| ECA | 低 | 低 | 优 | 效率优先 |
| Spatial | 低 | 低 | 中 | 目标定位 |
| CBAM | 中 | 中 | 优 | 性能优先 |
| Transformer | 高 | 高 | 优 | 全局建模 |

---

## 最佳实践

### 1. 选择合适的注意力类型

- **通道注意力 (SE/ECA)**: 适合增强重要特征通道
- **空间注意力**: 适合目标定位和显著性检测
- **CBAM**: 适合需要同时关注通道和空间的场景
- **Transformer**: 适合需要全局建模的场景

### 2. 注意力位置

- **浅层**: 关注低级特征（边缘、纹理）
- **深层**: 关注高级语义特征
- **多层**: 在多个层级添加注意力

### 3. 超参数调优

- **reduction**: SE 的降维比例，通常 8-16
- **num_heads**: Transformer 的头数，通常 4-16
- **loss_weight**: 注意力监督的权重，通常 0.01-0.1

### 4. 训练技巧

- 先训练主任务，再添加注意力监督
- 使用较小的学习率训练注意力模块
- 监控注意力权重的分布

---

## 常见问题

### Q1: 如何选择注意力类型？

**A**: 根据任务需求：
- 分类任务：SE 或 ECA
- 检测任务：CBAM 或 Spatial
- 分割任务：Spatial 或 Transformer
- 通用场景：CBAM

### Q2: 注意力监督是否必需？

**A**: 不是必需的，但推荐使用：
- 提高可解释性
- 改善小样本学习
- 加速收敛

### Q3: 如何可视化注意力？

**A**: 使用 `get_attention_weights()` 方法：

```python
# 获取注意力权重
weights = attention.get_attention_weights(x)

# 可视化
import matplotlib.pyplot as plt
plt.imshow(weights[0].detach().cpu().numpy())
plt.colorbar()
plt.show()
```

### Q4: 注意力模块会增加多少计算量？

**A**: 
- SE/ECA: <1% 额外计算
- Spatial: <1% 额外计算
- CBAM: ~1-2% 额外计算
- Transformer: 5-10% 额外计算

---

## 参考资源

### 论文

1. SE-Net: Hu et al. "Squeeze-and-Excitation Networks" CVPR 2018
2. ECA-Net: Wang et al. "ECA-Net: Efficient Channel Attention for Deep CNNs" CVPR 2020
3. CBAM: Woo et al. "CBAM: Convolutional Block Attention Module" ECCV 2018
4. Transformer: Vaswani et al. "Attention Is All You Need" NeurIPS 2017

### 代码

- `med_core/attention_supervision/advanced_attention.py` - 注意力模块实现
- `med_core/attention_supervision/advanced_supervision.py` - 注意力监督实现
- `examples/advanced_attention_demo.py` - 使用示例
- `tests/test_advanced_attention.py` - 单元测试

### 文档

- [API 文档](../api/attention_supervision.md)
- [注意力监督指南](attention_supervision.md)
- [性能优化指南](performance_optimization.md)

---

## 更新日志

### v0.2.0 (2026-02-20)

- ✨ 新增 SE 注意力
- ✨ 新增 ECA 注意力
- ✨ 新增空间注意力
- ✨ 新增 CBAM
- ✨ 新增 Transformer 注意力
- ✨ 新增通道注意力监督
- ✨ 新增空间注意力监督
- ✨ 新增 Transformer 注意力监督
- ✨ 新增混合注意力监督
- ✨ 新增工厂函数
- 📝 完善文档和示例
- ✅ 添加完整的单元测试
