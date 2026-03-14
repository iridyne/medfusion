# Fusion API

多模态融合策略模块，提供多种特征融合方法。

## 概述

Fusion 模块提供了 8 种融合策略，用于组合来自不同模态（如影像和表格数据）的特征：

- **基础融合**: Concatenate, Gated, Bilinear
- **注意力融合**: Attention, Cross-Attention
- **高级融合**: Kronecker, Fused-Attention, Self-Attention

所有融合模块继承自 `BaseFusion` 基类，支持统一的接口。

## 核心类

### BaseFusion

所有融合策略的抽象基类。

**方法：**
- `forward(vision_features, tabular_features)` - 融合两个模态的特征
- `get_output_dim()` - 返回融合后的特征维度

### ConcatenateFusion

简单的特征拼接融合。

**参数：**
- `vision_dim` (int): 视觉特征维度
- `tabular_dim` (int): 表格特征维度
- `output_dim` (int): 输出维度，默认 96
- `dropout` (float): Dropout 率，默认 0.3

**特性：**
- 最简单的融合策略
- 将两个模态特征拼接后投影到输出维度
- 使用 ReLU + LayerNorm + Dropout

**示例：**
```python
from med_core.fusion import ConcatenateFusion

fusion = ConcatenateFusion(
    vision_dim=512,
    tabular_dim=64,
    output_dim=256,
    dropout=0.3
)

# 前向传播
fused, aux = fusion(vision_features, tabular_features)
```

### GatedFusion

带有可学习门控机制的融合。

**参数：**
- `vision_dim` (int): 视觉特征维度
- `tabular_dim` (int): 表格特征维度
- `output_dim` (int): 输出维度，默认 96
- `initial_vision_weight` (float): 视觉模态初始权重，默认 0.3
- `initial_tabular_weight` (float): 表格模态初始权重，默认 0.7
- `learnable_weights` (bool): 全局权重是否可学习，默认 True
- `dropout` (float): Dropout 率，默认 0.3

**特性：**
- 学习每个样本的最优模态平衡
- 全局权重 (alpha) + 实例级门控 (z)
- 返回门控值用于可解释性分析

**示例：**
```python
from med_core.fusion import GatedFusion

fusion = GatedFusion(
    vision_dim=512,
    tabular_dim=64,
    output_dim=256,
    initial_vision_weight=0.3,
    initial_tabular_weight=0.7
)

fused, aux = fusion(vision_features, tabular_features)
# aux 包含: gate_values, vision_weight, tabular_weight

# 获取当前模态权重
weights = fusion.get_modality_weights()
print(f"Vision: {weights['vision_weight']:.3f}")
print(f"Tabular: {weights['tabular_weight']:.3f}")
```

### AttentionFusion

基于自注意力的融合。

**参数：**
- `vision_dim` (int): 视觉特征维度
- `tabular_dim` (int): 表格特征维度
- `output_dim` (int): 输出维度，默认 96
- `num_heads` (int): 注意力头数，默认 4
- `dropout` (float): Dropout 率，默认 0.3

**特性：**
- 将每个模态视为一个 token
- 使用可学习的 [CLS] token 聚合信息
- 多头自注意力捕获跨模态交互
- 返回注意力权重用于可视化

**示例：**
```python
from med_core.fusion import AttentionFusion

fusion = AttentionFusion(
    vision_dim=512,
    tabular_dim=64,
    output_dim=256,
    num_heads=8
)

fused, aux = fusion(vision_features, tabular_features)
# aux 包含: attention_weights (B, num_heads, 3, 3)

# 获取注意力权重
attn_weights = fusion.get_attention_weights()
```

### CrossAttentionFusion

跨模态注意力融合。

**参数：**
- `vision_dim` (int): 视觉特征维度
- `tabular_dim` (int): 表格特征维度
- `output_dim` (int): 输出维度，默认 96
- `num_heads` (int): 注意力头数，默认 4
- `dropout` (float): Dropout 率，默认 0.3

**特性：**
- 视觉特征关注表格特征（vision → tabular）
- 表格特征关注视觉特征（tabular → vision）
- 双向交叉注意力捕获模态间依赖
- 返回双向注意力权重

**示例：**
```python
from med_core.fusion import CrossAttentionFusion

fusion = CrossAttentionFusion(
    vision_dim=512,
    tabular_dim=64,
    output_dim=256,
    num_heads=4
)

fused, aux = fusion(vision_features, tabular_features)
# aux 包含: vision_to_tabular_attn, tabular_to_vision_attn
```

### BilinearFusion

双线性池化融合。

**参数：**
- `vision_dim` (int): 视觉特征维度
- `tabular_dim` (int): 表格特征维度
- `output_dim` (int): 输出维度，默认 96
- `rank` (int): 低秩近似的秩，默认 16
- `dropout` (float): Dropout 率，默认 0.3

**特性：**
- 捕获模态间的乘性交互
- 使用低秩分解提高效率
- 包含残差连接保留原始信息

**示例：**
```python
from med_core.fusion import BilinearFusion

fusion = BilinearFusion(
    vision_dim=512,
    tabular_dim=64,
    output_dim=256,
    rank=32  # 更高的秩捕获更多交互
)

fused, aux = fusion(vision_features, tabular_features)
```

### KroneckerFusion

基于 Kronecker 积的融合。

**特性：**
- 高效的张量积操作
- 捕获高阶特征交互
- 适用于需要丰富特征组合的场景

### FusedAttentionFusion

融合注意力机制。

**特性：**
- 结合自注意力和交叉注意力
- 多层注意力堆叠
- 适用于复杂的多模态场景

### SelfAttentionFusion

自注意力融合变体。

**包含：**
- `AdditiveAttentionFusion` - 加性注意力
- `BilinearAttentionFusion` - 双线性注意力
- `GatedAttentionFusion` - 门控注意力

## 工厂函数

### create_fusion_module

创建融合模块的工厂函数。

**参数：**
- `fusion_type` (str): 融合策略类型
- `vision_dim` (int): 视觉特征维度
- `tabular_dim` (int): 表格特征维度
- `output_dim` (int): 输出维度，默认 96
- `**kwargs`: 传递给具体融合类的额外参数

**支持的融合类型：**
- `"concatenate"` - 拼接融合
- `"gated"` - 门控融合
- `"attention"` - 注意力融合
- `"cross_attention"` - 交叉注意力融合
- `"bilinear"` - 双线性融合

**别名：**
- `"concat"` → `"concatenate"`
- `"attn"` → `"attention"`
- `"cross_attn"` → `"cross_attention"`
- `"gate"` → `"gated"`

**示例：**
```python
from med_core.fusion import create_fusion_module

# 创建门控融合
fusion = create_fusion_module(
    fusion_type="gated",
    vision_dim=512,
    tabular_dim=64,
    output_dim=256,
    initial_vision_weight=0.4,
    initial_tabular_weight=0.6
)

# 使用别名
fusion = create_fusion_module(
    fusion_type="attn",  # 等同于 "attention"
    vision_dim=512,
    tabular_dim=64,
    output_dim=256,
    num_heads=8
)
```

### list_available_fusions

列出所有可用的融合策略。

**返回：**
- `list[str]`: 可用融合策略名称列表

**示例：**
```python
from med_core.fusion import list_available_fusions

strategies = list_available_fusions()
print(strategies)
# ['concatenate', 'gated', 'attention', 'cross_attention', 'bilinear']
```

## 配置示例

在 YAML 配置文件中使用融合模块：

```yaml
model:
  fusion:
    strategy: attention
    output_dim: 256
    num_heads: 8
    dropout: 0.3
```

```yaml
model:
  fusion:
    strategy: gated
    output_dim: 256
    initial_vision_weight: 0.3
    initial_tabular_weight: 0.7
    learnable_weights: true
```

## 选择指南

**简单场景：**
- 使用 `ConcatenateFusion` - 快速基线

**需要模态平衡：**
- 使用 `GatedFusion` - 自动学习模态重要性

**需要跨模态交互：**
- 使用 `AttentionFusion` - 捕获全局依赖
- 使用 `CrossAttentionFusion` - 双向模态交互

**需要高阶交互：**
- 使用 `BilinearFusion` - 乘性特征组合
- 使用 `KroneckerFusion` - 张量积操作

## 参考

完整实现请参考：
- `/home/yixian/Projects/med-ml/medfusion/med_core/fusion/strategies.py` - 基础融合策略
- `/home/yixian/Projects/med-ml/medfusion/med_core/fusion/kronecker.py` - Kronecker 融合
- `/home/yixian/Projects/med-ml/medfusion/med_core/fusion/fused_attention.py` - 融合注意力
- `/home/yixian/Projects/med-ml/medfusion/med_core/fusion/self_attention.py` - 自注意力变体
