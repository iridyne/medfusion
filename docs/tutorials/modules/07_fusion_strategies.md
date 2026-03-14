# 融合策略对比

**预计时间：20分钟**

本教程详细介绍 MedFusion 支持的 8 种融合策略，帮助你选择最适合你任务的融合方法。

## 融合策略概览

融合策略负责将视觉特征（如 CT 图像）和表格特征（如临床数据）组合成统一的表示。不同策略适用于不同的医学场景。

### 支持的融合策略

| 策略 | 复杂度 | 参数量 | 速度 | 适用场景 |
|------|--------|--------|------|----------|
| Concatenate | 低 | 少 | 快 | 基线模型、快速实验 |
| Gated | 中 | 中 | 中 | 模态重要性不均衡 |
| Attention | 高 | 多 | 慢 | 需要可解释性 |
| Cross-Attention | 高 | 多 | 慢 | 强跨模态交互 |
| Bilinear | 中 | 中 | 中 | 捕获乘性交互 |
| Kronecker | 高 | 多 | 慢 | 高阶特征交互 |
| Fused-Attention | 高 | 多 | 慢 | 复杂多模态场景 |
| Self-Attention | 高 | 多 | 慢 | 多模态序列数据 |

## 1. Concatenate Fusion（拼接融合）

### 原理
最简单的融合策略，直接将视觉和表格特征拼接后通过全连接层投影。

### 优点
- 实现简单，训练快速
- 参数量少，不易过拟合
- 适合作为基线模型

### 缺点
- 无法学习模态间交互
- 假设所有特征同等重要
- 表达能力有限

### 配置示例
```yaml
model:
  fusion:
    fusion_type: concatenate
    hidden_dim: 96
    dropout: 0.3
```

### 代码示例
```python
from med_core.fusion import create_fusion_module

fusion = create_fusion_module(
    fusion_type="concatenate",
    vision_dim=512,
    tabular_dim=32,
    output_dim=96,
    dropout=0.3
)
```

### 使用场景
- 快速原型验证
- 数据量较小时
- 作为其他策略的对照组

## 2. Gated Fusion（门控融合）

### 原理
使用可学习的门控机制动态调节视觉和表格模态的权重，包含全局权重和实例级门控。

### 优点
- 自动学习模态重要性
- 支持实例级自适应
- 可解释性较好（可查看门控值）

### 缺点
- 比拼接稍复杂
- 需要更多训练数据

### 配置示例
```yaml
model:
  fusion:
    fusion_type: gated
    hidden_dim: 96
    initial_vision_weight: 0.3
    initial_tabular_weight: 0.7
    learnable_weights: true
    dropout: 0.3
```

### 代码示例
```python
fusion = create_fusion_module(
    fusion_type="gated",
    vision_dim=512,
    tabular_dim=32,
    output_dim=96,
    initial_vision_weight=0.3,
    initial_tabular_weight=0.7,
    learnable_weights=True
)

# 训练后查看学到的权重
weights = fusion.get_modality_weights()
print(f"Vision weight: {weights['vision_weight']:.3f}")
print(f"Tabular weight: {weights['tabular_weight']:.3f}")
```

### 使用场景
- 模态重要性不确定时
- 需要了解模态贡献度
- 不同样本模态重要性差异大

## 3. Attention Fusion（自注意力融合）

### 原理
将每个模态视为一个 token，使用 Transformer 的自注意力机制学习跨模态交互，通过 CLS token 聚合信息。

### 优点
- 强大的跨模态建模能力
- 可视化注意力权重
- 适合多模态场景扩展

### 缺点
- 参数量较大
- 训练时间较长
- 需要较多数据

### 配置示例
```yaml
model:
  fusion:
    fusion_type: attention
    hidden_dim: 96
    num_heads: 4
    dropout: 0.3
```

### 代码示例
```python
fusion = create_fusion_module(
    fusion_type="attention",
    vision_dim=512,
    tabular_dim=32,
    output_dim=96,
    num_heads=4,
    dropout=0.3
)

# 推理时获取注意力权重
model.eval()
with torch.no_grad():
    output, aux = fusion(vision_feat, tabular_feat)
    attn_weights = aux['attention_weights']  # (B, num_heads, 3, 3)
```

### 使用场景
- 需要可解释性分析
- 多模态数据（>2 种）
- 有充足训练数据

## 4. Cross-Attention Fusion（交叉注意力融合）

### 原理
视觉特征关注表格特征，表格特征关注视觉特征，双向交叉注意力捕获细粒度交互。

### 优点
- 最强的跨模态交互能力
- 双向信息流动
- 适合模态互补性强的场景

### 缺点
- 计算开销最大
- 参数量最多
- 容易过拟合

### 配置示例
```yaml
model:
  fusion:
    fusion_type: cross_attention
    hidden_dim: 96
    num_heads: 4
    dropout: 0.3
```

### 代码示例
```python
fusion = create_fusion_module(
    fusion_type="cross_attention",
    vision_dim=512,
    tabular_dim=32,
    output_dim=96,
    num_heads=4
)

# 获取双向注意力权重
output, aux = fusion(vision_feat, tabular_feat)
v2t_attn = aux['vision_to_tabular_attn']  # 视觉→表格
t2v_attn = aux['tabular_to_vision_attn']  # 表格→视觉
```

### 使用场景
- 影像和临床数据高度互补
- 需要深度跨模态理解
- 数据量充足（>5000 样本）

## 5. Bilinear Fusion（双线性融合）

### 原理
通过低秩双线性池化捕获模态间的乘性交互，使用跳跃连接保留原始信息。

### 优点
- 捕获二阶交互
- 低秩近似保证效率
- 适合特征维度较高的场景

### 缺点
- 需要调节 rank 参数
- 对初始化敏感

### 配置示例
```yaml
model:
  fusion:
    fusion_type: bilinear
    hidden_dim: 96
    rank: 16
    dropout: 0.3
```

### 代码示例
```python
fusion = create_fusion_module(
    fusion_type="bilinear",
    vision_dim=512,
    tabular_dim=32,
    output_dim=96,
    rank=16  # 低秩近似的秩
)
```

### 使用场景
- 需要捕获特征乘性关系
- 视觉和表格特征维度都较高
- 对线性融合效果不满意

## 性能对比

### 实验设置
- 数据集：肺癌诊断（CT + 临床数据）
- 样本量：5000 训练 / 1000 验证 / 1000 测试
- 骨干网络：ResNet50 + MLP
- 训练：50 epochs, AdamW, lr=1e-4

### 结果对比

| 策略 | AUC | 准确率 | 训练时间 | 推理速度 | 参数量 |
|------|-----|--------|----------|----------|--------|
| Concatenate | 0.856 | 81.2% | 1.0x | 100 ms | 1.2M |
| Gated | 0.872 | 83.5% | 1.2x | 105 ms | 1.5M |
| Attention | 0.881 | 84.8% | 1.8x | 130 ms | 2.1M |
| Cross-Attention | 0.889 | 85.6% | 2.3x | 160 ms | 2.8M |
| Bilinear | 0.868 | 82.9% | 1.4x | 110 ms | 1.7M |

**结论：**
- Concatenate 适合快速实验
- Gated 性价比最高
- Cross-Attention 精度最高但开销大

## 选择建议

### 决策树

```
开始
├─ 数据量 < 1000？
│  └─ 使用 Concatenate 或 Gated
├─ 需要可解释性？
│  └─ 使用 Attention 或 Gated
├─ 模态互补性强？
│  └─ 使用 Cross-Attention
├─ 追求最高精度？
│  └─ 使用 Cross-Attention 或 Attention
└─ 追求速度？
   └─ 使用 Concatenate 或 Bilinear
```

### 实践建议

1. **从简单开始**：先用 Concatenate 建立基线
2. **逐步升级**：Concatenate → Gated → Attention
3. **对比实验**：在验证集上对比多种策略
4. **考虑资源**：GPU 内存有限时避免 Cross-Attention
5. **数据量匹配**：小数据集避免复杂策略

## 实战示例

### 完整训练流程

```python
from med_core.models import MultiModalModelBuilder

# 方案 1：快速原型（Concatenate）
builder = MultiModalModelBuilder(num_classes=2)
builder.add_modality("ct", backbone="resnet18", pretrained=True)
builder.add_modality("clinical", backbone="mlp", input_dim=32)
builder.set_fusion("concatenate", hidden_dim=64)
model = builder.build()

# 方案 2：平衡方案（Gated）
builder.set_fusion("gated", hidden_dim=96,
                   initial_vision_weight=0.3,
                   initial_tabular_weight=0.7)
model = builder.build()

# 方案 3：高精度方案（Cross-Attention）
builder.set_fusion("cross_attention", hidden_dim=128, num_heads=8)
model = builder.build()
```

### 配置文件示例

```yaml
# configs/fusion_comparison.yaml
experiment_name: fusion_strategy_comparison

model:
  num_classes: 2
  vision:
    backbone: resnet50
    pretrained: true
    feature_dim: 512
  tabular:
    hidden_dims: [64, 32]
    output_dim: 32
  fusion:
    fusion_type: gated  # 修改这里切换策略
    hidden_dim: 96
    dropout: 0.3

training:
  num_epochs: 50
  optimizer:
    type: adamw
    learning_rate: 0.0001
```

## 调试技巧

### 检查融合模块输出

```python
# 打印融合模块信息
print(fusion)

# 检查输出维度
dummy_vision = torch.randn(4, 512)
dummy_tabular = torch.randn(4, 32)
output, aux = fusion(dummy_vision, dummy_tabular)
print(f"Output shape: {output.shape}")  # 应该是 (4, 96)

# 检查辅助输出
if aux is not None:
    for key, value in aux.items():
        print(f"{key}: {value.shape if hasattr(value, 'shape') else value}")
```

### 可视化注意力权重

```python
import matplotlib.pyplot as plt

# 仅适用于 Attention/Cross-Attention
if hasattr(fusion, 'get_attention_weights'):
    attn = fusion.get_attention_weights()
    if attn is not None:
        plt.imshow(attn[0, 0].cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title('Attention Weights')
        plt.savefig('attention_map.png')
```

## 常见问题

**Q: 如何选择 num_heads？**
A: 通常 4-8 个头即可，过多会增加计算量但提升有限。

**Q: Bilinear 的 rank 如何设置？**
A: 建议从 16 开始，根据验证集表现调整到 8-32。

**Q: 训练时 loss 不下降？**
A: 尝试降低学习率，或先用 Concatenate 验证数据和模型是否正常。

**Q: 不同策略可以组合吗？**
A: 可以，但需要自定义融合模块。建议先单独测试每种策略。

## 下一步

- [训练工作流](08_training_workflow.md) - 学习完整训练流程
- [模型评估](../guides/evaluation.md) - 评估和对比不同融合策略
- [超参数调优](../guides/hyperparameter_tuning.md) - 优化融合策略参数
