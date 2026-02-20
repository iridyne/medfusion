# 梯度检查点架构设计文档

## 📋 概述

本文档描述 MedFusion 框架中梯度检查点功能的架构设计，旨在为未来扩展提供清晰的指导。

---

## 🏗️ 架构设计

### 1. 核心组件

```
med_core/
├── utils/
│   └── gradient_checkpointing.py    # 通用工具函数
└── backbones/
    ├── base.py                       # 基类接口定义
    ├── vision.py                     # ResNet 实现
    ├── swin_2d.py                    # Swin 2D 实现
    ├── swin_3d.py                    # Swin 3D 实现
    ├── efficientnet.py               # 待实现
    ├── convnext.py                   # 待实现
    └── vit.py                        # 待实现
```

### 2. 设计原则

#### 2.1 分层设计

```
┌─────────────────────────────────────┐
│   BaseVisionBackbone (基类)         │
│   - enable_gradient_checkpointing() │
│   - disable_gradient_checkpointing()│
│   - is_gradient_checkpointing_enabled()│
└─────────────────────────────────────┘
              ▲
              │ 继承
              │
┌─────────────┴─────────────┐
│                            │
│  具体 Backbone 实现        │
│  - 重写 enable_gradient_   │
│    checkpointing()         │
│  - 实现模型特定的检查点逻辑│
└────────────────────────────┘
```

#### 2.2 接口统一

所有 backbone 必须实现以下接口：

```python
class BaseVisionBackbone(nn.Module):
    def enable_gradient_checkpointing(self, segments: int | None = None) -> None:
        """启用梯度检查点"""
        
    def disable_gradient_checkpointing(self) -> None:
        """禁用梯度检查点"""
        
    def is_gradient_checkpointing_enabled(self) -> bool:
        """检查是否启用"""
```

#### 2.3 工具函数复用

```python
# med_core/utils/gradient_checkpointing.py
def checkpoint_sequential(functions, segments, input, **kwargs):
    """对顺序模块应用检查点"""
    
def apply_gradient_checkpointing(model, segments):
    """自动应用检查点到模型"""
    
def estimate_memory_savings(model, input_shape, device):
    """估算内存节省"""
```

---

## 🔧 实现模式

### 模式 1: 顺序层模型（ResNet, EfficientNet, MobileNet）

**特点**: 模型由顺序的层组成

**实现策略**:
1. 捕获原始层列表
2. 将层分段
3. 对每段应用 `checkpoint_sequential`

**代码模板**:
```python
def enable_gradient_checkpointing(self, segments: int | None = None) -> None:
    super().enable_gradient_checkpointing(segments)
    
    if segments is None:
        segments = 4  # 默认段数
    
    from med_core.utils.gradient_checkpointing import checkpoint_sequential
    
    # 捕获原始层
    original_layers = list(self._backbone.children())
    
    def checkpointed_forward(x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self._gradient_checkpointing_enabled:
            # 正常前向传播
            for layer in original_layers:
                x = layer(x)
            return x
        
        # 使用检查点
        x = checkpoint_sequential(
            original_layers,
            segments=segments,
            input=x,
            use_reentrant=False,
        )
        return x
    
    # 替换 forward 方法
    self._backbone.forward = checkpointed_forward
```

**适用模型**:
- ✅ ResNet (已实现)
- ⏳ EfficientNet (待实现)
- ⏳ MobileNet (待实现)
- ⏳ DenseNet (待实现)

---

### 模式 2: Transformer 模型（Swin, ViT）

**特点**: 模型由多个 stage/block 组成，每个 stage 包含多个 transformer block

**实现策略**:
1. 捕获 patch embedding、position encoding、transformer stages、normalization
2. 对 transformer stages 应用检查点
3. 保持其他组件正常运行

**代码模板**:
```python
def enable_gradient_checkpointing(self, segments: int | None = None) -> None:
    super().enable_gradient_checkpointing(segments)
    
    if segments is None:
        segments = len(self._backbone.layers)  # 默认每个 stage 一个段
    
    from med_core.utils.gradient_checkpointing import checkpoint_sequential
    
    # 捕获原始组件
    patch_embed = self._backbone.patch_embed
    pos_drop = self._backbone.pos_drop
    layers = list(self._backbone.layers)
    norm = self._backbone.norm
    
    def checkpointed_forward(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if not self.training or not self._gradient_checkpointing_enabled:
            # 正常前向传播
            x = patch_embed(x)
            x = pos_drop(x)
            for layer in layers:
                x = layer(x)
            if normalize:
                x = norm(x)
            return x
        
        # Patch embedding (不使用检查点)
        x = patch_embed(x)
        x = pos_drop(x)
        
        # Transformer stages (使用检查点)
        x = checkpoint_sequential(
            layers,
            segments=min(segments, len(layers)),
            input=x,
            use_reentrant=False,
        )
        
        # Normalization (不使用检查点)
        if normalize:
            x = norm(x)
        
        return x
    
    # 替换 forward 方法
    self._backbone.forward = checkpointed_forward
```

**适用模型**:
- ✅ Swin Transformer 2D (已实现)
- ✅ Swin Transformer 3D (已实现)
- ⏳ ViT (待实现)
- ⏳ MaxViT (待实现)

---

### 模式 3: 混合架构（ConvNeXt）

**特点**: 结合了卷积和现代架构设计

**实现策略**:
1. 识别模型的主要 stage
2. 对每个 stage 应用检查点
3. 保持 stem 和 head 正常运行

**代码模板**:
```python
def enable_gradient_checkpointing(self, segments: int | None = None) -> None:
    super().enable_gradient_checkpointing(segments)
    
    if segments is None:
        segments = 4  # ConvNeXt 通常有 4 个 stage
    
    from med_core.utils.gradient_checkpointing import checkpoint_sequential
    
    # 捕获原始组件
    stem = self._backbone.stem
    stages = list(self._backbone.stages)
    norm = self._backbone.norm
    
    def checkpointed_forward(x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self._gradient_checkpointing_enabled:
            # 正常前向传播
            x = stem(x)
            for stage in stages:
                x = stage(x)
            x = norm(x)
            return x
        
        # Stem (不使用检查点)
        x = stem(x)
        
        # Stages (使用检查点)
        x = checkpoint_sequential(
            stages,
            segments=min(segments, len(stages)),
            input=x,
            use_reentrant=False,
        )
        
        # Normalization (不使用检查点)
        x = norm(x)
        
        return x
    
    # 替换 forward 方法
    self._backbone.forward = checkpointed_forward
```

**适用模型**:
- ⏳ ConvNeXt (待实现)
- ⏳ ConvNeXt V2 (待实现)

---

## 📝 实现清单

### 已完成 ✅
- [x] 核心工具模块 (`gradient_checkpointing.py`)
- [x] 基类接口 (`BaseVisionBackbone`)
- [x] ResNet 系列
- [x] Swin Transformer 2D
- [x] Swin Transformer 3D
- [x] 测试套件
- [x] 使用文档
- [x] 演示脚本

### 待实现 ⏳

#### 高优先级
- [ ] EfficientNet 系列 (模式 1)
- [ ] ConvNeXt 系列 (模式 3)
- [ ] ViT 系列 (模式 2)

#### 中优先级
- [ ] MobileNet 系列 (模式 1)
- [ ] MaxViT (模式 2)
- [ ] RegNet 系列 (模式 1)

#### 低优先级
- [ ] DenseNet 系列 (模式 1)
- [ ] 其他自定义 backbone

---

## 🎯 扩展指南

### 为新 Backbone 添加梯度检查点支持

#### 步骤 1: 分析模型结构

```python
# 打印模型结构
model = YourBackbone()
print(model)

# 查看子模块
for name, module in model.named_children():
    print(f"{name}: {type(module)}")
```

#### 步骤 2: 确定实现模式

- 顺序层模型 → 使用模式 1
- Transformer 模型 → 使用模式 2
- 混合架构 → 使用模式 3

#### 步骤 3: 实现 `enable_gradient_checkpointing`

```python
def enable_gradient_checkpointing(self, segments: int | None = None) -> None:
    """
    为 YourBackbone 启用梯度检查点。
    
    Args:
        segments: 检查点段数（None = 自动选择）
    """
    super().enable_gradient_checkpointing(segments)
    
    # 1. 设置默认段数
    if segments is None:
        segments = 4  # 根据模型结构调整
    
    # 2. 导入工具函数
    from med_core.utils.gradient_checkpointing import checkpoint_sequential
    
    # 3. 捕获原始组件
    # ... 根据模型结构捕获
    
    # 4. 定义检查点 forward
    def checkpointed_forward(x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self._gradient_checkpointing_enabled:
            # 正常前向传播
            pass
        else:
            # 使用检查点
            pass
    
    # 5. 替换 forward 方法
    self._backbone.forward = checkpointed_forward
```

#### 步骤 4: 添加测试

```python
# tests/test_gradient_checkpointing.py

def test_your_backbone_gradient_checkpointing():
    """测试 YourBackbone 的梯度检查点。"""
    backbone = YourBackbone(variant="base")
    
    # 启用检查点
    backbone.enable_gradient_checkpointing()
    assert backbone.is_gradient_checkpointing_enabled()
    
    # 测试前向传播
    backbone.train()
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    output = backbone(x)
    
    # 测试反向传播
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
```

#### 步骤 5: 更新文档

在 `docs/guides/gradient_checkpointing.md` 中添加：

```markdown
### YourBackbone

YourBackbone 的特点...

```python
from med_core.backbones.your_backbone import YourBackbone

backbone = YourBackbone(variant="base")
backbone.enable_gradient_checkpointing()
```

**内存节省**: ~XX%
```

---

## ⚠️ 常见陷阱

### 1. 递归错误

❌ **错误做法**:
```python
original_forward = self._backbone.forward
def new_forward(x):
    return original_forward(x)  # 会递归调用自己！
```

✅ **正确做法**:
```python
original_layers = list(self._backbone.children())
def new_forward(x):
    for layer in original_layers:
        x = layer(x)
    return x
```

### 2. 忘记检查训练模式

❌ **错误做法**:
```python
def checkpointed_forward(x):
    # 总是使用检查点，推理时也会变慢
    return checkpoint_sequential(layers, segments, x)
```

✅ **正确做法**:
```python
def checkpointed_forward(x):
    if not self.training or not self._gradient_checkpointing_enabled:
        # 推理时不使用检查点
        return normal_forward(x)
    return checkpoint_sequential(layers, segments, x)
```

### 3. 段数设置不当

❌ **错误做法**:
```python
segments = 100  # 太多段，开销大于收益
```

✅ **正确做法**:
```python
# 根据模型结构选择合理的段数
if segments is None:
    segments = len(self._backbone.stages)  # 通常 4-8 段
```

---

## 📊 性能指标

### 内存节省 vs 训练时间

| 段数 | 内存节省 | 训练时间增加 | 推荐场景 |
|------|----------|--------------|----------|
| 2    | ~20%     | ~10%         | 显存充足，追求速度 |
| 4    | ~35%     | ~20%         | 平衡（推荐） |
| 8    | ~45%     | ~30%         | 显存极度受限 |
| 16   | ~50%     | ~40%         | 不推荐（收益递减） |

### 不同模型的最佳段数

| 模型 | 推荐段数 | 原因 |
|------|----------|------|
| ResNet18/34 | 4 | 4 个主要 stage |
| ResNet50/101 | 4 | 4 个主要 stage |
| Swin-Tiny | 4 | 4 个 transformer stage |
| Swin-Base | 4-8 | 更深的网络可以用更多段 |
| ViT-Base | 6-12 | 12 个 transformer block |
| EfficientNet | 4-7 | 根据变体调整 |

---

## 🔄 版本历史

### v0.2.0 (2024-02)
- ✅ 初始实现
- ✅ 支持 ResNet 系列
- ✅ 支持 Swin Transformer 2D/3D
- ✅ 完整的测试和文档

### v0.3.0 (计划中)
- ⏳ 支持 EfficientNet
- ⏳ 支持 ConvNeXt
- ⏳ 支持 ViT
- ⏳ 支持 MobileNet

---

## 📚 参考资料

- [PyTorch Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
- [使用指南](gradient_checkpointing.md)
- [API 文档](../api/utils.md#gradient-checkpointing)

---

**维护者**: MedFusion Team  
**最后更新**: 2024-02-20
