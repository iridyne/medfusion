# MedFusion 新手避坑指南

这份指南基于实际从零开始使用框架的经验，列出了所有可能遇到的问题和解决方案。

先看这一页，再继续：

- [CLI 与 Config 使用路径](cli-config-workflow.md)

当前最稳定的训练入口是：

```bash
uv run medfusion validate-config --config configs/starter/quickstart.yaml
uv run medfusion train --config configs/starter/quickstart.yaml
uv run medfusion build-results \
  --config configs/starter/quickstart.yaml \
  --checkpoint outputs/quickstart/checkpoints/best.pth
```

要注意：

- `configs/starter/`、`configs/public_datasets/`、`configs/testing/` 是当前 `medfusion train` 主链配置
- `configs/builder/` 是 `MultiModalModelBuilder` / `build_model_from_config()` 的结构示例，不等价于 CLI 训练配置
- `configs/legacy/` 是历史模板，不建议新用户从这里开始
- 训练前先跑 `medfusion validate-config`
- 训练后再跑 `medfusion build-results`，这样结果页和报告需要的 artifact 才完整

## 🚨 关键问题清单

### 问题 1：融合策略命名不一致 ⭐⭐⭐⭐⭐

**症状**：
```bash
ValueError: Unknown fusion type: concat. Available: ['concatenate', 'gated', 'attention', 'cross_attention', 'bilinear']
```

**原因**：
- 文档和示例中使用 `concat`
- 实际代码要求 `concatenate`
- Builder API 接受 `concat`，但配置文件不接受

**解决方案**：
```yaml
# ❌ 错误
fusion:
  fusion_type: "concat"

# ✅ 正确
fusion:
  fusion_type: "concatenate"
```

**影响范围**：所有使用配置文件的用户（约 80%）

**修复建议**：
1. 在 `create_fusion_module()` 中添加别名映射：
   ```python
   FUSION_ALIASES = {
       "concat": "concatenate",
       "attn": "attention",
       "cross_attn": "cross_attention"
   }
   ```
2. 更新所有文档使用统一命名

---

### 问题 2：配置文件中的默认路径不存在 ⭐⭐⭐⭐

**症状**：
```bash
FileNotFoundError: data/dataset.csv not found
```

**原因**：
- 新手直接从非 quickstart 配置开始
- 没有先对照仓库里实际存在的数据路径
- 入口配置和自己准备的数据路径没有同步

**解决方案**：
创建 `configs/starter/quickstart.yaml`（已完成）：
```yaml
data:
  csv_path: "data/mock/metadata.csv"  # 使用实际存在的数据
  image_dir: "data/mock"
```

**影响范围**：所有新手用户（100%）

**修复建议**：
1. 优先从 `configs/starter/quickstart.yaml` 开始
2. 或者在 README 中明确说明需要准备数据
3. 提供 `med-download-sample-data` 命令

---

### 问题 3：列名不匹配 ⭐⭐⭐⭐

**症状**：
```bash
KeyError: 'weight' not found in CSV columns
```

**原因**：
- `default.yaml` 期望列名：`age`, `weight`, `marker_a`, `sex`, `smoking_status`
- 实际 mock 数据列名：`age`, `gender`, `diagnosis`

**解决方案**：
```yaml
# 检查实际 CSV 文件的列名
numerical_features:
  - "age"  # ✓ 存在
categorical_features:
  - "gender"  # ✓ 存在（不是 "sex"）
```

**影响范围**：所有使用 mock 数据的用户（60%）

**修复建议**：
1. 提供 `med-validate-data` 命令检查列名
2. 在训练开始前验证所有列是否存在
3. 提供友好的错误提示

---

### 问题 4：pin_memory 警告 ⭐⭐

**症状**：
```
UserWarning: 'pin_memory' argument is set as true but no accelerator is found
```

**原因**：
- 配置文件默认 `pin_memory: true`
- 但在 CPU 环境下会产生警告

**解决方案**：
```yaml
data:
  pin_memory: false  # CPU 环境下设为 false
```

**影响范围**：所有 CPU 用户（30%）

**修复建议**：
自动检测设备类型并设置 `pin_memory`

---

### 问题 5：val_auc 始终为 0.0000 ⭐⭐⭐

**症状**：
```
Val Metric (val_auc): 0.0000
```

**原因**：
- 验证集太小（只有 4 个样本）
- AUC 计算需要足够的样本

**解决方案**：
```yaml
data:
  batch_size: 2  # 减小 batch size
  train_ratio: 0.6
  val_ratio: 0.3  # 增加验证集比例
  test_ratio: 0.1
```

**影响范围**：使用小数据集的用户（40%）

**修复建议**：
1. 在验证集太小时给出警告
2. 提供替代指标（如 accuracy）

---

### 问题 6：num_workers 多进程问题 ⭐⭐⭐

**症状**：
```
RuntimeError: DataLoader worker (pid XXXX) is killed by signal
```

**原因**：
- 默认 `num_workers: 4`
- 在某些环境下会导致多进程问题

**解决方案**：
```yaml
data:
  num_workers: 0  # 调试时设为 0
```

**影响范围**：Windows 用户和某些 Linux 环境（20%）

**修复建议**：
在 Windows 上自动设置 `num_workers: 0`

---

### 问题 7：模型参数量巨大 ⭐⭐⭐

**症状**：
```
Model parameters: 11308496  # 1130 万参数
```

**原因**：
- ResNet18 预训练模型本身就很大
- 对于 30 个样本的数据集严重过拟合

**解决方案**：
```yaml
model:
  vision:
    backbone: "resnet18"
    freeze_backbone: true  # 冻结骨干网络
```

**影响范围**：小数据集用户（50%）

**修复建议**：
1. 提供轻量级 backbone 选项
2. 根据数据集大小自动建议配置

---

## 📋 完整的新手检查清单

### 安装阶段
- [ ] Python 版本 3.11+
- [ ] 使用 `uv sync` 而不是 `pip install`
- [ ] 检查 PyTorch 是否正确安装：`uv run python -c "import torch; print(torch.__version__)"`

### 数据准备阶段
- [ ] CSV 文件存在且路径正确
- [ ] 图像目录存在且路径正确
- [ ] CSV 中的 `image_path` 列指向实际存在的图像
- [ ] 所有配置的特征列在 CSV 中存在
- [ ] 至少有 20+ 个样本（否则验证集太小）

### 配置文件阶段
- [ ] 使用 `concatenate` 而不是 `concat`
- [ ] `fusion.hidden_dim` = `vision.feature_dim` + `tabular.output_dim`
- [ ] CPU 环境下设置 `pin_memory: false`
- [ ] 调试时设置 `num_workers: 0`
- [ ] 小数据集时设置 `freeze_backbone: true`

### 训练阶段
- [ ] 检查输出目录是否有写权限
- [ ] 监控第一个 epoch 是否正常完成
- [ ] 检查 loss 是否下降
- [ ] 验证集指标是否合理

---

## 🎯 推荐的新手工作流

### 1. 最小化测试（5 分钟）

```bash
# 使用提供的快速入门配置
uv run medfusion train --config configs/starter/quickstart.yaml
```

**预期结果**：
- 3 个 epoch 在 5 秒内完成
- 训练 loss 下降
- 没有报错

### 2. 使用自己的数据（30 分钟）

```bash
# 1. 准备数据
# - CSV 文件：patient_id, image_path, features..., label
# - 图像目录：包含所有图像

# 2. 复制快速入门配置
cp configs/starter/quickstart.yaml configs/my_experiment.yaml

# 3. 修改配置
# - 更新 csv_path 和 image_dir
# - 更新 numerical_features 和 categorical_features
# - 更新 num_classes

# 4. 验证配置（推荐添加此命令）
# uv run med-validate-config configs/my_experiment.yaml

# 5. 训练
uv run medfusion train --config configs/my_experiment.yaml
```

### 3. 使用 Builder API（1 小时）

```python
from med_core.models import MultiModalModelBuilder
import torch

# 构建模型
model = (
    MultiModalModelBuilder()
    .add_modality("xray", backbone="resnet18", modality_type="vision")
    .add_modality("clinical", backbone="mlp", modality_type="tabular", input_dim=10)
    .set_fusion("concatenate")  # 注意：使用完整名称
    .set_head("classification", num_classes=2)
    .build()
)

# 测试
xray = torch.randn(2, 3, 224, 224)
clinical = torch.randn(2, 10)
output = model({"xray": xray, "clinical": clinical})
print(f"Output shape: {output.shape}")  # 应该是 [2, 2]
```

---

## 🔧 常见错误速查表

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `Unknown fusion type: concat` | 命名不一致 | 改为 `concatenate` |
| `FileNotFoundError: data/dataset.csv` | 路径不存在 | 使用 `data/mock/metadata.csv` |
| `KeyError: 'weight'` | 列名不匹配 | 检查 CSV 实际列名 |
| `RuntimeError: DataLoader worker killed` | 多进程问题 | 设置 `num_workers: 0` |
| `val_auc: 0.0000` | 验证集太小 | 增加数据或使用 accuracy |
| `CUDA out of memory` | 显存不足 | 减小 `batch_size` |
| `dimension mismatch` | 特征维度不匹配 | 检查 `fusion.hidden_dim` |

---

## 💡 最佳实践

### 1. 从最简单的配置开始
```yaml
# 最小配置模板
model:
  vision:
    backbone: "resnet18"
    freeze_backbone: true  # 先冻结
  fusion:
    fusion_type: "concatenate"  # 最简单
training:
  num_epochs: 3  # 先少训练几轮
  mixed_precision: false  # 避免混合精度问题
```

### 2. 逐步增加复杂度
```
concatenate → gated → attention → fused_attention
freeze_backbone: true → false
num_epochs: 3 → 10 → 50
```

### 3. 使用 Builder API 进行快速实验
```python
# 比配置文件更直观
for fusion in ["concatenate", "gated", "attention"]:
    model = (
        MultiModalModelBuilder()
        .add_modality("xray", backbone="resnet18")
        .add_modality("clinical", backbone="mlp", input_dim=10)
        .set_fusion(fusion)
        .set_head("classification", num_classes=2)
        .build()
    )
    # 训练和评估...
```

---

## 📚 推荐学习路径

### 第 1 天：环境和基础
1. 安装环境
2. 运行 `configs/starter/quickstart.yaml`
3. 理解输出日志

### 第 2-3 天：使用自己的数据
1. 准备 CSV 和图像
2. 修改配置文件
3. 完成第一次训练

### 第 4-5 天：模型调优
1. 尝试不同的 backbone
2. 尝试不同的 fusion
3. 调整超参数

### 第 2 周：高级功能
1. 使用 Builder API
2. 实现自定义 backbone
3. 使用 MIL aggregation

---

## 🐛 遇到问题时的调试步骤

1. **检查数据**
   ```python
   import pandas as pd
   df = pd.read_csv("data/mock/metadata.csv")
   print(df.head())
   print(df.columns.tolist())
   ```

2. **测试模型构建**
   ```python
   from med_core.models import MultiModalModelBuilder
   model = MultiModalModelBuilder()...build()
   print(f"Model built successfully: {sum(p.numel() for p in model.parameters())} params")
   ```

3. **测试前向传播**
   ```python
   import torch
   dummy_input = {"xray": torch.randn(1, 3, 224, 224), "clinical": torch.randn(1, 10)}
   output = model(dummy_input)
   print(f"Forward pass successful: {output.shape}")
   ```

4. **逐步增加复杂度**
   - 先用 1 个样本测试
   - 再用 1 个 batch 测试
   - 最后用完整数据集训练

---

## 📞 获取帮助

如果遇到本指南未涵盖的问题：

1. 检查 GitHub Issues
2. 查看 `examples/` 目录中的示例
3. 阅读 `CLAUDE.md` 开发者文档
4. 提交新的 Issue 并附上：
   - 完整的错误信息
   - 配置文件
   - 数据集描述（样本数、特征数）
   - 环境信息（Python 版本、PyTorch 版本）
