# 配置模板使用指南

本目录包含常用场景的配置模板，帮助你快速启动新项目。

## 📋 可用模板

### 1. 病理图像分类 (`pathology_classification.yaml`)

**适用场景：**
- 病理切片分类（H&E 染色）
- 组织类型识别
- 肿瘤分级

**特点：**
- 使用 ResNet50 骨干网络
- 支持 CBAM 注意力机制
- 中等强度数据增强
- 推荐图像尺寸：224-512

**快速开始：**
```bash
# 1. 复制模板
cp configs/templates/pathology_classification.yaml configs/my_project.yaml

# 2. 修改数据路径
# 编辑 my_project.yaml，修改：
#   - data.csv_path
#   - data.image_dir
#   - model.num_classes

# 3. 开始训练
uv run med-train --config configs/my_project.yaml
```

---

### 2. 影像生存分析 (`radiology_survival.yaml`)

**适用场景：**
- 基于 CT/MRI 预测生存时间
- 肿瘤预后评估
- 治疗效果预测

**特点：**
- 支持 3D 影像处理
- Cox 比例风险模型
- C-index 评估指标
- Kaplan-Meier 曲线可视化

**数据格式要求：**
```csv
patient_id,scan_path,survival_time,event,age,gender,stage
P001,scans/p001.nii.gz,365,1,65,M,III
P002,scans/p002.nii.gz,730,0,58,F,II
```

**注意事项：**
- `survival_time`: 生存时间（天数）
- `event`: 1=死亡，0=删失
- 3D 数据内存占用大，建议 batch_size=4

---

### 3. 多模态融合 (`multimodal_fusion.yaml`)

**适用场景：**
- 影像 + 病理 + 临床数据融合
- 多组学研究
- 精准医疗

**特点：**
- 支持多种模态（CT/MRI + WSI + 临床 + 基因组）
- Cross-attention 融合策略
- 渐进式训练
- 模态缺失处理

**数据格式示例：**
```csv
patient_id,ct_path,wsi_path,age,gender,diagnosis
P001,radiology/p001.nii.gz,pathology/p001.svs,65,M,malignant
P002,radiology/p002.nii.gz,pathology/p002.svs,58,F,benign
P003,radiology/p003.nii.gz,,62,M,malignant  # 缺少病理数据
```

**高级功能：**
- 模态 dropout（提升鲁棒性）
- 差异化学习率
- 辅助损失
- 注意力权重可视化

---

## 🚀 使用流程

### 步骤 1：选择模板

根据你的任务类型选择合适的模板：

| 任务类型 | 推荐模板 | 数据类型 |
|---------|---------|---------|
| 病理图像分类 | `pathology_classification.yaml` | 2D 图像 + 临床数据 |
| 生存时间预测 | `radiology_survival.yaml` | 3D 影像 + 临床数据 |
| 多模态诊断 | `multimodal_fusion.yaml` | 多种模态 |

### 步骤 2：准备数据

确保你的数据格式符合模板要求：

```
data/
├── metadata.csv          # 元数据文件
└── images/              # 图像目录
    ├── patient_001.png
    ├── patient_002.png
    └── ...
```

### 步骤 3：修改配置

复制模板并修改关键参数：

```yaml
# 必须修改的参数
data:
  csv_path: "data/your_data.csv"        # 你的数据路径
  image_dir: "data/your_images"         # 你的图像目录

model:
  num_classes: 3                        # 你的类别数

# 可选修改的参数
training:
  num_epochs: 50                        # 训练轮数
  batch_size: 32                        # 批次大小
```

### 步骤 4：验证配置

在训练前验证配置文件：

```bash
# 检查配置文件语法
uv run python -c "import yaml; yaml.safe_load(open('configs/my_project.yaml'))"

# 检查数据路径是否存在
ls data/your_data.csv
ls data/your_images/
```

### 步骤 5：开始训练

```bash
uv run med-train --config configs/my_project.yaml
```

---

## 💡 常见问题

### Q1: 如何调整 batch_size？

根据 GPU 内存调整：

| GPU 内存 | 2D 图像 (224x224) | 3D 图像 (128x128x128) |
|---------|------------------|---------------------|
| 8GB     | 16-32            | 2-4                 |
| 16GB    | 32-64            | 4-8                 |
| 24GB    | 64-128           | 8-16                |

如果遇到 OOM（内存不足）错误，减小 batch_size。

### Q2: 如何选择融合策略？

| 融合策略 | 适用场景 | 复杂度 | 性能 |
|---------|---------|-------|------|
| `concatenate` | 简单任务，快速原型 | 低 | 中 |
| `gated` | 需要动态权重 | 中 | 高 |
| `attention` | 单模态内部融合 | 中 | 高 |
| `cross_attention` | 多模态交互 | 高 | 最高 |

**推荐：**
- 初学者：`concatenate`
- 双模态：`gated`
- 多模态：`cross_attention`

### Q3: 如何处理类别不平衡？

在配置文件中添加：

```yaml
training:
  loss:
    type: "focal_loss"
    alpha: 0.25
    gamma: 2.0

  # 或使用类别权重
  class_weights: [1.0, 2.0, 3.0]  # 根据类别比例调整
```

### Q4: 如何加速训练？

```yaml
training:
  mixed_precision: true          # 启用混合精度
  gradient_accumulation: 2       # 梯度累积

data:
  num_workers: 4                 # 增加数据加载线程
  pin_memory: true               # 启用内存锁定
```

### Q5: 如何只使用图像，不使用临床数据？

注释掉 tabular 和 fusion 部分：

```yaml
model:
  # tabular:  # 注释掉
  #   ...

  # fusion:   # 注释掉
  #   ...
```

---

## 🎯 最佳实践

### 1. 数据准备

✅ **推荐做法：**
- 使用标准化的 CSV 格式
- 图像路径使用相对路径
- 包含必要的元数据列

❌ **避免：**
- 使用绝对路径
- 缺少必要的列
- 数据格式不一致

### 2. 模型选择

✅ **推荐做法：**
- 从简单模型开始（ResNet18）
- 逐步增加复杂度
- 使用预训练权重

❌ **避免：**
- 一开始就用最复杂的模型
- 跳过预训练
- 过度拟合小数据集

### 3. 训练策略

✅ **推荐做法：**
- 使用早停（early stopping）
- 监控验证集指标
- 保存多个检查点

❌ **避免：**
- 训练到固定 epoch 数
- 只看训练集损失
- 只保存最后一个模型

### 4. 超参数调优

**优先级顺序：**
1. 学习率（最重要）
2. Batch size
3. 数据增强强度
4. Dropout 率
5. 模型架构

**推荐工具：**
- Weights & Biases（wandb）
- TensorBoard
- Optuna（自动调参）

---

## 📚 进阶主题

### 自定义模板

如果现有模板不满足需求，可以基于模板创建自定义配置：

```bash
# 1. 复制最接近的模板
cp configs/templates/pathology_classification.yaml configs/custom.yaml

# 2. 修改关键部分
# 例如：添加新的数据增强、修改网络结构等

# 3. 测试配置
uv run med-train --config configs/custom.yaml --dry-run
```

### 配置继承

可以使用 YAML 锚点实现配置复用：

```yaml
# 定义基础配置
base_training: &base_training
  num_epochs: 50
  mixed_precision: true

# 继承并覆盖
training:
  <<: *base_training
  num_epochs: 100  # 覆盖
```

### 环境变量

支持在配置中使用环境变量：

```yaml
data:
  csv_path: "${DATA_ROOT}/metadata.csv"
  image_dir: "${DATA_ROOT}/images"
```

```bash
export DATA_ROOT=/path/to/data
uv run med-train --config configs/my_project.yaml
```

---

## 🔗 相关文档

- [快速入门指南](../../docs/QUICKSTART_GUIDE.md)
- [配置系统文档](../../docs/configuration.md)
- [API 参考](../../docs/api/)
- [常见问题](../../docs/FAQ_CLIENTS.md)

---

## 📝 反馈

如果你发现模板有问题或有改进建议，请：

1. 提交 Issue
2. 发送 Pull Request
3. 联系维护者

**最后更新：** 2025-02-27
