# MedFusion 代码库分析报告

**分析日期**: 2026-02-21
**总代码行数**: ~31,396 行 Python 代码
**前端代码**: 24,901 个文件
**Web 目录大小**: 331MB

---

## 📊 模块代码行数统计

| 模块 | 代码行数 | 占比 | 分类 |
|------|---------|------|------|
| **backbones** | 4,932 | 15.7% | 🟡 部分核心 |
| **web** | 4,714 | 15.0% | 🔴 可删除 |
| **utils** | 3,315 | 10.6% | 🟢 核心 |
| **datasets** | 3,152 | 10.0% | 🟢 核心 |
| **attention_supervision** | 2,678 | 8.5% | 🔴 可删除 |
| **fusion** | 2,476 | 7.9% | 🟢 核心 |
| **shared** | 1,702 | 5.4% | 🟢 核心 |
| **configs** | 1,258 | 4.0% | 🟢 核心 |
| **trainers** | 1,207 | 3.8% | 🟢 核心 |
| **models** | 1,096 | 3.5% | 🟢 核心 |
| **heads** | 988 | 3.1% | 🟡 部分核心 |
| **evaluation** | 877 | 2.8% | 🟢 核心 |
| **preprocessing** | 536 | 1.7% | 🟢 核心 |
| **visualization** | 483 | 1.5% | 🟡 部分核心 |
| **aggregators** | 480 | 1.5% | 🔴 可删除 |
| **extractors** | 462 | 1.5% | 🔴 可删除 |
| **cli** | 453 | 1.4% | 🟢 核心 |
| **其他** | 83 | 0.3% | - |

---

## 🎯 功能分类

### 🟢 核心功能（必须保留）- 约 13,000 行

#### 1. 数据处理 (3,152 行)
- `datasets/medical.py` (538 行) - 医学数据集加载
- `datasets/transforms.py` - 数据增强
- `datasets/base.py` - 基础数据集类
- **保留理由**: 任何项目都需要数据加载

#### 2. 模型架构 (2,476 行)
- `fusion/strategies.py` (509 行) - 融合策略
- `fusion/base.py` - 基础融合模块
- **保留理由**: 多模态融合是核心价值

#### 3. 训练流程 (1,207 行)
- `trainers/multimodal.py` (566 行) - 多模态训练器
- `trainers/base.py` - 基础训练器
- **保留理由**: 标准化训练流程

#### 4. 评估和报告 (877 行)
- `evaluation/metrics.py` - 评估指标
- `evaluation/visualization.py` - 可视化
- **保留理由**: 自动生成报告是核心价值

#### 5. 配置系统 (1,258 行)
- `configs/` - YAML 配置管理
- **保留理由**: 配置驱动的设计

#### 6. 工具函数 (3,315 行)
- `utils/` - 各种工具函数
- **保留理由**: 支撑其他模块

#### 7. CLI 工具 (453 行)
- `cli/` - 命令行接口
- **保留理由**: 开发者主要使用方式

#### 8. 预处理 (536 行)
- `preprocessing/image.py` - 图像预处理
- **保留理由**: 医学图像需要特殊处理

---

### 🟡 部分核心（简化保留）- 约 6,400 行

#### 1. Backbone 模型 (4,932 行)
**当前**: 9 种架构，29 个变体
- ResNet (5 个): resnet18, 34, 50, 101, 152
- MobileNet (3 个): v2, v3_small, v3_large
- EfficientNet (8 个): b0-b7
- EfficientNetV2 (3 个): s, m, l
- ConvNeXt (4 个): tiny, small, base, large
- RegNet (7 个): y_400mf ~ y_32gf
- MaxViT (1 个): maxvit_t
- ViT (4 个): b_16, b_32, l_16, l_32
- Swin (3 个): t, s, b

**建议**: 保留 3-5 种常用
- ResNet50 (通用)
- EfficientNet-B0 (轻量)
- ViT-B16 (Transformer)
- 可选: ConvNeXt-Tiny, Swin-T

**可删除**:
- `backbones/swin_components.py` (878 行)
- `backbones/swin_2d.py` (534 行)
- `backbones/swin_3d.py`
- RegNet 系列
- MaxViT
- 大部分 EfficientNet 变体

**节省**: ~2,500 行

#### 2. 分类头 (988 行)
**当前**: `heads/survival.py` (504 行) - 生存分析
**建议**: 保留基础分类头，删除生存分析
**节省**: ~500 行

#### 3. 可视化 (483 行)
**当前**: `visualization/attention_viz.py` (483 行)
**建议**: 保留基础可视化，删除注意力可视化
**节省**: ~300 行

---

### 🔴 可删除功能（非核心）- 约 12,000 行

#### 1. Web UI (4,714 行 Python + 331MB 前端)
**包含**:
- FastAPI 后端 (4,714 行)
- React 前端 (24,901 个文件)
- 工作流编辑器
- 实验对比界面
- 报告生成 Web 界面

**删除理由**:
- 开发者用 CLI/API 更快
- 小客户不会直接用
- 维护成本高
- 8 周开发，未经实战验证

**节省**: 4,714 行 + 331MB

#### 2. 注意力监督 (2,678 行)
**包含**:
- `attention_supervision/mask_supervision.py` (511 行)
- `attention_supervision/advanced_supervision.py` (493 行)
- `attention_supervision/advanced_attention.py` (548 行)

**删除理由**:
- 高级功能，小项目用不到
- 需要额外的标注数据（mask）
- 增加复杂度

**节省**: 2,678 行

#### 3. 多视图聚合器 (480 行)
**包含**:
- `aggregators/` - 5 种聚合策略
- `backbones/multiview_vision.py`
- `datasets/multiview_*.py`

**删除理由**:
- 特殊场景才需要（多角度 CT）
- 小项目通常是单视图
- 等有需求再加

**节省**: ~1,500 行（包含相关代码）

#### 4. 特征提取器 (462 行)
**包含**: `extractors/` - 特征提取模块

**删除理由**:
- 不清楚用途
- 可能是过度设计

**节省**: 462 行

#### 5. 其他高级功能
- `utils/distributed.py` (475 行) - 分布式训练
- `utils/benchmark.py` (540 行) - 性能基准测试
- `datasets/cache.py` (487 行) - 复杂缓存系统
- `preprocessing/quality.py` (486 行) - 数据质量检查

**删除理由**: 过早优化，小项目用不到
**节省**: ~2,000 行

---

## 📉 简化后的代码规模估算

| 类别 | 当前行数 | 简化后 | 减少 |
|------|---------|--------|------|
| 核心功能 | 13,000 | 13,000 | 0 |
| 部分核心（简化） | 6,400 | 3,000 | -3,400 |
| 可删除功能 | 12,000 | 0 | -12,000 |
| **总计** | **31,400** | **16,000** | **-15,400 (49%)** |

**进一步优化**: 目标 < 10,000 行
- 简化 utils（3,315 → 1,500）
- 简化 datasets（3,152 → 2,000）
- 简化 fusion（2,476 → 1,500）

**最终目标**: ~8,000-10,000 行核心代码

---

## 🎯 MedFusion Lite 功能清单

### 保留功能

#### 1. 数据处理
- CSV + 图片路径加载
- 基础数据增强
- 自动 train/val/test 划分

#### 2. 模型架构
- 3-5 种 backbone（ResNet50, EfficientNet-B0, ViT-B16）
- 简单 fusion（拼接 + MLP）
- 基础分类头

#### 3. 训练流程
- 标准训练循环
- 混合精度训练（AMP）
- 早停和检查点

#### 4. 评估和报告
- 基础指标（Acc, AUC, F1）
- 混淆矩阵
- ROC/PR 曲线
- 自动生成 PDF/Word 报告 ⭐ 核心价值

#### 5. 配置系统
- YAML 配置
- 命令行参数覆盖

#### 6. CLI 工具
```bash
medfusion train --config config.yaml
medfusion evaluate --checkpoint model.pt
medfusion report --results results.json
```

### 删除功能

- ❌ Web UI（全部）
- ❌ 注意力监督
- ❌ 多视图支持
- ❌ 20+ 种 backbone
- ❌ 分布式训练
- ❌ 复杂缓存系统
- ❌ 性能基准测试
- ❌ 生存分析
- ❌ 特征提取器

---

## 💡 建议

### 立即行动

1. **创建 `medfusion-lite` 分支**
   ```bash
   git checkout -b medfusion-lite
   ```

2. **删除非核心模块**
   ```bash
   rm -rf med_core/web/
   rm -rf med_core/attention_supervision/
   rm -rf med_core/aggregators/
   rm -rf med_core/extractors/
   rm -rf web/
   ```

3. **简化 backbone**
   - 只保留 ResNet50, EfficientNet-B0, ViT-B16
   - 删除其他变体

4. **测试核心功能**
   - 运行简化后的测试
   - 确保核心功能可用

### 验证标准

使用 MedFusion Lite 完成一个模拟项目：
- 数据准备: < 10 分钟
- 配置模型: < 5 分钟
- 训练启动: 1 行命令
- 报告生成: 自动

**目标**: 从数据到报告 < 1 小时（不含训练时间）

---

## 🚨 风险提示

### 沉没成本谬误
- 已经写了 3.1 万行代码
- 花了 8 周开发 Web UI
- 但如果不好用，就应该删除

### 决策原则
**不要问**: "我花了多少时间？"
**要问**: "这个功能对下一个项目有用吗？"

如果答案是"不确定"或"可能有用"，就删除。
只保留"肯定有用"的功能。

---

**结论**: MedFusion 当前有 ~50% 的代码是 over-engineering，应该大幅简化。
