# MedFusion: Medical Multimodal Fusion Framework

> **Fusing Medical Intelligence** - A modular framework for medical multimodal fusion with 29 vision backbones and 5 fusion strategies.

MedFusion 是一个高度抽象、可插拔、模块化的医学多模态深度学习研究框架。它将模型架构、数据加载和训练逻辑解耦，使研究人员能够通过最少的代码修改在不同的骨干网络（如 ResNet vs. ViT）、融合策略和数据集之间切换。

## 🚀 核心特性

*   **解耦架构**：完全分离的骨干网络、数据加载器、训练器和配置。
*   **可插拔组件**：
    *   **视觉模块**：14 种骨干网络（ResNet、MobileNet、EfficientNet、EfficientNetV2、ConvNeXt、MaxViT、RegNet、ViT、Swin Transformer），共 29 个变体。
    *   **表格模块**：自适应 MLP、残差 MLP、特征分词器。
    *   **融合策略**：拼接、门控融合、注意力、交叉注意力、双线性。
    *   **注意力机制**：CBAM、SE Block、ECA Block。
*   **多视图支持** ⭐ NEW：
    *   **5 种多图片类型**：多角度 CT、时间序列、多模态、多切片、自定义视图。
    *   **5 种聚合策略**：Max、Mean、Attention、CrossView Attention、Learned Weight。
    *   **灵活处理**：支持缺失视图、权重共享、渐进式训练。
    *   **详细文档**：参见 `docs/MULTIVIEW_TYPES_GUIDE.md` 和 `docs/MULTIVIEW_TYPES_SUMMARY.md`。
*   **注意力监督** ⭐ NEW：
    *   **引导模型关注**：使模型学习关注临床相关区域。
    *   **3 种监督方法**：掩码引导、CAM 自监督、一致性约束。
    *   **可选功能**：零性能开销，按需启用。
    *   **详细文档**：参见 `docs/ATTENTION_MECHANISM_GUIDE.md`。
*   **医学 SOP 集成**：
    *   **预处理**：自动归一化、ROI 裁剪、伪影去除。
    *   **评估**：自动生成 ROC 曲线、PR 曲线、混淆矩阵和详细指标报告。
    *   **可解释性**：集成 Grad-CAM 和��意力可视化。
*   **配置驱动**：只需更改 YAML 配置文件即可从"皮肤病变"切换到"肺癌"项目。

## 🛠️ 安装

本项目使用 `uv` 进行依赖管理。

```bash
# 进入框架目录
cd medfusion

# 安装依赖和包
uv sync
```

## ⚡ 快速开始

### 1. 预处理数据

清理医学图像、归一化强度并去除伪影。

```bash
uv run medfusion-preprocess \
    --input-dir data/raw_images \
    --output-dir data/processed_images \
    --normalize percentile \
    --remove-artifacts
```

### 2. 训练模型

使用配置文件运行训练实验。

```bash
uv run medfusion-train --config configs/default.yaml
```

### 3. 评估

在特定数据集划分（val/test）上评估训练好的模型。

```bash
uv run medfusion-evaluate \
    --config configs/default.yaml \
    --checkpoint outputs/checkpoints/best.pth \
    --split test
```

## ⚙️ 配置

框架由 YAML 配置文件驱动。完整示例请参见 `configs/default.yaml`。

### 基础配置

```yaml
project_name: "medical-multimodal"
experiment_name: "resnet18_gated_fusion"

model:
  num_classes: 2
  vision:
    backbone: "resnet18"      # 选项: resnet50, efficientnet_b0, convnext_tiny, vit_b_16...
    pretrained: true
    attention_type: "cbam"    # 选项: cbam, se, eca, none
  tabular:
    hidden_dims: [64, 64]
  fusion:
    fusion_type: "gated"      # 选项: concatenate, attention, cross_attention, bilinear

training:
  num_epochs: 50
  mixed_precision: true
  use_progressive_training: true  # 阶段1: 冻结表格，训练视觉 -> 阶段2: 微调
```

### 多视图配置 ⭐ NEW

```yaml
# CT 多角度扫描
data:
  enable_multiview: true
  view_names: ["axial", "coronal", "sagittal"]
  view_path_columns:
    axial: "axial_path"
    coronal: "coronal_path"
    sagittal: "sagittal_path"
  missing_view_strategy: "zero"  # skip, zero, duplicate

model:
  vision:
    enable_multiview: true
    aggregator_type: "attention"  # max, mean, attention, cross_view_attention, learned_weight
    share_backbone_weights: true
```

或使用预设配置：

```python
from medfusion.configs import create_ct_multiview_config

config = create_ct_multiview_config(
    view_names=["axial", "coronal", "sagittal"],
    aggregator_type="attention",
    backbone="resnet50",
)
```

### 注意力监督配置 ⭐ NEW

```yaml
model:
  vision:
    enable_attention_supervision: true
    attention_type: "cbam"  # 必须使用 CBAM

training:
  use_attention_supervision: true
  attention_loss_weight: 0.1
  attention_supervision_method: "mask_guided"  # mask_guided, cam_based, consistency
```

**支持的场景：**
- **多角度 CT**：`["axial", "coronal", "sagittal"]`
- **时间序列**：`["baseline", "followup"]` 或 `["week_0", "week_4", "week_8"]`
- **多模态**：`["CT", "MRI", "PET"]` 或 `["T1", "T2", "FLAIR", "DWI"]`
- **多切片**：`["slice_1", "slice_2", "slice_3", ...]`
- **自定义**：任意视图名称

**详细文档：**
- 多视图完整指南：`docs/MULTIVIEW_TYPES_GUIDE.md`
- 多视图速查表：`docs/MULTIVIEW_TYPES_SUMMARY.md`
- 注意力机制指南：`docs/ATTENTION_MECHANISM_GUIDE.md`

## 📂 项目结构

```
medfusion/
├── medfusion/              # 核心框架包
│   ├── backbones/          # 视觉和表格骨干网络
│   ├── configs/            # 配置逻辑
│   ├── datasets/           # 医学数据集和变换
│   ├── evaluation/         # 指标、可视化、报告
│   ├── fusion/             # 融合策略
│   ├── preprocessing/      # 图像清理流程
│   └── trainers/           # 训练循环
├── configs/                # YAML 配置模板
├── examples/               # 演示脚本
└── tests/                  # 单元测试
```

## 🐍 Python API 使用

您也可以在 Python 脚本中直接使用框架组件：

### 基础用法

```python
from medfusion.backbones import create_vision_backbone, create_tabular_backbone
from medfusion.fusion import create_fusion_module, MultiModalFusionModel

# 1. 定义组件
vision = create_vision_backbone("resnet50", pretrained=True)
tabular = create_tabular_backbone(input_dim=10, output_dim=32)
fusion = create_fusion_module("cross_attention", vision_dim=2048, tabular_dim=32)

# 2. 构建模型
model = MultiModalFusionModel(
    vision_backbone=vision,
    tabular_backbone=tabular,
    fusion_module=fusion,
    num_classes=2
)

# 3. 准备使用标准 PyTorch 循环或 MedFusion Trainer 进行训练
```

### 多视图用法 ⭐ NEW

```python
from medfusion.configs import create_ct_multiview_config
from medfusion.datasets import MedicalMultiViewDataset
from medfusion.fusion import create_multiview_fusion_model
from medfusion.trainers import create_multiview_trainer

# 1. 配置
config = create_ct_multiview_config(
    view_names=["axial", "coronal", "sagittal"],
    aggregator_type="attention",
    backbone="resnet50",
)

# 2. 数据集
dataset = MedicalMultiViewDataset.from_csv_multiview(
    csv_path="data.csv",
    view_columns={
        "axial": "axial_path",
        "coronal": "coronal_path",
        "sagittal": "sagittal_path",
    },
    tabular_columns=["age", "gender"],
    label_column="label",
    view_config=config.data,
)

# 3. 模型
model = create_multiview_fusion_model(
    vision_backbone_name="resnet50",
    tabular_input_dim=2,
    fusion_type="gated",
    num_classes=2,
    aggregator_type="attention",
    view_names=config.data.view_names,
)

# 4. 训练
trainer = create_multiview_trainer(model, train_loader, val_loader, config)
trainer.train()
```

### 注意力监督用法 ⭐ NEW

```python
from medfusion.configs import ExperimentConfig

# 配置注意力监督
config = ExperimentConfig()
config.model.vision.enable_attention_supervision = True
config.model.vision.attention_type = "cbam"  # 必须使用 CBAM
config.training.use_attention_supervision = True
config.training.attention_loss_weight = 0.1
config.training.attention_supervision_method = "mask_guided"

# 数据集需要提供掩码
# CSV 格式: patient_id,image_path,mask_path,age,gender,label
dataset = MedicalMultimodalDataset.from_csv(
    csv_path="data_with_masks.csv",
    # ... 其他参数
)

# 训练时自动使用注意力监督
trainer = create_trainer(model, train_loader, val_loader, config)
trainer.train()
```

## 📊 评估报告

框架自动生成可发表的 Markdown 格式报告，包括：
*   系统信息和配置
*   指标表（准确率、AUC、F1、敏感性、特异性）及置信区间
*   混淆矩阵
*   ROC 和 PR 曲线
*   训练动态

生成位置：`outputs/results/report.md`

## 📚 组件库

MedFusion 提供丰富的预构建组件：

### 视觉 Backbone（14 种，29 个变体）
- **ResNet 系列**：resnet18, resnet34, resnet50, resnet101, resnet152
- **MobileNet 系列**：mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- **EfficientNet 系列**：efficientnet_b0 ~ b7
- **EfficientNetV2 系列**：efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
- **ConvNeXt 系列**：convnext_tiny, convnext_small, convnext_base, convnext_large
- **RegNet 系列**：regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf, regnet_y_8gf, regnet_y_16gf, regnet_y_32gf
- **MaxViT**：maxvit_t
- **ViT**：vit_b_16, vit_b_32, vit_l_16, vit_l_32
- **Swin Transformer**：swin_t, swin_s, swin_b

### 融合策略（5 种）
- **Concatenate**：简单拼接
- **Gated**：门控融合（可学习权重）
- **Attention**：自注意力融合
- **CrossAttention**：跨模态注意力
- **Bilinear**：双线性池化

### 视图聚合器（5 种）
- **MaxPool**：最大池化
- **MeanPool**：平均池化（支持 mask）
- **Attention**：可学习注意力权重
- **CrossViewAttention**：跨视图自注意力
- **LearnedWeight**：每个视图独立权重

### 注意力机制（3 种）
- **CBAM**：通道 + 空间注意力（支持注意力监督）
- **SE Block**：通道注意力
- **ECA Block**：高效通道注意力

**组合能力：** 14 种 backbone × 5 种融合 × 5 种聚合 = **350+ 种配置组合**

详细说明：`docs/component-library-overview.md`

## 📖 文档资源

- **快速开始指南**：`docs/quick-start-guide.md`
- **多视图完整指南**：`docs/MULTIVIEW_TYPES_GUIDE.md`
- **多视图速查表**：`docs/MULTIVIEW_TYPES_SUMMARY.md`
- **注意力机制指南**：`docs/ATTENTION_MECHANISM_GUIDE.md`
- **注意力监督指南**：`docs/ATTENTION_SUPERVISION_GUIDE.md`
- **组件库概览**：`docs/component-library-overview.md`
- **竞争力分析**：`docs/competitive-analysis.md`
- **代码质量报告**：`docs/code-quality-report.md`

## 🎯 使用场景

MedFusion 适用于以下医学影像任务：

- ✅ **疾病分类**：肺癌、皮肤病变、脑肿瘤等
- ✅ **多角度诊断**：CT 多平面重建（MPR）
- ✅ **治疗效果评估**：治疗前后对比
- ✅ **多模态融合**：影像 + 临床数据
- ✅ **时间序列分析**：疾病进展追踪
- ✅ **可解释性研究**：注意力可视化、Grad-CAM

## 🤝 贡献

欢迎贡献！请查看 `CONTRIBUTING.md` 了解详情。

## 📄 许可证

本项目采用 MIT 许可证。详见 `LICENSE` 文件。

## 📧 联系方式

如有问题或建议，请提交 Issue 或联系维护团队。

---

**版本：** 0.1.0  
**最后更新：** 2026-02-13  
**维护者：** Medical AI Research Team
