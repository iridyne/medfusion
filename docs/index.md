---
layout: home

hero:
  name: MedFusion
  text: 医学多模态深度学习框架
  tagline: 高度模块化 · 29+ 骨干网络 · 5+ 融合策略 · 配置驱动
  image:
    src: /logo.svg
    alt: MedFusion
  actions:
    - theme: brand
      text: 📚 开始学习教程
      link: /contents/tutorials/README
    - theme: alt
      text: 快速开始
      link: /contents/user-guides/QUICKSTART_GUIDE
    - theme: alt
      text: API 文档
      link: /contents/api/med_core

features:
  - icon: 🔧
    title: 高度模块化
    details: 骨干网络、融合策略、聚合器完全解耦，可独立替换。14 种 backbone × 5 种融合 × 5 种聚合 = 350+ 种配置组合。

  - icon: 📊
    title: 多视图支持
    details: 支持多角度 CT、时间序列、多模态、多切片等 5 种复杂医学数据场景，内置 MIL 聚合器。

  - icon: 🎯
    title: 配置驱动
    details: 通过 YAML ��置文件快速切换组件，无需修改代码。所有模型架构可通过配置定义。

  - icon: 🌐
    title: Web UI
    details: 实时训练监控、模型管理、工作流编辑器。FastAPI 后端 + React 前端，WebSocket 实时通信。

  - icon: ⚡
    title: 性能优化
    details: 混合精度训练、梯度累积、数据缓存、分布式训练。支持 TorchScript、ONNX、TensorRT 部署。

  - icon: 🧪
    title: 生产就绪
    details: 699+ 测试用例、Docker 支持、CI/CD 流程、完整的错误代码系统（E001-E028）。
---

## 核心架构

```
Model = Backbones + Fusion + Head + (Optional) MIL Aggregators
```

### 支持的组件

**视觉骨干网络 (29+)**
- ResNet, EfficientNet, ViT, Swin Transformer (2D/3D)
- DenseNet, ConvNeXt, MaxViT, RegNet, MobileNet

**融合策略 (5+)**
- Concatenate, Gated, Attention, Cross-Attention, Bilinear

**任务头**
- 分类：ClassificationHead, MultiLabel, Ordinal
- 生存分析：Cox, DeepSurvival, DiscreteTime

**MIL 聚合器**
- Mean, Max, Attention-based, Gated Attention

## 快速示例

```python
from med_core.models import MultiModalModelBuilder

# 构建多模态模型
builder = MultiModalModelBuilder(num_classes=2)
builder.add_modality("ct", backbone="swin3d_tiny", input_channels=1)
builder.add_modality("pathology", backbone="resnet50", pretrained=True)
builder.set_fusion("attention", hidden_dim=256)
builder.set_head("classification")
model = builder.build()

# 训练
outputs = model({"ct": ct_tensor, "pathology": path_tensor})
```

## 为什么选择 MedFusion？

::: tip 研究友好
配置驱动的设计让你可以快速尝试不同的模型架构组合，无需修改代码。
:::

::: tip 工业级质量
完整的测试覆盖、错误处理、文档和部署支持，可直接用于生产环境。
:::

::: tip 社区驱动
开源、文档完善、持续维护。欢迎贡献代码和反馈。
:::

## 开始使用

<div class="vp-doc">

### 安装

```bash
# 克隆仓库
git clone https://github.com/iridite/medfusion.git
cd medfusion

# 安装依赖（推荐使用 uv）
uv sync

# 或使用 pip
pip install -e ".[dev,web]"
```

### 训练模型

```bash
# 使用默认配置训练
uv run med-train --config configs/default.yaml

# 评估模型
uv run med-evaluate --checkpoint outputs/best_model.pth

# 启动 Web UI
./start-webui.sh
```

</div>

## 文档导航

- **[教程](/contents/tutorials/README)** - 从入门到精通的完整学习路径
- **[快速入门](/contents/user-guides/QUICKSTART_GUIDE)** - 新手必读
- **[API 文档](/contents/api/med_core)** - 完整的 API 参考
- **[用户指南](/contents/guides/quick_reference)** - 详细的功能指南
- **[架构设计](/contents/architecture/WEB_UI_ARCHITECTURE)** - 系统架构文档

## 社区

- [GitHub Issues](https://github.com/iridite/medfusion/issues) - 问题反馈
- [GitHub Discussions](https://github.com/iridite/medfusion/discussions) - 讨论交流
