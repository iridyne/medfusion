# MedFusion

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

高度模块化的医学多模态深度学习研究框架，支持 29 种视觉骨干网络和 5 种融合策略。

## ✨ 核心特性

- 🔧 **高度模块化**: 骨干网络、融合策略、聚合器完全解耦
- 📊 **多视图支持**: 多角度 CT、时间序列、多模态、多切片等 5 种场景
- 🎯 **配置驱动**: 通过 YAML 配置文件快速切换组件，无需修改代码
- 🌐 **Web UI**: 实时训练监控、模型管理、工作流编辑器
- ⚡ **Rust 加速**: 性能关键模块使用 Rust 实现

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/medfusion.git
cd medfusion

# 安装依赖（推荐使用 uv）
uv sync

# 安装开发依赖
uv sync --extra dev

# 安装 Web UI 依赖
uv sync --extra web

# 或使用 pip
pip install -e ".[dev,web]"
```

### 基础使用

```bash
# 1) 训练前先做配置与数据体检
uv run medfusion validate-config --config configs/starter/quickstart.yaml

# 2) 最快跑通训练主链
uv run medfusion train --config configs/starter/quickstart.yaml

# 3) 训练后生成结果页需要的 validation / 图表 / 报告 artifact
uv run medfusion build-results \
  --config configs/starter/quickstart.yaml \
  --checkpoint outputs/quickstart/checkpoints/best.pth \
  --split train

# 4) 评估模型
uv run medfusion evaluate \
  --config configs/starter/quickstart.yaml \
  --checkpoint outputs/quickstart/checkpoints/best.pth

# 数据预处理
uv run medfusion preprocess --input-dir data/raw --output-dir data/processed
```

### CLI 与 Config 路径

当前先分清三条使用路径：

1. `medfusion train` 直接可用的配置：
   - `configs/starter/`
   - `configs/public_datasets/`
   - `configs/testing/`
2. `MultiModalModelBuilder` / `build_model_from_config()` 用的结构示例：
   - `configs/builder/`
3. 历史模板：
   - `configs/legacy/`

入口说明：

- [configs/README.md](configs/README.md)
- [CLI 与 Config 使用路径](docs/contents/getting-started/cli-config-workflow.md)

推荐主链：

1. `medfusion validate-config --config ...`
2. `medfusion train --config ...`
3. `medfusion build-results --config ... --checkpoint ...`
4. 再进入 `Web UI` 或 `medfusion evaluate`

### 启动 Web UI

```bash
# 推荐：统一入口，直接进入工作台首页
uv run medfusion start

# 高级用法：指定主机、端口、热重载
uv run medfusion start --host 0.0.0.0 --port 8080 --reload

# 兼容旧入口
uv run medfusion web

# 访问 http://localhost:8000
```

进入工作台后，当前推荐顺序是：

1. 打开“训练配置向导”生成真实训练 YAML
2. 执行 `medfusion validate-config` / `medfusion train`
3. 跑完后再用 `medfusion build-results` 或工作台导入结果

## 🧪 公开数据集快速验证

如果你还没有私有医学数据，建议先用公开数据集验证框架和 Web UI 闭环。

推荐顺序：

1. `最快上手`：从 [MedMNIST](https://medmnist.com/v2) 开始，下载成本低，适合快速验证训练、评估和结果展示链路。
2. `表格任务`：用 [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) 先验证结构化输入和基础分类流程。
3. `真实医学影像`：再切到 [ISIC Challenge / HAM10000](https://challenge.isic-archive.com/data/) 或 [NIH ChestXray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) 做更接近公开论文复现的实验。

第一批推荐数据集：

| Dataset | 模态 | 典型任务 | 为什么适合第一轮验证 |
| --- | --- | --- | --- |
| MedMNIST（如 PathMNIST / ChestMNIST / BreastMNIST） | 2D / 3D 医学图像 | 分类、多标签分类 | 下载最省事，最适合先验证训练与结果页 |
| UCI Heart Disease | 表格 | 二分类 | 适合快速验证非图像主链、指标输出和报告 |
| ISIC 2018 / 2019（含 HAM10000 来源） | 皮肤镜图像 | 分类、分割 | 公共医学图像里很常见，适合做对外演示 |
| NIH ChestXray14 | X-ray | 多标签分类 | 经典胸片基准，适合后续做更真实的公开验证 |
| ISIC MILK10k | 双图像 / 多视图 | 病灶分类 | 更接近多视图 / 多模态内容表达，适合后续传播 |

详细清单、下载入口和推荐验证路径见：

- [公开数据集快速验证清单](docs/contents/getting-started/public-datasets.md)

最快复制命令：

```bash
# 1) 最快验证图像训练主链：PathMNIST
uv pip install medmnist
uv run python scripts/prepare_public_dataset.py medmnist-pathmnist --overwrite
uv run medfusion train --config configs/public_datasets/pathmnist_quickstart.yaml

# 2) 最快验证表格指标主链：UCI Heart Disease
uv run python scripts/prepare_public_dataset.py uci-heart-disease --overwrite
uv run medfusion train --config configs/public_datasets/uci_heart_disease_quickstart.yaml
```

说明：

- `PathMNIST` 会写到 `data/public/medmnist/pathmnist-demo/`，配置文件可直接使用。
- `UCI Heart Disease` 会写到 `data/public/uci/heart-disease-demo/`，配置文件可直接使用。
- 当前 CLI 主链仍按统一多模态输入处理，所以 `PathMNIST` 走 dummy tabular fallback，`UCI Heart Disease` 会自动生成一张中性占位图。

### 代码示例

**使用模型构建器创建多模态模型：**

```python
from med_core.models import MultiModalModelBuilder

# 构建模型
builder = MultiModalModelBuilder(num_classes=2)
builder.add_modality("ct", backbone="swin3d_tiny", input_channels=1)
builder.add_modality("pathology", backbone="resnet50", pretrained=True)
builder.set_fusion("attention", hidden_dim=256)
builder.set_head("classification")
model = builder.build()

# 训练
outputs = model({"ct": ct_tensor, "pathology": path_tensor})
```

**从配置文件构建模型：**

```python
from med_core.models import build_model_from_config
import yaml

with open("configs/builder/smurf.yaml") as f:
    config = yaml.safe_load(f)

model = build_model_from_config(config)
```

## 📖 文档

- [完整文档](docs/README.md)
- [API 参考](docs/contents/api/med_core.md)
- [配置指南](docs/contents/tutorials/fundamentals/configs.md)
- [开发指南](docs/contents/guides/development/contributing.md)

## 🏗️ 架构

### 核心组件

MedFusion 采用高度模块化的设计，核心公式为：

```
Model = Backbones + Fusion + Head + (Optional) MIL Aggregators
```

**组件说明：**

- **Backbones** (`med_core/backbones/`): 特征提取器
  - 视觉：ResNet, EfficientNet, ViT, Swin Transformer (2D/3D), DenseNet 等 29+ 种
  - 表格：MLP 网络，支持批归一化和 Dropout

- **Fusion** (`med_core/fusion/`): 多模态融合策略
  - 8 种融合方式：Concatenate, Gated, Attention, Cross-Attention, Bilinear, Kronecker, Fused-Attention, Self-Attention

- **Heads** (`med_core/heads/`): 任务特定输出层
  - 分类：ClassificationHead
  - 生存分析：CoxSurvivalHead, DeepSurvivalHead, DiscreteTimeSurvivalHead

- **MIL Aggregators** (`med_core/aggregators/`): 多实例学习聚合器
  - Mean, Max, Attention-based, Gated Attention

### 目录结构

```
medfusion/
├── med_core/                    # 核心 Python 库
│   ├── models/                  # 模型架构（Builder, SMuRF）
│   ├── backbones/               # 骨干网络（Vision, Tabular）
│   ├── fusion/                  # 融合策略
│   ├── heads/                   # 任务头（分类、生存分析）
│   ├── aggregators/             # MIL 聚合器
│   ├── attention_supervision/   # 注意力监督
│   ├── datasets/                # 数据加载器
│   ├── trainers/                # 训练器（Multimodal, MultiView）
│   ├── evaluation/              # 评估指标和可视化
│   ├── preprocessing/           # 数据预处理
│   ├── utils/                   # 工具函数
│   ├── configs/                 # 配置验证
│   ├── web/                     # Web 服务（FastAPI）
│   └── cli/                     # 命令行接口
├── configs/                     # 配置模板
├── tests/                       # 测试套件
├── examples/                    # 使用示例
└── docs/                        # 文档
```

## 🧪 测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试文件
uv run pytest tests/test_models.py

# 运行特定测试函数
uv run pytest tests/test_models.py::test_model_builder

# 运行匹配模式的测试
uv run pytest -k "fusion"

# 生成覆盖率报告
uv run pytest --cov=med_core --cov-report=html

# 查看详细输出
uv run pytest -v
```

## 🔧 开发

### 代码质量检查

```bash
# 代码检查
ruff check med_core/

# 自动修复问题
ruff check med_core/ --fix

# 代码格式化
ruff format med_core/

# 类型检查
mypy med_core/
```

### 项目要求

- Python 3.11+
- PyTorch 2.0+
- 使用现代类型注解（PEP 585/604）
- 所有函数必须有完整的类型注解
- 遵循 88 字符行长度限制

详细开发指南请参考 [CLAUDE.md](CLAUDE.md)。

## ⚡ 性能优化

### 优化优先级

遇到性能问题时，按以下顺序优化：

1. **算法层面**：混合精度训练、梯度累积、模型剪枝/量化
2. **工程层面**：数据缓存、预计算特征、优化 DataLoader
3. **基础设施**：更好的 GPU、分布式训练、NVMe SSD
4. **部署优化**：TorchScript、ONNX、TensorRT
5. **自定义算核**：Triton CUDA kernel、C++ 扩展

### 常见瓶颈解决方案

- **数据加载慢**：增加 `num_workers`、使用数据缓存、更快的存储
- **GPU 利用率低**：增大 batch size、优化 DataLoader、检查 CPU 预处理
- **显存不足**：梯度累积、混合精度、减小 batch size
- **训练时间长**：分布式训练、更好的 GPU、优化模型架构

**注意**：不建议过早迁移到 Rust。PyTorch 核心已经是 C++/CUDA 优化的，大部分性能瓶颈在 I/O 和 GPU 利用率，而非 Python 开销。详见 [CLAUDE.md](CLAUDE.md) 的性能优化章节。

## 🤝 贡献

欢迎贡献！请查看 [贡献指南](CONTRIBUTING.md)。

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 📮 联系方式

- 问题反馈: [GitHub Issues](https://github.com/yourusername/medfusion/issues)
- 邮件: your.email@example.com

## 🙏 致谢

感谢所有贡献者和开源社区的支持。
