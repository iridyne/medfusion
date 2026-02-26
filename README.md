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

# 安装依赖（使用 uv）
uv sync

# 或使用 pip
pip install -e .
```

### 训练模型

```bash
# 使用默认配置训练
uv run medfusion-train --config configs/default.yaml

# 自定义配置
uv run medfusion-train --config configs/multiview_resnet.yaml
```

### 启动 Web UI

```bash
./start-webui.sh
# 访问 http://localhost:8000
```

## 📖 文档

- [完整文档](docs/README.md)
- [API 参考](docs/api/README.md)
- [配置指南](docs/guides/configuration.md)
- [开发指南](docs/development/README.md)

## 🏗️ 架构

```
medfusion/
├── med_core/              # 核心 Python 库
│   ├── models/            # 模型架构
│   ├── datasets/          # 数据加载器
│   ├── trainers/          # 训练逻辑
│   └── web/               # Web 服务
├── med_core_rs/           # Rust 加速模块
├── web/frontend/          # React 前端
├── configs/               # 配置模板
├── examples/              # 使用示例
└── tests/                 # 测试套件
```

## 🧪 测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试
uv run pytest tests/test_models.py

# 生成覆盖率报告
uv run pytest --cov=med_core --cov-report=html
```

## 🤝 贡献

欢迎贡献！请查看 [贡献指南](CONTRIBUTING.md)。

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 📮 联系方式

- 问题反馈: [GitHub Issues](https://github.com/yourusername/medfusion/issues)
- 邮件: your.email@example.com

## 🙏 致谢

感谢所有贡献者和开源社区的支持。
