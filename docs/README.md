# MedFusion 文档

欢迎使用 MedFusion 多模态医学影像融合框架！

## 📚 文档导航

### 🎓 教程

**从入门到精通的完整学习路径：**
- [教程总览](contents/tutorials/README.md) - 三种学习路径（快速入门/完整学习/深度学习）
- [你的第一个模型](contents/tutorials/modules/02_first_model.md) - 30分钟快速上手
- [配置文件详解](contents/tutorials/modules/03_understanding_configs.md) - 理解配置系统
- [数据准备指南](contents/tutorials/modules/04_data_preparation.md) - 准备医学影像数据
- [模型构建器 API](contents/tutorials/modules/05_builder_api.md) - 使用 Builder API
- [查看所有教程模块](contents/tutorials/README.md#-所有教程模块)

### 🚀 快速开始

**新手必读：**
- [快速入门指南](contents/user-guides/QUICKSTART_GUIDE.md) - 新手避坑指南，常见问题和解决方案
- [Docker 部署指南](contents/user-guides/DOCKER_GUIDE.md) - 使用 Docker 部署 MedFusion
- [Web UI 快速入门](contents/user-guides/WEB_UI_QUICKSTART.md) - Web 界面使用指南

### 📖 API 文档

完整的 API 参考文档：

**核心模块：**
- [med_core](contents/api/med_core.md) - 核心模块总览
- [models](contents/api/models.md) - 模型构建器
- [backbones](contents/api/backbones.md) - 骨干网络
- [fusion](contents/api/fusion.md) - 融合策略
- [heads](contents/api/heads.md) - 任务头

**数据和训练：**
- [datasets](contents/api/datasets.md) - 数据加载器
- [trainers](contents/api/trainers.md) - 训练器
- [preprocessing](contents/api/preprocessing.md) - 数据预处理

**高级功能：**
- [aggregators](contents/api/aggregators.md) - MIL 聚合器
- [attention_supervision](contents/api/attention_supervision.md) - 注意力监督
- [evaluation](contents/api/evaluation.md) - 评估指标
- [utils](contents/api/utils.md) - 工具函数

### 📘 功能指南

**核心指南：**
- [快速参考](contents/guides/quick_reference.md) - 常用命令速查
- [FAQ 和故障排除](contents/guides/faq_troubleshooting.md) - 常见问题解答

**高级功能��**
- [分布式训练](contents/guides/distributed_training.md) - 多 GPU 和多节点训练
- [梯度检查点](contents/guides/gradient_checkpointing_guide.md) - 内存优化技术
- [模型压缩](contents/guides/model_compression.md) - 模型剪枝和量化
- [模型导出](contents/guides/model_export.md) - ONNX 和 TorchScript 导出
- [数据缓存](contents/guides/data_caching.md) - 加速数据加载
- [性能基准测试](contents/guides/performance_benchmarking.md) - 性能评估
- [CI/CD 流程](contents/guides/ci_cd.md) - 持续集成和部署

**专题指南：**
- [注意力机制](contents/guides/attention/mechanism.md) - CBAM, SE, ECA 详解
- [多视图支持](contents/guides/multiview/overview.md) - 多角度 CT、时间序列等

### 🏗️ 架构文档

深入了解系统设计：

- [Web UI 架构](contents/architecture/WEB_UI_ARCHITECTURE.md) - Web 界面架构设计
- [工作流设计](contents/architecture/WORKFLOW_DESIGN.md) - 工作流引擎设计

### 📋 参考文档

- [错误代码](contents/reference/error_codes.md) - 框架错误代码参考

---

## 🎯 按场景查找

### 模型构建
1. [模型构建器 API](contents/api/models.md)
2. [骨干网络 API](contents/api/backbones.md)
3. [融合策略 API](contents/api/fusion.md)

### 数据处理
1. [数据加载器 API](contents/api/datasets.md)
2. [数据预处理 API](contents/api/preprocessing.md)
3. [数据缓存指南](contents/guides/data_caching.md)

### 训练和优化
1. [训练器 API](contents/api/trainers.md)
2. [分布式训练](contents/guides/distributed_training.md)
3. [梯度检查点](contents/guides/gradient_checkpointing_guide.md)
4. [性能基准测试](contents/guides/performance_benchmarking.md)

### 部署
1. [Docker 部署](contents/user-guides/DOCKER_GUIDE.md)
2. [模型导出](contents/guides/model_export.md)
3. [CI/CD 流程](contents/guides/ci_cd.md)

---

## 🔗 相关资源

- **GitHub 仓库**: [iridite/medfusion](https://github.com/iridyne/medfusion)
- **配置模板**: `configs/` - YAML 配置文件示例
- **示例代码**: `examples/` - 使用示例
- **测试代码**: `tests/` - 单元测试和集成测试

---

**最后更新：** 2026-03-14
