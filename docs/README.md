# MedFusion 文档

欢迎使用 MedFusion 多模态医学影像融合框架！

## 📚 文档导航

### 🚀 快速开始

**新手必读：**
- [快速入门指南](user-guides/QUICKSTART_GUIDE.md) - 新手避坑指南，常见问题和解决方案
- [Docker 部署指南](user-guides/DOCKER_GUIDE.md) - 使用 Docker 部署 MedFusion
- [Web UI 快速入门](user-guides/WEB_UI_QUICKSTART.md) - Web 界面使用指南

### 📖 API 文档

完整的 API 参考文档：

**核心模块：**
- [med_core](api/med_core.md) - 核心模块总览
- [models](api/models.md) - 模型构建器
- [backbones](api/backbones.md) - 骨干网络
- [fusion](api/fusion.md) - 融合策略
- [heads](api/heads.md) - 任务头

**数据和训练：**
- [datasets](api/datasets.md) - 数据加载器
- [trainers](api/trainers.md) - 训练器
- [preprocessing](api/preprocessing.md) - 数据预处理

**高级功能：**
- [aggregators](api/aggregators.md) - MIL 聚合器
- [attention_supervision](api/attention_supervision.md) - 注意力监督
- [evaluation](api/evaluation.md) - 评估指标
- [utils](api/utils.md) - 工具函数

### 📘 功能指南

**核心指南：**
- [快速参考](guides/quick_reference.md) - 常用命令速查
- [FAQ 和故障排除](guides/faq_troubleshooting.md) - 常见问题解答

**高级功能：**
- [分布式训练](guides/distributed_training.md) - 多 GPU 和多节点训练
- [梯度检查点](guides/gradient_checkpointing_guide.md) - 内存优化技术
- [模型压缩](guides/model_compression.md) - 模型剪枝和量化
- [模型导出](guides/model_export.md) - ONNX 和 TorchScript 导出
- [数据缓存](guides/data_caching.md) - 加速数据加载
- [性能基准测试](guides/performance_benchmarking.md) - 性能评估
- [CI/CD 流程](guides/ci_cd.md) - 持续集成和部署

**专题指南：**
- [注意力机制](guides/attention/mechanism.md) - CBAM, SE, ECA 详解
- [多视图支持](guides/multiview/overview.md) - 多角度 CT、时间序列等

### 🏗️ 架构文档

深入了解系统设计：

- [Web UI 架构](architecture/WEB_UI_ARCHITECTURE.md) - Web 界面架构设计
- [工作流设计](architecture/WORKFLOW_DESIGN.md) - 工作流引擎设计

### 📋 参考文档

- [错误代码](reference/error_codes.md) - 框架错误代码参考

---

## 🎯 按场景查找

### 模型构建
1. [模型构建器 API](api/models.md)
2. [骨干网络 API](api/backbones.md)
3. [融合策略 API](api/fusion.md)

### 数据处理
1. [数据加载器 API](api/datasets.md)
2. [数据预处理 API](api/preprocessing.md)
3. [数据缓存指南](guides/data_caching.md)

### 训练和优化
1. [训练器 API](api/trainers.md)
2. [分布式训练](guides/distributed_training.md)
3. [梯度检查点](guides/gradient_checkpointing_guide.md)
4. [性能基准测试](guides/performance_benchmarking.md)

### 部署
1. [Docker 部署](user-guides/DOCKER_GUIDE.md)
2. [模型导出](guides/model_export.md)
3. [CI/CD 流程](guides/ci_cd.md)

---

## 🔗 相关资源

- **GitHub 仓库**: [iridite/medfusion](https://github.com/iridite/medfusion)
- **配置模板**: `configs/` - YAML 配置文件示例
- **示例代码**: `examples/` - 使用示例
- **测试代码**: `tests/` - 单元测试和集成测试

---

## 📝 文档维护

查看 [文档结构说明](DOCUMENTATION_STRUCTURE.md) 了解文档组织方式。

**文档组织原则：**
- **用户指南**：面向使用者，注重实用性
- **API 文档**：完整的 API 参考
- **功能指南**：详细的功能使用说明
- **架构文档**：系统设计和原理
- **参考文档**：错误代码、数据字典等

---

**最后更新：** 2026-03-14
