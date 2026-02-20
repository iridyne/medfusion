# MedFusion 文档

欢迎使用 MedFusion - 医学多模态深度学习框架的文档中心。

**版本**: v0.3.0  
**最后更新**: 2026-02-20

## 📚 文档导航

### 🚀 快速开始

- [项目状态报告](PROJECT_STATUS.md) - 当前开发进度和功能完成度
- [快速参考](guides/quick_reference.md) - 常用命令和配置速查
- [FAQ 和故障排除](guides/faq_troubleshooting.md) - 常见问题解答

### 🌐 Web UI

- [Web UI 快速入门](WEB_UI_QUICKSTART.md) - 5 分钟上手 Web 界面
- [Web UI 架构设计](WEB_UI_ARCHITECTURE.md) - 完整的架构说明和设计决策

### 📖 使用指南

#### 核心功能
- [多视图概览](guides/multiview/overview.md) - 多视图数据处理入门
- [多视图类型完整指南](guides/multiview/types_complete.md) - 所有支持的多视图类型
- [多视图快速参考](guides/multiview/types_quickref.md) - 配置速查表

#### 注意力机制
- [注意力机制指南](guides/attention/mechanism.md) - CBAM、SE Block、ECA Block
- [注意力监督](guides/attention/supervision.md) - Mask-guided、CAM-based 监督

#### 性能优化
- [梯度检查点指南](guides/gradient_checkpointing_guide.md) - 减少内存使用
- [数据缓存](guides/data_caching.md) - 加速数据加载
- [性能基准测试](guides/performance_benchmarking.md) - 性能测试和优化

#### 模型管理
- [模型导出](guides/model_export.md) - 导出为 ONNX、TorchScript
- [模型压缩](guides/model_compression.md) - 量化、剪枝、蒸馏

#### 部署
- [Docker 部署](guides/docker_deployment.md) - 容器化部署指南
- [分布式训练](guides/distributed_training.md) - 多 GPU 和多节点训练
- [CI/CD 集成](guides/ci_cd.md) - 持续集成和部署

### 🔧 API 参考

完整的 Python API 文档：

- [med_core](api/med_core.md) - 核心模块
- [backbones](api/backbones.md) - 视觉骨干网络（29 个变体）
- [fusion](api/fusion.md) - 融合策略（5 种）
- [aggregators](api/aggregators.md) - 多视图聚合器（5 种）
- [attention_supervision](api/attention_supervision.md) - 注意力监督
- [datasets](api/datasets.md) - 数据集加载器
- [trainers](api/trainers.md) - 训练器
- [models](api/models.md) - 模型定义
- [heads](api/heads.md) - 分类/回归头
- [evaluation](api/evaluation.md) - 评估指标
- [preprocessing](api/preprocessing.md) - 数据预处理
- [utils](api/utils.md) - 工具函数

### 🏗️ 架构设计

深入了解 MedFusion 的设计理念：

- [设计架构分析](architecture/design_architecture_analysis.md) - 整体架构设计
- [梯度检查点设计](architecture/gradient_checkpointing_design.md) - 内存优化设计
- [优化路线图](architecture/optimization_roadmap.md) - 性能优化计划

### 📋 参考资料

- [错误代码](reference/error_codes.md) - 完整的错误代码列表和解决方案
- [API 文档](guides/api_documentation.md) - API 使用说明

## 🎯 按角色查找文档

### 新用户
1. [项目状态报告](PROJECT_STATUS.md) - 了解项目概况
2. [Web UI 快速入门](WEB_UI_QUICKSTART.md) - 体验可视化界面
3. [快速参考](guides/quick_reference.md) - 学习基本命令

### 研究人员
1. [多视图完整指南](guides/multiview/types_complete.md) - 处理多视图数据
2. [注意力机制指南](guides/attention/mechanism.md) - 使用注意力机制
3. [性能基准测试](guides/performance_benchmarking.md) - 优化实验性能

### 开发者
1. [Web UI 架构设计](WEB_UI_ARCHITECTURE.md) - 理解系统架构
2. [API 参考](api/) - 查阅 API 文档
3. [架构设计](architecture/) - 深入了解设计决策

### 运维人员
1. [Docker 部署](guides/docker_deployment.md) - 容器化部署
2. [分布式训练](guides/distributed_training.md) - 多节点部署
3. [CI/CD 集成](guides/ci_cd.md) - 自动化流程

## 📊 文档统计

- **总文档数**: 37 个
- **API 参考**: 12 个
- **使用指南**: 14 个
- **架构设计**: 3 个
- **参考资料**: 1 个
- **其他**: 7 个

## 🔄 最近更新

### 2026-02-20
- ✅ 完成 Web UI 架构整理
- ✅ 清理重复文档（47 → 37）
- ✅ 删除临时和过时文档
- ✅ 更新文档索引

### 文档清理详情
- 删除重复的 Web UI 文档（4 个）
- 删除重复的错误代码文档（1 个）
- 删除重复的 Docker 文档（1 个）
- 删除重复的注意力机制文档（2 个）
- 删除临时文档（2 个）

## 📝 文档贡献

欢迎改进文档！请遵循以下原则：

1. **避免重复** - 一个主题只需一个文档
2. **保持更新** - 及时更新过时内容
3. **清晰简洁** - 使用简单明了的语言
4. **示例丰富** - 提供可运行的代码示例

## 🔗 相关资源

- [主 README](../README.md) - 项目主页
- [AGENTS.md](../AGENTS.md) - AI 辅助开发记录
- [CHANGELOG.md](../CHANGELOG.md) - 版本更新日志
- [GitHub Issues](https://github.com/your-org/medfusion/issues) - 问题反馈

## 📞 获取帮助

- 查看 [FAQ](guides/faq_troubleshooting.md)
- 查看 [错误代码](reference/error_codes.md)
- 提交 [GitHub Issue](https://github.com/your-org/medfusion/issues)

---

**维护者**: Medical AI Research Team  
**文档版本**: v0.3.0  
**最后更新**: 2026-02-20