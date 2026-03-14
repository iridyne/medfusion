# MedFusion 文档结构整理

## 📚 文档总览

当前文档共 **54 个文件**，分为 **15 个目录**。

### 1. 用户文档 (`user-guides/`)
面向框架使用者和新手

| 文件 | 描述 |
|------|------|
| `QUICKSTART_GUIDE.md` | 快速入门，常见问题解决 |
| `DOCKER_GUIDE.md` | Docker 部署指南 |
| `WEB_UI_QUICKSTART.md` | Web 界面使用指南 |

### 2. API 文档 (`api/`)
完整的 API 参考文档

| 文件 | 模块 |
|------|------|
| `med_core.md` | 核心模块总览 |
| `backbones.md` | 骨干网络 (Vision, Tabular, 3D) |
| `fusion.md` | 融合策略 (Concat, Gated, Attention 等) |
| `heads.md` | 任务头 (Classification, Survival) |
| `aggregators.md` | MIL 聚合器 |
| `attention_supervision.md` | 注意力监督 |
| `datasets.md` | 数据加载器 |
| `trainers.md` | 训练器 |
| `evaluation.md` | 评估指标 |
| `preprocessing.md` | 数据预处理 |
| `utils.md` | 工具函数 |

### 3. 架构文档 (`architecture/`)
系统设计和架构分析

| 文件 | 内容 |
|------|------|
| `WEB_UI_ARCHITECTURE.md` | Web UI 架构设计 |
| `WORKFLOW_DESIGN.md` | 工作流引擎设计 |
| `design_architecture_analysis.md` | 代码库架构分析 |
| `gradient_checkpointing_design.md` | 梯度检查点设计 |
| `optimization_roadmap.md` | 优化路线图 |

### 4. 开发文档 (`development/`)
面向贡献者和维护者

| 文件 | 内容 |
|------|------|
| `DEVELOPMENT_STRATEGY.md` | 开发策略和时间分配 |
| `COMPETITOR_ANALYSIS.md` | 竞品分析和最佳实践 |
| `P0_FIXES_REPORT.md` | 关键问题修复记录 |
| `REMAINING_ISSUES.md` | 当前问题清单和优先级 |
| `ISSUES_FOUND.md` | 已知问题列表 |
| `CODEBASE_ANALYSIS.md` | 代码库结构分析 |

### 5. 指南文档 (`guides/`)
详细的功能指南

**主要指南：**
- `quick_reference.md` - 快速参考
- `faq_troubleshooting.md` - FAQ 和故障排除
- `api_documentation.md` - API 文档指南

**功能指南：**
- `distributed_training.md` - 分布式训练
- `gradient_checkpointing_guide.md` - 梯度检查点
- `model_compression.md` - 模型压缩
- `model_export.md` - 模型导出 (ONNX, TorchScript)
- `data_caching.md` - 数据缓存
- `performance_benchmarking.md` - 性能基准测试

**部署指南：**
- `docker_deployment.md` - Docker 部署
- `ci_cd.md` - CI/CD 流程

**子目录：**
- `attention/` - 注意力机制文档
  - `mechanism.md` - 注意力机制详解
  - `supervision.md` - 注意力监督
- `multiview/` - 多视图文档
  - `overview.md` - 多视图概览
  - `types_complete.md` - 完整类型说明
  - `types_quickref.md` - 快速参考

### 6. 参考文档 (`reference/`)
数据字典和错误代码

| 文件 | 内容 |
|------|------|
| `error_codes.md` | 框架错误代码 (E001-E028) |
| `data_dictionary.yaml` | 数据字典 |

### 7. 历史记录 (`archive/`)
已完成的测试和会话记录

- `SESSION_SUMMARY_2026-02-21.md` - 会话总结
- `SIMULATION_TEST_RESULTS.md` - 模拟测试结果
- `FULL_WORKFLOW_TEST_RESULTS.md` - 完整工作流测试
- `BUG_FIX_SUMMARY.md` - Bug 修复总结
- `ATTENTION_SUPERVISION_TEST.md` - 注意力监督测试

### 8. 演示文档 (`presentations/`)
Reveal.js 演示文件

- `slides.md` - 演示幻灯片
- 相关配置和样式文件

### 9. 根目录文档
- `README.md` - 文档导航和快速开始
- `index.md` - Sphinx 文档入口
- `conf.py` - Sphinx 配置

---

## 🎯 文档使用指南

### 按用户角色

**新手用户** → `user-guides/QUICKSTART_GUIDE.md`
**开发者** → `development/DEVELOPMENT_STRATEGY.md`
**架构师** → `architecture/WEB_UI_ARCHITECTURE.md`
**API 使用者** → `api/med_core.md`

### 按功能

**模型构建** → `api/models.md` + `api/backbones.md` + `api/fusion.md`
**数据处理** → `api/datasets.md` + `guides/data_caching.md`
**训练** → `api/trainers.md` + `guides/distributed_training.md`
**部署** → `guides/docker_deployment.md` + `guides/model_export.md`
**性能优化** → `guides/performance_benchmarking.md` + `guides/gradient_checkpointing_guide.md`

---

## 📊 文档统计

| 类别 | 数量 |
|------|------|
| 用户文档 | 3 |
| API 文档 | 11 |
| 架构文档 | 5 |
| 开发文档 | 6 |
| 指南文档 | 15 |
| 参考文档 | 2 |
| 历史记录 | 5 |
| 演示文档 | 1 |
| **总计** | **54** |

---

## 🔄 文档维护建议

### 当前状态
- ✅ 结构清晰，分类合理
- ✅ 覆盖面广（API、架构、开发、用户）
- ⚠️ 部分文档可能需要更新（最后更新时间：2025-02-27）
- ⚠️ 演示文档与主文档分离

### 改进方向
1. **统一更新时间** - 标记每个文档的最后更新时间
2. **交叉引用** - 在相关文档间添加链接
3. **版本管理** - 为重要文档添加版本号
4. **搜索优化** - 添加关键词和标签
5. **定期审查** - 建立文档审查流程

---

**最后整理时间：** 2026-03-14
