# MedFusion 文档结构

## 📚 文档总览

精简后的文档共 **32 个文件**，分为 **6 个目录**。

### 1. 用户指南 (`user-guides/`) - 3 文件
面向框架使用者和新手

| 文件 | 描述 |
|------|------|
| `QUICKSTART_GUIDE.md` | 快速入门，常见问题解决 |
| `DOCKER_GUIDE.md` | Docker 部署指南 |
| `WEB_UI_QUICKSTART.md` | Web 界面使用指南 |

### 2. API 文档 (`api/`) - 12 文件
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
| `models.md` | 模型构建器 |

### 3. 功能指南 (`guides/`) - 11 文件
详细的功能使用指南

**核心指南：**
- `quick_reference.md` - 快速参考
- `faq_troubleshooting.md` - FAQ 和故障排除

**高级功能：**
- `distributed_training.md` - 分布式训练
- `gradient_checkpointing_guide.md` - 梯度检查点
- `model_compression.md` - 模型压缩
- `model_export.md` - 模型导出 (ONNX, TorchScript)
- `data_caching.md` - 数据缓存
- `performance_benchmarking.md` - 性能基准测试
- `ci_cd.md` - CI/CD 流程

**专题指南：**
- `attention/mechanism.md` - 注意力机制详解
- `multiview/overview.md` - 多视图支持

### 4. 架构文档 (`architecture/`) - 2 文件
系统设计和架构分析

| 文件 | 内容 |
|------|------|
| `WEB_UI_ARCHITECTURE.md` | Web UI 架构设计 |
| `WORKFLOW_DESIGN.md` | 工作流引擎设计 |

### 5. 参考文档 (`reference/`) - 1 文件
数据字典和错误代码

| 文件 | 内容 |
|------|------|
| `error_codes.md` | 框架错误代码 (MED-DATA-xxx, MED-MODEL-xxx) |

### 6. 根目录文档 - 3 文件
- `index.md` - VitePress 首页
- `README.md` - 文档导航
- `DOCUMENTATION_STRUCTURE.md` - 本文件

---

## 🎯 文档使用指南

### 按用户角色

**新手用户** → [快速入门指南](user-guides/QUICKSTART_GUIDE.md)
**开发者** → [快速参考](guides/quick_reference.md) + [API 文档](api/med_core.md)
**架构师** → [Web UI 架构](architecture/WEB_UI_ARCHITECTURE.md)
**运维人员** → [Docker 部署](user-guides/DOCKER_GUIDE.md)

### 按功能

**模型构建** → [API: models](api/models.md) + [API: backbones](api/backbones.md) + [API: fusion](api/fusion.md)
**数据处理** → [API: datasets](api/datasets.md) + [数据缓存](guides/data_caching.md)
**训练** → [API: trainers](api/trainers.md) + [分布式训练](guides/distributed_training.md)
**部署** → [Docker 部署](user-guides/DOCKER_GUIDE.md) + [模型导出](guides/model_export.md)
**性能优化** → [性能基准测试](guides/performance_benchmarking.md) + [梯度检查点](guides/gradient_checkpointing_guide.md)

---

## 📊 文档统计

| 类别 | 数量 |
|------|------|
| 用户指南 | 3 |
| API 文档 | 12 |
| 功能指南 | 11 |
| 架构文档 | 2 |
| 参考文档 | 1 |
| 根目录 | 3 |
| **总计** | **32** |

---

## 🔄 文档维护

### 清理说明

本次清理删除了以下内容：
- ❌ `archive/` - 历史会话记录和测试结果 (5 文件)
- ❌ `development/` - 内部开发文档 (6 文件)
- ❌ 重复的架构设计文档 (3 文件)
- ❌ 重复的指南文件 (5 文件)

**结果：** 从 52 个文件精简到 32 个文件，减少 38%

### 维护原则

1. **用户文档**：面向使用者，注重实用性和可操作性
2. **API 文档**：完整的 API 参考，使用 Sphinx autodoc 格式
3. **功能指南**：详细的功能使用说明，包含示例代码
4. **架构文档**：系统设计和原理，面向架构师和高级开发者
5. **参考文档**：错误代码、数据字典等查询资料

### 文档更新流程

- 所有文档使用中文编写
- 代码示例使用英文注释
- 每个文档包含最后更新时间
- 使用 VitePress 构建在线文档

---

**最后更新时间：** 2026-03-14
