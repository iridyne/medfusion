# MedFusion 教程

欢迎来到 MedFusion 教程！本教程体系采用**模块化设计**，每个教程都是独立的学习单元，你可以按需选择学习路径。

## 🎯 三种学习路径

### ⚡ 路径 1：快速入门（30 分钟）

**目标**：最快速度训练出第一个模型

**适合人群**：想要快速体验框架的用户

**学习模块**：
1. [环境安装](modules/01_installation.md)（5 分钟）
2. [你的第一个模型](modules/02_first_model.md)（30 分钟）⭐

**特点**：
- 使用合成数据，零门槛
- 端到端完整流程
- 立即看到结果

---

### 📚 路径 2：完整学习（2-3 小时）

**目标**：系统掌握框架核心功能

**适合人群**：希望全面了解框架的研究人员和工程师

**学习模块**：
1. [环境安装](modules/01_installation.md)（5 分钟）
2. [你的第一个模型](modules/02_first_model.md)（30 分钟）
3. [配置文件详解](modules/03_understanding_configs.md)（20 分钟）
4. [数据准备指南](modules/04_data_preparation.md)（30 分钟）
5. [模型构建器 API](modules/05_builder_api.md)（25 分钟）
6. [融合策略对比](modules/07_fusion_strategies.md)（20 分钟）
7. [训练工作流](modules/08_training_workflow.md)（30 分钟）
8. [模型导出](modules/13_model_export.md)（15 分钟）

**总时长**：约 2.5 小时

---

### 🎓 路径 3：深度学习（5-8 小时）

**目标**：掌握高级功能和真实案例

**适合人群**：需要在生产环境中使用框架的专业人员

**学习模块**：

**基础部分**（路径 2 的所有模块）

**高级功能**：
- [选择骨干网络](modules/06_choosing_backbones.md)（15 分钟）
- [监控训练进度](modules/09_monitoring_progress.md)（15 分钟）
- [超参数调优](modules/10_hyperparameter_tuning.md)（25 分钟）
- [注意力监督](modules/11_attention_supervision.md)（20 分钟）
- [多视图支持](modules/12_multiview_support.md)（25 分钟）
- [Docker 部署](modules/14_docker_deployment.md)（20 分钟）
- [生产环境清单](modules/15_production_checklist.md)（15 分钟）

**案例研究**（选择 1-2 个）：
- [肺结节检测](case_studies/01_lung_nodule_detection.md)（60 分钟）
- [乳腺癌分类](case_studies/02_breast_cancer_classification.md)（75 分钟）
- [生存预测](case_studies/03_survival_prediction.md)（90 分钟）

**总时长**：5-8 小时（取决于选择的案例数量）

---

## 📖 所有教程模块

### 入门模块

| 模块 | 时长 | 难度 | 描述 |
|------|------|------|------|
| [01. 环境安装](modules/01_installation.md) | 5 分钟 | ⭐ | 安装 MedFusion 和依赖 |
| [02. 你的第一个模型](modules/02_first_model.md) | 30 分钟 | ⭐ | 端到端训练第一个模型 |
| [03. 配置文件详解](modules/03_understanding_configs.md) | 20 分钟 | ⭐⭐ | 理解配置文件结构 |
| [04. 数据准备指南](modules/04_data_preparation.md) | 30 分钟 | ⭐⭐ | 准备医学影像数据 |

### 模型构建模块

| 模块 | 时长 | 难度 | 描述 |
|------|------|------|------|
| [05. 模型构建器 API](modules/05_builder_api.md) | 25 分钟 | ⭐⭐ | 使用 Builder API 构建模型 |
| [06. 选择骨干网络](modules/06_choosing_backbones.md) | 15 分钟 | ⭐⭐ | 如何选择合适的骨干网络 |
| [07. 融合策略对比](modules/07_fusion_strategies.md) | 20 分钟 | ⭐⭐⭐ | 8 种融合策略详解 |

### 训练模块

| 模块 | 时长 | 难度 | 描述 |
|------|------|------|------|
| [08. 训练工作流](modules/08_training_workflow.md) | 30 分钟 | ⭐⭐ | 完整训练流程 |
| [09. 监控训练进度](modules/09_monitoring_progress.md) | 15 分钟 | ⭐⭐ | TensorBoard 和 WandB |
| [10. 超参数调优](modules/10_hyperparameter_tuning.md) | 25 分钟 | ⭐⭐⭐ | 系统化调优策略 |

### 高级功能模块

| 模块 | 时长 | 难度 | 描述 |
|------|------|------|------|
| [11. 注意力监督](modules/11_attention_supervision.md) | 20 分钟 | ⭐⭐⭐ | 引导模型关注正确区域 |
| [12. 多视图支持](modules/12_multiview_support.md) | 25 分钟 | ⭐⭐⭐ | 多角度 CT、时间序列等 |

### 部署模块

| 模块 | 时长 | 难度 | 描述 |
|------|------|------|------|
| [13. 模型导出](modules/13_model_export.md) | 15 分钟 | ⭐⭐ | ONNX 和 TorchScript |
| [14. Docker 部署](modules/14_docker_deployment.md) | 20 分钟 | ⭐⭐ | 容器化部署 |
| [15. 生产环境清单](modules/15_production_checklist.md) | 15 分钟 | ⭐⭐⭐ | 上线前检查 |

### 案例研究

| 案例 | 时长 | 难度 | 描述 |
|------|------|------|------|
| [肺结节检测](case_studies/01_lung_nodule_detection.md) | 60 分钟 | ⭐⭐⭐ | CT 影像分类任务 |
| [乳腺癌分类](case_studies/02_breast_cancer_classification.md) | 75 分钟 | ⭐⭐⭐⭐ | 多模态融合案例 |
| [生存预测](case_studies/03_survival_prediction.md) | 90 分钟 | ⭐⭐⭐⭐ | 时间序列 + 临床数据 |

---

## 💡 如何使用本教程

### 按用户类型选择

**🆕 新手用户**
- 从[快速入门路径](#-路径-1快速入门30-分钟)开始
- 完成后可以继续学习完整路径

**👨‍💻 有深度学习基础的研究人员**
- 直接学习[完整学习路径](#-路径-2完整学习2-3-小时)
- 重点关注医学影像特有的功能（多视图、注意力监督）

**🏢 生产环境部署人员**
- 学习[深度学习路径](#-路径-3深度学习5-8-小时)
- 重点关注部署模块和生产环境清单

### 按需学习

每个模块都是独立的，你可以：
- 跳过已经熟悉的内容
- 只学习感兴趣的模块
- 按任意顺序学习（建议先完成入门模块）

---

## 📋 前置知识

### 必需
- Python 基础（变量、函数、类）
- 基本的命令行操作

### 推荐
- PyTorch 基础（张量、模型、训练循环）
- 深度学习基础概念（卷积、损失函数、优化器）
- 医学影像基础（CT、MRI、病理切片）

### 不需要
- 不需要医学背景（教程会解释医学概念）
- 不需要高级深度学习知识（教程会从基础开始）

---

## 🔗 相关资源

- [API 文档](../api/med_core.md) - 完整的 API 参考
- [用户指南](../user-guides/QUICKSTART_GUIDE.md) - 新手避坑指南
- [功能指南](../guides/quick_reference.md) - 快速参考
- [FAQ](../guides/faq_troubleshooting.md) - 常见问题解答

---

## 🤝 反馈和贡献

如果你在学习过程中遇到问题或有改进建议，欢迎：
- 提交 [GitHub Issue](https://github.com/iridite/medfusion/issues)
- 参与 [GitHub Discussions](https://github.com/iridite/medfusion/discussions)

---

**开始学习**：选择一个学习路径，点击第一个模块开始吧！🚀
