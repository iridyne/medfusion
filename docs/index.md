---
layout: home

hero:
  name: MedFusion
  text: 医学 AI 研究验证核心运行时
  tagline: 真实训练 · 真实结果 · 结构化评估摘要与结果报告输出
  actions:
    - theme: brand
      text: 先跑通一次
      link: /contents/getting-started/quickstart
    - theme: alt
      text: 如何新建 YAML
      link: /contents/getting-started/model-creation-paths
    - theme: alt
      text: 公开数据集验证
      link: /contents/getting-started/public-datasets

features:
  - icon: ✅
    title: 真实训练主链
    details: 从配置检查、训练、评估到结果构建，围绕同一条可执行主链组织，适合研究验证、课题复现与内部原型推进。

  - icon: 📦
    title: 结构化结果输出
    details: 标准化产出 checkpoint、logs/history.json、metrics/metrics.json、metrics/validation.json、reports/summary.json、reports/report.md，以及区分能力曲线、混淆矩阵、注意力图等结果图示。

  - icon: 🚀
    title: 公开数据集快速验证
    details: 没有私有数据也能直接上手。当前可通过 public-datasets 入口快速验证 PathMNIST、BreastMNIST、UCI Heart Disease 等示例路径。

  - icon: 🧩
    title: 可复用 runtime contract
    details: 同一套配置、训练和结果 contract 可被 CLI、Web、结果页、模型库和上层产品复用，不必为不同入口重复造轮子。

  - icon: 🛠️
    title: 模块化建模能力
    details: 支持 backbone、fusion、head、trainer 等组件解耦组合，适合多模态医学 AI 研究中的快速试验与迭代。

  - icon: 📚
    title: 以主路径组织文档
    details: 文档优先服务上手路径。先跑通主链，再看 API、架构和高级能力，而不是一开始就陷进实现细节。
---

## MedFusion 是什么

MedFusion 是一个面向 **医学 AI 研究验证** 的开源运行时。

它的重点不是只提供若干模型组件，而是把一条稳定的实验主链组织清楚：

```text
配置检查 → 数据加载 → 模型训练 / 评估 → 结果构建 → 评估摘要 / 结果报告输出
```

如果你需要的是：

- 快速验证一个医学 AI 任务能不能跑通
- 在不同 backbone / fusion / head 之间做实验组合
- 为上层 Web、结果页或产品层沉淀稳定结果结构
- 把训练、validation 和报告产物规范化

那 MedFusion 的设计就是为这些事服务的。

第一次使用建议先做一件事：

- 先跑通一次主链
- 没有私有数据就先走公开数据集
- 成功一次之后再回头看配置、Builder 和架构

如果你想自己新建 YAML 或模型，先看
[如何新建模型与 YAML](/contents/getting-started/model-creation-paths)。
当前建议很直接：

- 普通用户复制主链模板
- 高级用户走 Builder / 代码路径
- 真正新的模型能力先扩 runtime，再扩 YAML

## 你可以用它做什么

### 1. 跑通自己的医学 AI 训练任务

如果你已经有自己的数据和任务定义，可以从配置驱动主链直接开始：

- [CLI 与 Config 使用路径](/contents/getting-started/cli-config-workflow)
- [快速入门指南](/contents/getting-started/quickstart)

最小命令示例：

```bash
uv run medfusion validate-config --config configs/starter/quickstart.yaml
uv run medfusion train --config configs/starter/quickstart.yaml
uv run medfusion build-results \
  --config configs/starter/quickstart.yaml \
  --checkpoint outputs/quickstart/checkpoints/best.pth
```

### 2. 没有私有数据时先做 quick validation

如果你只是想先验证环境、流程和结果结构，当前首推对外演示路径就是：`public-datasets -> train -> build-results`

- [公开数据集快速验证清单](/contents/getting-started/public-datasets)

最小命令示例：

```bash
uv run medfusion public-datasets list
uv run medfusion public-datasets prepare medmnist-breastmnist --overwrite
uv run medfusion train --config configs/public_datasets/breastmnist_quickstart.yaml
uv run medfusion build-results \
  --config configs/public_datasets/breastmnist_quickstart.yaml \
  --checkpoint outputs/public_datasets/breastmnist_quickstart/checkpoints/best.pth
```

### 3. 为上层系统提供稳定结果 contract

MedFusion 输出的不只是模型 checkpoint，还包括一组更适合被系统消费的结果文件，例如：

- `metrics/metrics.json`
- `metrics/validation.json`
- `reports/summary.json`
- `reports/report.md`
- 区分能力曲线 / 混淆矩阵 / 校准曲线 / 注意力图等结果图示

这让它既适合研究验证，也适合作为上层产品或工作台的执行核心。

## 推荐阅读路径

### 新用户

1. [如何新建模型与 YAML](/contents/getting-started/model-creation-paths)
2. [CLI 与 Config 使用路径](/contents/getting-started/cli-config-workflow)
3. [快速入门指南](/contents/getting-started/quickstart)
4. [公开数据集快速验证清单](/contents/getting-started/public-datasets)
5. [Web UI 快速入门](/contents/getting-started/web-ui)

### 研究 / 工程用户

1. [Core Runtime Architecture](/contents/architecture/CORE_RUNTIME_ARCHITECTURE)
2. [API 文档总览](/contents/api/med_core)
3. [快速参考](/contents/guides/core/quick-reference)

### 准备做演示或推广

1. [OSS 对外推广准备清单](/contents/guides/core/oss-go-to-market-checklist)
2. [公开数据集快速验证清单](/contents/getting-started/public-datasets)
3. [examples/README.md](https://github.com/iridyne/medfusion/blob/main/examples/README.md)

## 核心能力概览

**配置驱动实验组织**
- 通过 YAML 管理训练任务、数据、模型和输出路径
- 先做配置检查，再进入训练与结果构建

**模块化模型能力**
- 支持多种 backbone、fusion、head 和聚合方式
- 适合多模态、多视图和医学任务原型迭代

**结果与报告闭环**
- 输出标准化指标、逐例评估摘要和可读报告
- 方便结果页、模型库和后续分析流程复用

**面向上层承接**
- CLI 可直接运行
- Web 与产品层可围绕统一 contract 对接

## 文档入口

- **[快速开始](/contents/getting-started/cli-config-workflow)** — 先理解当前最稳定的上手主链
- **[如何新建模型与 YAML](/contents/getting-started/model-creation-paths)** — 先判断自己该复制 YAML、进 Builder，还是先扩 runtime
- **[公开数据集验证](/contents/getting-started/public-datasets)** — 没有私有数据时的最快入口
- **[教程](/contents/tutorials/README)** — 从基础概念到高级能力的系统学习路径
- **[API 文档](/contents/api/med_core)** — 模块与接口参考
- **[架构设计](/contents/architecture/CORE_RUNTIME_ARCHITECTURE)** — 理解运行时组织方式与边界
- **[功能指南](/contents/guides/core/quick-reference)** — 常用命令、FAQ 与高级功能入口

## 社区与反馈

- [GitHub Issues](https://github.com/iridyne/medfusion/issues) — 问题反馈
- [GitHub Discussions](https://github.com/iridyne/medfusion/discussions) — 使用讨论与经验交流

## OSS WebUI 设计

- [OSS WebUI 工作流 Spec](/contents/architecture/OSS_WEBUI_WORKFLOW_SPEC) — 固定新的工作流导向信息架构、tabs 职责和现有页面映射
- [OSS WebUI 单页骨架图](/contents/architecture/OSS_WEBUI_WORKFLOW_OVERVIEW) — 用一张总览图快速解释整个 WebUI 的结构
