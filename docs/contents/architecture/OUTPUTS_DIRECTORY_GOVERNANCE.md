# Outputs Directory Governance

## Why This Document Exists

`oss/outputs/` 当前承载了两类不同语义的内容：

1. 标准训练 / 导入 / 评估 run 的正式产物
2. 脚本测试、演示素材、临时验证结果

第一类已经在代码层有明确结构；第二类目前没有统一出口，因此逐步堆积到 `outputs/` 顶层，造成目录语义混杂。

这份文档的目标是明确一件事：

`outputs/` 不是通用杂物箱，而是 **标准 run archive 根目录**。

---

## Canonical Responsibility

`oss/outputs/` 只负责保存 **一次可复现 run 的根目录**。

这里的 run 包括：

- CLI `train` 产生的训练输出
- `build-results` / `import-run` 产生的结果归档
- 需要被复盘、对比、汇报或导入到结果页的评估产物

不应进入 `outputs/` 的内容包括：

- 一次性脚本验证结果
- 报告模块自测产物
- 字体、海报、宣传图等展示素材
- web demo seed data
- 临时调试日志
- 结构不符合标准 run layout 的散落文件

判断标准只有一个：

如果这个目录不能被当作“某次 run 的完整归档根目录”理解，它就不应该进入 `outputs/`。

---

## Canonical Run Layout

标准结构由 [`med_core/output_layout.py`](/home/yixian/Projects/medfusion/oss/med_core/output_layout.py) 定义。

每个 run root 应满足如下布局：

```text
outputs/<category>/<run_name>/
├── checkpoints/
├── logs/
├── metrics/
├── reports/
└── artifacts/
```

各目录职责：

- `checkpoints/`
  保存权重文件，例如 `best.pth`、`last.pth`
- `logs/`
  保存过程日志，例如 `training.log`、`history.json`
- `metrics/`
  保存结构化指标，例如 `metrics.json`、`validation.json`
- `reports/`
  保存汇总产物，例如 `summary.json`、`report.md`
- `artifacts/`
  保存图表、可视化、配置快照等附加产物

可选扩展结构：

- `seeds/`
  多 seed 稳定性实验的子运行目录
- `stability/`
  seed 聚合后的汇总结果

这意味着 `outputs/` 顶层不应再出现：

- 顶层 `checkpoints/`
- 顶层 `logs/`
- 顶层 `history.json`
- 顶层 `metrics.json`
- 顶层测试日志文本

这些都属于旧式残留结构，而不是当前标准布局。

---

## Admission Rules

一个目录只有同时满足下面条件，才应该进入 `outputs/`：

1. 它对应一次明确的 run，而不是脚本副产物
2. 它能被 CLI / 文档 / 人类复盘者稳定引用
3. 它有清晰的 `run_name`，并且能落入标准 layout
4. 它包含或预期包含 checkpoint、logs、metrics、reports、artifacts 中的若干项
5. 它的内容对后续复现、展示、比较或导入有价值

反过来，以下目录不应进入 `outputs/`：

- 只为了验证某个函数是否工作
- 只包含若干 PNG、TXT、HTML，且不属于某次正式 run
- 主要用途是 web 演示数据预置
- 主要用途是内容素材制作
- 主要用途是开发期临时观察

---

## Recommended Top-Level Taxonomy

`outputs/` 的一级目录建议只保留“运行来源分类”，而不是“任意脚本名”或“资产名”。

推荐分类：

- `outputs/starter/`
  面向最小跑通链路的入门 run
- `outputs/public_datasets/`
  面向公开数据集 quickstart / baseline run
- `outputs/demo/`
  面向可展示但仍属于正式 run 的 demo 结果
- `outputs/testing/`
  面向需要保留归档结果的集成测试 / smoke run
- `outputs/experiments/`
  面向真实研究性实验 run

命名规则建议：

- 使用稳定语义名，不使用 “test1” / “new” / “final_final”
- 目录名反映来源和用途，而不是脚本实现细节
- 尽量由配置名或实验名直接映射

示例：

```text
outputs/starter/quickstart/
outputs/public_datasets/breastmnist_quickstart/
outputs/demo/smurf_lite_dr_z/
outputs/testing/smoke_cli_pathmnist/
outputs/experiments/lung_fusion_baseline_v2/
```

---

## Classification Of Current Remaining Directories

基于当前仓库状态，`outputs/` 中剩余目录可以分为两类。

### A. Should Stay In `outputs/`

这些目录符合“正式 run archive”的方向，应保留在 `outputs/`，但部分命名仍可继续规范化。

- `quickstart`
  对应 starter 主链 run，属于标准训练产物
- `public_datasets`
  对应公开数据集 quickstart run 集合，属于标准训练产物
- `smoke`
  若内部保存的是实际 smoke run 结果，应保留，但建议后续更明确地归到 `outputs/testing/`
- `smurf_lite_dr_z_demo`
  若其内部满足标准 layout，并被 demo / 结果页引用，则应视为 demo run archive

### B. Should Move Out Of `outputs/`

这些目录虽然有内容，但语义不是“标准 run archive”。

- `demo-web-data`
  更像 web 演示预置数据，不是一次训练 run 的归档
- `promo-assets`
  明显属于传播 / 展示素材
- `font-preview`
  明显属于视觉或字体预览结果
- `enhanced_report_test`
  属于报告模块功能验证产物
- `latex_test`
  属于 LaTeX 报告功能验证产物
- `statistical_test`
  属于统计报告功能验证产物
- `full_workflow_test`
  属于脚本级工作流验证结果；除非后续明确把它升级为正式集成测试 run，否则不应与正式产物并列

---

## Suggested Destinations For Non-Run Outputs

为了避免再次污染 `outputs/`，建议把非 run 产物迁移到语义更清楚的目录。

推荐归位：

- `demo-web-data`
  建议迁到 `web/dev-data/` 或 `demo_assets/web-data/`
- `promo-assets`
  建议迁到 `docs/assets/` 或 `demo_assets/promo/`
- `font-preview`
  建议迁到 `docs/assets/font-preview/` 或 `demo_assets/font-preview/`
- `enhanced_report_test`
  建议迁到 `artifacts/dev/report-tests/enhanced/`
- `latex_test`
  建议迁到 `artifacts/dev/report-tests/latex/`
- `statistical_test`
  建议迁到 `artifacts/dev/report-tests/statistical/`
- `full_workflow_test`
  若只是开发验证，建议迁到 `artifacts/dev/workflow-tests/full/`
  若后续要作为正式集成测试沉淀，则应改造成 `outputs/testing/<run_name>/` 的标准 layout

目录选择原则：

- 面向产品或页面演示的数据，放到 `web/` 或 `demo_assets/`
- 面向文档和内容展示的素材，放到 `docs/assets/`
- 面向开发期临时验证的产物，放到 `artifacts/dev/` 或 `.tmp/`
- 只有正式 run 才进入 `outputs/`

---

## Root Cause Of The Current Drift

当前混乱并不是因为 `outputs/` 概念错误，而是因为规范只落实了一半。

已经完成的部分：

- `RunOutputLayout` 已经把正式 run 的结构标准化
- CLI 主链已经在使用这套结构
- docs 首页也已经开始按 `checkpoints / logs / metrics / reports / artifacts` 描述输出

尚未完成的部分：

- 若干脚本直接把测试结果写到 `outputs/<script_name>`
- 演示数据和素材没有单独归档根目录
- README 仍保留旧式结构描述，容易误导后续贡献者继续往顶层写散落文件

所以问题的本质不是“目录太多”，而是：

正式 run 的规范存在，但非 run 产物缺少出口，结果全部被倒进了 `outputs/`。

---

## Immediate Recommendations

下一阶段建议按以下顺序推进。

1. 统一文档口径
   把 README 中仍然使用旧输出结构的描述改成与 `RunOutputLayout` 一致

2. 收敛脚本输出边界
   把 `scripts/test_enhanced_report.py`、`scripts/full_workflow_test.py`、`scripts/quick_simulation_test.py` 等脚本的默认输出路径迁到 `artifacts/dev/` 或 `outputs/testing/`

3. 给演示数据单独立根
   为 web demo / promo / preview 内容建立独立目录，不再与 run archive 并列

4. 保持 `outputs/` 的 run-only 约束
   后续任何新增目录，只要不是标准 run root，就不允许进入 `outputs/`

---

## Practical Rule Of Thumb

以后遇到一个新目录，先问一句：

“这是某次训练 / 导入 / 评估 run 的完整归档根目录吗？”

如果答案是：

- “是”，放进 `outputs/`
- “不是”，放到 `artifacts/dev/`、`docs/assets/`、`web/dev-data/` 或其他更具体的位置

这条规则比记忆目录名更重要。
