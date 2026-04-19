# Why MedFusion OSS（定位对比）

> 文档状态：**Stable**

这页只回答一个问题：在常见替代方案面前，MedFusion OSS 的定位是什么。

## 对比 1：通用“训练仓库”

通用训练仓库通常强在“可训练”，但训练后结果组织和验证产物不一定稳定。

MedFusion OSS 的区别：
- 默认入口先是 `medfusion start`，会把推荐首跑链路和成功标准讲清楚
- 默认主链是 `validate-config -> train -> build-results`
- 默认产物是结构化契约（`metrics/metrics.json / metrics/validation.json / reports/summary.json / reports/report.md`）
- 更强调“可复盘和可交付”，不只强调“跑起来”
- 当前正式版 preview 已经开始把这条主链挂到 GUI 前台，而不是只停留在 CLI

## 对比 2：AutoML/拖拽式平台

AutoML 平台强在交互易用和低门槛，但在可控性、可替换性和工程接入边界上，往往需要平台约束。

MedFusion OSS 的区别：
- Web 负责 onboarding 和结果理解，CLI / YAML 仍然负责真实执行
- 高级模式已有节点图 preview，但节点图会先编译成配置候选，而不是直接充当执行真源
- 明确的可替换点：`backbone / fusion / head / trainer`
- 面向工程接入而不是只面向页面操作
- 输出契约适合接到上层系统或报告流程

所以它当前更像：

- 一个 GUI-first 的模型搭建前台
- 一个 engine-first 的研究运行时
- 一个结果可回流、可解释、可交付的结果后台

而不是一个“任意模型都能自由拖拽出来”的全能 builder

## 对比 3：临床部署软件

临床部署软件关注合规、流程治理和上线运行保障，目标与研究 runtime 不同。

MedFusion OSS 的区别：
- 当前定位是研究验证与工程复盘
- 不承诺临床合规部署能力
- 强调可重复实验和可解释结果沉淀

## 一句话结论

MedFusion OSS 不是“功能最多”的那个，而是“先把路径讲清楚，再把训练-验证-结果闭环做稳”的那个。

如果从正式版 preview 来说，它当前最准确的外部理解是：

**研究运行时 + GUI 模型搭建主链 + 真实结果闭环**
