# 结果解读与交付检查

> 文档状态：**Stable**

目标：把 run 结果从“我看懂了”升级到“别人也能复核”。

## 必看文件

- `metrics/metrics.json`：核心指标
- `metrics/validation.json`：验证细节与切分信息
- `reports/summary.json`：最终汇报摘要
- `reports/report.md`：可读报告

## 交付前检查清单

- [ ] 配置文件路径可追溯（知道这次 run 用了哪个 config）
- [ ] checkpoint 可追溯（`best.pth` 路径明确）
- [ ] 主指标和次指标都已解释（不只报一个最高分）
- [ ] 数据划分与样本量已标注
- [ ] 异常点或不稳定点已注明
- [ ] 结论与 Non-goals 一致（不过度承诺）

## 推荐对外表达模板

- 输入：本次基于 `<dataset/config>` 执行标准主链
- 过程：完成 `validate-config -> train -> build-results`
- 输出：形成结构化 artifacts 与可读报告
- 结论：当前结果用于研究验证与工程复盘，不等同临床部署结论
