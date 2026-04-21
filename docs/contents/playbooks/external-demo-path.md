# 对外 Demo 路径

> 文档状态：**Stable**

目标：准备一条对外演示时可重复、可解释、可快速复现的标准路径。

## 推荐路径

1. 先说明入口：默认从 `uv run medfusion start` 进入 `Getting Started`
2. 再说明前台：先走 `Run Wizard` 做问题收敛；如果要展示高级模式，再进入 `/config/advanced/canvas`
3. 再说明执行：高级模式节点图会先编译成正式配置候选，再直接创建真实训练任务
4. 再展示结果后台：训练完成后直接跳到 `Model Library` 和结果详情
5. 最后展示产物：`reports/summary.json + reports/report.md + artifacts/visualizations`

如果现场时间很短，可以把 `start` 只作为“解释路径”的第一步，真正执行仍回到下面这组 CLI 命令。

## 正式版高级模式演示路径

如果你希望对外展示的不只是“CLI 能跑”，而是“GUI 模型搭建主链”：

1. `medfusion start`
2. 进入 `Getting Started`
3. 进入 `Run Wizard` 或 `/config/advanced/canvas`
4. 选择一条 compile-ready blueprint
5. 展示：
   - 组件注册表
   - 连接约束
   - `ExperimentConfig` contract 校验
   - 直接创建训练任务
6. 训练完成后，直接跳到结果后台
7. 在结果详情页说明：
   - 这次结果来自 `advanced_builder`
   - 入口是 `advanced-builder-canvas`
   - blueprint 是哪一个
   - 结果如何回流成 summary / validation / report / visual artifacts

## 3 分钟固定演示脚本

1. `0:00 - 0:40`
   先从 `medfusion start` 进入 `Getting Started`，明确默认路径与高级模式边界。
2. `0:40 - 1:30`
   进入 `/config/advanced/canvas`，选 compile-ready blueprint，演示图编译与 `ExperimentConfig` contract 校验。
3. `1:30 - 2:10`
   直接创建真实训练任务，等待任务完成后从训练页跳转到结果详情。
4. `2:10 - 3:00`
   按固定四层口径讲结果详情，并补充高级模式来源链字段。

## 结果详情固定口径（演示与文档一致）

1. 结论层：`reports/summary.json` 的核心结论与主指标
2. 指标层：`metrics/metrics.json` 与 `metrics/validation.json`
3. 可视化层：ROC / 混淆矩阵 / 注意力等图示 artifact
4. 文件层：可下载与可归档文件（`report.md`、图示 artifact）

高级模式 run 还要显式展示来源链：
- `source_type=advanced_builder`
- `entrypoint=advanced-builder-canvas`
- `blueprint_id=<当前蓝图>`
- `recommended_preset=<当前蓝图对应正式 preset>`
- `compile_boundary=<当前蓝图所在编译边界>`

## 演示最小证据点

- 训练任务完成后可直接跳 `Model Library` 结果详情
- 结果详情能看到 summary / metrics / validation / report / artifact
- 结果详情里能看到高级模式来源链字段
- 文件系统中能看到 `metrics/`、`reports/`、`artifacts/` 标准目录

## 演示命令

```bash
uv run medfusion validate-config --config configs/public_datasets/breastmnist_quickstart.yaml
uv run medfusion train --config configs/public_datasets/breastmnist_quickstart.yaml
uv run medfusion build-results \
  --config configs/public_datasets/breastmnist_quickstart.yaml \
  --checkpoint outputs/public_datasets/breastmnist_quickstart/checkpoints/best.pth
```

## 演示重点（建议顺序）

- 不是只展示“训练能跑”，而是展示“结果可复盘”
- 用 `reports/summary.json` 讲结论
- 用 `metrics/validation.json` 讲稳定性和风险点
- 用 `reports/report.md` 讲可交付性（可读、可归档、可复核）

## 不建议这样讲

- 夸大为“临床可直接部署”
- 夸大为“全面 benchmark 平台”
- 把 demo 脚本当作产品能力边界
