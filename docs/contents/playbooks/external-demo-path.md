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
