# 对外 Demo 路径

> 文档状态：**Stable**

目标：准备一条对外演示时可重复、可解释、可快速复现的标准路径。

## 推荐路径

1. 先说明输入：使用公开数据集 quickstart 配置
2. 再展示执行：`validate-config -> train -> build-results`
3. 最后展示产物：`reports/summary.json + reports/report.md + artifacts/visualizations`

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
