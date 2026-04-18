# 对外 Demo 路径

> 文档状态：**Stable**

目标：准备一条对外演示时可重复、可解释、可快速复现的标准路径。

## 推荐路径

1. 先说明入口：默认从 `uv run medfusion start` 进入 `Getting Started`
2. 再说明输入：使用公开数据集 quickstart 配置
3. 再展示执行：`validate-config -> train -> build-results`
4. 最后展示产物：`reports/summary.json + reports/report.md + artifacts/visualizations`

如果现场时间很短，可以把 `start` 只作为“解释路径”的第一步，真正执行仍回到下面这组 CLI 命令。

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

补充关注的 doctor-interest 产物：

- `artifacts/visualizations/doctor_interest/manifest.json`
- `metrics/case_explanations.json`

这些 doctor-interest overlays 表示模型建议医生复核的关注区域，不是分割轮廓，也不应表述成病灶边界。对外展示时建议统一说成“建议关注区”。

## 不建议这样讲

- 夸大为“临床可直接部署”
- 夸大为“全面 benchmark 平台”
- 把 demo 脚本当作产品能力边界
