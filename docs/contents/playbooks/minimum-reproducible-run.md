# 最小可复现实验（MRE）

> 文档状态：**Stable**

目标：在最短路径内产出一份可复盘的 run 结果。

## 前置条件

- 已安装依赖（见 `getting-started/installation.md`）
- 可访问公开数据集下载源

如果你是第一次进入项目，建议先跑一次：

```bash
uv run medfusion start
```

先看 `Getting Started` 和 `Quickstart Run`，确认这次 MRE 的推荐数据集、阶段顺序和预期产物，再执行下面的 CLI 命令。

## 步骤

```bash
uv run medfusion public-datasets list
uv run medfusion public-datasets prepare medmnist-breastmnist --overwrite
uv run medfusion train --config configs/public_datasets/breastmnist_quickstart.yaml
uv run medfusion build-results \
  --config configs/public_datasets/breastmnist_quickstart.yaml \
  --checkpoint outputs/public_datasets/breastmnist_quickstart/checkpoints/best.pth
```

## 完成标准

至少看到以下产物：

- `outputs/public_datasets/breastmnist_quickstart/checkpoints/best.pth`
- `outputs/public_datasets/breastmnist_quickstart/logs/history.json`
- `outputs/public_datasets/breastmnist_quickstart/metrics/metrics.json`
- `outputs/public_datasets/breastmnist_quickstart/metrics/validation.json`
- `outputs/public_datasets/breastmnist_quickstart/reports/summary.json`
- `outputs/public_datasets/breastmnist_quickstart/reports/report.md`
- `outputs/public_datasets/breastmnist_quickstart/artifacts/`

## 常见失败

- 数据下载失败：重试 `public-datasets prepare`，检查网络与磁盘空间
- checkpoint 路径错误：确认 `build-results --checkpoint` 指向 `best.pth`
