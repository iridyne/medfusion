# 最小可复现实验（MRE）

> 文档状态：**Stable**

目标：在最短路径内产出一份可复盘的 run 结果。

## 前置条件

- 已安装依赖（见 `getting-started/installation.md`）
- 可访问公开数据集下载源

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
- `outputs/public_datasets/breastmnist_quickstart/results/metrics.json`
- `outputs/public_datasets/breastmnist_quickstart/results/validation.json`
- `outputs/public_datasets/breastmnist_quickstart/results/summary.json`
- `outputs/public_datasets/breastmnist_quickstart/results/report.md`

## 常见失败

- 数据下载失败：重试 `public-datasets prepare`，检查网络与磁盘空间
- checkpoint 路径错误：确认 `build-results --checkpoint` 指向 `best.pth`
