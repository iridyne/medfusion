# 公开数据集 Quickstart Configs

这三份配置的目标不是追求 benchmark，而是让公开数据集先顺利走通 MedFusion 当前的训练主链。

## 可用配置

- `pathmnist_quickstart.yaml`
  - 适合最快验证图像训练、结果页和 artifact 输出
  - 推荐先跑 `medfusion public-datasets prepare medmnist-pathmnist`
- `breastmnist_quickstart.yaml`
  - 适合最小二分类图像 quick validation
  - 推荐先跑 `medfusion public-datasets prepare medmnist-breastmnist`
- `uci_heart_disease_quickstart.yaml`
  - 适合最快验证表格指标链路
  - 推荐先跑 `medfusion public-datasets prepare uci-heart-disease`

## 推荐命令

```bash
uv run medfusion public-datasets prepare medmnist-pathmnist --overwrite
uv run medfusion train --config configs/public_datasets/pathmnist_quickstart.yaml

uv run medfusion public-datasets prepare medmnist-breastmnist --overwrite
uv run medfusion train --config configs/public_datasets/breastmnist_quickstart.yaml

uv run medfusion public-datasets prepare uci-heart-disease --overwrite
uv run medfusion train --config configs/public_datasets/uci_heart_disease_quickstart.yaml
```

## 重要说明

当前 CLI 主链按统一的多模态输入处理，因此这里做了两层适配：

1. `PathMNIST`
   - 不额外引入表格特征
   - 通过数据加载器的 dummy tabular fallback 走统一训练链

2. `UCI Heart Disease`
   - 保留真实表格特征
   - 自动写入一张中性占位图，让表格任务也能进入当前多模态主链

这是一层 MVP 适配，不是最终的数据接入形态。
