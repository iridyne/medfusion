# MedFusion 快速参考

这页只保留当前 OSS MVP 最应该记住的入口、命令和输出 contract。

## 推荐入口

```bash
uv run medfusion start
```

它会先进入 `Getting Started`，再把你带到 `Quickstart Run`，帮助你理解推荐 quickstart、主链阶段和预期产物。

## YAML 主线

### 本地 quickstart

```bash
uv run medfusion validate-config --config configs/starter/quickstart.yaml
uv run medfusion train --config configs/starter/quickstart.yaml
uv run medfusion build-results \
  --config configs/starter/quickstart.yaml \
  --checkpoint outputs/quickstart/checkpoints/best.pth
```

### 公开数据 quickstart

```bash
uv run medfusion public-datasets prepare medmnist-breastmnist --overwrite
uv run medfusion train --config configs/public_datasets/breastmnist_quickstart.yaml
uv run medfusion build-results \
  --config configs/public_datasets/breastmnist_quickstart.yaml \
  --checkpoint outputs/public_datasets/breastmnist_quickstart/checkpoints/best.pth
```

### 自定义配置

```bash
uv run medfusion validate-config --config configs/my_config.yaml
uv run medfusion train --config configs/my_config.yaml
uv run medfusion build-results \
  --config configs/my_config.yaml \
  --checkpoint outputs/my_run/checkpoints/best.pth
```

## `validate-config` 会告诉你什么

除了检查 YAML 和数据准备，它现在还会直接打印当前配置的 mainline contract：

- `model_type`
- `vision_backbone`
- `fusion_type`
- `output_dir`
- `best.pth / summary.json / validation.json`
- 下一步推荐命令：`train / build-results / import-run`

如果这一步打印出来的 contract 就不对，不要继续训练，先改 YAML。

## 预期输出

标准 run 至少应该落这些文件：

- `outputs/<run_name>/checkpoints/best.pth`
- `outputs/<run_name>/logs/history.json`
- `outputs/<run_name>/metrics/metrics.json`
- `outputs/<run_name>/metrics/validation.json`
- `outputs/<run_name>/reports/summary.json`
- `outputs/<run_name>/reports/report.md`
- `outputs/<run_name>/artifacts/`

## 常用命令

### 评估模型

```bash
uv run medfusion evaluate \
  --config configs/starter/quickstart.yaml \
  --checkpoint outputs/quickstart/checkpoints/best.pth
```

### 导入结果

```bash
uv run medfusion import-run \
  --config configs/starter/quickstart.yaml \
  --checkpoint outputs/quickstart/checkpoints/best.pth
```

### 指定输出目录

```bash
uv run medfusion train \
  --config configs/starter/quickstart.yaml \
  --output-dir outputs/exp1
```

## 最小排错

- `Unknown fusion type: concat`
  - 用 `concatenate`，不要用 `concat`
- `FileNotFoundError`
  - 先跑 `validate-config`，优先检查它打印的 `output_dir`、CSV 路径和图像目录
- `KeyError: <column>`
  - YAML 里的特征列必须与 CSV 列名一致
- CPU 下 `pin_memory` 警告
  - 设成 `false`
- 只看到了 checkpoint，没有 metrics / reports
  - 还没执行 `build-results`，或者 `--checkpoint` 路径传错了

## 调试

```bash
export MEDFUSION_LOG_LEVEL=DEBUG
uv run medfusion train --config configs/starter/quickstart.yaml

python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.memory_summary())"
```

## 相关文档

- `docs/contents/getting-started/quickstart.md`
- `docs/contents/getting-started/cli-config-workflow.md`
- `docs/contents/getting-started/web-ui.md`
- `docs/contents/getting-started/public-datasets.md`
- `docs/contents/tutorials/deployment/production.md`
