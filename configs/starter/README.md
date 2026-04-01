# Starter Configs

这里是当前 `medfusion train` 最适合新用户直接使用的配置。

## 官方支持矩阵

当前 starter 目录进入第一版 MVP 官方支持矩阵的配置是：

- `quickstart.yaml`
- `default.yaml`

它们都属于当前 `YAML 主链` 的正式入口，用于：

- 从最小模板开始做本地数据实验
- 在现有 schema 内复现和调整成熟实验
- 作为后续扩展自己 YAML 的起点

## 推荐顺序

1. `quickstart.yaml`
   - 最适合第一次验证 CLI 主链
   - 使用仓库自带 mock 数据
   - 训练轮数更少，默认更稳

2. `default.yaml`
   - 作为更完整的基线配置参考
   - 仍然兼容当前训练主链
   - 不建议新用户把它当作第一条命令

## 直接运行

```bash
uv run medfusion train --config configs/starter/quickstart.yaml
```

推荐完整链路：

```bash
uv run medfusion validate-config --config configs/starter/quickstart.yaml
uv run medfusion train --config configs/starter/quickstart.yaml
uv run medfusion build-results --config configs/starter/quickstart.yaml --checkpoint outputs/quickstart/checkpoints/best.pth
```
