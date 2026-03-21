# Starter Configs

这里是当前 `medfusion train` 最适合新用户直接使用的配置。

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
