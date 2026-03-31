# Dev Scripts

`scripts/dev/` 用来放偏工程内部、偏开发者自测、或不适合作为普通用户入口展示的脚本。

这里的脚本通常具有这些特征：

- 主要用于验证某个工程能力或辅助模块
- 不属于当前官方 CLI / Web 训练主链
- 不适合作为低学习成本的模型搭建入口

如果你的目标是稳定跑通 MedFusion，请优先使用：

- `configs/starter/`
- `configs/public_datasets/`
- `medfusion train`
- `medfusion build-results`
