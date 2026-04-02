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

当前这里也包含一个典型示例：

- `scripts/dev/model_stack_diagnostic.py`
  用于开发阶段检查模型堆栈、张量维度和梯度流；它不是仓库官方 smoke 入口，项目级自检仍应使用 `bash test/smoke.sh`
