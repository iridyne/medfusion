# Config Layout

`configs/` 现在按“执行链路”分组，而不是把所有 YAML 混在一层。

## 目录说明

- `starter/`
  - 给 `medfusion train` 直接使用的入门配置
  - 新用户先看这里
- `public_datasets/`
  - 公开数据集快速验证配置
  - 配合 `scripts/prepare_public_dataset.py` 使用
- `testing/`
  - 供 smoke / simulation / 可插拔性脚本使用的测试配置
  - 不建议当作对外主入口
- `builder/`
  - 面向 `MultiModalModelBuilder` / `build_model_from_config()` 的模型结构示例
  - 这里的 YAML 不等价于当前 `medfusion train` 主链配置
- `legacy/`
  - 历史模板或未收敛到当前主链的旧配置
  - 默认不建议新用户从这里开始

## 推荐起点

### 1. 训练 CLI

```bash
uv run medfusion train --config configs/starter/quickstart.yaml
```

### 2. 公开数据集快速验证

```bash
uv run python scripts/prepare_public_dataset.py uci-heart-disease --overwrite
uv run medfusion train --config configs/public_datasets/uci_heart_disease_quickstart.yaml
```

### 3. Builder / 模型结构实验

如果你要验证更灵活的模态拼装或参考 SMuRF 风格结构，看：

- `configs/builder/`

但这条线和当前 `medfusion train` 的 dataclass 配置不是同一套 schema。
