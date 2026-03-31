# Builder Configs

这里是 `MultiModalModelBuilder` / `build_model_from_config()` 的结构示例。

要点：

- 这些 YAML 主要描述模态、融合、任务头等结构
- 它们不等价于当前 `medfusion train` 主链配置
- 如果你想直接训练，请回到：
  - `configs/starter/`
  - `configs/public_datasets/`
  - `configs/testing/`

## 当前内容

- `generic_multimodal.yaml`
- `templates/`

## 典型用法

```python
import yaml
from med_core.models import build_model_from_config

with open("configs/builder/generic_multimodal.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model = build_model_from_config(config)
```
