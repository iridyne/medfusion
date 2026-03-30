# 多 seed 稳定性汇报

> 文档状态：**Beta**

目标：避免单次 seed 偶然性，输出可汇报的稳定性结论（mean/std）。

## 适用场景

- 版本定版前稳定性检查
- 对外评审需要提供方差信息
- 研究结论需要可重复性证明

## 执行方式（主线）

当前 CLI 没有独立的 `medfusion stability` 子命令。

推荐做法是基于共享的 `med_core.stability.run_stability_study`，把主线
`medfusion train` / `medfusion build-results` 串起来：

```python
from copy import deepcopy
from pathlib import Path
import subprocess

import yaml

from med_core.stability import run_stability_study

base_config_path = Path("configs/demo/three_phase_ct_mvi_demo.yaml")
study_dir = Path("outputs/three_phase_ct_mvi_demo_stability")

with base_config_path.open("r", encoding="utf-8") as f:
    base_config = yaml.safe_load(f)


def run_seed(seed: int, output_dir: Path) -> None:
    seed_config = deepcopy(base_config)
    seed_config["seed"] = seed
    seed_config.setdefault("data", {})["random_seed"] = seed
    seed_config.setdefault("logging", {})["output_dir"] = str(output_dir)

    config_path = output_dir / "config.seed.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(seed_config, f, sort_keys=False, allow_unicode=True)

    subprocess.run(
        ["uv", "run", "medfusion", "train", "--config", str(config_path)],
        check=True,
    )
    subprocess.run(
        [
            "uv",
            "run",
            "medfusion",
            "build-results",
            "--config",
            str(config_path),
            "--checkpoint",
            str(output_dir / "checkpoints" / "best.pth"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )


run_stability_study(
    seeds=[13, 21, 34],
    study_dir=study_dir,
    run_seed=run_seed,
    study_name="Three-Phase CT MVI Stability",
)
```

## 结果目录

```text
<study_root>/
├── seeds/
│   ├── seed-0013/
│   ├── seed-0021/
│   └── seed-0034/
└── stability/
    ├── summary.json
    ├── summary.csv
    └── summary.md
```

## 汇报建议

- 主指标：mean ± std
- 辅助指标：各 seed 明细分布
- 结论表达：先说稳定区间，再说单次最佳值
