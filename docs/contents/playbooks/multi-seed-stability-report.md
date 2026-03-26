# 多 seed 稳定性汇报

> 文档状态：**Beta**

目标：避免单次 seed 偶然性，输出可汇报的稳定性结论（mean/std）。

## 适用场景

- 版本定版前稳定性检查
- 对外评审需要提供方差信息
- 研究结论需要可重复性证明

## 执行方式（smurf_e2e）

```bash
# 使用配置中的 stability.seeds
bash demo/smurf_e2e/run_single_ct.sh stable stability

# 临时覆盖 seeds
SEEDS=13,21,34 bash demo/smurf_e2e/run_single_ct.sh stable stability
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
