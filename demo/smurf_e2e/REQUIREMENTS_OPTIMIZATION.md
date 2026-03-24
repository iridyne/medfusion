# SMuRF E2E（单CT）要求与优化清单

## A. 演示可用性要求（医生沟通场景）

- [x] 单命令跑通：训练 → 评估 → 报告
- [x] 固定输出目录，避免覆盖历史结果
- [x] 必须产出：`metrics.json`、`predictions.csv`、`reports/smurf_e2e_report.md`
- [x] 必须产出可解释图并汇总到 `visualizations/gallery/`

## B. 工程稳定性要求

- [x] 配置分档：`fast`（快速出结果） / `stable`（正式复现）
- [x] 训练新增 early stopping（可配置开关）
- [x] 一键脚本优先使用 `.venv-medml/bin/python`，不存在再回退 `uv run python`

## C. 本次已落地优化

1. 新增配置：
   - `config.elbow_single_ct_fast.yaml`
   - `config.elbow_single_ct_stable.yaml`
2. 训练支持 early stopping：
   - `training.early_stopping.enabled`
   - `training.early_stopping.patience`
   - `training.early_stopping.min_delta`
3. 新增一键脚本：
   - `run_single_ct.sh`（`fast|stable|base` 三档）
4. README 已补充分档说明与推荐命令。

## D. 推荐执行命令

```bash
# 快速演示（默认）
bash demo/smurf_e2e/run_single_ct.sh fast

# 稳定复现（自动 early stopping）
SKIP_PREPARE=1 bash demo/smurf_e2e/run_single_ct.sh stable
```

## E. 下一轮可做（如果你同意）

- [ ] 增加“最低可接受指标阈值”检查（如 AUC、Accuracy 下限）
- [ ] 训练结束自动生成一页「医生结论摘要」JSON
- [ ] 稳定档支持多次 seed 运行并汇总均值/方差
