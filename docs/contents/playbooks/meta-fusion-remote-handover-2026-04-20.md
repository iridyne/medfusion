# Meta Fusion 远程接力交接（2026-04-20）

> 文档状态：**Draft**

该文档用于设备切换时的开发接力，和本地暂存同步版本：
- `handover/2026-04-20/meta-fusion-remote-handover.md`

## 1. 交接范围

本次交接覆盖最近 6 个 commit（按时间顺序）：

1. `a38dfde` test: fix cross-platform training signal assertions
2. `0b23cd3` feat(web): prefer db-backed experiments API over mock data
3. `a8d0f14` feat(web): switch experiment comparison UI to real experiments API
4. `81a74fd` feat(web): prefer experiment artifacts for metrics and charts
5. `030260e` feat(web): resolve run experiment artifacts from experiments config
6. `f96113b` refactor(web): remove mock stats from experiment report payload

## 2. 主要落地内容

### 2.1 Experiments 主路径改为真实数据优先

- 列表优先读取 `experiments/model_info`，仅无真实记录时 fallback。
- `model-*` / `run-*` 的收藏与删除行为落地。
- 训练历史、混淆矩阵、ROC 曲线优先读取 artifact JSON。
- `run-*` 在缺少 `model_info` 映射时，也可从 `experiments.config.artifact_paths` 读取产物。

涉及文件：
- `med_core/web/routers/experiments.py`
- `tests/test_experiments_api.py`

### 2.2 报告接口去除 mock 统计

- `compare/report` 共用 `_build_comparison_payload` 真实比较结果。
- 删除报告中硬编码的假 `t_test/wilcoxon` 统计值。
- 当前无已验证统计流程时，`statistical_tests` 为空对象。

涉及文件：
- `med_core/web/routers/experiments.py`
- `tests/test_experiments_api.py`

### 2.3 前端对比页切换到后端 API

涉及文件：
- `web/frontend/src/pages/ExperimentComparison.tsx`
- `web/frontend/src/components/experiment/ConfusionMatrix.tsx`
- `web/frontend/src/components/experiment/ROCCurve.tsx`

## 3. 验证记录

```bash
~/.local/bin/uv run pytest tests/test_experiments_api.py tests/test_training_control_api.py tests/test_web_api_minimal.py tests/test_workflow_api.py -q
# 15 passed
```

说明：当前环境无 `npm`，前端 build 尚未在本机复验。

## 4. 数据现状

真实 DB：`~/.medfusion/medfusion.db`

核心表：
- `dataset_info`
- `experiments`
- `model_info`
- `training_jobs`

本轮未新增实体、未改表结构。

## 5. 接力建议

1. 远程设备先 `git pull --ff-only origin main`。
2. 安装/使用 `uv` 跑回归子集。
3. 继续按 roadmap 做“mock 收敛”，避免引入新实体或过度工程化。

