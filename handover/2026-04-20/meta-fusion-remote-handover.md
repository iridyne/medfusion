# Meta Fusion 远程接力交接（2026-04-20）

更新时间：2026-04-20 21:22 CST
仓库：`/Users/yixian/Projects/medfusion`
当前分支：`main`

## 1. 本轮已落地（已提交）

本次设备切换前，已完成并提交以下 6 个 commit（按时间顺序）：

1. `a38dfde` test: fix cross-platform training signal assertions
2. `0b23cd3` feat(web): prefer db-backed experiments API over mock data
3. `a8d0f14` feat(web): switch experiment comparison UI to real experiments API
4. `81a74fd` feat(web): prefer experiment artifacts for metrics and charts
5. `030260e` feat(web): resolve run experiment artifacts from experiments config
6. `f96113b` refactor(web): remove mock stats from experiment report payload

## 2. 关键交付内容

### 2.1 Experiments API 从 mock 迁到真实数据优先

文件：
- `med_core/web/routers/experiments.py`
- `tests/test_experiments_api.py`

核心变化：
- 实验列表优先读 DB（`experiments` / `model_info`），仅在无真实数据时 fallback mock。
- `favorite/delete` 对 `model-*`、`run-*` 生效，兼容 legacy `exp-*`。
- `/metrics`、`/confusion-matrix`、`/roc-curve` 优先读取 artifact JSON：
  - `history_path`
  - `confusion_matrix_json_path`
  - `roc_curve_json_path`
- `run-*` 场景：若无 `model_info` 匹配，也会从 `experiments.config.artifact_paths` 读取 artifact。

### 2.2 报告链路去除硬编码 mock 统计值

文件：
- `med_core/web/routers/experiments.py`

核心变化：
- 抽出 `_build_comparison_payload`，`/compare` 与 `/report` 共用真实比较结果。
- `report` 不再写死 `t_test/wilcoxon` 假 p-value；当前 `statistical_tests` 为空对象（避免伪造统计结论）。

### 2.3 前端实验对比页改为真实 API

文件：
- `web/frontend/src/pages/ExperimentComparison.tsx`
- `web/frontend/src/components/experiment/ConfusionMatrix.tsx`
- `web/frontend/src/components/experiment/ROCCurve.tsx`

核心变化：
- compare/report/favorite 走后端接口。
- 混淆矩阵、ROC 数据由后端提供，不再本地 mock 生成。

## 3. 已验证状态

已通过测试（本机最后一次）：

```bash
~/.local/bin/uv run pytest tests/test_experiments_api.py tests/test_training_control_api.py tests/test_web_api_minimal.py tests/test_workflow_api.py -q
# 结果：15 passed
```

说明：
- `uv` 已安装在 `~/.local/bin/uv`。
- 该环境缺少 `npm`，因此未在本机完成前端 build 验证（后端/接口回归已通过）。

## 4. 真实 DB 与 mock 现状（你关心的点）

- 真实 DB 文件：`~/.medfusion/medfusion.db`
- 现有核心表：`dataset_info`、`experiments`、`model_info`、`training_jobs`
- 代码中的 mock 目前只作为兼容 fallback，不是主路径。

## 5. 路线图对齐说明（本轮原则）

执行原则：
- 严格按当前 roadmap 的“收敛/清理/真实主链”方向推进。
- 不新增实体、不扩表、不引入新基础设施。
- 只在现有模型/接口上做真实数据优先与 mock 缩减。

## 6. 远程设备接力建议（可直接执行）

1. 同步代码：
```bash
git fetch origin
git checkout main
git pull --ff-only origin main
```

2. 准备运行环境：
```bash
python3 -m pip install --user uv
~/.local/bin/uv sync --extra dev --extra web
```

3. 快速回归：
```bash
~/.local/bin/uv run pytest tests/test_experiments_api.py tests/test_training_control_api.py tests/test_web_api_minimal.py tests/test_workflow_api.py -q
```

## 7. 下一步建议（继续 roadmap，低风险顺序）

1. 继续缩减 `experiments` 路由内剩余 mock fallback 覆盖面（保留必要兼容）。
2. 在不引入新实体前提下，补齐 experiments report API 的契约测试面（下载与格式侧）。
3. 在远程设备补做前端 build/test，确认 UI 链路端到端无回归。

