# Runtime Optimization Backlog (2026-03-30)

> 状态：Active backlog（延后优化）
>
> 目标：记录 `med_core` 当前已落地优化与后续待办，避免上下文丢失。

## 已完成（本轮）

1. 版本号单一来源
- `med_core/web/config.py` 与 `med_core/web/__init__.py` 改为复用 `med_core.version.__version__`。
- 消除 Web 子系统独立硬编码版本导致的漂移风险。

2. Workflow 执行依赖状态澄清
- `med_core/web/node_executors.py` 不再报不存在的伪模块路径。
- 改为明确提示：实验工作流执行 runtime 尚未接入 OSS 主链，建议使用稳定主链（`medfusion start` / `medfusion train` / Training API）。

## 待优化（延后）

3. 拆分超大文件，降低维护成本
- 优先目标：
  - `med_core/postprocessing/results.py`
  - `med_core/backbones/vision.py`
  - `med_core/web/api/training.py`
  - `med_core/models/builder.py`
- 建议先按“装配 / 校验 / I/O / 产物导出”拆分，保持对外 API 不变。

4. 收敛训练主链分叉
- 现状：`med_core/cli/train.py` 同时维护 three-phase 专用路径 + 通用路径。
- 建议：抽象为可注册的 runner（按 `dataset_type` / `model_type` 选择），减少命令层膨胀。

5. 强化 schema 边界体验
- 现状：builder schema 与 train schema 已有防呆，但错误提示仍偏“开发者视角”。
- 建议：增加稳定错误码、文档跳转和“下一步怎么做”的 CLI 提示模板。

6. 基准测试进入可选自动化
- 现状：`benchmarks/baseline.json` 主要用于手动比较。
- 建议：增加非阻塞 CI 任务（nightly 或 label 触发）执行：
  - `scripts/run_benchmarks.py`
  - `scripts/compare_benchmarks.py`

## 建议执行顺序

1. 第 3 项（拆分大文件）
2. 第 4 项（训练主链收敛）
3. 第 5 项（错误体验）
4. 第 6 项（benchmark 自动化）

