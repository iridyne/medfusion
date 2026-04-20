# MedFusion 启动流程与使用工作流（统一口径）

> 文档状态：Stable
>
> 目标：把用户路径收口成一条主线，并明确 ComfyUI 是主线默认配置适配层。

## 1. 先定边界

- MedFusion 的执行真源是 `runtime + CLI`。
- Web UI 负责引导、校验、编译、监控和结果回流展示。
- ComfyUI 是外部子系统，不替代 MedFusion 训练执行层。

---

## 2. 唯一主线（推荐）

适用场景：首次上手、常规训练、对外演示主链。当前在配置阶段默认使用 ComfyUI 适配配置语义。

### 2.1 页面路径

1. `/start`：进入默认主线入口
2. `/config`：问题向导生成配置（默认 ComfyUI 适配配置），并可一键带向导参数进入训练
3. `/training`：启动并观察训练
4. `/models`：查看结果、导入 run、交付复盘，并可基于当前结果重开配置
5. 从结果后台重开配置进入训练时，训练页会标记来源为结果驱动迭代链路
6. 若只做快速验证，可在结果页直接重跑训练（同样保留来源标记）
7. 在训练看板也可基于当前任务重开配置，再次进入训练时会标记为训练阶段驱动迭代链路
8. 跨页重开配置会尽量回填关键模型字段（`backbone`、`numClasses`），减少重复输入

### 2.2 CLI 对应动作

```bash
uv run medfusion validate-config --config configs/starter/quickstart.yaml
uv run medfusion train --config configs/starter/quickstart.yaml
uv run medfusion build-results --config configs/starter/quickstart.yaml --checkpoint outputs/quickstart/checkpoints/best.pth
```

## 3. 默认配置适配层：ComfyUI

适用场景：主线默认已采用 ComfyUI 适配配置语义；当你需要桥接连通性检查或适配档案选择时进入此页。

### 3.1 进入方式（挂在主线内）

1. 主线仍从 `/start -> /config -> /training -> /models` 走
2. 在 `/config/comfyui` 做连通性检查与适配档案选择
3. 可跳转 `/config/advanced/canvas?blueprint=...` 做组件语义对齐
4. 也可“带预填回到 `/config`”先微调向导字段
5. 页面顶部会提供“配置/训练/结果”快捷跳转，减少在桥接页停留后的路径断点
6. 完成后可一键带推荐参数进入 `/training`，再回流到 `/models`

### 3.2 关键约束

- ComfyUI 负责外部流程交互
- MedFusion 负责训练执行、配置合同、结果合同
- 不允许把 ComfyUI 当成 MedFusion 训练替代层

---

## 4. 使用建议

- 默认总是先走唯一主线
- 默认先按主线推进；仅在需要桥接检查或档案切换时进入 `/config/comfyui`
- 若要新增模型能力，先扩 runtime，再扩 UI

---

## 5. 验收标准（用户视角）

用户完成一次闭环时，至少要做到：

1. 能明确知道当前始终处在同一主线，ComfyUI 只承担配置适配与桥接检查
2. 能从页面路径回溯到对应 CLI 命令
3. 训练结果能在 `/models` 里被稳定看到和解释
