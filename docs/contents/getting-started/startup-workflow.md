# MedFusion 启动流程与使用工作流（统一口径）

> 文档状态：Stable
>
> 目标：把用户路径收口成一条主线，并明确 ComfyUI 只作为可选适配模块。

## 1. 先定边界

- MedFusion 的执行真源是 `runtime + CLI`。
- Web UI 负责引导、校验、编译、监控和结果回流展示。
- ComfyUI 是外部子系统，不替代 MedFusion 训练执行层。

---

## 2. 唯一主线（推荐）

适用场景：首次上手、常规训练、对外演示主链。

### 2.1 页面路径

1. `/start`：选择任务入口
2. `/config`：问题向导生成配置
3. `/training`：启动并观察训练
4. `/models`：查看结果、导入 run、交付复盘

### 2.2 CLI 对应动作

```bash
uv run medfusion validate-config --config configs/starter/quickstart.yaml
uv run medfusion train --config configs/starter/quickstart.yaml
uv run medfusion build-results --config configs/starter/quickstart.yaml --checkpoint outputs/quickstart/checkpoints/best.pth
```

## 3. 可选模块：ComfyUI 适配（预览）

适用场景：你需要 ComfyUI 参与前置流程编排，但不改变 MedFusion 主线。

### 3.1 进入方式（挂在主线内）

1. 主线仍从 `/start -> /config -> /training -> /models` 走
2. 若需要 ComfyUI，在 `/config/comfyui` 做连通性与适配档案选择
3. 可跳转 `/config/advanced/canvas?blueprint=...` 做组件语义对齐
4. 完成后继续回到主线训练与结果回流

### 3.2 关键约束

- ComfyUI 负责外部流程交互
- MedFusion 负责训练执行、配置合同、结果合同
- 不允许把 ComfyUI 当成 MedFusion 训练替代层

---

## 4. 使用建议

- 默认总是先走唯一主线
- 只有在需要外部流程联动时，再启用 ComfyUI 适配模块
- 若要新增模型能力，先扩 runtime，再扩 UI

---

## 5. 验收标准（用户视角）

用户完成一次闭环时，至少要做到：

1. 能明确知道当前是否处在主线或主线内的可选适配模块
2. 能从页面路径回溯到对应 CLI 命令
3. 训练结果能在 `/models` 里被稳定看到和解释
