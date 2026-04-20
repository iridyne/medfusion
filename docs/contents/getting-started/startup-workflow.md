# MedFusion 启动流程与使用工作流（统一口径）

> 文档状态：Stable
>
> 目标：把用户真正要走的路径收口成两条，避免入口过多导致的认知混乱。

## 1. 先定边界

- MedFusion 的执行真源是 `runtime + CLI`。
- Web UI 负责引导、校验、编译、监控和结果回流展示。
- ComfyUI 是外部子系统，不替代 MedFusion 训练执行层。

---

## 2. 标准主线（推荐，90% 用户）

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

---

## 3. ComfyUI 适配线（预览）

适用场景：需要 ComfyUI 作为外部流程编排层，但训练和结果仍回到 MedFusion 主链。

### 3.1 页面路径

1. `/config/comfyui`：检查 ComfyUI 连通性
2. 选择“适配档案”：把 ComfyUI 流程映射到 MedFusion 组件语义
3. 一键跳转 `/config/advanced/canvas?blueprint=...`：在对应骨架上编译与校验
4. 训练后回到 `/models`：结果回流与交付复盘

### 3.2 这条线的关键约束

- ComfyUI 负责外部流程交互
- MedFusion 负责训练执行、配置合同、结果合同
- 不允许把 ComfyUI 当成 MedFusion 训练替代层

---

## 4. 什么时候走哪条

- 你只想跑通结果：走“标准主线”
- 你要做 ComfyUI 适配联动：走“ComfyUI 适配线”
- 你要做新模型能力：先扩 runtime，再扩 UI

---

## 5. 验收标准（用户视角）

用户完成一次闭环时，至少要做到：

1. 能明确知道当前在走哪条线
2. 能从页面路径回溯到对应 CLI 命令
3. 训练结果能在 `/models` 里被稳定看到和解释
