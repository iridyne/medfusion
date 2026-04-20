# Web API Reference

> 文档状态：**Beta**
>
> 这页只记录当前仓库里真实存在、且与正式版主链直接相关的 API。

## Base URL

```text
http://127.0.0.1:8000
```

## 0. 健康检查

- `GET /health`

示例响应：

```json
{
  "status": "healthy",
  "version": "0.3.0",
  "data_dir": "C:/Users/Administrator/.medfusion"
}
```

## 1. 系统 API（稳定）

- `GET /api/system/features`
- `GET /api/system/info`
- `GET /api/system/version`
- `GET /api/system/resources`
- `GET /api/system/storage`

其中 `GET /api/system/features` 是正式版边界总入口，包含：

- 稳定主链页面
- 部署形态口径（local_browser / private_server / managed_cloud）
- 高级模式（preview）说明
- workflow 实验态开关状态（`MEDFUSION_ENABLE_EXPERIMENTAL_WORKFLOW`）

## 2. 训练 API（稳定）

- `POST /api/training/start`
- `GET /api/training/jobs`
- `GET /api/training/{job_id}/status`
- `GET /api/training/{job_id}/history`
- `POST /api/training/{job_id}/pause`
- `POST /api/training/{job_id}/resume`
- `POST /api/training/{job_id}/stop`
- `WS /api/training/ws/{job_id}`

`POST /api/training/start` 请求体（核心字段）：

```json
{
  "experiment_name": "my-run",
  "training_model_config": {
    "backbone": "resnet18",
    "num_classes": 2
  },
  "dataset_config": {
    "dataset_id": "1",
    "data_path": "data/public/medmnist/breastmnist-demo"
  },
  "training_config": {
    "epochs": 1,
    "batch_size": 16,
    "learning_rate": 0.001
  }
}
```

响应示例：

```json
{
  "job_id": "6e6a46f8-9ed5-4f46-a2c1-6d2c2f11f2b5",
  "status": "running",
  "message": "训练任务已启动"
}
```

WebSocket 推送关键类型：

- `status_update`
- `training_complete`
- `error`

## 3. 模型库 API（稳定）

- `GET /api/models/`
- `GET /api/models/search`
- `GET /api/models/statistics`
- `GET /api/models/backbones`
- `GET /api/models/formats`
- `GET /api/models/{model_id}`
- `POST /api/models/`
- `PUT /api/models/{model_id}`
- `DELETE /api/models/{model_id}`
- `POST /api/models/{model_id}/upload`
- `GET /api/models/{model_id}/download`
- `GET /api/models/{model_id}/artifacts/{artifact_key}`
- `POST /api/models/import-run`

`GET /api/models/{model_id}` 返回的核心结构：

- 基础信息：`id/name/architecture/accuracy/loss/...`
- `result_files`：可下载产物索引（含 `artifact_key` 和下载 URL）
- `training_history`
- `validation`
- `visualizations`

`visualizations` 当前已对齐到结果页展示能力，包含但不限于：

- `roc_curve`
- `confusion_matrix`
- `attention_maps`
- `feature_importance_bar`
- `feature_importance_beeswarm`
- `phase_importance`
- `case_explanations`
- `three_phase_heatmaps`

`three_phase_heatmaps` 示例：

```json
{
  "method": "gradcam",
  "phase_labels": ["arterial", "portal", "noncontrast"],
  "artifact_key": "heatmap_manifest",
  "artifact_url": "/api/models/12/artifacts/heatmap_manifest",
  "case_count": 8,
  "heatmap_count": 24,
  "cases": []
}
```

## 4. 数据集 API（稳定）

- `GET /api/datasets/`
- `GET /api/datasets/search`
- `GET /api/datasets/statistics`
- `GET /api/datasets/class-counts`
- `GET /api/datasets/{dataset_id}`
- `POST /api/datasets/`
- `PUT /api/datasets/{dataset_id}`
- `DELETE /api/datasets/{dataset_id}`
- `POST /api/datasets/{dataset_id}/analyze`

## 5. 高级模式 API（preview，正式版可见）

- `GET /api/advanced-builder/catalog`
- `POST /api/advanced-builder/compile`
- `POST /api/advanced-builder/start-training`

说明：

- 这是正式版高级模式预览能力，不是默认入口。
- `start-training` 会先做图编译和 contract 校验，校验通过后直接创建真实训练任务。
- `compile` 的 `issues[]` 采用结构化字段：`level`、`code`、`message`、`path`、`context`、`suggestion`，用于前端定位与修复提示。

## 6. ComfyUI 集成 API（preview）

- `GET /api/comfyui/health`
- `GET /api/comfyui/adapter-profiles`

说明：

- 用于探测 ComfyUI 服务连通性并返回最小上线提示（打开地址、推荐启动命令、回流提示）。
- 支持查询参数 `base_url`（可选），用于覆盖默认探测地址。
- `adapter-profiles` 返回 MedFusion 侧的适配档案（compile-ready blueprint、family chain、默认回流预填），用于把 ComfyUI 流程映射到正式版组件语义。

## 7. 实验态 API（experimental）

### Workflow（默认关闭）

- 前缀：`/api/workflows/*`
- 开关：`MEDFUSION_ENABLE_EXPERIMENTAL_WORKFLOW=true`
- 默认行为：返回 `workflow_experimental_disabled`

### Experiments（当前为示例/演示口径）

- 前缀：`/api/experiments/*`
- 主要用于实验比较与报告演示，不是当前正式版主链阻塞项。

## 8. 错误响应约定

常见错误响应体：

```json
{
  "detail": "错误描述"
}
```

`workflow` 关闭时示例：

```json
{
  "detail": {
    "code": "workflow_experimental_disabled",
    "message": "Workflow editor is experimental and disabled by default in the current MVP.",
    "enable_env": "MEDFUSION_ENABLE_EXPERIMENTAL_WORKFLOW=true"
  }
}
```
