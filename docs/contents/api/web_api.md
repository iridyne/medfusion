# Web API Reference

MedFusion provides a RESTful API and WebSocket interface for training management, model operations, and system monitoring.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. This may change in future versions.

## Training API

### Start Training Job

**POST** `/api/training/start`

Start a new training job with the specified configuration.

**Request Body:**
```json
{
  "config": {
    "data": {
      "train_csv": "data/train.csv",
      "val_csv": "data/val.csv"
    },
    "model": {
      "vision_backbone": "resnet50",
      "fusion_type": "attention"
    },
    "training": {
      "epochs": 50,
      "batch_size": 32
    }
  },
  "job_name": "experiment_001"
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "started",
  "message": "Training job started successfully"
}
```

### Get Training Status

**GET** `/api/training/{job_id}/status`

Get the current status of a training job.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "epoch": 15,
  "total_epochs": 50,
  "metrics": {
    "train_loss": 0.234,
    "val_loss": 0.267,
    "val_accuracy": 0.892
  }
}
```

### Stop Training Job

**POST** `/api/training/{job_id}/stop`

Stop a running training job.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "stopped",
  "message": "Training job stopped successfully"
}
```

## Model API

### List Models

**GET** `/api/models`

List all registered models and checkpoints.

**Response:**
```json
{
  "models": [
    {
      "model_id": "resnet50_attention_001",
      "architecture": "resnet50",
      "fusion_type": "attention",
      "created_at": "2026-03-15T10:30:00Z",
      "metrics": {
        "val_accuracy": 0.892,
        "val_auc": 0.945
      }
    }
  ]
}
```

### Evaluate Model

**POST** `/api/models/{model_id}/evaluate`

Evaluate a model on a specified dataset.

**Request Body:**
```json
{
  "checkpoint_path": "outputs/checkpoints/best.pth",
  "test_csv": "data/test.csv",
  "split": "test"
}
```

**Response:**
```json
{
  "model_id": "resnet50_attention_001",
  "metrics": {
    "accuracy": 0.887,
    "auc": 0.941,
    "f1_score": 0.865
  },
  "report_path": "outputs/reports/evaluation_report.html"
}
```

## Dataset API

### List Datasets

**GET** `/api/datasets`

List all available datasets.

**Status:** ⚠️ TODO - Not yet implemented

## System API

### Get System Info

**GET** `/api/system/info`

Get system information including GPU availability and disk usage.

**Response:**
```json
{
  "gpu": {
    "available": true,
    "count": 2,
    "devices": [
      {
        "id": 0,
        "name": "NVIDIA RTX 3090",
        "memory_total": "24GB",
        "memory_used": "8GB"
      }
    ]
  },
  "disk": {
    "total": "1TB",
    "used": "450GB",
    "free": "550GB"
  },
  "python_version": "3.11.5",
  "pytorch_version": "2.1.0"
}
```

### Health Check

**GET** `/api/system/health`

Check if the API server is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-03-15T10:30:00Z"
}
```

## WebSocket API

### Training Progress Stream

**WebSocket** `ws://localhost:8000/ws/training/{job_id}`

Connect to receive real-time training progress updates.

**Message Format:**
```json
{
  "type": "progress",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "epoch": 15,
  "batch": 120,
  "total_batches": 200,
  "metrics": {
    "loss": 0.234,
    "accuracy": 0.892
  }
}
```

**Event Types:**
- `progress` - Training progress update
- `epoch_end` - Epoch completed
- `validation` - Validation metrics
- `completed` - Training completed
- `error` - Error occurred

## Error Responses

All endpoints return standard error responses:

```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": "Additional error details"
}
```

**Common HTTP Status Codes:**
- `200` - Success
- `400` - Bad Request
- `404` - Not Found
- `500` - Internal Server Error

## Rate Limiting

Currently, no rate limiting is enforced. This may change in production deployments.

## Examples

### Start Training with cURL

```bash
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d @config.json
```

### Monitor Training with WebSocket (Python)

```python
import asyncio
import websockets
import json

async def monitor_training(job_id):
    uri = f"ws://localhost:8000/ws/training/{job_id}"
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            data = json.loads(message)
            print(f"Epoch {data['epoch']}: Loss={data['metrics']['loss']:.4f}")

asyncio.run(monitor_training("550e8400-e29b-41d4-a716-446655440000"))
```
