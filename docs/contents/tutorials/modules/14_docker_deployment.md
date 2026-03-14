# Docker 部署

**预计时间：20分钟**

本教程介绍如何使用 Docker 部署 MedFusion 模型和 Web UI。

## 快速开始

### 前提条件

- Docker 20.10+
- Docker Compose 2.0+（可选）
- NVIDIA Docker（GPU 支持，可选）

### 一键启动（CPU 版本）

```bash
# 构建镜像
docker build -t medfusion/medfusion:latest .

# 运行容器
docker run -d \
  --name medfusion \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/outputs:/app/outputs \
  medfusion/medfusion:latest

# 访问 Web UI
open http://localhost:8000
```

### 一键启动（GPU 版本）

```bash
# 运行容器（需要 NVIDIA Docker）
docker run -d \
  --name medfusion \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/outputs:/app/outputs \
  medfusion/medfusion:latest

# 访问 Web UI
open http://localhost:8000
```

## 镜像构建

### 基础镜像构建

```bash
# 构建 CPU 版本
docker build -t medfusion/medfusion:latest .

# 构建并指定版本标签
docker build -t medfusion/medfusion:0.3.0 .
```

### GPU 版本构建

```bash
# 使用 CUDA 基础镜像
docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 \
  -t medfusion/medfusion:latest-gpu .
```

## 容器运行

### 基本运行

```bash
docker run -d \
  --name medfusion \
  -p 8000:8000 \
  medfusion/medfusion:latest
```

### 完整配置运行

```bash
docker run -d \
  --name medfusion \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/configs:/app/configs:ro \
  -e MEDCORE_LOG_LEVEL=INFO \
  -e CUDA_VISIBLE_DEVICES=0 \
  --restart unless-stopped \
  medfusion/medfusion:latest
```

### GPU 支持

```bash
# 使用所有 GPU
docker run -d \
  --name medfusion \
  --gpus all \
  -p 8000:8000 \
  medfusion/medfusion:latest

# 使用指定 GPU
docker run -d \
  --name medfusion \
  --gpus '"device=0,1"' \
  -p 8000:8000 \
  medfusion/medfusion:latest

# 限制 GPU 内存
docker run -d \
  --name medfusion \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  --memory="16g" \
  -p 8000:8000 \
  medfusion/medfusion:latest
```

## Docker Compose

### 默认启动（GPU 版本）

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### CPU 版本启动

```bash
# 使用 CPU profile
docker-compose --profile cpu up -d

# 查看日志
docker-compose --profile cpu logs -f
```

### 开发模式

```bash
# 启动开发容器
docker-compose --profile dev up -d medfusion-dev

# 进入容器
docker exec -it medfusion-dev bash

# 在容器内开发
cd /app
medfusion web --reload
```

## 数据卷管理

### 推荐的目录结构

```
medfusion/
├── data/              # 数据集（只读挂载）
│   ├── train/
│   ├── val/
│   └── test/
├── outputs/           # 训练输出（读写）
│   ├── experiments/
│   └── tensorboard/
├── logs/              # 日志文件（读写）
├── checkpoints/       # 模型检查点（读写）
└── configs/           # 配置文件（只读挂载）
```

### 数据卷挂载

```bash
# 只读挂载（数据集）
-v $(pwd)/data:/app/data:ro

# 读写挂载（输出）
-v $(pwd)/outputs:/app/outputs

# 命名卷（持久化）
docker volume create medfusion-data
docker run -v medfusion-data:/app/data medfusion/medfusion:latest
```

### 数据备份

```bash
# 备份输出目录
docker run --rm \
  -v medfusion-outputs:/app/outputs \
  -v $(pwd)/backup:/backup \
  ubuntu tar czf /backup/outputs-$(date +%Y%m%d).tar.gz /app/outputs

# 恢复备份
docker run --rm \
  -v medfusion-outputs:/app/outputs \
  -v $(pwd)/backup:/backup \
  ubuntu tar xzf /backup/outputs-20260220.tar.gz -C /
```

## 环境变量

### MedFusion 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MEDCORE_DATA_DIR` | `/app/data` | 数据集目录 |
| `MEDCORE_OUTPUT_DIR` | `/app/outputs` | 输出目录 |
| `MEDCORE_LOG_DIR` | `/app/logs` | 日志目录 |
| `MEDCORE_LOG_LEVEL` | `INFO` | 日志级别 |

### Web UI 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MEDCORE_WEB_HOST` | `0.0.0.0` | 监听地址 |
| `MEDCORE_WEB_PORT` | `8000` | 监听端口 |
| `MEDCORE_WEB_RELOAD` | `false` | 热重载 |

### CUDA 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `CUDA_VISIBLE_DEVICES` | `0` | 可见的 GPU 设备 |

### 使用 .env 文件

```bash
# 创建 .env 文件
cat > .env << EOF
MEDCORE_LOG_LEVEL=DEBUG
MEDCORE_WEB_RELOAD=true
CUDA_VISIBLE_DEVICES=0,1
EOF

# 使用 .env 文件启动
docker-compose --env-file .env up -d
```

## 故障排查

### 容器无法启动

```bash
# 查看容器日志
docker logs medfusion

# 查看详细错误
docker logs --tail 100 medfusion

# 检查容器状态
docker inspect medfusion
```

### 端口冲突

```bash
# 检查端口占用
lsof -i :8000

# 使用其他端口
docker run -p 8080:8000 medfusion/medfusion:latest
```

### GPU 不可用

```bash
# 检查 NVIDIA Docker 安装
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 检查 GPU 驱动
nvidia-smi

# 查看容器 GPU 使用情况
docker exec medfusion nvidia-smi
```

### 数据卷权限问题

```bash
# 修改目录权限
sudo chown -R 1000:1000 ./data ./outputs ./logs

# 或在容器内运行
docker exec -u root medfusion chown -R app:app /app/data
```

### 内存不足

```bash
# 限制容器内存
docker run --memory="16g" --memory-swap="16g" medfusion/medfusion:latest

# 查看容器资源使用
docker stats medfusion
```

## 最佳实践

### 1. 使用命名卷

```bash
# 创建命名卷
docker volume create medfusion-data
docker volume create medfusion-outputs

# 使用命名卷
docker run \
  -v medfusion-data:/app/data \
  -v medfusion-outputs:/app/outputs \
  medfusion/medfusion:latest
```

### 2. 限制资源使用

```bash
docker run \
  --cpus="4.0" \
  --memory="16g" \
  --memory-swap="16g" \
  medfusion/medfusion:latest
```

### 3. 使用 Docker Compose

推荐使用 Docker Compose 管理复杂配置。

### 4. 定期清理

```bash
# 清理未使用的镜像
docker image prune -a

# 清理未使用的卷
docker volume prune

# 清理所有未使用资源
docker system prune -a --volumes
```

### 5. 安全配置

```bash
# 使用非 root 用户运行
docker run --user 1000:1000 medfusion/medfusion:latest

# 只读根文件系统
docker run --read-only --tmpfs /tmp medfusion/medfusion:latest
```

## 生产环境部署

### 使用 Docker Compose

```yaml
# docker-compose.prod.yml
version: "3.8"

services:
  medfusion:
    image: medfusion/medfusion:latest
    restart: always
    ports:
      - "8000:8000"
    volumes:
      - medfusion-data:/app/data:ro
      - medfusion-outputs:/app/outputs
      - medfusion-logs:/app/logs
    environment:
      - MEDCORE_LOG_LEVEL=INFO
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  medfusion-data:
  medfusion-outputs:
  medfusion-logs:
```

```bash
# 启动生产环境
docker-compose -f docker-compose.prod.yml up -d
```

## 下一步

- [生产环境清单](15_production_checklist.md) - 部署前检查
- [性能优化](../../guides/performance_optimization.md) - 优化推理性能
- [监控和日志](../../guides/monitoring.md) - 生产环境监控

## 参考资源

详细的 Docker 部署指南请参考：
- [完整 Docker 指南](/home/yixian/Projects/med-ml/medfusion/docs/user-guides/DOCKER_GUIDE.md)
- [Docker 官方文档](https://docs.docker.com/)
- [NVIDIA Docker 文档](https://github.com/NVIDIA/nvidia-docker)
