# MedFusion Docker 部署指南

本文档介绍如何使用 Docker 部署 MedFusion Web UI。

> 当前建议把 Docker 理解成 **私有服务器 / 自建部署模式** 的一种实现，而不是另一套产品。
> 前端仍然是 React 静态资源，API/BFF 仍然是 FastAPI，训练仍然建议由独立 Python worker 执行。

## 📋 目录

- [快速开始](#快速开始)
- [镜像构建](#镜像构建)
- [容器运行](#容器运行)
- [Docker Compose](#docker-compose)
- [数据卷管理](#数据卷管理)
- [环境变量](#环境变量)
- [多架构支持](#多架构支持)
- [故障排查](#故障排查)

## 🚀 快速开始

### 前提条件

- Docker 20.10+
- Docker Compose 2.0+（可选）
- NVIDIA Docker（GPU 支持，可选）

## 当前定位

Docker 形态当前最适合：

- 私有服务器 / 自建部署
- 团队内部评估
- 把 Web/API 和训练执行拆到更清楚的进程边界

当前最不建议把 Docker 讲成：

- 一套和本机版完全不同的产品
- 一套需要 Node 后端才能成立的架构

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

## 🏗️ 镜像构建

### 基础镜像构建

```bash
# 构建 CPU 版本
docker build -t medfusion/medfusion:latest .

# 构建并指定版本标签
docker build -t medfusion/medfusion:0.3.0 .
```

### GPU 版本构建

MedFusion 的 Dockerfile 会自动检测 PyTorch 的 CUDA 支持。如果需要显式构建 GPU 版本：

```bash
# 使用 CUDA 基础镜像（需要修改 Dockerfile）
docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 \
  -t medfusion/medfusion:latest-gpu .
```

### 多架构构建

```bash
# 使用 buildx 构建多架构镜像
docker buildx create --use
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t medfusion/medfusion:latest \
  --push .
```

## 🐳 容器运行

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
  -e MEDFUSION_LOG_LEVEL=INFO \
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

### 交互式运行（调试）

```bash
# 进入容器 shell
docker run -it \
  --name medfusion-debug \
  -v $(pwd)/data:/app/data \
  medfusion/medfusion:latest \
  /bin/bash

# 在容器内运行命令
medfusion --help
medfusion start --host 0.0.0.0 --port 8000
```

## 🎼 Docker Compose

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
medfusion start --reload
```

### 多服务编排

```yaml
# 自定义 docker-compose.override.yml
version: "3.8"

services:
  medfusion-web:
    environment:
      - MEDFUSION_USE_WANDB=true
      - WANDB_API_KEY=${WANDB_API_KEY}
    
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=medfusion
      - POSTGRES_USER=medfusion
      - POSTGRES_PASSWORD=secret
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

```bash
# 使用自定义配置启动
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

## 💾 数据卷管理

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

## 🔧 环境变量

### MedFusion 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MEDFUSION_DATA_DIR` | `/app/data` | MedFusion 数据目录 |
| `MEDFUSION_LOG_LEVEL` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR） |

### Web UI 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MEDFUSION_HOST` | `0.0.0.0` | 监听地址 |
| `MEDFUSION_PORT` | `8000` | 监听端口 |

### CUDA 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `CUDA_VISIBLE_DEVICES` | `0` | 可见的 GPU 设备 |
| `CUDA_LAUNCH_BLOCKING` | - | 同步 CUDA 调用（调试） |

### 使用 .env 文件

```bash
# 创建 .env 文件
cat > .env << EOF
MEDFUSION_LOG_LEVEL=DEBUG
MEDFUSION_HOST=0.0.0.0
MEDFUSION_PORT=8000
CUDA_VISIBLE_DEVICES=0,1
EOF

# 使用 .env 文件启动
docker-compose --env-file .env up -d
```

## 🌐 多架构支持

### 支持的平台

- `linux/amd64`（x86_64）
- `linux/arm64`（ARM64/Apple Silicon）

### 构建多架构镜像

```bash
# 创建 buildx builder
docker buildx create --name medfusion-builder --use

# 构建并推送多架构镜像
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t medfusion/medfusion:latest \
  --push .

# 查看镜像信息
docker buildx imagetools inspect medfusion/medfusion:latest
```

### Apple Silicon (M1/M2) 支持

```bash
# 在 Apple Silicon Mac 上运行
docker run -d \
  --name medfusion \
  --platform linux/arm64 \
  -p 8000:8000 \
  medfusion/medfusion:latest

# 使用 Rosetta 2 运行 x86_64 镜像（性能较差）
docker run -d \
  --name medfusion \
  --platform linux/amd64 \
  -p 8000:8000 \
  medfusion/medfusion:latest
```

## 🔍 故障排查

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

### 健康检查失败

```bash
# 手动测试健康检查
docker exec medfusion curl -f http://localhost:8000/health

# 禁用健康检查
docker run --no-healthcheck medfusion/medfusion:latest
```

## 📚 最佳实践

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

推荐使用 Docker Compose 管理复杂配置，而不是长命令行。

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

# 限制网络访问
docker run --network none medfusion/medfusion:latest
```

## 🔗 相关资源

- [Docker 官方文档](https://docs.docker.com/)
- [NVIDIA Docker 文档](https://github.com/NVIDIA/nvidia-docker)
- [Docker Compose 文档](https://docs.docker.com/compose/)
- [MedFusion 主文档](../README.md)
- [Web UI 快速入门](../../getting-started/web-ui.md)

## 📝 更新日志

- **2026-02-20**: 初始版本，支持 Web UI 一体化部署
- **v0.3.0**: 重写 Dockerfile 和 docker-compose.yml

---

**维护者**: Medical AI Research Team  
**最后更新**: 2026-02-20
