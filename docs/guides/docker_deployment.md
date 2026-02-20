# Docker 部署指南

本指南介绍如何使用 Docker 部署和运行 MedFusion 框架。

## 目录

- [前置要求](#前置要求)
- [快速开始](#快速开始)
- [服务说明](#服务说明)
- [使用场景](#使用场景)
- [配置选项](#配置选项)
- [故障排查](#故障排查)

---

## 前置要求

### 必需

- Docker Engine 20.10+
- Docker Compose 2.0+

### 可选（GPU 支持）

- NVIDIA GPU
- NVIDIA Docker Runtime (nvidia-docker2)
- CUDA 11.0+

### 安装 NVIDIA Docker Runtime

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 验证安装
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

---

## 快速开始

### 1. 构建镜像

```bash
# 构建 MedFusion 镜像
docker-compose build

# 或使用 Docker 直接构建
docker build -t medfusion:latest .
```

### 2. 准备数据

```bash
# 创建必要的目录
mkdir -p data outputs logs notebooks

# 将数据放入 data 目录
cp -r /path/to/your/data/* data/
```

### 3. 运行训练

```bash
# 使用默认配置运行训练
docker-compose up medfusion-train

# 后台运行
docker-compose up -d medfusion-train

# 查看日志
docker-compose logs -f medfusion-train
```

---

## 服务说明

### medfusion-train

**用途**: 模型训练

**启动**:
```bash
docker-compose up medfusion-train
```

**自定义配置**:
```bash
# 使用自定义配置文件
docker-compose run --rm medfusion-train \
    python -m med_core.cli train --config /app/configs/medical_config.yaml
```

**环境变量**:
- `CUDA_VISIBLE_DEVICES`: GPU 设备 ID（默认: 0）
- `MEDCORE_LOG_LEVEL`: 日志级别（默认: INFO）
- `MEDCORE_USE_WANDB`: 是否使用 W&B（默认: false）

---

### medfusion-eval

**用途**: 模型评估

**启动**:
```bash
docker-compose --profile eval up medfusion-eval
```

**自定义评估**:
```bash
docker-compose run --rm medfusion-eval \
    python -m med_core.cli evaluate \
    --checkpoint /app/outputs/best_model.pth \
    --data-dir /app/data/test
```

---

### tensorboard

**用途**: 可视化训练过程

**启动**:
```bash
docker-compose --profile monitoring up tensorboard
```

**访问**: http://localhost:6006

---

### jupyter

**用途**: 交互式开发和实验

**启动**:
```bash
docker-compose --profile dev up jupyter
```

**访问**: http://localhost:8888

**特性**:
- JupyterLab 界面
- GPU 支持
- 预装 MedFusion
- 访问示例代码

---

### medfusion-dev

**用途**: 开发环境

**启动**:
```bash
docker-compose --profile dev up -d medfusion-dev
docker-compose exec medfusion-dev /bin/bash
```

**特性**:
- 代码热重载
- 完整开发工具
- 交互式 shell

---

## 使用场景

### 场景 1: 训练模型

```bash
# 1. 准备配置文件
cp configs/default.yaml configs/my_experiment.yaml
# 编辑 my_experiment.yaml

# 2. 运行训练
docker-compose run --rm medfusion-train \
    python -m med_core.cli train --config /app/configs/my_experiment.yaml

# 3. 监控训练（另一个终端）
docker-compose --profile monitoring up tensorboard
```

### 场景 2: 评估模型

```bash
# 评估最佳模型
docker-compose --profile eval run --rm medfusion-eval \
    python -m med_core.cli evaluate \
    --checkpoint /app/outputs/best_model.pth \
    --output-dir /app/evaluation_results
```

### 场景 3: 交互式开发

```bash
# 启动 Jupyter
docker-compose --profile dev up -d jupyter

# 访问 http://localhost:8888
# 在 notebooks/ 目录创建新笔记本
```

### 场景 4: 调试

```bash
# 启动开发容器
docker-compose --profile dev run --rm medfusion-dev /bin/bash

# 在容器内运行测试
pytest tests/

# 运行示例
python examples/train_demo.py
```

### 场景 5: 多 GPU 训练

```bash
# 修改 docker-compose.yml 中的 CUDA_VISIBLE_DEVICES
# 或使用环境变量
CUDA_VISIBLE_DEVICES=0,1,2,3 docker-compose up medfusion-train
```

---

## 配置选项

### 环境变量

在 `docker-compose.yml` 或 `.env` 文件中设置：

```yaml
environment:
  # GPU 配置
  - CUDA_VISIBLE_DEVICES=0,1
  
  # 日志配置
  - MEDCORE_LOG_LEVEL=DEBUG
  - MEDCORE_LOG_DIR=/app/logs
  
  # 数据配置
  - MEDCORE_DATA_DIR=/app/data
  - MEDCORE_OUTPUT_DIR=/app/outputs
  
  # W&B 配置
  - MEDCORE_USE_WANDB=true
  - WANDB_API_KEY=your_api_key
  - WANDB_PROJECT=medfusion
  
  # 性能配置
  - OMP_NUM_THREADS=4
  - MKL_NUM_THREADS=4
```

### 卷挂载

```yaml
volumes:
  # 只读数据
  - ./data:/app/data:ro
  
  # 读写输出
  - ./outputs:/app/outputs
  
  # 日志
  - ./logs:/app/logs
  
  # 自定义配置
  - ./configs:/app/configs:ro
  
  # 开发模式：挂载代码
  - ./med_core:/app/med_core
```

### 资源限制

```yaml
deploy:
  resources:
    limits:
      cpus: '8'
      memory: 32G
    reservations:
      cpus: '4'
      memory: 16G
      devices:
        - driver: nvidia
          count: 2
          capabilities: [gpu]
```

---

## 高级用法

### 自定义 Dockerfile

如果需要额外的依赖：

```dockerfile
# Dockerfile.custom
FROM medfusion:latest

# 安装额外的 Python 包
RUN pip install --no-cache-dir \
    optuna \
    mlflow \
    fastapi

# 安装系统包
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*
```

构建：
```bash
docker build -f Dockerfile.custom -t medfusion:custom .
```

### 多阶段训练

```bash
# 阶段 1: 预训练
docker-compose run --rm medfusion-train \
    python -m med_core.cli train \
    --config /app/configs/pretrain.yaml \
    --output-dir /app/outputs/pretrain

# 阶段 2: 微调
docker-compose run --rm medfusion-train \
    python -m med_core.cli train \
    --config /app/configs/finetune.yaml \
    --checkpoint /app/outputs/pretrain/best_model.pth \
    --output-dir /app/outputs/finetune
```

### 分布式训练

```bash
# 使用 torchrun
docker-compose run --rm medfusion-train \
    torchrun --nproc_per_node=4 \
    -m med_core.cli train \
    --config /app/configs/distributed.yaml
```

### 导出模型

```bash
# 导出为 ONNX
docker-compose run --rm medfusion-eval \
    python -m med_core.cli export \
    --checkpoint /app/outputs/best_model.pth \
    --format onnx \
    --output /app/outputs/model.onnx
```

---

## 故障排查

### 问题 1: GPU 不可用

**症状**:
```
RuntimeError: CUDA not available
```

**解决方案**:
```bash
# 检查 NVIDIA Docker 运行时
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# 确保 docker-compose.yml 中配置了 GPU
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### 问题 2: 权限错误

**症状**:
```
PermissionError: [Errno 13] Permission denied: '/app/outputs'
```

**解决方案**:
```bash
# 修改目录权限
chmod -R 777 outputs logs

# 或在 Dockerfile 中设置用户
USER 1000:1000
```

### 问题 3: 内存不足

**症状**:
```
RuntimeError: CUDA out of memory
```

**解决方案**:
```yaml
# 在配置文件中减小 batch size
training:
  batch_size: 8  # 从 32 减小到 8

# 或使用梯度累积
training:
  batch_size: 8
  gradient_accumulation_steps: 4
```

### 问题 4: 容器无法启动

**症状**:
```
Error response from daemon: failed to create shim
```

**解决方案**:
```bash
# 清理 Docker 资源
docker system prune -a

# 重启 Docker 服务
sudo systemctl restart docker

# 重新构建镜像
docker-compose build --no-cache
```

### 问题 5: 数据加载慢

**症状**: 训练速度慢，GPU 利用率低

**解决方案**:
```yaml
# 增加数据加载线程
data:
  num_workers: 8  # 增加 worker 数量
  pin_memory: true
  persistent_workers: true

# 使用更快的存储
volumes:
  - /fast/ssd/data:/app/data:ro  # 使用 SSD
```

---

## 生产部署

### 使用 Docker Swarm

```bash
# 初始化 Swarm
docker swarm init

# 部署服务栈
docker stack deploy -c docker-compose.yml medfusion

# 扩展服务
docker service scale medfusion_medfusion-train=3

# 查看服务状态
docker service ls
docker service ps medfusion_medfusion-train
```

### 使用 Kubernetes

```bash
# 转换 docker-compose 为 Kubernetes 配置
kompose convert -f docker-compose.yml

# 部署到 Kubernetes
kubectl apply -f medfusion-train-deployment.yaml
kubectl apply -f medfusion-train-service.yaml

# 查看状态
kubectl get pods
kubectl logs -f medfusion-train-xxxxx
```

---

## 最佳实践

### 1. 使用 .env 文件管理环境变量

```bash
# .env
CUDA_VISIBLE_DEVICES=0
MEDCORE_LOG_LEVEL=INFO
WANDB_API_KEY=your_key_here
```

### 2. 定期清理

```bash
# 清理未使用的镜像
docker image prune -a

# 清理未使用的卷
docker volume prune

# 完整清理
docker system prune -a --volumes
```

### 3. 使用健康检查

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import med_core; print('OK')"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 5s
```

### 4. 日志管理

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### 5. 安全性

```yaml
# 使用非 root 用户
user: "1000:1000"

# 只读文件系统
read_only: true

# 限制能力
cap_drop:
  - ALL
cap_add:
  - NET_BIND_SERVICE
```

---

## 参考资源

- [Docker 官方文档](https://docs.docker.com/)
- [Docker Compose 文档](https://docs.docker.com/compose/)
- [NVIDIA Docker 文档](https://github.com/NVIDIA/nvidia-docker)
- [MedFusion 主文档](../README.md)

---

## 支持

如有问题，请：
1. 查看[故障排查](#故障排查)部分
2. 检查 [GitHub Issues](https://github.com/your-org/medfusion/issues)
3. 提交新的 Issue

---

**最后更新**: 2026-02-20
