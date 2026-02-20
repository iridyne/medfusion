# Docker 快速参考

## 常用命令

### 构建和启动

```bash
# 构建镜像
docker-compose build

# 启动训练
docker-compose up medfusion-train

# 后台运行
docker-compose up -d medfusion-train

# 启动所有服务
docker-compose up
```

### 查看状态

```bash
# 查看运行中的容器
docker-compose ps

# 查看日志
docker-compose logs -f medfusion-train

# 查看资源使用
docker stats
```

### 停止和清理

```bash
# 停止服务
docker-compose stop

# 停止并删除容器
docker-compose down

# 删除所有（包括卷）
docker-compose down -v
```

### 交互式使用

```bash
# 进入运行中的容器
docker-compose exec medfusion-train /bin/bash

# 运行一次性命令
docker-compose run --rm medfusion-train python --version

# 启动开发环境
docker-compose --profile dev run --rm medfusion-dev /bin/bash
```

## 服务配置

### 训练服务

```bash
# 默认训练
docker-compose up medfusion-train

# 自定义配置
docker-compose run --rm medfusion-train \
    python -m med_core.cli train --config /app/configs/custom.yaml

# 指定 GPU
CUDA_VISIBLE_DEVICES=1 docker-compose up medfusion-train
```

### 评估服务

```bash
# 启动评估
docker-compose --profile eval up medfusion-eval

# 自定义评估
docker-compose run --rm medfusion-eval \
    python -m med_core.cli evaluate --checkpoint /app/outputs/model.pth
```

### TensorBoard

```bash
# 启动 TensorBoard
docker-compose --profile monitoring up tensorboard

# 访问: http://localhost:6006
```

### Jupyter

```bash
# 启动 Jupyter
docker-compose --profile dev up jupyter

# 访问: http://localhost:8888
```

## 环境变量

```bash
# 设置 GPU
CUDA_VISIBLE_DEVICES=0,1

# 日志级别
MEDCORE_LOG_LEVEL=DEBUG

# W&B 配置
MEDCORE_USE_WANDB=true
WANDB_API_KEY=your_key
```

## 卷挂载

```yaml
volumes:
  - ./data:/app/data:ro          # 数据（只读）
  - ./outputs:/app/outputs        # 输出
  - ./logs:/app/logs              # 日志
  - ./configs:/app/configs:ro     # 配置（只读）
```

## 故障排查

```bash
# 检查 GPU
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# 重建镜像
docker-compose build --no-cache

# 清理资源
docker system prune -a

# 查看详细日志
docker-compose logs --tail=100 medfusion-train
```

## 最佳实践

1. **使用 .env 文件**: 管理环境变量
2. **定期清理**: `docker system prune`
3. **使用卷**: 持久化数据和输出
4. **监控资源**: `docker stats`
5. **查看日志**: `docker-compose logs -f`

## 完整工作流示例

```bash
# 1. 构建镜像
docker-compose build

# 2. 准备数据
mkdir -p data outputs logs
cp -r /path/to/data/* data/

# 3. 运行训练
docker-compose up -d medfusion-train

# 4. 监控训练
docker-compose --profile monitoring up -d tensorboard
# 访问 http://localhost:6006

# 5. 查看日志
docker-compose logs -f medfusion-train

# 6. 训练完成后评估
docker-compose --profile eval run --rm medfusion-eval

# 7. 清理
docker-compose down
```
