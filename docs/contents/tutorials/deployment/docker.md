# MedFusion Docker 部署指南

本文档对应当前仓库的真实 Docker 真源：

- `docker/Dockerfile`
- `docker/docker-compose.yml`

目标是提供 **私有服务器 / 自建部署模式**，并保持与本机模式同一套执行语义（FastAPI BFF + Python worker）。

## 前提

- Docker 20.10+
- Docker Compose v2+

## 先做 Dry-Run（不启动容器）

如果当前机器不能跑 Docker daemon，先执行：

```bash
uv run python scripts/release_smoke.py --mode docker-dry-run
```

这一步会校验：

1. `docker/Dockerfile` 关键片段是否齐全
2. `docker/docker-compose.yml` 是否能被解析
3. 是否包含 `medfusion-web / postgres / redis`
4. `medfusion-web` 是否声明数据库与队列关键环境变量
5. 各服务是否包含 healthcheck

## 构建镜像

```bash
docker build -f docker/Dockerfile -t medfusion/medfusion:0.4.0 .
```

## 使用 Compose 启动（推荐）

```bash
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml logs -f
```

停止：

```bash
docker compose -f docker/docker-compose.yml down
```

当前 compose 默认包含：

1. `medfusion-web`
2. `postgres`
3. `redis`

其中 `medfusion-web` 默认使用：

- `MEDFUSION_DATABASE_URL=postgresql://...@postgres:5432/medfusion`
- `MEDFUSION_REDIS_URL=redis://redis:6379/0`
- `MEDFUSION_TRAINING_QUEUE_BACKEND=redis`

## 认证配置（可选）

默认 `MEDFUSION_AUTH_ENABLED=false`。如果要在私有部署时上锁 API，建议通过 `.env` 提供：

```bash
MEDFUSION_AUTH_ENABLED=true
MEDFUSION_AUTH_USERNAME=admin
MEDFUSION_AUTH_PASSWORD=change-me
MEDFUSION_SECRET_KEY=please-use-a-long-random-secret
```

然后：

```bash
docker compose -f docker/docker-compose.yml --env-file .env up -d
```

## 健康检查

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/start
curl http://127.0.0.1:8000/evaluation
```

## 真实 Docker Smoke（可选）

当机器已具备 Docker runtime，再跑：

```bash
uv run python scripts/release_smoke.py --mode docker
```

## 常见问题

### 1) 当前机器没有 Docker

先使用 dry-run：

```bash
uv run python scripts/release_smoke.py --mode docker-dry-run
```

### 2) 数据库连接失败

确认 `medfusion-web` 与 `postgres` 在同一 compose 网络，且 `MEDFUSION_DATABASE_URL` 指向 `postgres` 服务名而不是 `localhost`。

### 3) 队列未启用 Redis

确认：

```bash
MEDFUSION_TRAINING_QUEUE_BACKEND=redis
MEDFUSION_REDIS_URL=redis://redis:6379/0
```

## 参考

- [正式版 Smoke Matrix](../../playbooks/release-smoke-matrix.md)
- [生产部署检查](production.md)
- [Web UI 快速入门](../../getting-started/web-ui.md)
