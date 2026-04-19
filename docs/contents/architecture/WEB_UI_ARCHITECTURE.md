# MedFusion Web UI 架构设计规划

> 文档状态：**Beta**

**版本**: v0.3.0  
**更新日期**: 2026-02-20  
**状态**: 设计阶段 → 实施中

## 📋 目录

- [1. 概述](#1-概述)
- [2. 场景分析](#2-场景分析)
- [3. 架构方案](#3-架构方案)
- [4. 技术栈](#4-技术栈)
- [5. 目录结构](#5-目录结构)
- [6. 实施路线图](#6-实施路线图)
- [7. 潜在问题与优化](#7-潜在问题与优化)
- [8. 开发指南](#8-开发指南)

---

## 1. 概述

### 1.1 设计目标

MedFusion Web UI 旨在提供一个**易用、灵活、可扩展**的可视化界面，支持：

- ✅ **本地使用**：个人研究者在本地电脑上训练和管理模型
- ✅ **团队协作**：医院/研究机构多人共享使用
- ✅ **云服务**：未来支持 SaaS 部署

### 1.2 核心原则

1. **简单优先**：本地使用应该像 TensorBoard 一样简单（一个命令启动）
2. **渐进增强**：从简单的本地版本逐步扩展到企业版
3. **可选组件**：Web UI 是可选功能，不影响核心库使用
4. **零配置**：默认配置应该开箱即用

### 1.3 当前正式版实现方向

正式版当前不引入 Node 后端。

推荐底座是：

- 前端：`React + TypeScript + Vite`
- API/BFF：`FastAPI`
- 执行层：独立 Python worker / subprocess

也就是说：

```text
React Web App
  -> FastAPI API/BFF
    -> Metadata DB
    -> Python Training Worker
    -> Artifact Storage
    -> AI Adapter (future)
```

当前最重要的工程原则不是 SSR，而是：

- 训练不要直接跑在 Web 进程里
- runtime / config 才是执行真源
- 前端节点图不是执行真源，只是问题收敛和结构编辑前台
- 先把 Web API 层和训练执行层拆开，再考虑更大规模的队列与多 worker

---

## 2. 场景分析

### 2.1 场景 1：本地使用（个人研究者）

**用户画像：**
- 医学影像研究者、学生
- 在个人电脑上进行实验
- 不熟悉 Docker 和复杂部署

**需求：**
- 快速启动，无需复杂配置
- 可视化训练过程
- 管理实验和模型
- 不需要多用户支持

**技术要求：**
- 单进程运行
- 轻量级数据库（SQLite）
- 无需外部依赖

**使用方式：**
```bash
pip install medfusion[web]
medfusion start
# 自动打开浏览器 http://localhost:8000
```

---

### 2.2 场景 2：团队协作（医院/研究机构）

**用户画像：**
- 医院影像科、研究机构
- 5-20 人团队共享使用
- 需要统一管理数据和模型

**需求：**
- 多用户并发访问
- 数据持久化和备份
- 权限管理
- 资源隔离

**技术要求：**
- Docker 部署
- PostgreSQL 数据库
- Redis 任务队列
- 用户认证系统

**使用方式：**
```bash
docker-compose up -d
# 访问 http://your-server:8000
```

---

### 2.3 场景 3：云服务（SaaS）

**用户画像：**
- 云服务提供商
- 数百到数千用户
- 需要高可用和弹性扩展

**需求：**
- 微服务架构
- 负载均衡
- 自动扩缩容
- 监控和告警

**技术要求：**
- Kubernetes 部署
- 分布式数据库
- 对象存储（S3/OSS）
- API 网关

---

## 3. 架构方案

### 3.1 方案对比

#### 方案 A：集成架构（推荐用于 v0.3.0）✅

**架构图：**
```
┌─────────────────────────────────────┐
│     MedFusion Python 包              │
│                                      │
│  ┌────────────────────────────────┐ │
│  │  med_core/web/                 │ │
│  │  ├── app.py (FastAPI)          │ │
│  │  ├── api/ (REST API)           │ │
│  │  └── static/ (前端构建产物)    │ │
│  └────────────────────────────────┘ │
│                                      │
│  ┌────────────────────────────────┐ │
│  │  med_core/                     │ │
│  │  ├── models/                   │ │
│  │  ├── trainers/                 │ │
│  │  └── ...                       │ │
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘
```

**优势：**
- ✅ 一个命令启动（`medfusion start`）
- ✅ 不需要 Docker
- ✅ 不需要单独的后端进程
- ✅ 前端打包成静态文件，嵌入 Python 包
- ✅ 类似 TensorBoard 的用户体验

**劣势：**
- ⚠️ 前端修改需要重新构建
- ⚠️ Python 包体积增大（约 5-10MB）

**适用场景：**
- 本地使用（个人研究者）
- 快速原型验证
- 教学和演示

---

#### 方案 B：分离架构（用于开发）

**架构图：**
```
┌──────────────────┐      ┌──────────────────┐
│  前端开发服务器   │      │  后端 API 服务    │
│  (Vite)          │◄────►│  (FastAPI)       │
│  localhost:5173  │      │  localhost:8000  │
└──────────────────┘      └──────────────────┘
```

**优势：**
- ✅ 前端开发体验好（热重载）
- ✅ 前后端独立开发
- ✅ 便于调试

**劣势：**
- ❌ 需要启动两个进程
- ❌ 用户体验复杂

**适用场景：**
- 前端开发阶段
- 调试和测试

---

### 3.2 推荐方案：混合架构

**策略：**
- **开发时**：使用分离架构（方案 B）
- **发布时**：使用集成架构（方案 A）
- **执行时**：继续拆分 `API/BFF` 和 `Python worker`，避免把真实训练绑死在 Web 进程里

**实现：**
```bash
# 开发模式
npm run dev          # 前端开发服务器（5173）
uv run uvicorn med_core.web.app:app --reload --host 127.0.0.1 --port 8000

# 生产模式
npm run build        # 构建前端 → dist/
cp -r dist/* med_core/web/static/
uv run medfusion start  # 启动集成服务器（8000）
```

---

## 4. 技术栈

### 4.1 后端技术栈

| 组件 | 技术选型 | 版本 | 说明 |
|------|---------|------|------|
| Web 框架 | FastAPI | 0.104+ | 高性能异步框架 |
| ASGI 服务器 | Uvicorn | 0.24+ | 生产级服务器 |
| 数据库 | SQLite | 3.35+ | 本地使用（默认） |
| 数据库 | PostgreSQL | 15+ | 团队使用（可选） |
| ORM | SQLAlchemy | 2.0+ | 数据库抽象层 |
| 缓存 | Redis | 7+ | 任务队列（可选） |
| 任务队列 | Celery | 5.3+ | 后台任务（可选） |

### 4.2 前端技术栈

| 组件 | 技术选型 | 版本 | 说明 |
|------|---------|------|------|
| 框架 | React | 18+ | UI 框架 |
| 语言 | TypeScript | 5+ | 类型安全 |
| 构建工具 | Vite | 5+ | 快速构建 |
| UI 库 | Ant Design | 5+ | 企业级组件 |
| 图表库 | ECharts | 5+ | 数据可视化 |
| 路由 | React Router | 6+ | 前端路由 |
| 状态管理 | Zustand | 4+ | 轻量级状态管理 |
| HTTP 客户端 | Axios | 1.6+ | API 请求 |
| WebSocket / SSE | 原生 WebSocket / SSE | - | 训练状态与日志流 |

### 4.4 后端分层建议

当前正式版建议按 4 层理解后端：

1. `Core Runtime`
   `med_core` 内的训练、构模、配置、结果构建
2. `Application Layer`
   问题向导、模板推荐、图编译、run 管理、artifact 管理、后续 AI 助手
3. `API/BFF Layer`
   FastAPI，对前端暴露 REST / WebSocket
4. `Worker Layer`
   独立 Python 训练进程，不在 Web 进程里直接执行长任务

当前仓库已经开始往这个方向收口：

- `med_core/web/api/` 负责前端 API/BFF
- `med_core/web/application/` 负责训练运行时与业务编排
- `med_core/web/workers/` 负责本地 worker / subprocess 生命周期
- 高级模式图编译也开始下沉到 API/BFF：`/api/advanced-builder/compile`
- 高级模式当前还能返回 `ExperimentConfig` contract 校验结果，而不是只停在前端草案

### 4.3 开发工具

| 工具 | 用途 |
|------|------|
| uv | Python 包管理 |
| npm | 前端包管理 |
| Ruff | Python 代码检查 |
| ESLint | TypeScript 代码检查 |
| Prettier | 代码格式化 |
| pytest | Python 测试 |
| Vitest | 前端测试 |

---

## 5. 目录结构

### 5.1 当前结构（v0.3.0）

```
medfusion/
├── med_core/                    # 核心库
│   ├── models/                  # 模型定义
│   ├── trainers/                # 训练器
│   ├── datasets/                # 数据集
│   └── web/                     # Web 模块（集成）✅
│       ├── __init__.py
│       ├── app.py               # FastAPI 应用
│       ├── cli.py               # CLI 命令
│       ├── config.py            # 配置管理
│       ├── database.py          # 数据库连接
│       ├── api/                 # API 路由
│       │   ├── __init__.py
│       │   ├── training.py      # 训练 API
│       │   ├── models.py        # 模型 API
│       │   ├── datasets.py      # 数据集 API
│       │   ├── experiments.py   # 实验 API
│       │   └── system.py        # 系统 API
│       ├── models/              # 数据库模型
│       │   ├── __init__.py
│       │   ├── training.py
│       │   ├── model.py
│       │   └── experiment.py
│       ├── services/            # 业务逻辑
│       │   ├── __init__.py
│       │   ├── training.py
│       │   ├── model.py
│       │   └── system.py
│       └── static/              # 前端构建产物 ✅
│           ├── index.html
│           └── assets/
│               ├── index-*.js
│               └── index-*.css
│
├── web/                         # Web 开发目录
│   ├── frontend/                # 前端源码
│   │   ├── src/
│   │   │   ├── pages/           # 页面组件
│   │   │   ├── components/      # 可复用组件
│   │   │   ├── api/             # API 客户端
│   │   │   ├── utils/           # 工具函数
│   │   │   ├── hooks/           # React Hooks
│   │   │   ├── stores/          # 状态管理
│   │   │   └── App.tsx
│   │   ├── package.json
│   │   ├── vite.config.ts
│   │   └── tsconfig.json
│   │
│   └── backend/                 # 旧的独立后端（待清理）⚠️
│       └── app/
│
├── docs/                        # 文档站与架构文档
│   ├── contents/getting-started/web-ui.md
│   ├── contents/architecture/WEB_UI_ARCHITECTURE.md
│   └── ...
│
├── start-webui.sh               # 启动脚本 ✅
└── pyproject.toml               # 项目配置
```

### 5.2 需要清理的内容

**待删除：**
```
web/backend/                     # 旧的独立后端
web/start-webui.sh               # 旧的启动脚本
web/stop-webui.sh                # 旧的停止脚本
```

**原因：**
- 已经迁移到 `med_core/web/`
- 避免混淆
- 简化项目结构

---

## 6. 实施路线图

### 6.1 阶段 1：v0.3.0（当前）- 本地使用优先 ✅

**目标：**
- 提供简单易用的本地 Web UI
- 类似 TensorBoard 的用户体验
- 不需要 Docker 和复杂配置

**已完成：**
- ✅ 前端构建和部署
- ✅ 后端 API 实现
- ✅ 静态文件服务
- ✅ 一键启动脚本
- ✅ 基础文档

**待完成：**
- [ ] 清理旧的 `web/backend/` 目录
- [ ] 继续收敛和精简 Web 启动命令面（默认保持 `medfusion start`）
- [ ] 添加版本检查
- [ ] 优化首次启动体验
- [ ] 添加数据管理命令

**技术实现：**
```python
# med_core/web/app.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="MedFusion Web UI", version="0.3.0")

# API 路由
app.include_router(training_router, prefix="/api/training")
app.include_router(models_router, prefix="/api/models")
app.include_router(datasets_router, prefix="/api/datasets")
app.include_router(system_router, prefix="/api/system")

# 静态文件服务
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
else:
    @app.get("/")
    async def root():
        return {"message": "前端资源未构建，请运行: npm run build"}
```

**CLI 命令：**
```python
# med_core/web/cli.py
import click
import uvicorn

@click.group()
def web():
    """Web UI 管理命令"""
    pass

@web.command()
@click.option("--host", default="127.0.0.1", help="监听地址")
@click.option("--port", default=8000, help="监听端口")
@click.option("--reload", is_flag=True, help="开发模式（热重载）")
def start(host, port, reload):
    """启动 Web UI"""
    click.echo(f"🚀 启动 MedFusion Web UI")
    click.echo(f"   访问地址: http://{host}:{port}")
    click.echo(f"   API 文档: http://{host}:{port}/docs")
    
    uvicorn.run(
        "med_core.web.app:app",
        host=host,
        port=port,
        reload=reload
    )

@web.command()
def info():
    """显示 Web UI 信息"""
    from med_core.web.config import get_data_dir
    data_dir = get_data_dir()
    
    click.echo("📊 MedFusion Web UI 信息")
    click.echo(f"   数据目录: {data_dir}")
    click.echo(f"   数据库: {data_dir / 'medfusion.db'}")
    click.echo(f"   日志目录: {data_dir / 'logs'}")
```

---

### 6.2 阶段 2：v0.4.0 - 支持团队部署

**目标：**
- 支持多用户并发访问
- 添加用户认证和权限管理
- 提供 Docker 部署方案

**计划功能：**
- [ ] PostgreSQL 支持
- [ ] Redis 任务队列
- [ ] 用户认证系统（JWT）
- [ ] 权限管理（RBAC）
- [ ] Docker 镜像
- [ ] docker-compose 配置
- [ ] 数据备份和恢复

**技术实现：**
```yaml
# docker-compose.yml
version: '3.8'

services:
  medfusion:
    image: medfusion:0.4.0
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data
      - ./models:/models
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/medfusion
      - REDIS_URL=redis://redis:6379
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    volumes:
      - postgres-data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=medfusion
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data

volumes:
  postgres-data:
  redis-data:
```

---

### 6.3 阶段 3：v1.0.0 - 企业版和云服务

**目标：**
- 支持大规模部署
- 提供 SaaS 服务
- 完善的监控和告警

**计划功能：**
- [ ] Kubernetes 部署
- [ ] 微服务架构
- [ ] 对象存储集成（S3/OSS）
- [ ] 分布式训练支持
- [ ] 监控和告警（Prometheus + Grafana）
- [ ] 日志聚合（ELK）
- [ ] API 网关
- [ ] 负载均衡

---

## 7. 潜在问题与优化

### 7.1 技术架构问题

#### 问题 1：前端静态文件打包到 Python 包

**风险：**
- Python 包体积膨胀（React 构建产物约 5-10MB）
- PyPI 上传限制（单个文件 100MB，总包 60MB 建议）
- 用户安装时间变长

**优化方案：**
```python
# 方案 A: 按需下载（推荐）
@click.command()
def start():
    static_dir = Path(__file__).parent / "static"
    if not static_dir.exists():
        click.echo("⏳ 首次运行，正在下载前端资源...")
        download_frontend_assets()
    
    # 启动服务器
    uvicorn.run(...)

def download_frontend_assets():
    """从 GitHub Releases 下载前端资源"""
    url = "https://github.com/medfusion/releases/download/v0.3.0/web-ui.tar.gz"
    # 下载并解压到 static/
```

```toml
# 方案 B: 可选依赖
[project.optional-dependencies]
web = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
]
web-full = [
    "medfusion[web]",
    "medfusion-web-ui>=0.3.0",  # 单独的前端包
]
```

---

#### 问题 2：SQLite 并发性能

**风险：**
- SQLite 不支持高并发写入
- 多用户同时训练可能冲突

**优化方案：**
```python
# 自动检测并建议升级
from med_core.web.database import get_db_stats

@app.on_event("startup")
async def check_database():
    stats = get_db_stats()
    if stats["concurrent_users"] > 5:
        logger.warning(
            "检测到多用户使用，建议升级到 PostgreSQL\n"
            "运行: uv run uvicorn med_core.web.app:app --host 0.0.0.0 --port 8000"
        )
```

---

#### 问题 3：长时间训练任务阻塞

**风险：**
- FastAPI 同步调用训练会阻塞其他请求
- 用户关闭浏览器训练中断

**优化方案：**
```python
# 使用后台任务
from fastapi import BackgroundTasks
from concurrent.futures import ProcessPoolExecutor

executor = ProcessPoolExecutor(max_workers=4)

@app.post("/api/training/start")
async def start_training(config: dict, background_tasks: BackgroundTasks):
    job_id = generate_job_id()
    
    # 方案 A: BackgroundTasks（轻量级）
    background_tasks.add_task(run_training, job_id, config)
    
    # 方案 B: ProcessPoolExecutor（隔离性更好）
    executor.submit(run_training, job_id, config)
    
    return {"job_id": job_id, "status": "queued"}
```

---

### 7.2 用户体验问题

#### 问题 4：首次启动慢

**风险：**
- 需要初始化数据库
- 需要下载前端资源（如果采用按需下载）
- 用户可能以为卡住了

**优化方案：**
```python
# 添加进度提示
@click.command()
def start():
    with click.progressbar(
        length=100,
        label="初始化 MedFusion Web UI"
    ) as bar:
        # 检查数据库
        bar.update(20)
        init_database()
        
        # 检查前端资源
        bar.update(40)
        if not check_frontend():
            download_frontend_assets()
        
        # 启动服务器
        bar.update(40)
    
    click.echo("✅ 启动成功！")
    click.echo(f"🌐 访问: http://localhost:8000")
```

---

#### 问题 5：端口冲突

**风险：**
- 默认端口 8000 可能被占用
- 用户不知道如何修改

**优化方案：**
```python
import socket

def find_free_port(start_port=8000, max_attempts=100):
    """查找可用端口"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError("无法找到可用端口")

@click.command()
@click.option("--port", default=None, type=int)
def start(port):
    if port is None:
        port = find_free_port()
        if port != 8000:
            click.echo(f"⚠️  端口 8000 被占用，使用端口 {port}")
    
    uvicorn.run(..., port=port)
```

---

#### 问题 6：数据持久化位置不明确

**风险：**
- 用户不知道数据存在哪里
- 卸载时可能丢失数据

**优化方案：**
```python
# 明确的数据目录
from pathlib import Path

def get_data_dir() -> Path:
    """获取数据目录"""
    data_dir = Path.home() / ".medfusion"
    data_dir.mkdir(exist_ok=True)
    return data_dir

# 数据目录结构
# ~/.medfusion/
# ├── medfusion.db          # 数据库
# ├── models/               # 模型文件
# ├── experiments/          # 实验记录
# ├── logs/                 # 日志
# └── web-ui/              # 前端资源

# 提供管理命令
@click.command()
def info():
    """显示数据信息"""
    data_dir = get_data_dir()
    size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
    
    click.echo(f"📊 数据目录: {data_dir}")
    click.echo(f"📦 总大小: {size / 1024 / 1024:.2f} MB")
    click.echo(f"📁 子目录:")
    for subdir in ["models", "experiments", "logs"]:
        path = data_dir / subdir
        if path.exists():
            count = len(list(path.iterdir()))
            click.echo(f"   - {subdir}: {count} 项")

@click.command()
@click.option("--output", type=click.Path(), required=True)
def backup(output):
    """备份数据"""
    import shutil
    data_dir = get_data_dir()
    shutil.make_archive(output, "gztar", data_dir)
    click.echo(f"✅ 备份完成: {output}.tar.gz")
```

---

### 7.3 安全性问题

#### 问题 7：无认证机制

**风险：**
- 本地启动后任何人都能访问
- 局域网内其他人可能误操作

**优化方案：**
```python
# 默认只监听 localhost
@click.command()
@click.option("--host", default="127.0.0.1")
@click.option("--token", default=None, help="访问令牌")
def start(host, token):
    if host != "127.0.0.1" and token is None:
        click.echo("⚠️  警告: 监听公网地址但未设置令牌")
        if not click.confirm("是否继续？"):
            return
    
    if token:
        # 添加简单的 token 认证
        from fastapi import Security, HTTPException
        from fastapi.security import HTTPBearer
        
        security = HTTPBearer()
        
        async def verify_token(credentials = Security(security)):
            if credentials.credentials != token:
                raise HTTPException(status_code=401)
        
        # 应用到所有路由
        app.dependency_overrides[verify_token] = verify_token
```

---

#### 问题 8：文件上传安全

**风险：**
- 用户上传恶意文件
- 路径遍历攻击

**优化方案：**
```python
from pathlib import Path
import magic

# 严格的文件类型检查
ALLOWED_EXTENSIONS = {'.jpg', '.png', '.dcm', '.nii', '.csv', '.yaml'}
ALLOWED_MIME_TYPES = {
    'image/jpeg', 'image/png', 'application/dicom',
    'application/x-nifti', 'text/csv', 'text/yaml'
}

# 文件大小限制
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

@app.post("/api/upload")
async def upload_file(file: UploadFile):
    # 检查文件扩展名
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件类型: {ext}")
    
    # 检查 MIME 类型
    content = await file.read(1024)
    mime_type = magic.from_buffer(content, mime=True)
    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(400, f"不支持的 MIME 类型: {mime_type}")
    
    # 检查文件大小
    file.file.seek(0, 2)
    size = file.file.tell()
    if size > MAX_FILE_SIZE:
        raise HTTPException(400, f"文件过大: {size / 1024 / 1024:.2f} MB")
    
    # 安全的文件名
    safe_filename = secure_filename(file.filename)
    
    # 隔离的上传目录
    upload_dir = get_data_dir() / "uploads" / generate_uuid()
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存文件
    file_path = upload_dir / safe_filename
    with open(file_path, "wb") as f:
        file.file.seek(0)
        shutil.copyfileobj(file.file, f)
    
    return {"file_id": str(upload_dir.name), "filename": safe_filename}
```

---

### 7.4 性能问题

#### 问题 9：大文件上传慢

**风险：**
- 医学影像文件通常很大（几百 MB）
- 上传超时
- 内存占用过高

**优化方案：**
```python
# 分块上传
@app.post("/api/upload/chunk")
async def upload_chunk(
    file_id: str,
    chunk_index: int,
    total_chunks: int,
    chunk: UploadFile
):
    """分块上传"""
    chunk_dir = get_data_dir() / "uploads" / "chunks" / file_id
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存分块
    chunk_path = chunk_dir / f"chunk_{chunk_index}"
    with open(chunk_path, "wb") as f:
        shutil.copyfileobj(chunk.file, f)
    
    # 检查是否所有分块都已上传
    if chunk_index == total_chunks - 1:
        # 合并分块
        output_path = get_data_dir() / "uploads" / file_id
        with open(output_path, "wb") as outfile:
            for i in range(total_chunks):
                chunk_path = chunk_dir / f"chunk_{i}"
                with open(chunk_path, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)
        
        # 清理分块
        shutil.rmtree(chunk_dir)
        
        return {"status": "completed", "file_id": file_id}
    
    return {"status": "uploading", "progress": (chunk_index + 1) / total_chunks}

# 流式上传
@app.post("/api/upload/stream")
async def upload_stream(request: Request, filename: str):
    """流式上传（不占用内存）"""
    file_id = generate_uuid()
    file_path = get_data_dir() / "uploads" / file_id
    
    with open(file_path, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
    
    return {"file_id": file_id, "filename": filename}
```

---

#### 问题 10：实时监控性能开销

**风险：**
- WebSocket 连接过多
- 频繁推送数据消耗资源
- 影响训练性能

**优化方案：**
```python
# 限制推送频率
import time
from collections import defaultdict

last_update_time = defaultdict(float)
MIN_UPDATE_INTERVAL = 1.0  # 最多每秒更新一次

async def send_metrics(websocket, job_id, metrics):
    """限流的指标推送"""
    current_time = time.time()
    if current_time - last_update_time[job_id] < MIN_UPDATE_INTERVAL:
        return  # 跳过本次推送
    
    last_update_time[job_id] = current_time
    await websocket.send_json(metrics)

# 数据采样
def should_send_update(step: int, total_steps: int) -> bool:
    """智能采样：早期密集，后期稀疏"""
    if step < 100:
        return step % 10 == 0  # 每 10 步
    elif step < 1000:
        return step % 50 == 0  # 每 50 步
    else:
        return step % 100 == 0  # 每 100 步

# 连接数限制
from fastapi import WebSocket, WebSocketDisconnect

MAX_WEBSOCKET_CONNECTIONS = 100
active_connections = set()

@app.websocket("/ws/training/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    if len(active_connections) >= MAX_WEBSOCKET_CONNECTIONS:
        await websocket.close(code=1008, reason="连接数已达上限")
        return
    
    await websocket.accept()
    active_connections.add(websocket)
    
    try:
        # 处理消息
        pass
    finally:
        active_connections.remove(websocket)
```

---

### 7.5 兼容性问题

#### 问题 11：Python 版本兼容

**风险：**
- FastAPI 需要 Python 3.8+
- 某些医院可能使用旧版本 Python

**优化方案：**
```toml
# pyproject.toml
[project]
requires-python = ">=3.8"

[project.optional-dependencies]
web = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "sqlalchemy>=2.0.0",
]
```

```python
# 运行时检查
import sys

if sys.version_info < (3, 8):
    raise RuntimeError(
        "MedFusion Web UI 需要 Python 3.8 或更高版本\n"
        f"当前版本: {sys.version_info.major}.{sys.version_info.minor}"
    )
```

---

#### 问题 12：浏览器兼容性

**风险：**
- 某些医院使用旧版 IE 浏览器
- 现代 JS 特性不支持

**优化方案：**
```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import legacy from '@vitejs/plugin-legacy';

export default defineConfig({
  plugins: [
    react(),
    legacy({
      targets: ['defaults', 'not IE 11'],
      additionalLegacyPolyfills: ['regenerator-runtime/runtime']
    })
  ],
  build: {
    target: 'es2015',
    cssTarget: 'chrome80',
  }
});
```

```typescript
// 浏览器检测
const checkBrowser = () => {
  const ua = navigator.userAgent;
  
  // 检测 IE
  if (ua.indexOf('MSIE') !== -1 || ua.indexOf('Trident/') !== -1) {
    alert('不支持 Internet Explorer，请使用现代浏览器（Chrome、Firefox、Safari、Edge）');
    return false;
  }
  
  // 检测必要特性
  if (!window.fetch || !window.Promise || !window.WebSocket) {
    alert('浏览器版本过旧，请升级到最新版本');
    return false;
  }
  
  return true;
};

// 应用启动时检查
if (!checkBrowser()) {
  document.body.innerHTML = '<h1>浏览器不兼容</h1><p>请使用 Chrome 90+, Firefox 88+, Safari 14+, Edge 90+</p>';
}
```

---

### 7.6 可维护性问题

#### 问题 13：前后端版本不匹配

**风险：**
- 后端更新但前端未更新
- API 不兼容导致错误

**优化方案：**
```python
# 版本检查 API
@app.get("/api/version")
async def get_version():
    return {
        "backend": "0.3.0",
        "frontend": "0.3.0",
        "api": "v1",
        "min_frontend_version": "0.3.0"
    }
```

```typescript
// 前端版本检查
const APP_VERSION = '0.3.0';

async function checkVersion() {
  try {
    const response = await fetch('/api/version');
    const { backend, min_frontend_version } = await response.json();
    
    if (APP_VERSION < min_frontend_version) {
      console.error('前端版本过旧，请刷新页面');
      showUpdateNotification();
    }
    
    if (APP_VERSION !== backend) {
      console.warn(`版本不匹配: 前端 ${APP_VERSION}, 后端 ${backend}`);
    }
  } catch (error) {
    console.error('版本检查失败', error);
  }
}

// 应用启动时检查
checkVersion();
```

---

#### 问题 14：日志管理混乱

**风险：**
- 训练日志、Web 日志、系统日志混在一起
- 难以排查问题

**优化方案：**
```python
# 分离的日志配置
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging():
    """配置日志系统"""
    log_dir = get_data_dir() / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Web 服务日志
    web_handler = RotatingFileHandler(
        log_dir / "web.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    web_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    web_logger = logging.getLogger("med_core.web")
    web_logger.addHandler(web_handler)
    web_logger.setLevel(logging.INFO)
    
    # 训练日志（每个任务单独文件）
    training_logger = logging.getLogger("med_core.training")
    training_logger.setLevel(logging.INFO)
    
    # 系统日志
    system_handler = RotatingFileHandler(
        log_dir / "system.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    system_logger = logging.getLogger("med_core")
    system_logger.addHandler(system_handler)
    system_logger.setLevel(logging.WARNING)

# 日志目录结构
# ~/.medfusion/logs/
# ├── web.log           # Web 服务日志
# ├── web.log.1         # 轮转备份
# ├── training/         # 训练日志
# │   ├── job_001.log
# │   └── job_002.log
# └── system.log        # 系统日志
```

---

### 7.7 扩展性问题

#### 问题 15：插件系统缺失

**风险：**
- 用户无法添加自定义节点
- 功能扩展困难

**优化方案：**
```python
# 插件接口
from abc import ABC, abstractmethod
from typing import Dict, Any

class NodePlugin(ABC):
    """节点插件基类"""
    name: str
    category: str
    description: str
    
    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行节点逻辑"""
        pass
    
    def validate(self, inputs: Dict[str, Any]) -> bool:
        """验证输入"""
        return True

# 插件注册表
NODE_REGISTRY: Dict[str, type[NodePlugin]] = {}

def register_node(name: str, category: str = "custom"):
    """注册节点装饰器"""
    def decorator(cls):
        NODE_REGISTRY[name] = cls
        cls.name = name
        cls.category = category
        return cls
    return decorator

# 用户自定义节点
@register_node("my_custom_preprocessing", category="preprocessing")
class MyCustomPreprocessing(NodePlugin):
    description = "自定义预处理节点"
    
    def execute(self, inputs):
        image = inputs["image"]
        # 自定义处理逻辑
        processed = custom_process(image)
        return {"output": processed}

# 加载插件
def load_plugins(plugin_dir: Path):
    """从目录加载插件"""
    import importlib.util
    
    for plugin_file in plugin_dir.glob("*.py"):
        spec = importlib.util.spec_from_file_location(
            plugin_file.stem, plugin_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    
    return NODE_REGISTRY
```

---

#### 问题 16：API 版本管理

**风险：**
- API 变更破坏兼容性
- 旧客户端无法使用

**优化方案：**
```python
# API 版本控制
from fastapi import APIRouter

# v1 API
v1_router = APIRouter(prefix="/api/v1", tags=["v1"])

@v1_router.get("/models")
async def list_models_v1():
    """v1 版本的模型列表"""
    return {"models": [...]}

# v2 API（新增字段）
v2_router = APIRouter(prefix="/api/v2", tags=["v2"])

@v2_router.get("/models")
async def list_models_v2():
    """v2 版本的模型列表（包含更多信息）"""
    return {
        "models": [...],
        "total": 10,
        "page": 1
    }

# 注册路由
app.include_router(v1_router)
app.include_router(v2_router)

# 默认使用最新版本
app.include_router(v2_router, prefix="/api")

# 废弃警告
from functools import wraps
import warnings

def deprecated(version: str, alternative: str):
    """标记 API 为废弃"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            warnings.warn(
                f"此 API 将在 {version} 版本移除，请使用 {alternative}",
                DeprecationWarning
            )
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@v1_router.get("/old-endpoint")
@deprecated(version="0.4.0", alternative="/api/v2/new-endpoint")
async def old_endpoint():
    pass
```

---

## 8. 开发指南

### 8.1 环境搭建

#### 后端开发环境

```bash
# 1. 克隆项目
git clone https://github.com/your-org/medfusion.git
cd medfusion

# 2. 创建虚拟环境
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. 安装依赖（包含 Web 模块）
uv pip install -e ".[web,dev]"

# 4. 启动开发服务器
uv run uvicorn med_core.web.app:app --reload --host 127.0.0.1 --port 8000
```

#### 前端开发环境

```bash
# 1. 进入前端目录
cd web/frontend

# 2. 安装依赖
npm install

# 3. 启动开发服务器
npm run dev
# 访问 http://localhost:5173
```

---

### 8.2 构建和部署

#### 构建前端

```bash
cd web/frontend
npm run build
# 输出到 dist/
```

#### 部署到后端

```bash
# 复制构建产物到后端静态目录
cp -r web/frontend/dist/* med_core/web/static/

# 或使用脚本
python scripts/deploy_frontend.py
```

#### 构建 Python 包

```bash
# 构建
python -m build

# 安装本地包
pip install dist/medfusion-0.3.0-py3-none-any.whl
```

---

### 8.3 测试

#### 后端测试

```bash
# 运行所有测试
pytest tests/

# 运行 Web 模块测试
pytest tests/web/

# 生成覆盖率报告
pytest --cov=med_core.web --cov-report=html
```

#### 前端测试

```bash
cd web/frontend

# 单元测试
npm run test

# E2E 测试
npm run test:e2e

# 覆盖率
npm run test:coverage
```

---

### 8.4 代码规范

#### Python 代码规范

```bash
# 代码检查
ruff check med_core/

# 代码格式化
ruff format med_core/

# 类型检查
mypy med_core/
```

#### TypeScript 代码规范

```bash
cd web/frontend

# 代码检查
npm run lint

# 代码格式化
npm run format

# 类型检查
npm run type-check
```

---

### 8.5 调试技巧

#### 后端调试

```python
# 使用 debugpy
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()

# 或使用 pdb
import pdb; pdb.set_trace()
```

#### 前端调试

```typescript
// 使用 React DevTools
// Chrome 扩展: React Developer Tools

// 使用 console
console.log('Debug info:', data);

// 使用 debugger
debugger;
```

---

## 9. 总结

### 9.1 核心决策

1. **采用集成架构**：前端打包到 Python 包，提供类似 TensorBoard 的体验
2. **渐进增强**：从简单的本地版本逐步扩展到企业版
3. **可选组件**：Web UI 不影响核心库使用
4. **零配置**：默认使用 SQLite，无需额外依赖

### 9.2 实施优先级

**v0.3.0（当前）：**
- ✅ 集成架构实现
- ✅ 基础 Web UI
- [ ] CLI 命令完善
- [ ] 文档完善

**v0.4.0（下一步）：**
- [ ] Docker 支持
- [ ] PostgreSQL 支持
- [ ] 用户认证
- [ ] 权限管理

**v1.0.0（长期）：**
- [ ] Kubernetes 部署
- [ ] 微服务架构
- [ ] 云服务支持

### 9.3 参考资源

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [React 文档](https://react.dev/)
- [Ant Design 文档](https://ant.design/)
- [ECharts 文档](https://echarts.apache.org/)
- [TensorBoard 设计](https://www.tensorflow.org/tensorboard)

---

**文档版本**: v0.3.0  
**最后更新**: 2026-02-20  
**维护者**: Medical AI Research Team  
**反馈**: [GitHub Issues](https://github.com/your-org/medfusion/issues)
