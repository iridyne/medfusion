# MedFusion Web UI 快速入门

> 文档状态：**Beta**

本文档介绍如何快速启动和使用 MedFusion Web UI。

## 📋 目录

- [功能特性](#功能特性)
- [快速启动](#快速启动)
- [访问界面](#访问界面)
- [主要功能](#主要功能)
- [开发指南](#开发指南)
- [故障排除](#故障排除)

## ✨ 功能特性

MedFusion Web UI 提供了一个现代化的 Web 界面，用于管理和监控深度学习训练任务：

- 🎯 **训练监控** - 实时查看训练进度、损失曲线、准确率等指标
- 📊 **模型管理** - 浏览、上传、下载和管理训练好的模型
- 🔄 **工作流编辑器** - 受限 preview，可在单条主线上编排 `data -> model -> training -> optional evaluation`
- 🧪 **独立评估** - 基于现有 `config + checkpoint` 单独补跑评估与结果构建
- 📁 **数据集管理** - 管理和预处理医学影像数据集
- 💻 **系统监控** - 查看 GPU、内存、CPU 使用情况
- 🌐 **RESTful API** - 完整的 API 支持，可用于自动化和集成

## 🚀 快速启动

如果你只看一页工作流，先看：

- [启动流程与使用工作流（统一口径）](startup-workflow.md)

### 方法 1：使用统一 CLI 入口（推荐）

```bash
# 推荐入口：先进入 Getting Started 引导页
uv run medfusion start

# 指定选项
uv run medfusion start --host 0.0.0.0 --port 8080 --reload
```

`medfusion start` 会把 Web UI 作为默认入口收口起来，更适合新用户和回归验证场景。

你也可以先做预检，再启动服务：

```bash
# 只做资源与端口预检，不启动服务
uv run medfusion start --check-only

# 本地版本一致性检查（CLI / 本地 Web 资源）
uv run medfusion version-check --skip-server --json

# 运行中服务版本检查（默认检查 http://127.0.0.1:8000）
uv run medfusion version-check
```

如果你在本机维护数据目录，当前也支持备份/恢复：

```bash
# 备份数据目录（默认包含数据库元数据快照）
uv run medfusion data backup medfusion-data-backup

# 只备份文件目录（不导出数据库快照）
uv run medfusion data backup medfusion-data-backup --without-db-snapshot

# 预演恢复，不落盘
uv run medfusion data restore medfusion-data-backup.tar.gz --dry-run

# 覆盖恢复（默认恢复文件 + 数据库元数据快照）
uv run medfusion data restore medfusion-data-backup.tar.gz --overwrite

# 仅恢复文件，跳过数据库快照
uv run medfusion data restore medfusion-data-backup.tar.gz --overwrite --skip-db
```

如果你在私有部署模式启用 PostgreSQL，建议在启动前先执行一次 schema 升级：

```bash
# 读取 MEDFUSION_DATABASE_URL 并升级到 head
uv run medfusion db upgrade

# 查看当前 revision
uv run medfusion db current
```

数据库 URL 支持以下写法（会统一归一化到 `postgresql+psycopg://`）：

- `MEDFUSION_DATABASE_URL=postgres://user:pass@host:5432/medfusion`
- `MEDFUSION_DATABASE_URL=postgresql://user:pass@host:5432/medfusion`

### 认证与权限（v0.4 主线）

默认本机模式仍然是不开认证。需要对 API 上锁时：

```bash
# 启用认证并自动生成静态 Bearer token（适合单机/快速联调）
uv run medfusion start --auth

# 或显式指定静态 token
uv run medfusion start --auth --token your-static-token
```

如果你希望使用 JWT 登录发 token：

```bash
set MEDFUSION_AUTH_ENABLED=true
set MEDFUSION_AUTH_USERNAME=admin
set MEDFUSION_AUTH_PASSWORD=change-me
uv run medfusion start --no-browser
```

说明：JWT 发 token 依赖 `PyJWT`，建议直接安装 `medfusion[web]`。

然后调用：

```bash
POST /api/auth/token
{
  "username": "admin",
  "password": "change-me"
}
```

RBAC 角色约束：

- `viewer`: 只读 API
- `operator`: 读写 API
- `admin`: 全量读写（含静态 token 默认角色）

### 训练队列后端（local / redis）

默认队列后端是 `local`（进程内调度）。如果要切换为 Redis 队列：

```bash
set MEDFUSION_TRAINING_QUEUE_BACKEND=redis
set MEDFUSION_REDIS_URL=redis://127.0.0.1:6379/0
set MEDFUSION_REDIS_QUEUE_NAME=medfusion:training:jobs
uv run medfusion start --no-browser
```

如果 Redis 不可用，服务会自动回退到本地调度并记录 warning，不会阻塞训练入口。

当前默认第一页不再是假设你已经熟悉所有页面的工作台首页，而是 `Getting Started` 引导页。
它的职责是：

1. 介绍正式版当前有哪些组件和页面职责
2. 解释默认模式与高级模式的边界
3. 把你带到问题向导与主线配置入口
4. 把你带到后续训练与结果页面

大多数新手只需要这一条命令。

当前 Getting Started 页后的推荐路径：

1. 先进入 `Run Wizard`，从问题定义出发拿到推荐骨架
   - 当前默认会先落到 ComfyUI 适配配置语义（仍在 MedFusion 主线内）
   - 这里会直接显示模型数据库 contract 的 `recommended preset / compile boundary / compile notes / patch target hints`
2. 继续到训练监控页启动一次带默认参数的推荐训练
3. 训练完成后执行 `medfusion build-results`，再到模型库查看 artifact
4. 如需理解高级边界，再进入 `/config/advanced` 查看组件注册表和连接约束
5. 需要 first-run 演示材料时，再进入 `Quickstart Run` 页面查看公开数据 profile、命令链和预期产物

如果你是发布前自检，建议直接跑统一脚本：

```bash
# 本机主链 smoke
uv run python scripts/release_smoke.py --mode local

# Docker 配置 dry-run（不启动容器）
uv run python scripts/release_smoke.py --mode docker-dry-run
```

### Web 入口与 YAML 主链的关系

这里需要把边界说清楚：

- `medfusion start` 是新手引导入口
- `YAML 主链` 仍然是长期实验、复现和模型装配的正式路径
- Web 不会发明第二套执行 runtime，只负责解释、检查和结果 handoff

- 当前向导更准确地说是一个 **RunSpec / ExperimentConfig 生成器**
- 它帮助你填写当前主链已经支持的配置
- 它不是一个无代码模型发明器

如果你要新建模型，请先看 [如何新建模型与 YAML](model-creation-paths.md)。
当前 Web 向导**不会替你发明一个全新的模型能力**，它只是帮你减少现有 schema 的手写成本。

高级或兼容场景再看下面几种方式。

### 当前推荐的部署形态

当前正式版默认按下面三种形态来理解，其中第一种是当前最推荐入口：

1. **本机浏览器模式**
   - React 构建产物由 FastAPI 提供
   - FastAPI 负责 API/BFF
   - 本地 Python subprocess worker 执行训练
   - SQLite + 本地 artifact 目录
2. **私有服务器 / 自建部署模式**
   - 静态前端与 FastAPI 可分开部署
   - FastAPI 继续作为 API/BFF
   - Python worker 独立部署到 GPU 主机
   - PostgreSQL + 对象存储 / 共享文件系统
3. **托管云模式**
   - 静态前端 + 网关 / CDN
   - FastAPI API/BFF
   - 多 Python worker
   - PostgreSQL + S3 / OSS / MinIO

关键点不变：

- 不引入 Node 后端
- 训练不要直接跑在 Web 进程里
- runtime / config 才是执行真源

### 方法 2：使用启动脚本（兼容）

最简单的方式是使用提供的启动脚本：

```bash
# 在项目根目录下
./start-webui.sh

# 指定端口和主机
./start-webui.sh 8080 0.0.0.0
```

脚本会自动：
1. 检查 Python 环境和依赖
2. 构建前端（如果需要）
3. 启动 Web 服务器

### 方法 3：手动启动（开发调试）

如果你想手动控制每个步骤：

```bash
# 1. 确保依赖已安装
uv pip install -e ".[web]"

# 2. 构建前端（首次运行或前端有更新时）
cd web/frontend
npm install
npm run build
cp -r dist/* ../../med_core/web/static/
cd ../..

# 3. 启动后端服务器
uv run uvicorn med_core.web.app:app --host 127.0.0.1 --port 8000
```

### 方法 4：兼容旧 CLI 入口

```bash
# 兼容旧入口
uv run medfusion web

# 指定选项
uv run medfusion web --host 0.0.0.0 --port 8080 --reload
```

## 🌐 访问界面

启动成功后，在浏览器中访问：

- **Web UI**: http://127.0.0.1:8000
- **API 文档**: http://127.0.0.1:8000/docs
- **健康检查**: http://127.0.0.1:8000/health

## 📱 主要功能

### OSS 主链页面范围（固定）

当前 OSS Web 默认主链固定为 7 个页面与 1 条可选引导子路径：

1. `/start`（Getting Started）
2. `/workbench`（运行后概览）
3. `/datasets`（数据集管理）
4. `/config`（RunSpec 向导）
5. `/training`（训练监控）
6. `/models`（模型与结果）
7. `/system`（系统监控）
8. `/evaluation`（独立评估）

引导子路径：

- `/quickstart-run`：承接第一次运行链路说明，不单独放入主导航，也不作为默认主线下一步

高级模式路径：

- `/config/advanced`：正式版高级模式的组件注册表与连接约束页，不作为默认首页，但用于承接后续节点式结构编辑
- `/config/advanced/canvas`：高级模式节点图入口，当前允许在正式版组件边界内做结构编辑、编译检查，并在 contract 校验通过后直接创建真实训练任务
- `/config/comfyui`：ComfyUI 集成入口，提供连通性检查、快速打开和主链回流提示
  - 入口方式：可直接从 `/start` 首页按钮进入，无需手动输入 URL
  - 页面内可直接返回 `/config` 或进入 `/training`，减少“停在桥接页”后的路径断点

当前进展：

1. 高级模式节点图现在可以编译出 RunSpec 草案
2. 后端会继续做正式 `ExperimentConfig` contract 校验
3. 当校验通过时，可以直接从高级模式创建真实训练任务
4. 训练完成后，任务状态会附带结果 handoff 信息，并可继续跳到模型库查看结果
5. ComfyUI 已有独立上线入口页，可先检查连通性再进入外部画布联调
6. ComfyUI 入口页支持把 `config/checkpoint/output` 参数预填后，一键跳到结果后台导入弹窗
7. ComfyUI 入口页支持选择“适配档案”，并一键跳到对应的 MedFusion 高级模式组件骨架画布
8. ComfyUI 入口页支持一键“带推荐参数进入训练监控”，减少手工回填
9. Run Wizard 导出动作支持一键“带向导参数进入训练监控”，并在训练页显示来源提示
10. ComfyUI 入口页支持“带预填回到配置向导”，用于先微调配置再进入训练
11. ComfyUI 入口页顶部提供“主线步骤提示 + 配置/训练/结果快捷跳转”
12. 结果后台支持“基于当前结果重开配置”，直接进入下一轮主线迭代
13. 从结果后台重开配置后，训练页会显示 `model-library` 来源提示，方便区分迭代链路
14. 结果后台支持“基于当前结果直接重跑训练”，用于快速迭代验证
15. 训练看板支持“基于当前任务重开配置”，并在训练页显示 `training-monitor` 来源提示
16. 从 ComfyUI/结果后台/训练看板重开配置时，会预填关键字段（如 `backbone`、`numClasses`）
17. `/evaluation` 已作为独立评估入口开放，可基于现有 `config + checkpoint` 单独补跑 `validation / summary / report`
18. 独立评估当前只支持单次、单 checkpoint；多模型对比、批量评估和评估模板中心后续再做

结果详情页的推荐解读顺序也已固定：

1. 结论层：先看 `summary.json` 的主结论
2. 指标层：再看 `metrics.json` 与 `validation.json`
3. 可视化层：再看 ROC / 混淆矩阵 / 注意力等图示 artifact
4. 文件层：最后回到可下载文件用于复盘与交付

如果是高级模式发起的 run，结果详情会补充来源链（`source_type`、`entrypoint`、`blueprint_id`），并补充该 blueprint 对应的 `recommended_preset` 与 `compile_boundary`，用于对外演示和内部复核。

非主链路径（如 `/preprocessing`）仍保持实验或降级状态，不作为 OSS 默认入口叙事；`/workflow` 已升级为受限 preview，`/evaluation` 已升级为正式单次评估入口。

### 1. 训练监控

实时监控训练任务的进度和性能指标：

- **任务列表** - 查看所有训练任务及其状态
- **实时图表** - 损失曲线、准确率曲线、学习率变化
- **任务控制** - 暂停、恢复、停止训练任务
- **WebSocket 连接** - 实时更新，无需刷新页面

**访问路径**: `/training`

### 2. 模型管理

管理训练好的模型：

- **模型列表** - 浏览所有已保存的模型
- **搜索和筛选** - 按 backbone、格式、准确率等筛选
- **模型详情** - 查看模型参数、性能指标、训练信息
- **上传/下载** - 上传新模型或下载现有模型
- **统计信息** - 总模型数、总大小、平均准确率

**访问路径**: `/models`

### 3. 工作流编辑器（受限 preview）

可视化构建训练流程：

- **拖拽式编辑** - 通过拖拽节点构建工作流
- **节点类型** - 数据加载、模型、训练、评估等
- **配置面板** - 为每个节点配置参数
- **保存和加载** - 保存工作流配置供后续使用

当前状态：

1. 工作流能力当前仍不是默认首页，但已经作为受限 preview 正式暴露。
2. 当前只支持单条主线：`dataLoader -> model -> training -> optional evaluation`。
3. 真实执行仍回到统一 training runtime，不把节点图当执行真源。

对外推荐路径：

1. `/workbench`
2. `/datasets`
3. `/config`
4. `/training`
5. `/models`
6. `/system`
7. `/evaluation`

### 4. 独立评估（正式单次入口）

适用场景：

- 已经有 `config + checkpoint`
- 不想为了补结果产物再重跑训练
- 需要单独生成 `validation.json / summary.json / report.md`

当前边界：

1. 只支持单次、单 checkpoint 评估
2. 直接复用现有 `build-results` 结果构建链
3. 可选择是否把结果导入 `/models`
4. 多模型对比、批量评估和评估模板中心后续再做

### 5. 数据预处理（未进入当前 MVP 主链）

管理和预处理数据集：

- **数据集列表** - 查看所有可用数据集
- **预处理任务** - 创建和监控预处理任务
- **数据增强** - 配置数据增强策略
- **数据统计** - 查看数据集统计信息

当前默认导航不会直接开放 `/preprocessing` 页面，建议优先使用工作台和数据管理页完成当前 MVP 闭环。

### 6. 系统监控

监控系统资源使用情况：

- **GPU 监控** - GPU 使用率、显存、温度
- **内存监控** - 系统内存和交换空间使用
- **CPU 监控** - CPU 使用率和负载
- **磁盘监控** - 磁盘空间使用情况

**访问路径**: `/system`

## 🛠️ 开发指南

### 项目结构

```
medfusion/
├── med_core/
│   └── web/              # 后端代码
│       ├── api/          # API 路由
│       ├── models/       # 数据库模型
│       ├── services/     # 业务逻辑
│       ├── static/       # 前端构建产物
│       ├── app.py        # FastAPI 应用
│       ├── cli.py        # CLI 命令
│       └── config.py     # 配置管理
└── web/
    └── frontend/         # 前端代码
        ├── src/
        │   ├── api/      # API 客户端
        │   ├── components/ # React 组件
        │   ├── pages/    # 页面组件
        │   ├── utils/    # 工具函数
        │   └── App.tsx   # 主应用
        ├── package.json
        └── vite.config.ts
```

### 后端开发

后端使用 FastAPI 框架：

```bash
# 开发模式（自动重载）
uv run uvicorn med_core.web.app:app --reload --host 127.0.0.1 --port 8000

# 查看 API 文档
# 访问 http://127.0.0.1:8000/docs
```

**添加新的 API 端点**:

1. 在 `med_core/web/api/` 下创建或编辑路由文件
2. 在 `med_core/web/app.py` 中注册路由
3. 如需数据库，在 `med_core/web/models/` 中定义模型

### 前端开发

前端使用 React + TypeScript + Vite：

```bash
cd web/frontend

# 安装依赖
npm install

# 开发模式（热重载）
npm run dev

# 构建生产版本
npm run build

# 预览构建结果
npm run preview
```

**开发服务器配置**:

前端开发服务器（端口 3000）会自动代理 API 请求到后端（端口 8000）。

**添加新页面**:

1. 在 `src/pages/` 下创建新的页面组件
2. 在 `src/App.tsx` 中添加路由
3. 在 `src/components/Sidebar.tsx` 中添加导航链接

### 技术栈

**后端**:
- FastAPI - Web 框架
- SQLAlchemy - ORM
- Pydantic - 数据验证
- Uvicorn - ASGI 服务器

**前端**:
- React 18 - UI 框架
- TypeScript - 类型安全
- Vite - 构建工具
- Ant Design - UI 组件库
- ECharts - 图表库
- React Router - 路由管理
- Axios - HTTP 客户端
- Socket.IO - WebSocket 通信

## 🔧 故障排除

### 问题 1: 端口被占用

**错误信息**: `Address already in use`

**解决方案**:
```bash
# 查找占用端口的进程
lsof -i :8000

# 杀死进程
kill -9 <PID>

# 新手推荐直接换官方入口的端口
uv run medfusion start --port 8080
```

### 问题 2: 前端资源未找到

**错误信息**: `404 Not Found` 或空白页面

**解决方案**:
```bash
# 重新构建前端
cd web/frontend
npm run build
cp -r dist/* ../../med_core/web/static/
```

### 问题 3: API 请求失败

**错误信息**: `Network Error` 或 `CORS Error`

**解决方案**:
1. 确保后端服务器正在运行
2. 检查 API 地址是否正确
3. 查看浏览器控制台的错误信息
4. 检查后端日志

### 问题 4: WebSocket 连接失败

**错误信息**: `WebSocket connection failed`

**解决方案**:
1. 确保后端支持 WebSocket
2. 检查防火墙设置
3. 如果使用反向代理，确保配置了 WebSocket 支持

### 问题 5: 依赖安装失败

**解决方案**:
```bash
# 清理并重新安装
rm -rf .venv
uv venv
uv pip install -e ".[web]"

# 前端依赖
cd web/frontend
rm -rf node_modules package-lock.json
npm install
```

## 📚 更多资源

- [完整文档](../../README.md)
- [API 文档](http://127.0.0.1:8000/docs)
- [Web 工作区说明](../../../web/README.md)
- [开发者指南](../../../CONTRIBUTING.md)

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

- 提交 Issue: [GitHub Issues](https://github.com/iridyne/medfusion/issues)
- 提交 PR: [GitHub Pull Requests](https://github.com/iridyne/medfusion/pulls)

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](../../../LICENSE) 文件。

---

**最后更新**: 2026-04-22  
**版本**: 0.3.0
