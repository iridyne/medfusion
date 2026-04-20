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
- 🔄 **工作流编辑器** - 实验功能，默认不在当前 MVP 导航中开放
- 📁 **数据集管理** - 管理和预处理医学影像数据集
- 💻 **系统监控** - 查看 GPU、内存、CPU 使用情况
- 🌐 **RESTful API** - 完整的 API 支持，可用于自动化和集成

## 🚀 快速启动

### 方法 1：使用统一 CLI 入口（推荐）

```bash
# 推荐入口：先进入 Getting Started 引导页
uv run medfusion start

# 指定选项
uv run medfusion start --host 0.0.0.0 --port 8080 --reload
```

`medfusion start` 会把 Web UI 作为默认入口收口起来，更适合新用户和回归验证场景。

当前默认第一页不再是假设你已经熟悉所有页面的工作台首页，而是 `Getting Started` 引导页。
它的职责是：

1. 介绍正式版当前有哪些组件和页面职责
2. 解释默认模式与高级模式的边界
3. 把你带到问题向导与第一次运行链路
4. 把你带到后续训练与结果页面

大多数新手只需要这一条命令。

当前 Getting Started 页后的推荐路径：

1. 先进入 `Run Wizard`，从问题定义出发拿到推荐骨架
2. 如需理解高级边界，再进入 `/config/advanced` 查看组件注册表和连接约束
3. 需要 first-run 演示时，再进入 `Quickstart Run` 页面确认公开数据 profile、命令链和预期产物
4. 执行 `medfusion public-datasets prepare` 与 `medfusion validate-config`
5. 继续到训练监控页启动一次带默认参数的推荐训练
6. 训练完成后执行 `medfusion build-results`，再到模型库查看 artifact

如果你是发布前自检，建议直接跑统一脚本：

```bash
uv run python scripts/release_smoke.py --mode local
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

当前 OSS Web 默认主链固定为 7 个页面与 1 条引导子路径：

1. `/start`（Getting Started）
2. `/workbench`（运行后概览）
3. `/datasets`（数据集管理）
4. `/config`（RunSpec 向导）
5. `/training`（训练监控）
6. `/models`（模型与结果）
7. `/system`（系统监控）

引导子路径：

- `/quickstart-run`：承接第一次运行链路说明，不单独放入主导航，但作为 Getting Started 的下一步

高级模式路径：

- `/config/advanced`：正式版高级模式的组件注册表与连接约束页，不作为默认首页，但用于承接后续节点式结构编辑
- `/config/advanced/canvas`：高级模式节点图入口，当前允许在正式版组件边界内做结构编辑、编译检查，并在 contract 校验通过后直接创建真实训练任务

当前进展：

1. 高级模式节点图现在可以编译出 RunSpec 草案
2. 后端会继续做正式 `ExperimentConfig` contract 校验
3. 当校验通过时，可以直接从高级模式创建真实训练任务
4. 训练完成后，任务状态会附带结果 handoff 信息，并可继续跳到模型库查看结果

结果详情页的推荐解读顺序也已固定：

1. 结论层：先看 `summary.json` 的主结论
2. 指标层：再看 `metrics.json` 与 `validation.json`
3. 可视化层：再看 ROC / 混淆矩阵 / 注意力等图示 artifact
4. 文件层：最后回到可下载文件用于复盘与交付

如果是高级模式发起的 run，结果详情会补充来源链（`source_type`、`entrypoint`、`blueprint_id`），用于对外演示和内部复核。

非主链路径（如 `/workflow`、`/preprocessing`）保持实验或降级状态，不作为 OSS 默认入口叙事。

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

### 3. 工作流编辑器（实验态，默认关闭）

可视化构建训练流程：

- **拖拽式编辑** - 通过拖拽节点构建工作流
- **节点类型** - 数据加载、模型、训练、评估等
- **配置面板** - 为每个节点配置参数
- **保存和加载** - 保存工作流配置供后续使用

当前状态：

1. 工作流能力仍处于实验态，不属于当前对外演示主链。
2. 默认导航不会暴露 `/workflow`，前端也会回退到工作台首页。
3. 后端 API 默认返回实验态关闭提示，只有在显式设置 `MEDFUSION_ENABLE_EXPERIMENTAL_WORKFLOW=true` 时才开放。

对外推荐路径：

1. `/workbench`
2. `/datasets`
3. `/config`
4. `/training`
5. `/models`
6. `/system`

### 4. 数据预处理（未进入当前 MVP 主链）

管理和预处理数据集：

- **数据集列表** - 查看所有可用数据集
- **预处理任务** - 创建和监控预处理任务
- **数据增强** - 配置数据增强策略
- **数据统计** - 查看数据集统计信息

当前默认导航不会直接开放 `/preprocessing` 页面，建议优先使用工作台和数据管理页完成当前 MVP 闭环。

### 5. 系统监控

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

**最后更新**: 2026-02-20  
**版本**: 0.3.0
