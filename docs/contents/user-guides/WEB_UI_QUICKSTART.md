# MedFusion Web UI 快速入门

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
- 🔄 **工作流编辑器** - 可视化构建训练流程（开发中）
- 📁 **数据集管理** - 管理和预处理医学影像数据集
- 💻 **系统监控** - 查看 GPU、内存、CPU 使用情况
- 🌐 **RESTful API** - 完整的 API 支持，可用于自动化和集成

## 🚀 快速启动

### 方法 1：使用启动脚本（推荐）

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

### 方法 2：手动启动

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

### 方法 3：使用 CLI 命令

```bash
# 使用 MedFusion CLI
uv run python med_core/cli.py web start

# 指定选项
uv run python med_core/cli.py web start --host 0.0.0.0 --port 8080 --reload
```

## 🌐 访问界面

启动成功后，在浏览器中访问：

- **Web UI**: http://127.0.0.1:8000
- **API 文档**: http://127.0.0.1:8000/docs
- **健康检查**: http://127.0.0.1:8000/health

## 📱 主要功能

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

### 3. 工作流编辑器（开发中）

可视化构建训练流程：

- **拖拽式编辑** - 通过拖拽节点构建工作流
- **节点类型** - 数据加载、模型、训练、评估等
- **配置面板** - 为每个节点配置参数
- **保存和加载** - 保存工作流配置供后续使用

**访问路径**: `/workflow`

### 4. 数据预处理

管理和预处理数据集：

- **数据集列表** - 查看所有可用数据集
- **预处理任务** - 创建和监控预处理任务
- **数据增强** - 配置数据增强策略
- **数据统计** - 查看数据集统计信息

**访问路径**: `/preprocessing`

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

# 或使用其他端口
./start-webui.sh 8080
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

- [完整文档](../README.md)
- [API 文档](http://127.0.0.1:8000/docs)
- [CLI 使用指南](../web/CLI_GUIDE.md)
- [开发者指南](../CONTRIBUTING.md)

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

- 提交 Issue: [GitHub Issues](https://github.com/your-repo/medfusion/issues)
- 提交 PR: [GitHub Pull Requests](https://github.com/your-repo/medfusion/pulls)

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](../LICENSE) 文件。

---

**最后更新**: 2026-02-20  
**版本**: 0.3.0