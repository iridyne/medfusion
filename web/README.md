# MedFusion Web UI

**架构**: 集成架构（方案 A）  
**状态**: 生产就绪 ✅  
**版本**: v0.3.0

## 📂 目录结构

```
web/
├── frontend/              # 前端源码（React + TypeScript）
│   ├── src/              # 源代码
│   │   ├── pages/        # 页面组件
│   │   ├── components/   # 可复用组件
│   │   ├── api/          # API 客户端
│   │   └── utils/        # 工具函数
│   ├── dist/             # 构建产物（自动生成）
│   ├── package.json
│   └── vite.config.ts
├── docs/                 # Web UI 相关文档
└── README.md             # 本文档
```

**注意**: 后端代码已迁移到 `med_core/web/`（集成架构）

## 🚀 使用方式

### 用户使用（生产模式）

```bash
# 1. 安装（包含 Web UI）
pip install medfusion[web]

# 2. 启动（单进程，集成架构）
./start-webui.sh  # 项目根目录
# 或
medfusion web start

# 3. 访问
http://localhost:8000
```

### 开发者使用（开发模式）

```bash
# 后端开发（热重载）
uv run uvicorn med_core.web.app:app --reload --host 127.0.0.1 --port 8000

# 前端开发（热重载）
cd web/frontend
npm install
npm run dev  # 访问 http://localhost:5173
```

## 🔨 构建和部署

### 构建前端

```bash
cd web/frontend

# 安装依赖
npm install

# 构建生产版本
npm run build
# 输出到 dist/
```

### 部署到后端

```bash
# 从项目根目录执行
cp -r web/frontend/dist/* med_core/web/static/

# 验证部署
ls -lh med_core/web/static/
```

### 完整流程

```bash
# 一键构建和部署
cd web/frontend && npm run build && cd ../.. && \
cp -r web/frontend/dist/* med_core/web/static/ && \
echo "✅ 部署完成"
```

## 📋 架构说明

### 集成架构（当前）

```
┌─────────────────────────────────┐
│  MedFusion Python 包             │
│                                  │
│  med_core/web/                  │
│  ├── app.py (FastAPI)           │
│  ├── api/ (REST API)            │
│  └── static/ (前端构建产物)     │
│      ├── index.html             │
│      └── assets/                │
└─────────────────────────────────┘
```

**特点**:
- 单进程运行
- 前后端一体
- 类似 TensorBoard 体验
- 一个命令启动

### 开发流程

```
web/frontend/src/  →  npm run build  →  web/frontend/dist/  →  复制  →  med_core/web/static/
   (源代码)              (构建)            (构建产物)          (部署)      (生产环境)
```

## 📚 相关文档

- [架构设计文档](../docs/WEB_UI_ARCHITECTURE.md) - 详细的架构说明和设计决策
- [快速入门指南](../docs/WEB_UI_QUICKSTART.md) - 用户使用教程
- [项目状态报告](../docs/PROJECT_STATUS.md) - 当前开发进度

## 🛠️ 技术栈

### 前端
- React 18 + TypeScript
- Vite 5 (构建工具)
- Ant Design 5 (UI 组件)
- ECharts 5 (图表)
- React Router 6 (路由)
- Axios (HTTP 客户端)

### 后端
- FastAPI (Web 框架)
- Uvicorn (ASGI 服务器)
- SQLite (数据库)
- SQLAlchemy (ORM)

## 🔧 开发命令

```bash
# 前端开发
cd web/frontend
npm run dev          # 启动开发服务器
npm run build        # 构建生产版本
npm run preview      # 预览构建结果
npm run lint         # 代码检查
npm run type-check   # 类型检查

# 后端开发
uv run uvicorn med_core.web.app:app --reload  # 开发模式
uv run pytest tests/web/                       # 运行测试
```

## ❓ 常见问题

### Q: 为什么前端和后端分开？

A: 前端源码（`web/frontend/`）用于开发，构建产物部署到后端（`med_core/web/static/`）用于生产。这样既保持开发灵活性，又提供简单的用户体验。

### Q: 如何更新前端？

A: 修改 `web/frontend/src/` 中的代码，然后运行 `npm run build` 并复制到 `med_core/web/static/`。

### Q: 旧的 web/backend/ 去哪了？

A: 已迁移到 `med_core/web/`（集成架构）。旧代码已删除，备份在 `backups/web-backend-20260220.tar.gz`。

---

**最后更新**: 2026-02-20  
**维护者**: Medical AI Research Team