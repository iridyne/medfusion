# MedFusion Web Workspace

这个目录现在只承担一件事：

**存放 Web 前端源码与少量历史占位说明。**

如果你打开 `web/` 只是想知道“现在应该看哪里、改哪里、怎么启动”，这份 README 就是唯一推荐入口。

## 当前边界

- `web/frontend/`：前端源码工作区
- `med_core/web/`：当前真实后端与生产静态资源承载位置
- `web/backend/`：历史占位目录，不再承载当前后端实现

换句话说：

- 要改前端，去 `web/frontend/`
- 要看当前 Web 使用方式，去 `docs/contents/getting-started/web-ui.md`
- 要看当前后端实现，去 `med_core/web/`

## 目录说明

```text
web/
├── frontend/              # React + TypeScript 前端源码
│   ├── src/               # 页面、组件、API 客户端、工具函数
│   ├── package.json       # 前端依赖与脚本
│   ├── vite.config.ts     # Vite 开发与构建配置
│   └── dist/              # 本地构建产物，不是源码
├── backend/
│   └── DEPRECATED.md      # 旧独立后端的历史占位说明
├── docker-compose.yml     # Web 相关容器编排参考
└── README.md              # 本文档
```

## 唯一推荐入口

对大多数用户和回归场景，唯一推荐入口是：

```bash
uv run medfusion start
```

不要把 `web/` 目录理解成“用户启动 Web 的地方”。
当前官方 Web 入口已经收口到 CLI，而不是 `web/` 目录下的脚本或多套指南。

## 开发入口

### 前端开发

```bash
cd web/frontend
npm install
npm run dev
```

默认开发服务器由 Vite 提供，当前配置下前端端口是 `3000`，并代理到本地 `8000` 端口后端。

### 后端开发

```bash
uv run uvicorn med_core.web.app:app --reload --host 127.0.0.1 --port 8000
```

这里的重点是：

- 后端源码不在 `web/backend/`
- 当前后端实现已经在 `med_core/web/`

## 构建产物与本地文件

下面这些内容都不是需要整理进源码叙事的“正式结构”：

- `web/frontend/dist/`
- `web/frontend/node_modules/`
- `web/frontend/.env.local`

它们分别代表：

- 前端构建产物
- 前端依赖目录
- 本地环境配置

这些文件或目录可以存在于本机，但不应该被当成 Web 工作区的核心结构来理解。

## 当前真实的 Web 测试入口

不要再从 `web/` 目录下找独立测试脚本。

当前真实的 Web 测试入口在仓库根目录 `tests/` 下，例如：

- `tests/test_web_api_minimal.py`
- `tests/test_web_training_controls.py`
- `tests/test_workflow_api.py`

## 生产静态资源链路

当前链路是：

```text
web/frontend/src/
  -> npm run build
  -> web/frontend/dist/
  -> med_core/web/static/
```

也就是说：

- `web/frontend/` 负责开发
- `med_core/web/static/` 负责承载生产静态资源

## 当前建议的整理原则

为了避免 `web/` 继续膨胀，后续保持这几个原则：

1. 新的用户说明尽量写到主文档站，而不是继续往 `web/` 下堆独立指南。
2. `web/README.md` 保持为 `web/` 唯一总入口。
3. `web/backend/` 只保留历史占位，不再新增真实后端代码。
4. `dist/`、`node_modules/`、本地 `.env.local` 一律视为本地产物，不纳入目录治理叙事。

## 相关文档

- [Web UI 快速入门](../docs/contents/getting-started/web-ui.md)
- [Web UI 架构](../docs/contents/architecture/WEB_UI_ARCHITECTURE.md)
- [项目快速开始](../docs/contents/getting-started/quickstart.md)
