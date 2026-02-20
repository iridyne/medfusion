# MedFusion Web UI - CLI 使用指南

> 简化的命令行工具，像 `tensorboard`, `mlflow ui` 一样简单易用。

**版本**: v0.1.0  
**更新日期**: 2026-02-20

---

## 🎯 为什么使用 CLI 命令？

### 旧方式 ❌
```bash
./start-webui.sh    # 需要 shell 脚本
./stop-webui.sh     # 需要记住多个脚本
```

**问题**：
- 不符合 Python 生态习惯
- 需要维护多个 shell 脚本
- 跨平台兼容性差
- 不够直观

### 新方式 ✅
```bash
medfusion-web start   # 一个命令搞定
medfusion-web stop    # 简洁直观
medfusion-web status  # 查看状态
```

**优势**：
- ✅ 符合 Python 生态标准（类似 `tensorboard`, `mlflow ui`）
- ✅ 跨平台兼容（Windows/Linux/macOS）
- ✅ 统一的命令接口
- ✅ 更好的用户体验

---

## 📦 安装

### 方式 1: 开发模式安装（推荐）

```bash
# 进入后端目录
cd medfusion-web/backend

# 开发模式安装（可编辑）
pip install -e .

# 验证安装
medfusion-web --version
```

### 方式 2: 正式安装

```bash
cd medfusion-web/backend
pip install .
```

### 方式 3: 使用 uv（推荐）

```bash
cd medfusion-web/backend
uv pip install -e .
```

---

## 🚀 快速开始

### 1. 初始化环境（首次使用）

```bash
medfusion-web init
```

这会自动：
- 安装后端依赖
- 安装前端依赖
- 初始化数据库

### 2. 启动服务

```bash
# 启动完整服务（前端 + 后端）
medfusion-web start

# 或者后台运行
medfusion-web start --daemon
```

### 3. 访问界面

打开浏览器访问：
- **前端界面**: http://localhost:5173
- **后端 API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

### 4. 停止服务

```bash
medfusion-web stop
```

---

## 📚 命令参考

### `medfusion-web start`

启动完整的 Web UI 服务（前端 + 后端）

**选项**：
```bash
--backend-host TEXT      后端服务主机地址 [默认: 0.0.0.0]
--backend-port INTEGER   后端服务端口 [默认: 8000]
--frontend-port INTEGER  前端服务端口 [默认: 5173]
--reload                 开发模式（热重载）
--daemon                 后台运行
```

**示例**：
```bash
# 默认启动
medfusion-web start

# 后台运行
medfusion-web start --daemon

# 开发模式（热重载）
medfusion-web start --reload

# 自定义端口
medfusion-web start --backend-port 8080 --frontend-port 3000
```

---

### `medfusion-web start-backend`

只启动后端 API 服务

**选项**：
```bash
--host TEXT       主机地址 [默认: 0.0.0.0]
--port INTEGER    端口 [默认: 8000]
--reload          开发模式（热重载）
--daemon          后台运行
```

**示例**：
```bash
# 前台运行（开发调试）
medfusion-web start-backend

# 后台运行
medfusion-web start-backend --daemon

# 开发模式
medfusion-web start-backend --reload
```

---

### `medfusion-web start-frontend`

只启动前端开发服务器

**选项**：
```bash
--port INTEGER    端口 [默认: 5173]
--daemon          后台运行
```

**示例**：
```bash
# 前台运行
medfusion-web start-frontend

# 后台运行
medfusion-web start-frontend --daemon

# 自定义端口
medfusion-web start-frontend --port 3000
```

---

### `medfusion-web stop`

停止 Web UI 服务

**选项**：
```bash
--service [backend|frontend|all]  要停止的服务 [默认: all]
```

**示例**：
```bash
# 停止所有服务
medfusion-web stop

# 只停止后端
medfusion-web stop --service backend

# 只停止前端
medfusion-web stop --service frontend
```

---

### `medfusion-web status`

查看服务状态

**示例**：
```bash
medfusion-web status
```

**输出示例**：
```
📊 服务状态

  Backend: ✅ 运行中 (PID: 12345, CPU: 2.3%, 内存: 156.2MB)
  Frontend: ✅ 运行中 (PID: 12346, CPU: 0.8%, 内存: 89.5MB)

  后端端口 8000: ✅ 可访问
  前端端口 5173: ✅ 可访问
```

---

### `medfusion-web logs`

查看服务日志

**选项**：
```bash
--service [backend|frontend|all]  要查看的日志 [默认: all]
--follow, -f                      实时跟踪日志
--lines, -n INTEGER               显示的行数 [默认: 50]
```

**示例**：
```bash
# 查看所有日志（最近 50 行）
medfusion-web logs

# 实时跟踪日志
medfusion-web logs -f

# 只查看后端日志
medfusion-web logs --service backend

# 查看最近 100 行
medfusion-web logs -n 100
```

---

### `medfusion-web init`

初始化 Web UI 环境

**功能**：
- 安装后端 Python 依赖
- 安装前端 npm 依赖
- 初始化数据库

**示例**：
```bash
medfusion-web init
```

---

## 🔧 常见使用场景

### 场景 1: 开发调试

```bash
# 启动后端（热重载）
medfusion-web start-backend --reload

# 在另一个终端启动前端
medfusion-web start-frontend

# 实时查看日志
medfusion-web logs -f
```

### 场景 2: 生产部署

```bash
# 后台运行所有服务
medfusion-web start --daemon

# 查看状态
medfusion-web status

# 查看日志
medfusion-web logs -n 100
```

### 场景 3: 快速演示

```bash
# 一键启动（前台运行）
medfusion-web start

# 访问 http://localhost:5173
# 按 Ctrl+C 停止
```

### 场景 4: 只使用 API

```bash
# 只启动后端
medfusion-web start-backend --daemon

# 访问 API 文档
# http://localhost:8000/docs
```

---

## 🆚 新旧方式对比

| 操作 | 旧方式（Shell 脚本） | 新方式（CLI 命令） |
|------|---------------------|-------------------|
| 启动服务 | `./start-webui.sh` | `medfusion-web start` |
| 停止服务 | `./stop-webui.sh` | `medfusion-web stop` |
| 查看状态 | ❌ 不支持 | `medfusion-web status` |
| 查看日志 | `tail -f logs/*.log` | `medfusion-web logs -f` |
| 初始化 | 手动执行多个命令 | `medfusion-web init` |
| 只启动后端 | ❌ 不支持 | `medfusion-web start-backend` |
| 只启动前端 | ❌ 不支持 | `medfusion-web start-frontend` |
| 跨平台 | ❌ 仅 Linux/macOS | ✅ Windows/Linux/macOS |

---

## 💡 最佳实践

### 1. 开发环境

```bash
# 使用开发模式安装
pip install -e .

# 启动时使用热重载
medfusion-web start --reload

# 实时查看日志
medfusion-web logs -f
```

### 2. 生产环境

```bash
# 正式安装
pip install .

# 后台运行
medfusion-web start --daemon

# 定期检查状态
medfusion-web status

# 查看日志排查问题
medfusion-web logs -n 200
```

### 3. 自定义配置

```bash
# 使用环境变量
export BACKEND_PORT=8080
export FRONTEND_PORT=3000

# 或者使用命令行参数
medfusion-web start --backend-port 8080 --frontend-port 3000
```

---

## 🐛 故障排查

### 问题 1: 命令未找到

```bash
$ medfusion-web
bash: medfusion-web: command not found
```

**解决**：
```bash
# 确保已安装
pip install -e .

# 或者使用完整路径
python -m app.cli
```

### 问题 2: 端口被占用

```bash
❌ 端口 8000 已被占用
```

**解决**：
```bash
# 方式 1: 停止占用端口的进程
lsof -ti:8000 | xargs kill -9

# 方式 2: 使用其他端口
medfusion-web start --backend-port 8080
```

### 问题 3: 服务无法启动

```bash
# 查看详细日志
medfusion-web logs

# 检查依赖是否安装
pip list | grep fastapi

# 重新初始化
medfusion-web init
```

---

## 📖 与其他工具对比

### TensorBoard
```bash
tensorboard --logdir=./logs
```

### MLflow
```bash
mlflow ui
```

### Streamlit
```bash
streamlit run app.py
```

### MedFusion Web UI
```bash
medfusion-web start
```

**一致的体验** ✨

---

## 🔄 迁移指南

### 从 Shell 脚本迁移

**旧方式**：
```bash
./start-webui.sh
# 等待启动...
# 访问 http://localhost:5173
./stop-webui.sh
```

**新方式**：
```bash
# 首次使用
pip install -e backend/

# 启动
medfusion-web start --daemon

# 查看状态
medfusion-web status

# 停止
medfusion-web stop
```

**Shell 脚本保留**：
- 作为备选方案
- 用于 CI/CD 环境
- 特殊部署场景

---

## 📝 总结

### 核心优势

1. **简洁直观** - 一个命令搞定所有操作
2. **符合标准** - 遵循 Python 生态习惯
3. **跨平台** - Windows/Linux/macOS 通用
4. **功能完整** - 启动、停止、状态、日志全覆盖
5. **易于维护** - 统一的代码管理

### 推荐使用

✅ **日常开发**: `medfusion-web start --reload`  
✅ **生产部署**: `medfusion-web start --daemon`  
✅ **快速演示**: `medfusion-web start`  
✅ **问题排查**: `medfusion-web logs -f`

---

**文档版本**: v1.0  
**最后更新**: 2026-02-20  
**维护者**: MedFusion Team