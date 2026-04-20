# Windows 手工安装验证记录

> 文档状态：**Beta**
>
> 记录日期：2026-04-20

目标：给 Phase 3.4 提供第一轮 Windows 手工安装主链证据（不依赖安装脚本）。

## 环境

- OS: Windows（当前开发机）
- Repo: `C:\Users\Administrator\Projects\medfusion`
- Python runtime: `uv run python`

## 执行命令与结果

1. 安装依赖：

```powershell
uv sync --extra dev --extra web
```

结果：成功（依赖同步完成）。

2. 校验 CLI：

```powershell
uv run medfusion --version
```

结果：成功，返回 `MedFusion 0.2.0`。

3. 启动 Web 并校验可达：

```powershell
uv run medfusion start --host 127.0.0.1 --port 18080 --no-browser
```

结果：成功。验证到：

- `GET /health` -> `200`
- `GET /start` -> `200`

## 本轮发现与修复

1. `medfusion start` 在 Windows 非 UTF-8 控制台（GBK）下会因为 Rich 输出字符编码失败，导致启动中断。
2. Web 静态服务对 SPA 路由缺少回退，`/start` 之前可能返回 `404`。

已修复：

- `med_core/web/cli.py`：移除会触发编码问题的 Rich spinner/emoji 输出，改为纯文本初始化日志。
- `med_core/web/app.py`：引入 `SPAStaticFiles`，为非文件型前端路由回退 `index.html`。
- `tests/test_web_api_minimal.py`：补充 `/start` 返回 `200` 的最小契约断言。

## 下一步（继续按 roadmap）

1. 补 Windows 手工链路的 smoke 证据（`validate-config -> train -> build-results`）。
2. 将该验证记录纳入发布前检查入口。
