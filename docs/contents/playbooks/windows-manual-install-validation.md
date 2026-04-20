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

## 当前状态（按 roadmap 同步）

1. Windows 手工链路的 smoke 证据已补齐（`validate-config -> train -> build-results`）。
2. 本记录已纳入发布前检查入口并与主链实录互链。

---

## 第二轮补充（2026-04-20）

### A. Smoke 证据已补

执行命令：

```powershell
uv run python scripts/release_smoke.py --mode local
```

结果：成功（`Formal-release smoke completed.`）。

关键证据：

- `outputs/evidence/mainline-web-flow/2026-04-20/release-smoke-local.log`
- `outputs/public_datasets/breastmnist_quickstart/checkpoints/best.pth`
- `outputs/public_datasets/breastmnist_quickstart/reports/summary.json`
- `outputs/public_datasets/breastmnist_quickstart/reports/report.md`

### B. 卸载环节（dry-run + 沙箱实跑）

当前 CLI 已提供独立 `uninstall` 命令：

```powershell
uv run medfusion uninstall --yes
uv run medfusion uninstall --purge-data --yes
```

本轮已执行 dry-run 验证（避免误删当前开发环境）：

```powershell
uv run medfusion uninstall --dry-run --yes --json
uv run medfusion uninstall --purge-data --dry-run --yes --json
```

证据：

- `outputs/evidence/mainline-web-flow/2026-04-20/uninstall-dry-run.json`
- `outputs/evidence/mainline-web-flow/2026-04-20/uninstall-purge-dry-run.json`

隔离沙箱实跑（非 dry-run）：

- 在 `tmp/uninstall-sandbox-2026-04-20-b/` 造假数据目录
- 执行：

```powershell
uv run --project C:\Users\Administrator\Projects\medfusion medfusion uninstall --purge-data --yes --user-data-dir .\user-data --json
```

证据：

- `outputs/evidence/mainline-web-flow/2026-04-20/uninstall-sandbox-run.json`
- `outputs/evidence/mainline-web-flow/2026-04-20/uninstall-sandbox-postcheck.json`

### C. 真实用户目录场景实跑（完成）

执行目录与数据目录（模拟真实用户目录，不触碰仓库运行产物）：

- CWD：`C:\Users\Administrator\medfusion-uninstall-userdir-sim-2026-04-20`
- user-data-dir：`C:\Users\Administrator\.medfusion-uninstall-sim`

执行：

```powershell
uv run --project C:\Users\Administrator\Projects\medfusion medfusion uninstall --purge-data --yes --user-data-dir C:\Users\Administrator\.medfusion-uninstall-sim --json
```

证据：

- `outputs/evidence/mainline-web-flow/2026-04-20/uninstall-userdir-sim-run.json`
- `outputs/evidence/mainline-web-flow/2026-04-20/uninstall-userdir-sim-postcheck.json`

结论：

- Windows 链路中的 `install -> start -> smoke -> uninstall` 证据已闭环。
- `uninstall` 命令入口、dry-run、沙箱实跑、真实用户目录场景实跑均已落地。

## 双向跳转索引

- 主链实录模板：`docs/contents/playbooks/mainline-web-flow-record.md`
- 主链实录样例：`docs/contents/playbooks/mainline-web-flow-record-2026-04-20.md`
- 发布烟雾检查矩阵：`docs/contents/playbooks/release-smoke-matrix.md`
