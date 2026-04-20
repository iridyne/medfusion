# MedFusion 全平台安装 / 部署 / 卸载路线图（发布向）

> Last updated: 2026-04-20
>
> 目标：把“能跑”升级为“可安装、可部署、可卸载、可复现”。
> Windows 是当前市场推广第一优先平台。

## 1. 发布级范围（不做过度工程化）

### 当前执行焦点（2026-04-20 起）

- 只优先推进 Windows 安装与卸载闭环。
- Linux 与 Docker 的安装/卸载落地放到后续里程碑，不作为当前实现阻塞。

### P0（本轮发布阻塞项）

- Windows 10/11 x64：安装、启动、smoke、卸载闭环

### P1（下一阶段）

- Linux（Ubuntu 22.04+）：安装、启动、smoke、卸载闭环
- Docker（私有部署）：部署、启动、smoke、停止/清理闭环
- macOS（Apple Silicon）安装与卸载闭环
- Windows `winget` 分发清单与升级路径

### Non-goals（本轮不阻塞）

- 企业级多租户安装器
- GUI 安装器重设计
- 云托管商业化交付流程

## 2. 统一发布合同（所有平台同语义）

每个平台都必须能执行同一条逻辑链：

1. `install`: 安装依赖并完成 MedFusion 可运行环境
2. `start`: 能启动 `medfusion start`
3. `smoke`: 至少完成一条正式版 smoke path（本机或 Docker）
4. `uninstall`: 支持保留数据卸载与彻底清理卸载
5. `verify`: 卸载后残留状态符合预期

## 3. Windows 详细规划（优先级最高）

### 3.1 交付件

- `scripts/install/windows/install-medfusion.ps1`
- `scripts/uninstall/windows/uninstall-medfusion.ps1`
- `docs/contents/getting-started/installation.md`（Windows 主入口）
- `docs/contents/getting-started/web-ui.md`（Windows 启动与验证）
- `scripts/release_smoke.py`（可调用 Windows 路径）

### 3.2 安装合同

- 必检：
  - Python 3.11+
  - `uv` 可用
  - 项目依赖可安装（含 Web 依赖）
- 安装结果：
  - `uv run medfusion --help` 可用
  - `uv run medfusion start --no-browser` 可启动
  - 前端静态资源可访问（`/start`）

### 3.3 卸载合同

支持两种模式：

- `--keep-data`（默认推荐）
  - 删除运行环境/入口
  - 保留用户数据目录（如 `~/.medfusion` 与输出目录）
- `--purge-data`
  - 删除运行环境/入口
  - 同时删除用户数据目录与本地缓存

卸载后验收：

- `medfusion` 命令不可用（或显式提示未安装）
- 已选择保留的数据仍可访问，或已按 `--purge-data` 清空

## 4. Linux 规划（后置阶段，与 Windows 对齐）

### 4.1 交付件

- `scripts/install/linux/install-medfusion.sh`
- `scripts/uninstall/linux/uninstall-medfusion.sh`
- 文档入口与 Windows 保持同结构

### 4.2 合同

- 仍然执行 `install -> start -> smoke -> uninstall`
- 命令与参数语义尽量与 Windows 对齐（避免“平台特化命令宇宙”）

## 5. Docker 规划（后置阶段，部署形态）

### 5.1 交付件

- `docker/Dockerfile`
- `docker/docker-compose.yml`
- `docs/contents/tutorials/deployment/docker.md`

### 5.2 合同

- `deploy`: 镜像可构建、容器可启动
- `start`: `/health` 与 `/start` 可访问
- `smoke`: 可执行 Docker 形态最小 smoke
- `cleanup`: 容器、网络、临时卷可按文档清理

## 6. 验收矩阵（发布前必须打勾）

- [ ] Windows 安装脚本可重复执行（幂等）
- [ ] Windows 卸载脚本支持 `--keep-data` / `--purge-data`
- [ ] Windows 能跑通 `release_smoke.py --mode local`
- [ ] 文档中安装、部署、卸载命令与脚本入口一致
- [ ] CI 至少有一条 Windows smoke 任务

后置阶段验收项：

- [ ] Linux 安装/卸载脚本可重复执行
- [ ] Linux 能跑通 `release_smoke.py --mode local`
- [ ] Docker 能跑通 `release_smoke.py --mode docker`
- [ ] CI 补 Docker smoke 任务

## 7. 执行里程碑（建议）

### M1：Windows 闭环（发布阻塞）

- 落地 Windows install/uninstall 脚本
- 补 Windows smoke 证据（命令 + 输出）

### M2：Linux 与 Docker 对齐

- 补 Linux install/uninstall 脚本
- 固化 Docker smoke 与清理步骤

### M3：CI 固化

- 将 Windows + Docker smoke 进入发布前必过清单
- 报错回流到同一排障入口

## 8. 风险与缓解

- 风险：Windows 环境碎片化（PATH、权限、杀软）
  - 缓解：脚本内置前置检查、失败分级提示、可重入执行
- 风险：卸载误删用户训练结果
  - 缓解：默认 `--keep-data`，`--purge-data` 需要明确确认
- 风险：文档与脚本漂移
  - 缓解：所有文档只引用脚本入口，不维护多套手工命令版本
