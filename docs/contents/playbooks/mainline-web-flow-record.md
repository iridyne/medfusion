# Web 主链可复现实录（/start -> /config -> /training -> /models）

> 文档状态：Beta
>
> 目标：沉淀一份可重复执行的主链实录模板，用于对齐产品演示、回归验证和发布前证据归档。

## 1. 适用范围

- 只覆盖正式版唯一主线：`/start -> /config -> /training -> /models`
- 配置阶段默认使用 ComfyUI 适配语义（仍在 MedFusion 主链内）
- 不覆盖高级模式探索和自定义节点实验

## 2. 执行前准备

1. 安装依赖并确认 `uv run medfusion start --help` 可用。
2. 准备一条可运行配置（建议 `configs/starter/quickstart.yaml`）。
3. 预先创建证据目录（建议）：
   - `outputs/evidence/mainline-web-flow/<YYYY-MM-DD>/screenshots/`
   - `outputs/evidence/mainline-web-flow/<YYYY-MM-DD>/notes.md`

## 3. 主链执行步骤

1. 启动 Web：
   - `uv run medfusion start --no-browser`
2. 打开 `/start`：
   - 记录“唯一主线 + 默认 ComfyUI 适配配置”入口文案截图
3. 进入 `/config`：
   - 用默认路径生成配置并导出 YAML
   - 记录关键字段截图（项目名、实验名、输出目录）
4. 进入 `/training`：
   - 启动训练任务并记录任务创建成功状态
5. 进入 `/models`：
   - 验证结果回流
   - 记录 summary / metrics / report / artifacts 是否可见

## 4. 验收检查表（执行后打勾）

- [ ] `/start` 能看到单主线入口叙事
- [ ] `/config` 默认路径为 ComfyUI 适配语义
- [ ] 配置可导出为有效 YAML
- [ ] `/training` 能创建并显示真实训练任务
- [ ] `/models` 能看到回流结果并进入详情
- [ ] 详情中包含 summary / metrics / report / artifacts

## 5. 证据登记（建议格式）

| 项目 | 记录 |
|---|---|
| 执行日期 | `YYYY-MM-DD` |
| 执行环境 | `Windows 11 / Python 3.11+ / uv x.y.z` |
| 入口命令 | `uv run medfusion start --no-browser` |
| 配置文件 | `configs/starter/quickstart.yaml` 或导出路径 |
| 结果目录 | `outputs/...` |
| 截图目录 | `outputs/evidence/mainline-web-flow/<date>/screenshots/` |
| 备注 | 异常、重试和差异说明 |

## 6. 与发布前清单的关系

- 这份文档负责“主链实录”。
- Windows 安装/卸载闭环证据继续维护在：
  - `docs/contents/playbooks/windows-manual-install-validation.md`
- 发布前审阅时，建议并排查看主链实录与安装实录，避免只有安装证据没有主链证据。

## 7. 实录索引

- `docs/contents/playbooks/mainline-web-flow-record-2026-04-20.md`

## 8. 双向跳转索引

- 安装/卸载发布证据：`docs/contents/playbooks/windows-manual-install-validation.md`
- 发布烟雾检查矩阵：`docs/contents/playbooks/release-smoke-matrix.md`
