# MedFusion OSS Roadmap Status

> Last updated: 2026-04-20
>
> Source of truth plan:
> `docs/superpowers/plans/2026-04-19-medfusion-formal-release-execution-plan.md`

## 已完成（按路线图主线）

- [x] Phase 0：正式版定义、默认入口与页面职责收口
- [x] Phase 1：`/start` + `Run Wizard` 问题优先入口收口
- [x] Phase 2：高级模式图编译、contract 校验、真实训练任务创建与结果回流
- [x] Phase 2：结果详情页补齐三期解释 artifact 摘要展示（phase importance / case explanations / heatmap manifest）
- [x] Phase 2.2：高级模式编译问题结构化增强（`code/path/context/suggestion` + 画布定位）
- [x] Phase 2.2：ComfyUI 上线入口最小落地（`/config/comfyui` + `/api/comfyui/health`）
- [x] Phase 2.2：ComfyUI 适配层首版（`/api/comfyui/adapter-profiles` + 适配档案跳转组件画布）

## 进行中（最小发布补齐）

- [x] Phase 3.3：新增统一 smoke 执行脚本 `scripts/release_smoke.py`（本机 + Docker）
- [ ] Phase 3.1：安装/启动文档进一步统一到同一口径
- [x] Phase 3.1：Windows 启动兼容性修复（GBK 控制台编码 + `/start` 路由回退）
- [x] Phase 3.2：Web API 文档与实现字段首轮对齐（含结果可视化字段与 experimental 边界）
- [ ] Phase 3.3：沉淀 Windows 与 Docker 两条路径的可复现执行证据（命令 + 结果记录）
- [x] Phase 3.4：新增全平台安装/部署/卸载完整规划（Windows 优先）
- [x] Phase 3.4：安装策略收口为 Windows 手工命令路径（不推荐脚本安装）
- [x] Phase 3.4：补 Windows 手工安装实录首轮证据（install -> start 可达）
- [ ] Phase 3.4：补 Windows 完整实录（install -> start -> smoke -> uninstall）并沉淀证据
- [ ] Phase 3.4（后置）：补 Linux 对等 install/uninstall 脚本并与 Windows 语义对齐
- [ ] Phase 3.4（后置）：补 Docker 安装/卸载口径与 CI smoke 对齐

## 下一步建议（按最小可落地顺序）

1. 先补 Windows 完整主链实录中的 `smoke + uninstall` 两段证据。
2. 把 Windows 手工安装/卸载命令继续收口到安装文档单一入口。
3. Linux 与 Docker 脚本放到后置里程碑，不阻塞当前 Windows 主线。
