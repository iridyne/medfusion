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

## 进行中（最小发布补齐）

- [x] Phase 3.3：新增统一 smoke 执行脚本 `scripts/release_smoke.py`（本机 + Docker）
- [ ] Phase 3.1：安装/启动文档进一步统一到同一口径
- [x] Phase 3.2：Web API 文档与实现字段首轮对齐（含结果可视化字段与 experimental 边界）
- [ ] Phase 3.3：沉淀 Windows 与 Docker 两条路径的可复现执行证据（命令 + 结果记录）
- [x] Phase 3.4：新增全平台安装/部署/卸载完整规划（Windows 优先）
- [ ] Phase 3.4：落地 Windows install/uninstall 脚本与双模式卸载（keep-data / purge-data）
- [ ] Phase 3.4：补 Linux 对等 install/uninstall 脚本并与 Windows 语义对齐

## 下一步建议（按最小可落地顺序）

1. 先落地 Windows 安装与卸载脚本，并在本机完成一次 `install -> start -> smoke -> uninstall` 实录。
2. 再补 Linux 对等脚本，保持参数语义与 Windows 一致。
3. 最后把 Windows + Docker smoke 纳入发布前 CI 必过项。
