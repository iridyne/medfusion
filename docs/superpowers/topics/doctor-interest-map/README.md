# Doctor Interest Map Docs Index

日期：2026-04-17

本页用于归类 `doctor-interest-map` 主题相关文档，避免三期 CT 主线、旧热图导出、当前设计和执行计划散落在不同目录里。

## 1. 当前权威文档

这些是本轮工作的主文档，后续实现与 review 以这里为准。

- 设计说明：
  [2026-04-17-doctor-interest-map-design.md](/C:/Users/Administrator/.config/superpowers/worktrees/medfusion/doctor-interest-map/docs/superpowers/specs/2026-04-17-doctor-interest-map-design.md)
- 实施计划：
  [2026-04-17-doctor-interest-map-implementation-plan.md](/C:/Users/Administrator/.config/superpowers/worktrees/medfusion/doctor-interest-map/docs/superpowers/plans/2026-04-17-doctor-interest-map-implementation-plan.md)

## 2. 上游主线背景

这些文档描述的是当前 three-phase CT + clinical 主线能力，是理解本次 doctor-interest-map 改造的基础。

- 三期 CT 主线设计：
  [2026-03-31-smurf-feature-network-design.md](/C:/Users/Administrator/.config/superpowers/worktrees/medfusion/doctor-interest-map/docs/superpowers/specs/2026-03-31-smurf-feature-network-design.md)
- 三期 CT 主线实施计划：
  [2026-03-31-smurf-feature-network-implementation-plan.md](/C:/Users/Administrator/.config/superpowers/worktrees/medfusion/doctor-interest-map/docs/superpowers/plans/2026-03-31-smurf-feature-network-implementation-plan.md)
- Demo 复盘与汇报底稿：
  [three_phase_ct_mvi_demo_review.md](/C:/Users/Administrator/.config/superpowers/worktrees/medfusion/doctor-interest-map/docs/slides/three_phase_ct_mvi_demo_review.md)

## 3. 解释性与图像叠加相关文档

这些文档和当前主题不是一回事，但和“如何把模型关注区展示给医生”直接相关，属于需要并行参考的资料。

- 原始切片叠加图实施计划：
  [2026-03-31-original-slice-overlay-implementation-plan.md](/C:/Users/Administrator/.config/superpowers/worktrees/medfusion/doctor-interest-map/docs/superpowers/plans/2026-03-31-original-slice-overlay-implementation-plan.md)

## 4. 文档分工

建议以后按下面的规则维护，避免再次混乱：

- `docs/superpowers/specs/`
  - 放设计和方案决策
- `docs/superpowers/plans/`
  - 放可执行 implementation plan
- `docs/superpowers/topics/doctor-interest-map/`
  - 放该主题的导航页、索引页、汇总页
- `docs/slides/`
  - 放汇报稿、演示稿、复盘稿，不作为实现真源

## 5. 当前推荐阅读顺序

如果是第一次接手这个主题，按这个顺序看：

1. [2026-04-17-doctor-interest-map-design.md](/C:/Users/Administrator/.config/superpowers/worktrees/medfusion/doctor-interest-map/docs/superpowers/specs/2026-04-17-doctor-interest-map-design.md)
2. [2026-04-17-doctor-interest-map-implementation-plan.md](/C:/Users/Administrator/.config/superpowers/worktrees/medfusion/doctor-interest-map/docs/superpowers/plans/2026-04-17-doctor-interest-map-implementation-plan.md)
3. [2026-03-31-smurf-feature-network-design.md](/C:/Users/Administrator/.config/superpowers/worktrees/medfusion/doctor-interest-map/docs/superpowers/specs/2026-03-31-smurf-feature-network-design.md)
4. [2026-03-31-original-slice-overlay-implementation-plan.md](/C:/Users/Administrator/.config/superpowers/worktrees/medfusion/doctor-interest-map/docs/superpowers/plans/2026-03-31-original-slice-overlay-implementation-plan.md)
5. [three_phase_ct_mvi_demo_review.md](/C:/Users/Administrator/.config/superpowers/worktrees/medfusion/doctor-interest-map/docs/slides/three_phase_ct_mvi_demo_review.md)

## 6. 当前执行入口

本轮实现默认在独立 worktree 中进行：

- 分支：`codex/doctor-interest-map`
- 工作区：
  `C:\Users\Administrator\.config\superpowers\worktrees\medfusion\doctor-interest-map`

执行方式：

- `subagent-driven`
- 先按 implementation plan 的 Task 1 开始
