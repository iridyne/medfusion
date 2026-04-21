# MedFusion 发布前清单

> 文档状态：**Beta**

这页不讨论“如何做一个通用推理服务”，而是回答更具体的问题：

**如果要把当前 MedFusion 正式版往外发，发布前到底要检查什么。**

当前正式版的推荐底座已经明确：

- 前端：`React + TypeScript + Vite`
- API/BFF：`FastAPI`
- 训练执行：独立 `Python worker / subprocess`
- 执行真源：`runtime / config`
- 节点图：前台交互和结构编辑层，不是执行真源

所以这份清单也围绕这条链路展开，而不是围绕一个假设中的 Node/SSR/纯推理产品。

---

## 1. 先确认你要发的是哪种形态

当前建议区分 3 种形态：

### 本机浏览器模式

- React 构建产物由 FastAPI 提供
- FastAPI 同时承担 API/BFF
- 本地 Python subprocess worker 执行训练
- SQLite 存 metadata
- 本地文件系统存 artifacts
- 用户自定义模型模板默认落在当前用户数据目录下的本地文件系统

这是当前最推荐、最完整、最容易验证的正式版形态。

### 私有服务器 / 自建部署模式

- 静态前端可独立部署
- FastAPI 继续做 API/BFF
- Python worker 独立部署到 GPU 主机
- PostgreSQL 存 metadata
- 对象存储或共享文件系统存 artifacts
- 用户自定义模型来源默认仍按“用户私有来源”处理，而不是一开始就升级成共享模型库

这条线的重点不是换技术栈，而是把 Web/API 和训练执行拆到清楚的进程边界。

### 托管云模式

- 静态前端 + 网关 / CDN
- FastAPI API/BFF
- 多 Python worker
- PostgreSQL + S3 / OSS / MinIO

这是未来方向，不是当前必须先补完的商业化形态。

---

## 2. 不要在发布前摇摆的技术判断

发布前先把这些判断定死：

- 不引入 Node 后端
- 不把训练直接跑在 Web 进程里
- 不把节点图当执行真源
- 不把 workflow editor 当默认首页
- 不把高级模式当成“任意模型都能画出来”的卖点
- 不把用户自定义模型默认做成浏览器 localStorage 缓存能力
- 不把自定义模型一开始就做成团队共享模型库

如果这些判断还在摇摆，就不是真正进入发布阶段。

---

## 3. 信息层检查

发布前先检查说法是否统一：

- [ ] README 首屏已经说明 `GUI-first for users, engine-first internally, Web-first for deployment`
- [ ] README 已明确 3 种部署形态
- [ ] `medfusion start` 仍然是默认推荐入口
- [ ] Web UI 文档已经说明 `FastAPI BFF + Python worker`
- [ ] 高级模式的边界已清楚：注册表 / 连接约束 / 图编译 / contract 校验 / 真正训练
- [ ] 模型数据库已经说明“官方最小组织单元 + 官方成品模型 + 用户私有自定义模型来源”的关系
- [ ] 独立评估模块已经说明“训练后独立补跑评估与结果构建”，而不是新的默认训练入口
- [ ] 对外 demo 路径已经说明“高级模式来源链会进入结果详情页”

如果信息层不统一，发布时用户会把正式版误解成：

- 另一套 Node 后端产品
- 一个纯空画布 builder
- 一个只会给 checkpoint、不负责结果交付的研究脚手架

---

## 4. 运行层检查

### A. 默认主链

至少保证下面这条链能跑通：

```bash
uv run medfusion start
```

然后确认：

- [ ] 能进入 `Getting Started`
- [ ] 能进入 `Run Wizard`
- [ ] 能进入 `Training Monitor`
- [ ] 能进入 `Model Library`

### B. YAML 主链

至少保证下面这条 CLI 主链能跑通：

```bash
uv run medfusion validate-config --config configs/starter/quickstart.yaml
uv run medfusion train --config configs/starter/quickstart.yaml
uv run medfusion build-results \
  --config configs/starter/quickstart.yaml \
  --checkpoint outputs/quickstart/checkpoints/best.pth
```

并确认：

- [ ] `validate-config` 会打印 mainline contract
- [ ] `outputs/` 结构稳定
- [ ] `summary.json / validation.json / report.md` 都落盘

### C. 高级模式主链

至少保证下面这条正式版高级模式链能走通：

1. 进入 `/config/advanced`
2. 打开 `/config/advanced/canvas`
3. 选择一条 compile-ready blueprint
4. 通过后端 `contract` 校验
5. 直接创建真实训练任务
6. 训练完成后跳到结果后台
7. 结果详情显示来源信息：
   - `source_type=advanced_builder`
   - `entrypoint=advanced-builder-canvas`
   - `blueprint_id`
   - `recommended_preset`
   - `compile_boundary`

这条链如果走不通，就不能对外讲“高级模式已经进入正式版 preview”。

---

## 5. 结果交付检查

正式版不是只验证“训练会不会动”，还要验证“结果能不能交付”。

至少确认：

- [ ] 训练完成后，job 状态里能拿到 `result_model_id`
- [ ] 训练页能直接跳结果详情
- [ ] Model Library 能承接训练 deep link
- [ ] `/evaluation` 能基于现有 `config + checkpoint` 独立生成结果，并可选导入结果后台
- [ ] 结果详情按四层展示：
  - 结论层
  - 指标层
  - 可视化层
  - 文件层
- [ ] 结果详情里能看到来源链信息

如果训练完成后用户还得自己去文件夹找产物，这条正式版链就还没闭环。

---

## 6. 部署检查

### 本机浏览器模式

- [ ] `medfusion start` 一条命令可起
- [ ] SQLite 正常写入 job / model metadata
- [ ] 本地 Python worker 能拉起训练
- [ ] 结果能落回本地 artifacts
- [ ] 用户自定义模型已实际落盘到本机用户数据目录，而不是仅存在浏览器缓存
- [ ] 自定义模型文件带 `schema_version`，并明确当前仅作为内部状态文件使用
- [ ] 自定义模型目录已接入本地 git 快照历史，可查看和恢复保存记录
- [ ] 恢复历史版本时已明确采用“覆盖当前模型并写入新的本地 git 记录”的交互
- [ ] 历史保留策略已明确为“全局一套”，默认按条数保留最多 40 条，并尽量保证当前仍存在的模型至少保留最近若干条历史
- [ ] 回收站当前只提供恢复入口，不提供手动彻底清除入口
- [ ] UI 偏好设置已明确为机器级设置，不做多用户拆分
- [ ] 语言、主题和历史显示方式都已切到机器级本地文件设置，不再依赖浏览器缓存作为主存储
- [ ] `friendly / technical` 显示方式当前只作用于自定义模型的版本历史相关页面
- [ ] 设置页已提供“恢复默认设置”，便于客户在本机快速回到默认偏好
- [ ] 旧版本里残留在浏览器缓存中的语言 / 主题设置，已支持一次性迁移到机器级设置文件

### 私有服务器 / 自建部署模式

- [ ] Docker 文档说明仍以 `FastAPI BFF + Python worker` 为中心
- [ ] 没有把 Docker 讲成另一套产品
- [ ] 已明确前端/API/worker 的拆分建议
- [ ] 已明确 PostgreSQL / 对象存储是推荐方向，而不是强制本轮完成项
- [ ] 已明确用户自定义模型默认还是用户私有来源，后续再决定是否进入共享模型库

### 托管云模式

- [ ] 当前只作为方向说明，不夸大为已完成能力
- [ ] 文档里没有暗示“已经多租户上线”

---

## 7. 测试检查

至少确认这些测试或等价 smoke path 已通过：

- [ ] Web/API 边界测试
- [ ] 训练控制 API 测试
- [ ] 高级模式 API 测试
- [ ] Web 最小闭环测试
- [ ] 前端结果 handoff / 结果详情测试
- [ ] 文档入口一致性测试

这一步的重点不是“测试越多越好”，而是：

**主叙事、主入口、主链路都已经有自动化约束。**

---

## 8. 当前不该当发布阻塞项的东西

不要把这些东西误判成“发布前必须先做完”：

- 一个新的 Node BFF
- SSR / Next.js 改造
- 泛聊天框
- 任意模型自由拖拽
- workflow editor 升格为默认首页
- 完整云商业化能力
- 团队共享模型库
- 浏览器 localStorage 级别的模型模板持久化方案
- 用户自定义模型的正式导入 / 导出资产格式
- 团队共享的自定义模型版本库
- 多模型对比 / 批量评估中心
- 评估模板中心

这些都可能是后续演进方向，但不是当前正式版主链的先决条件。

---

## 9. 最小发布结论

如果你要判断“现在能不能对外发一个正式版 preview”，可以用下面这个标准：

### 可以发的最低条件

- 默认入口已经清楚
- 高级模式边界已经清楚
- `FastAPI BFF + Python worker` 方向已经清楚
- 默认主链、高级模式主链、结果交付主链已经打通
- 文档说法与代码现状一致

### 还不能发的信号

- 默认入口还在摇摆
- 高级模式还只是空画布
- 结果后台还不能解释来源链
- 文档还在混用旧架构和新架构
- 发布口径里还在暗示 Node 后端 / SSR / 任意模型 builder

---

## 10. 推荐连读

- [Web UI 快速入门](../../getting-started/web-ui.md)
- [Web UI 架构](../../architecture/WEB_UI_ARCHITECTURE.md)
- [Docker 部署指南](docker.md)
- [正式版 Smoke Matrix](../../playbooks/release-smoke-matrix.md)
- [对外 Demo 路径](../../playbooks/external-demo-path.md)
