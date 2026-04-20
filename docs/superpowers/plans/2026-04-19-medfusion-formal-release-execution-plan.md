# 2026-04-19 MedFusion 正式版执行分解计划

## 目标

把路线图里的“大方向”压成一条能按仓库逐步落实的执行链，并优先落 Phase 0 / Phase 1 的最小正式版切片：

1. 固定正式版定义与边界
2. 固定正式版默认入口与页面职责
3. 让模型搭建前台先从“问题定义”进入
4. 让结果后台继续围绕真实 run 的回流与展示

## 当前执行原则

- `GUI-first for users, engine-first internally`
- `Web-first`，但不重新发明 runtime
- 默认模式先走：组件介绍 -> 问题定义 -> 模板骨架 -> 参数编辑 -> 训练 -> 结果
- 节点式编辑当前是高级模式，不抢默认首页
- 先扩 runtime，再扩前台

## Phase 0：先把正式版定义钉死

### Step 0.1 固定一句话定义

- 动作：
  - 在 README、入口页、对外材料里统一正式版定义
  - 明确正式版是 GUI-first 的模型搭建主链，不是泛平台、不是真正 AI Native 成品
- 主要文件：
  - `README.md`
  - `docs/contents/guides/core/oss-go-to-market-checklist.md`
  - `web/frontend/src/pages/GettingStarted.tsx`
- 验收：
  - 用户第一次进入时，能用同一套说法理解产品

### Step 0.2 固定默认路径与高级模式边界

- 动作：
  - 明确 `/start` 是正式版入口
  - 明确 `Run Wizard` 是默认模型搭建入口第一阶段
  - 明确节点式编辑是高级模式，不作为默认首页
- 主要文件：
  - `web/frontend/src/pages/GettingStarted.tsx`
  - `web/frontend/src/pages/RunWizard.tsx`
  - `web/frontend/src/pages/Workbench.tsx`
  - `med_core/web/api/system.py`
- 验收：
  - 默认首页不再把 quickstart 或 workflow 误讲成正式版前台

### Step 0.3 固定页面职责

- 动作：
  - 给 `/start`、`/config`、`/training`、`/models`、`/workbench` 分别下定义
  - 继续让 workflow/experiments/preprocessing 退居实验边缘
- 主要文件：
  - `web/frontend/src/config/navigation.tsx`
  - `web/frontend/src/pages/GettingStarted.tsx`
  - `web/frontend/src/pages/Workbench.tsx`
  - `web/frontend/src/pages/ModelLibrary.tsx`
- 验收：
  - 用户能判断“去哪搭模型”“去哪看结果”“去哪看总览”

## Phase 1：把模型搭建前台做实

### Step 1.1 把入口改成组件介绍页

- 动作：
  - 首页先展示正式版有哪些组件
  - 明确每个组件负责什么
  - 主按钮先去问题向导，不再先去字段表单或空白画布
- 主要文件：
  - `web/frontend/src/pages/GettingStarted.tsx`
  - `web/frontend/src/config/startExperience.ts`
- 验收：
  - 首页先讲组件和主链，再引导进入搭建

### Step 1.2 把 Run Wizard 改成问题向导

- 动作：
  - 第一步先问“你要解决什么问题”
  - 根据问题路径映射到受支持的 preset / 模型骨架
  - 保留当前 runtime 边界说明，不假装支持任意模型空间
- 主要文件：
  - `web/frontend/src/pages/RunWizard.tsx`
  - `web/frontend/src/utils/runSpec.ts`
- 验收：
  - 用户不是先看到字段，而是先看到问题路径和推荐骨架

### Step 1.3 默认模式与高级模式分层

- 动作：
  - 默认模式保留参数级编辑
  - 高级模式仅保留为节点式结构编辑入口
  - 文案上明确“当前不是任意模型 builder”
- 主要文件：
  - `web/frontend/src/pages/RunWizard.tsx`
  - `web/frontend/src/pages/GettingStarted.tsx`
  - `docs/contents/getting-started/model-creation-paths.md`
- 验收：
  - 前台默认路径对非技术用户更轻，不再强迫用户理解空白节点画布

### Step 1.4 模板库与常见路径收口

- 动作：
  - 把 quickstart / clinical baseline / result audit 这种高频路径固定成可解释模板
  - 后续再扩展为更清楚的模板库
- 主要文件：
  - `web/frontend/src/utils/runSpec.ts`
  - `configs/starter/`
  - `configs/public_datasets/`
- 验收：
  - 新手、常规研究者、结果审查路径至少各有一个稳定起点

## Phase 2：图编译层与结果后台

### Step 2.1 组件注册表与约束系统

- 动作：
  - 定义前台可见组件集合
  - 定义连接约束和非法组合
  - 定义图什么时候只能保存草稿，什么时候可以训练
- 主要文件：
  - `med_core/web/workflow_engine.py`
  - `med_core/web/node_executors.py`
  - `web/frontend/src/pages/WorkflowEditor.tsx`
  - 后续新的正式版节点编辑页
- 依赖：
  - Phase 0 / Phase 1 先把默认入口收口

### Step 2.2 图到配置的编译层

- 动作：
  - 把图结构降级编译到当前 runtime 支持的配置
  - 把图校验错误翻成人类可读提示
- 主要文件：
  - `med_core/web/routers/workflow.py`
  - `med_core/configs/base_config.py`
  - `web/frontend/src/api/workflow.ts`

### Step 2.3 把 Model Library 收成结果后台

- 动作：
  - 把 `import-run` 固定为结果回流标准动作
  - 把 summary / validation / report / artifacts 的展示层级固定下来
- 主要文件：
  - `web/frontend/src/pages/ModelLibrary.tsx`
  - `web/frontend/src/components/model/ModelResultPanel.tsx`
  - `med_core/web/model_registry.py`
- 验收：
  - 任意真实 run 都能回流并稳定展示

## Phase 3：发布级补齐

### Step 3.1 安装与启动路径统一

- 动作：
  - 补齐 Windows / Linux / Docker 的主链说明
  - 补齐前端资源安装与分发逻辑
- 主要文件：
  - `med_core/web/cli.py`
  - `web/README.md`
  - `docs/contents/getting-started/web-ui.md`
  - `docs/contents/tutorials/deployment/docker.md`

### Step 3.2 API 文档与真实实现对齐

- 动作：
  - 清理 Web API 文档和实际能力之间的错位
  - 把实验态能力明确标成 experimental
- 主要文件：
  - `med_core/web/api/*`
  - `docs/contents/api/web_api.md`

### Step 3.3 做正式版 smoke path

- 动作：
  - 校验安装
  - 校验 `medfusion start`
  - 校验向导导出配置
  - 校验训练
  - 校验结果回流
- 验收：
  - 至少 Windows 和 Docker 各有一条可复现主链

### Step 3.4 全平台安装 / 部署 / 卸载闭环（Windows 优先）

- 动作：
  - 定义 Windows 的发布级最小合同：`install -> start -> smoke -> uninstall`
  - 当前 Windows 主线不采用脚本安装作为推荐路径，优先手工命令流程
  - 把 Linux / Docker 的对等合同放入后置里程碑，不阻塞当前 Windows 主线
  - 明确“卸载”不是删除程序文件就结束，而是要区分保留数据与彻底清理两种模式
  - 将安装、部署、卸载步骤沉淀为命令入口 + 文档入口 + smoke 验收入口三位一体
- 主要文件：
  - `docs/roadmap/oss/platform-install-deploy-uninstall-plan.md`
  - `docs/contents/getting-started/installation.md`
  - `docs/contents/getting-started/web-ui.md`
  - `docs/contents/tutorials/deployment/docker.md`
  - `scripts/release_smoke.py`
- 验收：
  - 当前阶段：Windows 路径能稳定完成安装、启动、一次最小 smoke、可选保留数据卸载、彻底清理卸载
  - 后置阶段：Linux / Docker 再补同语义的可复现安装与卸载路径

## Phase 4：市场与演示包装

### Step 4.1 README 首屏收口

- 动作：
  - 统一一句话定位
  - 统一主卖点顺序
  - 明确正式版与 Pro、AI Native 的关系

### Step 4.2 做固定演示脚本与素材

- 动作：
  - 首页截图
  - 向导截图
  - 结果页截图
  - artifact 结构图
  - 3 分钟演示脚本

## 本次 rollout 已开始落实的切片

- [x] 建立正式版执行分解计划
- [x] 把 `/start` 收口成组件介绍 + 默认路径说明
- [x] 把 `Run Wizard` 第一阶段改成问题优先的骨架入口
- [x] 把 `Model Library` 和 `ModelResultPanel` 收到正式版结果后台叙事
- [x] 给高级模式补上组件注册表、连接约束和 `/config/advanced` 承接页
- [x] 把高级模式推进到受约束的节点图原型（`/config/advanced/canvas`）
- [x] 给节点图原型补上图到 RunSpec/YAML 的前端编译草案
- [x] 把 Web 后端开始拆成 API/BFF 与 Python worker 两层代码结构
- [x] 把本地 `data/mock` fixture 真正补进仓库，打通默认 quickstart/mock 路径
- [x] 把高级模式图编译层下沉成 FastAPI API，并接回前端节点图原型
- [x] 给高级模式编译结果补上 ExperimentConfig contract 校验与 mainline contract 摘要
- [x] 让高级模式可以直接创建真实训练任务，并把完成后的结果 handoff 到结果后台
- [x] 把高级模式来源信息（source_context / blueprint）带进结果详情页
- [x] 用前端测试把新叙事钉住
- [x] 下一步继续把结果详情与 artifact 展示口径同步到文档入口
- [x] 下一步把高级模式的训练结果回流做成更强的结果详情与文档演示路径
- [x] 下一步把全平台安装 / 部署 / 卸载（Windows 优先）纳入正式路线图并形成完整规划
- [x] 下一步修复 Windows `medfusion start` 启动兼容问题（非 UTF-8 控制台 + SPA 路由回退）
