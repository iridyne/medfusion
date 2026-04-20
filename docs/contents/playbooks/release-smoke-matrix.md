# 正式版 Smoke Matrix

> 文档状态：**Beta**

这页只回答一个问题：

**在不同部署形态下，哪些最小路径必须真的跑通，才能说当前 MedFusion 正式版可以往外发。**

这里的 smoke matrix 不是完整回归测试清单，而是“最低可运行合同”。

---

## 为什么需要这张表

如果没有 smoke matrix，团队很容易陷入三种错觉：

1. 作者本机能跑，就以为别人按文档也能跑。
2. 某个页面存在，就以为主链已经闭环。
3. 某种部署方向被写进文档，就以为已经具备可发布可信度。

这张表的作用，是把“正式版最小可运行能力”讲死。

---

## 部署形态

当前按 3 种形态来理解：

1. **本机浏览器模式**
2. **私有服务器 / 自建部署模式**
3. **托管云模式**

其中：

- 当前最推荐、最完整、最可复现的是 **本机浏览器模式**
- 当前最适合作为发布前最小外部验证的是 **本机浏览器模式 + Docker 私有部署模式**
- **托管云模式** 目前更像方向说明，不是当前发版阻塞项

---

## Smoke Matrix

| 检查项 | 本机浏览器模式 | 私有服务器 / 自建部署模式 | 托管云模式 |
|---|---|---|---|
| `medfusion start` 可启动 | 必须通过 | 可替代为 API + 静态前端部署 | 不适用 |
| `Getting Started` 可打开 | 必须通过 | 必须通过 | 必须通过 |
| `Run Wizard` 可生成配置 | 必须通过 | 必须通过 | 必须通过 |
| 高级模式注册表可打开 | 必须通过 | 必须通过 | 必须通过 |
| 高级模式节点图可编译 | 必须通过 | 必须通过 | 必须通过 |
| `ExperimentConfig` contract 校验 | 必须通过 | 必须通过 | 必须通过 |
| 能直接创建真实训练任务 | 必须通过 | 必须通过 | 必须通过 |
| 训练不跑在 Web 进程里 | 必须通过 | 必须通过 | 必须通过 |
| 训练完成后结果可回流 | 必须通过 | 必须通过 | 必须通过 |
| 结果详情可显示来源链 | 必须通过 | 必须通过 | 必须通过 |
| `FastAPI BFF + Python worker` 边界清楚 | 必须通过 | 必须通过 | 必须通过 |

---

## 当前推荐验证顺序

### 0. 统一执行入口（推荐）

优先使用仓库内统一脚本，减少手工遗漏：

```bash
# 本机浏览器模式 + YAML 主链
uv run python scripts/release_smoke.py --mode local

# Docker 私有部署模式
uv run python scripts/release_smoke.py --mode docker
```

如果你需要一次跑完两条路径：

```bash
uv run python scripts/release_smoke.py --mode all
```

### 1. 本机浏览器模式

目标：验证“正式版默认主链”。

建议最少检查：

1. 运行：

```bash
uv run medfusion start
```

2. 确认可以进入：
   - `/start`
   - `/config`
   - `/training`
   - `/models`

3. 跑通最小 CLI 主链：

```bash
bash test/smoke.sh
```

4. 确认高级模式链：
   - `/config/advanced`
   - `/config/advanced/canvas`
   - compile-ready blueprint
   - 通过 contract 校验
   - 直接创建训练任务
   - 训练完成后跳结果后台

### 2. 私有服务器 / 自建部署模式

目标：验证“部署形态变化后，主链语义没有变”。

建议最少检查：

1. 前端静态资源可被提供
2. FastAPI API/BFF 可启动
3. Python worker 可独立执行训练
4. 结果可回流到 Model Library
5. Docker 文档不引入 Node 后端假设

### 3. 托管云模式

目标：验证“文档口径正确”，不是验证“当前已经商业可用”。

建议最少检查：

1. 文档里清楚写明这是未来方向
2. 文档里没有暗示多租户、计费、认证已经成熟
3. 技术口径仍然保持 `FastAPI BFF + Python worker`

---

## 当前仓库里已经存在的 smoke 入口

### Release smoke（统一入口）

```bash
uv run python scripts/release_smoke.py --mode local
uv run python scripts/release_smoke.py --mode docker
```

它会把本机 Web 启动检查和主链 smoke 串起来，并补上 Docker 形态的最小可运行检查。

### Shell smoke

```bash
bash test/smoke.sh
```

这条脚本继续作为 YAML 主链 smoke 的底层入口（`prepare -> validate-config -> train -> build-results`）。

### Web/API 最小闭环测试

Python 侧已经有几类关键测试：

- `tests/test_web_api_minimal.py`
- `tests/test_advanced_builder_api.py`
- `tests/test_training_control_api.py`
- `tests/test_web_training_controls.py`
- `tests/test_workflow_api.py`

这些测试共同覆盖：

- API/BFF
- Python worker 编排
- 高级模式编译
- 训练控制
- 结果回流

### 前端最小闭环测试

前端当前已经有一批定向 vitest，覆盖：

- `Getting Started`
- `Run Wizard`
- `Advanced Builder`
- `Training Monitor`
- `Model Library`
- `ModelResultPanel`

---

## 当前最小发布判断

如果下面两条同时成立，可以认为当前版本已经具备“正式版 preview 可对外展示”的最低可信度：

1. **本机浏览器模式** smoke path 已经打通
2. **私有服务器 / Docker 部署口径** 已经和当前代码实现一致

如果这两条都做不到，就还不算真正进入“能发”的阶段。

---

## 推荐连读

- [发布前清单](../tutorials/deployment/production.md)
- [对外 Demo 路径](./external-demo-path.md)
- [最小可复现实验（MRE）](./minimum-reproducible-run.md)
- [Web UI 快速入门](../getting-started/web-ui.md)
