# 如何新建模型与 YAML

> 文档状态：**Stable**

这一页只回答一个问题：

**如果你想“自己新建一个模型”或“自己写一份 YAML”，在 MedFusion 里到底应该走哪条路？**

先把结论讲清楚：

- **普通用户**：复制一份主链 config 模板，在现有组件里组合
- **高级用户**：走 Builder / 代码做结构实验
- **真正新的模型能力**：先扩 runtime，再扩 YAML

如果不先分清这三件事，很容易出现两种误解：

1. 以为任何模型都能只靠 YAML 从零发明出来
2. 以为 Web 里的向导可以替代 runtime 扩展

这两种理解当前都不成立。

## 先用 30 秒判断你属于哪条路径

### 路径 1：复制一份主链 config 模板

适合你如果：

- 你已经有自己的数据，想尽快跑通训练与结果产物
- 你要做的是现有能力范围内的组合，而不是发明全新机制
- 你希望继续使用 `validate-config -> train -> build-results` 这条官方主链

一句话理解：

**你不是在“发明新模型”，你是在“基于现有 runtime 组装一个新实验”。**

当前推荐做法：

1. 从最接近的模板开始复制
2. 修改数据路径、列名、类别数、训练参数
3. 在现有支持的 `backbone / fusion / head / trainer` 里做选择
4. 先跑 `validate-config`
5. 再跑 `train`
6. 最后跑 `build-results`

优先参考这些模板：

- `configs/starter/`
- `configs/public_datasets/`
- `configs/testing/`

这条路最适合：

- 新用户第一次上手
- 私有数据 baseline
- 对外 demo
- 结果页、报告、artifact 要一起交付的场景

## 路径 2：Builder / 代码做结构实验

适合你如果：

- 你想试验新的模态组合方式
- 你想快速做研究原型，而不是立刻接入官方 CLI 主链
- 你在比较不同结构设计，愿意接受“先做结构实验，再决定是否沉淀进主链”

一句话理解：

**你在做的是结构实验，不是直接写当前主链训练 YAML。**

当前入口主要是：

- `MultiModalModelBuilder`
- `build_model_from_config()`
- `configs/builder/`
- [模型构建器 API](../tutorials/fundamentals/builder-api.md)

这里最重要的边界是：

- `configs/builder/` 下的 YAML 更接近“模型结构描述”
- 它**不是当前 `medfusion train` 主链 YAML**
- 它不保证能直接走完整的 CLI / Web 训练闭环

所以如果你的目标是“今天就把训练、验证、结果报告一起跑通”，不要先从 Builder 路线起步。

## 路径 3：先扩 runtime，再扩 YAML

适合你如果：

- 你要的新能力在当前框架里根本不存在
- 你不是只换参数，而是要增加新的运行时能力

典型例子：

- 新的模态类型
- 新的 backbone / fusion / head / trainer 类型
- 新的数据读取契约
- 新的训练阶段逻辑
- 新的结果产物类型
- 新的 Web 向导字段与校验规则

这时正确顺序不是先写 YAML，而是：

1. 先把 runtime 能力实现出来
2. 再把它接到 config schema
3. 再补模板、文档和测试
4. 最后再考虑是否接入 Web 向导

一句话理解：

**YAML 只能配置已经存在的能力，不能凭空创造还不存在的能力。**

所以这里必须是：

**先扩 runtime，再扩 YAML。**

## Web UI / Run Wizard 的真实边界

Web 里的配置向导目前更准确的定位是：

- 一个 **RunSpec / ExperimentConfig 生成器**
- 一个帮助你填写现有主链配置的入口
- 一个让新用户少写 YAML、少踩字段坑的工具

正式版当前还额外补了一层 **高级模式注册表页**（`/config/advanced`）：

- 它先展示组件家族、连接约束和哪些骨架可编译
- 它的作用是把高级模式的边界讲清楚
- 它当前不是默认首页，也不是任意模型的可视化发明器

它现在**不是一个任意发明新模型的生成器**。

更直接地说：

- 它可以帮你生成当前 schema 内的训练配置
- 它可以帮你减少 YAML 手写成本
- 它**不会替你发明一个全新的模型能力**

如果你要的是“从 Web 上点几下就定义一种 runtime 里从没出现过的新模型”，当前并不支持。

## 新手最容易踩错的三件事

### 1. 把“写 YAML”误解成“发明新模型”

不是。

大多数情况下，你做的是：

- 复制主链模板
- 换数据
- 换现有组件
- 调训练参数

这属于“新实验配置”，不是“新增模型能力”。

### 2. 把 `configs/builder/` 当成 CLI 主链配置

也不是。

`configs/builder/` 更适合研究型结构实验。
如果你想走官方训练闭环，优先从 `configs/starter/` 或 `configs/public_datasets/` 开始。

### 3. 把 Web 向导当成无代码模型发明器

也不是。

它更像一个主链配置生成工具，而不是通用 AutoML 建模器。

## 最短行动建议

如果你现在就要开始，按这个判断：

- **我有自己的任务，只想先跑通**：复制一份主链 config 模板
- **我想做结构实验或研究原型**：走 Builder / 代码路径
- **我需要框架里还没有的新能力**：先扩 runtime，再把它暴露给 YAML

建议连读：

- [CLI 与 Config 使用路径](cli-config-workflow.md)
- [MedFusion 新手避坑指南](quickstart.md)
- [Web UI 快速入门](web-ui.md)
- [模型构建器 API](../tutorials/fundamentals/builder-api.md)
