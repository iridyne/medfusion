# OSS 对外推广准备清单

这份清单只服务一个目标：

**把 `oss` 以正确、克制、可复现的方式推向市场。**

不要把它讲成“什么都做完的平台”，而是讲成：

> 一个面向医学 AI 研究验证的核心运行时。  
> 能把训练、结果、validation、报告稳定串起来。

---

## 1. 当前最稳的对外定位

建议统一成下面这句：

> MedFusion OSS 是一个面向医学 AI 研究验证的可执行核心运行时，强调真实训练、真实结果和结构化 validation / report 输出。

可以扩展，但不要偏离这个中心。

### 可以重点讲的关键词

- 医学 AI 研究验证
- 真实训练闭环
- 真实结果闭环
- validation / report 结构化输出
- CLI + Web 协同
- 公开数据集快速验证入口
- 可复现、可审计、可沉淀 artifact

### 暂时不要讲太满的关键词

- 完整 benchmark 平台
- 完整公开数据中心
- 完整拖拽式建模平台
- 医生端成熟产品
- 已完成的全流程商业化平台

---

## 2. 对外内容里的主卖点顺序

建议顺序：

1. **真实训练**
2. **真实结果**
3. **真实 validation / report**
4. **公开数据集快速试跑**
5. **CLI / Web / 上层产品都能复用这套 runtime**

不要一上来先讲 builder、历史 demo、各种实验分支。

---

## 3. 最推荐的演示路径

### 路径 A：有数据用户

```bash
uv run medfusion validate-config --config configs/starter/quickstart.yaml
uv run medfusion train --config configs/starter/quickstart.yaml
uv run medfusion build-results \
  --config configs/starter/quickstart.yaml \
  --checkpoint outputs/quickstart/checkpoints/best.pth
```

适合讲：

- 配置体检
- 真实训练
- 真实结果沉淀
- 结果 artifact 可被结果页 / 模型库 / 上层产品消费

### 路径 B：无私有数据用户

```bash
uv run medfusion public-datasets list
uv run medfusion public-datasets prepare uci-heart-disease --overwrite
uv run medfusion train --config configs/public_datasets/uci_heart_disease_quickstart.yaml
```

适合讲：

- 没有私有数据也能先试
- 新用户可以先跑通最小闭环
- 公开数据集入口不是散落脚本，而是正式 CLI

---

## 4. README / 文档 / 发布文案要统一的句子

### 推荐说法

- MedFusion OSS 不是只给模型研究看代码的仓库，而是一套可执行研究运行时。
- 它当前最稳的价值是把训练到结果的闭环做实。
- 它适合研究验证、课题复现、对外演示和上层产品承接。
- 它已经提供第一批公开数据集快速验证入口。

### 不推荐说法

- 已经做成完整医学 AI 平台
- 已经覆盖大量 benchmark 一键跑
- 已经完成可视化拖拽建模主线
- 已经是医生可直接用的成熟产品

---

## 5. 发布前的最小检查清单

### 信息层

- [ ] README 首屏只讲当前主线，不混入 builder 叙事
- [ ] 文档首页给出新用户阅读顺序
- [ ] public-datasets 入口有明确文档
- [ ] examples 边界写清楚，不和主链混淆

### 运行层

- [ ] `validate-config -> train -> build-results` 路径可复现
- [ ] `public-datasets -> train` 路径可复现
- [ ] 输出目录结构稳定
- [ ] `metrics/metrics.json / metrics/validation.json / reports/summary.json / reports/report.md` contract 稳定

### 演示层

- [ ] 准备一条 3 分钟视频 / 录屏路径
- [ ] 准备一张结果页截图
- [ ] 准备一张 `report.md` / summary 截图
- [ ] 准备一张 artifact 目录结构图或说明

---

## 6. 当前阶段最合理的市场叙事

如果要一句话概括当前阶段：

> 先推广“研究运行时 + 真实结果闭环”，  
> 不要过早推广“完整平台 + 完整 builder + 完整公开验证中心”。

这是最稳的讲法，也最不容易把预期抬过头。

---

## 7. 下一阶段可以继续补什么

接下来如果继续往市场准备靠，可以优先补：

1. README 首屏继续压缩成更强的产品叙事
2. 公开数据集 quick validation 再补 1-2 个 profile
3. 工作台接入“先试公开数据”路径
4. 做一份固定 demo 脚本 / 演示路线
5. 准备一版面向 X / README / Slide 的统一素材口径
