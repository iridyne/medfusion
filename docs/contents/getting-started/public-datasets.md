# 公开数据集快速验证清单

这份清单的目标不是覆盖所有公开医学数据集，而是先给 MedFusion 用户一批最适合做“快速验证”的数据入口。

适用场景：

- 还没有私有数据，想先确认框架能不能跑起来
- 想给导师、同学、合作方演示 MedFusion 的闭环
- 想给 GitHub README、小红书、B 站内容提供一个可复现入口

筛选原则：

1. 公开可下载
2. 社区常见，容易解释
3. 任务定义明确
4. 适合当前 MVP 的训练 -> 结果 -> 报告链路

## 推荐优先级

### P0：最快验证

先用这些数据集验证“框架能跑 + 结果页能看 + 报告能出”。

| Dataset | 官方入口 | 模态 | 任务 | 规模/门槛 | 推荐用途 |
| --- | --- | --- | --- | --- | --- |
| MedMNIST | [medmnist.com/v2](https://medmnist.com/v2) / [GitHub](https://github.com/MedMNIST/MedMNIST) | 医学图像（2D/3D） | 分类、多标签分类、序数分类 | 低门槛，可通过 `pip install medmnist` 获取 | 最适合新用户第一轮验证 |
| UCI Heart Disease | [UCI 官方页](https://archive.ics.uci.edu/dataset/45/heart+disease) | 表格 | 二分类 | 很轻量 | 适合验证 tabular 主链和基础指标输出 |

### P1：真实公开医学影像

这些更接近常见论文和公开 benchmark，但下载和清洗成本更高。

| Dataset | 官方入口 | 模态 | 任务 | 规模/门槛 | 推荐用途 |
| --- | --- | --- | --- | --- | --- |
| ISIC Challenge 2018 / 2019 | [ISIC Challenge Data](https://challenge.isic-archive.com/data/) | 皮肤镜图像 | 分类、分割 | 中等 | 适合演示医学图像分类、分割和结果图表 |
| HAM10000 | [ISIC Challenge Data](https://challenge.isic-archive.com/data/) | 皮肤镜图像 | 多分类 | 中等 | 常见皮肤镜分类入门集，传播上辨识度高 |
| NIH ChestXray14 | [NIH 下载页](https://nihcc.app.box.com/v/ChestXray-NIHCC) | 胸部 X-ray | 多标签分类 | 较大 | 适合做更像真实医学影像项目的公开验证 |

### P2：更贴近多视图 / 多模态叙事

这些不是当前 MVP 的第一落点，但很适合后续内容升级。

| Dataset | 官方入口 | 模态 | 任务 | 规模/门槛 | 推荐用途 |
| --- | --- | --- | --- | --- | --- |
| ISIC MILK10k | [ISIC Archive](https://www.isic-archive.com/) / [ISIC Challenge Data](https://challenge.isic-archive.com/data/) | 成对图像 / 多视图 | 病灶分类 | 中等 | 适合讲多视图、节点式建模和“更像多模态”的内容 |

## 推荐验证路径

### 路径 A：10 分钟内跑通

目标：最快看到训练与结果产物。

1. 用 MedMNIST 下载一个最小子集，如 `PathMNIST`、`ChestMNIST` 或 `BreastMNIST`
2. 跑一次基础训练
3. 检查是否能稳定产出：
   - 训练历史
   - ROC / AUC
   - 混淆矩阵
   - validation 摘要
   - 报告文件

### 路径 B：先验证表格能力

目标：用最轻量的数据确认结构化输入链路。

1. 使用 UCI Heart Disease
2. 先跑 tabular baseline
3. 检查：
   - accuracy / precision / recall / F1
   - threshold analysis
   - calibration summary

### 路径 C：做对外演示素材

目标：产出更适合 README、小红书、B 站的图。

1. 使用 ISIC 2018 / 2019 或 HAM10000
2. 跑图像分类任务
3. 优先保留：
   - ROC 曲线
   - 归一化混淆矩阵
   - 注意力图
   - 结果摘要页截图

## 当前建议的第一批接入顺序

1. MedMNIST
2. UCI Heart Disease
3. ISIC 2018 / 2019
4. NIH ChestXray14
5. ISIC MILK10k

这个顺序的原因很简单：

- 先降低新用户上手门槛
- 再增加医学影像内容的可信度
- 最后补更贴多视图 / 多模态叙事的数据

## 可直接复制的最短命令

### PathMNIST

适合先验证图像训练、结果页和报告产物。

```bash
uv pip install medmnist
uv run python scripts/prepare_public_dataset.py medmnist-pathmnist --overwrite
uv run medfusion train --config configs/public_datasets/pathmnist_quickstart.yaml
```

输出目录固定为：

- `data/public/medmnist/pathmnist-demo/`
- `outputs/public_datasets/pathmnist_quickstart/`

### UCI Heart Disease

适合先验证 tabular 指标链路和二分类结果展示。

```bash
uv run python scripts/prepare_public_dataset.py uci-heart-disease --overwrite
uv run medfusion train --config configs/public_datasets/uci_heart_disease_quickstart.yaml
```

输出目录固定为：

- `data/public/uci/heart-disease-demo/`
- `outputs/public_datasets/uci_heart_disease_quickstart/`

## 当前适配说明

这里需要把实现边界讲清楚。

当前 MedFusion CLI 的稳定主链还是统一的“图像 + 表格”多模态训练接口，还不是分别为 image-only / tabular-only 单独收敛好的入口。

所以第一批公开数据集 quick validation 做了两层适配：

1. `PathMNIST`
   - 不强行伪造临床表格数据
   - 直接走数据加载器的 dummy tabular fallback
   - 目标是先验证图像训练、artifact 和结果展示链路

2. `UCI Heart Disease`
   - 保留真实表格特征
   - 自动生成一张中性 placeholder 图像
   - 目标是先验证 tabular 指标、validation 和报告链路

这层适配是为了让公开数据集尽快进入当前 MVP 主链，不是最终的数据接入形态。

## README 和内容侧的使用建议

在 README 里不要一次性堆太多数据集，建议只保留：

- 一个“最快开始”的数据集入口
- 一个“表格任务”入口
- 一个“真实医学影像”入口

在小红书和 B 站内容里可以这样分工：

- 小红书：优先展示 MedMNIST、ISIC 这类画面直观、容易理解的内容
- B 站：可以展开讲 ChestXray14、MILK10k 这类更贴真实研究场景的数据

## 后续建议

下一步最好继续补三类资产：

1. 每个数据集对应的最小 demo config
2. 数据下载后的目录约定
3. README 中可直接复制的最短命令
