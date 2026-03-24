# SMuRF E2E Demo（先跑通版）

这个目录现在支持两种用法：

1. **原始多分支模式**（更接近论文结构）
2. **单 CT 分支模式**（适合当前只有 2D CT 图像的数据）

---

## 这套流程会给你什么结果

主要有两类输出：

- **分类结果**：比如高风险/低风险，包含 `pred` 和 `confidence`
- **连续风险分数**：`smurf_score`（0 到 1，越高代表风险越高）

如果你打开生存分析（`enable_survival: true`），还会额外给出生存相关结果（如 C-index、生存曲线）。

---

## 先给医生看的“直白解释”

可以把它理解成：

- 输入：病人的影像 + 一些基础临床信息
- 输出：一个风险分数和一个分组结果
- 附加：告诉你模型主要看了哪些信息（解释图）

它不是替代医生判断，而是一个“辅助分层工具”。

---

## 单 CT 分支模式（推荐你现在先用这个）

你当前数据只有单张 2D CT，所以建议先用单分支模式。

### 需要的 CSV 列（最少）

- `label`：分类标签（0/1）
- `split`：train / val / test
- `ct_paths`：CT 序列路径（支持单路径、`|` 分隔、JSON 数组）
- 表格列：`age`, `bmi`, `crp`, `sex`, `smoking`

> 生存列 `survival_time`、`event` 可以先留着但不开启。

### 对应配置

使用：`demo/smurf_e2e/config.elbow_single_ct.yaml`

关键开关：

- `data.single_branch_mode: true`
- `data.ct_column: ct_paths`
- `data.enable_survival: false`（先求稳定跑通）

---

## elbow 数据 dry run（当前可直接复现）

我们准备了一个适配脚本：

- `demo/smurf_e2e/build_elbow_dryrun.py`

图像主要来源已设为：

- **`imgs/cropped_748/coronal/*.jpg`（优先）**
- `imgs/manual_preprocessed/coronal/*.jpg`（兜底）

### 一键顺序

```bash
uv run python demo/smurf_e2e/build_elbow_dryrun.py
uv run python demo/smurf_e2e/smurf_e2e.py --config demo/smurf_e2e/config.elbow_single_ct.yaml train
uv run python demo/smurf_e2e/smurf_e2e.py --config demo/smurf_e2e/config.elbow_single_ct.yaml evaluate
uv run python demo/smurf_e2e/smurf_e2e.py --config demo/smurf_e2e/config.elbow_single_ct.yaml report
```

---

## 输出文件在哪里看

以单 CT 模式为例：`demo/smurf_e2e/outputs/elbow_single_ct/`

- `checkpoints/best_smurf_multiregion.pth`：模型参数
- `history.json`：训练过程
- `metrics.json`：主要指标
- `predictions.csv`：每个样本的预测结果（含 `smurf_score`）
- `analysis_summary.json`：解释分析汇总
- `reports/smurf_e2e_report.md`：可直接读的报告

可视化图一般在：

- `visualizations/attention/`
- `visualizations/shap/`
- `visualizations/survival/`（仅在开启生存分析时）

---

## 和论文原版 SMuRF 的差异（务必说明）

论文原版使用了：

- CT 两个区域（原发灶 + 淋巴结）
- 病理 WSI（含 HIPT 多尺度）
- 生存和分级双任务

你当前这版是“受限复现”：

- 目前只有单 CT 分支 + 临床表格
- 没有 WSI，所以不能复现病理相关那部分图
- 但流程可完整跑通，并能产出风险分数、ROC、SHAP、注意力图（CT）等可解释结果

这版适合先做流程验证和临床沟通，后续再逐步补齐多区域/多模态。

---

## 原始多分支模式（保留兼容）

如果你有完整多模态数据，仍然可以使用原来的 `config.yaml` 走多分支流程。
