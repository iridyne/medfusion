# SMuRF E2E Demo (Multi-Region + Risk Score)

这个版本按你要的方向做了“双输出”：

1. **分类输出**（pred / confidence）
2. **连续风险分数**（`smurf_score`，风险头输出经 sigmoid 映射）

输入形态（可插拔）：

1. 表格化临床数据
2. 第一区域病理/CT 非连续序列
3. 第二区域病理/CT 非连续序列

病理分支支持两种编码：
- `patch_mil`：直接吃病理 patch 图像
- `hipt`：吃离线 HIPT embedding（`prepare-hipt` 生成）

## 技术核心

- 模型组装使用 `medfusion` 的 `MultiModalModelBuilder`
- 模态定义：
  - `clinical`（tabular）
  - `region1_ct`（vision + MIL）
  - `region1_pathology`（`patch_mil`: vision+MIL / `hipt`: embedding+MIL）
  - `region2_ct`（vision + MIL）
  - `region2_pathology`（`patch_mil`: vision+MIL / `hipt`: embedding+MIL）
- 融合策略默认：`fused_attention`
- 在融合特征上额外挂一个 `risk_head` 输出连续风险分
- 训练损失：`classification CE + survival_loss_weight * Cox loss`

## 非连续图像字段格式

每个序列列支持三种写法：

- 单路径：`a/b/c.npy`
- `|` 分隔：`a.npy|b.npy|c.npy`
- JSON 数组：`["a.npy", "b.npy", "c.npy"]`

脚本会均匀抽样到 `max_instances_per_modality`，不足则 padding。

## 快速跑通

```bash
bash demo/smurf_e2e/run_all.sh
```

流程：
1. `prepare-mock` 生成 mock 多区域数据（含 `survival_time/event`）
2. （可选）`prepare-hipt` 离线生成病理 embedding 并改写 CSV 路径
3. `train` 训练双头模型
4. `evaluate` 评估并导出 `smurf_score`
5. `report` 出报告

## 输出文件

在 `demo/smurf_e2e/outputs/`：

- `checkpoints/best_smurf_multiregion.pth`
- `history.json`
- `metrics.json`（含分类指标，开启 survival 时含 `c_index`）
- `predictions.csv`（含 `smurf_score`, `risk_logit`）
- `reports/smurf_e2e_report.md`

## 接入真实数据

你只需要保证 CSV 有这些列：

- 分类：`label`, `split`
- 生存：`survival_time`, `event`（可在 config 改列名）
- 表格：`tabular_numerical_columns` / `tabular_categorical_columns` 里声明的列
- 图像序列：
  - `region1_ct_paths`
  - `region1_pathology_paths`
  - `region2_ct_paths`
  - `region2_pathology_paths`

如果你用 `hipt` 模式：
1) 先把 `model.pathology_encoder.type` 设为 `hipt`
2) 运行 `prepare-hipt` 生成离线 embedding 并改写病理列

然后跑：

```bash
uv run python demo/smurf_e2e/smurf_e2e.py --config demo/smurf_e2e/config.yaml prepare-hipt  # hipt模式可先执行
uv run python demo/smurf_e2e/smurf_e2e.py --config demo/smurf_e2e/config.yaml train
uv run python demo/smurf_e2e/smurf_e2e.py --config demo/smurf_e2e/config.yaml evaluate
uv run python demo/smurf_e2e/smurf_e2e.py --config demo/smurf_e2e/config.yaml report
```
