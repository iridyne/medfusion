# SMURF Demo Feature Network Design

日期：2026-03-31

状态：Approved for implementation

范围：`configs/demo/three_phase_ct_*.yaml` 对应的主线 three-phase CT + clinical demo

## 1. 背景与目标

当前 `smurf` demo 已经并入主线运行路径：

- `medfusion validate-config --config configs/demo/three_phase_ct_*.yaml`
- `medfusion train --config configs/demo/three_phase_ct_*.yaml`
- `medfusion build-results --config configs/demo/three_phase_ct_*.yaml`

当前版本可以完成最小训练和结果产物，但特征网络仍然偏 baseline：

- 三期 CT 编码器过浅，整体更像 smoke demo 而不是可信医学建模方案
- 三期之间仅支持简单 `concatenate` / `mean`
- 临床变量分支过弱，缺少规范的预处理与缺失机制
- 模型没有为后续热力图和病例级解释显式保留中间表示
- 对医生展示来说，结论可以读，但“判断依据”仍然偏薄

本设计的目标不是追求 SOTA，而是把当前 demo 提升到“医生展示友好、结果更稳、后续可解释性可扩展”的形态。

## 2. 设计原则

本次改造遵守以下边界：

- 不新增 demo 专用训练入口脚本
- 不改变主线 CLI 使用方式
- 继续由 YAML 驱动运行
- 通用能力进入 `med_core/`，不带业务专名
- 优先让模型结构和产物更像严肃医学 demo，而不是引入过重的研究原型堆料

实现上的核心原则：

- 轻量升级，不做激进重构
- 先为解释性设计结构，再增强建模细节
- 所有新增输出都应服务于病例级解释和结果展示
- 保持向后兼容：已有 `three_phase_ct_*` 配置应能迁移或小改后继续运行

## 3. 当前实现概况

### 3.1 数据输入

当前数据集实现位于：

- `med_core/datasets/three_phase_ct.py`

输入为单病例级别的三期 CT + 临床特征：

- `arterial`
- `portal`
- `noncontrast`
- `clinical`
- `label`

影像预处理当前包含：

- DICOM 读取
- HU 转换
- `window_preset` 窗宽窗位
- slice 排序
- 插值到统一 `target_shape`

临床变量当前仅做：

- CSV 读列
- 缺失值补 `0.0`
- 转 float

### 3.2 模型

当前模型位于：

- `med_core/models/three_phase_ct_fusion.py`

结构为：

- 每一期单独一个浅层 3D encoder
- 得到三期向量
- `concatenate` 或 `mean` 融合
- 临床 MLP 编码
- 多模态融合
- 分类头 + 可选 `risk_head`

### 3.3 结果产物

当前 `build-results` 已支持：

- ROC / AUC
- confusion matrix
- SHAP-style surrogate 临床变量重要性
- `summary.json`
- `report.md`
- `predictions.json`

但影像内部解释仍为空白，尚未形成：

- 单期贡献
- 跨期贡献
- 热力图预留接口
- 病例级解释数据结构

## 4. 总体方案

本方案采用“轻量版 3”路线：

- 保留现有主线入口与 YAML 驱动方式
- 重构特征网络组织方式
- 为热力图和解释性保留模型中间输出
- 增强临床分支的稳定性
- 把结果产物升级为更适合医生阅读的结构

整体结构升级为四层：

1. `phase encoder`
2. `phase fusion`
3. `clinical encoder`
4. `decision head`

## 5. 模型设计

### 5.1 Phase Encoder

每一期 CT 使用同构的 3D 编码器，替代当前过浅的 `_PhaseEncoder`。

目标：

- 保持轻量，不引入重型 backbone
- 输出稳定的期相特征向量
- 暴露最后卷积特征图，供后续 Grad-CAM / activation map 使用
- 支持共享和不共享参数

建议结构：

- `Conv3d -> Norm -> ReLU`
- `Conv3d -> Norm -> ReLU -> Downsample`
- `Conv3d -> Norm -> ReLU`
- `AdaptiveAvgPool3d`
- `Linear projection`

建议新增输出：

- `phase_embedding`
- `phase_feature_map`

其中：

- `phase_embedding` 用于训练、融合、结果分析
- `phase_feature_map` 用于热力图和后续影像解释

### 5.2 Phase Fusion

当前三期之间只有 `concatenate` 和 `mean` 两种融合，表达过于粗糙。

改造目标：

- 保留 `concatenate` 作为基线与回退方案
- 新增 `gated phase fusion` 作为默认推荐
- 输出可解释的期相贡献权重

建议机制：

- 三期各自产生 `phase_embedding`
- 通过轻量 gating 网络得到三期权重
- 以加权融合或 gated concat 形式生成 `phase_features`

新增输出：

- `phase_importance`

该输出用于后续展示：

- 该病例判断更依赖哪一期
- 三期贡献是否均衡

### 5.3 Clinical Encoder

临床分支升级重点不在“变深”，而在“更像规范医疗建模”。

建议改动：

- 增加训练期标准化
- 显式保留缺失掩码
- 允许临床输入由 `原始值 + 缺失标记` 共同组成

建议输入组成：

- `clinical_values`
- `clinical_missing_mask`

建议输出：

- `clinical_features`

这样可以解决两个问题：

- 单纯补 `0.0` 带来的伪信号
- 医生质疑“缺失化验是否被错误当成真实值”时缺少解释

### 5.4 Decision Head

保留二分类主任务，但要求输出结构为解释性服务。

建议最终输出：

- `logits`
- `probability`
- `risk_score`
- `phase_features`
- `clinical_features`
- `fused_features`
- `phase_importance`
- `feature_maps`

其中：

- `feature_maps` 为后续热力图提供输入
- `phase_importance` 为结果页提供期相依据
- `clinical_features` / `fused_features` 为病例级解释和 debug 提供中间信息

## 6. 数据预处理设计

### 6.1 保留项

继续保留当前影像预处理主干：

- HU 转换
- 窗宽窗位
- slice 排序
- 统一形状插值

### 6.2 新增项

新增以下规范化能力：

- `clinical normalization`
- `clinical missing mask`
- `phase quality checks`

说明：

- `clinical normalization` 训练时拟合统计量，推理与 build-results 复用
- `clinical missing mask` 明确哪些字段缺失
- `phase quality checks` 在 doctor / build-results 阶段报告异常切片数、缺失目录、体积形状异常

### 6.3 暂缓项

以下能力不纳入本阶段：

- ROI 分割
- lesion crop
- radiomics 手工特征
- 配准级跨期对齐
- 重型预训练医学 backbone

这些能力有价值，但超出当前 demo 的稳妥改造范围。

## 7. 可解释性设计

本设计把可解释性分成三层。

### 7.1 Clinical Importance

继续复用现有：

- SHAP-style surrogate feature importance

但对外话术应统一为：

- 关键影响因素
- 临床变量贡献

### 7.2 Phase Importance

新增：

- 三期贡献权重输出

目标是让结果报告能够说明：

- 该病例主要依赖哪一期判断
- 是否存在某一期显著主导

### 7.3 Heatmap Readiness

本阶段不要求一次性做完高质量热力图展示，但模型结构必须支持：

- 对单期 CT 生成 Grad-CAM 风格热力图
- 将热力图与病例级输出关联

为此模型必须：

- 保留最后卷积特征图
- 保留可回溯到单期分支的梯度路径

## 8. 结果产物设计

### 8.1 新增结构化产物

建议在 `build-results` 中新增或扩展：

- `phase_importance.json`
- `case_explanations.json`

`case_explanations.json` 每例至少包含：

- `case_id`
- `predicted_label`
- `pred_probability`
- `risk_score`
- `phase_importance`
- `top_clinical_factors`

后续热力图接入时可再扩展：

- `heatmap_artifacts`

### 8.2 报告展示方式

`report.md` 和 Web/导出层应优先使用医生更熟悉的表达：

- 本例判断倾向
- 哪一期影像贡献更明显
- 哪些临床因素更关键
- 结果对应的风险概率

避免直接把网络术语堆给终端读者。

## 9. 配置设计

### 9.1 保持主线入口不变

仍然使用：

- `configs/demo/three_phase_ct_mvi_demo.yaml`
- `configs/demo/three_phase_ct_mvi_dr_z.yaml`

### 9.2 建议新增配置段

建议新增以下配置能力：

- `model.phase_encoder`
- `model.phase_fusion`
- `data.clinical_preprocessing`
- `explainability`

示意：

```yaml
model:
  model_type: three_phase_ct_fusion
  phase_encoder:
    base_channels: 16
    num_blocks: 3
    norm: batch
    dropout: 0.1
    share_weights: false
  phase_fusion:
    mode: gated
    hidden_dim: 64

data:
  clinical_preprocessing:
    normalize: true
    missing_value_strategy: zero_with_mask

explainability:
  export_phase_importance: true
  export_case_explanations: true
  heatmap_ready: true
```

约束：

- 对外尽量少暴露细碎调参
- 默认值应适合当前 demo 小样本场景

## 10. 需要修改的模块

预计涉及：

- `med_core/datasets/three_phase_ct.py`
- `med_core/models/three_phase_ct_fusion.py`
- `med_core/cli/train.py`
- `med_core/postprocessing/results.py`
- `med_core/configs/base_config.py`
- `med_core/configs/validation.py`

可能新增通用模块：

- `med_core/shared/preprocessing/clinical.py`
- `med_core/shared/visualization/heatmaps.py`

命名要求：

- 通用模块命名，不带 `smurf`
- 代码可服务未来其他三期 CT / 多时相影像任务

## 11. 错误处理与健壮性

本次方案要求补强以下错误处理：

- 临床特征维度与配置不一致
- 全部临床列缺失
- 某一期 DICOM 目录缺失或空目录
- 三期体积 shape 明显异常
- `phase_importance` 输出维度错误
- 热力图模式开启但模型未提供 feature map

所有错误应尽量在：

- `validate-config`
- `doctor`
- `build-results`

尽早暴露，而不是拖到训练中后段。

## 12. 测试策略

### 12.1 单元测试

新增或扩展：

- 编码器输出 shape
- `phase_importance` 合法性
- 临床 normalization / missing mask 逻辑
- feature map 导出存在性

### 12.2 集成测试

覆盖：

- 主线训练仍可跑通
- build-results 输出新增 JSON
- report 中出现三期贡献与临床因素摘要

### 12.3 回归重点

必须确认不回归：

- `device: auto`
- 训练/验证/测试 split 行为
- 现有 summary/report artifact 契约

## 13. 分阶段实施

### Phase 1：特征网络与临床分支重构

目标：

- 升级 phase encoder
- 加 clinical normalization + missing mask
- 保持训练主线可跑

### Phase 2：解释性中间输出

目标：

- 输出 `phase_importance`
- 输出 `case_explanations.json`
- 在 report 中呈现期相贡献和关键临床因素

### Phase 3：热力图接入准备

目标：

- 暴露 feature maps
- 预留 heatmap artifact 协议
- 为后续 Grad-CAM 类实现铺底

## 14. 明确不做的事情

本轮不做：

- 大型预训练 3D backbone 接入
- 分割模型联动
- 复杂跨期 transformer
- 直接面向临床部署的风险分层声明
- 过度定制某一专病话术

这保证 demo 仍然是通用 OSS 能力上的可信示范，而不是私有研究分支重新长出来。

## 15. 成功标准

完成后，demo 应满足：

- 仍可通过主线 CLI 和 YAML 直接运行
- 模型结构不再显得像纯 smoke baseline
- 结果页能回答“哪一期更重要、哪些临床因素更关键”
- 后续热力图工作不需要推倒重来
- 新增代码仍保持通用模块属性

## 16. 推荐实施顺序

推荐按以下顺序开发：

1. 重构 `three_phase_ct_fusion.py`
2. 补临床预处理模块
3. 扩展 train / build-results 契约
4. 补测试
5. 最后再接热力图实现

这样可以先把模型和结果语义打稳，再继续向可视化解释延伸。
