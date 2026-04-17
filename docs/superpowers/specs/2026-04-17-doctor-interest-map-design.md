# Doctor Interest Map Design

日期：2026-04-17

状态：Recommended baseline for implementation planning

范围：`configs/demo/three_phase_ct_mvi_dr_z.yaml` 对应的主线 three-phase CT + clinical MVI demo

## 1. 问题定义

当前主线已经具备：

- 三期 CT + clinical 二分类
- `phase_importance`
- 基于梯度的热图导出
- 原始切片叠加展示

但当前“热图”本质上仍然是：

- 模型事后解释
- 依赖 Grad-CAM
- 容易落在边缘、体表或不稳定层面

这和业务上真正想要的能力并不完全一致。

本次要补的是一种新的模型能力：

- 不依赖医生手工标注感兴趣区域
- 模型在推理时直接输出“建议医生重点复核的区域”
- 输出形式优先是连续热区图，其次可派生候选关注区域

这里的目标不是还原“医生真实眼动轨迹”，而是让模型生成：

- 与当前判别任务强相关
- 对医生复核有帮助
- 比普通 Grad-CAM 更稳定、结构上内生的关注区域

## 2. 设计目标

本次设计目标：

- 在现有三期 CT 主线上引入“医生建议关注区”能力
- 不增加人工区域标注依赖
- 保持现有 `validate-config/train/build-results` 入口不变
- 继续输出医生可展示的病例级解释产物
- 保证第一版实现复杂度可控

非目标：

- 不承诺恢复医生真实 gaze
- 不承诺等价于病灶分割
- 不在第一版做精确候选框检测器
- 不要求新增病灶级真值标注

## 3. 当前代码约束

当前主线关键实现：

- `med_core/models/three_phase_ct_fusion.py`
- `med_core/postprocessing/results.py`
- `med_core/shared/visualization/heatmaps.py`
- `med_core/attention_supervision/cam_supervision.py`
- `med_core/extractors/multi_region.py`

现状总结：

- 三期模型已有每期独立 3D encoder 和 `phase_importance`
- 结果构建已支持 Grad-CAM 叠图，但这是事后解释，不是内生关注头
- 仓库已有无掩码的 `CAMSelfSupervision`
- 仓库已有 `AdaptiveRegionExtractor` / `MultiRegionExtractor` 的概念积木

因此，本次不是推翻重做，而是在主线模型上新增：

- `doctor_interest_map` 头
- 基于兴趣图的局部汇聚
- 稳定兴趣图的无标注约束损失

## 4. 备选路线

## 4.1 路线 A：升级为内生连续热区图

做法：

- 每一期 encoder 输出 3D feature map
- 新增一个轻量 `interest head`
- 直接预测每一期的低分辨率 3D interest map
- 用 interest map 对 feature map 做加权池化
- 最终分类依赖“全局特征 + 兴趣区域特征”

优点：

- 最贴近当前主线
- 改动最小
- 最容易替代当前单纯 Grad-CAM 的角色

缺点：

- 只有连续热区图，没有天然候选区域结构
- 如果没有额外约束，可能仍会漂向边缘

## 4.2 路线 B：全局分支 + 兴趣图分支 + Top-K 关注区域

做法：

- 先预测连续 interest map
- 再从 interest map 中选取 Top-K 峰值位置
- 对这些局部区域提取 patch / cube 级特征
- 用局部特征和全局特征共同完成分类

优点：

- 同时得到连续热区图和候选关注区域
- 更接近“模型建议医生看这几处”
- 后续做 UI 非常自然

缺点：

- 比路线 A 更复杂
- patch 提取和局部重编码要处理好显存与稳定性

## 4.3 路线 C：原型网络 / 部件网络

做法：

- 学习若干“典型关注模式”原型
- 输出“本例最像哪几个原型”
- 再将原型响应投回空间区域

优点：

- 理论上解释性最强
- 容易形成“这类征象”式解释

缺点：

- 研究味太重
- 对小样本三期 CT demo 风险偏高
- 第一版落地成本不合适

## 5. 推荐路线

推荐采用路线 B，但第一版产品形态先以“连续热区图”为主。

也就是：

- 模型内部使用 `interest map + Top-K local focus`
- 对外第一版先展示 `doctor_interest_map`
- 如需候选区域列表，再从同一张图派生

推荐它的原因：

- 比路线 A 更像“模型内生能力”，不是只多了一个图
- 比路线 C 更现实，能落在现有仓库主线
- 能和现有 `phase_importance`、heatmap artifact、case explanation 自然衔接

## 6. 推荐模型：DIM-Net

推荐模型名采用通用命名：

- `DIM-Net`
- 全称：`Doctor Interest Map Network`

这里的 `Doctor` 是展示语义，不表示它学到真实医生 gaze。

其真实含义是：

- 用于生成医生建议复核区域的模型内生兴趣图网络

## 7. DIM-Net 总体结构

整体结构分四层：

1. 三期全局编码
2. 三期兴趣图生成
3. Top-K 局部关注区域汇聚
4. 影像-临床融合决策

结构摘要：

- 每一期 CT 经过 3D encoder 得到 `F_p`
- 由 `F_p` 生成一张 3D `doctor_interest_map`
- 用这张图做两件事：
- 加权池化得到该期的兴趣特征
- 选出 Top-K 关注区域得到局部 patch 特征
- 每一期输出一个增强后的 `phase_embedding`
- 三期再做 gated fusion，和 clinical 分支一起进入分类头

## 8. Phase Encoder 设计

保留现有三期结构，但升级每一期的输出语义。

每一期 encoder 输出：

- `feature_map`
- `global_embedding`

其中：

- `feature_map` 形状为 `[B, C, D, H, W]`
- `global_embedding` 是常规全局池化后的相位级表示

第一版不要求换成重型 backbone。

优先策略：

- 复用现有 `_PhaseEncoder3D`
- 只在其后增加解释性分支

这样可以降低改动范围，并避免影响当前训练入口。

## 9. Doctor Interest Map Head

对每一期 `feature_map` 增加一个轻量 3D 头：

- `1x1x1 conv`
- `norm`
- `relu`
- `1x1x1 conv -> 1 channel`

得到：

- `interest_logits_p`

再经过温度化归一化得到：

- `interest_map_p`

推荐形式：

- `sigmoid` 版本用于可视化
- `spatial softmax` 版本用于加权池化和 loss

保留两种视图：

- `interest_map_raw`
- `interest_map_norm`

这样训练和展示可以解耦。

## 10. Interest-Guided Pooling

每一期在常规全局池化之外，再做一次兴趣图引导池化：

- `global_feature_p = GAP(F_p)`
- `interest_feature_p = WeightedPool(F_p, interest_map_p)`

然后拼接：

- `phase_embedding_p = MLP([global_feature_p, interest_feature_p])`

这样可以保证：

- 模型不是只“画图”
- 而是实质上利用兴趣区域来完成分类

这点是和普通 Grad-CAM 的核心区别。

## 11. Top-K Local Focus 分支

为了让模型更像“建议医生看哪几处”，每一期再增加一个轻量局部分支。

步骤：

- 从 `interest_map_p` 中做非极大值抑制
- 选择 Top-K 峰值
- 在 `feature_map` 或原始输入体积上抽取固定大小的局部 cube
- 对局部 cube 做共享编码
- 聚合为 `local_focus_feature_p`

第一版推荐：

- `K = 3`
- patch 在 feature map 空间抽取，不直接回到原始体素空间

原因：

- 显存更稳
- 与当前主线编码器更容易接

该分支输出：

- `topk_focus_centers`
- `topk_focus_scores`
- `local_focus_feature_p`

## 12. Phase Fusion

沿用当前三期 gated fusion 思路，但输入换成增强后的相位表示。

每一期最终相位向量：

- `phase_embedding_p = MLP([global_feature_p, interest_feature_p, local_focus_feature_p])`

然后：

- `phase_gate -> phase_importance`
- `fused_image_feature = sum(alpha_p * phase_embedding_p)`

这样 `phase_importance` 保持兼容现有结果产物。

## 13. Clinical Fusion

clinical 分支保持当前规范：

- `clinical_values`
- `clinical_missing_mask`
- `clinical_encoder`

融合方式保持轻量：

- `fused_image_feature + clinical_feature -> multimodal_fusion`

理由：

- 本次重点是兴趣图能力
- 不需要同时大改临床分支

## 14. 训练目标

第一版训练损失建议由五部分组成：

### 14.1 分类损失

- `L_cls`

主任务损失，保持当前二分类设置。

### 14.2 CAM 自蒸馏对齐

- `L_cam_align`

目的：

- 在训练早期用现有 CAM 思路稳定兴趣图
- 避免 interest head 一开始完全漂移

实现方式：

- 用当前分类分支生成 CAM 或 Grad-CAM teacher
- 让 `interest_map_norm` 与 teacher 对齐

这一项不是最终解释依据，只是训练稳定器。

### 14.3 一致性损失

- `L_consistency`

目的：

- 让兴趣图在弱增强前后保持稳定
- 降低同一病例不同扰动下热图乱跳

可采用：

- augmentation consistency
- 相邻 slice consistency

### 14.4 稀疏/熵损失

- `L_sparse`

目的：

- 防止兴趣图过于弥散
- 让模型更像给出“重点区域”

可采用：

- entropy penalty
- top-mass regularization

### 14.5 Top-K 分离损失

- `L_diverse`

目的：

- 防止多个关注点塌缩到同一个位置

对 `Top-K` 峰值加入最小间距约束即可。

总损失：

- `L = L_cls + w1*L_cam_align + w2*L_consistency + w3*L_sparse + w4*L_diverse`

## 15. 无标注前提下的关键稳定器

如果完全没有任何先验，兴趣图容易学到错误区域。

因此第一版建议加两个“非人工标注”的约束：

### 15.1 Body Mask 约束

从 CT 体积自动生成粗 body mask：

- 基于 HU 阈值
- 最大连通域

用途：

- 抑制空气和体外区域
- 减少热点漂到图像外周

这不需要人工标注。

### 15.2 弱解剖位置先验

对三期 MVI 任务，可以加入非常弱的位置先验：

- 关注中上腹主体区域
- 抑制最顶端、最底端、最外周的极端响应

它不能替代器官分割，但足够防止第一版出现明显不合理热点。

这部分应设计为可配置开关，而不是硬编码业务规则。

## 16. 输出契约

模型 forward 推荐新增输出：

- `logits`
- `probability`
- `risk_score`
- `phase_importance`
- `feature_maps`
- `doctor_interest_maps`
- `topk_focus_centers`
- `topk_focus_scores`
- `phase_embeddings`
- `clinical_features`
- `fused_features`

其中：

- `doctor_interest_maps` 是对外主解释产物
- `topk_focus_centers` 是后续派生候选区域列表的基础

## 17. 结果导出设计

`build-results` 第一版新增两类解释产物：

### 17.1 Doctor Interest Overlay

每一期导出：

- `doctor_interest_overlay.png`
- `doctor_interest_original_overlay.png`
- `doctor_interest_original_slice.png`

与当前 heatmap 并行，但语义明确区分：

- 当前 `Grad-CAM`: 事后解释
- 新的 `Doctor Interest Map`: 内生关注建议

### 17.2 Focus Region JSON

病例级 `case_explanations.json` 扩展：

- `doctor_interest_artifacts`
- `topk_focus_regions`
- `focus_region_scores`

这样前端后续可以选择：

- 只显示连续热图
- 或显示“建议优先复核的 3 个区域”

## 18. 推荐实施顺序

推荐按四步实施：

### Phase 1

在 `three_phase_ct_fusion.py` 中加入 `doctor_interest_map` 头和 interest-guided pooling。

目标：

- 先让模型具备连续热区图能力

### Phase 2

加入 `CAM` 对齐、一致性、稀疏正则。

目标：

- 先把兴趣图训稳

### Phase 3

加入 `Top-K local focus` 分支。

目标：

- 让模型从“热区图”升级为“热区图 + 关注区域候选”

### Phase 4

扩展 `build-results` 和可视化产物。

目标：

- 让新能力可以直接用于 demo 展示

## 19. 成功标准

成功标准不只看分类指标，还要看解释质量。

第一版至少满足：

- 分类主线不回归
- 兴趣图稳定性明显好于当前纯 Grad-CAM
- 热点大多数情况下落在合理解剖区域内
- 同一病例轻微增强前后兴趣图不明显漂移
- 可以导出病例级连续热区图

更高一层的成功标准：

- 能从热区图中稳定抽出 3 个左右候选关注区域
- 医生人工抽查后，多数案例认为“值得看”

## 20. 不建议的第一版做法

第一版不建议：

- 直接做 prototype network
- 直接依赖器官或肿瘤真值分割
- 直接把输出定义为检测框
- 只换个更漂亮的 Grad-CAM 就当完成

原因：

- 要么成本过高
- 要么没有真正增加模型内生能力

## 21. 最终建议

最终建议是：

- 对外展示：先做连续 `doctor_interest_map`
- 对内模型：采用 `interest map + Top-K local focus` 双层结构
- 训练稳定：用 `CAM 自蒸馏 + 一致性 + 稀疏 + 分离` 组合
- 解剖约束：加自动 body mask 和弱位置先验，不引入人工区域标注

这条路线最适合当前仓库和当前任务边界。

一句话概括：

- 不是把 Grad-CAM 做得更像医生
- 而是把“建议医生看哪里”做成模型结构本身的一部分
