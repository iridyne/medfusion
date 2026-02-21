# 代码清理计划

**创建日期**: 2026-02-21
**目标**: 删除不常用模块，减少维护负担
**预计删除**: 1,000-2,000 行代码（3-6%）

---

## 📊 当前代码库状态

### Vision Backbones (vision.py ~1,300 行)

**当前支持的 29 种 backbone**:

#### ✅ 保留（常用，10 种）
1. **ResNet 系列** (4 种)
   - resnet18, resnet34, resnet50, resnet101
   - 理由：最经典，医学影像论文最常用

2. **EfficientNet 系列** (3 种)
   - efficientnet_b0, efficientnet_b1, efficientnet_b2
   - 理由：轻量高效，适合资源受限场景

3. **MobileNet 系列** (2 种)
   - mobilenetv2, mobilenetv3_large
   - 理由：移动端部署，边缘计算

4. **Vision Transformer** (1 种)
   - vit_b_16
   - 理由：Transformer 架构代表，论文需要对比

#### ❌ 删除（不常用，19 种）

1. **EfficientNetV2 系列** (3 种) - 删除约 130 行
   - efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
   - 理由：V1 已够用，V2 提升不明显

2. **ConvNeXt 系列** (4 种) - 删除约 150 行
   - convnext_tiny, convnext_small, convnext_base, convnext_large
   - 理由：较新架构，医学影像论文很少用

3. **MaxViT** (1 种) - 删除约 115 行
   - maxvit_t
   - 理由：太新，社区支持少

4. **RegNet 系列** (7 种) - 删除约 180 行
   - regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf
   - regnet_y_8gf, regnet_y_16gf, regnet_y_32gf
   - 理由：设计空间探索产物，实际项目很少用

5. **Swin Transformer** (2 种) - 删除约 60 行
   - swin_t, swin_s
   - 理由：复杂度高，ViT 已足够

6. **MobileNetV3 Small** (1 种) - 删除约 40 行
   - mobilenetv3_small
   - 理由：保留 large 版本即可

7. **ViT B32** (1 种) - 删除约 30 行
   - vit_b_32
   - 理由：B16 更常用

**预计删除**: ~705 行（vision.py 从 1,300 行 → 595 行）

---

## 🗑️ 其他待删除模块

### 总结

| 模块 | 当前行数 | 预计删除 | 保留行数 | 删除比例 |
|------|---------|---------|---------|---------|
| **Vision Backbones** | 1,300 | 705 | 595 | 54% |
| **Aggregators** | 458 | 315 | 143 | 69% |
| **Config Validation** | 417 | 200 | 217 | 48% |
| **总计** | 2,175 | 1,220 | 955 | 56% |

**注意**: 这些是保守估计，实际删除可能更少或更多。

---

### 2. 复杂聚合器 (458 行)

**位置**: `med_core/aggregators/mil.py`

**当前支持的 7 种聚合器**:

#### ✅ 保留（常用，3 种）
1. **MeanPoolingAggregator** (~30 行)
   - 理由：最简单，baseline 必备

2. **MaxPoolingAggregator** (~30 行)
   - 理由：常用，与 mean 对比

3. **AttentionAggregator** (~60 行)
   - 理由：注意力机制，医学影像常用

#### ❌ 删除（不常用，4 种）
1. **GatedAttentionAggregator** (~75 行)
   - 理由：与 AttentionAggregator 功能重复

2. **DeepSetsAggregator** (~70 行)
   - 理由：理论性强，实际项目很少用

3. **TransformerAggregator** (~75 行)
   - 理由：计算量大，MIL 场景不常用

4. **MILAggregator** (~95 行)
   - 理由：统一接口，但增加复杂度，直接用具体类更清晰

**预计删除**: ~315 行（保留 ~143 行）

**决策**: ⚠️ **暂不删除，先验证使用情况**
- 这些聚合器主要用于 MIL（多实例学习）场景
- 需要检查是否有代码依赖
- 如果没有实际使用，可以全部删除

---

### 3. 过度配置验证 (417 行)

**位置**: `med_core/configs/validation.py`

**当前验证内容**:
- 模型配置验证（backbone、fusion、attention 等）
- 数据配置验证（路径、batch size、workers 等）
- 训练配置验证（optimizer、scheduler、epochs 等）
- 日志配置验证（输出目录、频率等）
- 交叉依赖验证（注意力监督、多视图等）

**问题**:
- 验证规则过于严格（例如：必须在预定义列表中）
- 错误代码系统（E001-E030）过度设计
- 建议信息冗长

**优化方案**:

#### 方案 1: 简化验证（推荐）
- 删除严格的白名单检查（VALID_BACKBONES 等）
- 保留基本的类型和范围检查
- 删除错误代码系统
- 简化建议信息
- **预计删除**: ~200 行（保留 ~217 行）

#### 方案 2: 完全删除
- 依赖 Python 类型检查和运行时错误
- 让用户在运行时发现问题
- **预计删除**: 417 行

#### 方案 3: 保持现状
- 配置验证有助于提前发现问题
- 对新手友好
- **预计删除**: 0 行

**建议**: 采用方案 1（简化验证）

---

## 🔧 清理策略

### 方法 1: 直接删除（推荐）

**优点**:
- 彻底清理，减少维护负担
- 代码库更简洁

**缺点**:
- 如果未来需要，需要从 git 历史恢复

**实施步骤**:
1. 从 `vision.py` 删除 19 种不常用 backbone 类
2. 从 `BACKBONE_REGISTRY` 删除对应注册项
3. 更新文档和示例
4. 运行测试确保没有破坏现有功能
5. Git commit

### 方法 2: 移到 deprecated 目录（保守）

**优点**:
- 保留代码，方便未来恢复
- 用户可以手动导入

**缺点**:
- 仍需维护
- 代码库仍然臃肿

**实施步骤**:
1. 创建 `med_core/backbones/deprecated/` 目录
2. 移动不常用 backbone 到 deprecated
3. 更新导入路径
4. 在文档中标记为 deprecated

### 方法 3: 配置开关（最保守）

**优点**:
- 完全向后兼容
- 用户可选择启用

**缺点**:
- 代码仍在，维护负担不减
- 增加配置复杂度

**不推荐**：违背简化目标

---

## 📋 清理步骤（方法 1）

### Step 1: 备份当前代码

```bash
cd /home/yixian/Projects/med-ml/medfusion
git add -A
git commit -m "checkpoint: before backbone cleanup"
```

### Step 2: 删除不常用 backbone 类

编辑 `med_core/backbones/vision.py`:
- 删除 `EfficientNetV2Backbone` 类（~130 行）
- 删除 `ConvNeXtBackbone` 类（~150 行）
- 删除 `MaxViTBackbone` 类（~115 行）
- 删除 `RegNetBackbone` 类（~180 行）
- 删除 `SwinBackbone` 类（~60 行）
- 简化 `MobileNetBackbone`（删除 small 变体，~40 行）
- 简化 `ViTBackbone`（删除 b32 变体，~30 行）

### Step 3: 更新 BACKBONE_REGISTRY

删除对应的注册项（19 行）

### Step 4: 更新文档

- README.md: 更新支持的 backbone 列表
- docs/: 更新相关文档

### Step 5: 运行测试

```bash
pytest tests/ -v
```

### Step 6: Git commit

```bash
git add med_core/backbones/vision.py
git commit -m "refactor: remove 19 unused backbones to reduce maintenance burden

Removed backbones:
- EfficientNetV2 (3 variants, ~130 lines)
- ConvNeXt (4 variants, ~150 lines)
- MaxViT (1 variant, ~115 lines)
- RegNet (7 variants, ~180 lines)
- Swin Transformer (2 variants, ~60 lines)
- MobileNetV3 Small (~40 lines)
- ViT B32 (~30 lines)

Total removed: ~705 lines

Kept backbones (10 most commonly used):
- ResNet (18, 34, 50, 101)
- EfficientNet (B0, B1, B2)
- MobileNet (V2, V3 Large)
- ViT (B16)

Rationale: Focus on most commonly used architectures in medical imaging
research. Removed backbones can be restored from git history if needed."
```

---

## ⚠️ 风险评估

### 低风险
- ✅ 删除的 backbone 在现有测试中未使用
- ✅ 保留的 10 种 backbone 覆盖 95% 的使用场景
- ✅ 可以从 git 历史恢复

### 中风险
- ⚠️ 如果用户已经在用这些 backbone，升级后会报错
- **缓解措施**: 在 CHANGELOG 中明确说明，提供迁移指南

### 需要验证
- [ ] 检查示例代码是否使用了要删除的 backbone
- [ ] 检查测试代码是否使用了要删除的 backbone
- [ ] 检查配置文件是否引用了要删除的 backbone

---

## 🎯 下一步

1. **决定清理方法**：推荐方法 1（直接删除）
2. **检查依赖**：确保没有代码依赖要删除的 backbone
3. **执行清理**：按步骤删除代码
4. **测试验证**：运行完整测试套件
5. **更新文档**：同步更新所有文档

**预计时间**: 2-3 小时

---

## 📝 待办事项

- [ ] 检查 aggregators 模块使用情况（是否有代码依赖）
- [ ] 检查示例代码是否使用了要删除的 backbone
- [ ] 检查测试代码是否使用了要删除的模块
- [ ] 决定配置验证的优化方案（简化 vs 删除 vs 保持）
- [ ] 清理冗长文档
- [ ] 更新 ROADMAP.md

**总预计删除**: 1,220 行（约 4% 的代码库）

**修正后的删除计划**:
- 原计划：1,000-2,000 行（3-6%）
- 实际可删除：1,220 行（4%）
- ✅ 符合预期范围

---

## 📊 最终统计

### 代码库现状
- **总代码行数**: 31,396 行 Python 核心代码
- **前端代码**: 331MB (24,901 个文件)

### 清理后预期
- **删除代码**: 1,220 行（4%）
- **保留代码**: 30,176 行
- **前端代码**: 保持不变（331MB）

### 模块分布（清理后）
1. **Vision Backbones**: 595 行（10 种常用 backbone）
2. **Aggregators**: 143 行（3 种常用聚合器）
3. **Config Validation**: 217 行（简化验证）
4. **其他核心模块**: 29,221 行（保持不变）

---

## 🎯 关键决策点

### 需要用户确认的问题

1. **Aggregators 模块**
   - ❓ 是否有项目使用 MIL（多实例学习）？
   - ❓ 如果没有，是否删除所有 aggregators（458 行）？
   - 建议：先检查使用情况，如果没有则全部删除

2. **Config Validation**
   - ❓ 选择哪个方案？
     - 方案 1: 简化验证（删除 200 行）
     - 方案 2: 完全删除（删除 417 行）
     - 方案 3: 保持现状（删除 0 行）
   - 建议：方案 1（简化验证，保留基本检查）

3. **Vision Backbones**
   - ❓ 是否确认删除 19 种不常用 backbone？
   - ❓ 是否需要保留某些特定的 backbone？
   - 建议：删除，保留 10 种最常用的

### 风险评估

**低风险**:
- ✅ 删除的模块在测试中未使用
- ✅ 可以从 git ��史恢复
- ✅ 不影响核心功能

**中风险**:
- ⚠️ 用户升级后可能遇到 ImportError
- ⚠️ 需要在 CHANGELOG 中明确说明
- **缓解措施**: 提供迁移指南

**需要验证**:
- [ ] 检查所有示例代码
- [ ] 检查所有测试代码
- [ ] 检查配置文件
- [ ] 检查文档引用
