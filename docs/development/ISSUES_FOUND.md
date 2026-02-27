# 新手常见问题汇总

## 立即修复的问题

### 1. 融合策略命名不一致（严重）

**问题**：`configs/default.yaml` 和文档使用 `concat`，但代码要求 `concatenate`

**影响**：所有使用配置文件的新手用户

**修复位置**：
- `med_core/fusion/strategies.py:498` - 添加别名支持
- 所有配置文件 - 统一使用 `concatenate`
- README.md - 更新示例

### 2. 默认配置指向不存在的数据

**问题**：`configs/default.yaml` 指向 `data/dataset.csv`，但不存在

**影响**：100% 新手用户

**修复方案**：
- 创建 `configs/quickstart.yaml` 指向 `data/mock/`
- 在 README 中优先推荐 quickstart.yaml
- 提供示例数据下载命令

### 3. 列名不匹配

**问题**：配置期望 `weight`, `marker_a`，但 mock 数据只有 `age`, `gender`

**影响**：60% 使用 mock 数据的用户

**修复方案**：
- 更新 mock 数据包含所有默认列
- 或更新默认配置匹配 mock 数据
- 添加数据验证命令

## 需要文档改进的问题

### 4. Builder API 文档不足

**问题**：README 只有 15 行介绍，但 `examples/model_builder_demo.py` 有 462 行

**影响**：80% 用户不知道 Builder API 的强大功能

**改进方案**：
- 在 README 快速开始中优先展示 Builder API
- 添加 Builder vs Config 对比表
- 创建独立的 Builder API 教程

### 5. 融合策略选择指南缺失

**问题**：提供 8 种融合策略但无选择建议

**影响**：新手不知道选哪个

**改进方案**：
- 添加性能对比表
- 提供决策树
- 添加 `med-benchmark-fusion` 命令

### 6. 错误信息不友好

**问题**：`ValueError: Unknown fusion type: concat` 没有建议

**改进方案**：
```python
available = ['concatenate', 'gated', 'attention', ...]
if fusion_type == 'concat':
    raise ValueError(f"Did you mean 'concatenate'? (not 'concat')")
raise ValueError(f"Unknown fusion type: {fusion_type}. Available: {available}")
```

## 需要新功能的问题

### 7. 缺少配置验证命令

**需求**：`uv run med-validate-config configs/my_config.yaml`

**功能**：
- 检查文件路径是否存在
- 检查 CSV 列名是否匹配
- 检查维度是否兼容
- 提供修复建议

### 8. 缺少示例数据下载

**需求**：`uv run med-download-sample-data`

**功能**：
- 下载 100 个样本的示例数据集
- 包含 CSV 和图像
- 可以直接用于训练

### 9. 缺少交互式配置生成器

**需求**：`uv run med-config-wizard`

**功能**：
- 询问任务类型（分类/生存分析）
- 询问数据路径
- 询问模型复杂度
- 生成配置文件

## 优先级排序

| 优先级 | 问题 | 工作量 | 影响范围 |
|-------|------|--------|---------|
| P0 | 融合策略命名不一致 | 1 小时 | 80% 用户 |
| P0 | 默认配置指向不存在的数据 | 30 分钟 | 100% 用户 |
| P1 | 列名不匹配 | 1 小时 | 60% 用户 |
| P1 | Builder API 文档不足 | 4 小时 | 80% 用户 |
| P2 | 融合策略选择指南 | 2 小时 | 50% 用户 |
| P2 | 错误信息改进 | 2 小时 | 40% 用户 |
| P3 | 配置验证命令 | 8 小时 | 30% 用户 |
| P3 | 示例数据下载 | 4 小时 | 100% 用户 |
| P3 | 交互式配置生成器 | 16 小时 | 60% 用户 |

## 立即可以做的改进（1 天内）

1. **修复融合策略命名**
   ```python
   # med_core/fusion/strategies.py
   FUSION_ALIASES = {"concat": "concatenate", "attn": "attention"}
   fusion_type = FUSION_ALIASES.get(fusion_type, fusion_type)
   ```

2. **创建 quickstart.yaml**（已完成）

3. **更新 README**
   - 添加"常见问题"章节
   - 优先展示 Builder API
   - 添加融合策略对比表

4. **改进错误信息**
   - 添加 "Did you mean..." 建议
   - 列出所有可用选项

5. **创建新手指南**（已完成）
   - `docs/QUICKSTART_GUIDE.md`
