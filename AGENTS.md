# MedFusion 项目知识库

> 这是 OpenHands AI Agent 的持久化记忆文件，用于存储项目相关的知识、经验和最佳实践。

---

## 📋 项目概览

**项目名称**: MedFusion  
**项目类型**: 医学多模态深度学习研究框架  
**版本**: 0.2.0  
**语言**: Python 3.11+  
**主要框架**: PyTorch 2.0+

### 核心统计
- 代码总量: 40,496 行
- 文档总量: 16,324 行
- 核心模块: 100 个 Python 文件
- 测试套件: 37 个
- 示例脚本: 16 个

---

## 🏗️ 项目架构

### 目录结构
```
medfusion/
├── med_core/           # 核心代码库
│   ├── backbones/      # 29个预训练模型
│   ├── fusion/         # 5种融合策略
│   ├── datasets/       # 数据处理
│   ├── trainers/       # 训练器
│   ├── evaluation/     # 评估工具
│   ├── configs/        # 配置管理
│   └── utils/          # 工具函数
├── tests/              # 测试代码
├── examples/           # 示例脚本
├── docs/               # 文档
├── scripts/            # 工具脚本
└── configs/            # YAML配置文件
```

### 核心模块

1. **Backbones** (29个模型变体)
   - ResNet, MobileNet, EfficientNet, ConvNeXt, RegNet
   - ViT, Swin Transformer, MaxViT
   - 集成 CBAM, SE Block, ECA Block 注意力机制

2. **Fusion** (5种策略)
   - Concatenate, Gated, Attention, Cross-Attention, Bilinear
   - 支持 Kronecker 融合、Fused Attention

3. **Datasets**
   - 多视图数据集支持
   - 注意力监督数据集
   - 智能缓存系统

4. **Trainers**
   - 混合精度训练 (AMP)
   - 渐进式训练
   - 差异化学习率

---

## 🎯 核心特性

### 1. 多视图支持
- 5种聚合策略: MaxPool, MeanPool, Attention, CrossViewAttention, LearnedWeight
- 支持场景: 多角度CT, 时间序列, 多模态, 多切片
- 缺失视图处理: skip, zero, duplicate

### 2. 注意力监督
- 3种监督方法: Mask-Guided, CAM-Based, Consistency
- 提高模型可解释性
- 零性能开销（可选）

### 3. 配置驱动
- YAML配置文件
- 30+配置验证规则
- 无需修改代码即可切换实验

---

## 💡 开发经验和最佳实践

### 文档管理

**经验**: 项目分析和临时文档会污染根目录

**解决方案**:
1. 创建 `.analysis_archive/` 目录存放临时分析文档
2. 将该目录添加到 `.gitignore`
3. 保持根目录只有核心文档: `README.md`, `CHANGELOG.md`

**实施**:
```bash
mkdir -p .analysis_archive
echo ".analysis_archive/" >> .gitignore
mv *_ANALYSIS.md *_SUMMARY.md .analysis_archive/
```

### 依赖管理

**工具**: uv (现代 Python 包管理器)

**常用命令**:
```bash
uv sync                    # 同步依赖
uv add <package>           # 添加依赖
uv run pytest              # 运行测试
uv run python -m med_core  # 运行模块
```

### 测试策略

**测试覆盖**: 37个测试文件，覆盖所有核心模块

**测试类型**:
- 单元测试: 测试单个组件
- 集成测试: 测试模块间交互
- 端到端测试: 测试完整流程

**运行测试**:
```bash
uv run pytest                          # 运行所有测试
uv run pytest tests/test_backbones.py  # 运行特定测试
uv run pytest -v                       # 详细输出
```

### 配置管理

**配置文件位置**: `configs/*.yaml`

**配置验证**: 使用 `med_core/configs/validation.py` 进行验证

**最佳实践**:
- 使用 YAML 配置文件而非硬编码
- 为每个实验创建独立配置文件
- 使用配置继承减少重复

### Docker 使用

**服务**:
- `train`: 训练服务
- `eval`: 评估服务
- `tensorboard`: 监控服务
- `jupyter`: 交互式开发
- `dev`: 开发环境

**常用命令**:
```bash
docker-compose up train        # 启动训练
docker-compose up tensorboard  # 启动监控
docker-compose up jupyter      # 启动 Jupyter
```

---

## 🔧 常见任务

### 添加新的 Backbone

1. 在 `med_core/backbones/` 创建新文件
2. 继承 `BaseVisionBackbone`
3. 实现 `forward()` 和 `output_dim` 属性
4. 在 `__init__.py` 中注册
5. 添加测试到 `tests/test_backbones.py`

### 添加新的融合策略

1. 在 `med_core/fusion/` 创建新文件
2. 继承 `BaseFusion`
3. 实现 `forward()` 方法
4. 在 `__init__.py` 中注册
5. 添加测试到 `tests/test_fusion.py`

### 运行训练

```bash
# 使用配置文件
uv run med-train --config configs/medical_config.yaml

# 使用 CLI 参数
uv run med-train \
  --data-dir data/lung_cancer \
  --backbone resnet18 \
  --fusion-type gated \
  --epochs 50
```

### 运行评估

```bash
uv run med-evaluate \
  --checkpoint outputs/lung_cancer/best_model.pth \
  --data-dir data/lung_cancer \
  --output-dir evaluation_results
```

---

## 🐛 常见问题

### 问题: 导入错误

**症状**: `ModuleNotFoundError: No module named 'med_core'`

**解决**:
```bash
uv sync                    # 同步依赖
uv pip install -e .        # 开发模式安装
```

### 问题: CUDA 内存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决**:
1. 减小 batch size
2. 启用混合精度训练 (`use_amp: true`)
3. 启用梯度累积 (`gradient_accumulation_steps: 4`)
4. 使用更小的 backbone

### 问题: 配置验证失败

**症状**: `ConfigValidationError`

**解决**:
1. 检查配置文件语法
2. 运行配置验证: `uv run python -m med_core.configs.validation configs/your_config.yaml`
3. 参考 `configs/default.yaml` 作为模板

---

## 📚 重要文件位置

### 核心代码
- 主入口: `med_core/__init__.py`
- CLI: `med_core/cli.py`
- 异常定义: `med_core/exceptions.py`
- 版本信息: `med_core/version.py`

### 配置
- 默认配置: `configs/default.yaml`
- 医学配置: `configs/medical_config.yaml`
- 测试配置: `configs/test_*.yaml`

### 文档
- 主文档: `README.md`
- 变更日志: `CHANGELOG.md`
- API文档: `docs/api/`
- 用户指南: `docs/guides/`
- 架构分析: `docs/architecture/analysis.md`

### 脚本
- 生成模拟数据: `scripts/generate_mock_data.py`
- 冒烟测试: `scripts/smoke_test.py`
- 基准测试: `scripts/run_benchmarks.py`
- 文档构建: `scripts/build_docs.sh`

---

## 🚀 性能优化技巧

### 训练加速
1. 使用混合精度训练 (AMP)
2. 启用数据缓存
3. 增加 DataLoader workers
4. 使用 pin_memory
5. 启用梯度累积

### 内存优化
1. 减小 batch size
2. 使用梯度检查点
3. 清理不需要的中间变量
4. 使用更小的模型

### 数据加载优化
1. 启用智能缓存: `use_cache: true`
2. 预处理数据并保存
3. 使用多进程加载: `num_workers: 4`
4. 使用 SSD 存储数据

---

## 🔍 代码风格

### 工具
- **Ruff**: 代码检查和格式化
- **MyPy**: 类型检查
- **pytest**: 测试框架

### 规范
- 遵循 PEP 8
- 使用类型注解
- 编写文档字符串
- 保持函数简洁 (<50行)

### 检查命令
```bash
ruff check .           # 代码检查
ruff format .          # 代码格式化
mypy med_core          # 类型检查
pytest --cov=med_core  # 测试覆盖率
```

---

## 📊 项目评级

| 维度 | 评级 | 说明 |
|------|------|------|
| 代码质量 | ⭐⭐⭐⭐⭐ | 规范、类型注解完整 |
| 测试覆盖 | ⭐⭐⭐⭐⭐ | 37个测试文件 |
| 文档完整性 | ⭐⭐⭐⭐⭐ | 95%+覆盖率 |
| DevOps支持 | ⭐⭐⭐⭐⭐ | Docker, CI/CD完备 |
| 可扩展性 | ⭐⭐⭐⭐⭐ | 350+种配置组合 |
| 生产就绪度 | ⭐⭐⭐⭐⭐ | 可直接部署 |

**综合评分**: ⭐⭐⭐⭐⭐ (5/5)

---

## 🎓 学习资源

### 内部文档
- [快速开始](docs/guides/quickstart.md)
- [多视图指南](docs/guides/multiview/overview.md)
- [注意力监督](docs/guides/attention/supervision.md)
- [配置指南](docs/guides/configuration.md)

### 示例代码
- [训练示例](examples/train_demo.py)
- [多视图示例](examples/attention_quick_start.py)
- [缓存示例](examples/cache_demo.py)
- [配置验证示例](examples/config_validation_demo.py)

---

## 📝 更新日志

### 2026-02-20
- ✅ 完成项目深度分析
- ✅ 创建 `.analysis_archive/` 目录管理临时文档
- ✅ 将临时分析文档移出根目录
- ✅ 添加 `.analysis_archive/` 到 `.gitignore`
- ✅ 创建 `AGENTS.md` 记忆系统
- ✅ 清理 `docs/` 目录结构
- ✅ 创建 `docs/.archive/` 归档临时文档
- ✅ 从 Git 跟踪中移除 12 个临时分析文档

### 经验总结

#### 1. 文档管理最佳实践
**问题**: 临时分析文档会污染项目目录结构

**解决方案**:
- 根目录: 创建 `.analysis_archive/` 存放项目级临时文档
- docs 目录: 创建 `docs/.archive/` 存放文档级临时文档
- 将归档目录添加到 `.gitignore`
- 在归档目录中添加 README.md 说明用途

**实施步骤**:
```bash
# 根目录清理
mkdir -p .analysis_archive
echo ".analysis_archive/" >> .gitignore
mv *_ANALYSIS.md *_SUMMARY.md .analysis_archive/

# docs 目录清理
mkdir -p docs/.archive
echo ".archive/" >> docs/.gitignore
mv docs/*_2026-*.md docs/.archive/
```

#### 2. 项目结构原则
- **根目录**: 只保留核心文档 (README.md, CHANGELOG.md, AGENTS.md)
- **docs 目录**: 保持清晰的分类结构 (api/, guides/, reference/, architecture/, reviews/)
- **临时文档**: 统一归档到 `.archive/` 目录，不纳入版本控制

#### 3. 记忆系统
- 使用 `AGENTS.md` 持久化项目知识和经验
- 记录常见问题和解决方案
- 记录最佳实践和开发经验
- 定期更新，保持知识库的时效性

---

**最后更新**: 2026-02-20  
**维护者**: OpenHands AI Agent  
**项目状态**: 活跃开发中
