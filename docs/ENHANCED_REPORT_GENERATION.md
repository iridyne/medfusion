# 增强报告生成功能

**更新日期**: 2026-02-21
**版本**: v0.4.1

---

## 📋 概述

增强报告生成器在原有报告功能基础上，新增了三大核心功能：

1. **高分辨率图表生成**（300 DPI）- 适合论文发表
2. **统计显著性检验** - 自动计算 p-value
3. **LaTeX 报告输出** - 直接用于论文写作

这些功能专为医学科研论文发表设计，满足期刊对图表质量和统计分析的要求。

---

## 🎯 核心功能

### 1. 高分辨率图表（300 DPI）

**问题**: 默认图表分辨率（72-100 DPI）不满足期刊要求（通常要求 300 DPI）

**解决方案**:
- 自动配置 matplotlib 为 300 DPI
- 所有图表保存为高分辨率 PNG
- 字体大小优化，确保清晰可读

**使用示例**:
```python
from med_core.evaluation import generate_enhanced_report

report_path = generate_enhanced_report(
    metrics=metrics,
    output_dir="results",
    experiment_name="My Experiment",
    dpi=300,  # 高分辨率
)
```

### 2. 统计显著性检验

**问题**: 需要手动计算统计检验，容易出错

**解决方案**:
- 自动进行 Z-test for proportions
- 计算 p-value 和置信区间
- 判断统计显著性（p < 0.05）

**使用示例**:
```python
from med_core.evaluation import generate_enhanced_report

# 提供 baseline 进行对比
report_path = generate_enhanced_report(
    metrics=improved_metrics,
    baseline_metrics=baseline_metrics,  # 对比基线
    output_dir="results",
    experiment_name="Model Comparison",
)
```

**输出示例**:
```markdown
## Statistical Significance Tests

| Comparison | Current | Baseline | Difference | p-value | Significant |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline | 0.9250 | 0.8500 | 0.0750 | 0.0168 | ✅ Yes |

*Note: p < 0.05 indicates statistical significance*
```

### 3. LaTeX 报告生成

**问题**: 需要手动将结果转换为 LaTeX 格式

**解决方案**:
- 自动生成完整的 LaTeX 文档
- 包含格式化的表格和图表
- 可直接编译为 PDF

**使用示例**:
```python
from med_core.evaluation import EnhancedReportGenerator

generator = EnhancedReportGenerator(
    experiment_name="My Experiment",
    output_dir="results",
    description="Study description",
    dpi=300,
)

generator.add_metrics(metrics)
generator.add_plot("ROC Curve", "roc.png")

# 生成 LaTeX 报告
latex_path = generator.generate_latex_report()
```

**LaTeX 输出示例**:
```latex
\begin{table}[htbp]
\centering
\caption{Performance metrics for My Experiment}
\label{tab:my_experiment}
\begin{tabular}{lcc}
\hline
Metric & Value & 95\% CI \\
\hline
AUC-ROC & 0.950 & (0.920, 0.980) \\
Accuracy & 0.920 & (0.890, 0.950) \\
Sensitivity & 0.940 & (0.910, 0.970) \\
Specificity & 0.900 & (0.870, 0.930) \\
\hline
\end{tabular}
\end{table}
```

---

## 🚀 快速开始

### 基础用法

```python
from med_core.evaluation import calculate_binary_metrics, generate_enhanced_report

# 1. 计算指标
metrics = calculate_binary_metrics(y_true, y_pred, y_scores)

# 2. 生成增强报告
report_path = generate_enhanced_report(
    metrics=metrics,
    output_dir="results",
    experiment_name="Pneumonia Detection",
    dpi=300,
)

# 输出:
# - results/report.md (Markdown 报告)
# - results/report.tex (LaTeX 报告)
```

### 高级用法：统计对比

```python
from med_core.evaluation import EnhancedReportGenerator

# 创建生成器
generator = EnhancedReportGenerator(
    experiment_name="Model Comparison",
    output_dir="results",
    description="Comparing ResNet18 vs ResNet50",
    dpi=300,
    enable_statistical_tests=True,
)

# 添加当前模型指标
generator.add_metrics(resnet50_metrics)

# 添加基线模型指标（用于对比）
generator.add_comparison_metrics(resnet18_metrics, "ResNet18 Baseline")

# 添加图表
generator.add_plot("ROC Curve", "roc_curve.png")
generator.add_plot("Confusion Matrix", "confusion_matrix.png")

# 添加配置
generator.add_config({
    "model": "ResNet50",
    "dataset": "ChestX-ray14",
    "epochs": 50,
})

# 生成报告（同时生成 Markdown 和 LaTeX）
report_path = generator.generate()
```

---

## 📊 完整示例

参考 `scripts/test_enhanced_report.py` 查看完整示例，包括：

1. **基础增强报告生成** - 高分辨率图表
2. **统计显著性检验** - 模型对比
3. **LaTeX 报告生成** - 论文格式

运行测试：
```bash
python scripts/test_enhanced_report.py
```

---

## 🎨 生成的文件

### Markdown 报告 (`report.md`)

包含：
- 系统信息（时间戳、版本、硬件）
- 性能指标表格（带置信区间）
- 混淆矩阵
- 可视化图表（嵌入式）
- 统计检验结果（如果启用）
- 完整配置（JSON 格式）

### LaTeX 报告 (`report.tex`)

包含：
- 格式化的表格（booktabs 样式）
- 高分辨率图表引用
- 统计检验表格
- 可直接编译为 PDF

编译 LaTeX：
```bash
pdflatex report.tex
```

---

## 🔧 配置选项

### EnhancedReportGenerator 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `experiment_name` | str | 必需 | 实验名称 |
| `output_dir` | str/Path | 必需 | 输出目录 |
| `description` | str | "" | 实验描述 |
| `dpi` | int | 300 | 图表分辨率 |
| `enable_statistical_tests` | bool | True | 是否启用统计检验 |

### generate_enhanced_report 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `metrics` | object | 必需 | 指标对象 |
| `output_dir` | str/Path | 必需 | 输出目录 |
| `experiment_name` | str | "Evaluation" | 实验名称 |
| `plots` | dict | None | 图表字典 {名称: 路径} |
| `config` | dict | None | 配置字典 |
| `baseline_metrics` | object | None | 基线指标（用于对比） |
| `dpi` | int | 300 | 图表分辨率 |

---

## 📈 性能指标

测试结果（基于 `test_enhanced_report.py`）：

| 功能 | 状态 | 耗时 |
|------|------|------|
| 高分辨率图表生成 | ✅ | ~200ms/图 |
| 统计显著性检验 | ✅ | ~5ms |
| LaTeX 报告生成 | ✅ | ~10ms |
| Markdown 报告生成 | ✅ | ~5ms |

---

## 🎯 适用场景

### 1. 论文投稿

- 生成 300 DPI 高分辨率图表
- LaTeX 表格直接复制到论文
- 自动计算统计显著性

### 2. 模型对比

- 对比多个模型性能
- 自动进行统计检验
- 生成对比报告

### 3. 客户交付

- 专业的 Markdown 报告
- 高质量可视化
- 完整的实验配置

---

## 🔄 与原报告生成器的对比

| 功能 | 原版 | 增强版 |
|------|------|--------|
| Markdown 报告 | ✅ | ✅ |
| 基础图表 | ✅ (72 DPI) | ✅ (300 DPI) |
| LaTeX 输出 | ❌ | ✅ |
| 统计检验 | ❌ | ✅ |
| 模型对比 | ❌ | ✅ |
| 置信区间 | ✅ | ✅ |
| 配置记录 | ✅ | ✅ |

---

## 📝 注意事项

1. **DPI 设置**: 300 DPI 适合打印和论文，但文件较大（~500KB/图）
2. **统计检验**: 目前仅支持二分类任务的 Z-test
3. **LaTeX 编译**: 需要安装 LaTeX 发行版（如 TeX Live）
4. **图表路径**: LaTeX 中的图表路径为绝对路径，需要确保文件存在

---

## 🚧 未来改进

- [ ] 支持多分类任务的统计检验
- [ ] 添加更多统计检验方法（McNemar's test, DeLong's test）
- [ ] 支持自定义 LaTeX 模板
- [ ] 添加 Word 文档输出（.docx）
- [ ] 支持批量报告生成
- [ ] 添加报告预览功能

---

## 📚 参考资料

- [Matplotlib DPI 设置](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)
- [统计检验方法](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)
- [LaTeX 表格格式](https://www.overleaf.com/learn/latex/Tables)
- [医学期刊图表要求](https://www.nejm.org/author-center/artwork-images)

---

## 💡 示例输出

查看 `outputs/` 目录下的示例报告：

1. `outputs/enhanced_report_test/` - 基础增强报告
2. `outputs/statistical_test/` - 统计检验报告
3. `outputs/latex_test/` - LaTeX 报告

每个目录包含：
- `report.md` - Markdown 报告
- `report.tex` - LaTeX 报告
- `*.png` - 高分辨率图表（300 DPI）
