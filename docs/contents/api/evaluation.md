# Evaluation API

模型评估模块，提供全面的评估工具和报告生成。

## 概述

Evaluation 模块提供了医学研究所需的完整评估工具链：

- **分类指标**: Accuracy, AUC, F1, Sensitivity, Specificity, PPV, NPV
- **可视化**: ROC 曲线, PR 曲线, 混淆矩阵
- **可解释性**: Grad-CAM, 注意力可视化
- **报告生成**: 自动生成符合医学发表标准的评估报告

## 指标计算

### calculate_binary_metrics

计算二分类任务的所有指标。

**参数：**
- `y_true` (np.ndarray): 真实标签 (N,)
- `y_pred` (np.ndarray): 预测标签 (N,)
- `y_score` (np.ndarray): 预测概率 (N,)

**返回：**
- `dict[str, float]`: 指标字典

**包含指标：**
- `accuracy` - 准确率
- `auc` - ROC 曲线下面积
- `f1` - F1 分数
- `precision` - 精确率
- `recall` - 召回率（灵敏度）
- `specificity` - 特异度
- `ppv` - 阳性预测值
- `npv` - 阴性预测值

**示例：**
```python
from med_core.evaluation import calculate_binary_metrics
import numpy as np

# 预测结果
y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.3, 0.85, 0.6])

# 计算指标
metrics = calculate_binary_metrics(y_true, y_pred, y_score)

print(f"准确率: {metrics['accuracy']:.3f}")
print(f"AUC: {metrics['auc']:.3f}")
print(f"F1: {metrics['f1']:.3f}")
print(f"灵敏度: {metrics['recall']:.3f}")
print(f"特异度: {metrics['specificity']:.3f}")
```

### MetricsCalculator

指标计算器类，支持批量计算和聚合。

**方法：**
- `add_batch(y_true, y_pred, y_score)` - 添加一个批次的预测
- `compute()` - 计算所有批次的聚合指标
- `reset()` - 重置计算器

**示例：**
```python
from med_core.evaluation import MetricsCalculator

calculator = MetricsCalculator()

# 逐批次添加预测
for batch in test_loader:
    images, tabular, labels = batch
    outputs = model(images, tabular)
    probs = torch.softmax(outputs, dim=1)[:, 1]
    preds = outputs.argmax(dim=1)

    calculator.add_batch(
        y_true=labels.cpu().numpy(),
        y_pred=preds.cpu().numpy(),
        y_score=probs.cpu().numpy()
    )

# 计算最终指标
metrics = calculator.compute()
print(metrics)
```

## 可视化

### plot_roc_curve

绘制 ROC 曲线。

**参数：**
- `y_true` (np.ndarray): 真实标签
- `y_score` (np.ndarray): 预测概率
- `save_path` (str | Path): 保存路径，默认 None
- `title` (str): 图表标题
- `show_ci` (bool): 是否显示置信区间，默认 False

**返回：**
- `matplotlib.figure.Figure`: 图表对象

**示例：**
```python
from med_core.evaluation import plot_roc_curve

fig = plot_roc_curve(
    y_true=y_true,
    y_score=y_score,
    save_path="outputs/roc_curve.png",
    title="ROC Curve - CT Diagnosis",
    show_ci=True
)
```

### plot_confusion_matrix

绘制混淆矩阵。

**参数：**
- `y_true` (np.ndarray): 真实标签
- `y_pred` (np.ndarray): 预测标签
- `class_names` (list[str]): 类别名称
- `save_path` (str | Path): 保存路径
- `normalize` (bool): 是否归一化，默认 False

**示例：**
```python
from med_core.evaluation import plot_confusion_matrix

fig = plot_confusion_matrix(
    y_true=y_true,
    y_pred=y_pred,
    class_names=["Benign", "Malignant"],
    save_path="outputs/confusion_matrix.png",
    normalize=True
)
```

### ReportVisualizer

报告可视化工具类。

**功能：**
- 生成多种评估图表
- 统一的样式和格式
- 支持批量导出

**示例：**
```python
from med_core.evaluation import ReportVisualizer

visualizer = ReportVisualizer(output_dir="outputs/figures")

# 生成所有可视化
visualizer.plot_roc_curve(y_true, y_score)
visualizer.plot_pr_curve(y_true, y_score)
visualizer.plot_confusion_matrix(y_true, y_pred)
visualizer.plot_calibration_curve(y_true, y_score)

# 保存所有图表
visualizer.save_all()
```

## 可解释性

### GradCAM

Grad-CAM 类激活映射。

**参数：**
- `model` (nn.Module): 模型
- `target_layer` (nn.Module): 目标层（通常是最后一个卷积层）

**方法：**
- `generate_cam(input, target_class)` - 生成类激活图
- `visualize(input, target_class, original_image)` - 可视化 CAM

**示例：**
```python
from med_core.evaluation import GradCAM

# 创建 Grad-CAM
gradcam = GradCAM(
    model=model,
    target_layer=model.vision_backbone.layer4[-1]
)

# 生成 CAM
cam = gradcam.generate_cam(
    input=image_tensor,
    target_class=1  # 恶性
)

# 可视化
fig = gradcam.visualize(
    input=image_tensor,
    target_class=1,
    original_image=original_image
)
```

### visualize_gradcam

便捷的 Grad-CAM 可视化函数。

**参数：**
- `model` (nn.Module): 模型
- `image` (torch.Tensor): 输入图像
- `target_class` (int): 目标类别
- `target_layer_name` (str): 目标层名称

**返回：**
- `np.ndarray`: CAM 热力图

**示例：**
```python
from med_core.evaluation import visualize_gradcam

cam = visualize_gradcam(
    model=model,
    image=image_tensor,
    target_class=1,
    target_layer_name="vision_backbone.layer4"
)

# 保存热力图
import matplotlib.pyplot as plt
plt.imsave("outputs/gradcam.png", cam, cmap="jet")
```

### visualize_attention_weights

可视化注意力权重。

**参数：**
- `attention_weights` (torch.Tensor): 注意力权重
- `modality_names` (list[str]): 模态名称
- `save_path` (str | Path): 保存路径

**示例：**
```python
from med_core.evaluation import visualize_attention_weights

# 获取注意力权重
_, aux = model(images, tabular)
attention_weights = aux['attention_weights']

# 可视化
visualize_attention_weights(
    attention_weights=attention_weights,
    modality_names=["CT", "Clinical Data"],
    save_path="outputs/attention.png"
)
```

## 报告生成

### generate_evaluation_report

生成完整的评估报告。

**参数：**
- `y_true` (np.ndarray): 真实标签
- `y_pred` (np.ndarray): 预测标签
- `y_score` (np.ndarray): 预测概率
- `output_dir` (str | Path): 输出目录
- `class_names` (list[str]): 类别名称
- `model_name` (str): 模型名称

**生成内容：**
- `report.json` - 指标 JSON 文件
- `report.txt` - 文本报告
- `roc_curve.png` - ROC 曲线
- `confusion_matrix.png` - 混淆矩阵
- `pr_curve.png` - PR 曲线

**示例：**
```python
from med_core.evaluation import generate_evaluation_report

report = generate_evaluation_report(
    y_true=y_true,
    y_pred=y_pred,
    y_score=y_score,
    output_dir="outputs/evaluation",
    class_names=["Benign", "Malignant"],
    model_name="SMuRF-ResNet50"
)

print(report)
```

### ReportGenerator

报告生成器类。

**功能：**
- 生成结构化报告
- 支持多种输出格式（JSON, TXT, HTML）
- 自动生成图表

**示例：**
```python
from med_core.evaluation import ReportGenerator

generator = ReportGenerator(output_dir="outputs/reports")

# 添加评估结果
generator.add_results(
    split="test",
    y_true=y_true,
    y_pred=y_pred,
    y_score=y_score
)

# 生成报告
generator.generate(
    model_name="SMuRF",
    class_names=["Benign", "Malignant"],
    format="html"
)
```

### EnhancedReportGenerator

增强版报告生成器。

**额外功能：**
- 统计显著性检验
- 置信区间计算
- 子组分析
- 模型比较

**示例：**
```python
from med_core.evaluation import EnhancedReportGenerator

generator = EnhancedReportGenerator(output_dir="outputs/reports")

# 添加多个模型的结果
generator.add_model_results("SMuRF", y_true, y_pred1, y_score1)
generator.add_model_results("ResNet50", y_true, y_pred2, y_score2)

# 生成比较报告
report = generator.generate_comparison_report(
    include_statistical_tests=True,
    confidence_level=0.95
)
```

## 完整评估流程

```python
from med_core.evaluation import (
    MetricsCalculator,
    plot_roc_curve,
    plot_confusion_matrix,
    visualize_gradcam,
    generate_evaluation_report
)

# 1. 收集预测结果
calculator = MetricsCalculator()
all_images = []
all_labels = []
all_preds = []
all_scores = []

model.eval()
with torch.no_grad():
    for images, tabular, labels in test_loader:
        images = images.to(device)
        tabular = tabular.to(device)

        outputs = model(images, tabular)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)

        all_images.append(images.cpu())
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_scores.append(probs.cpu().numpy())

# 2. 合并结果
y_true = np.concatenate(all_labels)
y_pred = np.concatenate(all_preds)
y_score = np.concatenate(all_scores)

# 3. 生成完整报告
report = generate_evaluation_report(
    y_true=y_true,
    y_pred=y_pred,
    y_score=y_score,
    output_dir="outputs/evaluation",
    class_names=["Benign", "Malignant"],
    model_name="SMuRF-ResNet50"
)

# 4. 生成 Grad-CAM（可选）
sample_image = all_images[0][0:1].to(device)
cam = visualize_gradcam(
    model=model,
    image=sample_image,
    target_class=1,
    target_layer_name="vision_backbone.layer4"
)

print("评估完成！")
print(f"AUC: {report['auc']:.3f}")
print(f"准确率: {report['accuracy']:.3f}")
print(f"F1: {report['f1']:.3f}")
```

## 参考

完整实现请参考：
- `med_core/evaluation/metrics.py` - 指标计算
- `med_core/evaluation/visualization.py` - 可视化
- `med_core/evaluation/interpretability.py` - 可解释性
- `med_core/evaluation/report_generator.py` - 报告生成
