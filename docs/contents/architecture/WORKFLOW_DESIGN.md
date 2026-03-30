# MedFusion 节点化工作流设计文档

> 文档状态：**Beta**

**版本**: v0.1.0  
**日期**: 2026-02-20  
**状态**: 设计阶段

## 📋 目录

- [设计理念](#设计理念)
- [参考项目](#参考项目)
- [节点类型定义](#节点类型定义)
- [数据流设计](#数据流设计)
- [工作流执行引擎](#工作流执行引擎)
- [工作流保存格式](#工作流保存格式)
- [技术实现](#技术实现)
- [示例工作流](#示例工作流)
- [实现路线图](#实现路线图)

---

## 设计理念

### 核心目标

1. **零代码操作**: 用户通过拖拽节点和连线完成整个训练流程，无需编写代码
2. **可视化数据流**: 清晰展示数据在各个处理步骤之间的流动
3. **模块化设计**: 每个节点代表一个独立的操作，易于理解和复用
4. **工作流复用**: 支持保存、加载、分享工作流模板
5. **实时反馈**: 节点执行状态实时更新，支持中断和恢复

### 设计原则

- **简单优先**: 常用场景应该只需要 3-5 个节点
- **渐进式复杂**: 支持从简单到复杂的工作流构建
- **类型安全**: 节点连接时自动验证数据类型
- **错误友好**: 清晰的错误提示和修复建议
- **性能优化**: 只重新执行变化的节点，缓存中间结果

---

## 参考项目

### ComfyUI

**优势**:
- 成熟的节点化界面（基于 LiteGraph.js）
- 高效的执行引擎（只执行变化部分）
- 强大的插件系统（custom_nodes）
- 工作流保存为 JSON 格式
- 支持从生成的图像中提取工作流

**借鉴点**:
- 节点化界面设计
- 数据流可视化
- 工作流保存格式
- 执行引擎优化策略

### 决策链（Statsape）

**优势**:
- 统计分析专用节点
- 模板市场（统计流云市场）
- 自动报告生成
- 零代码操作
- 双架构部署（桌面版 + 网页版）

**借鉴点**:
- 模板市场设计
- 自动报告生成
- 统计分析节点
- 用户体验优化

### MedFusion 的差异化

- **医学影像专用**: 支持 DICOM、NIfTI 等医学格式
- **多模态融合**: 图像 + 表格数据的融合处理
- **深度学习训练**: 完整的训练、验证、测试流程
- **模型库集成**: 29 种 Backbone、5 种 Fusion 策略
- **质量控制**: 数据质量检查、模型评估、统计检验

---

## 节点类型定义

### 1. 数据源节点 (Data Source Nodes)

#### 1.1 LoadDataset
- **功能**: 加载已上传的数据集
- **输入**: 无
- **输出**: 
  - `dataset`: Dataset 对象
  - `metadata`: 数据集元信息
- **配置**:
  - `dataset_id`: 数据集 ID（下拉选择）
  - `split`: 数据集划分（train/val/test/all）
  - `shuffle`: 是否打乱
  - `seed`: 随机种子

#### 1.2 LoadImage
- **功能**: 加载单张或批量图像
- **输入**: 无
- **输出**: 
  - `images`: 图像张量
  - `paths`: 文件路径列表
- **配置**:
  - `path`: 图像路径或目录
  - `format`: 图像格式（DICOM/NIfTI/PNG/JPG）
  - `recursive`: 是否递归搜索

#### 1.3 LoadTabular
- **功能**: 加载表格数据
- **输入**: 无
- **输出**: 
  - `dataframe`: Pandas DataFrame
  - `columns`: 列名列表
- **配置**:
  - `path`: 文件路径
  - `format`: 文件格式（CSV/Excel/JSON）
  - `encoding`: 编码格式

### 2. 预处理节点 (Preprocessing Nodes)

#### 2.1 ImagePreprocess
- **功能**: 图像预处理
- **输入**: 
  - `images`: 图像张量
- **输出**: 
  - `processed_images`: 处理后的图像
- **配置**:
  - `resize`: 目标尺寸 (width, height)
  - `normalize`: 归一化方法（zscore/minmax/imagenet）
  - `augmentation`: 数据增强（随机翻转、旋转、裁剪等）

#### 2.2 TabularPreprocess
- **功能**: 表格数据预处理
- **输入**: 
  - `dataframe`: Pandas DataFrame
- **输出**: 
  - `processed_data`: 处理后的数据
  - `scaler`: 标准化器（用于推理）
- **配置**:
  - `missing_strategy`: 缺失值处理（drop/mean/median/mode）
  - `encoding`: 类别编码（onehot/label/target）
  - `scaling`: 数值缩放（standard/minmax/robust）
  - `feature_selection`: 特征选择方法

#### 2.3 DataAugmentation
- **功能**: 高级数据增强
- **输入**: 
  - `images`: 图像张量
- **输出**: 
  - `augmented_images`: 增强后的图像
- **配置**:
  - `methods`: 增强方法列表
  - `probability`: 应用概率
  - `intensity`: 增强强度

### 3. 模型节点 (Model Nodes)

#### 3.1 SelectBackbone
- **功能**: 选择视觉骨干网络
- **输入**: 无
- **输出**: 
  - `backbone_config`: Backbone 配置
- **配置**:
  - `model_name`: 模型名称（ResNet50/EfficientNet/ViT/Swin 等）
  - `pretrained`: 是否使用预训练权重
  - `freeze_layers`: 冻结层数
  - `output_dim`: 输出维度

#### 3.2 SelectFusion
- **功能**: 选择融合策略
- **输入**: 无
- **输出**: 
  - `fusion_config`: Fusion 配置
- **配置**:
  - `strategy`: 融合策略（concatenate/gated/attention/cross_attention/bilinear）
  - `hidden_dim`: 隐藏层维度
  - `num_heads`: 注意力头数（attention 策略）
  - `dropout`: Dropout 率

#### 3.3 SelectAggregator
- **功能**: 选择多视图聚合器
- **输入**: 无
- **输出**: 
  - `aggregator_config`: Aggregator 配置
- **配置**:
  - `method`: 聚合方法（mean/max/attention/cross_view/learned_weight）
  - `learnable`: 是否可学习
  - `missing_strategy`: 缺失视图处理

#### 3.4 BuildModel
- **功能**: 构建完整模型
- **输入**: 
  - `backbone_config`: Backbone 配置
  - `fusion_config`: Fusion 配置
  - `aggregator_config`: Aggregator 配置（可选）
- **输出**: 
  - `model`: PyTorch 模型
  - `model_summary`: 模型摘要
- **配置**:
  - `num_classes`: 分类类别数
  - `task_type`: 任务类型（classification/regression）

### 4. 训练节点 (Training Nodes)

#### 4.1 ConfigureTraining
- **功能**: 配置训练参数
- **输入**: 无
- **输出**: 
  - `training_config`: 训练配置
- **配置**:
  - `learning_rate`: 学习率
  - `batch_size`: 批次大小
  - `epochs`: 训练轮数
  - `optimizer`: 优化器（Adam/AdamW/SGD）
  - `scheduler`: 学习率调度器
  - `loss_function`: 损失函数
  - `mixed_precision`: 混合精度训练

#### 4.2 TrainModel
- **功能**: 训练模型
- **输入**: 
  - `model`: PyTorch 模型
  - `train_dataset`: 训练数据集
  - `val_dataset`: 验证数据集（可选）
  - `training_config`: 训练配置
- **输出**: 
  - `trained_model`: 训练后的模型
  - `training_history`: 训练历史
  - `best_checkpoint`: 最佳检查点路径
- **配置**:
  - `early_stopping`: 早停策略
  - `checkpoint_interval`: 检查点保存间隔
  - `log_interval`: 日志记录间隔

#### 4.3 ResumeTraining
- **功能**: 从检查点恢复训练
- **输入**: 
  - `checkpoint_path`: 检查点路径
  - `train_dataset`: 训练数据集
  - `val_dataset`: 验证数据集
- **输出**: 
  - `trained_model`: 训练后的模型
  - `training_history`: 训练历史

### 5. 评估节点 (Evaluation Nodes)

#### 5.1 EvaluateModel
- **功能**: 评估模型性能
- **输入**: 
  - `model`: PyTorch 模型
  - `test_dataset`: 测试数据集
- **输出**: 
  - `metrics`: 评估指标
  - `predictions`: 预测结果
  - `confusion_matrix`: 混淆矩阵
- **配置**:
  - `metrics`: 评估指标列表（accuracy/precision/recall/f1/auc）
  - `save_predictions`: 是否保存预测结果

#### 5.2 VisualizeResults
- **功能**: 可视化评估结果
- **输入**: 
  - `metrics`: 评估指标
  - `confusion_matrix`: 混淆矩阵
  - `training_history`: 训练历史（可选）
- **输出**: 
  - `plots`: 图表列表
- **配置**:
  - `plot_types`: 图表类型（confusion_matrix/roc_curve/pr_curve/learning_curve）
  - `save_path`: 保存路径

#### 5.3 GenerateReport
- **功能**: 生成评估报告
- **输入**: 
  - `metrics`: 评估指标
  - `plots`: 图表列表
  - `model_summary`: 模型摘要
- **输出**: 
  - `report_path`: 报告文件路径
- **配置**:
  - `format`: 报告格式（PDF/Word/HTML）
  - `template`: 报告模板
  - `include_sections`: 包含的章节

### 6. 输出节点 (Output Nodes)

#### 6.1 SaveModel
- **功能**: 保存模型
- **输入**: 
  - `model`: PyTorch 模型
- **输出**: 
  - `model_path`: 模型保存路径
- **配置**:
  - `path`: 保存路径
  - `format`: 保存格式（PyTorch/ONNX/TorchScript）
  - `quantization`: 量化选项

#### 6.2 ExportPredictions
- **功能**: 导出预测结果
- **输入**: 
  - `predictions`: 预测结果
- **输出**: 
  - `export_path`: 导出文件路径
- **配置**:
  - `path`: 导出路径
  - `format`: 导出格式（CSV/JSON/Excel）

#### 6.3 SaveWorkflow
- **功能**: 保存工作流
- **输入**: 
  - `workflow`: 当前工作流
- **输出**: 
  - `workflow_path`: 工作流保存路径
- **配置**:
  - `path`: 保存路径
  - `name`: 工作流名称
  - `description`: 工作流描述

### 7. 工具节点 (Utility Nodes)

#### 7.1 DataSplit
- **功能**: 数据集划分
- **输入**: 
  - `dataset`: 数据集
- **输出**: 
  - `train_dataset`: 训练集
  - `val_dataset`: 验证集
  - `test_dataset`: 测试集
- **配置**:
  - `train_ratio`: 训练集比例
  - `val_ratio`: 验证集比例
  - `test_ratio`: 测试集比例
  - `stratify`: 是否分层采样
  - `seed`: 随机种子

#### 7.2 MergeDatasets
- **功能**: 合并多个数据集
- **输入**: 
  - `dataset1`: 数据集 1
  - `dataset2`: 数据集 2
  - `...`: 更多数据集
- **输出**: 
  - `merged_dataset`: 合并后的数据集
- **配置**:
  - `merge_strategy`: 合并策略（concat/interleave）

#### 7.3 FilterData
- **功能**: 数据过滤
- **输入**: 
  - `dataset`: 数据集
- **输出**: 
  - `filtered_dataset`: 过滤后的数据集
- **配置**:
  - `filter_condition`: 过滤条件
  - `filter_column`: 过滤列（表格数据）

---

## 数据流设计

### 数据类型系统

```python
# 基础数据类型
class DataType(Enum):
    IMAGE = "image"              # 图像张量 (B, C, H, W)
    TABULAR = "tabular"          # 表格数据 (DataFrame)
    DATASET = "dataset"          # 数据集对象
    MODEL = "model"              # PyTorch 模型
    CONFIG = "config"            # 配置字典
    METRICS = "metrics"          # 评估指标
    PATH = "path"                # 文件路径
    TENSOR = "tensor"            # 通用张量
    ANY = "any"                  # 任意类型

# 复合数据类型
class CompositeDataType:
    MULTIMODAL = [DataType.IMAGE, DataType.TABULAR]  # 多模态数据
    TRAINING_DATA = [DataType.DATASET, DataType.CONFIG]  # 训练数据
```

### 连接规则

1. **类型匹配**: 输出类型必须与输入类型兼容
2. **多输入**: 节点可以有多个输入端口
3. **多输出**: 节点可以有多个输出端口
4. **可选输入**: 某些输入端口可以为空
5. **类型转换**: 自动进行兼容类型的转换

### 数据流示例

```
LoadDataset → ImagePreprocess → SelectBackbone → BuildModel → TrainModel → EvaluateModel → GenerateReport
                                      ↓              ↓             ↓
                                 SelectFusion → ConfigureTraining
```

---

## 工作流执行引擎

### 执行策略

#### 1. 拓扑排序
- 根据节点依赖关系确定执行顺序
- 检测循环依赖并报错
- 支持并行执行独立节点

#### 2. 增量执行
- 只执行发生变化的节点及其下游节点
- 缓存未变化节点的输出结果
- 支持手动标记节点为"需要重新执行"

#### 3. 错误处理
- 节点执行失败时停止工作流
- 显示详细的错误信息和堆栈跟踪
- 支持从失败节点恢复执行

#### 4. 进度监控
- 实时显示每个节点的执行状态
- 显示整体工作流进度
- 支持取消正在执行的工作流

### 执行状态

```python
class NodeStatus(Enum):
    PENDING = "pending"          # 等待执行
    RUNNING = "running"          # 正在执行
    SUCCESS = "success"          # 执行成功
    ERROR = "error"              # 执行失败
    SKIPPED = "skipped"          # 跳过执行
    CACHED = "cached"            # 使用缓存结果
```

### 缓存机制

```python
# 缓存键生成
cache_key = hash(node_id + node_config + input_hashes)

# 缓存策略
- 内存缓存: 小数据（配置、指标等）
- 磁盘缓存: 大数据（模型、数据集等）
- 缓存过期: 基于时间或手动清除
```

---

## 工作流保存格式

### JSON 格式

```json
{
  "version": "1.0.0",
  "metadata": {
    "name": "Chest X-Ray Classification",
    "description": "使用 ResNet50 进行胸部 X 光分类",
    "author": "user@example.com",
    "created_at": "2026-02-20T10:30:00Z",
    "tags": ["classification", "medical-imaging", "resnet"]
  },
  "nodes": [
    {
      "id": "node_1",
      "type": "LoadDataset",
      "position": {"x": 100, "y": 100},
      "config": {
        "dataset_id": "chest-xray-001",
        "split": "train",
        "shuffle": true,
        "seed": 42
      }
    },
    {
      "id": "node_2",
      "type": "ImagePreprocess",
      "position": {"x": 300, "y": 100},
      "config": {
        "resize": [224, 224],
        "normalize": "imagenet",
        "augmentation": ["random_flip", "random_rotation"]
      }
    },
    {
      "id": "node_3",
      "type": "SelectBackbone",
      "position": {"x": 500, "y": 100},
      "config": {
        "model_name": "resnet50",
        "pretrained": true,
        "freeze_layers": 0,
        "output_dim": 512
      }
    }
  ],
  "connections": [
    {
      "id": "conn_1",
      "source": "node_1",
      "source_port": "dataset",
      "target": "node_2",
      "target_port": "images"
    },
    {
      "id": "conn_2",
      "source": "node_2",
      "source_port": "processed_images",
      "target": "node_3",
      "target_port": "input"
    }
  ],
  "execution_history": [
    {
      "timestamp": "2026-02-20T10:35:00Z",
      "status": "success",
      "duration": 300.5,
      "metrics": {
        "accuracy": 0.92,
        "loss": 0.25
      }
    }
  ]
}
```

### .medfusion 工程文件

```
workflow.medfusion
├── workflow.json          # 工作流定义
├── config.yaml           # 全局配置
├── checkpoints/          # 模型检查点
│   ├── epoch_10.pth
│   └── best_model.pth
├── logs/                 # 训练日志
│   └── training.log
├── metrics/              # 结构化指标与 validation
│   ├── metrics.json
│   └── validation.json
├── reports/              # 汇总与可读报告
│   ├── summary.json
│   └── report.md
├── artifacts/            # 图表与可视化产物
│   └── visualizations/
└── cache/                # 节点缓存
    ├── node_1.pkl
    └── node_2.pkl
```

---

## 技术实现

### 前端技术栈

#### ReactFlow
- **优势**: 
  - React 生态，易于集成
  - 丰富的节点和连线定制
  - 支持缩放、平移、选择等交互
  - 活跃的社区和文档
- **使用场景**:
  - 节点画布渲染
  - 节点拖拽和连接
  - 工作流可视化

#### 替代方案对比

| 方案 | 优势 | 劣势 | 推荐度 |
|------|------|------|--------|
| ReactFlow | React 生态、易用、文档好 | 性能一般 | ⭐⭐⭐⭐⭐ |
| LiteGraph.js | 高性能、ComfyUI 使用 | 非 React、文档少 | ⭐⭐⭐ |
| Rete.js | 功能强大、插件丰富 | 学习曲线陡峭 | ⭐⭐⭐⭐ |
| D3.js | 完全自定义 | 开发成本高 | ⭐⭐ |

**最终选择**: ReactFlow（与现有 React 技术栈一致）

### 后端技术栈

#### 工作流执行引擎

```python
# med_core/workflow/engine.py

class WorkflowEngine:
    def __init__(self):
        self.nodes = {}
        self.connections = []
        self.cache = WorkflowCache()
        
    def add_node(self, node: Node):
        """添加节点"""
        self.nodes[node.id] = node
        
    def add_connection(self, connection: Connection):
        """添加连接"""
        self.validate_connection(connection)
        self.connections.append(connection)
        
    def execute(self, start_node_id: str = None):
        """执行工作流"""
        # 1. 拓扑排序
        execution_order = self.topological_sort()
        
        # 2. 执行节点
        for node_id in execution_order:
            node = self.nodes[node_id]
            
            # 检查缓存
            if self.cache.has(node_id):
                node.status = NodeStatus.CACHED
                continue
                
            # 执行节点
            try:
                node.status = NodeStatus.RUNNING
                outputs = node.execute()
                node.status = NodeStatus.SUCCESS
                
                # 缓存结果
                self.cache.set(node_id, outputs)
            except Exception as e:
                node.status = NodeStatus.ERROR
                node.error = str(e)
                raise
                
    def topological_sort(self) -> List[str]:
        """拓扑排序"""
        # 实现拓扑排序算法
        pass
```

#### 节点基类

```python
# med_core/workflow/nodes/base.py

class Node(ABC):
    def __init__(self, node_id: str, node_type: str, config: dict):
        self.id = node_id
        self.type = node_type
        self.config = config
        self.status = NodeStatus.PENDING
        self.inputs = {}
        self.outputs = {}
        self.error = None
        
    @abstractmethod
    def execute(self) -> dict:
        """执行节点逻辑"""
        pass
        
    @abstractmethod
    def validate(self) -> bool:
        """验证节点配置"""
        pass
        
    def get_input(self, port_name: str):
        """获取输入数据"""
        return self.inputs.get(port_name)
        
    def set_output(self, port_name: str, data):
        """设置输出数据"""
        self.outputs[port_name] = data
```

### API 端点

```python
# POST /api/workflow/create
# 创建新工作流

# GET /api/workflow/{workflow_id}
# 获取工作流详情

# PUT /api/workflow/{workflow_id}
# 更新工作流

# POST /api/workflow/{workflow_id}/execute
# 执行工作流

# GET /api/workflow/{workflow_id}/status
# 获取执行状态

# POST /api/workflow/{workflow_id}/cancel
# 取消执行

# GET /api/workflow/templates
# 获取工作流模板列表

# POST /api/workflow/import
# 导入工作流

# GET /api/workflow/{workflow_id}/export
# 导出工作流
```

---

## 示例工作流

### 示例 1: 简单图像分类

```
LoadDataset → ImagePreprocess → SelectBackbone → BuildModel → TrainModel → EvaluateModel
                                      ↓              ↓
                                 ConfigureTraining
```

**节点配置**:
1. LoadDataset: 选择 Chest X-Ray 数据集
2. ImagePreprocess: 调整大小到 224x224，ImageNet 归一化
3. SelectBackbone: ResNet50，预训练权重
4. ConfigureTraining: 学习率 1e-4，批次大小 32，100 轮
5. BuildModel: 2 分类（Normal/Pneumonia）
6. TrainModel: 训练模型
7. EvaluateModel: 计算准确率、AUC 等指标

### 示例 2: 多模态融合

```
LoadDataset → ImagePreprocess → SelectBackbone ↘
                                                  → SelectFusion → BuildModel → TrainModel
LoadTabular → TabularPreprocess ----------------↗                    ↓
                                                            ConfigureTraining
```

**节点配置**:
1. LoadDataset: 加载多模态数据集（图像 + 临床数据）
2. ImagePreprocess: 图像预处理
3. TabularPreprocess: 表格数据预处理
4. SelectBackbone: EfficientNet-B0
5. SelectFusion: Gated Fusion
6. ConfigureTraining: 学习率 1e-3，批次大小 16
7. BuildModel: 构建多模态模型
8. TrainModel: 训练

### 示例 3: 多视图聚合

```
LoadDataset → ImagePreprocess → SelectBackbone → SelectAggregator → BuildModel → TrainModel
   (多视图)                                            ↓
                                              ConfigureTraining
```

**节点配置**:
1. LoadDataset: 加载多视图数据集（CT 多角度）
2. ImagePreprocess: 统一预处理
3. SelectBackbone: Swin Transformer
4. SelectAggregator: Attention Aggregator
5. ConfigureTraining: 学习率 5e-5，批次大小 8
6. BuildModel: 构建多视图模型
7. TrainModel: 训练

### 示例 4: 完整流程（数据 → 训练 → 评估 → 报告）

```
LoadDataset → DataSplit → ImagePreprocess → SelectBackbone → BuildModel → TrainModel → EvaluateModel → VisualizeResults → GenerateReport
                ↓                                 ↓              ↓             ↓                                    ↓
              (train)                        SelectFusion  ConfigureTraining                                  SaveModel
                ↓
              (val)
                ↓
              (test)
```

---

## 实现路线图

### Phase 1: 基础框架（2 周）

**目标**: 实现基本的节点系统和工作流执行引擎

- [ ] ReactFlow 集成
- [ ] 节点基类和注册系统
- [ ] 工作流执行引擎（拓扑排序、基本执行）
- [ ] 数据类型系统和连接验证
- [ ] 工作流保存/加载（JSON 格式）

**交付物**:
- 可以拖拽节点和连线
- 可以执行简单的工作流（3-5 个节点）
- 可以保存和加载工作流

### Phase 2: 核心节点（2 周）

**目标**: 实现常用的核心节点

- [ ] 数
