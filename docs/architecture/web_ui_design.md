# MedFusion Web UI 设计方案

## 📋 概述

基于 ComfyUI 的节点式工作流设计理念，为 MedFusion 框架打造一个可视化的 Web 界面，提供统一的训练、推理和框架搭建功能。

---

## 🎯 设计目标

### 核心理念
- **可视化工作流**: 拖拽式节点编辑器，直观展示数据流
- **零代码配置**: 通过 UI 完成所有配置，无需编写代码
- **实时反馈**: 训练过程实时监控，即时查看结果
- **模块化设计**: 节点化的组件，易于扩展和组合
- **本地优先**: 基于本地 Python 后端，数据安全可控

### 参考 ComfyUI 的优点
- ✅ 节点式工作流编辑器
- ✅ 拖拽连接，可视化数据流
- ✅ 实时预览和反馈
- ✅ 支持保存/加载工作流
- ✅ 插件化扩展

---

## 🏗️ 技术架构

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                     前端 (Web UI)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ 工作流编辑器  │  │ 训练监控面板  │  │ 模型管理界面  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         React + React Flow + Ant Design / Chakra UI     │
└─────────────────────────────────────────────────────────┘
                            ▲
                            │ HTTP / WebSocket
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   后端 (Python API)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  REST API    │  │  WebSocket   │  │  任务队列     │  │
│  │  (FastAPI)   │  │  (实时通信)   │  │  (Celery)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            ▲
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  MedFusion 核心框架                      │
│         Backbones / Fusion / Trainers / Datasets        │
└─────────────────────────────────────────────────────────┘
```

### 技术栈选择

#### 前端
```yaml
框架: React 18+
状态管理: Zustand / Redux Toolkit
节点编辑器: React Flow (类似 ComfyUI 的节点系统)
UI 组件库: Ant Design / Chakra UI
图表可视化: ECharts / Recharts
实时通信: Socket.IO Client
构建工具: Vite
```

#### 后端
```yaml
Web 框架: FastAPI (高性能、异步、自动文档)
WebSocket: FastAPI WebSocket / Socket.IO
任务队列: Celery + Redis (处理长时间训练任务)
数据库: SQLite / PostgreSQL (存储配置、历史记录)
文件存储: 本地文件系统
```

---

## 🎨 界面设计

### 主界面布局

```
┌─────────────────────────────────────────────────────────────┐
│  MedFusion Studio                    [用户] [设置] [帮助]    │
├─────────────────────────────────────────────────────────────┤
│ [工作流] [训练] [推理] [数据集] [模型库] [实验管理]          │
├──────────┬──────────────────────────────────────────────────┤
│          │                                                   │
│  节点库   │              工作流画布                           │
│          │                                                   │
│ 📦 数据   │    ┌─────┐      ┌─────┐      ┌─────┐            │
│  - 加载   │    │数据 │─────▶│模型 │─────▶│训练 │            │
│  - 预处理 │    └─────┘      └─────┘      └─────┘            │
│          │                                                   │
│ 🧠 模型   │    ┌─────┐                                       │
│  - 骨干网络│    │评估 │                                       │
│  - 融合   │    └─────┘                                       │
│  - 头部   │                                                   │
│          │                                                   │
│ 🎯 训练   │                                                   │
│  - 训练器 │                                                   │
│  - 优化器 │                                                   │
│  - 损失   │                                                   │
│          │                                                   │
│ 📊 评估   │                                                   │
│  - 指标   │                                                   │
│  - 可视化 │                                                   │
│          │                                                   │
└──────────┴──────────────────────────────────────────────────┘
│  属性面板: 选中节点的详细配置                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 核心功能模块

### 1. 节点系统

#### 节点类型定义

```typescript
// 节点基类
interface BaseNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: {
    label: string;
    config: Record<string, any>;
    status: 'idle' | 'running' | 'success' | 'error';
  };
}

// 节点类别
enum NodeCategory {
  DATA = 'data',           // 数据节点
  MODEL = 'model',         // 模型节点
  TRAINING = 'training',   // 训练节点
  EVALUATION = 'evaluation', // 评估节点
  EXPORT = 'export',       // 导出节点
}
```

#### 具体节点类型

**数据节点**
```yaml
- DatasetLoader: 加载数据集
  输入: 数据路径、配置
  输出: 数据集对象
  
- DataPreprocessing: 数据预处理
  输入: 原始数据
  输出: 预处理后的数据
  
- DataAugmentation: 数据增强
  输入: 数据
  输出: 增强后的数据
  
- DataSplit: 数据划分
  输入: 数据集
  输出: 训练集、验证集、测试集
```

**模型节点**
```yaml
- BackboneSelector: 选择骨干网络
  配置: ResNet/Swin/EfficientNet/等
  输出: Backbone 模型
  
- FusionModule: 融合模块
  输入: 多个特征
  配置: Concatenate/Gated/Attention/等
  输出: 融合后的特征
  
- HeadModule: 分类/回归头
  输入: 特征
  配置: 类别数、头部类型
  输出: 预测结果
  
- FullModel: 完整模型
  输入: Backbone + Fusion + Head
  输出: 完整模型
```

**训练节点**
```yaml
- Trainer: 训练器
  输入: 模型、数据、配置
  输出: 训练好的模型
  
- Optimizer: 优化器
  配置: Adam/SGD/AdamW/等
  输出: 优化器对象
  
- LossFunction: 损失函数
  配置: CrossEntropy/MSE/等
  输出: 损失函数
  
- LearningRateScheduler: 学习率调度
  配置: StepLR/CosineAnnealing/等
  输出: 调度器
```

**评估节点**
```yaml
- Evaluator: 评估器
  输入: 模型、测试数据
  输出: 评估指标
  
- MetricsCalculator: 指标计算
  输入: 预测、真实标签
  输出: Accuracy/F1/AUC/等
  
- Visualizer: 可视化
  输入: 结果
  输出: 图表、混淆矩阵、ROC 曲线
```

**导出节点**
```yaml
- ModelExporter: 模型导出
  输入: 训练好的模型
  配置: ONNX/TorchScript/等
  输出: 导出的模型文件
  
- CheckpointSaver: 保存检查点
  输入: 模型、优化器状态
  输出: 检查点文件
```

---

### 2. 工作流编辑器

#### 核心功能

```typescript
// 工作流管理
class WorkflowManager {
  // 创建新工作流
  createWorkflow(): Workflow;
  
  // 保存工作流
  saveWorkflow(workflow: Workflow): void;
  
  // 加载工作流
  loadWorkflow(id: string): Workflow;
  
  // 执行工作流
  executeWorkflow(workflow: Workflow): Promise<Result>;
  
  // 验证工作流
  validateWorkflow(workflow: Workflow): ValidationResult;
}

// 节点操作
class NodeOperations {
  // 添加节点
  addNode(type: string, position: Position): Node;
  
  // 删除节点
  deleteNode(nodeId: string): void;
  
  // 连接节点
  connectNodes(sourceId: string, targetId: string): void;
  
  // 更新节点配置
  updateNodeConfig(nodeId: string, config: any): void;
}
```

#### 交互设计

```yaml
拖拽操作:
  - 从节点库拖拽节点到画布
  - 拖拽节点调整位置
  - 拖拽连接线连接节点

右键菜单:
  - 删除节点
  - 复制节点
  - 查看节点文档
  - 运行到此节点

快捷键:
  - Ctrl+S: 保存工作流
  - Ctrl+Z: 撤销
  - Ctrl+Y: 重做
  - Delete: 删除选中节点
  - Ctrl+C/V: 复制粘贴节点
```

---

### 3. 训练监控面板

#### 实时监控

```typescript
interface TrainingMonitor {
  // 训练状态
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed';
  
  // 当前进度
  progress: {
    currentEpoch: number;
    totalEpochs: number;
    currentBatch: number;
    totalBatches: number;
    percentage: number;
  };
  
  // 实时指标
  metrics: {
    loss: number[];
    accuracy: number[];
    learningRate: number[];
    // ... 其他指标
  };
  
  // 系统资源
  resources: {
    gpuUsage: number;
    gpuMemory: number;
    cpuUsage: number;
    ramUsage: number;
  };
}
```

#### 可视化组件

```yaml
训练曲线:
  - Loss 曲线（训练/验证）
  - Accuracy 曲线
  - Learning Rate 曲线
  - 自定义指标曲线

系统监控:
  - GPU 使用率
  - GPU 显存
  - CPU 使用率
  - 内存使用率

日志输出:
  - 实时训练日志
  - 错误信息
  - 警告信息
```

---

### 4. 模型管理

#### 模型库

```yaml
功能:
  - 浏览所有可用模型
  - 查看模型详情（参数量、FLOPs、性能）
  - 下载预训练模型
  - 上传自定义模型
  - 模型版本管理

展示信息:
  - 模型名称和变体
  - 输入输出尺寸
  - 参数量
  - 预训练数据集
  - 性能指标
  - 使用示例
```

#### 实验管理

```yaml
功能:
  - 查看所有实验
  - 对比实验结果
  - 导出实验报告
  - 恢复实验配置

实验信息:
  - 实验名称和描述
  - 创建时间
  - 配置参数
  - 训练指标
  - 最佳模型
  - 日志文件
```

---

## 🔌 API 设计

### REST API 端点

```python
# 工作流管理
POST   /api/workflows              # 创建工作流
GET    /api/workflows              # 获取工作流列表
GET    /api/workflows/{id}         # 获取工作流详情
PUT    /api/workflows/{id}         # 更新工作流
DELETE /api/workflows/{id}         # 删除工作流
POST   /api/workflows/{id}/execute # 执行工作流

# 节点管理
GET    /api/nodes                  # 获取所有节点类型
GET    /api/nodes/{type}           # 获取节点详情
POST   /api/nodes/validate         # 验证节点配置

# 训练管理
POST   /api/training/start         # 开始训练
POST   /api/training/pause         # 暂停训练
POST   /api/training/resume        # 恢复训练
POST   /api/training/stop          # 停止训练
GET    /api/training/status        # 获取训练状态

# 模型管理
GET    /api/models                 # 获取模型列表
GET    /api/models/{id}            # 获取模型详情
POST   /api/models/upload          # 上传模型
DELETE /api/models/{id}            # 删除模型

# 数据集管理
GET    /api/datasets               # 获取数据集列表
POST   /api/datasets/upload        # 上传数据集
GET    /api/datasets/{id}/preview  # 预览数据集

# 实验管理
GET    /api/experiments            # 获取实验列表
GET    /api/experiments/{id}       # 获取实验详情
POST   /api/experiments/{id}/compare # 对比实验
```

### WebSocket 事件

```python
# 客户端 -> 服务器
'subscribe_training'    # 订阅训练更新
'unsubscribe_training'  # 取消订阅

# 服务器 -> 客户端
'training_progress'     # 训练进度更新
'training_metrics'      # 训练指标更新
'training_log'          # 训练日志
'training_completed'    # 训练完成
'training_error'        # 训练错误
'system_resources'      # 系统资源更新
```

---

## 📦 项目结构

```
medfusion-web/
├── frontend/                    # 前端项目
│   ├── src/
│   │   ├── components/          # React 组件
│   │   │   ├── WorkflowEditor/  # 工作流编辑器
│   │   │   ├── NodeLibrary/     # 节点库
│   │   │   ├── TrainingMonitor/ # 训练监控
│   │   │   ├── ModelManager/    # 模型管理
│   │   │   └── ExperimentPanel/ # 实验面板
│   │   ├── nodes/               # 节点定义
│   │   │   ├── DataNodes.tsx
│   │   │   ├── ModelNodes.tsx
│   │   │   ├── TrainingNodes.tsx
│   │   │   └── EvaluationNodes.tsx
│   │   ├── stores/              # 状态管理
│   │   │   ├── workflowStore.ts
│   │   │   ├── trainingStore.ts
│   │   │   └── modelStore.ts
│   │   ├── api/                 # API 调用
│   │   │   ├── workflow.ts
│   │   │   ├── training.ts
│   │   │   └── websocket.ts
│   │   ├── types/               # TypeScript 类型
│   │   ├── utils/               # 工具函数
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
│
├── backend/                     # 后端项目
│   ├── app/
│   │   ├── api/                 # API 路由
│   │   │   ├── workflows.py
│   │   │   ├── training.py
│   │   │   ├── models.py
│   │   │   └── websocket.py
│   │   ├── core/                # 核心逻辑
│   │   │   ├── workflow_engine.py  # 工作流执行引擎
│   │   │   ├── node_registry.py    # 节点注册表
│   │   │   └── task_manager.py     # 任务管理
│   │   ├── models/              # 数据模型
│   │   │   ├── workflow.py
│   │   │   ├── experiment.py
│   │   │   └── training_job.py
│   │   ├── services/            # 业务逻辑
│   │   │   ├── training_service.py
│   │   │   ├── model_service.py
│   │   │   └── dataset_service.py
│   │   ├── nodes/               # 节点实现
│   │   │   ├── data_nodes.py
│   │   │   ├── model_nodes.py
│   │   │   ├── training_nodes.py
│   │   │   └── evaluation_nodes.py
│   │   ├── utils/               # 工具函数
│   │   └── main.py              # FastAPI 应用入口
│   ├── requirements.txt
│   └── pyproject.toml
│
├── docker/                      # Docker 配置
│   ├── Dockerfile.frontend
│   ├── Dockerfile.backend
│   └── docker-compose.yml
│
└── README.md
```

---

## 🚀 实施计划

### Phase 1: MVP (最小可行产品) - 2 周

**目标**: 实现基本的工作流编辑和训练功能

```yaml
Week 1:
  后端:
    - ✅ FastAPI 项目搭建
    - ✅ 基础 API 端点
    - ✅ 节点注册系统
    - ✅ 简单的工作流执行引擎
  
  前端:
    - ✅ React 项目搭建
    - ✅ React Flow 集成
    - ✅ 基础节点库（5-10 个核心节点）
    - ✅ 工作流画布

Week 2:
  后端:
    - ✅ WebSocket 实时通信
    - ✅ 训练任务管理
    - ✅ 与 MedFusion 核心集成
  
  前端:
    - ✅ 训练监控面板
    - ✅ 实时指标展示
    - ✅ 工作流保存/加载
    - ✅ 基础 UI 优化
```

### Phase 2: 功能完善 - 2 周

```yaml
Week 3:
  - ✅ 完整的节点库（30+ 节点）
  - ✅ 模型管理界面
  - ✅ 数据集管理
  - ✅ 实验对比功能

Week 4:
  - ✅ 高级训练功能（分布式、混合精度）
  - ✅ 模型导出功能
  - ✅ 可视化增强
  - ✅ 用户文档
```

### Phase 3: 优化和扩展 - 2 周

```yaml
Week 5:
  - ✅ 性能优化
  - ✅ 错误处理和恢复
  - ✅ 插件系统
  - ✅ 自定义节点支持

Week 6:
  - ✅ 用户体验优化
  - ✅ 测试和 Bug 修复
  - ✅ 部署文档
  - ✅ 发布 v1.0
```

---

## 💡 核心特性

### 1. 零代码训练

用户只需：
1. 拖拽"数据加载"节点，选择数据集
2. 拖拽"ResNet50"节点，配置参数
3. 拖拽"训练器"节点，设置 epochs
4. 连接节点
5. 点击"运行"

### 2. 实时反馈

- 训练进度实时更新
- Loss/Accuracy 曲线实时绘制
- GPU 使用率实时监控
- 日志实时输出

### 3. 工作流复用

- 保存工作流为模板
- 分享工作流配置
- 导入社区工作流
- 版本控制

### 4. 实验管理

- 自动记录所有实验
- 对比不同配置的结果
- 导出实验报告
- 最佳模型自动保存

---

## 🎨 UI 设计参考

### 配色方案

```css
/* 主题色 */
--primary: #1890ff;      /* 蓝色 - 主要操作 */
--success: #52c41a;      /* 绿色 - 成功状态 */
--warning: #faad14;      /* 橙色 - 警告 */
--error: #f5222d;        /* 红色 - 错误 */

/* 背景色 */
--bg-primary: #ffffff;   /* 主背景 */
--bg-secondary: #f5f5f5; /* 次背景 */
--bg-dark: #001529;      /* 深色背景 */

/* 节点颜色 */
--node-data: #52c41a;    /* 数据节点 - 绿色 */
--node-model: #1890ff;   /* 模型节点 - 蓝色 */
--node-training: #fa8c16;/* 训练节点 - 橙色 */
--node-eval: #722ed1;    /* 评估节点 - 紫色 */
```

### 节点样式

```yaml
节点外观:
  - 圆角矩形
  - 左侧输入端口，右侧输出端口
  - 顶部显示节点类型图标
  - 中间显示节点名称
  - 底部显示状态指示器

状态指示:
  - 灰色: 未运行
  - 蓝色: 运行中（带动画）
  - 绿色: 成功
  - 红色: 错误
```

---

## 🔒 安全考虑

```yaml
认证授权:
  - 可选的用户登录系统
  - API Token 认证
  - 工作流权限控制

数据安全:
  - 本地数据存储
  - 敏感信息加密
  - 文件上传大小限制

资源限制:
  - 并发训练任务限制
  - GPU 资源分配
  - 磁盘空间监控
```

---

## 📊 性能优化

```yaml
前端:
  - 虚拟滚动（大量节点）
  - 懒加载组件
  - 图表数据采样
  - WebSocket 消息节流

后端:
  - 异步任务处理
  - 数据库连接池
  - 缓存常用数据
  - 静态文件 CDN
```

---

## 🎯 与 ComfyUI 的对比

| 特性 | ComfyUI | MedFusion Web UI |
|------|---------|------------------|
| 节点式工作流 | ✅ | ✅ |
| 拖拽操作 | ✅ | ✅ |
| 实时预览 | ✅ | ✅ |
| 工作流保存 | ✅ | ✅ |
| 领域 | 图像生成 | 医学深度学习 |
| 训练监控 | ❌ | ✅ |
| 实验管理 | ❌ | ✅ |
| 模型对比 | ❌ | ✅ |
| 分布式训练 | ❌ | ✅ |

---

## 📝 总结

### 优势

1. **降低门槛**: 零代码配置，适合非编程用户
2. **可视化**: 直观展示数据流和训练过程
3. **高效**: 拖拽式操作，快速搭建实验
4. **可复用**: 工作流模板，避免重复配置
5. **可扩展**: 插件化设计，易于添加新功能

### 技术亮点

1. **React Flow**: 强大的节点编辑器库
2. **FastAPI**: 高性能异步 Web 框架
3. **WebSocket**: 实时双向通信
4. **Celery**: 分布式任务队列
5. **模块化**: 清晰的前后端分离

### 下一步

1. 评估设计方案
2. 确定技术栈
3. 搭建项目骨架
4. 实现 MVP
5. 迭代优化

---

**设计者**: MedFusion Team  
**日期**: 2024-02-20  
**版本**: v1.0
