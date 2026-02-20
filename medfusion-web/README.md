# MedFusion Web UI

医学深度学习框架的 Web 界面，提供可视化工作流编辑、训练监控和模型管理功能。

## 功能特性

- 🎨 **可视化工作流编辑器**: 拖拽式节点编辑，类似 ComfyUI
- 🚀 **智能工作流执行**: 依赖解析、并行执行、错误处理
- 🔥 **真实训练集成**: 集成 med_core 训练器，支持混合精度和梯度检查点
- 📊 **实时训练监控**: WebSocket 实时推送训练指标和进度
- 🎮 **训练控制**: 支持暂停/恢复/停止训练
- 🗂️ **模型库管理**: 浏览和管理预训练模型
- 💻 **系统资源监控**: CPU、内存、GPU 使用情况
- 🎭 **主题切换**: 支持亮色/暗色主题
- 🐳 **Docker 部署**: 一键启动完整服务

## 技术栈

### 后端
- FastAPI: 高性能 Web 框架
- WebSocket: 实时通信
- Celery + Redis: 异步任务队列
- SQLAlchemy: ORM
- Pydantic: 数据验证

### 前端
- React 18 + TypeScript
- Vite: 构建工具
- Ant Design: UI 组件库
- ReactFlow: 工作流编辑器
- ECharts: 数据可视化
- Zustand: 状态管理

## 快速开始

### 使用 Docker (推荐)

```bash
# 启动所有服务
docker-compose up -d

# 访问应用
# 前端: http://localhost
# 后端 API: http://localhost:8000
# API 文档: http://localhost:8000/docs
```

### 本地开发

#### 后端

```bash
cd backend

# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 前端

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

## 项目结构

```
medfusion-web/
├── backend/                 # 后端服务
│   ├── app/
│   │   ├── api/            # API 路由
│   │   ├── core/           # 核心功能
│   │   ├── models/         # 数据模型
│   │   ├── services/       # 业务逻辑
│   │   ├── nodes/          # 工作流节点
│   │   └── main.py         # 应用入口
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/               # 前端应用
│   ├── src/
│   │   ├── api/           # API 客户端
│   │   ├── components/    # React 组件
│   │   ├── pages/         # 页面组件
│   │   ├── stores/        # 状态管理
│   │   ├── types/         # TypeScript 类型
│   │   └── utils/         # 工具函数
│   ├── package.json
│   └── Dockerfile
└── docker-compose.yml      # Docker 编排
```

## API 文档

启动后端服务后，访问 http://localhost:8000/docs 查看完整的 API 文档。

### 主要 API 端点

- `GET /api/workflows/nodes` - 获取可用节点列表
- `POST /api/workflows/execute` - 执行工作流
- `POST /api/training/start` - 开始训练
- `GET /api/training/status/{job_id}` - 获取训练状态
- `WS /api/training/ws/{job_id}` - 训练进度 WebSocket
- `GET /api/models/` - 获取模型列表
- `GET /api/system/resources` - 获取系统资源

## 工作流节点

### 数据节点
- **Dataset Loader**: 加载医学图像数据集

### 模型节点
- **Backbone Selector**: 选择预训练骨干网络

### 训练节点
- **Trainer**: 训练深度学习模型

### 评估节点
- **Evaluator**: 评估模型性能

### 导出节点
- **Model Exporter**: 导出训练好的模型

## 开发指南

### 添加新节点

1. 在 `backend/app/nodes/` 创建节点类
2. 继承 `NodePlugin` 基类
3. 使用 `@register_node` 装饰器注册

```python
from app.core.node_registry import NodePlugin, register_node

@register_node("my_node")
class MyNode(NodePlugin):
    name = "My Node"
    category = "custom"
    
    @property
    def inputs(self):
        return ["input1", "input2"]
    
    @property
    def outputs(self):
        return ["output1"]
    
    async def execute(self, inputs):
        # 实现节点逻辑
        return {"output1": result}
```

### 添加新页面

1. 在 `frontend/src/pages/` 创建页面组件
2. 在 `App.tsx` 添加路由
3. 在 `Sidebar.tsx` 添加菜单项

## 性能优化

- 前端使用 React.memo 和 useMemo 优化渲染
- 后端使用异步 I/O 和连接池
- WebSocket 用于实时数据推送，减少轮询
- 图表使用虚拟滚动和数据采样
- Docker 多阶段构建减小镜像体积

## 部署

### 生产环境配置

1. 修改 `backend/app/core/config.py` 设置生产环境变量
2. 设置 `DEBUG=False`
3. 配置 HTTPS 和域名
4. 使用 Nginx 反向代理
5. 配置数据库持久化

### 环境变量

```bash
# 后端
DEBUG=False
HOST=0.0.0.0
PORT=8000
DATABASE_URL=postgresql://user:pass@localhost/medfusion
REDIS_URL=redis://localhost:6379/0

# 前端
VITE_API_URL=https://api.yourdomain.com
```

## 故障排除

### 后端无法启动
- 检查 Python 版本 (需要 3.11+)
- 确认所有依赖已安装
- 检查端口 8000 是否被占用

### 前端无法连接后端
- 确认后端服务已启动
- 检查 CORS 配置
- 查看浏览器控制台错误

### Docker 构建失败
- 确认 Docker 和 Docker Compose 已安装
- 检查网络连接
- 清理旧镜像: `docker system prune -a`

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 联系方式

- 项目主页: https://github.com/yourusername/medfusion
- 文档: https://medfusion.readthedocs.io
