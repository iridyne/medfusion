# MedFusion Web UI 完整指南

## 📋 目录

- [快速开始](#快速开始)
- [系统要求](#系统要求)
- [安装步骤](#安装步骤)
- [启动服务](#启动服务)
- [API 文档](#api-文档)
- [功能特性](#功能特性)
- [故障排除](#故障排除)
- [开发指南](#开发指南)

---

## 🚀 快速开始

### 一键启动（推荐）

```bash
cd medfusion-web
./start-webui.sh
```

启动后访问：
- **前端界面**: http://localhost:5173
- **后端 API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

### 停止服务

```bash
./stop-webui.sh
```

---

## 💻 系统要求

### 必需
- **Python**: 3.8+
- **Node.js**: 16+
- **npm**: 8+

### 推荐
- **操作系统**: Linux / macOS / Windows (WSL2)
- **内存**: 4GB+
- **磁盘空间**: 2GB+

---

## 📦 安装步骤

### 方法 1: 使用启动脚本（推荐）

启动脚本会自动完成所有安装步骤：

```bash
./start-webui.sh
```

### 方法 2: 手动安装

#### 1. 安装后端依赖

```bash
cd backend

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 初始化数据库
python scripts/init_db.py
```

#### 2. 安装前端依赖

```bash
cd frontend

# 安装依赖
npm install
```

---

## 🎯 启动服务

### 开发模式

#### 启动后端

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 启动前端

```bash
cd frontend
npm run dev
```

### 生产模式

#### 使用 Docker Compose

```bash
docker-compose up -d
```

#### 手动启动

```bash
# 后端
cd backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# 前端（构建后使用 nginx 等服务器）
cd frontend
npm run build
```

---

## 📚 API 文档

### API 端点总览

| 模块 | 端点数量 | 前缀 |
|------|---------|------|
| 工作流 | 9 | `/api/workflows` |
| 训练 | 7 | `/api/training` |
| 模型 | 11 | `/api/models` |
| 数据集 | 9 | `/api/datasets` |
| 系统 | 2 | `/api/system` |
| **总计** | **40** | - |

### 交互式 API 文档

访问 http://localhost:8000/docs 查看完整的交互式 API 文档（Swagger UI）

### 主要 API 端点

#### 工作流 API

```bash
# 获取所有可用节点
GET /api/workflows/nodes

# 创建工作流
POST /api/workflows/

# 执行工作流
POST /api/workflows/execute
```

#### 训练 API

```bash
# 启动训练
POST /api/training/start

# 获取训练状态
GET /api/training/status/{job_id}

# 停止训练
POST /api/training/stop/{job_id}
```

#### 模型 API

```bash
# 获取模型列表
GET /api/models/

# 上传模型文件
POST /api/models/{model_id}/upload

# 下载模型
GET /api/models/{model_id}/download
```

#### 数据集 API

```bash
# 获取数据集列表
GET /api/datasets/

# 创建数据集
POST /api/datasets/

# 分析数据集
POST /api/datasets/{dataset_id}/analyze
```

---

## ✨ 功能特性

### 1. 工作流编辑器

- ✅ 可视化拖拽编辑
- ✅ 4 种自定义节点（数据加载、模型、训练、评估）
- ✅ 节点配置面板
- ✅ 工作流保存和加载
- ✅ 实时执行和监控

### 2. 训练监控

- ✅ 训练任务列表
- ✅ 实时进度监控
- ✅ 训练曲线可视化
- ✅ 训练控制（暂停/恢复/停止）
- ✅ 日志查看

### 3. 模型库

- ✅ 模型列表和搜索
- ✅ 模型详情查看
- ✅ 模型上传和下载
- ✅ 模型统计信息
- ✅ 模型标签管理

### 4. 数据集管理

- ✅ 数据集列表和搜索
- ✅ 数据集详情查看
- ✅ 数据集统计信息
- ✅ 数据集分析
- ✅ 数据集标签管理

### 5. 系统监控

- ✅ 系统信息查看
- ✅ GPU 状态监控
- ✅ 资源使用统计

---

## 🔧 故障排除

### 问题 1: 端口被占用

**症状**: `Address already in use`

**解决方案**:

```bash
# 查找占用端口的进程
lsof -i :8000  # 后端
lsof -i :5173  # 前端

# 停止进程
kill -9 <PID>
```

### 问题 2: 数据库连接失败

**症状**: `Could not connect to database`

**解决方案**:

```bash
# 重新初始化数据库
cd backend
python scripts/init_db.py
```

### 问题 3: 前端无法连接后端

**症状**: `Network Error` 或 `CORS Error`

**解决方案**:

1. 检查后端是否启动：`curl http://localhost:8000/health`
2. 检查 CORS 配置：`backend/app/core/config.py`
3. 检查前端 API 地址：`frontend/src/api/index.ts`

### 问题 4: 依赖安装失败

**症状**: `pip install` 或 `npm install` 失败

**解决方案**:

```bash
# Python 依赖
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir

# Node.js 依赖
npm cache clean --force
npm install
```

### 问题 5: 虚拟环境激活失败

**症状**: `venv/bin/activate: No such file or directory`

**解决方案**:

```bash
# 重新创建虚拟环境
cd backend
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 👨‍💻 开发指南

### 项目结构

```
medfusion-web/
├── backend/                 # 后端服务
│   ├── app/
│   │   ├── api/            # API 路由
│   │   ├── crud/           # 数据库操作
│   │   ├── models/         # 数据模型
│   │   ├── core/           # 核心配置
│   │   └── main.py         # 应用入口
│   ├── scripts/            # 工具脚本
│   ├── requirements.txt    # Python 依赖
│   └── medfusion.db        # SQLite 数据库
│
├── frontend/               # 前端应用
│   ├── src/
│   │   ├── api/           # API 客户端
│   │   ├── components/    # React 组件
│   │   ├── pages/         # 页面
│   │   ├── hooks/         # 自定义 Hooks
│   │   └── App.tsx        # 应用入口
│   ├── package.json       # Node.js 依赖
│   └── vite.config.ts     # Vite 配置
│
├── start-webui.sh         # 启动脚本
├── stop-webui.sh          # 停止脚本
└── docker-compose.yml     # Docker 配置
```

### 添加新的 API 端点

#### 1. 创建数据模型（如果需要）

```python
# backend/app/models/database.py
class NewModel(Base):
    __tablename__ = "new_models"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
```

#### 2. 创建 CRUD 操作

```python
# backend/app/crud/new_model.py
class NewModelCRUD:
    @staticmethod
    def create(db: Session, name: str):
        model = NewModel(name=name)
        db.add(model)
        db.commit()
        return model
```

#### 3. 创建 API 路由

```python
# backend/app/api/new_model.py
from fastapi import APIRouter

router = APIRouter()

@router.post("/")
async def create_new_model(name: str):
    # 实现逻辑
    pass
```

#### 4. 注册路由

```python
# backend/app/main.py
from app.api import new_model

app.include_router(new_model.router, prefix="/api/new-model", tags=["new-model"])
```

#### 5. 创建前端 API 客户端

```typescript
// frontend/src/api/newModel.ts
export const createNewModel = async (name: string) => {
  const response = await api.post('/new-model/', { name })
  return response.data
}
```

### 代码规范

#### 后端（Python）

- 使用 **Black** 格式化代码
- 使用 **Ruff** 进行代码检查
- 遵循 **PEP 8** 规范
- 添加类型注解

```bash
# 格式化代码
black app/

# 代码检查
ruff check app/
```

#### 前端（TypeScript）

- 使用 **ESLint** 进行代码检查
- 使用 **Prettier** 格式化代码
- 遵循 **Airbnb** 风格指南

```bash
# 格式化代码
npm run format

# 代码检查
npm run lint
```

### 测试

#### 后端测试

```bash
cd backend
pytest tests/
```

#### 前端测试

```bash
cd frontend
npm run test
```

#### API 集成测试

```bash
cd backend
python test_api_integration.py
```

---

## 🔐 安全配置

### 生产环境配置

#### 1. 修改默认密钥

```python
# backend/app/core/config.py
SECRET_KEY = "your-secret-key-here"  # 使用强密钥
```

#### 2. 配置 CORS

```python
# backend/app/core/config.py
CORS_ORIGINS = [
    "https://your-domain.com",  # 只允许特定域名
]
```

#### 3. 使用 HTTPS

```bash
# 使用 nginx 反向代理
# 配置 SSL 证书
```

#### 4. 数据库安全

```bash
# 使用 PostgreSQL 替代 SQLite
# 配置数据库访问权限
```

---

## 📊 性能优化

### 后端优化

1. **使用多进程**

```bash
uvicorn app.main:app --workers 4
```

2. **启用缓存**

```python
# 使用 Redis 缓存
```

3. **数据库优化**

```python
# 添加索引
# 使用连接池
```

### 前端优化

1. **代码分割**

```typescript
// 使用动态导入
const Component = lazy(() => import('./Component'))
```

2. **资源压缩**

```bash
npm run build  # 自动压缩
```

3. **CDN 加速**

```html
<!-- 使用 CDN 加载静态资源 -->
```

---

## 📝 更新日志

### v0.2.0 (2024-02-20)

- ✅ 添加数据集管理功能
- ✅ 完善模型管理 API
- ✅ 创建一键启动脚本
- ✅ 完善文档

### v0.1.0 (2024-02-19)

- ✅ 完成后端 API 集成
- ✅ 完成前端核心功能
- ✅ 实现工作流编辑器
- ✅ 实现训练监控
- ✅ 实现模型库

---

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证。

---

## 📞 联系方式

- **项目主页**: https://github.com/your-org/medfusion
- **问题反馈**: https://github.com/your-org/medfusion/issues
- **文档**: https://docs.medfusion.ai

---

## 🙏 致谢

感谢所有贡献者和使用者！

---

**最后更新**: 2024-02-20  
**版本**: 0.2.0
