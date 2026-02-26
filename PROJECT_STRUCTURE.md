# MedFusion 项目结构

## 根目录文件

### 配置文件
- `pyproject.toml` - Python 项目配置和依赖管理
- `uv.lock` - 依赖锁定文件
- `.nvmrc` - Node.js 版本锁定
- `.editorconfig` - 编辑器配置
- `.gitignore` - Git 忽略规则
- `.dockerignore` - Docker 构建忽略规则
- `.pre-commit-config.yaml` - Git pre-commit hooks

### 文档文件
- `README.md` - 项目主文档
- `CONTRIBUTING.md` - 贡献指南
- `CHANGELOG.md` - 版本变更日志
- `LICENSE` - MIT 许可证

### 构建和部署
- `Dockerfile` - Docker 镜像构建
- `docker-compose.yml` - Docker 服务编排
- `Makefile` - 开发命令快捷方式
- `start-webui.sh` - Web UI 启动脚本

## 核心目录

### `med_core/` - Python 核心库
```
med_core/
├── models/              # 模型架构
│   ├── backbones/       # 视觉骨干网络 (29 种)
│   ├── fusion/          # 融合策略 (5 种)
│   ├── aggregators/     # 特征聚合器
│   └── heads/           # 分类和生存分析头
├── datasets/            # 数据加载器
├── trainers/            # 训练逻辑
├── configs/             # 配置系统
├── preprocessing/       # 图像预处理
├── evaluation/          # 评估和报告
├── attention_supervision/  # 注意力监督
├── web/                 # FastAPI Web 服务
│   ├── app.py           # 主应用入口
│   ├── api/             # API 端点
│   ├── routers/         # 路由
│   └── models/          # 数据模型
└── utils/               # 工具函数
```

### `med_core_rs/` - Rust 加速模块
使用 PyO3 集成的性能关键模块。

### `web/frontend/` - React 前端
```
web/frontend/
├── src/
│   ├── components/      # React 组件
│   ├── pages/           # 页面
│   ├── services/        # API 服务
│   └── stores/          # 状态管理 (Zustand)
├── public/              # 静态资源
└── package.json         # Node.js 依赖
```

### `configs/` - YAML 配置模板
预定义的训练配置文件，支持快速实验切换。

### `examples/` - 使用示例
演示如何使用框架的各种功能。

### `tests/` - 测试套件
单元测试和集成测试。

### `scripts/` - 辅助脚本
数据处理、模型转换等工具脚本。

### `docs/` - 文档
```
docs/
├── api/                 # API 参考文档
├── guides/              # 使用指南
├── development/         # 开发文档
│   └── AGENTS.md        # AI Agent 开发记录
└── ROADMAP.md           # 项目路线图
```

## 数据和输出目录

### `data/` - 数据目录
```
data/
├── raw/                 # 原始数据 (gitignored)
├── processed/           # 预处理后数据 (gitignored)
└── examples/            # 示例数据
```

### `outputs/` - 训练输出
```
outputs/
├── checkpoints/         # 模型检查点
├── logs/                # 训练日志
└── reports/             # 评估报告
```

### `notebooks/` - Jupyter Notebooks
实验和分析笔记本。

### `benchmarks/` - 性能基准测试
模型性能和速度基准。

## 特殊目录

### `.archive/` - 归档文件
- 临时文档和会话状态
- 备份文件
- 旧版本代码 (legacy-src)

### `.github/` - GitHub 配置
CI/CD 工作流和 issue 模板。

### `.claude/` - Claude Code 配置
AI 辅助开发的项目规则和记忆。

## 缓存目录 (gitignored)
- `.venv/` - Python 虚拟环境
- `.mypy_cache/` - mypy 类型检查缓存
- `.pytest_cache/` - pytest 测试缓存
- `.ruff_cache/` - ruff linter 缓存

## 开发工作流

1. **安装依赖**: `make install` 或 `uv sync`
2. **开发模式**: `make dev`
3. **运行测试**: `make test`
4. **代码格式化**: `make format`
5. **类型检查**: `make typecheck`
6. **启动 Web UI**: `./start-webui.sh`

## 配置驱动架构

所有实验通过 `configs/*.yaml` 定义：
- 选择骨干网络 (29 种)
- 选择融合策略 (5 种)
- 选择聚合器 (5 种)
- 配置训练参数
- 无需修改代码

## 模块化设计

- **解耦**: 骨干网络、融合策略、聚合器完全独立
- **可插拔**: 通过配置文件快速切换组件
- **可扩展**: 易于添加新的模型和策略
