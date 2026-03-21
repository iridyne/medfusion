# MedFusion 快速参考

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-org/medfusion.git
cd medfusion

# 安装依赖
uv pip install -e ".[dev]"

# 验证安装
python -c "import med_core; print(med_core.__version__)"
```

## 基本使用

### 训练模型

```bash
# 最快开始
medfusion train --config configs/starter/quickstart.yaml

# 使用自定义配置
medfusion train --config configs/my_config.yaml

# 指定输出目录
medfusion train --config configs/starter/quickstart.yaml --output-dir outputs/exp1
```

### 评估模型

```bash
# 评估检查点
medfusion evaluate \
  --config configs/starter/quickstart.yaml \
  --checkpoint outputs/quickstart/checkpoints/best.pth
```

## 配置文件模板

```yaml
# configs/my_config.yaml
project_name: "my-project"
experiment_name: "my-exp"

model:
  num_classes: 2

  vision:
    backbone: "resnet18"
    pretrained: true
    freeze_backbone: false
    feature_dim: 128

  tabular:
    hidden_dims: [32]
    output_dim: 16

data:
  csv_path: "data/mock/metadata.csv"
  image_dir: "data/mock"
  image_path_column: "image_path"
  target_column: "diagnosis"
  batch_size: 32
  num_workers: 4
  image_size: 224

training:
  num_epochs: 50
  optimizer:
    optimizer: "adamw"
    learning_rate: 0.001
  mixed_precision: true
```

## 常用命令

### 数据准备

```bash
# 生成模拟数据
python scripts/generate_mock_data.py

# 验证数据格式
python -c "from med_core.datasets import load_dataset; ds = load_dataset('data/')"
```

### 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_config_validation.py -v

# 带覆盖率
pytest tests/ --cov=med_core --cov-report=html
```

### 代码质量

```bash
# Linting
ruff check med_core/ tests/

# 格式化
ruff format med_core/ tests/

# 类型检查
mypy med_core/ --ignore-missing-imports

# Pre-commit
pre-commit run --all-files
```

## Docker

```bash
# 构建镜像
docker-compose build

# 启动 Web UI（GPU 版本）
docker compose -f docker/docker-compose.yml up medfusion-web

# 启动 CPU 版本
docker compose -f docker/docker-compose.yml --profile cpu up medfusion-web-cpu

# 启动开发容器
docker compose -f docker/docker-compose.yml --profile dev up medfusion-dev
```

## 调试

```bash
# 启用详细日志（Web UI）
export MEDFUSION_LOG_LEVEL=DEBUG
medfusion train --config configs/starter/quickstart.yaml

# 检查 GPU
python -c "import torch; print(torch.cuda.is_available())"

# 内存分析
python -c "import torch; print(torch.cuda.memory_summary())"
```

## 错误代码

| 代码 | 类型 | 描述 |
|------|------|------|
| E001-E030 | 配置 | 配置验证错误 |
| E100-E199 | 配置 | 配置加载错误 |
| E200-E299 | 数据 | 数据集错误 |
| E300-E399 | 模型 | 模型错误 |
| E400-E499 | 训练 | 训练错误 |

## 环境变量

```bash
# 日志级别
export MEDFUSION_LOG_LEVEL=INFO

# Web UI 数据目录
export MEDFUSION_DATA_DIR=/path/to/data

# GPU 设备
export CUDA_VISIBLE_DEVICES=0,1
```

## 性能优化

```yaml
# 混合精度训练
training:
  use_amp: true

# 梯度累积
training:
  gradient_accumulation_steps: 4

# 数据加载优化
data:
  num_workers: 8
  pin_memory: true
  persistent_workers: true
```

## 常见问题快速修复

### CUDA out of memory
```yaml
training:
  batch_size: 8  # 减小
  use_amp: true  # 启用混合精度
```

### Loss 变成 NaN
```yaml
training:
  optimizer:
    lr: 0.0001  # 降低学习率
  max_grad_norm: 1.0  # 梯度裁剪
```

### 训练速度慢
```yaml
data:
  num_workers: 8  # 增加 workers
  prefetch_factor: 2
training:
  use_amp: true  # 混合精度
```

## 资源链接

- 📖 完整文档: `docs/`
- 🐛 报告问题: GitHub Issues
- 💬 讨论: GitHub Discussions
- 📧 联系: your-email@example.com

## 版本信息

```bash
# 查看版本
python -c "import med_core; print(med_core.__version__)"

# 查看依赖
uv pip list
```
