# 环境安装

**预计时间：5 分钟**

本教程将指导你快速安装 MedFusion 及其依赖项。

## 前置要求

- Python 3.11 或更高���本
- Git
- （推荐）uv 包管理器

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/medfusion.git
cd medfusion
```

### 2. 安装依赖

**推荐方式：使用 uv**

```bash
# 安装基础依赖
uv sync

# 安装开发依赖（包含测试、代码检查工具）
uv sync --extra dev

# 安装 Web UI 依赖
uv sync --extra web
```

**备选方式：使用 pip**

```bash
# 安装基础依赖
pip install -e .

# 安装所有可选依赖
pip install -e ".[dev,web]"
```

### 3. 验证安装

运行以下命令验证安装是否成功：

```bash
# 检查命令行工具
uv run med-train --help

# 运行测试（可选）
uv run pytest -q
```

**预期输出：**

```
usage: med-train [-h] --config CONFIG [--resume RESUME]

Train a medical multimodal model

optional arguments:
  -h, --help         show this help message and exit
  --config CONFIG    Path to config file
  --resume RESUME    Path to checkpoint to resume from
```

## 常见问题

### Q: 如何安装 uv？

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Q: 遇到 CUDA 相关错误怎么办？

确保你的 PyTorch 版本与 CUDA 版本匹配。访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取正确的安装命令。

### Q: 安装速度很慢？

可以使用国内镜像源：

```bash
uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 下一步

安装完成后，继续学习 [你的第一个模型](02_first_model.md) 教程，在 30 分钟内完成第一个多模态模型的训练。

## 更多信息

- [完整安装文档](../../README.md#安装)
- [开发环境配置](../../development/setup.md)
- [故障排除指南](../../guides/troubleshooting.md)
