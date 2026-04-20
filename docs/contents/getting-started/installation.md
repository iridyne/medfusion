# 环境安装

> 文档状态：**Stable**

**预计时间：5 分钟**

本教程将指导你快速安装 MedFusion 及其依赖项。

## 前置要求

- Python 3.11 或更高版本
- Git
- （推荐）uv 包管理器

## 安装步骤

### Windows（当前优先，推荐手工命令安装）

在仓库根目录执行：

```powershell
# 1) 安装依赖
uv sync --extra dev --extra web

# 2) 验证 CLI
uv run medfusion --version

# 3) 启动 Web（首次建议不自动开浏览器）
uv run medfusion start --host 127.0.0.1 --port 8000 --no-browser
```

Windows 卸载（保留数据，推荐）：

```powershell
uv run medfusion uninstall --yes
```

Windows 彻底清理卸载（删除数据）：

```powershell
uv run medfusion uninstall --purge-data --yes
```

手工兜底（当卸载命令不可用时）：

```powershell
Remove-Item -LiteralPath .venv -Recurse -Force
Remove-Item -LiteralPath .venv, outputs, logs, checkpoints -Recurse -Force
Remove-Item -LiteralPath "$env:USERPROFILE\.medfusion" -Recurse -Force
```

> 说明：当前路线图优先推进 Windows 主线，Linux 与 Docker 安装落地在后置阶段。

### 1. 克隆仓库

```bash
git clone https://github.com/iridyne/medfusion.git
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
uv run medfusion train --help

# 可选：跑仓库 smoke 主链
bash test/smoke.sh
```

**预期输出：**

```
usage: medfusion train [-h] --config CONFIG [--output-dir OUTPUT_DIR]

Train a medical multimodal model

optional arguments:
  -h, --help         show this help message and exit
  --config CONFIG    Path to YAML configuration file
  --output-dir OUTPUT_DIR
                     Override output directory
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

安装完成后，继续学习 [你的第一个模型](first-model.md) 教程，在 30 分钟内完成第一个多模态模型的训练。

## 更多信息

- [文档总入口](../../README.md)
- [贡献与开发指南](../guides/development/contributing.md)
- [FAQ 和故障排除](../guides/core/faq.md)
