# CI/CD 管道文档

本文档介绍 MedFusion 项目的持续集成和持续部署（CI/CD）流程。

## 目录

- [概述](#概述)
- [工作流](#工作流)
- [Pre-commit Hooks](#pre-commit-hooks)
- [本地开发](#本地开发)
- [故障排查](#故障排查)

---

## 概述

MedFusion 使用 GitHub Actions 实现自动化的 CI/CD 流程，包括：

- ✅ 代码质量检查（linting, formatting, type checking）
- ✅ 自动化测试（单元测试、集成测试）
- ✅ 安全扫描
- ✅ Docker 镜像构建
- ✅ 文档构建
- ✅ 自动发布

---

## 工作流

### 1. CI Pipeline (`.github/workflows/ci.yml`)

**触发条件**:
- Push 到 `main` 或 `develop` 分支
- Pull Request 到 `main` 或 `develop` 分支
- 手动触发

**包含的任务**:

#### 代码质量检查
```yaml
- Ruff linting (代码规范)
- Ruff formatting (代码格式)
- Mypy type checking (类型检查)
```

#### 测试
```yaml
- Python 3.10, 3.11, 3.12 多版本测试
- 单元测试
- 代码覆盖率报告
- 上传到 Codecov
```

#### 集成测试
```yaml
- 生成模拟数据
- 端到端测试
- 冒烟测试
```

#### Docker 构建
```yaml
- 构建 Docker 镜像
- 测试镜像可用性
- 使用 BuildKit 缓存
```

#### 安全扫描
```yaml
- Bandit (安全漏洞检测)
- Safety (依赖安全检查)
```

#### 文档
```yaml
- 检查文档链接
- 验证示例代码
```

**查看状态**:
```bash
# 在 GitHub 仓库页面查看
https://github.com/your-org/medfusion/actions
```

---

### 2. Release Pipeline (`.github/workflows/release.yml`)

**触发条件**:
- Push 带有 `v*.*.*` 格式的 tag
- 手动触发

**包含的任务**:

#### 创建发布
```yaml
- 运行完整测试套件
- 构建 Python 包
- 生成变更日志
- 创建 GitHub Release
```

#### Docker 发布
```yaml
- 构建多架构镜像 (amd64, arm64)
- 推送到 GitHub Container Registry
- 标签: version, major.minor, major, latest
```

#### PyPI 发布（可选）
```yaml
- 构建 wheel 和 sdist
- 发布到 PyPI
```

#### 文档发布
```yaml
- 构建文档
- 部署到 GitHub Pages
```

**发布流程**:

```bash
# 1. 更新版本号
# 编辑 med_core/version.py
__version__ = "1.2.0"

# 2. 提交更改
git add med_core/version.py
git commit -m "chore: bump version to 1.2.0"

# 3. 创建标签
git tag -a v1.2.0 -m "Release version 1.2.0"

# 4. 推送标签
git push origin v1.2.0

# 5. GitHub Actions 自动执行发布流程
```

---

### 3. Code Quality Pipeline (`.github/workflows/code-quality.yml`)

**触发条件**:
- Pull Request
- Push 到主分支

**包含的检查**:

#### Ruff Linting
```yaml
- 代码规范检查
- 格式检查
- 自动修复建议
```

#### Type Checking
```yaml
- Mypy 类型检查
- 严格模式
- 类型覆盖率
```

#### 代码复杂度
```yaml
- 圈复杂度分析
- 可维护性指数
- 复杂度阈值检查
```

#### 文档质量
```yaml
- Docstring 覆盖率
- Docstring 风格检查
```

#### 导入顺序
```yaml
- isort 检查
- 自动排序
```

#### 死代码检测
```yaml
- Vulture 检测未使用代码
```

#### 代码重复
```yaml
- PMD CPD 检测重复代码
```

---

## Pre-commit Hooks

Pre-commit hooks 在提交代码前自动运行检查，确保代码质量。

### 安装

```bash
# 安装 pre-commit
pip install pre-commit

# 安装 hooks
pre-commit install

# 安装 commit-msg hook
pre-commit install --hook-type commit-msg
```

### 使用

```bash
# 自动运行（每次 git commit 时）
git commit -m "feat: add new feature"

# 手动运行所有文件
pre-commit run --all-files

# 运行特定 hook
pre-commit run ruff --all-files

# 跳过 hooks（不推荐）
git commit -m "message" --no-verify
```

### 包含的 Hooks

1. **Ruff**: 代码检查和格式化
2. **Mypy**: 类型检查
3. **isort**: 导入排序
4. **Bandit**: 安全检查
5. **通用检查**: 
   - 大文件检查
   - 合并冲突检查
   - YAML/TOML/JSON 验证
   - 私钥检测
   - 文件末尾修复
   - 尾随空格修复
6. **Markdown**: Markdown 格式化
7. **Dockerfile**: Dockerfile 检查
8. **Shell**: Shell 脚本检查
9. **Commitizen**: 提交消息规范

### 配置

编辑 `.pre-commit-config.yaml` 自定义配置：

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
```

---

## 本地开发

### 运行测试

```bash
# 所有测试
pytest tests/ -v

# 带覆盖率
pytest tests/ --cov=med_core --cov-report=html

# 特定测试
pytest tests/test_config_validation.py -v

# 并行测试
pytest tests/ -n auto
```

### 代码质量检查

```bash
# Ruff 检查
ruff check med_core/ tests/

# Ruff 格式化
ruff format med_core/ tests/

# Mypy 类型检查
mypy med_core/ --ignore-missing-imports

# 所有检查
pre-commit run --all-files
```

### 构建 Docker 镜像

```bash
# 构建
docker build -t medfusion:dev .

# 测试
docker run --rm medfusion:dev python -c "import med_core; print(med_core.__version__)"

# 使用 docker-compose
docker-compose build
```

### 生成覆盖率报告

```bash
# 运行测试并生成报告
pytest tests/ --cov=med_core --cov-report=html

# 查看报告
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

---

## 徽章

在 README.md 中添加状态徽章：

```markdown
![CI](https://github.com/your-org/medfusion/workflows/CI%2FCD%20Pipeline/badge.svg)
![Code Quality](https://github.com/your-org/medfusion/workflows/Code%20Quality/badge.svg)
[![codecov](https://codecov.io/gh/your-org/medfusion/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/medfusion)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
```

---

## 故障排查

### 问题 1: Pre-commit hooks 失败

**症状**:
```
[INFO] Installing environment for https://github.com/astral-sh/ruff-pre-commit.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
An unexpected error has occurred
```

**解决方案**:
```bash
# 清理 pre-commit 缓存
pre-commit clean

# 重新安装
pre-commit install --install-hooks

# 更新 hooks
pre-commit autoupdate
```

### 问题 2: GitHub Actions 失败

**症状**: CI 工作流失败

**解决方案**:
```bash
# 1. 查看详细日志
# 在 GitHub Actions 页面查看失败的步骤

# 2. 本地复现
# 使用 act 在本地运行 GitHub Actions
brew install act  # macOS
act -j test  # 运行 test job

# 3. 检查依赖
uv pip list
uv pip check
```

### 问题 3: Docker 构建失败

**症状**: Docker 镜像构建失败

**解决方案**:
```bash
# 清理 Docker 缓存
docker builder prune -a

# 无缓存构建
docker build --no-cache -t medfusion:test .

# 检查 Dockerfile 语法
docker build --check -t medfusion:test .
```

### 问题 4: 测试覆盖率低

**症状**: 覆盖率低于预期

**解决方案**:
```bash
# 查看未覆盖的代码
pytest tests/ --cov=med_core --cov-report=term-missing

# 生成详细报告
pytest tests/ --cov=med_core --cov-report=html
open htmlcov/index.html

# 添加更多测试
# 编辑 tests/ 目录下的测试文件
```

### 问题 5: Mypy 类型错误

**症状**: 类型检查失败

**解决方案**:
```bash
# 查看详细错误
mypy med_core/ --show-error-codes

# 忽略特定错误（临时）
# type: ignore[error-code]

# 添加类型注解
def function(x: int) -> str:
    return str(x)
```

---

## 最佳实践

### 1. 提交前检查

```bash
# 运行所有检查
pre-commit run --all-files

# 运行测试
pytest tests/ -v

# 检查覆盖率
pytest tests/ --cov=med_core --cov-report=term
```

### 2. 提交消息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```bash
# 功能
git commit -m "feat: add new attention mechanism"

# 修复
git commit -m "fix: resolve memory leak in data loader"

# 文档
git commit -m "docs: update installation guide"

# 样式
git commit -m "style: format code with ruff"

# 重构
git commit -m "refactor: simplify config validation"

# 测试
git commit -m "test: add tests for fusion module"

# 构建
git commit -m "chore: update dependencies"
```

### 3. Pull Request 流程

```bash
# 1. 创建分支
git checkout -b feature/new-feature

# 2. 开发和提交
git add .
git commit -m "feat: implement new feature"

# 3. 推送到远程
git push origin feature/new-feature

# 4. 创建 Pull Request
# 在 GitHub 上创建 PR

# 5. 等待 CI 通过
# 查看 GitHub Actions 状态

# 6. 代码审查
# 响应审查意见

# 7. 合并
# 由维护者合并到主分支
```

### 4. 发布流程

```bash
# 1. 确保所有测试通过
pytest tests/ -v

# 2. 更新版本号
# 编辑 med_core/version.py

# 3. 更新 CHANGELOG
# 编辑 CHANGELOG.md

# 4. 提交更改
git add .
git commit -m "chore: prepare release v1.2.0"

# 5. 创建标签
git tag -a v1.2.0 -m "Release v1.2.0"

# 6. 推送
git push origin main --tags

# 7. 等待自动发布
# GitHub Actions 自动创建 release
```

---

## 配置文件

### pyproject.toml

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "C", "N", "D"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.coverage.run]
source = ["med_core"]
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]
```

---

## 参考资源

- [GitHub Actions 文档](https://docs.github.com/en/actions)
- [Pre-commit 文档](https://pre-commit.com/)
- [Ruff 文档](https://docs.astral.sh/ruff/)
- [Pytest 文档](https://docs.pytest.org/)
- [Codecov 文档](https://docs.codecov.com/)

---

**最后更新**: 2026-02-20
