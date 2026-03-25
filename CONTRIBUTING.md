# 贡献指南

感谢你对 MedFusion 项目的关注！

## 开发环境设置

1. Fork 并克隆仓库
2. 安装依赖：`uv sync`
3. 如需本地钩子，可安装：`pre-commit install`

## 开发流程

建议按这个节奏来：

1. 创建功能分支：`git checkout -b feature/your-feature`
2. 编写代码和测试
3. 先跑快速验证：`bash scripts/full_regression.sh --quick`
4. 准备提交较大改动时，再跑：`bash scripts/full_regression.sh --ci`
5. 如需更重的本地整套检查，再跑：`bash scripts/full_regression.sh --full`
6. 提交更改：`git commit -m "feat: your feature"`
7. 推送分支：`git push origin feature/your-feature`
8. 创建 Pull Request

## 验证模式说明

仓库当前的统一验证入口是：

```bash
bash scripts/full_regression.sh --help
```

### quick

```bash
bash scripts/full_regression.sh --quick
```

适合日常开发后的最小自检。它当前只覆盖验证工作流相关的关键文件和最小测试集，不负责一次性清理整个仓库的历史格式化债。

### ci

```bash
bash scripts/full_regression.sh --ci
```

尽量对齐当前 GitHub CI。这个模式会显式忽略：

- `tests/test_config_validation.py`
- `tests/test_export.py`

这两项是当前 CI 对齐约定的一部分。

### full

```bash
bash scripts/full_regression.sh --full
```

会调用当前更完整的本地检查入口：`scripts/local_ci_test.sh`。

## 代码规范

- 遵循 PEP 8 风格指南
- 使用类型注解
- 编写单元测试
- 保持函数简洁
- 添加必要的文档字符串

## 提交信息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

- `feat:` 新功能
- `fix:` 错误修复
- `docs:` 文档更新
- `refactor:` 代码重构
- `test:` 测试相关
- `chore:` 构建/工具相关

## 测试要求

- 所有新功能必须包含测试
- 提交前至少运行 `bash scripts/full_regression.sh --quick`
- 涉及较大改动时，建议运行 `--ci` 或 `--full`
- 新增流程性约定时，请同步更新脚本和文档，避免只存在于口头约定里

## 问题反馈

使用 GitHub Issues 报告问题，请包含：

- 问题描述
- 复现步骤
- 预期行为
- 实际行为
- 环境信息（Python 版本、PyTorch 版本等）
