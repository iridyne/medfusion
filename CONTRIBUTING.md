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
4. 提交前跑本地 smoke / handoff：`bash scripts/full_regression.sh --ci`
5. 如需更重的本地非-pytest 检查，再跑：`bash scripts/full_regression.sh --full`
6. 推送分支后让 GitHub Actions CI 跑 `pytest`
7. 如果 CI 失败，检查 Actions 日志，或运行：`bash scripts/inspect_ci_failure.sh`
8. 提交更改：`git commit -m "feat: your feature"`
9. 推送分支：`git push origin feature/your-feature`
10. 创建 Pull Request

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
`pytest` 不在这个模式里运行。

### ci

```bash
bash scripts/full_regression.sh --ci
```

这个模式负责本地 smoke 和提交流程 handoff，不在本地执行 `pytest`。
`pytest` 的真源已经迁移到 GitHub Actions CI。

### full

```bash
bash scripts/full_regression.sh --full
```

当前 `--full` 已内置更完整的本地非-pytest 检查；`scripts/local_ci_test.sh` 仅作为兼容包装保留。

## CI 失败排查

`pytest` 当前固定在 GitHub Actions CI 里执行：

- `.github/workflows/ci.yml`

如需直接查看最近失败日志：

```bash
bash scripts/inspect_ci_failure.sh
```

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
- 提交前建议运行 `bash scripts/full_regression.sh --ci`
- `pytest` 由 GitHub Actions CI 负责，失败时按 CI 日志修复
- 如需更完整的本地非-pytest 检查，再运行 `--full`
- 新增流程性约定时，请同步更新脚本和文档，避免只存在于口头约定里

## 问题反馈

使用 GitHub Issues 报告问题，请包含：

- 问题描述
- 复现步骤
- 预期行为
- 实际行为
- 环境信息（Python 版本、PyTorch 版本等）
