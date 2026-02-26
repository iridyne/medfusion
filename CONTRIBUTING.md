# 贡献指南

感谢你对 MedFusion 项目的关注！

## 开发环境设置

1. Fork 并克隆仓库
2. 安装依赖: `uv sync`
3. 安装 pre-commit hooks: `pre-commit install`

## 开发流程

1. 创建功能分支: `git checkout -b feature/your-feature`
2. 编写代码和测试
3. 运行测试: `uv run pytest`
4. 代码格式化: `ruff format .`
5. 类型检查: `mypy med_core`
6. 提交更改: `git commit -m "feat: your feature"`
7. 推送分支: `git push origin feature/your-feature`
8. 创建 Pull Request

## 代码规范

- 遵循 PEP 8 风格指南
- 使用类型注解
- 编写单元测试
- 保持函数简洁（< 50 行）
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
- 测试覆盖率不低于 80%
- 确保所有测试通过

## 问题反馈

使用 GitHub Issues 报告问题，请包含：

- 问题描述
- 复现步骤
- 预期行为
- 实际行为
- 环境信息（Python 版本、PyTorch 版本等）
