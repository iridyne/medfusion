# MedFusion 文档

这套文档服务于 MedFusion OSS：一个面向医学 AI 训练与验证的 open-core runtime。

如果你是第一次来，先不要全看，按你的目标选一条路径就行。

文档状态说明：
- **Stable**：接口和流程相对稳定，推荐优先参考
- **Beta**：仍在迭代，可能随版本调整
- **Legacy**：仅历史参考，不建议新项目按此落地

---

## 从这里开始（3 条路径）

### 1) 我想先把主链跑通（推荐）

- [CLI 与 Config 使用路径](contents/getting-started/cli-config-workflow.md)
- [快速上手](contents/getting-started/quickstart.md)
- [公开数据集快速验证](contents/getting-started/public-datasets.md)

适合：新用户、对外 demo、先要结果再深入。

### 2) 我想理解系统怎么工作的

- [Core Runtime Architecture](contents/architecture/CORE_RUNTIME_ARCHITECTURE.md)
- [Workflow 设计](contents/architecture/WORKFLOW_DESIGN.md)
- [Web UI 架构](contents/architecture/WEB_UI_ARCHITECTURE.md)

适合：架构评审、技术选型、二次开发前调研。

### 3) 我要做开发或贡献

- [安装](contents/getting-started/installation.md)
- [教程总览](contents/tutorials/README.md)
- [贡献指南](../CONTRIBUTING.md)

适合：准备改代码、补测试、提交 PR。

---

## 深入文档入口

- API 参考：
  - [med_core](contents/api/med_core.md)
  - [models](contents/api/models.md)
  - [backbones](contents/api/backbones.md)
  - [fusion](contents/api/fusion.md)
  - [heads](contents/api/heads.md)
  - [datasets](contents/api/datasets.md)
  - [trainers](contents/api/trainers.md)
  - [preprocessing](contents/api/preprocessing.md)
  - [evaluation](contents/api/evaluation.md)
  - [web_api](contents/api/web_api.md)

- 参考：
  - [错误代码](contents/reference/error_codes.md)

---

## 文档使用建议

- 先走“主链跑通”，再看架构和 API。
- 如果你要判断是否接入，优先看输出契约：`metrics.json / validation.json / summary.json / report.md`。
- 如果链接失效或内容过期，欢迎直接提 Issue 或 PR。
