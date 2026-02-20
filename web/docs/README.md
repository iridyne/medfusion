# MedFusion WebUI 文档目录

本目录包含 MedFusion WebUI 的技术文档、集成报告和项目状态文档。

## 📁 目录结构

```
docs/
├── README.md                 # 本文件 - 文档索引
├── reports/                  # 技术集成报告
│   ├── API_INTEGRATION_REPORT.md
│   ├── BACKEND_API_INTEGRATION_SUMMARY.md
│   ├── DATABASE_INTEGRATION.md
│   ├── WEBSOCKET_INTEGRATION.md
│   ├── FRONTEND_ENHANCEMENT.md
│   ├── FRONTEND_OPTIMIZATION.md
│   └── OPTIMIZATION_REPORT.md
└── status/                   # 项目状态文档
    ├── FRONTEND_SUMMARY.md
    ├── PROJECT_STATUS.md
    └── WEBUI_COMPLETION_SUMMARY.md
```

## 📚 核心文档

以下核心文档位于项目根目录（`web/`）：

- **[README.md](../README.md)** - 项目主文档，包含项目介绍、功能特性、技术栈
- **[QUICKSTART.md](../QUICKSTART.md)** - 快速开始指南，5 分钟上手
- **[WEB_UI_GUIDE.md](../WEB_UI_GUIDE.md)** - 完整用户指南，详细功能说明

## 📊 技术集成报告 (`reports/`)

记录各个功能模块的集成过程和技术细节：

### 后端集成

- **[API_INTEGRATION_REPORT.md](reports/API_INTEGRATION_REPORT.md)**
  - 完整的 API 集成报告
  - 40 个 API 端点详细说明
  - 测试结果和使用示例

- **[BACKEND_API_INTEGRATION_SUMMARY.md](reports/BACKEND_API_INTEGRATION_SUMMARY.md)**
  - 后端 API 集成总结
  - 架构设计和实现要点

- **[DATABASE_INTEGRATION.md](reports/DATABASE_INTEGRATION.md)**
  - 数据库集成完成报告
  - 6 个数据表设计
  - CRUD 操作实现

- **[WEBSOCKET_INTEGRATION.md](reports/WEBSOCKET_INTEGRATION.md)**
  - WebSocket 实时通信集成
  - 双向控制实现
  - 自动重连机制

### 前端优化

- **[FRONTEND_ENHANCEMENT.md](reports/FRONTEND_ENHANCEMENT.md)**
  - 前端功能增强报告
  - 工作流编辑器、训练监控、模型库实现

- **[FRONTEND_OPTIMIZATION.md](reports/FRONTEND_OPTIMIZATION.md)**
  - 前端性能优化报告（491 行）
  - 虚拟滚动、懒加载、国际化、主题系统
  - 性能指标和最佳实践

- **[OPTIMIZATION_REPORT.md](reports/OPTIMIZATION_REPORT.md)**
  - 综合优化报告
  - 性能提升数据

## 📈 项目状态文档 (`status/`)

记录项目开发进度和完成情况：

- **[FRONTEND_SUMMARY.md](status/FRONTEND_SUMMARY.md)**
  - 前端开发总结
  - 功能清单和完成度

- **[PROJECT_STATUS.md](status/PROJECT_STATUS.md)**
  - 项目整体状态
  - 开发进度追踪

- **[WEBUI_COMPLETION_SUMMARY.md](status/WEBUI_COMPLETION_SUMMARY.md)**
  - WebUI 完成总结
  - 成果统计和下一步计划

## 🔍 快速导航

### 我想了解...

- **如何快速开始使用？** → [QUICKSTART.md](../QUICKSTART.md)
- **完整的功能说明？** → [WEB_UI_GUIDE.md](../WEB_UI_GUIDE.md)
- **API 接口文档？** → [API_INTEGRATION_REPORT.md](reports/API_INTEGRATION_REPORT.md)
- **数据库设计？** → [DATABASE_INTEGRATION.md](reports/DATABASE_INTEGRATION.md)
- **前端性能优化？** → [FRONTEND_OPTIMIZATION.md](reports/FRONTEND_OPTIMIZATION.md)
- **WebSocket 实时通信？** → [WEBSOCKET_INTEGRATION.md](reports/WEBSOCKET_INTEGRATION.md)
- **项目完成情况？** → [WEBUI_COMPLETION_SUMMARY.md](status/WEBUI_COMPLETION_SUMMARY.md)

## 📝 文档维护

### 文档分类原则

- **核心文档**：放在项目根目录，面向用户
- **技术报告**：放在 `docs/reports/`，面向开发者
- **状态文档**：放在 `docs/status/`，记录项目进度

### 更新频率

- 核心文档：功能变更时更新
- 技术报告：功能完成时创建，后续按需更新
- 状态文档：里程碑完成时更新

---

**最后更新**: 2026-02-20  
**维护者**: MedFusion Team