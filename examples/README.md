# Developer Examples Guide

`examples/` 里的脚本主要用于演示 API、能力模块和历史搭建方式。

它的定位是 **开发者参考集合**，不是面向普通用户的模型搭建主入口。

如果你的目标是稳定地跑通当前 MedFusion，请先走下面这条主链：

```bash
uv run medfusion validate-config --config configs/starter/quickstart.yaml
uv run medfusion train --config configs/starter/quickstart.yaml
uv run medfusion build-results \
  --config configs/starter/quickstart.yaml \
  --checkpoint outputs/quickstart/checkpoints/best.pth
uv run medfusion import-run \
  --config configs/starter/quickstart.yaml \
  --checkpoint outputs/quickstart/checkpoints/best.pth
```

说明：

- 这是当前官方推荐先走的 CLI / Web 训练主链。
- 如果你的目标是“尽量少写代码、尽量快地完成模型搭建”，请优先停留在这条主链，不要从 `examples/` 起步。
- `configs/starter/`、`configs/public_datasets/`、`configs/testing/` 属于这条主链的配置。
- `configs/builder/` 属于 `MultiModalModelBuilder` / `build_model_from_config()` 的结构表达，不是当前 CLI/Web 训练主链的 schema。
- `examples/` 适合读代码、理解模块关系、做局部功能验证；不要默认把任意示例都当成 `medfusion train` 的等价替代。

## 谁应该看这个目录

- 想做二次开发，想看底层 Python API 如何拼装
- 想理解某个专题能力的大致调用方式
- 想验证某个模块级功能，而不是走完整主链

如果你只是想把模型搭起来、拿到标准结果目录、接到 Web 结果页，这个目录可以先忽略。

## 已迁出的工程演示

低用户价值、偏工程内部能力的 demo 会逐步迁到 `scripts/dev/`，避免继续占用 `examples/` 这个对外可见目录。

## 示例分类

| 文件 | 类型 | 用途 | 与当前主链的关系 |
| --- | --- | --- | --- |
| `train_demo.py` | 历史端到端脚本 | 展示最早期的 Python API 手工拼装流程 | 能帮助理解框架如何起盘，但不是当前官方训练入口 |
| `model_builder_demo.py` | Builder API 演示 | 展示 fluent builder 的模型结构拼装 | 不是 `medfusion train` 的配置 schema，也不等价于 CLI/Web 主链 |
| `smurf_usage.py` | 模型 API 演示 | 展示 SMuRF / SMuRF with MIL 的直接调用 | 适合理解模型接口，不是训练主链 |
| `attention_quick_start.py` | 功能速览 | 快速了解注意力监督配置思路 | 偏概念说明，不是完整训练闭环 |
| `attention_supervision_example.py` | 专题功能演示 | 深入演示 mask / CAM 注意力监督 | 偏研究特性演示，不是主入口 |
| `advanced_attention_demo.py` | 专题功能演示 | 展示 SE / ECA / CBAM / Transformer attention | 模块能力演示，不是主入口 |
| `model_export_demo.py` | 部署演示 | 演示 ONNX / TorchScript 导出 | 部署专题，不是训练主链 |

## 建议阅读顺序

1. 先用 `configs/starter/quickstart.yaml` 跑通 CLI 主链。
2. 再看 `train_demo.py`，理解底层是如何手工组装 dataset、fusion 和 trainer 的。
3. 如果你关心灵活拼装模型，再看 `model_builder_demo.py` 和 `configs/builder/`。
4. 如果你关心专项能力，再按主题看 attention、cache、distributed、export 等示例。

## 什么时候不要从 examples 开始

- 你想验证当前 Web UI / dashboard 是否可用。
- 你想确认某个 YAML 能不能直接交给 `medfusion train`。
- 你要做对外演示，且希望路径和文档、CLI、结果页完全一致。

这三种情况都应该先走 `configs/starter/` 或 `configs/public_datasets/`。
