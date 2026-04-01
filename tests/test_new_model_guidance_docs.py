"""Documentation guards for new-model guidance in beginner-facing docs."""

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


GUIDE_PATH = "docs/contents/getting-started/model-creation-paths.md"


def test_model_creation_guide_declares_three_supported_paths() -> None:
    text = _read_text(GUIDE_PATH)

    assert "复制一份主链 config 模板" in text
    assert "Builder / 代码做结构实验" in text
    assert "先扩 runtime，再扩 YAML" in text
    assert "Run Wizard" in text
    assert "不是一个任意发明新模型的生成器" in text


def test_beginner_facing_docs_link_to_model_creation_guide() -> None:
    expected_link = "model-creation-paths"
    beginner_docs = [
        "README.md",
        "docs/README.md",
        "docs/index.md",
        "docs/.vitepress/config.mts",
        "docs/contents/getting-started/quickstart.md",
        "docs/contents/getting-started/cli-config-workflow.md",
        "docs/contents/getting-started/public-datasets.md",
        "docs/contents/getting-started/web-ui.md",
        "docs/contents/getting-started/first-model.md",
        "docs/contents/tutorials/README.md",
        "docs/contents/tutorials/fundamentals/configs.md",
        "docs/contents/tutorials/fundamentals/builder-api.md",
        "docs/contents/tutorials/training/workflow.md",
    ]

    for path in beginner_docs:
        assert expected_link in _read_text(path), path


def test_cli_and_tutorial_docs_repeat_the_path_boundary_clearly() -> None:
    expectations = {
        "docs/contents/getting-started/cli-config-workflow.md": [
            "复制一份最接近的主链 YAML 模板",
            "Builder / 代码路径",
            "先扩 runtime，再把它暴露给 YAML",
        ],
        "docs/contents/getting-started/web-ui.md": [
            "RunSpec / ExperimentConfig 生成器",
            "不会替你发明一个全新的模型能力",
        ],
        "docs/contents/getting-started/first-model.md": [
            "不是当前最推荐的新手起点",
            "复制主链模板",
        ],
        "docs/contents/tutorials/README.md": [
            "普通用户",
            "高级用户",
            "真正新的模型能力",
        ],
        "docs/contents/tutorials/fundamentals/configs.md": [
            "只能在当前 runtime 已经支持的组件范围内组合",
            "先扩 runtime，再扩 YAML",
        ],
        "docs/contents/tutorials/fundamentals/builder-api.md": [
            "不是当前 `medfusion train` 主链 YAML",
            "结构实验",
        ],
        "docs/contents/tutorials/training/workflow.md": [
            "新建模型 YAML",
            "复制主链模板",
        ],
    }

    for path, markers in expectations.items():
        text = _read_text(path)
        for marker in markers:
            assert marker in text, (path, marker)


def test_beginner_learning_path_prefers_mainline_over_handwritten_demo() -> None:
    text = _read_text("docs/contents/tutorials/README.md")

    assert "CLI 与 Config 使用路径" in text
    assert "快速上手" in text
    assert "公开数据集快速验证" in text
    assert "第一个模型" in text
    assert "教学型补充" in text


def test_web_docs_name_one_default_entry_and_demote_legacy_paths() -> None:
    web_text = _read_text("docs/contents/getting-started/web-ui.md")
    workflow_text = _read_text("docs/contents/tutorials/training/workflow.md")

    assert "大多数新手只需要这一条命令" in web_text
    assert "uv run medfusion start" in web_text
    assert "高级或兼容场景再看下面几种方式" in web_text
    assert "uv run medfusion start" in workflow_text
    assert "兼容或开发调试时再看旧入口" in workflow_text


def test_onboarding_docs_describe_start_as_guided_entry_and_yaml_as_mainline() -> None:
    readme_text = _read_text("README.md")
    web_text = _read_text("docs/contents/getting-started/web-ui.md")

    assert "Getting Started" in web_text
    assert "YAML 主链" in web_text
    assert "medfusion start" in readme_text
    assert "先成功跑通一次，再迁移到自己的 YAML" in readme_text


def test_configs_tutorial_starts_from_minimal_mainline_template() -> None:
    text = _read_text("docs/contents/tutorials/fundamentals/configs.md")

    assert "最小可运行模板" in text
    assert "configs/starter/quickstart.yaml" in text
    assert "第一次上手只需要先改这几个地方" in text
    assert "后面的章节属于扩展字段参考" in text
