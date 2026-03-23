"""Tests guarding examples positioning and mainline guidance."""

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_examples_readme_declares_mainline_boundary() -> None:
    text = _read_text("examples/README.md")

    assert "官方推荐先走的 CLI / Web 训练主链" in text
    assert "configs/starter/" in text
    assert "configs/builder/" in text
    assert "不是当前 CLI/Web 训练主链" in text


def test_root_and_quickstart_link_to_examples_guide() -> None:
    assert "examples/README.md" in _read_text("README.md")
    assert "examples/README.md" in _read_text(
        "docs/contents/getting-started/quickstart.md",
    )


def test_key_examples_explain_their_scope() -> None:
    expectations = {
        "examples/train_demo.py": "Not the current official CLI / Web training mainline",
        "examples/model_builder_demo.py": (
            "Not equivalent to the current `medfusion train` config schema"
        ),
        "examples/smurf_usage.py": (
            "Not the current official end-to-end training entrypoint"
        ),
        "examples/attention_quick_start.py": "不属于当前官方 CLI / Web 训练主链",
        "examples/attention_supervision_example.py": "不属于当前官方 CLI / Web 训练主链",
    }

    for path, marker in expectations.items():
        assert marker in _read_text(path), path
