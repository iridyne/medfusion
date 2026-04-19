from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_readme_front_screen_declares_gui_runtime_results_positioning() -> None:
    text = _read_text("README.md")
    assert "研究运行时 + GUI 模型搭建主链 + 真实结果闭环" in text
    assert "GUI-first for users, engine-first internally, Web-first for deployment" in text
    assert "Node 后端" in text


def test_why_doc_demotes_full_builder_narrative() -> None:
    text = _read_text("docs/contents/guides/core/why-medfusion-oss.md")
    assert "高级模式已有节点图 preview" in text
    assert "全能 builder" in text
    assert "GUI-first 的模型搭建前台" in text


def test_faq_declares_current_product_boundary() -> None:
    text = _read_text("docs/contents/guides/core/faq.md")
    assert "正式版高级模式 preview" in text
    assert "为什么当前不引入 Node 后端" in text
    assert "直接创建真实训练任务" in text
