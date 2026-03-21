"""
Tests for CLI module structure and imports.
"""

from pathlib import Path
import tomllib


def test_cli_imports():
    """Test that CLI functions can be imported."""
    from med_core.cli import evaluate, import_run, preprocess, train

    assert callable(train)
    assert callable(evaluate)
    assert callable(preprocess)
    assert callable(import_run)


def test_cli_submodule_imports():
    """Test that CLI submodules can be imported directly."""
    from med_core.cli.evaluate import evaluate
    from med_core.cli.import_run import import_run
    from med_core.cli.preprocess import preprocess
    from med_core.cli.train import train

    assert callable(train)
    assert callable(evaluate)
    assert callable(preprocess)
    assert callable(import_run)


def test_cli_backward_compatibility():
    """Test that old import path still works."""
    from med_core.cli import evaluate, import_run, preprocess, train

    # Verify functions are accessible
    assert train.__module__ == "med_core.cli.train"
    assert evaluate.__module__ == "med_core.cli.evaluate"
    assert preprocess.__module__ == "med_core.cli.preprocess"
    assert import_run.__module__ == "med_core.cli.import_run"


def test_cli_module_structure():
    """Test CLI module structure."""
    import med_core.cli as cli_module

    # Check __all__ exports
    assert hasattr(cli_module, "__all__")
    assert "train" in cli_module.__all__
    assert "evaluate" in cli_module.__all__
    assert "preprocess" in cli_module.__all__

    # Check all exported functions exist
    for func_name in cli_module.__all__:
        assert hasattr(cli_module, func_name)
        assert callable(getattr(cli_module, func_name))


def test_console_script_targets_are_explicit_functions():
    """Console scripts should point to concrete callables, not package attrs."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    scripts = project["scripts"]

    assert scripts["medfusion"] == "med_core.cli:main"
    assert scripts["med-train"] == "med_core.cli.train:train"
    assert scripts["med-evaluate"] == "med_core.cli.evaluate:evaluate"
    assert scripts["med-preprocess"] == "med_core.cli.preprocess:preprocess"
