"""
Tests for CLI module structure and imports.
"""


def test_cli_imports():
    """Test that CLI functions can be imported."""
    from med_core.cli import evaluate, preprocess, train

    assert callable(train)
    assert callable(evaluate)
    assert callable(preprocess)


def test_cli_submodule_imports():
    """Test that CLI submodules can be imported directly."""
    from med_core.cli.evaluate import evaluate
    from med_core.cli.preprocess import preprocess
    from med_core.cli.train import train

    assert callable(train)
    assert callable(evaluate)
    assert callable(preprocess)


def test_cli_backward_compatibility():
    """Test that old import path still works."""
    from med_core.cli import evaluate, preprocess, train

    # Verify functions are accessible
    assert train.__module__ == "med_core.cli.train"
    assert evaluate.__module__ == "med_core.cli.evaluate"
    assert preprocess.__module__ == "med_core.cli.preprocess"


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
