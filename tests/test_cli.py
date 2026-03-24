"""
Tests for CLI module structure and imports.
"""

import json
from pathlib import Path
import tomllib

from test_build_results import _create_checkpoint_and_logs


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


def test_build_results_cli_supports_survival_and_importance_options(
    tmp_path,
    capsys,
):
    from med_core.cli.build_results import build_results

    config_path, checkpoint_path = _create_checkpoint_and_logs(
        tmp_path,
        include_survival=True,
    )

    build_results(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "train",
            "--survival-time-column",
            "survival_time",
            "--survival-event-column",
            "event",
            "--importance-sample-limit",
            "8",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["validation"]["survival"]["c_index"] is not None
    assert payload["validation"]["global_feature_importance"]["top_features"]
