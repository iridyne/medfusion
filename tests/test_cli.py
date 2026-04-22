"""
Tests for CLI module structure and imports.
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path

import pytest

os.environ.setdefault("MEDFUSION_DATA_DIR", tempfile.mkdtemp(prefix="medfusion-cli-test-"))
os.environ.setdefault("MPLBACKEND", "Agg")

from test_build_results import _create_checkpoint_and_logs


def _assert_path_equal(actual: str, expected: Path) -> None:
    assert Path(actual) == expected


def test_cli_imports():
    """Test that CLI functions can be imported."""
    from med_core.cli import (
        evaluate,
        import_run,
        preprocess,
        public_datasets,
        run,
        train,
    )

    assert callable(train)
    assert callable(run)
    assert callable(evaluate)
    assert callable(preprocess)
    assert callable(import_run)
    assert callable(public_datasets)


def test_cli_submodule_imports():
    """Test that CLI submodules can be imported directly."""
    from med_core.cli.evaluate import evaluate
    from med_core.cli.import_run import import_run
    from med_core.cli.preprocess import preprocess
    from med_core.cli.public_datasets import public_datasets
    from med_core.cli.run import run
    from med_core.cli.train import train

    assert callable(train)
    assert callable(run)
    assert callable(evaluate)
    assert callable(preprocess)
    assert callable(import_run)
    assert callable(public_datasets)


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
    assert "run" in cli_module.__all__
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


def test_start_help_matches_mvp_contract(capsys):
    from med_core.cli import main

    original_argv = sys.argv[:]
    try:
        sys.argv = ["medfusion", "--help"]
        main()
    finally:
        sys.argv = original_argv

    output = capsys.readouterr().out

    assert "medfusion start" in output
    assert "run" in output
    assert "validate-config" in output
    assert "build-results" in output
    assert "uninstall" in output
    assert "version-check" in output
    assert "YAML" in output


def test_run_help_matches_mvp_contract(capsys):
    from med_core.cli.run import run

    with pytest.raises(SystemExit) as exc_info:
        run(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out

    assert "--config" in output
    assert "--output-dir" in output
    assert "--skip-build-results" in output
    assert "--json" in output
    assert "validate-config" in output
    assert "build-results" in output


def test_main_dispatches_run_command(monkeypatch):
    import med_core.cli as cli_module

    captured = {}

    def _fake_run(argv=None, prog="medfusion run"):
        captured["argv"] = list(argv or [])
        captured["prog"] = prog

    monkeypatch.setattr(cli_module, "run", _fake_run)

    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "medfusion",
            "run",
            "--config",
            "configs/starter/quickstart.yaml",
            "--skip-build-results",
        ]
        cli_module.main()
    finally:
        sys.argv = original_argv

    assert captured == {
        "argv": ["--config", "configs/starter/quickstart.yaml", "--skip-build-results"],
        "prog": "medfusion run",
    }


def test_main_dispatches_uninstall_command(monkeypatch):
    import med_core.cli as cli_module

    captured = {}

    def _fake_uninstall(argv=None, prog="medfusion uninstall"):
        captured["argv"] = list(argv or [])
        captured["prog"] = prog

    monkeypatch.setattr(cli_module, "uninstall", _fake_uninstall)

    original_argv = sys.argv[:]
    try:
        sys.argv = ["medfusion", "uninstall", "--purge-data", "--yes"]
        cli_module.main()
    finally:
        sys.argv = original_argv

    assert captured == {
        "argv": ["--purge-data", "--yes"],
        "prog": "medfusion uninstall",
    }


def test_main_dispatches_version_check_command(monkeypatch):
    import med_core.cli as cli_module

    captured = {}

    def _fake_version_check(argv=None, prog="medfusion version-check"):
        captured["argv"] = list(argv or [])
        captured["prog"] = prog

    monkeypatch.setattr(cli_module, "version_check", _fake_version_check)

    original_argv = sys.argv[:]
    try:
        sys.argv = ["medfusion", "version-check", "--skip-server", "--json"]
        cli_module.main()
    finally:
        sys.argv = original_argv

    assert captured == {
        "argv": ["--skip-server", "--json"],
        "prog": "medfusion version-check",
    }


def test_version_check_cli_supports_local_only_json_output(capsys):
    from med_core.cli.version_check import version_check

    version_check(["--skip-server", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["local"]["cli_version"] == payload["local"]["web_settings_version"]
    assert payload["server"]["checked"] is False
    assert payload["server"]["reason"] == "skip-server"


def test_web_data_commands_include_restore() -> None:
    from med_core.web.cli import data

    assert "restore" in data.commands


def test_start_entrypoint_hint_detects_legacy_web_start() -> None:
    from med_core.web.cli import _start_entrypoint_hint

    assert _start_entrypoint_hint("medfusion start") is None
    hint = _start_entrypoint_hint("medfusion web start")
    assert hint is not None
    assert "medfusion start" in hint


def test_uninstall_cli_supports_keep_and_purge_modes(tmp_path, monkeypatch, capsys):
    from med_core.cli.uninstall import uninstall

    monkeypatch.chdir(tmp_path)

    # Keep-data mode should only remove .venv by default.
    (tmp_path / ".venv" / "Scripts").mkdir(parents=True)
    (tmp_path / "outputs").mkdir(parents=True)
    user_data_dir = tmp_path / "user-data"
    user_data_dir.mkdir(parents=True)

    uninstall(
        [
            "--yes",
            "--json",
            "--venv-path",
            ".venv",
            "--user-data-dir",
            str(user_data_dir),
        ]
    )
    keep_payload = json.loads(capsys.readouterr().out)
    assert keep_payload["ok"] is True
    assert keep_payload["mode"] == "keep-data"
    assert not (tmp_path / ".venv").exists()
    assert (tmp_path / "outputs").exists()
    assert user_data_dir.exists()

    # Purge-data mode should remove project artifacts and user data directory.
    (tmp_path / ".venv").mkdir(parents=True)
    (tmp_path / "logs").mkdir(parents=True)
    (tmp_path / "checkpoints").mkdir(parents=True)

    uninstall(
        [
            "--yes",
            "--json",
            "--purge-data",
            "--venv-path",
            ".venv",
            "--user-data-dir",
            str(user_data_dir),
        ]
    )
    purge_payload = json.loads(capsys.readouterr().out)
    assert purge_payload["ok"] is True
    assert purge_payload["mode"] == "purge-data"
    assert not (tmp_path / ".venv").exists()
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "logs").exists()
    assert not (tmp_path / "checkpoints").exists()
    assert not user_data_dir.exists()


def test_validate_config_cli_surfaces_yaml_mainline_contract(capsys):
    from med_core.cli.doctor import validate_config

    validate_config(["--config", "configs/starter/quickstart.yaml"])
    output = capsys.readouterr().out

    assert "Mainline contract" in output
    assert "multimodal_fusion" in output
    assert "resnet18" in output
    assert "concatenate" in output
    assert "outputs/quickstart" in output
    assert "medfusion train --config configs/starter/quickstart.yaml" in output
    assert "medfusion build-results --config configs/starter/quickstart.yaml" in output


def test_validate_config_json_includes_model_and_artifact_contract(capsys):
    from med_core.cli.doctor import validate_config

    validate_config(["--config", "configs/starter/quickstart.yaml", "--json"])
    payload = json.loads(capsys.readouterr().out)

    contract = payload["summary"]["mainline_contract"]
    model = contract["model"]
    artifacts = contract["artifacts"]

    assert contract["output_dir"] == "outputs/quickstart"
    assert model["model_type"] == "multimodal_fusion"
    assert model["vision_backbone"] == "resnet18"
    assert model["fusion_type"] == "concatenate"
    assert artifacts["checkpoint"] == "outputs/quickstart/checkpoints/best.pth"
    assert artifacts["summary"] == "outputs/quickstart/reports/summary.json"
    assert contract["recommended_commands"]["validate"] == (
        "medfusion validate-config --config configs/starter/quickstart.yaml"
    )


def test_validate_config_cli_allows_repo_relative_data_paths_outside_oss(
    tmp_path,
    monkeypatch,
    capsys,
):
    from med_core.cli.doctor import validate_config

    repo_root = Path(__file__).resolve().parents[1]
    outside_cwd = tmp_path / "outside-validate-config"
    outside_cwd.mkdir()
    monkeypatch.chdir(outside_cwd)

    validate_config(
        ["--config", str(repo_root / "configs/starter/quickstart.yaml"), "--json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["summary"]["dataset_rows"] > 0


def test_validate_config_cli_resolves_repo_relative_config_outside_oss(
    tmp_path,
    monkeypatch,
    capsys,
):
    from med_core.cli.doctor import validate_config

    outside_cwd = tmp_path / "outside-relative-config"
    outside_cwd.mkdir()
    monkeypatch.chdir(outside_cwd)

    validate_config(["--config", "configs/starter/quickstart.yaml", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["summary"]["mainline_contract"]["output_dir"] == "outputs/quickstart"


def test_run_cli_executes_validate_train_and_build_results(
    tmp_path,
    monkeypatch,
    capsys,
):
    import importlib

    from med_core.cli.run import run
    from med_core.output_layout import RunOutputLayout

    run_module = importlib.import_module("med_core.cli.run")
    output_dir = tmp_path / "run-default-output"
    calls: list[str] = []
    layout = RunOutputLayout(output_dir).ensure_exists()
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    root_handler = logging.StreamHandler(sys.stdout)
    root_logger.handlers = [root_handler]
    root_logger.setLevel(logging.INFO)

    def _fake_validate_config(argv=None, prog="medfusion validate-config"):
        calls.append("validate-config")
        assert argv is not None
        assert "--config" in argv
        assert "--json" in argv
        print(json.dumps({"ok": True, "summary": {}}))

    def _fake_train(argv=None, prog="medfusion train"):
        calls.append("train")
        assert argv is not None
        assert "--config" in argv
        assert "--output-dir" in argv
        print("TRAIN NOISE SHOULD NOT LEAK")
        layout.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        layout.logs_dir.mkdir(parents=True, exist_ok=True)
        (layout.checkpoints_dir / "best.pth").write_bytes(b"checkpoint")
        layout.history_path.write_text("{}", encoding="utf-8")

    def _fake_build_results(argv=None, prog="medfusion build-results"):
        calls.append("build-results")
        assert argv is not None
        assert "--checkpoint" in argv
        assert "--json" in argv
        layout.reports_dir.mkdir(parents=True, exist_ok=True)
        layout.metrics_dir.mkdir(parents=True, exist_ok=True)
        layout.summary_path.write_text("{}", encoding="utf-8")
        layout.validation_path.write_text("{}", encoding="utf-8")
        layout.report_path.write_text("# report", encoding="utf-8")
        layout.metrics_path.write_text("{}", encoding="utf-8")
        root_logger.info("BUILD RESULTS NOISE SHOULD NOT LEAK")
        print(
            json.dumps(
                {
                    "output_dir": str(layout.root_dir),
                    "artifact_paths": {
                        "summary_path": str(layout.summary_path),
                        "validation_path": str(layout.validation_path),
                        "report_path": str(layout.report_path),
                        "metrics_path": str(layout.metrics_path),
                    },
                    "metrics": {},
                    "validation": {},
                }
            )
        )

    monkeypatch.setattr(run_module, "validate_config", _fake_validate_config)
    monkeypatch.setattr(run_module, "train", _fake_train)
    monkeypatch.setattr(run_module, "build_results", _fake_build_results)

    try:
        run(
            [
                "--config",
                "configs/starter/quickstart.yaml",
                "--output-dir",
                str(output_dir),
                "--json",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)

    assert calls == ["validate-config", "train", "build-results"]
    _assert_path_equal(payload["output_dir"], layout.root_dir)
    _assert_path_equal(payload["checkpoint"], layout.checkpoints_dir / "best.pth")
    assert payload["build_results_skipped"] is False
    _assert_path_equal(payload["artifact_paths"]["summary"], layout.summary_path)
    _assert_path_equal(payload["artifact_paths"]["validation"], layout.validation_path)
    _assert_path_equal(payload["artifact_paths"]["report"], layout.report_path)


def test_run_cli_can_skip_build_results(tmp_path, monkeypatch, capsys):
    import importlib

    from med_core.cli.run import run
    from med_core.output_layout import RunOutputLayout

    run_module = importlib.import_module("med_core.cli.run")
    output_dir = tmp_path / "run-skip-output"
    calls: list[str] = []
    layout = RunOutputLayout(output_dir).ensure_exists()

    def _fake_validate_config(argv=None, prog="medfusion validate-config"):
        calls.append("validate-config")
        print(json.dumps({"ok": True, "summary": {}}))

    def _fake_train(argv=None, prog="medfusion train"):
        calls.append("train")
        (layout.checkpoints_dir / "best.pth").write_bytes(b"checkpoint")
        layout.history_path.write_text("{}", encoding="utf-8")

    def _fail_build_results(argv=None, prog="medfusion build-results"):
        raise AssertionError("build-results should not be called when skipped")

    monkeypatch.setattr(run_module, "validate_config", _fake_validate_config)
    monkeypatch.setattr(run_module, "train", _fake_train)
    monkeypatch.setattr(run_module, "build_results", _fail_build_results)

    run(
        [
            "--config",
            "configs/starter/quickstart.yaml",
            "--output-dir",
            str(output_dir),
            "--skip-build-results",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert calls == ["validate-config", "train"]
    assert payload["build_results_skipped"] is True
    _assert_path_equal(
        payload["artifact_paths"]["checkpoint"],
        layout.checkpoints_dir / "best.pth",
    )
    _assert_path_equal(payload["artifact_paths"]["history"], layout.history_path)


def test_docs_promote_run_as_recommended_cli_mainline() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    workflow = Path("docs/contents/getting-started/cli-config-workflow.md").read_text(
        encoding="utf-8"
    )
    quickstart = Path("docs/contents/getting-started/quickstart.md").read_text(
        encoding="utf-8"
    )

    assert "uv run medfusion run --config configs/starter/quickstart.yaml" in readme
    assert "uv run medfusion run --config configs/starter/quickstart.yaml" in workflow
    assert "uv run medfusion run --config configs/starter/quickstart.yaml" in quickstart
    assert "validate-config -> train -> build-results" in workflow


def test_evaluate_cli_uses_canonical_result_contract(tmp_path):
    from med_core.cli.evaluate import evaluate

    config_path, checkpoint_path = _create_checkpoint_and_logs(tmp_path)
    output_dir = tmp_path / "evaluate-output"

    evaluate(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "train",
            "--output-dir",
            str(output_dir),
        ]
    )

    metrics_path = output_dir / "metrics" / "metrics.json"
    validation_path = output_dir / "metrics" / "validation.json"
    summary_path = output_dir / "reports" / "summary.json"
    report_path = output_dir / "reports" / "report.md"

    for path in (metrics_path, validation_path, summary_path, report_path):
        assert path.exists(), f"Missing evaluate artifact: {path}"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report_text = report_path.read_text(encoding="utf-8")

    assert metrics["meta"]["generated_by"] == "medfusion.build_results"
    assert validation["meta"] == metrics["meta"]
    assert summary["meta"] == metrics["meta"]
    _assert_path_equal(summary["artifacts"]["metrics_path"], metrics_path)
    _assert_path_equal(summary["artifacts"]["validation_path"], validation_path)
    _assert_path_equal(summary["artifacts"]["report_path"], report_path)
    assert validation["overview"]["split"] == "train"
    assert "## Contract Metadata" in report_text
    assert "## Artifact Index" in report_text


def test_public_datasets_cli_lists_available_quick_validation_sets(capsys):
    from med_core.cli.public_datasets import public_datasets

    public_datasets(["list", "--json"])
    payload = json.loads(capsys.readouterr().out)

    ids = {item["id"] for item in payload}
    assert "medmnist-pathmnist" in ids
    assert "medmnist-breastmnist" in ids
    assert "uci-heart-disease" in ids


def test_public_datasets_cli_prepare_dry_run(capsys):
    from med_core.cli.public_datasets import public_datasets

    public_datasets([
        "prepare",
        "medmnist-breastmnist",
        "--dry-run",
        "--json",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert payload["dataset"]["id"] == "medmnist-breastmnist"
    assert payload["recommended_commands"]["prepare"].startswith(
        "medfusion public-datasets prepare medmnist-breastmnist"
    )
    assert payload["recommended_commands"]["train"] == (
        "medfusion train --config "
        "configs/public_datasets/breastmnist_quickstart.yaml"
    )
    assert payload["split_limits"] == {"train": 270, "val": 45, "test": 45}


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


def test_train_cli_anchors_relative_output_override_to_repo_root(
    tmp_path,
    monkeypatch,
):
    from med_core.cli.train import MedicalMultimodalDataset, train
    from med_core.configs import load_config, save_config

    class StopAfterOutputDir(RuntimeError):
        pass

    def _stop_after_output_dir(*args, **kwargs):
        raise StopAfterOutputDir

    config = load_config("configs/starter/quickstart.yaml")
    config.logging.output_dir = str(tmp_path / "config-output")
    config_path = tmp_path / "train-anchor.yaml"
    save_config(config, config_path)

    repo_root = Path(__file__).resolve().parents[1]
    relative_output_dir = f"outputs/pytest-train-anchor-{tmp_path.name}"
    target_dir = repo_root / relative_output_dir
    shutil.rmtree(target_dir, ignore_errors=True)

    outside_cwd = tmp_path / "outside-train"
    outside_cwd.mkdir()
    monkeypatch.chdir(outside_cwd)
    monkeypatch.setattr(
        MedicalMultimodalDataset,
        "from_csv",
        staticmethod(_stop_after_output_dir),
    )

    with pytest.raises(StopAfterOutputDir):
        train(["--config", str(config_path), "--output-dir", relative_output_dir])

    assert target_dir.exists()


def test_train_cli_anchors_relative_data_paths_to_repo_root(
    tmp_path,
    monkeypatch,
):
    from med_core.cli.train import MedicalMultimodalDataset, train
    from med_core.configs import load_config, save_config

    class StopAfterDataPathCheck(RuntimeError):
        pass

    repo_root = Path(__file__).resolve().parents[1]

    def _assert_data_paths(*args, **kwargs):
        assert Path(kwargs["csv_path"]) == repo_root / "data/mock/metadata.csv"
        assert Path(kwargs["image_dir"]) == repo_root / "data/mock"
        raise StopAfterDataPathCheck

    config = load_config("configs/starter/quickstart.yaml")
    config.logging.output_dir = str(tmp_path / "train-data-paths-output")
    config_path = tmp_path / "train-data-paths.yaml"
    save_config(config, config_path)

    outside_cwd = tmp_path / "outside-train-data-paths"
    outside_cwd.mkdir()
    monkeypatch.chdir(outside_cwd)
    monkeypatch.setattr(
        MedicalMultimodalDataset,
        "from_csv",
        staticmethod(_assert_data_paths),
    )

    with pytest.raises(StopAfterDataPathCheck):
        train(["--config", str(config_path)])


def test_build_results_cli_anchors_relative_output_override_to_repo_root(
    tmp_path,
    monkeypatch,
    capsys,
):
    from med_core.cli.build_results import build_results

    config_path, checkpoint_path = _create_checkpoint_and_logs(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    relative_output_dir = f"outputs/pytest-build-results-anchor-{tmp_path.name}"
    target_dir = repo_root / relative_output_dir
    shutil.rmtree(target_dir, ignore_errors=True)

    outside_cwd = tmp_path / "outside-build-results"
    outside_cwd.mkdir()
    monkeypatch.chdir(outside_cwd)

    try:
        build_results(
            [
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--output-dir",
                relative_output_dir,
                "--attention-samples",
                "0",
                "--json",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert Path(payload["output_dir"]) == target_dir
        assert (target_dir / "reports" / "summary.json").exists()
    finally:
        shutil.rmtree(target_dir, ignore_errors=True)


def test_import_run_cli_preserves_result_contract_and_is_idempotent(tmp_path, capsys):
    from med_core.cli.import_run import import_run

    config_path, checkpoint_path = _create_checkpoint_and_logs(
        tmp_path,
        include_survival=True,
    )

    base_args = [
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--split",
        "train",
        "--attention-samples",
        "2",
        "--survival-time-column",
        "survival_time",
        "--survival-event-column",
        "event",
        "--importance-sample-limit",
        "8",
        "--name",
        "ci-import-run-model",
        "--tag",
        "ci-import-run",
        "--json",
    ]

    import_run(base_args)
    first_payload = json.loads(capsys.readouterr().out)
    artifact_paths = first_payload["artifact_paths"]

    for key in (
        "metrics_path",
        "validation_path",
        "summary_path",
        "report_path",
        "config_path",
        "prediction_path",
    ):
        assert key in artifact_paths
        assert Path(artifact_paths[key]).exists(), f"Missing import artifact: {key}"

    summary = json.loads(Path(artifact_paths["summary_path"]).read_text(encoding="utf-8"))
    validation = json.loads(
        Path(artifact_paths["validation_path"]).read_text(encoding="utf-8")
    )

    assert summary["meta"]["generated_by"] == "medfusion.build_results"
    assert summary["meta"]["split"] == "train"
    assert summary["meta"] == validation["meta"]
    assert validation["survival"]["c_index"] is not None
    assert validation["global_feature_importance"]["top_features"]
    assert set(first_payload["tags"]) >= {"imported", "split:train", "ci-import-run"}

    import_run(base_args)
    second_payload = json.loads(capsys.readouterr().out)
    assert second_payload["model_id"] == first_payload["model_id"]


def test_import_run_cli_anchors_relative_output_override_to_repo_root(
    tmp_path,
    monkeypatch,
    capsys,
):
    from med_core.cli.import_run import import_run

    config_path, checkpoint_path = _create_checkpoint_and_logs(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    relative_output_dir = f"outputs/pytest-import-run-anchor-{tmp_path.name}"
    target_dir = repo_root / relative_output_dir
    shutil.rmtree(target_dir, ignore_errors=True)

    outside_cwd = tmp_path / "outside-import-run"
    outside_cwd.mkdir()
    monkeypatch.chdir(outside_cwd)

    try:
        import_run(
            [
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--output-dir",
                relative_output_dir,
                "--attention-samples",
                "0",
                "--json",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        artifact_paths = payload["artifact_paths"]
        assert Path(artifact_paths["summary_path"]).parent.parent == target_dir
        assert (target_dir / "reports" / "summary.json").exists()
    finally:
        shutil.rmtree(target_dir, ignore_errors=True)
