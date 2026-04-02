"""
Tests for CLI module structure and imports.
"""

import json
import os
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path

import pytest

from test_build_results import _create_checkpoint_and_logs

os.environ.setdefault("MEDFUSION_DATA_DIR", tempfile.mkdtemp(prefix="medfusion-cli-test-"))


def test_cli_imports():
    """Test that CLI functions can be imported."""
    from med_core.cli import evaluate, import_run, preprocess, public_datasets, train

    assert callable(train)
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
    from med_core.cli.train import train

    assert callable(train)
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
    assert "validate-config" in output
    assert "build-results" in output
    assert "YAML" in output


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
    assert summary["artifacts"]["metrics_path"] == str(metrics_path)
    assert summary["artifacts"]["validation_path"] == str(validation_path)
    assert summary["artifacts"]["report_path"] == str(report_path)
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
