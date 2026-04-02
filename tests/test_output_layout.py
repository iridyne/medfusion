from pathlib import Path

import pytest

from med_core.output_layout import RunOutputLayout, resolve_run_output_dir


def test_run_output_layout_creates_structured_subdirectories(tmp_path: Path) -> None:
    run_dir = tmp_path / "demo-run"

    layout = RunOutputLayout(run_dir)
    layout.ensure_exists()

    assert layout.root_dir == run_dir
    assert layout.checkpoints_dir == run_dir / "checkpoints"
    assert layout.logs_dir == run_dir / "logs"
    assert layout.reports_dir == run_dir / "reports"
    assert layout.metrics_dir == run_dir / "metrics"
    assert layout.artifacts_dir == run_dir / "artifacts"
    assert layout.history_path == run_dir / "logs" / "history.json"
    assert layout.training_log_path == run_dir / "logs" / "training.log"

    for path in (
        layout.root_dir,
        layout.checkpoints_dir,
        layout.logs_dir,
        layout.reports_dir,
        layout.metrics_dir,
        layout.artifacts_dir,
    ):
        assert path.exists()


def test_run_output_layout_exposes_stability_paths(tmp_path: Path) -> None:
    layout = RunOutputLayout(tmp_path / "stability-study")

    assert layout.seed_runs_dir == layout.root_dir / "seeds"
    assert layout.seed_output_dir(7) == layout.seed_runs_dir / "seed-0007"
    assert layout.stability_dir == layout.root_dir / "stability"
    assert layout.stability_summary_json_path == layout.stability_dir / "summary.json"
    assert layout.stability_summary_csv_path == layout.stability_dir / "summary.csv"
    assert layout.stability_summary_md_path == layout.stability_dir / "summary.md"


def test_resolve_run_output_dir_preserves_explicit_override(tmp_path: Path) -> None:
    configured_run_dir = tmp_path / "configured-run"
    checkpoint_path = configured_run_dir / "checkpoints" / "best.pth"
    override_dir = tmp_path / "manual-run"

    resolved = resolve_run_output_dir(
        config_output_dir=configured_run_dir,
        checkpoint_path=checkpoint_path,
        override=override_dir,
    )

    assert resolved == override_dir


def test_resolve_run_output_dir_infers_run_root_from_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "inferred-run"
    checkpoint_path = run_dir / "checkpoints" / "best.pth"

    resolved = resolve_run_output_dir(
        config_output_dir=tmp_path / "unused-config-root",
        checkpoint_path=checkpoint_path,
        override=None,
    )

    assert resolved == run_dir


def test_run_output_layout_anchors_relative_root_to_oss_repo() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    layout = RunOutputLayout("outputs/anchored-layout")

    assert layout.root_dir == repo_root / "outputs" / "anchored-layout"


def test_resolve_run_output_dir_anchors_relative_config_to_oss_repo() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    resolved = resolve_run_output_dir(
        config_output_dir="outputs/anchored-run",
        checkpoint_path=Path("/tmp/external-checkpoint.pth"),
        override=None,
    )

    assert resolved == repo_root / "outputs" / "anchored-run"


def test_web_training_job_output_uses_shared_run_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("fastapi")

    from med_core.web.api import training as training_api

    monkeypatch.setattr(training_api.settings, "data_dir", tmp_path)

    output_dir, log_file = training_api._prepare_job_output("job-123")

    assert output_dir == str(tmp_path / "experiments" / "job-123")
    assert log_file == str(
        tmp_path / "experiments" / "job-123" / "logs" / "training.log"
    )
