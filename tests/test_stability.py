import json
from pathlib import Path

import numpy as np
import pytest

from med_core.stability import (
    SeedRunArtifacts,
    default_seed_output_dir,
    parse_seeds,
    run_stability_study,
)


def test_parse_seeds_accepts_csv_and_iterables() -> None:
    assert parse_seeds("7, 11, 7, 3") == [7, 11, 3]
    assert parse_seeds([5, 2, 5]) == [5, 2]

    with pytest.raises(ValueError):
        parse_seeds("")


def test_default_seed_output_dir_uses_stable_layout(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"

    assert default_seed_output_dir(study_dir, 42) == study_dir / "seeds" / "seed-0042"


def test_run_stability_study_writes_summary_artifacts(tmp_path: Path) -> None:
    study_dir = tmp_path / "study"
    metric_payloads = {
        11: {"accuracy": 0.71, "loss": 0.91, "auc": 0.82},
        17: {"accuracy": 0.79, "loss": 0.74, "auc": 0.88},
    }
    history_payloads = {
        11: {"entries": [{"epoch": 1}, {"epoch": 2}], "best_val_acc": 0.73, "best_epoch": 2},
        17: {"entries": [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}], "best_val_acc": 0.81, "best_epoch": 3},
    }

    def fake_run(seed: int, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(metric_payloads[seed], indent=2),
            encoding="utf-8",
        )
        (output_dir / "history.json").write_text(
            json.dumps(history_payloads[seed], indent=2),
            encoding="utf-8",
        )

    result = run_stability_study(
        seeds=[11, 17],
        study_dir=study_dir,
        run_seed=fake_run,
        resolve_seed_artifacts=lambda seed, output_dir: SeedRunArtifacts(
            seed=seed,
            output_dir=output_dir,
            metrics_path=output_dir / "metrics.json",
            history_path=output_dir / "history.json",
        ),
        study_name="SMuRF Single-CT",
    )

    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))

    assert result.summary_csv_path.exists()
    assert result.summary_md_path.exists()
    assert result.seed_dirs == {
        11: study_dir / "seeds" / "seed-0011",
        17: study_dir / "seeds" / "seed-0017",
    }
    assert summary["study_name"] == "SMuRF Single-CT"
    assert [row["seed"] for row in summary["per_seed"]] == [11, 17]

    accuracy = summary["aggregates"]["accuracy"]
    assert accuracy["goal"] == "maximize"
    assert accuracy["mean"] == pytest.approx(0.75)
    assert accuracy["std"] == pytest.approx(float(np.std([0.71, 0.79], ddof=0)))
    assert accuracy["best_seed"] == 17
    assert accuracy["worst_seed"] == 11

    loss = summary["aggregates"]["loss"]
    assert loss["goal"] == "minimize"
    assert loss["best_seed"] == 17
    assert loss["worst_seed"] == 11

    history_best = summary["aggregates"]["history.best_val_acc"]
    assert history_best["mean"] == pytest.approx(0.77)
    assert history_best["best_seed"] == 17

    epochs_completed = summary["aggregates"]["history.epochs_completed"]
    assert epochs_completed["max_seed"] == 17
    assert epochs_completed["min_seed"] == 11

    csv_text = result.summary_csv_path.read_text(encoding="utf-8")
    assert "aggregate,,accuracy" in csv_text
    assert "per_seed,11,accuracy,0.71" in csv_text

    md_text = result.summary_md_path.read_text(encoding="utf-8")
    assert "# SMuRF Single-CT Stability Summary" in md_text
    assert "| 11 |" in md_text
    assert "| accuracy |" in md_text
