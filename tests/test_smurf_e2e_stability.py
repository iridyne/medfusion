import importlib.util
import json
import sys
from pathlib import Path


def _load_smurf_e2e_module():
    module_path = Path(__file__).resolve().parents[1] / "demo" / "smurf_e2e" / "smurf_e2e.py"
    spec = importlib.util.spec_from_file_location("smurf_e2e_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_stability_pipeline_reuses_shared_runner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_smurf_e2e_module()
    study_dir = tmp_path / "smurf-study"
    config = {
        "seed": 42,
        "output_dir": str(study_dir),
        "stability": {"seeds": [3, 5]},
    }
    calls: list[tuple[str, int, Path]] = []

    def fake_train(config_payload, project_root, output_dir=None):
        seed_dir = Path(config_payload["output_dir"])
        calls.append(("train", int(config_payload["seed"]), seed_dir))
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "history.json").write_text(
            json.dumps(
                {
                    "entries": [{"epoch": 1}],
                    "best_val_acc": 0.7 + (0.01 * int(config_payload["seed"])),
                    "best_epoch": 1,
                }
            ),
            encoding="utf-8",
        )

    def fake_evaluate(config_payload, project_root, output_dir=None, checkpoint=None):
        seed_dir = Path(config_payload["output_dir"])
        calls.append(("evaluate", int(config_payload["seed"]), seed_dir))
        (seed_dir / "metrics.json").write_text(
            json.dumps({"accuracy": 0.8 + (0.01 * int(config_payload["seed"]))}),
            encoding="utf-8",
        )

    def fake_report(config_payload, project_root, output_dir=None):
        seed_dir = Path(config_payload["output_dir"])
        calls.append(("report", int(config_payload["seed"]), seed_dir))
        reports_dir = seed_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "smurf_e2e_report.md").write_text("# report", encoding="utf-8")

    monkeypatch.setattr(module, "run_train_pipeline", fake_train)
    monkeypatch.setattr(module, "run_evaluate_pipeline", fake_evaluate)
    monkeypatch.setattr(module, "run_report_pipeline", fake_report)

    result = module.run_stability_pipeline(config, tmp_path)
    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))

    assert summary["seeds"] == [3, 5]
    assert calls == [
        ("train", 3, study_dir / "seeds" / "seed-0003"),
        ("evaluate", 3, study_dir / "seeds" / "seed-0003"),
        ("report", 3, study_dir / "seeds" / "seed-0003"),
        ("train", 5, study_dir / "seeds" / "seed-0005"),
        ("evaluate", 5, study_dir / "seeds" / "seed-0005"),
        ("report", 5, study_dir / "seeds" / "seed-0005"),
    ]
