from pathlib import Path


OSS_ROOT = Path(__file__).resolve().parents[1]


def _read_script(relative_path: str) -> str:
    return (OSS_ROOT / relative_path).read_text(encoding="utf-8")


def test_non_run_scripts_write_to_artifacts_dev_paths() -> None:
    enhanced_report = _read_script("scripts/test_enhanced_report.py")
    full_workflow = _read_script("scripts/full_workflow_test.py")
    quick_simulation = _read_script("scripts/quick_simulation_test.py")

    assert 'Path("artifacts/dev/report-tests/enhanced")' in enhanced_report
    assert 'Path("artifacts/dev/report-tests/statistical")' in enhanced_report
    assert 'Path("artifacts/dev/report-tests/latex")' in enhanced_report
    assert 'FULL_WORKFLOW_OUTPUT_DIR = "artifacts/dev/workflow-tests/full_workflow_test"' in full_workflow
    assert "config.logging.output_dir = FULL_WORKFLOW_OUTPUT_DIR" in full_workflow
    assert 'output_dir: "artifacts/dev/simulation_test/run"' in quick_simulation
    assert 'Path("artifacts/dev/simulation_test/simulation_test_results.json")' in quick_simulation


def test_non_run_scripts_no_longer_use_outputs_root() -> None:
    enhanced_report = _read_script("scripts/test_enhanced_report.py")
    full_workflow = _read_script("scripts/full_workflow_test.py")
    quick_simulation = _read_script("scripts/quick_simulation_test.py")

    assert "outputs/enhanced_report_test" not in enhanced_report
    assert "outputs/statistical_test" not in enhanced_report
    assert "outputs/latex_test" not in enhanced_report
    assert "outputs/full_workflow_test" not in full_workflow
    assert "outputs/simulation_test" not in quick_simulation
    assert "outputs/simulation_test_results.json" not in quick_simulation
