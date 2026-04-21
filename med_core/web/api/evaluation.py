"""Independent post-run evaluation API."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from med_core.configs import load_config
from med_core.postprocessing import build_results_artifacts

from ..database import get_db_session
from ..model_registry import register_model_artifacts

router = APIRouter()
logger = logging.getLogger(__name__)


class EvaluationRequest(BaseModel):
    config_path: str
    checkpoint_path: str
    output_dir: str | None = None
    split: Literal["train", "val", "test", "all"] = "test"
    attention_samples: int = 4
    enable_survival: bool = True
    survival_time_column: str | None = None
    survival_event_column: str | None = None
    enable_importance: bool = True
    importance_sample_limit: int = 128
    import_to_model_library: bool = False
    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None


def _read_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@router.post("/run")
async def run_independent_evaluation(
    request: EvaluationRequest,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """Evaluate a real checkpoint without going through a training job."""
    config_path = Path(request.config_path)
    checkpoint_path = Path(request.checkpoint_path)
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint not found: {checkpoint_path}",
        )

    try:
        config = load_config(config_path)
        result = build_results_artifacts(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_dir=request.output_dir,
            split=request.split,
            attention_samples=max(request.attention_samples, 0),
            enable_survival=request.enable_survival,
            survival_time_column=request.survival_time_column,
            survival_event_column=request.survival_event_column,
            enable_importance=request.enable_importance,
            importance_sample_limit=max(request.importance_sample_limit, 0),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced to client
        logger.exception("Independent evaluation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    imported_model = None
    if request.import_to_model_library:
        summary_payload = _read_json(result.artifact_paths.get("summary_path"))
        config_snapshot = _read_json(result.artifact_paths.get("config_path"))
        experiment_name = (
            summary_payload.get("experiment_name")
            or config.experiment_name
            or checkpoint_path.stem
        )
        model_name = request.name or (
            experiment_name
            if experiment_name.endswith("-model")
            else f"{experiment_name}-model"
        )
        imported_model = register_model_artifacts(
            db=db,
            checkpoint_path=checkpoint_path,
            artifact_paths=result.artifact_paths,
            metrics=result.metrics,
            validation=result.validation,
            name=model_name,
            description=request.description
            or "由独立评估入口生成并导入的结果产物",
            architecture=config.model.vision.backbone,
            config_path=result.artifact_paths.get("config_path"),
            tags=["evaluated", f"split:{request.split}", *(request.tags or [])],
            extra_config={
                "import_source": "evaluation_api",
                "source_config_path": str(config_path),
                "config_snapshot": config_snapshot,
                "source_context": {
                    "source_type": "evaluation",
                    "entrypoint": "evaluation-center",
                    "split": request.split,
                },
            },
        )

    overview = result.validation.get("overview", {})
    return {
        "status": "completed",
        "mode": (
            "evaluate_and_import"
            if request.import_to_model_library
            else "evaluate_only"
        ),
        "output_dir": result.output_dir,
        "artifact_paths": result.artifact_paths,
        "metrics": result.metrics,
        "validation": result.validation,
        "summary": {
            "experiment_name": config.experiment_name,
            "split": request.split,
            "sample_count": overview.get("sample_count"),
            "accuracy": overview.get("accuracy", result.metrics.get("best_accuracy")),
            "auc": overview.get("auc", result.metrics.get("auc")),
            "macro_f1": overview.get("macro_f1", result.metrics.get("macro_f1")),
        },
        "model_library_import": (
            {
                "imported": True,
                "model_id": imported_model.id,
                "model_name": imported_model.name,
            }
            if imported_model is not None
            else {
                "imported": False,
                "model_id": None,
                "model_name": None,
            }
        ),
        "next_step": (
            "结果已进入 Model Library，可直接在结果后台继续复盘。"
            if imported_model is not None
            else "评估已完成；如需进入结果后台，可勾选导入结果后台后重新执行。"
        ),
    }
