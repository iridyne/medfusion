"""项目工作区 API。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import asc, desc
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db_session
from ..models import DatasetInfo, ModelInfo, ProjectInfo, TrainingJob
from ..time_utils import utcnow

router = APIRouter()

TaskType = Literal["binary_classification", "cox_survival", "multimodal_research"]

PROJECT_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "binary_basic",
        "name": "二分类预测",
        "task_type": "binary_classification",
        "description": "面向基础二分类科研任务的默认模板，适合快速验证图像 + 表格主链。",
        "required_fields": ["target_column"],
        "recommended_backbone": "resnet18",
        "recommended_fusion": "concatenate",
        "expected_outputs": ["训练曲线", "ROC 曲线", "混淆矩阵", "Word/PDF 报告"],
        "warnings": ["默认按二分类处理，请确认目标列只有两个有效类别。"],
    },
    {
        "id": "cox_survival",
        "name": "Cox 生存分析",
        "task_type": "cox_survival",
        "description": "面向预后建模和生存分析的模板，强调 survival_time / event 字段映射。",
        "required_fields": ["survival_time", "event"],
        "recommended_backbone": "resnet18",
        "recommended_fusion": "attention",
        "expected_outputs": ["风险分层结果", "训练曲线", "Word/PDF 报告"],
        "warnings": ["当前 v1 不承诺自动生成列线图 / nomogram。"],
    },
    {
        "id": "multimodal_research",
        "name": "多模态研究工作流",
        "task_type": "multimodal_research",
        "description": "面向图像 + 表格的研究工作流模板，适合展示 attention 和多模态结果。",
        "required_fields": ["image_path_column", "target_column"],
        "recommended_backbone": "resnet50",
        "recommended_fusion": "attention",
        "expected_outputs": ["训练曲线", "ROC 曲线", "attention 可视化", "项目导出包"],
        "warnings": ["v1 以图像 + tabular 主链为主，不覆盖任意模态组合。"],
    },
]


class ProjectCreate(BaseModel):
    name: str
    description: str | None = None
    task_type: TaskType
    template_id: str
    dataset_id: int | None = None
    config_path: str | None = None
    output_dir: str | None = None
    tags: list[str] | None = None
    project_meta: dict[str, Any] | None = None
    status: str = "draft"


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    task_type: TaskType | None = None
    template_id: str | None = None
    dataset_id: int | None = None
    config_path: str | None = None
    output_dir: str | None = None
    latest_job_id: str | None = None
    latest_model_id: int | None = None
    tags: list[str] | None = None
    project_meta: dict[str, Any] | None = None
    status: str | None = None


def _get_project_or_404(db: Session, project_id: int) -> ProjectInfo:
    project = db.query(ProjectInfo).filter(ProjectInfo.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project


def _extract_job_project_id(job: TrainingJob) -> int | None:
    config = job.config or {}
    value = config.get("project_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _extract_model_project_id(model: ModelInfo) -> int | None:
    config = model.config or {}
    value = config.get("project_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _project_dataset_name(db: Session, project: ProjectInfo) -> str | None:
    if project.dataset_id is None:
        return None
    dataset = db.query(DatasetInfo).filter(DatasetInfo.id == project.dataset_id).first()
    return dataset.name if dataset else None


def _job_summary(job: TrainingJob) -> dict[str, Any]:
    config = job.config or {}
    training_model_config = config.get("training_model_config", {})
    return {
        "id": job.id,
        "job_id": job.job_id,
        "experiment_name": config.get("experiment_name"),
        "backbone": training_model_config.get("backbone"),
        "status": job.status,
        "progress": job.progress,
        "current_epoch": job.current_epoch,
        "total_epochs": job.total_epochs,
        "current_loss": job.current_loss,
        "current_accuracy": job.current_accuracy,
        "created_at": job.created_at.isoformat(),
    }


def _model_summary(model: ModelInfo) -> dict[str, Any]:
    return {
        "id": model.id,
        "name": model.name,
        "architecture": model.architecture,
        "accuracy": model.accuracy,
        "loss": model.loss,
        "checkpoint_path": model.checkpoint_path,
        "created_at": model.created_at.isoformat(),
    }


def _project_jobs(db: Session, project_id: int) -> list[TrainingJob]:
    jobs = db.query(TrainingJob).order_by(TrainingJob.created_at.desc()).all()
    return [job for job in jobs if _extract_job_project_id(job) == project_id]


def _project_models(db: Session, project_id: int) -> list[ModelInfo]:
    models = db.query(ModelInfo).order_by(ModelInfo.created_at.desc()).all()
    return [model for model in models if _extract_model_project_id(model) == project_id]


def _to_payload(
    db: Session,
    project: ProjectInfo,
    *,
    include_related: bool = False,
) -> dict[str, Any]:
    jobs = _project_jobs(db, project.id)
    models = _project_models(db, project.id)
    latest_job = jobs[0] if jobs else None
    latest_model = models[0] if models else None

    payload = {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "task_type": project.task_type,
        "template_id": project.template_id,
        "status": project.status,
        "dataset_id": project.dataset_id,
        "dataset_name": _project_dataset_name(db, project),
        "config_path": project.config_path,
        "output_dir": project.output_dir,
        "latest_job_id": project.latest_job_id,
        "latest_model_id": project.latest_model_id,
        "tags": project.tags or [],
        "project_meta": project.project_meta or {},
        "job_count": len(jobs),
        "model_count": len(models),
        "latest_job": _job_summary(latest_job) if latest_job else None,
        "latest_model": _model_summary(latest_model) if latest_model else None,
        "created_at": project.created_at.isoformat(),
        "updated_at": project.updated_at.isoformat() if project.updated_at else None,
    }

    if include_related:
        payload["jobs"] = [_job_summary(job) for job in jobs[:10]]
        payload["models"] = [_model_summary(model) for model in models[:10]]

    return payload


def _build_export_bundle(project: ProjectInfo) -> Path:
    export_root = settings.data_dir / "exports" / "projects"
    export_root.mkdir(parents=True, exist_ok=True)

    staging_dir = export_root / f"project-{project.id}"
    archive_base = export_root / f"project-{project.id}-{int(utcnow().timestamp())}"

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    if project.output_dir:
        output_dir = Path(project.output_dir)
        if output_dir.exists():
            shutil.copytree(output_dir, staging_dir / "output", dirs_exist_ok=True)

    if project.config_path:
        config_path = Path(project.config_path)
        if config_path.exists():
            config_target = staging_dir / "config"
            config_target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_path, config_target / config_path.name)

    manifest = {
        "project_id": project.id,
        "name": project.name,
        "task_type": project.task_type,
        "template_id": project.template_id,
        "status": project.status,
        "generated_at": utcnow().isoformat(),
    }
    (staging_dir / "project.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=staging_dir)
    return Path(archive_path)


@router.get("/templates")
async def list_project_templates() -> dict[str, list[dict[str, Any]]]:
    """列出本地专业版模板。"""
    return {"templates": PROJECT_TEMPLATES}


@router.get("/templates/{template_id}")
async def get_project_template(template_id: str) -> dict[str, Any]:
    """获取单个模板说明。"""
    for template in PROJECT_TEMPLATES:
        if template["id"] == template_id:
            return template
    raise HTTPException(status_code=404, detail="模板不存在")


@router.get("/")
async def list_projects(
    skip: int = 0,
    limit: int = 50,
    task_type: str | None = None,
    status: str | None = None,
    sort_by: str = "updated_at",
    order: str = "desc",
    db: Session = Depends(get_db_session),
) -> list[dict[str, Any]]:
    """列出项目。"""
    query = db.query(ProjectInfo)
    if task_type:
        query = query.filter(ProjectInfo.task_type == task_type)
    if status:
        query = query.filter(ProjectInfo.status == status)

    sort_column = getattr(ProjectInfo, sort_by, ProjectInfo.updated_at)
    sort_expr = desc(sort_column) if order.lower() == "desc" else asc(sort_column)
    projects = query.order_by(sort_expr).offset(skip).limit(limit).all()
    return [_to_payload(db, project) for project in projects]


@router.post("/")
async def create_project(
    payload: ProjectCreate,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """创建项目。"""
    project = ProjectInfo(
        name=payload.name,
        description=payload.description,
        task_type=payload.task_type,
        template_id=payload.template_id,
        status=payload.status,
        dataset_id=payload.dataset_id,
        config_path=payload.config_path,
        output_dir=payload.output_dir,
        tags=payload.tags,
        project_meta=payload.project_meta,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return _to_payload(db, project)


@router.get("/{project_id}")
async def get_project(
    project_id: int,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """获取项目详情。"""
    project = _get_project_or_404(db, project_id)
    return _to_payload(db, project, include_related=True)


@router.patch("/{project_id}")
async def update_project(
    project_id: int,
    payload: ProjectUpdate,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """更新项目。"""
    project = _get_project_or_404(db, project_id)
    updates = payload.model_dump(exclude_unset=True)
    for key, value in updates.items():
        setattr(project, key, value)
    project.updated_at = utcnow()
    db.commit()
    db.refresh(project)
    return _to_payload(db, project)


@router.delete("/{project_id}")
async def delete_project(
    project_id: int,
    db: Session = Depends(get_db_session),
) -> dict[str, str]:
    """删除项目。"""
    project = _get_project_or_404(db, project_id)
    db.delete(project)
    db.commit()
    return {"message": "项目已删除"}


@router.get("/{project_id}/runs")
async def get_project_runs(
    project_id: int,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """获取项目相关训练和模型记录。"""
    project = _get_project_or_404(db, project_id)
    jobs = _project_jobs(db, project.id)
    models = _project_models(db, project.id)
    return {
        "project_id": project.id,
        "jobs": [_job_summary(job) for job in jobs],
        "models": [_model_summary(model) for model in models],
    }


@router.post("/{project_id}/export")
async def export_project_bundle(
    project_id: int,
    db: Session = Depends(get_db_session),
) -> dict[str, Any]:
    """导出项目交付包。"""
    project = _get_project_or_404(db, project_id)

    if not project.output_dir and not project.config_path:
        raise HTTPException(status_code=400, detail="项目暂无可导出的结果")

    archive_path = _build_export_bundle(project)
    return {
        "project_id": project.id,
        "archive_path": str(archive_path),
        "download_url": f"/api/projects/{project.id}/export/download?path={archive_path.name}",
    }


@router.get("/{project_id}/export/download")
async def download_project_bundle(
    project_id: int,
    path: str = Query(...),
    db: Session = Depends(get_db_session),
) -> FileResponse:
    """下载项目交付包。"""
    _get_project_or_404(db, project_id)
    archive_path = settings.data_dir / "exports" / "projects" / path
    if not archive_path.exists():
        raise HTTPException(status_code=404, detail="项目导出包不存在")
    return FileResponse(path=archive_path, filename=archive_path.name)
