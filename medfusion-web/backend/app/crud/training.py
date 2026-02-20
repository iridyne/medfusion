"""训练任务 CRUD 操作"""
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.models.database import TrainingCheckpoint, TrainingJob


class TrainingJobCRUD:
    """训练任务 CRUD 操作"""

    @staticmethod
    def create(
        db: Session,
        job_id: str,
        model_config: dict[str, Any],
        data_config: dict[str, Any],
        training_config: dict[str, Any],
        name: str | None = None,
        description: str | None = None,
    ) -> TrainingJob:
        """创建训练任务"""
        job = TrainingJob(
            job_id=job_id,
            name=name,
            description=description,
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            status="pending",
            total_epochs=training_config.get("epochs", 10),
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def get(db: Session, job_id: str) -> TrainingJob | None:
        """获取训练任务"""
        return db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()

    @staticmethod
    def get_by_id(db: Session, id: int) -> TrainingJob | None:
        """根据 ID 获取训练任务"""
        return db.query(TrainingJob).filter(TrainingJob.id == id).first()

    @staticmethod
    def list(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        status: str | None = None,
    ) -> list[TrainingJob]:
        """列出训练任务"""
        query = db.query(TrainingJob)

        if status:
            query = query.filter(TrainingJob.status == status)

        return query.order_by(TrainingJob.created_at.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def update_status(
        db: Session,
        job_id: str,
        status: str,
        error: str | None = None,
    ) -> TrainingJob | None:
        """更新任务状态"""
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()

        if not job:
            return None

        job.status = status

        if error is not None:
            job.error = error

        if status == "running" and not job.started_at:
            job.started_at = datetime.utcnow()

        if status in ["completed", "failed", "stopped"]:
            job.completed_at = datetime.utcnow()
            if job.started_at:
                job.duration = (job.completed_at - job.started_at).total_seconds()

        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def update_progress(
        db: Session,
        job_id: str,
        progress: float,
        current_epoch: int,
        current_metrics: dict[str, Any] | None = None,
    ) -> TrainingJob | None:
        """更新训练进度"""
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()

        if not job:
            return None

        job.progress = progress
        job.current_epoch = current_epoch

        if current_metrics is not None:
            job.current_metrics = current_metrics

        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def update_history(
        db: Session,
        job_id: str,
        history: dict[str, Any],
    ) -> TrainingJob | None:
        """更新训练历史"""
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()

        if not job:
            return None

        job.history = history

        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def update_model_path(
        db: Session,
        job_id: str,
        model_path: str,
        checkpoint_path: str | None = None,
    ) -> TrainingJob | None:
        """更新模型路径"""
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()

        if not job:
            return None

        job.model_path = model_path

        if checkpoint_path is not None:
            job.checkpoint_path = checkpoint_path

        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def delete(db: Session, job_id: str) -> bool:
        """删除训练任务"""
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()

        if not job:
            return False

        db.delete(job)
        db.commit()
        return True


class TrainingCheckpointCRUD:
    """训练检查点 CRUD 操作"""

    @staticmethod
    def create(
        db: Session,
        job_id: int,
        epoch: int,
        checkpoint_path: str,
        metrics: dict[str, Any] | None = None,
        step: int | None = None,
        file_size: int | None = None,
        is_best: bool = False,
    ) -> TrainingCheckpoint:
        """创建检查点"""
        checkpoint = TrainingCheckpoint(
            job_id=job_id,
            epoch=epoch,
            step=step,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            file_size=file_size,
            is_best=is_best,
        )
        db.add(checkpoint)
        db.commit()
        db.refresh(checkpoint)
        return checkpoint

    @staticmethod
    def get(db: Session, checkpoint_id: int) -> TrainingCheckpoint | None:
        """获取检查点"""
        return db.query(TrainingCheckpoint).filter(TrainingCheckpoint.id == checkpoint_id).first()

    @staticmethod
    def list_by_job(
        db: Session,
        job_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> list[TrainingCheckpoint]:
        """列出任务的检查点"""
        return (
            db.query(TrainingCheckpoint)
            .filter(TrainingCheckpoint.job_id == job_id)
            .order_by(TrainingCheckpoint.epoch.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_best(db: Session, job_id: int) -> TrainingCheckpoint | None:
        """获取最佳检查点"""
        return (
            db.query(TrainingCheckpoint)
            .filter(TrainingCheckpoint.job_id == job_id, TrainingCheckpoint.is_best)
            .first()
        )

    @staticmethod
    def mark_as_best(db: Session, checkpoint_id: int) -> TrainingCheckpoint | None:
        """标记为最佳检查点"""
        checkpoint = db.query(TrainingCheckpoint).filter(TrainingCheckpoint.id == checkpoint_id).first()

        if not checkpoint:
            return None

        # 取消同一任务的其他最佳标记
        db.query(TrainingCheckpoint).filter(
            TrainingCheckpoint.job_id == checkpoint.job_id,
            TrainingCheckpoint.is_best
        ).update({"is_best": False})

        checkpoint.is_best = True

        db.commit()
        db.refresh(checkpoint)
        return checkpoint

    @staticmethod
    def delete(db: Session, checkpoint_id: int) -> bool:
        """删除检查点"""
        checkpoint = db.query(TrainingCheckpoint).filter(TrainingCheckpoint.id == checkpoint_id).first()

        if not checkpoint:
            return False

        db.delete(checkpoint)
        db.commit()
        return True
