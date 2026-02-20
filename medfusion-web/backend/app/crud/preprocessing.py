"""预处理任务 CRUD 操作"""

from typing import Dict, List, Optional

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.models.database import PreprocessingTask


class PreprocessingTaskCRUD:
    """预处理任务数据库操作"""

    @staticmethod
    def create(
        db: Session,
        task_id: str,
        name: str,
        input_dir: str,
        output_dir: str,
        config: Dict,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> PreprocessingTask:
        """创建预处理任务"""
        task = PreprocessingTask(
            task_id=task_id,
            name=name,
            description=description,
            input_dir=input_dir,
            output_dir=output_dir,
            config=config,
            status="pending",
            progress=0.0,
            total_images=0,
            processed_images=0,
            failed_images=0,
            created_by=created_by,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        return task

    @staticmethod
    def get(db: Session, task_id: int) -> Optional[PreprocessingTask]:
        """根据 ID 获取预处理任务"""
        return (
            db.query(PreprocessingTask).filter(PreprocessingTask.id == task_id).first()
        )

    @staticmethod
    def get_by_task_id(db: Session, task_id: str) -> Optional[PreprocessingTask]:
        """根据 task_id 获取预处理任务"""
        return (
            db.query(PreprocessingTask)
            .filter(PreprocessingTask.task_id == task_id)
            .first()
        )

    @staticmethod
    def list(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None,
        sort_by: str = "created_at",
        order: str = "desc",
    ) -> List[PreprocessingTask]:
        """列出预处理任务"""
        query = db.query(PreprocessingTask)

        # 筛选
        if status:
            query = query.filter(PreprocessingTask.status == status)

        # 排序
        if order == "desc":
            query = query.order_by(desc(getattr(PreprocessingTask, sort_by)))
        else:
            query = query.order_by(getattr(PreprocessingTask, sort_by))

        return query.offset(skip).limit(limit).all()

    @staticmethod
    def search(
        db: Session,
        keyword: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[PreprocessingTask]:
        """搜索预处理任务"""
        query = db.query(PreprocessingTask).filter(
            (PreprocessingTask.name.contains(keyword))
            | (PreprocessingTask.description.contains(keyword))
        )
        return query.offset(skip).limit(limit).all()

    @staticmethod
    def update_status(
        db: Session,
        task_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> Optional[PreprocessingTask]:
        """更新任务状态"""
        task = PreprocessingTaskCRUD.get_by_task_id(db, task_id)
        if task:
            task.status = status
            if error:
                task.error = error
            db.commit()
            db.refresh(task)
        return task

    @staticmethod
    def update_progress(
        db: Session,
        task_id: str,
        progress: float,
        processed_images: int,
        failed_images: int = 0,
    ) -> Optional[PreprocessingTask]:
        """更新任务进度"""
        task = PreprocessingTaskCRUD.get_by_task_id(db, task_id)
        if task:
            task.progress = progress
            task.processed_images = processed_images
            task.failed_images = failed_images
            db.commit()
            db.refresh(task)
        return task

    @staticmethod
    def update(
        db: Session,
        task_id: int,
        **kwargs,
    ) -> Optional[PreprocessingTask]:
        """更新预处理任务"""
        task = PreprocessingTaskCRUD.get(db, task_id)
        if task:
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            db.commit()
            db.refresh(task)
        return task

    @staticmethod
    def delete(db: Session, task_id: int) -> bool:
        """删除预处理任务"""
        task = PreprocessingTaskCRUD.get(db, task_id)
        if task:
            db.delete(task)
            db.commit()
            return True
        return False

    @staticmethod
    def get_statistics(db: Session) -> Dict:
        """获取预处理任务统计信息"""
        total_tasks = db.query(func.count(PreprocessingTask.id)).scalar()

        # 按状态统计
        status_counts = {}
        for status in ["pending", "running", "completed", "failed", "cancelled"]:
            count = (
                db.query(func.count(PreprocessingTask.id))
                .filter(PreprocessingTask.status == status)
                .scalar()
            )
            status_counts[status] = count

        # 总处理图像数
        total_processed = (
            db.query(func.sum(PreprocessingTask.processed_images)).scalar() or 0
        )
        total_failed = db.query(func.sum(PreprocessingTask.failed_images)).scalar() or 0

        return {
            "total_tasks": total_tasks,
            "status_counts": status_counts,
            "total_processed_images": total_processed,
            "total_failed_images": total_failed,
        }

    @staticmethod
    def count(db: Session, status: Optional[str] = None) -> int:
        """统计任务数量"""
        query = db.query(func.count(PreprocessingTask.id))
        if status:
            query = query.filter(PreprocessingTask.status == status)
        return query.scalar()
