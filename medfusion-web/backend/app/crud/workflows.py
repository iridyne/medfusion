"""工作流 CRUD 操作"""
import builtins
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.models.database import Workflow, WorkflowExecution


class WorkflowCRUD:
    """工作流 CRUD 操作"""

    @staticmethod
    def create(
        db: Session,
        name: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        description: str | None = None,
        created_by: str | None = None,
    ) -> Workflow:
        """创建工作流"""
        workflow = Workflow(
            name=name,
            description=description,
            nodes=nodes,
            edges=edges,
            created_by=created_by,
        )
        db.add(workflow)
        db.commit()
        db.refresh(workflow)
        return workflow

    @staticmethod
    def get(db: Session, workflow_id: int) -> Workflow | None:
        """获取工作流"""
        return db.query(Workflow).filter(Workflow.id == workflow_id).first()

    @staticmethod
    def get_by_name(db: Session, name: str) -> Workflow | None:
        """根据名称获取工作流"""
        return db.query(Workflow).filter(Workflow.name == name).first()

    @staticmethod
    def list(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        created_by: str | None = None,
    ) -> list[Workflow]:
        """列出工作流"""
        query = db.query(Workflow)

        if created_by:
            query = query.filter(Workflow.created_by == created_by)

        return query.order_by(Workflow.updated_at.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def update(
        db: Session,
        workflow_id: int,
        name: str | None = None,
        description: str | None = None,
        nodes: builtins.list[dict[str, Any]] | None = None,
        edges: builtins.list[dict[str, Any]] | None = None,
    ) -> Workflow | None:
        """更新工作流"""
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()

        if not workflow:
            return None

        if name is not None:
            workflow.name = name
        if description is not None:
            workflow.description = description
        if nodes is not None:
            workflow.nodes = nodes
        if edges is not None:
            workflow.edges = edges

        workflow.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(workflow)
        return workflow

    @staticmethod
    def delete(db: Session, workflow_id: int) -> bool:
        """删除工作流"""
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()

        if not workflow:
            return False

        db.delete(workflow)
        db.commit()
        return True

    @staticmethod
    def increment_execution_count(db: Session, workflow_id: int):
        """增加执行次数"""
        workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()

        if workflow:
            workflow.execution_count += 1
            workflow.last_executed_at = datetime.utcnow()
            db.commit()


class WorkflowExecutionCRUD:
    """工作流执行记录 CRUD 操作"""

    @staticmethod
    def create(
        db: Session,
        workflow_id: int,
        status: str = "pending",
    ) -> WorkflowExecution:
        """创建执行记录"""
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            status=status,
            started_at=datetime.utcnow(),
        )
        db.add(execution)
        db.commit()
        db.refresh(execution)
        return execution

    @staticmethod
    def get(db: Session, execution_id: int) -> WorkflowExecution | None:
        """获取执行记录"""
        return db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()

    @staticmethod
    def list_by_workflow(
        db: Session,
        workflow_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> list[WorkflowExecution]:
        """列出工作流的执行记录"""
        return (
            db.query(WorkflowExecution)
            .filter(WorkflowExecution.workflow_id == workflow_id)
            .order_by(WorkflowExecution.started_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    @staticmethod
    def update_status(
        db: Session,
        execution_id: int,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> WorkflowExecution | None:
        """更新执行状态"""
        execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()

        if not execution:
            return None

        execution.status = status

        if result is not None:
            execution.result = result

        if error is not None:
            execution.error = error

        if status in ["completed", "failed"]:
            execution.completed_at = datetime.utcnow()
            if execution.started_at:
                execution.duration = (execution.completed_at - execution.started_at).total_seconds()

        db.commit()
        db.refresh(execution)
        return execution
