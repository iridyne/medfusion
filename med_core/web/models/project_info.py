"""项目工作区模型。"""

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text

from ..database import Base
from ..time_utils import utcnow


class ProjectInfo(Base):
    """本地专业版项目记录。"""

    __tablename__ = "project_info"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    task_type = Column(String(64), nullable=False, index=True)
    template_id = Column(String(100), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="draft", index=True)

    dataset_id = Column(Integer, ForeignKey("dataset_info.id"), nullable=True)
    config_path = Column(String(500), nullable=True)
    output_dir = Column(String(500), nullable=True)
    latest_job_id = Column(String(100), nullable=True)
    latest_model_id = Column(Integer, nullable=True)

    tags = Column(JSON, nullable=True)
    project_meta = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=utcnow, index=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    def __repr__(self) -> str:
        return (
            f"<ProjectInfo(id={self.id}, name='{self.name}', "
            f"task_type='{self.task_type}', status='{self.status}')>"
        )
