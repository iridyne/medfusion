"""实验模型"""

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text

from ..database import Base


class Experiment(Base):
    """实验记录"""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # 配置信息
    config = Column(JSON, nullable=False)

    # 状态
    status = Column(String(50), default="created", index=True)  # created, running, completed, failed

    # 结果
    metrics = Column(JSON, nullable=True)
    best_metric = Column(String(100), nullable=True)
    best_value = Column(String(100), nullable=True)

    # 路径
    output_dir = Column(String(500), nullable=True)
    checkpoint_path = Column(String(500), nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # 标签
    tags = Column(JSON, nullable=True)

    def __repr__(self) -> str:
        return f"<Experiment(id={self.id}, name='{self.name}', status='{self.status}')>"
