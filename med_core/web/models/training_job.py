"""训练任务模型"""

from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, ForeignKey, Text
from datetime import datetime

from ..database import Base


class TrainingJob(Base):
    """训练任务"""

    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(100), unique=True, nullable=False, index=True)

    # 关联实验
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)

    # 配置
    config = Column(JSON, nullable=False)

    # 状态
    status = Column(String(50), default="queued", index=True)  # queued, running, paused, completed, failed, stopped

    # 进度
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, nullable=False)
    current_batch = Column(Integer, default=0)
    total_batches = Column(Integer, nullable=True)
    progress = Column(Float, default=0.0)  # 0-100

    # 当前指标
    current_loss = Column(Float, nullable=True)
    current_accuracy = Column(Float, nullable=True)
    current_lr = Column(Float, nullable=True)

    # 最佳指标
    best_loss = Column(Float, nullable=True)
    best_accuracy = Column(Float, nullable=True)
    best_epoch = Column(Integer, nullable=True)

    # 系统资源
    gpu_usage = Column(Float, nullable=True)
    gpu_memory = Column(Float, nullable=True)
    cpu_usage = Column(Float, nullable=True)
    ram_usage = Column(Float, nullable=True)

    # 路径
    output_dir = Column(String(500), nullable=True)
    log_file = Column(String(500), nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # 错误信息
    error_message = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<TrainingJob(job_id='{self.job_id}', status='{self.status}', progress={self.progress:.1f}%)>"
