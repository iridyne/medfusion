"""模型信息模型"""

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text

from ..database import Base


class ModelInfo(Base):
    """模型信息"""

    __tablename__ = "model_info"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # 模型类型
    model_type = Column(String(100), nullable=False)  # classification, segmentation, etc.
    architecture = Column(String(100), nullable=False)  # resnet50, swin_transformer, etc.

    # 模型配置
    config = Column(JSON, nullable=True)

    # 性能指标
    metrics = Column(JSON, nullable=True)
    accuracy = Column(Float, nullable=True)
    loss = Column(Float, nullable=True)

    # 模型大小
    num_parameters = Column(Integer, nullable=True)
    model_size_mb = Column(Float, nullable=True)

    # 文件路径
    checkpoint_path = Column(String(500), nullable=False)
    config_path = Column(String(500), nullable=True)

    # 训练信息
    trained_epochs = Column(Integer, nullable=True)
    training_time = Column(Float, nullable=True)  # 秒

    # 数据集信息
    dataset_name = Column(String(255), nullable=True)
    num_classes = Column(Integer, nullable=True)

    # 标签
    tags = Column(JSON, nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<ModelInfo(id={self.id}, name='{self.name}', architecture='{self.architecture}')>"
