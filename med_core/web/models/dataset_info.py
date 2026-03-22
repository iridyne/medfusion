"""数据集信息模型"""

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text

from ..database import Base
from ..time_utils import utcnow


class DatasetInfo(Base):
    """数据集元信息"""

    __tablename__ = "dataset_info"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # 数据路径与类型
    data_path = Column(String(500), nullable=False)
    dataset_type = Column(
        String(50),
        nullable=False,
        default="image",
        index=True,
    )  # image / tabular / multimodal
    status = Column(
        String(50),
        nullable=False,
        default="ready",
        index=True,
    )  # uploading / processing / ready / error

    # 数据统计
    size_bytes = Column(Integer, nullable=True)
    num_samples = Column(Integer, nullable=True)
    num_classes = Column(Integer, nullable=True)
    train_samples = Column(Integer, nullable=True)
    val_samples = Column(Integer, nullable=True)
    test_samples = Column(Integer, nullable=True)
    class_distribution = Column(JSON, nullable=True)

    # 附加信息
    tags = Column(JSON, nullable=True)
    analysis = Column(JSON, nullable=True)
    created_by = Column(String(255), nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=utcnow, index=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    def __repr__(self) -> str:
        return f"<DatasetInfo(id={self.id}, name='{self.name}', type='{self.dataset_type}')>"
