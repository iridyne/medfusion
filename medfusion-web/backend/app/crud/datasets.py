"""数据集 CRUD 操作"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, or_

from app.models.database import Dataset


class DatasetCRUD:
    """数据集 CRUD 操作类"""
    
    @staticmethod
    def create(
        db: Session,
        name: str,
        data_path: str,
        description: Optional[str] = None,
        num_samples: Optional[int] = None,
        num_classes: Optional[int] = None,
        train_samples: Optional[int] = None,
        val_samples: Optional[int] = None,
        test_samples: Optional[int] = None,
        class_distribution: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None,
    ) -> Dataset:
        """创建数据集记录"""
        dataset = Dataset(
            name=name,
            description=description,
            data_path=data_path,
            num_samples=num_samples,
            num_classes=num_classes,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            class_distribution=class_distribution,
            tags=tags,
            created_by=created_by,
        )
        
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        return dataset
    
    @staticmethod
    def get(db: Session, dataset_id: int) -> Optional[Dataset]:
        """根据 ID 获取数据集"""
        return db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    @staticmethod
    def get_by_name(db: Session, name: str) -> Optional[Dataset]:
        """根据名称获取数据集"""
        return db.query(Dataset).filter(Dataset.name == name).first()
    
    @staticmethod
    def list(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        num_classes: Optional[int] = None,
        sort_by: str = "created_at",
        order: str = "desc",
    ) -> List[Dataset]:
        """获取数据集列表"""
        query = db.query(Dataset)
        
        # 筛选
        if num_classes is not None:
            query = query.filter(Dataset.num_classes == num_classes)
        
        # 排序
        if order == "desc":
            query = query.order_by(getattr(Dataset, sort_by).desc())
        else:
            query = query.order_by(getattr(Dataset, sort_by).asc())
        
        # 分页
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def search(
        db: Session,
        keyword: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Dataset]:
        """搜索数据集"""
        query = db.query(Dataset).filter(
            or_(
                Dataset.name.ilike(f"%{keyword}%"),
                Dataset.description.ilike(f"%{keyword}%"),
            )
        )
        
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def update(
        db: Session,
        dataset_id: int,
        **kwargs
    ) -> Optional[Dataset]:
        """更新数据集信息"""
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        
        if not dataset:
            return None
        
        # 更新字段
        for key, value in kwargs.items():
            if hasattr(dataset, key) and value is not None:
                setattr(dataset, key, value)
        
        db.commit()
        db.refresh(dataset)
        
        return dataset
    
    @staticmethod
    def delete(db: Session, dataset_id: int) -> bool:
        """删除数据集"""
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        
        if not dataset:
            return False
        
        db.delete(dataset)
        db.commit()
        
        return True
    
    @staticmethod
    def get_statistics(db: Session) -> Dict[str, Any]:
        """获取数据集统计信息"""
        total_count = db.query(func.count(Dataset.id)).scalar()
        total_samples = db.query(func.sum(Dataset.num_samples)).scalar() or 0
        avg_samples = db.query(func.avg(Dataset.num_samples)).scalar() or 0
        
        return {
            "total_count": total_count,
            "total_samples": total_samples,
            "avg_samples": round(avg_samples, 2),
        }
    
    @staticmethod
    def get_class_counts(db: Session) -> List[int]:
        """获取所有数据集的类别数"""
        results = db.query(Dataset.num_classes).distinct().all()
        return sorted([r[0] for r in results if r[0] is not None])
