"""模型 CRUD 操作"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models.database import Model


class ModelCRUD:
    """模型 CRUD 操作类"""
    
    @staticmethod
    def create(
        db: Session,
        name: str,
        backbone: str,
        num_classes: int,
        model_path: str,
        description: Optional[str] = None,
        accuracy: Optional[float] = None,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        file_size: Optional[int] = None,
        format: Optional[str] = "pytorch",
        input_shape: Optional[List[int]] = None,
        training_job_id: Optional[int] = None,
        trained_epochs: Optional[int] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None,
    ) -> Model:
        """创建模型"""
        model = Model(
            name=name,
            description=description,
            backbone=backbone,
            num_classes=num_classes,
            input_shape=input_shape,
            accuracy=accuracy,
            loss=loss,
            metrics=metrics,
            model_path=model_path,
            file_size=file_size,
            format=format,
            training_job_id=training_job_id,
            trained_epochs=trained_epochs,
            tags=tags,
            created_by=created_by,
        )
        
        db.add(model)
        db.commit()
        db.refresh(model)
        
        return model
    
    @staticmethod
    def get(db: Session, model_id: int) -> Optional[Model]:
        """根据 ID 获取模型"""
        return db.query(Model).filter(Model.id == model_id).first()
    
    @staticmethod
    def get_by_name(db: Session, name: str) -> Optional[Model]:
        """根据名称获取模型"""
        return db.query(Model).filter(Model.name == name).first()
    
    @staticmethod
    def list(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        backbone: Optional[str] = None,
        format: Optional[str] = None,
        sort_by: str = "created_at",
        order: str = "desc",
    ) -> List[Model]:
        """列出所有模型"""
        query = db.query(Model)
        
        # 筛选
        if backbone:
            query = query.filter(Model.backbone == backbone)
        if format:
            query = query.filter(Model.format == format)
        
        # 排序
        if order == "desc":
            query = query.order_by(desc(getattr(Model, sort_by)))
        else:
            query = query.order_by(getattr(Model, sort_by))
        
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def search(
        db: Session,
        keyword: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Model]:
        """搜索模型"""
        query = db.query(Model).filter(
            (Model.name.contains(keyword)) | 
            (Model.description.contains(keyword))
        )
        
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def update(
        db: Session,
        model_id: int,
        **kwargs
    ) -> Optional[Model]:
        """更新模型"""
        model = db.query(Model).filter(Model.id == model_id).first()
        
        if not model:
            return None
        
        for key, value in kwargs.items():
            if hasattr(model, key) and value is not None:
                setattr(model, key, value)
        
        db.commit()
        db.refresh(model)
        
        return model
    
    @staticmethod
    def delete(db: Session, model_id: int) -> bool:
        """删除模型"""
        model = db.query(Model).filter(Model.id == model_id).first()
        
        if not model:
            return False
        
        db.delete(model)
        db.commit()
        
        return True
    
    @staticmethod
    def get_statistics(db: Session) -> Dict[str, Any]:
        """获取模型统计信息"""
        total_count = db.query(Model).count()
        
        # 总参数量（需要从模型配置中计算，这里简化处理）
        models = db.query(Model).all()
        total_params = 0
        total_size = 0
        total_accuracy = 0.0
        accuracy_count = 0
        
        for model in models:
            if model.file_size:
                total_size += model.file_size
            if model.accuracy:
                total_accuracy += model.accuracy
                accuracy_count += 1
        
        avg_accuracy = total_accuracy / accuracy_count if accuracy_count > 0 else 0.0
        
        return {
            "total_count": total_count,
            "total_size": total_size,
            "avg_accuracy": avg_accuracy,
        }
    
    @staticmethod
    def get_backbones(db: Session) -> List[str]:
        """获取所有使用的 Backbone"""
        result = db.query(Model.backbone).distinct().all()
        return [r[0] for r in result]
    
    @staticmethod
    def get_formats(db: Session) -> List[str]:
        """获取所有模型格式"""
        result = db.query(Model.format).distinct().all()
        return [r[0] for r in result if r[0]]
