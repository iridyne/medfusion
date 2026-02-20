#!/usr/bin/env python3
"""测试数据库集成"""
import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.database import SessionLocal
from app.crud import WorkflowCRUD, TrainingJobCRUD


def test_workflow_crud():
    """测试工作流 CRUD 操作"""
    print("\n=== 测试工作流 CRUD ===")
    
    db = SessionLocal()
    
    try:
        # 创建工作流
        print("1. 创建工作流...")
        workflow = WorkflowCRUD.create(
            db=db,
            name="测试工作流",
            description="这是一个测试工作流",
            nodes=[
                {"id": "1", "type": "data_loader", "data": {"path": "/data"}},
                {"id": "2", "type": "model", "data": {"backbone": "resnet18"}},
            ],
            edges=[
                {"id": "e1", "source": "1", "target": "2"},
            ],
        )
        print(f"   ✅ 创建成功: ID={workflow.id}, Name={workflow.name}")
        
        # 获取工作流
        print("\n2. 获取工作流...")
        retrieved = WorkflowCRUD.get(db, workflow.id)
        print(f"   ✅ 获取成功: {retrieved.name}")
        print(f"      节点数: {len(retrieved.nodes)}")
        print(f"      边数: {len(retrieved.edges)}")
        
        # 列出所有工作流
        print("\n3. 列出所有工作流...")
        workflows = WorkflowCRUD.list(db)
        print(f"   ✅ 找到 {len(workflows)} 个工作流")
        
        # 更新工作流
        print("\n4. 更新工作流...")
        updated = WorkflowCRUD.update(
            db=db,
            workflow_id=workflow.id,
            name="更新后的工作流",
            description="描述已更新",
            nodes=retrieved.nodes,
            edges=retrieved.edges,
        )
        print(f"   ✅ 更新成功: {updated.name}")
        
        # 删除工作流
        print("\n5. 删除工作流...")
        success = WorkflowCRUD.delete(db, workflow.id)
        print(f"   ✅ 删除成功: {success}")
        
    finally:
        db.close()


def test_training_crud():
    """测试训练任务 CRUD 操作"""
    print("\n=== 测试训练任务 CRUD ===")
    
    db = SessionLocal()
    
    try:
        # 创建训练任务
        print("1. 创建训练任务...")
        job = TrainingJobCRUD.create(
            db=db,
            job_id="test_job_001",
            name="测试训练任务",
            description="这是一个测试训练任务",
            model_config={"backbone": "resnet18", "num_classes": 10},
            data_config={"batch_size": 32, "num_workers": 4},
            training_config={"epochs": 50, "lr": 0.001},
        )
        print(f"   ✅ 创建成功: ID={job.id}, JobID={job.job_id}")
        
        # 更新状态
        print("\n2. 更新训练状态...")
        TrainingJobCRUD.update_status(db, job.job_id, "running")
        updated = TrainingJobCRUD.get(db, job.job_id)
        print(f"   ✅ 状态更新: {updated.status}")
        
        # 更新进度
        print("\n3. 更新训练进度...")
        TrainingJobCRUD.update_progress(
            db=db,
            job_id=job.job_id,
            progress=0.5,
            current_epoch=25,
            current_metrics={"loss": 0.5, "accuracy": 0.85},
        )
        updated = TrainingJobCRUD.get(db, job.job_id)
        print(f"   ✅ 进度更新: {updated.progress * 100}%")
        print(f"      当前 epoch: {updated.current_epoch}/{updated.total_epochs}")
        print(f"      指标: {updated.current_metrics}")
        
        # 列出所有任务
        print("\n4. 列出所有训练任务...")
        jobs = TrainingJobCRUD.list(db)
        print(f"   ✅ 找到 {len(jobs)} 个训练任务")
        
        # 按状态筛选
        print("\n5. 按状态筛选...")
        running_jobs = TrainingJobCRUD.list(db, status="running")
        print(f"   ✅ 找到 {len(running_jobs)} 个运行中的任务")
        
        # 删除任务
        print("\n6. 删除训练任务...")
        success = TrainingJobCRUD.delete(db, job.job_id)
        print(f"   ✅ 删除成功: {success}")
        
    finally:
        db.close()


def main():
    """运行所有测试"""
    print("=" * 60)
    print("数据库集成测试")
    print("=" * 60)
    
    try:
        test_workflow_crud()
        test_training_crud()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
