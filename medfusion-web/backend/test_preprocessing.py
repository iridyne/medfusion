"""预处理功能测试脚本

测试预处理 API 的完整功能
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from app.crud.preprocessing import PreprocessingTaskCRUD
from app.models.database import Base
from app.services.preprocessing_service import preprocessing_service
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 创建测试数据库
TEST_DB_URL = "sqlite:///./test_preprocessing.db"
engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建表
Base.metadata.create_all(bind=engine)


def test_crud_operations():
    """测试 CRUD 操作"""
    print("\n" + "=" * 60)
    print("测试 1: CRUD 操作")
    print("=" * 60)

    db = SessionLocal()

    try:
        # 1. 创建任务
        print("\n1. 创建预处理任务...")
        task = PreprocessingTaskCRUD.create(
            db=db,
            task_id="test_task_001",
            name="测试预处理任务",
            description="这是一个测试任务",
            input_dir="/tmp/input",
            output_dir="/tmp/output",
            config={
                "size": 224,
                "normalize": "percentile",
                "remove_artifacts": False,
                "enhance_contrast": True,
            },
            created_by="test_user",
        )
        print(f"✅ 创建成功: ID={task.id}, task_id={task.task_id}")

        # 2. 获取任务
        print("\n2. 获取任务...")
        retrieved_task = PreprocessingTaskCRUD.get(db, task.id)
        assert retrieved_task is not None
        assert retrieved_task.task_id == "test_task_001"
        print(f"✅ 获取成功: {retrieved_task.name}")

        # 3. 通过 task_id 获取
        print("\n3. 通过 task_id 获取...")
        task_by_id = PreprocessingTaskCRUD.get_by_task_id(db, "test_task_001")
        assert task_by_id is not None
        print(f"✅ 获取成功: {task_by_id.name}")

        # 4. 更新状态
        print("\n4. 更新任务状态...")
        updated_task = PreprocessingTaskCRUD.update_status(
            db, "test_task_001", "running"
        )
        assert updated_task.status == "running"
        print(f"✅ 状态更新成功: {updated_task.status}")

        # 5. 更新进度
        print("\n5. 更新任务进度...")
        updated_task = PreprocessingTaskCRUD.update_progress(
            db, "test_task_001", progress=0.5, processed_images=50, failed_images=2
        )
        assert updated_task.progress == 0.5
        assert updated_task.processed_images == 50
        print(f"✅ 进度更新成功: {updated_task.progress * 100}%")

        # 6. 列出任务
        print("\n6. 列出所有任务...")
        tasks = PreprocessingTaskCRUD.list(db, skip=0, limit=10)
        print(f"✅ 找到 {len(tasks)} 个任务")

        # 7. 搜索任务
        print("\n7. 搜索任务...")
        search_results = PreprocessingTaskCRUD.search(db, "测试")
        print(f"✅ 搜索到 {len(search_results)} 个任务")

        # 8. 获取统计信息
        print("\n8. 获取统计信息...")
        stats = PreprocessingTaskCRUD.get_statistics(db)
        print("✅ 统计信息:")
        print(f"   - 总任务数: {stats['total_tasks']}")
        print(f"   - 状态分布: {stats['status_counts']}")
        print(f"   - 已处理图像: {stats['total_processed_images']}")

        # 9. 删除任务
        print("\n9. 删除任务...")
        success = PreprocessingTaskCRUD.delete(db, task.id)
        assert success
        print("✅ 删除成功")

        print("\n" + "=" * 60)
        print("✅ 所有 CRUD 测试通过!")
        print("=" * 60)

    finally:
        db.close()


async def test_preprocessing_service():
    """测试预处理服务"""
    print("\n" + "=" * 60)
    print("测试 2: 预处理服务")
    print("=" * 60)

    # 创建测试目录和图像
    import tempfile

    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # 创建测试图像
        print("\n1. 创建测试图像...")
        num_images = 5
        for i in range(num_images):
            img = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            img.save(input_dir / f"test_image_{i}.jpg")
        print(f"✅ 创建了 {num_images} 张测试图像")

        # 定义进度回调
        progress_updates = []

        async def progress_callback(data):
            progress_updates.append(data)
            print(f"   进度更新: {data['type']}")
            if data["type"] == "progress":
                print(
                    f"      - 进度: {data['progress'] * 100:.1f}%"
                    f" ({data['processed_images']}/{data.get('total_images', '?')})"
                )

        # 启动预处理
        print("\n2. 启动预处理...")
        config = {
            "size": 128,
            "normalize": "minmax",
            "remove_artifacts": False,
            "enhance_contrast": False,
        }

        result = await preprocessing_service.start_preprocessing(
            task_id="test_service_001",
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            config=config,
            progress_callback=progress_callback,
        )

        print("\n✅ 预处理完成!")
        print(f"   - 状态: {result['status']}")
        print(f"   - 总图像数: {result['total_images']}")
        print(f"   - 已处理: {result['processed_images']}")
        print(f"   - 失败: {result['failed_images']}")
        print(f"   - 耗时: {result['duration']:.2f}s")

        # 验证输出
        print("\n3. 验证输出...")
        output_files = list(output_dir.glob("*.jpg"))
        assert len(output_files) == num_images
        print(f"✅ 输出文件数量正确: {len(output_files)}")

        # 验证进度回调
        print("\n4. 验证进度回调...")
        assert len(progress_updates) > 0
        assert any(u["type"] == "started" for u in progress_updates)
        assert any(u["type"] == "completed" for u in progress_updates)
        print(f"✅ 收到 {len(progress_updates)} 次进度更新")

        print("\n" + "=" * 60)
        print("✅ 预处理服务测试通过!")
        print("=" * 60)


async def test_cancellation():
    """测试任务取消"""
    print("\n" + "=" * 60)
    print("测试 3: 任务取消")
    print("=" * 60)

    import tempfile

    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # 创建大量测试图像
        print("\n1. 创建测试图像...")
        num_images = 100
        for i in range(num_images):
            img = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            img.save(input_dir / f"test_image_{i}.jpg")
        print(f"✅ 创建了 {num_images} 张测试图像")

        # 定义进度回调
        async def progress_callback(data):
            if data["type"] == "progress":
                print(
                    f"   进度: {data['progress'] * 100:.1f}% "
                    f"({data['processed_images']}/{data.get('total_images', '?')})"
                )

        # 启动预处理任务
        print("\n2. 启动预处理任务...")
        task_id = "test_cancel_001"
        config = {
            "size": 128,
            "normalize": "minmax",
            "remove_artifacts": False,
            "enhance_contrast": False,
        }

        # 创建异步任务
        task = asyncio.create_task(
            preprocessing_service.start_preprocessing(
                task_id=task_id,
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                config=config,
                progress_callback=progress_callback,
            )
        )
        preprocessing_service.register_task(task_id, task)

        # 等待一小段时间
        await asyncio.sleep(0.5)

        # 取消任务
        print("\n3. 取消任务...")
        success = preprocessing_service.cancel_task(task_id)
        assert success
        print("✅ 取消请求已发送")

        # 等待任务完成
        result = await task
        print("\n✅ 任务已取消!")
        print(f"   - 状态: {result['status']}")
        print(f"   - 已处理: {result['processed_images']}/{result['total_images']}")

        assert result["status"] == "cancelled"
        assert result["processed_images"] < result["total_images"]

        print("\n" + "=" * 60)
        print("✅ 任务取消测试通过!")
        print("=" * 60)


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("预处理功能测试")
    print("=" * 60)

    try:
        # 测试 1: CRUD 操作
        test_crud_operations()

        # 测试 2: 预处理服务
        asyncio.run(test_preprocessing_service())

        # 测试 3: 任务取消
        asyncio.run(test_cancellation())

        print("\n" + "=" * 60)
        print("🎉 所有测试通过!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # 清理测试数据库
        Path("test_preprocessing.db").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
