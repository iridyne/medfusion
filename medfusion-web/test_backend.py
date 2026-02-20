"""
测试 Web UI 后端核心功能

测试工作流执行引擎和训练服务
"""
import asyncio
import sys
from pathlib import Path

# 添加后端路径
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))


async def test_workflow_engine():
    """测试工作流执行引擎"""
    print("=" * 80)
    print("测试工作流执行引擎")
    print("=" * 80)
    
    from app.core.workflow_engine import WorkflowEngine
    
    # 定义一个简单的工作流
    workflow = {
        "nodes": [
            {
                "id": "node1",
                "type": "dataset_loader",
                "data": {
                    "config": {
                        "data_path": "/path/to/data"
                    }
                }
            },
            {
                "id": "node2",
                "type": "backbone_selector",
                "data": {
                    "config": {
                        "backbone_type": "resnet18",
                        "pretrained": True
                    }
                }
            },
            {
                "id": "node3",
                "type": "trainer",
                "data": {
                    "config": {
                        "epochs": 2,
                        "batch_size": 32,
                        "learning_rate": 0.001
                    }
                }
            }
        ],
        "edges": [
            {
                "id": "e1",
                "source": "node1",
                "target": "node3"
            },
            {
                "id": "e2",
                "source": "node2",
                "target": "node3"
            }
        ]
    }
    
    # 创建执行引擎
    engine = WorkflowEngine(workflow)
    
    # 进度回调
    async def progress_callback(node_id, status, execution, progress):
        print(f"  [{progress:.1f}%] Node {node_id}: {status}")
        if execution and execution.outputs:
            print(f"    Outputs: {list(execution.outputs.keys())}")
    
    # 执行工作流
    print("\n执行工作流...")
    result = await engine.execute(progress_callback=progress_callback)
    
    print(f"\n执行结果:")
    print(f"  状态: {result['status']}")
    print(f"  执行节点数: {len(result['executions'])}")
    
    if result['status'] == 'success':
        print(f"  统计信息:")
        print(f"    总节点: {result['statistics']['total_nodes']}")
        print(f"    完成节点: {result['statistics']['completed_nodes']}")
        print(f"    总耗时: {result['statistics']['total_duration']:.2f}s")
        print("\n✅ 工作流执行引擎测试通过!")
    else:
        print(f"  错误: {result.get('error')}")
        print("\n❌ 工作流执行引擎测试失败!")
    
    return result['status'] == 'success'


async def test_training_service():
    """测试训练服务"""
    print("\n" + "=" * 80)
    print("测试训练服务")
    print("=" * 80)
    
    from app.services.training_service import TrainingService
    
    # 训练配置
    config = {
        "model_config": {
            "backbone": "resnet18",
            "num_classes": 10,
            "pretrained": False,  # 不使用预训练以加快测试
            "feature_dim": 128
        },
        "data_config": {
            "num_samples": 100  # 少量数据用于测试
        },
        "training_config": {
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "use_amp": False,  # 不使用混合精度以兼容 CPU
            "gradient_checkpointing": False,
            "use_scheduler": False,
            "save_model": False
        }
    }
    
    # 创建训练服务
    service = TrainingService("test_job", config)
    
    # 进度回调
    progress_updates = []
    
    async def progress_callback(data):
        msg_type = data.get("type")
        progress_updates.append(msg_type)
        
        if msg_type == "status_update":
            print(f"  状态: {data.get('status')}")
        elif msg_type == "batch_progress":
            epoch = data.get("epoch")
            batch = data.get("batch")
            total = data.get("total_batches")
            loss = data.get("loss")
            print(f"  Epoch {epoch}, Batch {batch}/{total}, Loss: {loss:.4f}")
        elif msg_type == "epoch_completed":
            metrics = data.get("metrics", {})
            print(f"  Epoch {metrics.get('epoch')} 完成:")
            print(f"    Train Loss: {metrics.get('train_loss', 0):.4f}")
            print(f"    Train Acc: {metrics.get('train_acc', 0):.2f}%")
            print(f"    Val Loss: {metrics.get('val_loss', 0):.4f}")
            print(f"    Val Acc: {metrics.get('val_acc', 0):.2f}%")
        elif msg_type == "training_completed":
            print(f"  训练完成!")
        elif msg_type == "training_failed":
            print(f"  训练失败: {data.get('error')}")
    
    # 运行训练
    print("\n开始训练...")
    try:
        await service.run(progress_callback=progress_callback)
        
        # 获取最终状态
        status = service.get_status()
        
        print(f"\n训练结果:")
        print(f"  状态: {status['status']}")
        print(f"  进度: {status['progress']:.1f}%")
        print(f"  Epoch: {status['current_epoch']}/{status['total_epochs']}")
        print(f"  耗时: {status['duration']:.2f}s")
        
        if status['status'] == 'completed':
            print(f"  最终指标:")
            metrics = status['metrics']
            print(f"    Train Loss: {metrics.get('train_loss', 0):.4f}")
            print(f"    Train Acc: {metrics.get('train_acc', 0):.2f}%")
            print(f"    Val Loss: {metrics.get('val_loss', 0):.4f}")
            print(f"    Val Acc: {metrics.get('val_acc', 0):.2f}%")
            print("\n✅ 训练服务测试通过!")
            return True
        else:
            print(f"  错误: {status.get('error')}")
            print("\n❌ 训练服务测试失败!")
            return False
    
    except Exception as e:
        print(f"\n❌ 训练服务测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """运行所有测试"""
    print("MedFusion Web UI 后端核心功能测试")
    print("=" * 80)
    
    results = []
    
    # 测试工作流引擎
    try:
        result = await test_workflow_engine()
        results.append(("工作流执行引擎", result))
    except Exception as e:
        print(f"\n❌ 工作流执行引擎测试异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(("工作流执行引擎", False))
    
    # 测试训练服务
    try:
        result = await test_training_service()
        results.append(("训练服务", result))
    except Exception as e:
        print(f"\n❌ 训练服务测试异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(("训练服务", False))
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 所有测试通过!")
    else:
        print("\n⚠️  部分测试失败")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
