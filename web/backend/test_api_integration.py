"""API 集成测试脚本

测试所有后端 API 端点的功能
"""
import asyncio
import json
from typing import Any

import httpx

BASE_URL = "http://localhost:8000"


class APITester:
    """API 测试器"""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def close(self):
        """关闭客户端"""
        await self.client.aclose()

    def print_result(self, test_name: str, success: bool, data: Any = None):
        """打印测试结果"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if data:
            print(f"   Response: {json.dumps(data, indent=2, ensure_ascii=False)[:200]}...")
        print()

    async def test_workflow_apis(self):
        """测试工作流 API"""
        print("=" * 60)
        print("测试工作流 API")
        print("=" * 60)

        # 1. 创建工作流
        workflow_data = {
            "name": "测试工作流",
            "description": "这是一个测试工作流",
            "nodes": [
                {
                    "id": "node1",
                    "type": "dataLoader",
                    "position": {"x": 100, "y": 100},
                    "data": {"path": "/data/test"}
                },
                {
                    "id": "node2",
                    "type": "model",
                    "position": {"x": 300, "y": 100},
                    "data": {"backbone": "resnet18"}
                }
            ],
            "edges": [
                {
                    "id": "edge1",
                    "source": "node1",
                    "target": "node2"
                }
            ]
        }

        try:
            response = await self.client.post("/api/workflows/", json=workflow_data)
            success = response.status_code == 200
            data = response.json() if success else None
            workflow_id = data.get("id") if data else None
            self.print_result("创建工作流", success, data)
        except Exception as e:
            self.print_result("创建工作流", False, {"error": str(e)})
            return

        # 2. 获取工作流列表
        try:
            response = await self.client.get("/api/workflows/")
            success = response.status_code == 200
            data = response.json() if success else None
            self.print_result("获取工作流列表", success, data)
        except Exception as e:
            self.print_result("获取工作流列表", False, {"error": str(e)})

        # 3. 获取工作流详情
        if workflow_id:
            try:
                response = await self.client.get(f"/api/workflows/{workflow_id}")
                success = response.status_code == 200
                data = response.json() if success else None
                self.print_result("获取工作流详情", success, data)
            except Exception as e:
                self.print_result("获取工作流详情", False, {"error": str(e)})

        # 4. 更新工作流
        if workflow_id:
            try:
                update_data = {
                    **workflow_data,
                    "description": "更新后的描述"
                }
                response = await self.client.put(f"/api/workflows/{workflow_id}", json=update_data)
                success = response.status_code == 200
                data = response.json() if success else None
                self.print_result("更新工作流", success, data)
            except Exception as e:
                self.print_result("更新工作流", False, {"error": str(e)})

        # 5. 删除工作流
        if workflow_id:
            try:
                response = await self.client.delete(f"/api/workflows/{workflow_id}")
                success = response.status_code == 200
                data = response.json() if success else None
                self.print_result("删除工作流", success, data)
            except Exception as e:
                self.print_result("删除工作流", False, {"error": str(e)})

    async def test_training_apis(self):
        """测试训练 API"""
        print("=" * 60)
        print("测试训练 API")
        print("=" * 60)

        # 1. 启动训练
        training_config = {
            "name": "测试训练任务",
            "description": "这是一个测试训练任务",
            "model_config": {
                "backbone": "resnet18",
                "num_classes": 3,
                "pretrained": True
            },
            "data_config": {
                "data_dir": "/data/test",
                "batch_size": 32
            },
            "training_config": {
                "epochs": 10,
                "learning_rate": 0.001,
                "optimizer": "adam"
            }
        }

        try:
            response = await self.client.post("/api/training/start", json=training_config)
            success = response.status_code == 200
            data = response.json() if success else None
            job_id = data.get("job_id") if data else None
            self.print_result("启动训练", success, data)
        except Exception as e:
            self.print_result("启动训练", False, {"error": str(e)})
            return

        # 等待一下让训练初始化
        await asyncio.sleep(1)

        # 2. 获取训练状态
        if job_id:
            try:
                response = await self.client.get(f"/api/training/status/{job_id}")
                success = response.status_code == 200
                data = response.json() if success else None
                self.print_result("获取训练状态", success, data)
            except Exception as e:
                self.print_result("获取训练状态", False, {"error": str(e)})

        # 3. 获取训练任务列表
        try:
            response = await self.client.get("/api/training/list")
            success = response.status_code == 200
            data = response.json() if success else None
            self.print_result("获取训练任务列表", success, data)
        except Exception as e:
            self.print_result("获取训练任务列表", False, {"error": str(e)})

        # 4. 暂停训练
        if job_id:
            try:
                response = await self.client.post(f"/api/training/pause/{job_id}")
                success = response.status_code == 200
                data = response.json() if success else None
                self.print_result("暂停训练", success, data)
            except Exception as e:
                self.print_result("暂停训练", False, {"error": str(e)})

        await asyncio.sleep(0.5)

        # 5. 恢复训练
        if job_id:
            try:
                response = await self.client.post(f"/api/training/resume/{job_id}")
                success = response.status_code == 200
                data = response.json() if success else None
                self.print_result("恢复训练", success, data)
            except Exception as e:
                self.print_result("恢复训练", False, {"error": str(e)})

        await asyncio.sleep(0.5)

        # 6. 停止训练
        if job_id:
            try:
                response = await self.client.post(f"/api/training/stop/{job_id}")
                success = response.status_code == 200
                data = response.json() if success else None
                self.print_result("停止训练", success, data)
            except Exception as e:
                self.print_result("停止训练", False, {"error": str(e)})

    async def test_model_apis(self):
        """测试模型 API"""
        print("=" * 60)
        print("测试模型 API")
        print("=" * 60)

        # 1. 创建模型
        model_data = {
            "name": "测试模型",
            "description": "这是一个测试模型",
            "backbone": "resnet18",
            "num_classes": 3,
            "accuracy": 0.92,
            "loss": 0.25,
            "metrics": {
                "precision": 0.91,
                "recall": 0.93,
                "f1": 0.92
            },
            "format": "pytorch",
            "trained_epochs": 50,
            "tags": ["test", "resnet"]
        }

        try:
            response = await self.client.post("/api/models/", json=model_data)
            success = response.status_code == 200
            data = response.json() if success else None
            model_id = data.get("id") if data else None
            self.print_result("创建模型", success, data)
        except Exception as e:
            self.print_result("创建模型", False, {"error": str(e)})
            return

        # 2. 获取模型列表
        try:
            response = await self.client.get("/api/models/")
            success = response.status_code == 200
            data = response.json() if success else None
            self.print_result("获取模型列表", success, data)
        except Exception as e:
            self.print_result("获取模型列表", False, {"error": str(e)})

        # 3. 获取模型详情
        if model_id:
            try:
                response = await self.client.get(f"/api/models/{model_id}")
                success = response.status_code == 200
                data = response.json() if success else None
                self.print_result("获取模型详情", success, data)
            except Exception as e:
                self.print_result("获取模型详情", False, {"error": str(e)})

        # 4. 搜索模型
        try:
            response = await self.client.get("/api/models/search?keyword=测试")
            success = response.status_code == 200
            data = response.json() if success else None
            self.print_result("搜索模型", success, data)
        except Exception as e:
            self.print_result("搜索模型", False, {"error": str(e)})

        # 5. 获取统计信息
        try:
            response = await self.client.get("/api/models/statistics")
            success = response.status_code == 200
            data = response.json() if success else None
            self.print_result("获取统计信息", success, data)
        except Exception as e:
            self.print_result("获取统计信息", False, {"error": str(e)})

        # 6. 获取 Backbone 列表
        try:
            response = await self.client.get("/api/models/backbones")
            success = response.status_code == 200
            data = response.json() if success else None
            self.print_result("获取 Backbone 列表", success, data)
        except Exception as e:
            self.print_result("获取 Backbone 列表", False, {"error": str(e)})

        # 7. 更新模型
        if model_id:
            try:
                update_data = {
                    "description": "更新后的描述",
                    "accuracy": 0.95
                }
                response = await self.client.put(f"/api/models/{model_id}", json=update_data)
                success = response.status_code == 200
                data = response.json() if success else None
                self.print_result("更新模型", success, data)
            except Exception as e:
                self.print_result("更新模型", False, {"error": str(e)})

        # 8. 删除模型
        if model_id:
            try:
                response = await self.client.delete(f"/api/models/{model_id}")
                success = response.status_code == 200
                data = response.json() if success else None
                self.print_result("删除模型", success, data)
            except Exception as e:
                self.print_result("删除模型", False, {"error": str(e)})

    async def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "=" * 60)
        print("开始 API 集成测试")
        print("=" * 60 + "\n")

        try:
            await self.test_workflow_apis()
            await self.test_training_apis()
            await self.test_model_apis()
            await self.test_dataset_apis()

            print("\n" + "=" * 60)
            print("所有测试完成！")
            print("=" * 60 + "\n")

        finally:
            await self.close()

    async def test_dataset_apis(self):
        """测试数据集 API"""
        print("=" * 60)
        print("测试数据集 API")
        print("=" * 60)

        # 1. 创建数据集
        dataset_data = {
            "name": "测试数据集",
            "description": "这是一个测试数据集",
            "data_path": "/data/test_dataset",
            "num_samples": 1000,
            "num_classes": 3,
            "train_samples": 700,
            "val_samples": 200,
            "test_samples": 100,
            "class_distribution": {
                "class_0": 400,
                "class_1": 350,
                "class_2": 250
            },
            "tags": ["test", "medical"]
        }

        try:
            response = await self.client.post("/api/datasets/", json=dataset_data)
            success = response.status_code == 200
            data = response.json() if success else None
            dataset_id = data.get("id") if data else None
            self.print_result("创建数据集", success, data)
        except Exception as e:
            self.print_result("创建数据集", False, {"error": str(e)})
            return

        # 2. 获取数据集列表
        try:
            response = await self.client.get("/api/datasets/")
            success = response.status_code == 200
            data = response.json() if success else None
            self.print_result("获取数据集列表", success, data)
        except Exception as e:
            self.print_result("获取数据集列表", False, {"error": str(e)})

        # 3. 获取数据集详情
        if dataset_id:
            try:
                response = await self.client.get(f"/api/datasets/{dataset_id}")
                success = response.status_code == 200
                data = response.json() if success else None
                self.print_result("获取数据集详情", success, data)
            except Exception as e:
                self.print_result("获取数据集详情", False, {"error": str(e)})

        # 4. 搜索数据集
        try:
            response = await self.client.get("/api/datasets/search?keyword=测试")
            success = response.status_code == 200
            data = response.json() if success else None
            self.print_result("搜索数据集", success, data)
        except Exception as e:
            self.print_result("搜索数据集", False, {"error": str(e)})

        # 5. 获取统计信息
        try:
            response = await self.client.get("/api/datasets/statistics")
            success = response.status_code == 200
            data = response.json() if success else None
            self.print_result("获取统计信息", success, data)
        except Exception as e:
            self.print_result("获取统计信息", False, {"error": str(e)})

        # 6. 获取类别数列表
        try:
            response = await self.client.get("/api/datasets/class-counts")
            success = response.status_code == 200
            data = response.json() if success else None
            self.print_result("获取类别数列表", success, data)
        except Exception as e:
            self.print_result("获取类别数列表", False, {"error": str(e)})

        # 7. 分析数据集
        if dataset_id:
            try:
                response = await self.client.post(f"/api/datasets/{dataset_id}/analyze")
                success = response.status_code == 200
                data = response.json() if success else None
                self.print_result("分析数据集", success, data)
            except Exception as e:
                self.print_result("分析数据集", False, {"error": str(e)})

        # 8. 更新数据集
        if dataset_id:
            try:
                update_data = {
                    "description": "更新后的描述",
                    "num_samples": 1100
                }
                response = await self.client.put(f"/api/datasets/{dataset_id}", json=update_data)
                success = response.status_code == 200
                data = response.json() if success else None
                self.print_result("更新数据集", success, data)
            except Exception as e:
                self.print_result("更新数据集", False, {"error": str(e)})

        # 9. 删除数据集
        if dataset_id:
            try:
                response = await self.client.delete(f"/api/datasets/{dataset_id}")
                success = response.status_code == 200
                data = response.json() if success else None
                self.print_result("删除数据集", success, data)
            except Exception as e:
                self.print_result("删除数据集", False, {"error": str(e)})


async def main():
    """主函数"""
    tester = APITester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
