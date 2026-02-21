"""
简化版模拟测试 - 直接测试核心功能

目标：用最简单的方式测试 MedFusion 是否能跑通
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from med_core.backbones import create_tabular_backbone, create_vision_backbone
from med_core.fusion import MultiModalFusionModel, create_fusion_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 测试记录
test_log = []


def log_test(stage, success, time_taken, notes=""):
    """记录测试结果"""
    test_log.append(
        {
            "stage": stage,
            "success": success,
            "time": f"{time_taken:.2f}s",
            "notes": notes,
        }
    )
    status = "✅" if success else "❌"
    logger.info(f"{status} {stage} - {time_taken:.2f}s - {notes}")


def test_stage_1_data_generation():
    """阶段 1: 生成测试数据"""
    logger.info("\n" + "=" * 60)
    logger.info("阶段 1: 生成测试数据")
    logger.info("=" * 60)
    start = time.time()

    try:
        # 创建目录
        data_dir = Path("data/simple_test")
        images_dir = data_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # 生成 100 张图片
        records = []
        for i in range(100):
            # 生成随机图片
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = images_dir / f"img_{i:03d}.png"
            img.save(img_path)

            # 生成标签和表格数据
            label = i % 2
            records.append(
                {
                    "image_path": str(img_path),
                    "age": np.random.randint(20, 80),
                    "sex": np.random.choice([0, 1]),
                    "label": label,
                }
            )

        # 保存 CSV
        df = pd.DataFrame(records)
        csv_path = data_dir / "data.csv"
        df.to_csv(csv_path, index=False)

        elapsed = time.time() - start
        log_test("数据生成", True, elapsed, f"生成 {len(df)} 条数据")
        return data_dir, csv_path, df

    except Exception as e:
        elapsed = time.time() - start
        log_test("数据生成", False, elapsed, str(e))
        raise


def test_stage_2_create_model():
    """阶段 2: 创建模型"""
    logger.info("\n" + "=" * 60)
    logger.info("阶段 2: 创建模型")
    logger.info("=" * 60)
    start = time.time()

    try:
        # 创建 vision backbone
        logger.info("创建 ResNet18 backbone...")
        vision_backbone = create_vision_backbone(
            backbone_name="resnet18",
            pretrained=False,  # 快速测试，不用预训练
            feature_dim=512,  # ResNet18 输出维度
            attention_type="none",  # 简化测试，不用注意力
        )

        # 创建 tabular backbone
        logger.info("创建 tabular backbone...")
        tabular_backbone = create_tabular_backbone(
            input_dim=2,  # age + sex
            hidden_dims=[16, 16],
            output_dim=16,
        )

        # 创建 fusion module
        logger.info("创建 fusion module...")
        fusion_module = create_fusion_module(
            fusion_type="concatenate",
            vision_dim=512,  # ResNet18 输出
            tabular_dim=16,
        )

        # 创建完整模型
        logger.info("组装完整模型...")
        model = MultiModalFusionModel(
            vision_backbone=vision_backbone,
            tabular_backbone=tabular_backbone,
            fusion_module=fusion_module,
            num_classes=2,
        )

        elapsed = time.time() - start
        log_test("模型创建", True, elapsed, "ResNet18 + MLP + Concatenate")
        return model

    except Exception as e:
        elapsed = time.time() - start
        log_test("模型创建", False, elapsed, str(e))
        raise


class SimpleDataset(Dataset):
    """简单的数据集类"""

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 加载图片
        img = Image.open(row["image_path"]).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        # 表格数据
        tabular = torch.tensor([row["age"] / 100.0, row["sex"]], dtype=torch.float32)

        # 标签
        label = torch.tensor(row["label"], dtype=torch.long)

        return img_tensor, tabular, label


def test_stage_3_training():
    """阶段 3: 训练模型"""
    logger.info("\n" + "=" * 60)
    logger.info("阶段 3: 训练模型（1 个 epoch）")
    logger.info("=" * 60)
    start = time.time()

    try:
        # 生成数据
        data_dir, csv_path, df = test_stage_1_data_generation()

        # 创建模型
        model = test_stage_2_create_model()

        # 创建数据加载器
        dataset = SimpleDataset(df)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        # 设置训练
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        logger.info(f"使用设备: {device}")
        logger.info("开始训练...")

        # 训练 1 个 epoch
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, tabular, labels) in enumerate(dataloader):
            images = images.to(device)
            tabular = tabular.to(device)
            labels = labels.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(images, tabular)

            # 模型可能返回字典，提取 logits
            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("output", outputs))
            else:
                logits = outputs

            loss = criterion(logits, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 2 == 0:
                logger.info(
                    f"Batch {batch_idx + 1}/{len(dataloader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Acc: {100.0 * correct / total:.2f}%"
                )

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total

        elapsed = time.time() - start
        log_test(
            "模型训练",
            True,
            elapsed,
            f"Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%",
        )

        return model, avg_loss, accuracy

    except Exception as e:
        elapsed = time.time() - start
        log_test("模型训练", False, elapsed, str(e))
        raise


def print_summary():
    """打印测试总结"""
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)

    total_time = sum(float(t["time"].replace("s", "")) for t in test_log)
    success_count = sum(1 for t in test_log if t["success"])

    logger.info(f"\n总耗时: {total_time:.2f} 秒")
    logger.info(f"成功: {success_count}/{len(test_log)}")

    logger.info("\n详细结果:")
    for t in test_log:
        status = "✅" if t["success"] else "❌"
        logger.info(f"  {status} {t['stage']}: {t['time']} - {t['notes']}")


def main():
    """主函数"""
    logger.info("🚀 开始 MedFusion 简化测试")
    logger.info("目标：验证核心功能是否能跑通\n")

    try:
        # 运行完整训练测试
        model, loss, acc = test_stage_3_training()

        logger.info("\n🎉 测试成功！MedFusion 核心功能可用")
        logger.info(f"最终结果: Loss={loss:.4f}, Accuracy={acc:.2f}%")

    except Exception as e:
        logger.error(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()

    finally:
        print_summary()


if __name__ == "__main__":
    main()
