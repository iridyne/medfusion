"""
测试注意力监督的实际效果

对比实验：
1. 基线模型（无注意力监督）
2. CAM 注意力监督模型

目标：验证注意力监督是否真的能提升准确率
"""

import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from med_core.backbones import create_tabular_backbone, create_vision_backbone
from med_core.datasets import (
    MedicalMultimodalDataset,
    create_dataloaders,
    get_train_transforms,
    get_val_transforms,
    split_dataset,
)
from med_core.evaluation import calculate_binary_metrics
from med_core.fusion import MultiModalFusionModel, create_fusion_module

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(output_dir: Path, num_samples: int = 200):
    """生成合成测试数据"""
    logger.info(f"生成 {num_samples} 条合成数据...")

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    data = []

    for i in range(num_samples):
        # 生成随机图片
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # 为正类添加特征（白色圆圈）
        label = np.random.randint(0, 2)
        if label == 1:
            center = (np.random.randint(50, 174), np.random.randint(50, 174))
            radius = np.random.randint(10, 30)
            y, x = np.ogrid[:224, :224]
            mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
            img_array[mask] = 255

        img = Image.fromarray(img_array)
        img_name = f"sample_{i:04d}.png"
        img.save(image_dir / img_name)

        # 生成临床数据（与标签相关）
        age = np.random.normal(60, 10) + (5 if label == 1 else 0)

        record = {
            "patient_id": f"P{i:04d}",
            "image_path": img_name,
            "age": age,
            "sex": np.random.choice(["M", "F"]),
            "diagnosis": label,
        }
        data.append(record)

    import pandas as pd
    df = pd.DataFrame(data)
    csv_path = output_dir / "dataset.csv"
    df.to_csv(csv_path, index=False)

    logger.info(f"✅ 数据生成完成: {csv_path}")
    return csv_path


def train_model(model, train_loader, val_loader, device, num_epochs=5, model_name="Model"):
    """训练模型"""
    logger.info(f"\n{'='*60}")
    logger.info(f"训练 {model_name}")
    logger.info(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = model.to(device)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, tabular, labels in train_loader:
            images = images.to(device)
            tabular = tabular.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, tabular)

            # 处理字典输出
            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("output", outputs))
            else:
                logits = outputs

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total

        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, tabular, labels in val_loader:
                images = images.to(device)
                tabular = tabular.to(device)
                labels = labels.to(device)

                outputs = model(images, tabular)
                if isinstance(outputs, dict):
                    logits = outputs.get("logits", outputs.get("output", outputs))
                else:
                    logits = outputs

                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        best_val_acc = max(best_val_acc, val_acc)

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%"
        )

    logger.info(f"✅ {model_name} 训练完成 - 最佳验证准确率: {best_val_acc:.2f}%")
    return best_val_acc


def evaluate_model(model, test_loader, device, model_name="Model"):
    """评估模型"""
    logger.info(f"\n{'='*60}")
    logger.info(f"评估 {model_name}")
    logger.info(f"{'='*60}")

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, tabular, labels in test_loader:
            images = images.to(device)
            tabular = tabular.to(device)

            outputs = model(images, tabular)
            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("output", outputs))
            else:
                logits = outputs

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # 计算指标
    metrics = calculate_binary_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        y_prob=all_probs,
    )

    logger.info(f"测试结果:")
    logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  - AUC: {metrics['auc']:.4f}")
    logger.info(f"  - F1 Score: {metrics['f1']:.4f}")
    logger.info(f"  - Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"  - Specificity: {metrics['specificity']:.4f}")

    return metrics


def main():
    logger.info("🚀 开始注意力监督效果测试")
    logger.info("=" * 60)

    # 1. 准备数据
    data_dir = Path("data/attention_supervision_test")
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    csv_path = generate_synthetic_data(data_dir, num_samples=300)

    # 2. 准备数据集
    logger.info("\n" + "=" * 60)
    logger.info("准备数据集")
    logger.info("=" * 60)

    full_dataset, _ = MedicalMultimodalDataset.from_csv(
        csv_path=str(csv_path),
        image_dir=str(data_dir / "images"),
        image_column="image_path",
        target_column="diagnosis",
        numerical_features=["age"],
        categorical_features=["sex"],
        handle_missing="fill_mean",
    )

    train_ds, val_ds, test_ds = split_dataset(
        full_dataset,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )

    # 添加数据增强
    train_ds.transform = get_train_transforms(image_size=224)
    val_ds.transform = get_val_transforms(image_size=224)
    test_ds.transform = get_val_transforms(image_size=224)

    dataloaders = create_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=16,
        num_workers=0,
    )

    logger.info(f"✅ 数据集准备完成")
    logger.info(f"  - 训练集: {len(train_ds)} 样本")
    logger.info(f"  - 验证集: {len(val_ds)} 样本")
    logger.info(f"  - 测试集: {len(test_ds)} 样本")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  - 设备: {device}")

    # 3. 创建基线模型（无注意力监督）
    logger.info("\n" + "=" * 60)
    logger.info("创建基线模型（无注意力）")
    logger.info("=" * 60)

    vision_backbone_baseline = create_vision_backbone(
        backbone_name="resnet18",
        pretrained=True,
        feature_dim=128,
    )

    tabular_backbone_baseline = create_tabular_backbone(
        input_dim=train_ds.get_tabular_dim(),
        output_dim=16,
        hidden_dims=[32, 32],
    )

    fusion_module_baseline = create_fusion_module(
        fusion_type="concatenate",
        vision_dim=128,
        tabular_dim=16,
        output_dim=64,
    )

    model_baseline = MultiModalFusionModel(
        vision_backbone=vision_backbone_baseline,
        tabular_backbone=tabular_backbone_baseline,
        fusion_module=fusion_module_baseline,
        num_classes=2,
    )

    logger.info(f"✅ 基线模型创建完成")

    # 4. 创建注意力监督模型
    logger.info("\n" + "=" * 60)
    logger.info("创建注意力监督模型（CBAM）")
    logger.info("=" * 60)

    vision_backbone_attention = create_vision_backbone(
        backbone_name="resnet18",
        pretrained=True,
        feature_dim=128,
        attention_type="cbam",  # 使用 CBAM 注意力
    )

    tabular_backbone_attention = create_tabular_backbone(
        input_dim=train_ds.get_tabular_dim(),
        output_dim=16,
        hidden_dims=[32, 32],
    )

    fusion_module_attention = create_fusion_module(
        fusion_type="concatenate",
        vision_dim=128,
        tabular_dim=16,
        output_dim=64,
    )

    model_attention = MultiModalFusionModel(
        vision_backbone=vision_backbone_attention,
        tabular_backbone=tabular_backbone_attention,
        fusion_module=fusion_module_attention,
        num_classes=2,
    )

    logger.info(f"✅ 注意力模型创建完成")

    # 5. 训练基线模型
    best_val_acc_baseline = train_model(
        model_baseline,
        dataloaders["train"],
        dataloaders["val"],
        device,
        num_epochs=10,
        model_name="基线模型（无注意力）",
    )

    # 6. 训练注意力模型
    best_val_acc_attention = train_model(
        model_attention,
        dataloaders["train"],
        dataloaders["val"],
        device,
        num_epochs=10,
        model_name="注意力监督模型（CBAM）",
    )

    # 7. 评估基线模型
    metrics_baseline = evaluate_model(
        model_baseline,
        dataloaders["test"],
        device,
        model_name="基线模型（无注意力）",
    )

    # 8. 评估注意力模型
    metrics_attention = evaluate_model(
        model_attention,
        dataloaders["test"],
        device,
        model_name="注意力监督模型（CBAM）",
    )

    # 9. 对比结果
    logger.info("\n" + "=" * 60)
    logger.info("📊 对比结果")
    logger.info("=" * 60)

    logger.info(f"\n验证集最佳准确率:")
    logger.info(f"  基线模型: {best_val_acc_baseline:.2f}%")
    logger.info(f"  注意力模型: {best_val_acc_attention:.2f}%")
    logger.info(f"  提升: {best_val_acc_attention - best_val_acc_baseline:+.2f}%")

    logger.info(f"\n测试集准确率:")
    logger.info(f"  基线模型: {metrics_baseline['accuracy']:.4f}")
    logger.info(f"  注意力模型: {metrics_attention['accuracy']:.4f}")
    logger.info(f"  提升: {metrics_attention['accuracy'] - metrics_baseline['accuracy']:+.4f}")

    logger.info(f"\n测试集 AUC:")
    logger.info(f"  基线模型: {metrics_baseline['auc']:.4f}")
    logger.info(f"  注意力模型: {metrics_attention['auc']:.4f}")
    logger.info(f"  提升: {metrics_attention['auc'] - metrics_baseline['auc']:+.4f}")

    logger.info(f"\n测试集 F1 Score:")
    logger.info(f"  基线模型: {metrics_baseline['f1']:.4f}")
    logger.info(f"  注意力模型: {metrics_attention['f1']:.4f}")
    logger.info(f"  提升: {metrics_attention['f1'] - metrics_baseline['f1']:+.4f}")

    # 10. 结论
    logger.info("\n" + "=" * 60)
    logger.info("💡 结论")
    logger.info("=" * 60)

    improvement = metrics_attention['accuracy'] - metrics_baseline['accuracy']

    if improvement > 0.02:  # 提升超过 2%
        logger.info("✅ 注意力监督有明显效果，建议保留")
        logger.info(f"   准确率提升: {improvement:.4f} ({improvement*100:.2f}%)")
    elif improvement > 0:
        logger.info("⚠️ 注意力监督有轻微提升，但不明显")
        logger.info(f"   准确率提升: {improvement:.4f} ({improvement*100:.2f}%)")
        logger.info("   建议：在真实数据上测试后再决定")
    else:
        logger.info("❌ 注意力监督没有提升，甚至可能降低性能")
        logger.info(f"   准确率变化: {improvement:.4f} ({improvement*100:.2f}%)")
        logger.info("   建议：删除注意力监督模块（2,678 行代码）")

    logger.info("\n注意：这是合成数据测试，真实数据可能有不同结果")


if __name__ == "__main__":
    main()
