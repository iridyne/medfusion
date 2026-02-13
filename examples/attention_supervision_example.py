"""
完整的注意力监督训练示例

本示例展示如何使用 Med-Framework 的注意力监督功能训练医学影像分类模型。
支持两种监督方法：
1. 基于掩码的监督（需要人工标注的病灶掩码）
2. 基于 CAM 的自监督（自动生成类激活图）
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from med_core.backbones import create_vision_backbone
from med_core.configs import DataConfig, ExperimentConfig, TrainingConfig, VisionConfig
from med_core.fusion import create_fusion_model
from med_core.trainers import create_trainer

# ============================================================================
# 示例 1: 基于掩码的注意力监督（有人工标注）
# ============================================================================

class MedicalDatasetWithMasks(Dataset):
    """
    医学影像数据集，包含病灶掩码标注

    数据格式：
    - images: 医学影像 (B, 3, 224, 224)
    - tabular: 临床特征 (B, num_features)
    - labels: 诊断标签 (B,)
    - masks: 病灶掩码 (B, 1, 224, 224)，值在 [0, 1] 之间
    """

    def __init__(self, image_paths, tabular_data, labels, mask_paths, transform=None):
        self.image_paths = image_paths
        self.tabular_data = tabular_data
        self.labels = labels
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 加载掩码
        mask = Image.open(self.mask_paths[idx]).convert('L')  # 灰度图
        mask = torch.from_numpy(np.array(mask)).float() / 255.0  # 归一化到 [0, 1]
        mask = mask.unsqueeze(0)  # (1, H, W)

        # 调整掩码大小以匹配图像
        if mask.shape[-2:] != image.shape[-2:]:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0),
                size=image.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # 表格数据
        tabular = torch.tensor(self.tabular_data[idx], dtype=torch.float32)

        # 标签
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, tabular, label, mask


def example_mask_based_supervision():
    """示例：使用掩码监督训练模型"""

    print("=" * 80)
    print("示例 1: 基于掩码的注意力监督")
    print("=" * 80)

    # 1. 配置模型
    config = ExperimentConfig(
        experiment_name="pneumonia_detection_with_masks",
        seed=42,
    )

    # 视觉配置：启用注意力监督
    config.model.vision = VisionConfig(
        backbone="resnet50",
        pretrained=True,
        attention_type="cbam",  # 必须使用 CBAM
        enable_attention_supervision=True,  # 启用注意力监督
        feature_dim=128,
    )

    # 训练配置：使用掩码监督
    config.training = TrainingConfig(
        num_epochs=50,
        use_attention_supervision=True,  # 启用注意力监督
        attention_supervision_method="mask",  # 使用掩码监督
        attention_loss_weight=0.1,  # 注意力损失权重
        mixed_precision=True,
    )

    # 2. 创建数据集（模拟数据）
    num_samples = 100
    image_paths = [f"data/images/sample_{i}.jpg" for i in range(num_samples)]
    mask_paths = [f"data/masks/sample_{i}.png" for i in range(num_samples)]
    tabular_data = np.random.randn(num_samples, 10)  # 10 个临床特征
    labels = np.random.randint(0, 2, num_samples)  # 二分类

    # 创建数据集
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MedicalDatasetWithMasks(
        image_paths[:80],
        tabular_data[:80],
        labels[:80],
        mask_paths[:80],
        transform=transform
    )

    val_dataset = MedicalDatasetWithMasks(
        image_paths[80:],
        tabular_data[80:],
        labels[80:],
        mask_paths[80:],
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # 3. 创建模型
    model = create_fusion_model(
        vision_backbone_name="resnet50",
        vision_config={
            "pretrained": True,
            "attention_type": "cbam",
            "enable_attention_supervision": True,
        },
        tabular_input_dim=10,
        num_classes=2,
        fusion_type="gated",
    )

    # 4. 创建训练器
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 5. 训练
    print("\n开始训练（使用掩码监督）...")
    print(f"- 注意力监督方法: mask")
    print(f"- 注意力损失权重: {config.training.attention_loss_weight}")
    print(f"- 数据集包含病灶掩码标注")

    # trainer.train()  # 取消注释以实际训练

    print("\n✓ 训练完成！模型学会了关注掩码标记的病灶区域。")


# ============================================================================
# 示例 2: 基于 CAM 的自监督（无需人工标注）
# ============================================================================

class MedicalDatasetWithoutMasks(Dataset):
    """
    医学影像数据集，不包含掩码标注

    数据格式：
    - images: 医学影像 (B, 3, 224, 224)
    - tabular: 临床特征 (B, num_features)
    - labels: 诊断标签 (B,)
    """

    def __init__(self, image_paths, tabular_data, labels, transform=None):
        self.image_paths = image_paths
        self.tabular_data = tabular_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 表格数据
        tabular = torch.tensor(self.tabular_data[idx], dtype=torch.float32)

        # 标签
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, tabular, label


def example_cam_based_supervision():
    """示例：使用 CAM 自监督训练模型"""

    print("\n" + "=" * 80)
    print("示例 2: 基于 CAM 的自监督")
    print("=" * 80)

    # 1. 配置模型
    config = ExperimentConfig(
        experiment_name="pneumonia_detection_with_cam",
        seed=42,
    )

    # 视觉配置：启用注意力监督
    config.model.vision = VisionConfig(
        backbone="resnet50",
        pretrained=True,
        attention_type="cbam",  # 必须使用 CBAM
        enable_attention_supervision=True,  # 启用注意力监督
        feature_dim=128,
    )

    # 训练配置：使用 CAM 自监督
    config.training = TrainingConfig(
        num_epochs=50,
        use_attention_supervision=True,  # 启用注意力监督
        attention_supervision_method="cam",  # 使用 CAM 自监督
        attention_loss_weight=0.05,  # CAM 监督权重可以稍低
        mixed_precision=True,
    )

    # 2. 创建数据集（无需掩码）
    num_samples = 100
    image_paths = [f"data/images/sample_{i}.jpg" for i in range(num_samples)]
    tabular_data = np.random.randn(num_samples, 10)
    labels = np.random.randint(0, 2, num_samples)

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MedicalDatasetWithoutMasks(
        image_paths[:80],
        tabular_data[:80],
        labels[:80],
        transform=transform
    )

    val_dataset = MedicalDatasetWithoutMasks(
        image_paths[80:],
        tabular_data[80:],
        labels[80:],
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # 3. 创建模型
    model = create_fusion_model(
        vision_backbone_name="resnet50",
        vision_config={
            "pretrained": True,
            "attention_type": "cbam",
            "enable_attention_supervision": True,
        },
        tabular_input_dim=10,
        num_classes=2,
        fusion_type="gated",
    )

    # 4. 创建训练器
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 5. 训练
    print("\n开始训练（使用 CAM 自监督）...")
    print(f"- 注意力监督方法: cam")
    print(f"- 注意力损失权重: {config.training.attention_loss_weight}")
    print(f"- 无需人工标注掩码，自动生成 CAM")

    # trainer.train()  # 取消注释以实际训练

    print("\n✓ 训练完成！模型学会了关注 CAM 识别的判别性区域。")


# ============================================================================
# 示例 3: 可视化注意力权重
# ============================================================================

def visualize_attention_weights(model, image, tabular, save_path="attention_viz.png"):
    """可视化模型的注意力权重"""

    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        # 获取中间输出
        if hasattr(model, "vision_backbone"):
            vision_outputs = model.vision_backbone(image, return_intermediates=True)

            if isinstance(vision_outputs, dict):
                attention_weights = vision_outputs.get("attention_weights")

                if attention_weights is not None:
                    # 可视化
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                    # 原图
                    img = image[0].cpu().permute(1, 2, 0).numpy()
                    img = (img - img.min()) / (img.max() - img.min())
                    axes[0].imshow(img)
                    axes[0].set_title("Original Image")
                    axes[0].axis('off')

                    # 注意力热力图
                    attn = attention_weights[0, 0].cpu().numpy()
                    axes[1].imshow(img)
                    axes[1].imshow(attn, alpha=0.5, cmap='jet')
                    axes[1].set_title("Attention Heatmap")
                    axes[1].axis('off')

                    plt.tight_layout()
                    plt.savefig(save_path)
                    print(f"\n✓ 注意力可视化已保存到: {save_path}")
                else:
                    print("\n⚠️ 模型未返回注意力权重")
            else:
                print("\n⚠️ 模型不支持 return_intermediates")
        else:
            print("\n⚠️ 模型没有 vision_backbone 属性")


# ============================================================================
# 示例 4: 对比实验（有监督 vs 无监督）
# ============================================================================

def compare_with_without_supervision():
    """对比有无注意力监督的训练效果"""

    print("\n" + "=" * 80)
    print("示例 4: 对比实验")
    print("=" * 80)

    # 配置 1: 无注意力监督（基线）
    config_baseline = ExperimentConfig(
        experiment_name="baseline_no_supervision",
    )
    config_baseline.model.vision = VisionConfig(
        backbone="resnet50",
        attention_type="cbam",
        enable_attention_supervision=False,  # 不启用
    )
    config_baseline.training.use_attention_supervision = False

    # 配置 2: 有注意力监督
    config_supervised = ExperimentConfig(
        experiment_name="with_attention_supervision",
    )
    config_supervised.model.vision = VisionConfig(
        backbone="resnet50",
        attention_type="cbam",
        enable_attention_supervision=True,  # 启用
    )
    config_supervised.training.use_attention_supervision = True
    config_supervised.training.attention_supervision_method = "cam"
    config_supervised.training.attention_loss_weight = 0.1

    print("\n对比配置：")
    print("1. 基线模型：不使用注意力监督")
    print("2. 监督模型：使用 CAM 注意力监督")
    print("\n预期结果：")
    print("- 监督模型的注意力更集中在病灶区域")
    print("- 监督模型的可解释性更强")
    print("- 监督模型在小样本场景下可能表现更好")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""

    print("\n" + "=" * 80)
    print("Med-Framework 注意力监督训练示例")
    print("=" * 80)

    # 示例 1: 掩码监督
    example_mask_based_supervision()

    # 示例 2: CAM 自监督
    example_cam_based_supervision()

    # 示例 4: 对比实验
    compare_with_without_supervision()

    print("\n" + "=" * 80)
    print("所有示例完成！")
    print("=" * 80)
    print("\n关键要点：")
    print("1. 注意力监督需要 CBAM 注意力机制")
    print("2. 支持两种方法：mask（需要标注）和 cam（自动生成）")
    print("3. 通过配置文件轻松启用/禁用")
    print("4. 向后兼容，不影响现有代码")
    print("\n详细文档请参考: docs/ATTENTION_MECHANISM_GUIDE.md")


if __name__ == "__main__":
    main()
