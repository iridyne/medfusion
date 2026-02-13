"""
注意力监督快速开始指南

这是一个最简化的示例，展示如何在 3 步内启用注意力监督训练。
"""

from med_core.configs import ExperimentConfig
from med_core.fusion import create_fusion_model

# ============================================================================
# 方法 1: 使用 CAM 自监督（推荐，无需额外标注）
# ============================================================================


def quick_start_cam():
    """3 步启用 CAM 注意力监督"""

    print("=" * 60)
    print("快速开始：CAM 注意力监督（无需掩码标注）")
    print("=" * 60)

    # 步骤 1: 配置启用注意力监督
    config = ExperimentConfig()
    config.model.vision.attention_type = "cbam"  # 必须使用 CBAM
    config.model.vision.enable_attention_supervision = True  # 启用
    config.training.use_attention_supervision = True
    config.training.attention_supervision_method = "cam"  # CAM 自动生成
    config.training.attention_loss_weight = 0.1

    print("\n✓ 步骤 1: 配置完成")
    print(f"  - 注意力类型: {config.model.vision.attention_type}")
    print(f"  - 监督方法: {config.training.attention_supervision_method}")
    print(f"  - 损失权重: {config.training.attention_loss_weight}")

    # 步骤 2: 创建模型（与普通模型相同）
    _model = create_fusion_model(
        vision_backbone_name="resnet50",
        vision_config={
            "attention_type": "cbam",
            "enable_attention_supervision": True,
        },
        tabular_input_dim=10,
        num_classes=2,
    )

    print("\n✓ 步骤 2: 模型创建完成")

    # 步骤 3: 训练（数据集无需包含掩码）
    # train_loader, val_loader = ...  # 标准数据加载器
    # trainer = create_trainer(model, train_loader, val_loader, config)
    # trainer.train()

    print("\n✓ 步骤 3: 准备训练")
    print("  - 数据集格式: (images, tabular, labels)")
    print("  - 无需掩码标注，CAM 自动生成")

    print("\n" + "=" * 60)
    print("完成！模型将自动学习关注判别性区域。")
    print("=" * 60)


# ============================================================================
# 方法 2: 使用掩码监督（精度更高，需要标注）
# ============================================================================


def quick_start_mask():
    """3 步启用掩码注意力监督"""

    print("\n" + "=" * 60)
    print("快速开始：掩码注意力监督（需要掩码标注）")
    print("=" * 60)

    # 步骤 1: 配置启用注意力监督
    config = ExperimentConfig()
    config.model.vision.attention_type = "cbam"
    config.model.vision.enable_attention_supervision = True
    config.training.use_attention_supervision = True
    config.training.attention_supervision_method = "mask"  # 使用掩码
    config.training.attention_loss_weight = 0.1

    print("\n✓ 步骤 1: 配置完成")
    print(f"  - 监督方法: {config.training.attention_supervision_method}")

    # 步骤 2: 创建模型（与普通模型相同）
    _model = create_fusion_model(
        vision_backbone_name="resnet50",
        vision_config={
            "attention_type": "cbam",
            "enable_attention_supervision": True,
        },
        tabular_input_dim=10,
        num_classes=2,
    )

    print("\n✓ 步骤 2: 模型创建完成")

    # 步骤 3: 训练（数据集需要包含掩码）
    print("\n✓ 步骤 3: 准备训练")
    print("  - 数据集格式: (images, tabular, labels, masks)")
    print("  - masks 形状: (B, 1, H, W)，值在 [0, 1]")
    print("  - 需要在 Dataset.__getitem__() 中返回 4 个元素")

    print("\n示例 Dataset 代码：")
    print("""
    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])
        tabular = self.tabular_data[idx]
        label = self.labels[idx]
        mask = load_mask(self.mask_paths[idx])  # 加载掩码
        return image, tabular, label, mask
    """)

    print("\n" + "=" * 60)
    print("完成！模型将学习关注掩码标记的区域。")
    print("=" * 60)


# ============================================================================
# 配置对比
# ============================================================================


def show_config_comparison():
    """展示不同配置的对比"""

    print("\n" + "=" * 60)
    print("配置对比")
    print("=" * 60)

    configs = {
        "无注意力监督（基线）": {
            "vision.attention_type": "cbam",
            "vision.enable_attention_supervision": False,
            "training.use_attention_supervision": False,
            "数据集要求": "(images, tabular, labels)",
            "优点": "简单，无额外开销",
            "缺点": "注意力不受约束",
        },
        "CAM 自监督": {
            "vision.attention_type": "cbam",
            "vision.enable_attention_supervision": True,
            "training.use_attention_supervision": True,
            "training.attention_supervision_method": "cam",
            "数据集要求": "(images, tabular, labels)",
            "优点": "无需标注，自动生成",
            "缺点": "精度不如掩码监督",
        },
        "掩码监督": {
            "vision.attention_type": "cbam",
            "vision.enable_attention_supervision": True,
            "training.use_attention_supervision": True,
            "training.attention_supervision_method": "mask",
            "数据集要求": "(images, tabular, labels, masks)",
            "优点": "精度最高，直接监督",
            "缺点": "需要人工标注掩码",
        },
    }

    for name, cfg in configs.items():
        print(f"\n【{name}】")
        for key, value in cfg.items():
            print(f"  {key}: {value}")


# ============================================================================
# 常见问题
# ============================================================================


def show_faq():
    """常见问题解答"""

    print("\n" + "=" * 60)
    print("常见问题")
    print("=" * 60)

    faqs = [
        {
            "Q": "为什么必须使用 CBAM？",
            "A": "只有 CBAM 有空间注意力权重可以返回。SE 和 ECA 只做通道注意力。",
        },
        {
            "Q": "CAM 和掩码监督哪个更好？",
            "A": "掩码监督精度更高但需要标注。CAM 无需标注但精度稍低。建议先用 CAM，有标注再用掩码。",
        },
        {
            "Q": "注意力损失权重设置多少合适？",
            "A": "建议 0.05-0.2。CAM 可以稍低（0.05-0.1），掩码可以稍高（0.1-0.2）。",
        },
        {
            "Q": "对性能有影响吗？",
            "A": "训练时增加约 5-10% 内存。推理时如果不调用 return_intermediates，零开销。",
        },
        {
            "Q": "可以用于 Transformer 吗？",
            "A": "不可以。ViT/Swin/MaxViT 不支持外部注意力模块。",
        },
    ]

    for i, faq in enumerate(faqs, 1):
        print(f"\n{i}. {faq['Q']}")
        print(f"   {faq['A']}")


# ============================================================================
# 主函数
# ============================================================================


def main():
    """运行快速开始示例"""

    print("\n" + "=" * 60)
    print("Med-Framework 注意力监督 - 快速开始")
    print("=" * 60)

    # 方法 1: CAM（推荐）
    quick_start_cam()

    # 方法 2: 掩码
    quick_start_mask()

    # 配置对比
    show_config_comparison()

    # 常见问题
    show_faq()

    print("\n" + "=" * 60)
    print("更多信息")
    print("=" * 60)
    print("- 完整示例: examples/attention_supervision_example.py")
    print("- 详细文档: docs/ATTENTION_MECHANISM_GUIDE.md")
    print("- API 参考: med_core/trainers/multimodal.py")


if __name__ == "__main__":
    main()
