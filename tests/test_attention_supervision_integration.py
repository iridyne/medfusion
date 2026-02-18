"""
注意力监督集成测试

测试注意力监督功能的端到端集成。
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from med_core.backbones import create_vision_backbone
from med_core.configs import ExperimentConfig
from med_core.fusion import create_fusion_model
from med_core.trainers import MultimodalTrainer


class TestAttentionSupervisionIntegration:
    """注意力监督集成测试"""

    @pytest.fixture
    def mock_data(self):
        """创建模拟数据"""
        batch_size = 8
        images = torch.randn(batch_size, 3, 224, 224)
        tabular = torch.randn(batch_size, 10)
        labels = torch.randint(0, 2, (batch_size,))
        masks = torch.rand(batch_size, 1, 224, 224)  # 模拟掩码
        return images, tabular, labels, masks

    @pytest.fixture
    def config_mask(self):
        """创建 Mask 监督配置"""
        config = ExperimentConfig()
        config.model.vision.backbone = "resnet18"
        config.model.vision.attention_type = "cbam"
        config.model.vision.enable_attention_supervision = True
        config.training.num_epochs = 2
        config.training.use_attention_supervision = True
        config.training.attention_supervision_method = "mask"
        config.training.attention_loss_weight = 0.1
        return config

    @pytest.fixture
    def config_cam(self):
        """创建 CAM 监督配置"""
        config = ExperimentConfig()
        config.model.vision.backbone = "resnet18"
        config.model.vision.attention_type = "cbam"
        config.model.vision.enable_attention_supervision = True
        config.training.num_epochs = 2
        config.training.use_attention_supervision = True
        config.training.attention_supervision_method = "cam"
        config.training.attention_loss_weight = 0.1
        return config

    def test_backbone_returns_intermediates(self):
        """测试 Backbone 返回中间结果"""
        backbone = create_vision_backbone(
            backbone_name="resnet18",
            attention_type="cbam",
            pretrained=False,
        )

        # 启用注意力监督
        backbone.enable_attention_supervision = True
        backbone._attention.return_attention_weights = True

        images = torch.randn(2, 3, 224, 224)

        # 测试返回中间结果
        outputs = backbone(images, return_intermediates=True)

        assert isinstance(outputs, dict)
        assert "features" in outputs
        assert "feature_maps" in outputs
        assert "attention_weights" in outputs

        # 检查形状
        assert outputs["features"].shape == (2, 128)  # feature_dim=128
        assert outputs["feature_maps"].shape[0] == 2
        assert outputs["attention_weights"] is not None

    def test_cbam_returns_weights(self):
        """测试 CBAM 返回注意力权重"""
        from med_core.backbones.attention import CBAM

        cbam = CBAM(
            in_channels=512,
            return_attention_weights=True,
        )

        x = torch.randn(2, 512, 7, 7)
        output, weights_dict = cbam(x)

        assert isinstance(weights_dict, dict)
        assert "channel_weights" in weights_dict
        assert "spatial_weights" in weights_dict
        assert weights_dict["spatial_weights"] is not None
        assert weights_dict["spatial_weights"].shape == (2, 1, 7, 7)

    def test_model_with_attention_supervision(self, config_mask):
        """测试模型支持注意力监督"""
        model = create_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            fusion_type="concatenate",
            num_classes=2,
            config=config_mask.model,
        )

        images = torch.randn(2, 3, 224, 224)
        tabular = torch.randn(2, 10)

        # 测试正常前向传播
        outputs = model(images, tabular)
        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 2)

        # 测试返回中间结果
        if hasattr(model, "vision_backbone"):
            vision_outputs = model.vision_backbone(images, return_intermediates=True)
            assert isinstance(vision_outputs, dict)
            assert "attention_weights" in vision_outputs

    def test_trainer_mask_supervision(self, config_mask, mock_data):
        """测试训练器 Mask 监督"""
        images, tabular, labels, masks = mock_data

        # 创建数据集（包含掩码）
        dataset = TensorDataset(images, tabular, labels, masks)
        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)

        # 创建模型
        model = create_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            fusion_type="concatenate",
            num_classes=2,
            config=config_mask.model,
        )

        # 创建训练器
        trainer = MultimodalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config_mask,
            device="cpu",
        )

        # 验证配置
        assert trainer.use_attention_supervision is True
        assert trainer.attention_supervision_method == "mask"
        assert trainer.attention_loss_weight == 0.1

        # 测试训练步骤
        batch = next(iter(train_loader))
        metrics = trainer.training_step(batch, 0)

        assert "loss" in metrics
        assert metrics["loss"].requires_grad

    def test_trainer_cam_supervision(self, config_cam, mock_data):
        """测试训练器 CAM 监督"""
        images, tabular, labels, _ = mock_data

        # 创建数据集（不包含掩码）
        dataset = TensorDataset(images, tabular, labels)
        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)

        # 创建模型
        model = create_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            fusion_type="concatenate",
            num_classes=2,
            config=config_cam.model,
        )

        # 创建训练器
        trainer = MultimodalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config_cam,
            device="cpu",
        )

        # 验证配置
        assert trainer.use_attention_supervision is True
        assert trainer.attention_supervision_method == "cam"

        # 测试训练步骤
        batch = next(iter(train_loader))
        metrics = trainer.training_step(batch, 0)

        assert "loss" in metrics
        assert metrics["loss"].requires_grad

    def test_cam_generation(self, config_cam):
        """测试 CAM 生成"""
        model = create_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            fusion_type="concatenate",
            num_classes=2,
            config=config_cam.model,
        )

        # 创建训练器
        images = torch.randn(2, 3, 224, 224)
        tabular = torch.randn(2, 10)
        labels = torch.tensor([0, 1])

        dataset = TensorDataset(images, tabular, labels)
        train_loader = DataLoader(dataset, batch_size=2)
        val_loader = DataLoader(dataset, batch_size=2)

        trainer = MultimodalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config_cam,
            device="cpu",
        )

        # 测试 CAM 生成
        if hasattr(model, "vision_backbone"):
            vision_outputs = model.vision_backbone(images, return_intermediates=True)
            feature_maps = vision_outputs.get("feature_maps")

            if feature_maps is not None:
                cam = trainer._generate_cam(feature_maps, labels)

                if cam is not None:
                    assert cam.shape[0] == 2
                    assert cam.shape[1] == 1
                    assert cam.min() >= 0
                    assert cam.max() <= 1

    def test_attention_loss_computation(self, config_mask):
        """测试注意力损失计算"""
        model = create_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            fusion_type="concatenate",
            num_classes=2,
            config=config_mask.model,
        )

        images = torch.randn(2, 3, 224, 224)
        tabular = torch.randn(2, 10)
        labels = torch.tensor([0, 1])
        masks = torch.rand(2, 1, 224, 224)

        dataset = TensorDataset(images, tabular, labels, masks)
        train_loader = DataLoader(dataset, batch_size=2)
        val_loader = DataLoader(dataset, batch_size=2)

        trainer = MultimodalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config_mask,
            device="cpu",
        )

        # 获取注意力权重
        if hasattr(model, "vision_backbone"):
            vision_outputs = model.vision_backbone(images, return_intermediates=True)
            attention_weights = vision_outputs.get("attention_weights")
            feature_maps = vision_outputs.get("feature_maps")

            if attention_weights is not None:
                # 计算注意力损失
                attention_loss = trainer._compute_attention_loss(
                    attention_weights=attention_weights,
                    feature_maps=feature_maps,
                    labels=labels,
                    masks=masks,
                )

                if attention_loss is not None:
                    assert isinstance(attention_loss, torch.Tensor)
                    assert attention_loss.requires_grad

    def test_config_validation(self):
        """测试配置验证"""
        config = ExperimentConfig()

        # 测试：启用注意力监督但未启用 vision 配置
        config.model.vision.attention_type = "cbam"
        config.model.vision.enable_attention_supervision = False  # 未启用
        config.training.use_attention_supervision = True  # 启用

        model = create_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            config=config.model,
        )

        dataset = TensorDataset(
            torch.randn(4, 3, 224, 224),
            torch.randn(4, 10),
            torch.tensor([0, 1, 0, 1]),
        )
        loader = DataLoader(dataset, batch_size=2)

        # 应该发出警告但不报错
        trainer = MultimodalTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=config,
            device="cpu",
        )

        # 验证警告后的状态
        assert trainer.use_attention_supervision is False  # 应该被禁用

    def test_se_attention_not_supported(self):
        """测试 SE 注意力不支持监督"""
        config = ExperimentConfig()
        config.model.vision.attention_type = "se"  # SE 不支持
        config.model.vision.enable_attention_supervision = True
        config.training.use_attention_supervision = True
        config.training.attention_supervision_method = "mask"

        model = create_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            config=config.model,
        )

        dataset = TensorDataset(
            torch.randn(4, 3, 224, 224),
            torch.randn(4, 10),
            torch.tensor([0, 1, 0, 1]),
        )
        loader = DataLoader(dataset, batch_size=2)

        # 应该发出警告
        trainer = MultimodalTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=config,
            device="cpu",
        )

        # SE 不支持，应该被禁用
        assert trainer.use_attention_supervision is False

    def test_backward_compatibility(self):
        """测试向后兼容性"""
        # 不启用注意力监督的情况
        config = ExperimentConfig()
        config.model.vision.attention_type = "cbam"
        config.model.vision.enable_attention_supervision = False
        config.training.use_attention_supervision = False

        model = create_fusion_model(
            vision_backbone_name="resnet18",
            tabular_input_dim=10,
            num_classes=2,
            config=config.model,
        )

        images = torch.randn(2, 3, 224, 224)
        tabular = torch.randn(2, 10)

        # 应该正常工作
        outputs = model(images, tabular)
        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
