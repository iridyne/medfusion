"""
测试梯度检查点功能。

测试内容：
1. 基本功能测试
2. 内存节省验证
3. 不同 backbone 的兼容性
4. 训练/推理模式切换
"""

import pytest
import torch
import torch.nn as nn

from med_core.backbones.swin_2d import SwinTransformer2DBackbone
from med_core.backbones.vision import ResNetBackbone
from med_core.utils.gradient_checkpointing import (
    CheckpointedSequential,
    checkpoint_sequential,
    create_checkpoint_wrapper,
    estimate_memory_savings,
)


class TestGradientCheckpointingUtils:
    """测试梯度检查点工具函数。"""

    def test_checkpoint_sequential(self):
        """测试顺序模块的梯度检查点。"""
        # 创建简单的模块列表
        layers = [
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        ]

        x = torch.randn(4, 128, requires_grad=True)

        # 使用梯度检查点
        output = checkpoint_sequential(
            layers, segments=2, input=x, use_reentrant=False
        )

        # 验证输出形状
        assert output.shape == (4, 128)

        # 验证可以反向传播
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_checkpointed_sequential(self):
        """测试 CheckpointedSequential 容器。"""
        model = CheckpointedSequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            segments=2,
        )

        # 训练模式
        model.train()
        x = torch.randn(2, 64, requires_grad=True)
        output = model(x)
        assert output.shape == (2, 64)

        # 验证反向传播
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

        # 推理模式（不使用检查点）
        model.eval()
        with torch.no_grad():
            output_eval = model(x)
            assert output_eval.shape == (2, 64)

    def test_create_checkpoint_wrapper(self):
        """测试检查点包装器。"""
        layer = nn.Linear(128, 128)
        wrapped_forward = create_checkpoint_wrapper(layer, use_reentrant=False)

        x = torch.randn(4, 128, requires_grad=True)
        output = wrapped_forward(x)

        assert output.shape == (4, 128)

        # 验证反向传播
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestResNetGradientCheckpointing:
    """测试 ResNet 的梯度检查点功能。"""

    @pytest.mark.parametrize("variant", ["resnet18", "resnet50"])
    def test_enable_gradient_checkpointing(self, variant):
        """测试启用梯度检查点。"""
        backbone = ResNetBackbone(
            variant=variant,
            pretrained=False,
            feature_dim=128,
        )

        # 默认未启用
        assert not backbone.is_gradient_checkpointing_enabled()

        # 启用梯度检查点
        backbone.enable_gradient_checkpointing()
        assert backbone.is_gradient_checkpointing_enabled()

        # 禁用梯度检查点
        backbone.disable_gradient_checkpointing()
        assert not backbone.is_gradient_checkpointing_enabled()

    @pytest.mark.parametrize("variant", ["resnet18", "resnet50"])
    def test_forward_with_checkpointing(self, variant):
        """测试启用检查点后的前向传播。"""
        backbone = ResNetBackbone(
            variant=variant,
            pretrained=False,
            feature_dim=128,
        )

        backbone.enable_gradient_checkpointing(segments=4)
        backbone.train()

        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = backbone(x)

        # 验证输出形状
        assert output.shape == (2, 128)

        # 验证反向传播
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_checkpointing_in_eval_mode(self):
        """测试推理模式下不使用检查点。"""
        backbone = ResNetBackbone(
            variant="resnet18",
            pretrained=False,
            feature_dim=128,
        )

        backbone.enable_gradient_checkpointing()
        backbone.eval()

        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output = backbone(x)
            assert output.shape == (2, 128)

    def test_checkpointing_with_attention(self):
        """测试带注意力机制的检查点。"""
        backbone = ResNetBackbone(
            variant="resnet18",
            pretrained=False,
            feature_dim=128,
            attention_type="cbam",
        )

        backbone.enable_gradient_checkpointing()
        backbone.train()

        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = backbone(x)

        assert output.shape == (2, 128)

        # 验证反向传播
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestSwinGradientCheckpointing:
    """测试 Swin Transformer 的梯度检查点功能。"""

    @pytest.mark.parametrize("variant", ["tiny", "small"])
    def test_swin2d_gradient_checkpointing(self, variant):
        """测试 Swin 2D 的梯度检查点。"""
        backbone = SwinTransformer2DBackbone(
            variant=variant,
            in_channels=3,
            feature_dim=128,
            pretrained=False,
        )

        # 启用梯度检查点
        backbone.enable_gradient_checkpointing()
        assert backbone.is_gradient_checkpointing_enabled()

        backbone.train()
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = backbone(x)

        assert output.shape == (2, 128)

        # 验证反向传播
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMemorySavings:
    """测试内存节省效果（需要 CUDA）。"""

    def test_estimate_memory_savings_resnet(self):
        """测试 ResNet 的内存节省估算。"""
        model = ResNetBackbone(
            variant="resnet50",
            pretrained=False,
            feature_dim=128,
        )

        stats = estimate_memory_savings(
            model,
            input_shape=(3, 224, 224),
            device="cuda",
        )

        # 验证返回的统计信息
        assert "without_checkpoint" in stats
        assert "with_checkpoint" in stats
        assert "savings" in stats
        assert "savings_percent" in stats

        # 应该有内存节省
        assert stats["savings"] >= 0
        assert stats["savings_percent"] >= 0

        print("\nResNet50 Memory Savings:")
        print(f"  Without checkpoint: {stats['without_checkpoint']:.2f} MB")
        print(f"  With checkpoint: {stats['with_checkpoint']:.2f} MB")
        print(f"  Savings: {stats['savings']:.2f} MB ({stats['savings_percent']:.1f}%)")


class TestIntegration:
    """集成测试。"""

    def test_full_training_loop(self):
        """测试完整的训练循环。"""
        # 创建模型
        model = ResNetBackbone(
            variant="resnet18",
            pretrained=False,
            feature_dim=10,
        )

        # 启用梯度检查点
        model.enable_gradient_checkpointing()
        model.train()

        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 模拟训练步骤
        for _ in range(3):
            x = torch.randn(4, 3, 224, 224)
            target = torch.randint(0, 10, (4,))

            # 前向传播
            output = model(x)

            # 计算损失
            loss = nn.functional.cross_entropy(output, target)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 验证梯度存在
            for param in model.parameters():
                if param.requires_grad:
                    assert param.grad is not None

    def test_checkpointing_with_frozen_layers(self):
        """测试冻结层与梯度检查点的兼容性。"""
        model = ResNetBackbone(
            variant="resnet18",
            pretrained=False,
            freeze=True,
            feature_dim=128,
        )

        # 启用梯度检查点
        model.enable_gradient_checkpointing()
        model.train()

        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = model(x)

        assert output.shape == (2, 128)

        # 验证反向传播
        loss = output.sum()
        loss.backward()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
