"""
测试高级注意力模块

测试 SE、ECA、Transformer 等注意力机制。
"""

import pytest
import torch
import torch.nn as nn

from med_core.attention_supervision import (
    SEAttention,
    ECAAttention,
    SpatialAttention,
    CBAM,
    MultiHeadSelfAttention,
    TransformerAttention2D,
    create_attention_module,
    ChannelAttentionSupervision,
    SpatialAttentionSupervision,
    TransformerAttentionSupervision,
    HybridAttentionSupervision,
    create_attention_supervision,
)


class TestSEAttention:
    """测试 SE 注意力"""
    
    def test_forward(self):
        """测试前向传播"""
        se = SEAttention(channels=256, reduction=16)
        x = torch.randn(2, 256, 14, 14)
        out = se(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
    
    def test_get_attention_weights(self):
        """测试获取注意力权重"""
        se = SEAttention(channels=256, reduction=16)
        x = torch.randn(2, 256, 14, 14)
        weights = se.get_attention_weights(x)
        
        assert weights.shape == (2, 256)
        assert (weights >= 0).all() and (weights <= 1).all()
    
    def test_different_reductions(self):
        """测试不同的降维比例"""
        for reduction in [4, 8, 16, 32]:
            se = SEAttention(channels=256, reduction=reduction)
            x = torch.randn(2, 256, 14, 14)
            out = se(x)
            assert out.shape == x.shape


class TestECAAttention:
    """测试 ECA 注意力"""
    
    def test_forward(self):
        """测试前向传播"""
        eca = ECAAttention(channels=256)
        x = torch.randn(2, 256, 14, 14)
        out = eca(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
    
    def test_auto_kernel_size(self):
        """测试自动计算卷积核大小"""
        for channels in [64, 128, 256, 512]:
            eca = ECAAttention(channels=channels)
            assert eca.kernel_size % 2 == 1  # 奇数
            assert eca.kernel_size > 0
    
    def test_manual_kernel_size(self):
        """测试手动指定卷积核大小"""
        eca = ECAAttention(channels=256, kernel_size=5)
        assert eca.kernel_size == 5
        
        x = torch.randn(2, 256, 14, 14)
        out = eca(x)
        assert out.shape == x.shape


class TestSpatialAttention:
    """测试空间注意力"""
    
    def test_forward(self):
        """测试前向传播"""
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 256, 14, 14)
        out = sa(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
    
    def test_get_attention_weights(self):
        """测试获取注意力权重"""
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 256, 14, 14)
        weights = sa.get_attention_weights(x)
        
        assert weights.shape == (2, 1, 14, 14)
        assert (weights >= 0).all() and (weights <= 1).all()


class TestCBAM:
    """测试 CBAM"""
    
    def test_forward(self):
        """测试前向传播"""
        cbam = CBAM(channels=256, reduction=16, spatial_kernel=7)
        x = torch.randn(2, 256, 14, 14)
        out = cbam(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
    
    def test_components(self):
        """测试组件"""
        cbam = CBAM(channels=256)
        
        assert hasattr(cbam, "channel_attention")
        assert hasattr(cbam, "spatial_attention")
        assert isinstance(cbam.channel_attention, SEAttention)
        assert isinstance(cbam.spatial_attention, SpatialAttention)


class TestMultiHeadSelfAttention:
    """测试多头自注意力"""
    
    def test_forward(self):
        """测试前向传播"""
        mhsa = MultiHeadSelfAttention(dim=256, num_heads=8)
        x = torch.randn(2, 196, 256)
        out = mhsa(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
    
    def test_get_attention_weights(self):
        """测试获取注意力权重"""
        mhsa = MultiHeadSelfAttention(dim=256, num_heads=8)
        x = torch.randn(2, 196, 256)
        weights = mhsa.get_attention_weights(x)
        
        assert weights.shape == (2, 8, 196, 196)
        # 检查是否归一化
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
    
    def test_different_heads(self):
        """测试不同的头数"""
        for num_heads in [1, 2, 4, 8, 16]:
            mhsa = MultiHeadSelfAttention(dim=256, num_heads=num_heads)
            x = torch.randn(2, 196, 256)
            out = mhsa(x)
            assert out.shape == x.shape


class TestTransformerAttention2D:
    """测试 2D Transformer 注意力"""
    
    def test_forward(self):
        """测试前向传播"""
        ta = TransformerAttention2D(channels=256, num_heads=8)
        x = torch.randn(2, 256, 14, 14)
        out = ta(x)
        
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
    
    def test_get_attention_weights(self):
        """测试获取注意力权重"""
        ta = TransformerAttention2D(channels=256, num_heads=8)
        x = torch.randn(2, 256, 14, 14)
        weights = ta.get_attention_weights(x)
        
        assert weights.shape == (2, 8, 196, 196)


class TestCreateAttentionModule:
    """测试注意力模块工厂函数"""
    
    def test_create_se(self):
        """测试创建 SE"""
        attn = create_attention_module("se", channels=256, reduction=16)
        assert isinstance(attn, SEAttention)
    
    def test_create_eca(self):
        """测试创建 ECA"""
        attn = create_attention_module("eca", channels=256)
        assert isinstance(attn, ECAAttention)
    
    def test_create_spatial(self):
        """测试创建空间注意力"""
        attn = create_attention_module("spatial", channels=256, kernel_size=7)
        assert isinstance(attn, SpatialAttention)
    
    def test_create_cbam(self):
        """测试创建 CBAM"""
        attn = create_attention_module("cbam", channels=256)
        assert isinstance(attn, CBAM)
    
    def test_create_transformer(self):
        """测试创建 Transformer"""
        attn = create_attention_module("transformer", channels=256, num_heads=8)
        assert isinstance(attn, TransformerAttention2D)
    
    def test_invalid_type(self):
        """测试无效类型"""
        with pytest.raises(ValueError):
            create_attention_module("invalid", channels=256)


class TestChannelAttentionSupervision:
    """测试通道注意力监督"""
    
    def test_forward(self):
        """测试前向传播"""
        supervision = ChannelAttentionSupervision(loss_weight=0.1)
        
        channel_weights = torch.sigmoid(torch.randn(2, 256))
        features = torch.randn(2, 256, 14, 14)
        
        loss = supervision(channel_weights, features)
        
        assert loss.total_loss.item() >= 0
        assert isinstance(loss.components, dict)
    
    def test_with_target_channels(self):
        """测试带目标通道"""
        supervision = ChannelAttentionSupervision(
            loss_weight=0.1,
            target_channels=[0, 1, 2],
        )
        
        channel_weights = torch.sigmoid(torch.randn(2, 256))
        features = torch.randn(2, 256, 14, 14)
        
        loss = supervision(channel_weights, features)
        
        assert "target_loss" in loss.components
    
    def test_diversity_loss(self):
        """测试多样性损失"""
        supervision = ChannelAttentionSupervision(
            loss_weight=0.1,
            diversity_weight=0.1,
        )
        
        channel_weights = torch.sigmoid(torch.randn(2, 256))
        features = torch.randn(2, 256, 14, 14)
        
        loss = supervision(channel_weights, features)
        
        assert "diversity_loss" in loss.components


class TestSpatialAttentionSupervision:
    """测试空间注意力监督"""
    
    def test_forward(self):
        """测试前向传播"""
        supervision = SpatialAttentionSupervision(loss_weight=0.1)
        
        spatial_weights = torch.sigmoid(torch.randn(2, 1, 14, 14))
        features = torch.randn(2, 256, 14, 14)
        
        loss = supervision(spatial_weights, features)
        
        assert loss.total_loss.item() >= 0
        assert isinstance(loss.components, dict)
    
    def test_with_targets(self):
        """测试带目标掩码"""
        supervision = SpatialAttentionSupervision(loss_weight=0.1)
        
        spatial_weights = torch.sigmoid(torch.randn(2, 1, 14, 14))
        features = torch.randn(2, 256, 14, 14)
        targets = torch.randint(0, 2, (2, 1, 14, 14)).float()
        
        loss = supervision(spatial_weights, features, targets)
        
        assert "target_loss" in loss.components


class TestTransformerAttentionSupervision:
    """测试 Transformer 注意力监督"""
    
    def test_forward(self):
        """测试前向传播"""
        supervision = TransformerAttentionSupervision(loss_weight=0.1)
        
        transformer_weights = torch.softmax(torch.randn(2, 8, 196, 196), dim=-1)
        features = torch.randn(2, 196, 256)
        
        loss = supervision(transformer_weights, features)
        
        assert loss.total_loss.item() >= 0
        assert isinstance(loss.components, dict)
    
    def test_head_diversity(self):
        """测试头多样性损失"""
        supervision = TransformerAttentionSupervision(
            loss_weight=0.1,
            head_diversity_weight=0.1,
        )
        
        transformer_weights = torch.softmax(torch.randn(2, 8, 196, 196), dim=-1)
        features = torch.randn(2, 196, 256)
        
        loss = supervision(transformer_weights, features)
        
        assert "head_diversity_loss" in loss.components


class TestHybridAttentionSupervision:
    """测试混合注意力监督"""
    
    def test_forward(self):
        """测试前向传播"""
        supervision = HybridAttentionSupervision(loss_weight=0.1)
        
        attentions = {
            "channel": torch.sigmoid(torch.randn(2, 256)),
            "spatial": torch.sigmoid(torch.randn(2, 1, 14, 14)),
        }
        features = torch.randn(2, 256, 14, 14)
        
        loss = supervision(attentions, features)
        
        assert loss.total_loss.item() >= 0
        assert isinstance(loss.components, dict)
    
    def test_all_attention_types(self):
        """测试所有注意力类型"""
        supervision = HybridAttentionSupervision(loss_weight=0.1)
        
        attentions = {
            "channel": torch.sigmoid(torch.randn(2, 256)),
            "spatial": torch.sigmoid(torch.randn(2, 1, 14, 14)),
            "transformer": torch.softmax(torch.randn(2, 8, 196, 196), dim=-1),
        }
        features = torch.randn(2, 256, 14, 14)
        
        loss = supervision(attentions, features)
        
        # 检查所有组件
        assert any("channel_" in k for k in loss.components.keys())
        assert any("spatial_" in k for k in loss.components.keys())
        assert any("transformer_" in k for k in loss.components.keys())


class TestCreateAttentionSupervision:
    """测试注意力监督工厂函数"""
    
    def test_create_channel(self):
        """测试创建通道监督"""
        sup = create_attention_supervision("channel", loss_weight=0.1)
        assert isinstance(sup, ChannelAttentionSupervision)
    
    def test_create_spatial(self):
        """测试创建空间监督"""
        sup = create_attention_supervision("spatial", loss_weight=0.1)
        assert isinstance(sup, SpatialAttentionSupervision)
    
    def test_create_transformer(self):
        """测试创建 Transformer 监督"""
        sup = create_attention_supervision("transformer", loss_weight=0.1)
        assert isinstance(sup, TransformerAttentionSupervision)
    
    def test_create_hybrid(self):
        """测试创建混合监督"""
        sup = create_attention_supervision("hybrid", loss_weight=0.1)
        assert isinstance(sup, HybridAttentionSupervision)
    
    def test_invalid_type(self):
        """测试无效类型"""
        with pytest.raises(ValueError):
            create_attention_supervision("invalid", loss_weight=0.1)


class TestIntegration:
    """集成测试"""
    
    def test_se_with_supervision(self):
        """测试 SE + 监督"""
        # 创建模块
        se = SEAttention(channels=256, reduction=16)
        supervision = ChannelAttentionSupervision(loss_weight=0.1)
        
        # 前向传播
        x = torch.randn(2, 256, 14, 14)
        out = se(x)
        
        # 获取注意力权重
        weights = se.get_attention_weights(x)
        
        # 计算监督损失
        loss = supervision(weights, x)
        
        assert out.shape == x.shape
        assert loss.total_loss.item() >= 0
    
    def test_cbam_with_supervision(self):
        """测试 CBAM + 监督"""
        # 创建模块
        cbam = CBAM(channels=256)
        supervision = HybridAttentionSupervision(loss_weight=0.1)
        
        # 前向传播
        x = torch.randn(2, 256, 14, 14)
        out = cbam(x)
        
        # 获取注意力权重
        channel_weights = cbam.channel_attention.get_attention_weights(x)
        spatial_weights = cbam.spatial_attention.get_attention_weights(out)
        
        attentions = {
            "channel": channel_weights,
            "spatial": spatial_weights,
        }
        
        # 计算监督损失
        loss = supervision(attentions, x)
        
        assert out.shape == x.shape
        assert loss.total_loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
