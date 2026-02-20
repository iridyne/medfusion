"""
高级注意力模块使用示例

演示如何使用 SE、ECA、Transformer 等注意力机制。
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn


def demo_se_attention():
    """演示 SE 注意力"""
    print("=" * 60)
    print("SE (Squeeze-and-Excitation) 注意力演示")
    print("=" * 60)
    
    from med_core.attention_supervision import SEAttention
    
    # 创建 SE 模块
    se = SEAttention(channels=256, reduction=16)
    
    # 输入特征
    x = torch.randn(2, 256, 14, 14)
    print(f"\n输入形状: {x.shape}")
    
    # 应用 SE 注意力
    out = se(x)
    print(f"输出形状: {out.shape}")
    
    # 获取注意力权重
    weights = se.get_attention_weights(x)
    print(f"通道注意力权重形状: {weights.shape}")
    print(f"权重范围: [{weights.min():.3f}, {weights.max():.3f}]")
    
    print("\n特点:")
    print("  • 通道注意力机制")
    print("  • 全局平均池化 + 两层全连接")
    print("  • 参数量小，计算高效")
    print("  • 适合增强重要通道")


def demo_eca_attention():
    """演示 ECA 注意力"""
    print("\n" + "=" * 60)
    print("ECA (Efficient Channel Attention) 注意力演示")
    print("=" * 60)
    
    from med_core.attention_supervision import ECAAttention
    
    # 创建 ECA 模块
    eca = ECAAttention(channels=256)
    
    # 输入特征
    x = torch.randn(2, 256, 14, 14)
    print(f"\n输入形状: {x.shape}")
    
    # 应用 ECA 注意力
    out = eca(x)
    print(f"输出形状: {out.shape}")
    
    # 获取注意力权重
    weights = eca.get_attention_weights(x)
    print(f"通道注意力权重形状: {weights.shape}")
    
    print(f"\n自动计算的卷积核大小: {eca.kernel_size}")
    
    print("\n特点:")
    print("  • 高效的通道注意力")
    print("  • 使用 1D 卷积，避免降维")
    print("  • 参数量更少")
    print("  • 性能优于 SE")


def demo_spatial_attention():
    """演示空间注意力"""
    print("\n" + "=" * 60)
    print("空间注意力演示")
    print("=" * 60)
    
    from med_core.attention_supervision import SpatialAttention
    
    # 创建空间注意力模块
    sa = SpatialAttention(kernel_size=7)
    
    # 输入特征
    x = torch.randn(2, 256, 14, 14)
    print(f"\n输入形状: {x.shape}")
    
    # 应用空间注意力
    out = sa(x)
    print(f"输出形状: {out.shape}")
    
    # 获取注意力权重
    weights = sa.get_attention_weights(x)
    print(f"空间注意力权重形状: {weights.shape}")
    
    print("\n特点:")
    print("  • 空间维度的注意力")
    print("  • 使用平均池化和最大池化")
    print("  • 关注重要的空间位置")
    print("  • 适合目标定位")


def demo_cbam():
    """演示 CBAM"""
    print("\n" + "=" * 60)
    print("CBAM (Convolutional Block Attention Module) 演示")
    print("=" * 60)
    
    from med_core.attention_supervision import CBAM
    
    # 创建 CBAM 模块
    cbam = CBAM(channels=256, reduction=16, spatial_kernel=7)
    
    # 输入特征
    x = torch.randn(2, 256, 14, 14)
    print(f"\n输入形状: {x.shape}")
    
    # 应用 CBAM
    out = cbam(x)
    print(f"输出形状: {out.shape}")
    
    print("\n特点:")
    print("  • 结合通道和空间注意力")
    print("  • 先通道后空间")
    print("  • 性能强大")
    print("  • 广泛应用于各种任务")


def demo_transformer_attention():
    """演示 Transformer 注意力"""
    print("\n" + "=" * 60)
    print("Transformer 注意力演示")
    print("=" * 60)
    
    from med_core.attention_supervision import TransformerAttention2D
    
    # 创建 Transformer 注意力模块
    ta = TransformerAttention2D(channels=256, num_heads=8)
    
    # 输入特征
    x = torch.randn(2, 256, 14, 14)
    print(f"\n输入形状: {x.shape}")
    
    # 应用 Transformer 注意力
    out = ta(x)
    print(f"输出形状: {out.shape}")
    
    # 获取注意力权重
    weights = ta.get_attention_weights(x)
    print(f"注意力权重形状: {weights.shape}")
    print(f"  (B, num_heads, N, N) = (batch, 头数, 序列长度, 序列长度)")
    
    print("\n特点:")
    print("  • 多头自注意力机制")
    print("  • 全局感受野")
    print("  • 捕获长距离依赖")
    print("  • 计算复杂度 O(N²)")


def demo_factory_function():
    """演示工厂函数"""
    print("\n" + "=" * 60)
    print("工厂函数演示")
    print("=" * 60)
    
    from med_core.attention_supervision import create_attention_module
    
    print("\n支持的注意力类型:")
    attention_types = ["se", "eca", "spatial", "cbam", "transformer"]
    
    for attn_type in attention_types:
        attn = create_attention_module(attn_type, channels=256)
        x = torch.randn(2, 256, 14, 14)
        out = attn(x)
        print(f"  • {attn_type:12s}: {x.shape} -> {out.shape}")
    
    print("\n使用示例:")
    code = '''
from med_core.attention_supervision import create_attention_module

# 创建注意力模块
attention = create_attention_module(
    attention_type="se",
    channels=256,
    reduction=16,
)

# 在模型中使用
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 256, 3, padding=1)
        self.attention = create_attention_module("se", channels=256)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.attention(x)  # 应用注意力
        return x
'''
    print(code)


def demo_attention_supervision():
    """演示注意力监督"""
    print("\n" + "=" * 60)
    print("注意力监督演示")
    print("=" * 60)
    
    from med_core.attention_supervision import (
        ChannelAttentionSupervision,
        SpatialAttentionSupervision,
        TransformerAttentionSupervision,
    )
    
    print("\n1. 通道注意力监督:")
    channel_sup = ChannelAttentionSupervision(
        loss_weight=0.1,
        diversity_weight=0.1,
        sparsity_weight=0.1,
    )
    
    channel_weights = torch.sigmoid(torch.randn(2, 256))
    features = torch.randn(2, 256, 14, 14)
    
    loss = channel_sup(channel_weights, features)
    print(f"   总损失: {loss.total_loss.item():.4f}")
    print(f"   损失组件: {list(loss.components.keys())}")
    
    print("\n2. 空间注意力监督:")
    spatial_sup = SpatialAttentionSupervision(
        loss_weight=0.1,
        consistency_weight=0.1,
        smoothness_weight=0.1,
    )
    
    spatial_weights = torch.sigmoid(torch.randn(2, 1, 14, 14))
    
    loss = spatial_sup(spatial_weights, features)
    print(f"   总损失: {loss.total_loss.item():.4f}")
    print(f"   损失组件: {list(loss.components.keys())}")
    
    print("\n3. Transformer 注意力监督:")
    transformer_sup = TransformerAttentionSupervision(
        loss_weight=0.1,
        head_diversity_weight=0.1,
        locality_weight=0.1,
    )
    
    transformer_weights = torch.softmax(torch.randn(2, 8, 196, 196), dim=-1)
    features_seq = torch.randn(2, 196, 256)
    
    loss = transformer_sup(transformer_weights, features_seq)
    print(f"   总损失: {loss.total_loss.item():.4f}")
    print(f"   损失组件: {list(loss.components.keys())}")


def demo_integration_example():
    """演示集成示例"""
    print("\n" + "=" * 60)
    print("集成示例")
    print("=" * 60)
    
    print("\n完整的模型集成示例:")
    code = '''
import torch.nn as nn
from med_core.attention_supervision import (
    SEAttention,
    ChannelAttentionSupervision,
)

class AttentionEnhancedModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 骨干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.ReLU(),
        )
        
        # SE 注意力
        self.attention = SEAttention(channels=256, reduction=16)
        
        # 注意力监督
        self.attention_supervision = ChannelAttentionSupervision(
            loss_weight=0.1,
            diversity_weight=0.1,
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 10),
        )
    
    def forward(self, x, return_attention=False):
        # 特征提取
        features = self.backbone(x)
        
        # 应用注意力
        attended_features = self.attention(features)
        
        # 分类
        logits = self.classifier(attended_features)
        
        if return_attention:
            # 获取注意力权重
            attn_weights = self.attention.get_attention_weights(features)
            return logits, attn_weights
        
        return logits
    
    def compute_loss(self, x, y):
        # 前向传播
        logits, attn_weights = self.forward(x, return_attention=True)
        
        # 分类损失
        cls_loss = nn.CrossEntropyLoss()(logits, y)
        
        # 注意力监督损失
        attn_loss = self.attention_supervision(
            attn_weights,
            self.backbone(x),
        )
        
        # 总损失
        total_loss = cls_loss + attn_loss.total_loss
        
        return total_loss, {
            "cls_loss": cls_loss.item(),
            "attn_loss": attn_loss.total_loss.item(),
            **{k: v.item() for k, v in attn_loss.components.items()},
        }

# 使用
model = AttentionEnhancedModel()
x = torch.randn(2, 3, 224, 224)
y = torch.randint(0, 10, (2,))

loss, loss_dict = model.compute_loss(x, y)
print(f"Total loss: {loss.item():.4f}")
print(f"Loss components: {loss_dict}")
'''
    print(code)


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("MedFusion 高级注意力模块演示")
    print("=" * 60)
    
    try:
        # 演示 1: SE 注意力
        demo_se_attention()
        
        # 演示 2: ECA 注意力
        demo_eca_attention()
        
        # 演示 3: 空间注意力
        demo_spatial_attention()
        
        # 演示 4: CBAM
        demo_cbam()
        
        # 演示 5: Transformer 注意力
        demo_transformer_attention()
        
        # 演示 6: 工厂函数
        demo_factory_function()
        
        # 演示 7: 注意力监督
        demo_attention_supervision()
        
        # 演示 8: 集成示例
        demo_integration_example()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
        print("\n💡 关键要点:")
        print("  1. SE/ECA 适合通道注意力")
        print("  2. 空间注意力适合目标定位")
        print("  3. CBAM 结合两者优势")
        print("  4. Transformer 适合全局建模")
        print("  5. 注意力监督提高可解释性")
        
        print("\n📖 相关资源:")
        print("  • med_core/attention_supervision/advanced_attention.py")
        print("  • med_core/attention_supervision/advanced_supervision.py")
        print("  • examples/advanced_attention_demo.py")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
