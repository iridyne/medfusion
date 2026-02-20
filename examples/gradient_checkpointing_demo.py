"""
梯度检查点演示脚本。

展示如何使用梯度检查点来降低训练时的内存占用。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from med_core.backbones.vision import ResNetBackbone
from med_core.utils.gradient_checkpointing import estimate_memory_savings


def create_dummy_dataloader(batch_size: int = 8, num_batches: int = 10):
    """创建模拟数据加载器。"""
    images = torch.randn(batch_size * num_batches, 3, 224, 224)
    labels = torch.randint(0, 10, (batch_size * num_batches,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_without_checkpointing():
    """不使用梯度检查点的训练。"""
    print("\n" + "=" * 60)
    print("训练模式 1: 不使用梯度检查点")
    print("=" * 60)

    # 创建模型
    model = ResNetBackbone(
        variant="resnet50",
        pretrained=False,
        feature_dim=10,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # 设置训练
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 创建数据
    train_loader = create_dummy_dataloader(batch_size=8, num_batches=5)

    # 训练一个 epoch
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 打印内存使用
        if device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"Batch {batch_idx + 1}: Loss={loss.item():.4f}, "
                  f"Memory: {memory_allocated:.2f}MB / {memory_reserved:.2f}MB")
        else:
            print(f"Batch {batch_idx + 1}: Loss={loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"\n平均损失: {avg_loss:.4f}")

    if device.type == "cuda":
        max_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"峰值内存: {max_memory:.2f}MB")

    return max_memory if device.type == "cuda" else 0


def train_with_checkpointing():
    """使用梯度检查点的训练。"""
    print("\n" + "=" * 60)
    print("训练模式 2: 使用梯度检查点")
    print("=" * 60)

    # 创建模型
    model = ResNetBackbone(
        variant="resnet50",
        pretrained=False,
        feature_dim=10,
    )

    # 启用梯度检查点
    model.enable_gradient_checkpointing(segments=4)
    print("✓ 梯度检查点已启用 (segments=4)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # 重置内存统计
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 设置训练
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 创建数据
    train_loader = create_dummy_dataloader(batch_size=8, num_batches=5)

    # 训练一个 epoch
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播（使用梯度检查点）
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 打印内存使用
        if device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"Batch {batch_idx + 1}: Loss={loss.item():.4f}, "
                  f"Memory: {memory_allocated:.2f}MB / {memory_reserved:.2f}MB")
        else:
            print(f"Batch {batch_idx + 1}: Loss={loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"\n平均损失: {avg_loss:.4f}")

    if device.type == "cuda":
        max_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"峰值内存: {max_memory:.2f}MB")

    return max_memory if device.type == "cuda" else 0


def compare_memory_usage():
    """比较内存使用情况。"""
    print("\n" + "=" * 60)
    print("内存使用对比")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，跳过内存对比")
        return

    # 训练不使用检查点
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    memory_without = train_without_checkpointing()

    # 训练使用检查点
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    memory_with = train_with_checkpointing()

    # 打印对比
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"不使用检查点: {memory_without:.2f}MB")
    print(f"使用检查点:   {memory_with:.2f}MB")
    savings = memory_without - memory_with
    savings_percent = (savings / memory_without * 100) if memory_without > 0 else 0
    print(f"节省内存:     {savings:.2f}MB ({savings_percent:.1f}%)")


def demo_estimate_memory_savings():
    """演示内存节省估算。"""
    print("\n" + "=" * 60)
    print("内存节省估算")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，跳过内存估算")
        return

    model = ResNetBackbone(
        variant="resnet50",
        pretrained=False,
        feature_dim=128,
    )

    print("正在估算内存节省...")
    stats = estimate_memory_savings(
        model=model,
        input_shape=(3, 224, 224),
        device="cuda",
    )

    print(f"\n不使用检查点: {stats['without_checkpoint']:.2f}MB")
    print(f"使用检查点:   {stats['with_checkpoint']:.2f}MB")
    print(f"节省内存:     {stats['savings']:.2f}MB ({stats['savings_percent']:.1f}%)")


def demo_dynamic_checkpointing():
    """演示动态启用/禁用检查点。"""
    print("\n" + "=" * 60)
    print("动态启用/禁用梯度检查点")
    print("=" * 60)

    model = ResNetBackbone(
        variant="resnet18",
        pretrained=False,
        feature_dim=10,
    )

    # 初始状态
    print(f"初始状态: {model.is_gradient_checkpointing_enabled()}")

    # 启用检查点
    model.enable_gradient_checkpointing()
    print(f"启用后: {model.is_gradient_checkpointing_enabled()}")

    # 禁用检查点
    model.disable_gradient_checkpointing()
    print(f"禁用后: {model.is_gradient_checkpointing_enabled()}")

    # 模拟训练场景
    print("\n模拟训练场景:")
    print("- Epoch 1-30: 使用检查点（内存受限）")
    model.enable_gradient_checkpointing()
    print(f"  检查点状态: {model.is_gradient_checkpointing_enabled()}")

    print("- Epoch 31-50: 禁用检查点（加速训练）")
    model.disable_gradient_checkpointing()
    print(f"  检查点状态: {model.is_gradient_checkpointing_enabled()}")


def main():
    """主函数。"""
    print("=" * 60)
    print("梯度检查点演示")
    print("=" * 60)

    # 1. 动态启用/禁用演示
    demo_dynamic_checkpointing()

    # 2. 内存节省估算
    demo_estimate_memory_savings()

    # 3. 实际训练对比
    if torch.cuda.is_available():
        compare_memory_usage()
    else:
        print("\n⚠️  CUDA 不可用，跳过训练对比")
        print("提示: 在 CPU 上也可以使用梯度检查点，但主要用于节省 GPU 内存")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n更多信息请参考: docs/guides/gradient_checkpointing.md")


if __name__ == "__main__":
    main()
