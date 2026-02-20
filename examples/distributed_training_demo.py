"""
分布式训练示例

演示如何使用 DDP 和 FSDP 进行分布式训练。
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def create_simple_model():
    """创建简单的测试模型"""
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )


def create_dataset(num_samples=1000):
    """创建测试数据集"""
    X = torch.randn(num_samples, 10)
    y = torch.randint(0, 2, (num_samples,))
    return TensorDataset(X, y)


def train_epoch(model, dataloader, criterion, optimizer, device, rank):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if batch_idx % 10 == 0:
            print(f"Rank {rank}: Batch {batch_idx}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def train_ddp(args):
    """使用 DDP 训练"""
    print("\n" + "=" * 60)
    print("DDP (DistributedDataParallel) 训练")
    print("=" * 60)

    from med_core.utils.distributed import (
        DDPWrapper,
        cleanup_distributed,
        is_main_process,
        reduce_dict,
        save_checkpoint,
        setup_distributed,
    )

    # 设置分布式环境
    rank, local_rank, world_size = setup_distributed(backend=args.backend)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = create_simple_model()
    model = DDPWrapper(model, find_unused_parameters=False)

    # 创建数据集和数据加载器
    dataset = create_dataset(args.num_samples)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
    )

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        avg_loss, accuracy = train_epoch(
            model, dataloader, criterion, optimizer, device, rank
        )

        # 归约指标
        metrics = {
            "loss": torch.tensor(avg_loss).to(device),
            "accuracy": torch.tensor(accuracy).to(device),
        }
        avg_metrics = reduce_dict(metrics)

        if is_main_process():
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print(f"  Average Loss: {avg_metrics['loss'].item():.4f}")
            print(f"  Average Accuracy: {avg_metrics['accuracy'].item():.2f}%")

        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                f"outputs/checkpoint_ddp_epoch_{epoch + 1}.pt",
                loss=avg_metrics['loss'].item(),
                accuracy=avg_metrics['accuracy'].item(),
            )

    # 清理
    cleanup_distributed()

    if is_main_process():
        print("\n✓ DDP 训练完成！")


def train_fsdp(args):
    """使用 FSDP 训练"""
    print("\n" + "=" * 60)
    print("FSDP (Fully Sharded Data Parallel) 训练")
    print("=" * 60)

    from med_core.utils.distributed import (
        FSDPWrapper,
        cleanup_distributed,
        is_main_process,
        reduce_dict,
        save_checkpoint,
        setup_distributed,
    )

    # 设置分布式环境
    rank, local_rank, world_size = setup_distributed(backend=args.backend)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = create_simple_model()
    model = FSDPWrapper(
        model,
        sharding_strategy=args.sharding_strategy,
        min_num_params=args.min_num_params,
    )

    # 创建数据集和数据加载器
    dataset = create_dataset(args.num_samples)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
    )

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        avg_loss, accuracy = train_epoch(
            model, dataloader, criterion, optimizer, device, rank
        )

        # 归约指标
        metrics = {
            "loss": torch.tensor(avg_loss).to(device),
            "accuracy": torch.tensor(accuracy).to(device),
        }
        avg_metrics = reduce_dict(metrics)

        if is_main_process():
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print(f"  Average Loss: {avg_metrics['loss'].item():.4f}")
            print(f"  Average Accuracy: {avg_metrics['accuracy'].item():.2f}%")

        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                f"outputs/checkpoint_fsdp_epoch_{epoch + 1}.pt",
                loss=avg_metrics['loss'].item(),
                accuracy=avg_metrics['accuracy'].item(),
            )

    # 清理
    cleanup_distributed()

    if is_main_process():
        print("\n✓ FSDP 训练完成！")


def demo_usage():
    """演示使用方法"""
    print("\n" + "=" * 60)
    print("分布式训练使用指南")
    print("=" * 60)

    print("\n1. 单机多卡 DDP 训练:")
    print("   torchrun --nproc_per_node=4 distributed_training_demo.py --strategy ddp")

    print("\n2. 多机多卡 DDP 训练:")
    print("   # 节点 0:")
    print("   torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \\")
    print("            --master_addr=192.168.1.1 --master_port=29500 \\")
    print("            distributed_training_demo.py --strategy ddp")
    print("   # 节点 1:")
    print("   torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \\")
    print("            --master_addr=192.168.1.1 --master_port=29500 \\")
    print("            distributed_training_demo.py --strategy ddp")

    print("\n3. FSDP 训练:")
    print("   torchrun --nproc_per_node=4 distributed_training_demo.py --strategy fsdp")

    print("\n4. 不同的分片策略:")
    print("   # 完全分片（最省内存）")
    print("   --sharding_strategy FULL_SHARD")
    print("   # 仅分片梯度和优化器状态")
    print("   --sharding_strategy SHARD_GRAD_OP")
    print("   # 不分片（类似 DDP）")
    print("   --sharding_strategy NO_SHARD")

    print("\n5. 环境变量:")
    print("   export MASTER_ADDR=localhost")
    print("   export MASTER_PORT=29500")
    print("   export WORLD_SIZE=4")
    print("   export RANK=0")
    print("   export LOCAL_RANK=0")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分布式训练示例")

    # 训练参数
    parser.add_argument("--strategy", type=str, default="ddp",
                        choices=["ddp", "fsdp"],
                        help="分布式策略")
    parser.add_argument("--backend", type=str, default="nccl",
                        choices=["nccl", "gloo", "mpi"],
                        help="分布式后端")
    parser.add_argument("--epochs", type=int, default=5,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="样本数量")
    parser.add_argument("--save_interval", type=int, default=2,
                        help="保存间隔")

    # FSDP 参数
    parser.add_argument("--sharding_strategy", type=str, default="FULL_SHARD",
                        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"],
                        help="FSDP 分片策略")
    parser.add_argument("--min_num_params", type=int, default=1000,
                        help="自动包装的最小参数数量")

    # 其他
    parser.add_argument("--demo", action="store_true",
                        help="显示使用示例")

    args = parser.parse_args()

    # 创建输出目录
    Path("outputs").mkdir(exist_ok=True)

    if args.demo:
        demo_usage()
        return

    # 根据策略选择训练函数
    if args.strategy == "ddp":
        train_ddp(args)
    elif args.strategy == "fsdp":
        train_fsdp(args)


if __name__ == "__main__":
    main()
