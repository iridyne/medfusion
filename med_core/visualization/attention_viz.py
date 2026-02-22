"""
注意力可视化工具

提供注意力权重和监督效果的可视化功能。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure


def visualize_attention_overlay(
    image: torch.Tensor | np.ndarray,
    attention: torch.Tensor | np.ndarray,
    alpha: float = 0.5,
    cmap: str = "jet",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """
    可视化注意力叠加在原图上

    Args:
        image: 原始图像 (C, H, W) 或 (H, W, C)
        attention: 注意力权重 (H, W)
        alpha: 注意力图的透明度
        cmap: 颜色映射
        title: 图像标题
        save_path: 保存路径

    Returns:
        matplotlib Figure 对象

    Example:
        >>> image = torch.randn(3, 224, 224)
        >>> attention = torch.randn(224, 224)
        >>> fig = visualize_attention_overlay(image, attention)
        >>> plt.show()
    """
    # 转换为 numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(attention, torch.Tensor):
        attention = attention.cpu().numpy()

    # 调整图像维度 (C, H, W) -> (H, W, C)
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))

    # 归一化图像到 [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # 如果是灰度图，转换为 RGB
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    # 归一化注意力到 [0, 1]
    attention = (attention - attention.min()) / (
        attention.max() - attention.min() + 1e-8
    )

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))

    # 显示原图
    ax.imshow(image)

    # 叠加注意力热图
    im = ax.imshow(attention, cmap=cmap, alpha=alpha)

    # 添加颜色条
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 设置标题
    if title:
        ax.set_title(title, fontsize=14)

    ax.axis("off")
    plt.tight_layout()

    # 保存
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_attention_comparison(
    image: torch.Tensor | np.ndarray,
    attention_before: torch.Tensor | np.ndarray,
    attention_after: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray | None = None,
    titles: list[str] | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """
    对比可视化：引导前后的注意力变化

    Args:
        image: 原始图像
        attention_before: 引导前的注意力
        attention_after: 引导后的注意力
        target: 目标掩码（可选）
        titles: 子图标题列表
        save_path: 保存路径

    Returns:
        matplotlib Figure 对象

    Example:
        >>> fig = visualize_attention_comparison(
        ...     image=image,
        ...     attention_before=attention_before,
        ...     attention_after=attention_after,
        ...     target=mask,
        ...     titles=["原图", "引导前", "引导后", "目标掩码"],
        ... )
    """
    # 转换为 numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(attention_before, torch.Tensor):
        attention_before = attention_before.cpu().numpy()
    if isinstance(attention_after, torch.Tensor):
        attention_after = attention_after.cpu().numpy()
    if target is not None and isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    # 调整图像维度
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))

    # 归一化
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    attention_before = (attention_before - attention_before.min()) / (
        attention_before.max() - attention_before.min() + 1e-8
    )
    attention_after = (attention_after - attention_after.min()) / (
        attention_after.max() - attention_after.min() + 1e-8
    )

    # 确定子图数量
    num_plots = 3 if target is None else 4

    # 默认标题
    if titles is None:
        titles = ["原图", "引导前", "引导后"]
        if target is not None:
            titles.append("目标掩码")

    # 创建图形
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

    # 原图
    axes[0].imshow(image)
    axes[0].set_title(titles[0], fontsize=12)
    axes[0].axis("off")

    # 引导前
    axes[1].imshow(image)
    im1 = axes[1].imshow(attention_before, cmap="jet", alpha=0.5)
    axes[1].set_title(titles[1], fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 引导后
    axes[2].imshow(image)
    im2 = axes[2].imshow(attention_after, cmap="jet", alpha=0.5)
    axes[2].set_title(titles[2], fontsize=12)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 目标掩码
    if target is not None:
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)
        axes[3].imshow(image)
        im3 = axes[3].imshow(target, cmap="Reds", alpha=0.5)
        axes[3].set_title(titles[3], fontsize=12)
        axes[3].axis("off")
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # 保存
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_attention_supervision_loss(
    image: torch.Tensor | np.ndarray,
    attention: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    loss_components: dict[str, float],
    save_path: str | Path | None = None,
) -> Figure:
    """
    可视化注意力监督损失

    Args:
        image: 原始图像
        attention: 注意力权重
        target: 目标分布
        loss_components: 损失组件字典
        save_path: 保存路径

    Returns:
        matplotlib Figure 对象

    Example:
        >>> fig = visualize_attention_supervision_loss(
        ...     image=image,
        ...     attention=attention,
        ...     target=mask,
        ...     loss_components={"main": 0.5, "smooth": 0.1},
        ... )
    """
    # 转换为 numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(attention, torch.Tensor):
        attention = attention.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    # 调整维度
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))

    # 归一化
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    attention = (attention - attention.min()) / (
        attention.max() - attention.min() + 1e-8
    )
    target = (target - target.min()) / (target.max() - target.min() + 1e-8)

    # 创建图形
    fig = plt.figure(figsize=(15, 5))

    # 子图1: 原图
    ax1 = plt.subplot(1, 4, 1)
    ax1.imshow(image)
    ax1.set_title("原图", fontsize=12)
    ax1.axis("off")

    # 子图2: 注意力
    ax2 = plt.subplot(1, 4, 2)
    ax2.imshow(image)
    im2 = ax2.imshow(attention, cmap="jet", alpha=0.5)
    ax2.set_title("模型注意力", fontsize=12)
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 子图3: 目标
    ax3 = plt.subplot(1, 4, 3)
    ax3.imshow(image)
    im3 = ax3.imshow(target, cmap="Reds", alpha=0.5)
    ax3.set_title("目标分布", fontsize=12)
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # 子图4: 损失组件
    ax4 = plt.subplot(1, 4, 4)
    loss_names = list(loss_components.keys())
    loss_values = list(loss_components.values())

    bars = ax4.bar(loss_names, loss_values, color="steelblue")
    ax4.set_title("损失组件", fontsize=12)
    ax4.set_ylabel("损失值", fontsize=10)
    ax4.tick_params(axis="x", rotation=45)

    # 在柱状图上显示数值
    for bar, value in zip(bars, loss_values):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    # 保存
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_mil_attention(
    image: torch.Tensor | np.ndarray,
    patch_attention: torch.Tensor | np.ndarray,
    grid_size: tuple[int, int],
    top_k: int = 5,
    save_path: str | Path | None = None,
) -> Figure:
    """
    可视化多实例学习的 patch 注意力

    Args:
        image: 原始图像
        patch_attention: Patch 注意力权重 (num_patches,)
        grid_size: Patch 网格尺寸 (num_patches_h, num_patches_w)
        top_k: 显示 top-k 个最重要的 patches
        save_path: 保存路径

    Returns:
        matplotlib Figure 对象

    Example:
        >>> fig = visualize_mil_attention(
        ...     image=image,
        ...     patch_attention=attention_weights,
        ...     grid_size=(14, 14),
        ...     top_k=5,
        ... )
    """
    # 转换为 numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(patch_attention, torch.Tensor):
        patch_attention = patch_attention.cpu().numpy()

    # 调整维度
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))

    # 归一化
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    # 重塑 patch 注意力为 2D 网格
    num_patches_h, num_patches_w = grid_size
    attention_map = patch_attention.reshape(num_patches_h, num_patches_w)

    # 找到 top-k patches
    top_k_indices = np.argsort(patch_attention)[-top_k:][::-1]
    top_k_coords = [
        (idx // num_patches_w, idx % num_patches_w) for idx in top_k_indices
    ]

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 子图1: 原图
    axes[0].imshow(image)
    axes[0].set_title("原图", fontsize=12)
    axes[0].axis("off")

    # 子图2: 注意力热图
    im = axes[1].imshow(attention_map, cmap="jet", interpolation="nearest")
    axes[1].set_title("Patch 注意力分布", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 子图3: Top-k patches 高亮
    axes[2].imshow(image)

    # 计算 patch 大小
    H, W = image.shape[:2]
    patch_h = H // num_patches_h
    patch_w = W // num_patches_w

    # 绘制 top-k patches 的边界框
    for rank, (i, j) in enumerate(top_k_coords):
        y = i * patch_h
        x = j * patch_w

        # 绘制矩形
        rect = plt.Rectangle(
            (x, y),
            patch_w,
            patch_h,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        axes[2].add_patch(rect)

        # 标注排名
        axes[2].text(
            x + patch_w / 2,
            y + patch_h / 2,
            f"{rank + 1}",
            color="yellow",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
        )

    axes[2].set_title(f"Top-{top_k} 重要 Patches", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()

    # 保存
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_attention_statistics(
    attention_history: list[torch.Tensor | np.ndarray],
    labels: list[str] | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """
    绘制注意力统计信息（熵、方差等）

    Args:
        attention_history: 注意力权重历史列表
        labels: 标签列表
        save_path: 保存路径

    Returns:
        matplotlib Figure 对象

    Example:
        >>> attention_history = [attention_epoch1, attention_epoch2, ...]
        >>> fig = plot_attention_statistics(
        ...     attention_history,
        ...     labels=["Epoch 1", "Epoch 2", ...],
        ... )
    """
    # 计算统计信息
    entropies = []
    variances = []
    max_values = []

    for attention in attention_history:
        if isinstance(attention, torch.Tensor):
            attention = attention.cpu().numpy()

        # 展平
        attention_flat = attention.flatten()

        # 归一化为概率分布
        attention_prob = attention_flat / (attention_flat.sum() + 1e-8)

        # 熵
        entropy = -(attention_prob * np.log(attention_prob + 1e-8)).sum()
        entropies.append(entropy)

        # 方差
        variance = attention_flat.var()
        variances.append(variance)

        # 最大值
        max_value = attention_flat.max()
        max_values.append(max_value)

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = range(len(attention_history))
    if labels is None:
        labels = [f"Step {i}" for i in x]

    # 熵
    axes[0].plot(x, entropies, marker="o", color="steelblue")
    axes[0].set_title("注意力熵（越小越集中）", fontsize=12)
    axes[0].set_xlabel("步骤", fontsize=10)
    axes[0].set_ylabel("熵", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 方差
    axes[1].plot(x, variances, marker="s", color="coral")
    axes[1].set_title("注意力方差（越大越集中）", fontsize=12)
    axes[1].set_xlabel("步骤", fontsize=10)
    axes[1].set_ylabel("方差", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # 最大值
    axes[2].plot(x, max_values, marker="^", color="green")
    axes[2].set_title("注意力最大值", fontsize=12)
    axes[2].set_xlabel("步骤", fontsize=10)
    axes[2].set_ylabel("最大值", fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
