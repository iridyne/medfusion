"""
支持注意力监督的数据集扩展

扩展基础数据集类，支持加载分割掩码、边界框等标注信息。
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from med_core.datasets.base import BaseMultimodalDataset


class AttentionSupervisedDataset(BaseMultimodalDataset):
    """
    支持注意力监督的数据集

    在基础多模态数据集的基础上，增加对分割掩码、边界框、关键点等标注的支持。

    Args:
        image_paths: 图像路径列表
        tabular_data: 表格数据
        labels: 标签
        mask_paths: 分割掩码路径列表（可选）
        bboxes: 边界框列表（可选）
        keypoints: 关键点列表（可选）
        transform: 图像变换
        target_transform: 标签变换
        mask_transform: 掩码变换
        return_mask: 是否返回掩码
        return_bbox: 是否返回边界框
        return_keypoint: 是否返回关键点

    Example:
        >>> dataset = AttentionSupervisedDataset(
        ...     image_paths=["img1.jpg", "img2.jpg"],
        ...     tabular_data=np.random.randn(2, 10),
        ...     labels=np.array([0, 1]),
        ...     mask_paths=["mask1.png", "mask2.png"],
        ...     return_mask=True,
        ... )
        >>>
        >>> image, tabular, label, mask = dataset[0]
    """

    def __init__(
        self,
        image_paths: list[str | Path],
        tabular_data: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        mask_paths: list[str | Path | None] | None = None,
        bboxes: list[list[float] | None] | None = None,
        keypoints: list[list[float] | None] | None = None,
        transform: Any = None,
        target_transform: Any = None,
        mask_transform: Any = None,
        return_mask: bool = False,
        return_bbox: bool = False,
        return_keypoint: bool = False,
    ):
        super().__init__(
            image_paths=image_paths,
            tabular_data=tabular_data,
            labels=labels,
            transform=transform,
            target_transform=target_transform,
        )

        # 分割掩码
        self.mask_paths = None
        if mask_paths is not None:
            if len(mask_paths) != len(image_paths):
                raise ValueError(
                    f"Mismatch between images ({len(image_paths)}) "
                    f"and masks ({len(mask_paths)})"
                )
            self.mask_paths = [Path(p) if p is not None else None for p in mask_paths]

        # 边界框
        self.bboxes = None
        if bboxes is not None:
            if len(bboxes) != len(image_paths):
                raise ValueError(
                    f"Mismatch between images ({len(image_paths)}) "
                    f"and bboxes ({len(bboxes)})"
                )
            self.bboxes = bboxes

        # 关键点
        self.keypoints = None
        if keypoints is not None:
            if len(keypoints) != len(image_paths):
                raise ValueError(
                    f"Mismatch between images ({len(image_paths)}) "
                    f"and keypoints ({len(keypoints)})"
                )
            self.keypoints = keypoints

        self.mask_transform = mask_transform
        self.return_mask = return_mask
        self.return_bbox = return_bbox
        self.return_keypoint = return_keypoint

    def load_image(self, path: Path) -> Image.Image:
        """加载图像"""
        return Image.open(path).convert("RGB")

    def load_mask(self, path: Path) -> Image.Image:
        """
        加载分割掩码

        Args:
            path: 掩码文件路径

        Returns:
            掩码图像（灰度图）
        """
        return Image.open(path).convert("L")  # 转换为灰度图

    def __getitem__(self, idx: int) -> tuple:
        """
        获取样本

        Returns:
            根据配置返回不同的元组：
            - 基础: (image, tabular, label)
            - +mask: (image, tabular, label, mask)
            - +bbox: (image, tabular, label, bbox)
            - +keypoint: (image, tabular, label, keypoint)
            - 全部: (image, tabular, label, mask, bbox, keypoint)
        """
        # 加载基础数据
        image, tabular, label = super().__getitem__(idx)

        result = [image, tabular, label]

        # 加载掩码
        if self.return_mask:
            if self.mask_paths is not None and self.mask_paths[idx] is not None:
                mask = self.load_mask(self.mask_paths[idx])

                # 应用掩码变换
                if self.mask_transform is not None:
                    mask = self.mask_transform(mask)
                elif self.transform is not None:
                    # 如果没有专门的掩码变换，使用图像变换（但跳过归一化等）
                    mask = self.transform(mask)

                result.append(mask)
            else:
                # 没有掩码，返回 None
                result.append(None)

        # 加载边界框
        if self.return_bbox:
            if self.bboxes is not None and self.bboxes[idx] is not None:
                bbox = torch.tensor(self.bboxes[idx], dtype=torch.float32)
                result.append(bbox)
            else:
                result.append(None)

        # 加载关键点
        if self.return_keypoint:
            if self.keypoints is not None and self.keypoints[idx] is not None:
                keypoint = torch.tensor(self.keypoints[idx], dtype=torch.float32)
                result.append(keypoint)
            else:
                result.append(None)

        return tuple(result)

    def has_masks(self) -> bool:
        """检查数据集是否有分割掩码"""
        return self.mask_paths is not None and any(p is not None for p in self.mask_paths)

    def has_bboxes(self) -> bool:
        """检查数据集是否有边界框"""
        return self.bboxes is not None and any(b is not None for b in self.bboxes)

    def has_keypoints(self) -> bool:
        """检查数据集是否有关键点"""
        return self.keypoints is not None and any(k is not None for k in self.keypoints)

    def get_mask_coverage(self) -> float:
        """
        获取掩码覆盖率（有多少样本有掩码）

        Returns:
            掩码覆盖率 [0, 1]
        """
        if not self.has_masks():
            return 0.0

        num_with_mask = sum(1 for p in self.mask_paths if p is not None)
        return num_with_mask / len(self.mask_paths)

    def get_statistics(self) -> dict[str, Any]:
        """
        获取数据集统计信息

        Returns:
            统计信息字典
        """
        stats = super().get_statistics()

        # 添加标注统计
        stats.update({
            "has_masks": self.has_masks(),
            "has_bboxes": self.has_bboxes(),
            "has_keypoints": self.has_keypoints(),
        })

        if self.has_masks():
            stats["mask_coverage"] = self.get_mask_coverage()

        return stats


class MedicalAttentionSupervisedDataset(AttentionSupervisedDataset):
    """
    医学影像注意力监督数据集

    专门用于医学影像的数据集，支持 DICOM 等医学影像格式。

    Args:
        image_paths: 图像路径列表
        tabular_data: 表格数据
        labels: 标签
        mask_paths: 分割掩码路径列表
        image_format: 图像格式 ("dicom", "nifti", "png", "jpg")
        transform: 图像变换
        target_transform: 标签变换
        mask_transform: 掩码变换
        return_mask: 是否返回掩码

    Example:
        >>> dataset = MedicalAttentionSupervisedDataset(
        ...     image_paths=["scan1.dcm", "scan2.dcm"],
        ...     tabular_data=patient_features,
        ...     labels=diagnoses,
        ...     mask_paths=["mask1.png", "mask2.png"],
        ...     image_format="dicom",
        ...     return_mask=True,
        ... )
    """

    def __init__(
        self,
        image_paths: list[str | Path],
        tabular_data: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        mask_paths: list[str | Path | None] | None = None,
        image_format: str = "dicom",
        transform: Any = None,
        target_transform: Any = None,
        mask_transform: Any = None,
        return_mask: bool = False,
    ):
        super().__init__(
            image_paths=image_paths,
            tabular_data=tabular_data,
            labels=labels,
            mask_paths=mask_paths,
            transform=transform,
            target_transform=target_transform,
            mask_transform=mask_transform,
            return_mask=return_mask,
        )

        self.image_format = image_format.lower()

    def load_image(self, path: Path) -> Image.Image | np.ndarray:
        """
        加载医学影像

        Args:
            path: 图像路径

        Returns:
            加载的图像
        """
        if self.image_format == "dicom":
            return self._load_dicom(path)
        elif self.image_format == "nifti":
            return self._load_nifti(path)
        else:
            # 标准图像格式
            return Image.open(path).convert("RGB")

    def _load_dicom(self, path: Path) -> np.ndarray:
        """
        加载 DICOM 文件

        Args:
            path: DICOM 文件路径

        Returns:
            图像数组
        """
        try:
            import pydicom
        except ImportError:
            raise ImportError(
                "pydicom is required to load DICOM files. "
                "Install it with: pip install pydicom"
            )

        # 读取 DICOM
        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array

        # 归一化到 [0, 255]
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
        image = image.astype(np.uint8)

        # 转换为 PIL Image
        if len(image.shape) == 2:
            # 灰度图转 RGB
            image = Image.fromarray(image).convert("RGB")
        else:
            image = Image.fromarray(image)

        return image

    def _load_nifti(self, path: Path) -> np.ndarray:
        """
        加载 NIfTI 文件

        Args:
            path: NIfTI 文件路径

        Returns:
            图像数组
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "nibabel is required to load NIfTI files. "
                "Install it with: pip install nibabel"
            )

        # 读取 NIfTI
        nii = nib.load(path)
        image = nii.get_fdata()

        # 如果是 3D，取中间切片
        if len(image.shape) == 3:
            image = image[:, :, image.shape[2] // 2]

        # 归一化到 [0, 255]
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
        image = image.astype(np.uint8)

        # 转换为 PIL Image
        image = Image.fromarray(image).convert("RGB")

        return image

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        image_dir: str | Path,
        mask_dir: str | Path | None = None,
        image_col: str = "image_path",
        mask_col: str = "mask_path",
        label_col: str = "label",
        tabular_cols: list[str] | None = None,
        image_format: str = "dicom",
        transform: Any = None,
        return_mask: bool = False,
    ) -> "MedicalAttentionSupervisedDataset":
        """
        从 CSV 文件创建数据集

        Args:
            csv_path: CSV 文件路径
            image_dir: 图像目录
            mask_dir: 掩码目录（可选）
            image_col: 图像路径列名
            mask_col: 掩码路径列名
            label_col: 标签列名
            tabular_cols: 表格特征列名列表
            image_format: 图像格式
            transform: 图像变换
            return_mask: 是否返回掩码

        Returns:
            数据集实例

        Example:
            >>> dataset = MedicalAttentionSupervisedDataset.from_csv(
            ...     csv_path="data.csv",
            ...     image_dir="images/",
            ...     mask_dir="masks/",
            ...     image_col="scan_path",
            ...     mask_col="lesion_mask",
            ...     label_col="diagnosis",
            ...     tabular_cols=["age", "gender", "symptoms"],
            ...     return_mask=True,
            ... )
        """
        import pandas as pd

        # 读取 CSV
        df = pd.read_csv(csv_path)

        # 图像路径
        image_dir = Path(image_dir)
        image_paths = [image_dir / p for p in df[image_col]]

        # 掩码路径
        mask_paths = None
        if mask_dir is not None and mask_col in df.columns:
            mask_dir = Path(mask_dir)
            mask_paths = []
            for mask_name in df[mask_col]:
                if pd.isna(mask_name) or mask_name == "":
                    mask_paths.append(None)
                else:
                    mask_paths.append(mask_dir / mask_name)

        # 标签
        labels = df[label_col].values

        # 表格数据
        if tabular_cols is None:
            # 使用除了图像、掩码、标签之外的所有列
            exclude_cols = {image_col, mask_col, label_col}
            tabular_cols = [c for c in df.columns if c not in exclude_cols]

        tabular_data = df[tabular_cols].values.astype(np.float32)

        return cls(
            image_paths=image_paths,
            tabular_data=tabular_data,
            labels=labels,
            mask_paths=mask_paths,
            image_format=image_format,
            transform=transform,
            return_mask=return_mask,
        )
