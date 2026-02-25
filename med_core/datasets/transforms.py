"""
Data augmentation and transformation utilities for medical images.

Provides configurable transforms for:
- Training (with augmentation)
- Validation/Testing (without augmentation)
- Medical-specific augmentations (e.g., intensity normalization)
"""

from typing import Any, Literal

import torch
from torchvision import transforms as T


class AddGaussianNoise:
    """Add Gaussian noise to an image tensor."""

    def __init__(self, mean: float = 0.0, std: float = 0.02):
        """
        Initialize Gaussian noise transform.

        Args:
            mean: Mean of the Gaussian noise
            std: Standard deviation of the Gaussian noise
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add noise to tensor."""
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class CLAHETransform:
    """
    Contrast Limited Adaptive Histogram Equalization.

    Useful for enhancing contrast in medical images.
    """

    def __init__(
        self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
    ):
        """
        Initialize CLAHE transform.

        Args:
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image: Any) -> Any:
        """Apply CLAHE to image."""
        try:
            import cv2
            import numpy as np
            from PIL import Image

            # Convert PIL to numpy
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image

            # Apply CLAHE
            if len(img_array.shape) == 3:
                # Convert to LAB color space for color images
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(
                    clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
                )
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # Grayscale
                clahe = cv2.createCLAHE(
                    clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
                )
                result = clahe.apply(img_array)

            return Image.fromarray(result)

        except ImportError:
            # Fallback if OpenCV not available
            return image


class RandomMedicalAugmentation:
    """
    Medical-specific random augmentations.

    Includes transformations that are common and safe for medical images.
    """

    def __init__(
        self,
        rotation_degrees: int = 15,
        translate: tuple[float, float] = (0.05, 0.05),
        scale: tuple[float, float] = (0.9, 1.1),
        shear: int = 5,
        flip_p: float = 0.5,
        brightness: float = 0.15,
        contrast: float = 0.15,
    ):
        """
        Initialize medical augmentation.

        Args:
            rotation_degrees: Maximum rotation angle
            translate: Maximum translation as fraction of image size
            scale: Scale range (min, max)
            shear: Maximum shear angle
            flip_p: Probability of horizontal flip
            brightness: Brightness jitter range
            contrast: Contrast jitter range
        """
        self.affine = T.RandomAffine(
            degrees=rotation_degrees,
            translate=translate,
            scale=scale,
            shear=shear,
        )
        self.flip = T.RandomHorizontalFlip(p=flip_p)
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=0.0,  # Usually not desired for medical images
            hue=0.0,
        )

    def __call__(self, image: Any) -> Any:
        """Apply augmentations."""
        image = self.affine(image)
        image = self.flip(image)
        image = self.color_jitter(image)
        return image


def get_train_transforms(
    image_size: int = 224,
    augmentation_strength: Literal["light", "medium", "heavy"] = "medium",
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    use_clahe: bool = False,
) -> T.Compose:
    """
    Get training transforms with augmentation.

    Args:
        image_size: Target image size
        augmentation_strength: Level of augmentation ("light", "medium", "heavy")
        normalize_mean: Mean for normalization (ImageNet default)
        normalize_std: Std for normalization (ImageNet default)
        use_clahe: Whether to apply CLAHE preprocessing

    Returns:
        Composed transform pipeline
    """
    transform_list = []

    # Optional CLAHE preprocessing
    if use_clahe:
        transform_list.append(CLAHETransform())

    # Resize
    transform_list.append(T.Resize((image_size, image_size)))

    # Augmentation based on strength
    if augmentation_strength == "light":
        transform_list.extend(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
            ]
        )

    elif augmentation_strength == "medium":
        transform_list.extend(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(
                    degrees=15,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    shear=5,
                ),
                T.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )

    elif augmentation_strength == "heavy":
        transform_list.extend(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.RandomAffine(
                    degrees=20,
                    translate=(0.1, 0.1),
                    scale=(0.8, 1.2),
                    shear=10,
                ),
                T.ColorJitter(brightness=0.2, contrast=0.25, saturation=0.1),
                T.RandomPerspective(distortion_scale=0.1, p=0.3),
            ]
        )

    # Convert to tensor and normalize
    transform_list.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )

    # Add Gaussian noise for medium/heavy augmentation
    if augmentation_strength in ("medium", "heavy"):
        transform_list.append(AddGaussianNoise(std=0.02))

    # Random erasing for heavy augmentation
    if augmentation_strength == "heavy":
        transform_list.append(T.RandomErasing(p=0.2, scale=(0.02, 0.1)))

    return T.Compose(transform_list)


def get_val_transforms(
    image_size: int = 224,
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    use_clahe: bool = False,
) -> T.Compose:
    """
    Get validation/test transforms (no augmentation).

    Args:
        image_size: Target image size
        normalize_mean: Mean for normalization
        normalize_std: Std for normalization
        use_clahe: Whether to apply CLAHE preprocessing

    Returns:
        Composed transform pipeline
    """
    transform_list = []

    if use_clahe:
        transform_list.append(CLAHETransform())

    transform_list.extend(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )

    return T.Compose(transform_list)


def get_medical_augmentation(
    image_size: int = 224,
    rotation_degrees: int = 15,
    translate: tuple[float, float] = (0.05, 0.05),
    scale: tuple[float, float] = (0.9, 1.1),
    flip_p: float = 0.5,
    brightness: float = 0.15,
    contrast: float = 0.15,
    gaussian_noise_std: float = 0.02,
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """
    Get customizable medical image augmentation pipeline.

    Allows fine-grained control over each augmentation parameter.

    Args:
        image_size: Target image size
        rotation_degrees: Maximum rotation angle
        translate: Translation range as fraction of image size
        scale: Scale range
        flip_p: Horizontal flip probability
        brightness: Brightness jitter range
        contrast: Contrast jitter range
        gaussian_noise_std: Standard deviation of Gaussian noise
        normalize_mean: Normalization mean
        normalize_std: Normalization std

    Returns:
        Composed transform pipeline
    """
    transform_list = [
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=flip_p),
        T.RandomAffine(
            degrees=rotation_degrees,
            translate=translate,
            scale=scale,
        ),
        T.ColorJitter(brightness=brightness, contrast=contrast),
        T.ToTensor(),
        T.Normalize(mean=normalize_mean, std=normalize_std),
    ]

    if gaussian_noise_std > 0:
        transform_list.append(AddGaussianNoise(std=gaussian_noise_std))

    return T.Compose(transform_list)


def denormalize(
    tensor: torch.Tensor,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Denormalize an image tensor for visualization.

    Args:
        tensor: Normalized image tensor (C, H, W) or (B, C, H, W)
        mean: Normalization mean
        std: Normalization std

    Returns:
        Denormalized tensor
    """
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean_tensor = mean_tensor.unsqueeze(0)
        std_tensor = std_tensor.unsqueeze(0)

    mean_tensor = mean_tensor.to(tensor.device)
    std_tensor = std_tensor.to(tensor.device)

    return tensor * std_tensor + mean_tensor
