"""
Multi-region feature extractors for medical imaging.

This module provides extractors that can process multiple regions of interest
(ROIs) from medical images, such as tumor regions, surrounding tissue, etc.
"""

import torch
import torch.nn.functional as F
from torch import nn


class MultiRegionExtractor(nn.Module):
    """
    Extract features from multiple regions of interest.

    Args:
        backbone: Feature extraction backbone
        num_regions: Number of regions to extract
        region_pooling: Pooling method for each region ('avg', 'max', 'adaptive')
        output_size: Output spatial size for adaptive pooling

    Example:
        >>> from med_core.backbones import swin3d_small
        >>> backbone = swin3d_small(in_channels=1, feature_dim=512)
        >>> extractor = MultiRegionExtractor(backbone, num_regions=3)
        >>> image = torch.randn(2, 1, 64, 128, 128)
        >>> masks = torch.randn(2, 3, 64, 128, 128)  # 3 region masks
        >>> features = extractor(image, masks)  # [2, 3, 512]
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_regions: int = 1,
        region_pooling: str = "adaptive",
        output_size: tuple = (4, 4, 4),
    ):
        super().__init__()

        self.backbone = backbone
        self.num_regions = num_regions
        self.region_pooling = region_pooling
        self.output_size = output_size

    def forward(
        self,
        x: torch.Tensor,
        region_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image [B, C, D, H, W] or [B, C, H, W]
            region_masks: Region masks [B, num_regions, D, H, W] or [B, num_regions, H, W]
                         If None, uses uniform regions

        Returns:
            Region features [B, num_regions, feature_dim]
        """
        # Extract features from backbone
        if hasattr(self.backbone, "forward_features"):
            # Get intermediate features
            features = self.backbone.forward_features(x)
        else:
            # Use full forward pass
            features = self.backbone(x)

        # If features are 1D, we need to reshape or use different strategy
        if features.dim() == 2:
            # Features are already pooled [B, feature_dim]
            # Replicate for each region (simple strategy)
            features = features.unsqueeze(1).expand(-1, self.num_regions, -1)
            return features

        # Extract region-specific features
        if region_masks is None:
            # Use uniform grid regions
            region_features = self._extract_uniform_regions(features)
        else:
            # Use provided masks
            region_features = self._extract_masked_regions(features, region_masks)

        return region_features

    def _extract_uniform_regions(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract features from uniform grid regions.

        Args:
            features: Feature maps [B, C, D, H, W] or [B, C, H, W]

        Returns:
            Region features [B, num_regions, C]
        """
        batch_size, channels = features.shape[:2]
        is_3d = features.dim() == 5

        if is_3d:
            # 3D case: divide into grid
            d, h, w = features.shape[2:]
            # Simple strategy: divide into num_regions parts along depth
            region_size = max(1, d // self.num_regions)
            region_features = []

            for i in range(self.num_regions):
                start = i * region_size
                end = min((i + 1) * region_size, d)
                region = features[:, :, start:end, :, :]
                # Pool region
                pooled = F.adaptive_avg_pool3d(region, (1, 1, 1))
                region_features.append(pooled.view(batch_size, channels))

            return torch.stack(region_features, dim=1)  # [B, num_regions, C]
        # 2D case
        h, w = features.shape[2:]
        region_size = max(1, h // self.num_regions)
        region_features = []

        for i in range(self.num_regions):
            start = i * region_size
            end = min((i + 1) * region_size, h)
            region = features[:, :, start:end, :]
            pooled = F.adaptive_avg_pool2d(region, (1, 1))
            region_features.append(pooled.view(batch_size, channels))

        return torch.stack(region_features, dim=1)

    def _extract_masked_regions(
        self, features: torch.Tensor, masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract features from masked regions.

        Args:
            features: Feature maps [B, C, D, H, W] or [B, C, H, W]
            masks: Region masks [B, num_regions, D, H, W] or [B, num_regions, H, W]

        Returns:
            Region features [B, num_regions, C]
        """
        batch_size, channels = features.shape[:2]
        is_3d = features.dim() == 5

        # Resize masks to match feature map size
        if is_3d:
            target_size = features.shape[2:]
            if masks.shape[2:] != target_size:
                masks = F.interpolate(
                    masks, size=target_size, mode="trilinear", align_corners=False,
                )
        else:
            target_size = features.shape[2:]
            if masks.shape[2:] != target_size:
                masks = F.interpolate(
                    masks, size=target_size, mode="bilinear", align_corners=False,
                )

        # Extract features for each region
        region_features = []
        for i in range(self.num_regions):
            mask = masks[:, i : i + 1, ...]  # [B, 1, ...]
            # Apply mask
            masked_features = features * mask
            # Pool
            if is_3d:
                pooled = F.adaptive_avg_pool3d(masked_features, (1, 1, 1))
            else:
                pooled = F.adaptive_avg_pool2d(masked_features, (1, 1))
            region_features.append(pooled.view(batch_size, channels))

        return torch.stack(region_features, dim=1)  # [B, num_regions, C]


class HierarchicalRegionExtractor(nn.Module):
    """
    Extract features from hierarchical regions (e.g., tumor, peritumoral, background).

    Args:
        backbone: Feature extraction backbone
        region_names: Names of regions
        aggregation: How to aggregate region features ('concat', 'attention', 'mean')

    Example:
        >>> extractor = HierarchicalRegionExtractor(
        ...     backbone=backbone,
        ...     region_names=['tumor', 'peritumoral', 'background']
        ... )
        >>> image = torch.randn(2, 1, 64, 128, 128)
        >>> masks = {'tumor': torch.randn(2, 1, 64, 128, 128),
        ...          'peritumoral': torch.randn(2, 1, 64, 128, 128),
        ...          'background': torch.randn(2, 1, 64, 128, 128)}
        >>> features = extractor(image, masks)
    """

    def __init__(
        self,
        backbone: nn.Module,
        region_names: list[str],
        aggregation: str = "concat",
        attention_dim: int = 128,
    ):
        super().__init__()

        self.backbone = backbone
        self.region_names = region_names
        self.num_regions = len(region_names)
        self.aggregation = aggregation

        # Get feature dimension from backbone
        if hasattr(backbone, "backbone_output_dim"):
            self.feature_dim = backbone.backbone_output_dim
        else:
            # Try to infer from a dummy forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, 1, 32, 32, 32)
                try:
                    dummy_output = backbone(dummy_input)
                    self.feature_dim = dummy_output.shape[-1]
                except Exception:
                    self.feature_dim = 512  # Default

        # Aggregation module
        if aggregation == "attention":
            self.attention = nn.Sequential(
                nn.Linear(self.feature_dim, attention_dim),
                nn.Tanh(),
                nn.Linear(attention_dim, 1),
            )
        elif aggregation == "concat":
            self.output_dim = self.feature_dim * self.num_regions
        elif aggregation == "mean":
            self.output_dim = self.feature_dim
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def forward(
        self,
        x: torch.Tensor,
        region_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image [B, C, D, H, W] or [B, C, H, W]
            region_masks: Dictionary of region masks {region_name: mask_tensor}

        Returns:
            Aggregated features [B, output_dim]
        """
        # Extract features for each region
        region_features = []

        for region_name in self.region_names:
            if region_name not in region_masks:
                raise ValueError(f"Missing mask for region: {region_name}")

            mask = region_masks[region_name]
            # Apply mask to input
            masked_input = x * mask

            # Extract features
            features = self.backbone(masked_input)
            region_features.append(features)

        # Stack region features
        region_features = torch.stack(
            region_features, dim=1,
        )  # [B, num_regions, feature_dim]

        # Aggregate
        if self.aggregation == "concat":
            # Concatenate all region features
            aggregated = region_features.view(region_features.size(0), -1)
        elif self.aggregation == "mean":
            # Average pooling
            aggregated = region_features.mean(dim=1)
        elif self.aggregation == "attention":
            # Attention-based aggregation
            attention_logits = self.attention(region_features)  # [B, num_regions, 1]
            attention_weights = F.softmax(attention_logits, dim=1)
            aggregated = (region_features * attention_weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return aggregated


class AdaptiveRegionExtractor(nn.Module):
    """
    Adaptive region extractor that learns to identify important regions.

    Args:
        backbone: Feature extraction backbone
        num_regions: Number of regions to extract
        region_size: Size of each region
        feature_dim: Feature dimension

    Example:
        >>> extractor = AdaptiveRegionExtractor(
        ...     backbone=backbone,
        ...     num_regions=5,
        ...     feature_dim=512
        ... )
        >>> image = torch.randn(2, 1, 64, 128, 128)
        >>> features, region_coords = extractor(image, return_coords=True)
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_regions: int = 5,
        region_size: int = 16,
        feature_dim: int = 512,
    ):
        super().__init__()

        self.backbone = backbone
        self.num_regions = num_regions
        self.region_size = region_size
        self.feature_dim = feature_dim

        # Region proposal network
        self.region_proposal = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_regions * 3),  # 3 coordinates (x, y, z) or (x, y)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_coords: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image [B, C, D, H, W] or [B, C, H, W]
            return_coords: Return region coordinates

        Returns:
            Region features [B, num_regions, feature_dim]
            If return_coords=True, also returns coordinates
        """
        # Extract global features
        global_features = self.backbone(x)  # [B, feature_dim]

        # Propose region coordinates
        region_coords = self.region_proposal(global_features)  # [B, num_regions * 3]
        region_coords = region_coords.view(
            -1, self.num_regions, 3,
        )  # [B, num_regions, 3]
        region_coords = torch.sigmoid(region_coords)  # Normalize to [0, 1]

        # Extract features from proposed regions
        # For simplicity, replicate global features
        # In practice, you would crop and process each region
        region_features = global_features.unsqueeze(1).expand(-1, self.num_regions, -1)

        if return_coords:
            return region_features, region_coords
        return region_features


class MultiScaleRegionExtractor(nn.Module):
    """
    Extract features from multiple scales and regions.

    Args:
        backbone: Feature extraction backbone
        scales: List of scales to extract features from
        num_regions_per_scale: Number of regions per scale

    Example:
        >>> extractor = MultiScaleRegionExtractor(
        ...     backbone=backbone,
        ...     scales=[1.0, 0.5, 0.25],
        ...     num_regions_per_scale=3
        ... )
        >>> image = torch.randn(2, 1, 64, 128, 128)
        >>> features = extractor(image)  # [2, 9, 512] (3 scales × 3 regions)
    """

    def __init__(
        self,
        backbone: nn.Module,
        scales: list[float] | None = None,
        num_regions_per_scale: int = 3,
    ):
        super().__init__()

        self.backbone = backbone
        self.scales = scales if scales is not None else [1.0, 0.5, 0.25]
        self.num_regions_per_scale = num_regions_per_scale
        self.total_regions = len(self.scales) * num_regions_per_scale

        # Region extractors for each scale
        self.region_extractors = nn.ModuleList(
            [
                MultiRegionExtractor(backbone, num_regions=num_regions_per_scale)
                for _ in scales
            ],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image [B, C, D, H, W] or [B, C, H, W]

        Returns:
            Multi-scale region features [B, total_regions, feature_dim]
        """
        all_features = []

        for scale, extractor in zip(self.scales, self.region_extractors):
            if scale != 1.0:
                # Resize input
                is_3d = x.dim() == 5
                if is_3d:
                    size = tuple(int(s * scale) for s in x.shape[2:])
                    x_scaled = F.interpolate(
                        x, size=size, mode="trilinear", align_corners=False,
                    )
                else:
                    size = tuple(int(s * scale) for s in x.shape[2:])
                    x_scaled = F.interpolate(
                        x, size=size, mode="bilinear", align_corners=False,
                    )
            else:
                x_scaled = x

            # Extract region features
            features = extractor(x_scaled)  # [B, num_regions_per_scale, feature_dim]
            all_features.append(features)

        # Concatenate all scale features
        all_features = torch.cat(all_features, dim=1)  # [B, total_regions, feature_dim]

        return all_features
