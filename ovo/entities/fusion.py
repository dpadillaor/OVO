"""
Fusion Strategy Pattern for Instance Comparison

This module implements the Strategy pattern for instance fusion logic,
allowing dynamic selection of different fusion algorithms (CLIP, DINO, PE, geometric).
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from ..utils.instance_utils import compute_pcd_overlap, compute_centroid_distance

import torch
import numpy as np
import open3d as o3d

class FusionStrategy(ABC):
    """Abstract base class for fusion strategies."""

    @abstractmethod
    def same_instance(
        self,
        instance1,
        instance2,
        points_centroid1: Tuple[torch.Tensor, torch.Tensor],
        points_centroid2: Tuple[torch.Tensor, torch.Tensor]
    ) -> bool:
        """
        Determine if two instances represent the same object.

        Args:
            instance1: First Instance3D object
            instance2: Second Instance3D object
            points_centroid1: Tuple of (points, centroid) for instance1
            points_centroid2: Tuple of (points, centroid) for instance2

        Returns:
            bool: True if instances should be fused, False otherwise
        """
        pass


class SemanticGeometricFusion(FusionStrategy):
    """
    Semantic + Geometric fusion strategy.

    Uses both semantic features (CLIP, DINO, PE) and geometric proximity
    to determine if instances should be fused.
    """

    def __init__(self, config: Dict[str, Any], feature_attr: str):
        """
        Initialize the semantic-geometric fusion strategy.

        Args:
            config: Configuration dictionary with thresholds
            feature_attr: Name of the feature attribute to use (e.g., 'clip_feature')
        """
        self.th_centroid = config.get("th_centroid", 1.5)
        self.th_cossim = config.get("th_cossim", 0.81)
        self.th_points = config.get("th_points", 0.1)
        self.feature_attr = feature_attr

    def same_instance(
        self,
        instance1,
        instance2,
        points_centroid1: Tuple[torch.Tensor, torch.Tensor],
        points_centroid2: Tuple[torch.Tensor, torch.Tensor]
    ) -> bool:
        """Determine if instances are the same using semantic + geometric criteria."""
        points1, centroid1 = points_centroid1
        points2, centroid2 = points_centroid2

        # Check centroid distance
        if compute_centroid_distance(centroid1, centroid2) > self.th_centroid:
            return False

        # Check semantic similarity
        feature1 = getattr(instance1, self.feature_attr)[0]
        feature2 = getattr(instance2, self.feature_attr)[0]
        cos_sim = torch.nn.functional.cosine_similarity(feature1, feature2, dim=0)
        if cos_sim < self.th_cossim:
            return False

        # Check point cloud proximity
        p_dist = compute_pcd_overlap(points1, points2, self.th_points)

        return p_dist > 0.5 or (cos_sim > 0.9 and p_dist > 0.2)


class GeometricOnlyFusion(FusionStrategy):
    """
    Geometric-only fusion strategy.

    Uses only spatial overlap and distance to determine if instances should be fused.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the geometric-only fusion strategy.

        Args:
            config: Configuration dictionary with thresholds
        """
        self.th_centroid = config.get("th_centroid", 1.5)
        self.th_points = config.get("th_points", 0.1)
        self.th_cossim = None

    def same_instance(
        self,
        instance1,
        instance2,
        points_centroid1: Tuple[torch.Tensor, torch.Tensor],
        points_centroid2: Tuple[torch.Tensor, torch.Tensor]
    ) -> bool:
        """Determine if instances are the same using only geometric criteria."""
        points1, centroid1 = points_centroid1
        points2, centroid2 = points_centroid2

        # Check centroid distance
        if compute_centroid_distance(centroid1, centroid2) > self.th_centroid:
            return False

        # Check point cloud proximity
        p_dist = compute_pcd_overlap(points1, points2, self.th_points)

        return p_dist > 0.5


def create_fusion_strategy(config: Dict[str, Any]) -> FusionStrategy:
    """
    Factory function to create the appropriate fusion strategy.

    Args:
        config: Configuration dictionary containing 'fusion_method' key

    Returns:
        FusionStrategy: The appropriate strategy instance

    Raises:
        ValueError: If fusion_method is not recognized
    """
    fusion_method = config.get("fusion_method", "clip").lower()

    strategy_map = {
        "clip": ("clip_feature", SemanticGeometricFusion),
        "dino": ("dino_feature", SemanticGeometricFusion),
        "pe": ("pe_feature", SemanticGeometricFusion),
        "geometric": (None, GeometricOnlyFusion),
    }

    if fusion_method not in strategy_map:
        raise ValueError(f"Unknown fusion method: {fusion_method}")

    feature_attr, strategy_class = strategy_map[fusion_method]

    if strategy_class == GeometricOnlyFusion:
        return strategy_class(config)
    else:
        return strategy_class(config, feature_attr=feature_attr)
