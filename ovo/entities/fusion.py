"""
Fusion Strategy Pattern for Instance Comparison

This module implements the Strategy pattern for instance fusion logic,
allowing dynamic selection of different fusion algorithms (CLIP, DINO, PE, geometric).
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from ..utils.instance_utils import compute_pcd_overlap, compute_centroid_distance

import logging
import torch
import numpy as np
import open3d as o3d

logger = logging.getLogger("ovo.fusion")

class FusionStrategy(ABC):
    """Abstract base class for fusion strategies."""

    def __init__(self):
        self._decisions: list = []

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

    def pop_decisions(self) -> list:
        """Return accumulated per-pair decisions and clear the buffer."""
        decisions, self._decisions = self._decisions, []
        return decisions


class SemanticGeometricFusion(FusionStrategy):
    """
    Semantic + Geometric fusion strategy.

    Uses both semantic features (CLIP, DINO, PE) and geometric proximity
    to determine if instances should be fused.
    """

    def __init__(self, config: Dict[str, Any], feature_attr: str):
        super().__init__()
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
        i1, i2 = instance1.id, instance2.id

        centroid_dist = compute_centroid_distance(centroid1, centroid2)
        if centroid_dist > self.th_centroid:
            logger.debug("REJECTED i1=%s i2=%s | centroid_dist=%.3f > th=%.3f", i1, i2, centroid_dist, self.th_centroid)
            self._decisions.append({"result": "REJECTED", "i1": i1, "i2": i2, "reason": "centroid", "centroid_dist": float(centroid_dist)})
            return False

        feature1 = getattr(instance1, self.feature_attr)[0]
        feature2 = getattr(instance2, self.feature_attr)[0]
        cos_sim = torch.nn.functional.cosine_similarity(feature1, feature2, dim=0).item()
        if cos_sim < self.th_cossim:
            logger.debug("REJECTED i1=%s i2=%s | cos_sim=%.3f < th=%.3f (centroid_dist=%.3f)", i1, i2, cos_sim, self.th_cossim, centroid_dist)
            self._decisions.append({"result": "REJECTED", "i1": i1, "i2": i2, "reason": "cos_sim", "centroid_dist": float(centroid_dist), "cos_sim": cos_sim})
            return False

        p_dist = compute_pcd_overlap(points1, points2, self.th_points)
        result = p_dist > 0.5 or (cos_sim > 0.9 and p_dist > 0.2)
        if result:
            logger.debug("ACCEPTED i1=%s i2=%s | cos_sim=%.3f centroid_dist=%.3f p_dist=%.3f", i1, i2, cos_sim, centroid_dist, p_dist)
            self._decisions.append({"result": "ACCEPTED", "i1": i1, "i2": i2, "centroid_dist": float(centroid_dist), "cos_sim": cos_sim, "p_dist": float(p_dist)})
        else:
            logger.debug("REJECTED i1=%s i2=%s | overlap: p_dist=%.3f (cos_sim=%.3f centroid_dist=%.3f)", i1, i2, p_dist, cos_sim, centroid_dist)
            self._decisions.append({"result": "REJECTED", "i1": i1, "i2": i2, "reason": "overlap", "centroid_dist": float(centroid_dist), "cos_sim": cos_sim, "p_dist": float(p_dist)})
        return result


class GeometricOnlyFusion(FusionStrategy):
    """
    Geometric-only fusion strategy.

    Uses only spatial overlap and distance to determine if instances should be fused.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
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
        i1, i2 = instance1.id, instance2.id

        centroid_dist = compute_centroid_distance(centroid1, centroid2)
        if centroid_dist > self.th_centroid:
            logger.debug("REJECTED i1=%s i2=%s | centroid_dist=%.3f > th=%.3f", i1, i2, centroid_dist, self.th_centroid)
            self._decisions.append({"result": "REJECTED", "i1": i1, "i2": i2, "reason": "centroid", "centroid_dist": float(centroid_dist)})
            return False

        p_dist = compute_pcd_overlap(points1, points2, self.th_points)
        if p_dist > 0.5:
            logger.debug("ACCEPTED i1=%s i2=%s | centroid_dist=%.3f p_dist=%.3f", i1, i2, centroid_dist, p_dist)
            self._decisions.append({"result": "ACCEPTED", "i1": i1, "i2": i2, "centroid_dist": float(centroid_dist), "p_dist": float(p_dist)})
            return True
        logger.debug("REJECTED i1=%s i2=%s | overlap: p_dist=%.3f < 0.5 (centroid_dist=%.3f)", i1, i2, p_dist, centroid_dist)
        self._decisions.append({"result": "REJECTED", "i1": i1, "i2": i2, "reason": "overlap", "centroid_dist": float(centroid_dist), "p_dist": float(p_dist)})
        return False


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
        "sam3": ("sam3_feature", SemanticGeometricFusion),
        "geometric": (None, GeometricOnlyFusion),
    }

    if fusion_method not in strategy_map:
        raise ValueError(f"Unknown fusion method: {fusion_method}")

    feature_attr, strategy_class = strategy_map[fusion_method]

    if strategy_class == GeometricOnlyFusion:
        return strategy_class(config)
    else:
        return strategy_class(config, feature_attr=feature_attr)
