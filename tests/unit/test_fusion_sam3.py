"""Tests for SAM3 fusion strategy integration."""

import pytest
import torch
from unittest.mock import MagicMock

from ovo.entities.fusion import (
    FusionStrategy,
    SemanticGeometricFusion,
    create_fusion_strategy
)


class TestFusionFactorySAM3:
    """Test fusion factory creates SAM3 strategy correctly."""

    def test_create_sam3_fusion_strategy(self):
        """Factory should return SemanticGeometricFusion for 'sam3' config."""
        config = {
            "fusion_method": "sam3",
            "th_centroid": 1.5,
            "th_cossim": 0.81,
            "th_points": 0.1
        }
        strategy = create_fusion_strategy(config)

        assert isinstance(strategy, SemanticGeometricFusion)
        assert strategy.feature_attr == "sam3_feature"

    def test_sam3_strategy_stores_thresholds(self):
        """SAM3 strategy should store threshold values from config."""
        config = {
            "fusion_method": "sam3",
            "th_centroid": 2.0,
            "th_cossim": 0.9,
            "th_points": 0.2
        }
        strategy = create_fusion_strategy(config)

        assert strategy.th_centroid == 2.0
        assert strategy.th_cossim == 0.9
        assert strategy.th_points == 0.2


class TestSemanticGeometricFusionSAM3:
    """Test SemanticGeometricFusion works with sam3_feature."""

    def test_same_instance_reads_sam3_feature(self, mock_instance_sam3, sample_points_centroid):
        """same_instance should read sam3_feature from instances."""
        config = {
            "th_centroid": 10.0,  # High threshold to pass
            "th_cossim": 0.5,    # Low threshold to pass
            "th_points": 1.0     # High distance threshold for overlap
        }
        strategy = SemanticGeometricFusion(config, feature_attr="sam3_feature")

        # Create instances with similar SAM3 features
        instance1 = mock_instance_sam3(1)
        instance2 = mock_instance_sam3(2)
        instance1.sam3_feature = torch.tensor([[1.0, 0.0, 0.0]])
        instance2.sam3_feature = torch.tensor([[1.0, 0.0, 0.0]])  # Same

        data1 = sample_points_centroid((0, 0, 0))
        data2 = sample_points_centroid((0.01, 0, 0))  # Very Close

        result = strategy.same_instance(instance1, instance2, data1, data2)

        # Should fuse because features are identical and points are very close
        assert result == True

    def test_same_instance_rejects_dissimilar_sam3_features(self, mock_instance_sam3, sample_points_centroid):
        """same_instance should reject instances with dissimilar sam3_features."""
        config = {
            "th_centroid": 10.0,
            "th_cossim": 0.9,  # High threshold
            "th_points": 0.01
        }
        strategy = SemanticGeometricFusion(config, feature_attr="sam3_feature")

        instance1 = mock_instance_sam3(1)
        instance2 = mock_instance_sam3(2)
        instance1.sam3_feature = torch.tensor([[1.0, 0.0, 0.0]])
        instance2.sam3_feature = torch.tensor([[0.0, 1.0, 0.0]])  # Orthogonal

        data1 = sample_points_centroid((0, 0, 0))
        data2 = sample_points_centroid((0.1, 0, 0))

        result = strategy.same_instance(instance1, instance2, data1, data2)

        assert result == False

    def test_same_instance_handles_1024_dim_features(self, mock_instance_sam3, sample_points_centroid):
        """same_instance should work with 1024-dimensional SAM3 features."""
        config = {
            "th_centroid": 10.0,
            "th_cossim": 0.8,
            "th_points": 0.01
        }
        strategy = SemanticGeometricFusion(config, feature_attr="sam3_feature")

        # Create normalized 1024-dim features
        feat1 = torch.randn(1, 1024)
        feat1 = feat1 / feat1.norm()
        feat2 = feat1 + 0.1 * torch.randn(1, 1024)  # Slightly perturbed
        feat2 = feat2 / feat2.norm()

        instance1 = mock_instance_sam3(1)
        instance2 = mock_instance_sam3(2)
        instance1.sam3_feature = feat1
        instance2.sam3_feature = feat2

        data1 = sample_points_centroid((0, 0, 0))
        data2 = sample_points_centroid((0.1, 0, 0))

        # Should not raise any dimension errors
        result = strategy.same_instance(instance1, instance2, data1, data2)
        assert isinstance(result, bool)
