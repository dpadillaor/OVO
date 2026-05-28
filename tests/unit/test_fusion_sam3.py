"""Tests for SAM3 fusion strategy integration."""

import pytest
import torch
from unittest.mock import MagicMock, patch

from ovo.entities.fusion import (
    FusionStrategy,
    CosSimilarityCriterion,
    create_fusion_strategy,
)
from ovo.utils.cooccurrence_graph import CooccurrenceGraph


def _graph():
    return CooccurrenceGraph()


class TestFusionFactorySAM3:
    """Test fusion factory creates SAM3 strategy correctly."""

    def test_create_sam3_fusion_strategy(self):
        config = {"fusion_method": "sam3", "th_centroid": 1.5, "th_cossim": 0.81, "th_points": 0.1}
        strategy = create_fusion_strategy(config, _graph())
        assert isinstance(strategy, FusionStrategy)
        cos = next(c for c in strategy.criteria if isinstance(c, CosSimilarityCriterion))
        assert cos.feature_attr == "sam3_feature"

    def test_sam3_strategy_stores_thresholds(self):
        config = {"fusion_method": "sam3", "th_centroid": 2.0, "th_cossim": 0.9, "th_points": 0.2}
        strategy = create_fusion_strategy(config, _graph())
        from ovo.entities.fusion import CentroidDistanceCriterion, PointOverlapCriterion
        centroid = next(c for c in strategy.criteria if isinstance(c, CentroidDistanceCriterion))
        cos = next(c for c in strategy.criteria if isinstance(c, CosSimilarityCriterion))
        overlap = next(c for c in strategy.criteria if isinstance(c, PointOverlapCriterion))
        assert centroid.threshold == 2.0
        assert cos.threshold == 0.9
        assert overlap.threshold == 0.2


class TestSAM3FusionBehavior:
    """Test fusion strategy works correctly with sam3_feature."""

    def _make_strategy(self, th_centroid=10.0, th_cossim=0.5, th_points=1.0):
        config = {
            "fusion_method": "sam3",
            "fusion_criteria": ["centroid", "cos_sim", "overlap"],
            "th_centroid": th_centroid,
            "th_cossim": th_cossim,
            "th_points": th_points,
        }
        return create_fusion_strategy(config, _graph())

    def test_same_instance_reads_sam3_feature(self, mock_instance_sam3, sample_points_centroid):
        strategy = self._make_strategy()

        instance1 = mock_instance_sam3(1)
        instance2 = mock_instance_sam3(2)
        instance1.sam3_feature = torch.tensor([[1.0, 0.0, 0.0]])
        instance2.sam3_feature = torch.tensor([[1.0, 0.0, 0.0]])

        data1 = sample_points_centroid((0, 0, 0))
        data2 = sample_points_centroid((0.01, 0, 0))

        result = strategy.same_instance(instance1, instance2, data1, data2)
        assert result is True

    def test_same_instance_rejects_dissimilar_sam3_features(self, mock_instance_sam3, sample_points_centroid):
        strategy = self._make_strategy(th_cossim=0.9)

        instance1 = mock_instance_sam3(1)
        instance2 = mock_instance_sam3(2)
        instance1.sam3_feature = torch.tensor([[1.0, 0.0, 0.0]])
        instance2.sam3_feature = torch.tensor([[0.0, 1.0, 0.0]])  # orthogonal

        data1 = sample_points_centroid((0, 0, 0))
        data2 = sample_points_centroid((0.1, 0, 0))

        result = strategy.same_instance(instance1, instance2, data1, data2)
        assert result is False

    def test_same_instance_handles_1024_dim_features(self, mock_instance_sam3, sample_points_centroid):
        strategy = self._make_strategy(th_cossim=0.8)

        feat1 = torch.randn(1, 1024)
        feat1 = feat1 / feat1.norm()
        feat2 = feat1 + 0.1 * torch.randn(1, 1024)
        feat2 = feat2 / feat2.norm()

        instance1 = mock_instance_sam3(1)
        instance2 = mock_instance_sam3(2)
        instance1.sam3_feature = feat1
        instance2.sam3_feature = feat2

        data1 = sample_points_centroid((0, 0, 0))
        data2 = sample_points_centroid((0.1, 0, 0))

        result = strategy.same_instance(instance1, instance2, data1, data2)
        assert isinstance(result, bool)
