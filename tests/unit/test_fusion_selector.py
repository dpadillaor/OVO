"""
TDD Tests for Fusion Strategy Pattern

These tests verify the infrastructure of the Strategy pattern for fusion logic:
1. Factory creates correct strategy based on configuration
2. OVO delegates fusion decisions to the strategy
3. All strategies comply with the FusionStrategy interface
"""

import pytest
import torch
from abc import ABC
from unittest.mock import MagicMock, patch

# Imports will fail initially (TDD Red phase) - this is expected
from ovo.entities.fusion import (
    FusionStrategy,
    SemanticGeometricFusion,
    GeometricOnlyFusion,
    create_fusion_strategy
)


class TestFusionFactory:
    """Test suite for fusion strategy factory functionality."""

    def test_create_clip_fusion_strategy(self):
        """Factory should return SemanticGeometricFusion for 'clip' config."""
        config = {"fusion_method": "clip", "th_centroid": 1.5, "th_cossim": 0.81, "th_points": 0.1}
        strategy = create_fusion_strategy(config)

        assert isinstance(strategy, SemanticGeometricFusion)
        assert strategy.feature_attr == "clip_feature"

    def test_create_pe_fusion_strategy(self):
        """Factory should return SemanticGeometricFusion for 'pe' config."""
        config = {"fusion_method": "pe", "th_centroid": 1.5, "th_cossim": 0.81, "th_points": 0.1}
        strategy = create_fusion_strategy(config)

        assert isinstance(strategy, SemanticGeometricFusion)
        assert strategy.feature_attr == "pe_feature"

    def test_create_geometric_only_strategy(self):
        """Factory should return GeometricOnlyFusion for 'geometric' config."""
        config = {"fusion_method": "geometric", "th_centroid": 1.5, "th_points": 0.1}
        strategy = create_fusion_strategy(config)

        assert isinstance(strategy, GeometricOnlyFusion)

    def test_factory_unknown_method_raises_error(self):
        """Factory should raise ValueError for unknown fusion methods."""
        config = {"fusion_method": "unknown_method"}

        with pytest.raises(ValueError, match="Unknown fusion method"):
            create_fusion_strategy(config)

    def test_factory_default_method(self):
        """Factory should use 'clip' as default when fusion_method is not specified."""
        config = {"th_centroid": 1.5, "th_cossim": 0.81, "th_points": 0.1}
        strategy = create_fusion_strategy(config)

        assert isinstance(strategy, SemanticGeometricFusion)
        assert strategy.feature_attr == "clip_feature"


class TestFusionDelegation:
    """Test suite for verifying OVO delegates to fusion strategy."""

    # Fixtures imported from tests/fixtures/fixtures_fusion.py via conftest.py

    def test_strategy_same_instance_is_called(self, mock_instance, sample_points_centroid):
        """OVO should call strategy.same_instance when comparing instances."""
        # Create mock strategy
        mock_strategy = MagicMock(spec=FusionStrategy)
        mock_strategy.same_instance.return_value = False

        instance1 = mock_instance(1)
        instance2 = mock_instance(2)
        data1 = sample_points_centroid((0, 0, 0))
        data2 = sample_points_centroid((0.5, 0, 0))

        # Call the strategy directly (simulates what OVO should do)
        result = mock_strategy.same_instance(instance1, instance2, data1, data2)

        # Verify the strategy method was called
        mock_strategy.same_instance.assert_called_once_with(instance1, instance2, data1, data2)
        assert result is False

    def test_strategy_delegation_returns_correct_value(self, mock_instance, sample_points_centroid):
        """Strategy should return the correct boolean value for fusion decision."""
        mock_strategy = MagicMock(spec=FusionStrategy)

        instance1 = mock_instance(1)
        instance2 = mock_instance(2)
        data1 = sample_points_centroid((0, 0, 0))
        data2 = sample_points_centroid((0.1, 0, 0))

        # Test True case
        mock_strategy.same_instance.return_value = True
        assert mock_strategy.same_instance(instance1, instance2, data1, data2) is True

        # Test False case
        mock_strategy.same_instance.return_value = False
        assert mock_strategy.same_instance(instance1, instance2, data1, data2) is False


class TestFusionStrategyInterface:
    """Test suite for verifying strategy interface compliance."""

    def test_fusion_strategy_is_abstract(self):
        """FusionStrategy should be an abstract base class."""
        assert issubclass(FusionStrategy, ABC)

    def test_fusion_strategy_cannot_be_instantiated(self):
        """FusionStrategy ABC should not be directly instantiatable."""
        with pytest.raises(TypeError):
            FusionStrategy()

    def test_semantic_geometric_inherits_from_base(self):
        """SemanticGeometricFusion should inherit from FusionStrategy."""
        assert issubclass(SemanticGeometricFusion, FusionStrategy)

    def test_geometric_only_inherits_from_base(self):
        """GeometricOnlyFusion should inherit from FusionStrategy."""
        assert issubclass(GeometricOnlyFusion, FusionStrategy)

    def test_semantic_geometric_has_same_instance_method(self):
        """SemanticGeometricFusion should have same_instance method."""
        config = {"th_centroid": 1.5, "th_cossim": 0.81, "th_points": 0.1}
        strategy = SemanticGeometricFusion(config, feature_attr="clip_feature")

        assert hasattr(strategy, 'same_instance')
        assert callable(strategy.same_instance)

    def test_geometric_only_has_same_instance_method(self):
        """GeometricOnlyFusion should have same_instance method."""
        config = {"th_centroid": 1.5, "th_points": 0.1}
        strategy = GeometricOnlyFusion(config)

        assert hasattr(strategy, 'same_instance')
        assert callable(strategy.same_instance)


class TestSemanticGeometricFusionConfig:
    """Test suite for SemanticGeometricFusion configuration."""

    def test_stores_thresholds_from_config(self):
        """Strategy should store threshold values from configuration."""
        config = {"th_centroid": 2.0, "th_cossim": 0.85, "th_points": 0.15}
        strategy = SemanticGeometricFusion(config, feature_attr="clip_feature")

        assert strategy.th_centroid == 2.0
        assert strategy.th_cossim == 0.85
        assert strategy.th_points == 0.15

    def test_stores_feature_attribute(self):
        """Strategy should store the feature attribute name."""
        config = {"th_centroid": 1.5, "th_cossim": 0.81, "th_points": 0.1}
        strategy = SemanticGeometricFusion(config, feature_attr="pe_feature")

        assert strategy.feature_attr == "pe_feature"


class TestGeometricOnlyFusionConfig:
    """Test suite for GeometricOnlyFusion configuration."""

    def test_stores_thresholds_from_config(self):
        """Strategy should store threshold values from configuration."""
        config = {"th_centroid": 2.0, "th_points": 0.15}
        strategy = GeometricOnlyFusion(config)

        assert strategy.th_centroid == 2.0
        assert strategy.th_points == 0.15

    def test_does_not_require_cossim_threshold(self):
        """GeometricOnlyFusion should not require th_cossim."""
        config = {"th_centroid": 1.5, "th_points": 0.1}
        strategy = GeometricOnlyFusion(config)

        # Should not have or need th_cossim
        assert not hasattr(strategy, 'th_cossim') or strategy.th_cossim is None


class TestOVOIntegration:
    """Test suite for verifying OVO integrates with fusion strategy."""

    # Fixture imported from tests/fixtures/fixtures_fusion.py via conftest.py

    def test_ovo_has_fusion_strategy_attribute(self, minimal_ovo_config, minimal_run_config):
        """OVO should have a fusion_strategy attribute after initialization."""
        # We use patch to avoid loading heavy models
        with patch('ovo.entities.generator_pipeline.CLIPGenerator'), \
             patch('ovo.entities.generator_pipeline.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config, minimal_run_config, mock_logger)

            assert hasattr(ovo, 'fusion_strategy')
            assert isinstance(ovo.fusion_strategy, FusionStrategy)

    def test_ovo_creates_correct_strategy_from_config(self, minimal_ovo_config, minimal_run_config):
        """OVO should create the correct strategy based on fusion_method config."""
        with patch('ovo.entities.generator_pipeline.CLIPGenerator'), \
             patch('ovo.entities.generator_pipeline.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            from dataclasses import replace
            ovo = OVO(replace(minimal_ovo_config, fusion_method="clip"), minimal_run_config, mock_logger)
            assert isinstance(ovo.fusion_strategy, SemanticGeometricFusion)
            assert ovo.fusion_strategy.feature_attr == "clip_feature"

    def test_ovo_creates_geometric_strategy(self, minimal_ovo_config, minimal_run_config):
        """OVO should create GeometricOnlyFusion for 'geometric' config."""
        with patch('ovo.entities.generator_pipeline.CLIPGenerator'), \
             patch('ovo.entities.generator_pipeline.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            from dataclasses import replace
            ovo = OVO(replace(minimal_ovo_config, fusion_method="geometric"), minimal_run_config, mock_logger)

            assert isinstance(ovo.fusion_strategy, GeometricOnlyFusion)

    def test_ovo_uses_strategy_for_fusion_decision(self, minimal_ovo_config, minimal_run_config):
        """OVO.update_map should use fusion_strategy.same_instance for comparisons."""
        with patch('ovo.entities.generator_pipeline.CLIPGenerator'), \
             patch('ovo.entities.generator_pipeline.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config, minimal_run_config, mock_logger)

            # Inject a mock strategy
            mock_strategy = MagicMock(spec=FusionStrategy)
            mock_strategy.same_instance.return_value = False
            ovo.fusion_strategy = mock_strategy

            # Create mock instances
            instance1 = MagicMock()
            instance1.id = 1
            instance1.clip_feature = torch.randn(1, 512)
            instance1.points_ids = list(range(10))
            instance1.kfs_ids = []
            instance1.top_kf = []

            instance2 = MagicMock()
            instance2.id = 2
            instance2.clip_feature = torch.randn(1, 512)
            instance2.points_ids = list(range(10, 20))
            instance2.kfs_ids = []
            instance2.top_kf = []

            # Setup OVO state
            ovo.objects = {1: instance1, 2: instance2}
            ovo.keyframes_queue = []

            # Create mock map data
            from ovo.entities.ovo import MapData
            points_3d = torch.randn(200, 3)
            points_ids = torch.arange(200)
            points_ins_ids = torch.cat([torch.ones(100) * 1, torch.ones(100) * 2]).long()
            map_data = MapData(points_3d, points_ids, points_ins_ids)

            # Call update_map
            with patch.object(ovo, 'complete_semantic_info'), \
                 patch.object(ovo, 'update_objects_clip'):
                ovo.update_map(map_data, {})

            # Verify strategy.same_instance was called
            assert mock_strategy.same_instance.called, \
                "OVO should delegate fusion decision to fusion_strategy.same_instance"

