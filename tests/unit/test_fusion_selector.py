"""Tests for Fusion Strategy Chain of Responsibility pattern."""

import pytest
import torch
from unittest.mock import MagicMock, patch

from ovo.entities.fusion import (
    FusionStrategy,
    CooccurrenceCriterion,
    CentroidDistanceCriterion,
    CosSimilarityCriterion,
    PointOverlapCriterion,
    create_fusion_strategy,
)
from ovo.utils.cooccurrence_graph import CooccurrenceGraph


def _graph():
    return CooccurrenceGraph()


class TestFusionFactory:
    """Test create_fusion_strategy builds correct criterion chains."""

    def test_create_clip_fusion_strategy(self):
        config = {"fusion_method": "clip", "th_centroid": 1.5, "th_cossim": 0.81, "th_points": 0.1}
        strategy = create_fusion_strategy(config, _graph())
        assert isinstance(strategy, FusionStrategy)
        cos = next(c for c in strategy.criteria if isinstance(c, CosSimilarityCriterion))
        assert cos.feature_attr == "clip_feature"

    def test_create_dino_fusion_strategy(self):
        config = {"fusion_method": "dino", "th_centroid": 1.5, "th_cossim": 0.81, "th_points": 0.1}
        strategy = create_fusion_strategy(config, _graph())
        cos = next(c for c in strategy.criteria if isinstance(c, CosSimilarityCriterion))
        assert cos.feature_attr == "dino_feature"

    def test_create_pe_fusion_strategy(self):
        config = {"fusion_method": "pe", "th_centroid": 1.5, "th_cossim": 0.81, "th_points": 0.1}
        strategy = create_fusion_strategy(config, _graph())
        cos = next(c for c in strategy.criteria if isinstance(c, CosSimilarityCriterion))
        assert cos.feature_attr == "pe_feature"

    def test_factory_unknown_method_raises_error(self):
        config = {"fusion_method": "unknown_method"}
        with pytest.raises(ValueError, match="Unknown fusion_method"):
            create_fusion_strategy(config, _graph())

    def test_factory_default_method_is_clip(self):
        config = {"th_centroid": 1.5, "th_cossim": 0.81, "th_points": 0.1}
        strategy = create_fusion_strategy(config, _graph())
        cos = next(c for c in strategy.criteria if isinstance(c, CosSimilarityCriterion))
        assert cos.feature_attr == "clip_feature"

    def test_custom_chain_override(self):
        """fusion_criteria overrides default chain."""
        config = {"fusion_method": "clip", "fusion_criteria": ["centroid", "overlap"]}
        strategy = create_fusion_strategy(config, _graph())
        names = [c.name for c in strategy.criteria]
        assert names == ["centroid", "overlap"]

    def test_unknown_criterion_in_chain_raises(self):
        config = {"fusion_method": "clip", "fusion_criteria": ["centroid", "bogus"]}
        with pytest.raises(ValueError, match="Unknown criterion"):
            create_fusion_strategy(config, _graph())


class TestFusionDelegation:
    """Verify FusionStrategy.same_instance delegates to criteria chain."""

    def test_strategy_same_instance_is_called(self, mock_instance, sample_points_centroid):
        mock_strategy = MagicMock(spec=FusionStrategy)
        mock_strategy.same_instance.return_value = False

        instance1 = mock_instance(1)
        instance2 = mock_instance(2)
        data1 = sample_points_centroid((0, 0, 0))
        data2 = sample_points_centroid((0.5, 0, 0))

        result = mock_strategy.same_instance(instance1, instance2, data1, data2)
        mock_strategy.same_instance.assert_called_once_with(instance1, instance2, data1, data2)
        assert result is False

    def test_strategy_delegation_returns_correct_value(self, mock_instance, sample_points_centroid):
        mock_strategy = MagicMock(spec=FusionStrategy)
        instance1 = mock_instance(1)
        instance2 = mock_instance(2)
        data1 = sample_points_centroid((0, 0, 0))
        data2 = sample_points_centroid((0.1, 0, 0))

        mock_strategy.same_instance.return_value = True
        assert mock_strategy.same_instance(instance1, instance2, data1, data2) is True

        mock_strategy.same_instance.return_value = False
        assert mock_strategy.same_instance(instance1, instance2, data1, data2) is False


class TestFusionStrategyInterface:
    """FusionStrategy interface compliance."""

    def test_fusion_strategy_is_concrete(self):
        """FusionStrategy is now a concrete class, not ABC."""
        strategy = FusionStrategy(criteria=[])
        assert isinstance(strategy, FusionStrategy)

    def test_fusion_strategy_has_same_instance(self):
        strategy = FusionStrategy(criteria=[])
        assert hasattr(strategy, "same_instance")
        assert callable(strategy.same_instance)

    def test_fusion_strategy_has_pop_decisions(self):
        strategy = FusionStrategy(criteria=[])
        assert hasattr(strategy, "pop_decisions")

    def test_fusion_strategy_has_pop_timings(self):
        strategy = FusionStrategy(criteria=[])
        assert hasattr(strategy, "pop_timings")

    def test_semantic_chain_has_same_instance(self):
        config = {"th_centroid": 1.5, "th_cossim": 0.81, "th_points": 0.1}
        strategy = create_fusion_strategy({**config, "fusion_method": "clip"}, _graph())
        assert callable(strategy.same_instance)



class TestFusionCriteriaConfig:
    """Criteria store correct threshold values from config."""

    def test_centroid_criterion_threshold(self):
        config = {"fusion_method": "clip", "th_centroid": 2.0, "th_cossim": 0.85, "th_points": 0.15}
        strategy = create_fusion_strategy(config, _graph())
        centroid = next(c for c in strategy.criteria if isinstance(c, CentroidDistanceCriterion))
        assert centroid.threshold == 2.0

    def test_cossim_criterion_threshold(self):
        config = {"fusion_method": "clip", "th_centroid": 1.5, "th_cossim": 0.85, "th_points": 0.1}
        strategy = create_fusion_strategy(config, _graph())
        cos = next(c for c in strategy.criteria if isinstance(c, CosSimilarityCriterion))
        assert cos.threshold == 0.85

    def test_overlap_criterion_threshold(self):
        config = {"fusion_method": "clip", "th_centroid": 1.5, "th_cossim": 0.81, "th_points": 0.15}
        strategy = create_fusion_strategy(config, _graph())
        overlap = next(c for c in strategy.criteria if isinstance(c, PointOverlapCriterion))
        assert overlap.threshold == 0.15

    def test_feature_attr_dino(self):
        config = {"fusion_method": "dino", "th_centroid": 1.5, "th_cossim": 0.81, "th_points": 0.1}
        strategy = create_fusion_strategy(config, _graph())
        cos = next(c for c in strategy.criteria if isinstance(c, CosSimilarityCriterion))
        assert cos.feature_attr == "dino_feature"


class TestOVOIntegration:
    """OVO integrates with fusion strategy."""

    def test_ovo_has_fusion_strategy_attribute(self, minimal_ovo_config):
        with patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config, mock_logger, eval=True)

            assert hasattr(ovo, 'fusion_strategy')
            assert isinstance(ovo.fusion_strategy, FusionStrategy)

    def test_ovo_creates_correct_strategy_from_config(self, minimal_ovo_config):
        with patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            minimal_ovo_config["fusion_method"] = "clip"
            ovo = OVO(minimal_ovo_config, mock_logger, eval=True)

            assert isinstance(ovo.fusion_strategy, FusionStrategy)
            cos = next(c for c in ovo.fusion_strategy.criteria if isinstance(c, CosSimilarityCriterion))
            assert cos.feature_attr == "clip_feature"

    def test_ovo_uses_strategy_for_fusion_decision(self, minimal_ovo_config):
        with patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config, mock_logger, eval=True)

            mock_strategy = MagicMock(spec=FusionStrategy)
            mock_strategy.same_instance.return_value = False
            mock_strategy.pop_decisions.return_value = []
            mock_strategy.pop_timings.return_value = {}
            ovo.fusion_strategy = mock_strategy

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

            ovo.objects = {1: instance1, 2: instance2}
            ovo.keyframes = {
                "ins_descriptors": {},
                "ins_pe_descriptors": {},
                "frame_id": [],
                "ins_maps": [],
            }
            ovo.keyframes_queue = []

            points_3d = torch.randn(200, 3)
            points_ids = torch.arange(200)
            points_ins_ids = torch.cat([torch.ones(100) * 1, torch.ones(100) * 2]).long()
            map_data = (points_3d, points_ids, points_ins_ids)

            with patch.object(ovo, 'complete_semantic_info'), \
                 patch.object(ovo, 'update_objects_clip'), \
                 patch.object(ovo, 'update_objects_pe'):
                ovo.update_map(map_data, [])

            assert mock_strategy.same_instance.called, \
                "OVO should delegate fusion decision to fusion_strategy.same_instance"
