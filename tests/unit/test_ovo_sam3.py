"""Tests for OVO SAM3 integration."""

import pytest
import torch
import numpy as np
from collections import deque
from unittest.mock import MagicMock, patch


class TestOVOSAM3Initialization:
    """Test OVO initializes SAM3Generator correctly."""

    def test_ovo_initializes_sam3_generator_when_config_present(self, minimal_ovo_config_sam3):
        """OVO should create SAM3Generator when 'sam3' key is in config."""
        with patch('ovo.entities.ovo.SAM3Generator') as MockSAM3Gen, \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            MockSAM3Gen.assert_called_once()
            assert ovo.sam3_generator is not None

    def test_ovo_sam3_generator_none_when_no_config(self, minimal_ovo_config):
        """OVO should have sam3_generator=None when 'sam3' not in config."""
        with patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            # minimal_ovo_config doesn't have 'sam3' key
            ovo = OVO(minimal_ovo_config, mock_logger, eval=True)

            assert ovo.sam3_generator is None

    def test_ovo_creates_sam3_fusion_strategy(self, minimal_ovo_config_sam3):
        """OVO should create SAM3 fusion strategy when fusion_method='sam3'."""
        with patch('ovo.entities.ovo.SAM3Generator'), \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger
            from ovo.entities.fusion import SemanticGeometricFusion

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            assert isinstance(ovo.fusion_strategy, SemanticGeometricFusion)
            assert ovo.fusion_strategy.feature_attr == "sam3_feature"


class TestOVOSAM3Keyframes:
    """Test OVO manages SAM3 keyframe descriptors."""

    def test_ovo_has_ins_sam3_descriptors_in_keyframes(self, minimal_ovo_config_sam3):
        """OVO.keyframes should have 'ins_sam3_descriptors' key."""
        with patch('ovo.entities.ovo.SAM3Generator'), \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            assert "ins_sam3_descriptors" in ovo.keyframes

    def test_ovo_stores_sam3_descriptors_per_keyframe(self, minimal_ovo_config_sam3):
        """OVO should store SAM3 descriptors indexed by keyframe ID."""
        with patch('ovo.entities.ovo.SAM3Generator') as MockSAM3Gen, \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            # Setup mock SAM3 generator
            mock_sam3 = MagicMock()
            mock_sam3.extract_sam3.return_value = torch.randn(5, 1024)
            MockSAM3Gen.return_value = mock_sam3

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            # Simulate adding descriptors for keyframe 0
            kf_id = 0
            descriptors = torch.randn(5, 1024)
            ovo.keyframes["ins_sam3_descriptors"][kf_id] = {
                i: descriptors[i:i+1] for i in range(5)
            }

            assert kf_id in ovo.keyframes["ins_sam3_descriptors"]
            assert len(ovo.keyframes["ins_sam3_descriptors"][kf_id]) == 5


class TestOVOUpdateObjectsSAM3:
    """Test OVO.update_objects_sam3() method."""

    def test_ovo_has_update_objects_sam3_method(self, minimal_ovo_config_sam3):
        """OVO should have update_objects_sam3 method."""
        with patch('ovo.entities.ovo.SAM3Generator'), \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            assert hasattr(ovo, 'update_objects_sam3')
            assert callable(ovo.update_objects_sam3)

    def test_update_objects_sam3_calls_instance_update(self, minimal_ovo_config_sam3):
        """update_objects_sam3 should call update_sam3 on each instance."""
        with patch('ovo.entities.ovo.SAM3Generator'), \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            # Create mock instances
            mock_instance1 = MagicMock()
            mock_instance2 = MagicMock()
            ovo.objects = {1: mock_instance1, 2: mock_instance2}

            ovo.update_objects_sam3()

            mock_instance1.update_sam3.assert_called_once()
            mock_instance2.update_sam3.assert_called_once()

    def test_update_objects_sam3_passes_keyframes(self, minimal_ovo_config_sam3):
        """update_objects_sam3 should pass ins_sam3_descriptors to instances."""
        with patch('ovo.entities.ovo.SAM3Generator'), \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            # Setup keyframes with SAM3 descriptors
            ovo.keyframes["ins_sam3_descriptors"] = {
                0: {1: torch.randn(1, 1024)}
            }

            mock_instance = MagicMock()
            ovo.objects = {1: mock_instance}

            ovo.update_objects_sam3()

            # Verify keyframes dict was passed
            call_args = mock_instance.update_sam3.call_args
            assert call_args[0][0] == ovo.keyframes["ins_sam3_descriptors"]


class TestOVOMapUpdateWithSAM3:
    """Test OVO.update_map integrates SAM3 feature extraction."""

    def test_update_map_extracts_sam3_features(self, minimal_ovo_config_sam3):
        """update_map should extract SAM3 features for new keyframes."""
        with patch('ovo.entities.ovo.SAM3Generator') as MockSAM3Gen, \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_sam3 = MagicMock()
            mock_sam3.extract_sam3.return_value = torch.randn(3, 1024)
            MockSAM3Gen.return_value = mock_sam3

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            # Setup for update_map
            # update_map calls complete_semantic_info which pops from keyframes_queue
            # and calls _compute_semantic_info which calls _extract_sam3
            ovo.keyframes_queue = deque([[
                [1, 2, 3], # matched_ins_ids
                torch.zeros(3, 100, 100), # binary_maps
                np.zeros((100, 100, 3)), # image
                0 # kf_id
            ]])
            ovo.keyframes["frame_id"] = [0]
            
            with patch.object(ovo, '_extract_clip', return_value=torch.randn(3, 512)), \
                 patch.object(ovo, '_update_matched_objects_clip'), \
                 patch.object(ovo, '_update_matched_objects_sam3'), \
                 patch.object(ovo, 'update_objects_clip'), \
                 patch.object(ovo, 'update_objects_pe'), \
                 patch.object(ovo, 'update_objects_sam3'):

                points_3d = torch.randn(100, 3)
                points_ids = torch.arange(100)
                points_ins_ids = torch.ones(100).long()
                
                ovo.update_map((points_3d, points_ids, points_ins_ids), [])

                # Verify SAM3 extraction was called
                mock_sam3.extract_sam3.assert_called()
