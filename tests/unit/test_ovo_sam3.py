"""
Unit tests for OVO's integration with SAM3FusionAdapter.

These tests verify that OVO correctly initializes and delegates to the
SAM3 fusion adapter, following the same pattern as PE integration.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from ovo.entities.ovo import OVO


class TestOVOSAM3Initialization:
    """Test OVO initializes SAM3Generator and adapter correctly."""

    def test_ovo_initializes_sam3_generator_when_config_present(self, minimal_ovo_config_sam3):
        """OVO should create SAM3Generator when 'sam3' key is in config."""
        with patch('ovo.entities.ovo.SAM3Generator') as MockSAM3Gen, \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            MockSAM3Gen.assert_called_once()
            assert ovo.sam3_generator is not None

    def test_ovo_sam3_generator_none_when_no_config(self, minimal_ovo_config):
        """OVO should have sam3_generator=None when 'sam3' not in config."""
        with patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config, mock_logger, eval=True)

            assert ovo.sam3_generator is None

    def test_ovo_creates_sam3_fusion_strategy(self, minimal_ovo_config_sam3):
        """OVO should create SAM3 fusion strategy when fusion_method='sam3'."""
        with patch('ovo.entities.ovo.SAM3Generator'), \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.logger import Logger
            from ovo.entities.fusion import SemanticGeometricFusion

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            assert isinstance(ovo.fusion_strategy, SemanticGeometricFusion)
            assert ovo.fusion_strategy.feature_attr == "sam3_feature"


class TestOVOSAM3FusionAdapter:
    """Test OVO uses SAM3FusionAdapter correctly."""

    def test_get_fusion_encoder_returns_sam3_adapter(self, minimal_ovo_config_sam3):
        """_get_fusion_encoder should return SAM3FusionAdapter when method is 'sam3'."""
        with patch('ovo.entities.ovo.SAM3Generator') as MockSAM3Gen, \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.logger import Logger
            from ovo.entities.fusion_encoders import SAM3FusionAdapter

            mock_sam3_gen = MagicMock()
            MockSAM3Gen.return_value = mock_sam3_gen

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            # Verify adapter is SAM3FusionAdapter
            assert isinstance(ovo.fusion_encoder, SAM3FusionAdapter)
            assert ovo.fusion_encoder.generator == mock_sam3_gen
            assert ovo.fusion_encoder.storage_key == "ins_sam3_descriptors"

    def test_validate_fusion_config_raises_on_missing_generator(self, minimal_ovo_config_sam3):
        """Should raise ValueError if fusion_method='sam3' but generator missing."""
        with patch('ovo.entities.ovo.SAM3Generator') as MockSAM3Gen, \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.logger import Logger

            # Make SAM3Generator return None
            MockSAM3Gen.return_value = None

            mock_logger = MagicMock(spec=Logger)

            # This should raise during OVO init when it calls _validate_fusion_config
            with pytest.raises(ValueError, match="requires SAM3 generator"):
                ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)


class TestOVOSAM3DelegationToAdapter:
    """Test OVO delegates SAM3 operations to adapter."""

    def test_compute_semantic_info_delegates_to_sam3_adapter(self, minimal_ovo_config_sam3):
        """_compute_semantic_info should call SAM3 adapter.compute_and_update."""
        with patch('ovo.entities.ovo.SAM3Generator') as MockSAM3Gen, \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.logger import Logger

            # Setup mocks
            mock_sam3_gen = MagicMock()
            MockSAM3Gen.return_value = mock_sam3_gen

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            # Mock adapter
            mock_adapter = MagicMock()
            ovo.fusion_encoder = mock_adapter

            # Mock queue return
            image = torch.randn(100, 100, 3).numpy()
            binary_maps = torch.ones(2, 100, 100)
            matched_ids = [1, 2]
            kf_id = 5
            ovo.keyframes_queue.popleft = MagicMock(return_value=(matched_ids, binary_maps, image, kf_id))

            # Mock CLIP extraction (always runs)
            ovo._extract_clip = MagicMock(return_value=torch.randn(2, 512))
            ovo._update_matched_objects_clip = MagicMock()

            # Call method
            ovo._compute_semantic_info()

            # Assertions
            ovo._extract_clip.assert_called() # CLIP always runs
            mock_adapter.compute_and_update.assert_called_with(
                image, binary_maps, matched_ids, kf_id, ovo.keyframes, ovo.objects
            )

    def test_update_map_delegates_update_to_sam3_adapter(self, minimal_ovo_config_sam3):
        """update_map should call SAM3 adapter.update_objects."""
        with patch('ovo.entities.ovo.SAM3Generator') as MockSAM3Gen, \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.logger import Logger

            mock_sam3_gen = MagicMock()
            MockSAM3Gen.return_value = mock_sam3_gen

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            # Mock adapter
            mock_adapter = MagicMock()
            ovo.fusion_encoder = mock_adapter

            # Setup minimal update_map dependencies
            ovo.complete_semantic_info = MagicMock()
            ovo.update_objects_clip = MagicMock()

            # Create map data
            points_3d = torch.randn(10, 3)
            points_ids = torch.arange(10)
            points_ins_ids = torch.zeros(10, dtype=torch.long)
            map_data = (points_3d, points_ids, points_ins_ids)

            ovo.objects = {}

            ovo.update_map(map_data, kfs=[])

            # Verify adapter.update_objects was called
            mock_adapter.update_objects.assert_called_with(ovo.objects, ovo.keyframes)

    def test_update_map_delegates_cleanup_to_sam3_adapter(self, minimal_ovo_config_sam3):
        """update_map should call SAM3 adapter.cleanup_keyframe for deleted KFs."""
        with patch('ovo.entities.ovo.SAM3Generator') as MockSAM3Gen, \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.logger import Logger

            mock_sam3_gen = MagicMock()
            MockSAM3Gen.return_value = mock_sam3_gen

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            # Mock adapter
            mock_adapter = MagicMock()
            ovo.fusion_encoder = mock_adapter

            ovo.complete_semantic_info = MagicMock()
            ovo.update_objects_clip = MagicMock()

            # Keyframes state: KF 1 and 2 exist
            ovo.keyframes["frame_id"] = [1, 2]
            ovo.keyframes["ins_descriptors"] = {1: {}, 2: {}}

            ovo.objects = {}

            # Update with only KF 2 remaining (KF 1 deleted)
            ovo.update_map((torch.zeros(1, 3), torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)), kfs=[2])

            # Adapter cleanup should be called for KF 1
            mock_adapter.cleanup_keyframe.assert_called_with(1, ovo.keyframes)

    def test_update_map_delegates_transfer_on_merge(self, minimal_ovo_config_sam3):
        """update_map should call SAM3 adapter.transfer_on_merge when instances merge."""
        with patch('ovo.entities.ovo.SAM3Generator') as MockSAM3Gen, \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.logger import Logger

            mock_sam3_gen = MagicMock()
            MockSAM3Gen.return_value = mock_sam3_gen

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            # Mock adapter
            mock_adapter = MagicMock()
            ovo.fusion_encoder = mock_adapter

            # Setup minimal update_map dependencies
            ovo.complete_semantic_info = MagicMock()
            ovo.update_objects_clip = MagicMock()

            # Create map data
            points_3d = torch.randn(10, 3)
            points_ids = torch.arange(10)
            points_ins_ids = torch.zeros(10, dtype=torch.long)
            map_data = (points_3d, points_ids, points_ins_ids)

            # Mock fusion strategy to FORCE a merge
            obj1 = MagicMock()
            obj1.id = 1
            obj1.kfs_ids = [0]
            obj2 = MagicMock()
            obj2.id = 2
            obj2.kfs_ids = [0]
            ovo.objects = {1: obj1, 2: obj2}
            ovo.keyframes["ins_descriptors"] = {0: {}}

            ovo.fusion_strategy.same_instance = MagicMock(return_value=True)

            # Mock instance_utils.fuse_instances to actually perform merge logic
            with patch('ovo.utils.instance_utils.fuse_instances', return_value=(obj1, points_ins_ids)):
                ovo.update_map(map_data, kfs=[])

                # Verify transfer called (obj2 merged into obj1)
                mock_adapter.transfer_on_merge.assert_called_with([2], 1, ovo.keyframes)


class TestOVOSAM3BackwardCompatibility:
    """Test OVO still works without SAM3 (backward compatibility)."""

    def test_ovo_works_without_sam3_config(self, minimal_ovo_config):
        """OVO should work normally when 'sam3' not in config."""
        with patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config, mock_logger, eval=True)

            assert ovo.sam3_generator is None
            # fusion_encoder should be None for CLIP-only mode
            assert ovo.fusion_encoder is None

    def test_ovo_no_sam3_methods_called_when_disabled(self, minimal_ovo_config):
        """OVO should not attempt SAM3 operations when sam3_generator is None."""
        with patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config, mock_logger, eval=True)

            # Mock queue return
            ovo.keyframes_queue.popleft = MagicMock(return_value=(
                [1], torch.ones(1, 10, 10), torch.randn(10, 10, 3).numpy(), 1
            ))

            # Mock CLIP
            ovo._extract_clip = MagicMock(return_value=torch.randn(1, 512))
            ovo._update_matched_objects_clip = MagicMock()

            # Should not raise error
            ovo._compute_semantic_info()

            ovo._extract_clip.assert_called()
