"""
Unit tests for OVO's integration with FusionEncoderAdapter.

These tests verify that OVO correctly initializes and delegates to the
fusion adapter for optional encoders (PE, SAM3), while maintaining
mandatory CLIP operations.
"""

import pytest
import torch
from dataclasses import replace
from unittest.mock import MagicMock, patch, call
from ovo.entities.ovo import OVO, MapData
from ovo.entities.semantic_config import SemanticConfig
from ovo.entities.fusion_encoders import FusionFrameInput

class TestOVOFusionIntegration:

    @pytest.fixture
    def mock_ovo(self, minimal_ovo_config, minimal_run_config):
        """Create a mocked OVO instance."""
        with patch('ovo.entities.generator_pipeline.CLIPGenerator'), \
             patch('ovo.entities.generator_pipeline.PEGenerator'), \
             patch('ovo.entities.generator_pipeline.MaskGenerator'):

            config = replace(minimal_ovo_config, pe_config={"model_card": "test"})

            mock_logger = MagicMock()
            ovo = OVO(config, minimal_run_config, mock_logger)

            ovo.keyframes_queue = MagicMock()
            ovo.objects = {}
            return ovo

    def test_get_fusion_encoder_returns_pe_adapter(self, mock_ovo):
        """Should return PEFusionAdapter when method is 'pe'."""
        mock_ovo.semantic_config = replace(mock_ovo.semantic_config, fusion_method="pe")
        mock_ovo.generators = MagicMock()
        mock_ovo.generators.pe = MagicMock()

        adapter = mock_ovo._get_fusion_encoder()

        from ovo.entities.fusion_encoders import PEFusionAdapter
        assert isinstance(adapter, PEFusionAdapter)
        assert adapter.generator == mock_ovo.generators.pe

    def test_get_fusion_encoder_returns_none_for_clip(self, mock_ovo):
        """Should return None when method is 'clip'."""
        mock_ovo.semantic_config = replace(mock_ovo.semantic_config, fusion_method="clip")

        adapter = mock_ovo._get_fusion_encoder()
        assert adapter is None

    def test_validate_fusion_config_raises_on_missing_generator(self, mock_ovo):
        """Should raise ValueError if fusion method selected but generator missing."""
        mock_ovo.semantic_config = replace(mock_ovo.semantic_config, fusion_method="pe")
        mock_ovo.generators = MagicMock()
        mock_ovo.generators.pe = None

        with pytest.raises(ValueError, match="requires PE generator"):
            mock_ovo._validate_fusion_config()

    def test_compute_semantic_info_delegates_to_adapter(self, mock_ovo):
        """_compute_semantic_info should call adapter.compute_and_update."""
        mock_adapter = MagicMock()
        mock_ovo.fusion_encoder = mock_adapter

        image = torch.randn(10, 10, 3)
        binary_maps = torch.ones(2, 10, 10)
        matched_ids = [1, 2]
        kf_id = 5
        mock_ovo.keyframes_queue.popleft.return_value = (matched_ids, binary_maps, image, kf_id)

        mock_ovo._extract_clip = MagicMock(return_value=torch.randn(2, 512))
        mock_ovo._update_matched_objects_clip = MagicMock()

        mock_ovo._compute_semantic_info()

        mock_ovo._extract_clip.assert_called()
        mock_adapter.compute_and_update.assert_called_with(
            FusionFrameInput(image=image, binary_maps=binary_maps, matched_ins_ids=matched_ids, kf_id=kf_id),
            mock_ovo.kf_store, mock_ovo.objects
        )

    def test_compute_semantic_info_runs_without_adapter(self, mock_ovo):
        """_compute_semantic_info should work fine when adapter is None (CLIP mode)."""
        mock_ovo.fusion_encoder = None

        mock_ovo.keyframes_queue.popleft.return_value = ([1], torch.ones(1, 10, 10), torch.randn(10, 10, 3), 1)

        mock_ovo._extract_clip = MagicMock(return_value=torch.randn(1, 512))
        mock_ovo._update_matched_objects_clip = MagicMock()

        mock_ovo._compute_semantic_info()

        mock_ovo._extract_clip.assert_called()

    def test_update_map_delegates_transfer_on_merge(self, mock_ovo):
        """update_map should call adapter.transfer_on_merge."""
        mock_adapter = MagicMock()
        mock_ovo.fusion_encoder = mock_adapter

        mock_ovo.complete_semantic_info = MagicMock()
        mock_ovo.update_objects_clip = MagicMock()

        points_3d = torch.randn(10, 3)
        points_ids = torch.arange(10)
        points_ins_ids = torch.zeros(10, dtype=torch.long)
        map_data = MapData(points_3d, points_ids, points_ins_ids)

        obj1 = MagicMock(); obj1.id = 1; obj1.kfs_ids = [0]
        obj2 = MagicMock(); obj2.id = 2; obj2.kfs_ids = [0]
        mock_ovo.objects = {1: obj1, 2: obj2}

        mock_ovo.fusion_strategy.same_instance = MagicMock(return_value=True)

        with patch('ovo.utils.instance_utils.fuse_instances', return_value=(obj1, points_ins_ids)):
            mock_ovo.update_map(map_data, kfs=[])
            pass

    def test_update_map_delegates_cleanup(self, mock_ovo):
        """update_map should call adapter.cleanup_keyframe for deleted KFs."""
        mock_adapter = MagicMock()
        mock_ovo.fusion_encoder = mock_adapter

        mock_ovo.complete_semantic_info = MagicMock()
        mock_ovo.update_objects_clip = MagicMock()

        mock_ovo.kf_store.frame_ids = [1, 2]

        mock_ovo.update_map(MapData(torch.zeros(1,3), torch.zeros(1), torch.zeros(1, dtype=torch.long)), kfs={2: {}})
