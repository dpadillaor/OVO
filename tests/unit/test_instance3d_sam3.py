"""Tests for Instance3D SAM3 feature integration."""

import pytest
import torch
from ovo.entities.instance3d import Instance3D


class TestInstance3DSAM3Attributes:
    """Test Instance3D has SAM3-related attributes."""

    def test_instance_has_sam3_feature_attribute(self):
        """Instance3D should have sam3_feature attribute initialized to None."""
        instance = Instance3D(id=1)

        assert hasattr(instance, 'sam3_feature')
        assert instance.sam3_feature is None

    def test_instance_has_sam3_feature_kf_attribute(self):
        """Instance3D should have sam3_feature_kf attribute initialized to None."""
        instance = Instance3D(id=1)

        assert hasattr(instance, 'sam3_feature_kf')
        assert instance.sam3_feature_kf is None

    def test_instance_has_to_update_sam3_flag(self):
        """Instance3D should have to_update_sam3 flag initialized to False."""
        instance = Instance3D(id=1)

        assert hasattr(instance, 'to_update_sam3')
        assert instance.to_update_sam3 == False

    def test_instance_update_sets_to_update_sam3_flag(self):
        """Instance3D.update() should set to_update_sam3 to True."""
        instance = Instance3D(id=1)
        instance.update(points_ids=[1, 2, 3], kf_id=0, area=100)

        assert instance.to_update_sam3 == True


class TestInstance3DSAM3Update:
    """Test Instance3D.update_sam3() method."""

    def test_update_sam3_computes_feature(self):
        """update_sam3 should compute sam3_feature from keyframe descriptors."""
        instance = Instance3D(id=1)
        instance.update(points_ids=[1, 2, 3], kf_id=0, area=100)

        keyframes_sam3 = {
            0: {1: torch.randn(1, 1024)}
        }

        instance.update_sam3(keyframes_sam3)

        assert instance.sam3_feature is not None
        assert instance.sam3_feature.shape == (1, 1024)

    def test_update_sam3_sets_feature_kf(self):
        """update_sam3 should set sam3_feature_kf to the selected keyframe."""
        instance = Instance3D(id=1)
        instance.update(points_ids=[1, 2, 3], kf_id=0, area=100)

        keyframes_sam3 = {
            0: {1: torch.randn(1, 1024)}
        }

        instance.update_sam3(keyframes_sam3)

        assert instance.sam3_feature_kf is not None

    def test_update_sam3_clears_to_update_flag(self):
        """update_sam3 should set to_update_sam3 to False after update."""
        instance = Instance3D(id=1)
        instance.update(points_ids=[1, 2, 3], kf_id=0, area=100)

        keyframes_sam3 = {
            0: {1: torch.randn(1, 1024)}
        }

        instance.update_sam3(keyframes_sam3)

        assert instance.to_update_sam3 == False

    def test_update_sam3_skips_when_flag_false(self):
        """update_sam3 should skip computation when to_update_sam3 is False."""
        instance = Instance3D(id=1)
        instance.to_update_sam3 = False
        instance.sam3_feature = torch.randn(1, 1024)
        original_feature = instance.sam3_feature.clone()

        keyframes_sam3 = {
            0: {1: torch.randn(1, 1024)}  # Different feature
        }

        instance.update_sam3(keyframes_sam3)

        assert torch.equal(instance.sam3_feature, original_feature)

    def test_update_sam3_force_update(self):
        """update_sam3 should recompute when force_update=True."""
        instance = Instance3D(id=1)
        instance.kfs_ids = [0]
        instance.to_update_sam3 = False
        instance.sam3_feature = torch.zeros(1, 1024)

        keyframes_sam3 = {
            0: {1: torch.ones(1, 1024)}  # Different feature
        }

        instance.update_sam3(keyframes_sam3, force_update=True)

        assert torch.allclose(instance.sam3_feature, torch.ones(1, 1024))

    def test_update_sam3_multiple_keyframes_uses_median(self):
        """update_sam3 should select feature minimizing L1 norm (median)."""
        instance = Instance3D(id=1)
        instance.kfs_ids = [0, 1, 2]
        instance.to_update_sam3 = True

        # Create 3 features where the middle one is the median
        keyframes_sam3 = {
            0: {1: torch.tensor([[0.0, 0.0, 0.0, 0.0]])},
            1: {1: torch.tensor([[0.5, 0.5, 0.5, 0.5]])},  # Median
            2: {1: torch.tensor([[1.0, 1.0, 1.0, 1.0]])},
        }

        instance.update_sam3(keyframes_sam3)

        # Should select the median feature (index 1)
        expected = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        assert torch.allclose(instance.sam3_feature, expected)

    def test_update_sam3_empty_keyframes(self):
        """update_sam3 should handle empty keyframes gracefully."""
        instance = Instance3D(id=1)
        instance.kfs_ids = [0]
        instance.to_update_sam3 = True

        keyframes_sam3 = {}  # No keyframes

        instance.update_sam3(keyframes_sam3)

        assert instance.sam3_feature is None


class TestInstance3DSAM3Export:
    """Test Instance3D export/restore with SAM3 features."""

    def test_export_includes_sam3_feature(self):
        """export() should include sam3_feature in the dictionary."""
        instance = Instance3D(id=1)
        instance.sam3_feature = torch.randn(1, 1024)
        instance.sam3_feature_kf = 5

        exported = instance.export()

        assert f"ins3d_1_sam3_feature" in exported
        assert f"ins3d_1_sam3_feature_kf" in exported

    def test_restore_loads_sam3_feature(self):
        """restore() should load sam3_feature from dictionary."""
        instance = Instance3D(id=1)
        sam3_feat = torch.randn(1, 1024)

        obj_dict = {
            "ins3d_1_clip_feature": None,
            "ins3d_1_clip_feature_kf": None,
            "ins3d_1_pe_feature": None,
            "ins3d_1_pe_feature_kf": None,
            "ins3d_1_sam3_feature": sam3_feat,
            "ins3d_1_sam3_feature_kf": 3,
        }

        instance.restore(obj_dict, debug_info=False)

        assert torch.equal(instance.sam3_feature, sam3_feat)
        assert instance.sam3_feature_kf == 3

    def test_restore_sets_to_update_sam3_when_none(self):
        """restore() should set to_update_sam3=True when sam3_feature is None."""
        instance = Instance3D(id=1)

        obj_dict = {
            "ins3d_1_clip_feature": None,
            "ins3d_1_clip_feature_kf": None,
            "ins3d_1_pe_feature": None,
            "ins3d_1_pe_feature_kf": None,
            "ins3d_1_sam3_feature": None,
            "ins3d_1_sam3_feature_kf": None,
        }

        instance.restore(obj_dict, debug_info=False)

        assert instance.to_update_sam3 == True
