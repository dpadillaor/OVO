"""
Unit tests for FusionEncoderAdapter and its implementations.

These tests verify the adapter pattern for fusion encoders:
1. Adapter correctly delegates extraction to generators.
2. Adapter manages storage in keyframes correctly.
3. Adapter handles instance updates and merge transfers.
4. Edge cases (empty inputs, missing IDs) are handled gracefully.
"""

import pytest
import torch
from unittest.mock import MagicMock, call
from ovo.entities.fusion_encoders import PEFusionAdapter, DINOFusionAdapter, SAM3FusionAdapter

class TestPEFusionAdapter:
    """Tests for PEFusionAdapter logic."""

    def test_init_sets_correct_defaults(self, mock_pe_generator):
        """Adapter should initialize with correct generator and storage key."""
        adapter = PEFusionAdapter(mock_pe_generator)
        assert adapter.generator == mock_pe_generator
        assert adapter.storage_key == "ins_pe_descriptors"

    def test_init_with_wrong_generator_type_no_runtime_error(self):
        """
        Tests that passing a 'wrong' generator type does NOT raise a runtime error during __init__.
        Python type hints are for static analysis; runtime checks would need explicit implementation.
        The actual error should occur when a method (like 'extract_pe') is called.
        """
        class DummyWrongGenerator:
            # This class intentionally does not have an 'extract_pe' method
            pass

        wrong_generator_instance = DummyWrongGenerator()
        
        # We expect no exception to be raised during initialization
        adapter = PEFusionAdapter(wrong_generator_instance)
        assert adapter.generator == wrong_generator_instance
        assert adapter.storage_key == "ins_pe_descriptors"
        
        # The actual error would occur when 'extract_pe' is called.
        # Provide a valid keyframes structure for compute_and_update to avoid KeyError
        keyframes_with_storage = {adapter.storage_key: {}}
        with pytest.raises(AttributeError, match="object has no attribute 'extract_pe'"):
            adapter.compute_and_update(torch.randn(3, 100, 100), torch.ones(1, 100, 100), [0], 1, keyframes_with_storage, {0: MagicMock()})


    # --- Compute and Update Tests ---

    def test_compute_and_update_normal_flow(self, mock_pe_generator, mock_keyframes_data, mock_objects_dict):
        """Standard flow: extract -> store -> update objects."""
        adapter = PEFusionAdapter(mock_pe_generator)
        
        # Setup inputs
        image = torch.randn(3, 100, 100)
        binary_maps = torch.ones(3, 100, 100) # 3 masks
        matched_ins_ids = [0, 1, 2] # All exist in mock_objects_dict
        kf_id = 10
        
        # Mock generator output
        mock_pe_generator.extract_pe.return_value = torch.randn(3, 768)
        
        adapter.compute_and_update(image, binary_maps, matched_ins_ids, kf_id, mock_keyframes_data, mock_objects_dict)
        
        # 1. Verify Generator called
        mock_pe_generator.extract_pe.assert_called_once()
        
        # 2. Verify Keyframes storage
        assert kf_id in mock_keyframes_data["ins_pe_descriptors"]
        storage = mock_keyframes_data["ins_pe_descriptors"][kf_id]
        assert len(storage) == 3
        assert 0 in storage and 1 in storage and 2 in storage
        assert storage[0].shape == (1, 768)
        
        # 3. Verify Object update called
        mock_objects_dict[0].update_pe.assert_called()

    def test_compute_and_update_filters_invalid_ids(self, mock_pe_generator, mock_keyframes_data, mock_objects_dict):
        """Should not store descriptors for instance ID -1."""
        adapter = PEFusionAdapter(mock_pe_generator)
        
        matched_ins_ids = [0, -1, 1]
        binary_maps = torch.ones(3, 100, 100)
        kf_id = 5
        
        mock_pe_generator.extract_pe.return_value = torch.randn(3, 768)
        
        dummy_image = torch.randn(3, 100, 100)
        adapter.compute_and_update(dummy_image, binary_maps, matched_ins_ids, kf_id, mock_keyframes_data, mock_objects_dict)
        
        storage = mock_keyframes_data["ins_pe_descriptors"][kf_id]
        assert 0 in storage
        assert 1 in storage
        assert -1 not in storage
        assert len(storage) == 2

    def test_compute_and_update_empty_input(self, mock_pe_generator, mock_keyframes_data, mock_objects_dict):
        """Should handle empty inputs gracefully."""
        adapter = PEFusionAdapter(mock_pe_generator)
        
        adapter.compute_and_update(None, None, [], 5, mock_keyframes_data, mock_objects_dict)
        
        mock_pe_generator.extract_pe.assert_not_called()
        assert 5 not in mock_keyframes_data["ins_pe_descriptors"]

    def test_compute_and_update_object_missing_from_dict(self, mock_pe_generator, mock_keyframes_data, mock_objects_dict):
        """Should store descriptor in keyframes even if object is missing from objects dict."""
        adapter = PEFusionAdapter(mock_pe_generator)
        
        matched_ins_ids = [99] # 99 does not exist in mock_objects_dict
        binary_maps = torch.ones(1, 100, 100)
        kf_id = 7
        
        mock_pe_generator.extract_pe.return_value = torch.randn(1, 768)
        
        dummy_image = torch.randn(3, 100, 100)
        adapter.compute_and_update(dummy_image, binary_maps, matched_ins_ids, kf_id, mock_keyframes_data, mock_objects_dict)
        
        # Should be in keyframes
        assert 99 in mock_keyframes_data["ins_pe_descriptors"][kf_id]
        # Should not crash trying to access objects[99]
    
    # --- Update Objects Tests ---

    def test_update_objects_delegates_to_instance(self, mock_pe_generator, mock_keyframes_data, mock_objects_dict):
        """Should call update_pe on all objects with correct storage."""
        adapter = PEFusionAdapter(mock_pe_generator)
        
        adapter.update_objects(mock_objects_dict, mock_keyframes_data)
        
        # Check delegation
        for obj in mock_objects_dict.values():
            # In the strategy, we iterate all objects and call update_pe if needed.
            # Ideally the adapter passes the WHOLE keyframe dict for that encoder.
            obj.update_pe.assert_called_with(mock_keyframes_data["ins_pe_descriptors"])

    # --- Merge Transfer Tests ---

    def test_transfer_on_merge_moves_descriptors(self, mock_pe_generator, mock_keyframes_data):
        """Should move descriptors from source to target ID in all keyframes."""
        adapter = PEFusionAdapter(mock_pe_generator)
        
        kf1, kf2 = 10, 20
        src_id, tgt_id = 1, 2
        feat1 = torch.randn(1, 768)
        
        # Setup: Source exists in KF1, not KF2
        mock_keyframes_data["ins_pe_descriptors"] = {
            kf1: {src_id: feat1},
            
            kf2: {tgt_id: torch.randn(1, 768)} # Target exists here
        }
        
        adapter.transfer_on_merge([src_id], tgt_id, mock_keyframes_data)
        
        # KF1: Source should be gone, Target should have Source's feature
        assert src_id not in mock_keyframes_data["ins_pe_descriptors"][kf1]
        assert tgt_id in mock_keyframes_data["ins_pe_descriptors"][kf1]
        assert torch.equal(mock_keyframes_data["ins_pe_descriptors"][kf1][tgt_id], feat1)
        
        # KF2: Source wasn't there, Target remains untouched
        assert src_id not in mock_keyframes_data["ins_pe_descriptors"][kf2]
        assert tgt_id in mock_keyframes_data["ins_pe_descriptors"][kf2]

    def test_transfer_on_merge_handles_missing_keys(self, mock_pe_generator, mock_keyframes_data):
        """Should be safe if storage key is missing from keyframes."""
        adapter = PEFusionAdapter(mock_pe_generator)
        # Empty keyframes data (no "ins_pe_descriptors" populated properly or empty dict)
        mock_keyframes_data["ins_pe_descriptors"] = {}
        
        # Should not crash
        adapter.transfer_on_merge([1], 2, mock_keyframes_data)

    def test_transfer_on_merge_multiple_sources(self, mock_pe_generator, mock_keyframes_data):
        """Should handle multiple source IDs."""
        adapter = PEFusionAdapter(mock_pe_generator)
        kf1 = 1
        src1, src2, tgt = 10, 11, 20
        feat1 = torch.randn(1, 768)
        
        mock_keyframes_data["ins_pe_descriptors"] = {
            kf1: {src1: feat1, src2: torch.randn(1, 768)}
        }
        
        # Note: If multiple sources exist in same keyframe, last one processed overwrites target.
        # This is expected behavior for simple merge logic.
        adapter.transfer_on_merge([src1, src2], tgt, mock_keyframes_data)
        
        assert src1 not in mock_keyframes_data["ins_pe_descriptors"][kf1]
        assert src2 not in mock_keyframes_data["ins_pe_descriptors"][kf1]
        assert tgt in mock_keyframes_data["ins_pe_descriptors"][kf1]

    # --- Cleanup Tests ---

    def test_cleanup_keyframe_removes_entry(self, mock_pe_generator, mock_keyframes_data):
        """Should remove keyframe entry from storage."""
        adapter = PEFusionAdapter(mock_pe_generator)
        kf_id = 5
        mock_keyframes_data["ins_pe_descriptors"][kf_id] = {1: "data"}
        
        adapter.cleanup_keyframe(kf_id, mock_keyframes_data)
        assert kf_id not in mock_keyframes_data["ins_pe_descriptors"]

    def test_cleanup_keyframe_safe_if_missing(self, mock_pe_generator, mock_keyframes_data):
        """Should do nothing if keyframe not present."""
        adapter = PEFusionAdapter(mock_pe_generator)
        adapter.cleanup_keyframe(999, mock_keyframes_data)
        # No error raised


class TestDINOFusionAdapter:
    """Tests for DINOFusionAdapter configuration."""

    def test_init_sets_correct_defaults(self, mock_dino_generator):
        """Should initialize with DINO-specific storage key."""
        adapter = DINOFusionAdapter(mock_dino_generator)
        assert adapter.generator == mock_dino_generator
        assert adapter.storage_key == "ins_dino_descriptors"

    def test_methods_exist(self, mock_dino_generator):
        """Verify placeholder methods exist and follow interface."""
        adapter = DINOFusionAdapter(mock_dino_generator)
        assert hasattr(adapter, 'compute_and_update')
        assert hasattr(adapter, 'update_objects')
        assert hasattr(adapter, 'transfer_on_merge')
        assert hasattr(adapter, 'cleanup_keyframe')


class TestSAM3FusionAdapter:
    """Tests for SAM3FusionAdapter logic."""

    def test_init_sets_correct_defaults(self, mock_sam3_generator):
        """Adapter should initialize with correct generator and storage key."""
        adapter = SAM3FusionAdapter(mock_sam3_generator)
        assert adapter.generator == mock_sam3_generator
        assert adapter.storage_key == "ins_sam3_descriptors"

    def test_init_with_custom_storage_key(self, mock_sam3_generator):
        """Should allow custom storage key."""
        adapter = SAM3FusionAdapter(mock_sam3_generator, storage_key="custom_sam3_key")
        assert adapter.storage_key == "custom_sam3_key"

    # --- Compute and Update Tests ---

    def test_compute_and_update_normal_flow(self, mock_sam3_generator, mock_keyframes_data, mock_objects_dict):
        """Standard flow: extract -> store -> update objects."""
        adapter = SAM3FusionAdapter(mock_sam3_generator)

        # Setup inputs
        image = torch.randn(3, 100, 100)
        binary_maps = torch.ones(3, 100, 100) # 3 masks
        matched_ins_ids = [0, 1, 2] # All exist in mock_objects_dict
        kf_id = 10

        # Mock generator output
        mock_sam3_generator.extract_sam3.return_value = torch.randn(3, 1024)

        adapter.compute_and_update(image, binary_maps, matched_ins_ids, kf_id, mock_keyframes_data, mock_objects_dict)

        # 1. Verify Generator called
        mock_sam3_generator.extract_sam3.assert_called_once()

        # 2. Verify Keyframes storage
        assert kf_id in mock_keyframes_data["ins_sam3_descriptors"]
        storage = mock_keyframes_data["ins_sam3_descriptors"][kf_id]
        assert len(storage) == 3
        assert 0 in storage and 1 in storage and 2 in storage
        assert storage[0].shape == (1, 1024)

        # 3. Verify Object update called
        mock_objects_dict[0].update_sam3.assert_called()

    def test_compute_and_update_filters_invalid_ids(self, mock_sam3_generator, mock_keyframes_data, mock_objects_dict):
        """Should not store descriptors for instance ID -1."""
        adapter = SAM3FusionAdapter(mock_sam3_generator)

        matched_ins_ids = [0, -1, 1]
        binary_maps = torch.ones(3, 100, 100)
        kf_id = 5

        mock_sam3_generator.extract_sam3.return_value = torch.randn(3, 1024)

        dummy_image = torch.randn(3, 100, 100)
        adapter.compute_and_update(dummy_image, binary_maps, matched_ins_ids, kf_id, mock_keyframes_data, mock_objects_dict)

        storage = mock_keyframes_data["ins_sam3_descriptors"][kf_id]
        assert 0 in storage
        assert 1 in storage
        assert -1 not in storage
        assert len(storage) == 2

    def test_compute_and_update_empty_input(self, mock_sam3_generator, mock_keyframes_data, mock_objects_dict):
        """Should handle empty inputs gracefully."""
        adapter = SAM3FusionAdapter(mock_sam3_generator)

        adapter.compute_and_update(None, None, [], 5, mock_keyframes_data, mock_objects_dict)

        mock_sam3_generator.extract_sam3.assert_not_called()
        assert 5 not in mock_keyframes_data["ins_sam3_descriptors"]

    def test_compute_and_update_object_missing_from_dict(self, mock_sam3_generator, mock_keyframes_data, mock_objects_dict):
        """Should store descriptor in keyframes even if object is missing from objects dict."""
        adapter = SAM3FusionAdapter(mock_sam3_generator)

        matched_ins_ids = [99] # 99 does not exist in mock_objects_dict
        binary_maps = torch.ones(1, 100, 100)
        kf_id = 7

        mock_sam3_generator.extract_sam3.return_value = torch.randn(1, 1024)

        dummy_image = torch.randn(3, 100, 100)
        adapter.compute_and_update(dummy_image, binary_maps, matched_ins_ids, kf_id, mock_keyframes_data, mock_objects_dict)

        # Should be in keyframes
        assert 99 in mock_keyframes_data["ins_sam3_descriptors"][kf_id]
        # Should not crash trying to access objects[99]

    # --- Update Objects Tests ---

    def test_update_objects_delegates_to_instance(self, mock_sam3_generator, mock_keyframes_data, mock_objects_dict):
        """Should call update_sam3 on all objects with correct storage."""
        adapter = SAM3FusionAdapter(mock_sam3_generator)

        adapter.update_objects(mock_objects_dict, mock_keyframes_data)

        # Check delegation
        for obj in mock_objects_dict.values():
            obj.update_sam3.assert_called_with(mock_keyframes_data["ins_sam3_descriptors"])

    def test_update_objects_only_updates_flagged_instances(self, mock_sam3_generator, mock_keyframes_data):
        """Should only update instances with to_update_sam3=True."""
        adapter = SAM3FusionAdapter(mock_sam3_generator)

        # Create objects with different update flags
        obj1 = MagicMock()
        obj1.to_update_sam3 = True
        obj2 = MagicMock()
        obj2.to_update_sam3 = False

        objects = {1: obj1, 2: obj2}

        adapter.update_objects(objects, mock_keyframes_data)

        # Adapter checks the flag itself: only flagged instances are updated
        obj1.update_sam3.assert_called_once()
        obj2.update_sam3.assert_not_called()

    # --- Merge Transfer Tests ---

    def test_transfer_on_merge_moves_descriptors(self, mock_sam3_generator, mock_keyframes_data):
        """Should move descriptors from source to target ID in all keyframes."""
        adapter = SAM3FusionAdapter(mock_sam3_generator)

        kf1, kf2 = 10, 20
        src_id, tgt_id = 1, 2
        feat1 = torch.randn(1, 1024)

        # Setup: Source exists in KF1, not KF2
        mock_keyframes_data["ins_sam3_descriptors"] = {
            kf1: {src_id: feat1},
            kf2: {tgt_id: torch.randn(1, 1024)} # Target exists here
        }

        adapter.transfer_on_merge([src_id], tgt_id, mock_keyframes_data)

        # KF1: Source should be gone, Target should have Source's feature
        assert src_id not in mock_keyframes_data["ins_sam3_descriptors"][kf1]
        assert tgt_id in mock_keyframes_data["ins_sam3_descriptors"][kf1]
        assert torch.equal(mock_keyframes_data["ins_sam3_descriptors"][kf1][tgt_id], feat1)

        # KF2: Source wasn't there, Target remains untouched
        assert src_id not in mock_keyframes_data["ins_sam3_descriptors"][kf2]
        assert tgt_id in mock_keyframes_data["ins_sam3_descriptors"][kf2]

    def test_transfer_on_merge_handles_missing_keys(self, mock_sam3_generator, mock_keyframes_data):
        """Should be safe if storage key is missing from keyframes."""
        adapter = SAM3FusionAdapter(mock_sam3_generator)
        # Empty keyframes data
        mock_keyframes_data["ins_sam3_descriptors"] = {}

        # Should not crash
        adapter.transfer_on_merge([1], 2, mock_keyframes_data)

    def test_transfer_on_merge_multiple_sources(self, mock_sam3_generator, mock_keyframes_data):
        """Should handle multiple source IDs."""
        adapter = SAM3FusionAdapter(mock_sam3_generator)
        kf1 = 1
        src1, src2, tgt = 10, 11, 20
        feat1 = torch.randn(1, 1024)

        mock_keyframes_data["ins_sam3_descriptors"] = {
            kf1: {src1: feat1, src2: torch.randn(1, 1024)}
        }

        adapter.transfer_on_merge([src1, src2], tgt, mock_keyframes_data)

        assert src1 not in mock_keyframes_data["ins_sam3_descriptors"][kf1]
        assert src2 not in mock_keyframes_data["ins_sam3_descriptors"][kf1]
        assert tgt in mock_keyframes_data["ins_sam3_descriptors"][kf1]

    # --- Cleanup Tests ---

    def test_cleanup_keyframe_removes_entry(self, mock_sam3_generator, mock_keyframes_data):
        """Should remove keyframe entry from storage."""
        adapter = SAM3FusionAdapter(mock_sam3_generator)
        kf_id = 5
        mock_keyframes_data["ins_sam3_descriptors"][kf_id] = {1: "data"}

        adapter.cleanup_keyframe(kf_id, mock_keyframes_data)
        assert kf_id not in mock_keyframes_data["ins_sam3_descriptors"]

    def test_cleanup_keyframe_safe_if_missing(self, mock_sam3_generator, mock_keyframes_data):
        """Should do nothing if keyframe not present."""
        adapter = SAM3FusionAdapter(mock_sam3_generator)
        adapter.cleanup_keyframe(999, mock_keyframes_data)
        # No error raised