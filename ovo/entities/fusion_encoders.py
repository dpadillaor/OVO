from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
import torch

from ovo.entities.instance3d import Instance3D
from ovo.entities.pe_generator import PEGenerator

class FusionEncoderAdapter(ABC):
    """
    Adapter for optional fusion encoders (PE, DINO, etc.)

    Encapsulates all encoder-specific logic for:
    - Descriptor extraction
    - Storage management
    - Instance updates
    - Merge transfers
    """

    @abstractmethod
    def compute_and_update(self, image: torch.Tensor, binary_maps: torch.Tensor, matched_ins_ids: List[int], kf_id: int, keyframes: Dict[str, Any], objects: Dict[int, Instance3D]) -> None:
        """Extract embeddings and update instances."""
        pass

    @abstractmethod
    def update_objects(self, objects: Dict[int, Instance3D], keyframes: Dict[str, Any]) -> None:
        """Batch update all objects with fused descriptors."""
        pass

    @abstractmethod
    def transfer_on_merge(self, source_ids: List[int], target_id: int, keyframes: Dict[str, Any]) -> None:
        """Transfer descriptors during instance merge."""
        pass

    @abstractmethod
    def cleanup_keyframe(self, kf_id: int, keyframes: Dict[str, Any]) -> None:
        """Remove descriptors for deleted keyframe."""
        pass


class PEFusionAdapter(FusionEncoderAdapter):
    """Adapter for Perception Encoder fusion."""

    def __init__(self, pe_generator: PEGenerator, storage_key: str = "ins_pe_descriptors"):
        self.generator = pe_generator
        self.storage_key = storage_key

    def compute_and_update(self, image: torch.Tensor, binary_maps: torch.Tensor, matched_ins_ids: List[int], kf_id: int, keyframes: Dict[str, Any], objects: Dict[int, Instance3D]) -> None:
        """
        Extract PE embeddings and store in keyframes + update instances.
        
        Args:
            image: RGB image tensor/array.
            binary_maps: Binary masks for instances.
            matched_ins_ids: List of instance IDs matched to masks.
            kf_id: Current keyframe ID.
            keyframes: Global keyframes dictionary.
            objects: Global objects dictionary.
        """
        # Handle empty input
        if image is None or binary_maps is None or len(matched_ins_ids) == 0:
            return

        # Extract embeddings
        pe_embeds = self.generator.extract_pe(image, binary_maps).cpu()

        # Initialize storage for keyframe if needed
        if kf_id not in keyframes[self.storage_key]:
            keyframes[self.storage_key][kf_id] = {}

        # Store embeddings and update instances
        for idx, ins_id in enumerate(matched_ins_ids):
            if ins_id == -1:
                continue
                
            # Store in keyframes
            keyframes[self.storage_key][kf_id][ins_id] = pe_embeds[idx:idx+1]
            
            # Update object if it exists
            if ins_id in objects:
                objects[ins_id].update_pe(keyframes[self.storage_key])

    def update_objects(self, objects: Dict[int, Instance3D], keyframes: Dict[str, Any]) -> None:
        """
        Batch update all objects with fused PE descriptors.
        
        Args:
            objects: Dictionary of Instance3D objects.
            keyframes: Global keyframes dictionary.
        """
        for obj in objects.values():
            if obj.to_update_pe:
                obj.update_pe(keyframes[self.storage_key])

    def transfer_on_merge(self, source_ids: List[int], target_id: int, keyframes: Dict[str, Any]) -> None:
        """
        Transfer PE descriptors from source instances to target during merge.
        
        Args:
            source_ids: List of source instance IDs to merge from.
            target_id: Target instance ID to merge into.
            keyframes: Global keyframes dictionary.
        """
        if self.storage_key not in keyframes:
            return

        for kf_id in keyframes[self.storage_key]:
            for source_id in source_ids:
                if source_id in keyframes[self.storage_key][kf_id]:
                    # Move descriptor from source to target
                    # If multiple sources map to same target in same frame, last one wins (simple override)
                    keyframes[self.storage_key][kf_id][target_id] = \
                        keyframes[self.storage_key][kf_id].pop(source_id)

    def cleanup_keyframe(self, kf_id: int, keyframes: Dict[str, Any]) -> None:
        """
        Remove PE descriptors for deleted keyframe.
        
        Args:
            kf_id: ID of keyframe to remove.
            keyframes: Global keyframes dictionary.
        """
        if kf_id in keyframes[self.storage_key]:
            del keyframes[self.storage_key][kf_id]


class DINOFusionAdapter(FusionEncoderAdapter):
    """Adapter for DINO fusion (placeholder for future integration)."""

    def __init__(self, dino_generator: Any, storage_key: str = "ins_dino_descriptors"):
        self.generator = dino_generator
        self.storage_key = storage_key

    def compute_and_update(self, image: torch.Tensor, binary_maps: torch.Tensor, matched_ins_ids: List[int], kf_id: int, keyframes: Dict[str, Any], objects: Dict[int, Instance3D]) -> None:
        # Placeholder implementation
        pass

    def update_objects(self, objects: Dict[int, Instance3D], keyframes: Dict[str, Any]) -> None:
        # Placeholder implementation
        pass

    def transfer_on_merge(self, source_ids: List[int], target_id: int, keyframes: Dict[str, Any]) -> None:
        # Placeholder implementation
        pass

    def cleanup_keyframe(self, kf_id: int, keyframes: Dict[str, Any]) -> None:
        # Placeholder implementation
        pass