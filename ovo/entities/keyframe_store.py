from typing import Dict, List, Any
import numpy as np
import torch


class KeyframeStore:
    """Typed store for per-keyframe semantic descriptors and frame metadata.

    Supports dict-style access (keyframes[key]) for backward compat with
    FusionEncoderAdapter implementations that use storage_key string lookup.
    """

    _DESCRIPTOR_KEYS = ("ins_descriptors", "ins_pe_descriptors", "ins_sam3_descriptors")

    def __init__(self) -> None:
        self._store: Dict[str, Dict[int, Dict[int, torch.Tensor]]] = {
            "ins_descriptors": {},
            "ins_pe_descriptors": {},
            "ins_sam3_descriptors": {},
        }
        self.frame_ids: List[int | str] = []
        self.ins_maps: List[Any] = []  # debug only

    # ── Dict protocol (backward compat for FusionEncoderAdapter) ────────────

    def __getitem__(self, key: str) -> Dict:
        return self._store[key]

    def __setitem__(self, key: str, value: Dict) -> None:
        self._store[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._store

    # ── Typed properties ─────────────────────────────────────────────────────

    @property
    def clip(self) -> Dict[int, Dict[int, torch.Tensor]]:
        return self._store["ins_descriptors"]

    @property
    def pe(self) -> Dict[int, Dict[int, torch.Tensor]]:
        return self._store["ins_pe_descriptors"]

    @property
    def sam3(self) -> Dict[int, Dict[int, torch.Tensor]]:
        return self._store["ins_sam3_descriptors"]

    # ── Frame lifecycle ──────────────────────────────────────────────────────

    def register_frame(self, frame_id: int) -> None:
        self.frame_ids.append(frame_id)

    def mark_deleted(self, idx: int) -> None:
        """Mark kf at index idx as deleted; preserves kf_id indexing."""
        self.frame_ids[idx] = "Deleted"

    # ── Descriptor management ────────────────────────────────────────────────

    def cleanup_kf(self, kf_id: int) -> None:
        """Remove all descriptors for keyframe kf_id."""
        for key in self._DESCRIPTOR_KEYS:
            self._store[key].pop(kf_id, None)

    def cleanup_clip(self, kf_id: int) -> None:
        self._store["ins_descriptors"].pop(kf_id, None)

    def transfer_clip(self, from_id: int, to_id: int, kf_id: int) -> None:
        """Move CLIP descriptor from_id → to_id for kf_id (no-op if to_id exists)."""
        store = self._store["ins_descriptors"].get(kf_id, {})
        if from_id in store and to_id not in store:
            store[to_id] = store.pop(from_id)

    # ── Serialization ────────────────────────────────────────────────────────

    def capture(self, debug_info: bool) -> Dict[str, Any]:
        """Return serializable dict of keyframe state."""
        data: Dict[str, Any] = {}
        if not debug_info:
            return data
        data["frame_id"] = np.array(self.frame_ids)
        for kf_id, descs in self.clip.items():
            for ins_id, d in descs.items():
                data[f"kf_{kf_id}_ins3d_{ins_id}_clips"] = d.cpu().numpy()
        for kf_id, descs in self.pe.items():
            for ins_id, d in descs.items():
                data[f"kf_{kf_id}_ins3d_{ins_id}_pe"] = d.cpu().numpy()
        for kf_id, descs in self.sam3.items():
            for ins_id, d in descs.items():
                data[f"kf_{kf_id}_ins3d_{ins_id}_sam3"] = d.cpu().numpy()
        return data

    def restore(self, scene_dict: Dict[str, Any], obj_ids, device: str) -> None:
        """Restore keyframe state from a captured scene_dict."""
        self.frame_ids = list(scene_dict["frame_id"])
        for i in range(len(self.frame_ids)):
            self._store["ins_descriptors"][i] = {}
            self._store["ins_pe_descriptors"][i] = {}
            self._store["ins_sam3_descriptors"][i] = {}
            for ins_id in obj_ids:
                clip_d = scene_dict.get(f"kf_{i}_ins3d_{ins_id}_clips")
                if clip_d is not None:
                    self._store["ins_descriptors"][i][ins_id] = torch.tensor(clip_d, device=device)
                pe_d = scene_dict.get(f"kf_{i}_ins3d_{ins_id}_pe")
                if pe_d is not None:
                    self._store["ins_pe_descriptors"][i][ins_id] = torch.tensor(pe_d, device=device)
                sam3_d = scene_dict.get(f"kf_{i}_ins3d_{ins_id}_sam3")
                if sam3_d is not None:
                    self._store["ins_sam3_descriptors"][i][ins_id] = torch.tensor(sam3_d, device=device)
