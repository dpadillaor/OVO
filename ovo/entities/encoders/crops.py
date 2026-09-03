"""Crop strategies: turn a frame + per-instance masks into the crops an encoder sees.

Decoupling the crop from the encoder lets the same descriptor (e.g. PE) run on the masked
segment (tight, background removed) or on the bounding box (context kept), which changes how
well its features group fragments of one object. Crops come out in 0-255; the encoder
normalizes. `SegCrop` reproduces the previous single-crop path exactly.
"""

from abc import ABC, abstractmethod

import torch
import torchvision.transforms.functional as F

from ...utils import segment_utils


class CropStrategy(ABC):
    @abstractmethod
    def __call__(self, image: torch.Tensor, binary_maps: torch.Tensor) -> torch.Tensor:
        """(image 3xHxW 0-255, N masks) -> (N,3,out_l,out_l) crops in 0-255."""
        ...


class SegCrop(CropStrategy):
    """Masked segment: the object cropped to its mask, background set to black, padded square."""

    def __init__(self, out_l: int):
        self.out_l = out_l

    def __call__(self, image, binary_maps):
        return segment_utils.segmap2segimg(binary_maps, image, also_bbox=False, out_l=self.out_l)


class BBoxCrop(CropStrategy):
    """Bounding box with margin: the raw image inside the box, background kept, no mask."""

    def __init__(self, out_l: int, margin: int = 50):
        self.out_l = out_l
        self.margin = margin

    def __call__(self, image, binary_maps):
        bboxes_xyhw = segment_utils.batched_box_xyxy_to_xywh(segment_utils.batched_mask_to_box(binary_maps))
        crops = [
            F.resize(segment_utils.get_bbox_img(bboxes_xyhw[i], image, self.margin), (self.out_l, self.out_l))
            for i in range(binary_maps.shape[0])
        ]
        return torch.stack(crops, dim=0)


def build_crop(kind: str, out_l: int, margin: int = 50) -> CropStrategy:
    """`seg` (masked segment, default) or `bbox` (box with context)."""
    if kind == "seg":
        return SegCrop(out_l)
    if kind == "bbox":
        return BBoxCrop(out_l, margin)
    raise ValueError(f"Unknown crop strategy: {kind!r}. Valid: seg, bbox")
