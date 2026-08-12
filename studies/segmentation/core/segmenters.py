"""Frontera con el modelo: corre un AMG de SAM y devuelve máscaras CRUDAS (list[dict]).

Crudo = solo el NMS interno de SAM (cajas). El NMS externo de OVO se estudia aparte,
en `nms_decision`. Aquí vive el I/O de modelo (carga, warmup, GPU); nada de poda ni pintado.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ovo.utils.segment_utils import load_sam


@dataclass(frozen=True)
class SamConfig:
    """Parámetros de generación de un AMG de SAM (semántica de SAM2, no del NMS de OVO)."""
    sam_ckpt_path: str
    sam_version: str = "2.1"
    sam_encoder: str = "hiera_l"
    points_per_side: int = 16
    crop_n_layers: int = 0
    pred_iou_thresh: float = 0.8
    stability_score_thresh: float = 0.95
    min_mask_region_area: int = 0
    use_m2m: bool = False

    def _to_load_sam_dict(self) -> dict[str, Any]:
        # load_sam usa nombres propios ("nms_iou_th" -> pred_iou_thresh, etc.); traducimos aquí.
        return {
            "sam_version": self.sam_version,
            "sam_encoder": self.sam_encoder,
            "sam_ckpt_path": self.sam_ckpt_path,
            "points_per_side": self.points_per_side,
            "crop_n_layers": self.crop_n_layers,
            "nms_iou_th": self.pred_iou_thresh,
            "stability_score_th": self.stability_score_thresh,
            "min_mask_region_area": self.min_mask_region_area,
            "use_m2m": self.use_m2m,
        }


class SamSegmenter:
    """Carga un AMG de SAM una vez y segmenta frames. `segment` devuelve máscaras crudas."""

    def __init__(self, config: SamConfig, device: str = "cuda") -> None:
        self.config = config
        self.device = device
        self._dtype = torch.float32 if config.sam_version == "" else torch.bfloat16
        self._amg = load_sam(config._to_load_sam_dict(), device=device)
        self._warmup()

    def _warmup(self) -> None:
        dummy = np.random.rand(512, 512, 3).astype(np.uint8)
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=self._dtype):
            self._amg.generate(dummy)

    def segment(self, image: np.ndarray) -> list[dict]:
        """image (H,W,3) RGB uint8 -> máscaras crudas de SAM (cada dict: segmentation, predicted_iou, stability_score, ...)."""
        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=self._dtype):
            return self._amg.generate(image)
