"""Frontera con el modelo: corre un AMG de SAM y devuelve máscaras CRUDAS (list[dict]).

Crudo = solo el NMS interno de SAM (cajas). El NMS externo de OVO se estudia aparte,
en `nms_decision`. Aquí vive el I/O de modelo (carga, warmup, GPU); nada de poda ni pintado.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ovo.utils.segment_utils import load_sam


@dataclass(frozen=True)
class PointMasks:
    """Las 3 máscaras multimask que SAM devuelve al pinchar un punto, con sus scores."""
    point: tuple[int, int]
    masks: list[np.ndarray]   # 3 x (H, W) bool
    scores: list[float]


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

    def predict_point(self, image: np.ndarray, xy: tuple[int, int]) -> PointMasks:
        """Pincha un punto (x,y) y devuelve las 3 máscaras multimask de SAM. Reusa el predictor del AMG."""
        return _predict_point(self._amg.predictor, image, xy, self.device, self._dtype)


class Sam3PointPredictor:
    """Predictor interactivo de SAM3: carga el modelo una vez y pincha puntos (frente A)."""

    def __init__(self, device: str = "cuda", sam3_path: str = "thirdParty/sam3") -> None:
        self.device = device
        self._dtype = torch.bfloat16
        if sam3_path not in sys.path:
            sys.path.append(sam3_path)
        from sam3.model_builder import build_sam3_video_model
        from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor

        video_model = build_sam3_video_model(load_from_HF=True, device=device)
        tracker = video_model.tracker
        tracker.backbone = video_model.detector.backbone
        self._pred = SAM3InteractiveImagePredictor(tracker)

    def predict_point(self, image: np.ndarray, xy: tuple[int, int]) -> PointMasks:
        """image RGB uint8 + punto (x,y) -> las 3 máscaras multimask de SAM3."""
        return _predict_point(self._pred, image, xy, self.device, self._dtype)


def _predict_point(predictor, image: np.ndarray, xy: tuple[int, int],
                   device: str, dtype: torch.dtype) -> PointMasks:
    """Corre un predictor interactivo (SAM2/SAM3) sobre un punto. Mismo contrato en ambos."""
    coords = np.array([[xy[0], xy[1]]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=dtype):
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(point_coords=coords, point_labels=labels, multimask_output=True)
        predictor.reset_predictor()
    return PointMasks(
        point=(int(xy[0]), int(xy[1])),
        masks=[np.asarray(m).astype(bool) for m in masks],
        scores=[float(s) for s in np.asarray(scores).ravel()],
    )
