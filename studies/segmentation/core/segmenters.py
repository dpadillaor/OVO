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
    scores: list[float]       # predicted_iou (la 'confianza')
    stability: list[float]    # stability_score, el segundo umbral del AMG


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


class Sam3Segmenter:
    """AMG de SAM3: la maquinaria oficial del AMG de SAM2 (grid+filtros+NMS) con el predictor de SAM3."""

    def __init__(self, cfg: SamConfig, device: str = "cuda", sam3_path: str = "thirdParty/sam3") -> None:
        self.config = cfg
        self.device = device
        self._dtype = torch.bfloat16
        # Predictor interactivo del MODELO DE IMAGEN de SAM3 (misma puerta que el point predictor).
        if sam3_path not in sys.path:
            sys.path.append(sam3_path)
        import os
        import sam3
        from sam3 import build_sam3_image_model
        bpe = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
        model = build_sam3_image_model(bpe_path=bpe, enable_inst_interactivity=True)
        predictor = model.inst_interactive_predictor
        predictor.model.backbone = model.backbone
        self._amg = self._build_amg(predictor, cfg)
        self._warmup()

    @staticmethod
    def _build_amg(predictor, cfg: SamConfig):
        """Envuelve el predictor de SAM3 con el AMG. Delega en la fuente única de OVO para que el
        estudio y el pipeline construyan exactamente el mismo AMG (mismas máscaras garantizadas)."""
        from ovo.utils.segment_utils import build_amg_from_predictor
        return build_amg_from_predictor(
            predictor,
            points_per_side=cfg.points_per_side,
            crop_n_layers=cfg.crop_n_layers,
            pred_iou_thresh=cfg.pred_iou_thresh,
            stability_score_thresh=cfg.stability_score_thresh,
            min_mask_region_area=cfg.min_mask_region_area,
            use_m2m=cfg.use_m2m,
        )

    def _warmup(self) -> None:
        dummy = np.random.rand(512, 512, 3).astype(np.uint8)
        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=self._dtype):
            self._amg.generate(dummy)

    def segment(self, image: np.ndarray) -> list[dict]:
        """image (H,W,3) RGB uint8 -> máscaras crudas de SAM3 (mismo formato que SAM2)."""
        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=self._dtype):
            return self._amg.generate(image)

    def predict_point(self, image: np.ndarray, xy: tuple[int, int]) -> PointMasks:
        """Pincha un punto: reusa el predictor interactivo de SAM3 (iou + stability como el AMG)."""
        return _predict_point(self._amg.predictor, image, xy, self.device, self._dtype)


class Sam3PointPredictor:
    """Predictor de puntos de SAM3 (tarea SAM1), vía oficial de imagen del repo sam3."""

    def __init__(self, device: str = "cuda", sam3_path: str = "thirdParty/sam3") -> None:
        self.device = device
        self._dtype = torch.bfloat16
        if sam3_path not in sys.path:
            sys.path.append(sam3_path)
        import os
        import sam3
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        bpe = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
        self._model = build_sam3_image_model(bpe_path=bpe, enable_inst_interactivity=True)
        self._proc = Sam3Processor(self._model)

    def predict_point(self, image: np.ndarray, xy: tuple[int, int]) -> PointMasks:
        """image RGB uint8 + punto (x,y) -> las 3 máscaras multimask de SAM3 (predict_inst)."""
        from PIL import Image
        pil = Image.fromarray(image)
        coords = np.array([[xy[0], xy[1]]])
        labels = np.array([1])
        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=self._dtype):
            state = self._proc.set_image(pil)
            masks, scores, _ = self._model.predict_inst(
                state, point_coords=coords, point_labels=labels, multimask_output=True)
        return PointMasks(
            point=(int(xy[0]), int(xy[1])),
            masks=[np.asarray(m).astype(bool) for m in masks],
            scores=[float(s) for s in np.asarray(scores).ravel()],
        )


def _predict_point(predictor, image: np.ndarray, xy: tuple[int, int],
                   device: str, dtype: torch.dtype) -> PointMasks:
    """Corre un predictor interactivo (SAM2/SAM3) sobre un punto. Devuelve iou y stability como el AMG."""
    from sam2.utils.amg import calculate_stability_score
    coords = np.array([[xy[0], xy[1]]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=dtype):
        predictor.set_image(image)
        logits, scores, _ = predictor.predict(point_coords=coords, point_labels=labels,
                                              multimask_output=True, return_logits=True)
        if hasattr(predictor, "reset_predictor"):
            predictor.reset_predictor()
    thr = float(getattr(predictor, "mask_threshold", 0.0))
    lt = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    stability = calculate_stability_score(lt, thr, 1.0).numpy().ravel()
    return PointMasks(
        point=(int(xy[0]), int(xy[1])),
        masks=[(np.asarray(m) > thr).astype(bool) for m in logits],
        scores=[float(s) for s in np.asarray(scores).ravel()],
        stability=[float(s) for s in stability],
    )
