"""Galería: una imagen por cada máscara cruda de SAM2, sobre el frame.

Rendering puro para inspeccionar máscara a máscara. Si se pasa un DecisionBreakdown,
el estado OVO (kept/descarte) va en el nombre del archivo, no sobre la imagen. Ni modelo ni NMS aquí.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from studies.segmentation.core.nms_decision import DecisionBreakdown


def _highlight(image: np.ndarray, seg: np.ndarray, kept: bool = True,
               fade: float = 0.8, tint: float = 0.35) -> np.ndarray:
    """Fondo gris pálido translúcido; la máscara translúcida en verde (sobrevive) o rojo (se elimina)."""
    img = image.astype(np.float32)
    gray = img.mean(axis=2, keepdims=True)                      # desatura
    faded = gray * (1 - 0.4) + 255 * 0.4                        # y aclara hacia gris pálido
    out = img.copy()
    bg = ~seg
    out[bg] = img[bg] * (1 - fade) + faded[bg] * fade           # fondo: gris pálido
    color = np.array([40, 220, 90], np.float32) if kept else np.array([230, 40, 40], np.float32)
    out[seg] = img[seg] * (1 - tint) + color * tint            # máscara: verde/rojo según destino
    border = seg.astype(np.uint8) - cv2.erode(seg.astype(np.uint8), np.ones((3, 3), np.uint8))
    out[border > 0] = color
    return out.astype(np.uint8)


def render(image: np.ndarray, records: list[dict], out_dir: str,
           breakdown: DecisionBreakdown | None = None, fade: float = 0.8) -> None:
    """Guarda mask_{id}[_estado].png por cada máscara cruda, sin texto sobre la imagen."""
    os.makedirs(out_dir, exist_ok=True)
    verdict = {v.index: v for v in breakdown.verdicts} if breakdown else {}

    for i, rec in enumerate(records):
        seg = rec["segmentation"].astype(bool)
        tag, kept = "", True
        if i in verdict:
            v = verdict[i]
            kept = v.kept
            tag = "_kept" if v.kept else "_out-" + "-".join(k.kind for k in v.kills)
        rgb = _highlight(image, seg, kept, fade)
        cv2.imwrite(os.path.join(out_dir, f"mask_{i:02d}{tag}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
