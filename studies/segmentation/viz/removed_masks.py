"""Lente removed: por cada máscara descartada, quién la mató y la geometría del solape.

Rojo = eliminada · azul = asesina(s) · amarillo = solape. Rendering puro sobre records + breakdown.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from studies.segmentation.core.nms_decision import DecisionBreakdown, MaskVerdict


def _draw(image: np.ndarray, dead: np.ndarray, killers: list[np.ndarray]) -> np.ndarray:
    """Frame con la muerta en rojo, las asesinas en azul y el solape en amarillo."""
    img = image.astype(np.float32)
    overlap = np.zeros(dead.shape, bool)
    for k in killers:
        overlap |= dead & k
    for k in killers:
        only = k & ~overlap
        img[only] = img[only] * 0.5 + np.array([40, 90, 230], np.float32) * 0.5
    dead_only = dead & ~overlap
    img[dead_only] = img[dead_only] * 0.5 + np.array([230, 60, 60], np.float32) * 0.5
    img[overlap] = img[overlap] * 0.3 + np.array([255, 240, 60], np.float32) * 0.7
    for seg, col in [(dead, (255, 0, 0))] + [(k, (0, 120, 255)) for k in killers]:
        border = seg.astype(np.uint8) - cv2.erode(seg.astype(np.uint8), np.ones((3, 3), np.uint8))
        img[border > 0] = np.array(col, np.float32)
    return img.astype(np.uint8)


def _caption(v: MaskVerdict) -> str:
    lines = [f"máscara {v.index} descartada  (área={v.area}, score={v.score:.3f})"]
    for k in v.kills:
        if k.killer_index is None:
            lines.append(f"  · {k.kind}: score {k.value:.3f} <= umbral")
        else:
            lines.append(f"  · {k.kind}={k.value:.3f}  vs máscara {k.killer_index}")
    return "\n".join(lines)


def render(image: np.ndarray, records: list[dict], breakdown: DecisionBreakdown, out_dir: str) -> None:
    """Guarda un png por cada máscara descartada, con asesina(s) y métricas."""
    os.makedirs(out_dir, exist_ok=True)
    for v in breakdown.removed:
        dead = records[v.index]["segmentation"].astype(bool)
        killers = [records[k.killer_index]["segmentation"].astype(bool)
                   for k in v.kills if k.killer_index is not None]
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.imshow(_draw(image, dead, killers))
        ax.set_title(_caption(v), fontsize=12, loc="left")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"mask_{v.index:02d}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
