"""Galería: una imagen por cada máscara cruda de SAM2, sobre el frame.

Rendering puro para inspeccionar máscara a máscara. Si se pasa un DecisionBreakdown,
anota si OVO la conserva o la descarta (y por qué). Ni modelo ni NMS aquí.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from studies.segmentation.core.nms_decision import DecisionBreakdown


def _highlight(image: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Frame apagado con una sola máscara resaltada en verde + su borde."""
    img = image.astype(np.float32) * 0.4
    img[seg] = img[seg] * 0.4 + np.array([60, 220, 90], np.float32) * 0.6
    border = seg.astype(np.uint8) - cv2.erode(seg.astype(np.uint8), np.ones((3, 3), np.uint8))
    img[border > 0] = np.array([255, 255, 0], np.float32)
    return img.astype(np.uint8)


def render(image: np.ndarray, records: list[dict], out_dir: str, breakdown: DecisionBreakdown | None = None) -> None:
    """Guarda mask_{id}.png por cada máscara cruda en out_dir, con id/área/score y estado OVO."""
    os.makedirs(out_dir, exist_ok=True)
    verdict = {v.index: v for v in breakdown.verdicts} if breakdown else {}

    for i, rec in enumerate(records):
        seg = rec["segmentation"].astype(bool)
        area = int(seg.sum())
        score = float(rec["stability_score"] * rec["predicted_iou"])
        title = f"máscara {i}  ·  área={area}  ·  score={score:.3f}"
        if i in verdict:
            v = verdict[i]
            if v.kept:
                title += "  ·  OVO: KEPT"
            else:
                why = ", ".join(k.kind for k in v.kills)
                title += f"  ·  OVO: descartada ({why})"

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(_highlight(image, seg))
        ax.set_title(title, fontsize=13)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"mask_{i:02d}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)
