"""Lente point: pinchar UN punto -> las 3 máscaras dispares que devuelve SAM.

Muestra la ambigüedad de granularidad: el mismo punto puede ser objeto, parte o todo.
Rendering puro sobre un PointMasks. Ni modelo aquí.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from studies.segmentation.core.segmenters import PointMasks


def _panel(image: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Frame apagado con una máscara resaltada."""
    img = image.astype(np.float32) * 0.45
    img[seg] = img[seg] * 0.4 + np.array([60, 220, 90], np.float32) * 0.6
    return img.astype(np.uint8)


def render(image: np.ndarray, pm: PointMasks, out_path: str) -> None:
    """Guarda: prompt (con el punto) + las 3 máscaras multimask con su score."""
    order = np.argsort(pm.scores)[::-1]
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))

    axes[0].imshow(image)
    axes[0].scatter([pm.point[0]], [pm.point[1]], c="lime", s=200, edgecolors="black", zorder=5)
    axes[0].set_title(f"prompt  ({pm.point[0]}, {pm.point[1]})", fontsize=13)
    axes[0].axis("off")

    for ax, rank in zip(axes[1:], order):
        seg = pm.masks[rank]
        ax.imshow(_panel(image, seg))
        ax.set_title(f"máscara  score={pm.scores[rank]:.3f}  área={int(seg.sum())}", fontsize=13)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
