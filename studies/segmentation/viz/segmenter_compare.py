"""Lente compare: el segmap FINAL (tras poda OVO) de cada modelo, lado a lado.

Rendering puro. Recibe los binary_maps finales por modelo y los pinta junto al original.
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def colored(image: np.ndarray, binary_maps: np.ndarray) -> np.ndarray:
    """Frame apagado con las capas finales coloreadas (RGB uint8). Para juntar imágenes fuera."""
    return _segmap(image, binary_maps)


def _segmap(image: np.ndarray, binary_maps: np.ndarray) -> np.ndarray:
    """Colorea las capas finales sobre el frame apagado."""
    n = len(binary_maps)
    colors = (plt.cm.tab20(np.linspace(0, 1, max(n, 1)))[:, :3] * 255).astype(np.uint8)
    img = image.astype(np.float32) * 0.3
    for i, seg in enumerate(binary_maps):
        seg = seg.astype(bool)
        img[seg] = img[seg] * 0.7 + colors[i % len(colors)] * 0.3
    for seg in binary_maps:
        seg = seg.astype(np.uint8)
        border = seg - cv2.erode(seg, np.ones((3, 3), np.uint8))
        img[border > 0] = 0
    return img.astype(np.uint8)


def render(image: np.ndarray, named_binmaps: dict[str, np.ndarray], out_path: str) -> None:
    """Guarda [original | modelo_1 final | modelo_2 final ...] con el nº de capas finales."""
    labels = list(named_binmaps)
    fig, axes = plt.subplots(1, len(labels) + 1, figsize=(11 * (len(labels) + 1), 7))
    axes[0].imshow(image)
    axes[0].set_title("original", fontsize=14)
    axes[0].axis("off")
    for ax, label in zip(axes[1:], labels):
        bm = named_binmaps[label]
        ax.imshow(_segmap(image, bm))
        ax.set_title(f"{label} final ({len(bm)} capas)", fontsize=14)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
