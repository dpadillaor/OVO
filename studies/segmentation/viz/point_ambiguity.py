"""Lente point: pinchar UN punto -> las 3 máscaras dispares que devuelve SAM.

Muestra la ambigüedad de granularidad: el mismo punto puede ser objeto, parte o todo.
Rendering puro sobre un PointMasks. Ni modelo aquí.
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Liberation Serif"  # clon métrico de Times New Roman

from studies.segmentation.core.segmenters import PointMasks


def _panel(image: np.ndarray, seg: np.ndarray, point: tuple[int, int] | None = None) -> np.ndarray:
    """Frame visible con la máscara translúcida en verde + borde amarillo; marca el prompt."""
    seg = seg.astype(bool)
    img = image.astype(np.float32) * 0.6
    img[seg] = img[seg] * 0.55 + np.array([50, 220, 100], np.float32) * 0.45
    out = img.astype(np.uint8)
    m = seg.astype(np.uint8)
    border = m - cv2.erode(m, np.ones((3, 3), np.uint8))
    out[border > 0] = (255, 235, 0)
    if point is not None:
        cv2.circle(out, point, 9, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, point, 6, (255, 40, 40), -1, cv2.LINE_AA)
    return out


def _prompt_panel(image: np.ndarray, point: tuple[int, int]) -> np.ndarray:
    out = image.copy()
    cv2.circle(out, point, 10, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(out, point, 7, (255, 40, 40), -1, cv2.LINE_AA)
    return out


def render_compare(image: np.ndarray, named: dict[str, PointMasks], out_path: str) -> None:
    """Una fila por modelo (SAM2/SAM3): prompt + sus 3 máscaras ordenadas por score. Frente A."""
    labels = list(named)
    n = len(labels)
    col_w = 22 / 4
    row_h = col_w * image.shape[0] / image.shape[1]  # altura = ancho de columna * aspecto imagen
    fig, axes = plt.subplots(n, 4, figsize=(22, row_h * n),
                             gridspec_kw={"wspace": 0.015, "hspace": 0.015})
    if n == 1:
        axes = axes[None, :]
    for r, label in enumerate(labels):
        pm = named[label]
        pt = (int(pm.point[0]), int(pm.point[1]))
        axes[r, 0].imshow(_prompt_panel(image, pt))
        axes[r, 0].set_ylabel(label.upper(), fontsize=16, rotation=90, labelpad=10, fontweight="bold")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("Prompt", fontsize=14, fontweight="bold")
        for c, (ax, rank) in enumerate(zip(axes[r, 1:], np.argsort(pm.scores)[::-1])):
            seg = pm.masks[rank]
            ax.imshow(_panel(image, seg, pt))
            ax.text(0.02, 0.05, f"conf {pm.scores[rank]:.3f}\nstab {pm.stability[rank]:.3f}",
                    transform=ax.transAxes, fontsize=13, color="white", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.6))
            ax.axis("off")
            if r == 0:
                ax.set_title(f"Mask {c + 1}", fontsize=14, fontweight="bold")
    fig.subplots_adjust(wspace=0.015, hspace=0.015, left=0.02, right=0.999, top=0.95, bottom=0.005)
    fig.savefig(out_path, dpi=150, pad_inches=0)
    plt.close(fig)


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
