"""Lente Q1: el tríptico crudo | poda | segmap de un frame.

Rendering puro: recibe imagen + máscaras + DecisionBreakdown y pinta. Ni modelo ni NMS.
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from ovo.utils.segment_utils import mask2segmap
from studies.segmentation.core.nms_decision import DecisionBreakdown

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(h: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(_FONT, size=max(18, h // 35))
    except OSError:
        return ImageFont.load_default()


def _label(pil: Image.Image, seg: np.ndarray, text: str) -> None:
    ys, xs = np.where(seg)
    if len(ys) == 0:
        return
    cy, cx = int(ys.mean()), int(xs.mean())
    draw = ImageDraw.Draw(pil)
    font = _font(seg.shape[0])
    box = draw.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    draw.ellipse([cx - tw // 2 - 4, cy - th // 2 - 4, cx + tw // 2 + 4, cy + th // 2 + 4], fill=(0, 0, 0))
    draw.text((cx - tw // 2, cy - th // 2), text, fill=(255, 255, 255), font=font)


def _overlay(image: np.ndarray, segs: list[np.ndarray], removed: set[int]) -> np.ndarray:
    """Pinta cada máscara: kept en color, removed apagada en rojo, con su id."""
    colors = (plt.cm.tab20(np.linspace(0, 1, max(len(segs), 1)))[:, :3] * 255).astype(np.uint8)
    img = image.astype(np.float32) * 0.3
    for i, seg in enumerate(segs):
        if i in removed:
            img[seg] = img[seg] * 0.3 + np.array([200, 50, 50], np.float32) * 0.1
        else:
            img[seg] = img[seg] * 0.7 + colors[i % len(colors)] * 0.3
    for seg in segs:
        border = seg.astype(np.uint8) - cv2.erode(seg.astype(np.uint8), np.ones((3, 3), np.uint8))
        img[border > 0] = 0
    pil = Image.fromarray(img.astype(np.uint8))
    for i, seg in enumerate(segs):
        _label(pil, seg, "X" if i in removed else str(i))
    return np.array(pil)


def render(image: np.ndarray, records: list[dict], breakdown: DecisionBreakdown,
           out_path: str, raw_label: str = "SAM2") -> None:
    """Guarda el tríptico crudo(N) | poda(kept + X) | segmap(final) en out_path."""
    segs = [r["segmentation"].astype(bool) for r in records]
    removed = {v.index for v in breakdown.removed}
    kept_records = [records[v.index] for v in breakdown.kept]

    raw = _overlay(image, segs, removed=set())
    pruned = _overlay(image, segs, removed=removed)
    _, binary_maps = mask2segmap(kept_records, image, sort=True)
    segmap = _overlay(image, [b.astype(bool) for b in binary_maps], removed=set())

    titles = [
        f"1) {raw_label} crudo ({len(segs)})",
        f"2) NMS OVO ({len(kept_records)} kept, X={len(removed)})",
        f"3) segmap ({binary_maps.shape[0]} capas)",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(32, 9))
    for ax, im, t in zip(axes, [raw, pruned, segmap], titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=14)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
