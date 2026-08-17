"""Lente prompts: la rejilla de puntos que el AMG pincha en cada frame.

Rendering puro. Recibe los puntos (normalizados [0,1]) y los pinta sobre el frame.
"""
from __future__ import annotations

import cv2
import numpy as np


def render(image: np.ndarray, points_norm: np.ndarray, out_path: str,
           dim: float = 0.75, radius: int = 4) -> None:
    """Dibuja la rejilla de prompts (Nx2 en [0,1]) sobre el frame -> out_path."""
    h, w = image.shape[:2]
    img = (image.astype(np.float32) * dim).astype(np.uint8)
    pts = (points_norm * np.array([w, h])).astype(int)
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), radius + 1, (0, 0, 0), -1, cv2.LINE_AA)      # borde
        cv2.circle(img, (int(x), int(y)), radius, (255, 60, 60), -1, cv2.LINE_AA)      # punto
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
