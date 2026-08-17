"""Lente compare: el segmap FINAL (tras poda OVO) de cada modelo, lado a lado.

Rendering puro. Recibe los binary_maps finales por modelo y los pinta junto al original.
"""
from __future__ import annotations

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FLAT_BG = np.array([40, 40, 40], np.float32)  # gris oscuro para píxeles sin máscara (modo flat)


def colored(image: np.ndarray, binary_maps: np.ndarray, dim: float = 0.3,
            alpha: float = 0.3, bg: str = "frame") -> np.ndarray:
    """Frame con las capas finales coloreadas (RGB uint8). Para juntar imágenes fuera.

    dim: brillo del fondo (1.0 = original). alpha: opacidad del color.
    bg: 'frame' = frame atenuado bajo las máscaras; 'flat' = gris liso donde no hay máscara
        (resalta los huecos de cobertura).
    """
    return _segmap(image, binary_maps, dim, alpha, bg)


def _segmap(image: np.ndarray, binary_maps: np.ndarray, dim: float = 0.3,
            alpha: float = 0.3, bg: str = "frame") -> np.ndarray:
    """Colorea las capas finales. bg='frame' atenúa el frame; bg='flat' pinta gris lo no cubierto."""
    n = len(binary_maps)
    colors = (plt.cm.tab20(np.linspace(0, 1, max(n, 1)))[:, :3] * 255).astype(np.uint8)
    if bg == "flat":
        covered = np.any(binary_maps.astype(bool), axis=0) if n else np.zeros(image.shape[:2], bool)
        img = image.astype(np.float32) * dim
        img[~covered] = FLAT_BG
        h, w = image.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        hatch = ((xx + yy) % 14 < 2) & ~covered  # rayas diagonales cada 14 px sobre el hueco
        img[hatch] = FLAT_BG + 22
    else:
        img = image.astype(np.float32) * dim
    for i, seg in enumerate(binary_maps):
        seg = seg.astype(bool)
        img[seg] = img[seg] * (1 - alpha) + colors[i % len(colors)] * alpha
    for seg in binary_maps:
        seg = seg.astype(np.uint8)
        border = seg - cv2.erode(seg, np.ones((3, 3), np.uint8))
        img[border > 0] = 0
    return img.astype(np.uint8)


def contours(image: np.ndarray, binary_maps: np.ndarray, dim: float = 0.5,
             thickness: int = 1) -> np.ndarray:
    """Solo el borde de cada máscara (RGB uint8). Revela solapes: nada se tapa por relleno."""
    n = len(binary_maps)
    colors = (plt.cm.tab20(np.linspace(0, 1, max(n, 1)))[:, :3] * 255).astype(np.uint8)
    img = (image.astype(np.float32) * dim).astype(np.uint8)
    for i, seg in enumerate(binary_maps):
        cnts, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, tuple(int(c) for c in colors[i % len(colors)]),
                         thickness, cv2.LINE_AA)
    return img


def heat(image: np.ndarray, binary_maps: np.ndarray, dim: float = 0.35) -> np.ndarray:
    """Mapa de calor del nº de máscaras que cubren cada píxel. Rojo = muy redundante."""
    count = binary_maps.astype(np.uint16).sum(axis=0) if len(binary_maps) else \
        np.zeros(image.shape[:2], np.uint16)
    img = image.astype(np.float32) * dim
    cmax = max(int(count.max()), 1)
    norm = (count.astype(np.float32) / cmax)
    hot = (plt.cm.turbo(norm)[:, :, :3] * 255).astype(np.float32)
    m = count > 0
    img[m] = img[m] * 0.25 + hot[m] * 0.75
    return img.astype(np.uint8)


def removed_overlay(image: np.ndarray, kept: np.ndarray, removed: np.ndarray,
                    dim: float = 0.45) -> np.ndarray:
    """Las máscaras que el mask_nms de OVO descarta, en rojo; las que sobreviven, contorno gris."""
    img = image.astype(np.float32) * dim
    for seg in removed:                       # relleno rojo = descartada
        m = seg.astype(bool)
        img[m] = img[m] * 0.4 + np.array([230, 40, 40], np.float32) * 0.6
    out = img.astype(np.uint8)
    for seg in kept:                          # contorno gris = superviviente (contexto)
        cnts, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def _centroid(seg: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.nonzero(seg)
    return (int(xs.mean()), int(ys.mean())) if len(xs) else None


def number_masks(img: np.ndarray, binary_maps: np.ndarray, min_dist: int = 32,
                 scale: float = 0.8) -> np.ndarray:
    """Escribe el índice de cada máscara en su centroide; empuja la etiqueta si choca con otra."""
    out = img.copy()
    placed: list[tuple[int, int]] = []
    for i, seg in enumerate(binary_maps):
        c = _centroid(seg.astype(np.uint8))
        if c is None:
            continue
        x, y = c
        for _ in range(12):                       # anti-solape: sube la etiqueta hasta despejar
            if all(abs(x - px) + abs(y - py) >= min_dist for px, py in placed):
                break
            y -= 14
        placed.append((x, y))
        txt = str(i)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        org = (x - tw // 2, y + th // 2)
        cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def removed_pairs(image: np.ndarray, pairs: list[tuple[np.ndarray, np.ndarray | None]],
                  dim: float = 0.45) -> np.ndarray:
    """Cada máscara descartada (rojo) unida por una línea a la superviviente que la eliminó.

    pairs: (máscara_descartada, máscara_asesina|None). None = descartada por low_score (sin culpable).
    """
    img = image.astype(np.float32) * dim
    for rem, _ in pairs:
        m = rem.astype(bool)
        img[m] = img[m] * 0.5 + np.array([230, 40, 40], np.float32) * 0.5
    out = img.astype(np.uint8)
    for rem, kil in pairs:
        cr = _centroid(rem)
        if cr is None:
            continue
        if kil is None:                                   # low_score: sin asesina
            cv2.circle(out, cr, 5, (60, 120, 255), -1, cv2.LINE_AA)
            continue
        cnts, _ = cv2.findContours(kil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (60, 220, 60), 2, cv2.LINE_AA)  # asesina: contorno verde
        ck = _centroid(kil)
        if ck is not None:
            cv2.line(out, cr, ck, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(out, cr, 3, (230, 40, 40), -1, cv2.LINE_AA)
    return out


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
