"""Lente timing: barra horizontal apilada por modelo (encoder + decode + resto = total)."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from studies.segmentation.core.timing_stats import TimingStats

# segmento -> (atributo, color). Los 4 suman el total.
#   post = stability/umbral/box-NMS/RLE por crop · overhead = crop-boxes/cross-NMS/ensamblado
_SEGMENTS = [("encoder", "#2ca02c"), ("decode", "#1f77b4"), ("post", "#ff7f0e"), ("overhead", "#9e9e9e")]


def _parts(ts: TimingStats) -> dict[str, float]:
    return {"encoder": ts.encoder_ms.mean, "decode": ts.decode_ms.mean,
            "post": ts.post_ms.mean, "overhead": ts.overhead_ms.mean}


def render(named: dict[str, TimingStats], out_path: str, title: str = "") -> None:
    """Una barra horizontal apilada por modelo; longitud total = total_ms, segmentos etiquetados."""
    labels = list(named)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 1.6 * len(labels) + 1.5))

    for i, label in enumerate(labels):
        parts = _parts(named[label])
        left = 0.0
        for seg, color in _SEGMENTS:
            w = parts[seg]
            ax.barh(i, w, left=left, color=color, edgecolor="white")
            if w > named[label].total_ms.mean * 0.06:  # etiqueta solo si cabe
                ax.text(left + w / 2, i, f"{seg}\n{w:.0f}ms", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            left += w
        ax.text(left + 3, i, f"{named[label].total_ms.mean:.0f} ms", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}\n(crops={named[l].n_crops}, masks~{named[l].n_masks.mean:.0f}, "
                        f"vram~{named[l].peak_vram_mb.mean:.0f}MB)" for l in labels])
    ax.set_xlabel("ms")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
