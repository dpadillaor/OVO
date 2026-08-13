"""Lente timing: barra horizontal apilada por modelo (encoder + decode + resto = total)."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from studies.segmentation.core.timing_stats import TimingStats

# segmento -> (atributo, color). Los 3 suman el total.
#   encoder = backbone sobre la imagen · decode = máscaras por punto · filtering =
#   limpiar candidatas (pred_iou, stability, box-NMS, RLE) + glue (overhead), no-modelo.
_SEGMENTS = [("encoder", "#2ca02c"), ("decode", "#1f77b4"), ("filtering", "#ff7f0e")]


def _parts(ts: TimingStats) -> dict[str, float]:
    return {"encoder": ts.encoder_ms.mean, "decode": ts.decode_ms.mean,
            "filtering": ts.post_ms.mean + ts.overhead_ms.mean}


def render(named: dict[str, TimingStats], out_path: str, title: str = "") -> None:
    """Una barra horizontal apilada por modelo; longitud total = total_ms, segmentos etiquetados."""
    labels = list(named)
    y = np.arange(len(labels))
    max_total = max(named[l].total_ms.mean for l in labels)
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
    ax.set_yticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_xlabel("ms")
    ax.set_title(title)
    ax.set_xlim(0, max_total * 1.15)  # hueco a la derecha para el label del total
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
