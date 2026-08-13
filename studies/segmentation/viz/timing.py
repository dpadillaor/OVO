"""Lente timing: barras de coste (total/encoder/decode) por modelo, con desviación."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from studies.segmentation.core.timing_stats import TimingStats

_METRICS = [("total", "total_ms"), ("encoder", "encoder_ms"), ("decode", "decode_ms")]


def render(named: dict[str, TimingStats], out_path: str, title: str = "") -> None:
    """Barras agrupadas: por métrica (total/encoder/decode), una barra por modelo, yerr=std."""
    labels = list(named)
    x = np.arange(len(_METRICS))
    w = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(labels):
        ts = named[label]
        means = [getattr(ts, attr).mean for _, attr in _METRICS]
        stds = [getattr(ts, attr).std for _, attr in _METRICS]
        ax.bar(x + i * w, means, w, yerr=stds, capsize=4,
               label=f"{label}  (crops={ts.n_crops}, masks~{ts.n_masks.mean:.0f}, vram~{ts.peak_vram_mb.mean:.0f}MB)")
    ax.set_xticks(x + w * (len(labels) - 1) / 2)
    ax.set_xticklabels([m for m, _ in _METRICS])
    ax.set_ylabel("ms")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
