"""Lente scene: coste y nº de máscaras por frame a lo largo de una escena (líneas por modelo)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from studies.segmentation.core.profiling import FrameProfile

_STYLE = {"sam2": ("#e67e22", "o"), "sam3": ("#2980b9", "s")}


def render(frames: list[int], series: dict[str, list[FrameProfile]], out_path: str, title: str = "") -> None:
    """Dos paneles: tiempo total (ms) y nº de máscaras crudas por frame, una línea por modelo."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for model, profs in series.items():
        color, marker = _STYLE.get(model, (None, "o"))
        ax1.plot(frames, [p.total_ms for p in profs], marker=marker, color=color, ms=3, lw=1.2, label=model)
        ax2.plot(frames, [p.n_masks for p in profs], marker=marker, color=color, ms=3, lw=1.2, label=model)
    ax1.set_ylabel("tiempo total (ms)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("máscaras crudas")
    ax2.set_xlabel("frame")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
