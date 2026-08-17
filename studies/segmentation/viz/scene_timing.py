"""Lente scene: coste y nº de máscaras por frame a lo largo de una escena (líneas por modelo)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from studies.segmentation.core.profiling import FrameProfile

_STYLE = {"sam2": ("#e67e22", "o"), "sam3": ("#2980b9", "s")}


def render(frames: list[int], series: dict[str, list[FrameProfile]], out_path: str,
           coverage: dict[str, list[float]] | None = None, title: str = "") -> None:
    """Tres paneles por frame (línea por modelo): tiempo total (ms), nº de máscaras y cobertura."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for model, profs in series.items():
        color, marker = _STYLE.get(model, (None, "o"))
        ax1.plot(frames, [p.total_ms for p in profs], marker=marker, color=color, ms=3, lw=1.2, label=model)
        ax2.plot(frames, [p.n_masks for p in profs], marker=marker, color=color, ms=3, lw=1.2, label=model)
        if coverage:
            ax3.plot(frames, [c * 100 for c in coverage[model]], marker=marker, color=color,
                     ms=3, lw=1.2, label=model)
    ax1.set_ylabel("tiempo total (ms)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("máscaras crudas")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax3.set_ylabel("cobertura (% imagen)")
    ax3.set_xlabel("frame")
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
