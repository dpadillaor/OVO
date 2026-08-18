"""Lente cost: coste por frame de un AMG, tiempo apilado (encoder|mask decoder|filtering) + VRAM pico."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Liberation Serif"  # clon métrico de Times New Roman en Linux

# segmento -> (etiqueta legible, color). Los 3 suman el total, coherentes con el texto del cap. 5:
#   encoder de visión · mask decoder (máscaras por punto) · filtering (poda interna + OVO, no-modelo).
_SEGMENTS = [("encoder", "Encoder", "#6baed6"),
             ("decode", "Mask decoder", "#fd8d3c"),
             ("filtering", "Filtering", "#74c476")]


def _parts_ms(m: dict) -> dict[str, float]:
    """De un bloque del timing.json (medias) a los 3 segmentos en ms."""
    g = lambda k: m[k]["mean"]
    return {"encoder": g("encoder_ms"), "decode": g("decode_ms"),
            "filtering": g("post_ms") + g("overhead_ms")}


def render(data: dict, out_path: str) -> None:
    """`data` = {modelo: bloque timing.json}. Izq: tiempo apilado por modelo. Dcha: VRAM pico (GB)."""
    labels = list(data)
    y = np.arange(len(labels))
    fig, (ax_t, ax_m) = plt.subplots(
        1, 2, figsize=(11, 1.0 * len(labels) + 1.2), gridspec_kw={"width_ratios": [3, 1]})

    # --- izquierda: tiempo apilado ---
    max_total = max(sum(_parts_ms(data[l]).values()) for l in labels)
    for i, label in enumerate(labels):
        parts = _parts_ms(data[label])
        total = sum(parts.values())
        left = 0.0
        for key, name, color in _SEGMENTS:
            w = parts[key]
            ax_t.barh(i, w, left=left, color=color, edgecolor="white",
                      label=name if i == 0 else None)
            if w > max_total * 0.05:  # ms del tramo, solo si cabe
                ax_t.text(left + w / 2, i, f"{w:.0f}", ha="center", va="center",
                          color="white", fontsize=11, fontweight="bold")
            left += w
        ax_t.text(total + max_total * 0.01, i, f"{total:.0f} ms", va="center",
                  fontsize=12, fontweight="bold")
    ax_t.set_yticks(y)
    ax_t.set_yticklabels(labels, fontsize=14, fontweight="bold")
    ax_t.set_xlabel("Time per frame (ms)", fontsize=15)
    ax_t.tick_params(axis="x", labelsize=12)
    ax_t.set_xlim(0, max_total * 1.14)
    ax_t.invert_yaxis()
    ax_t.legend(loc="upper right", frameon=True, fontsize=11)

    # --- derecha: VRAM pico ---
    vram_gb = [data[l]["peak_vram_mb"]["mean"] / 1024.0 for l in labels]
    colors = [c for _, _, c in _SEGMENTS]
    bars = ax_m.bar(y, vram_gb, color=[colors[i % len(colors)] for i in range(len(labels))],
                    width=0.6, edgecolor="white")
    for rect, v in zip(bars, vram_gb):
        ax_m.text(rect.get_x() + rect.get_width() / 2, v + max(vram_gb) * 0.02,
                  f"{v:.1f} GB", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax_m.set_xticks(y)
    ax_m.set_xticklabels(labels, fontsize=13, fontweight="bold")
    ax_m.set_ylabel("Peak VRAM (GB)", fontsize=15)
    ax_m.tick_params(axis="y", labelsize=11)
    ax_m.set_ylim(0, max(vram_gb) * 1.18)
    ax_m.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
