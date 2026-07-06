"""Per-signal static figures (matplotlib -> SVG/PDF/PNG) for contest Tier2 telemetry.

One figure per logger signal: value vs KF index, with jump keyframes marked. Colour
encodes the signal so it reads the same across figures; jumps are dashed vlines.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .loader import SceneContestData, SIGNAL_META

# Okabe-Ito, colourblind-safe. Stable signal -> colour.
SIGNAL_COLORS = {
    "n_matched":    "#0072B2",
    "n_pre_assign": "#56B4E9",
    "n_used":       "#E69F00",
    "n_orphans":    "#D55E00",
    "n_births":     "#009E73",
    "n_robos":      "#CC79A7",
    "robo_rate":    "#CC79A7",
    "orphan_rate":  "#D55E00",
}
_FALLBACK = "#333333"

_RATE_META = {
    "robo_rate":   ("Grab rate (n_robos / n_pre_assign)", "rate"),
    "orphan_rate": ("Orphan rate (n_orphans / n_matched)", "rate"),
}


def _style(ax) -> None:
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _mark_jumps(ax, jumps: list[dict]) -> None:
    for j in jumps:
        ax.axvline(j["kf"], color="0.25", ls="--", lw=1.0, alpha=0.7, zorder=1)
        ax.annotate(j["label"], xy=(j["kf"], 1.0), xycoords=("data", "axes fraction"),
                    fontsize=7, color="0.25", rotation=90,
                    ha="right", va="top", xytext=(-2, -2), textcoords="offset points")


def signal_figure(data: SceneContestData, signal: str):
    """One line plot for a raw or derived signal. Returns the matplotlib Figure."""
    if signal in data.series:
        y = data.series[signal]
        title, ylabel = SIGNAL_META.get(signal, (signal, "value"))
        marker = "." if signal == "n_births" else None  # sparse count reads better with dots
    elif signal in data.derived:
        y = data.derived[signal]
        title, ylabel = _RATE_META.get(signal, (signal, "rate"))
        marker = None
    else:
        raise KeyError(f"Signal '{signal}' not in scene {data.exp_id}/{data.scene}")

    x = range(len(y))
    color = SIGNAL_COLORS.get(signal, _FALLBACK)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(x, y, "-", color=color, lw=1.2, marker=marker, ms=3, zorder=2)
    _mark_jumps(ax, data.jumps)
    _style(ax)
    ax.set_xlabel("KF idx", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f"{title}  ·  {data.exp_id}/{data.scene}", fontsize=9)
    ax.set_xlim(-0.5, max(len(y) - 0.5, 1))
    fig.tight_layout()
    return fig


def save_scene_figures(data: SceneContestData, figures_dir: str | pathlib.Path,
                       ext: str = "svg", derived: bool = True, dpi: int = 120) -> list[pathlib.Path]:
    """Write one figure per raw signal (+ derived rates) into figures_dir. Returns paths."""
    figures_dir = pathlib.Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    names = list(data.signals)
    if derived:
        names += list(data.derived.keys())

    saved: list[pathlib.Path] = []
    for signal in names:
        fig = signal_figure(data, signal)
        out = figures_dir / f"contest_{signal}.{ext}"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        saved.append(out)
    return saved
