"""Gráficas de actividad del contest: merges, splits e instancias por escena.

Reusa `query.summary.summarize` para los datos (verdicts + instancias, baseline vs contest).
Cada `chart_*` guarda una figura; `render_all` las vuelca todas para comparar.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..query.summary import summarize

plt.rcParams["font.family"] = "Liberation Serif"  # clon métrico de Times New Roman

_MERGE, _SPLIT = "#e45756", "#4c78a8"   # merges rojo, splits azul
_BASE, _CONT = "#bab0ac", "#59a14f"     # instancias base gris, contest verde


def _data(baseline: str, contest: str, scenes: list[str]) -> dict:
    return summarize(contest, scenes, baseline=baseline)


def _scenes_vals(rows, key):
    return [r[key] for r in rows]


def chart_merges_splits(data: dict, out_path: str) -> None:
    """Barras agrupadas por escena: merges vs splits."""
    rows = data["rows"]; scenes = [r["scene"] for r in rows]
    x = np.arange(len(scenes)); w = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - w / 2, _scenes_vals(rows, "merge"), w, color=_MERGE, label="Merges")
    ax.bar(x + w / 2, _scenes_vals(rows, "split"), w, color=_SPLIT, label="Splits")
    ax.set_xticks(x); ax.set_xticklabels(scenes, fontsize=11)
    ax.set_ylabel("Veredictos aplicados", fontsize=12)
    ax.legend(fontsize=11, frameon=False); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def chart_instances(data: dict, out_path: str, figsize=(6.6, 5.0), fs=15) -> None:
    """Barras agrupadas por escena: nº instancias baseline vs contest.

    figsize/fs pequeños y fuentes grandes: pensado para ir en un minipage a ~0.6\\textwidth
    sin que el escalado deje el texto ilegible.
    """
    rows = data["rows"]; scenes = [r["scene"] for r in rows]
    x = np.arange(len(scenes)); w = 0.4
    vb = _scenes_vals(rows, "ins_base"); vc = _scenes_vals(rows, "ins_contest")
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - w / 2, vb, w, color=_BASE, label="Baseline")
    ax.bar(x + w / 2, vc, w, color=_CONT, label="Contest")
    top = max(max(vb), max(vc))
    for xi, v in zip(x - w / 2, vb):
        ax.text(xi, v + top * 0.01, str(v), ha="center", va="bottom", fontsize=fs - 5)
    for xi, v in zip(x + w / 2, vc):
        ax.text(xi, v + top * 0.01, str(v), ha="center", va="bottom", fontsize=fs - 5)
    ax.set_xticks(x); ax.set_xticklabels(scenes, fontsize=fs - 2, rotation=35, ha="right")
    ax.set_ylabel("Number of instances", fontsize=fs)
    ax.tick_params(axis="y", labelsize=fs - 3)
    ax.set_ylim(0, top * 1.12)
    leg = ax.legend(fontsize=fs - 1, frameon=True, loc="upper right", edgecolor="black", facecolor="white")
    leg.get_frame().set_linewidth(1.0); leg.get_frame().set_alpha(1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def chart_combo(data: dict, out_path: str) -> None:
    """2 paneles: izq merges/splits, dcha instancias base vs contest."""
    rows = data["rows"]; scenes = [r["scene"] for r in rows]
    x = np.arange(len(scenes)); w = 0.4
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 4.4))
    a.bar(x - w / 2, _scenes_vals(rows, "merge"), w, color=_MERGE, label="Merges")
    a.bar(x + w / 2, _scenes_vals(rows, "split"), w, color=_SPLIT, label="Splits")
    a.set_title("Actividad", fontsize=14, fontweight="bold")
    a.set_ylabel("Veredictos aplicados", fontsize=12)
    b.bar(x - w / 2, _scenes_vals(rows, "ins_base"), w, color=_BASE, label="Baseline")
    b.bar(x + w / 2, _scenes_vals(rows, "ins_contest"), w, color=_CONT, label="Contest")
    b.set_title("Instancias del mapa", fontsize=14, fontweight="bold")
    b.set_ylabel("Nº de instancias", fontsize=12)
    for ax in (a, b):
        ax.set_xticks(x); ax.set_xticklabels(scenes, fontsize=10, rotation=30)
        ax.legend(fontsize=11, frameon=False); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def chart_stacked(data: dict, out_path: str) -> None:
    """Barras apiladas merges+splits por escena, con Δinstancias anotado."""
    rows = data["rows"]; scenes = [r["scene"] for r in rows]
    x = np.arange(len(scenes))
    m = _scenes_vals(rows, "merge"); s = _scenes_vals(rows, "split")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x, m, 0.6, color=_MERGE, label="Merges")
    ax.bar(x, s, 0.6, bottom=m, color=_SPLIT, label="Splits")
    for i, r in enumerate(rows):
        d = r["ins_contest"] - r["ins_base"]
        ax.text(i, m[i] + s[i] + 1.5, f"{d:+d} ins", ha="center", fontsize=8, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(scenes, fontsize=11)
    ax.set_ylabel("Veredictos aplicados", fontsize=12)
    ax.legend(fontsize=11, frameon=False); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def chart_scatter(data: dict, out_path: str) -> None:
    """Scatter: tamaño de escena (instancias baseline) vs actividad total (merges+splits)."""
    rows = data["rows"]
    size = np.array([r["ins_base"] for r in rows], float)
    act = np.array([r["merge"] + r["split"] for r in rows], float)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(size, act, s=70, color=_SPLIT, zorder=3)
    for r, xv, yv in zip(rows, size, act):
        ax.annotate(r["scene"], (xv, yv), textcoords="offset points", xytext=(6, 4), fontsize=10)
    if len(size) > 1:
        a, b = np.polyfit(size, act, 1)
        xs = np.linspace(size.min(), size.max(), 50)
        r2 = np.corrcoef(size, act)[0, 1] ** 2
        ax.plot(xs, a * xs + b, "--", color="#e45756", alpha=0.8, label=f"fit ($R^2$={r2:.2f})")
        ax.legend(fontsize=11, frameon=False)
    ax.set_xlabel("Instancias baseline (tamaño de escena)", fontsize=12)
    ax.set_ylabel("Merges + Splits", fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def render_all(baseline: str, contest: str, scenes: list[str], out_dir: str) -> list[str]:
    import os
    os.makedirs(out_dir, exist_ok=True)
    data = _data(baseline, contest, scenes)
    outs = []
    for name, fn in [("merges_splits", chart_merges_splits), ("instances", chart_instances),
                     ("combo", chart_combo), ("stacked", chart_stacked), ("scatter", chart_scatter)]:
        p = os.path.join(out_dir, f"activity_{name}.png")
        fn(data, p); outs.append(p)
    return outs
