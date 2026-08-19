"""Radar comparativo baseline vs contest: 3 paneles (mIoU / mAcc / AP_agn), una punta por escena.

Carga las métricas por escena de los artefactos de eval (statistics_<scene>.txt e
instance_ap_<scene>.txt), sin re-correr nada. Un polígono por experimento sobre cada radar.
"""
from __future__ import annotations

import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..common.resolve import resolve_exp_path

plt.rcParams["font.family"] = "Liberation Serif"  # clon métrico de Times New Roman

_METRICS = [("miou", "mIoU"), ("macc", "mAcc"), ("ap", "AP$_\\mathrm{agn}$")]


def _semantic(exp_path: pathlib.Path, scene: str) -> tuple[float, float]:
    """(mIoU, mAcc) en % = media sobre clases de statistics_<scene>.txt."""
    p = exp_path / "replica" / f"statistics_{scene}.txt"
    acc, iou = [], []
    for ln in p.read_text().splitlines():
        parts = [x.strip() for x in ln.split(",") if x.strip() != ""]
        if len(parts) < 3 or parts[0] == "label":
            continue
        acc.append(float(parts[1])); iou.append(float(parts[2]))
    return float(np.nanmean(iou)) * 100, float(np.nanmean(acc)) * 100


def _ap(exp_path: pathlib.Path, scene: str) -> float:
    """AP class-agnostic (%) de instance_ap_<scene>.txt."""
    p = exp_path / "replica" / f"instance_ap_{scene}.txt"
    for ln in p.read_text().splitlines():
        k, _, v = ln.partition(",")
        if k.strip() == "AP_agnostic":
            return float(v) * 100
    return float("nan")


def load_scene_metrics(exp: str, scenes: list[str]) -> dict[str, dict[str, float]]:
    """{'miou': {scene: val}, 'macc': {...}, 'ap': {...}} en % para un experimento."""
    ep = resolve_exp_path(exp)
    out = {"miou": {}, "macc": {}, "ap": {}}
    for s in scenes:
        mi, ma = _semantic(ep, s)
        out["miou"][s] = mi; out["macc"][s] = ma; out["ap"][s] = _ap(ep, s)
    return out


def radar_figure(baseline: str, contest: str, scenes: list[str], out_path: str) -> None:
    """3 radares (mIoU/mAcc/AP), una punta por escena, polígonos baseline vs contest."""
    b = load_scene_metrics(baseline, scenes)
    c = load_scene_metrics(contest, scenes)
    n = len(scenes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # cerrar el polígono

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), subplot_kw={"polar": True})
    c_base, c_cont = "#4c78a8", "#e45756"  # azul baseline, rojo contest
    for ax, (key, title) in zip(axes, _METRICS):
        vb = [b[key][s] for s in scenes]; vb += vb[:1]
        vc = [c[key][s] for s in scenes]; vc += vc[:1]
        vmax = max(max(vb), max(vc)) * 1.1
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(scenes, fontsize=11)
        ax.set_ylim(0, vmax)
        ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(alpha=0.4)
        ax.plot(angles, vb, color=c_base, linewidth=1.8, label="Baseline")
        ax.fill(angles, vb, color=c_base, alpha=0.15)
        ax.plot(angles, vc, color=c_cont, linewidth=1.8, label="Contest")
        ax.fill(angles, vc, color=c_cont, alpha=0.15)
    axes[0].legend(loc="upper left", bbox_to_anchor=(-0.25, 1.12), fontsize=12, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
