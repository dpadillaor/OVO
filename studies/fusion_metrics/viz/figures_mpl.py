"""Paper-grade static figures (matplotlib → SVG/PDF/PNG).

Plotly (charts.py) is for interactive dashboards; this module is for the printed
TFM/paper: vectorial, colorblind-safe (Okabe-Ito), grayscale-robust (colour =
series, marker = position), no interactivity.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .charts import _pr_trajectory
from .loader import SceneFusionData

# Okabe-Ito, colourblind-safe. Colour encodes the series (scene/run).
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]
# Marker encodes gate index (chain position) -> grayscale-robust.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
# Stable gate → colour so a gate reads the same across figures. overlap warm = slow.
GATE_COLORS = {"centroid": "#2a78d6", "cos_sim": "#1baf7a", "overlap": "#eb6834",
               "overlap_old": "#eb6834", "cooccurrence": "#4a3aa7", "aabb": "#eda100",
               "p_dist": "#e34948"}


def pr_trajectory_series(series: list[tuple[str, list[dict]]], title: str | None = None,
                         color_offset: int = 0):
    """One polyline per (label, by_criterion) series in the P-R plane.

    Colour = series, marker = gate index. Path runs right (recall 1, all pass) to
    left (accepted set). ``color_offset`` shifts the palette so a lone series keeps
    the colour it had in a combined figure. Returns the matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    gate_names: list[str] = []

    for i, (label, by_criterion) in enumerate(series):
        xs, ys, names = _pr_trajectory(by_criterion)
        if not xs:
            continue
        if len(names) > len(gate_names):
            gate_names = names
        color = OKABE_ITO[(i + color_offset) % len(OKABE_ITO)]
        ax.plot(xs, ys, "-", color=color, lw=1.5, zorder=2)
        for j, (x, y) in enumerate(zip(xs, ys)):
            ax.plot(x, y, MARKERS[j % len(MARKERS)], color=color, ms=6,
                    mfc="white", mew=1.4, zorder=3)
        ax.annotate(label, (xs[-1], ys[-1]), fontsize=8, color=color,
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("recall (good merges still alive / total)", fontsize=9)
    ax.set_ylabel("precision (good / alive)", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)

    # Marker legend = gate (chain order); colour legend is implicit via inline labels.
    if gate_names:
        handles = [Line2D([], [], marker=MARKERS[j % len(MARKERS)], color="0.3",
                          mfc="white", mew=1.4, ls="", ms=6, label=n)
                   for j, n in enumerate(gate_names)]
        ax.legend(handles=handles, title="gate", fontsize=7, title_fontsize=8,
                  loc="lower left", framealpha=0.9)

    fig.tight_layout()
    return fig


def pr_trajectory(scenes: list[SceneFusionData], title: str | None = None):
    """P-R trajectory, one polyline per scene."""
    return pr_trajectory_series([(s.scene, s.by_criterion) for s in scenes], title=title)


def pr_trajectory_pooled(pooled_by_criterion: list[dict],
                         scene_series: list[tuple[str, list[dict]]],
                         title: str | None = None, pooled_label: str = "pooled"):
    """Experiment-level P-R: faint per-scene polylines (spread) under one bold
    micro-average line (``pooled_by_criterion``, derived from pooled counts).

    Macro-averaging P-R points across scenes is meaningless when volumes differ,
    so the summary line comes from POOLED COUNTS, not an average of the curves.
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.4))

    # faint per-scene context, no markers/labels, low zorder
    for label, by_criterion in scene_series:
        xs, ys, _ = _pr_trajectory(by_criterion)
        if xs:
            ax.plot(xs, ys, "-", color="0.6", lw=1.0, alpha=0.35, zorder=1)

    # bold pooled micro-average line, markers per gate
    xs, ys, gate_names = _pr_trajectory(pooled_by_criterion)
    if xs:
        ax.plot(xs, ys, "-", color="#000000", lw=2.2, zorder=3)
        for j, (x, y) in enumerate(zip(xs, ys)):
            ax.plot(x, y, MARKERS[j % len(MARKERS)], color="#000000", ms=6,
                    mfc="white", mew=1.4, zorder=4)
        ax.annotate(pooled_label, (xs[-1], ys[-1]), fontsize=8, color="#000000",
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("recall (good merges still alive / total)", fontsize=9)
    ax.set_ylabel("precision (good / alive)", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)

    handles = [Line2D([], [], color="0.6", lw=1.0, alpha=0.6, label="per scene"),
               Line2D([], [], color="#000000", lw=2.2, label=pooled_label)]
    handles += [Line2D([], [], marker=MARKERS[j % len(MARKERS)], color="0.3",
                       mfc="white", mew=1.4, ls="", ms=6, label=n)
                for j, n in enumerate(gate_names)]
    ax.legend(handles=handles, fontsize=7, title_fontsize=8, loc="lower left",
              framealpha=0.9)

    fig.tight_layout()
    return fig


_EPOCH_ORDER = {"predrift_predrift": 0, "predrift_postdrift": 1, "postdrift_postdrift": 2}


def _sorted_epochs(scene: SceneFusionData) -> list[tuple[str, dict]]:
    return sorted(scene.by_epoch.items(), key=lambda kv: _EPOCH_ORDER.get(kv[0], 9))


def pr_trajectory_by_epoch(scene: SceneFusionData, title: str | None = None):
    """P-R trajectory for one scene, one polyline per drift epoch (all on one plane)."""
    series = [(ep, d.get("by_criterion", [])) for ep, d in _sorted_epochs(scene)]
    return pr_trajectory_series(series, title=title or f"{scene.scene} · by epoch")


def _save(fig, out: str | pathlib.Path) -> pathlib.Path:
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_pr_trajectory(scenes: list[SceneFusionData], out: str | pathlib.Path,
                       title: str | None = None) -> pathlib.Path:
    return _save(pr_trajectory(scenes, title=title), out)


def save_pr_trajectory_pooled(pooled_by_criterion: list[dict],
                              scene_series: list[tuple[str, list[dict]]],
                              out: str | pathlib.Path, title: str | None = None,
                              pooled_label: str = "pooled") -> pathlib.Path:
    return _save(pr_trajectory_pooled(pooled_by_criterion, scene_series,
                                      title=title, pooled_label=pooled_label), out)


def save_pr_trajectory_by_epoch(scene: SceneFusionData, out: str | pathlib.Path,
                                title: str | None = None) -> pathlib.Path:
    return _save(pr_trajectory_by_epoch(scene, title=title), out)


def save_pr_trajectory_epochs_separate(scene: SceneFusionData, figures_dir: str | pathlib.Path,
                                       ext: str = ".svg") -> list[pathlib.Path]:
    """One separate P-R figure per epoch (single polyline), palette matched to the combined."""
    figures_dir = pathlib.Path(figures_dir)
    saved = []
    for i, (ep, d) in enumerate(_sorted_epochs(scene)):
        fig = pr_trajectory_series([(ep, d.get("by_criterion", []))],
                                   title=f"{scene.scene} · {ep}", color_offset=i)
        saved.append(_save(fig, figures_dir / f"pr_trajectory_{ep}{ext}"))
    return saved


# --- cost / execution time ---

def _bubble_areas(values, r_min: float = 5.0, r_max: float = 13.0) -> list[float]:
    """Map values to scatter areas with the *radius* bounded to [r_min, r_max] points.

    Radius (not area) interpolates linearly over the value range, so even a 60× data
    spread yields at most an r_max/r_min (~3×) radius ratio — the big bubble stands out
    without swallowing the plot. Constant values collapse to r_max.
    """
    lo, hi = min(values), max(values)
    span = hi - lo
    rs = [(r_max if span == 0 else r_min + (r_max - r_min) * (v - lo) / span) for v in values]
    return [3.14159 * r * r for r in rs]


def cost_scatter(timings, title: str | None = None):
    """Decomposition view: each gate placed by volume × unit cost, bubble area = total time.

    X = pairs the gate handled (log), Y = ms per pair (log), bubble ∝ total seconds.
    Faint iso-cost diagonals (total = volume × unit) make the bubble area readable:
    a gate can be expensive by volume (cheap unit, huge N) or by unit (slow unit, small N).
    """
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ts = [t for t in timings if t.count and t.time_s]
    if not ts:
        ax.text(0.5, 0.5, "no timing logs", ha="center", va="center", transform=ax.transAxes)
        return fig

    xs = [t.count for t in ts]
    ys = [t.mean_s * 1e3 for t in ts]
    secs = [t.time_s for t in ts]
    smax = max(secs)
    areas = [90 + 2600 * (s / smax) for s in secs]  # bubble area ∝ total time
    colors = [GATE_COLORS.get(t.name, "#888780") for t in ts]

    ax.set_xscale("log")
    ax.set_yscale("log")
    # headroom so the top bubble + its label clear the title
    ax.set_xlim(min(xs) / 2.5, max(xs) * 3.2)
    ax.set_ylim(min(ys) / 3, max(ys) * 3.5)

    # iso-cost diagonals: total_s = count · (ms/1000) -> ms = 1000·total/count (slope -1 on log-log)
    xlo, xhi = ax.get_xlim()
    xlbl = xlo * 1.35  # labels hug the left end of each line, away from the bubbles
    for total_s in (0.1, 0.5, 1.0):
        ax.plot([xlo, xhi], [1e3 * total_s / xlo, 1e3 * total_s / xhi],
                ls=":", lw=0.8, color="0.6", zorder=1)
        ax.annotate(f"{total_s:g}s total", (xlbl, 1e3 * total_s / xlbl), fontsize=7,
                    color="0.5", ha="left", va="bottom")

    ax.scatter(xs, ys, s=areas, c=colors, alpha=0.75, edgecolors="white",
               linewidths=1.3, zorder=3)
    for t, x, y in zip(ts, xs, ys):
        # top-most bubble labels below-right to avoid the title; others above-right
        below = y > max(ys) / 1.5
        ax.annotate(f"{t.name}\n{t.mean_s * 1e3:.2g} ms · {t.time_s:.2f}s", (x, y),
                    fontsize=8, color=GATE_COLORS.get(t.name, "#444441"),
                    xytext=(9, -20 if below else 8), textcoords="offset points",
                    va="top" if below else "bottom")
    ax.set_xlabel("pairs handled (log)", fontsize=9)
    ax.set_ylabel("ms per pair (log)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="both", lw=0.3, alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def cost_unit_vs_total(timings, title: str | None = None, log: bool = True):
    """Just the two quantities that matter: unit cost (ms/pair) vs total time.

    X = ms per pair (intrinsic gate speed), Y = total seconds (realised cost); the pair
    count rides along in each label. ``log`` picks log-log (spreads decades) or linear.
    """
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ts = [t for t in timings if t.count and t.time_s]
    if not ts:
        ax.text(0.5, 0.5, "no timing logs", ha="center", va="center", transform=ax.transAxes)
        return fig

    xs = [t.mean_s * 1e3 for t in ts]
    ys = [t.time_s for t in ts]
    colors = [GATE_COLORS.get(t.name, "#888780") for t in ts]

    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min(xs) / 3, max(xs) * 3)
        ax.set_ylim(min(ys) / 3, max(ys) * 3.5)
    else:
        ax.set_xlim(0, max(xs) * 1.18)
        ax.set_ylim(0, max(ys) * 1.18)

    ax.scatter(xs, ys, s=140, c=colors, alpha=0.8, edgecolors="white",
               linewidths=1.3, zorder=3)
    for t, x, y in zip(ts, xs, ys):
        ax.annotate(f"{t.name}\n{t.time_s:.2f}s · {t.count:,} pairs", (x, y),
                    fontsize=8, color=GATE_COLORS.get(t.name, "#444441"),
                    xytext=(9, 8), textcoords="offset points")

    ax.set_xlabel("ms per pair", fontsize=9)
    ax.set_ylabel("total seconds", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="both", lw=0.3, alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def cost_attribution(timings, title: str | None = None):
    """Where the time goes: total seconds per gate, biggest first, with share labels."""
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ts = [t for t in timings if t.time_s]
    if not ts:
        ax.text(0.5, 0.5, "no timing logs", ha="center", va="center", transform=ax.transAxes)
        return fig

    ts = sorted(ts, key=lambda t: t.time_s, reverse=True)  # biggest on top
    total = sum(t.time_s for t in ts)
    names = [t.name for t in ts]
    secs = [t.time_s for t in ts]
    colors = [GATE_COLORS.get(t.name, "#888780") for t in ts]
    ypos = list(range(len(ts) - 1, -1, -1))  # first gate at the top

    ax.barh(ypos, secs, color=colors, height=0.6, zorder=3)
    for y, t in zip(ypos, ts):
        ax.annotate(f"{t.time_s:.2f}s · {t.time_s / total * 100:.0f}%",
                    (t.time_s, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=GATE_COLORS.get(t.name, "#444441"))

    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(f"total seconds  (Σ = {total:.2f}s)", fontsize=9)
    ax.set_xlim(0, max(secs) * 1.28)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="x", lw=0.3, alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def cost_total_vs_pairs(timings, title: str | None = None, log: bool = True):
    """Does time scale with work? Total seconds (Y) vs pairs handled (X).

    Mirror of unit-vs-total: here the ray slope from the origin is the unit cost
    (ms/pair). Gates on a common ray share a unit cost; a gate lifted above the others'
    ray is expensive per pair (overlap), one below is cheap (centroid). Unit cost rides
    in each label; ``log`` picks log-log or linear.
    """
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ts = [t for t in timings if t.count and t.time_s]
    if not ts:
        ax.text(0.5, 0.5, "no timing logs", ha="center", va="center", transform=ax.transAxes)
        return fig

    xs = [t.count for t in ts]
    ys = [t.time_s for t in ts]
    colors = [GATE_COLORS.get(t.name, "#888780") for t in ts]
    units = [t.mean_s for t in ts]
    areas = _bubble_areas(units)  # bubble ∝ ms/pair, radius bounded so the range can't explode

    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min(xs) / 3, max(xs) * 3)
        ax.set_ylim(min(ys) / 3, max(ys) * 3.5)
    else:
        ax.set_xlim(0, max(xs) * 1.18)
        ax.set_ylim(0, max(ys) * 1.18)

    ax.scatter(xs, ys, s=areas, c=colors, alpha=0.8, edgecolors="white",
               linewidths=1.3, zorder=3)
    for t, x, y, a in zip(ts, xs, ys, areas):
        pad = (a / 3.14159) ** 0.5 + 4  # clear the bubble edge (radius in points)
        ax.annotate(f"{t.name}\n{t.count:,} pairs\n{t.time_s:.2f}s · {t.mean_s * 1e3:.2g} ms/pair",
                    (x, y), fontsize=8, color=GATE_COLORS.get(t.name, "#444441"),
                    xytext=(pad, pad), textcoords="offset points")

    ax.set_xlabel("pairs handled", fontsize=9)
    ax.set_ylabel("total seconds", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="both", lw=0.3, alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def _load_timings(exp_path: str | pathlib.Path, scene: str, chain: list[str]):
    """Load per-gate timings for the given cascade names (handles overlap_old etc.)."""
    from ..core.timing.reader import load_criterion_timings  # sibling package, lazy
    return load_criterion_timings(pathlib.Path(exp_path), scene, criteria=chain or None)


def pool_timings(exp_path: str | pathlib.Path, scenes: list[SceneFusionData]):
    """Micro-average cost: per criterion, sum pairs (count) and seconds (time_s)
    across scenes. Totals are additive; the per-pair unit cost falls out of
    pooled_time / pooled_count (CriterionTiming.mean_s), never an average of
    per-scene means. Returns list[CriterionTiming] in chain order."""
    from ..core.timing.reader import load_criterion_timings, CriterionTiming  # lazy
    chain = [g["criterion"] for g in scenes[0].by_criterion] if scenes else []
    acc: dict[str, list] = {}
    order: list[str] = []
    for s in scenes:
        for t in load_criterion_timings(pathlib.Path(exp_path), s.scene, criteria=chain or None):
            if t.name not in acc:
                acc[t.name] = [0, 0.0]
                order.append(t.name)
            acc[t.name][0] += t.count
            acc[t.name][1] += t.time_s
    return [CriterionTiming(name=n, count=acc[n][0], time_s=acc[n][1]) for n in order]


_COST_VIEWS = {"decomp": cost_scatter, "unit-total": cost_unit_vs_total,
               "attribution": cost_attribution, "total-pairs": cost_total_vs_pairs}
_LOG_VIEWS = {"unit-total", "total-pairs"}


def _render_cost(timings, view: str, log: bool, title: str):
    builder = _COST_VIEWS[view]
    kwargs = {"log": log} if view in _LOG_VIEWS else {}
    return builder(timings, title=title, **kwargs)


def save_cost_scatter(scene_data: SceneFusionData, exp_path: str | pathlib.Path,
                      out: str | pathlib.Path, view: str = "decomp",
                      log: bool = True) -> pathlib.Path:
    chain = [g["criterion"] for g in scene_data.by_criterion]
    timings = _load_timings(exp_path, scene_data.scene, chain)
    return _save(_render_cost(timings, view, log, f"{scene_data.scene} · gate cost ({view})"), out)


def save_cost_scatter_pooled(exp_path: str | pathlib.Path, scenes: list[SceneFusionData],
                             out: str | pathlib.Path, view: str = "decomp",
                             log: bool = True) -> pathlib.Path:
    timings = pool_timings(exp_path, scenes)
    title = f"pooled · gate cost ({view}) ({len(scenes)} scenes)"
    return _save(_render_cost(timings, view, log, title), out)
