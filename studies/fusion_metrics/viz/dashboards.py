"""Composite multi-panel dashboards for scene, experiment, and comparison levels."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .loader import SceneFusionData, pool_verdicts
from .charts import (
    confusion_heatmap, gate_waterfall, accept_mode_bar, ap_curves,
    instance_composition, instance_scatter, epoch_heatmap, cascade_bars,
    scene_dots, compare_heatmap, compare_scatter, scene_summary_blurb,
    C_IMPROVED, C_WORSENED, C_UNCHANGED,
)


def scene_dashboard(data: SceneFusionData, mode: str = "objects") -> go.Figure:
    blurb = scene_summary_blurb(data)

    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            "Matriz de confusión", "FN por gate de rechazo", "Rama del OR que acepta",
            f"Curvas AP ({mode})", "Composición de instancias @0.5", "IoU pre vs post por instancia",
            "¿Qué deja pasar cada gate?", "Métricas por epoch", "",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    # Row 1: confusion heatmap, reject-gate waterfall, accept-branch bar
    _embed(fig, confusion_heatmap(data), row=1, col=1)
    _embed(fig, gate_waterfall(data), row=1, col=2)
    _embed(fig, accept_mode_bar(data), row=1, col=3)

    # Row 2: AP curves, instance composition, instance scatter
    _embed(fig, ap_curves(data, mode), row=2, col=1)
    _embed(fig, instance_composition(data, mode), row=2, col=2)
    _embed(fig, instance_scatter(data), row=2, col=3)

    # Row 3: per-gate cascade + epoch heatmap (drift-only; blank otherwise)
    _embed(fig, cascade_bars(data), row=3, col=1)
    _embed(fig, epoch_heatmap(data), row=3, col=2)

    fig.update_layout(
        title=dict(
            text=f"<b>{data.exp_id}</b> / {data.scene}<br>"
                 f"<sup>{blurb}</sup>",
            font={"size": 14},
        ),
        height=1400,
        margin=dict(t=120, b=30, l=20, r=20),
        showlegend=True,
    )
    return fig


def experiment_summary(scenes: list[SceneFusionData],
                       mode: str = "objects") -> go.Figure:
    if not scenes:
        return go.Figure()

    exp_id = scenes[0].exp_id

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "ΔAP50 por escena",
            "Veredictos agregados (pooled)",
            f"AP50 pre vs post ({mode})",
            "Instancias pre vs post",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.1,
    )

    # (1,1): ΔAP50 dots per scene
    _embed(fig, scene_dots(scenes, "delta_ap50", mode), row=1, col=1)

    # (1,2): Pooled verdicts as a small heatmap
    pooled = pool_verdicts(scenes)
    c = go.Figure(data=go.Heatmap(
        z=[[pooled["TP"], pooled["FP"]], [pooled["FN"], pooled["TN"]]],
        x=["SÍ", "NO"],
        y=["SÍ", "NO"],
        text=[[f"TP:{pooled['TP']}", f"FP:{pooled['FP']}"],
              [f"FN:{pooled['FN']}", f"TN:{pooled['TN']}"]],
        texttemplate="%{text}", textfont={"size": 13, "color": "white"},
        colorscale="Blues", showscale=False,
    ))
    c.update_layout(
        xaxis_title="<b>Fusion</b>", yaxis_title="<b>GT</b>",
        xaxis_side="top", width=350, height=280,
        margin=dict(t=50, b=20, l=60, r=20),
    )
    c.add_annotation(x=0.5, y=-0.30, xref="paper", yref="paper", showarrow=False,
                     text=(f"P={pooled['precision']:.3f} R={pooled['recall']:.3f} "
                           f"F1={pooled['f1']:.3f} | {pooled['pairs_scored']} pares"),
                     font={"size": 10, "color": "#555"})
    _embed(fig, c, row=1, col=2)

    # (2,1): Pre vs post AP50 bars per scene
    scenes_sorted = sorted(scenes, key=lambda s: s.agnostic_impact(mode).get("delta_ap50", 0))
    ap_pre = [s.agnostic_impact(mode).get("pre", {}).get("ap50", 0) for s in scenes_sorted]
    ap_post = [s.agnostic_impact(mode).get("post", {}).get("ap50", 0) for s in scenes_sorted]
    scene_names = [s.scene for s in scenes_sorted]

    ap_fig = go.Figure()
    ap_fig.add_trace(go.Bar(y=scene_names, x=ap_pre, name="pre", orientation="h",
                             marker_color="#3498db", width=0.35))
    ap_fig.add_trace(go.Bar(y=scene_names, x=ap_post, name="post", orientation="h",
                             marker_color="#e67e22", width=0.35))
    ap_fig.update_layout(
        barmode="group", height=max(200, 35 * len(scenes)),
        xaxis_title="AP50", margin=dict(l=80, r=20, t=10, b=30),
    )
    _embed(fig, ap_fig, row=2, col=1)

    # (2,2): Instances pre vs post bars per scene
    n_pre = [s.run_info.get("instances_pre", 0) for s in scenes_sorted]
    n_post = [s.run_info.get("instances_post", 0) for s in scenes_sorted]
    n_merges = [s.run_info.get("merges_applied", 0) for s in scenes_sorted]

    inst_fig = go.Figure()
    inst_fig.add_trace(go.Bar(y=scene_names, x=n_pre, name="pre", orientation="h",
                               marker_color="#3498db", width=0.35))
    inst_fig.add_trace(go.Bar(y=scene_names, x=n_post, name="post", orientation="h",
                               marker_color="#e67e22", width=0.35))
    inst_fig.add_trace(go.Bar(
        y=scene_names, x=n_merges, name="merges", orientation="h",
        marker_color="#2ecc71", width=0.35, visible="legendonly",
    ))
    inst_fig.update_layout(
        barmode="group", height=max(200, 35 * len(scenes)),
        xaxis_title="Instancias", margin=dict(l=80, r=20, t=10, b=30),
    )
    _embed(fig, inst_fig, row=2, col=2)

    fig.update_layout(
        title=dict(
            text=f"<b>{exp_id}</b> — resumen de experimento ({len(scenes)} escenas)",
            font={"size": 15},
        ),
        height=900,
        margin=dict(t=80, b=30, l=20, r=20),
        showlegend=True,
    )
    return fig


def compare_dashboard(experiments: dict[str, list[SceneFusionData]],
                      mode: str = "objects") -> go.Figure:
    if not experiments:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "ΔAP50 por experimento × escena",
            "Precision vs Recall (tamaño ∝ ΔAP50)",
            "AP50 pre vs post agregado",
            "Composición agregada (pooled matched/spurious/missed)",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.1,
    )

    _embed(fig, compare_heatmap(experiments, "delta_ap50", mode), row=1, col=1)
    _embed(fig, compare_scatter(experiments, mode), row=1, col=2)

    # (2,1): Aggregated AP50 pre vs post per experiment
    exp_ids = sorted(experiments.keys())
    pooled_pre = []
    pooled_post = []
    for eid in exp_ids:
        scenes = experiments[eid]
        pre_vals = [s.agnostic_impact(mode).get("pre", {}).get("ap50", 0) for s in scenes]
        post_vals = [s.agnostic_impact(mode).get("post", {}).get("ap50", 0) for s in scenes]
        pooled_pre.append(sum(pre_vals) / len(pre_vals) if pre_vals else 0)
        pooled_post.append(sum(post_vals) / len(post_vals) if post_vals else 0)

    ap_sum_fig = go.Figure()
    ap_sum_fig.add_trace(go.Bar(
        y=exp_ids, x=pooled_pre, name="pre", orientation="h",
        marker_color="#3498db", text=[f"{v:.3f}" for v in pooled_pre],
        textposition="outside",
    ))
    ap_sum_fig.add_trace(go.Bar(
        y=exp_ids, x=pooled_post, name="post", orientation="h",
        marker_color="#e67e22", text=[f"{v:.3f}" for v in pooled_post],
        textposition="outside",
    ))
    ap_sum_fig.update_layout(
        barmode="group", height=max(200, 50 * len(exp_ids)),
        xaxis_title="AP50 (media entre escenas)", margin=dict(l=300, r=60, t=10, b=30),
    )
    _embed(fig, ap_sum_fig, row=2, col=1)

    # (2,2): Pooled composition
    comp_data = []
    for eid in exp_ids:
        scenes = experiments[eid]
        n_matched = 0
        n_spurious = 0
        n_missed = 0
        for s in scenes:
            post50 = next((p for p in s.agnostic_impact(mode).get("post", {}).get("per_threshold", [])
                          if p["iou"] == 0.5), {})
            n_matched += post50.get("matched", 0)
            n_spurious += post50.get("spurious", 0)
            n_missed += post50.get("missed", 0)
        comp_data.append((n_matched, n_spurious, n_missed))

    comp_fig = go.Figure()
    comp_fig.add_trace(go.Bar(
        y=exp_ids, x=[c[0] for c in comp_data], name="matched",
        orientation="h", marker_color=C_IMPROVED,
        text=[str(c[0]) for c in comp_data], textposition="inside",
    ))
    comp_fig.add_trace(go.Bar(
        y=exp_ids, x=[c[1] for c in comp_data], name="spurious",
        orientation="h", marker_color=C_WORSENED,
        text=[str(c[1]) for c in comp_data], textposition="inside",
    ))
    comp_fig.add_trace(go.Bar(
        y=exp_ids, x=[c[2] for c in comp_data], name="missed",
        orientation="h", marker_color=C_UNCHANGED,
        text=[str(c[2]) for c in comp_data], textposition="inside",
    ))
    comp_fig.update_layout(
        barmode="stack", height=max(200, 50 * len(exp_ids)),
        xaxis_title="Instancias @0.5 (suma entre escenas)",
        margin=dict(l=300, r=20, t=10, b=30),
    )
    _embed(fig, comp_fig, row=2, col=2)

    fig.update_layout(
        title=dict(
            text=f"<b>Comparación de experimentos</b> ({len(exp_ids)} exps, modo={mode})",
            font={"size": 15},
        ),
        height=1100,
        margin=dict(t=80, b=30, l=20, r=20),
        showlegend=True,
    )
    return fig


def _embed(target: go.Figure, source: go.Figure, *, row: int, col: int) -> None:
    """Copy all traces and layout annotations from source into target at the given subplot position."""
    for trace in source.data:
        target.add_trace(trace, row=row, col=col)
    # Copy axis titles from source
    if source.layout.xaxis and source.layout.xaxis.title and source.layout.xaxis.title.text:
        target.update_xaxes(title_text=source.layout.xaxis.title.text, row=row, col=col)
    if source.layout.yaxis and source.layout.yaxis.title and source.layout.yaxis.title.text:
        target.update_yaxes(title_text=source.layout.yaxis.title.text, row=row, col=col)
    # Copy annotations not attached to data (e.g. static text)
    for ann in source.layout.annotations or []:
        # Only copy standalone annotations, not those tied to subplots
        if not hasattr(ann, 'xref') or ann.xref == "paper":
            try:
                target.add_annotation(
                    x=ann.x, y=ann.y, text=ann.text,
                    xref=f"x{source._grid_ref[row][col] if hasattr(source, '_grid_ref') else ''}" if ann.xref == "x" else ann.xref,
                    yref=ann.yref,
                    showarrow=ann.showarrow,
                    font=ann.font,
                    xanchor=ann.xanchor,
                    yanchor=ann.yanchor,
                )
            except Exception:
                pass
