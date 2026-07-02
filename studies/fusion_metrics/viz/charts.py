"""Visualization functions for fusion evaluation data. Each returns a plotly Figure."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px

from .loader import SceneFusionData, pool_verdicts

# --- colour palette ---
C_TP = "#2ecc71"
C_FP = "#e74c3c"
C_FN = "#f39c12"
C_TN = "#bdc3c7"
C_PRE = "#3498db"
C_POST = "#e67e22"
C_IMPROVED = "#27ae60"
C_WORSENED = "#c0392b"
C_UNCHANGED = "#95a5a6"


def confusion_heatmap(data: SceneFusionData) -> go.Figure:
    c = data.counts
    tp, fp, fn, tn = c.get("TP", 0), c.get("FP", 0), c.get("FN", 0), c.get("TN", 0)
    z = [[tp, fp], [fn, tn]]
    r = data.rates

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["SÍ (merge)", "NO (split)"],
            y=["SÍ", "NO"],
            text=[[f"TP: {tp}", f"FP: {fp}"], [f"FN: {fn}", f"TN: {tn}"]],
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            colorscale=[[0, C_TN], [0.25, C_FN], [0.5, C_FP], [0.75, C_TP], [1, C_TP]],
            showscale=False,
            zmin=0,
            zmax=max(tp, fp, fn, tn, 1),
        )
    )
    fig.update_layout(
        title="Fusion vs GT (verdicts)",
        xaxis_title="<b>Fusion decision</b>",
        yaxis_title="<b>GT: mismo objeto?</b>",
        xaxis_side="top",
        width=450,
        height=350,
        margin=dict(t=80, b=20, l=60, r=20),
    )
    fig.add_annotation(
        x=0.5, y=-0.22, xref="paper", yref="paper", showarrow=False,
        text=(f"Precision: {r.get('precision', 0):.3f}  |  "
              f"Recall: {r.get('recall', 0):.3f}  |  "
              f"F1: {r.get('f1', 0):.3f}"),
        font={"size": 11, "color": "#555"},
    )
    return fig


def gate_waterfall(data: SceneFusionData) -> go.Figure:
    """Horizontal stacked bars: TN + FN per rejection gate."""
    gates = sorted(data.by_criterion, key=lambda g: g.get("FN", 0), reverse=True)
    if not gates:
        return go.Figure()

    names = [g["criterion"] for g in gates]
    tn_vals = [g.get("TN", 0) for g in gates]
    fn_vals = [g.get("FN", 0) for g in gates]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=tn_vals, name="TN (correct rejection)",
        marker_color=C_TN, orientation="h",
        text=[f"TN: {v}" for v in tn_vals], textposition="inside",
        textfont={"color": "#555"},
    ))
    fig.add_trace(go.Bar(
        y=names, x=fn_vals, name="FN (missed merge)",
        marker_color=C_FN, orientation="h",
        text=[f"FN: {v}" for v in fn_vals], textposition="inside",
        textfont={"color": "white"},
    ))

    fig.update_layout(
        title="¿Dónde se pierde el recall? (FN por gate de rechazo)",
        barmode="stack",
        xaxis_title="Número de pares rechazados",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=250,
        margin=dict(t=60, b=20, l=100, r=20),
    )
    return fig


def accept_mode_bar(data: SceneFusionData) -> go.Figure:
    """Horizontal stacked bars: TP + FP per accept branch (OR mode A/B/AB).

    Reads the terminal gate's ``accept_modes`` from by_criterion. Pre-column runs
    carry a single '?' bucket.
    """
    terminal = next((g for g in reversed(data.by_criterion) if g.get("accept_modes")), None)
    if terminal is None:
        return go.Figure()

    order = {"A": 0, "B": 1, "AB": 2, "?": 3}
    modes = sorted(terminal["accept_modes"].items(), key=lambda x: order.get(x[0], 9))
    names = [m[0] for m in modes]
    tp_vals = [m[1].get("TP", 0) for m in modes]
    fp_vals = [m[1].get("FP", 0) for m in modes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=tp_vals, name="TP (merge correcto)",
        marker_color=C_TP, orientation="h",
        text=[f"TP: {v}" for v in tp_vals], textposition="inside",
        textfont={"color": "white"},
    ))
    fig.add_trace(go.Bar(
        y=names, x=fp_vals, name="FP (over-merge)",
        marker_color=C_FP, orientation="h",
        text=[f"FP: {v}" for v in fp_vals], textposition="inside",
        textfont={"color": "white"},
    ))
    fig.update_layout(
        title="¿Qué rama del OR acepta? (A=geom, B=sem, AB=ambas)",
        barmode="stack",
        xaxis_title="Número de pares aceptados",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=250,
        margin=dict(t=60, b=20, l=100, r=20),
    )
    return fig


def cascade_bars(data: SceneFusionData) -> go.Figure:
    """Per-gate cascade: TP + FP that each criterion lets pass (chain order, top-down)."""
    gates = data.by_criterion
    if not gates:
        return go.Figure()

    names = [g["criterion"] for g in gates]
    tp = [g.get("TP", 0) for g in gates]
    fp = [g.get("FP", 0) for g in gates]
    labels = [f"P{g.get('precision', 0) or 0:.2f} R{g.get('recall', 0) or 0:.2f}" for g in gates]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=tp, name="TP (pasa & bueno)", orientation="h", marker_color=C_TP,
        text=[f"TP:{v}" for v in tp], textposition="inside", textfont={"color": "white"},
    ))
    fig.add_trace(go.Bar(
        y=names, x=fp, name="FP (pasa & malo)", orientation="h", marker_color=C_FP,
        text=[f"FP:{v}" for v in fp], textposition="inside", textfont={"color": "white"},
    ))
    # precision/recall of each gate at the bar end
    for name, t, f, lab in zip(names, tp, fp, labels):
        fig.add_annotation(x=t + f, y=name, text=lab, showarrow=False,
                           xanchor="left", xshift=4, font={"size": 9, "color": "#555"})

    fig.update_layout(
        title="¿Qué deja pasar cada gate? (TP/FP + P/R)",
        barmode="stack",
        xaxis_title="Pares que pasan",
        yaxis=dict(autorange="reversed"),  # first gate on top (chain order)
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=250,
        margin=dict(t=60, b=20, l=100, r=60),
    )
    return fig


def ap_curves(data: SceneFusionData, mode: str = "objects") -> go.Figure:
    impact = data.agnostic_impact(mode)
    pre_pts = impact.get("pre", {}).get("per_threshold", [])
    post_pts = impact.get("post", {}).get("per_threshold", [])

    if not pre_pts or not post_pts:
        return go.Figure()

    ious = [p["iou"] for p in pre_pts]
    pre_ap = [p["ap"] for p in pre_pts]
    post_ap = [p["ap"] for p in post_pts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ious, y=pre_ap, mode="lines+markers", name="Pre-fusion",
        line=dict(color=C_PRE, width=2.5), marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=ious, y=post_ap, mode="lines+markers", name="Post-fusion",
        line=dict(color=C_POST, width=2.5), marker=dict(size=6),
    ))
    fig.add_annotation(
        x=0.5, y=0.05, xref="paper", yref="paper", showarrow=False,
        text=f"ΔAP50: {impact.get('delta_ap50', 0):+.4f}  |  "
             f"ΔAP_mean: {impact.get('delta_ap_mean', 0):+.4f}",
        font={"size": 10, "color": "#555"},
    )
    fig.update_layout(
        title=f"AP curves ({mode})",
        xaxis_title="IoU threshold",
        yaxis_title="AP",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=350,
        margin=dict(t=60, b=40, l=50, r=20),
    )
    return fig


def instance_composition(data: SceneFusionData, mode: str = "objects") -> go.Figure:
    impact = data.agnostic_impact(mode)
    pre = impact.get("pre", {}).get("per_threshold", [])
    post = impact.get("post", {}).get("per_threshold", [])

    # Use @0.5 for matched/spurious/missed
    pre50 = next((p for p in pre if p["iou"] == 0.5), pre[0] if pre else {})
    post50 = next((p for p in post if p["iou"] == 0.5), post[0] if post else {})

    categories = ["matched", "spurious", "missed"]
    pre_vals = [pre50.get(c, 0) for c in categories]
    post_vals = [post50.get(c, 0) for c in categories]
    deltas = [post_vals[i] - pre_vals[i] for i in range(3)]

    fig = go.Figure()
    x_pos = [0, 1, 2]
    bar_w = 0.35
    fig.add_trace(go.Bar(
        x=[p - bar_w / 2 for p in x_pos], y=pre_vals, name="Pre-fusion",
        marker_color=C_PRE, width=bar_w,
        text=pre_vals, textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=[p + bar_w / 2 for p in x_pos], y=post_vals, name="Post-fusion",
        marker_color=C_POST, width=bar_w,
        text=post_vals, textposition="outside",
    ))

    for i, d in enumerate(deltas):
        color = C_IMPROVED if d < 0 else (C_WORSENED if d > 0 else C_UNCHANGED)
        symbol = "↓" if d < 0 else ("↑" if d > 0 else "—")
        fig.add_annotation(
            x=x_pos[i], y=max(pre_vals[i], post_vals[i]) + max(pre_vals + post_vals) * 0.12,
            text=f"{symbol}{abs(d)}", showarrow=False,
            font={"color": color, "size": 12}, xshift=0,
        )

    fig.update_layout(
        title=f"Composición de instancias @0.5 ({mode})",
        xaxis=dict(tickmode="array", tickvals=x_pos, ticktext=categories),
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=350,
        margin=dict(t=60, b=40, l=50, r=20),
    )
    return fig


def instance_waterfall(data: SceneFusionData) -> go.Figure:
    df = data.non_background_stats
    if df.empty:
        return go.Figure()

    df = df.copy()
    df["delta_iou"] = df["iou_post"] - df["iou_pre"]
    df = df.sort_values("delta_iou")
    df["label"] = df.apply(
        lambda r: f"{r['gt_id']}: {r['name']}", axis=1
    )

    colors = df["status"].map({
        "improved": C_IMPROVED, "worsened": C_WORSENED, "unchanged": C_UNCHANGED,
    }).fillna(C_UNCHANGED)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["label"], x=df["delta_iou"],
        marker_color=colors.tolist(),
        orientation="h",
        text=[f"{v:+.3f}" for v in df["delta_iou"]],
        textposition="outside",
        textfont={"size": 10},
    ))

    fig.add_vline(x=0, line_dash="solid", line_color="#888", line_width=1)

    # Summary in subtitle
    n_imp = (df["status"] == "improved").sum()
    n_wor = (df["status"] == "worsened").sum()
    n_unc = (df["status"] == "unchanged").sum()

    fig.update_layout(
        title=f"Δ IoU por instancia ({n_imp}↑ {n_wor}↓ {n_unc}—)",
        xaxis_title="Δ IoU (post − pre)",
        yaxis_title="",
        height=max(300, 20 * len(df)),
        margin=dict(t=50, b=30, l=180, r=60),
        showlegend=False,
    )
    return fig


def instance_scatter(data: SceneFusionData) -> go.Figure:
    """Pre vs post IoU scatter with match_status coloring and diagonal."""
    df = data.non_background_stats
    if df.empty:
        return go.Figure()

    colors_map = {"KEPT": C_TP, "GAINED": C_IMPROVED, "LOST": C_WORSENED, "UNMATCHED": C_UNCHANGED}
    df = df.copy()
    df["color"] = df["match_status"].map(colors_map).fillna(C_UNCHANGED)

    max_iou = max(df["iou_pre"].max(), df["iou_post"].max(), 0.5) * 1.1

    fig = px.scatter(
        df, x="iou_pre", y="iou_post", color="match_status",
        color_discrete_map=colors_map,
        hover_data={"gt_id": True, "name": True, "iou_pre": ":.3f", "iou_post": ":.3f",
                     "status": True, "match_status": True},
        labels={"iou_pre": "IoU pre-fusion", "iou_post": "IoU post-fusion"},
    )

    # Diagonal: no change
    fig.add_trace(go.Scatter(
        x=[0, max_iou], y=[0, max_iou], mode="lines",
        line=dict(dash="dash", color="#888", width=1),
        name="sin cambio", showlegend=False,
    ))
    # AP threshold lines
    fig.add_hline(y=0.5, line_dash="dot", line_color="#aaa", annotation_text="IoU=0.5")
    fig.add_vline(x=0.5, line_dash="dot", line_color="#aaa")

    fig.update_layout(
        title="IoU pre vs post por instancia",
        xaxis=dict(range=[-0.02, max_iou]),
        yaxis=dict(range=[-0.02, max_iou]),
        height=500,
        margin=dict(t=50, b=40, l=50, r=20),
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    return fig


def epoch_heatmap(data: SceneFusionData) -> go.Figure:
    by_epoch = data.by_epoch
    if not by_epoch:
        return go.Figure()

    epochs = list(by_epoch.keys())
    metrics = ["TP", "FP", "FN", "TN", "precision", "recall", "f1"]

    z = []
    for epoch in epochs:
        e = by_epoch[epoch]
        row = []
        for m in metrics:
            if m in ("precision", "recall", "f1"):
                row.append(round(e.get("rates", {}).get(m, 0), 3))
            else:
                row.append(e.get("counts", {}).get(m, 0))
        z.append(row)

    # Format precision/recall/f1 to 3 decimals, counts as int
    cell_text = [[str(v) for v in row] for row in z]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=metrics,
        y=epochs,
        text=cell_text,
        texttemplate="%{text}",
        textfont={"size": 11},
        colorscale="RdYlGn",
        showscale=False,
    ))
    fig.update_layout(
        title="Métricas por epoch",
        xaxis_side="top",
        height=200,
        margin=dict(t=60, b=20, l=200, r=20),
    )
    return fig


def scene_dots(scenes: list[SceneFusionData], metric: str = "delta_ap50",
               mode: str = "objects") -> go.Figure:
    fig = go.Figure()

    x_vals = []
    y_vals = []
    texts = []
    for s in scenes:
        impact = s.agnostic_impact(mode)
        val = impact.get(metric, 0)
        x_vals.append(val)
        y_vals.append(s.scene)
        texts.append(f"{s.scene}: {val:+.4f}")

    colors = [C_IMPROVED if v > 0 else (C_WORSENED if v < 0 else C_UNCHANGED) for v in x_vals]

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="markers+text",
        marker=dict(size=14, color=colors, line=dict(width=1, color="white")),
        text=[f"{v:+.3f}" for v in x_vals],
        textposition="middle right",
        textfont={"size": 11},
        hovertext=texts,
        hoverinfo="text",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#888", line_width=1)
    fig.update_layout(
        title=f"{metric} por escena — {scenes[0].exp_id}" if scenes else "",
        xaxis_title=metric,
        yaxis_title="",
        height=max(150, 30 * len(scenes)),
        margin=dict(t=50, b=30, l=80, r=120),
        showlegend=False,
    )
    return fig


def compare_heatmap(experiments: dict[str, list[SceneFusionData]],
                    metric: str = "delta_ap50", mode: str = "objects") -> go.Figure:
    exps = sorted(experiments.keys())

    # Gather all scene names across all experiments
    all_scenes = set()
    for scenes in experiments.values():
        all_scenes.update(s.scene for s in scenes)
    all_scenes = sorted(all_scenes)

    z = []
    cell_text = []
    for exp in exps:
        scene_map = {s.scene: s for s in experiments[exp]}
        row = []
        row_txt = []
        for sc in all_scenes:
            if sc in scene_map:
                impact = scene_map[sc].agnostic_impact(mode)
                v = impact.get(metric, float("nan"))
            else:
                v = float("nan")
            row.append(v)
            row_txt.append(f"{v:+.3f}" if not (isinstance(v, float) and v != v) else "—")
        z.append(row)
        cell_text.append(row_txt)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=all_scenes, y=exps,
        text=cell_text, texttemplate="%{text}", textfont={"size": 11},
        colorscale="RdYlGn", zmid=0,
        showscale=True, colorbar=dict(title=metric),
    ))
    fig.update_layout(
        title=f"{metric} — experimentos × escenas",
        xaxis_side="top",
        height=max(200, 50 + 40 * len(exps)),
        margin=dict(t=80, b=20, l=300, r=60),
    )
    return fig


def compare_scatter(experiments: dict[str, list[SceneFusionData]],
                    mode: str = "objects") -> go.Figure:
    points = []
    for exp_id, scenes in experiments.items():
        pooled = pool_verdicts(scenes)
        # Aggregate AP impact
        total_delta_ap50 = 0.0
        total_pre_ap50 = 0.0
        total_post_ap50 = 0.0
        n = len(scenes)
        for s in scenes:
            imp = s.agnostic_impact(mode)
            total_delta_ap50 += imp.get("delta_ap50", 0)
            total_pre_ap50 += imp.get("pre", {}).get("ap50", 0)
            total_post_ap50 += imp.get("post", {}).get("ap50", 0)
        points.append(dict(
            exp_id=exp_id,
            precision=pooled["precision"],
            recall=pooled["recall"],
            delta_ap50=total_delta_ap50 / n if n > 0 else 0,
            pre_ap50=total_pre_ap50 / n if n > 0 else 0,
            post_ap50=total_post_ap50 / n if n > 0 else 0,
            n_scenes=n,
            TP=pooled["TP"], FP=pooled["FP"], FN=pooled["FN"],
        ))

    fig = go.Figure()
    for p in points:
        fig.add_trace(go.Scatter(
            x=[p["recall"]], y=[p["precision"]], mode="markers+text",
            name=p["exp_id"],
            text=p["exp_id"],
            textposition="top center",
            textfont={"size": 9},
            marker=dict(
                size=max(10, p["delta_ap50"] * 200 + 10),
                sizemode="area",
                line=dict(width=1, color="#333"),
            ),
            hovertemplate=(
                f"<b>{p['exp_id']}</b><br>"
                f"Precision: {p['precision']:.3f}<br>"
                f"Recall: {p['recall']:.3f}<br>"
                f"ΔAP50: {p['delta_ap50']:+.4f}<br>"
                f"Scenes: {p['n_scenes']}<br>"
                f"TP:{p['TP']} FP:{p['FP']} FN:{p['FN']}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Trade-off: precision vs recall (tamaño ∝ ΔAP50)",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=500,
        margin=dict(t=50, b=40, l=60, r=20),
        showlegend=False,
    )
    return fig


def scene_summary_blurb(data: SceneFusionData) -> str:
    """Single-sentence human-readable summary of the scene's fusion effect."""
    run = data.run_info
    imp = data.agnostic_impact("objects")
    r = data.rates
    return (
        f"{run.get('instances_pre', '?')}→{run.get('instances_post', '?')} instancias "
        f"({run.get('merges_applied', '?')} merges) | "
        f"AP50: {imp.get('pre', {}).get('ap50', 0):.3f}→{imp.get('post', {}).get('ap50', 0):.3f} "
        f"(Δ{imp.get('delta_ap50', 0):+.3f}) | "
        f"spurious: {imp.get('delta_spurious50', 0):+d} | "
        f"P={r.get('precision', 0):.2f} R={r.get('recall', 0):.3f}"
    )
