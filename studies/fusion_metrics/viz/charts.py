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


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# Sankey semantic palette — a traffic-light ramp matching the top-to-bottom band
# order TP→FP→FN→TN: green = go (correct merge), red = stop (correct reject), and
# the two errors sit as amber caution between them. Correct outcomes own the vivid
# extremes; mistakes are the softer transition at the split seam. Nodes stay neutral.
Q_TP, Q_FP, Q_FN, Q_TN = "#4caf7d", "#ef8e3b", "#f2c94c", "#d1495b"
NODE_SLATE = "#3d4b5c"
REJECT_NODE = "#a8384c"  # deeper garnet of the TN red — ties the sinks to "reject"


def gate_sankey(by_criterion: list[dict], garnet_reject_nodes: bool = True,
                title: str = "Fusion gate cascade") -> go.Figure:
    """Cascade funnel per gate: geometry = pass (up) vs reject (down), colour = TP/FP/FN/TN correctness.

    garnet_reject_nodes: reject sink nodes in garnet red (True) or neutral slate (False).
    """
    if not by_criterion:
        return go.Figure()
    reject_node = REJECT_NODE if garnet_reject_nodes else NODE_SLATE

    gates = by_criterion  # already in cascade order (eval[i] == pass[i-1])
    n = len(gates)
    eps = 1e-9  # nudge fixed positions off the 0/1 edges (plotly clamps exact edges)

    # Depths 0..n map into [X_MIN, X_MAX] (not full [0,1]) so neither edge column
    # touches the border: centroid gets breathing room on the left, and the result
    # column keeps its reject label on the right instead of plotly flipping it.
    X_MIN, X_MAX = 0.03, 0.82

    def depth_x(d: int) -> float:
        return X_MIN + d / n * (X_MAX - X_MIN)

    total = gates[0].get("eval", 0) or 1  # full inflow; scales node heights
    labels, node_colors, node_x, node_y = [], [], [], []

    def add(label: str, color: str, x: float, y: float) -> int:
        labels.append(label); node_colors.append(color)
        node_x.append(min(max(x, eps), 1 - eps))
        node_y.append(min(max(y, eps), 1 - eps))
        return len(labels) - 1

    # depth d in [0, n]. gate_i sits at depth i; the two outputs of gate_i land at
    # depth i+1: survivor (next gate / accept) on top, its reject sibling stacked
    # right below — same column, touching, heights ∝ volume, so the split reads
    # as one incoming flow cleanly bisected. y is each node's centre (0 = top).
    def col_y(above: int, height: int) -> float:
        return (above + height / 2) / total

    # gate/accept nodes stay unlabelled — the column header carries criterion +
    # eval total; only reject sinks label their leaked count. All reject sinks
    # hang from one waterline (baseline = first survivor band) instead of hugging
    # their shrinking survivor, so late gates don't crowd the top.
    baseline = gates[0].get("pass", 0)
    # Node labels left empty; reject counts are placed as annotations below so we
    # control which side they sit on (plotly auto-flips sink labels unpredictably).
    gate_idx, reject_idx = [], []
    for i, g in enumerate(gates):
        gate_idx.append(add("", NODE_SLATE, depth_x(i), col_y(0, g.get("eval", 0))))
        reject_idx.append(add("", reject_node,
                              depth_x(i + 1), col_y(baseline, g.get("reject", 0))))
    accept = add("", NODE_SLATE, depth_x(n), col_y(0, gates[-1].get("pass", 0)))

    # Each flow splits by confusion quadrant. Pass = TP (right merge) + FP (wrong
    # merge that slipped through); both go to the next gate. Reject = TN (right
    # reject) + FN (missed merge); both drop to the reject sink. Order good-first
    # so correct outcomes hug the survivor rail, errors band toward the split seam.
    src, tgt, val, link_colors, link_kind = [], [], [], [], []

    def link(s: int, t: int, v: int, color: str, kind: str) -> None:
        src.append(s); tgt.append(t); val.append(v)
        link_colors.append(color); link_kind.append(kind)

    for i, g in enumerate(gates):
        pass_tgt = gate_idx[i + 1] if i + 1 < n else accept
        link(gate_idx[i], pass_tgt, g.get("TP", 0), _hex_to_rgba(Q_TP, 0.65), "TP")
        link(gate_idx[i], pass_tgt, g.get("FP", 0), _hex_to_rgba(Q_FP, 0.65), "FP")
        link(gate_idx[i], reject_idx[i], g.get("FN", 0), _hex_to_rgba(Q_FN, 0.7), "FN")
        link(gate_idx[i], reject_idx[i], g.get("TN", 0), _hex_to_rgba(Q_TN, 0.55), "TN")

    assert len(src) == len(tgt) == len(val), "sankey link arrays must match"

    def pair(la: str, va: int, ca: str, lb: str, vb: int, cb: str) -> str:
        return (f"<span style='color:{ca}'>{la} {va:,}</span> · "
                f"<span style='color:{cb}'>{lb} {vb:,}</span>")

    # Headers carry only column identity (name + eval). The confusion counts live
    # at their terminal node instead: each reject sink is FN + TN, the final merges
    # are TP + FP — a complete, non-duplicated split with no header clutter.
    headers = [(depth_x(i), f"<b>{g['criterion']}</b><br>{g.get('eval', 0):,} eval")
               for i, g in enumerate(gates)]
    headers.append((depth_x(n), "<b>Result</b><br>&nbsp;"))  # blank eval line to align names
    annotations = [
        dict(x=x, y=1.03, xref="paper", yref="paper", text=text, showarrow=False,
             xanchor="center", yanchor="bottom", align="center",
             font=dict(size=11, color="#555"))
        for x, text in headers
    ]

    # Terminal counts placed by hand right of each node. Sankey node y is 0 = top,
    # so paper y (0 = bottom) is 1 - node_y; x nudged past the bar. A reject sink is
    # FN + TN, the accept node is TP + FP — shown as a sub-line under the total.
    def terminal(idx: int, total_txt: str, color: str, comp: str) -> dict:
        return dict(
            x=node_x[idx] + 0.025, y=1 - node_y[idx], xref="paper", yref="paper",
            text=f"<b>{total_txt}</b><br><span style='font-size:9px'>{comp}</span>",
            showarrow=False, xanchor="left", yanchor="middle", align="center",
            font=dict(size=11, color=color), bgcolor="rgba(255,255,255,0.75)",
            borderpad=1)

    for i, g in enumerate(gates):
        comp = pair("FN", g.get("FN", 0), Q_FN, "TN", g.get("TN", 0), Q_TN)
        annotations.append(terminal(reject_idx[i], f"−{g.get('reject', 0):,}", "#333", comp))
    last = gates[-1]
    comp = pair("TP", last.get("TP", 0), Q_TP, "FP", last.get("FP", 0), Q_FP)
    annotations.append(terminal(accept, f"{last.get('pass', 0):,} merges", "#2e8b57", comp))

    # Pass composition (TP + FP) for the intermediate transitions, floated on the
    # pass band in each gap — the final gate's pass already shows at the accept node.
    for i in range(n - 1):
        g = gates[i]
        y_pass = 1 - g.get("pass", 0) / (2 * total)  # centre of the pass band
        annotations.append(dict(
            x=(depth_x(i) + depth_x(i + 1)) / 2, y=y_pass, xref="paper", yref="paper",
            text=f"<span style='font-size:9px'>"
                 f"{pair('TP', g.get('TP', 0), Q_TP, 'FP', g.get('FP', 0), Q_FP)}</span>",
            showarrow=False, xanchor="center", yanchor="middle",
            bgcolor="rgba(255,255,255,0.75)", borderpad=1))

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        valueformat=",.0f",
        node=dict(label=labels, color=node_colors, x=node_x, y=node_y,
                  pad=2, thickness=18, line=dict(color="#888", width=0.5),
                  hovertemplate="%{label}<extra></extra>"),
        link=dict(source=src, target=tgt, value=val, color=link_colors,
                  customdata=link_kind,
                  hovertemplate="%{source.label} → %{target.label}<br>"
                                "%{customdata}: %{value}<extra></extra>"),
    ))

    # Sankey links don't populate a legend, so add dummy no-data scatter traces —
    # one per quadrant — purely to render a real, native, positionable legend.
    legend = [(Q_TP, "TP — correct merge"), (Q_FP, "FP — wrong merge"),
              (Q_FN, "FN — missed merge"), (Q_TN, "TN — correct reject")]
    for color, name in legend:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=name, showlegend=True,
            marker=dict(size=12, color=color, symbol="square")))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center",
                   y=0.975, yanchor="top", font=dict(size=15)),
        font=dict(family="Helvetica, Arial, sans-serif", size=12, color="#333"),
        width=820, height=560, margin=dict(t=110, b=50, l=40, r=30),
        shapes=[dict(type="line", xref="paper", yref="paper", x0=0.02, x1=0.98,
                     y0=1.18, y1=1.18, line=dict(color="#ccc", width=1))],
        annotations=annotations,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="white",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.04, yanchor="top",
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    itemsizing="constant"),
    )
    return fig


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


# Confusion quadrants in the Sankey's traffic-light order + labels, reused so the
# donut reads as the same four colours as the cascade.
_QUADRANTS = [
    ("TP", Q_TP, "TP — correct merge"),
    ("FP", Q_FP, "FP — wrong merge"),
    ("FN", Q_FN, "FN — missed merge"),
    ("TN", Q_TN, "TN — correct reject"),
]


def _donut_trace(counts: dict, hole_title: str) -> go.Pie:
    """A single confusion-split donut (same colours/order as the cascade)."""
    labels = [lab for _, _, lab in _QUADRANTS]
    values = [counts.get(k, 0) for k, _, _ in _QUADRANTS]
    colors = [c for _, c, _ in _QUADRANTS]
    return go.Pie(
        labels=labels, values=values, marker=dict(colors=colors, line=dict(color="white", width=1)),
        hole=0.55, sort=False, direction="clockwise", rotation=0,
        textinfo="percent", texttemplate="%{percent:.1%}", textposition="outside",
        title=dict(text=hole_title, font=dict(size=13, color="#333")),
        hovertemplate="%{label}<br>%{value:,} (%{percent:.1%})<extra></extra>",
    )


def confusion_donut(data: SceneFusionData, title: str = "Confusion split") -> go.Figure:
    """Donut of the global confusion split (TP/FP/FN/TN), cascade colours + percentages."""
    total = sum(data.counts.get(k, 0) for k, _, _ in _QUADRANTS)
    fig = go.Figure(_donut_trace(data.counts, f"{total:,}<br>pairs"))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center", font=dict(size=15)),
        width=460, height=440, margin=dict(t=70, b=60, l=20, r=20),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.08, yanchor="top",
                    font=dict(size=11)),
        paper_bgcolor="white",
    )
    return fig


def confusion_donut_by_epoch(by_epoch: dict[str, dict]) -> go.Figure:
    """Small-multiples: one confusion donut per drift epoch, shared colours/legend.

    Donuts sit in the lower band (domain y ≤ 0.80) so the outside percent labels of
    tiny TP/FP slices never reach the epoch header floating above each column.
    """
    order = {"predrift_predrift": 0, "predrift_postdrift": 1, "postdrift_postdrift": 2}
    epochs = sorted(by_epoch.items(), key=lambda kv: order.get(kv[0], 9))
    if not epochs:
        return go.Figure()

    n = len(epochs)
    gap = 0.04
    fig = go.Figure()
    annotations = []
    for i, (ep, d) in enumerate(epochs):
        counts = d.get("counts", {})
        total = sum(counts.get(k, 0) for k, _, _ in _QUADRANTS)
        tr = _donut_trace(counts, f"{total:,}")
        tr.showlegend = (i == 0)  # one shared legend
        x0, x1 = i / n + gap, (i + 1) / n - gap
        tr.domain = dict(x=[x0, x1], y=[0.0, 0.80])  # leave the top strip for the header
        fig.add_trace(tr)
        annotations.append(dict(
            x=(x0 + x1) / 2, y=0.97, xref="paper", yref="paper", showarrow=False,
            text=f"<b>{ep}</b>", font=dict(size=12, color="#333"),
            xanchor="center", yanchor="bottom"))

    fig.update_layout(
        title=dict(text="<b>Confusion split by epoch</b>", x=0.5, xanchor="center",
                   y=0.98, yanchor="top", font=dict(size=15)),
        annotations=annotations,
        width=320 * n, height=470, margin=dict(t=110, b=60, l=20, r=20),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.06, yanchor="top",
                    font=dict(size=11)),
        paper_bgcolor="white",
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


def _pr_trajectory(by_criterion: list[dict]) -> tuple[list[float], list[float], list[str]]:
    """Cascade as a path in the P-R plane: (recall_cum, precision, gate_names).

    recall_cum uses a FIXED denominator (all good pairs at entry) so it is monotone
    decreasing gate-to-gate; precision is the ``precision`` column (live-set precision).
    Prepends the entry point (recall 1, raw-stream precision). Gates with no ``pass``
    (undefined precision) are skipped.
    """
    if not by_criterion:
        return [], [], []
    g0 = by_criterion[0]
    good_total = g0.get("TP", 0) + g0.get("FN", 0)
    if good_total == 0 or g0.get("eval", 0) == 0:
        return [], [], []
    xs, ys, names = [1.0], [good_total / g0["eval"]], ["eval·in"]
    for row in by_criterion:
        if (row.get("TP", 0) + row.get("FP", 0)) == 0:  # nothing passed -> precision undefined
            continue
        xs.append(row["TP"] / good_total)
        ys.append(row["precision"])
        names.append(row["criterion"])
    return xs, ys, names


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
