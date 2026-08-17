"""Generate a standalone HTML comparison report across experiments.

Usage:
  python -m studies.fusion_metrics.viz.report --exps <id1> <id2> ... [--out report.html]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import plotly.graph_objects as go
import plotly.express as px

OUTPUT_ROOT = pathlib.Path("data/output/Replica")
SCENES = ["office0", "office1", "office2", "office3", "office4",
          "room0", "room1", "room2"]
EPOCHS = ["predrift_predrift", "predrift_postdrift", "postdrift_postdrift"]
EPOCH_LABELS = ["Pre-drift / Pre-drift", "Pre-drift / Post-drift", "Post-drift / Post-drift"]
METRICS = ["TP", "FP", "FN", "TN"]


def _resolve(exp_id: str) -> pathlib.Path:
    p = pathlib.Path(exp_id)
    if p.is_dir():
        return p.resolve()
    c = OUTPUT_ROOT / exp_id
    if c.is_dir():
        return c
    matches = sorted(OUTPUT_ROOT.glob(f"{exp_id}*"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Ambiguous '{exp_id}' matches: {[m.name for m in matches]}", file=sys.stderr)
        sys.exit(1)
    raise FileNotFoundError(exp_id)


def load_experiments(exps: list[str], labels: list[str] | None = None) -> dict[str, dict]:
    data = {}
    for i, eid in enumerate(exps):
        ep = _resolve(eid)
        label = labels[i] if labels else ep.name
        scenes_data = {}
        for s in SCENES:
            f = ep / s / "fusion" / "fusion_LC" / "fusion_eval_summary.json"
            if f.exists():
                scenes_data[s] = json.loads(f.read_text())
        data[label] = {"path": ep, "scenes": scenes_data}
    return data


def _table_header(cols, widths=None):
    html = '<tr>'
    for i, c in enumerate(cols):
        w = f' style="width:{widths[i]}px"' if widths else ''
        html += f'<th{w}>{c}</th>'
    html += '</tr>'
    return html


def _table_row(cells, bold=False, highlight=None):
    tag = 'th' if bold else 'td'
    cls = f' class="{highlight}"' if highlight else ''
    return f'<tr{cls}>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>'


def build_html(data: dict[str, dict], baseline: str | None = None) -> str:
    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
             max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
      h1 { color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 8px; }
      h2 { color: #16213e; margin-top: 40px; }
      h3 { color: #0f3460; margin-top: 28px; }
      .summary { display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; }
      .card { background: white; border-radius: 10px; padding: 16px 24px;
               box-shadow: 0 2px 8px rgba(0,0,0,0.08); flex: 1; min-width: 160px; }
      .card .val { font-size: 28px; font-weight: 700; line-height: 1.2; }
      .card .lbl { font-size: 13px; color: #666; }
      .card.highlight { border-left: 4px solid #e94560; }
      table { border-collapse: collapse; width: 100%; margin: 12px 0 24px;
              background: white; border-radius: 8px; overflow: hidden;
              box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
      th { background: #1a1a2e; color: white; padding: 10px 14px;
           text-align: center; font-size: 13px; font-weight: 600; }
      td { padding: 8px 14px; text-align: center; border-bottom: 1px solid #eee;
           font-size: 13px; }
      tr:hover td { background: #f0f0f8; }
      .best { font-weight: 700; color: #e94560; }
      tr.section td { background: #f8f8fc; font-weight: 600; text-align: left;
                       padding: 6px 14px; }
      .epoch-tabs { margin: 12px 0; }
      .epoch-content { display: none; }
      .epoch-content.active { display: block; }
      .tab-btn { padding: 8px 16px; border: none; background: #ddd; cursor: pointer;
                  border-radius: 6px 6px 0 0; margin-right: 4px; font-size: 13px; }
      .tab-btn.active { background: #1a1a2e; color: white; }
    </style>
    <script>
    function switchEpoch(name) {
      document.querySelectorAll('.epoch-content').forEach(e => e.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.getElementById('epoch-' + name).classList.add('active');
      event.target.classList.add('active');
    }
    </script>
    """

    labels = list(data.keys())
    n = len(labels)

    # Aggregate totals
    totals = {lbl: {m: sum(d["scenes"][s]["verdicts"]["counts"][m]
                          for s in d["scenes"] if s in d["scenes"])
                     for m in METRICS}
              for lbl, d in data.items()}

    def pr(tp, fp): return tp / (tp + fp) * 100 if (tp + fp) else 0
    def re(tp, fn): return tp / (tp + fn) * 100 if (tp + fn) else 0
    def f1(p, r):   return 2 * p * r / (p + r) if (p + r) else 0

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <title>OVO Fusion — Comparison Report</title>{css}</head><body>
    <h1>OVO Fusion — Cross-Experiment Comparison</h1>
    <p>Experiments: {', '.join(labels)} · Scenes: {', '.join(SCENES)}</p>
    """

    # === Table 1: Total verdicts ===
    html += '<h2>1. Total Verdicts</h2><table>'
    html += _table_header(["Experiment"] + METRICS + ["Pairs", "Precision", "Recall", "F1"])
    best_f1 = max(f1(pr(totals[l]["TP"], totals[l]["FP"]), re(totals[l]["TP"], totals[l]["FN"])) for l in labels)
    for lbl in labels:
        t = totals[lbl]
        p = pr(t["TP"], t["FP"])
        r = re(t["TP"], t["FN"])
        f = f1(p, r)
        hl = ' class="best"' if abs(f - best_f1) < 0.001 else ''
        html += f'<tr{hl}><td style="text-align:left">{lbl}</td>'
        html += ''.join(f'<td>{t[m]}</td>' for m in METRICS)
        html += f'<td>{sum(t.values())}</td>'
        html += f'<td>{p:.1f}%</td><td>{r:.1f}%</td><td>{f:.1f}%</td></tr>'
    html += '</table>'

    # === Table 2: By Epoch (with global + criterion breakdown) ===
    html += '<h2>2. By Epoch</h2>'
    html += '<div class="epoch-tabs">'

    all_tabs = [("global", "Global")] + list(zip(EPOCHS, EPOCH_LABELS))
    for i, (key, label) in enumerate(all_tabs):
        cls = ' class="tab-btn active"' if i == 0 else ''
        html += f'<button{cls} onclick="switchEpoch(\'{key}\')">{label}</button>'
    html += '</div>'

    for i, (key, label) in enumerate(all_tabs):
        is_global = key == "global"
        cls = ' epoch-content active' if i == 0 else ' epoch-content'
        html += f'<div id="epoch-{key}" class="{cls}">'

        # Summary sub-table
        html += '<h3>Summary</h3><table>'
        html += _table_header(["Experiment"] + METRICS + ["Precision", "Recall", "F1"])
        for lbl in labels:
            if is_global:
                t = totals[lbl]
            else:
                t = {m: sum(data[lbl]["scenes"][s]["verdicts"]["by_epoch"][key]["counts"][m]
                            for s in data[lbl]["scenes"] if s in data[lbl]["scenes"])
                     for m in METRICS}
            p = pr(t["TP"], t["FP"])
            r = re(t["TP"], t["FN"])
            f = f1(p, r)
            html += f'<tr><td style="text-align:left">{lbl}</td>'
            html += ''.join(f'<td>{t[m]}</td>' for m in METRICS)
            html += f'<td>{p:.1f}%</td><td>{r:.1f}%</td><td>{f:.1f}%</td></tr>'
        html += '</table>'

        # Criterion breakdown per experiment
        html += '<h3>Impact by Criterion</h3>'
        for lbl in labels:
            html += f'<h4>{lbl}</h4><table>'
            html += _table_header(["Criterion", "Total", "TP", "FP", "FN", "TN",
                                   "TP%", "FP%", "FN%", "TN%"])
            bg = {}
            for s in data[lbl]["scenes"]:
                verdicts = data[lbl]["scenes"][s]["verdicts"]
                cascade = (verdicts.get("by_criterion", []) if is_global
                           else verdicts["by_epoch"][key].get("by_criterion", []))
                for g in cascade:
                    agg = bg.setdefault(g["criterion"], {"eval": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0})
                    for k in ("eval", "TP", "TN", "FP", "FN"):
                        agg[k] += g.get(k, 0)
            for grp in sorted(bg):
                d = bg[grp]
                total = d["eval"] or 1  # eval = pairs entering the gate
                html += f'<tr><td style="text-align:left">{grp}</td><td>{d["eval"]}</td>'
                html += ''.join(f'<td>{d.get(m, 0)}</td>' for m in METRICS)
                for m in METRICS:
                    html += f'<td>{d.get(m, 0) / total * 100:.1f}%</td>'
                html += '</tr>'
            html += '</table>'

        # Chart for this tab
        fig_ep = go.Figure()
        for lbl in labels:
            if is_global:
                t = totals[lbl]
            else:
                t = {m: sum(data[lbl]["scenes"][s]["verdicts"]["by_epoch"][key]["counts"][m]
                            for s in data[lbl]["scenes"] if s in data[lbl]["scenes"])
                     for m in METRICS}
            fig_ep.add_trace(go.Bar(name=lbl, x=METRICS,
                                    y=[t[m] for m in METRICS],
                                    text=[str(t[m]) for m in METRICS],
                                    textposition="auto"))
        fig_ep.update_layout(barmode="group", title=f"Verdicts – {label}",
                             yaxis_type="log", height=350,
                             margin=dict(l=40, r=20, t=40, b=40))
        html += fig_ep.to_html(full_html=False, include_plotlyjs="cdn")

        # F1 chart
        fig_ep_f1 = go.Figure()
        for metric, color, symbol in [("Precision", "#4e79a7", "circle"),
                                       ("Recall", "#f28e2b", "square"),
                                       ("F1", "#e15759", "diamond")]:
            vals = []
            for lbl in labels:
                if is_global:
                    t = totals[lbl]
                else:
                    t = {m: sum(data[lbl]["scenes"][s]["verdicts"]["by_epoch"][key]["counts"][m]
                                for s in data[lbl]["scenes"] if s in data[lbl]["scenes"])
                         for m in METRICS}
                p = pr(t["TP"], t["FP"])
                r = re(t["TP"], t["FN"])
                v = {"Precision": p, "Recall": r, "F1": f1(p, r)}[metric]
                vals.append(round(v, 1))
            fig_ep_f1.add_trace(go.Scatter(name=metric, x=labels, y=vals,
                                           mode="lines+markers",
                                           marker=dict(symbol=symbol, size=10),
                                           line=dict(width=2.5)))
        fig_ep_f1.update_layout(title=f"P/R/F1 – {label}", height=350,
                                yaxis=dict(title="%", ticksuffix="%"),
                                margin=dict(l=40, r=20, t=40, b=80))
        html += fig_ep_f1.to_html(full_html=False, include_plotlyjs=False)

        html += '</div>'

    # === Table 4: Per-scene mIoU (baseline-first with deltas) ===
    import csv, io

    def _scene_miou(lbl, s):
        f = data[lbl]["path"] / "replica" / f"statistics_{s}.txt"
        if not f.exists():
            return None
        reader = csv.DictReader(f.read_text().splitlines(), skipinitialspace=True)
        ious = [float(r["iou"]) for r in reader if r["iou"].strip() not in ("", "nan")]
        return sum(ious) / len(ious) if ious else None

    # Determine column order: baseline first, then the rest
    other_labels = [l for l in labels if l != baseline]
    col_order = ([baseline] + other_labels) if baseline else labels

    html += '<h2>3. Per-Scene mIoU</h2>'
    if baseline:
        html += f'<p>Baseline: <strong>{baseline}</strong> · Δ shown for others (green=better, red=worse)</p>'

    # Header row
    html += '<table>'
    cols = ["Scene"] + col_order
    html += _table_header(cols)

    for s in SCENES:
        baseline_miou = _scene_miou(baseline, s) if baseline else None
        html += f'<tr><td style="text-align:left;font-weight:600">{s}</td>'
        for lbl in col_order:
            m = _scene_miou(lbl, s)
            if m is None:
                html += '<td>—</td>'
            elif baseline and lbl != baseline and baseline_miou is not None:
                delta = m - baseline_miou
                color = "green" if delta > 0.001 else ("red" if delta < -0.001 else "gray")
                html += f'<td>{m:.4f} <span style="color:{color};font-size:0.85em">({delta:+.4f})</span></td>'
            else:
                html += f'<td style="font-weight:700">{m:.4f}</td>'
        html += '</tr>'
    html += '</table>'

    # === Table 4: Verdict Distribution by Epoch ===
    html += '<h2>4. Verdict Distribution by Epoch</h2>'
    html += '<p>100% stacked bars: how each verdict type is distributed across drift epochs for each experiment.</p>'

    epoch_colors = {"predrift_predrift": "#4e79a7",
                    "predrift_postdrift": "#f28e2b",
                    "postdrift_postdrift": "#e15759"}

    # Single chart: one group per verdict type, each bar = 100% stacked by epoch
    fig_ed = go.Figure()
    for ep, color in epoch_colors.items():
        for lbl in col_order:
            vals = []
            for metric in METRICS:
                t = {m: sum(data[lbl]["scenes"][s]["verdicts"]["by_epoch"][ep]["counts"][m]
                            for s in data[lbl]["scenes"] if s in data[lbl]["scenes"])
                     for m in METRICS}
                vals.append(t[metric])
            total = sum(vals)
            pcts = [v / total * 100 if total else 0 for v in vals]
            label = f"{ep.split('_')[0]}_{lbl}"
            fig_ed.add_trace(go.Bar(name=f"{lbl} – {EPOCH_LABELS[EPOCHS.index(ep)]}",
                                     x=[f"{lbl}_{m}" for m in METRICS],
                                     y=pcts,
                                     marker_color=color,
                                     hovertemplate=f"{lbl}<br>{{{{x}}}}<br>%{{{{y:.1f}}}}%<extra></extra>"))

    fig_ed.update_layout(barmode="relative", title="Epoch Share per Verdict Type (100% stacked)",
                         xaxis=dict(tickmode="array",
                                    tickvals=[f"{l}_{m}" for l in col_order for m in METRICS],
                                    ticktext=[f"{l}<br>{m}" for l in col_order for m in METRICS],
                                    tickangle=0),
                         yaxis=dict(title="%", ticksuffix="%", range=[0, 105]),
                         height=max(350, 250 * len(col_order) // 2),
                         legend=dict(orientation="h", y=-0.25),
                         margin=dict(l=40, r=20, t=40, b=120))
    html += fig_ed.to_html(full_html=False, include_plotlyjs="cdn")

    # Per-experiment stacked bars: each experiment = one bar-group of 4 metrics, 100% stacked by epoch
    html += '<h3>Per-Experiment View</h3>'
    for lbl in col_order:
        fig_exp = go.Figure()
        for ep, color in epoch_colors.items():
            t = {m: sum(data[lbl]["scenes"][s]["verdicts"]["by_epoch"][ep]["counts"][m]
                        for s in data[lbl]["scenes"] if s in data[lbl]["scenes"])
                 for m in METRICS}
            vals = [t[m] for m in METRICS]
            total = sum(vals)
            pcts = [v / total * 100 if total else 0 for v in vals]
            fig_exp.add_trace(go.Bar(name=EPOCH_LABELS[EPOCHS.index(ep)],
                                     x=METRICS, y=pcts,
                                     text=[f"{v} ({p:.0f}%)" for v, p in zip(vals, pcts)],
                                     textposition="inside",
                                     textfont_color="white",
                                     marker_color=color))
        fig_exp.update_layout(barmode="relative", title=f"{lbl} – Verdict Epoch Composition",
                              yaxis=dict(title="%", ticksuffix="%", range=[0, 105]),
                              height=350,
                              legend=dict(orientation="h", y=-0.25),
                              margin=dict(l=40, r=20, t=40, b=100))
        html += fig_exp.to_html(full_html=False, include_plotlyjs=False)

    # === Table 5: Class-agnostic AP (post-fusion, with deltas vs baseline) ===
    html += '<h2>5. Class-Agnostic AP</h2>'
    if baseline:
        html += f'<p>Baseline: <strong>{baseline}</strong> · Δ shown (green=better, red=worse)</p>'

    # Compute mean per experiment across scenes
    ap_fields = ["ap_mean", "ap50", "ap25"]
    delta_fields = ["delta_ap_mean", "delta_ap50", "delta_ap25"]
    ap_headers = ["AP mean", "AP@50", "AP@25"]
    sp_fields = ["matched50", "spurious50", "missed50"]
    sp_headers = ["Matched@50", "Spurious@50", "Missed@50"]

    def _mean_ap(lbl, field):
        vals = []
        for s in SCENES:
            d = data[lbl]["scenes"].get(s, {})
            if not d:
                continue
            v = d.get("agnostic_impact", {}).get("objects", {}).get("post", {}).get(field)
            if v is not None:
                vals.append(v)
        return sum(vals) / len(vals) if vals else None

    def _mean_delta(lbl, field):
        vals = []
        for s in SCENES:
            d = data[lbl]["scenes"].get(s, {})
            if not d:
                continue
            v = d.get("agnostic_impact", {}).get("objects", {}).get(field)
            if v is not None:
                vals.append(v)
        return sum(vals) / len(vals) if vals else None

    col_order = ([baseline] + other_labels) if baseline else labels

    # --- AP values table ---
    html += '<h3>Post-Fusion AP (objects mode)</h3><table>'
    html += _table_header(["Experiment"] + ap_headers)
    for lbl in col_order:
        html += f'<tr><td style="text-align:left;font-weight:{"bold" if lbl == baseline else "normal"}">{lbl}</td>'
        for field in ap_fields:
            v = _mean_ap(lbl, field)
            if v is None:
                html += '<td>—</td>'
            elif baseline and lbl != baseline:
                bv = _mean_ap(baseline, field)
                if bv is not None:
                    delta = v - bv
                    color = "green" if delta > 0.001 else ("red" if delta < -0.001 else "gray")
                    html += f'<td>{v:.3f} <span style="color:{color};font-size:0.85em">({delta:+.3f})</span></td>'
                else:
                    html += f'<td>{v:.3f}</td>'
            else:
                html += f'<td style="font-weight:700">{v:.3f}</td>'
        html += '</tr>'
    html += '</table>'

    # --- Delta table (how much fusion improved each experiment) ---
    html += '<h3>Fusion Impact (Δ post – pre, objects mode)</h3><table>'
    html += _table_header(["Experiment"] + ["Δ " + h for h in ap_headers])
    for lbl in col_order:
        html += f'<tr><td style="text-align:left">{lbl}</td>'
        for field in delta_fields:
            v = _mean_delta(lbl, field)
            if v is None:
                html += '<td>—</td>'
            else:
                color = "green" if v > 0.001 else ("red" if v < -0.001 else "gray")
                html += f'<td><span style="color:{color}">{v:+.3f}</span></td>'
        html += '</tr>'
    html += '</table>'

    def _per_threshold_50(lbl, field, mode="post"):
        """Read matched/spurious/missed at IoU=0.5 from per_threshold array."""
        vals = []
        for s in SCENES:
            d = data[lbl]["scenes"].get(s, {}).get("agnostic_impact", {}).get("objects", {}).get(mode, {})
            pt = d.get("per_threshold", [])
            entry = next((e for e in pt if e.get("iou") == 0.5), None)
            if entry is not None:
                v = entry.get(field)
                if v is not None:
                    vals.append(v)
        return sum(vals) / len(vals) if vals else None

    # --- Matched/Spurious/Missed table ---
    html += '<h3>Matched / Spurious / Missed @50 (post-fusion)</h3><table>'
    html += _table_header(["Experiment"] + sp_headers)
    for lbl in col_order:
        html += f'<tr><td style="text-align:left">{lbl}</td>'
        for field in sp_fields:
            mean_v = _per_threshold_50(lbl, field)
            if mean_v is None:
                html += '<td>—</td>'
            elif baseline and lbl != baseline:
                b_mean = _per_threshold_50(baseline, field)
                if b_mean is not None:
                    delta = mean_v - b_mean
                    if field == "spurious50":
                        color = "green" if delta < -0.5 else ("red" if delta > 0.5 else "gray")
                    else:
                        color = "green" if delta > 0.5 else ("red" if delta < -0.5 else "gray")
                    html += f'<td>{mean_v:.0f} <span style="color:{color};font-size:0.85em">({delta:+.0f})</span></td>'
                else:
                    html += f'<td>{mean_v:.0f}</td>'
            else:
                html += f'<td style="font-weight:700">{mean_v:.0f}</td>'
        html += '</tr>'
    html += '</table>'

    # === Plotly charts ===
    html += '<h2>6. Charts</h2>'

    # Chart 1: TP/FP/FN bar
    fig1 = go.Figure()
    for lbl in labels:
        t = totals[lbl]
        fig1.add_trace(go.Bar(name=lbl, x=METRICS,
                              y=[t[m] for m in METRICS],
                              text=[str(t[m]) for m in METRICS],
                              textposition="auto"))
    fig1.update_layout(barmode="group", title="Verdict Counts by Experiment",
                       yaxis_type="log", height=400)
    html += fig1.to_html(full_html=False, include_plotlyjs="cdn")

    # Chart 2: Precision/Recall/F1
    fig2 = go.Figure()
    for metric, color in [("Precision", "#4e79a7"), ("Recall", "#f28e2b"), ("F1", "#e15759")]:
        vals = []
        for lbl in labels:
            t = totals[lbl]
            p = pr(t["TP"], t["FP"])
            r = re(t["TP"], t["FN"])
            v = {"Precision": p, "Recall": r, "F1": f1(p, r)}[metric]
            vals.append(round(v, 1))
        fig2.add_trace(go.Bar(name=metric, x=labels, y=vals,
                              text=[f"{v}%" for v in vals],
                              textposition="auto", marker_color=color))
    fig2.update_layout(barmode="group", title="Precision / Recall / F1", height=400)
    html += fig2.to_html(full_html=False, include_plotlyjs=False)

    # Chart 3: Epoch F1
    fig3 = go.Figure()
    for lbl in labels:
        ep_f1s = []
        for ep in EPOCHS:
            t = {m: sum(data[lbl]["scenes"][s]["verdicts"]["by_epoch"][ep]["counts"][m]
                        for s in data[lbl]["scenes"] if s in data[lbl]["scenes"])
                 for m in METRICS}
            p = pr(t["TP"], t["FP"])
            r = re(t["TP"], t["FN"])
            ep_f1s.append(round(f1(p, r), 1))
        fig3.add_trace(go.Scatter(name=lbl, x=EPOCH_LABELS, y=ep_f1s,
                                  mode="lines+markers", line=dict(width=3)))
    fig3.update_layout(title="F1 by Epoch", height=400,
                       yaxis=dict(title="F1 (%)", ticksuffix="%"))
    html += fig3.to_html(full_html=False, include_plotlyjs=False)

    html += "</body></html>"
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exps", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Short display labels (same order as --exps)")
    parser.add_argument("--baseline", default=None,
                        help="Label to use as baseline (first column, show deltas)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    labels = args.labels or args.exps
    if len(labels) != len(args.exps):
        print("error: --labels count must match --exps count", file=sys.stderr)
        sys.exit(1)
    data = load_experiments(args.exps, labels)
    if args.baseline and args.baseline not in labels:
        print(f"error: baseline '{args.baseline}' not in labels {labels}", file=sys.stderr)
        sys.exit(1)
    html = build_html(data, baseline=args.baseline)
    out = args.out or (list(data.values())[0]["path"] / "_viz" / "comparison_report.html")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(html)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
