"""Focused comparison: task18-jump (Salto) vs task1718-jump-cooc (Salto + Grafo).

3 comparativas: J1-T0.1, J1-T0.05, J2-T0.1.
Auto-detects new experiments as they appear on disk (60s TTL cache + Refresh button).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ovo.utils.results_utils import load_experiments, load_scene_results

OUTPUT_DIR = ROOT / "data" / "output"
DATE = "20260422"

VARIANTS = ["task18-jump", "task1718-jump-cooc"]
VARIANT_LONG = {"task18-jump": "Salto", "task1718-jump-cooc": "Salto + Grafo"}
VARIANT_SHORT = {"task18-jump": "Salto", "task1718-jump-cooc": "Salto + Grafo"}

# The 3 explicit comparatives (J2-T0.05 not planned)
EXPECTED_CONFIGS: list[tuple[int, float]] = [
    (1, 0.1),
    (1, 0.05),
    (2, 0.1),
]
COMP_LABELS = {
    (1, 0.1):  "Comparativa 1 · J1 T0.10",
    (1, 0.05): "Comparativa 2 · J1 T0.05",
    (2, 0.1):  "Comparativa 3 · J2 T0.10",
}

SEMANTIC_METRICS = ["mIoU", "mAcc", "Head_mIoU", "Common_mIoU", "Tail_mIoU"]
AP_METRICS = ["AP_agnostic", "AP_agnostic_50", "AP_agnostic_25", "AP", "AP_50", "AP_25"]
FUSION_COLS = [
    "Fusion_Total", "Fusion_Accepted", "Fusion_Accept_Rate",
    "Fusion_Reject_Centroid", "Fusion_Reject_CosSim", "Fusion_Reject_Overlap",
]


def _config_label(jump: int | float, trans: float) -> str:
    t_str = f"{trans:.2f}".rstrip("0").rstrip(".")
    return f"J{int(jump)}-T{t_str}"


@st.cache_data(ttl=60)
def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_exp, df_class = load_experiments(OUTPUT_DIR, date_filter=DATE)
    df_scene, _ = load_scene_results(OUTPUT_DIR)
    if not df_scene.empty and "Date" in df_scene.columns:
        df_scene = df_scene[df_scene["Date"] == DATE]
    return df_exp, df_class, df_scene


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Variant"] = df["Label"].map(VARIANT_SHORT)
    df["Config"] = df.apply(
        lambda r: _config_label(r["Jump_Count"], r["Trans_Noise"]), axis=1
    )
    return df


# ── Page setup ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Salto vs Salto+Grafo", layout="wide")
st.title("Salto vs Salto + Grafo — barrido de parámetros de salto")
st.caption("Auto-refresca cada 60 s. Pulsa **Refresh** para forzar recarga.")

if st.button("Refresh"):
    st.cache_data.clear()
    st.rerun()

df_exp, df_class, df_scene = _load()

# Filter to this comparison's experiments
df = _enrich(df_exp[df_exp["Label"].isin(VARIANTS)]) if not df_exp.empty else pd.DataFrame()

# ── Status table — 3 comparativas ────────────────────────────────────────────
st.header("Estado de experimentos")

_METRIC_COLS = ["mIoU", "mAcc", "AP_agnostic", "AP_agnostic_50", "Fusion_Accept_Rate"]

def _status_row(df_all: pd.DataFrame, jc: int, tn: float, variant: str) -> dict:
    match = df_all[
        (df_all["Jump_Count"] == jc) & (df_all["Trans_Noise"] == tn) & (df_all["Label"] == variant)
    ]
    done = not match.empty
    r = match.iloc[0] if done else None
    return {
        "Estado": "✓ Listo" if done else "⏳ Corriendo",
        "mIoU":   f"{r['mIoU']:.4f}"            if done else "—",
        "mAcc":   f"{r['mAcc']:.4f}"            if done else "—",
        "AP_agn": f"{r['AP_agnostic']:.4f}"     if done and pd.notna(r["AP_agnostic"]) else "—",
        "AP_agn50": f"{r['AP_agnostic_50']:.4f}" if done and pd.notna(r.get("AP_agnostic_50")) else "—",
        "F_accept": f"{r['Fusion_Accept_Rate']:.4f}" if done and pd.notna(r["Fusion_Accept_Rate"]) else "—",
    }

for i, (jc, tn) in enumerate(EXPECTED_CONFIGS, 1):
    comp_title = COMP_LABELS[(jc, tn)]
    st.subheader(comp_title)
    rows = []
    for variant in VARIANTS:
        row = {"Variante": VARIANT_LONG[variant], "Saltos": int(jc), "T": tn}
        row.update(_status_row(df, jc, tn, variant))
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Only keep experiments that belong to the 3 planned comparatives (exclude J2-T0.05)
_mask_planned = df.apply(
    lambda r: (int(r["Jump_Count"]), r["Trans_Noise"]) in EXPECTED_CONFIGS, axis=1
)
df_done = df[_mask_planned].copy() if not df.empty else pd.DataFrame()

if df_done.empty:
    st.info("No hay experimentos completos todavía.")
    st.stop()

# ── Delta tables per comparativa ──────────────────────────────────────────────
st.header("Deltas: Salto+Grafo − Salto")

_DELTA_METRICS = [
    ("mIoU",           "mIoU",           True),
    ("mAcc",           "mAcc",           True),
    ("Head_mIoU",      "Head_mIoU",      True),
    ("Common_mIoU",    "Common_mIoU",    True),
    ("Tail_mIoU",      "Tail_mIoU",      True),
    ("AP_agnostic",    "AP_agnostic",    True),
    ("AP_agnostic_50", "AP_agnostic_50", True),
    ("AP_agnostic_25", "AP_agnostic_25", True),
    ("AP",             "AP",             True),
    ("AP_50",          "AP_50",          True),
    ("AP_25",          "AP_25",          True),
    ("Fusion_Accept_Rate", "F_Accept_Rate", True),
    ("Fusion_Reject_Centroid", "F_Rej_Centroid", False),
    ("Fusion_Reject_CosSim",   "F_Rej_CosSim",   False),
    ("Fusion_Reject_Overlap",  "F_Rej_Overlap",  False),
]

def _fmt(val: float, is_int: bool = False) -> str:
    if pd.isna(val):
        return "—"
    return f"{int(val)}" if is_int else f"{val:.4f}"

def _delta_color(delta: float, higher_is_better: bool | None) -> str:
    if pd.isna(delta) or higher_is_better is None:
        return ""
    if delta > 0:
        return "color:green;font-weight:bold" if higher_is_better else "color:red;font-weight:bold"
    if delta < 0:
        return "color:red;font-weight:bold" if higher_is_better else "color:green;font-weight:bold"
    return ""

delta_cols = st.columns(len(EXPECTED_CONFIGS))
for col_widget, (jc, tn) in zip(delta_cols, EXPECTED_CONFIGS):
    with col_widget:
        st.markdown(f"**{COMP_LABELS[(jc, tn)]}**")
        base_row = df_done[(df_done["Jump_Count"] == jc) & (df_done["Trans_Noise"] == tn) & (df_done["Label"] == "task18-jump")]
        cooc_row = df_done[(df_done["Jump_Count"] == jc) & (df_done["Trans_Noise"] == tn) & (df_done["Label"] == "task1718-jump-cooc")]

        if base_row.empty and cooc_row.empty:
            st.caption("Ninguno completado")
            continue

        table_rows = []
        for col_key, col_label, higher_better in _DELTA_METRICS:
            if col_key not in df_done.columns:
                continue
            is_int = col_key in ("Num_Instances", "Fusion_Reject_Centroid",
                                  "Fusion_Reject_CosSim", "Fusion_Reject_Overlap",
                                  "Fusion_Total", "Fusion_Accepted")
            b = base_row.iloc[0][col_key] if not base_row.empty else float("nan")
            c = cooc_row.iloc[0][col_key] if not cooc_row.empty else float("nan")
            delta = c - b if (pd.notna(b) and pd.notna(c)) else float("nan")
            delta_str = ("—" if pd.isna(delta)
                         else (f"{int(delta):+d}" if is_int else f"{delta:+.4f}"))
            color = _delta_color(delta, higher_better)
            table_rows.append({
                "Métrica":    col_label,
                "Salto":      _fmt(b, is_int),
                "Salto+Grafo": _fmt(c, is_int),
                "Δ":          delta_str,
                "_color":     color,
            })

        df_tbl = pd.DataFrame(table_rows)
        # Render with inline HTML for color
        html_rows = ""
        for _, r in df_tbl.iterrows():
            html_rows += (
                f"<tr><td>{r['Métrica']}</td>"
                f"<td>{r['Salto']}</td>"
                f"<td>{r['Salto+Grafo']}</td>"
                f"<td style='{r['_color']}'>{r['Δ']}</td></tr>"
            )
        st.markdown(
            f"""<table style="font-size:12px;width:100%">
            <thead><tr><th>Métrica</th><th>Salto</th><th>Salto+Grafo</th><th>Δ</th></tr></thead>
            <tbody>{html_rows}</tbody></table>""",
            unsafe_allow_html=True,
        )

# ── Semantic metrics — radar ──────────────────────────────────────────────────
st.header("Métricas semánticas")

_RADAR_METRICS = ["mIoU", "mAcc", "Head_mIoU", "Common_mIoU", "Tail_mIoU"]
_RADAR_LABELS  = ["mIoU", "mAcc", "Head", "Common", "Tail"]

avail_radar = [m for m in _RADAR_METRICS if m in df_done.columns and df_done[m].notna().any()]

if avail_radar:
    _labels_used = [_RADAR_LABELS[_RADAR_METRICS.index(m)] for m in avail_radar]
    n = len(avail_radar)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    # One radar per comparativa, side by side
    radar_cols = st.columns(len(EXPECTED_CONFIGS))

    # Consistent colors: Salto=blue, Salto+Grafo=orange; line style by config already via separate radars
    _VAR_COLORS = {"Salto": "#3a7ebf", "Salto + Grafo": "#e07030"}
    _VAR_FILL   = {"Salto": 0.10,      "Salto + Grafo": 0.15}

    for radar_col, (jc, tn) in zip(radar_cols, EXPECTED_CONFIGS):
        with radar_col:
            st.markdown(f"**{COMP_LABELS[(jc, tn)]}**")
            subset = df_done[(df_done["Jump_Count"] == jc) & (df_done["Trans_Noise"] == tn)]

            fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw={"polar": True})
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles)
            ax.set_xticklabels(_labels_used, size=9)
            ax.set_ylim(0, 0.6)
            ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            ax.set_yticklabels(["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"], size=6, color="grey")
            ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

            for _, row in subset.iterrows():
                variant = row["Variant"]
                vals = [float(row[m]) if pd.notna(row.get(m)) else 0.0 for m in avail_radar]
                vals_closed = vals + vals[:1]
                color = _VAR_COLORS.get(variant, "#888888")
                ax.plot(angles_closed, vals_closed, color=color, linewidth=1.2, label=variant)
                ax.fill(angles_closed, vals_closed, color=color, alpha=_VAR_FILL.get(variant, 0.1))

            ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Value table below radar
            tbl_rows = {"Métrica": _labels_used}
            for _, row in subset.iterrows():
                variant = row["Variant"]
                color_hex = _VAR_COLORS.get(variant, "#888")
                vals = [float(row[m]) if pd.notna(row.get(m)) else float("nan") for m in avail_radar]
                tbl_rows[variant] = [f"{v:.3f}" if not np.isnan(v) else "—" for v in vals]

            df_tbl = pd.DataFrame(tbl_rows)
            # Render colored headers
            header_html = "<tr><th>Métrica</th>"
            for variant in subset["Variant"].tolist():
                c = _VAR_COLORS.get(variant, "#888")
                header_html += f"<th style='color:{c}'>{variant}</th>"
            header_html += "</tr>"
            body_html = ""
            for _, r in df_tbl.iterrows():
                body_html += f"<tr><td><b>{r['Métrica']}</b></td>"
                for variant in subset["Variant"].tolist():
                    body_html += f"<td style='text-align:center'>{r[variant]}</td>"
                body_html += "</tr>"
            st.markdown(
                f"<table style='font-size:11px;width:100%;margin-top:4px'>"
                f"<thead>{header_html}</thead><tbody>{body_html}</tbody></table>",
                unsafe_allow_html=True,
            )

# ── Instance AP ───────────────────────────────────────────────────────────────
st.header("Instance AP (segmentación de instancias)")
st.caption(
    "**AP**: instancia correcta **y** clase correcta. "
    "**AP_agnostic**: solo calidad de máscara, ignora clase predicha. "
    "Mayor = mejor en ambos casos."
)

avail_ap = [m for m in AP_METRICS if m in df_done.columns and df_done[m].notna().any()]
if avail_ap:
    cols = st.columns(min(3, len(avail_ap)))
    for i, metric in enumerate(avail_ap):
        with cols[i % len(cols)]:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(data=df_done, x="Config", y=metric, hue="Variant",
                        hue_order=["Salto", "Salto + Grafo"],
                        palette=["#3a7ebf", "#e07030"], ax=ax)
            ymax = df_done[metric].dropna().max()
            ax.set_ylim(0, max(ymax * 1.25, 0.01))
            for bar in ax.patches:
                h = bar.get_height()
                if h > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, h + ymax * 0.02,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold",
                    )
            ax.set_title(metric, fontsize=11)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=30)
            ax.legend(title="", fontsize=8, loc="upper right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ── Fusion decisions ──────────────────────────────────────────────────────────
st.header("Decisiones de fusión")

avail_fusion = [c for c in FUSION_COLS if c in df_done.columns]
if avail_fusion:
    # ── Summary table — paired comparison per config ─────────────────────────
    st.subheader("Resumen por experimento (suma todas las escenas)")
    st.caption("Una fila por comparativa · columnas Salto (S) vs Salto+Grafo (S+G)")

    def _fusion_vals(row) -> dict:
        total  = int(row.get("Fusion_Total", 0))
        accept = int(row.get("Fusion_Accepted", 0))
        rate   = row.get("Fusion_Accept_Rate", float("nan"))
        rej    = total - accept
        pct = lambda n: f"{100*n/rej:.1f}%" if rej > 0 else "—"
        return {
            "total":  total,
            "accept": accept,
            "rate":   rate,
            "rej":    rej,
            "cen":    int(row.get("Fusion_Reject_Centroid", 0)),
            "cos":    int(row.get("Fusion_Reject_CosSim",   0)),
            "ovl":    int(row.get("Fusion_Reject_Overlap",  0)),
            "pct":    pct,
        }

    summary_rows = []
    for jc, tn in EXPECTED_CONFIGS:
        base = df_done[(df_done["Jump_Count"] == jc) & (df_done["Trans_Noise"] == tn) & (df_done["Label"] == "task18-jump")]
        cooc = df_done[(df_done["Jump_Count"] == jc) & (df_done["Trans_Noise"] == tn) & (df_done["Label"] == "task1718-jump-cooc")]
        b = _fusion_vals(base.iloc[0]) if not base.empty else None
        c = _fusion_vals(cooc.iloc[0]) if not cooc.empty else None

        def _f(d, key): return f"{d[key]:,}" if d else "—"
        def _pct(d, key): return d["pct"](d[key]) if d else "—"
        def _rate(d): return f"{100*d['rate']:.2f}%" if d and pd.notna(d["rate"]) else "—"
        def _delta_rate():
            if b and c and pd.notna(b["rate"]) and pd.notna(c["rate"]):
                diff = (c["rate"] - b["rate"]) * 100
                return f"{diff:+.2f}pp"
            return "—"

        summary_rows.append({
            "Comparativa":       COMP_LABELS[(jc, tn)],
            "S Evaluados":       _f(b, "total"),
            "S+G Evaluados":     _f(c, "total"),
            "S Fusionados":      _f(b, "accept"),
            "S+G Fusionados":    _f(c, "accept"),
            "S Rechazados":      _f(b, "rej"),
            "S+G Rechazados":    _f(c, "rej"),
            "S Tasa%":           _rate(b),
            "S+G Tasa%":         _rate(c),
            "S Centroide%":      _pct(b, "cen"),
            "S+G Centroide%":    _pct(c, "cen"),
            "S CosSim%":         _pct(b, "cos"),
            "S+G CosSim%":       _pct(c, "cos"),
            "S Overlap%":        _pct(b, "ovl"),
            "S+G Overlap%":      _pct(c, "ovl"),
        })

    df_sum = pd.DataFrame(summary_rows)

    s_cols       = [c for c in df_sum.columns if c.startswith("S ")]
    sg_cols      = [c for c in df_sum.columns if c.startswith("S+G ")]
    fus_cols     = [c for c in df_sum.columns if "Fusionados" in c]
    rej_cols     = [c for c in df_sum.columns if "Rechazados" in c]

    styled_sum = (
        df_sum.style
        .set_properties(subset=s_cols,   **{"background-color": "#dbeeff", "color": "#1a4a7a"})
        .set_properties(subset=sg_cols,  **{"background-color": "#fde8d0", "color": "#7a3010"})
        .set_properties(subset=fus_cols, **{"font-weight": "bold", "color": "#1a7a1a"})
        .set_properties(subset=rej_cols, **{"font-weight": "bold", "color": "#c0392b"})
    )
    st.dataframe(styled_sum, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Tasa de aceptación")
        if "Fusion_Accept_Rate" in df_done.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=df_done, x="Config", y="Fusion_Accept_Rate", hue="Variant",
                        hue_order=["Salto", "Salto + Grafo"],
                        palette=["#3a7ebf", "#e07030"], ax=ax)
            ymax = df_done["Fusion_Accept_Rate"].dropna().max()
            ax.set_ylim(0, max(ymax * 1.3, 0.01))
            for bar in ax.patches:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h * 1.02,
                            f"{100*h:.2f}%", ha="center", va="bottom", fontsize=8)
            ax.set_ylabel("Tasa aceptación")
            ax.set_xlabel("")
            ax.set_title("Tasa de aceptación de fusión")
            ax.tick_params(axis="x", rotation=30)
            ax.legend(title="Variante", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col_b:
        st.subheader("Razones de rechazo (% sobre rechazadas)")
        reason_cols = ["Fusion_Reject_Centroid", "Fusion_Reject_CosSim", "Fusion_Reject_Overlap"]
        avail_r = [c for c in reason_cols if c in df_done.columns]
        if avail_r:
            df_r = df_done[["Config", "Variant", "Fusion_Total", "Fusion_Accepted"] + avail_r].copy()
            df_r["Rejected"] = df_r["Fusion_Total"] - df_r["Fusion_Accepted"]
            for c in avail_r:
                df_r[c] = df_r[c] / df_r["Rejected"].replace(0, float("nan")) * 100
            df_r = df_r.set_index(["Config", "Variant"])[avail_r]
            df_r.columns = [c.replace("Fusion_Reject_", "") for c in df_r.columns]
            fig, ax = plt.subplots(figsize=(7, 4))
            df_r.plot(kind="bar", stacked=True, ax=ax,
                      color=["#e07070", "#70a0e0", "#70c070"])
            ax.set_ylim(0, 110)
            ax.set_title("% razón de rechazo (sobre total rechazadas)")
            ax.set_xlabel("Config / Variante")
            ax.tick_params(axis="x", rotation=45)
            ax.legend(title="Razón", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
            for bar in ax.patches:
                h = bar.get_height()
                if h > 5:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_y() + h/2, f"{h:.0f}%",
                            ha="center", va="center", fontsize=7, color="white", fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ── Per-scene breakdown ───────────────────────────────────────────────────────
st.header("Desglose por escena")

if not df_scene.empty:
    df_s = _enrich(df_scene[df_scene["Label"].isin(VARIANTS)])
    df_s = df_s[df_s.apply(
        lambda r: (int(r["Jump_Count"]), r["Trans_Noise"]) in EXPECTED_CONFIGS, axis=1
    )]

    if not df_s.empty:
        # (col, label, normalize_to_data_max, fixed_max or None)
        _SCENE_RADAR_METRICS = [
            ("mIoU",          "mIoU",        False, 0.6),
            ("mAcc",          "mAcc",        False, 0.7),
            ("AP_agnostic",   "AP_agnostic", False, 0.4),
            ("Num_Instances", "N_Instancias", True,  None),
        ]
        avail_scene_metrics = [
            (col, lbl, norm, fmax) for col, lbl, norm, fmax in _SCENE_RADAR_METRICS
            if col in df_s.columns and df_s[col].notna().any()
        ]

        scenes_ordered = sorted(df_s["Scene"].unique())
        n_scenes = len(scenes_ordered)
        s_angles = np.linspace(0, 2 * np.pi, n_scenes, endpoint=False).tolist()
        s_angles_closed = s_angles + s_angles[:1]

        _VAR_COLORS = {"Salto": "#3a7ebf", "Salto + Grafo": "#e07030"}

        # Layout: one row per metric, 3 columns per row (one per comparativa)
        for col_key, col_label, normalize, fixed_max in avail_scene_metrics:
            st.subheader(col_label)
            if fixed_max is not None:
                global_max = fixed_max
            else:
                global_max = df_s[col_key].dropna().max() if normalize else 1.0
            if global_max == 0:
                global_max = 1.0

            scene_cols = st.columns(len(EXPECTED_CONFIGS))
            for scene_col, (jc, tn) in zip(scene_cols, EXPECTED_CONFIGS):
                with scene_col:
                    st.caption(COMP_LABELS[(jc, tn)])
                    subset = df_s[(df_s["Jump_Count"] == jc) & (df_s["Trans_Noise"] == tn)]

                    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"polar": True})
                    ax.set_theta_offset(np.pi / 2)
                    ax.set_theta_direction(-1)
                    ax.set_xticks(s_angles)
                    ax.set_xticklabels(scenes_ordered, size=7)
                    ax.set_ylim(0, global_max)
                    if normalize:
                        _ticks = [global_max * f for f in [0.25, 0.5, 0.75, 1.0]]
                        _tick_labels = [f"{int(t)}" for t in _ticks]
                    else:
                        _ticks = [v for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] if v <= global_max]
                        _tick_labels = [str(v) for v in _ticks]
                    ax.set_yticks(_ticks)
                    ax.set_yticklabels(_tick_labels, size=5, color="grey")
                    ax.grid(color="grey", linestyle="--", linewidth=0.4, alpha=0.5)

                    for variant in VARIANT_SHORT.values():
                        var_data = subset[subset["Variant"] == variant]
                        if var_data.empty:
                            continue
                        color = _VAR_COLORS.get(variant, "#888")
                        vals = []
                        for scene in scenes_ordered:
                            scene_row = var_data[var_data["Scene"] == scene]
                            v = float(scene_row.iloc[0][col_key]) if not scene_row.empty and pd.notna(scene_row.iloc[0][col_key]) else 0.0
                            vals.append(v)
                        vals_closed = vals + vals[:1]
                        ax.plot(s_angles_closed, vals_closed, color=color, linewidth=1.2, label=variant)
                        ax.fill(s_angles_closed, vals_closed, color=color, alpha=0.12)

                    ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.15), fontsize=7)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

        # ── Tablas comparativas (una por comparativa, full-width) ────────────
        st.subheader("Tabla comparativa por escena")
        for jc, tn in EXPECTED_CONFIGS:
            st.markdown(f"**{COMP_LABELS[(jc, tn)]}**")
            subset = df_s[(df_s["Jump_Count"] == jc) & (df_s["Trans_Noise"] == tn)]
            tbl_records = []
            for scene in scenes_ordered:
                row_base = subset[(subset["Scene"] == scene) & (subset["Label"] == "task18-jump")]
                row_cooc = subset[(subset["Scene"] == scene) & (subset["Label"] == "task1718-jump-cooc")]
                rec = {"Escena": scene}
                for col_key, col_label, _, _fmax in avail_scene_metrics:
                    is_int = col_key == "Num_Instances"
                    b = float(row_base.iloc[0][col_key]) if not row_base.empty and pd.notna(row_base.iloc[0].get(col_key)) else float("nan")
                    c = float(row_cooc.iloc[0][col_key]) if not row_cooc.empty and pd.notna(row_cooc.iloc[0].get(col_key)) else float("nan")
                    delta = c - b if (pd.notna(b) and pd.notna(c)) else float("nan")
                    fmt = (lambda x: f"{int(x)}" if not np.isnan(x) else "—") if is_int else (lambda x: f"{x:.4f}" if not np.isnan(x) else "—")
                    rec[f"{col_label} S"]   = fmt(b)
                    rec[f"{col_label} S+G"] = fmt(c)
                    rec[f"Δ {col_label}"]   = ("—" if np.isnan(delta) else (f"{int(delta):+d}" if is_int else f"{delta:+.4f}"))
                tbl_records.append(rec)
            df_tbl = pd.DataFrame(tbl_records)
            delta_cols_names = [c for c in df_tbl.columns if c.startswith("Δ")]

            def _color_delta(val: str) -> str:
                if isinstance(val, str) and val not in ("—", ""):
                    try:
                        num = float(val.replace("+", ""))
                        if num > 0:
                            return "color: green; font-weight: bold"
                        if num < 0:
                            return "color: red; font-weight: bold"
                    except ValueError:
                        pass
                return ""

            styled = df_tbl.style.applymap(_color_delta, subset=delta_cols_names)
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.divider()

# ── Fusion por escena ─────────────────────────────────────────────────────────
st.header("Fusión por escena")

if not df_scene.empty:
    df_fs = _enrich(df_scene[df_scene["Label"].isin(VARIANTS)])
    df_fs = df_fs[df_fs.apply(
        lambda r: (int(r["Jump_Count"]), r["Trans_Noise"]) in EXPECTED_CONFIGS, axis=1
    )]
    if not df_fs.empty:
        scenes_fs = sorted(df_fs["Scene"].unique())
        for jc, tn in EXPECTED_CONFIGS:
            st.markdown(f"**{COMP_LABELS[(jc, tn)]}**")
            subset = df_fs[(df_fs["Jump_Count"] == jc) & (df_fs["Trans_Noise"] == tn)]
            rows = []
            for scene in scenes_fs:
                b = subset[(subset["Scene"] == scene) & (subset["Label"] == "task18-jump")]
                c = subset[(subset["Scene"] == scene) & (subset["Label"] == "task1718-jump-cooc")]
                def _fv(df, col): return df.iloc[0][col] if not df.empty and col in df.columns and pd.notna(df.iloc[0][col]) else float("nan")
                def _fi(df, col): return int(_fv(df, col)) if not pd.isna(_fv(df, col)) else None
                def _pct(n, rej): return f"{100*n/rej:.1f}%" if rej and rej > 0 else "—"

                b_tot  = _fi(b, "Fusion_Total") or 0
                c_tot  = _fi(c, "Fusion_Total") or 0
                b_acc  = _fi(b, "Fusion_Accepted") or 0
                c_acc  = _fi(c, "Fusion_Accepted") or 0
                b_rej  = b_tot - b_acc
                c_rej  = c_tot - c_acc
                b_rate = _fv(b, "Fusion_Accept_Rate")
                c_rate = _fv(c, "Fusion_Accept_Rate")
                b_cen  = _fi(b, "Fusion_Reject_Centroid") or 0
                c_cen  = _fi(c, "Fusion_Reject_Centroid") or 0
                b_cos  = _fi(b, "Fusion_Reject_CosSim")   or 0
                c_cos  = _fi(c, "Fusion_Reject_CosSim")   or 0
                b_ovl  = _fi(b, "Fusion_Reject_Overlap")  or 0
                c_ovl  = _fi(c, "Fusion_Reject_Overlap")  or 0

                rows.append({
                    "Escena":            scene,
                    "S Evaluados":       f"{b_tot:,}"  if b_tot else "—",
                    "S+G Evaluados":     f"{c_tot:,}"  if c_tot else "—",
                    "S Fusionados":      f"{b_acc:,}"  if b_acc else "—",
                    "S+G Fusionados":    f"{c_acc:,}"  if c_acc else "—",
                    "S Rechazados":      f"{b_rej:,}"  if b_rej else "—",
                    "S+G Rechazados":    f"{c_rej:,}"  if c_rej else "—",
                    "S Tasa%":           f"{100*b_rate:.2f}%" if pd.notna(b_rate) else "—",
                    "S+G Tasa%":         f"{100*c_rate:.2f}%" if pd.notna(c_rate) else "—",
                    "S Centroide%":      _pct(b_cen, b_rej),
                    "S+G Centroide%":    _pct(c_cen, c_rej),
                    "S CosSim%":         _pct(b_cos, b_rej),
                    "S+G CosSim%":       _pct(c_cos, c_rej),
                    "S Overlap%":        _pct(b_ovl, b_rej),
                    "S+G Overlap%":      _pct(c_ovl, c_rej),
                })

            df_tbl = pd.DataFrame(rows)
            s_cols   = [col for col in df_tbl.columns if col.startswith("S ")]
            sg_cols  = [col for col in df_tbl.columns if col.startswith("S+G ")]
            fus_cols = [col for col in df_tbl.columns if "Fusionados" in col]
            rej_cols = [col for col in df_tbl.columns if "Rechazados" in col]
            styled = (
                df_tbl.style
                .set_properties(subset=s_cols,   **{"background-color": "#dbeeff", "color": "#1a4a7a"})
                .set_properties(subset=sg_cols,  **{"background-color": "#fde8d0", "color": "#7a3010"})
                .set_properties(subset=fus_cols, **{"font-weight": "bold", "color": "#1a7a1a"})
                .set_properties(subset=rej_cols, **{"font-weight": "bold", "color": "#c0392b"})
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.divider()

# ── Per-class IoU heatmap ─────────────────────────────────────────────────────
st.header("Heatmap IoU por clase")

if not df_class.empty:
    df_cl = _enrich(df_class[df_class["Label"].isin(VARIANTS) & (df_class["Date"] == DATE)])
    df_cl = df_cl[df_cl.apply(
        lambda r: (int(r["Jump_Count"]), r["Trans_Noise"]) in EXPECTED_CONFIGS, axis=1
    )]
    if not df_cl.empty:
        df_cl["ExpLabel"] = df_cl["Variant"] + "\n" + df_cl["Config"]

        # Build full pivot to get consistent non-zero class list
        pivot_all = df_cl.pivot_table(index="Class", columns="ExpLabel", values="IoU", aggfunc="mean")
        active_classes = pivot_all.index[pivot_all.max(axis=1).fillna(0) > 0]
        n_rows = len(active_classes)

        with st.expander(f"Ver heatmaps ({n_rows} clases con datos)", expanded=True):
            heat_cols = st.columns(len(EXPECTED_CONFIGS))
            for heat_col, (jc, tn) in zip(heat_cols, EXPECTED_CONFIGS):
                with heat_col:
                    st.caption(COMP_LABELS[(jc, tn)])
                    mask = (df_cl["Jump_Count"] == jc) & (df_cl["Trans_Noise"] == tn)
                    pivot = df_cl[mask].pivot_table(
                        index="Class", columns="ExpLabel", values="IoU", aggfunc="mean"
                    )
                    pivot = pivot.loc[pivot.index.isin(active_classes)]
                    if pivot.empty:
                        st.caption("Sin datos")
                        continue
                    fig, ax = plt.subplots(figsize=(3.5, max(2.5, n_rows * 0.22)))
                    sns.heatmap(
                        pivot, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                        linewidths=0.3, linecolor="grey",
                        annot=True, fmt=".2f", annot_kws={"size": 7},
                        cbar=False,
                    )
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    ax.tick_params(axis="x", labelsize=8, rotation=30)
                    ax.tick_params(axis="y", labelsize=7)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=False)
                    plt.close(fig)

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("Datos brutos"):
    display_cols = [
        "Config", "Variant", "mIoU", "mAcc",
        "Head_mIoU", "Common_mIoU", "Tail_mIoU",
        "AP_agnostic", "AP_agnostic_50", "AP_agnostic_25",
        "AP", "AP_50", "AP_25",
        "Fusion_Total", "Fusion_Accepted", "Fusion_Accept_Rate",
        "Fusion_Reject_Centroid", "Fusion_Reject_CosSim", "Fusion_Reject_Overlap",
        "Experiment_ID",
    ]
    avail = [c for c in display_cols if c in df_done.columns]
    st.dataframe(
        df_done[avail].sort_values(["Config", "Variant"]).reset_index(drop=True),
        use_container_width=True,
    )
