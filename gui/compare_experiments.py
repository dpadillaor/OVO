"""Multi-experiment comparator (up to 3 selected experiments).

Generalises gui/compare_cooc.py: instead of two fixed labels swept over three
fixed (Jump_Count, Trans_Noise) configs, the user picks 1-3 arbitrary
experiments from data/output/Replica/ and the page renders status, deltas,
radars, instance AP, fusion breakdown, per-scene radars/tables and IoU
heatmaps for that selection.
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
MAX_SELECT = 3

# Stable colour palette assigned by position in the selection list.
PALETTE = ["#3a7ebf", "#e07030", "#2ca02c"]
PALETTE_BG = ["#dbeeff", "#fde8d0", "#dcefdc"]
PALETTE_FG = ["#1a4a7a", "#7a3010", "#1f5a1f"]

SEMANTIC_METRICS = ["mIoU", "mAcc", "Head_mIoU", "Common_mIoU", "Tail_mIoU"]
AP_METRICS = ["AP", "AP_50", "AP_25", "AP_agnostic", "AP_agnostic_50", "AP_agnostic_25"]
FUSION_REJECT_COLS = [
    "Fusion_Reject_Centroid", "Fusion_Reject_AABB",
    "Fusion_Reject_CosSim", "Fusion_Reject_Overlap",
    "Fusion_Reject_Cooccurrence",
]


@st.cache_data(ttl=60)
def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_exp, df_class = load_experiments(OUTPUT_DIR, dataset_filter="Replica")
    df_scene, _ = load_scene_results(OUTPUT_DIR)
    if not df_scene.empty:
        df_scene = df_scene[df_scene["Dataset"] == "Replica"]
    return df_exp, df_class, df_scene


def _default_selection(exp_ids: list[str]) -> list[str]:
    """Pick up to MAX_SELECT defaults: a 'baseline'-tagged ID first, then most recent."""
    if not exp_ids:
        return []
    picks: list[str] = []
    baseline = next((e for e in exp_ids if e.endswith("_baseline")), None)
    if baseline:
        picks.append(baseline)
    sorted_recent = sorted(exp_ids, reverse=True)
    for eid in sorted_recent:
        if len(picks) >= MAX_SELECT:
            break
        if eid not in picks:
            picks.append(eid)
    return picks[:MAX_SELECT]


def _fmt(val, is_int: bool = False) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{int(val):,}" if is_int else f"{val:.4f}"


def _fmt_pct(num, denom) -> str:
    if denom is None or denom == 0 or pd.isna(denom):
        return "—"
    return f"{100 * num / denom:.1f}%"


def _delta_str(delta: float, is_int: bool = False) -> str:
    if pd.isna(delta):
        return "—"
    return f"{int(delta):+d}" if is_int else f"{delta:+.4f}"


def _delta_color(delta: float, higher_is_better: bool | None) -> str:
    if pd.isna(delta) or higher_is_better is None:
        return ""
    if delta > 0:
        return "color:green;font-weight:bold" if higher_is_better else "color:red;font-weight:bold"
    if delta < 0:
        return "color:red;font-weight:bold" if higher_is_better else "color:green;font-weight:bold"
    return ""


# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Comparador de experimentos", layout="wide")
st.title("Comparador de experimentos OVO — hasta 3 simultáneos")
st.caption(f"Auto-refresca cada 60 s · Selección máx. {MAX_SELECT} · Replica únicamente")

if st.button("Refresh"):
    st.cache_data.clear()
    st.rerun()

df_exp, df_class, df_scene = _load()

if df_exp.empty:
    st.warning("No se encontraron experimentos en data/output/Replica/.")
    st.stop()

all_ids = sorted(df_exp["Experiment_ID"].dropna().unique().tolist())
default_pick = _default_selection(all_ids)

selected = st.multiselect(
    f"Experimentos a comparar (máx. {MAX_SELECT})",
    options=all_ids,
    default=default_pick,
    max_selections=MAX_SELECT,
    help="Primer seleccionado actúa como baseline para las deltas.",
)

if not selected:
    st.info("Selecciona al menos un experimento.")
    st.stop()

# Preserve selection order, stable colours by position.
df_sel = df_exp[df_exp["Experiment_ID"].isin(selected)].copy()
df_sel["_order"] = df_sel["Experiment_ID"].map({eid: i for i, eid in enumerate(selected)})
df_sel = df_sel.sort_values("_order").reset_index(drop=True)

# Ensure all reject columns exist (fill 0 for experiments with old/different schema).
for _rc in FUSION_REJECT_COLS:
    if _rc not in df_sel.columns:
        df_sel[_rc] = 0

color_map  = {eid: PALETTE[i % len(PALETTE)] for i, eid in enumerate(selected)}
bg_map     = {eid: PALETTE_BG[i % len(PALETTE_BG)] for i, eid in enumerate(selected)}
fg_map     = {eid: PALETTE_FG[i % len(PALETTE_FG)] for i, eid in enumerate(selected)}
letter_map = {eid: chr(ord("A") + i) for i, eid in enumerate(selected)}


# ── Status table ─────────────────────────────────────────────────────────────
st.header("Estado de experimentos seleccionados")

status_rows = []
for _, r in df_sel.iterrows():
    eid = r["Experiment_ID"]
    status_rows.append({
        "Experiment_ID":      eid,
        "Label":              r.get("Label", ""),
        "SLAM_Config":        r.get("SLAM_Config", ""),
        "Fusion":             r.get("Fusion", ""),
        "mIoU":               _fmt(r.get("mIoU")),
        "mAcc":               _fmt(r.get("mAcc")),
        "AP_agnostic":        _fmt(r.get("AP_agnostic")),
        "AP_agnostic_50":     _fmt(r.get("AP_agnostic_50")),
        "Fusion_Accept_Rate": _fmt(r.get("Fusion_Accept_Rate")),
        "Num_Instances":      _fmt(r.get("Num_Instances"), is_int=True),
    })
df_status = pd.DataFrame(status_rows)

# Colour each row by its assigned palette colour.
def _row_style(row: pd.Series) -> list[str]:
    eid = row["Experiment_ID"]
    return [f"background-color: {bg_map[eid]}; color: {fg_map[eid]}"] * len(row)

st.dataframe(
    df_status.style.apply(_row_style, axis=1),
    use_container_width=True, hide_index=True,
)


# ── Delta table ──────────────────────────────────────────────────────────────
if len(selected) >= 2:
    st.header("Deltas vs baseline (primer seleccionado)")
    baseline_id = selected[0]
    base_row = df_sel[df_sel["Experiment_ID"] == baseline_id].iloc[0]

    _DELTA_METRICS = [
        ("mIoU",           True),
        ("mAcc",           True),
        ("Head_mIoU",      True),
        ("Common_mIoU",    True),
        ("Tail_mIoU",      True),
        ("AP",             True),
        ("AP_50",          True),
        ("AP_25",          True),
        ("AP_agnostic",    True),
        ("AP_agnostic_50", True),
        ("AP_agnostic_25", True),
        ("Fusion_Accept_Rate",         True),
        ("Fusion_Reject_Centroid",     None),
        ("Fusion_Reject_AABB",         None),
        ("Fusion_Reject_CosSim",       None),
        ("Fusion_Reject_Overlap",      None),
        ("Fusion_Reject_Cooccurrence", None),
    ]

    other_ids = selected[1:]
    delta_cols = st.columns(len(other_ids))
    for col_widget, eid in zip(delta_cols, other_ids):
        other_row = df_sel[df_sel["Experiment_ID"] == eid].iloc[0]
        with col_widget:
            st.markdown(
                f"**vs <span style='color:{color_map[eid]}'>{eid}</span>**",
                unsafe_allow_html=True,
            )
            tbl_rows = []
            for col_key, higher_better in _DELTA_METRICS:
                if col_key not in df_sel.columns:
                    continue
                is_int = col_key.startswith("Fusion_Reject_")
                b = base_row.get(col_key, float("nan"))
                c = other_row.get(col_key, float("nan"))
                delta = (c - b) if (pd.notna(b) and pd.notna(c)) else float("nan")
                tbl_rows.append({
                    "Métrica":  col_key,
                    "Baseline": _fmt(b, is_int),
                    "Otro":     _fmt(c, is_int),
                    "Δ":        _delta_str(delta, is_int),
                    "_color":   _delta_color(delta, higher_better),
                })
            html = "<table style='font-size:12px;width:100%'>"
            html += "<thead><tr><th>Métrica</th><th>Baseline</th><th>Otro</th><th>Δ</th></tr></thead><tbody>"
            for r in tbl_rows:
                html += (
                    f"<tr><td>{r['Métrica']}</td>"
                    f"<td>{r['Baseline']}</td>"
                    f"<td>{r['Otro']}</td>"
                    f"<td style='{r['_color']}'>{r['Δ']}</td></tr>"
                )
            html += "</tbody></table>"
            st.markdown(html, unsafe_allow_html=True)


# ── Semantic metrics radar ───────────────────────────────────────────────────
st.header("Métricas semánticas (radar)")

avail_radar = [m for m in SEMANTIC_METRICS if m in df_sel.columns and df_sel[m].notna().any()]
if avail_radar:
    radar_labels = [m.replace("_mIoU", "") if m.endswith("_mIoU") else m for m in avail_radar]
    n = len(avail_radar)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(radar_labels, size=9)
    ax.set_ylim(0, 0.6)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_yticklabels(["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"], size=6, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    for _, row in df_sel.iterrows():
        eid = row["Experiment_ID"]
        vals = [float(row[m]) if pd.notna(row.get(m)) else 0.0 for m in avail_radar]
        vals_closed = vals + vals[:1]
        c = color_map[eid]
        ax.plot(angles_closed, vals_closed, color=c, linewidth=1.4, label=eid)
        ax.fill(angles_closed, vals_closed, color=c, alpha=0.12)

    ax.legend(loc="upper right", bbox_to_anchor=(1.6, 1.1), fontsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Instance AP bars ─────────────────────────────────────────────────────────
st.header("Instance AP (segmentación de instancias)")
st.caption(
    "**AP**: instancia correcta **y** clase correcta. "
    "**AP_agnostic**: solo calidad de máscara, ignora clase predicha. Mayor = mejor."
)

avail_ap = [m for m in AP_METRICS if m in df_sel.columns and df_sel[m].notna().any()]
if avail_ap:
    n_cols = min(3, len(avail_ap))
    cols = st.columns(n_cols)
    for i, metric in enumerate(avail_ap):
        with cols[i % n_cols]:
            fig, ax = plt.subplots(figsize=(5, 4))
            x = np.arange(len(df_sel))
            heights = df_sel[metric].fillna(0).values
            colors = [color_map[eid] for eid in df_sel["Experiment_ID"]]
            bars = ax.bar(x, heights, color=colors)
            ymax = max(heights.max(), 0.01)
            ax.set_ylim(0, ymax * 1.25)
            ax.set_xticks(x)
            ax.set_xticklabels(df_sel["Experiment_ID"], rotation=30, ha="right", fontsize=7)
            for bar, h in zip(bars, heights):
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + ymax * 0.02,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
            ax.set_title(metric, fontsize=11)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ── Fusion decisions ─────────────────────────────────────────────────────────
st.header("Decisiones de fusión")
if "Fusion_Total" in df_sel.columns:
    summary_rows = []
    for _, r in df_sel.iterrows():
        total  = int(r.get("Fusion_Total") or 0)
        accept = int(r.get("Fusion_Accepted") or 0)
        rej    = total - accept
        rate   = r.get("Fusion_Accept_Rate", float("nan"))
        summary_rows.append({
            "Experiment_ID": r["Experiment_ID"],
            "Evaluados":     f"{total:,}" if total else "—",
            "Fusionados":    f"{accept:,}" if accept else "—",
            "Rechazados":    f"{rej:,}" if rej else "—",
            "Tasa%":         f"{100 * rate:.2f}%" if pd.notna(rate) else "—",
            "Centroide%":    _fmt_pct(r.get("Fusion_Reject_Centroid", 0), rej),
            "AABB%":         _fmt_pct(r.get("Fusion_Reject_AABB", 0), rej),
            "CosSim%":       _fmt_pct(r.get("Fusion_Reject_CosSim", 0), rej),
            "Overlap%":      _fmt_pct(r.get("Fusion_Reject_Overlap", 0), rej),
            "Cooccur%":      _fmt_pct(r.get("Fusion_Reject_Cooccurrence", 0), rej),
        })
    df_fus = pd.DataFrame(summary_rows)
    st.dataframe(
        df_fus.style.apply(_row_style, axis=1),
        use_container_width=True, hide_index=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Tasa de aceptación")
        if "Fusion_Accept_Rate" in df_sel.columns and df_sel["Fusion_Accept_Rate"].notna().any():
            fig, ax = plt.subplots(figsize=(6, 4))
            heights = df_sel["Fusion_Accept_Rate"].fillna(0).values
            colors = [color_map[eid] for eid in df_sel["Experiment_ID"]]
            bars = ax.bar(np.arange(len(df_sel)), heights, color=colors)
            ymax = max(heights.max(), 0.01)
            ax.set_ylim(0, ymax * 1.3)
            ax.set_xticks(np.arange(len(df_sel)))
            ax.set_xticklabels(df_sel["Experiment_ID"], rotation=30, ha="right", fontsize=7)
            for bar, h in zip(bars, heights):
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                            f"{100*h:.2f}%", ha="center", va="bottom", fontsize=8)
            ax.set_ylabel("Tasa aceptación")
            ax.set_title("Fusion accept rate")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col_b:
        st.subheader("Razones de rechazo (% sobre rechazadas)")
        reason_cols = [c for c in FUSION_REJECT_COLS if c in df_sel.columns]
        if reason_cols:
            df_r = df_sel[["Experiment_ID", "Fusion_Total", "Fusion_Accepted"] + reason_cols].copy()
            df_r["Rejected"] = df_r["Fusion_Total"] - df_r["Fusion_Accepted"]
            for c in reason_cols:
                df_r[c] = df_r[c] / df_r["Rejected"].replace(0, float("nan")) * 100
            df_r_plot = df_r.set_index("Experiment_ID")[reason_cols]
            df_r_plot.columns = [c.replace("Fusion_Reject_", "") for c in df_r_plot.columns]
            fig, ax = plt.subplots(figsize=(6, 4))
            df_r_plot.plot(kind="bar", stacked=True, ax=ax,
                           color=["#e07070", "#70a0e0", "#70c070", "#c070d0", "#f0a030"][:len(reason_cols)])
            ax.set_ylim(0, 110)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=30)
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


# ── Per-scene breakdown ──────────────────────────────────────────────────────
st.header("Desglose por escena")

df_scene_sel = df_scene[df_scene["Experiment_ID"].isin(selected)].copy() if not df_scene.empty else pd.DataFrame()

if df_scene_sel.empty:
    st.info("Sin datos por escena disponibles para la selección.")
else:
    scenes_ordered = sorted(df_scene_sel["Scene"].unique())
    n_scenes = len(scenes_ordered)

    # Radar per metric, one column per experiment
    _SCENE_RADAR_METRICS = [
        ("mIoU",          "mIoU",         False, 0.4),
        ("mAcc",          "mAcc",         False, 0.5),
        ("AP_agnostic",   "AP_agnostic",  False, 0.3),
        ("Num_Instances", "N_Instancias", True,  140),
    ]
    avail_scene = [
        (k, lbl, norm, fmx) for k, lbl, norm, fmx in _SCENE_RADAR_METRICS
        if k in df_scene_sel.columns and df_scene_sel[k].notna().any()
    ]

    if avail_scene and n_scenes >= 2:
        s_angles = np.linspace(0, 2 * np.pi, n_scenes, endpoint=False).tolist()
        s_angles_closed = s_angles + s_angles[:1]

        for col_key, col_label, normalize, fixed_max in avail_scene:
            st.subheader(col_label)
            if fixed_max is not None:
                global_max = fixed_max
            else:
                global_max = df_scene_sel[col_key].dropna().max() if normalize else 1.0
            if not global_max or global_max == 0:
                global_max = 1.0

            radar_layout = st.columns([1, 2, 1])
            with radar_layout[1]:
                fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"polar": True})
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                ax.set_xticks(s_angles)
                ax.set_xticklabels(scenes_ordered, size=8)
                ax.set_ylim(0, global_max)
                if normalize:
                    ticks = [global_max * f for f in [0.25, 0.5, 0.75, 1.0]]
                    tick_labels = [f"{int(t)}" for t in ticks]
                else:
                    ticks = [v for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] if v <= global_max]
                    tick_labels = [str(v) for v in ticks]
                ax.set_yticks(ticks)
                ax.set_yticklabels(tick_labels, size=6, color="grey")
                ax.grid(color="grey", linestyle="--", linewidth=0.4, alpha=0.5)

                for eid in selected:
                    sub = df_scene_sel[df_scene_sel["Experiment_ID"] == eid]
                    vals = []
                    for scene in scenes_ordered:
                        scene_row = sub[sub["Scene"] == scene]
                        v = float(scene_row.iloc[0][col_key]) if not scene_row.empty and pd.notna(scene_row.iloc[0][col_key]) else 0.0
                        vals.append(v)
                    vals_closed = vals + vals[:1]
                    c = color_map[eid]
                    ax.plot(s_angles_closed, vals_closed, color=c, linewidth=1.3, label=eid)
                    ax.fill(s_angles_closed, vals_closed, color=c, alpha=0.12)

                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
                          ncol=1, fontsize=7, frameon=False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # Per-scene comparative table: one row per scene, one block per metric
    # with values for each experiment + Δ vs baseline (first selected).
    st.subheader("Tabla comparativa por escena")

    legend_parts = [
        f"<span style='color:{color_map[eid]};font-weight:bold'>{letter_map[eid]}</span> = {eid}"
        for eid in selected
    ]
    st.markdown(
        "<div style='font-size:13px;margin-bottom:8px;line-height:1.5'>" + "<br>".join(legend_parts) + "</div>",
        unsafe_allow_html=True,
    )

    baseline_id = selected[0]
    others = selected[1:]

    tbl_rows = []
    for scene in scenes_ordered:
        rec = {"Escena": scene}
        for col_key, col_label, _, _ in avail_scene:
            is_int = col_key == "Num_Instances"
            fmt_v = (
                (lambda x: "—" if pd.isna(x) else f"{int(x)}") if is_int
                else (lambda x: "—" if pd.isna(x) else f"{x:.4f}")
            )
            vals = {}
            for eid in selected:
                sub = df_scene_sel[
                    (df_scene_sel["Experiment_ID"] == eid) & (df_scene_sel["Scene"] == scene)
                ]
                v = float(sub.iloc[0][col_key]) if not sub.empty and pd.notna(sub.iloc[0].get(col_key)) else float("nan")
                vals[eid] = v
                rec[f"{col_label} [{letter_map[eid]}]"] = fmt_v(v)
            for eid in others:
                d = (vals[eid] - vals[baseline_id]
                     if pd.notna(vals[eid]) and pd.notna(vals[baseline_id]) else float("nan"))
                rec[f"Δ{col_label} [{letter_map[eid]}]"] = (
                    "—" if pd.isna(d) else (f"{int(d):+d}" if is_int else f"{d:+.4f}")
                )
        tbl_rows.append(rec)

    df_scene_tbl = pd.DataFrame(tbl_rows)

    # higher_is_better per metric: None = ambiguous (no color)
    _HIB = {"mIoU": True, "mAcc": True, "AP_agnostic": True, "N_Instancias": False}
    delta_meta = {}  # delta col name -> higher_is_better
    for c in df_scene_tbl.columns:
        if c.startswith("Δ"):
            mname = c[1:].split(" [")[0]
            delta_meta[c] = _HIB.get(mname)

    def _color_delta(val: str, higher_is_better: bool | None) -> str:
        if higher_is_better is None:
            return ""
        if isinstance(val, str) and val not in ("—", ""):
            try:
                num = float(val.replace("+", ""))
                if num > 0:
                    return "color: green; font-weight: bold" if higher_is_better else "color: red; font-weight: bold"
                if num < 0:
                    return "color: red; font-weight: bold" if higher_is_better else "color: green; font-weight: bold"
            except ValueError:
                pass
        return ""

    styled = df_scene_tbl.style
    for col, hib in delta_meta.items():
        styled = styled.applymap(lambda v, hib=hib: _color_delta(v, hib), subset=[col])
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ── Fusion per scene ─────────────────────────────────────────────────────────
st.header("Fusión por escena")
if not df_scene_sel.empty and "Fusion_Total" in df_scene_sel.columns:
    def _fi(row, col):
        v = row.get(col) if row is not None else None
        return int(v) if v is not None and pd.notna(v) else 0

    def _pct(n, rej): return f"{100*n/rej:.1f}%" if rej > 0 else "—"

    fus_rows = []
    for scene in sorted(df_scene_sel["Scene"].unique()):
        rec = {"Escena": scene}
        for eid in selected:
            L = letter_map[eid]
            sub = df_scene_sel[
                (df_scene_sel["Experiment_ID"] == eid) & (df_scene_sel["Scene"] == scene)
            ]
            rr = sub.iloc[0] if not sub.empty else None
            tot  = _fi(rr, "Fusion_Total")
            acc  = _fi(rr, "Fusion_Accepted")
            rej  = tot - acc
            rate = rr.get("Fusion_Accept_Rate") if rr is not None else None
            cen  = _fi(rr, "Fusion_Reject_Centroid")
            aab  = _fi(rr, "Fusion_Reject_AABB")
            cos  = _fi(rr, "Fusion_Reject_CosSim")
            ovl  = _fi(rr, "Fusion_Reject_Overlap")
            coo  = _fi(rr, "Fusion_Reject_Cooccurrence")

            rec[f"Eval [{L}]"]      = f"{tot:,}" if tot else "—"
            rec[f"Fus [{L}]"]       = f"{acc:,}" if acc else "—"
            rec[f"Rej [{L}]"]       = f"{rej:,}" if rej else "—"
            rec[f"Tasa% [{L}]"]     = f"{100*rate:.2f}%" if pd.notna(rate) else "—"
            rec[f"Cen% [{L}]"]      = _pct(cen, rej)
            rec[f"AABB% [{L}]"]     = _pct(aab, rej)
            rec[f"Cos% [{L}]"]      = _pct(cos, rej)
            rec[f"Ovl% [{L}]"]      = _pct(ovl, rej)
            rec[f"Coo% [{L}]"]      = _pct(coo, rej)
        fus_rows.append(rec)

    df_fus_tbl = pd.DataFrame(fus_rows)

    legend_parts = [
        f"<span style='color:{color_map[eid]};font-weight:bold'>{letter_map[eid]}</span> = {eid}"
        for eid in selected
    ]
    st.markdown(
        "<div style='font-size:13px;margin-bottom:8px;line-height:1.5'>" + "<br>".join(legend_parts) + "</div>",
        unsafe_allow_html=True,
    )

    styled = df_fus_tbl.style
    for eid in selected:
        L = letter_map[eid]
        cols_eid = [c for c in df_fus_tbl.columns if c.endswith(f"[{L}]")]
        styled = styled.set_properties(
            subset=cols_eid,
            **{"background-color": bg_map[eid], "color": fg_map[eid]},
        )
        fus_cols_eid = [c for c in cols_eid if c.startswith("Fus ")]
        rej_cols_eid = [c for c in cols_eid if c.startswith("Rej ")]
        styled = styled.set_properties(
            subset=fus_cols_eid, **{"font-weight": "bold", "color": "#1a7a1a"},
        )
        styled = styled.set_properties(
            subset=rej_cols_eid, **{"font-weight": "bold", "color": "#c0392b"},
        )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ── Per-class IoU heatmap ────────────────────────────────────────────────────
st.header("Heatmap IoU por clase")
df_class_sel = df_class[df_class["Experiment_ID"].isin(selected)].copy() if not df_class.empty else pd.DataFrame()
if not df_class_sel.empty:
    pivot_all = df_class_sel.pivot_table(
        index="Class", columns="Experiment_ID", values="IoU", aggfunc="mean"
    )
    # Keep ordering by selection
    cols_in_order = [eid for eid in selected if eid in pivot_all.columns]
    pivot_all = pivot_all[cols_in_order]
    active = pivot_all.index[pivot_all.max(axis=1).fillna(0) > 0]
    pivot_all = pivot_all.loc[active]
    n_rows = len(pivot_all)

    if n_rows > 0:
        with st.expander(f"Ver heatmap ({n_rows} clases con datos)", expanded=True):
            fig, ax = plt.subplots(figsize=(2 + 1.2 * len(cols_in_order), max(3, n_rows * 0.22)))
            sns.heatmap(
                pivot_all, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.3, linecolor="grey",
                annot=True, fmt=".2f", annot_kws={"size": 7},
                cbar=True,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelsize=8, rotation=20)
            ax.tick_params(axis="y", labelsize=7)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)


# ── Raw data ─────────────────────────────────────────────────────────────────
with st.expander("Datos brutos"):
    display_cols = [
        "Experiment_ID", "Date", "SLAM_Config", "Fusion", "Label",
        "Jump_Count", "Trans_Noise", "Rot_Noise",
        "mIoU", "mAcc", "Head_mIoU", "Common_mIoU", "Tail_mIoU",
        "AP", "AP_50", "AP_25", "AP_agnostic", "AP_agnostic_50", "AP_agnostic_25",
        "Num_Instances",
        "Fusion_Total", "Fusion_Accepted", "Fusion_Accept_Rate",
        "Fusion_Reject_Centroid", "Fusion_Reject_AABB",
        "Fusion_Reject_CosSim", "Fusion_Reject_Overlap",
        "Fusion_Reject_Cooccurrence",
    ]
    avail_disp = [c for c in display_cols if c in df_sel.columns]
    st.dataframe(df_sel[avail_disp], use_container_width=True, hide_index=True)
